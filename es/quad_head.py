import nevergrad as ng
import numpy as np
from tqdm import tqdm
import ray
import multiprocessing as mp
from argparse import Namespace

from quad_worker import QuadWorker

def run_quad_fdm(conf: Namespace, _run=None):
    optim_list = conf.optim_method
    for optim in optim_list:
        if conf.pipeline == 'all':
            conf.score_type = 'all'
            run_quad_fdm_with_optim_all_params(conf, optim, _run)

def run_quad_fdm_with_optim_all_params(conf: Namespace, optimizer, _run=None):
    # seeding
    np.random.seed(conf.seed)
    
    # saving the optimizer
    import os
    if not os.path.exists('iccps_runs/npzs'):
        os.makedirs('iccps_runs/npzs')
    if not os.path.exists('iccps_runs/optims_pkl'):
        os.makedirs('iccps_runs/optims_pkl')

    filename = 'iccps_runs/npzs/' + conf.run_name + '_' + optimizer + '_budget' + str(conf.budget) + '.npz'
    filename_optim = 'iccps_runs/optims_pkl/' + conf.run_name + '_' + optimizer + '_budget' + str(conf.budget) + '_optim.pkl'
    
    if not os.path.exists(conf.base_folder):
        os.makedirs(conf.base_folder)

    num_cores = mp.cpu_count()

    # setting up parameter space
    param = ng.p.Dict()

    # discrete components
    for i in range(conf.design_space['battery'][0]):
        param['battery' + str(i)] = ng.p.Choice(np.arange(conf.design_space['battery'][1], dtype=int))
    for i in range(conf.design_space['esc'][0]):
        param['esc' + str(i)] = ng.p.Choice(np.arange(conf.design_space['esc'][1], dtype=int))
    for i in range(conf.design_space['arm'][0]):
        param['arm' + str(i)] = ng.p.Choice(np.arange(conf.design_space['arm'][1], dtype=int))
    for i in range(conf.design_space['prop'][0]):
        param['prop' + str(i)] = ng.p.Choice(np.arange(conf.design_space['prop'][1], dtype=int))
    for i in range(conf.design_space['motor'][0]):
        param['motor' + str(i)] = ng.p.Choice(np.arange(conf.design_space['motor'][1], dtype=int))
    for i in range(conf.design_space['support'][0]):
        param['support' + str(i)] = ng.p.Choice(np.arange(conf.design_space['support'][1], dtype=int))

    # continuous components
    for i in range(conf.design_space['arm_length'][0]):
        param['arm_length' + str(i)] = ng.p.Scalar(lower=conf.design_space['arm_length'][1], upper=conf.design_space['arm_length'][2])
    for i in range(conf.design_space['support_length'][0]):
        param['support_length' + str(i)] = ng.p.Scalar(lower=conf.design_space['support_length'][1], upper=conf.design_space['support_length'][2])

    # control params
    # all parameters tuned in the same run, with raw score
    param['lqr_vector1'] = ng.p.Array(shape=(conf.design_space['LQR_1'][0], ), lower=conf.design_space['LQR_1'][1], upper=conf.design_space['LQR_1'][2])
    param['lqr_vector3'] = ng.p.Array(shape=(conf.design_space['LQR_3'][0], ), lower=conf.design_space['LQR_3'][1], upper=conf.design_space['LQR_3'][2])
    param['lqr_vector4'] = ng.p.Array(shape=(conf.design_space['LQR_4'][0], ), lower=conf.design_space['LQR_4'][1], upper=conf.design_space['LQR_4'][2])
    param['lqr_vector5'] = ng.p.Array(shape=(conf.design_space['LQR_5'][0], ), lower=conf.design_space['LQR_5'][1], upper=conf.design_space['LQR_5'][2])
    param['lat_vel'] = ng.p.Array(shape=(conf.design_space['lateral_velocity'][0], ), lower=conf.design_space['lateral_velocity'][1], upper=conf.design_space['lateral_velocity'][2])
    param['vert_vel'] = ng.p.Array(shape=(conf.design_space['vertical_velocity'][0], ), lower=conf.design_space['vertical_velocity'][1], upper=conf.design_space['vertical_velocity'][2])

    # setting up optimizer with hyperparams
    optim = ng.optimizers.registry[optimizer](parametrization=param, budget=conf.budget, num_workers=num_cores)

    # seeding
    optim.parametrization.random_state = np.random.RandomState(conf.seed)
    print('Optimizer: ', optim)

    # setting up workers
    workers = [QuadWorker.remote(conf, worker_id) for worker_id in range(num_cores)]

    # all scores
    all_scores = []
    all_individuals = []

    # work distribution loop
    eval_id = 0
    for prog in tqdm(range(conf.budget // num_cores)):
        individuals = [optim.ask() for _ in range(num_cores)]
        results = []

        # distribute
        for ind, worker in zip(individuals, workers):
            work = ind.args[0]
            work['eval_id'] = eval_id
            #ind.args[0]['eval_id'] = eval_id
            worker.run_sim.remote(work)
            eval_id += 1

        # collect
        future_results = [worker.collect.remote() for worker in workers]
        results = ray.get(future_results)

        # update optimization
        # negate since we want to maximize scores
        for ind, score in zip(individuals, results):
            if conf.score_type != 'trim':
                optim.tell(ind, 1600.0 - np.sum(score))
            else:
                optim.tell(ind, np.sum(score[:-1]))

        # collect all
        all_scores.extend(results)
        all_individuals.extend(individuals)

        if prog % 10 == 0:
            score_all_np = np.asarray(all_scores)
            latvel_all_np = score_all_np[:, -1]
            score_all_np = score_all_np[:, :-1]
            if conf.score_type != 'trim':
                print("Current High Score: " + str(np.max(np.sum(score_all_np, axis=1))))
                print("At index: " + str(str(np.argmax(np.sum(score_all_np, axis=1)))))
            else:
                print('Current Trim Only Best Score: ' + str(np.min(np.sum(score_all_np, axis=1))))
                print("At index: " + str(str(np.argmin(np.sum(score_all_np, axis=1)))))

            selected_vectors = []
            for indi in all_individuals:
                current_vec = []
                d = indi.args[0]
                for key in d:
                    if isinstance(d[key], np.ndarray) or isinstance(d[key], list):
                        current_vec.extend(list(d[key]))
                    else:
                        current_vec.append(d[key])
                selected_vectors.append(current_vec)

            vector_all_np = np.asarray(selected_vectors)
            np.savez_compressed(filename, scores=score_all_np, vectors=vector_all_np, latvels=latvel_all_np)
            # _run.add_artifact(filename)
            optim.dump(filename_optim)
            # _run.add_artifact(filename_optim)

    # storing as npz, while running as sacred experiment, the directory iccps_runs should've been created
    # column 0 is eval 1 score, column 1-3 is eval 3-5 score
    score_all_np = np.asarray(all_scores)
    latvel_all_np = score_all_np[:, -1]
    score_all_np = score_all_np[:, :-1]
    if not conf.score_type == 'trim':
        print("Current High Score: " + str(np.max(np.sum(score_all_np, axis=1))))
        print("At index: " + str(str(np.argmax(np.sum(score_all_np, axis=1)))))
    else:
        print('Current Trim Only Best Score: ' + str(np.min(np.sum(score_all_np, axis=1))))
        print("At index: " + str(str(np.argmin(np.sum(score_all_np, axis=1)))))

    selected_vectors = []
    for indi in all_individuals:
        current_vec = []
        d = indi.args[0]
        for key in d:
            if isinstance(d[key], np.ndarray) or isinstance(d[key], list):
                current_vec.extend(list(d[key]))
            else:
                current_vec.append(d[key])
        selected_vectors.append(current_vec)
    
    vector_all_np = np.asarray(selected_vectors)
    np.savez_compressed(filename, scores=score_all_np, vectors=vector_all_np, latvels=latvel_all_np)
    # _run.add_artifact(filename)
    optim.dump(filename_optim)
    # _run.add_artifact(filename_optim)

def run_quad_fdm_with_optim_seq(conf: Namespace, optimizer, _run=None):
    # seeding
    np.random.seed(conf.seed)
    
    # saving the optimizer
    import os
    if not os.path.exists('iccps_runs/npzs'):
        os.makedirs('iccps_runs/npzs')
    if not os.path.exists('iccps_runs/optims_pkl'):
        os.makedirs('iccps_runs/optims_pkl')

    filename = 'iccps_runs/npzs/' + conf.run_name + '_' + optimizer + '_budget' + str(conf.budget) + '.npz'
    filename_optim = 'iccps_runs/optims_pkl/' + conf.run_name + '_' + optimizer + '_budget' + str(conf.budget) + '_optim.pkl'
    
    if not os.path.exists(conf.base_folder):
        os.makedirs(conf.base_folder)

    num_cores = mp.cpu_count()

    # setting up parameter space
    param = ng.p.Dict()

    # discrete components
    for i in range(conf.design_space['battery'][0]):
        param['battery' + str(i)] = ng.p.Choice(np.arange(conf.design_space['battery'][1], dtype=int))
    for i in range(conf.design_space['esc'][0]):
        param['esc' + str(i)] = ng.p.Choice(np.arange(conf.design_space['esc'][1], dtype=int))
    for i in range(conf.design_space['arm'][0]):
        param['arm' + str(i)] = ng.p.Choice(np.arange(conf.design_space['arm'][1], dtype=int))
    for i in range(conf.design_space['prop'][0]):
        param['prop' + str(i)] = ng.p.Choice(np.arange(conf.design_space['prop'][1], dtype=int))
    for i in range(conf.design_space['motor'][0]):
        param['motor' + str(i)] = ng.p.Choice(np.arange(conf.design_space['motor'][1], dtype=int))
    for i in range(conf.design_space['support'][0]):
        param['support' + str(i)] = ng.p.Choice(np.arange(conf.design_space['support'][1], dtype=int))

    # continuous components
    for i in range(conf.design_space['arm_length'][0]):
        param['arm_length' + str(i)] = ng.p.Scalar(lower=conf.design_space['arm_length'][1], upper=conf.design_space['arm_length'][2])
    for i in range(conf.design_space['support_length'][0]):
        param['support_length' + str(i)] = ng.p.Scalar(lower=conf.design_space['support_length'][1], upper=conf.design_space['support_length'][2])

    # control params
    # all parameters tuned in the same run, with raw score
    param['lqr_vector1'] = ng.p.Array(shape=(conf.design_space['LQR_1'][0], ), lower=conf.design_space['LQR_1'][1], upper=conf.design_space['LQR_1'][2])
    param['lqr_vector3'] = ng.p.Array(shape=(conf.design_space['LQR_3'][0], ), lower=conf.design_space['LQR_3'][1], upper=conf.design_space['LQR_3'][2])
    param['lqr_vector4'] = ng.p.Array(shape=(conf.design_space['LQR_4'][0], ), lower=conf.design_space['LQR_4'][1], upper=conf.design_space['LQR_4'][2])
    param['lqr_vector5'] = ng.p.Array(shape=(conf.design_space['LQR_5'][0], ), lower=conf.design_space['LQR_5'][1], upper=conf.design_space['LQR_5'][2])
    param['lat_vel'] = ng.p.Array(shape=(conf.design_space['lateral_velocity'][0], ), lower=conf.design_space['lateral_velocity'][1], upper=conf.design_space['lateral_velocity'][2])
    param['vert_vel'] = ng.p.Array(shape=(conf.design_space['vertical_velocity'][0], ), lower=conf.design_space['vertical_velocity'][1], upper=conf.design_space['vertical_velocity'][2])

    # setting up optimizer with hyperparams
    optim = ng.optimizers.registry[optimizer](parametrization=param, budget=conf.budget, num_workers=num_cores)

    # seeding
    optim.parametrization.random_state = np.random.RandomState(conf.seed)
    print('Optimizer: ', optim)

    # setting up workers
    workers = [QuadWorker.remote(conf, worker_id) for worker_id in range(num_cores)]

    # all scores
    all_scores = []
    all_individuals = []

    # work distribution loop
    eval_id = 0
    for prog in tqdm(range(conf.budget // num_cores)):
        individuals = [optim.ask() for _ in range(num_cores)]
        results = []

        # distribute
        for ind, worker in zip(individuals, workers):
            work = ind.args[0]
            work['eval_id'] = eval_id
            #ind.args[0]['eval_id'] = eval_id
            worker.run_sim.remote(work)
            eval_id += 1

        # collect
        future_results = [worker.collect.remote() for worker in workers]
        results = ray.get(future_results)

        # update optimization
        # negate since we want to maximize scores
        for ind, score in zip(individuals, results):
            if (not conf.trim_only) and (not conf.trim_discrete_only):
                optim.tell(ind, 1640.0 - np.sum(score))
            else:
                optim.tell(ind, np.sum(score))

        # collect all
        all_scores.extend(results)
        all_individuals.extend(individuals)

        if prog % 5 == 0:
            score_all_np = np.asarray(all_scores)
            if (not conf.trim_only) and (not conf.trim_discrete_only) and (not conf.trim_arm_only):
                print("Current High Score: " + str(np.max(np.sum(score_all_np, axis=1))))
                print("At index: " + str(str(np.argmax(np.sum(score_all_np, axis=1)))))
            else:
                print('Current Trim Only Best Score: ' + str(np.min(np.sum(score_all_np, axis=1))))
                print("At index: " + str(str(np.argmin(np.sum(score_all_np, axis=1)))))
            selected_vectors = []
            for indi in all_individuals:
                current_vec = []
                d = indi.args[0]
                for key in d:
                    if isinstance(d[key], np.ndarray) or isinstance(d[key], list):
                        current_vec.extend(list(d[key]))
                    else:
                        current_vec.append(d[key])
                selected_vectors.append(current_vec)

            vector_all_np = np.asarray(selected_vectors)
            np.savez_compressed(filename, scores=score_all_np, vectors=vector_all_np)
            _run.add_artifact(filename)
            optim.dump(filename_optim)
            _run.add_artifact(filename_optim)

    # storing as npz, while running as sacred experiment, the directory iccps_runs should've been created
    # column 0 is eval 1 score, column 1-3 is eval 3-5 score
    score_all_np = np.asarray(all_scores)
    if (not conf.trim_only) and (not conf.trim_discrete_only) and (not conf.trim_arm_only):
        print("Current High Score: " + str(np.max(np.sum(score_all_np, axis=1))))
        print("At index: " + str(str(np.argmax(np.sum(score_all_np, axis=1)))))
    else:
        print('Current Trim Only Best Score: ' + str(np.min(np.sum(score_all_np, axis=1))))
        print("At index: " + str(str(np.argmin(np.sum(score_all_np, axis=1)))))
    selected_vectors = []
    for indi in all_individuals:
        current_vec = []
        d = indi.args[0]
        for key in d:
            if isinstance(d[key], np.ndarray) or isinstance(d[key], list):
                current_vec.extend(list(d[key]))
            else:
                current_vec.append(d[key])
        selected_vectors.append(current_vec)
    
    vector_all_np = np.asarray(selected_vectors)
    np.savez_compressed(filename, scores=score_all_np, vectors=vector_all_np)
    _run.add_artifact(filename)
    optim.dump(filename_optim)
    _run.add_artifact(filename_optim)
