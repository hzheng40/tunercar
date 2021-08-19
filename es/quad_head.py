import nevergrad as ng
import numpy as np
from tqdm import tqdm
import ray
import multiprocessing as mp
from argparse import Namespace

from quad_worker import QuadWorker

def run_quad_fdm(conf: Namespace, _run=None):
    # seeding
    np.random.seed(conf.seed)
    
    # saving the optimizer
    import os
    if not os.path.exists('quad_fdm_runs/npzs'):
        os.makedirs('quad_fdm_runs/npzs')
    if not os.path.exists('quad_fdm_runs/optims_pkl'):
        os.makedirs('quad_fdm_runs/optims_pkl')
    filename = 'quad_fdm_runs/npzs/' + conf.run_name + '_' + conf.optim_method + '_budget' + str(conf.budget) + '.npz'
    filename_optim = 'quad_fdm_runs/optims_pkl/' + conf.run_name + '_' + conf.optim_method + '_budget' + str(conf.budget) + '_optim.pkl'
    if not os.path.exists(conf.base_folder):
        os.makedirs(conf.base_folder)

    num_cores = mp.cpu_count()

    # setting up parameter space, hplane has more nodes
    if conf.vehicle != 'hplane':
        param = ng.p.Dict()

        if not conf.warm_start:
            # include discrete choices if not warm start
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

            # continuous parameters
            if not conf.discrete_only:
                for i in range(conf.design_space['arm_length'][0]):
                    param['arm_length' + str(i)] = ng.p.Scalar(lower=conf.design_space['arm_length'][1], upper=conf.design_space['arm_length'][2])
                for i in range(conf.design_space['support_length'][0]):
                    param['support_length' + str(i)] = ng.p.Scalar(lower=conf.design_space['support_length'][1], upper=conf.design_space['support_length'][2])
                param['lqr_vector'] = ng.p.Array(shape=(conf.design_space['LQR'][0], ), lower=conf.design_space['LQR'][1], upper=conf.design_space['LQR'][2])
                param['lat_vel'] = ng.p.Array(shape=(conf.design_space['lateral_velocity'][0], ), lower=conf.design_space['lateral_velocity'][1], upper=conf.design_space['lateral_velocity'][2])
                param['vert_vel'] = ng.p.Array(shape=(conf.design_space['vertical_velocity'][0], ), lower=conf.design_space['vertical_velocity'][1], upper=conf.design_space['vertical_velocity'][2])
            else:
                import baselines
                # use general continuous params
                lqr = list(eval('baselines.default_lqr'))
                latvel = list(eval('baselines.default_latvel'))
                vertvel = list(eval('baselines.default_vertvel'))
                arm_lengths = list(eval('baselines.' + conf.vehicle + '_arm_lengths'))
                support_lengths = list(eval('baselines.' + conf.vehicle + '_support_lengths'))
                continunous_baseline = [*arm_lengths, *support_lengths, *lqr, *latvel, *vertvel]
                param['continunous_baseline'] = continunous_baseline
            
        else:
            # continuous parameters
            for i in range(conf.design_space['arm_length'][0]):
                param['arm_length' + str(i)] = ng.p.Scalar(lower=conf.design_space['arm_length'][1], upper=conf.design_space['arm_length'][2])
            for i in range(conf.design_space['support_length'][0]):
                param['support_length' + str(i)] = ng.p.Scalar(lower=conf.design_space['support_length'][1], upper=conf.design_space['support_length'][2])
            param['lqr_vector'] = ng.p.Array(shape=(conf.design_space['LQR'][0], ), lower=conf.design_space['LQR'][1], upper=conf.design_space['LQR'][2])
            param['lat_vel'] = ng.p.Array(shape=(conf.design_space['lateral_velocity'][0], ), lower=conf.design_space['lateral_velocity'][1], upper=conf.design_space['lateral_velocity'][2])
            param['vert_vel'] = ng.p.Array(shape=(conf.design_space['vertical_velocity'][0], ), lower=conf.design_space['vertical_velocity'][1], upper=conf.design_space['vertical_velocity'][2])

            import baselines
            # load warm start baseline for discrete parameters
            num_discrete = conf.design_space['battery'][0] + conf.design_space['esc'][0] + conf.design_space['arm'][0] + conf.design_space['prop'][0] + conf.design_space['motor'][0] + conf.design_space['support'][0]
            discrete_baseline = list((eval('baselines.' + conf.warm_start_params['baseline'])[:num_discrete]).astype(int))
            print(discrete_baseline)
            param['discrete_baseline'] = discrete_baseline
            # initialize continuous params from baseline
            vert_size = conf.design_space['vertical_velocity'][0]
            lat_size = conf.design_space['lateral_velocity'][0]
            lqr_size = conf.design_space['LQR'][0]
            param['vert_vel'].value = eval('baselines.' + conf.warm_start_params['baseline'])[-vert_size:]
            param['lat_vel'].value = eval('baselines.' + conf.warm_start_params['baseline'])[-(vert_size + lat_size):-vert_size]
            param['lqr_vector'].value = eval('baselines.' + conf.warm_start_params['baseline'])[-(vert_size + lat_size + lqr_size):-(vert_size + lat_size)]

    else:
        param = ng.p.Dict()
        for i in range(conf.design_space['battery'][0]):
            param['battery' + str(i)] = ng.p.Choice(np.arange(conf.design_space['battery'][1], dtype=int))
        for i in range(conf.design_space['esc'][0]):
            param['esc' + str(i)] = ng.p.Choice(np.arange(conf.design_space['esc'][1], dtype=int))
        for i in range(conf.design_space['arm'][0]):
            param['arm' + str(i)] = ng.p.Choice(np.arange(conf.design_space['arm'][1], dtype=int))
        for i in range(conf.design_space['wing_support'][0]):
            param['wing_support' + str(i)] = ng.p.Choice(np.arange(conf.design_space['wing_support'][1], dtype=int))
        for i in range(conf.design_space['prop'][0]):
            param['prop' + str(i)] = ng.p.Choice(np.arange(conf.design_space['prop'][1], dtype=int))
        for i in range(conf.design_space['motor'][0]):
            param['motor' + str(i)] = ng.p.Choice(np.arange(conf.design_space['motor'][1], dtype=int))
        for i in range(conf.design_space['flange_support'][0]):
            param['flange_support' + str(i)] = ng.p.Choice(np.arange(conf.design_space['flange_support'][1], dtype=int))
        for i in range(conf.design_space['wing'][0]):
            param['wing' + str(i)] = ng.p.Choice(np.arange(conf.design_space['wing'][1], dtype=int))
        for i in range(conf.design_space['servo'][0]):
            param['servo' + str(i)] = ng.p.Choice(np.arange(conf.design_space['servo'][1], dtype=int))
        for i in range(conf.design_space['arm_length'][0]):
            param['arm_length' + str(i)] = ng.p.Scalar(lower=conf.design_space['arm_length'][1], upper=conf.design_space['arm_length'][2])
        for i in range(conf.design_space['flange_support_length'][0]):
            param['flange_support_length' + str(i)] = ng.p.Scalar(lower=conf.design_space['flange_support_length'][1], upper=conf.design_space['flange_support_length'][2])
        for i in range(conf.design_space['wing_offset'][0]):
            param['wing_offset' + str(i)] = ng.p.Scalar(lower=conf.design_space['wing_offset'][1], upper=conf.design_space['wing_offset'][2])
        for i in range(conf.design_space['wing_span'][0]):
            param['wing_span' + str(i)] = ng.p.Scalar(lower=conf.design_space['wing_span'][1], upper=conf.design_space['wing_span'][2])
        for i in range(conf.design_space['wing_chord'][0]):
            param['wing_chord' + str(i)] = ng.p.Scalar(lower=conf.design_space['wing_chord'][1], upper=conf.design_space['wing_chord'][2])
        param['lqr_vector'] = ng.p.Array(shape=(conf.design_space['LQR'][0], ), lower=conf.design_space['LQR'][1], upper=conf.design_space['LQR'][2])
        param['lat_vel'] = ng.p.Array(shape=(conf.design_space['lateral_velocity'][0], ), lower=conf.design_space['lateral_velocity'][1], upper=conf.design_space['lateral_velocity'][2])
        param['vert_vel'] = ng.p.Array(shape=(conf.design_space['vertical_velocity'][0], ), lower=conf.design_space['vertical_velocity'][1], upper=conf.design_space['vertical_velocity'][2])

    # setting up optimizer with hyperparams
    if conf.optim_method == 'CMA' and conf.optim_params['popsize'] != 'default':
        # configured CMA, popsize tested and working
        optim = ng.optimizers.registry[conf.optim_method](parametrization=param, budget=conf.budget, num_workers=num_cores)
        optim._popsize = conf.optim_params['popsize']

    elif conf.optim_method == 'NoisyDE' and conf.optim_params['popsize'] != 'default':
        # configured DE, noisy recommendation
        optim = ng.optimizers.DifferentialEvolution(recommendation='noisy', popsize=conf.optim_params['popsize'])(parametrization=param, budget=conf.budget, num_workers=num_cores)

    elif conf.optim_method == 'TwoPointsDE' and conf.optim_params['popsize'] != 'default':
        # configured DE, twopoints crossover
        optim = ng.optimizers.DifferentialEvolution(crossover='twopoints', popsize=conf.optim_params['popsize'])(parametrization=param, budget=conf.budget, num_workers=num_cores)

    elif conf.optim_method == 'PSO' and conf.optim_params['popsize'] != 'default':
        # configured PSO
        optim = ng.optimizers.ConfiguredPSO(popsize=conf.optim_params['popsize'])(parametrization=param, budget=conf.budget, num_workers=num_cores)
    if conf.optim_method == 'Chaining':
        # chaining optimizers
        chain_optims = []
        for name in conf.optim_params['chain_optims']:
            chain_optims.append(eval('ng.optimizers.' + name))
        chain = ng.optimizers.Chaining(chain_optims, conf.optim_params['chain_budget'])
        # chain = ng.optimizers.Chaining([ng.optimizers.PortfolioDiscreteOnePlusOne, ng.optimizers.CMA], ['third'])
        optim = chain(parametrization=param, budget=conf.budget, num_workers=num_cores)
    else:
        optim = ng.optimizers.registry[conf.optim_method](parametrization=param, budget=conf.budget, num_workers=num_cores)

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
            optim.tell(ind, 1640.0 - np.sum(score))

        # collect all
        all_scores.extend(results)
        all_individuals.extend(individuals)

        if prog % 5 == 0:
            score_all_np = np.asarray(all_scores)
            print("Current High Score: " + str(np.max(np.sum(score_all_np, axis=1))))
            print("At index: " + str(str(np.argmax(np.sum(score_all_np, axis=1)))))
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

    # storing as npz, while running as sacred experiment, the directory quad_fdm_runs should've been created
    # column 0 is eval 1 score, column 1-3 is eval 3-5 score
    score_all_np = np.asarray(all_scores)
    selected_vectors = []
    for indi in all_individuals:
        current_vec = []
        d = indi.args[0]
        for key in d:
            if isinstance(d[key], np.ndarray):
                current_vec.extend(list(d[key]))
            else:
                current_vec.append(d[key])
        selected_vectors.append(current_vec)
    
    vector_all_np = np.asarray(selected_vectors)
    np.savez_compressed(filename, scores=score_all_np, vectors=vector_all_np)
    _run.add_artifact(filename)
    optim.dump(filename_optim)
    _run.add_artifact(filename_optim)
