import nevergrad as ng
import numpy as np
from tqdm import tqdm
import ray
import multiprocessing as mp
from argparse import Namespace

from arch_worker import ArchWorker

def run_arch_fdm(conf: Namespace, _run=None):
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

    # setting up parameter space, choice for meta optimization
    param = ng.p.Choice(np.arange(conf.num_choices), repetitions=conf.num_max_selections)

    if conf.optim_method == 'Chaining':
        # chaining optimizers
        chain_optims = []
        for name in conf.optim_params['chain_optims']:
            chain_optims.append(eval('ng.optimizers.' + name))
        chain = ng.optimizers.Chaining(chain_optims, conf.optim_params['chain_budget'])
        # chain = ng.optimizers.Chaining([ng.optimizers.PortfolioDiscreteOnePlusOne, ng.optimizers.CMA], ['third'])
        optim = chain(parametrization=param, budget=conf.budget, num_workers=num_cores)
    else:
        optim = ng.optimizers.registry[conf.optim_method](parametrization=param, budget=conf.meta_budget, num_workers=num_cores)

    # seeding
    optim.parametrization.random_state = np.random.RandomState(conf.seed)
    print('Optimizer: ', optim)

    # setting up workers
    workers = [ArchWorker.remote(conf, worker_id) for worker_id in range(num_cores)]

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

    # storing as npz, while running as sacred experiment, the directory quad_fdm_runs should've been created
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
