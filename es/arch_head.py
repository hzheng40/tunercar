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
    if not os.path.exists('iccps_runs/npzs'):
        os.makedirs('iccps_runs/npzs')
    if not os.path.exists('iccps_runs/optims_pkl'):
        os.makedirs('iccps_runs/optims_pkl')
    filename = 'iccps_runs/npzs/' + conf.run_name + '_' + conf.optim_method + '_budget' + str(conf.budget) + '.npz'
    filename_optim = 'iccps_runs/optims_pkl/' + conf.run_name + '_' + conf.optim_method + '_budget' + str(conf.budget) + '_optim.pkl'
    if not os.path.exists(conf.base_folder):
        os.makedirs(conf.base_folder)

    num_cores = conf.num_meta_workers

    # setting up parameter space, choice for meta optimization
    param = ng.p.Dict(
        base_node=ng.p.Choice(np.arange(conf.num_base_choices)),
        low_selections=ng.p.Choice(np.arange(conf.num_low_choices), repetitions=conf.num_max_selections),
        high_selections=ng.p.Choice(np.arange(conf.num_high_choices), repetitions=conf.num_max_selections)
        )

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
    all_vectors = []

    # work distribution loop
    eval_id = 0
    for prog in tqdm(range(conf.budget // num_cores)):
        individuals = [optim.ask() for _ in range(num_cores)]
        results = []

        # distribute
        for ind, worker in zip(individuals, workers):
            work = ind.args[0]
            worker.run_sim.remote(work, eval_id)
            eval_id += 1

        # collect
        # future_vectors = []
        future_results = []
        for worker in workers:
            score = worker.collect.remote()
            # future_vectors.append(vec)
            future_results.append(score)
        # future_results = [worker.collect.remote() for worker in workers]
        results = ray.get(future_results)
        # vectors = ray.get(future_vectors)

        # update optimization, objective value is trim score
        for ind, score in zip(individuals, results):
            optim.tell(ind, np.sum(score))

        # collect all
        all_scores.extend(results)
        all_individuals.extend([[ind.args[0]['base_node'], *(ind.args[0]['low_selections']), *(ind.args[0]['high_selections'])] for ind in individuals])
        # all_vectors.extend(vectors)

        if prog % 5 == 0:
            score_all_np = np.asarray(all_scores)
            print('Current Trim Only Best Score: ' + str(np.min(np.sum(score_all_np, axis=1))))
            print("At index: " + str(str(np.argmin(np.sum(score_all_np, axis=1)))))
            # vector_all_np = -1 * np.ones((len(all_vectors), len(max(all_vectors, key = lambda x: len(x)))), dtype=int)
            # for i, j in enumerate(all_vectors):
            #     vector_all_np[i][0:len(j)] = j
            # vector_all_np = vector_all_np.astype(int)
            selection_all_np = np.array(all_individuals).astype(int)

            # np.savez_compressed(filename, scores=score_all_np, vectors=vector_all_np, selections=selection_all_np)
            np.savez_compressed(filename, scores=score_all_np, selections=selection_all_np)
            # _run.add_artifact(filename)
            optim.dump(filename_optim)
            # _run.add_artifact(filename_optim)

    score_all_np = np.asarray(all_scores)
    print('Current Trim Only Best Score: ' + str(np.min(np.sum(score_all_np, axis=1))))
    print("At index: " + str(str(np.argmin(np.sum(score_all_np, axis=1)))))
    # vector_all_np = -1 * np.ones((len(all_vectors), len(max(all_vectors, key = lambda x: len(x)))), dtype=int)
    # for i, j in enumerate(all_vectors):
    #     vector_all_np[i][0:len(j)] = j
    # vector_all_np = vector_all_np.astype(int)
    selection_all_np = np.array(all_individuals).astype(int)

    # np.savez_compressed(filename, scores=score_all_np, vectors=vector_all_np, selections=selection_all_np)
    np.savez_compressed(filename, scores=score_all_np, selections=selection_all_np)
    # _run.add_artifact(filename)
    optim.dump(filename_optim)
    # _run.add_artifact(filename_optim)
