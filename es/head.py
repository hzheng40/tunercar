import nevergrad as ng
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import ray
import multiprocessing as mp
from argparse import Namespace

from worker import GymWorker

def run_tunercar(conf: Namespace, _run=None):
    # set up stuff and seeding
    np.random.seed(conf.seed)
    import os
    if not os.path.exists('tunercar_runs/npzs'):
        os.makedirs('tunercar_runs/npzs')
    if not os.path.exists('tunercar_runs/optims_pkl'):
        os.makedirs('tunercar_runs/optims_pkl')
    filename = 'tunercar_runs/npzs/' + conf.run_name + '_' + conf.optim_method + '_budget' + str(conf.budget) + '.npz'
    filename_optim = 'tunercar_runs/optims_pkl/' + conf.run_name + '_' + conf.optim_method + '_budget' + str(conf.budget) + '_optim.pkl'

    num_cores = mp.cpu_count()

    # setting up optimizer
    param = ng.p.Dict(
        mass=ng.p.Scalar(lower=conf.mass_min, upper=conf.mass_max),
        lf=ng.p.Scalar(lower=conf.lf_min, upper=conf.lf_max),
        tlad=ng.p.Scalar(lower=conf.tlad_min, upper=conf.tlad_max),
        vgain=ng.p.Scalar(lower=conf.vgain_min, upper=conf.vgain_max))
    optim = ng.optimizers.registry[conf.optim_method](parametrization=param, budget=conf.budget)
    optim._popsize = conf.popsize

    if conf.optim_method == 'CMA':
        popsize = optim._popsize

    # setting up workers
    workers = [GymWorker.remote(conf) for _ in range(num_cores)]

    # all scores
    all_scores = []
    score_log = []
    all_individuals = []

    # work distribution loop
    for _ in tqdm(range(conf.budget//num_cores)):
        individuals = [optim.ask() for _ in range(num_cores)]
        results = []
        
        # distribute
        for ind, worker in zip(individuals, workers):
            worker.run_sim.remote(ind.args[0])

        # collect
        future_results = [worker.collect.remote() for worker in workers]
        results = ray.get(future_results)

        # update optimization
        for ind, score in zip(individuals, results):
            optim.tell(ind, score)

        # collect all
        all_scores.extend(results)
        all_individuals.extend(individuals)
        # score_log.extend(results)
        # # logging
        # if len(score_log) >= popsize and conf.optim_method == 'CMA':
        #     current_gen_scores = score_log[:popsize]
        #     # remove processed
        #     score_log = score_log[popsize:]
        #     print('Current gen lap time:', score_log)


    # storing as npz, while running as sacred experiment, the directory tunercar_runs should've been created
    score_all_np = np.asarray(all_scores)
    params_all_np = np.empty((len(param.args[0].keys()), score_all_np.shape[0]))
    for i, indi in enumerate(all_individuals):
        params_all_np[0, i] = indi['mass'].value
        params_all_np[1, i] = indi['lf'].value
        params_all_np[2, i] = indi['tlad'].value
        params_all_np[3, i] = indi['vgain'].value
    perf_nums_np = conf.perf_num * np.ones(score_all_np.shape[0])
    np.savez_compressed(filename, lap_times=score_all_np, params=params_all_np, perf_nums=perf_nums_np)
    _run.add_artifact(filename)
    optim.dump(filename_optim)
    _run.add_artifact(filename_optim)