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
    num_cores = mp.cpu_count()

    # setting up optimizer
    param = ng.p.Dict(
        mass=ng.p.Scalar(lower=conf.mass_min, upper=conf.mass_max),
        lf=ng.p.Scalar(lower=conf.lf_min, upper=conf.lf_max),
        wlad=ng.p.Scalar(lower=conf.wlad_min, upper=conf.wlad_max),
        tdiv=ng.p.Scalar(lower=conf.tdivisor_min, upper=conf.tdivisor_max),
        vgain=ng.p.Scalar(lower=conf.vgain_min, upper=conf.vgain_max))
    optim = ng.optimizer.registry[conf.optim_method](parameterization=param, budget=conf.budget)

    popsize = optim._popsize

    # setting up workers
    workers = [GymWorker.remote() for _ in range(num_cores)]

    # all scores
    all_scores = []
    all_individuals = []

    # work distribution loop
    for _ in tqdm(range(budget//num_cores)):
        individuals = [optim.ask() for _ in range(num_cores)]
        results = []
        
        # distribute
        for ind, worker in zip(individuals, workers):
            worker.run_sim.remote(ind.args[0])

        # collect
        future_results = [worker.collect().remote() for worker in workers]
        results = ray.get(future_results)

        # update optimization
        for ind, score in zip(individuals, results):
            optim.tell(ind, score)

        # collect all
        all_scores.extend(scores)
        all_individuals.extend(individuals)

        # logging
        if len(all_scores) >= pop_size:
            current_gen_scores = all_scores[:pop_size]

            # remove processed?
            all_scores = all_scores[pop_size:]

            # mean score