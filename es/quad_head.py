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

    num_cores = mp.cpu_count()

    # setting up parameter space
    # 5 parameters:
    #   arm length: float
    #   num batteries: int
    #   battery voltage: choice
    #   battery capacity: float
    #   battery mass: float

    param = ng.p.Dict(
        arm_length=ng.p.Scalar(lower=conf.arm_length_min, upper=conf.arm_length_max),
        num_batt=ng.p.Choice(conf.num_batteries),
        batt_v=ng.p.Choice(conf.battery_voltage),
        batt_cap=ng.p.Scalar(lower=conf.battery_capacity_min, upper=conf.battery_capacity_max),
        batt_m=ng.p.Scalar(lower=conf.battery_mass_min, upper=conf.battery_mass_max))

    # setting up optimizer with hyperparams
    # TODO: currently only checks if the popsize is default
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

    # storing as npz, while running as sacred experiment, the directory tunercar_runs should've been created
    score_all_np = np.asarray(all_scores)
    params_general_scalars_np = np.empty((4, score_all_np.shape[0]))
    num_controller_params = 5 if conf.controller=='lqr' else 1
    params_controller_np = np.empty((num_controller_params, score_all_np.shape[0]))
    params_perturb_np = np.empty((conf.num_ctrl, score_all_np.shape[0]))

    for i, indi in enumerate(all_individuals):
        if conf.normalize_param:
            params_np = indi.value
            params_perturb_np[:, i] = params_np[:conf.num_ctrl]
            params_general_scalars_np[:, i] = params_np[conf.num_ctrl:conf.num_ctrl + 4]
            if conf.controller == 'lqr':
                params_controller_np[:, i] = params_np[-6:-1]
            else:
                params_controller_np[:, i] = params_np[-1]
        else:
            # log common scalar parameters
            params_general_scalars_np[0, i] = indi['mass'].value
            params_general_scalars_np[1, i] = indi['lf'].value
            params_general_scalars_np[2, i] = indi['vel_min'].value
            params_general_scalars_np[3, i] = indi['vel_max'].value

            # log controller specific parameters
            if conf.controller == 'stanley':
                params_controller_np[0, i] = indi['kpath'].value
            elif conf.controller == 'lqr':
                params_controller_np[0, i] = indi['q1'].value
                params_controller_np[1, i] = indi['q2'].value
                params_controller_np[2, i] = indi['q3'].value
                params_controller_np[3, i] = indi['q4'].value
                params_controller_np[4, i] = indi['r'].value
            else:
                params_controller_np[0, i] = indi['tlad'].value

            # log perturb vector parameters
            params_perturb_np[:, i] = indi['perturb'].value

        # log vector parameters, perturb
    np.savez_compressed(filename, lap_times=score_all_np, general_params=params_general_scalars_np, controller_params=params_controller_np, perturb_params=params_perturb_np, controller=np.array([conf.controller]))
    _run.add_artifact(filename)
    optim.dump(filename_optim)
    _run.add_artifact(filename_optim)