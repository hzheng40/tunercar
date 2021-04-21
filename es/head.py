import nevergrad as ng
import numpy as np
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

    # setting up parameter space, based on controller
    # TODO: another thing to try is to only search a vector that's normalized, all elements between 0 and 1, then figure out the actual number based on bounds later
    if conf.controller == 'stanley':
        if conf.normalize_param:
            # one vector with elements all between 0 and 1 for min and max
            # order: perturb, mass, lf, vel_min, vel_max, kpath
            num_params = 4 + conf.num_ctrl + 1
            init_arr = 0.5 * np.ones((num_params, ))
            param = ng.p.Array(init=init_arr).set_bounds(0, 1)
        else:
            param = ng.p.Dict(
                mass=ng.p.Scalar(lower=conf.mass_min, upper=conf.mass_max),
                lf=ng.p.Scalar(lower=conf.lf_min, upper=conf.lf_max),
                kpath=ng.p.Scalar(lower=conf.kpath_min, upper=conf.kpath_max),
                perturb=ng.p.Array(init=np.zeros(conf.num_ctrl), mutable_sigma=True).set_bounds(conf.left_bound, conf.right_bound),
                vel_min=ng.p.Scalar(lower=conf.v_lower_min, upper=conf.v_lower_max),
                vel_max=ng.p.Scalar(lower=conf.v_upper_min, upper=conf.v_upper_max))
    elif conf.controller == 'lqr':
        if conf.normalize_param:
            # one vector with elements all between 0 and 1 for min and max
            # order: perturb, mass, lf, vel_min, vel_max, q1, q2, q3, q4, r
            num_params = 4 + conf.num_ctrl + 5
            init_arr = 0.5 * np.ones((num_params, ))
            param = ng.p.Array(init=init_arr).set_bounds(0, 1)
        else:
            param = ng.p.Dict(
                mass=ng.p.Scalar(lower=conf.mass_min, upper=conf.mass_max),
                lf=ng.p.Scalar(lower=conf.lf_min, upper=conf.lf_max),
                q1=ng.p.Scalar(lower=conf.q1_min, upper=conf.q1_max),
                q2=ng.p.Scalar(lower=conf.q2_min, upper=conf.q2_max),
                q3=ng.p.Scalar(lower=conf.q3_min, upper=conf.q3_max),
                q4=ng.p.Scalar(lower=conf.q4_min, upper=conf.q4_max),
                r=ng.p.Scalar(lower=conf.r_min, upper=conf.r_max),
                perturb=ng.p.Array(init=np.zeros(conf.num_ctrl), mutable_sigma=True).set_bounds(conf.left_bound, conf.right_bound),
                vel_min=ng.p.Scalar(lower=conf.v_lower_min, upper=conf.v_lower_max),
                vel_max=ng.p.Scalar(lower=conf.v_upper_min, upper=conf.v_upper_max))
    else:
        # defaults to pure pursuit
        if conf.normalize_param:
            # one vector with elements all between 0 and 1 for min and max
            # order: perturb, mass, lf, vel_min, vel_max, wpt_lad
            num_params = 4 + conf.num_ctrl + 1
            init_arr = 0.5 * np.ones((num_params, ))
            param = ng.p.Array(init=init_arr).set_bounds(0, 1)
        else:
            param = ng.p.Dict(
                mass=ng.p.Scalar(lower=conf.mass_min, upper=conf.mass_max),
                lf=ng.p.Scalar(lower=conf.lf_min, upper=conf.lf_max),
                tlad=ng.p.Scalar(lower=conf.tlad_min, upper=conf.tlad_max),
                perturb=ng.p.Array(init=np.zeros(conf.num_ctrl), mutable_sigma=True).set_bounds(conf.left_bound, conf.right_bound),
                vel_min=ng.p.Scalar(lower=conf.v_lower_min, upper=conf.v_lower_max),
                vel_max=ng.p.Scalar(lower=conf.v_upper_min, upper=conf.v_upper_max))

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

    else:
        optim = ng.optimizers.registry[conf.optim_method](parametrization=param, budget=conf.budget, num_workers=num_cores)

    # seeding
    optim.parametrization.random_state = np.random.RandomState(conf.seed)
    print('Optimizer: ', optim)

    # setting up workers
    workers = [GymWorker.remote(conf, worker_id) for worker_id in range(num_cores)]

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