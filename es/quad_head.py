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

    # setting up parameter space
    # 5 parameters:
    #   arm length: float
    #   num batteries: choice
    #   battery voltage: choice
    #   battery capacity: float
    #   battery mass: float

    # TODO: easier way instead of writing everything out?
    param = ng.p.Dict(
        battery=ng.p.Choice(np.arange(conf.design_space['battery'], dtype=int)),
        esc1=ng.p.Choice(np.arange(conf.design_space['esc'][0], dtype=int)),
        esc2=ng.p.Choice(np.arange(conf.design_space['esc'][1], dtype=int)),
        esc3=ng.p.Choice(np.arange(conf.design_space['esc'][2], dtype=int)),
        esc4=ng.p.Choice(np.arange(conf.design_space['esc'][3], dtype=int)),
        arm1=ng.p.Choice(np.arange(conf.design_space['arm'][0], dtype=int)),
        arm2=ng.p.Choice(np.arange(conf.design_space['arm'][1], dtype=int)),
        arm3=ng.p.Choice(np.arange(conf.design_space['arm'][2], dtype=int)),
        arm4=ng.p.Choice(np.arange(conf.design_space['arm'][3], dtype=int)),
        prop1=ng.p.Choice(np.arange(conf.design_space['prop'][0], dtype=int)),
        prop2=ng.p.Choice(np.arange(conf.design_space['prop'][1], dtype=int)),
        prop3=ng.p.Choice(np.arange(conf.design_space['prop'][2], dtype=int)),
        prop4=ng.p.Choice(np.arange(conf.design_space['prop'][3], dtype=int)),
        motor1=ng.p.Choice(np.arange(conf.design_space['motor'][0], dtype=int)),
        motor2=ng.p.Choice(np.arange(conf.design_space['motor'][1], dtype=int)),
        motor3=ng.p.Choice(np.arange(conf.design_space['motor'][2], dtype=int)),
        motor4=ng.p.Choice(np.arange(conf.design_space['motor'][3], dtype=int)),
        support1=ng.p.Choice(np.arange(conf.design_space['support'][0], dtype=int)),
        support2=ng.p.Choice(np.arange(conf.design_space['support'][1], dtype=int)),
        support3=ng.p.Choice(np.arange(conf.design_space['support'][2], dtype=int)),
        support4=ng.p.Choice(np.arange(conf.design_space['support'][3], dtype=int)),
        arm_length1=ng.p.Scalar(lower=100.0, upper=conf.design_space['arm_length'][0]),
        arm_length2=ng.p.Scalar(lower=100.0, upper=conf.design_space['arm_length'][1]),
        arm_length3=ng.p.Scalar(lower=100.0, upper=conf.design_space['arm_length'][2]),
        arm_length4=ng.p.Scalar(lower=100.0, upper=conf.design_space['arm_length'][3]),
        support_length1=ng.p.Scalar(lower=0.0, upper=conf.design_space['support_length'][0]),
        support_length2=ng.p.Scalar(lower=0.0, upper=conf.design_space['support_length'][1]),
        support_length3=ng.p.Scalar(lower=0.0, upper=conf.design_space['support_length'][2]),
        support_length4=ng.p.Scalar(lower=0.0, upper=conf.design_space['support_length'][3]),
        Q_position=ng.p.Scalar(lower=0.0, upper=conf.design_space['Q_position']),
        Q_velocity=ng.p.Scalar(lower=0.0, upper=conf.design_space['Q_velocity']),
        Q_angular_velocity=ng.p.Scalar(lower=0.0, upper=conf.design_space['Q_angular_velocity']),
        Q_angles=ng.p.Scalar(lower=0.0, upper=conf.design_space['Q_angles']),
        control_R=ng.p.Scalar(lower=0.0, upper=conf.design_space['control_R']))

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

    # setting number of objectives
    # optim.num_objectives = 2

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
    for _ in tqdm(range(conf.budget // num_cores)):
        individuals = [optim.ask() for _ in range(num_cores)]
        results = []

        # distribute
        for ind, worker in zip(individuals, workers):
            work = ind.args[0]
            work['eval_id'] = eval_id
            worker.run_sim.remote(work)
            eval_id += 1

        # collect
        future_results = [worker.collect.remote() for worker in workers]
        results = ray.get(future_results)

        # update optimization
        # negate since we want to maximize scores
        for ind, score in zip(individuals, results):
            optim.tell(ind, -np.sum(score))

        # collect all
        all_scores.extend(results)
        all_individuals.extend(individuals)

    # storing as npz, while running as sacred experiment, the directory quad_fdm_runs should've been created
    # column 0 is eval 1 score, column 1-3 is eval 3-5 score
    score_all_np = np.asarray(all_scores)
    selected_vectors = [[indi['battery'],
                         indi['esc1'],
                         indi['esc2'],
                         indi['esc3'],
                         indi['esc4'],
                         indi['arm1'],
                         indi['arm2'],
                         indi['arm3'],
                         indi['arm4'],
                         indi['prop1'],
                         indi['prop2'],
                         indi['prop3'],
                         indi['prop4'],
                         indi['motor1'],
                         indi['motor2'],
                         indi['motor3'],
                         indi['motor4'],
                         indi['support1'],
                         indi['support2'],
                         indi['support3'],
                         indi['support4'],
                         indi['arm_length1'],
                         indi['arm_length2'],
                         indi['arm_length3'],
                         indi['arm_length4'],
                         indi['support_length1'],
                         indi['support_length2'],
                         indi['support_length3'],
                         indi['support_length4'],
                         0.0, 0.0, 0.0, 0.0, 0.0] for indi in all_individuals]
    vector_all_np = np.asarray(selected_vectors)
    eval_id = [indi['eval_id'] for indi in all_individuals]
    eval_id_np = np.array(eval_id)
    np.savez_compressed(filename, scores=score_all_np, vectors=vector_all_np, eval_id=eval_id_np)
    _run.add_artifact(filename)
    optim.dump(filename_optim)
    _run.add_artifact(filename_optim)


def run_quad_fdm_simple(conf: Namespace, _run=None):
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
    #   num batteries: choice
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

    # setting number of objectives
    optim.num_objectives = 2

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
    for _ in tqdm(range(conf.budget//num_cores)):
        individuals = [optim.ask() for _ in range(num_cores)]
        results = []

        # distribute
        for ind, worker in zip(individuals, workers):
            work = ind.args[0]
            work['eval_id'] = eval_id
            worker.run_sim.remote(work)
            eval_id += 1

        # collect
        future_results = [worker.collect.remote() for worker in workers]
        results = ray.get(future_results)

        # update optimization
        for ind, score in zip(individuals, results):
            optim.tell(ind, score)

        # collect all
        all_scores.extend(results)
        all_individuals.extend(individuals)

    # storing as npz, while running as sacred experiment, the directory quad_fdm_runs should've been created
    # column 0 is max distance, column 1 is max hover time, negate to return actual seconds
    score_all_np = -np.asarray(all_scores)
    arm_length_np = np.asarray([indi.args[0]['arm_length'] for indi in all_individuals])
    num_batt_np = np.asarray([indi.args[0]['num_batt'] for indi in all_individuals])
    batt_v_np = np.asarray([indi.args[0]['batt_v'] for indi in all_individuals])
    batt_cap_np = np.asarray([indi.args[0]['batt_cap'] for indi in all_individuals])
    batt_m_np = np.asarray([indi.args[0]['batt_m'] for indi in all_individuals])

    np.savez_compressed(filename, scores=score_all_np, arm_length=arm_length_np, num_batt=num_batt_np, batt_v=batt_v_np, batt_cap=batt_cap_np, batt_m=batt_m_np)
    _run.add_artifact(filename)
    optim.dump(filename_optim)
    _run.add_artifact(filename_optim)