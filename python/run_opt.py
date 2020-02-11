import nevergrad as ng
import numpy as np
from future import ProcessPoolExecutor
import gym
from mpc import lattice_planner

from speed_opt import optimal_time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
parser.add_argument('--opt_method', type=str, required=True, help='cma, pso, ce, de, fastga')
parser.add_argument('--num_workers', type=int, default=1)
ARGS = parser.parse_args()








def get_time_sim(spline, car_params, racecar_env):
    speed = get_vel(spline, car_params['mass'], (0.3302-car_params['l_r'])/0.3302)
    trajectory = np.zeros((spline.shape[1], 3))
    trajectory[:, 0] = spline[0, :].T
    trajectory[:, 1] = spline[1, :].T
    trajectory[:, 2] = speed

    work = {'id': 'bigmattyboi', 'traj': trajectory.tolist(),'wpt_lad': car_params['wpt_lad'], 'track_lad': car_params['track_lad'], 'mu': car_params['mu'], 'h_cg': car_params['h_cg'], 'l_r': car_params['l_r'], 'cs_f': car_params['cs_f'], 'cs_r': car_params['cs_r'], 'I_z': car_params['I_z'], 'mass': car_params['mass'], 'speed_gain': car_params['speed_gain']}
    result = task_worker.eval(work, racecar_env)
    lap_time = result['lap_time']
    in_collision = result['collision']
    collision_angle = result['collision_angle']
    return lap_time, in_collision, collision_angle