import nevergrad as ng
import numpy as np
from concurrent import futures
import gym
from mpc import lattice_planner
from utils import *
from speed_opt import optimal_time
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True, help='path to config yaml file')
parser.add_argument('--opt_method', type=str, required=True, help='cma, pso, ce, de, fastga')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for optimization')
parser.add_argument('--seed', type=int, default=12345, help='random seed for numpy and random')
ARGS = parser.parse_args()

# seeding
np.random.seed(ARGS.seed)
random.seed(ARGS.seed)

# read config
with open(ARGS.config_path, 'r') as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as ex:
        print(ex)

# TODO:
#   1. read in map, get metadata, read centerline csv, make the boxes
boxes, num_boxes = utils.get_boxes(CONFIG)
#   2. create parameterization with array of size (num_pts + num_params), set bounds of the variable
instrum = ng.p.Instrumentation(ng.p.Array())


#   2. 



def get_vel(spline, mass, Wf):
    spline = spline[:, ::5]

    # mass = 3.51 # mass
    # Wf = 0.515 # weight percentage of front
    muf = 0.523
    gravity = 9.81

    path = optimal_time.define_path(spline[0, :], spline[1, :])
    params = optimal_time.define_params(mass, Wf, muf, gravity)
    B, A, U, v, topt = optimal_time.optimize(path=path, params=params)
    v = v.repeat(5)
    return v


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