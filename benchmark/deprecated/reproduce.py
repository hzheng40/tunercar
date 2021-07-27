import argparse
from argparse import Namespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nevergrad as ng
import gym
import yaml
import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir + '/../es')
from planners import PurePursuitPlanner, StanleyPlanner, LQRPlanner
from utils import perturb, interpolate_velocity, subsample

# args
def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True, nargs='+', help='Name of the experiment without the npz extension.')
parser.add_argument('--npz_dir', type=str, default='../es/tunercar_runs/npzs/')
parser.add_argument('--pkl_dir', type=str, default='../es/tunercar_runs/optims_pkl/')
parser.add_argument('--run_ind', type=int, required=True, nargs='+', 'The indices of experiments to reproduce, can be multiple.')
parser.add_argument('--conf_file', type=str, required=True, nargs='+', help='Path of the config yaml file, needs to match the experiments in order.')
parser.add_argument('--use_best', type=str2bool, default=False, help='Use best laptime indices in each npz, if set to true  the run_ind arguments will be ignored.')
args = parser.parse_args()

def rerun(npz, indices, conf):
    # seeding and constant
    np.random.seed(conf.seed)
    wb = 0.17145 + 0.15875

    # translate into work dict
    work = {}

    # gym env
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)

    # new param
    new_params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': work['lf'], 'lr': wb - work['lf'], 'h': 0.074, 'm': work['mass'], 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

    # controller
    if conf.controller == 'lqr':
        planner = LQRPlanner(conf, wb)
    elif conf.controller == 'stanley':
        planner = StanleyPlanner(conf, wb)
    else:
        planner = PurePursuitPlanner(conf, wb)

    # waypoints and perturb
    waypoints = planner.waypoints
    pert_vec = work['perturb']
    sub_ind = subsample(planner.waypoints.shape[0], conf.num_ctrl)
    pert_waypoints = perturb(pert_vec, planner.waypoints[sub_ind, :], conf.track_width)

if __name__ == '__main__':
    npz_name_list = args.exp_name
    indices = args.run_ind
    conf_name = args.conf_file
    use_best = args.use_best

    with open('../es/cofigs/' + conf_name + '.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    npz_list = []
    for npz_name in npz_name_list:
        npz_list.append(np.load(args.npz_dir + npz_name + '.npz'))

    rerun(npz_list, indices, conf)