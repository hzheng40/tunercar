import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nevergrad as ng
import argparse
from argparse import Namespace
import yaml
from reconstruction_utils import recover_params
import sys
import os
import gym

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
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--npz_dir', type=str, default='../es/tunercar_runs/npzs/')
parser.add_argument('--pkl_dir', type=str, default='../es/tunercar_runs/optims_pkl/')
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--num_jostle', type=int, default=100)
parser.add_argument('--conf', type=str, required=True)
args = parser.parse_args()

def main():
    print('Sensitivity Analysis on: ' + args.exp_name)
    print('With configuration file: ' + args.conf)

    # seeding
    np.random.seed(args.seed)

    # grabbing config
    with open('../es/configs/' + args.conf) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    if not conf.nomalize_param:
        raise NotImplementedError('Unnormalized analysis not supported yet.')

    # new rng
    rng = np.random.default_rng(args.seed)

    # load
    optim = ng.optimizers.base.Optimizer.load(args.pkl_dir + args.exp_name + '_optim.pkl')

    # recommendation
    rec = optim.provide_recommendation()
    rec_np = rec.value

    # jostle around recommendation
    # gaussian noise, stddev 0.1 centered at 0
    noise = rng.normal(loc=0.0, scale=0.1, size=(args.num_jostle, rec_np.shape[0]))
    rec_mat = np.tile(rec_np, (args.num_jostle, 1))
    jostled_rec = rec_mat + noise

    assert jostled_rec.shape[0] == args.num_jostle
    assert jostled_rec.shape[1] == rec_np.shape[0]

    # setup sim
    # default params
    default_params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
    wb = default_params['lf'] + default_params['lr']
    # controllers
    if conf.controller == 'stanley':
        planner = StanleyPlanner(conf, wb)
    elif conf.controller == 'lqr':
        planner = LQRPlanner(conf, wb)
    else:
        # init pure pursuit planner with raceline
        planner = PurePursuitPlanner(conf, wb)
    # env
    env = gym.make('f110_gym:f110-v0', seed=conf.seed, map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    _, _, _, _ = env.reset(np.array([[0., 0., 0.]]))

    # loop across all noises
    for i in range(args.num_jostle):
        # recover actual parameters
        work = recover_params(jostled_rec[i, :], conf)

if __name__ == '__main__':
    main()