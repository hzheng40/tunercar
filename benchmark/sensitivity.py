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
from tqdm import tqdm

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
parser.add_argument('--noise_scale', type=float, default=0.1)
parser.add_argument('--conf', type=str, required=True)
parser.add_argument('--rerun', type=str2bool, default=True)
parser.add_argument('--saved_npz', type=str, nargs='+')
parser.add_argument('--portion', type=str, default='all')
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

    if not conf.normalize_param:
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
    noise = rng.normal(loc=0.0, scale=args.noise_scale, size=(args.num_jostle, rec_np.shape[0]))
    rec_mat = np.tile(rec_np, (args.num_jostle, 1))
    
    # move rec around depending on portion arg
    if args.portion == 'all':
        jostled_rec = rec_mat + noise
    elif args.portion == 'physical':
        jostled_rec = rec_mat[:, conf.num_ctrl:conf.num_ctrl + 2] + noise[:, conf.num_ctrl:conf.num_ctrl + 2]
    elif args.portion == 'path':
        jostled_rec = rec_mat[:, :conf.num_ctrl] + noise[:, :conf.num_ctrl]
    elif args.portion == 'control':
        jostled_rec = rec_mat[:, conf.num_ctrl + 2:] + noise[:, conf.num_ctrl + 2:]
    else:
        # default to all
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

    # book keeping
    all_score = []
    all_collision = []
    # all_traj_x = []
    # all_traj_y = []
    # all_traj_th = []

    # loop across all noises
    for i in tqdm(range(args.num_jostle)):
        # book keeping
        # traj_x = []
        # traj_y = []
        # traj_th = []
        # recover actual parameters
        work = recover_params(jostled_rec[i, :], conf)
        # perturb waypoints
        planner.reset_waypoints()
        sub_ind = subsample(planner.waypoints.shape[0], conf.num_ctrl)
        pert_waypoints = perturb(work['perturb'], planner.waypoints[sub_ind, :], conf.track_width)
        vel = interpolate_velocity(work['vel_min'], work['vel_max'], pert_waypoints[:, 4], method='sigmoid')
        new_waypoints = np.hstack((pert_waypoints, vel[:, None]))
        planner.waypoints = new_waypoints
        # start pose
        start_x = new_waypoints[conf.start_ind, conf.wpt_xind]
        start_y = new_waypoints[conf.start_ind, conf.wpt_yind]
        start_th = new_waypoints[conf.start_ind, conf.wpt_thind]
        # new params update
        new_params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': work['lf'], 'lr': wb - work['lf'], 'h': 0.074, 'm': work['mass'], 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
        env.update_params(new_params)
        # reset env
        obs, step_reward, done, info = env.reset(np.array([[start_x, start_y, start_th]]))
        laptime = 0.0
        # sim loop
        while not done:
            # actuation from planner
            if conf.controller == 'stanley':
                speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], work['kpath'])
            elif conf.controller == 'lqr':
                speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], env.timestep, work['q1'], work['q2'], work['q3'], work['q4'], work['r'], conf.iteration, conf.eps)
            else:
                # default to pure pursuit
                speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'])
            # step with action
            obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
            laptime += step_reward
            # book keeping
            # traj_x.append(obs['poses_x'][0])
            # traj_y.append(obs['poses_y'][0])
            # traj_th.append(obs['poses_theta'][0])

        # book keeping
        all_collision.append(obs['collisions'][0])
        laptime = np.around(laptime, 2)
        all_score.append(laptime)
        # all_traj_x.append(traj_x)
        # all_traj_y.append(traj_y)
        # all_traj_th.append(traj_th)
    fname = args.exp_name + '_scale' + str(args.noise_scale) + '_' + str(args.num_jostle) + '_points_around.npz' if args.portion == 'all' else 'portion' + args.portion + '_' + args.exp_name + '_scale' + str(args.noise_scale) + '_' + str(args.num_jostle) + '_points_around.npz'
    np.savez_compressed(fname,
                        score=np.array(all_score),
                        collision=np.array(all_collision),
                        scale=np.array(args.noise_scale),
                        noisy_rec=jostled_rec)

def visualize(npz_list):
    scales = []
    nocrash_per = []
    df = pd.DataFrame(columns=['Noise Std. dev.', 'Success rate'])
    for npz in npz_list:
        # load
        data = np.load(npz)
        score = data['score']
        collision = data['collision']
        df = df.append({'Noise Std. dev.': data['scale'], 'Success rate': 1 - np.sum(collision)/collision.shape[0]}, ignore_index=True)
        scales.append(data['scale'])
        nocrash_per.append(1 - np.sum(collision)/collision.shape[0])

    # visualize
    print(df)
    sns.set_style('white')
    sns.set_style('ticks')
    sns.set_context('poster')
    # sns.lineplot(data=df, x='Noise Std. dev.', y='Success rate', palette='Paired')
    # sns.lineplot(data=df, markers=True)
    plt.plot(scales, nocrash_per, marker='^', markersize=30)
    plt.xlabel('Noise Standard Deviation')
    plt.ylabel('Success Rate')
    plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    if args.rerun:
        main()
    else:
        if args.saved_npz is None:
            raise Exception('Please specify saved npz path.')
        visualize(args.saved_npz)