import numpy as np
import yaml
import gym
import os
import argparse
import multiprocessing as mp
import random
import sys
sys.path.append('../../python')
from utils import make_spline, make_waypoints, get_speed
# import task_worker, zmq_worker
from run_opt import simulation_loop
import pango_visualizer

def worker_func(config, score_q, perturb_chunk, mu, h_cg, l_r, cs_f, cs_r, I_z, mass, speed_gain, track_lad, grid_lad, best_waypoints, viz):
    if viz:
        pango_viz = pango_visualizer.PangoViz(config['zoom'], '../../maps/'+config['map_prefix']+config['map_img_ext'], '../../maps/'+config['map_name'], best_waypoints, show_laser=True)
    else:
        pango_viz = None
    racecar_env = gym.make('f110_gym:f110-v0')
    racecar_env.init_map('../../maps/' + config['map_name'], config['map_img_ext'], False, False)
    racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, '../../build/')
    for perturb in perturb_chunk:
        score = simulation_loop(perturb, racecar_env, best_waypoints, '../../maps/levine.yaml', '.pgm', '../../python/mpc', speed_gain, track_lad, grid_lad, pango_viz)
        print(score)
        score_q.put(score)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_npz_path', type=str, required=True)
    parser.add_argument('--save_stat', type=int, default=1)
    parser.add_argument('--num_eval', type=int, default=100)
    parser.add_argument('--stat_npz_path', type=str)
    parser.add_argument('--eval_config_path', type=str, default='./eval_config.yaml')
    parser.add_argument('--viz', type=int, default=1)
    args = parser.parse_args()

    with open(args.eval_config_path, 'r') as yaml_stream:
        try:
            CONFIG = yaml.safe_load(yaml_stream)
        except yaml.YAMLError as ex:
            print(ex)

    eval_npz = np.load(args.result_npz_path)
    best_idx = np.argmin(eval_npz['time_list_hist'][-1,:])
    best_params = eval_npz['params_list_history'][-1, :, best_idx]
    best_points = eval_npz['points_list_history'][-1, :, :, best_idx]

    best_spline, best_theta, best_curvature = make_spline(best_points, CONFIG)
    # best_spline[:, 1]*=-1.
    best_speed = get_speed(best_spline, best_params[0], best_params[1]/0.3302)
    best_waypoints = make_waypoints(best_spline, best_theta, best_speed)
    best_param_eval = {'mass': best_params[0], 'l_r': 0.3302-best_params[1], 'wpt_lad':best_params[2], 'track_lad':best_params[3], 'speed_gain': best_params[4], 'I_z': 0.04712, 'mu':0.523, 'h_cg':0.074, 'cs_f':4.718, 'cs_r':5.4562}
    mu = 0.523
    h_cg = 0.074
    l_r = 0.3302-best_params[1]
    cs_f = 4.718
    cs_r = 5.4562
    I_z = 0.04712
    mass = best_params[0]
    track_lad = best_params[3]
    grid_lad = best_params[2]
    speed_gain = best_params[4]

    # seeding
    np.random.seed(12345)
    random.seed(12345)
    # random perturbs to start
    perturb = 0.01*np.random.rand(args.num_eval, 3)
    
    num_workers = 1
    jobs = []
    score_q = mp.Queue()
    scores = []
    chunk_size = int(args.num_eval/num_workers)

    worker_func(CONFIG, score_q, perturb, mu, h_cg, l_r, cs_f, cs_r, I_z, mass, speed_gain, track_lad, grid_lad, best_waypoints, args.viz)

    # for i in range(num_workers):
    #     perturb_chunk = perturb[i*chunk_size:(i+1)*chunk_size, :]
    #     p = mp.Process(target=worker_func, args=(CONFIG, score_q, perturb_chunk, mu, h_cg, l_r, cs_f, cs_r, I_z, mass))
    #     jobs.append(p)
    
    # while True:
    #     while not score_q.empty():
    #         one_out = score_q.get_nowait()
    #         scores.append(one_out)
    #     all_dead = True
    #     for proc in jobs:
    #         if proc.is_alive():
    #             all_dead = False
    #             break
    #     if all_dead:
    #         break
    #     print('num scores', len(scores))
    # assert len(scores) == args.num_eval

    # scores = np.array(scores)
    # valid_scores = scores[scores<100]
    # avg_score = np.mean(valid_scores)
    # std_dev = np.std(valid_scores)

    # if args.save_stat:
    #     if args.stat_npz_path is None:
    #         print('Need npz path if save stat')
    #         sys.exit()
    #     np.savez_compressed(args.stat_npz_path, scores=valid_scores)