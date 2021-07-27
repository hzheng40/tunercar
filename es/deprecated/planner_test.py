import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from planners import PurePursuitPlanner, StanleyPlanner, LQRPlanner
from utils import perturb, interpolate_velocity, subsample

np.random.seed(12345)
wb = 0.17145+0.15875
# work_np = np.array([3.89436991, 0.14868024, 2.78745207, 0.68066784])
work_np = np.array([ 3.78502832,  0.16085193,  1.32167479, 10.15579699])
work = {'mass': work_np[0], 'lf': work_np[1], 'vel_min': work_np[2], 'vel_max': work_np[3], 'tlad': 0.89968225}
with open('../es/configs/default.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)
print(conf.optim_params['popsize'])

env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
new_params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': work['lf'], 'lr': wb - work['lf'], 'h': 0.074, 'm': work['mass'], 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

# planner = StanleyPlanner(conf, wb)
planner = PurePursuitPlanner(conf, wb)
# planner = LQRPlanner(conf, wb)

waypoints = planner.waypoints
pert_vec = np.array([ 1.97560142e-01, -6.41282462e-02, -2.43020293e-01, -5.47139809e-01, 7.48741956e-01,  1.07811702e-01, -2.33666769e-01,  2.10451481e-01, 3.68013715e-01,  1.05102550e-02, -3.89631297e-01,  2.34700300e-02, 1.27007698e-01, -1.58935421e-01, -1.60792378e-01,  8.07350333e-01, 6.35899249e-01,  5.18949338e-02,  3.52344923e-01, -1.43301655e-01, 1.46843528e-01,  3.33649763e-01, -3.62758115e-01,  8.53656018e-02, 3.20816195e-01, -2.65058809e-01,  1.42044223e-01,  8.93768522e-02, 9.59712763e-02, -1.25626595e-01, -1.96302337e-01, -6.11180686e-01, 6.43611998e-01, -4.75233262e-01, -1.02171903e-02,  2.17185490e-01, 4.18462944e-02, -6.78229635e-02,  2.02075185e-01, -2.08411438e-01, 6.40227780e-01, -3.63453483e-01, -5.09091217e-01, -3.31760479e-01, 1.09105829e-01, -3.43035239e-01,  6.46929682e-01,  1.60850743e-01, 4.04741394e-02, -3.10782851e-02, -2.88009490e-01, -2.27732516e-01, 8.77537793e-01, -1.94211498e-01,  4.99660814e-01, -4.10329922e-01, 4.24992357e-01, -2.26735851e-01, -1.59029368e-01,  4.01855158e-02, 5.97183740e-01,  5.08797690e-01,  2.42761209e-01, -1.02621674e-01, 5.16884117e-02, -4.16160731e-01,  2.29517100e-01,  5.48830829e-01, 6.32206852e-01,  8.58798783e-01,  2.19443281e-01,  6.10045955e-01, 6.27219754e-02, -6.18875354e-01, -7.64076555e-01, -1.37395876e-01, 1.20395313e-01,  1.02077268e-01, -7.12480441e-01,  3.15707402e-02, 2.88815575e-02,  2.63209162e-01,  3.48179813e-01,  3.94671379e-02, 5.07245612e-01,  2.12093388e-01, -3.82514237e-01, -2.88002861e-01, 3.80675274e-01,  4.96519933e-01,  4.00804450e-01, -1.61011712e-01, 2.70681743e-01, -2.42058128e-01,  2.54429701e-01, -3.09554914e-01, 2.91022569e-02, -3.85532191e-01,  8.24495006e-03,  9.69544036e-02, 1.81556819e-01, -7.52294018e-02,  3.99812047e-01, -3.57721272e-02, 4.28587129e-01, -4.48237782e-03, -8.71589789e-02,  8.09324100e-02, 5.12581930e-01,  6.32100946e-01, -4.90182149e-01,  9.50132900e-02, 5.78887657e-01,  3.19523298e-01,  3.80865314e-01,  6.31824926e-01, 5.36039522e-01,  1.86821428e-01, -8.49206697e-02,  2.74607329e-01, 2.47748649e-01, -2.27321740e-01,  7.05215858e-01, -2.73906633e-01, 6.05438095e-02, -1.34321636e-01,  5.89223199e-01, -5.54957508e-02, 3.08202777e-04,  7.09190551e-02, -3.92321493e-01,  1.07717901e-01, 3.65605085e-01,  3.45205188e-01,  5.45494792e-02,  2.20929693e-01, 8.69752980e-02,  2.17889796e-02,  2.70243703e-01,  5.30641919e-01, 3.41472204e-02,  3.02831309e-02,  1.90615423e-01, -4.07204103e-01, 4.03961158e-01,  1.22609848e-02, -1.99363595e-01,  1.48223429e-01, 5.86274208e-01,  4.82590366e-01,  5.92371782e-01,  2.58486101e-01, 1.79031862e-01, -1.74688330e-01,  1.28938722e-01, -5.76548143e-01, 3.62196434e-01, -4.17102434e-01,  2.49969868e-02,  1.68364849e-01, 2.96452154e-01,  2.34204260e-01, -4.99770417e-01, -5.20059394e-01, 6.16306161e-01, -1.93649696e-01, -7.35444095e-01, -2.06057392e-01, 5.98730468e-01,  2.60576757e-01, -1.65898445e-01, -3.66272571e-01, 3.83257639e-01, -1.38817658e-01, -5.86642731e-01,  4.73987637e-01, 2.22388846e-01,  8.52537494e-01, -2.50314083e-01,  5.06075439e-01, 3.52093482e-01, -1.30163493e-01, -4.34863544e-01,  4.10650743e-01, 1.15046629e-02, -1.16631008e-01, -1.38075337e-01, -3.23159873e-01, 4.41325592e-01, -2.58510807e-01, -2.22061689e-01, -8.31117133e-01, 2.61086739e-03, -2.18091694e-01, -3.85455952e-01, -5.62686421e-01, 1.20279005e-01, -2.14109258e-01, -9.64041921e-02, -1.61467773e-01])
# assert pert_vec.shape[0] == conf.num_ctrl
sub_ind = subsample(planner.waypoints.shape[0], pert_vec.shape[0])
pert_waypoints = perturb(pert_vec, waypoints[sub_ind, :], conf.track_width)
vel = interpolate_velocity(work['vel_min'], work['vel_max'], pert_waypoints[:, 4])
new_waypoints = np.hstack((pert_waypoints, vel[:, None]))

import matplotlib.pyplot as plt
plt.scatter(new_waypoints[:, 1], new_waypoints[:, 2], s=0.5)
plt.scatter(waypoints[:, 0], waypoints[:, 1], s=0.5)
plt.show()

planner.waypoints = new_waypoints
# np.set_printoptions(suppress=True, precision=4)
# print(planner.waypoints[:10, :])

env.update_params(new_params)
obs, step_reward, done, info = env.reset(new_waypoints[conf.start_ind, 1:4][None, :])
env.render()
env.render_waypoints(new_waypoints[:, 1:3])

lqr_params = {'matrix_q_1': 0.7, 'matrix_q_2': 0.0, 'matrix_q_3': 1.2, 'matrix_q_4': 0.0, 'matrix_r': 5., 'iterations': 50, 'eps': 0.001}

# after testing to find min and max
# matrix_r: 35 max, 1 min
# matrix_q_1: 7 max, 0.1 min
# matrix_q_2: 0.3 max, 0.0 min
# matrix_q_3: 10 max, 0.0 min
# matrix_q_4: 0.5 max, 0.0 min

laptime = 0.0
start = time.time()
trajx = []
trajy = []
trajx1 = []
trajy1 = []
trajthe = []
speeds = []
steers = []

while not done:
    if obs['lap_counts'][0] < 1:
        trajx.append(obs['poses_x'][0])
        trajy.append(obs['poses_y'][0])
    else:
        trajx1.append(obs['poses_x'][0])
        trajy1.append(obs['poses_y'][0])
    trajthe.append(obs['poses_theta'][0])
    speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'])
    # speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], env.timestep,
    #                                 lqr_params['matrix_q_1'],lqr_params['matrix_q_2'],lqr_params['matrix_q_3'], lqr_params['matrix_q_4'],
    #                                 lqr_params['matrix_r'], lqr_params['iterations'], lqr_params['eps'])
    # speeds.append(speed)
    # steers.append(steer)
    obs, step_reward, done, info = env.step(np.array([[steer, 0.1*speed]]))
    laptime += np.around(step_reward, 2)
    env.render('human')

print(laptime, np.around(laptime, 2), time.time()-start)
# import matplotlib.pyplot as plt
# plt.scatter(trajx, trajy, c='red')
# plt.scatter(trajx1, trajy1, c='blue')
# plt.show()


