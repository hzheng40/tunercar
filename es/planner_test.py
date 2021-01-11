import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from planners import PurePursuitPlanner

work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
with open('../es/configs/config_map0.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
obs, step_reward, done, info = env.reset(np.array([[0., 0., 1.57079632679]]))
# env.render()
planner = PurePursuitPlanner(conf, 0.17145+0.15875)

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
    speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
    speeds.append(speed)
    steers.append(steer)
    obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
    laptime += step_reward
    # env.render()
print(laptime, time.time()-start)
import matplotlib.pyplot as plt
plt.scatter(trajx, trajy, c='red')
plt.scatter(trajx1, trajy1, c='blue')
plt.show()