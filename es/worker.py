import ray
import gym
import numpy as np
import warnings
from scipy.signal import resample

from planners import PurePursuitPlanner, load_waypoints
from utils import perturb

@ray.remote
class GymWorker:
    """
    Ray remote gym worker, each worker has its own gym instance
    """
    def __init__(self, conf):
        # score
        self.curr_laptime = 0.0
        self.rollout_done = False

        self.conf = conf

        # default params
        self.default_params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
        self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

        # init worker's associated gym instance
        self.env = gym.make('f110_gym:f110-v0', seed=conf.seed, map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        # reset gym instance
        obs, step_reward, done, info = self.env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

        # init pure pursuit planner with raceline
        self.planner = PurePursuitPlanner(conf, self.params['lf']+self.params['lr'])

    def run_sim(self, work):
        """
        Run simulation with given work

        Args:
            work (dict): {'mass', 'lf', 'tlad', 'vgain'}

        Returns:
            None
        """

        # get work detail
        mass = work['mass']
        lf = work['lf']
        tlad = work['tlad']
        xy = work['xy']
        velocities = work['velocities']
        vgain = 1
        # vgain = work['vgain']

        track_width = 1.1
        og_xy_wpts = load_waypoints(self.conf)[:, [self.conf.wpt_xind, self.conf.wpt_yind]]
        og_xy_wpts_subsampled = resample(og_xy_wpts, self.conf.num_subsampled_wpts)
        xy_wpts = perturb(xy, og_xy_wpts_subsampled, track_width, smoothing=20)

        new_wpts = np.zeros([xy_wpts.shape[0], self.planner.waypoints.shape[1]])
        new_wpts[:, self.conf.wpt_vind] = velocities
        new_wpts[:, [self.conf.wpt_xind, self.conf.wpt_yind]] = xy_wpts
        self.planner.waypoints = new_wpts


        # set new params
        wheelbase = self.params['lf'] + self.params['lr']
        self.params['lf'] = lf
        self.params['lr'] = wheelbase - lf
        self.params['m'] = mass

        # update params for gym instance
        self.env.update_params(self.params)

        # reset env
        obs, step_reward, done, info = self.env.reset(np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]]))

        # reset score
        self.curr_laptime = 0.0
        self.rollout_done = False

        cummulated_laptime = 0.0
        while not done:
            # get actuation from planner
            speed, steer = self.planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], tlad, vgain)

            is_catch_warning = True
            # is_catch_warning = False
            if is_catch_warning:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        # step environment
                        obs, step_reward, done, info = self.env.step(np.array([[steer, speed]]))
                    except:
                        f_name = '/home/brandon/tunercar_ws/src/tunercar/es/tunercar_runs/npzs/work_load.npz'
                        np.savez_compressed(f_name, mass=mass, lf=lf, tlad=tlad, xy=xy, velocities=velocities, og_xy_wpts=og_xy_wpts, new_wpts=new_wpts)
                        print('Saving work load')
                        exit()
            else:
                obs, step_reward, done, info = self.env.step(np.array([[steer, speed]]))

            # increment laptime
            cummulated_laptime += step_reward

            if cummulated_laptime >= 300.:
                cummulated_laptime = 99999.
                break

        # out of sim loop, check if collision
        if obs['collisions'][0]:
            cummulated_laptime = 99999.
        else:
            pass

        self.curr_laptime = cummulated_laptime
        self.rollout_done = True
        # print('Lap time:', cummulated_laptime)

    def collect(self):
        """
        Collect function, called when lap time is requested
        Resets worker instance after called

        Args:
            None

        Returns:
            curr_laptime (float): score/laptime of the current rollout
        """
        while not self.rollout_done:
            continue
        
        return self.curr_laptime