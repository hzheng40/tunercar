import ray

# import gym
# import numpy as np
# from planners import PurePursuitPlanner, StanleyPlanner, LQRPlanner
# from utils import perturb, interpolate_velocity, subsample

@ray.remote
class QuadWorker:
    """
    Ray remote gym worker, each worker has its own gym instance
    """
    def __init__(self, conf, worker_id):
        # score
        self.curr_laptime = 0.0
        self.rollout_done = False

        self.conf = conf
        self.worker_id = worker_id

        # default params
        self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
        self.wb = self.params['lf'] + self.params['lr']

        # init worker's associated gym instance
        self.env = gym.make('f110_gym:f110-v0', seed=conf.seed, map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        self.env.update_params(self.params)
        # reset gym instance
        obs, step_reward, done, info = self.env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

        if conf.controller == 'stanley':
            self.planner = StanleyPlanner(conf, self.wb)
        elif conf.controller == 'lqr':
            self.planner = LQRPlanner(conf, self.wb)
        else:
            # init pure pursuit planner with raceline
            self.planner = PurePursuitPlanner(conf, self.wb)

    def run_sim(self, raw_work):
        """
        Run simulation with given work

        Args:
            work (dict): genome to be evaluated

        Returns:
            None
        """

        if self.conf.normalize_param:
            # print(raw_work)
            # reconstruct work dict if normalization is used
            work = {}
            # general
            work['perturb'] = self.conf.left_bound + raw_work[:self.conf.num_ctrl]*(self.conf.right_bound - self.conf.left_bound)
            work['mass'] = self.conf.mass_min + raw_work[self.conf.num_ctrl]*(self.conf.mass_max - self.conf.mass_min)
            work['lf'] = self.conf.lf_min + raw_work[self.conf.num_ctrl + 1]*(self.conf.lf_max - self.conf.lf_min)
            work['vel_min'] = self.conf.v_lower_min + raw_work[self.conf.num_ctrl + 2]*(self.conf.v_lower_max - self.conf.v_lower_min)
            work['vel_max'] = self.conf.v_upper_min + raw_work[self.conf.num_ctrl + 3]*(self.conf.v_upper_max - self.conf.v_upper_min)
            # controller
            if self.conf.controller == 'stanley':
                work['kpath'] = self.conf.kpath_min + raw_work[-1]*(self.conf.kpath_max - self.conf.kpath_min)
            elif self.conf.controller == 'lqr':
                work['q1'] = self.conf.q1_min + raw_work[-5]*(self.conf.q1_max - self.conf.q1_min)
                work['q2'] = self.conf.q2_min + raw_work[-4]*(self.conf.q2_max - self.conf.q2_min)
                work['q3'] = self.conf.q3_min + raw_work[-3]*(self.conf.q3_max - self.conf.q3_min)
                work['q4'] = self.conf.q4_min + raw_work[-2]*(self.conf.q4_max - self.conf.q4_min)
                work['r'] = self.conf.r_min + raw_work[-1]*(self.conf.r_max - self.conf.r_min)
            else:
                # default to pure pursuit
                work['tlad'] = self.conf.tlad_min + raw_work[-1]*(self.conf.tlad_max - self.conf.tlad_min)
        else:
            work = raw_work

        if self.conf.controller == 'stanley':
            kpath = work['kpath']
        elif self.conf.controller == 'lqr':
            q1 = work['q1']
            q2 = work['q2']
            q3 = work['q3']
            q4 = work['q4']
            r = work['r']
        else:
            # default to pure pursuit
            tlad = work['tlad']

        # reset waypoints in planner to centerline
        self.planner.reset_waypoints()

        # perturb waypoints, TODO: might need to do rejection sampling
        sub_ind = subsample(self.planner.waypoints.shape[0], self.conf.num_ctrl)
        try:
            pert_waypoints = perturb(work['perturb'], self.planner.waypoints[sub_ind, :], self.conf.track_width)
        except:
            self.curr_laptime = np.around(99999., 2)
            self.rollout_done = True
            return

        vel = interpolate_velocity(work['vel_min'], work['vel_max'], pert_waypoints[:, 4], method='sigmoid')
        new_waypoints = np.hstack((pert_waypoints, vel[:, None]))

        # import matplotlib.pyplot as plt
        # plt.scatter(new_waypoints[:, 1], new_waypoints[:, 2], s=0.5)
        # plt.show()
        # print(work['perturb'])
        # print(work)

        # update planner waypoints
        self.planner.waypoints = new_waypoints

        # get starting pose from conf, should default to 0
        start_x = new_waypoints[self.conf.start_ind, self.conf.wpt_xind]
        start_y = new_waypoints[self.conf.start_ind, self.conf.wpt_yind]
        start_th = new_waypoints[self.conf.start_ind, self.conf.wpt_thind]

        # set new params
        new_params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': work['lf'], 'lr': self.wb - work['lf'], 'h': 0.074, 'm': work['mass'], 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

        # update params for gym instance
        self.env.update_params(new_params)

        # reset env
        obs, step_reward, done, info = self.env.reset(np.array([[start_x, start_y, start_th]]))

        # reset score
        self.curr_laptime = 0.0
        self.rollout_done = False

        cummulated_laptime = 0.0
        if self.worker_id == self.conf.render_worker_id and self.conf.render:
            self.env.render('human')
            self.env.render_waypoints(new_waypoints[:, 1:3])
        while not done:
            # get actuation from planner
            if self.conf.controller == 'stanley':
                speed, steer = self.planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], kpath)
            elif self.conf.controller == 'lqr':
                speed, steer = self.planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], self.env.timestep, q1, q2, q3, q4, r, self.conf.iteration, self.conf.eps)
            else:
                # default to pure pursuit
                speed, steer = self.planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], tlad)

            # step environment
            obs, step_reward, done, info = self.env.step(np.array([[steer, speed]]))

            # render worker_id 0
            if self.worker_id == self.conf.render_worker_id and self.conf.render:
                self.env.render('human')

            # increment laptime
            cummulated_laptime += step_reward

            if cummulated_laptime >= 240.:
                cummulated_laptime = 99999.
                break

        # out of sim loop, check if collision
        if obs['collisions'][0]:
            cummulated_laptime = 99999.
            # cummulated_laptime = 600 - cummulated_laptime
            # cummulated_laptime = 240.
        else:
            pass

        self.curr_laptime = np.around(cummulated_laptime, 2)

        # # catch short laptime error
        # if self.curr_laptime < 10.:
        #     # TODO: error is that spline goes insane, and car was reset at zeroth index
        #     print(obs)
        #     import matplotlib.pyplot as plt
        #     plt.scatter(new_waypoints[:, 1], new_waypoints[:, 2], s=0.5)
        #     plt.show()
        #     print(work)
        #     raise(KeyboardInterrupt)

        # if self.curr_laptime < 300:
        #     print('something worked')

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