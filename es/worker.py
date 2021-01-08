import ray
import gym
from pure_pursuit_utils import *

@ray.remote
class GymWorker:
    """
    Ray remote gym worker, each worker has its own gym instance
    """
    def __init__(self):
        # TODO: store something?
        self.curr_laptime = 0.0

        # TODO: init associated gym instance
        self.env = gym.make('f110_gym:f110-v0')

    def run_sim(self, work):
        """
        Run simulation with given work

        Args:
            work (dict): {'mass', 'lf', 'wlad', 'tdiv', 'vgain'}

        Returns:
            None
        """

        # TODO: get work detail
        mass = work['mass']
        lf = work['lf']
        wlad = work['wlad']
        tdiv = work['tdiv']
        vgain = work['vgain']

        # TODO: update params for gym instance
        self.env.update_params()

        # TODO: reset env
        obs, step_reward, done, info = self.env.reset()

        while not done:
            # TODO: get actuation from planner
            speed, steer = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase)

            # TODO: step environment
            obs, step_reward, done, info = self.env.step(speed, steer)

            # TODO: increment laptime
            self.curr_laptime += step_reward

        # TODO: out of sim loop, check if collision
        if obs['in_collision']:
            self.curr_laptime = 99999.
        else:
            pass

    def collect(self):
        """
        Collect function, called when lap time is requested
        Resets worker instance after called

        Args:
            None

        Returns:
            score (float): score/laptime of the current rollout
        """

        # TODO: make backup of laptime
        score = self.curr_laptime

        # TODO: reset everything
        self.curr_laptime = 0.0

        # TODO: something about gym reset?
        self.env.reset()

        return score