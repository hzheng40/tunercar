import ray
import numpy as np
import os
import sys
from itertools import cycle

@ray.remote
class QuadWorker:
    """
    Ray remote worker, each worker has its own simulation/binary instance
    Objective values are max_distance and max_hover_time
    """
    def __init__(self, conf, worker_id):
        # score keeping
        # self.max_distance = 0.0
        # self.max_hover_time = 0.0
        self.score = []
        self.eval_done = False

        self.conf = conf
        self.worker_id = worker_id

        self.sim = None

    def run_sim(self, raw_work):
        """
        Run simulation with given work

        Args:
            raw_work (dict): 

        Returns:
            None
        """

        # if self.sim is None:
        # initialize simulation
        wrapper_path = self.conf.fdm_wrapper_path
        sys.path.append(wrapper_path)
        from fdm_wrapper.components.propulsion_block import PropulsionBlock
        from fdm_wrapper.components.battery import Battery
        from fdm_wrapper.simulation import Simulation
        from fdm_wrapper.design import Design
        self.sim = Simulation(eval_id=raw_work['eval_id'], create_folder=True)

        # extract current genome
        arm_length = raw_work['arm_length']
        num_batt = raw_work['num_batt']
        batt_v = raw_work['batt_v']
        batt_cap = raw_work['batt_cap']
        batt_m = raw_work['batt_m']

        # create design for eval
        design = Design()

        # create and add components to design
        prob_blocks = [PropulsionBlock(arm_length=arm_length)] * 4
        for prob in prob_blocks:
            design.add_propulsion_block(prob)
        batteries = [Battery(mass=batt_m, voltage=batt_v, capacity=batt_cap,
                             c_peak=150.0, c_continuous=75.0, rm=13.0)] * num_batt
        for batt in batteries:
            design.add_battery(batt)

        # finalize desgin
        design.finalize()
        
        # assign propulsion blocks to a battery
        zipped_prop = zip(cycle([*range(1, num_batt + 1)]), prob_blocks)
        for zip_batt, zip_prob in zipped_prop:
            zip_prob.assign_to_battery(zip_batt)

        # fdm eval
        responses = self.sim.evaluate_design(design)
        
        # store multi-objective score, negate because want to maximize
        max_dist = responses['max_distance']
        max_hov = responses['max_hover_time']
        self.score = [0.0 if (np.isnan(max_dist) or np.isinf(max_dist)) else -max_dist, 0.0 if (np.isnan(max_hov) or np.isinf(max_hov)) else -max_hov]
        self.eval_done = True

    def collect(self):
        """
        Collect function, called when lap time is requested
        Resets worker instance after called

        Args:
            None

        Returns:
            curr_laptime (float): score/laptime of the current rollout
        """
        while not self.eval_done:
            continue
        return self.score