import ray
import numpy as np
import os
import sys
from itertools import cycle

from quadspider import construct_baseline_quad_spider_design
from quad import construct_baseline_quad_rotor_design
from hcopter import construct_baseline_hcopter_design
from hexring import construct_baseline_hexring_design
from hex import construct_baseline_hex_rotor_design
from hplane import construct_baseline_hplane_design
from prob_design_generator.space import DesignSpace
from uav_simulator.simulation import Simulation
from multiprocessing import Process
from multiprocessing import Manager
import pickle as pk
import os

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

        self.mapping = {
            "quadspider": construct_baseline_quad_spider_design,
            "quad": construct_baseline_quad_rotor_design,
            "hcopter": construct_baseline_hcopter_design,
            "hexring": construct_baseline_hexring_design,
            "hplane": construct_baseline_hplane_design,
            "hex": construct_baseline_hex_rotor_design,
        }

    def run_sim(self, raw_work):
        """
        Runs the full SwRI simulation with LQR parameters

        Args:
            raw_work (numpy.ndarray (N, )): sampled current candidate, size dependends on vehicle

        Returns:
            None
        """

        # extract the selected vector
        selected_vector = [raw_work['battery'],
                           raw_work['esc1'],
                           raw_work['esc2'],
                           raw_work['esc3'],
                           raw_work['esc4'],
                           raw_work['arm1'],
                           raw_work['arm2'],
                           raw_work['arm3'],
                           raw_work['arm4'],
                           raw_work['prop1'],
                           raw_work['prop2'],
                           raw_work['prop3'],
                           raw_work['prop4'],
                           raw_work['motor1'],
                           raw_work['motor2'],
                           raw_work['motor3'],
                           raw_work['motor4'],
                           raw_work['support1'],
                           raw_work['support2'],
                           raw_work['support3'],
                           raw_work['support4'],
                           raw_work['arm_length1'],
                           raw_work['arm_length2'],
                           raw_work['arm_length3'],
                           raw_work['arm_length4'],
                           raw_work['support_length1'],
                           raw_work['support_length2'],
                           raw_work['support_length3'],
                           raw_work['support_length4'],
                           raw_work['Q_position'],
                           raw_work['Q_velocity'],
                           raw_work['Q_angular_velocity'],
                           raw_work['Q_angles'],
                           raw_work['control_R']]

        callback = self.mapping[self.conf.vehicle]
        try:
            space = DesignSpace(self.conf.acel_path)
            design_graph = callback(space, selected_vector, is_selected=True)
            simulation = Simulation(eval_id=raw_work['eval_id'],
                                    base_folder=self.conf.base_folder,
                                    create_folder=True)

            manager = Manager()
            responses = manager.dict()
            process = Process(target=simulation.evaluate_design,
                              args=(design_graph, responses))
            process.start()
            process.join()
            # print(responses)
            # responses = simulation.evaluate_design(design_graph)
            self.score = [responses[1]['score'],
                          responses[3]['score'],
                          responses[4]['score'],
                          responses[5]['score']]
            output_path = os.path.join(simulation.eval_folder, "design_graph.pk")
            with open(output_path, "wb") as fout:
                pk.dump(design_graph, fout)
        except:
            self.score = [0.0, 0.0, 0.0, 0.0]
        self.eval_done = True

    def run_sim_simple(self, raw_work):
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