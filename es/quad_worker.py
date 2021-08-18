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
        # reset score before sim
        self.score = []
        self.eval_done = False

        # extract the selected vector
        # selected_vector = [raw_work['battery'],
        #                    raw_work['esc1'],
        #                    raw_work['esc2'],
        #                    raw_work['esc3'],
        #                    raw_work['esc4'],
        #                    raw_work['arm1'],
        #                    raw_work['arm2'],
        #                    raw_work['arm3'],
        #                    raw_work['arm4'],
        #                    raw_work['prop1'],
        #                    raw_work['prop2'],
        #                    raw_work['prop3'],
        #                    raw_work['prop4'],
        #                    raw_work['motor1'],
        #                    raw_work['motor2'],
        #                    raw_work['motor3'],
        #                    raw_work['motor4'],
        #                    raw_work['support1'],
        #                    raw_work['support2'],
        #                    raw_work['support3'],
        #                    raw_work['support4'],
        #                    raw_work['arm_length1'],
        #                    raw_work['arm_length2'],
        #                    raw_work['arm_length3'],
        #                    raw_work['arm_length4'],
        #                    raw_work['support_length1'],
        #                    raw_work['support_length2'],
        #                    raw_work['support_length3'],
        #                    raw_work['support_length4'],
        #                    *raw_work['lqr_vector'],
        #                    *raw_work['lat_vel'],
        #                    *raw_work['vert_vel']
        #                    ]
        selected_vector = []
        for key in raw_work:
            if isinstance(raw_work[key], np.ndarray):
                selected_vector.extend(list(raw_work[key]))
            else:
                selected_vector.append(raw_work[key])

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
            # responses = simulation.evaluate_design(design_graph)
            # self.score = [responses[1]['score'],
            #               responses[3]['score'],
            #               responses[4]['score'],
            #               responses[5]['score']]
            if not bool(responses):
                self.score = [0.0, 0.0, 0.0, 0.0]
            else:
                for key in responses:
                    self.score.append(responses[key]['score'] + 10.)
            output_path = os.path.join(simulation.eval_folder, "design_graph.pk")
            with open(output_path, "wb") as fout:
                pk.dump(design_graph, fout)
        except Exception as e:
            print(e)
            self.score = [-1000.0, -1000.0, -1000.0, -1000.0]
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