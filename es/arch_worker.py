import ray
import numpy as np
import os
import sys
import shutil

from prob_design_generator.space import DesignSpace
from uav_simulator.simulation import Simulation
import networkx as nx
import pickle as pk
from multiprocessing import Process
from multiprocessing import Manager
from quad_worker import QuadWorker
from generate_design import Design

@ray.remote
class ArchWorker:
    """
    Ray remote worker, each worker has its own simulation/binary instance
    Objective values are max_distance and max_hover_time
    """
    def __init__(self, conf, worker_id):
        # score keeping
        self.score = []
        self.eval_done = False

        self.conf = conf
        self.worker_id = worker_id

        # trim workers
        # self.workers = [QuadWorker().remote(conf, i) for i in conf.num_workers]

        self.space = DesignSpace(self.conf.acel_path)

    def _get_trim_score(self, responses):
        """
        Updates the worker's score when in trim only scenario

        Args:
            responses (dict{pandas.DataFrame}): return from simulation

        Returns:
            None
        """
        # if response is empty, trim is not found
        if not bool(responses):
            self.score = 8 * [99999.]
            return

        # forward trim
        forward_df = responses['forward']
        forward_dist_obj = (2000.0 - forward_df['Distance'].max())
        forward_time_obj = (410.0 - forward_df['Flight time'].max())
        forward_frac_obj = 500.0 * (forward_df[['Frac pow', 'Frac amp', 'Frac current']] >= 1.0).sum().sum()

        # turn radius 500 trim
        turn_500_df = responses['turn_500']
        turn_500_dist_obj = (3142.0 - turn_500_df['Distance'].max())
        turn_500_frac_obj = 500.0 * (turn_500_df[['Frac pow', 'Frac amp', 'Frac current']] >= 1.0).sum().sum()

        # turn radius 300 trim
        turn_300_df = responses['turn_300']
        turn_300_dist_obj = (3500.0 - turn_300_df['Distance'].max())
        turn_300_speed_obj = - turn_300_df['Speed'].max()
        turn_300_frac_obj = 500.0 * (turn_300_df[['Frac pow', 'Frac amp', 'Frac current']] >= 1.0).sum().sum()

        self.score = [forward_dist_obj, forward_time_obj, forward_frac_obj, turn_500_dist_obj, turn_500_frac_obj, turn_300_dist_obj, turn_300_speed_obj, turn_300_frac_obj]

    def _generate_design(self, selections):
        design = Design(self.conf.node_options, self.conf.end_options)
        design.generate_by_selections(selections)
        design_graph = design.to_design_graph(self.space)
        return design_graph

    def run_sim(self, selections, eid):
        """
        Runs the full SwRI simulation with LQR parameters

        Args:
            selections (list (N, )): sampled current candidate

        Returns:
            None
        """
        # reset score before sim
        self.score = []
        self.eval_done = False

        design_graph = self._generate_design(selections)

        try:
            simulation = Simulation(eval_id=eid,
                                    base_folder=self.conf.base_folder,
                                    create_folder=True)
            manager = Manager()
            responses = manager.dict()
            # run trim only
            process = Process(target=simulation.evaluate_design,
                              args=(design_graph, True, True, [], True, False, responses))
            process.start()
            process.join()

            self._get_trim_score(responses)
            
            output_path = os.path.join(simulation.eval_folder, "design_graph.pk")
            with open(output_path, "wb") as fout:
                pk.dump(design_graph, fout)

            # shutil.rmtree(os.path.join(simulation.eval_folder, "assembly/"))
        except Exception as e:
            print(e)
            self.score = 8 * [99999.]

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
