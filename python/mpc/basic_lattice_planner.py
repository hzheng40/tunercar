from mpc import trajectory_generator_utils, pure_pursuit_utils
import numpy as np
from PIL import Image
from numpy import linalg as LA
import math
from numba import njit
from numba.typed import List, Dict
from mpc import AStrajectory_generator as trajectory_generator
from mpc import transformations
import os
import yaml

class BasicLatticePlanner(object):
    def __init__(self, map_path, map_img_ext, waypoints, directory, track_lad, grid_lad):
        self.prev_traj = None
        self.prev_pp_traj = None
        self.prev_param = None
        self.prev_steer = 0.
        self.waypoints = waypoints
        self.wheelbase = 0.3302
        self.max_reacquire = 10
        self.safe_speed = 2.5

        self.track_lad = track_lad
        self.grid_lad = grid_lad
        self.STEER_LP = 0.99

        lut_all = np.load(directory + '/lut_inuse_nokappa.npz')
        self.lut_x = lut_all['x']
        self.lut_y = lut_all['y']
        self.lut_theta = lut_all['theta']
        self.lut = lut_all['lut']

        map_prefix = os.path.splitext(map_path)[0]
        map_img_path = map_prefix + map_img_ext
        map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        map_img = map_img.astype(np.float64)
        inflation = 3
        self.costmap = trajectory_generator_utils.fill_occgrid_map(map_img, inflation)

        with open(map_path, 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
            except yaml.YAMLError as ex:
                print(ex)


        self.map_resolution = map_metadata['resolution']
        self.map_origin_x = map_metadata['origin'][0]
        self.map_origin_y = map_metadata['origin'][1]

    def _rotation_matrix(self, angle, direction, point=None):
        sina = math.sin(angle)
        cosa = math.cos(angle)
        direction = self._unit_vector(direction[:3])
        # rotation matrix around unit vector
        R = np.array(((cosa, 0.0,  0.0),
                         (0.0,  cosa, 0.0),
                         (0.0,  0.0,  cosa)), dtype=np.float64)
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array((( 0.0,         -direction[2],  direction[1]),
                          ( direction[2], 0.0,          -direction[0]),
                          (-direction[1], direction[0],  0.0)),
                         dtype=np.float64)
        M = np.identity(4)
        M[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            M[:3, 3] = point - np.dot(R, point)
        return M

    def _unit_vector(self, data, axis=None, out=None):
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= math.sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data*data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = waypoints[:, 0:2]
        # position = np.array(position[0:2])
        nearest_point, nearest_dist, t, i = pure_pursuit_utils.nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = pure_pursuit_utils.first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty(waypoints[i2, :].shape)
            # x, y
            current_waypoint[0:2] = waypoints[i2, 0:2]
            # theta
            current_waypoint[3] = waypoints[i2, 3]
            # speed
            current_waypoint[2] = waypoints[i, 2]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return waypoints[i, :]
        else:
            return None

    def _get_current_speed(self, position, theta):
        wpts = self.waypoints[:, 0:2]
        position = position[0:2]
        nearest_point, nearest_dist, t, i = pure_pursuit_utils.nearest_point_on_trajectory_py2(position, wpts)
        speed = self.waypoints[i, 2]
        return speed, nearest_dist

    def _pure_pursuit(self, pose_x, pose_y, pose_theta, trajectory, lookahead_distance):
        # returns speed, steering_angle pair

        if trajectory is None:
            # trajectory gone, moving straight forward
            return self.safe_speed, 0.0
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(trajectory, lookahead_distance, position, pose_theta)
        if lookahead_point is None:
            # no lookahead point, slow down
            return self.safe_speed, 0.0

        speed, steering_angle = pure_pursuit_utils.get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)

        # if abs(steering_angle) > 0.4189:
            # print('clipped')
            # steering_angle = (steering_angle/abs(steering_angle))*0.4189
        return speed, steering_angle


    def plan(self, pose, current_vel):
        current_vel = max(0.01, current_vel)
        pose_x, pose_y, pose_theta = pose
       
        lookahead_point = self._get_current_waypoint(self.waypoints, self.grid_lad, np.array([pose_x, pose_y]), pose_theta)
        if lookahead_point is None:
            return self.safe_speed, self.prev_steer


        goal_x, goal_y, speed, waypoint_theta = lookahead_point
        # quat = transformations.quaternion_from_euler(0.0, 0.0, pose_theta)
        # goal_quat = transformations.quaternion_from_euler(0.0, 0.0, waypoint_theta)
        # qr = transformations.quaternion_multiply(goal_quat, quat)

        if waypoint_theta < 0:
            waypoint_theta_temp = 2*np.pi + waypoint_theta
        else:
            waypoint_theta_temp = waypoint_theta
        goal_theta = waypoint_theta_temp - pose_theta
        # goal_theta = waypoint_theta - pose_theta
        if goal_theta > np.pi:
            goal_theta = goal_theta - (2*np.pi)
        if goal_theta < -np.pi:
            goal_theta = goal_theta + (2*np.pi)

        if abs(goal_theta) > np.pi/2:
            goal_theta = (goal_theta/abs(goal_theta))*np.pi/2

        rot = transformations.rotation_matrix(pose_theta, (0, 0, 1))
        grid_rot = transformations.rotation_matrix(goal_theta, (0, 0, 1))
        rot_cw = transformations.rotation_matrix(-np.pi/2, (0, 0, 1))
        rot_ccw = transformations.rotation_matrix(np.pi/2, (0, 0, 1))

        goal_grid = trajectory_generator_utils.create_grid_map_only(goal_x, goal_y, 0.3, np.array([goal_theta]), grid_rot, rot_cw, rot_ccw, 0.01, 0.8, self.grid_lad/1.5, np.ascontiguousarray(self.costmap), self.map_origin_x, self.map_origin_y, self.map_resolution, np.array([[pose_x], [pose_y], [0.0]]), rot)

        states_list_local = trajectory_generator_utils.grid_lookup(goal_grid, self.lut_x, self.lut_y, self.lut_theta, self.lut)

        states_list = trajectory_generator_utils.trans_traj_list(states_list_local, np.array([[pose_x], [pose_y], [0.0]]), rot)

        free_traj_list = trajectory_generator.sample_parallel_map_only(states_list, np.ascontiguousarray(self.costmap), self.map_origin_x, self.map_origin_y, self.map_resolution)

        free_traj_idx = np.where(free_traj_list)[0]
        free_traj_sorted = np.sort(free_traj_idx)

        if len(free_traj_sorted) == 0:
            next_speed, next_steer = self._pure_pursuit(pose_x, pose_y, pose_theta, self.prev_pp_traj, self.track_lad)
            return next_speed, next_steer

        best_traj_idx = free_traj_sorted[0]
        best_traj = states_list[best_traj_idx*trajectory_generator.NUM_STEPS:(best_traj_idx+1)*trajectory_generator.NUM_STEPS, 0:2]

        pp_traj = np.empty((best_traj.shape[0], best_traj.shape[1]+2))
        pp_traj[:, 0:2] = best_traj
        pp_traj[:, 2] = speed*np.ones(best_traj.shape[0])
        pp_traj[:, 3] = np.zeros(best_traj.shape[0])

        next_speed, next_steer = self._pure_pursuit(pose_x, pose_y, pose_theta, pp_traj, self.track_lad)

        return next_speed, next_steer


    def compute_action(self, pp_traj, safety_flag, pose, off_policy=False):
        pose_x, pose_y, pose_theta = pose

        if safety_flag:
            next_speed, next_steer = self._pure_pursuit(pose_x, pose_y, pose_theta, pp_traj, self.track_lad)
            next_steer = self.STEER_LP*next_steer+(1-self.STEER_LP)*self.prev_steer

        elif off_policy:
            next_speed, next_steer = self._pure_pursuit(pose_x, pose_y, pose_theta, self.waypoints[:, 0:4], self.track_lad)
            next_steer = self.STEER_LP*next_steer+(1-self.STEER_LP)*self.prev_steer

        else:
            next_speed, next_steer = self._pure_pursuit(pose_x, pose_y, pose_theta, pp_traj, self.track_lad)

        self.prev_steer = next_steer

        return next_speed, next_steer