import numpy as np

from pure_pursuit_utils import *

class PurePursuitPlanner:
    def __init__(self):
        pass

    def load_waypoints(self, config):
        # load waypoints
        waypoints = np.loadtxt(config[''])


    def plan(self, obs):
        pass

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

        if abs(steering_angle) > 0.4189:
            # print('clipped')
            steering_angle = (steering_angle/abs(steering_angle))*0.4189
        return speed, steering_angle