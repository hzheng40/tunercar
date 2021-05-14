import numpy as np

import pure_pursuit_utils


def load_waypoints(conf):
    return np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

class PurePursuitPlanner:
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.waypoints = load_waypoints(conf)
        self.max_reacquire = 20.

    # TODO: This method doesn't use theta
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = pure_pursuit_utils.nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = pure_pursuit_utils.first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        # TODO: Why pass self.waypoints as a method argument?
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = pure_pursuit_utils.get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle