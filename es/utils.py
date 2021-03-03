import numpy as np
import scipy.interpolate
from numba import njit

def perturb(vec, waypoints, track_width, smoothing=20):
    """
    Perturbs the waypoints at control points and recreate the waypoing spline

    Args:
        vec (numpy.ndarray (n, )): perturbation vector of length n, each element between -1 and 1
        waypoints (numpy.ndarray (n, 2)): waypoint control points to perturb of length n, columns are (x, y)
        track_width (float): width of the track in meters
        smoothing (float): smoothing factor for reinterpolation. see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html for more details (s argument)
    """
    # get perp direction for each control point
    # diffs between points for vector
    diffs = waypoints[1:, :] - waypoints[:-1, :]
    # append last->first to the list
    diffs = np.vstack((diffs, waypoints[0,:] - waypoints[-1,:]))
    # get perps
    perps = np.zeros(diffs.shape)
    perps[:, 0] = diffs[:, 1]
    perps[:, 1] = -1 * diffs[:, 0]
    # normalize
    norm = np.linalg.norm(perps, axis=1)
    perps[:, 0] = perps[:, 0]/norm
    perps[:, 1] = perps[:, 1]/norm

    # shift control point in perp direction
    vec = track_width * vec.reshape((vec.shape[0], 1))
    new_waypoints = waypoints + np.multiply(perps, vec)

    # re-interpolate
    tck, u = scipy.interpolate.splprep([new_waypoints[:, 0], new_waypoints[:, 1]], s=smoothing, k=5, per=True)
    unew = np.arange(0, 1.0, 0.001)
    out_smooth = np.asarray(scipy.interpolate.splev(unew, tck)).T

    return out_smooth

def get_curvature(waypoints):
    """
    Returns the curvature of each point on the list of waypoints

    Args:
        waypoints (numpy.ndarray (n, 2)): list of waypoints

    Returns:
        curvature (numpy.ndarray (n, )): curvature at each point of the waypoint
    """
    
    # no smoothing, since the given waypoints should already be smoothed
    tck, u = scipy.interpolate.splprep([waypoints[:, 0], waypoints[:, 1]], s=0, k=5, per=True)
    unew = np.arange(0, 1.0, 0.001)
    derivs = scipy.interpolate.spalde(unew, tck)

    # getting the curvature
    Dx = np.asarray(derivs[0])
    Dy = np.asarray(derivs[1])
    dx = Dx[:, 1]
    dy = Dy[:, 1]
    ddx = Dx[:, 2]
    ddy = Dy[:, 2]
    curvature = (dx*ddy - dy*ddx)/((dx*dx + dy*dy)**1.5)

    return curvature

def interpolate_velocity(min_vel, max_vel, waypoints, method='linear', optional_args=[]):
    """
    Interpolate to find velocity along the waypoints based on curvature

    Args:
        min_vel (float): Minimum velocity at maximum curvature
        max_vel (float): Maximum velocity at minimum curvature
        waypoints (numpy.ndarray(n, 2)): Waypoints to find velocity on, columns are (x, y), after perturb
        method (str, default='linear'): method of interpolation, default is linear, options include ['linear', 'sigmoid', ]
        optional_args (list(str), default=[]): optional arguments for different interpolation methods

    Returns:
        vel (numpy.ndarray(n, )):
    """
    # interp data
    curvature = np.abs(get_curvature(waypoints))
    x = np.array([np.min(curvature), np.max(curvature)])
    y = np.array([max_vel, min_vel])
    
    if method == 'linear':
        f = scipy.interpolate.interp1d(x, y, kind='linear')
        vel = f(curvature)

    return vel

# unit tests
import unittest
import matplotlib.pyplot as plt

class PerturbTest(unittest.TestCase):
    def setUp(self):
        # seed
        np.random.seed(1234)

        # load waypoints
        raw_waypoints = np.loadtxt('maps/f1tenth_racetracks/Austin/Austin_map_waypoints.csv', delimiter=',', skiprows=1)
        self.waypoints = raw_waypoints[:, 0:2]
        self.track_width = raw_waypoints[0, 2] = raw_waypoints[0, 3]

        # min max vel
        self.min_vel = 2.0
        self.max_vel = 15.0

    def test_perturb(self):
        # random perturbation
        vec = 0.5*(2.0*np.random.rand(self.waypoints.shape[0]) - 1.0)
        vec1 = 1.0 * np.ones(self.waypoints.shape[0])
        vec2 = -1.0 * np.ones(self.waypoints.shape[0])
        # random
        new_waypoints = perturb(vec, self.waypoints, self.track_width)
        # outer
        new_waypoints1 = perturb(vec1, self.waypoints, self.track_width, smoothing=5)
        # inner
        new_waypoints2 = perturb(vec2, self.waypoints, self.track_width, smoothing=5)

        plt.scatter(self.waypoints[:, 0], self.waypoints[:, 1], c='red')
        plt.scatter(new_waypoints1[:, 0], new_waypoints1[:, 1], c='blue')
        plt.scatter(new_waypoints2[:, 0], new_waypoints2[:, 1])
        plt.scatter(new_waypoints[:, 0], new_waypoints[:, 1])
        plt.axes().set_aspect('equal')
        plt.show()

    def test_curvature(self):
        vec = np.zeros(self.waypoints.shape[0])
        new_waypoints = perturb(vec, self.waypoints, self.track_width)
        curvature = get_curvature(new_waypoints)
        plt.scatter(new_waypoints[:, 0], new_waypoints[:, 1], c=curvature)
        plt.show()
        self.assertEqual(1000, curvature.shape[0])

    def test_vel_interp(self):
        vec = np.zeros(self.waypoints.shape[0])
        new_waypoints = perturb(vec, self.waypoints, self.track_width)
        vel = interpolate_velocity(self.min_vel, self.max_vel, new_waypoints)
        plt.scatter(new_waypoints[:, 0], new_waypoints[:, 1], c=vel)
        plt.show()
        self.assertEqual(1000, vel.shape[0])

if __name__ == '__main__':
    unittest.main()