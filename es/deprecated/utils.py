import numpy as np
import scipy.interpolate

def perturb(vec, waypoints, track_width, smoothing=0, num_waypoints=1000):
    """
    Perturbs the waypoints at control points and recreate the waypoint spline with arc length, xy, heading, and curvature

    Args:
        vec (numpy.ndarray (n, )): perturbation vector of length n, each element between -1 and 1
        waypoints (numpy.ndarray (n, 2)): waypoint control points to perturb of length n, columns are (x, y)
        track_width (float): width of the track in meters
        smoothing (float, default=5): smoothing factor for reinterpolation. see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html for more details (s argument)
        num_waypoints (int, default=1000): number of new waypoints in the perturbed new waypoint
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
    tck, u = scipy.interpolate.splprep([new_waypoints[:, 0], new_waypoints[:, 1]], s=smoothing, k=5, per=True, quiet=5)
    unew = np.arange(0, 1.0, 1/num_waypoints)
    out_new = np.asarray(scipy.interpolate.splev(unew, tck)).T

    # arc length
    diffs_new = np.linalg.norm(out_new[1:, :] - out_new[:-1, :], axis=1)
    s = np.cumsum(diffs_new)
    s = np.insert(s, 0, 0)

    # derivatives
    derivs = scipy.interpolate.spalde(unew, tck)
    Dx = np.asarray(derivs[0])
    Dy = np.asarray(derivs[1])
    dx = Dx[:, 1]
    dy = Dy[:, 1]
    ddx = Dx[:, 2]
    ddy = Dy[:, 2]

    # heading
    theta = np.arctan2(dy, dx)
    theta[theta > 2*np.pi] -= 2*np.pi
    theta[theta < 0] += 2*np.pi

    # curvature
    curvature = (dx*ddy - dy*ddx)/((dx*dx + dy*dy)**1.5)

    # organize
    new_wpts = np.empty((out_new.shape[0], 5))
    new_wpts[:, 0] = s
    new_wpts[:, 1:3] = out_new
    new_wpts[:, 3] = theta
    new_wpts[:, 4] = curvature

    return new_wpts

def interpolate_velocity(min_vel, max_vel, curvatures, method='linear', optional_args=[]):
    """
    Interpolate to find velocity along the waypoints based on curvature

    Args:
        min_vel (float): Minimum velocity at maximum curvature
        max_vel (float): Maximum velocity at minimum curvature
        curvatures (numpy.ndarray(n, )): list of curvature along waypoints
        method (str, default='linear'): method of interpolation, default is linear, options include ['linear', 'sigmoid', ]
        optional_args (list(str), default=[]): optional arguments for different interpolation methods

    Returns:
        vel (numpy.ndarray(n, )):
    """
    # interp data
    curvature = np.abs(curvatures)
    x = np.array([np.min(curvature), np.max(curvature)])
    y = np.array([max_vel, min_vel])

    if method == 'linear':
        f = scipy.interpolate.interp1d(x, y, kind='linear')
        vel = f(curvature)

    # TODO: might not be the best to use linear, need to compare with raceline vel to see what's going on

    elif method == 'sigmoid':
        kernel = np.ones(50)/50
        f = lambda x: -2/(1+np.exp(-10*x))+2
        vel = min_vel + f(curvature)*(max_vel-min_vel)
        # plt.plot(vel)
        vel_padded = np.hstack((vel[0]*np.ones(50), vel, vel[-1]*np.ones(50)))
        vel_padded = np.convolve(vel_padded, kernel, mode='same')
        vel = vel_padded[50:-50]
        # vel = np.convolve(vel, kernel, mode='same')
        # print(vel[0], vel[-1], vel[1], vel[-2])
        # plt.plot(vel)
        # plt.show()

    return vel

def subsample(orig_length, num_samples, shift=0):
    """
    Find the index of subsamples in a length of array with given number of samples

    Args:
        orig_length (int): Length of axis of original array for subsampling
        num_samples (int): Number of subsamples
        shift (int, default=0): Shift in indices for subsamples for rejection sampling

    Returns:
        indices (np.array(num_samples, )): Indices of the samples
    """

    # making sure number of samples is less than original length
    assert (orig_length >= num_samples), "Number of subsamples must be less than the original length."

    indices = np.linspace(0, orig_length-1, num=num_samples, dtype=int)
    indices = indices + shift
    indices[indices > orig_length - 1] -= orig_length - 1
    return indices

def check_collision(new_wpts, wpts, track_width, smoothing=5, num_waypoints=1000):
    """
    Check if perturbed waypoints intersects the track boundary

    Args:
        new_wpts (numpy.ndarray(num_new_wpts, 5)): waypoints after purturbation
        wpts (numpy.ndarray(n, 2)): original centerline
        track_width (float): track width to one side of centerline
        smoothing (int): floating factor for spline interpolation
        num_waypoints (int): number of waypoints after perturbation
    """
    vec_right = np.ones(wpts.shape[0])
    vec_left = -1. * np.ones(wpts.shape[0])
    track_right = perturb(vec_right, wpts, track_width)
    track_left = perturb(vec_left, wpts, track_width)
    # print(track_right.shape, track_left.shape, new_wpts.shape)
    # dists =

# unit tests
import unittest
import matplotlib.pyplot as plt
import matplotlib

class PerturbTest(unittest.TestCase):
    def setUp(self):
        # seed
        np.random.seed(1234)

        # load waypoints
        raw_waypoints = np.loadtxt('maps/f1tenth_racetracks/Spielberg/Spielberg_centerline.csv', delimiter=',', skiprows=1)
        self.compare_raceline = np.loadtxt('maps/f1tenth_racetracks/Spielberg/Spielberg_raceline.csv', delimiter=';', skiprows=3)
        self.waypoints = raw_waypoints[:, 0:2]
        self.track_width = raw_waypoints[0, 2]
        # min max vel
        self.min_vel = 4.0
        self.max_vel = 8.0

    def test_perturb(self):
        # random perturbation
        vec = 0.5*(2.0*np.random.rand(self.waypoints.shape[0]) - 1.0)
        # vec = np.zeros(self.waypoints.shape[0])
        vec1 = 1.0 * np.ones(self.waypoints.shape[0])
        vec2 = -1.0 * np.ones(self.waypoints.shape[0])
        # random
        # good number for silverstone
        num_samples = 100
        # num_samples = 400

        sub_ind = subsample(vec.shape[0], num_samples, shift=0)
        print(sub_ind)
        new_waypoints = perturb(vec[sub_ind], self.waypoints[sub_ind, :], self.track_width, smoothing=0)
        check_collision(new_waypoints, self.waypoints, self.track_width)
        plt.plot(new_waypoints[:, 3])
        plt.show()
        # outer
        new_waypoints1 = perturb(vec1, self.waypoints, self.track_width, smoothing=5)
        # inner
        new_waypoints2 = perturb(vec2, self.waypoints, self.track_width, smoothing=5)

        plt.scatter(self.waypoints[:, 0], self.waypoints[:, 1], c='red', s=0.3)
        plt.scatter(new_waypoints1[:, 1], new_waypoints1[:, 2], c='blue', s=0.3)
        plt.scatter(new_waypoints2[:, 1], new_waypoints2[:, 2], s=0.3)
        plt.scatter(new_waypoints[:, 1], new_waypoints[:, 2], s=0.3)
        plt.scatter(self.waypoints[sub_ind, 0], self.waypoints[sub_ind, 1], s=5.0, c='green')
        plt.axes().set_aspect('equal')
        plt.show()

    # def test_curvature(self):
    #     vec = np.zeros(self.waypoints.shape[0])
    #     new_waypoints = perturb(vec, self.waypoints, self.track_width, smoothing=0)
    #     plt.scatter(new_waypoints[:, 1], new_waypoints[:, 2], c=np.abs(new_waypoints[:, 4]))
    #     plt.show()
    #     self.assertEqual(1000, new_waypoints.shape[0])

    # def test_vel_interp(self):
    #     # vec = np.zeros(self.waypoints.shape[0])
    #     # new_waypoints = perturb(vec, self.waypoints, self.track_width, smoothing=0, num_waypoints=2233)
    #     new_waypoints = self.compare_raceline
    #     vel = interpolate_velocity(self.min_vel, self.max_vel, new_waypoints[:, 4], method='sigmoid')
    #     traffic = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "C2"])
    #     fig, axs = plt.subplots(1, 2, sharey=True)
    #     scatter1 = axs[0].scatter(new_waypoints[:, 1], new_waypoints[:, 2], c=vel, cmap=traffic)
    #     axs[0].set_title('interped')
    #     scatter2 = axs[1].scatter(self.compare_raceline[:, 1], self.compare_raceline[:, 2], c=self.compare_raceline[:, 5], cmap=traffic)
    #     axs[1].set_title('ref')
    #     plt.colorbar(scatter2, ax=axs)
    #     plt.show()

    #     plt.plot(vel)
    #     plt.plot(self.compare_raceline[:, 5])
    #     plt.show()
    #     self.assertEqual(1000, vel.shape[0])

if __name__ == '__main__':
    unittest.main()