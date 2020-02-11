from __future__ import print_function
import numpy as np
import numpy.linalg as LA
import math
import copy
import matplotlib.pyplot as plt
import os.path
import sys
from numba import jit
from numba import njit
from numba import jitclass
from numba import float64, boolean, void, int32, int64, uint8, prange
from numba import deferred_type
# from numba.typed import List


# ----------- FLAGS ----------
USE_HEURISTIC_INIT = True
CUBIC_STABLE = True
QUINTIC_STABLE = False
FIRST_ORDER = False

# ----------- CONSTANTS -------------

# max/min curvature (rad)
# KMAX = 30000000.0
# KMIN = -30000000.0
KMAX = 30.0
KMIN = -30.0
# max/min curvature rate (rad/sec)
# DKMAX = 300000000.0
# DKMIN = -300000000.0
DKMAX = 30.0
DKMIN = 30.0
# max/min accel/decel (m/s^2)
DVMAX = 2.000
DVMIN = -6.000
# control latency (sec)
TDELAY = 0.0800
# speed control logic a coeff
ASCL = 0.1681
# ASCL = 16.81
# speed control logic b coeff
BSCL = -0.0049
# speed control logic threshold (m/s)
VSCL = 4.000
# max curvature for speed (rad)
KVMAX = 0.1485
# speed control logic safety factor
SF = 1.000

# ----------- TERMINATION CRITERIA -------------

# minimum number of iterations
MIN_ITER = 1
MAX_ITER = 6000
# allowable crosstrack error (m)
CROSSTRACK_E = 0.001
# allowable inline error (m)
INLINE_E = 0.001
# allowable heading error (rad)
HEADING_E = 0.1
# allowable curvature error (m^-1)
CURVATURE_E = 0.005
# general error, ill-defined heuristic (unitless)
GENERAL_E = 0.05
E_MAX = 10000


# ----------- PARAMETER PERTURBATION -------------
# for estimation of partial deriv
# perturbation for a
H_SX = 0.001
# perturbation for b
H_SY = 0.001
# perturbation for d
H_THETA = 0.001
# perturbation for s
H_K = 0.001
# if all params perturbed equally, heuristic (unitless)
H_GLOBAL = 0.001


# ----------- NEWTONS METHOD ---------------------
STEP_GAIN = 0.01


# ----------- INTEGRATION STEP SIZE CONTROL-----------
# time step
STEP_SIZE = 0.0001
# STEP_SIZE = 0.00001
# time step for plotting
PLOT_STEP_SIZE = 0.1
# number of equally spaced points along spline
NUM_STEPS = 100

# ----------------- DEBUGS ------------------
DEBUG_ERROR = False
DEBUG_PLOT = False
DEBUG_BUILD_LUT = True
DEBUG_CONVERGENCE = False
REBUILD_LUT = False



# -------------- Data Structures --------------
state_spec = [('sx', float64),
              ('sy', float64),
              ('theta', float64),
              ('kappa', float64),
              ('v', float64)]

@jitclass(state_spec)
class State(object):
    """
    car state
    """
    def __init__(self, sx=0.0, sy=0.0, theta=0.0, kappa=0.0, v=0.0):
        self.sx = sx
        self.sy = sy
        self.theta = theta
        self.kappa = kappa
        self.v = v

spline_spec = [('s', float64),
              ('kappa_0', float64),
              ('kappa_1', float64),
              ('kappa_2', float64),
              ('kappa_3', float64)]

@jitclass(spline_spec)
class Spline(object):
    """
    parameterized spline with 5 parameters
    notation from Howard 09
    """
    def __init__(self, s=0.0, kappa_0=0.0, kappa_1=0.0, kappa_2=0.0, kappa_3=0.0):
        self.s = s
        self.kappa_0 = kappa_0
        self.kappa_1 = kappa_1
        self.kappa_2 = kappa_2
        self.kappa_3 = kappa_3

# ------------ Fucntions ---------------------
@njit(Spline.class_type.instance_type(State.class_type.instance_type, State.class_type.instance_type))
def init_params(veh, goal):
    """
    Function to initialize Spline parameters
    Use heuristic to init (Nagy 2001), necessary for cubic splines

    Args:
        veh (State): initial vehicle state
        goal (State): goal vehicle state
    Returns:
        parameterized_control_input (Spline): estimated control params
    """


    # init
    sx_f = goal.sx
    sy_f = goal.sy
    theta_f = goal.theta
    kappa_0 = veh.kappa
    kappa_f = goal.kappa

    parameterized_control_input = Spline(0., 0., 0., 0., 0.)

    # heuristic for initiall guess from Nagy and Kelly 2001
    d = math.sqrt(sx_f**2 + sy_f**2)
    d_theta = abs(theta_f)
    s = d*(d_theta**2/5.0 + 1.0) + (2.0/5.0)*d_theta
    c = 0.0
    a = (6.0*theta_f/(s**2)) - (2*kappa_0/s) + (4*kappa_f/s)
    b = (3.0/(s**2)) * (kappa_0+kappa_f) + (6.0*theta_f/(s**3))

    si = 0.0
    # fill in estimated parameters
    parameterized_control_input.kappa_0 = kappa_0
    parameterized_control_input.kappa_3 = kappa_f
    parameterized_control_input.s = s

    parameterized_control_input.kappa_1 = (1.0/49.0)*(8.0*b*si - 8.0*b*s - 26.0*kappa_0 - kappa_f)
    parameterized_control_input.kappa_2 = 0.25*(kappa_f - 2.0*kappa_0 + 5.0*parameterized_control_input.kappa_1)
    # parameterized_control_input.kappa_1 = 0.5
    # parameterized_control_input.kappa_2 = 0.5
    return parameterized_control_input

@njit(State.class_type.instance_type(State.class_type.instance_type))
def speed_control_logic(veh_next):
    """
    Function to compute safe/feasible speed and curvature

    Args:
        veh_next (State): target next state
    Returns:
        veh_next (State): updated target next state
    """
    vcmd = abs(veh_next.v)
    kappa_next = veh_next.kappa

    # compute safe speed
    compare_v = (kappa_next-ASCL)/BSCL
    vcmd_max = max(VSCL, compare_v)

    # compute safe curvature
    compare_kappa = ASCL + (BSCL*vcmd)
    kmax_scl = min(KMAX, compare_kappa)

    # check if max curvatre for speed is exceeded
    if kappa_next >= kmax_scl:
        vcmd = SF * vcmd_max

    # update velocity command
    veh_next.v = vcmd

    return veh_next

@njit(State.class_type.instance_type(State.class_type.instance_type, State.class_type.instance_type, float64))
def response_to_control_inputs(veh, veh_next, dt):
    """ Function for computing the vehicles response to control inputs
        Mainly deals with delays associated with controls
        Also considers upper and lower safety bounds

        Args:
            veh (State): current vehicle state,
            veh_next (State): next vehicle state,
            dt (float): sampling time
        Returns:
            veh_next (State): updated next vehicle state
    """
    # variable lookup
    kappa = veh.kappa
    kappa_next = veh_next.kappa
    v = veh.v
    v_next = veh_next.v

    # compute curvature rate command
    kdot = (kappa_next - kappa)/dt

    # check against upper/lower bound on curvature rate
    kdot = min(kdot, DKMAX)
    kdot = max(kdot, DKMIN)

    # call speed control logic for safe speed
    veh_next = speed_control_logic(veh_next)

    # compute curvature at the next vehicle state
    kappa_next = kappa + kdot*dt

    # check upper/lower bound on curvature
    kappa_next = min(kappa_next, KMAX)
    kappa_next = max(kappa_next, KMIN)

    # compute acceleration command
    vdot = (v_next - v)/dt

    # check upper/lower bound on acceleration
    vdot = min(vdot, DVMAX)
    vdot = max(vdot, DVMIN)

    # compute velocity at next state
    veh_next.v = v + vdot*dt

    return veh_next

@njit(float64(Spline.class_type.instance_type, float64, float64))
def get_curvature_command(parameterized_control_input, v, t):
    """
    Function to calculate the curvature at the end of the spline

    Args:
        parameterized_control_input (Spline): current spline params
        v (float): current velocity
        t (float): time from start to end of the spline
    Returns:
        k_next_cmd (float): curvature at the end of the spline
    """

    # estimate arc length travelled for the time given, with initial arc length at 0
    st = v*t
    si = 0.0

    # cubic stable paths, found in McNaughton p76 and several others
    # notations are slightly different, the kappa_0~3 is equivalent
    # to p0~3 in McNaughton
    kappa_0 = parameterized_control_input.kappa_0
    kappa_1 = parameterized_control_input.kappa_1
    kappa_2 = parameterized_control_input.kappa_2
    kappa_3 = parameterized_control_input.kappa_3
    s = parameterized_control_input.s

    a = kappa_0
    b = (-0.5)*(-2*kappa_3 + 11.0*kappa_0 - 18.0*kappa_1 + 9.0*kappa_2)/(s-si)
    c = (4.5)*(-kappa_3 + 2.0*kappa_0 - 5.0*kappa_1 + 4.0*kappa_2)/((s-si)**2)
    d = (-4.5)*(-kappa_3 + kappa_0 - 3.0*kappa_1 + 3.0*kappa_2)/((s-si)**3)

    # get curvature at arc length st
    k_next_cmd = a + b*st + c*st**2 + d*st**3

    return k_next_cmd

@njit(float64(float64, float64, float64))
def get_velocity_command(v_goal, v, dt):
    """
    Function that updates the vehicle velocity based on velocity profile

    Args:
        v_goal (float): goal velocity
        v (float): current velocity
        dt (float): sampling time
    Returns:
        v_next_cmd (float): next commanded velocity
    """

    # TODO: change velocity profile to actual profile from sys id
    # yo
    accel = 50.0
    decel = -80.0
    v_next_cmd = 0.0
    if v < v_goal:
        v_next_cmd = v + accel*dt
    elif v > v_goal:
        v_next_cmd = v + decel*dt
    else:
        v_next_cmd = v

    return v_next_cmd


@njit(State.class_type.instance_type(State.class_type.instance_type, State.class_type.instance_type, Spline.class_type.instance_type, float64))
def motion_model(veh, goal, parameterized_control_input, dt):
    """
    Function that computes update to vehicle state

    Args:
        veh (State): current vehicle state
        goal (State): goal vehicle state
        parameterized_control_input (Spline): parameterized control input
        dt (float): sampling time
    Returns:
        veh_next (State): next vehicle state
    """


    # get motion model predictive horizon, assuming constant accel/decel
    horizon = 0
    if goal.v == 0 and veh.v == 0:
        # triangular velocity profile, use speed limit
        horizon = (2.0*parameterized_control_input.s)/VSCL
    else:
        # trapezoidal velocity profile
        horizon = (2.0*parameterized_control_input.s)/(veh.v+goal.v)

    v_goal = goal.v

    # init elapsed predicting time
    t = 0.0

    # doing this because numba doesn't know copy
    current_veh = State(0., 0., 0., 0., 0.)
    current_veh.sx = veh.sx
    current_veh.sy = veh.sy
    current_veh.theta = veh.theta
    current_veh.kappa = veh.kappa
    current_veh.v = veh.v
    # current_veh = copy.deepcopy(veh)

    veh_next = State(0., 0., 0., 0., 0.)

    while t < horizon:
        # get current state
        sx = current_veh.sx
        sy = current_veh.sy
        v = current_veh.v
        theta = current_veh.theta
        kappa = current_veh.kappa

        # change in x-position
        sx_next = sx + (v*math.cos(theta)*dt)
        veh_next.sx = sx_next

        # change in y-position
        sy_next = sy + (v*math.sin(theta)*dt)
        veh_next.sy = sy_next

        # change in orientation
        theta_next = theta + (v*kappa*dt)
        veh_next.theta = theta_next

        # get curvature command
        kappa_next = get_curvature_command(parameterized_control_input, v, t)
        veh_next.kappa = kappa_next

        # get velocity command
        v_next = get_velocity_command(v_goal, v, dt)
        veh_next.v = v_next

        # get acceleration command
        # not used on f110?
        # a_next_cmd = 0.0

        # estimate response
        veh_next = response_to_control_inputs(current_veh, veh_next, dt)

        # increment timestep
        t = t+dt

        # update current state
        # current_veh = copy.deepcopy(veh_next)
        current_veh = veh_next

    # return the state at the end of the trajectory
    return veh_next

@njit(boolean(State.class_type.instance_type, State.class_type.instance_type))
def check_convergence(veh_next, goal):
    """
    Function that checkes if the next state calculated is close enough to goal

    Args:
        veh_next (State): vehicle state to check
        goal (State): goal state
    Returns:
        convergence (bool): if algorithm has converged
    """

    # state errors
    sx_error = abs(veh_next.sx - goal.sx)
    sy_error = abs(veh_next.sy - goal.sy)
    theta_error = abs(veh_next.theta - goal.theta)
    if DEBUG_ERROR:
        print('errors: ', sx_error, sy_error, theta_error)
    # checks
    if sx_error < GENERAL_E and sy_error < GENERAL_E and theta_error < GENERAL_E:
        return True
    else:
        return False

@njit(Spline.class_type.instance_type(State.class_type.instance_type, State.class_type.instance_type, Spline.class_type.instance_type))
def opt_step(veh_next, goal, parameterized_control_input):
    """
    Function that calculates the Jacobian matrix and correct for next vehicle state

    Args:
        veh (State): current vehicle state
        veh_next (State): vehicle state at endpoint of spline
        goal (State): goal vehicle state
        parameterized_control_input (Spline): current spline params
    Returns:
        veh_next (State): corrected vehicle state for next iteration
    """

    # implementation of Kelly and Nagy, 2003, Reactive Nonholonomic Trajectory Generation
    # plus the Jacobian from McNaughton thesis

    # pre calc a, b, c, d from equally spaced knots for stable cubic paths, same as in get_curvature_command
    kappa_0 = parameterized_control_input.kappa_0
    kappa_1 = parameterized_control_input.kappa_1
    kappa_2 = parameterized_control_input.kappa_2
    kappa_3 = parameterized_control_input.kappa_3
    s = parameterized_control_input.s

    a = kappa_0
    b = (-0.5)*(-2*kappa_3 + 11.0*kappa_0 - 18.0*kappa_1 + 9.0*kappa_2)/s
    c = (4.5)*(-kappa_3 + 2.0*kappa_0 - 5.0*kappa_1 + 4.0*kappa_2)/(s**2)
    d = (-4.5)*(-kappa_3 + kappa_0 - 3.0*kappa_1 + 3.0*kappa_2)/(s**3)

    # init Jacobian, 3x3 because only p1/kappa_1, p2/kappa_2, and s taking deriv
    J = np.empty((3,3))

    # pre calc some vectors
    n = 8.
    k = np.arange(0., n+1.)
    # weight vector, 1 for first and last, 4 for even, 2 for odd
    w = np.array([1, 4, 2, 4, 2, 4, 2, 4, 1])
    # arc length vectors
    # not using this because numba doesn't like linspace?
    # s_vec = np.linspace(1./n, (n+1.)/n, n+1.).astype(np.float64)
    s_vec = np.array([0., 1./8., 2./8., 3./8., 4./8., 5./8., 6./8., 7./8., 1.])
    s_vec = s*s_vec
    s_vec_sq = s_vec**2
    s_vec_cube = s_vec**3
    # theta vector
    theta_vec = a*s_vec + b*s_vec**2/2 + c*s_vec**3/3 + d*s_vec**4/4
    # cos vec
    f_vec = np.cos(theta_vec)
    # sin vec
    g_vec = np.sin(theta_vec)
    # position x
    x_vec = np.multiply(w, f_vec)
    # position y
    y_vec = np.multiply(w, g_vec)
    # higher orders
    F2_vec = np.multiply(np.multiply(w, s_vec_sq), f_vec)
    G2_vec = np.multiply(np.multiply(w, s_vec_sq), g_vec)
    F3_vec = np.multiply(np.multiply(w, s_vec_cube), f_vec)
    G3_vec = np.multiply(np.multiply(w, s_vec_cube), g_vec)
    # summing for Jacobian
    F2 = np.sum(F2_vec)
    G2 = np.sum(G2_vec)
    F3 = np.sum(F3_vec)
    G3 = np.sum(G3_vec)
    f = f_vec[-1]
    g = g_vec[-1]
    # partial derivs of theta, equation (63) from kelly and nagy 2003
    dtheta_s = a + b*s + c*s**2 + d*s**3
    dtheta_p1 = s**2/2
    dtheta_p2 = s**3/3
    # fill in Jacobian
    J[0, 0] = -0.5*G2
    J[0, 1] = -(1/3)*G3
    J[0, 2] = f
    J[1, 0] = 0.5*F2
    J[1, 1] = (1/3)*F3
    J[1, 2] = g
    J[2, 0] = dtheta_p1
    J[2, 1] = dtheta_p2
    J[2, 2] = dtheta_s
    # update scheme, from McNaughton thesis (3.43)
    # delta between goal and predicted next state
    delta_sx = goal.sx - veh_next.sx
    delta_sy = goal.sy - veh_next.sy
    delta_theta = goal.theta - veh_next.theta
    delta_q = np.array([[delta_sx],[delta_sy],[delta_theta]])
    J_inv = LA.pinv(J)
    delta_param = np.dot(J_inv, delta_q)
    corrected_control_input = Spline(0., 0., 0., 0., 0.)
    corrected_control_input.kappa_0 = kappa_0
    corrected_control_input.kappa_1 = kappa_1 + STEP_GAIN*delta_param[0, 0]
    corrected_control_input.kappa_2 = kappa_2 + STEP_GAIN*delta_param[1, 0]
    corrected_control_input.kappa_3 = kappa_3
    corrected_control_input.s = s + delta_param[2, 0]

    return corrected_control_input

@njit(float64[:,:](Spline.class_type.instance_type))
def integrate_path(parameterized_control_input):
    """
    Function that integrates to find equally spaced points along the spline
    McNaughton thesis trapezoidal integration approach, (3.47, 3.48)

    Args:
        parameterized_control_input (Spline): spline params
    Returns:
        states (np.ndarray (Nx4)): full state of the vehicle at each interval
    """

    # convenience
    kappa_0 = float64(parameterized_control_input.kappa_0)
    kappa_1 = float64(parameterized_control_input.kappa_1)
    kappa_2 = float64(parameterized_control_input.kappa_2)
    kappa_3 = float64(parameterized_control_input.kappa_3)
    s = float64(parameterized_control_input.s)
    N = NUM_STEPS

    a = kappa_0
    b = (-0.5)*(-2*kappa_3 + 11.0*kappa_0 - 18.0*kappa_1 + 9.0*kappa_2)/s
    c = (4.5)*(-kappa_3 + 2.0*kappa_0 - 5.0*kappa_1 + 4.0*kappa_2)/(s**2)
    d = (-4.5)*(-kappa_3 + kappa_0 - 3.0*kappa_1 + 3.0*kappa_2)/(s**3)

    # init return state array
    states = np.zeros((N, 4), dtype=np.float64)

    # at each step states are: [x, y, theta, kappa]
    # start state is assumed all zero
    states[0, :] = np.zeros((1, 4), dtype=np.float64)
    delta_x_k = 0.
    delta_y_k = 0.
    x_k = 0.
    y_k = 0.
    prev_theta_k = 0.
    for k in range(1, N):
        # get arc length at step k
        s_k = s*float64(k)/float64(N)
        # get theta at step k
        theta_k = float64(a*s_k + b*s_k**2./2. + c*s_k**3./3. + d*s_k**4./4.)
        # get kappa at step k
        kappa_k = float64(a + b*s_k + c*s_k**2. + d*s_k**3.)
        # get position at step k
        delta_x_k = float64(delta_x_k*((k-1.)/k) + (np.cos(theta_k)+np.cos(prev_theta_k))/(2.*k))
        delta_y_k = float64(delta_y_k*((k-1.)/k) + (np.sin(theta_k)+np.sin(prev_theta_k))/(2.*k))
        # print('deltas', delta_x_k, delta_y_k)
        # x_k = float64(x_k + s_k*delta_x_k)
        # y_k = float64(y_k + s_k*delta_y_k)
        x_k = float64(s_k*delta_x_k)
        y_k = float64(s_k*delta_y_k)
        # print('state', x_k, y_k)
        # add to full state array
        states[k, :] = np.array([x_k, y_k, theta_k, kappa_k])
        # store theta for next step
        prev_theta_k = theta_k
    return states


# ----------------- Warm Start LUT --------------
@njit(Spline.class_type.instance_type(float64, float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:]))
def lookup(x, y, theta, lut_x, lut_y, lut_theta, lut):
    """
    Function that looks up the look up table for warm start
    range array loaded from .npz file
    Only 3d for now

    Args:
        x, y, theta (float): delta from start state
    Returns:
        initial_guess (Spline): initial guess for the spline parameters
    """
    x_idx = np.abs(lut_x - x).argmin()
    y_idx = np.abs(lut_y - y).argmin()
    theta_idx = np.abs(lut_theta - theta).argmin()

    # params should be stored as [s, k0, k1, k2, k3]
    params = lut[x_idx, y_idx, theta_idx]

    return Spline(params[0], params[1], params[2], params[3], params[4])


# ------------------ Standalone Traj Generators --------------
@njit(Spline.class_type.instance_type(float64, float64, float64, float64, float64, float64))
def trajectory_generator_coldstart(goal_x, goal_y, goal_theta, goal_v, goal_kappa, start_kappa):
    convergence = False
    status = True
    goal = State(goal_x, goal_y, goal_theta, goal_kappa, goal_v)
    veh = State(0., 0., 0., start_kappa, goal_v)

    parameterized_control_input = init_params(veh, goal)
    # iter counter init
    iteration = 0
    # time stepsize
    dt = STEP_SIZE
    # trajectory optimization loop
    while ((not convergence) or (iteration < MIN_ITER)) and (iteration < MAX_ITER):
        # run motion model to get end point of trajectory
        veh_next = motion_model(veh, goal, parameterized_control_input, dt)
        # determine convergence
        convergence = check_convergence(veh_next, goal)
        # if not converged, run trajectory optimization
        if convergence:
            break
        else:
            parameterized_control_input = opt_step(veh_next, goal, parameterized_control_input)
            iteration = iteration + 1
            if parameterized_control_input.s < 0 or parameterized_control_input.s > 4*np.sqrt(goal_x**2+goal_y**2):
                status = False
                break

    if DEBUG_CONVERGENCE:
        if iteration == MAX_ITER:
            print("Optimization stopped at max iteration.")
        elif not status:
            print("Optimization failed, negative or arc length too long.")
        else:
            # print("Converged at " + str(iteration) + "th iteration.")
            print('converged')
    if not status:
        parameterized_control_input.s = -1
        parameterized_control_input.kappa_0 = -1
        parameterized_control_input.kappa_1 = -1
        parameterized_control_input.kappa_2 = -1
        parameterized_control_input.kappa_3 = -1

    return parameterized_control_input

@njit(Spline.class_type.instance_type(float64, float64, float64, float64, float64, float64, Spline.class_type.instance_type))
def trajectory_generator_warmstart(goal_x, goal_y, goal_theta, goal_v, goal_kappa, start_kappa, param_init):
    convergence = False
    status = True
    goal = State(goal_x, goal_y, goal_theta, goal_kappa, goal_v)
    veh = State(0., 0., 0., start_kappa, goal_v)

    parameterized_control_input = param_init
    # iter counter init
    iteration = 0
    # time stepsize
    dt = STEP_SIZE
    # trajectory optimization loop
    while ((not convergence) or (iteration < MIN_ITER)) and (iteration < MAX_ITER):
        # run motion model to get end point of trajectory
        veh_next = motion_model(veh, goal, parameterized_control_input, dt)
        # determine convergence
        convergence = check_convergence(veh_next, goal)
        # if not converged, run trajectory optimization
        if convergence:
            break
        else:
            parameterized_control_input = opt_step(veh_next, goal, parameterized_control_input)
            iteration = iteration + 1
            if parameterized_control_input.s < 0 or parameterized_control_input.s > 4*np.sqrt(goal_x**2+goal_y**2):
                status = False
                break

    if DEBUG_CONVERGENCE:
        if iteration == MAX_ITER:
            print("Optimization stopped at max iteration.")
        elif not status:
            print("Optimization failed, negative or arc length too long.")
        else:
            # print("Converged at " + str(iteration) + "th iteration.")
            print('converged')
    if not status:
        parameterized_control_input.s = -1
        parameterized_control_input.kappa_0 = -1
        parameterized_control_input.kappa_1 = -1
        parameterized_control_input.kappa_2 = -1
        parameterized_control_input.kappa_3 = -1

    return parameterized_control_input


@njit(Spline.class_type.instance_type(float64, float64, float64, float64, float64, float64, float64[:], float64[:], float64[:], float64[:,:,:,:], boolean))
def trajectory_generator(goal_x, goal_y, goal_theta, goal_v, goal_kappa, start_kappa, lut_x, lut_y, lut_theta, lut, warm_start=True):
    """
    Function that generates parameterized control input with given goal (in car frame)
    TODO: consider actual current vehicle state, now assuming all zero, and same vel as goal
    TODO: maybe need starting curvature?

    Args:
        all_args (float): goal state
    Returns:
        parameterized_control_input (Spline): generated spline params
    """

    # if DEBUG_PLOT:
    #     plt.ion()
    #     plt.show()

    # var init
    convergence = False
    status = True
    goal = State(goal_x, goal_y, goal_theta, goal_kappa, goal_v)
    veh = State(0., 0., 0., start_kappa, goal_v)
    # check if using warm start lut
    if warm_start:
        parameterized_control_input = lookup(goal_x, goal_y, goal_theta, lut_x, lut_y, lut_theta, lut)
    else:
        # heuristics init for spline params
        parameterized_control_input = init_params(veh, goal)
    # iter counter init
    iteration = 0
    # time stepsize
    dt = STEP_SIZE
    # trajectory optimization loop
    while ((not convergence) or (iteration < MIN_ITER)) and (iteration < MAX_ITER):
        # if DEBUG_PLOT:
        #     plt.clf()
        #     states = integrate_path(parameterized_control_input)
        #     x = states[:, 0]
        #     y = states[:, 1]
        #     plt.plot(x, y)
        #     plt.xlim(0, 20)
        #     plt.ylim(-10, 10)
        #     plt.draw()
        #     plt.pause(0.0001)


        # if iteration%10 == 0:
            # print('Trajectory optimization iteration: ' + str(iteration))
            # print(parameterized_control_input.kappa_0, parameterized_control_input.kappa_1, parameterized_control_input.kappa_2, parameterized_control_input.kappa_3, parameterized_control_input.s)
        # run motion model to get end point of trajectory
        veh_next = motion_model(veh, goal, parameterized_control_input, dt)
        # determine convergence
        convergence = check_convergence(veh_next, goal)
        # if not converged, run trajectory optimization
        if convergence:
            break
        else:
            parameterized_control_input = opt_step(veh_next, goal, parameterized_control_input)
            iteration = iteration + 1
            if parameterized_control_input.s < 0 or parameterized_control_input.s > 4*np.sqrt(goal_x**2+goal_y**2):
                status = False
                break

    if DEBUG_CONVERGENCE:
        if iteration == MAX_ITER:
            print("Optimization stopped at max iteration.")
        elif not status:
            print("Optimization failed, negative or arc length too long.")
        else:
            # print("Converged at " + str(iteration) + "th iteration.")
            print('converged')

    return parameterized_control_input

def save_np(delta_x, delta_y, delta_theta, lut_new):
    # only 3d now
    np.savez_compressed('lut_finer.npz', x=delta_x, y=delta_y, theta=delta_theta, lut=lut_new)


@njit(float64[:,:,:,:](int32, float64[:], float64[:], float64[:], float64, float64))
def build_lut(n, delta_x, delta_y, delta_theta, kappa, v_max):
    """
    Function that builds and saves a look up table for warm start if there isn't one
    The LUT has 5 dimensions: delta_x, delta_y, delta_theta, initial_curvature, constant_velocity

    McNaughton thesis has a sampling scheme:
    steps: 16
    kappa_0, kappa_1: [-0.19, 0.19]
    delta_x: [1, 50]
    delta_y: [-50, 50]
    delta_theta: [-pi/2, pi/2]

    we could also add one for velocity
    v: [1, 10] # TODO ?

    In the lookup table, the order of indices is: [x, y, theta, kappa_0, kappa_1, v]

    Args:
        n (int): discretization steps for each axis
        delta_x/y/theta (float[]): range of delta states
        kappa, v_max (float): curvature for start/goal, and speed limit on the traj

    Retruns:
        lut_new (float[n x n x n x 5]): the 3d look up table

    """

    # check if file already exist
    # if os.path.exists('lut.npz'):
    #     print('LUT already built, skipping...')
    #     return True

    # if no existing lut


    # 3D lut
    n = int64(n)
    lut_new = np.zeros((n, n, n, int64(5)))
    for x_idx in range(n):
        for y_idx in range(n):
            for theta_idx in range(n):
                if DEBUG_BUILD_LUT:
                    print('Finding solution for:', delta_x[x_idx], delta_y[y_idx], delta_theta[theta_idx])
                # TODO: real ugly here, half of args not even used
                current_spline = trajectory_generator(delta_x[x_idx], delta_y[y_idx], delta_theta[theta_idx], v_max, kappa, kappa, delta_x, delta_y, delta_theta, lut_new, warm_start=False)
                if DEBUG_BUILD_LUT:
                    print('Solution returned:', current_spline.s, current_spline.kappa_0, current_spline.kappa_1, current_spline.kappa_2, current_spline.kappa_3)
                # if success:
                params_list = [current_spline.s, current_spline.kappa_0, current_spline.kappa_1, current_spline.kappa_2, current_spline.kappa_3]
                # else:
                #     params_list = [-1]*5
                lut_new[x_idx, y_idx, theta_idx] = params_list

    return lut_new


# util functions for sampling the costmap in parallel
# looping over samples, and integrate the trajs in parallel
# @njit(float64[:,:](float64[:,:]), parallel=True)
@njit(float64[:,:](float64[:,:]))
def integrate_parallel(curvature_list):
    """
    Function that integrates multiple to find trajectories in parallel
    each trajectory will be N_samplesx4 array

    Args:
        curvature_list (Spine[]): List of parameterized control input
    Returns:
        points_list ((N*N_samples)x4 ndarray): List of integrated points on the trajs from the list
    """

    points_list = np.empty((curvature_list.shape[0]*NUM_STEPS, 4))
    for i in range(curvature_list.shape[0]):
        s = float64(curvature_list[i, 0])
        k0 = float64(curvature_list[i, 1])
        k1 = float64(curvature_list[i, 2])
        k2 = float64(curvature_list[i, 3])
        k3 = float64(curvature_list[i, 4])
        current_spline = Spline(s, k0, k1, k2, k3)
        # this returns a trajectory as Nx4 array
        points = integrate_path(current_spline)
        points_list[i*NUM_STEPS:i*NUM_STEPS+NUM_STEPS, :] = points
    return points_list


# sampling
@njit(boolean(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64, int64))
def sample(traj, env_layer, static_layer, dynamic_layer, origin_x, origin_y, map_resolution, map_width):
    """
    Function for F1/10 occupancy grid, three layers, returns True for not free, and False otherwise

    Args:
        traj (Nx4 ndarray): states after integration
        env_layer, static_layer, dynamic_layer (N, ndarray): costmaps
        origin_x, origin_y, map_resolution, map_width (float): map info

    Returns:
        free (bool): whether trajectory is in free space
    """
    static_thresh = 50.
    free = True
    traj_x = traj[:, 0]
    traj_y = traj[:, 1]
    col = ((traj_x - origin_x)/map_resolution).astype(np.intp)
    row = ((traj_y - origin_y)/map_resolution).astype(np.intp)
    # advanced indexing not support by numba
    # free = (np.all(env_layer[row, col] == 0)) and (np.all(dynamic_layer[row, col] == 0)) and (np.all(static_layer[row, col] <= static_thresh))
    for i in prange(col.shape[0]):
        if (env_layer[row[i], col[i]] != 0) or (dynamic_layer[row[i], col[i]] != 0) or (static_layer[row[i], col[i]] >= static_thresh):
            free = False
    return free

@njit(boolean(float64[:,:], float64[:,:], float64, float64, float64))
def sample_map_only(traj, env_layer, origin_x, origin_y, map_resolution):
    """
    Function for F1/10 occupancy grid, one layers, returns True for not free, and False otherwise

    Args:
        traj (Nx4 ndarray): states after integration
        env_layer, static_layer, dynamic_layer (N, ndarray): costmaps
        origin_x, origin_y, map_resolution, map_width (float): map info

    Returns:
        free (bool): whether trajectory is in free space
    """
    free = True
    traj_x = traj[:, 0]
    traj_y = traj[:, 1]
    row = ((traj_y - origin_y)/map_resolution).astype(np.intp)
    col = ((traj_x - origin_x)/map_resolution).astype(np.intp)
    # advanced indexing not support by numba
    # free = (np.all(env_layer[row, col] == 0)) and (np.all(dynamic_layer[row, col] == 0)) and (np.all(static_layer[row, col] <= static_thresh))
    for i in prange(col.shape[0]):
        if (env_layer[row[i], col[i]] != 0):
            free = False
    return free



# looping over samples, return the indices(rows) that are free
@njit(boolean[:](float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64, int64))
def sample_parallel(traj_list, env_layer, static_layer, dynamic_layer, origin_x, origin_y, map_resolution, map_width):
    """
    Function for checking trajectories against F1/10 occupancy grid, return indices (row) of trajectories in free space

    Args:
        traj_list ((N*N_samples)x4 ndarray): list of points on trajectories
        env_layer, static_layer, dynamic_layer (MxM ndarray): costmap layers
        origin_x, origin_y, map_resolution, map_width (float): map meta data

    Returns:
        free_traj_list (N, ndarray): row
    """
    static_thresh = 50.
    num_traj = int(traj_list.shape[0]/NUM_STEPS)
    free_traj_list = np.empty((num_traj)).astype(boolean)
    for i in prange(num_traj):
        free = sample(traj_list[i*NUM_STEPS:i*NUM_STEPS+NUM_STEPS, :], env_layer, static_layer, dynamic_layer, origin_x, origin_y, map_resolution, map_width)
        free_traj_list[i] = free
        # if free:
            # free_traj_list.append(int64(i))

    # return np.array(free_traj_list)
    return free_traj_list

@njit(boolean[:](float64[:,:], float64[:,:], float64, float64, float64))
def sample_parallel_map_only(traj_list, env_layer, origin_x, origin_y, map_resolution):
    """
    Function for checking trajectories against F1/10 occupancy grid, return indices (row) of trajectories in free space

    Args:
        traj_list ((N*N_samples)x4 ndarray): list of points on trajectories
        env_layer, static_layer, dynamic_layer (MxM ndarray): costmap layers
        origin_x, origin_y, map_resolution, map_width (float): map meta data

    Returns:
        free_traj_list (N, ndarray): row
    """
    static_thresh = 50.
    num_traj = int(traj_list.shape[0]/NUM_STEPS)
    free_traj_list = np.empty((num_traj)).astype(boolean)
    for i in prange(num_traj):
        free = sample_map_only(traj_list[i*NUM_STEPS:i*NUM_STEPS+NUM_STEPS, :], env_layer, origin_x, origin_y, map_resolution)
        free_traj_list[i] = free
        # if free:
            # free_traj_list.append(int64(i))

    # return np.array(free_traj_list)
    return free_traj_list



# -------------------- Plotting ---------------------
def plot_traj(parameterized_control_input):
    """
    Function that plots the generated spline


    Args:
        parameterized_control_input (Spline): spline params
    Returns:
    """

    states = integrate_path(parameterized_control_input)
    x = states[:, 0]
    y = states[:, 1]
    plt.xlim(0, 5)
    plt.ylim(-5, 5)
    plt.plot(x, y)
    plt.show()

def plot_traj_list_parallel(params_list):
    # plt.xlim(0, 5)
    # plt.ylim(-5, 5)
    plt.axis('equal')
    # start = time.time()
    states_list = integrate_parallel(params_list)
    # print('Integration time: ' + str(time.time()-start))
    x = states_list[:, 0]
    y = states_list[:, 1]
    plt.scatter(x, y, color='blue', s=0.01)
    # for param in params_list:
    #     states = integrate_path(param)
    #     x = states[:, 0]
    #     y = states[:, 1]
    #     plt.plot(x, y, color='blue')
    plt.show()

def plot_traj_list(param_list):
    start = time.time()
    for param in param_list:
        states = integrate_path(param)
    print('Integration time: ' + str(time.time()-start))


    return


# def main():
#     # build sampling ranges
#     n = 16
#     # note that kappa used twice for both initial and final curvature
#     delta_x = np.linspace(0.5, 10, n)
#     delta_y = np.linspace(-10., 10., n)
#     delta_theta = np.linspace(-np.pi/2, np.pi/2, n)
#     kappa = np.linspace(-0.19, 0.19, n)
#     v = np.linspace(1, 10, n)

#     # use constant v and kappa
#     v_max = v[-1]
#     kappa_const = 0.

#     # warm start
#     lut_loaded = False
#     lut_all = None
#     if os.path.exists('lut_latest.npz'):
#         if not REBUILD_LUT:
#             print('Loading LUT...')
#             try:
#                 lut_all = np.load('lut_latest.npz')
#             except:
#                 print('LUT loading failed, exiting...')
#                 sys.exit()
#         else:
#             lut_new = build_lut(n, delta_x, delta_y, delta_theta, kappa_const, v_max)
#             save_np(delta_x, delta_y, delta_theta, lut_new)
#     else:
#         lut_new = build_lut(n, delta_x, delta_y, delta_theta, kappa_const, v_max)
#         save_np(delta_x, delta_y, delta_theta, lut_new)

#     lut_all = np.load('lut_comp.npz')

#     # load table and indices into arrays
#     lut_x = lut_all['x']
#     lut_y = lut_all['y']
#     lut_theta = lut_all['theta']
#     lut = lut_all['lut']

#     # load example cost map
#     costmap_all = np.load('example_costmap.npz')
#     env_layer = costmap_all['env']
#     static_layer = costmap_all['stat']
#     dynamic_layer = costmap_all['dyn']
#     map_resolution = 0.1
#     origin_x = -51.224998
#     origin_y = -51.224998
#     map_width = 1024

#     goal_x = 0.0
#     goal_y = 0.0
#     goal_y_list = np.linspace(0.0, 8.0, 100)
#     goal_theta = np.pi/2.0
#     goal_theta_list = np.linspace(-np.pi/3, np.pi/3, 20)
#     goal_v = 10.0
#     goal_v_list = np.linspace(1, 30, 3)
#     goal_kappa = 0.0
#     start_kappa = 0.0

#     # param_list = List(Spline.class_type.instance_type, reflected=True)
#     param_list = np.empty((goal_y_list.shape[0], 5))
#     # param_list = []
#     # different goal_y

#     # iter_length = 500
#     # lookup_time = 0
#     # int_time = 0
#     # sample_time = 0

#     # for iteration in range(iter_length):
#         # print('Running iteration ' + str(iteration))
#         # start_time = time.time()
#     best_param = None
#     for i, goal_y in enumerate(goal_y_list):
#         # param = trajectory_generator_coldstart(goal_x, goal_y, goal_theta, goal_v, goal_kappa, 0.39)
#         param = lookup(goal_x, goal_y, goal_theta, lut_x, lut_y, lut_theta, lut)
#         # if lookup_param.s > 0 and lookup_param.s < 4*(math.sqrt(goal_x**2+goal_y**2)):
#         #     best_param = lookup_param
#         # if best_param is not None:
#         #     param = trajectory_generator_warmstart(goal_x, goal_y, goal_theta, goal_v, goal_kappa, start_kappa, best_param)
#         # else:
#         #     param = trajectory_generator(goal_x, goal_y, goal_theta, goal_v, goal_kappa, start_kappa, lut_x, lut_y, lut_theta, lut, warm_start=False)

#         # current_states = integrate_path(param)
#         # if math.sqrt((current_states[-1, 0]-goal_x)**2+(current_states[-1, 1]-goal_y)**2) <= 0.005:
#         #     best_param = param
#         # param = lookup(goal_x, goal_y, goal_theta, lut_x, lut_y, lut_theta, lut)
#         # print(goal_y)
#         # print(param.s, param.kappa_0, param.kappa_1, param.kappa_2, param.kappa_3)
#         param_list[i,:] = np.array([param.s, param.kappa_0, param.kappa_1, param.kappa_2, param.kappa_3])

#         # print('Converged in ' + str(time.time() - start_time) + ' seconds.')
#         # param_list.append(param)

#     # print('Lookup time: ' + str(time.time()-start_time))
#     # lookup_time = lookup_time + (time.time()-start_time)
#     # print(param.__dict__)
#     # param_arr = np.array(param_list)
#     plot_traj_list_parallel(param_list)
    # plot_traj_list(param_list)

    #     start = time.time()
    #     states_list = integrate_parallel(param_list)
    #     int_time = int_time + (time.time()-start)
    #     # print('Integration time: ' + str(time.time()-start))

    #     sample_start_time = time.time()
    #     free_traj_list = sample_parallel(states_list, env_layer, static_layer, dynamic_layer, origin_x, origin_y, map_resolution, map_width)
    #     # print('Sampling time: ' + str(time.time()-sample_start_time))
    #     sample_time = sample_time + (time.time() - sample_start_time)
    # avg_lookup_time = lookup_time / iter_length
    # avg_int_time = int_time / iter_length
    # avg_sample_time = sample_time / iter_length
    # print('Over ' + str(iter_length) + ' iteration, average lookup time: ' + str(avg_lookup_time) + ', average integration time: ' + str(avg_int_time) + ', average sampling time: ' + str(avg_sample_time))

    # param_list = []
    # # different theta
    # goal_y = 0.0
    # start_time = time.time()
    # for goal_theta in goal_theta_list:

    #     param = trajectory_generator(goal_x, goal_y, goal_theta, goal_v, goal_kappa, start_kappa, lut_x, lut_y, lut_theta, lut, warm_start=True)
    #     param_list.append(param)
    # # print(param.__dict__)
    # print(time.time() - start_time)
    # plot_traj_list(param_list)

# if __name__ == '__main__':
   # main()
