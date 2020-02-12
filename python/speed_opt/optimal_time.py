import numpy as np
from numba import njit
import cvxpy as cv
import mosek
from scipy.integrate import ode, odeint
from rk6 import odeintRK6
import argparse
import sys
import zmq
import time


# vehicle parameters
mass = 5            # vehicle mass [kg]
Wf = .5             # percent of weight on front tires
muf = .8            # coefficient of friction between tires and ground
gravity = 9.81      # gravity constant (m/s^2)

@njit(cache=True)
def define_params(mass, Wf, muf, gravity):
    """
    params derived from vehicle config
    """

    params = {}
    Fmax = muf*mass*gravity
    params['mass'] = mass
    params['Fmax'] = Fmax
    params['Flongmax'] = Fmax*Wf
    return params

@njit(cache=True)
def define_path(x, y):
    """
    calculate s, s_prime, s_dprime using way points
    """

    num_wpts = np.size(x)
    # print('no of way points: {}'.format(num_wpts))

    theta = np.linspace(0, 1, num_wpts)
    dtheta = 1/(num_wpts-1)
    S = np.array([x, y])

    S_middle = np.zeros([2,num_wpts-1])
    S_prime = np.zeros([2,num_wpts-1])
    S_dprime= np.zeros([2,num_wpts-1])

    for j in range(num_wpts-1):

        S_middle[:,j] = (S[:,j] + S[:,j+1])/2
        S_prime[:,j] = (S[:,j+1] - S[:,j])/dtheta

        if j==0:
            S_dprime[:,j] = (S[:,j]/2 - S[:,j+1] + S[:,j+2]/2)/(dtheta**2)
        elif j==1 or j==num_wpts-3:
            S_dprime[:,j] = (S[:,j-1] - S[:,j] - S[:,j+1] + S[:,j+2])/2/(dtheta**2)
        # elif j==num_wpts-3:
        #     S_dprime[:,j] = (S[:,j-2] - S[:,j-1] - S[:,j] + S[:,j+1])/(2*dtheta**2)
        elif j==num_wpts-2:
            S_dprime[:,j] = (S[:,j-1]/2 - S[:,j] + S[:,j+1]/2)/(dtheta**2)
        else:
            S_dprime[:,j] = (- 5/48*S[:,j-2] + 13/16*S[:,j-1] - 17/24*S[:,j] - 17/24*S[:,j+1] + 13/16*S[:,j+2] - 5/48*S[:,j+3])/(dtheta**2)

    path = {
            'theta': theta,
            'dtheta': dtheta,
            'S': S,
            'S_middle': S_middle,
            'S_prime': S_prime,
            'S_dprime': S_dprime,
            }

    return path

@njit(cache=True)
def dynamics(phi, params):
    """
    dynamics (non-linear)
    """

    mass = params['mass']

    R = np.zeros((2,2))
    R[0,0] = np.cos(phi)
    R[0,1] = -np.sin(phi)
    R[1,0] = np.sin(phi)
    R[1,1] = np.cos(phi)

    M = np.zeros((2,2))
    M[0,0] = mass
    M[1,1] = mass

    C = np.zeros((2,2))
    d = np.zeros((2))

    return R, M, C, d

@njit(cache=True)
def dynamics_cvx(S_prime, S_dprime, params):
    """
    dynamics (convexified)
    """

    phi = np.arctan2(S_prime[1],S_prime[0])

    R, M, C, d = dynamics(phi, params)
    C = np.dot(M, S_dprime) + np.dot(C, S_prime)
    M = np.dot(M, S_prime)
    return R, M, C, d


def friction_circle(Fmax):
    t = np.linspace(0, 2*np.pi, num=100)
    x = Fmax*np.cos(t)
    y = Fmax*np.sin(t)
    return x, y

@njit(cache=True)
def diffequation(t, x, u, R, M, C, d):
    """
    write as first order ode
    """
    x0dot = x[2:]
    # inefficient
    # x1dot = np.dot(np.linalg.inv(M), np.dot(R, u) - np.dot(C, x[2:]) - d)
    x1dot = np.linalg.solve(M, np.dot(R, u) - np.dot(C, x[2:]) - d)
    return np.concatenate([x0dot, x1dot], axis=0)


# simulate control inputs
@njit(cache=True)
def simulate(b, a, u, path, params):
    """
    integrate using ode solver, rk6
    """

    theta = path['theta']
    dtheta = path['dtheta']
    S = path['S']
    S_prime = path['S_prime']
    S_dprime = path['S_dprime']
    num_wpts = theta.size

    # initialize position, velocity
    x, y = np.zeros([num_wpts]), np.zeros([num_wpts])
    x[0], y[0] = S[0,0], S[1,0]
    vx, vy = np.zeros([num_wpts]), np.zeros([num_wpts])
    # vx[0], vy[0] = S_prime[0,0], S_prime[1,0]

    # calculate time for each index
    bsqrt = np.sqrt(b)
    dt = 2*dtheta/(bsqrt[0:num_wpts-1]+bsqrt[1:num_wpts])
    t = np.zeros([num_wpts])
    for j in range(1, num_wpts):
        t[j] = t[j-1] + dt[j-1]
    # print('The optimal time to traverse is {:.4f}s'.format(t[-1]))

    # integrate
    # print('using Runge Kutta sixth order integration')
    for j in range(num_wpts-1):
        phi = np.arctan2(S_prime[1,j],S_prime[0,j])
        R, M, C, d = dynamics(phi, params)
        odesol = odeintRK6(diffequation, [x[j], y[j], vx[j], vy[j]], [t[j], t[j+1]],
                        args=(u[:,j], R, M, C, d))
        x[j+1], y[j+1], vx[j+1], vy[j+1] = odesol[-1,:]

    return np.sqrt(vx**2+vy**2), np.sum(dt)


def optimize(path, params):
    """
    main function to optimize trajectory
    solves convex optimization
    """

    theta = path['theta']
    dtheta = path['dtheta']
    S = path['S']
    S_prime = path['S_prime']
    S_dprime = path['S_dprime']
    num_wpts = theta.size

    # opt vars
    A = cv.Variable((num_wpts-1))
    B = cv.Variable((num_wpts))
    U = cv.Variable((2, num_wpts-1))

    cost = 0
    constr = []

    # no constr on A[0], U[:,0], defined on mid points
    
    # TODO: constr could be vectorized?

    constr += [B[0] == 0]
    for j in range(num_wpts-1):

        cost += 2*dtheta*cv.inv_pos(cv.power(B[j],0.5) + cv.power(B[j+1],0.5))

        R, M, C, d = dynamics_cvx(S_prime[:,j], S_dprime[:,j], params)
        constr += [R*U[:,j] == M*A[j] + C*((B[j] + B[j+1])/2) + d]
        constr += [B[j] >= 0]
        constr += [cv.norm(U[:,j],2) <= params['Fmax']]
        constr += [U[0,j] <= params['Flongmax']]
        constr += [B[j+1] - B[j] == 2*A[j]*dtheta]

    # problem_define_time = time.time()
    problem = cv.Problem(cv.Minimize(cost), constr)
    # problem_define_done = time.time()
    solution = problem.solve(solver=cv.MOSEK, verbose=False)
    # problem_solve_done = time.time()
    B, A, U = B.value, A.value, U.value
    B = abs(B)

    vopt, topt = simulate(B, A, U, path, params)
    # cvx_simulate_done_time = time.time()
    # print('Problem defn time: ' + str(problem_define_done - problem_define_time) + ', problem solve time: ' + str(problem_solve_done - problem_define_done) + ', cvx sim time: ' + str(cvx_simulate_done_time - problem_solve_done))
    return B, A, U, vopt, topt


def solve(x, y):

    path = define_path(x, y)
    params = define_params(mass, Wf, muf, gravity)
    B, A, U, vopt, topt = optimize(path=path, params=params, plot_results=False)

    return x, y, vopt, topt
