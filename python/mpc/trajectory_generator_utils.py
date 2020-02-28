from __future__ import print_function
# import sys
# sys.path.append('./mpc')
import numpy as np
from numba import njit
from numba import float64, int64, prange, boolean
from numba.typed import List
from mpc import AStrajectory_generator as trajectory_generator
from mpc import pure_pursuit_utils
# from robust import linear_chi_square
# from robust import robust_utils
# import GJK
import time
# import trajectory_generator

WAYPOINT_SPEED = 4.
N_SHIFT = 5
N_CULL = int(trajectory_generator.NUM_STEPS / 2)
N_KNOT = 3
T_INTERP = 0.05

CAR_WIDTH = 0.31*1.1
CAR_LENGTH = 0.58*1.1
T_CD_HORZ_SHORT = 1.
T_CD_HORZ_LONG = 3.
DISCOUNT = np.ascontiguousarray(np.insert(np.cumprod(0.9*np.ones(int(T_CD_HORZ_LONG/T_INTERP))), 0, 1.0))
CD_CONST = 1.
CS_CONST = .1
S_CONST = 1.
MIN_BOUNDARY_DIST = 0.5*np.sqrt(CAR_WIDTH**2+CAR_LENGTH**2)


@njit(fastmath=False, cache=True)
def fill_occgrid_map(map_array, inflation):
    map_array = np.ascontiguousarray(map_array)
    env_layer = np.zeros((map_array.shape))
    inflation_range = np.arange(-inflation, inflation+1, 1)
    for r in range(map_array.shape[0]):
        for c in range(map_array.shape[1]):
            if map_array[r, c] <= 250.0:
                env_layer[r, c] = 1.0
                for offset_r in inflation_range:
                    for offset_c in inflation_range:
                        current_row = r+offset_r
                        current_col = c+offset_c
                        if (current_row >= 0 and current_row < map_array.shape[0] and current_col >= 0 and current_col < map_array.shape[1]):
                            env_layer[current_row, current_col] = 1.0
    return env_layer

@njit(fastmath=False, cache=True)
def fill_occgrid(ranges, angles, rot, trans, inflation, static_thresh, env_layer, static_layer, origin_x, origin_y, map_resolution, map_height, map_width):
    env_layer = np.ascontiguousarray(env_layer)
    static_layer = np.ascontiguousarray(static_layer)
    layers = np.empty((env_layer.shape[0], env_layer.shape[1], 2))
    layers[:, :, 0] = static_layer
    valid_angles = angles[~(np.isnan(ranges) | np.isinf(ranges))]
    valid_ranges = ranges[~(np.isnan(ranges) | np.isinf(ranges))]
    # polar to cartesian
    x_local = valid_ranges*np.cos(valid_angles)
    y_local = valid_ranges*np.sin(valid_angles)
    # homogeneous
    homo_xy = np.vstack((x_local, y_local, np.zeros(x_local.shape[0]), np.ones(x_local.shape[0])))
    # rotation
    rotated_xy = np.dot(np.ascontiguousarray(rot), np.ascontiguousarray(homo_xy))
    rotated_xy = rotated_xy / rotated_xy[3, :]
    # translation
    translated_xy = rotated_xy[0:3, :] + trans
    # coord to row col
    cols = ((translated_xy[0,:]-origin_x)/map_resolution).astype(np.int64)
    rows = ((translated_xy[1,:]-origin_y)/map_resolution).astype(np.int64)
    # add inflation
    inflation_range = np.arange(-inflation, inflation+1, 1)
    for offset_r in inflation_range:
        for offset_c in inflation_range:
            current_rows = rows + offset_r
            current_cols = cols + offset_c
            valid_rows = current_rows[(current_rows>=0)&(current_rows<map_height)&(current_cols>=0)&(current_cols<map_width)]
            valid_cols = current_cols[(current_rows>=0)&(current_rows<map_height)&(current_cols>=0)&(current_cols<map_width)]
            for i in range(valid_rows.shape[0]):
                layers[valid_rows[i], valid_cols[i], 1] = 100
            # advanced indexing not implemented in numba
            # layers[valid_rows, valid_cols, 1] = 100

    # filtering
    valid_rows = rows[(rows>=0) & (rows<map_height) & (cols>=0) & (cols<map_width)]
    valid_cols = cols[(rows>=0) & (rows<map_height) & (cols>=0) & (cols<map_width)]
    for i in range(layers.shape[0]):
        for j in range(layers.shape[1]):
            if env_layer[i, j] > 0 and layers[i, j, 1] > 0:
                layers[i, j, 1] = 0
            if layers[i, j, 1] > 0 and static_layer[i, j] < 100:
                layers[i, j, 0] = static_layer[i, j] + 1
            if env_layer[i, j] == 0 and layers[i, j, 1] == 0 and static_layer[i, j] > 0:
                layers[i, j, 0] = static_layer[i, j] - 1
            if layers[i, j, 1] > 0 and static_layer[i, j] >= static_thresh:
                layers[i, j, 1] = 0
            if static_layer[i, j] >= static_thresh and env_layer[i, j] > 0:
                layers[i, j, 0] = 0
    return layers

@njit(fastmath=False, cache=True)
def create_grid(goal_x, goal_y, goal_res, goal_theta_range, rot, rot_cw, rot_ccw, ray_res, max_lateral, max_longitudinal, env_layer, static_layer, dynamic_layer, origin_x, origin_y, map_resolution, static_thresh, trans, pose_rot):
    # goal x y theta in car frame
    # returns Nx3 list of goals [[x, y, theta], ...]
    # need maximum ranges if doesn't hit occgrid

    rot_orig = np.ascontiguousarray(rot)
    rot = np.ascontiguousarray(rot[0:3, 0:3])
    rot_cw = np.ascontiguousarray(rot_cw[0:3, 0:3])
    rot_ccw = np.ascontiguousarray(rot_ccw[0:3, 0:3])

    # unit vec of goal pose in global frame
    x_axis = np.ascontiguousarray(np.array([[1.], [0.], [0.]]))
    vec_rotated = np.dot(rot, x_axis)
    # vec_rotated = vec_rotated / vec_rotated[3]
    vec_rotated = vec_rotated / np.linalg.norm(vec_rotated)
    # vec_translated = vec_rotated + np.array([[goal_x], [goal_y], [0.]])

    # rotate unit vec to find two perpendicular unit vecs
    vec_ccw = np.dot(rot_ccw, np.ascontiguousarray(vec_rotated))
    # vec_ccw = vec_ccw / vec_ccw[3]
    vec_ccw = vec_ccw / np.linalg.norm(vec_ccw)
    vec_cw = np.dot(rot_cw, np.ascontiguousarray(vec_rotated))
    # vec_cw = vec_cw / vec_cw[3]
    vec_cw = vec_cw / np.linalg.norm(vec_cw)

    # find vector in opposite direction
    # vec_translated = vec_translated / np.linalg.norm(vec_translated)
    vec_neg = -vec_rotated
    # ray marching on lateral direction
    hit_point_ccw = np.array([[goal_x], [goal_y], [0.]])
    dist_ccw = 0.
    hit_point_cw = np.array([[goal_x], [goal_y], [0.]])
    dist_cw = 0.
    # keep a list of checked points in lateral
    ray_marched_list_lateral = [hit_point_ccw]
    while dist_ccw <= max_lateral:
        # ray march ccw rotated vec
        dist_ccw += ray_res
        hit_point_ccw = hit_point_ccw + ray_res * vec_ccw
        ray_marched_list_lateral.append(hit_point_ccw)
        hit_point_ccw_global = np.dot(np.ascontiguousarray(pose_rot[0:3, 0:3]), np.ascontiguousarray(hit_point_ccw)) + trans
        # get current cell coord
        current_col = int64((hit_point_ccw[0,0] - origin_x)/map_resolution)
        current_row = int64((hit_point_ccw[1,0] - origin_y)/map_resolution)
        # check occgrid occupancy
        if env_layer[current_row, current_col] != 0 or static_layer[current_row, current_col] >= static_thresh or dynamic_layer[current_row, current_col] !=0:
            break

    while dist_cw <= max_lateral:
        # ray march cw rotated vec
        dist_cw += ray_res
        hit_point_cw = hit_point_cw + ray_res * vec_cw
        ray_marched_list_lateral.append(hit_point_cw)
        hit_point_cw_global = np.dot(np.ascontiguousarray(pose_rot[0:3, 0:3]), np.ascontiguousarray(hit_point_cw)) + trans
        # get current cell coord
        current_col = int64((hit_point_cw[0,0] - origin_x)/map_resolution)
        current_row = int64((hit_point_cw[1,0] - origin_y)/map_resolution)
        # check occgrid occupancy
        if env_layer[current_row, current_col] != 0 or static_layer[current_row, current_col] >= static_thresh or dynamic_layer[current_row, current_col] !=0:
            break



    # ray marching in longitudinal direction
    hit_point_forward = np.array([[goal_x], [goal_y], [0.]])
    dist_forward = 0.
    hit_point_backward = np.array([[goal_x], [goal_y], [0.]])
    dist_backward = 0.
    # keep a list of checked points in longitude
    ray_marched_list_longitudinal = [hit_point_forward]
    # while dist_forward <= max_longitudinal:
    #     # ray march forward
    #     dist_forward += ray_res
    #     hit_point_forward = hit_point_forward + ray_res * vec_rotated
    #     ray_marched_list_longitudinal.append(hit_point_forward)
    #     hit_point_forward_global = np.dot(pose_rot[0:3, 0:3], hit_point_forward) + trans
    #     # get current cell coord
    #     current_col = int64((hit_point_forward_global[0,0] - origin_x)/map_resolution)
    #     current_row = int64((hit_point_forward_global[1,0] - origin_y)/map_resolution)
    #     # check occgrid occupancy
    #     if env_layer[current_row, current_col] != 0 or static_layer[current_row, current_col] >= static_thresh or dynamic_layer[current_row, current_col] !=0:
    #         break

    while dist_backward <= max_longitudinal:
        # ray march backward
        dist_backward += ray_res
        hit_point_backward = hit_point_backward + ray_res * vec_neg
        ray_marched_list_longitudinal.append(hit_point_backward)
        hit_point_backward_global = np.dot(np.ascontiguousarray(pose_rot[0:3, 0:3]), np.ascontiguousarray(hit_point_backward)) + trans
        # get current cell coord
        current_col = int64((hit_point_backward_global[0,0] - origin_x)/map_resolution)
        current_row = int64((hit_point_backward_global[1,0] - origin_y)/map_resolution)
        # check occgrid occupancy
        if env_layer[current_row, current_col] != 0 or static_layer[current_row, current_col] >= static_thresh or dynamic_layer[current_row, current_col] !=0:
            break


    # create a grid
    # max_y = dist_ccw, min_y = -dist_cw, max_x = dist_forward, min_x = -dist_backward
    y_range = np.arange(-dist_cw, dist_ccw+goal_res, goal_res)
    # sort the array of abs val so that zero gets the min index
    y_range_abs = np.abs(y_range)
    abs_idx = np.argsort(y_range_abs)
    y_range = y_range[abs_idx]

    # x_range = np.arange(-dist_backward, dist_forward+goal_res, goal_res)
    x_range = np.arange(0.0, -dist_backward-goal_res, -goal_res)


    grid_homo = np.zeros((4, y_range.shape[0]*x_range.shape[0]))
    for i in range(x_range.shape[0]):
        for j in range(y_range.shape[0]):
            idx = i*y_range.shape[0] + j
            grid_homo[0, idx] = x_range[i]
            grid_homo[1, idx] = y_range[j]
            grid_homo[2, idx] = 0.
            grid_homo[3, idx] = 1.
    # rotate and translate the grid to the goal pose
    grid_homo_rotated = np.dot(np.ascontiguousarray(rot_orig), np.ascontiguousarray(grid_homo))
    grid_homo_rotated = grid_homo_rotated / grid_homo_rotated[3, :]
    grid_homo_translated = grid_homo_rotated[0:3, :] + np.array([[goal_x], [goal_y], [0.]])

    # add a range of theta to the grid
    grid = np.zeros((3, grid_homo_translated.shape[1]*goal_theta_range.shape[0]))
    for i in range(grid_homo_translated.shape[1]):
        for j in range(goal_theta_range.shape[0]):
            idx = i*goal_theta_range.shape[0] + j
            grid[0, idx] = grid_homo_translated[0, i]
            grid[1, idx] = grid_homo_translated[1, i]
            grid[2, idx] = goal_theta_range[j]
    # grid [[x, y, theta], ...]

    return grid

@njit(fastmath=False, cache=True)
def create_grid_map_only(goal_x, goal_y, goal_res, goal_theta_range, rot, rot_cw, rot_ccw, ray_res, max_lateral, max_longitudinal, env_layer, origin_x, origin_y, map_resolution, trans, pose_rot):
    # goal x y theta in car frame
    # returns Nx3 list of goals [[x, y, theta], ...]
    # need maximum ranges if doesn't hit occgrid

    rot_orig = np.ascontiguousarray(rot)
    rot = np.ascontiguousarray(rot[0:3, 0:3])
    rot_cw = np.ascontiguousarray(rot_cw[0:3, 0:3])
    rot_ccw = np.ascontiguousarray(rot_ccw[0:3, 0:3])

    # unit vec of goal pose in global frame
    x_axis = np.ascontiguousarray(np.array([[1.], [0.], [0.]]))
    vec_rotated = np.dot(rot, x_axis)
    # vec_rotated = vec_rotated / vec_rotated[3]
    vec_rotated = vec_rotated / np.linalg.norm(vec_rotated)
    # vec_translated = vec_rotated + np.array([[goal_x], [goal_y], [0.]])

    # rotate unit vec to find two perpendicular unit vecs
    vec_ccw = np.dot(rot_ccw, np.ascontiguousarray(vec_rotated))
    # vec_ccw = vec_ccw / vec_ccw[3]
    vec_ccw = vec_ccw / np.linalg.norm(vec_ccw)
    vec_cw = np.dot(rot_cw, np.ascontiguousarray(vec_rotated))
    # vec_cw = vec_cw / vec_cw[3]
    vec_cw = vec_cw / np.linalg.norm(vec_cw)

    # find vector in opposite direction
    # vec_translated = vec_translated / np.linalg.norm(vec_translated)
    vec_neg = -vec_rotated
    # ray marching on lateral direction
    hit_point_ccw = np.array([[goal_x], [goal_y], [0.]])
    dist_ccw = 0.
    hit_point_cw = np.array([[goal_x], [goal_y], [0.]])
    dist_cw = 0.
    # keep a list of checked points in lateral
    ray_marched_list_lateral = [hit_point_ccw]
    # while dist_ccw <= max_lateral:
    #     # ray march ccw rotated vec
    #     dist_ccw += ray_res
    #     hit_point_ccw = hit_point_ccw + ray_res * vec_ccw
    #     ray_marched_list_lateral.append(hit_point_ccw)
    #     hit_point_ccw_global = np.dot(np.ascontiguousarray(pose_rot[0:3, 0:3]), np.ascontiguousarray(hit_point_ccw)) + trans
    #     # get current cell coord
    #     current_col = int64((hit_point_ccw[0,0] - origin_x)/map_resolution)
    #     current_row = int64((hit_point_ccw[1,0] - origin_y)/map_resolution)
    #     # check occgrid occupancy
    #     if env_layer[current_row, current_col] != 0:
    #         break

    # while dist_cw <= max_lateral:
    #     # ray march cw rotated vec
    #     dist_cw += ray_res
    #     hit_point_cw = hit_point_cw + ray_res * vec_cw
    #     ray_marched_list_lateral.append(hit_point_cw)
    #     hit_point_cw_global = np.dot(np.ascontiguousarray(pose_rot[0:3, 0:3]), np.ascontiguousarray(hit_point_cw)) + trans
    #     # get current cell coord
    #     current_col = int64((hit_point_cw[0,0] - origin_x)/map_resolution)
    #     current_row = int64((hit_point_cw[1,0] - origin_y)/map_resolution)
    #     # check occgrid occupancy
    #     if env_layer[current_row, current_col] != 0:
    #         break



    # ray marching in longitudinal direction
    hit_point_forward = np.array([[goal_x], [goal_y], [0.]])
    dist_forward = 0.
    hit_point_backward = np.array([[goal_x], [goal_y], [0.]])
    dist_backward = 0.
    # keep a list of checked points in longitude
    ray_marched_list_longitudinal = [hit_point_forward]
    # while dist_forward <= max_longitudinal:
    #     # ray march forward
    #     dist_forward += ray_res
    #     hit_point_forward = hit_point_forward + ray_res * vec_rotated
    #     ray_marched_list_longitudinal.append(hit_point_forward)
    #     hit_point_forward_global = np.dot(pose_rot[0:3, 0:3], hit_point_forward) + trans
    #     # get current cell coord
    #     current_col = int64((hit_point_forward_global[0,0] - origin_x)/map_resolution)
    #     current_row = int64((hit_point_forward_global[1,0] - origin_y)/map_resolution)
    #     # check occgrid occupancy
    #     if env_layer[current_row, current_col] != 0 or static_layer[current_row, current_col] >= static_thresh or dynamic_layer[current_row, current_col] !=0:
    #         break

    while dist_backward <= max_longitudinal:
        # ray march backward
        dist_backward += ray_res
        hit_point_backward = hit_point_backward + ray_res * vec_neg
        ray_marched_list_longitudinal.append(hit_point_backward)
        hit_point_backward_global = np.dot(np.ascontiguousarray(pose_rot[0:3, 0:3]), np.ascontiguousarray(hit_point_backward)) + trans
        # get current cell coord
        current_col = int64((hit_point_backward_global[0,0] - origin_x)/map_resolution)
        current_row = int64((hit_point_backward_global[1,0] - origin_y)/map_resolution)
        # check occgrid occupancy
        if env_layer[current_row, current_col] != 0:
            break


    # create a grid
    # max_y = dist_ccw, min_y = -dist_cw, max_x = dist_forward, min_x = -dist_backward
    y_range = np.arange(-max_lateral, max_lateral+goal_res, goal_res)
    # sort the array of abs val so that zero gets the min index
    y_range_abs = np.abs(y_range)
    abs_idx = np.argsort(y_range_abs)
    y_range = y_range[abs_idx]

    # x_range = np.arange(-dist_backward, dist_forward+goal_res, goal_res)
    x_range = np.arange(0.0, -dist_backward-goal_res, -goal_res)


    grid_homo = np.zeros((4, y_range.shape[0]*x_range.shape[0]))
    for i in range(x_range.shape[0]):
        for j in range(y_range.shape[0]):
            idx = i*y_range.shape[0] + j
            grid_homo[0, idx] = x_range[i]
            grid_homo[1, idx] = y_range[j]
            grid_homo[2, idx] = 0.
            grid_homo[3, idx] = 1.
    # rotate and translate the grid to the goal pose
    grid_homo_rotated = np.dot(np.ascontiguousarray(rot_orig), np.ascontiguousarray(grid_homo))
    grid_homo_rotated = grid_homo_rotated / grid_homo_rotated[3, :]
    grid_homo_translated = grid_homo_rotated[0:3, :] + np.array([[goal_x], [goal_y], [0.]])

    # add a range of theta to the grid
    grid = np.zeros((3, grid_homo_translated.shape[1]*goal_theta_range.shape[0]))
    for i in range(grid_homo_translated.shape[1]):
        for j in range(goal_theta_range.shape[0]):
            idx = i*goal_theta_range.shape[0] + j
            grid[0, idx] = grid_homo_translated[0, i]
            grid[1, idx] = grid_homo_translated[1, i]
            grid[2, idx] = goal_theta_range[j]
    # grid [[x, y, theta], ...]

    return grid

# @njit(fastmath=False, cache=True)(float64[:, :], float64[:, :](float64[:,:], float64[:], float64[:], float64[:], float64[:,:,:,:]), cache=True)
@njit(fastmath=False, cache=True)
def grid_lookup(goal_grid, lut_x, lut_y, lut_theta, lut_kappa, lut, flow, s, kappa0, lut_stepsizes):
    params_list = np.empty((goal_grid.shape[0], 5))
    for i in range(goal_grid.shape[0]):
        param = trajectory_generator.lookup(goal_grid[i, 0], goal_grid[i, 1], goal_grid[i, 2], kappa0, lut_x, lut_y, lut_theta, lut_kappa, lut, lut_stepsizes)
        if (param[0] < 3*np.linalg.norm(goal_grid[i,:2])) and (param[0] > 0):
            params_list[i, :] = param
        else:
            params_list[i, 0] = 0.
    idx = params_list[:, 0] >= 0.0001
    params_list = params_list[idx]
    flow = flow[idx]
    s_filtered = s[idx]
    lookup_grid_filtered = goal_grid[idx]
    states_list_local = trajectory_generator.integrate_all(params_list)
    return states_list_local, params_list, flow, lookup_grid_filtered, s_filtered

@njit(fastmath=False, cache=True)
def basic_grid_lookup(goal_grid, lut_x, lut_y, lut_theta, lut, lut_stepsizes):
    params_list = np.empty((goal_grid.shape[0], 5))
    for i in range(goal_grid.shape[0]):
        param = trajectory_generator.basic_lookup(goal_grid[i, 0], goal_grid[i, 1], goal_grid[i, 2], lut_x, lut_y, lut_theta, lut, lut_stepsizes)
        if (param[0] < 3*np.linalg.norm(goal_grid[i,:2])) and (param[0] > 0):
            params_list[i, :] = param
        else:
            params_list[i, 0] = 0.
    idx = params_list[:, 0] >= 0.0001
    params_list = params_list[idx]
    lookup_grid_filtered = goal_grid[idx]
    states_list_local = trajectory_generator.integrate_all(params_list)
    return states_list_local, params_list, lookup_grid_filtered

@njit(fastmath=False, cache=True)
def grid_lookup_parallel(goal_grid, lut_x, lut_y, lut_theta, lut_kappa, lut, kappa0, lut_stepsizes, num_guys, flow):
    flow_list = List()
    for i in range(flow.shape[0]):
        flow_list.append(flow[i])
    params_list = np.empty((goal_grid.shape[0], 5))
    for i in range(goal_grid.shape[0]):
        param = trajectory_generator.lookup(goal_grid[i, 0], goal_grid[i, 1], goal_grid[i, 2], kappa0, lut_x, lut_y, lut_theta, lut_kappa, lut, lut_stepsizes)
        if (param[0] < 3*np.linalg.norm(goal_grid[i,:2])) and (param[0] > 0):
            params_list[i, :] = param
        else:
            params_list[i, 0] = 0.

    params_list = params_list.reshape((num_guys, int(params_list.shape[0]/num_guys), -1))
    params_list_out = List()
    for i in range(params_list.shape[0]):
        params_list_out.append(params_list[i])
    num_traj_list = List()
    for i in range(num_guys):
        idx = params_list_out[i][:, 0] >= 0.0001
        params_list_out[i] = params_list_out[i][idx]
        flow_list[i] = flow_list[i][idx]
        num_traj_list.append(params_list_out[i].shape[0])
    states_list_local = trajectory_generator.integrate_parallel(params_list_out)
    return states_list_local, params_list_out, flow_list, num_traj_list

# @njit(fastmath=False, cache=True)(float64[:,:](float64[:,:], float64[:,:]))
# def filter_states(states_list, goal_grid):
#     for i in range(goal_grid.shape[1]):


@njit(fastmath=False, cache=True)
def trans_traj_list(traj_list, trans, rot):
    # input traj_list is N*N_samples X 4 ndarray
    # xy_list = np.ascontiguousarray(traj_list[:, 0:2].T)
    xy_list = traj_list[:, 0:2].T
    rot = np.ascontiguousarray(rot)
    trans = np.ascontiguousarray(trans)
    # get homogeneous coords
    homo_xy = np.ascontiguousarray(np.vstack((xy_list, np.zeros((1, traj_list.shape[0])), np.ones((1, traj_list.shape[0])))))
    # apply rotation
    rotated_xy = np.dot(rot, homo_xy)
    rotated_xy = rotated_xy / rotated_xy[3, :]
    # apply translation
    translated_xy = rotated_xy[0:3, :] + trans
    new_traj_list = np.zeros(traj_list.shape)
    new_traj_list[:, 0:2] = translated_xy[0:2, :].T
    new_traj_list[:, 2:4] = traj_list[:, 2:4]
    return new_traj_list

@njit(fastmath=False, cache=True)
def trans_traj_list_multiple(traj_list_all, trans, rot):
    out = List()
    for traj_list in traj_list_all:
        # if it's one of the guys with no traj
        if traj_list.shape == (1, 1):
            out.append(traj_list)
        else:
            out.append(trans_traj_list(traj_list, trans, rot))
    return out

@njit(fastmath=False, cache=True)
def get_length_cost(param_list):
    # not division by zero, grid lookup only returns s >= 0.
    return 1./param_list[:, 0]

@njit(fastmath=False, cache=True)
def get_max_curvature(traj_list, num_traj):
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        out[i] = np.max(np.abs(traj_list[i*trajectory_generator.NUM_STEPS:(i+1)*trajectory_generator.NUM_STEPS, 3]))
    return out

# @njit(fastmath=False, cache=True)
# def get_min_curvature(traj_list, num_traj):
#     out = np.empty((num_traj, ))
#     for i in range(num_traj):
#         out[i] = np.min(np.abs(traj_list[i*trajectory_generator.NUM_STEPS:(i+1)*trajectory_generator.NUM_STEPS, 3]))
#     return out

@njit(fastmath=False, cache=True)
def get_mean_curvature(traj_list, num_traj):
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        out[i] = np.mean(np.abs(traj_list[i*trajectory_generator.NUM_STEPS:(i+1)*trajectory_generator.NUM_STEPS, 3]))
    return out

# @njit(fastmath=False, cache=True)
# def get_s_cost(traj_list, num_traj, wpts):
#     # find the nearest waypoint to the current pose, nearest waypoint to the endpoint, take difference in indices
#     # pose [x,y]
#     N = trajectory_generator.NUM_STEPS
#     out = np.empty((num_traj, ))
#     min_dist, min_idx = min_sqr_dist_to_wpts(traj_list[0, 0:2], wpts)
#     for i in range(num_traj):
#         end_point = traj_list[(i+1)*N - 1, 0:2]
#         min_dist_e, min_idx_e = min_sqr_dist_to_wpts(end_point, wpts)
#         out[i] = (min_idx_e - min_idx) % len(wpts)
#     print('s', out)
#     return out

@njit(fastmath=False, cache=True)
def get_s_cost_wlut(traj_list, num_traj, wpts, speed_lut, lut_resolution):
    N = trajectory_generator.NUM_STEPS
    out = np.empty((num_traj, ))
    start_pt = traj_list[0, 0:2]
    start_query = (int(np.round(start_pt[0]/lut_resolution)), int(np.round(start_pt[1]/lut_resolution)))
    if start_query in speed_lut:
        start_s = speed_lut[start_query][4]
    else:
        return np.inf*np.ones((num_traj, ))

    for i in range(num_traj):
        query = (int(np.round(traj_list[(i+1)*N - 1, 0]/lut_resolution)), int(np.round(traj_list[(i+1)*N - 1, 1]/lut_resolution)))
        if query in speed_lut:
            traj_s = speed_lut[query][4]
        else:
            out[i] = 10.
            continue

        temp = traj_s - start_s

        if temp >= 0.0:
            out[i] = 1./(temp+0.1)
        else:
            temp2 = temp + wpts[-1, 4]
            if temp2 < wpts[-1, 4]/2.:
                out[i] = 1./(temp2+0.1)
            else:
                out[i] = 10.
    return out

@njit(fastmath=False, cache=True)
def get_s_cost_wlut_multiple(traj_cost_list, traj_list_all, num_traj_list, wpts, speed_lut, lut_resolution):
    for i in range(len(traj_cost_list)):
        traj_cost_list[i][4, :] = get_s_cost_wlut(traj_list_all[i], num_traj_list[i], wpts, speed_lut, lut_resolution)


@njit(fastmath=False, cache=True)
def get_similarity_cost(traj_list, prev_path, num_traj):
    N = trajectory_generator.NUM_STEPS
    prev_shifted = prev_path[N_SHIFT:-N_CULL, 2]
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        traj = traj_list[i*N:(i+1)*N, 2]
        traj_shifted = traj[:- N_SHIFT - N_CULL]
        out[i] = np.sum(np.square((traj_shifted - prev_shifted)))
    return out

@njit(fastmath=False, cache=True)
def get_acceleration_cost(param_list, new_traj_list, num_traj):
    N = trajectory_generator.NUM_STEPS
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        vel = np.ascontiguousarray(new_traj_list[i*N:(i+1)*N, 4])
        out[i] = np.amax(np.abs(np.diff(vel, n=1)*vel[:-1]/(param_list[i, 0]/N)))
    return out

@njit(fastmath=False, cache=True)
def get_delta_curvature_cost(param_list, new_traj_list, num_traj):
    N = trajectory_generator.NUM_STEPS
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        kappa = np.ascontiguousarray(new_traj_list[i*N:(i+1)*N, 3])
        vel = new_traj_list[i*N:(i+1)*N, 4]
        out[i] = np.amax(np.abs(np.diff(kappa, n=1)*vel[:-1]/(param_list[i, 0]/N)))
    return out

@njit(fastmath=False, cache=True)
def get_lataccel_cost(new_traj_list, num_traj):
    N = trajectory_generator.NUM_STEPS
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        vel = new_traj_list[i*N:(i+1)*N, 4]
        kappa = new_traj_list[i*N:(i+1)*N, 3]
        out[i] = np.amax(np.abs(kappa) * np.square(vel))
    return out

@njit(fastmath=False, cache=True)
def get_speed_cost(new_traj_list, num_traj):
    N = trajectory_generator.NUM_STEPS
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        min_speed = np.amin(new_traj_list[i*N:(i+1)*N, 4])
        if min_speed < 0.01:
            # print('min_speed close to zero')
            out[i] = np.inf
        else:
            out[i] = 1/min_speed
    return out


# c dynamic obs
@njit(fastmath=False, cache=True)
def multiInterp2(x, xp, fp):
    # i = np.arange(fp.shape[1])
    j = np.searchsorted(xp, x) - 1
    j[j < 0] = 0
    j[j >= len(xp) - 1] = len(xp) - 2
    d = np.expand_dims(((x - xp[j]) / (xp[j + 1] - xp[j])), axis=1)
    return (1 - d) * fp[j, :] + fp[j + 1, :] * d

@njit(fastmath=False, cache=True)
def get_bbox(states):
    # states (Nx3), returns (4xNx2)
    poses = np.expand_dims(states[:, :2], axis=0)
    Lengths = np.expand_dims(CAR_LENGTH*np.ones((states.shape[0])), axis=1)
    Widths = np.expand_dims(CAR_WIDTH*np.ones((states.shape[0])), axis=1)
    x = np.expand_dims((Lengths/2.)*np.vstack((np.cos(states[:,2]), np.sin(states[:,2]))).T, axis=0)
    y = np.expand_dims((Widths/2.)*np.vstack((-np.sin(states[:,2]), np.cos(states[:,2]))).T, axis=0)
    corners = np.concatenate((x-y, x+y, y-x, -x-y), axis=0)
    return corners + poses

# @njit(fastmath=False, cache=True)
# def get_current_s(position, wpts):
#     return 0.
#     nearest_point, nearest_dist, t, i = pure_pursuit_utils.nearest_point_on_trajectory(position, wpts[:, 0:2])
#     s = wpts[i, 4] + t*(wpts[i+1, 4]-wpts[i, 4])
#     return s

@njit(fastmath=False, cache=True)
def get_cdynamic_cost_helper(new_traj_list, parameters_list, num_traj, opp_traj, opp_param, opp_collision, wpts):
    N = trajectory_generator.NUM_STEPS
    out = np.empty((2, num_traj))
    end_xy = np.empty((num_traj, 4))
    opp_vel = opp_traj[:, 4]
    #opp_t = np.hstack((np.zeros((1,)), np.cumsum(opp_param[0] / opp_vel / N)))
    opp_t = np.cumsum(opp_param[0] / opp_vel / N)
    opp_t -= opp_t[0]
    t_end = min(opp_t[-1], T_CD_HORZ_LONG)
    # short_t = np.arange(0, T_CD_HORZ_SHORT, T_INTERP)
    opp_t_all = np.arange(0, t_end+T_INTERP*0.00001, T_INTERP)

    opp_states_interped = multiInterp2(opp_t_all, opp_t, opp_traj[:,:3])
    opp_bbox = get_bbox(opp_states_interped)
    for i in range(num_traj):
        # get timestamps
        vel = new_traj_list[i*N:(i+1)*N, 4]
        #t = np.hstack((np.zeros((1,)), np.cumsum(parameters_list[i, 0] / vel / N)))
        t = np.cumsum(parameters_list[i, 0] / vel / N)
        t -= t[0]
        t_end = min(t_end, t[-1])
        long_t = np.arange(0, t_end+T_INTERP*0.00001, T_INTERP)
        states_interped = multiInterp2(long_t, t, new_traj_list[i*N:(i+1)*N, :3])

        s_idx = states_interped.shape[0] - 1
        opp_end_xy = opp_states_interped[s_idx, 0:2]
        ego_end_xy = states_interped[s_idx, 0:2]
        end_xy[i, :] = np.concatenate((opp_end_xy, ego_end_xy))

        bbox = get_bbox(states_interped)
        dist = np.empty((states_interped.shape[0]))
        for j in range(states_interped.shape[0]):
            if opp_collision:
                dist[j] = GJK.collision_dist(np.ascontiguousarray(bbox[:,j,:]), np.ascontiguousarray(opp_bbox[:,0,:]))
            else:
                dist[j] = GJK.collision_dist(np.ascontiguousarray(bbox[:,j,:]), np.ascontiguousarray(opp_bbox[:,j,:]))
            # print(dist[j])
        # short cost
        end_idx = int(min(T_CD_HORZ_SHORT, t_end) / T_INTERP)
        if np.any(dist[:end_idx] < 1e-10):
            # print('cdynamics short cost inf')
            out[0, i] = np.inf
        else:
            out[0, i] = 0.0
        # long cost
        out[1, i] = CD_CONST * np.dot(1/(dist+0.1), DISCOUNT[:len(dist)])
    return out, end_xy

@njit(fastmath=False, cache=True)
def get_cdynamic_cost(new_traj_list, parameters_list, num_traj, opp_traj_list, opp_param_list, opp_weights, opp_collision, wpts):
    N_OPP = opp_param_list.shape[0]
    N = trajectory_generator.NUM_STEPS
    out = np.zeros((2, num_traj))
    end_xy = np.empty((N_OPP, num_traj, 4))
    for i in range(N_OPP):
        temp, temp_xy = get_cdynamic_cost_helper(new_traj_list, parameters_list, num_traj, opp_traj_list[i*N:(i+1)*N, :], opp_param_list[i], opp_collision, wpts)
        # temp, temp_xy = get_cdynamic_cost_helper(new_traj_list, parameters_list, num_traj, opp_traj_list[i], opp_param_list[i], opp_collision, wpts)
        out += opp_weights[i] * temp
        # robust cost, new weighted sum
        # TODO: return list of temp (N_OPP x num_traj) instead of out (only the long cost)
        end_xy[i,:,:] = temp_xy
    return out, end_xy

@njit(fastmath=False, cache=True)
def get_cdynamic_cost_robust(new_traj_list, parameters_list, num_traj, opp_traj_list, opp_param_list, opp_weights, wpts):
    opp_collision = False
    N_OPP = opp_param_list.shape[0]
    N = trajectory_generator.NUM_STEPS
    short_cost = np.zeros((num_traj, ))
    long_cost = np.empty((N_OPP, num_traj))
    end_xy = np.empty((N_OPP, num_traj, 4))
    for i in range(N_OPP):
        temp, temp_xy = get_cdynamic_cost_helper(new_traj_list, parameters_list, num_traj, opp_traj_list[i*N:(i+1)*N, :], opp_param_list[i], opp_collision, wpts)
        # temp here is (2 x num_traj)
        short_cost += opp_weights[i]*temp[0, :]
        long_cost[i, :] = temp[1, :]
        end_xy[i,:,:] = temp_xy
    return short_cost, long_cost, end_xy

@njit(fastmath=False, cache=True)
def min_sqr_dist_to_wpts(point, wpts):
    """
    Function that calculates the minimum distance from a point to the race line

    Args:
        x, y (float64): point that we're checking the distance
        wpts (ndarray(Nx2)): race line
    Returns:
        dist (float64): minimum distance from point to race line
    """

    # waypoints = np.ascontiguousarray(wpts)
    # point = np.array([[x, y]])
    diff = wpts - point
    diff_squared = np.square(diff)
    dists = np.sum(diff_squared, axis=1)
    min_idx = np.argmin(dists)
    min_dist = dists[min_idx]
    return min_dist, min_idx

# @njit(fastmath=False, cache=True)
# def get_lane_cost_traj(traj, wpts, order):
#     """
#     Function that calculates the cost from deviating from the race line

#     Args:
#         traj (ndarray(N_samples x 4)): trajectory being evaluated
#         wpts (ndarray(Nx2)): race line
#         order (int64): controls the shape of the cost function
#                        higher order cost increases faster
#     Returns:
#         cost (float64): cost from deviating from race line for a trajectory
#     """
#     # traj_arr = np.ascontiguousarray(traj)
#     # waypoints = np.ascontiguousarray(wpts)
#     N = trajectory_generator.NUM_STEPS
#     # cost_arr = np.ascontiguousarray(np.zeros((N, 1)))
#     cost = 0
#     for i in range(N):
#         pt = traj[i, 0:2]
#         min_dist, _ = min_sqr_dist_to_wpts(pt, wpts)
#         # cost_arr[i, 0] = min_dist**order
#         cost += min_dist**(order/2.)
#     # cost = np.sum(cost_arr)
#     return cost



# @njit(fastmath=False, cache=True)(float64[:](float64[:,:], float64[:,:], int64, int64), cache=True)
# @njit(fastmath=False, cache=True)
# def get_lane_cost_traj_list(traj_list, wpts, order, num_traj):
#     """
#     Function that calculates the lane cost for all trajectories in a list

#     Args:
#         traj_list (ndarray(N*N_samples x 4)): a list of trajectories being evaluated
#         wpts (ndarray(Nx2)): race line
#         order (int64): controls the shape of the cost function
#                        higher order cost increases faster
#     Returns:
#         cost (ndarray(Nx1)): list of lane costs for all trajectories
#     """

#     N = trajectory_generator.NUM_STEPS
#     cost_list = np.empty((num_traj,))
#     for i in range(num_traj):
#         traj_eval = traj_list[i*N:(i+1)*N]
#         cost_list[i] = get_lane_cost_traj(traj_eval, wpts, order)
#     return cost_list

# fucking try except doesn't work, way too many keys for if else
# @njit(fastmath=False, cache=True)
def get_lane_cost_traj_list_nonnumba(traj_list, num_traj, speed_lut, reso):
    N = trajectory_generator.NUM_STEPS
    dist_squared = np.empty((traj_list.shape[0],))
    for i in range(traj_list.shape[0]):
        query = (int(np.round(traj_list[i, 0]/reso)), int(np.round(traj_list[i, 1]/reso)))
        try:
            dist_squared[i] = speed_lut[query][1]**2
        except:
            # print('min dist query outside lut')
            dist_squared[i] = np.inf
    out = np.empty((num_traj,))
    for j in range(num_traj):
        out[j] = np.sum(dist_squared[j*N:(j+1)*N])
    return out

@njit(fastmath=False, cache=True)
def get_progress_costs(end_xy, opp_relative_weights, num_traj, speed_lut, lut_resolution):
    N_OPP = opp_relative_weights.shape[0]
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        temp = 0.
        for j in range(N_OPP):
            opp_point = end_xy[j, i, 0:2]
            ego_point = end_xy[j, i, 2:]
            opp_query = (int(np.round(opp_point[0]/lut_resolution)), int(np.round(opp_point[1]/lut_resolution)))
            ego_query = (int(np.round(ego_point[0]/lut_resolution)), int(np.round(ego_point[1]/lut_resolution)))
            if opp_query in speed_lut:
                opp_s = speed_lut[opp_query][4]
                opp_flag = False
            else:
                opp_flag = True

            if ego_query in speed_lut:
                ego_s = speed_lut[ego_query][4]
                ego_flag = False
            else:
                ego_flag = True

            if ego_flag:
                temp = np.inf
                break

            if opp_flag:
                temp += 0.0
            else:
                temp += opp_relative_weights[j] * (S_CONST*max(0., (opp_s - ego_s)))

        out[i] = temp
    return out

@njit(fastmath=False, cache=True)
def get_progress_costs_multiple(traj_cost_list, end_xy_list, opp_relative_weights, num_traj_list, speed_lut, lut_resolution):
    for i in range(len(traj_cost_list)):
        traj_cost_list[i][12, :] = get_progress_costs(end_xy_list[i], opp_relative_weights, num_traj_list[i], speed_lut, lut_resolution)

@njit(fastmath=False, cache=True)
def get_progress_costs_robust(end_xy, N_OPP, num_traj, speed_lut, lut_resolution):
    out = np.empty((N_OPP, num_traj))
    for i in range(num_traj):
        for j in range(N_OPP):
            opp_point = end_xy[j, i, 0:2]
            ego_point = end_xy[j, i, 2:]
            opp_query = (int(np.round(opp_point[0]/lut_resolution)), int(np.round(opp_point[1]/lut_resolution)))
            ego_query = (int(np.round(ego_point[0]/lut_resolution)), int(np.round(ego_point[1]/lut_resolution)))
            if opp_query in speed_lut:
                opp_s = speed_lut[opp_query][4]
                opp_flag = False
            else:
                opp_flag = True

            if ego_query in speed_lut:
                ego_s = speed_lut[ego_query][4]
                ego_flag = False
            else:
                ego_flag = True

            if ego_flag:
                out[:, i] = 1e6
                break

            if opp_flag:
                out[j, i] = 0.0
            else:
                out[j, i] = (S_CONST*max(0., (opp_s - ego_s)))
    return out

# same, fucking try except not supported yet
@njit(fastmath=False, cache=True)
def get_range_costs(traj_list, num_traj, range_lut, reso):
    N = trajectory_generator.NUM_STEPS
    # range cost rows are: min, max, mean
    range_costs = np.empty((1, traj_list.shape[0]))
    for i in range(traj_list.shape[0]):
        query = (int(np.round(traj_list[i, 0]/reso)), int(np.round(traj_list[i, 1]/reso)))
        if query in range_lut:
            vals = range_lut[query]
            if vals[0] <= MIN_BOUNDARY_DIST:
                range_costs[:, i] = 0.
            else:
                range_costs[:, i] = vals[0]
        else:
            # print('range query outside lut')
            range_costs[:, i] = 0.
    # print('min range 0th', range_costs[0, 0:N])
    # print('min range 1th', range_costs[0, N:])
    out = np.empty((1, num_traj))
    for j in range(num_traj):
        # out[:, j] = np.sum(range_costs[:, j*N:(j+1)*N], axis=1)
        temp = range_costs[0 , j*N:(j+1)*N]
        if np.any(temp < MIN_BOUNDARY_DIST):
            out[0, j] = np.inf
        else:
            out[0, j] = np.sum(CS_CONST/(temp+0.1))
    # print('lowest min range cost idx', np.argmin(out[0]))
    # print('lowest min range cost', np.min(out[0]))
    return out

@njit(fastmath=False, cache=True)
def get_range_costs_multiple(traj_cost_list, traj_list_all, num_traj_list, range_lut, reso):
    for i in range(len(traj_cost_list)):
        traj_cost_list[i][9, :] = get_range_costs(traj_list_all[i], num_traj_list[i], range_lut, reso)


@njit(fastmath=False, cache=True)
def lookup_velocity(position, wpts):
    min_dist, min_idx = min_sqr_dist_to_wpts(position, wpts[:,0:2])
    speed = wpts[min_idx, 2]
    return speed

@njit(fastmath=False, cache=True)
def get_velocity_profile(traj_list, wpts, flow_delta, num_traj, current_vel):
    # flow_delta (num_traj, N_KNOT)
    N = trajectory_generator.NUM_STEPS
    steps = int(N/N_KNOT)
    idx = np.arange(N_KNOT+1) * steps
    idx[-1] = N-1

    # new_traj_list = np.empty((traj_list.shape[0], traj_list.shape[1]+1))
    new_traj_list = np.copy(traj_list)
    new_traj_list = np.hstack((new_traj_list, np.empty((traj_list.shape[0], 1))))

    # get speed at knot points
    for i in range(num_traj):
        for j in range(N_KNOT+1):
            if j == 0:
                new_traj_list[i * N + idx[j], 4] = max(current_vel, 0.01)
            else:
                temp = flow_delta[i, j-1]
                #new_traj_list[i * N + idx[j], 4] = max(lookup_velocity(traj_list[i * N + idx[j], 0:2], wpts) + temp, 0.01)
                # this can be const only because we know that the wpts all have const speed. use above for a new map with diff speeds
                new_traj_list[i * N + idx[j], 4] = max(WAYPOINT_SPEED + temp, 0.01)

    for i in range(num_traj):
        counter = 0
        for j in range(N-1):
            if j == idx[counter]:
                low_vel = new_traj_list[i*N+idx[counter], 4]
                high_vel = new_traj_list[i*N+idx[counter+1], 4]
                step_diff = idx[counter+1] - idx[counter]
                start = idx[counter]
                counter += 1
            new_traj_list[i*N+j, 4] = low_vel + (high_vel - low_vel) * (j - start) / step_diff

    return new_traj_list

@njit(fastmath=False, cache=True)
def get_velocity_profile_multiple(traj_list_all, wpts, flow_delta_all, num_traj_list, current_vel):
    out = List()
    for traj_list, flow_delta, num_traj in zip(traj_list_all, flow_delta_all, num_traj_list):
        if num_traj == 0:
            # this should be one extra column longer than the normal ones
            out.append(traj_list)
        else:
            out.append(get_velocity_profile(traj_list, wpts, flow_delta, num_traj, current_vel))
    return out

# @njit(fastmath=False, cache=True)(float64[:, :](float64[:, :], float64[:, :], float64[:], float64[:, :], float64[:, :], float64[:, :]), cache=True)
@njit(fastmath=False, cache=True)
def get_traj_list_cost(traj_list, new_traj_list, cost_weights, wpts, prev_path, param_list, opp_traj_list, opp_param_list, opp_weights, opp_collision):
    num_traj = int(traj_list.shape[0]/trajectory_generator.NUM_STEPS)
    cost_mat = np.empty((len(cost_weights), num_traj))
    cost_mat[0, :] = get_length_cost(param_list)
    cost_mat[1, :] = get_max_curvature(traj_list, num_traj)
    cost_mat[2, :] = get_mean_curvature(traj_list, num_traj)
    cost_mat[3, :] = get_similarity_cost(traj_list, prev_path, num_traj)
    cost_mat[5, :] = get_acceleration_cost(param_list, new_traj_list, num_traj)
    cost_mat[6, :] = get_delta_curvature_cost(param_list, new_traj_list, num_traj)
    cost_mat[7, :] = get_lataccel_cost(new_traj_list, num_traj)
    cost_mat[8, :] = get_speed_cost(new_traj_list, num_traj)
    temp, end_xy = get_cdynamic_cost(new_traj_list, param_list, num_traj, opp_traj_list, opp_param_list, opp_weights, opp_collision, wpts)
    cost_mat[10:12, :] = temp
    return cost_mat, end_xy

@njit(fastmath=False, cache=True)
def get_traj_list_cost_robust(traj_list, new_traj_list, cost_weights, wpts, prev_path, param_list, opp_traj_list, opp_param_list, opp_weights):
    num_traj = int(traj_list.shape[0]/trajectory_generator.NUM_STEPS)
    cost_mat = np.empty((len(cost_weights)-2, num_traj))
    cost_mat[0, :] = get_length_cost(param_list)
    cost_mat[1, :] = get_max_curvature(traj_list, num_traj)
    cost_mat[2, :] = get_mean_curvature(traj_list, num_traj)
    cost_mat[3, :] = get_similarity_cost(traj_list, prev_path, num_traj)
    cost_mat[5, :] = get_acceleration_cost(param_list, new_traj_list, num_traj)
    cost_mat[6, :] = get_delta_curvature_cost(param_list, new_traj_list, num_traj)
    cost_mat[7, :] = get_lataccel_cost(new_traj_list, num_traj)
    cost_mat[8, :] = get_speed_cost(new_traj_list, num_traj)
    short_cost, long_cost, end_xy = get_cdynamic_cost_robust(new_traj_list, param_list, num_traj, opp_traj_list, opp_param_list, opp_weights, wpts)
    cost_mat[10, :] = short_cost
    return cost_mat, end_xy, long_cost

@njit(fastmath=False, cache=True)
def get_traj_list_cost_multiple(traj_list_all, new_traj_list_all, cost_weights_all, wpts, prev_path, param_list_all, opp_traj_list, opp_param_list, opp_weights):
    opp_collision = False
    out_cost = List()
    out_end = List()
    for i in range(cost_weights_all.shape[0]):
        cost_mat, end_xy = get_traj_list_cost(traj_list_all[i], new_traj_list_all[i], cost_weights_all[i], wpts, prev_path, param_list_all[i], opp_traj_list, opp_param_list, opp_weights, opp_collision)
        out_cost.append(cost_mat)
        out_end.append(end_xy)
    return out_cost, out_end

@njit(fastmath=False, cache=True)
def get_robust_cost(long_cost, long_cost_w, progress_cost, progress_cost_w, picked_idx_count, rho):
    num_traj = long_cost.shape[1]
    n_opp = np.sum(picked_idx_count)
    uni = 1. * np.ones((n_opp, ))/n_opp
    out = np.empty((num_traj, ))
    for i in range(num_traj):
        costs = long_cost_w*long_cost[:, i] + progress_cost_w*progress_cost[:, i]
        repeated_costs = np.repeat(costs, picked_idx_count).astype(np.float64)
        q = robust_utils.linear_chi_square(-repeated_costs, uni, rho)
        out[i] = np.dot(q, repeated_costs)
    return out

@njit(fastmath=False, cache=True)
def sum_cost(cost_mat, cost_weights):
    return np.sum(cost_mat*np.expand_dims(cost_weights, axis=1), axis=0)


@njit(fastmath=False, cache=True)
def sum_cost_multiple(traj_costs_list, cost_weights_list):
    out = List()
    for i in range(len(traj_costs_list)):
        out.append(sum_cost(traj_costs_list[i], cost_weights_list[i]))
    return out

# LOOP FUSION DOES ABSOLUTELY NOTHING

# @njit(fastmath=False, cache=True)
# def get_traj_list_cost(traj_list, new_traj_list, costmap, cost_weights, wpts, prev_path, param_list, lut_x, lut_y, lut):
#     num_traj = int(traj_list.shape[0]/trajectory_generator.NUM_STEPS)
#     out_static = get_static_cost(param_list, traj_list, num_traj, wpts[:, 0:2], prev_path)

#     return out_static

# @njit(fastmath=False, cache=True)
# def get_static_cost(param_list, traj_list, num_traj, wpts, prev_path):
#     out = np.empty((7, num_traj))
#     # order: lane_cost_traj, length, max_curv, min_curv, mean_curv, similarity, s_cost
#     N = trajectory_generator.NUM_STEPS
#     prev_shifted = prev_path[N_SHIFT:-N_CULL, 2]
#     min_dist, min_idx = min_sqr_dist_to_wpts(traj_list[0, 0:2], wpts)
#     # loop through traj_list
#     for i in range(num_traj):
#         traj_eval = traj_list[i*N:(i+1)*N]
#         traj_shifted = traj_eval[:-N_SHIFT-N_CULL, 2]
#         out[0, i] = get_lane_cost_traj(traj_eval, wpts, 2)
#         out[1, i] = param_list[i, 0]
#         out[2, i] = np.max(np.abs(traj_eval[:, 3]))
#         out[3, i] = np.min(np.abs(traj_eval[:, 3]))
#         out[4, i] = np.mean(np.abs(traj_eval[:, 3]))
#         out[5, i] = np.sum(np.square((traj_shifted - prev_shifted)))
#         end_point = traj_eval[-1, 0:2]
#         min_dist_e, min_idx_e = min_sqr_dist_to_wpts(end_point, wpts)
#         out[6, i] = (min_idx_e - min_idx) % len(wpts)

#     return out












# @njit(fastmath=False, cache=True)(float64[:, :](float64[:, :], int64, int64, int64, float64, float64, float64), cache=True)
# def fill_map_cost_table(map_array, d1, d2, d3, lethal_cost, high_cost, low_cost):
#     """
#     Function that fills in the cost table for the occgrid
#     three distances, d1 for lethal, d2 for high, d3 for low

#     Args:
#         map_array (ndarray(NxM)): map image
#         d1, d2, d3 (int64): distances for different level of cost filling
#         lethal_cost, high_cost, low_cost (float64): costs for three different levels
#     Returns:
#         cost_table (ndarray(NxM)): cost table for each pixel, same size as map
#     """
#     map_arr = np.ascontiguousarray(map_array)
#     map_cost_table = np.ascontiguousarray(np.zeros(map_arr.shape))
#     rows, cols = map_arr.shape
#     for r in range(rows):
#         for c in range(cols):
#             if map_arr[r, c] <= 250.0:
#                 # here's a hit in the map
#                 break
#     cost_table = np.ones(map_arr.shape)
#     return cost_table



# waypoint getting helpers
# @njit(fastmath=False, cache=True)(float64[:](float64[:,:],float64, float64))
# def dist(waypoints, position_x, position_y):
#     return np.sqrt((waypoints[:, 0]-position_x)**2+(waypoints[:,1]-position_y)**2)

# @njit(fastmath=False, cache=True)(boolean[:](float64[:,:], float64[:], float64))
# def dir(dist_valid_points, pos, theta):
#     same_dir = np.full(dist_valid_points.shape[0], False)
#     for i in range(dist_valid_points.shape[0]):
#         point = dist_valid_points[i,:]
#         vec = np.asarray(point[0:2])-np.asarray(pos[0:2])
#         unit_vec = vec/np.linalg.norm(vec)
#         rotated_x = np.asarray([np.cos(theta), np.sin(theta)])
#         cosang = np.dot(np.ascontiguousarray(unit_vec), np.ascontiguousarray(rotated_x))
#         # crossed = np.cross(unit_vec, rotated_x)
#         # sinang = np.linalg.norm(crossed)
#         # angle = np.arctan2(sinang, cosang)
#         angle = np.arccos(cosang)
#         same_dir[i] = boolean(abs(angle) <= np.pi/2.5)

#     return same_dir

# def get_current_waypoint(waypoints, position, theta, look_ahead_distance):
#     dists = dist(waypoints, position[0], position[1])
#     dist_valid_

