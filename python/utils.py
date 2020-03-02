import numpy as np
from scipy.interpolate import splprep, splev, spalde
import csv
import yaml
from PIL import Image
from speed_opt import optimal_time

def preprocess_centerline(raw_centerline, config):
    # [x, y, theta, s]
    tck, u = splprep(raw_centerline[:, 0], raw_centerline[:, 1], s=config['preproc_smooth'], k=5, per=True)
    # unew = np.arange(0, 1.0, 0.002)
    unew = np.linspace(0., 1., config['num_preproc_points'])
    new_centerline = np.asarray(splev(unew, tck)).T
    diffs = np.linagl.norm(new_centerline[1:]-new_centerline[:-1], axis=1)
    s = np.cumsum(diffs)
    s = np.insert(s, 0, 0)
    derivs = spalde(unew, tck)
    Dx = np.asarray(derivs[0])
    Dy = np.asarray(derivs[1])
    dx = Dx[:, 1]
    dy = Dy[:, 1]
    theta = np.arctan2(dy, dx)
    waypoints = np.empty((new_centerline.shape[0], 4))
    waypoints[:, 0:2] = new_centerline
    waypoints[:, 2] = theta
    waypoints[:, 3] = s
    return waypoints

def get_map_img(config):
    map_img = np.array(Image.open('../maps/'+config['map_name']+config['map_img_ext']).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
    if len(map_img.shape) > 2:
        print('Use grayscale image!')
        return None
    else:
        return map_img

def get_centerline(config):
    with open('../maps'+config['map_csv_name']+'.csv', 'r') as f:
        waypoints = [tuple(line) for line in csv.reader(f)]
        centerline = np.array([(float(pt[0]), float(pt[1])) for pt in waypoints])
    centerline = centerline[::config['skip_factor'], :]
    return centerline

def make_spline(points, config):
    # TODO: periodic in new version
    # tck, u = splprep([points[0, :], points[1, :]], k=3, s=points.shape[1] * config['s_factor'], per=1)
    points[1, :] *= -1.
    tck, u = splprep([points[0, :], points[1, :]], k=3, s=points.shape[1] * config['s_factor'], per=1)
    spline_size = (points.shape[1] - 1) * config['interp_factor']
    # make a u vector that has much more values for interpolation
    new_u = np.arange(spline_size)
    new_u = new_u / float(spline_size - 1)

    # evaluate spline, out is Nx2
    spline = np.asarray(splev(new_u, tck)).T

    # get thetas
    derivs = spalde(new_u, tck)
    Dx = np.asarray(derivs[0])
    Dy = np.asarray(derivs[1])
    dx = Dx[:, 1]
    dy = Dy[:, 1]
    ddx = Dx[:, 2]
    ddy = Dy[:, 2]
    curvature = (dx*ddy - dy*ddx)/((dx*dx + dy*dy)**1.5)
    theta = np.arctan2(dy, dx)

    return spline, theta, curvature

# @njit(cache=True)
def get_bbox(waypoints, box_width, box_height):
    # waypoints (Nx3) [x, y, theta], returns (4xNx2)
    poses = np.expand_dims(waypoints[:, :2], axis=0)
    Lengths = np.expand_dims(box_height*np.ones((waypoints.shape[0])), axis=1)
    Widths = np.expand_dims(box_width*np.ones((waypoints.shape[0])), axis=1)
    x = np.expand_dims((Lengths/2.)*np.vstack((np.cos(waypoints[:,2]), np.sin(waypoints[:,2]))).T, axis=0)
    y = np.expand_dims((Widths/2.)*np.vstack((-np.sin(waypoints[:,2]), np.cos(waypoints[:,2]))).T, axis=0)
    corners = np.concatenate((x-y, x+y, y-x, -x-y), axis=0)
    # 4xNx4, middle dim is num boxes
    return corners + poses

def get_width(center_line, config):
    diff = center_line[1:, :] - center_line[0:-1, :]
    width_sum = np.sum(np.linalg.norm(diff, axis=1), axis=0)
    width = width_sum / (center_line.shape[0]-1)
    return config['width_scale'] * width

def get_boxes(centerline, map_img, config):
    num_points = centerline.shape[0]
    width = get_width(centerline, config)
    # TODO: put box_height = 0.8 into config yaml
    height = config['box_height']
    boxes = np.empty((7, num_points))
    # TODO: not sure what this is?
    prev_angle = 10.0
    sterak = 1
    ctr = 0

def make_waypoints(spline, theta, speed):
    waypoints = np.zeros((spline.shape[0], 4))
    waypoints[:, 0:2] = spline
    waypoints[:, 2] = speed
    waypoints[:, 3] = theta
    return waypoints

def get_speed(spline, mass, Wf):
    # spline = spline[::5, :]

    muf = 0.523
    gravity = 9.81
    path = optimal_time.define_path(spline[:, 0], spline[:, 1])
    params = optimal_time.define_params(mass, Wf, muf, gravity)
    B, A, U, v, topt = optimal_time.optimize(path, params)

    # v = v.repeat(5)
    return v

def tweak_t(config, sub_waypoints, ts):
    angle = sub_waypoints[:, 2]
    angle_arr = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    
    return tweaked_waypoints

def bound_pop(config, population):
    t_bound = config['box_height']
    mass_bound = config['mass']
    l_f_bound = config['l_f_bound']
    wpt_lad_bound = config['wpt_lad_bound']
    track_lad_div_bound = config['track_lad_div_bound']
    speed_gain_bound = config['speed_gain_bound']
    population[0] = mass_bound[0] + (mass_bound[1] - mass_bound[0]) * population[0]
    population[1] = l_f_bound[0] + (l_f_bound[1] - l_f_bound[0]) * population[1]
    population[2] = wpt_lad_bound[0] + (wpt_lad_bound[1] - wpt_lad_bound[0]) * population[2]
    population[3] = track_lad_div_bound[0] + (track_lad_div_bound[1] - track_lad_div_bound[0]) * population[3]
    population[4] = speed_gain_bound[0] + (speed_gain_bound[1] - speed_gain_bound[0]) * population[4]
    population[5:] = -t_bound + 2*t_bound*population[5:]
    return population

def get_population(config):
    # params [mass, l_f, wpt_lad, track_lad_div, speed_gain]
    vector_size = config['num_params'] + config['num_control_points']
    population = np.random.rand(vector_size)
    bounded_population = bound_pop(config, population)
    return bounded_population

def population_to_params(config, waypoints, map_img):
    # points(fixed s, variable t, bounded by track width)
    # vel prof (search, on/off by flag)
    # params (bounded)

    num_skip = int(waypoints.shape[0] / config['num_control_points'])
    sub_waypoints = waypoints[::num_skip, :]