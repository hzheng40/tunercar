import numpy as np
from scipy.interpolate import splprep, splev, spalde
import csv
import yaml
from PIL import Image

def preprocess_centerline(raw_centerline, config):
    # TODO: do splprep and splev to get equispaced centerline
    pass

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
    tck, u = splprep([points[0, :], points[1, :]], k=3, s=points.shape[1] * config['s_factor'])
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

def make_waypoints(spline, theta, curvature):
    waypoints = np.zeros((spline.shape[0], 5))
    waypoints[:, 0:2] = spline
    waypoints[:, 3] = theta
    waypoints[:, 4] = curvature
    return waypoints