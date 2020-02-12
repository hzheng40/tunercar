import numpy as np
from scipy.interpolate import splprep, splev
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
    tck, u = splprep([points[0, :], points[1, :]], k=3, s=points.shape[1] * config['s_factor'], per=1)
    spline_size = (points.shape[1] - 1) * config['interp_factor']
    # make a u vector that has much more values for interpolation
    new_u = np.arange(spline_size)
    new_u = new_u / float(spline_size - 1)

    # evaluate spline
    spline = np.asarray(splev(new_u, tck))

    return spline

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
