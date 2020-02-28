import numpy as np
from PIL import Image
import OpenGL.GL as gl
import pypangolin as pangolin

from mpc import transformations
import yaml

class PangoViz(object):
    def __init__(self, zoom, map_img_path, map_yaml_path, waypoints, show_laser=False):
        # load map params
        with open(map_yaml_path, 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                map_resolution = map_metadata['resolution']
                origin = map_metadata['origin']
                map_origin_x = origin[0]
                map_origin_y = origin[1]
            except yaml.YAMLError as ex:
                print(ex)

        self.waypoints = waypoints
        self.waypoints_plot = np.copy(waypoints[:, 0:3])
        self.waypoints_plot[:, 2]*=0.

        # toggle for laser viz
        self.show_laser = show_laser
        # load map
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        if len(self.map_img.shape) > 2:
            print('map image not grayscale')
            self.map_img = np.dot(self.map_img[...,:3], [0.29, 0.57, 0.14])
            self.map_img = np.floor(self.map_img)
        range_x = np.arange(self.map_img.shape[1])
        range_y = np.arange(self.map_img.shape[0])
        map_x, map_y = np.meshgrid(range_x, range_y)
        map_x = (map_x*map_resolution+map_origin_x).flatten()
        map_y = (map_y*map_resolution+map_origin_y).flatten()
        map_z = np.zeros(map_y.shape)
        map_coords = np.vstack((map_x, map_y, map_z))
        map_mask = self.map_img == 0.0
        map_mask_flat = map_mask.flatten()
        self.map_points = map_coords[:, map_mask_flat].T

        # init pangolin
        pangolin.CreateWindowAndBind('sim', 930, 1080)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # define projection and initial modelview matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 120, 120, 320, 280, 0.2, 200),
            pangolin.ModelViewLookAt(-0.1, 0, zoom, 0, 0, 0, pangolin.AxisDirection.AxisZ
            ))
        self.handler = pangolin.Handler3D(self.scam)

        # create interactive view in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(pangolin.Attach(0.0), pangolin.Attach(1.0), pangolin.Attach(0.0), pangolin.Attach(1.0), -1920.0 / 1080)
        self.dcam.SetHandler(self.handler)

        # scan params
        angle_min = -4.7/2
        angle_max = 4.7/2
        num_beams = 1080
        self.scan_angles = np.linspace(angle_min, angle_max, num_beams)


    def update(self, obs, ego_all_states, ego_picked_state, ego_xy_grid):
        # clear buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(37/255, 37/255, 38/255, 1.0)
        self.dcam.Activate(self.scam)
        # grab observations
        # ignoring scans for now
        # car poses
        ego_x = obs['poses_x'][0]
        ego_y = obs['poses_y'][0]
        ego_theta = obs['poses_theta'][0]

        # print('ego', ego_x, ego_y, ego_theta)
        # print('opp', op_x, op_y, op_theta)

        # Draw boxes for agents
        # ego_pose = np.identity(4)
        ego_pose = transformations.rotation_matrix(ego_theta, (0, 0, 1))
        ego_pose[0, 3] = ego_x
        ego_pose[1, 3] = ego_y
        # ego_pose[2, 3] = 0.1
        # op_pose = np.identity(4)

        ego_size = np.array([0.58, 0.31, 0.1])
        gl.glLineWidth(1)
        # ego is blue-ish
        gl.glColor3f(0.0, 0.5, 1.0)
        pangolin.DrawBoxes([ego_pose], [ego_size])

        # Draw map
        gl.glPointSize(2)
        gl.glColor3f(0.2, 0.2, 0.2)
        pangolin.DrawPoints(self.map_points)

        # draw flow samples
        # ego
        gl.glPointSize(2)
        gl.glColor3f(0.0, 0.5, 1.0)
        # print('hello \n')
        # print('st', ego_grid[0:5])
        if ego_xy_grid is None:
            pangolin.FinishFrame()
            return
        gl.glPointSize(2)
        gl.glColor3f(0.0, 0.5, 1.0)
        # print('xytheta', ego_xythetas[0:5])
        rot = np.array([[np.cos(ego_theta), np.sin(ego_theta)],[-np.sin(ego_theta), np.cos(ego_theta)]])
        xy_grid = np.dot(ego_xy_grid[:,:2], rot)
        temp = np.hstack([xy_grid, np.zeros((xy_grid.shape[0],1))])
        # print('xyz', ego_xythetas[0:5])
        pangolin.DrawPoints(temp + np.array([ego_x, ego_y, 0.0])[None,:])

        # opp

        # Draw laser scans
        # Red for ego, Blue for op
        # Could be turned off
        if self.show_laser:
            rot_mat = transformations.rotation_matrix(ego_theta, (0, 0, 1))
            ego_scan = obs['scans'][0]
            ego_scan = np.asarray(ego_scan)
            ego_scan_x = np.multiply(ego_scan, np.sin(self.scan_angles))
            ego_scan_y = np.multiply(ego_scan, np.cos(self.scan_angles))

            ego_scan_arr = np.zeros((ego_scan_x.shape[0], 3))
            ego_scan_arr[:, 0] = ego_scan_y
            ego_scan_arr[:, 1] = ego_scan_x
            ego_scan_arr = np.dot(rot_mat[0:3,0:3], ego_scan_arr.T)
            ego_scan_arr = ego_scan_arr + np.array([[ego_x], [ego_y], [0]])

            gl.glPointSize(1)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawPoints(ego_scan_arr.T)


        # Draw splines
        if ego_all_states is not None:
            gl.glPointSize(2)
            gl.glColor3f(0.8, 0.0, 0.5)
            # print('num traj', ego_all_states.shape[0]/100)
            pangolin.DrawPoints(np.hstack([ego_all_states[:, 0:2], np.zeros((ego_all_states.shape[0],1))]))
        if ego_picked_state is not None:
            gl.glPointSize(5)
            if ego_all_states is None:
                gl.glColor3f(1., 0., 0.)
            else:
                gl.glColor3f(1., 1., 1.)
            pangolin.DrawPoints(np.hstack([ego_picked_state[:,0:2], np.zeros((ego_picked_state.shape[0],1))]))


        # draw waypoints
        gl.glPointSize(2)
        gl.glColor3f(1., 1., 1.)
        pangolin.DrawPoints(self.waypoints_plot)

        # render
        pangolin.FinishFrame()


        # print('hi \n')
        # time.sleep(1)