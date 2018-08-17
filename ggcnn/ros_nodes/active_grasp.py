#!/usr/bin/env python

from __future__ import division

import rospy

import numpy as np
import cv2
from scipy.signal import convolve2d as conv2d
from scipy.ndimage.filters import gaussian_filter

from skimage.feature import peak_local_max

import matplotlib.pyplot as plt

from dougsm_helpers.tf_helpers import current_robot_pose
from tf import transformations as tft
from dougsm_helpers.timeit import TimeIt

from ggcnn.aegrasp import predict
from ggcnn.grasp import detect_grasps
from ggcnn.grasp_stats import update_batch, update_batch_single_sample, update_histogram_angle
from ggcnn.heightmap import HeightMap
from ggcnn.gridshow import gridshow

from ggcnn.srv import NextViewpoint
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo

import cv_bridge
bridge = cv_bridge.CvBridge()

# cv2.namedWindow('Display', cv2.WINDOW_NORMAL)

class ViewpointEntropyCalculator:
    def __init__(self):
        self.hist_bins_q = 10
        self.hist_bins_a = 18

        self.best_dist_scale = 0.25
        self.best_gain = 0.25
        self.prev_view_cutoff = 0.1
        self.prev_view_gain = 0.1

        self.height = (0.5, 0.25)

        # Used to convert cells to positions etc.
        hm_bounds = np.array([
            [-0.2, -0.8],
            [0.2, -0.4]
        ])
        hm_res = 0.005
        self.hm = HeightMap(hm_bounds, hm_res)
        self.hm.add_map('count', 0.0)
        self.hm.add_map('visited', 0.0)
        self.count_hist = np.zeros((self.hm.maps['count'].shape[0], self.hm.maps['count'].shape[1], self.hist_bins_a, self.hist_bins_q))
        self.count_hist[:, :, :, :] = 1

        self.position_history = []

        # Useful meshgrid
        xs = np.arange(self.hm.bounds[0, 0], self.hm.bounds[1, 0], self.hm.res[0]) + self.hm.res[0] / 2
        ys = np.arange(self.hm.bounds[0, 1], self.hm.bounds[1, 1], self.hm.res[1]) + self.hm.res[1] / 2
        self.xv, self.yv = np.meshgrid(xs, ys)

        self.img_pub = rospy.Publisher('~ap_image', Image, queue_size=1)


    def get_next_viewpoint(self, req):
        print(0)
        camera_pose = current_robot_pose('panda_link0', 'camera_depth_optical_frame')
        cam_x = camera_pose.position.x
        cam_y = camera_pose.position.y
        cam_z = camera_pose.position.z

        newpos_pixel = self.hm.pos_to_cell([[cam_x, cam_y]])[0]
        self.hm.visited[newpos_pixel[0], newpos_pixel[1]] = self.hm.visited.max() + 1

        self.position_history.append((cam_x, cam_y, cam_z))
        print(1)
        depth_msg = rospy.wait_for_message('/camera/depth/image_meters/', Image)
        depth = bridge.imgmsg_to_cv2(depth_msg)

        crop_size = 460
        out_size = 300
        imh, imw = depth.shape
        depth_crop = cv2.resize(depth[(imh - crop_size) // 2:(imh - crop_size) // 2 + crop_size,
                                (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size], (out_size, out_size))
        depth_crop = depth_crop.copy()
        depth_nan = np.isnan(depth_crop).copy()
        depth_crop[depth_nan] = 0
        depth = depth_crop
        points, angle, width_img = predict(depth, 300)

        print()
        caminfo = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo)
        K = np.array(caminfo.K).reshape((3, 3))
        x = ((np.vstack((np.linspace(90, 550, depth.shape[1], np.float), )*depth.shape[0]) - K[0, 2])/K[0, 0] * depth).flatten()
        y = ((np.vstack((np.linspace(10, 470, depth.shape[0], np.float), )*depth.shape[1]).T - K[1,2])/K[1, 1] * depth).flatten()

        print(x.min(), x.max(), y.min(), y.max())

        cq = camera_pose.orientation
        camera_rot = tft.quaternion_matrix([cq.x, cq.y, cq.z, cq.w])[0:3, 0:3]
        pos = np.dot(camera_rot, np.stack((x, y, depth.flatten()))) + np.array([[cam_x, cam_y, cam_z]]).T
        pos = pos.T.astype(np.float32)

        print(pos.min(axis=0), pos.max(axis=0))

        pf = points.flatten()
        angle = angle.flatten() + np.pi/2

        try:
            cell_ids = self.hm.pos_to_cell(pos[:, :2])
        except ValueError:
            print('FAILED')
            return Point()

        print(cell_ids.min(axis=0), cell_ids.max(axis=0))

        update_histogram_angle(pf, angle, cell_ids, self.count_hist)

        # Marginalise over all angles.
        hist_sum_q = np.sum(self.count_hist, axis=2)
        weights = np.arange(0.5/self.hist_bins_q, 1.0, 1/self.hist_bins_q)
        hist_mean = np.sum(hist_sum_q * weights.reshape((1, 1, -1)), axis=2)/(np.sum(hist_sum_q, axis=2) + 1e-6)

        # Average quality per angle per x,y
        best_angle = np.argmax(
                np.sum(self.count_hist * weights.reshape((1, 1, 1, -1)), axis=3)/(np.sum(self.count_hist, axis=3) + 1e-6),
                axis=2
        ).astype(np.float)

        hist_p = hist_sum_q / np.expand_dims(np.sum(hist_sum_q, axis=2) + 1e-6, -1)
        hist_ent = -np.sum(hist_p * np.log(hist_p+1e-6), axis=2)

        d_from_robot = np.sqrt((self.xv - cam_x)**2 + (self.yv - cam_y)**2)
        d_robot_scalar = (d_from_robot - d_from_robot.min())/(d_from_robot.max() - d_from_robot.min())

        q_am = np.unravel_index(np.argmax(hist_mean), hist_mean.shape)
        q_am_pos = self.hm.cell_to_pos([q_am])[0]
        d_from_best_q = np.sqrt((self.xv - q_am_pos[0])**2 + (self.yv - q_am_pos[1])**2)  # Cost of moving away from the best grasp.

        # Calculated expected information gain.
        fov = int(cam_z * np.tan(55.0*400.0/480.0 / 180.0 * np.pi) / self.hm.res[0])  # Field of view in heightmap pixels.
        exp_inf_gain = gaussian_filter(hist_ent, fov/6, mode='nearest', truncate=3)

        d_from_prev_view = np.zeros_like(self.hm.count)
        for x, y, z in self.position_history:
            d_from_prev_view += np.clip(1 - (np.sqrt((self.xv - x)**2 + (self.yv - y)**2 + (cam_z - z)**2)/self.prev_view_cutoff), 0, 1)

        height_weight = (cam_z - self.height[1])/(self.height[0]-self.height[1]) + 1e-2
        best_cost = (d_from_best_q / (self.best_dist_scale * height_weight)) * self.best_gain

        prev_view_cost = d_from_prev_view * self.prev_view_gain

        exp_inf_gain -= best_cost
        exp_inf_gain -= prev_view_cost

        exp_inf_gain_mask = exp_inf_gain.copy()
        exp_inf_gain_mask[d_from_robot > 0.10] = exp_inf_gain.min()
        ig_am = np.unravel_index(np.argmax(exp_inf_gain_mask), exp_inf_gain.shape)
        maxpos = self.hm.cell_to_pos([ig_am])[0]
        diff = maxpos - np.array([cam_x, cam_y])
        move_amt = 0.05
        if np.linalg.norm(diff) > move_amt:
            diff = diff/np.linalg.norm(diff) * move_amt

        p = Point()
        p.x = diff[0]
        p.y = diff[1]
        p.z = -1 * ((move_amt - np.linalg.norm(diff))/move_amt * 0.01 + 0.01)

        show = gridshow('Display',
                 [cv2.resize(points, hist_ent.shape), hist_mean, hist_ent, np.exp(exp_inf_gain), best_cost, self.hm.visited],
                 [None, None, None, None, None, None],
                 [cv2.COLORMAP_BONE] + [cv2.COLORMAP_JET, ] * 4 + [cv2.COLORMAP_BONE],
                 3,
                 False)

        self.img_pub.publish(bridge.cv2_to_imgmsg(show))

        return p


if __name__ == '__main__':
    rospy.init_node('panda_active_grasp')
    VEC = ViewpointEntropyCalculator()
    rospy.Service('~get_next_viewpoint', NextViewpoint, VEC.get_next_viewpoint)
    rospy.spin()
