#!/usr/bin/env python

from __future__ import division, print_function

import rospy
import rosparam

import numpy as np
import cv2
from scipy.signal import convolve2d as conv2d
from scipy.ndimage.filters import gaussian_filter

from skimage.feature import peak_local_max

from dougsm_helpers.tf_helpers import current_robot_pose
from tf import transformations as tft
from dougsm_helpers.timeit import TimeIt

from ggcnn.ggcnn import predict
from ggcnn.grasp import detect_grasps
from mvp_grasping.grasp_stats import update_batch, update_batch_single_sample, update_histogram_angle
from mvp_grasping.gridworld import GridWorld
from dougsm_helpers.gridshow import gridshow

from mvp_grasping.srv import NextViewpoint, NextViewpointResponse
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray

import cv_bridge
bridge = cv_bridge.CvBridge()


class ViewpointEntropyCalculator:
    def __init__(self):
        self.hist_bins_q = rosparam.get('~histogram/bins/quality')
        self.hist_bins_a = rosparam.get('~histogram/bins/angle')

        self.dist_from_best_scale = rosparam.get('~histogram/cost/dist_from_best_scale')
        self.dist_from_best_gain = rosparam.get('~histogram/cost/dist_from_best_gain')
        self.dist_from_prev_view_scale = rosparam.get('~histogram/cost/dist_from_view_scale')
        self.dist_from_prev_view_gain = rosparam.get('~histogram/cost/dist_from_view_gain')

        self.height = (rosparam.get('~height/z1'), rosparam.get('~height/z2'))

        # Create a GridWorld where we will store values.
        gw_bounds = np.array([
            [rosparam.get('~histogram/bounds/x1'), rosparam.get('~histogram/bounds/y1')],
            [rosparam.get('~histogram/bounds/x2'), rosparam.get('~histogram/bounds/y2')]
        ])
        gw_res = rosparam.get('~histogram/resolution')

        self.gw = GridWorld(gw_bounds, gw_res)
        self.gw.add_map('visited', 0.0)
        self.gw.add_map('hist', 1.0, extra_dims=(self.hist_bins_q, self.hist_bins_a))

        self.gw.hist = np.zeros((self.gw.maps['count'].shape[0], self.gw.maps['count'].shape[1], self.hist_bins_a, self.hist_bins_q))
        self.hist_mean = 0

        # Useful meshgrid for distance calculations
        xs = np.arange(self.gw.bounds[0, 0], self.gw.bounds[1, 0], self.gw.res) + self.gw.res / 2
        ys = np.arange(self.gw.bounds[0, 1], self.gw.bounds[1, 1], self.gw.res) + self.gw.res / 2
        self._xv, self._yv = np.meshgrid(xs, ys)

        # Track the position of the camera.
        self.position_history = []

        # Get the camera parameters
        cam_info_topic = rosparam.get('~camera/info_topic')
        camera_info_msg = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.cam_K = np.array(camera_info_msg.K).reshape((3, 3))

        self.img_pub = rospy.Publisher('~visualisation', Image, queue_size=1)
        rospy.Service('~update', NextViewpoint, self.update_service_handler)

    def update_service_handler(self, req):
        depth_msg = rospy.wait_for_message('/camera/depth/image_meters/', Image)
        camera_pose = current_robot_pose('panda_link0', 'camera_depth_optical_frame')
        cam_x = camera_pose.position.x
        cam_y = camera_pose.position.y
        cam_z = camera_pose.position.z
        self.position_history.append(np.array([cam_x, cam_y, cam_z])

        newpos_pixel = self.gw.pos_to_cell([[cam_x, cam_y]])[0]
        self.gw.visited[newpos_pixel[0], newpos_pixel[1]] = self.gw.visited.max() + 1

        depth = bridge.imgmsg_to_cv2(depth_msg)

        crop_size = 350
        out_size = 300

        depth_crop, depth_nan_mask = ggcnn.proces_depth_image(depth, crop_size, out_size, return_mask=True)

        points, angle, width_img, _ = predict(depth, process_depth=False)
        pf = points.flatten()
        angle = angle.flatten() + np.pi/2

        # Convert the points to 3D coordinates.
        self.cam_K = np.array(caminfo.self.cam_K).reshape((3, 3))
        x = ((np.vstack((np.linspace((imh - crop_size) // 2, (imh - crop_size) // 2 + crop_size, depth.shape[1], np.float), )*depth.shape[0]) - self.cam_K[0, 2])/self.cam_K[0, 0] * depth).flatten()
        y = ((np.vstack((np.linspace((imw - crop_size) // 2, (imw - crop_size) // 2 + crop_size, depth.shape[0], np.float), )*depth.shape[1]).T - self.cam_K[1,2])/self.cam_K[1, 1] * depth).flatten()
        cq = camera_pose.orientation
        camera_rot = tft.quaternion_matrix([cq.x, cq.y, cq.z, cq.w])[0:3, 0:3]
        pos = np.dot(camera_rot, np.stack((x, y, depth.flatten()))) + np.array([[cam_x, cam_y, cam_z]])
        pos[depth_nan_mask.flatten() == 1] = 0

        try:
            cell_ids = self.gw.pos_to_cell(pos[:, :2])
        except ValueError:
            print('FAILED')
            return Point()

        update_histogram_angle(pf, angle, cell_ids, self.gw.hist)

        # Marginalise over all angles.
        hist_sum_q = np.sum(self.gw.hist, axis=2)
        weights = np.arange(0.5/self.hist_bins_q, 1.0, 1/self.hist_bins_q)
        hist_mean = np.sum(hist_sum_q * weights.reshape((1, 1, -1)), axis=2)/(np.sum(hist_sum_q, axis=2) + 1e-6)

        # Average quality per angle per x,y
        best_angle = np.argmax(
                np.sum(self.gw.hist * weights.reshape((1, 1, 1, -1)), axis=3)/(np.sum(self.gw.hist, axis=3) + 1e-6),
                axis=2
        ).astype(np.float)

        hist_p = hist_sum_q / np.expand_dims(np.sum(hist_sum_q, axis=2) + 1e-6, -1)
        hist_ent = -np.sum(hist_p * np.log(hist_p+1e-6), axis=2)

        d_from_robot = np.sqrt((self.xv - cam_x)**2 + (self.yv - cam_y)**2)
        d_robot_scalar = (d_from_robot - d_from_robot.min())/(d_from_robot.max() - d_from_robot.min())

        q_am = np.unravel_index(np.argmax(hist_mean), hist_mean.shape)
        q_am_pos = self.gw.cell_to_pos([q_am])[0]
        d_from_best_q = np.sqrt((self.xv - q_am_pos[0])**2 + (self.yv - q_am_pos[1])**2)  # Cost of moving away from the best grasp.

        q_am_ang = best_angle[q_am]/self.hist_bins_a * np.pi - np.pi/2

        # Calculated expected information gain.
        fov = int(cam_z * np.tan(65.5*crop_size/480.0 / 180.0 * np.pi) / self.gw.res)  # Field of view in heightmap pixels.
        exp_inf_gain = gaussian_filter(hist_ent, fov/6, mode='nearest', truncate=3)

        d_from_prev_view = np.zeros_like(self.gw.count)
        for x, y, z in self.position_history:
            d_from_prev_view += np.clip(1 - (np.sqrt((self.xv - x)**2 + (self.yv - y)**2 + (cam_z - z)**2)/self.dist_from_prev_view_scale), 0, 1)

        height_weight = (cam_z - self.height[1])/(self.height[0]-self.height[1]) + 1e-2
        best_cost = (d_from_best_q / (self.dist_from_best_scale * height_weight)) * self.dist_from_best_gain

        prev_view_cost = d_from_prev_view * self.dist_from_prev_view_gain

        exp_inf_gain -= best_cost
        exp_inf_gain -= prev_view_cost

        exp_inf_gain_mask = exp_inf_gain.copy()
        exp_inf_gain_mask[d_from_robot > 0.10] = exp_inf_gain.min()
        ig_am = np.unravel_index(np.argmax(exp_inf_gain_mask), exp_inf_gain.shape)
        maxpos = self.gw.cell_to_pos([ig_am])[0]
        diff = maxpos - np.array([cam_x, cam_y])
        move_amt = 0.05
        if np.linalg.norm(diff) > move_amt:
            diff = diff/np.linalg.norm(diff) * move_amt

        ret = NextViewpointResponse()

        p = Point()
        p.x = diff[0]
        p.y = diff[1]
        p.z = -1 * ((move_amt - np.linalg.norm(diff))/move_amt * 0.01 + 0.01)
        ret.viewpoint = p
        ret.best_grasp = Float32MultiArray()
        ret.best_grasp.data = [q_am_pos[0], q_am_pos[1], 0.03, q_am_ang]

        show = gridshow('Display',
                 [cv2.resize(points, hist_ent.shape), hist_mean, hist_ent, np.exp(exp_inf_gain), best_cost, self.gw.visited],
                 [None, None, None, None, None, None],
                 [cv2.COLORMAP_BONE] + [cv2.COLORMAP_JET, ] * 4 + [cv2.COLORMAP_BONE],
                 3,
                 False)

        self.img_pub.publish(bridge.cv2_to_imgmsg(show))

        return ret

    def reset():


if __name__ == '__main__':
    rospy.init_node('panda_active_grasp')
    VEC = ViewpointEntropyCalculator()
    rospy.spin()
