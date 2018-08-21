#!/usr/bin/env python

from __future__ import division, print_function

import rospy

import time

import numpy as np
import cv2
from scipy.signal import convolve2d as conv2d
from scipy.ndimage.filters import gaussian_filter

from skimage.feature import peak_local_max

from dougsm_helpers.tf_helpers import current_robot_pose
from tf import transformations as tft
from dougsm_helpers.timeit import TimeIt

from ggcnn.ggcnn import predict, process_depth_image
from ggcnn.grasp import detect_grasps
from mvp_grasping.grasp_stats import update_batch, update_histogram_angle
from mvp_grasping.gridworld import GridWorld
from dougsm_helpers.gridshow import gridshow

from mvp_grasping.srv import NextViewpoint, NextViewpointResponse
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty as EmptySrv, EmptyResponse as EmptySrvResponse

import cv_bridge
bridge = cv_bridge.CvBridge()


class ViewpointEntropyCalculator:
    def __init__(self):
        self.hist_bins_q = rospy.get_param('~histogram/bins/quality')
        self.hist_bins_a = rospy.get_param('~histogram/bins/angle')

        self.dist_from_best_scale = rospy.get_param('~cost/dist_from_best_scale')
        self.dist_from_best_gain = rospy.get_param('~cost/dist_from_best_gain')
        self.dist_from_prev_view_scale = rospy.get_param('~cost/dist_from_prev_view_scale')
        self.dist_from_prev_view_gain = rospy.get_param('~cost/dist_from_prev_view_gain')

        self.height = (rospy.get_param('~height/z1'), rospy.get_param('~height/z2'))

        # Create a GridWorld where we will store values.
        self.gw_bounds = np.array([
            [rospy.get_param('~histogram/bounds/x1'), rospy.get_param('~histogram/bounds/y1')],
            [rospy.get_param('~histogram/bounds/x2'), rospy.get_param('~histogram/bounds/y2')]
        ])
        self.gw_res = rospy.get_param('~histogram/resolution')

        self.reset_gridworld(EmptySrv())
        self.hist_mean = 0

        # Useful meshgrid for distance calculations
        xs = np.arange(self.gw.bounds[0, 0], self.gw.bounds[1, 0] - 1e-6, self.gw.res) + self.gw.res / 2
        ys = np.arange(self.gw.bounds[0, 1], self.gw.bounds[1, 1] - 1e-6, self.gw.res) + self.gw.res / 2
        self._xv, self._yv = np.meshgrid(xs, ys)

        # Get the camera parameters
        cam_info_topic = rospy.get_param('~camera/info_topic')
        camera_info_msg = rospy.wait_for_message(cam_info_topic, CameraInfo)
        self.cam_K = np.array(camera_info_msg.K).reshape((3, 3))

        self.img_pub = rospy.Publisher('~visualisation', Image, queue_size=1)
        rospy.Service('~update_grid', NextViewpoint, self.update_service_handler)
        rospy.Service('~reset_grid', EmptySrv, self.reset_gridworld)

        self.base_frame = rospy.get_param('~camera/base_frame')
        self.camera_frame = rospy.get_param('~camera/camera_frame')
        self.img_crop_size = rospy.get_param('~camera/crop_size')
        self.img_crop_y_offset = rospy.get_param('~camera/crop_y_offset')
        self.cam_fov = rospy.get_param('~camera/fov')

        self.curr_depth_img = None
        self.curr_img_time = 0
        self.last_image_pose = None
        rospy.Subscriber(rospy.get_param('~camera/depth_topic'), Image, self._depth_img_callback, queue_size=1)

    def _depth_img_callback(self, msg):
        self.curr_img_time = time.time()
        self.last_image_pose = current_robot_pose(self.base_frame, self.camera_frame)
        self.curr_depth_img = bridge.imgmsg_to_cv2(msg)

    def update_service_handler(self, req):
        if self.curr_depth_img is None:
            rospy.logerr('No depth image received yet.')
            rospy.sleep(0.5)

        if time.time() - self.curr_img_time > 0.5:
            rospy.logerr('The Realsense node has died')
            return NextViewpointResponse()

        with TimeIt('Total'):
            with TimeIt('Update Histogram'):
                depth = self.curr_depth_img.copy()
                camera_pose = self.last_image_pose
                cam_p = camera_pose.position
                self.position_history.append(np.array([cam_p.x, cam_p.y, cam_p.z]))

                # For display purposes.
                newpos_pixel = self.gw.pos_to_cell(np.array([[cam_p.x, cam_p.y]]))[0]
                self.gw.visited[newpos_pixel[0], newpos_pixel[1]] = self.gw.visited.max() + 1

                cq = camera_pose.orientation
                camera_rot = tft.quaternion_matrix([cq.x, cq.y, cq.z, cq.w])[0:3, 0:3]

                # Do grasp prediction
                depth_crop, depth_nan_mask = process_depth_image(depth, self.img_crop_size, 300, return_mask=True, crop_y_offset=self.img_crop_y_offset)
                points, angle, width_img, _ = predict(depth_crop, process_depth=False, depth_nan_mask=depth_nan_mask)
                angle -= np.arcsin(camera_rot[0, 1])  # Correct for the rotation of the camera
                angle = (angle + np.pi/2) % np.pi  # Wrap [0, pi]

                # Convert to 3D positions.
                imh, imw = depth.shape
                x = ((np.vstack((np.linspace((imw - self.img_crop_size) // 2, (imw - self.img_crop_size) // 2 + self.img_crop_size, depth_crop.shape[1], np.float), )*depth_crop.shape[0]) - self.cam_K[0, 2])/self.cam_K[0, 0] * depth_crop).flatten()
                y = ((np.vstack((np.linspace((imh - self.img_crop_size) // 2 - self.img_crop_y_offset, (imh - self.img_crop_size) // 2 + self.img_crop_size - self.img_crop_y_offset, depth_crop.shape[0], np.float), )*depth_crop.shape[1]).T - self.cam_K[1,2])/self.cam_K[1, 1] * depth_crop).flatten()

                pos = np.dot(camera_rot, np.stack((x, y, depth_crop.flatten()))).T + np.array([[camera_pose.position.x, camera_pose.position.y, camera_pose.position.z]])
                pos[depth_nan_mask.flatten() == 1, :] = 0  # Get rid of NaNs
                pos[pos[:, 2] > 0.10, :] = 0  # Ignore obvious noise.

                cell_ids = self.gw.pos_to_cell(pos[:, :2])
                width_m =  2 / depth_crop * np.tan(self.cam_fov/300.0 * width_img / 2.0 / 180.0 * np.pi)
                width_m = width_img / 300.0 * 2.0 * depth_crop * np.tan(self.cam_fov * self.img_crop_size/depth.shape[0] / 2.0 / 180.0 * np.pi)

                update_batch([pos[:, 2], width_m.flatten()], cell_ids, self.gw.count, [self.gw.depth_mean, self.gw.width_mean], [self.gw.depth_var, self.gw.width_var])
                update_histogram_angle(points.flatten(), angle.flatten(), cell_ids, self.gw.hist)


            with TimeIt('Calculate Best Grasp'):
                # Marginalise over all angles.
                hist_sum_q = np.sum(self.gw.hist, axis=2)
                weights = np.arange(0.5/self.hist_bins_q, 1.0, 1/self.hist_bins_q)
                hist_mean = np.sum(hist_sum_q * weights.reshape((1, 1, -1)), axis=2)/(np.sum(hist_sum_q, axis=2) + 1e-6)
                hist_mean[self.gw.count == 0] = 0  # Ignore areas we haven't seen yet.
                hist_mean[0, :] = 0
                hist_mean[-1, :] = 0
                hist_mean[:, 0] = 0
                hist_mean[:, -1] = 0
                q_am = np.unravel_index(np.argmax(hist_mean), hist_mean.shape)
                # q_am_pos = self.gw.cell_to_pos([q_am])[0]

                # Interpolate position between the nearest neighbours.
                q_ama = np.array(q_am)
                conn_neighbours = np.array([q_ama + [-1, -1],
                                            q_ama + [-1, 0],
                                            q_ama + [-1, 1],
                                            q_ama + [0, -1],
                                            q_ama + [0, 0],
                                            q_ama + [0, 1],
                                            q_ama + [1, -1],
                                            q_ama + [1, 0],
                                            q_ama + [1, 1],
                                            ])
                neighbour_weights = hist_mean[conn_neighbours[:, 0], conn_neighbours[:, 1]]
                q_am_neigh = self.gw.cell_to_pos(conn_neighbours)
                q_am_neigh = np.average(q_am_neigh, weights=neighbour_weights, axis=0)
                q_am_pos = (q_am_neigh[0], q_am_neigh[1])

                best_grasp_hist = self.gw.hist[q_am[0], q_am[1], :, :]
                angle_weights = np.sum(best_grasp_hist - 1 * weights.reshape((1, -1)), axis=1)#/(np.sum(best_grasp_hist, axis=1) + 1e-6)
                ang_bins = np.arange(0.5/self.hist_bins_a, 1.0, 1/self.hist_bins_a) * np.pi
                q_am_ang = np.arctan2(
                    np.sum(np.sin(ang_bins) * angle_weights),
                    np.sum(np.cos(ang_bins) * angle_weights)
                ) - np.pi/2

                q_am_dep = self.gw.depth_mean[q_am]
                q_am_wid = self.gw.width_mean[q_am]

            with TimeIt('Calculate Information Gain'):
                hist_p = hist_sum_q / np.expand_dims(np.sum(hist_sum_q, axis=2) + 1e-6, -1)
                hist_ent = -np.sum(hist_p * np.log(hist_p+1e-6), axis=2)

                fov = int(cam_p.z * np.tan(self.cam_fov*self.img_crop_size/depth.shape[0] / 180.0 * np.pi) / self.gw.res)  # Field of view in gridworld cells
                exp_inf_gain = gaussian_filter(hist_ent, fov/6, mode='nearest', truncate=3)

            with TimeIt('Calculate Travel Cost'):
                # Distance from current robot pos.
                d_from_robot = np.sqrt((self._xv - cam_p.x)**2 + (self._yv - cam_p.y)**2)
                d_robot_scalar = (d_from_robot - d_from_robot.min())/(d_from_robot.max() - d_from_robot.min())

                # Distance from best detected grasp.
                d_from_best_q = np.sqrt((self._xv - q_am_pos[0])**2 + (self._yv - q_am_pos[1])**2)  # Cost of moving away from the best grasp.
                height_weight = (cam_p.z - self.height[1])/(self.height[0]-self.height[1]) + 1e-2
                best_cost = (d_from_best_q / (self.dist_from_best_scale * height_weight)) * self.dist_from_best_gain

                # Distance from previous viewpoints.
                d_from_prev_view = np.zeros(self.gw.shape)
                for x, y, z in self.position_history:
                    d_from_prev_view += np.clip(1 - (np.sqrt((self._xv - x)**2 + (self._yv - y)**2 + (cam_p.z - z)**2)/self.dist_from_prev_view_scale), 0, 1)
                prev_view_cost = d_from_prev_view * self.dist_from_prev_view_gain

                # Calculate total reward.
                exp_inf_gain -= best_cost
                exp_inf_gain -= prev_view_cost

                # Generate Command
                exp_inf_gain_mask = exp_inf_gain.copy()
                exp_inf_gain_mask[d_from_robot > 0.05] = exp_inf_gain.min()
                ig_am = np.unravel_index(np.argmax(exp_inf_gain_mask), exp_inf_gain.shape)
                maxpos = self.gw.cell_to_pos([ig_am])[0]
                diff = maxpos - np.array([cam_p.x, cam_p.y])
                move_amt = 0.05
                if np.linalg.norm(diff) > move_amt:
                    diff = diff/np.linalg.norm(diff) * move_amt

            with TimeIt('Response'):
                ret = NextViewpointResponse()

                p = Point()
                p.x = diff[0]
                p.y = diff[1]
                p.z = -1 * ((move_amt - np.linalg.norm(diff))/move_amt * 0.01 + 0.01)
                ret.viewpoint = p
                ret.best_grasp = Float32MultiArray()
                ret.best_grasp.data = [q_am_pos[0], q_am_pos[1], q_am_dep, q_am_ang, q_am_wid]

                show = gridshow('Display',
                         [cv2.resize(points, hist_ent.shape), hist_mean, hist_ent, np.exp(exp_inf_gain), best_cost, self.gw.visited],
                         [None, None, None, None, None, None],
                         [cv2.COLORMAP_BONE] + [cv2.COLORMAP_JET, ] * 4 + [cv2.COLORMAP_BONE],
                         3,
                         False)

                self.img_pub.publish(bridge.cv2_to_imgmsg(show))

        return ret

    def reset_gridworld(self, req):
        self.gw = GridWorld(self.gw_bounds, self.gw_res)
        self.gw.add_grid('visited', 0.0)
        self.gw.add_grid('hist', 1.0, extra_dims=(self.hist_bins_a, self.hist_bins_q))
        self.gw.add_grid('depth_mean', 0.0)
        self.gw.add_grid('depth_var', 0.0)
        self.gw.add_grid('width_mean', 0.0)
        self.gw.add_grid('width_var', 0.0)
        self.gw.add_grid('count', 0.0)
        self.position_history = []
        return EmptySrvResponse()


if __name__ == '__main__':
    rospy.init_node('grasp_entropy_node')
    VEC = ViewpointEntropyCalculator()
    rospy.spin()
