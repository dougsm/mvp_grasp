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

from mvp_grasping.srv import NextViewpoint, NextViewpointResponse, AddFailurePoint, AddFailurePointResponse
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty as EmptySrv, EmptyResponse as EmptySrvResponse

import cv_bridge
bridge = cv_bridge.CvBridge()

TimeIt.print_output = False

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
        self.fgw = GridWorld(self.gw_bounds, self.gw_res)
        self.fgw.add_grid('failures', 0.0)

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
        rospy.Service('~add_failure_point', AddFailurePoint, self.add_failure_point_callback)

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
                self.position_history.append(np.array([cam_p.x, cam_p.y, cam_p.z, 0]))

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
                # Marginalise over all anglenp.sum(hist_p * np.log((hist_p + 1e-6)/(hist_pt1 + 1e-6)), axis=2)s.
                hist_sum_q = np.sum(self.gw.hist, axis=2)
                weights = np.arange(0.5/self.hist_bins_q, 1.0, 1/self.hist_bins_q)
                hist_mean = np.sum(hist_sum_q * weights.reshape((1, 1, -1)), axis=2)/(np.sum(hist_sum_q, axis=2) + 1e-6)
                hist_mean[self.gw.count == 0] = 0  # Ignore areas we haven't seen yet.
                hist_mean[0, :] = 0
                hist_mean[-1, :] = 0
                hist_mean[:, 0] = 0
                hist_mean[:, -1] = 0
                hist_mean -= self.fgw.failures
                hist_mean = np.clip(hist_mean, 0.0, 1.0)

                # d_from_robot = np.sqrt((self._xv - cam_p.x)**2 + (self._yv - cam_p.y)**2)
                # d_from_robot_scalar = (d_from_robot - d_from_robot.min())/(d_from_robot.max() - d_from_robot.min())
                # hist_mean *= (1 - d_from_robot_scalar) * 0.2 + 0.8

                q_am = np.unravel_index(np.argmax(hist_mean), hist_mean.shape)
                q_am_pos = self.gw.cell_to_pos([q_am])[0]

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
                q_am_neigh_avg = np.average(q_am_neigh, weights=neighbour_weights, axis=0)
                q_am_pos = (q_am_neigh_avg[0], q_am_neigh_avg[1])

                # best_grasp_hist = self.gw.hist[q_am[0], q_am[1], :, :]
                best_grasp_hist = self.gw.hist[conn_neighbours[:, 0], conn_neighbours[:, 1], :, :]
                angle_weights = np.sum(best_grasp_hist - 1 * weights.reshape((1, 1, -1)), axis=2)
                ang_bins = (np.arange(0.5/self.hist_bins_a, 1.0, 1/self.hist_bins_a) * np.pi).reshape(1, -1)
                q_am_ang = np.arctan2(
                    np.sum(np.sin(ang_bins) * angle_weights * neighbour_weights.reshape(-1, 1)),
                    np.sum(np.cos(ang_bins) * angle_weights * neighbour_weights.reshape(-1, 1))
                ) - np.pi/2
                # q_am_ang = np.average(q_am_ang, weights=neighbour_weights)
                # print(q_am_ang)

                q_am_dep = self.gw.depth_mean[q_am]
                q_am_wid = self.gw.width_mean[q_am]

            with TimeIt('Calculate Information Gain'):
                hist_p = hist_sum_q / np.expand_dims(np.sum(hist_sum_q, axis=2) + 1e-6, -1)
                hist_ent = -np.sum(hist_p * np.log(hist_p+1e-6), axis=2)
                # Camera field of view in grid cells.
                fov = int(cam_p.z * 2 * np.tan(self.cam_fov*self.img_crop_size/depth.shape[0]/2.0 / 180.0 * np.pi) / self.gw.res)  # Field of view in gridworld cells
                exp_inf_gain = gaussian_filter(hist_ent, fov/6, truncate=3)

                kl_divergence = np.sum(hist_p * np.log((hist_p+1e-6)/(self.gw.hist_p_prev+1e-6)), axis=2)
                # exp_inf_gain = gaussian_filter(1-np.exp(-kl_divergence), fov/6, truncate=3)

                hist_ent_prev = -np.sum(self.gw.hist_p_prev * np.log(self.gw.hist_p_prev+1e-6), axis=2)
                change_in_entropy = np.abs(hist_ent_prev - hist_ent)
                # exp_inf_gain = gaussian_filter(change_in_entropy, fov/6, truncate=3.0)

                self.gw.hist_p_prev = hist_p
                kl_divergence[0, :] = 0
                kl_divergence[-1, :] = 0
                kl_divergence[:, 0] = 0
                kl_divergence[:, -1] = 0
                norm_i_gain = 1 - np.exp(-1 * kl_divergence.sum())
                self.position_history[-1][-1] = norm_i_gain

            with TimeIt('Calculate Travel Cost'):
                # Distance from current robot pos.
                d_from_robot = np.sqrt((self._xv - cam_p.x)**2 + (self._yv - cam_p.y)**2)
                d_robot_scalar = (d_from_robot - d_from_robot.min())/(d_from_robot.max() - d_from_robot.min())

                # Distance from best detected grasp.
                d_from_best_q = np.sqrt((self._xv - q_am_pos[0])**2 + (self._yv - q_am_pos[1])**2)  # Cost of moving away from the best grasp.
                height_weight = (cam_p.z - self.height[1])/(self.height[0]-self.height[1]) + 1e-2
                height_weight = max(min(height_weight, 1.0), 0.0)
                # best_cost = (d_from_best_q / (self.dist_from_best_scale * height_weight)) * self.dist_from_best_gain
                best_cost = (d_from_best_q / self.dist_from_best_scale) * (1-height_weight) * self.dist_from_best_gain

                # Distance from previous viewpoints.
                d_from_prev_view = np.zeros(self.gw.shape)
                for x, y, z, kl in self.position_history:
                    d_from_prev_view += np.clip(1 - (np.sqrt((self._xv - x)**2 + (self._yv - y)**2 + 0*(cam_p.z - z)**2)/self.dist_from_prev_view_scale), 0, 1) * (1-kl)
                prev_view_cost = d_from_prev_view * self.dist_from_prev_view_gain

                # Calculate total reward.
                exp_inf_gain_before = exp_inf_gain.copy()
                exp_inf_gain -= best_cost
                exp_inf_gain -= prev_view_cost
                print(exp_inf_gain_before.max(), (best_cost).max(), exp_inf_gain.max())

                # Generate Command
                exp_inf_gain_mask = exp_inf_gain.copy()
                exp_inf_gain_mask[d_from_robot > 0.10] = exp_inf_gain.min()
                ig_am = np.unravel_index(np.argmax(exp_inf_gain_mask), exp_inf_gain.shape)
                maxpos = self.gw.cell_to_pos([ig_am])[0]
                diff = maxpos - np.array([cam_p.x, cam_p.y])
                move_amt = 0.05
                if np.linalg.norm(diff) > move_amt:
                    diff = diff/np.linalg.norm(diff) * move_amt

                # diff_q = q_am_pos - np.array([cam_p.x, cam_p.y])
                # if np.linalg.norm(diff_q) > move_amt:
                #     diff_q = diff_q/np.linalg.norm(diff_q) * move_amt
                #
                # TimeIt.print_output = False
                # gamma = 10
                # diff = height_weight**gamma * diff + (1 - height_weight**gamma) * diff_q

            # with TimeIt('Normals'):
            #     pts = np.vstack((q_am_neigh.T, self.gw.depth_mean[conn_neighbours[:, 0], conn_neighbours[:, 1]]))
            #     C = np.cov(pts)
            #     eigvals, v = np.linalg.eig(C)
            #     i = np.argmin(eigvals)
            #     normals = v[:, i]/np.linalg.norm(v[:, i]) * (-1 if v[2, i] < 0 else 1)
            #     print(normals)

            with TimeIt('Response'):
                ret = NextViewpointResponse()
                ret.velocity_cmd.linear.x = diff[0]
                ret.velocity_cmd.linear.y = diff[1]
                ret.velocity_cmd.linear.z = -1 * (np.sqrt(move_amt**2 - p.x**2 - p.y**2)) * 0.5

                ret.best_grasp.pose.position.x = q_am_pos[0]
                ret.best_grasp.pose.position.y = q_am_pos[1]
                ret.best_grasp.pose.position.z = q_am_dep

                q = tft.quaternion_from_euler(np.pi, 0, q_am_angle - np.pi/2)
                ret.best_grasp.pose.orientation.x = q[0]
                ret.best_grasp.pose.orientation.y = q[1]
                ret.best_grasp.pose.orientation.z = q[2]
                ret.best_grasp.pose.orientation.w = q[3]

                ret.best_grasp.quality = hist_mean[q_am[0], q_am[1]]
                ret.best_grasp.width = q_am_wid

                exp_inf_gain = (exp_inf_gain - exp_inf_gain.min())/(exp_inf_gain.max()-exp_inf_gain.min())*(exp_inf_gain_before.max()-exp_inf_gain_before.min())
                show = gridshow('Display',
                         [cv2.resize(points, hist_ent.shape), hist_mean, change_in_entropy, exp_inf_gain, exp_inf_gain_before, self.gw.visited],
                         [None, None, None, (exp_inf_gain.min(), exp_inf_gain_before.max()), (exp_inf_gain.min(), exp_inf_gain_before.max()), None],
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
        self.gw.add_grid('hist_p_prev', 1.0/self.hist_bins_q, extra_dims=(self.hist_bins_q, ))
        self.position_history = []
        return EmptySrvResponse()

    def add_failure_point_callback(self, req):
        new_fp = np.zeros_like(self.fgw.failures)
        cell_id = self.gw.pos_to_cell(np.array([[req.point.x, req.point.y]]))[0]
        new_fp[cell_id[0], cell_id[1]] = 1.0
        new_fp = gaussian_filter(new_fp, 1, mode='nearest', truncate=3)
        self.fgw.failures = 0.5*self.fgw.failures + 0.5* new_fp/new_fp.max()
        return AddFailurePointResponse()

if __name__ == '__main__':
    rospy.init_node('grasp_entropy_node')
    VEC = ViewpointEntropyCalculator()
    rospy.spin()
