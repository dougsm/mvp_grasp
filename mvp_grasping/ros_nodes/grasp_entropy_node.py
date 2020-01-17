#!/usr/bin/env python

from __future__ import division, print_function

import rospy

import time

import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter

import dougsm_helpers.tf_helpers as tfh
from tf import transformations as tft
from dougsm_helpers.timeit import TimeIt

from ggcnn.ggcnn import predict, process_depth_image
from mvp_grasping.grasp_stats import update_batch, update_histogram_angle
from mvp_grasping.gridworld import GridWorld
from dougsm_helpers.gridshow import gridshow

from mvp_grasping.srv import NextViewpoint, NextViewpointResponse, AddFailurePoint, AddFailurePointResponse
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Empty as EmptySrv, EmptyResponse as EmptySrvResponse

import cv_bridge
bridge = cv_bridge.CvBridge()

TimeIt.print_output = False


class ViewpointEntropyCalculator:
    """
    This class implements the Grid World portion of the Multi-View controller.
    """
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

        self.counter = 0
        self.curr_depth_img = None
        self.curr_img_time = 0
        self.last_image_pose = None
        rospy.Subscriber(rospy.get_param('~camera/depth_topic'), Image, self._depth_img_callback, queue_size=1)

    def _depth_img_callback(self, msg):
        """
        Doing a rospy.wait_for_message is super slow, compared to just subscribing and keeping the newest one.
        """
        self.curr_img_time = time.time()
        self.last_image_pose = tfh.current_robot_pose(self.base_frame, self.camera_frame)
        self.curr_depth_img = bridge.imgmsg_to_cv2(msg)

    def update_service_handler(self, req):
        """
        Update the GridWorld with a new observation, compute the viewpoint entropy and generate a new command.
        :param req: Ignored
        :return: NextViewpointResponse (success flag, best grsap, velocity command)
        """

        # Some initial checks
        if self.curr_depth_img is None:
            rospy.logerr('No depth image received yet.')
            rospy.sleep(0.5)

        if time.time() - self.curr_img_time > 0.5:
            rospy.logerr('The Realsense node has died')
            return NextViewpointResponse()

        with TimeIt('Total'):
            with TimeIt('Update Histogram'):
                # Step 1: Perform a GG-CNN prediction and update the grid world with the observations

                self.no_viewpoints += 1
                depth = self.curr_depth_img.copy()
                camera_pose = self.last_image_pose
                cam_p = camera_pose.position
                self.position_history.append(np.array([cam_p.x, cam_p.y, cam_p.z, 0]))

                # For display purposes.
                newpos_pixel = self.gw.pos_to_cell(np.array([[cam_p.x, cam_p.y]]))[0]
                self.gw.visited[newpos_pixel[0], newpos_pixel[1]] = self.gw.visited.max() + 1

                camera_rot = tft.quaternion_matrix(tfh.quaternion_to_list(camera_pose.orientation))[0:3, 0:3]

                # Do grasp prediction
                depth_crop, depth_nan_mask = process_depth_image(depth, self.img_crop_size, 300, return_mask=True, crop_y_offset=self.img_crop_y_offset)
                points, angle, width_img, _ = predict(depth_crop, process_depth=False, depth_nan_mask=depth_nan_mask)
                angle -= np.arcsin(camera_rot[0, 1])  # Correct for the rotation of the camera
                angle = (angle + np.pi/2) % np.pi  # Wrap [0, pi]

                # Convert to 3D positions.
                imh, imw = depth.shape
                x = ((np.vstack((np.linspace((imw - self.img_crop_size) // 2, (imw - self.img_crop_size) // 2 + self.img_crop_size, depth_crop.shape[1], np.float), )*depth_crop.shape[0]) - self.cam_K[0, 2])/self.cam_K[0, 0] * depth_crop).flatten()
                y = ((np.vstack((np.linspace((imh - self.img_crop_size) // 2 - self.img_crop_y_offset, (imh - self.img_crop_size) // 2 + self.img_crop_size - self.img_crop_y_offset, depth_crop.shape[0], np.float), )*depth_crop.shape[1]).T - self.cam_K[1,2])/self.cam_K[1, 1] * depth_crop).flatten()
                pos = np.dot(camera_rot, np.stack((x, y, depth_crop.flatten()))).T + np.array([[cam_p.x, cam_p.y, cam_p.z]])

                # Clean the data a bit.
                pos[depth_nan_mask.flatten() == 1, :] = 0  # Get rid of NaNs
                pos[pos[:, 2] > 0.17, :] = 0  # Ignore obvious noise.
                pos[pos[:, 2] < 0.0, :] = 0  # Ignore obvious noise.

                cell_ids = self.gw.pos_to_cell(pos[:, :2])
                width_m = width_img / 300.0 * 2.0 * depth_crop * np.tan(self.cam_fov * self.img_crop_size/depth.shape[0] / 2.0 / 180.0 * np.pi)

                update_batch([pos[:, 2], width_m.flatten()], cell_ids, self.gw.count, [self.gw.depth_mean, self.gw.width_mean], [self.gw.depth_var, self.gw.width_var])
                update_histogram_angle(points.flatten(), angle.flatten(), cell_ids, self.gw.hist)

            with TimeIt('Calculate Best Grasp'):
                # Step 2: Compute the position of the best grasp in the GridWorld

                # Sum over all angles to get the grasp quality only.
                hist_sum_q = np.sum(self.gw.hist, axis=2)
                weights = np.arange(0.5/self.hist_bins_q, 1.0, 1/self.hist_bins_q)
                hist_mean = np.sum(hist_sum_q * weights.reshape((1, 1, -1)), axis=2)/(np.sum(hist_sum_q, axis=2) + 1e-6)
                hist_mean[self.gw.count == 0] = 0  # Ignore areas we haven't seen yet.
                hist_mean[0, :] = 0  # Ignore single pixel along each edge.
                hist_mean[-1, :] = 0
                hist_mean[:, 0] = 0
                hist_mean[:, -1] = 0
                hist_mean -= self.fgw.failures
                hist_mean = np.clip(hist_mean, 0.0, 1.0)

                # ArgMax of grasp quality
                q_am = np.unravel_index(np.argmax(hist_mean), hist_mean.shape)

                # Interpolate position between the neighbours of the best grasp, weighted by quality
                q_ama = np.array(q_am)
                conn_neighbours = np.array([q_ama])  # Disable rounding
                neighbour_weights = hist_mean[conn_neighbours[:, 0], conn_neighbours[:, 1]]
                q_am_neigh = self.gw.cell_to_pos(conn_neighbours)
                q_am_neigh_avg = np.average(q_am_neigh, weights=neighbour_weights, axis=0)
                q_am_pos = (q_am_neigh_avg[0], q_am_neigh_avg[1])  # This is the grasp center

                # Perform same weighted averaging of the angles.
                best_grasp_hist = self.gw.hist[conn_neighbours[:, 0], conn_neighbours[:, 1], :, :]
                angle_weights = np.sum((best_grasp_hist - 1) * weights.reshape((1, 1, -1)), axis=2)
                ang_bins = (np.arange(0.5/self.hist_bins_a, 1.0, 1/self.hist_bins_a) * np.pi).reshape(1, -1)

                # Compute the weighted vector mean of the sin/cos components of the angle predictions
                # Do double angles so that -np.pi/2 == np.pi/2, then unwrap
                q_am_ang = np.arctan2(
                    np.sum(np.sin(ang_bins*2) * angle_weights * neighbour_weights.reshape(-1, 1)),
                    np.sum(np.cos(ang_bins*2) * angle_weights * neighbour_weights.reshape(-1, 1))
                )
                if q_am_ang < 0:
                    q_am_ang += 2*np.pi
                q_am_ang = q_am_ang/2.0 - np.pi/2

                # Get the depth and width at the grasp center
                q_am_dep = self.gw.depth_mean[q_am]
                q_am_wid = self.gw.width_mean[q_am]

            with TimeIt('Calculate Information Gain'):
                # Step 3: Compute the expected information gain from a viewpoint above every cell in the GridWorld

                # Compute entropy per cell.
                hist_p = hist_sum_q / np.expand_dims(np.sum(hist_sum_q, axis=2) + 1e-6, -1)
                hist_ent = -np.sum(hist_p * np.log(hist_p+1e-6), axis=2)

                # Treat camera field of view as a Gaussian
                # Field of view in number gridworld cells
                fov = int(cam_p.z * 2 * np.tan(self.cam_fov*self.img_crop_size/depth.shape[0]/2.0 / 180.0 * np.pi) / self.gw.res)
                exp_inf_gain = gaussian_filter(hist_ent, fov/6, truncate=3)

                # Track changes by KL Divergence (not used/disabled by default)
                kl_divergence = np.sum(hist_p * np.log((hist_p+1e-6)/(self.gw.hist_p_prev+1e-6)), axis=2)
                self.gw.hist_p_prev = hist_p
                kl_divergence[0, :] = 0
                kl_divergence[-1, :] = 0
                kl_divergence[:, 0] = 0
                kl_divergence[:, -1] = 0
                norm_i_gain = 1 - np.exp(-1 * kl_divergence.sum())
                self.position_history[-1][-1] = norm_i_gain

            with TimeIt('Calculate Travel Cost'):
                # Step 4: Compute cost of moving away from the best detected grasp.

                # Distance from current robot pos.
                d_from_robot = np.sqrt((self._xv - cam_p.x)**2 + (self._yv - cam_p.y)**2)

                # Distance from best detected grasp, weighted by the robot's current height (Z axis)
                d_from_best_q = np.sqrt((self._xv - q_am_pos[0])**2 + (self._yv - q_am_pos[1])**2)  # Cost of moving away from the best grasp.
                height_weight = (cam_p.z - self.height[1])/(self.height[0]-self.height[1]) + 1e-2
                height_weight = max(min(height_weight, 1.0), 0.0)
                best_cost = (d_from_best_q / self.dist_from_best_scale) * (1-height_weight) * self.dist_from_best_gain

                # Distance from previous viewpoints (dist_from_prev_view_gain is 0 by default)
                d_from_prev_view = np.zeros(self.gw.shape)
                for x, y, z, kl in self.position_history:
                    d_from_prev_view += np.clip(1 - (np.sqrt((self._xv - x)**2 + (self._yv - y)**2 + 0*(cam_p.z - z)**2)/self.dist_from_prev_view_scale), 0, 1) * (1-kl)
                prev_view_cost = d_from_prev_view * self.dist_from_prev_view_gain

                # Calculate total expected information gain.
                exp_inf_gain_before = exp_inf_gain.copy()
                exp_inf_gain -= best_cost
                exp_inf_gain -= prev_view_cost

                # Compute local direction of maximum information gain
                exp_inf_gain_mask = exp_inf_gain.copy()
                greedy_window = 0.1
                exp_inf_gain_mask[d_from_robot > greedy_window] = exp_inf_gain.min()
                ig_am = np.unravel_index(np.argmax(exp_inf_gain_mask), exp_inf_gain.shape)
                maxpos = self.gw.cell_to_pos([ig_am])[0]
                diff = (maxpos - np.array([cam_p.x, cam_p.y]))/greedy_window
                # Maximum of 1
                if np.linalg.norm(diff) > 1.0:
                    diff = diff/np.linalg.norm(diff)

            with TimeIt('Response'):
                # Step 5: Generate a Response

                ret = NextViewpointResponse()
                ret.success = True
                ret.no_viewpoints = self.no_viewpoints

                # xyz velocity normalised to 1
                ret.velocity_cmd.linear.x = diff[0]
                ret.velocity_cmd.linear.y = diff[1]
                ret.velocity_cmd.linear.z = -1 * (np.sqrt(1 - diff[0]**2 - diff[1]**2))

                # Grasp pose
                ret.best_grasp.pose.position.x = q_am_pos[0]
                ret.best_grasp.pose.position.y = q_am_pos[1]
                ret.best_grasp.pose.position.z = q_am_dep
                q = tft.quaternion_from_euler(np.pi, 0, q_am_ang - np.pi/2)
                ret.best_grasp.pose.orientation = tfh.list_to_quaternion(q)

                ret.best_grasp.quality = hist_mean[q_am[0], q_am[1]]
                ret.best_grasp.width = q_am_wid
                ret.best_grasp.entropy = hist_ent[q_am[0], q_am[1]]

                # Normalise for plotting purposes
                exp_inf_gain = (exp_inf_gain - exp_inf_gain.min())/(exp_inf_gain.max()-exp_inf_gain.min())*(exp_inf_gain_before.max()-exp_inf_gain_before.min())
                show = gridshow('Display',
                         [cv2.resize(points, hist_ent.shape), hist_mean, hist_ent, exp_inf_gain, exp_inf_gain_before, self.gw.visited],
                         [None, None, None, (exp_inf_gain.min(), exp_inf_gain_before.max()), (exp_inf_gain.min(), exp_inf_gain_before.max()), None],
                         [cv2.COLORMAP_JET] + [cv2.COLORMAP_JET, ] * 4 + [cv2.COLORMAP_BONE],
                         3,
                         False)

                self.img_pub.publish(bridge.cv2_to_imgmsg(show))

        # For dumping things to npz files
        if False:
            kwargs = {
                'M': self.gw.hist,
                'depth_crop': depth_crop,
                'points': points,
                'hist_sum_q': hist_sum_q,
                'hist_mean': hist_mean,
                'q_am': q_am,
                'q_am_pos': q_am_pos,
                'best_grasp_hist': best_grasp_hist,
                'hist_ent': hist_ent,
                'best_cost': best_cost,
                'exp_inf_gain': exp_inf_gain,
                'pos_history': np.array(self.position_history),
                'visited': self.gw.visited,
                'depth': depth_crop,
                'v': diff
            }
            np.savez('/home/guest/numpy_out/%d.npz'%self.counter, **kwargs)
            self.counter += 1

        return ret

    def reset_gridworld(self, req):
        """
        Reset gridworld to initial conditions.
        :param req: Ignored (trigger)
        :return: Empty Response
        """
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
        self.no_viewpoints = 0
        self.counter = 0
        return EmptySrvResponse()

    def add_failure_point_callback(self, req):
        """
        If a grasp fails, be able to blacklist that area on the next attempt.
        :param req: AddFailurePointRequest (Point to blacklist)
        :return: AddFailurePointResponse (Empty)
        """
        new_fp = np.zeros_like(self.fgw.failures)
        cell_id = self.gw.pos_to_cell(np.array([[req.point.x, req.point.y]]))[0]
        new_fp[cell_id[0], cell_id[1]] = 1.0
        new_fp = gaussian_filter(new_fp, 0.0075/self.gw.res, mode='nearest', truncate=3)
        self.fgw.failures = 0.5*self.fgw.failures + 0.5* new_fp/new_fp.max()
        return AddFailurePointResponse()


if __name__ == '__main__':
    rospy.init_node('grasp_entropy_node')
    VEC = ViewpointEntropyCalculator()
    rospy.spin()
