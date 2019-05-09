#! /usr/bin/env python

from __future__ import division, print_function

import rospy

import os
import time
import datetime


from std_msgs.msg import Int16
from geometry_msgs.msg import Twist
from franka_msgs.msg import FrankaState, Errors as FrankaErrors

from franka_control_wrappers.panda_commander import PandaCommander

import dougsm_helpers.tf_helpers as tfh
from dougsm_helpers.ros_control import ControlSwitcher

from ggcnn.msg import Grasp
from ggcnn.srv import GraspPrediction

from mvp_grasping.panda_base_grasping_controller import Logger, Run, Experiment

Run.log_properties = ['success', 'time', 'quality']
Experiment.log_properties = ['success_rate', 'mpph']


class PandaOpenLoopGraspController(object):
    """
    Perform open-loop grasps from a single viewpoint using the Panda robot.
    """
    def __init__(self):
        ggcnn_service_name = '/ggcnn_service'
        rospy.wait_for_service(ggcnn_service_name + '/predict')
        self.ggcnn_srv = rospy.ServiceProxy(ggcnn_service_name + '/predict', GraspPrediction)

        self.curr_velocity_publish_rate = 100.0  # Hz
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.max_velo = 0.10
        self.curr_velo = Twist()
        self.best_grasp = Grasp()

        self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                                   'velocity': 'cartesian_velocity_node_controller'})
        self.cs.switch_controller('moveit')
        self.pc = PandaCommander(group_name='panda_arm_hand')

        self.robot_state = None
        self.ROBOT_ERROR_DETECTED = False
        self.BAD_UPDATE = False
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

        # Centre and above the bin
        self.pregrasp_pose = [(rospy.get_param('/grasp_entropy_node/histogram/bounds/x2') + rospy.get_param('/grasp_entropy_node/histogram/bounds/x1'))/2 - 0.03,
                              (rospy.get_param('/grasp_entropy_node/histogram/bounds/y2') + rospy.get_param('/grasp_entropy_node/histogram/bounds/y1'))/2 + 0.10,
                              rospy.get_param('/grasp_entropy_node/height/z1') + 0.05,
                              2**0.5/2, -2**0.5/2, 0, 0]

        self.last_weight = 0
        self.__weight_increase_check()

        self.experiment = Experiment()

    def __recover_robot_from_error(self):
        rospy.logerr('Recovering')
        self.pc.recover()
        rospy.logerr('Done')
        self.ROBOT_ERROR_DETECTED = False

    def __weight_increase_check(self):
        try:
            w = rospy.wait_for_message('/scales/weight', Int16, timeout=2).data
            increased = w > self.last_weight
            self.last_weight = w
            return increased
        except:
            return raw_input('No weight. Success? [1/0]') == '1'

    def __robot_state_callback(self, msg):
        self.robot_state = msg
        if any(self.robot_state.cartesian_collision):
            if not self.ROBOT_ERROR_DETECTED:
                rospy.logerr('Detected Cartesian Collision')
            self.ROBOT_ERROR_DETECTED = True
        for s in FrankaErrors.__slots__:
            if getattr(msg.current_errors, s):
                self.stop()
                if not self.ROBOT_ERROR_DETECTED:
                    rospy.logerr('Robot Error Detected')
                self.ROBOT_ERROR_DETECTED = True

    def __execute_best_grasp(self):
            self.cs.switch_controller('moveit')

            ret = self.ggcnn_srv.call()
            if not ret.success:
                return False
            best_grasp = ret.best_grasp
            self.best_grasp = best_grasp

            tfh.publish_pose_as_transform(best_grasp.pose, 'panda_link0', 'G', 0.5)

            if raw_input('Continue?') == '0':
                return False

            # Offset for initial pose.
            initial_offset = 0.10
            LINK_EE_OFFSET = 0.138

            # Add some limits, plus a starting offset.
            best_grasp.pose.position.z = max(best_grasp.pose.position.z - 0.01, 0.026)  # 0.021 = collision with ground
            best_grasp.pose.position.z += initial_offset + LINK_EE_OFFSET  # Offset from end efector position to

            self.pc.set_gripper(best_grasp.width, wait=False)
            rospy.sleep(0.1)
            self.pc.goto_pose(best_grasp.pose, velocity=0.1)

            # Reset the position
            best_grasp.pose.position.z -= initial_offset + LINK_EE_OFFSET

            self.cs.switch_controller('velocity')
            v = Twist()
            v.linear.z = -0.05

            # Monitor robot state and descend
            while self.robot_state.O_T_EE[-2] > best_grasp.pose.position.z and not any(self.robot_state.cartesian_contact) and not self.ROBOT_ERROR_DETECTED:
                self.curr_velo_pub.publish(v)
                rospy.sleep(0.01)
            v.linear.z = 0
            self.curr_velo_pub.publish(v)

            # Check for collisions
            if self.ROBOT_ERROR_DETECTED:
                return False

            # close the fingers.
            rospy.sleep(0.2)
            self.pc.grasp(0, force=2)

            # Sometimes triggered by closing on something that pushes the robot
            if self.ROBOT_ERROR_DETECTED:
                return False

            return True

    def stop(self):
        self.pc.stop()
        self.curr_velo = Twist()
        self.curr_velo_pub.publish(self.curr_velo)

    def go(self):
        raw_input('Press Enter to Start.')
        while not rospy.is_shutdown():
            self.cs.switch_controller('moveit')
            self.pc.goto_named_pose('grip_ready', velocity=0.25)
            self.pc.goto_pose(self.pregrasp_pose, velocity=0.25)
            self.pc.set_gripper(0.1)

            self.cs.switch_controller('velocity')

            run = self.experiment.new_run()
            run.start()
            grasp_ret = self.__execute_best_grasp()
            run.stop()

            if not grasp_ret or self.ROBOT_ERROR_DETECTED:
                rospy.logerr('Something went wrong, aborting this run')
                if self.ROBOT_ERROR_DETECTED:
                    self.__recover_robot_from_error()
                continue

            # Release Object
            self.cs.switch_controller('moveit')
            self.pc.goto_named_pose('grip_ready', velocity=0.5)
            self.pc.goto_named_pose('drop_box', velocity=0.5)
            self.pc.set_gripper(0.07)

            # Check success using the scales.
            rospy.sleep(1.0)
            grasp_success = self.__weight_increase_check()
            if not grasp_success:
                rospy.logerr("Failed Grasp")
            else:
                rospy.logerr("Successful Grasp")

            run.success = grasp_success
            run.quality = self.best_grasp.quality
            run.save()


if __name__ == '__main__':
    rospy.init_node('panda_open_loop_grasp')
    pg = PandaOpenLoopGraspController()
    pg.go()
