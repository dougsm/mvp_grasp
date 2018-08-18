#! /usr/bin/env python

from __future__ import division, print_function

import rospy
import tf.transformations as tft

import numpy as np

from std_msgs.msg import Empty
from geometry_msgs.msg import Twist, Pose
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from std_srvs.srv import Empty as EmptySrv

from franka_control_wrappers.panda_commander import PandaCommander

from dougsm_helpers.tf_helpers import current_robot_pose, publish_tf_quaterion_as_transform, convert_pose, publish_pose_as_transform
from dougsm_helpers.timeit import TimeIt
from dougsm_helpers.ros_control import ControlSwitcher

from mvp_grasping.srv import NextViewpoint, NextViewpointRequest

DIE = False

class ActiveGraspController:
    def __init__(self):
        entropy_node_name = '/grasp_entropy_node'
        rospy.wait_for_service(entropy_node_name + '/update_grid')
        self.entropy_srv = rospy.ServiceProxy(entropy_node_name + '/update_grid', NextViewpoint)
        rospy.wait_for_service(entropy_node_name + '/reset_grid')
        self.entropy_reset_srv = rospy.ServiceProxy(entropy_node_name + '/reset_grid', EmptySrv)

        self.curr_velocity_publish_rate = 100.0  # Hz
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.curr_velo = Twist()
        self.best_grasp = Pose()

        self.update_rate = 4.0  # Hz
        update_topic_name = '~/update'
        self.update_pub = rospy.Publisher(update_topic_name, Empty, queue_size=1)
        rospy.Subscriber(update_topic_name, Empty, self.__update_callback, queue_size=1)

        self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                                   'velocity': 'cartesian_velocity_node_controller'})
        self.cs.switch_controller('moveit')
        self.pc = PandaCommander(group_name='panda_arm_hand')

        self.robot_state = None
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

    def __robot_state_callback(self, msg):
        self.robot_state = msg
        for s in FrankaErrors.__slots__:
            if getattr(msg.current_errors, s):
                self.stop()
                rospy.logerr('Robot Error Detected')
                DIE = True
                raise SystemExit()

    def __update_callback(self, msg):
        res = self.entropy_srv()
        delta = res.viewpoint

        vscale = 2

        self.curr_velo.linear.x = delta.x * vscale
        self.curr_velo.linear.y = delta.y * vscale
        self.curr_velo.linear.z = delta.z * vscale

        # Publish pose.
        gp = Pose()
        gp.position.x = res.best_grasp.data[0]
        gp.position.y = res.best_grasp.data[1]
        gp.position.z = res.best_grasp.data[2]
        ang = res.best_grasp.data[3]
        if ang < 0:
            ang += np.pi
        q = tft.quaternion_from_euler(np.pi, 0, -1 * ang)
        gp.orientation.x = q[0]
        gp.orientation.y = q[1]
        gp.orientation.z = q[2]
        gp.orientation.w = q[3]
        publish_pose_as_transform(gp, 'panda_link0', 'G', 0.05)
        self.best_grasp = gp

    def __trigger_update(self):
        self.update_pub.publish(Empty())

    def __velo_control_loop(self):
        ctr = 0
        r = rospy.Rate(self.curr_velocity_publish_rate)
        while not rospy.is_shutdown() and not DIE:
            ctr += 1
            if ctr >= self.curr_velocity_publish_rate/self.update_rate:
                ctr = 0
            self.__trigger_update()

            # End effector Z height
            if self.robot_state.O_T_EE[-2] < 0.2:
                self.stop()
                return True

            # Cartesian Contact
            if any(self.robot_state.cartesian_contact):
                self.stop()
                rospy.logerr('Detected cartesian contact during velocity control loop.')
                return False

            self.curr_velo_pub.publish(self.curr_velo)
            r.sleep()

        return not rospy.is_shutdown()

    def __execute_grasp(self, grasp_pose):
            if raw_input('Looks Good? Y/n') not in ['', 'y', 'Y']:
                return

            self.cs.switch_controller('moveit')

            # Offset for initial pose.
            initial_offset = 0.15
            link_ee_offset = 0.138
            grasp_pose.position.z = max(grasp_pose.position.z - 0.01, 0.01)
            grasp_pose.position.z += initial_offset + link_ee_offset  # Offset from end efector position to

            self.pc.goto_pose(grasp_pose, velocity=0.25)

            # Reset the position
            grasp_pose.position.z -= initial_offset + link_ee_offset
            #self.pc.goto_pose_cartesian(grasp_pose, velocity=0.1)

            self.cs.switch_controller('velocity')
            v = Twist()
            v.linear.z = -0.05
            while self.robot_state.O_T_EE[-2] > grasp_pose.position.z and not any(self.robot_state.cartesian_contact):
                self.curr_velo_pub.publish(v)
                rospy.sleep(0.01)
            v.linear.z = 0
            self.curr_velo_pub.publish(v)

            # close the fingers.
            rospy.sleep(0.5)
            return self.pc.grasp(0)

    def stop(self):
        self.pc.stop()
        self.curr_velo = Twist()
        self.curr_velo_pub.publish(self.curr_velo)

    def go(self):
        while not rospy.is_shutdown() and not DIE:
            self.cs.switch_controller('moveit')
            self.pc.goto_named_pose('grip_ready', velocity=0.5)
            self.pc.set_gripper(0.1)

            self.cs.switch_controller('velocity')
            raw_input('Press Enter to Start.')
            self.entropy_reset_srv.call()
            self.__trigger_update()

            # Perform the velocity control portion.
            success = self.__velo_control_loop()
            if not success:
                continue

            self.__execute_grasp(self.best_grasp)


if __name__ == '__main__':
    rospy.init_node('ap_grasping_velo')
    agc = ActiveGraspController()
    agc.go()
