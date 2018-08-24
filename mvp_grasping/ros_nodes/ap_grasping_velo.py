#! /usr/bin/env python

from __future__ import division, print_function

import rospy
import tf.transformations as tft

import time

import numpy as np

from std_msgs.msg import Empty, Int16
from geometry_msgs.msg import Twist, Pose
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from std_srvs.srv import Empty as EmptySrv

from franka_control_wrappers.panda_commander import PandaCommander

from dougsm_helpers.tf_helpers import current_robot_pose, publish_tf_quaterion_as_transform, convert_pose, publish_pose_as_transform
from dougsm_helpers.timeit import TimeIt
from dougsm_helpers.ros_control import ControlSwitcher

from mvp_grasping.srv import NextViewpoint, NextViewpointRequest, AddFailurePoint, AddFailurePointRequest


class ActiveGraspController:
    def __init__(self):
        entropy_node_name = '/grasp_entropy_node'
        rospy.wait_for_service(entropy_node_name + '/update_grid')
        self.entropy_srv = rospy.ServiceProxy(entropy_node_name + '/update_grid', NextViewpoint)
        rospy.wait_for_service(entropy_node_name + '/reset_grid')
        self.entropy_reset_srv = rospy.ServiceProxy(entropy_node_name + '/reset_grid', EmptySrv)
        rospy.wait_for_service(entropy_node_name + '/add_failure_point')
        self.add_failure_point_srv = rospy.ServiceProxy(entropy_node_name + '/add_failure_point', AddFailurePoint)

        self.curr_velocity_publish_rate = 100.0  # Hz
        self.curr_velo_pub = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', Twist, queue_size=1)
        self.curr_velo = Twist()
        self.best_grasp = Pose()
        # self.grasp_normal = np.array([0, 0, 1])
        self.grasp_width = 0.10
        self._in_velo_loop = False

        self.update_rate = 10.0  # Hz
        update_topic_name = '~/update'
        self.update_pub = rospy.Publisher(update_topic_name, Empty, queue_size=1)
        rospy.Subscriber(update_topic_name, Empty, self.__update_callback, queue_size=1)

        self.cs = ControlSwitcher({'moveit': 'position_joint_trajectory_controller',
                                   'velocity': 'cartesian_velocity_node_controller'})
        self.cs.switch_controller('moveit')
        self.pc = PandaCommander(group_name='panda_arm_hand')

        self.robot_state = None
        self.ERROR_DETECTED = False
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

        self.pregrasp_pose = [(rospy.get_param('/grasp_entropy_node/histogram/bounds/x2') + rospy.get_param('/grasp_entropy_node/histogram/bounds/x1'))/2,
                              (rospy.get_param('/grasp_entropy_node/histogram/bounds/y2') + rospy.get_param('/grasp_entropy_node/histogram/bounds/y1'))/2 + 0.08,
                              0.60,
                              2**0.5/2, -2**0.5/2, 0, 0]

        self.last_weight = 0
        self.__weight_increase_check()

    def __weight_increase_check(self):
        try:
            w = rospy.wait_for_message('/scales/weight', Int16, timeout=2).data
            increased = (w - 5) > self.last_weight
            self.last_weight = w
            return increased
        except:
            return raw_input('No weight. Success? [1/0]') == '1'

    def __robot_state_callback(self, msg):
        self.robot_state = msg
        if any(self.robot_state.cartesian_collision):
            if not self.ERROR_DETECTED:
                rospy.logerr('Detected Cartesian Collision')
            self.ERROR_DETECTED = True
        for s in FrankaErrors.__slots__:
            if getattr(msg.current_errors, s):
                self.stop()
                if not self.ERROR_DETECTED:
                    rospy.logerr('Robot Error Detected')
                self.ERROR_DETECTED = True

    def __update_callback(self, msg):
        if not self._in_velo_loop:
            # Stop the callback lagging behind
            return
        res = self.entropy_srv()
        delta = res.viewpoint

        vscale = 3

        self.curr_velo.linear.x = delta.x * vscale
        self.curr_velo.linear.y = delta.y * vscale
        self.curr_velo.linear.z = delta.z * vscale
        # self.curr_velo.linear.z = -1 * max(0.01, 0.075 - (delta.x**2 + delta.y**2)**0.5)

        if len(res.best_grasp.data) > 0:
            # max_angle = 10 / 180 * np.pi
            # if res.best_grasp.data[7] < np.cos(max_angle):
            #     s = np.array([res.best_grasp.data[5], res.best_grasp.data[6], np.cos(max_angle)])
            #     s[0:2] = s[0:2]/(np.sqrt((s[0]**2 + s[1]**2)/(1-s[2]**2)))
            #     print(np.linalg.norm(s))
            #     # xyzw = [0, 0, 0, 1]
            # else:
            #     self.grasp_normal = np.array(res.best_grasp.data[5:])
            # a = np.cross([0.0, 0.0, 1.0], self.grasp_normal)
            # w = np.dot([0.0, 0.0, 1.0], self.grasp_normal)
            # xyzw = np.array([a[0], a[1], a[2], w])
            # xyzw /= np.linalg.norm(xyzw)

            # Publish pose.
            gp = Pose()
            gp.position.x = res.best_grasp.data[0]
            gp.position.y = res.best_grasp.data[1]
            gp.position.z = res.best_grasp.data[2]
            ang = res.best_grasp.data[3]
            q = tft.quaternion_from_euler(np.pi, 0, ang - np.pi/2)

            # q = tft.quaternion_multiply(xyzw, q)

            gp.orientation.x = q[0]
            gp.orientation.y = q[1]
            gp.orientation.z = q[2]
            gp.orientation.w = q[3]
            self.best_grasp = gp
            self.grasp_width = res.best_grasp.data[4]

            curr_R = np.array(self.robot_state.O_T_EE).reshape((4, 4)).T
            cpq = tft.quaternion_from_matrix(curr_R)
            dq = tft.quaternion_multiply(q, tft.quaternion_conjugate(cpq))
            d_euler = tft.euler_from_quaternion(dq)
            self.curr_velo.angular.z = (d_euler[2])

            publish_pose_as_transform(gp, 'panda_link0', 'G', 0.05)
        else:
            self.curr_velo.angular.z = 0

    def __trigger_update(self):
        self.update_pub.publish(Empty())

    def __velo_control_loop(self):
        ctr = 0
        r = rospy.Rate(self.curr_velocity_publish_rate)
        while not rospy.is_shutdown():
            if self.ERROR_DETECTED:
                return False

            # End effector Z height
            if self.robot_state.O_T_EE[-2] < 0.175: # - self.best_grasp.position.z < 0.15:
                # self.stop()
                rospy.sleep(0.1)
                return True

            ctr += 1
            if ctr >= self.curr_velocity_publish_rate/self.update_rate:
                ctr = 0
            self.__trigger_update()

            # Cartesian Contact
            if any(self.robot_state.cartesian_contact):
                self.stop()
                rospy.logerr('Detected cartesian contact during velocity control loop.')
                return False

            self.curr_velo_pub.publish(self.curr_velo)
            r.sleep()

        return not rospy.is_shutdown()

    def __execute_grasp(self, grasp_pose):
            # if raw_input('Looks Good? Y/n') not in ['', 'y', 'Y']:
            #     return
            if self.grasp_width == 0.0:
                # Something is wrong.
                return False

            self.cs.switch_controller('moveit')

            # Offset for initial pose.
            initial_offset = 0.1
            link_ee_offset = 0.138

            # o = grasp_pose.orientation
            # n = np.dot(tft.quaternion_matrix([o.x, o.y, o.z, o.w]), np.array([[0, 0, initial_offset + link_ee_offset, 0]]).T).flatten()
            # angled_offset = n * -1

            grasp_pose.position.z = max(grasp_pose.position.z - 0.01, 0.026)  # 0.021 = collision with ground
            grasp_pose.position.z += initial_offset + link_ee_offset  # Offset from end efector position to
            # grasp_pose.position.x += angled_offset[0]
            # grasp_pose.position.y += angled_offset[1]
            # grasp_pose.position.z += angled_offset[2]

            self.pc.set_gripper(self.grasp_width, wait=False)
            rospy.sleep(0.1)
            self.pc.goto_pose(grasp_pose, velocity=0.25)

            # Reset the position
            grasp_pose.position.z -= initial_offset + link_ee_offset
            # grasp_pose.position.x -= angled_offset[0]
            # grasp_pose.position.y -= angled_offset[1]
            # grasp_pose.position.z -= angled_offset[2]
            #self.pc.goto_pose_cartesian(grasp_pose, velocity=0.1)

            self.cs.switch_controller('velocity')
            v = Twist()
            # angled_velo = -0.05 * angled_offset
            v.linear.z = -0.05
            # v.linear.x = angled_velo[0]
            # v.linear.y = angled_velo[1]
            # v.linear.z = angled_velo[2]
            while self.robot_state.O_T_EE[-2] > grasp_pose.position.z and not any(self.robot_state.cartesian_contact) and not self.ERROR_DETECTED:
                self.curr_velo_pub.publish(v)
                rospy.sleep(0.01)
            v.linear.z = 0
            self.curr_velo_pub.publish(v)

            # Check for collisions
            if self.ERROR_DETECTED:
                self.pc.recover()
                self.ERROR_DETECTED = False
                return False

            # close the fingers.
            rospy.sleep(0.2)
            self.pc.grasp(0, force=2)

            if self.ERROR_DETECTED:
                self.pc.recover()
                self.ERROR_DETECTED = False

            return True

    def stop(self):
        self.pc.stop()
        self.curr_velo = Twist()
        self.curr_velo_pub.publish(self.curr_velo)

    def go(self):
        run_name = raw_input('Name of this run? ')
        with open('/home/acrv/doug_logs/%s.txt' % run_name, 'w') as f:
            f.write('Grasp\tSuccess\tTime\n')
        i = 1

        raw_input('Press Enter to Start.')
        while not rospy.is_shutdown():
            self.cs.switch_controller('moveit')
            self.pc.goto_named_pose('grip_ready', velocity=0.25)
            start_pose = list(self.pregrasp_pose)
            start_pose[0] += np.random.randn() * 0.05
            start_pose[1] += np.random.randn() * 0.05
            self.pc.goto_pose(start_pose, velocity=0.25)
            self.pc.set_gripper(0.1)

            self.cs.switch_controller('velocity')

            self.entropy_reset_srv.call()
            self.__trigger_update()

            t0 = time.time()
            # Perform the velocity control portion.
            self._in_velo_loop = True
            success = self.__velo_control_loop()
            self._in_velo_loop = False
            if not success:
                rospy.sleep(1.0)
                if self.ERROR_DETECTED:
                    rospy.logerr('Recovering')
                    self.pc.recover()
                    rospy.logerr('Done')
                    self.ERROR_DETECTED = False
                continue

            grasp_ret = gs = self.__execute_grasp(self.best_grasp)
            t1 = time.time()
            if gs:
                self.cs.switch_controller('moveit')
                self.pc.goto_named_pose('grip_ready', velocity=0.5)
                self.pc.goto_named_pose('drop_box', velocity=0.5)
                self.pc.set_gripper(0.07)

            if grasp_ret:
                # success = raw_input('Success? ')
                rospy.sleep(1.0)
                success = self.__weight_increase_check()
                if not success:
                    rospy.logerr("Failed Grasp")
                    m = AddFailurePointRequest()
                    m.point.x = self.best_grasp.position.x
                    m.point.y = self.best_grasp.position.y
                    self.add_failure_point_srv.call(m)
                else:
                    rospy.logerr("Successful Grasp")
                with open('/home/acrv/doug_logs/%s.txt' % run_name, 'a') as f:
                    f.write('%d\t%s\t%f\n' % (i, success, t1-t0))
                i += 1


if __name__ == '__main__':
    rospy.init_node('ap_grasping_velo')
    agc = ActiveGraspController()
    agc.go()
