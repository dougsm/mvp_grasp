#! /usr/bin/env python

from __future__ import division, print_function

import rospy
import tf.transformations as tft

import os
import time
import datetime

import numpy as np

from std_msgs.msg import Empty, Int16
from geometry_msgs.msg import Twist, Pose
from franka_msgs.msg import FrankaState, Errors as FrankaErrors
from std_srvs.srv import Empty as EmptySrv

from franka_control_wrappers.panda_commander import PandaCommander

import dougsm_helpers.tf_helpers as tfh
from dougsm_helpers.ros_control import ControlSwitcher

from mvp_grasping.srv import NextViewpoint, AddFailurePoint, AddFailurePointRequest


class Logger:
    def __init__(self, output_desc='run', output_dir='~'):
        dt = datetime.datetime.now().strftime('%m%d_%H%M%S')
        self.out_file = os.path.join(output_dir, '%s_%s.txt' % (dt, output_desc))

    def write_line(self, l):
        with open(self.out_file, 'a') as f:
            f.write(l)
            f.write('\n')

    def log_list(self, l):
        o = []
        for i in l:
            if isinstance(i, float):
                o.append('%0.2f' % i)
            else:
                o.append(i)
        self.write_line('\t'.join([str(i) for i in o]))

    def log_params(self, params):
        for p in params:
            self.log_list([
                p,
                rospy.get_param(p)
            ])

    def log_dict(self, d, s=[]):
        if isinstance(d, dict):
            for k in d:
                self.log_dict(d[k], s + [k])
        else:
            self.log_list([
                '/'.join(s),
                d
            ])


class Run:
    log_properties = [
        'success',
        'time',
        'viewpoints',
        'quality',
        'entropy'
    ]

    def __init__(self, experiment):
        self.experiment = experiment
        self.viewpoints = None
        self.t0 = 0
        self.t1 = 0
        self.success = False
        self.qualtiy = None
        self.entropy = None

    def start(self):
        self.t0 = time.time()

    def stop(self):
        self.t1 = time.time()

    @property
    def time(self):
        return self.t1 - self.t0

    @property
    def log_list(self):
        return [getattr(self, p) for p in Run.log_properties]

    def save(self):
        self.experiment.save_run(self)


class Experiment:
    log_properties = [
        'success_rate',
        'mpph'
    ]

    def __init__(self):
        self.runs = []
        exp_name = raw_input('Name of this experiment? ')
        self.logger = Logger(exp_name)
        self.logger.log_dict(rospy.get_param('/grasp_entropy_node'))
        self.logger.log_list(Run.log_properties + Experiment.log_properties)
        self.successes = 0

    def new_run(self):
        return Run(self)

    def save_run(self, run):
        if run.success:
            self.successes += 1
        self.runs.append(run)
        self.log_run(run)

    @property
    def success_rate(self):
        return self.successes/len(self.runs)

    @property
    def mpph(self):
        return (3600 / (sum([r.time for r in self.runs]) / len(self.runs))) * self.success_rate

    @property
    def log_list(self):
        return [getattr(self, p) for p in Experiment.log_properties]

    def log_run(self, run):
        self.logger.log_list(run.log_list + self.log_list)
        print(self.success_rate, self.mpph)


class BaseGraspController(object):
    """
    An base class for performing grasping experiments with the Panda robot.
    """
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
        self.max_velo = 0.1
        self.curr_velo = Twist()
        self.best_grasp = Pose()
        self.viewpoints = 0
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
        self.ROBOT_ERROR_DETECTED = False
        self.BAD_UPDATE = False
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.__robot_state_callback, queue_size=1)

        # Centre and above the bin
        self.pregrasp_pose = [(rospy.get_param('/grasp_entropy_node/histogram/bounds/x2') + rospy.get_param('/grasp_entropy_node/histogram/bounds/x1'))/2,
                              (rospy.get_param('/grasp_entropy_node/histogram/bounds/y2') + rospy.get_param('/grasp_entropy_node/histogram/bounds/y1'))/2 + 0.08,
                              rospy.get_param('/grasp_entropy_node/height/z1') + 0.066,
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

    def __update_callback(self, msg):
        # Update the MVP Controller asynchronously
        if not self._in_velo_loop:
            # Stop the callback lagging behind
            return

        res = self.entropy_srv()
        if not res.success:
            # Something has gone wrong, 0 velocity.
            self.BAD_UPDATE = True
            self.curr_velo = Twist()
            return

        self.viewpoints = res.no_viewpoints

        # Calculate the required angular velocity to match the best grasp.
        q = tfh.quaternion_to_list(res.best_grasp.pose.orientation)
        curr_R = np.array(self.robot_state.O_T_EE).reshape((4, 4)).T
        cpq = tft.quaternion_from_matrix(curr_R)
        dq = tft.quaternion_multiply(q, tft.quaternion_conjugate(cpq))
        d_euler = tft.euler_from_quaternion(dq)
        res.velocity_cmd.angular.z = d_euler[2]

        self.best_grasp = res.best_grasp
        self.curr_velo = res.velocity_cmd

        tfh.publish_pose_as_transform(self.best_grasp.pose, 'panda_link0', 'G', 0.05)

    def __trigger_update(self):
        # Let ROS handle the threading for me.
        self.update_pub.publish(Empty())

    def __velo_control_loop(self):
        raise NotImplementedError()

    def __execute_best_grasp(self):
            self.cs.switch_controller('moveit')

            # Offset for initial pose.
            initial_offset = 0.05
            LINK_EE_OFFSET = 0.138

            # Add some limits, plus a starting offset.
            self.best_grasp.pose.position.z = max(self.best_grasp.pose.position.z - 0.01, 0.026)  # 0.021 = collision with ground
            self.best_grasp.pose.position.z += initial_offset + LINK_EE_OFFSET  # Offset from end efector position to

            self.pc.set_gripper(self.best_grasp.width, wait=False)
            rospy.sleep(0.1)
            self.pc.goto_pose(self.best_grasp.pose, velocity=0.5)

            # Reset the position
            self.best_grasp.pose.position.z -= initial_offset + LINK_EE_OFFSET

            self.cs.switch_controller('velocity')
            v = Twist()
            v.linear.z = -0.05

            # Monitor robot state and descend
            while self.robot_state.O_T_EE[-2] > self.best_grasp.pose.position.z and not any(self.robot_state.cartesian_contact) and not self.ROBOT_ERROR_DETECTED:
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
            start_pose = list(self.pregrasp_pose)
            start_pose[0] += np.random.randn() * 0.05
            start_pose[1] += np.random.randn() * 0.05
            self.pc.goto_pose(start_pose, velocity=0.25)
            self.pc.set_gripper(0.1)

            self.cs.switch_controller('velocity')

            self.entropy_reset_srv.call()
            self.__trigger_update()

            run = self.experiment.new_run()
            run.start()

            # Perform the velocity control portion.
            self._in_velo_loop = True
            velo_ok = self.__velo_control_loop()
            self._in_velo_loop = False
            if not velo_ok:
                rospy.sleep(1.0)
                if self.BAD_UPDATE:
                    raw_input('Fix Me! Enter to Continue')
                    self.BAD_UPDATE = False
                if self.ROBOT_ERROR_DETECTED:
                    self.__recover_robot_from_error()
                rospy.logerr('Aborting this Run.')
                continue

            # Execute the Grasp
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
            self.pc.set_gripper(0.1)

            # Check success using the scales.
            rospy.sleep(1.0)
            grasp_success = self.__weight_increase_check()
            if not grasp_success:
                rospy.logerr("Failed Grasp")
                m = AddFailurePointRequest()
                m.point.x = self.best_grasp.pose.position.x
                m.point.y = self.best_grasp.pose.position.y
                self.add_failure_point_srv.call(m)
            else:
                rospy.logerr("Successful Grasp")

            run.success = grasp_success
            run.quality = self.best_grasp.quality
            run.entropy = self.best_grasp.entropy
            run.viewpoints = self.viewpoints
            run.save()

