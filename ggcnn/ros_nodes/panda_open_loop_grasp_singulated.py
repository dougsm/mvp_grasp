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
from dougsm_helpers.timeit import TimeIt
from dougsm_helpers.ros_control import ControlSwitcher

from ggcnn.msg import Grasp
from ggcnn.srv import GraspPrediction, GraspPredictionRequest

class Logger:
    def __init__(self, output_desc='run', output_dir='/home/acrv/doug_logs/'):
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
        'object',
        'success',
        'time',
        'quality',
    ]

    def __init__(self, experiment, object=''):
        self.experiment = experiment
        self.viewpoints = None
        self.t0 = 0
        self.t1 = 0
        self.success = False
        self.qualtiy = None
        self.object = object

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
    def __init__(self, objects=[]):
        self.runs = []
        exp_name = raw_input('Name of this experiment? ')
        self.logger = Logger(exp_name)
        self.logger.log_dict(rospy.get_param('/grasp_entropy_node'))
        self.logger.log_list(Run.log_properties + Experiment.log_properties)
        self.successes = 0
        self.objects = objects
        self.object_id = 0
        self.object_count = 0

    def new_run(self):
        # self.object_count += 1
        if self.object_count == 10:
            self.object_count = 0
            self.object_id += 1
            if self.object_count >= len(self.objects):
                rospy.logerr("THAT'S ALL FOLKS")
                exit()
            rospy.logerr("NEW OBJECT: %s" % self.objects[self.object_id])
            raw_input('Enter To Confirm')
        return Run(self, object=self.objects[self.object_id])

    def save_run(self, run):
        if run.success:
            self.successes += 1
        self.object_count += 1
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


class PandaOpenLoopGraspController:
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

        self.pregrasp_pose = [(rospy.get_param('/grasp_entropy_node/histogram/bounds/x2') + rospy.get_param('/grasp_entropy_node/histogram/bounds/x1'))/2 - 0.03,
                              (rospy.get_param('/grasp_entropy_node/histogram/bounds/y2') + rospy.get_param('/grasp_entropy_node/histogram/bounds/y1'))/2 + 0.10,
                              rospy.get_param('/grasp_entropy_node/height/z1') + 0.0,
                              2**0.5/2, -2**0.5/2, 0, 0]

        self.last_weight = 0
        self.__weight_increase_check()

        objects = ['clamp', 'gearbox', 'nozzle', 'part1', 'part3', 'pawn', 'turbine', 'vase']
        # objects = ['Tape', 'Brush', 'Bear', 'Duck', 'Toothbrush', 'Ball', 'Die', 'Screwdriver', 'Clamp', 'Pen', 'Mug', 'Cable']
        self.experiment = Experiment(objects=objects)

    def __recover_robot_from_error(self):
        rospy.logerr('Recovering')
        self.pc.recover()
        rospy.logerr('Done')
        self.ROBOT_ERROR_DETECTED = False

    def __weight_increase_check(self):
        try:
            w = rospy.wait_for_message('/scales/weight', Int16, timeout=2).data
            increased = w < self.last_weight
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

            # if raw_input('Continue?') == '0':
            #     return False

            # Offset for initial pose.
            initial_offset = 0.10
            LINK_EE_OFFSET = 0.138

            # Add some limits, plus a starting offset.
            best_grasp.pose.position.z = max(best_grasp.pose.position.z - 0.01, 0.026)  # 0.021 = collision with ground
            best_grasp.pose.position.z += initial_offset + LINK_EE_OFFSET  # Offset from end efector position to

            print('Width: ', best_grasp.width)
            self.pc.set_gripper(best_grasp.width, wait=False)
            rospy.sleep(0.1)
            self.pc.goto_pose(best_grasp.pose, velocity=0.25)

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
            self.pc.grasp(0, force=5)

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
            self.pc.goto_named_pose('grip_ready', velocity=0.5)
            self.pc.goto_pose(self.pregrasp_pose, velocity=0.5)
            self.pc.set_gripper(0.1)
            rospy.sleep(2.0)

            self.__weight_increase_check()
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
            # self.pc.goto_named_pose('drop_box', velocity=0.5)

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
