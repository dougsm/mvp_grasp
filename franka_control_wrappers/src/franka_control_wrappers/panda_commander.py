"""
PandaCommander is a class which wraps some basic moveit functions
for the Panda Robot.
"""

import rospy
import actionlib

from math import pi

import moveit_commander
from moveit_commander.conversions import pose_to_list, list_to_pose

import geometry_msgs.msg
import franka_gripper.msg
from franka_control.msg import ErrorRecoveryActionGoal
from std_msgs.msg import Empty


class PandaCommander(object):
    def __init__(self, group_name=None):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.groups = {}
        self.active_group = None
        self.set_group(group_name)

        self.reset_publisher = rospy.Publisher('/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=1)

    def print_debug_info(self):
        if self.active_group:
            planning_frame = self.active_group.get_planning_frame()
            print("============ Reference frame: %s" % planning_frame)
            eef_link = self.active_group.get_end_effector_link()
            print("============ End effector: %s" % eef_link)
        else:
            print("============ No active planning group.")
        group_names = self.robot.get_group_names()
        print("============ Robot Groups:", self.robot.get_group_names())
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")

    def set_group(self, group_name):
        self.active_group = group_name
        if group_name is None:
            self.active_group = None
            return
        else:
            if group_name not in self.groups:
                if group_name not in self.robot.get_group_names():
                    raise ValueError('Group name %s is not valid. Options are %s' % (group_name, self.robot.get_group_names()))
                self.groups[group_name] = moveit_commander.MoveGroupCommander(group_name)
            self.active_group = self.groups[group_name]

    def goto_joints(self, joint_values, group_name=None, wait=True):
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        joint_goal = self.active_group.get_current_joint_values()
        if len(joint_goal) != len(joint_values):
            raise IndexError('Expected %d Joint Values, got %d' % (len(joint_goal), len(joint_values)))
        for i, v in enumerate(joint_values):
            joint_goal[i] = v

        success = self.active_group.go(joint_goal, wait)
        self.active_group.stop()
        return success

    def goto_pose(self, pose, velocity=1.0, group_name=None, wait=True):
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        if type(pose) is list:
            pose = list_to_pose(pose)
        self.active_group.set_max_velocity_scaling_factor(velocity)
        self.active_group.set_pose_target(pose)
        success = self.active_group.go(wait=wait)
        self.active_group.stop()
        self.active_group.clear_pose_targets()
        return success

    def goto_pose_cartesian(self, pose, velocity=1.0, group_name=None, wait=True):
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        if type(pose) is list:
            pose = list_to_pose(pose)

        self.active_group.set_max_velocity_scaling_factor(velocity)
        (plan, fraction) = self.active_group.compute_cartesian_path(
                                           [pose],   # waypoints to follow
                                           0.005,        # eef_step
                                           0.0)         # jump_threshold
        if fraction != 1.0:
            raise ValueError('Unable to plan entire path!')
            return False

        success = self.active_group.execute(plan, wait=wait)
        self.active_group.stop()
        self.active_group.clear_pose_targets()
        return success


    def goto_named_pose(self, pose_name, velocity=1.0, group_name=None, wait=True):
        if group_name:
            self.set_group(group_name)
        if not self.active_group:
            raise ValueError('No active Planning Group')

        self.active_group.set_max_velocity_scaling_factor(velocity)
        self.active_group.set_named_target(pose_name)
        success = self.active_group.go(wait=wait)
        self.active_group.stop()
        return success

    def home_gripper(self):
        client = actionlib.SimpleActionClient('franka_gripper/homing', franka_gripper.msg.HomingAction)
        client.wait_for_server()
        client.send_goal(franka_gripper.msg.HomingGoal())
        return client.wait_for_result()

    def set_gripper(self, width, speed=0.1, wait=True):
        client = actionlib.SimpleActionClient('franka_gripper/move', franka_gripper.msg.MoveAction)
        client.wait_for_server()
        client.send_goal(franka_gripper.msg.MoveGoal(width, speed))
        if wait:
            return client.wait_for_result()
        else:
            return True

    def grasp(self, width=0, e_inner=0.1, e_outer=0.1, speed=0.1, force=15):
        client = actionlib.SimpleActionClient('franka_gripper/grasp', franka_gripper.msg.GraspAction)
        client.wait_for_server()
        client.send_goal(
            franka_gripper.msg.GraspGoal(
                width,
                franka_gripper.msg.GraspEpsilon(e_inner, e_outer),
                speed,
                force
            )
        )
        return client.wait_for_result()

    def stop(self):
        if self.active_group:
            self.active_group.stop()

    def recover(self):
        self.reset_publisher.publish(ErrorRecoveryActionGoal())
        rospy.sleep(3.0)

if __name__ == '__main__':
    rospy.init_node('panda_commander_test', anonymous=True)
    pc = PandaCommander(group_name='panda_arm_hand')

    pc.goto_pose_cartesian(1)
    exit()

    pc.home_gripper()
    pc.grasp()
    rospy.sleep(2)
    pc.set_gripper(0.15, 0.02)

    # pc.print_debug_info()
    #
    # pc.goto_joints([0, -pi/4, 0, -pi/2, 0, pi/3, 0], 'panda_arm_hand')
    #
    # pose_goal = geometry_msgs.msg.Pose()
    # pose_goal.orientation.x = 1.0
    # pose_goal.position.x = 0.4
    # pose_goal.position.y = 0.1
    # pose_goal.position.z = 0.4
    # pc.goto_pose(pose_goal)
    #
    pc.goto_pose([0.3, -0.1, 0.3, 1.0, 0, 0, 0])

    pc.goto_named_pose('ready')
