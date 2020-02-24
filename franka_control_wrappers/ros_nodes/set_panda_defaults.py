#! /usr/bin/env python

import rospy
from franka_control.srv import SetForceTorqueCollisionBehavior, SetForceTorqueCollisionBehaviorRequest
from franka_control.srv import SetEEFrame, SetEEFrameRequest

rospy.init_node('set_panda_defaults')

# Set some sensible defaults
rospy.wait_for_service('/franka_control/set_force_torque_collision_behavior')
ftcb_srv = rospy.ServiceProxy('/franka_control/set_force_torque_collision_behavior', SetForceTorqueCollisionBehavior)
ftcb_msg = SetForceTorqueCollisionBehaviorRequest()
ftcb_msg.lower_torque_thresholds_nominal = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
ftcb_msg.upper_torque_thresholds_nominal = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
ftcb_msg.lower_force_thresholds_nominal = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]  # These will be used by the velocity controller to stop movement
ftcb_msg.upper_force_thresholds_nominal = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

res = ftcb_srv.call(ftcb_msg).success
if not res:
    rospy.logerr('Failed to set Force/Torque Collision Behaviour Thresholds')
else:
    rospy.loginfo('Successfully set Force/Torque Collision Behaviour Thresholds')


rospy.wait_for_service('/franka_control/set_EE_frame')
eef_srv = rospy.ServiceProxy('/franka_control/set_EE_frame', SetEEFrame)
eef_msg = SetEEFrameRequest()

# franka_msgs.msg.FrankaState keeps track of the updated F_T_EE, so the default transform has to be hardcoded
default_F_T_EE = [0.7071, -0.7071, 0.0, 0.0, 0.7071, 0.7071, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1034, 1.0]

# Add the configured gripper height to the z-axis in the translation vector part of the transform
gripper_offset = rospy.get_param("/panda_setup/camera_mount_extra_height", 0.035)

current_F_T_EE = default_F_T_EE
current_F_T_EE[14] += gripper_offset
eef_msg.F_T_EE = current_F_T_EE

res = eef_srv.call(eef_msg).success
if not res:
    rospy.logerr('Failed to set EE Frame')
else:
    rospy.loginfo('Successfully set EE Frame')
