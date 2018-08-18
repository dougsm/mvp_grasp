#! /usr/bin/env python

import rospy
from franka_control.srv import SetForceTorqueCollisionBehavior, SetForceTorqueCollisionBehaviorRequest
from franka_control.srv import SetEEFrame, SetEEFrameRequest

rospy.init_node('set_panda_defaults')

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
# This is the default settings + 35mm down for the gripper
eef_msg.F_T_EE = [0.707099974155426, -0.707099974155426, 0.0, 0.0, 0.707099974155426, 0.707099974155426, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.035 + 0.10339999943971634, 1.0]

res = eef_srv.call(eef_msg).success
if not res:
    rospy.logerr('Failed to set EE Frame')
else:
    rospy.loginfo('Successfully set EE Frame')
