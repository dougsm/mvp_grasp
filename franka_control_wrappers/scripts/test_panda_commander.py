#! /usr/bin/env python

import rospy
from franka_control_wrappers.panda_commander import PandaCommander

rospy.init_node('test_panda_commander')

PandaCommander()

print('Woo!')
