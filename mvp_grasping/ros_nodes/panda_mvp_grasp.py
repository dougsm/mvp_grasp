#! /usr/bin/env python

from __future__ import division, print_function

import rospy
from geometry_msgs.msg import Twist

from mvp_grasping.panda_base_grasping_controller import BaseGraspController


class MVPGraspController(BaseGraspController):
    """
    Perform grasping using the Multi-View Controller.
    All of the common functionality is implemented in BaseGraspController
    """
    def __init__(self):
        super(MVPGraspController, self).__init__()
        self.update_rate = 10.0  # Hz



if __name__ == '__main__':
    rospy.init_node('ap_grasping_velo')
    agc = MVPGraspController()
    agc.go()
