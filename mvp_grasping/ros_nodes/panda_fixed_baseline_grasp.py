#! /usr/bin/env python

from __future__ import division, print_function

import rospy

import numpy as np

from geometry_msgs.msg import Twist

from mvp_grasping.panda_base_grasping_controller import BaseGraspController


class FixedBaselineGraspController(BaseGraspController):
    """
    Perform a fixed, spiral data collection trajectory before grasping.
    All of the common functionality is implemented in BaseGraspController
    """
    def __init__(self):
        super(FixedBaselineGraspController, self).__init__()
        self.update_rate = 10.0 * 50 / 65  # Hz  (perform 50 updates during the trajectory)

    def __velo_control_loop(self):
        """
        Perform velocity control along a spiral trajectory
        """
        ctr = 0
        r = rospy.Rate(self.curr_velocity_publish_rate)
        while not rospy.is_shutdown():
            if self.ROBOT_ERROR_DETECTED or self.BAD_UPDATE:
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

            v = Twist()

            path_amt = (self.robot_state.O_T_EE[-2] - 0.175)/(0.465-0.175)

            # Follow a spiral
            vx = np.cos(2*np.pi*path_amt)
            vy = np.sin(2*np.pi*path_amt) * -1
            vz = -0.5
            absv = (vx**2 + vy**2 + vz**2)**0.5
            vx /= absv
            vy /= absv
            vz /= absv

            v.linear.x = vx * self.max_velo
            v.linear.y = vy * self.max_velo
            v.linear.z = vz * self.max_velo

            self.curr_velo_pub.publish(v)
            r.sleep()

        return not rospy.is_shutdown()


if __name__ == '__main__':
    rospy.init_node('ap_grasping_velo')
    agc = FixedBaselineGraspController()
    agc.go()
