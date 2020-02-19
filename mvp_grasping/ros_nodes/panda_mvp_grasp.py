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

    def trigger_update(self):
        super(MVPGraspController, self).trigger_update()

    def __velo_control_loop(self):
        """
        Perform velocity control using the MVP controller.
        """
        ctr = 0
        r = rospy.Rate(self.curr_velocity_publish_rate)
        while not rospy.is_shutdown():
            if self.ROBOT_ERROR_DETECTED or self.BAD_UPDATE:
                return False

            # End effector Z height
            if self.robot_state.O_T_EE[-2] < 0.175:
                # self.stop()
                rospy.sleep(0.1)
                return True

            ctr += 1
            if ctr >= self.curr_velocity_publish_rate/self.update_rate:
                ctr = 0
                self.trigger_update()

            # Cartesian Contact
            if any(self.robot_state.cartesian_contact):
                self.stop()
                rospy.logerr('Detected cartesian contact during velocity control loop.')
                return False

            v = Twist()
            v.linear.x = self.curr_velo.linear.x * self.max_velo
            v.linear.y = self.curr_velo.linear.y * self.max_velo
            v.linear.z = self.curr_velo.linear.z * self.max_velo
            v.angular = self.curr_velo.angular

            self.curr_velo_pub.publish(v)
            r.sleep()

        return not rospy.is_shutdown()


    def setup_velocity_control_loop(self):
        super(MVPGraspController, self).setup_velocity_control_loop()

    def execute_velocity_control_loop(self):
        """Perform the velocity control portion."""
        self._in_velo_loop = True
        velo_ok = self.__velo_control_loop()
        self._in_velo_loop = False
        return velo_ok


    def execute_best_grasp(self):
        super(MVPGraspController, self).execute_best_grasp()


    def release_object(self):
        super(MVPGraspController, self).release_object()


    def check_success_using_scales(self, run):
        super(MVPGraspController, self).check_success_using_scales(run)

    def go(self):
        super(MVPGraspController, self).go()


if __name__ == '__main__':
    rospy.init_node('ap_grasping_velo')
    agc = MVPGraspController()
    agc.go()
