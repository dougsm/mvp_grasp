#! /usr/bin/env python

import rospy
import tf.transformations as tft

import numpy as np

# import kinova_msgs.msg
# import kinova_msgs.srv
import std_msgs.msg
import std_srvs.srv
import geometry_msgs.msg

from dougsm_helpers.tf_helpers import current_robot_pose, publish_tf_quaterion_as_transform, convert_pose, publish_pose_as_transform

from franka_control_wrappers.panda_commander import PandaCommander
from ggcnn.srv import NextViewpoint

MOVING = False  # Flag whether the robot is moving under velocity control.
CURR_Z = 0  # Current end-effector z height.

def execute_grasp(pc, gp_base):
    # Execute a grasp.
    global MOVING
    global CURR_Z
    global start_force_srv
    global stop_force_srv

    if raw_input('Looks Good? Y/n') not in ['', 'y', 'Y']:
        return

    # Offset for initial pose.
    initial_offset = 0.15
    gp_base.position.z = max(gp_base.position.z, 0.01)
    gp_base.position.z += initial_offset + 0.125

    pc.goto_pose(gp_base, velocity=0.25)

    # Disable force control, makes the robot more accurate.
    # stop_force_srv.call(kinova_msgs.srv.StopRequest())

    # move_to_pose(gp_base)
    # rospy.sleep(0.1)

    # Start force control, helps prevent bad collisions.
    # start_force_srv.call(kinova_msgs.srv.StartRequest())

    # rospy.sleep(0.25)

    # Reset the position
    gp_base.position.z -= initial_offset
    pc.goto_pose_cartesian(gp_base, velocity=0.1)

    # Flag to check for collisions.
    # MOVING = True

    # Generate a nonlinearity for the controller.
    # cart_cov = generate_cartesian_covariance(0)

    # Move straight down under velocity control.
    # velo_pub = rospy.Publisher('/m1n6s200_driver/in/cartesian_velocity', kinova_msgs.msg.PoseVelocity, queue_size=1)
    # while MOVING and CURR_Z - 0.02 > gp_base.position.z:
    #     dz = gp_base.position.z - CURR_Z - 0.03   # Offset by a few cm for the fingertips.
    #     MAX_VELO_Z = 0.08
    #
    #     dz = max(min(dz, MAX_VELO_Z), -1.0*MAX_VELO_Z)
    #
    #     v = np.array([0, 0, dz])
    #     vc = list(np.dot(v, cart_cov)) + [0, 0, 0]
    #     velo_pub.publish(kinova_msgs.msg.PoseVelocity(*vc))
    #     rospy.sleep(1/100.0)

    # MOVING = False

    # close the fingers.
    rospy.sleep(1.0)
    pc.grasp(0)

    # Move back up to initial position.
    # gp_base.position.z += initial_offset
    # gp_base.orientation.x = 1
    # gp_base.orientation.y = 0
    # gp_base.orientation.z = 0
    # gp_base.orientation.w = 0
    # move_to_pose(gp_base)

    # stop_force_srv.call(kinova_msgs.srv.StopRequest())

    return

PREGRASP_POSE = [0.0, -0.45, 0.55, 2**0.5/2, -2**0.5/2, 0, 0]

if __name__ == '__main__':
    rospy.init_node('panda_open_loop_grasp', anonymous=True)

    next_view_srv = rospy.ServiceProxy('/panda_active_grasp/get_next_viewpoint',  NextViewpoint)

    pc = PandaCommander(group_name='panda_arm_hand')
    pc.set_gripper(0.1)

    # Home position.
    # pc.goto_pose(PREGRASP_POSE)
    pc.goto_named_pose('grip_ready', velocity=0.5)

    raw_input('Press Enter to Start.')

    while not rospy.is_shutdown():

        res = next_view_srv()
        delta = res.viewpoint

        print(delta)

        curr_pose = current_robot_pose('panda_link0', 'panda_hand')
        curr_pose.position.x += delta.x
        curr_pose.position.y += delta.y
        curr_pose.position.z += delta.z

        gp = geometry_msgs.msg.Pose()
        gp.position.x = res.best_grasp.data[0]
        gp.position.y = res.best_grasp.data[1]
        gp.position.z = res.best_grasp.data[2]
        q = tft.quaternion_from_euler(np.pi, 0, -1 * res.best_grasp.data[3] + np.pi)
        gp.orientation.x = q[0]
        gp.orientation.y = q[1]
        gp.orientation.z = q[2]
        gp.orientation.w = q[3]

        publish_pose_as_transform(gp, 'panda_link0', 'G', 0.5)

        if curr_pose.position.z < 0.35:
            execute_grasp(pc, gp)
            pc.goto_named_pose('grip_ready', velocity=0.2)
            rospy.sleep(1.0)
            pc.set_gripper(0.1)
            break

        else:
            p = curr_pose.position
            pc.goto_pose([p.x, p.y, p.z, 2**0.5/2, -2**0.5/2, 0, 0], velocity=0.5)
