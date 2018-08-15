#! /usr/bin/env python

import rospy
import tf.transformations as tft

import numpy as np

# import kinova_msgs.msg
# import kinova_msgs.srv
import std_msgs.msg
import std_srvs.srv
import geometry_msgs.msg

# from helpers.gripper_action_client import set_finger_positions
# from helpers.position_action_client import position_client, move_to_position
from helpers.transforms import current_robot_pose, publish_tf_quaterion_as_transform, convert_pose, publish_pose_as_transform
# from helpers.covariance import generate_cartesian_covariance

from helpers.panda_commander import PandaCommander

MOVING = False  # Flag whether the robot is moving under velocity control.
CURR_Z = 0  # Current end-effector z height.


def execute_grasp(pc):
    # Execute a grasp.
    global MOVING
    global CURR_Z
    global start_force_srv
    global stop_force_srv

    # Get the positions.
    msg = rospy.wait_for_message('/ggcnn/out/command', std_msgs.msg.Float32MultiArray)
    d = list(msg.data)

    # Calculate the gripper width.
    grip_width = d[4]
    # Convert width in pixels to m.
    # 0.07 is distance from end effector (CURR_Z) to camera.
    # 0.1 is approx degrees per pixel for the realsense.
    g_width = 2 * ((CURR_Z + 0.07)) * np.tan(0.1 * grip_width / 2.0 / 180.0 * np.pi)
    # Convert into motor positions.
    #g = min((1 - (min(g_width, 70)/70)) * (6800-4000) + 4000, 5500)


    # pc.set_gripper(g_width + 0.01)


    # Construct the Pose in the frame of the camera.
    gp = geometry_msgs.msg.Pose()
    gp.position.x = d[0]
    gp.position.y = d[1]
    gp.position.z = d[2]
    q = tft.quaternion_from_euler(0, 0, -1 * d[3] - np.pi/2)
    gp.orientation.x = q[0]
    gp.orientation.y = q[1]
    gp.orientation.z = q[2]
    gp.orientation.w = q[3]

    print(gp)
    gp_base = convert_pose(gp, 'camera_depth_optical_frame', 'panda_link0')
    gpbo = gp_base.orientation
    e = tft.euler_from_quaternion([gpbo.x, gpbo.y, gpbo.z, gpbo.w])

    q = tft.quaternion_from_euler(np.pi, 0, e[2])
    gp_base.orientation.x = q[0]
    gp_base.orientation.y = q[1]
    gp_base.orientation.z = q[2]
    gp_base.orientation.w = q[3]

    publish_pose_as_transform(gp_base, 'panda_link0', 'G', 0.5)

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

    # Robot Monitors.
    # wrench_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_wrench', geometry_msgs.msg.WrenchStamped, robot_wrench_callback, queue_size=1)
    # position_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_pose', geometry_msgs.msg.PoseStamped, robot_position_callback, queue_size=1)

    # https://github.com/dougsm/rosbag_recording_services
    # start_record_srv = rospy.ServiceProxy('/data_recording/start_recording', std_srvs.srv.Trigger)
    # stop_record_srv = rospy.ServiceProxy('/data_recording/stop_recording', std_srvs.srv.Trigger)

    # Enable/disable force control.
    # start_force_srv = rospy.ServiceProxy('/m1n6s200_driver/in/start_force_control', kinova_msgs.srv.Start)
    # stop_force_srv = rospy.ServiceProxy('/m1n6s200_driver/in/stop_force_control', kinova_msgs.srv.Stop)

    pc = PandaCommander(group_name='panda_arm_hand')
    pc.set_gripper(0.1)
    pc.home_gripper()

    # Home position.
    # pc.goto_pose(PREGRASP_POSE)
    pc.goto_named_pose('grip_ready', velocity=0.5)


    while not rospy.is_shutdown():

        pc.set_gripper(0.1)

        raw_input('Press Enter to Start.')

        # start_record_srv(std_srvs.srv.TriggerRequest())
        rospy.sleep(3.0)
        execute_grasp(pc)
        pc.goto_named_pose('grip_ready', velocity=0.25)
        rospy.sleep(3.0)
        # stop_record_srv(std_srvs.srv.TriggerRequest())
