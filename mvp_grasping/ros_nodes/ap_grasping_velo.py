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
from dougsm_helpers.timeit import TimeIt

from ggcnn.srv import NextViewpoint

def move_to_pose_velo(d_pose):
    p = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', geometry_msgs.msg.Twist, queue_size=1)

    velo = geometry_msgs.msg.Twist()

    scale_linear = 0.7
    scale_angular = 0.5
    threshold = 0.005

    dx = 1
    dy = 1
    dz = 1
    droll = 1
    dpitch = 1
    dyaw = 1

    while abs(droll) > threshold or abs(dpitch) > threshold or abs(dyaw) > threshold or abs(dx) > threshold or abs(dy) > threshold or abs(dz) > threshold:
        curr_pose = current_robot_pose('panda_link0', 'panda_EE')

        # Cartesian
        dx = d_pose[0] - curr_pose.position.x
        dy = d_pose[1] - curr_pose.position.y
        dz = d_pose[2] - curr_pose.position.z

        velo.linear.x = dx * scale_linear
        velo.linear.y = dy * scale_linear
        velo.linear.z = dz * scale_linear

        # Angular
        cpq = curr_pose.orientation
        dq = tft.quaternion_multiply(d_pose[3:], tft.quaternion_conjugate([cpq.x, cpq.y, cpq.z, cpq.w]))
        d_euler = tft.euler_from_quaternion(dq)
        droll = (d_euler[0]) * 1
        dpitch = (d_euler[1]) * 1
        dyaw = (d_euler[2]) * 1

        velo.angular.x = droll * scale_angular
        velo.angular.y = dpitch * scale_angular
        velo.angular.z = dyaw * scale_angular

        p.publish(velo)
        rospy.sleep(0.001)

    velo.linear.x = 0
    velo.linear.y = 0
    velo.linear.z = 0
    velo.angular.x = 0
    velo.angular.y = 0
    velo.angular.z = 0
    p.publish(velo)


def update_callback(msg):
    with TimeIt('Callback'):
        res = next_view_srv()
        delta = res.viewpoint

        vscale = 2

        velo.linear.x = delta.x * vscale
        velo.linear.y = delta.y * vscale
        velo.linear.z = delta.z * vscale

        gp = geometry_msgs.msg.Pose()
        gp.position.x = res.best_grasp.data[0]
        gp.position.y = res.best_grasp.data[1]
        gp.position.z = res.best_grasp.data[2]
        ang = res.best_grasp.data[3]
        if ang < 0:
            ang += np.pi
        q = tft.quaternion_from_euler(np.pi, 0, -1 * ang)
        gp.orientation.x = q[0]
        gp.orientation.y = q[1]
        gp.orientation.z = q[2]
        gp.orientation.w = q[3]
        publish_pose_as_transform(gp, 'panda_link0', 'G', 0.05)

if __name__ == '__main__':
    PREGRASP_POSE = [0.0, -0.45, 0.55, 2**0.5/2, -2**0.5/2, 0, 0]

    rospy.init_node('panda_open_loop_grasp', anonymous=True)

    next_view_srv = rospy.ServiceProxy('/grasp_entropy_node/update_grid', NextViewpoint)

    raw_input('Press Enter to Start.')

    move_to_pose_velo(PREGRASP_POSE)
    exit()

    r = rospy.Rate(100)

    p = rospy.Publisher('/cartesian_velocity_node_controller/cartesian_velocity', geometry_msgs.msg.Twist, queue_size=1)
    s = rospy.Subscriber('~/update', std_msgs.msg.Empty, update_callback, queue_size=1)
    p2 = rospy.Publisher('~/update', std_msgs.msg.Empty, queue_size=1)

    i = 0

    velo = geometry_msgs.msg.Twist()

    while not rospy.is_shutdown():
        i += 1
        curr_pose = current_robot_pose('panda_link0', 'panda_hand')
        if i >= 25:
            i = 0
            p2.publish(std_msgs.msg.Empty())

        if curr_pose.position.z < 0.25:
            velo.linear.x = 0
            velo.linear.y = 0
            velo.linear.z = 0
            p.publish(velo)
            break

        p.publish(velo)

        r.sleep()
