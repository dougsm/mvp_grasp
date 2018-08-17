#! /usr/bin/env python

import time
from os import path
import rospy

import numpy as np
import tensorflow as tf
from keras.models import load_model

import cv2
import scipy.ndimage as ndimage
from skimage.draw import circle
from skimage.feature import peak_local_max

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray

from ggcnn.aegrasp import predict

bridge = CvBridge()

# Load the Network.
# MODEL_FILE = 'active_perception/models/epoch_29_model.hdf5'
# model = load_model(path.join(path.dirname(__file__), MODEL_FILE))

rospy.init_node('ggcnn_detection')

# Output publishers.
grasp_pub = rospy.Publisher('ggcnn/img/grasp', Image, queue_size=1)
grasp_plain_pub = rospy.Publisher('ggcnn/img/grasp_plain', Image, queue_size=1)
depth_pub = rospy.Publisher('ggcnn/img/depth', Image, queue_size=1)
ang_pub = rospy.Publisher('ggcnn/img/ang', Image, queue_size=1)
cmd_pub = rospy.Publisher('ggcnn/out/command', Float32MultiArray, queue_size=1)

# Initialise some globals.
prev_mp = np.array([150, 150])
ROBOT_Z = 0

# Tensorflow graph to allow use in callback.
graph = tf.get_default_graph()

# Get the camera parameters
camera_info_msg = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo)
K = camera_info_msg.K
fx = K[0]
cx = K[2]
fy = K[4]
cy = K[5]


# Execution Timing
class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = False

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))


def robot_pos_callback(data):
    global ROBOT_Z
    ROBOT_Z = data.pose.position.z


def depth_callback(depth_message):
    global model
    global graph
    global prev_mp
    global ROBOT_Z
    global fx, cx, fy, cy

    with TimeIt('Crop'):
        depth = bridge.imgmsg_to_cv2(depth_message)
    #  Crop a square out of the middle of the depth and resize it to 300*300
        crop_size = 400
    #     depth_crop = cv2.resize(depth[(480-crop_size)//2:(480-crop_size)//2+crop_size, (640-crop_size)//2:(640-crop_size)//2+crop_size], (300, 300))
    #
    #     # Replace nan with 0 for inpainting.
    #     depth_crop = depth_crop.copy()
    #     depth_nan = np.isnan(depth_crop).copy()
    #     depth_crop[depth_nan] = 0
    #
    # with TimeIt('Inpaint'):
    #     # open cv inpainting does weird things at the border.
    #     depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    #
    #     mask = (depth_crop == 0).astype(np.uint8)
    #     # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    #     depth_scale = np.abs(depth_crop).max()
    #     depth_crop = depth_crop.astype(np.float32)/depth_scale  # Has to be float32, 64 not supported.
    #
    #     depth_crop = cv2.inpaint(depth_crop, mask, 1, cv2.INPAINT_NS)
    #
    #     # Back to original size and value range.
    #     depth_crop = depth_crop[1:-1, 1:-1]
    #     depth_crop = depth_crop * depth_scale
    #
    #
    # with TimeIt('Inference'):
    #     # Run it through the network.
    #     depth_crop = np.clip((depth_crop - depth_crop.mean()), -1, 1)
    #     with graph.as_default():
    #         pred_out = model.predict(depth_crop.reshape((1, 300, 300, 1)))
    #
    #     points_out = pred_out[0].squeeze()
    #     points_out[depth_nan] = 0
    #     np.clip(points_out, 0.0, 1.0, points_out)
    #     points_out = points_out ** 2
    #
    # with TimeIt('Trig'):
    #     # Calculate the angle map.
    #     cos_out = pred_out[1].squeeze()
    #     sin_out = pred_out[2].squeeze()
    #     ang_out = np.arctan2(sin_out, cos_out)/2.0
    #
    #     width_out = pred_out[3].squeeze() * 150.0  # Scaled 0-150:0-1
    #
    # with TimeIt('Filter'):
    #     # Filter the outputs.
    #     points_out = ndimage.filters.gaussian_filter(points_out, 5.0)  # 3.0
    #     ang_out = ndimage.filters.gaussian_filter(ang_out, 2.0)

    points_out, ang_out, width_out, depth_crop = predict(depth, crop_size)

    with TimeIt('Calculate Depth'):
        # Figure out roughly the depth in mm of the part between the grippers for collision avoidance.
        depth_center = depth_crop[100:141, 130:171].flatten()
        depth_center.sort()
        depth_center = depth_center[:10].mean() * 1000.0

    with TimeIt('Control'):
        # Calculate the best pose from the camera intrinsics.
        maxes = None

        ALWAYS_MAX = True  # Use ALWAYS_MAX = True for the open-loop solution.

        if ROBOT_Z > 0.34 or ALWAYS_MAX:  # > 0.34 initialises the max tracking when the robot is reset.
            # Track the global max.
            max_pixel = np.array(np.unravel_index(np.argmax(points_out), points_out.shape))
            prev_mp = max_pixel.astype(np.int)
        else:
            # Calculate a set of local maxes.  Choose the one that is closes to the previous one.
            maxes = peak_local_max(points_out, min_distance=10, threshold_abs=0.1, num_peaks=3)
            if maxes.shape[0] == 0:
                return
            max_pixel = maxes[np.argmin(np.linalg.norm(maxes - prev_mp, axis=1))]

            # Keep a global copy for next iteration.
            prev_mp = (max_pixel * 0.25 + prev_mp * 0.75).astype(np.int)

        ang = ang_out[max_pixel[0], max_pixel[1]]
        width = width_out[max_pixel[0], max_pixel[1]]

        # Convert max_pixel back to uncropped/resized image coordinates in order to do the camera transform.
        max_pixel = ((np.array(max_pixel) / 300.0 * crop_size) + np.array([(480 - crop_size)//2, (640 - crop_size) // 2]))
        max_pixel = np.round(max_pixel).astype(np.int)

        point_depth = depth[max_pixel[0], max_pixel[1]]

        # These magic numbers are my camera intrinsic parameters.
        x = (max_pixel[1] - cx)/(fx) * point_depth
        y = (max_pixel[0] - cy)/(fy) * point_depth
        z = point_depth

        if np.isnan(z):
            return

    with TimeIt('Draw'):
        # Draw grasp markers on the points_out and publish it. (for visualisation)
        grasp_img = np.zeros((300, 300, 3), dtype=np.uint8)
        grasp_img[:,:,2] = (points_out * 255.0)

        grasp_img_plain = grasp_img.copy()

        rr, cc = circle(prev_mp[0], prev_mp[1], 5)
        grasp_img[rr, cc, 0] = 0
        grasp_img[rr, cc, 1] = 255
        grasp_img[rr, cc, 2] = 0

    with TimeIt('Publish'):
        # Publish the output images (not used for control, only visualisation)
        grasp_img = bridge.cv2_to_imgmsg(grasp_img, 'bgr8')
        grasp_img.header = depth_message.header
        grasp_pub.publish(grasp_img)

        grasp_img_plain = bridge.cv2_to_imgmsg(grasp_img_plain, 'bgr8')
        grasp_img_plain.header = depth_message.header
        grasp_plain_pub.publish(grasp_img_plain)

        depth_pub.publish(bridge.cv2_to_imgmsg(depth_crop))

        ang_pub.publish(bridge.cv2_to_imgmsg(ang_out))

        # Output the best grasp pose relative to camera.
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [x, y, z, ang, width, depth_center]
        cmd_pub.publish(cmd_msg)


depth_sub = rospy.Subscriber('/camera/depth/image_meters', Image, depth_callback, queue_size=1)
# robot_pos_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_pose', PoseStamped, robot_pos_callback, queue_size=1)

while not rospy.is_shutdown():
    rospy.spin()
