#!/bin/bash
rosrun dynamic_reconfigure dynparam set realsense2_camera_manager rs435_depth_enable_auto_exposure 0
rosrun dynamic_reconfigure dynparam set realsense2_camera_manager rs435_depth_enable_auto_exposure 1

# Workaround for the D435 autoexposure not working on system init
# https://github.com/intel-ros/realsense/issues/318
