// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_control_wrappers/cartesian_velocity_node_controller.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_control_wrappers {

bool CartesianVelocityNodeController::init(hardware_interface::RobotHW* robot_hardware,
                                              ros::NodeHandle& node_handle) {
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("CartesianVelocityNodeController: Could not get parameter arm_id");
    return false;
  }

  velocity_cartesian_interface_ =
      robot_hardware->get<franka_hw::FrankaVelocityCartesianInterface>();
  if (velocity_cartesian_interface_ == nullptr) {
    ROS_ERROR(
        "CartesianVelocityNodeController: Could not get Cartesian velocity interface from "
        "hardware");
    return false;
  }
  try {
    velocity_cartesian_handle_ = std::make_unique<franka_hw::FrankaCartesianVelocityHandle>(
        velocity_cartesian_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianVelocityNodeController: Exception getting Cartesian handle: " << e.what());
    return false;
  }

  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("CartesianVelocityNodeController: Could not get state interface from hardware");
    return false;
  }

  try {
    auto state_handle = state_interface->getHandle(arm_id + "_robot");
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianVelocityNodeController: Exception getting state handle: " << e.what());
    return false;
  }

  velocity_command_subscriber = node_handle.subscribe("~cartesian_velocity",
                                                       0,
                                                       &CartesianVelocityNodeController::cartesian_velocity_callback,
                                                       this);

  return true;
}

void CartesianVelocityNodeController::starting(const ros::Time& /* time */) {
  time_since_last_command = ros::Duration(0.0);
  velocity_command = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
}

void CartesianVelocityNodeController::cartesian_velocity_callback(const geometry_msgs::Twist::ConstPtr& msg) {
  velocity_command[0] = msg->linear.x;
  velocity_command[1] = msg->linear.y;
  velocity_command[2] = msg->linear.z;
  velocity_command[3] = msg->angular.x;
  velocity_command[4] = msg->angular.y;
  velocity_command[5] = msg->angular.z;

  time_since_last_command = ros::Duration(0.0);
}

void CartesianVelocityNodeController::update(const ros::Time& /* time */,
                                                const ros::Duration& period) {
  time_since_last_command += period;

  if(time_since_last_command.toSec() > 0.1) {
    velocity_command = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  }
  velocity_cartesian_handle_->setCommand(velocity_command);
}

void CartesianVelocityNodeController::stopping(const ros::Time& /*time*/) {
  // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
  // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
  // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

}  // namespace franka_control_wrappers

PLUGINLIB_EXPORT_CLASS(franka_control_wrappers::CartesianVelocityNodeController,
                       controller_interface::ControllerBase)
