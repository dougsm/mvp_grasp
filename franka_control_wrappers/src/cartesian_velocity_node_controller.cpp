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
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianVelocityNodeController: Exception getting state handle: " << e.what());
    return false;
  }


  node_handle.param<double>("max_duration_between_commands", max_duration_between_commands, 0.01);

  // Rate Limiting
  if(!node_handle.getParam("rate_limiting/linear/velocity", max_velocity_linear)) {
    ROS_ERROR("CartesianVelocityNodeController: Could not get parameter rate_limiting/linear/velocity");
    return false;
  }
  if(!node_handle.getParam("rate_limiting/linear/acceleration", max_acceleration_linear)) {
    ROS_ERROR("CartesianVelocityNodeController: Could not get parameter rate_limiting/acc/acceleration");
    return false;
  }
  if(!node_handle.getParam("rate_limiting/linear/jerk", max_jerk_linear)) {
    ROS_ERROR("CartesianVelocityNodeController: Could not get parameter rate_limiting/linear/jerk");
    return false;
  }
  if(!node_handle.getParam("rate_limiting/angular/velocity", max_velocity_angular)) {
    ROS_ERROR("CartesianVelocityNodeController: Could not get parameter rate_limiting/angular/velocity");
    return false;
  }
  if(!node_handle.getParam("rate_limiting/angular/acceleration", max_acceleration_angular)) {
    ROS_ERROR("CartesianVelocityNodeController: Could not get parameter rate_limiting/acc/acceleration");
    return false;
  }
  if(!node_handle.getParam("rate_limiting/angular/jerk", max_jerk_angular)) {
    ROS_ERROR("CartesianVelocityNodeController: Could not get parameter rate_limiting/angular/jerk");
    return false;
  }

  node_handle.param<bool>("stop_on_contact", stop_on_contact, true);

  velocity_command_subscriber = node_handle.subscribe("cartesian_velocity",
                                                       10,
                                                       &CartesianVelocityNodeController::cartesian_velocity_callback,
                                                       this);

  return true;
}

void CartesianVelocityNodeController::starting(const ros::Time& /* time */) {
  time_since_last_command = ros::Duration(0.0);
  velocity_command = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  last_sent_velocity = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
}

void CartesianVelocityNodeController::cartesian_velocity_callback(const geometry_msgs::Twist::ConstPtr& msg) {
  // Callback for ROS message
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
  // Update the controller at 1kHz
  time_since_last_command += period;

  // If no message received in set time,
  if(time_since_last_command.toSec() > max_duration_between_commands) {
    velocity_command = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  }

  auto state = state_handle_->getRobotState();

  // Check for contacts
  if(stop_on_contact) {
    for (size_t i = 0; i < state.cartesian_contact.size(); i++) {
      if(state.cartesian_contact[i]) {
        velocity_command = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
        // ROS_ERROR_STREAM("Detected Cartesian Contact in Direction "  << i);
      }
    }
  }

  last_sent_velocity = franka::limitRate(
    max_velocity_linear,
    max_acceleration_linear,
    max_jerk_linear,
    max_velocity_angular,
    max_acceleration_angular,
    max_jerk_angular,
    velocity_command,
    state.O_dP_EE_c,
    state.O_ddP_EE_c
  );

  velocity_cartesian_handle_->setCommand(last_sent_velocity);
}

void CartesianVelocityNodeController::stopping(const ros::Time& /*time*/) {
  // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
  // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
  // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

}  // namespace franka_control_wrappers

PLUGINLIB_EXPORT_CLASS(franka_control_wrappers::CartesianVelocityNodeController,
                       controller_interface::ControllerBase)
