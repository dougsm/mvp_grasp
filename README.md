# GG-CNN + Multi-View Picking

This repository contains the implementation of the Multi-View Picking system and experimental code for running on a [Franka Emika Panda](https://www.franka.de/panda/) Robot from the paper:

**Multi-View Picking: Next-best-view Reaching for Improved Grasping in Clutter**

*[Douglas Morrison](http://dougsm.com), [Peter Corke](http://petercorke.com), [Jürgen Leitner](http://juxi.net)*

International Conference on Robotics and Automation (ICRA), 2019

[arXiv](https://arxiv.org/abs/1809.08564) | [Video](https://www.youtube.com/watch?v=Vn3vSPKlaEk)

For more information about GG-CNN, see [this repository](https://github.com/dougsm/ggcnn) or [this arXiv paper](https://arxiv.org/abs/1804.05172).

If you use this work, please cite the following as appropriate:

```text
@inproceedings{morrison2019multiview, 
	title={{Multi-View Picking: Next-best-view Reaching for Improved Grasping in Clutter}}, 
	author={Morrison, Douglas and Corke, Peter and Leitner, J\"urgen}, 
	booktitle={2019 IEEE International Conference on Robotics and Automation (ICRA)}, 
	year={2019} 
}

@inproceedings{morrison2018closing, 
	title={{Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach}}, 
	author={Morrison, Douglas and Corke, Peter and Leitner, J\"urgen}, 
	booktitle={Proc.\ of Robotics: Science and Systems (RSS)}, 
	year={2018} 
}
```

**Contact**

Any questions or comments contact [Doug Morrison](mailto:doug.morrison@roboticvision.org).

## Setup

**Hardware:**

This code is designed around a Franka Emika Panda robot using an Intel Realsense D435 camera mounted on the wrist.  A 3D-printalbe camera mount is available in the `cad` folder. DYMO M10 scales are used to detect grasp success (*Optional*.  See the scales_interface directry for more information).

**The following external packages are required to run everything completely:**
* [ROS Kinetic](http://wiki.ros.org/kinetic/Installation)
* [Franka ROS 0.5.0](https://github.com/frankaemika/franka_ros/tree/0.5.0)
* [Realsense ROS](https://github.com/IntelRealSense/realsense-ros#installation-instructions)

**Installation:**

Clone this repository into your ROS worksapce and run `rosdep install --from-paths src --ignore-src --rosdistro=<your_rosdistro> -y` and then `catkin_make`/`catkin build`.

**Local python requirements can be installed by:**

```bash
pip install -r requirements.txt
```

## Packages Overview

* `dougsm_helpers`: A set of common functions for dealing with ROS and TF that are used throughout.
* `scales_interface`: A simple interface to a set of DYMO scales for reading weight.
* `ggcnn`: Service and Node for running GG-CNN.  Provides two 
* `franka_control_wrappers`: Add a simple velocity controller node and MoveIt commander for controlling the Panda robot.
* `mvp_grasping`: ROS nodes for executing grasps using the [Multi-View Picking](https://arxiv.org/abs/1809.08564) approach, including baselines.

## Running

To run grasping experiments:

```bash
# Start the robot and required extras.
roslaunch mvp_grasping robot_bringup.launch

# Start the camera, depth conversion and static transform
roslaunch mvp_grasping wrist_realsense.launch

# # Start the scales interface (disabled by default, useful if you have compatible scales)
# roslaunch scales_interface scales.launch

# Start the Multi-View Picking backend
roslaunch mvp_grasping grasp_entropy_service.launch
 
## Execute Grasping Experiment

# For Multi-View Picking
rosrun mvp_grasping panda_mvp_grasp.py

# For Fixed data-collection baseline
rosrun mvp_grasping panda_fixed_baseline.py

# For single-view open-loop grasping baseline
roslaunch ggcnn ggcnn_service.launch
rosrun mvp_grasping panda_open_loop_grasp.py
```


## Configuration

While this code has been written with specific hardware in mind, different physical settings or cameras may be used by customising `ggcnn/cfg/ggcnn_service.yaml` and `mvp_grasping/cfg/mvp_grasp.yaml`.
New robots and cameras will require major changes.
