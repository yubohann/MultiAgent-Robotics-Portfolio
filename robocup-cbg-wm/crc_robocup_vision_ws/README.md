# CRC RoboCup Vision ROS2 Workspace

This workspace is the ROS2 portfolio version of the China Robot Competition and RoboCup China robot vision challenge robot. It is the runtime side of the IsaacLab-to-ROS2 project, with navigation, target detection, shooter control, sensor fusion and behavior orchestration behind ROS2 interfaces.

Target platform.

- Ubuntu 24.04
- ROS2 Jazzy
- `colcon` as the build tool

Build.

```bash
cd crc_robocup_vision_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Main launch.

```bash
ros2 launch rcvrl_bringup competition.launch.py
```

Hardware-free launch check.

```bash
ros2 launch rcvrl_bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

Record the real or simulated sensor contract for drift and fusion analysis.

```bash
ros2 bag record /tf /tf_static /scan /imu/data_raw /wheel/odom /odometry/filtered \
  /camera/image_raw /camera/camera_info /target_detection /cmd_vel \
  /range/front_left /range/front_right /bumper/front_left /bumper/front_right
```

WSL note. Build from a native Linux path such as `~/crc_robocup_vision_ws`, because ROSIDL generation requires an ASCII path.

Package layout.

- `rcvrl_bringup` holds system launch files and top-level runtime wiring.
- `rcvrl_navigation` stores Nav2, slam_toolbox, maps and target route configuration.
- `rcvrl_motion` records motion telemetry and map-drift data.
- `rcvrl_vision` detects AprilTag Tag36h11 targets from camera images.
- `rcvrl_shooter` controls the serial laser module.
- `rcvrl_behavior` runs the competition state machine.
- `rcvrl_description` publishes the robot URDF and frame description.
- `rcvrl_interfaces` defines custom ROS2 message definitions.
- `rcvrl_docs` collects project documentation used for portfolio submission.

Runtime strategy notes.

- Robots receive opponent pose and track information, and relocalization triggers follow localization confidence.
- Shooter control preserves the rule model used in IsaacLab, with opponent targets only, a 5-50 cm normal-target shooter-outlet range, a 20-80 cm recessed-base range, line-of-sight blocking and at least 0.80 s laser dwell.
- Blue armor blockers protect the base targets, and the behavior layer treats the base target as visible once the armor route opens a legal shot.
- The RL bridge trains with Sim2Real domain randomization and an action shield, and ROS2 integration exposes real odometry, IMU, lidar, camera, ToF, bumper and shooter-state signals.
