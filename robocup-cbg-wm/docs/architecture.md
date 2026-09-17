# Architecture

RoboCup VisionRL is organized as a ROS2 Jazzy system with clear package interfaces and a Sim2Real contract shared by IsaacLab training, ROS2 dry runs, and real-robot deployment.

## Runtime Graph

```text
/camera/image_raw + /camera/camera_info
        |
        v
rcvrl_vision/apriltag_detector
        |
        v
/target_detection  --------------------+
                                       |
Nav2 /navigate_to_pose action <--- rcvrl_behavior/competition_behavior
                                       |
/cmd_vel ------------------------------+
                                       |
/shooter/enable / /fire / /disable <---+
        |
        v
rcvrl_shooter/shooter_controller -> serial laser module
```

The behavior node owns the competition rule gate and requires `team_color` and `target_owner` to describe opposite teams before a fire command passes. Own base detections are rejected regardless of route file content.

Sensor fusion is a separate runtime layer. Wheel odometry and IMU feed `robot_localization/ekf_node`, lidar feeds Nav2 costmaps, and camera, depth, ToF and bumper topics remain available to behavior or future RL policy adapters.

## Packages

- `rcvrl_bringup` starts the complete competition stack.
- `rcvrl_description` publishes the robot model and TF frames.
- `rcvrl_navigation` stores Nav2, slam_toolbox, map and target route configuration.
- `rcvrl_vision` detects AprilTag Tag36h11 targets from the camera stream.
- `rcvrl_shooter` controls the fixed 5V laser module through parameterized serial commands.
- `rcvrl_behavior` owns the competition state machine and keeps perception code separate from control code.
- `rcvrl_interfaces` defines `TargetDetection`.

## Topics, Services and Actions

| Name | Type | Owner | Purpose |
| --- | --- | --- | --- |
| `/camera/image_raw` | `sensor_msgs/Image` | camera driver | Camera frames for visual target detection |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | camera driver | Camera intrinsic calibration |
| `/depth/image_raw` | `sensor_msgs/Image` | depth camera driver | Short-range depth for obstacle and target context |
| `/scan` | `sensor_msgs/LaserScan` | 2D lidar | Mapping, localization and Nav2 obstacle layers |
| `/imu/data_raw` | `sensor_msgs/Imu` | IMU driver | Yaw rate and acceleration for EKF |
| `/wheel/odom` | `nav_msgs/Odometry` | base controller | Wheel encoder odometry |
| `/odometry/filtered` | `nav_msgs/Odometry` | `robot_localization` | Fused odometry from wheel encoders and IMU |
| `/range/front_left`, `/range/front_right` | `sensor_msgs/Range` | ToF drivers | Close obstacle and ring-side checks |
| `/bumper/front_left`, `/bumper/front_right` | `std_msgs/Bool` | bumper contacts | Collision and localization-health triggers |
| `/target_detection` | `rcvrl_interfaces/TargetDetection` | `rcvrl_vision` | Structured target observation |
| `/cmd_vel` | `geometry_msgs/Twist` | `rcvrl_behavior`, Nav2 | Base velocity control |
| `/shooter/enable` | `std_srvs/Trigger` | `rcvrl_shooter` | Enable the laser module |
| `/shooter/fire` | `std_srvs/Trigger` | `rcvrl_shooter` | Send the firing command |
| `/shooter/disable` | `std_srvs/Trigger` | `rcvrl_shooter` | Disable the laser module |
| `navigate_to_pose` | `nav2_msgs/NavigateToPose` | Nav2 | Navigate to a configured target pose |

## Competition Config

- `rcvrl_navigation/config/targets.elimination.yellow.yaml` keeps yellow on the yellow side with blue-side targets only.
- `rcvrl_navigation/config/targets.elimination.blue.yaml` keeps blue on the blue side with yellow-side targets only.
- `rcvrl_bringup/config/sim2real.yaml` holds field, robot, sensor, shooter and randomization parameters for real-world calibration.

## State Machine

```text
INIT -> NAVIGATE -> SEARCH -> ALIGN -> FIRE -> NEXT_TARGET
                         ^                    |
                         |                    v
                         +--------- retry / skip

NAVIGATE/SEARCH/ALIGN -> RECOVER_LOCALIZATION -> NAVIGATE

NEXT_TARGET -> NAVIGATE or RETURN_HOME -> END
```

Timeouts are parameterized for the 3-minute match window and the 20-second no-progress rule. `RECOVER_LOCALIZATION` spins in place to rebuild lidar, IMU and camera consistency after collision or repeated navigation failure.
