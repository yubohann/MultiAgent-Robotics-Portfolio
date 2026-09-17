# Sim2Real Plan

This project transfers a defined Sim2Real layer, the ROS2 contract, with `/cmd_vel`, `/target_detection`, Nav2 goals, shooter services, TF frames and the opponent-target safety gate.

## Public Evidence Status

The repository publishes simulation, rule-environment scoring and IsaacLab replay evidence, and the 1v1 two-robot line has subsequently been exercised on real robots. Publicly validated here are the ROS2 deployment contract, the calibration procedure, the domain-randomization plan, rule tests, IsaacLab replay, simulation scoring and 1v1 real-robot experiment coverage.

A quantified real-robot claim package builds on hardware setup, calibration logs, rosbag2 recordings, run counts, success definitions, task success rate, collision and stuck statistics and arena condition notes. The exact capability list is maintained in `docs/capabilities.md`.

## Rule Contract

- Yellow robot attacks blue-side targets only.
- Blue robot attacks yellow-side targets only.
- Own base detections are rejected before firing.
- Route files carry `target_owner`, and the behavior node skips a target when `target_owner == team_color`.
- A normal target hit removes one opponent armor plate in order. A base target hit ends the match for the firing side when it is the opponent base, and hitting own base loses the match.

## Calibration Order

1. Set the `map` origin at the south-west arena corner, and verify the 3.0m x 3.0m arena extent, 0.5m start zones, 0.5m base zones, inner fences, and 0.3m cube obstacles.
2. Measure wheel radius, track width, chassis footprint, camera pose, lidar pose, shooter pose, and base link height, and update `description` and `bringup/config/sim2real.yaml`.
3. Mount and model IMU, wheel encoders, 2D lidar, RGB camera, depth and ToF range sensors, bumper contacts, and the fixed laser module with the same frames as the real robot.
4. Drive straight 1m and rotate 360 degrees on the real robot, and fit wheel radius, track width, motor deadband, max speed and acceleration limits until odom and measured motion agree.
5. Fuse `/wheel/odom` and `/imu/data_raw` with `robot_localization`, verify yaw stability after collision, and check that odom drift stays bounded during a 360-degree spin.
6. Record `camera_info`, tag size 0.05m, camera-to-base TF, exposure, focus and lighting, and check target distance and center error with AprilTag boards at 0.3m, 0.5m, 0.8m and 1.2m.
7. Measure camera-to-beam offset, serial command latency, hit radius and fire repeat interval, and save the offset in `sim2real.yaml` and the command bytes in `shooter/config/shooter.yaml`.
8. Compare Nav2 paths in sim and on the real arena, and keep the robot path clear of fences, cube obstacles and the other robot footprint.
9. Hit the bumper or push the chassis sideways to force localization confidence low, then verify the robot spins in place to rebuild the map before continuing.
10. Run yellow and blue separately with their own route files, and confirm both sides fire on opponent targets only.

## Domain Randomization

Randomize the values that are unstable in the real arena.

- lighting from 600 to 1200 lux
- tag yaw at 30, 45, and 60 degrees
- camera latency, image blur, exposure noise, and partial tag occlusion
- wheel slip, friction, motor deadband, and battery voltage scale
- obstacle and target placement error within the rule tolerance
- shooter fire latency and small beam-camera offset drift

## RL Interface

Use object-centric world-model SAC Flow self-play for elimination matches.

- observation covers robot pose estimate, opponent-target bearing and distance, tag visibility, own and opponent armor count, time remaining and obstacle distances.
- action covers high-level tactical controls for target selection, base-rush timing, blocking, recovery, fire gating and risk preference.
- reward covers progress toward opponent targets, valid target detection, clean alignment, successful hit, legal blocking, collision penalty, own-target penalty, timeout penalty and safe push-box progress.
- recovery uses blocked motion and bumper contact to reduce localization confidence, and the policy earns reward for spinning in place when confidence is poor.
- training uses many fast parallel rule environments before replaying the learned high-level policy in IsaacLab.
- deployment keeps Nav2, AprilTag detection, EKF and shooter services in the loop, with the policy acting through the ROS2 contract.

## Verification Ladder

1. Fast Python rule environment for world-model SAC Flow self-play and contract tests.
2. IsaacLab headless match replay.
3. IsaacLab GUI visual check covering target fall, armor motion, robot collision and laser blocking at walls.
4. ROS2 dry run with `shooter_dry_run:=true`.
5. Real robot tethered test at low speed.
6. One opponent normal target drill.
7. Four opponent normal targets plus opponent base target.
8. Full 180s elimination match with rosbag2 logging.

## Required Logs

Record these for every real run.

- `/tf`, `/tf_static`, `/odom`, `/scan`, `/camera/image_raw`, `/camera/camera_info`
- `/target_detection`, `/cmd_vel`, Nav2 action feedback, shooter service calls
- target hit timestamps, armor removal order, collision and stuck events, final winner
