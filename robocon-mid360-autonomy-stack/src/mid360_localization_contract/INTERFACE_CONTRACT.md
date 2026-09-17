# MID-360 Interface Contract

## Evidence Status

This contract is implemented as an isolated ROS 2 Humble package. Topic availability, QoS compatibility, external FAST-LIO2 frame names, calibration, map registration and target computer timing carry `TBD` entries pending Linux, ROS 2 and hardware evidence.

## Frame Ownership

```text
map -> odom                  mid360_map_odom_anchor
odom -> base_link            mid360_pose_bridge
base_link -> imu_link        mid360_static_sensor_frames (calibration gate)
imu_link -> lidar_mid360     mid360_static_sensor_frames (calibration gate)
```

External FAST-LIO2 `odom -> base_link` publication stays disabled or renamed, and its source TF is private input.

## Topics

| Topic | Type | Producer | Consumer | Contract |
| --- | --- | --- | --- | --- |
| `/livox/lidar` | `livox_ros_driver2/msg/CustomMsg` | official driver | FAST-LIO2, input guard | `timebase` and every `offset_time` retained, with full message conversion |
| `/livox/imu` | `sensor_msgs/msg/Imu` | official driver | FAST-LIO2, input guard | finite acceleration and gyro, a non-zero timestamp and sensor QoS |
| `/Odometry` | `nav_msgs/msg/Odometry` | selected FAST-LIO2 | pose bridge, preflight | source frame and child frame configured explicitly, private upstream topic |
| `/mid360/local_odometry` | `nav_msgs/msg/Odometry` | pose bridge | map anchor, control adapter | `odom -> base_link` with a complete normalized quaternion and a timestamp copied from source |
| `/mid360/localization_odometry` | `nav_msgs/msg/Odometry` | map anchor | competition logic, vision and control | `map -> base_link`, emitted after the anchor exists |
| `/mid360/pose_valid` | `std_msgs/msg/Bool` | pose bridge | map anchor, preflight, control | false for invalid or stale source, with diagnostics published separately |
| `/mid360/input_valid` | `std_msgs/msg/Bool` | input guard | preflight, supervisor | true only when both streams are structurally valid and fresh |
| `/mid360/map_locked` | `std_msgs/msg/Bool` | map anchor | preflight, supervisor | true only after accepted `/initialpose` or verified correction |
| `/mid360/pose_status` | `std_msgs/msg/String` JSON v2 | map anchor | supervisor, diagnostics and UI, recorder | versioned tracking state, map lock, pose age, anchor source and reason, and a `quality` object containing stream drops, map fitness and scan points, degeneracy, dynamic, overlap, resource and protection fields |
| `/mid360/relocalization_request` | `std_msgs/msg/Bool` | operator and UI | map anchor | true enters `RELOCALIZING`, and a valid `/initialpose` or verified correction clears it |
| `/mid360/preflight_ready` | `std_msgs/msg/Bool` | preflight | competition supervisor | absolute-pose actions require true |

## Diagnostic Fields

The current prototype publishes `diagnostic_msgs/DiagnosticArray` alongside the compatibility booleans. Required fields grow incrementally.

- stream fields cover `point_count`, `offset_span_ns`, `timebase`, `lidar_ok`, `imu_ok`, `lidar_fresh` and `imu_fresh`.
- pose fields cover `pose_age_sec`, `source_sequence`, `accepted_sequence`, `drop_count`, `frame_ok` and `quality_reason`.
- map fields cover `tracking_state` with `UNINITIALIZED`, `TRACKING`, `RELOCALIZING` and `LOST`, plus `map_locked`, `anchor_source`, `anchor_age_sec` and `correction_rejected_reason`.
- preflight fields cover topic types, all prerequisite booleans, freshness and readiness reason.
- quality fields cover `degeneracy_score`, `dynamic_point_ratio`, `scan_map_overlap`, `map_update_allowed`, `map_update_veto_reason`, `protection_level`, `uncertainty_bound_m`, resource watermarks, input drops and map matcher fitness and scan points.

`unknown` or `TBD` marks any value that awaits a producer and real evidence.

## QoS And Timing

Sensor subscriptions use `qos_profile_sensor_data`. Exact driver QoS is checked with `ros2 topic info -v` on Ubuntu. The initial competition gate is 0.30 s pose and input silence.
