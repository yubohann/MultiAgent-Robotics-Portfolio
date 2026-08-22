# MID-360 Interface Contract

## Evidence Status

This contract is implemented as an isolated ROS 2 Humble package. Topic availability, QoS compatibility, external FAST-LIO2 frame names, calibration, map registration, and target computer timing remain `TBD` until Linux/ROS 2 and hardware evidence exist.

## Frame Ownership

```text
map -> odom                  mid360_map_odom_anchor
odom -> base_link            mid360_pose_bridge
base_link -> imu_link        mid360_static_sensor_frames (calibration gate)
imu_link -> lidar_mid360     mid360_static_sensor_frames (calibration gate)
```

No external FAST-LIO2 node may publish `odom -> base_link`. Its source TF is private input and must be either disabled or renamed.

## Topics

| Topic | Type | Producer | Consumer | Contract |
| --- | --- | --- | --- | --- |
| `/livox/lidar` | `livox_ros_driver2/msg/CustomMsg` | official driver | FAST-LIO2, input guard | `timebase` and every `offset_time` retained; no header-only conversion |
| `/livox/imu` | `sensor_msgs/msg/Imu` | official driver | FAST-LIO2, input guard | finite acceleration/gyro; non-zero timestamp; sensor QoS |
| `/Odometry` | `nav_msgs/msg/Odometry` | selected FAST-LIO2 | pose bridge, preflight | source frame/child frame configured explicitly; private upstream topic |
| `/mid360/local_odometry` | `nav_msgs/msg/Odometry` | pose bridge | map anchor, control adapter | `odom -> base_link`; complete normalized quaternion; timestamp copied from source |
| `/mid360/localization_odometry` | `nav_msgs/msg/Odometry` | map anchor | competition logic, vision/control | `map -> base_link`; emitted only after anchor exists |
| `/mid360/pose_valid` | `std_msgs/msg/Bool` | pose bridge | map anchor, preflight, control | false for invalid/stale source; not a replacement for diagnostics |
| `/mid360/input_valid` | `std_msgs/msg/Bool` | input guard | preflight, supervisor | true only when both streams are structurally valid and fresh |
| `/mid360/map_locked` | `std_msgs/msg/Bool` | map anchor | preflight, supervisor | true only after accepted `/initialpose` or verified correction |
| `/mid360/pose_status` | `std_msgs/msg/String` JSON v2 | map anchor | supervisor, diagnostics/UI, recorder | versioned tracking state, map lock, pose age, anchor source/reason, and a `quality` object containing stream drops, map fitness/scan points, degeneracy/dynamic/overlap/resource/protection fields; unavailable measurements are `unknown` |
| `/mid360/relocalization_request` | `std_msgs/msg/Bool` | operator/UI | map anchor | true enters `RELOCALIZING`; a valid `/initialpose` or verified correction clears it |
| `/mid360/preflight_ready` | `std_msgs/msg/Bool` | preflight | competition supervisor | absolute-pose actions require true |

## Diagnostic Fields

The current prototype publishes `diagnostic_msgs/DiagnosticArray` alongside the compatibility booleans. Required fields are being expanded incrementally and must not be silently removed:

- stream: `point_count`, `offset_span_ns`, `timebase`, `lidar_ok`, `imu_ok`, `lidar_fresh`, `imu_fresh`;
- pose: `pose_age_sec`, `source_sequence`, `accepted_sequence`, `drop_count`, `frame_ok`, `quality_reason`;
- map: `tracking_state` (`UNINITIALIZED`, `TRACKING`, `RELOCALIZING`, `LOST`), `map_locked`, `anchor_source`, `anchor_age_sec`, `correction_rejected_reason`;
- preflight: topic types, all prerequisite booleans, freshness and readiness reason.
- quality contract: `degeneracy_score`, `dynamic_point_ratio`, `scan_map_overlap`, `map_update_allowed`, `map_update_veto_reason`, `protection_level`, `uncertainty_bound_m`, resource watermarks, input drops and map matcher fitness/scan points. Values are `unknown` until a producer and real evidence exist.

Unknown values must be represented as `unknown` or `TBD`, never as a fabricated zero measurement.

## QoS And Timing

Sensor subscriptions use `qos_profile_sensor_data`. Exact driver QoS is checked with `ros2 topic info -v` on Ubuntu. The initial competition gate is 0.30 s pose/input silence.
