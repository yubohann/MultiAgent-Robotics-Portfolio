# ROBOCON MID-360 Autonomy Stack

## Research Report and Interview Memory Notes

**Project:** Dual-robot autonomy for the ABU ROBOCON 2025 Robot Basketball task  
**Primary sensor:** Livox MID-360 LiDAR with built-in IMU  
**Software:** ROS 2 Humble, FAST-LIO2, RGB-D/YOLO, ROS 2 launch and diagnostics  
**Execution boundary:** NUC high-level coordination -> UART/ESP-NOW -> STM32/CAN real-time control  
**Source workspaces:** `G:\mid360_ws_test_20260819`, `G:\MID360_ROBOCON_C_archive_20260825`  
**Portfolio references:** `docs/project-deep-dives/robocon-mid360-autonomy-stack*.md`

> This document is written for research discussion, graduate interviews, and technical portfolio review. It describes the current source that can be inspected directly, then connects it to the broader physical competition system documented in the portfolio. Values marked `TBD`, `unknown`, or `synthetic` remain evidence fields awaiting the corresponding Linux, bag, calibration, or hardware record.

---

## 1. Remember This First

### One sentence

I built a rule-aware dual-robot autonomy stack in which MID-360 LiDAR and IMU measurements are converted by FAST-LIO2 into a quality-checked local pose, a fixed-map anchor supplies the field frame, and a ROS 2 supervisor turns fresh geometric and semantic evidence into acknowledged, safety-gated basketball actions executed by STM32/CAN controllers.

### Five keywords

`MID-360` · `FAST-LIO2` · `fixed-map localization` · `quality contracts` · `dual-robot task supervision`

### Thirty-second version

The project addresses a practical competition problem: two robots must move, receive, pass, aim, and shoot while their viewpoint, vibration, network, and mechanism state change continuously. I separated the system into sensing, estimation, perception, task reasoning, communication, embedded execution, and safety layers. The MID-360 `CustomMsg` keeps its `timebase` and per-point `offset_time`, preserving the information required for motion compensation. A pose bridge converts the FAST-LIO tracking frame into one canonical `odom -> base_link` stream, while a map anchor maintains `map -> odom`. The supervisor accepts a shot only when pose, map lock, target freshness, ball presence, mechanism health, and teammate safety are all current. This gives the team a system that can be calibrated, inspected, replayed, and recovered on a real competition robot.

### Core research claim

Reliable competition autonomy comes from making every transition evidence-bearing. A measurement has a frame, timestamp, quality state, and diagnostic reason; an action has a permission gate, transport acknowledgement, physical completion signal, timeout, and recovery path.

---

## 2. Abstract

ROBOCON Robot Basketball couples continuous motion with discrete rules: possession and passing deadlines, field zones, target geometry, shooting conditions, and a match clock. The project develops an end-to-end autonomy stack for this setting. A Livox MID-360 and its internal IMU feed a FAST-LIO2 LiDAR-inertial odometry front end. A fixed field map provides a global reference without interrupting the locally continuous trajectory. RGB-D/YOLO supplies basket observations and depth. ROS 2 connects estimation, perception, task supervision, dual-robot communication, and high-level control. UART and ESP-NOW carry versioned messages between the high-level computers and the two robots; STM32/CAN retains deterministic motor and mechanism control, including PID, limits, current protection, timeout handling, and emergency stop.

The inspected research workspace turns this architecture into explicit software contracts. `mid360_input_guard` validates Livox packet structure and IMU health. `mid360_pose_bridge` enforces source frame names, normalizes quaternions, transforms a tracking-origin pose to the robot base, and publishes freshness and sequence diagnostics. `mid360_map_odom_anchor` computes `T_map_odom` from `/initialpose` or a verified correction and exposes a four-state tracking machine. `game_supervisor` implements competition lifecycle states, ordered pass evidence, a `FireShot` safety gate, and an expiring, idempotent message protocol.

The engineering result is a reproducible interface layer around external algorithms and hardware. The research value lies in the explicit propagation of uncertainty and failure information from raw sensor timing to task-level decisions. The same interfaces support synthetic tests, bag replay, NUC deployment, and later field validation.

---

## 3. Evidence Scope and Version Map

### 3.1 Directly inspectable source

The current G-drive workspace contains:

| Package or dependency | Evidence in source | Role |
| --- | --- | --- |
| `mid360_localization_contract` | Python ROS 2 nodes, YAML templates, launch files, unit tests | Input validation, pose conversion, map anchor, preflight readiness |
| `game_supervisor` | Python ROS 2 adapter, deterministic core, protocol and tests | Match lifecycle, action gates, dual-robot message semantics |
| `vendor_fast_lio` | FAST-LIO2 ROS package and MID-360 profile | External LiDAR-inertial odometry front end |
| `vendor_livox_ros_driver2` | Official Livox ROS 2 driver | MID-360 transport and `CustomMsg` publication |

The workspace is deliberately isolated. Private maps, network addresses, serial numbers, firmware versions, measured extrinsics, and hardware statistics remain outside the checked-in templates.

### 3.2 Broader physical-system evidence

The portfolio's detailed technical documents describe the complete competition robot integration: RGB-D/YOLO basket perception, dual-robot coordination, UART/ESP-NOW, STM32/CAN, chassis and mechanism tests, field debugging, and national second prizes in the Robot Basketball and Robot Basketball Shooting events. Those documents provide system context and field experience; the G-drive prototype provides the inspectable localization and supervisor implementation.

### 3.3 Evolution of the source

The project materials include both an early integrated ROBOCON system description and a later isolated contract workspace. The relationship is evolutionary:

```text
physical competition integration
    -> interface audit and failure analysis
    -> isolated MID-360 / FAST-LIO2 contracts
    -> deterministic game supervisor and deployment templates
    -> measured hardware validation and replay artifacts
```

This report keeps those stages connected while identifying which statement comes from code, a template, a design document, or a field result.

---

## 4. Competition Problem as a Research System

### 4.1 Operational challenge

Robot Basketball requires a robot pair to maintain a shared task state while each platform is moving and sensing independently. The robot must answer four questions repeatedly:

1. Where am I in the field coordinate system?
2. Is the target observation current and geometrically usable?
3. Does the rulebook permit this action at this moment?
4. Did the physical mechanism complete the requested action?

The questions form a closed loop rather than a sequence of independent modules. A pose jump changes aiming geometry. A stale target changes shot permission. A teammate timeout changes pass readiness. A missing mechanism acknowledgement changes the next task.

### 4.2 Rule values used by the documented system

The detailed project report records the ABU ROBOCON 2025 Robot Basketball rule revision dated 2024-08-14. The following values explain the architecture:

| Rule item | Documented value | System consequence |
| --- | ---: | --- |
| Playing field | `15.0 m x 8.0 m` | Map scale, boundary geometry, and initial pose share one frame |
| Fence | `0.10 m` high, `0.05 m` wide | Boundary geometry enters mapping and local motion checks |
| Backboard | `1.80 m x 1.05 m` | Target perception uses a structured physical object |
| Basket height | `2.43 m` | Shooting model uses a fixed target height |
| Inner basket diameter | `0.450-0.459 m` | Target-center estimation and mechanism repeatability matter |
| Preliminary match | `120 s` | Match clock is a supervisor state |
| Knockout match | `160 s` | Match phase belongs in a versioned rule profile |
| Possession | `20 s` | Ball state has a deadline and expiry path |
| Advance | `8 s` | Transition to the offensive side has a dedicated timer |
| Minimum pass distance | `1.0 m` | Two-robot synchronized pose is part of pass preparation |
| Dribble hand height | `>= 0.70 m` | Mechanism or physical feedback must support evidence |
| Ordinary scores | `2` and `3` points | Shooting zone and shot type enter task reasoning |
| Dunk score | `7` points | Dunk requires additional spatial and motion evidence |

The numeric profile belongs in a private, versioned game configuration. The checked-in `game_profile.template.yaml` intentionally uses `TBD` placeholders until the official rule revision and robot hardware are verified together.

### 4.3 Research questions

The project can be framed through three research questions:

**RQ1 - Geometric trust:** How can a high-rate LiDAR-inertial estimator remain useful to task control when source frames, timestamps, extrinsics, or odometry freshness are uncertain?

**RQ2 - Semantic actionability:** How can target observations and mechanism feedback become explicit evidence instead of loosely coupled callbacks?

**RQ3 - System recovery:** How can localization loss, stale communication, or actuator faults move the robot into a bounded recovery state and later restore progress?

The implementation answers these questions with contracts, state machines, and diagnostic fields that make quality visible at the consumer boundary.

---

## 5. Overall Architecture

```text
Livox MID-360 CustomMsg + built-in IMU
                |
                v
        mid360_input_guard
                |
                v
       FAST-LIO2 local odometry
          private source frame
                |
                v
        mid360_pose_bridge
     canonical odom -> base_link
                |
                v
       mid360_map_odom_anchor
       fixed map -> odom anchor
                |
                +----------------------+
                |                      |
                v                      v
   localization / preflight       RGB-D + YOLO basket target
                |                      |
                +----------+-----------+
                           v
                 game_supervisor
              rule gates, roles, timers, recovery
                           |
                           v
                 dual-robot communication
             UART / ESP-NOW, ACK, expiry, dedup
                           |
                           v
                  STM32 / CAN execution
              chassis, shooter, mechanism feedback
```

### 5.1 Layer responsibilities

| Layer | Inputs | Outputs | Design decision |
| --- | --- | --- | --- |
| Sensor transport | MID-360 packets, IMU | ROS 2 sensor topics | Preserve native timing and message type |
| Input quality | `CustomMsg`, `Imu` | `input_valid`, diagnostics | Reject malformed or stale streams early |
| Local estimation | LiDAR, IMU | FAST-LIO odometry | Keep local motion continuous and high-rate |
| Pose contract | source odometry, calibration | `/mid360/local_odometry`, TF | One canonical `odom -> base_link` owner |
| Global localization | local pose, `/initialpose`, correction | `/mid360/localization_odometry`, `map -> odom` | Add field reference without resetting local motion |
| Semantic perception | RGB-D frames | target detection and freshness | Convert pixels and depth into task evidence |
| Task supervision | pose, map, target, ball, mechanism, teammate | action decisions | Require a complete evidence set before action |
| Transport protocol | task messages | ACKs, sequence and heartbeat | Expire and deduplicate messages deterministically |
| Real-time control | high-level actions | motor/mechanism feedback | Keep safety-critical loops on STM32/CAN |

The interfaces make module ownership explicit. This is valuable during field debugging because a failure can be localized to transport, estimation, perception, task logic, or actuation without hiding the original reason.

---

## 6. MID-360 Data Path and Time Semantics

### 6.1 Native Livox message

The driver publishes `/livox/lidar` as `livox_ros_driver2/msg/CustomMsg` and `/livox/imu` as `sensor_msgs/msg/Imu`. `CustomMsg` contains:

```text
header       ROS header
timebase     timestamp of the first point
point_num    number of points
lidar_id     device identifier
points[]     CustomPoint records
```

Each `CustomPoint` contains `offset_time` together with position, reflectivity, tag, and scan line. The per-point time is important because a rotating LiDAR observes different points at different instants. FAST-LIO2 can use those offsets for motion compensation; copying one frame timestamp onto every point would remove that information.

### 6.2 `mid360_input_guard`

The implementation in `mid360_localization_contract/input_guard.py` performs the following checks:

- `point_num` equals the received point-array length;
- point count meets `minimum_points`;
- `timebase` is non-zero and increases across accepted packets;
- the LiDAR header timestamp is non-zero and increasing;
- every `offset_time` is monotonically nondecreasing;
- IMU timestamps are non-zero and increasing;
- all angular velocity and linear acceleration values are finite;
- LiDAR and IMU arrival times remain inside their configured silence limits.

The node publishes `/mid360/input_valid` and a `DiagnosticArray` with point count, offset span, stream freshness, accepted sequence, received sequence, drop count, and the latest reason. This turns “the sensor seems unstable” into an observable contract.

### 6.3 Configuration and timing

The competition profile sets a `0.30 s` target silence limit for both LiDAR and IMU. The value is a deployment threshold to validate on the robot. FAST-LIO2's template uses a `10 Hz` scan rate, MID-360 LiDAR type, a `0.5 m` blind range, and an explicit `time_offset_lidar_to_imu` field. Extrinsics and time offsets remain calibration inputs rather than assumed facts.

---

## 7. FAST-LIO2 and Pose Contract

### 7.1 Estimation role

FAST-LIO2 remains responsible for LiDAR-inertial odometry. Conceptually, the estimator propagates pose and velocity with IMU measurements and corrects the state with LiDAR geometric residuals. A point-to-plane residual can be written as:

\[
r_i = n_i^T(Rp_i + p - q_i),
\]

where `p_i` is a point in the sensor frame, `R,p` describe the current pose, and `n_i,q_i` describe a local map plane. The actual FAST-LIO2 implementation is an external dependency; the project owns the message, frame, timing, and quality boundary around it.

### 7.2 `mid360_pose_bridge`

The bridge subscribes to a configured source `nav_msgs/Odometry` topic, usually `/Odometry`. It enforces the configured source odometry frame and tracking child frame. The default source names are `TBD` until the selected FAST-LIO2 build is inspected.

For a source pose expressed at the tracking origin, the bridge computes the base pose as:

\[
T_{odom\rightarrow base}
    = T_{odom\rightarrow tracking}
      \; T_{base\rightarrow tracking}^{-1}.
\]

The bridge then:

- normalizes the full quaternion and rejects a zero-norm quaternion;
- copies the source timestamp and covariance;
- transforms angular and linear velocity to the base origin;
- applies optional position-step, speed, yaw-rate, and height quality limits;
- publishes `/mid360/local_odometry` with `odom -> base_link`;
- publishes the only `odom -> base_link` TF;
- reports `pose_valid`, age, source sequence, accepted sequence, drop count, frame status, and the quality reason.

The design prevents a common integration failure: two nodes competing to publish `odom -> base_link`. FAST-LIO2 can publish a private tracking transform or disable its TF output, while the bridge owns the public contract.

### 7.3 Why full quaternion handling matters

Yaw is derived from the normalized quaternion through the complete rotation representation. Reading the quaternion `z` component as yaw works only in a narrow planar special case. The geometry tests include roll and pitch, scaled non-unit quaternions, zero-quaternion rejection, transform composition, and sensor-height removal.

---

## 8. Fixed-Map Localization and Recovery

### 8.1 Separating local motion from global reference

Local odometry is continuous and responsive to short-term motion. A fixed map supplies the field coordinate needed for target poses, pass positions, shooting zones, and repeatable competition geometry. The map anchor therefore estimates:

\[
T_{map\rightarrow odom}
    = T_{map\rightarrow base}
      \; T_{odom\rightarrow base}^{-1}.
\]

The anchor changes the global reference while preserving the local odometry trajectory.

### 8.2 `mid360_map_odom_anchor`

The node consumes:

- `/mid360/local_odometry` for the latest local pose;
- `/initialpose` as a `PoseWithCovarianceStamped` in the `map` frame;
- `/mid360/map_to_odom_correction` as a verified `map -> odom` odometry correction;
- `/mid360/pose_valid` and `/mid360/relocalization_request` for quality and recovery control.

An initial pose is accepted when local odometry is present and fresh and `pose_valid` is true. A correction is accepted only with `header.frame_id=map` and `child_frame_id=odom`. The node then publishes:

- `/mid360/localization_odometry` with `map -> base_link`;
- `/mid360/map_locked`;
- `/mid360/pose_status` as versioned JSON;
- `DiagnosticArray` fields for tracking state, anchor source, anchor age, pose age, and rejection reason;
- the sole `map -> odom` TF while tracking is locked.

### 8.3 Tracking state machine

```text
UNINITIALIZED --accepted anchor--> TRACKING
TRACKING --invalid or stale pose--> LOST
TRACKING --relocalization request--> RELOCALIZING
LOST --valid anchor and fresh pose--> TRACKING
RELOCALIZING --new accepted anchor--> TRACKING
```

The implementation uses four explicit states:

- `UNINITIALIZED`: no accepted map anchor exists;
- `TRACKING`: anchor exists, pose is valid and odometry is fresh;
- `RELOCALIZING`: operator or system requested a new anchor;
- `LOST`: an existing anchor is temporarily unusable because pose quality or freshness failed.

Only `TRACKING` sets `map_locked=true`. This makes absolute-coordinate actions depend on evidence rather than on the continued existence of an old TF.

### 8.4 Preflight readiness

`mid360_preflight` checks topic types, input validity, pose validity, canonical odometry freshness, and map lock when required. It publishes `/mid360/preflight_ready` and a diagnostic reason. The default odometry silence target is `0.30 s`, with a `10 s` startup grace period. The competition supervisor consumes this readiness signal before entering `WAIT_START`.

---

## 9. ROS 2 Runtime and Launch Composition

### 9.1 Public topic contract

| Topic | Type | Purpose |
| --- | --- | --- |
| `/livox/lidar` | `livox_ros_driver2/msg/CustomMsg` | Native MID-360 packet stream |
| `/livox/imu` | `sensor_msgs/msg/Imu` | IMU stream for FAST-LIO2 and health checks |
| `/Odometry` | `nav_msgs/msg/Odometry` | Selected FAST-LIO2 source pose |
| `/mid360/local_odometry` | `nav_msgs/msg/Odometry` | Canonical `odom -> base_link` pose |
| `/mid360/localization_odometry` | `nav_msgs/msg/Odometry` | Canonical `map -> base_link` pose |
| `/mid360/input_valid` | `std_msgs/msg/Bool` | Input structure and freshness gate |
| `/mid360/pose_valid` | `std_msgs/msg/Bool` | Pose frame, quality, and freshness gate |
| `/mid360/map_locked` | `std_msgs/msg/Bool` | Fixed-map tracking state |
| `/mid360/preflight_ready` | `std_msgs/msg/Bool` | System-level prerequisite for match logic |
| `/mid360/pose_status` | `std_msgs/msg/String` | Versioned JSON tracking summary |

### 9.2 Launch entry points

`robocon_mid360_bringup.launch.py` composes the official Livox driver, the selected `fastlio_mapping` executable, and the localization contract launch. It registers process-exit handlers so that a failed driver or FAST-LIO2 process shuts down the top-level bringup coherently.

`robocon_competition.launch.py` composes the MID-360 bringup with `game_supervisor`. Driver configuration, FAST-LIO2 parameters, contract parameters, and supervisor parameters are launch arguments. This keeps private deployment values outside the public template.

### 9.3 Deployment sequence

1. Identify the NUC model, active NIC, MID-360 IP, driver version, and firmware.
2. Copy the public templates to private configuration files.
3. Verify `/livox/lidar` type, point timing, IMU stamps, and QoS with the official driver.
4. Record raw LiDAR/IMU bags for stationary, turning, and fast-transition calibration.
5. Measure LiDAR-IMU time offset and `base_link -> imu_link` / `imu_link -> lidar_mid360` extrinsics.
6. Configure FAST-LIO2 source frames and prevent duplicate `odom -> base_link` publication.
7. Build and verify the fixed map, then set `/initialpose`.
8. Run preflight and diagnostics before enabling absolute-coordinate competition actions.
9. Add RGB-D/YOLO target quality, dual-robot communication, and STM32/CAN feedback.
10. Capture a versioned `run_manifest` for every mapping, localization, and match run.

---

## 10. Semantic Perception and Shooting Loop

The detailed physical-system documentation describes an RGB-D camera with YOLO basket detection. The perception adapter has three responsibilities:

1. detect the basket or target in the camera frame;
2. estimate distance and bearing using depth and calibrated transforms;
3. publish a quality-bearing observation with timestamp, confidence, stability, and expiry.

The target observation joins the field pose through the TF chain. A shooting parameter can then be computed from target distance, heading error, basket height, and a calibrated mechanism model. The exact ballistic parameters belong to the robot-specific private profile.

The system treats “command transmitted” and “action completed” as separate events. A shot request enters the execution layer; release or mechanism feedback confirms completion; the task supervisor then advances or recovers. This distinction keeps the task state aligned with physical reality.

---

## 11. Game Supervisor and Dual-Robot Protocol

### 11.1 Lifecycle states

`GameSupervisor` defines:

```text
BOOT -> PREFLIGHT -> WAIT_START -> ACTIVE
                         |          |
                         v          v
                       FAULT      PAUSED
                                    |
                                    v
                                 RECOVERY

ACTIVE / PAUSED -> FINISHED
Any state -> ESTOP by emergency stop
```

The implementation uses `BOOT`, `PREFLIGHT`, `WAIT_START`, `ACTIVE`, `PAUSED`, `RECOVERY`, `FAULT`, `ESTOP`, and `FINISHED`. State transitions are checked by `_require`, so an operator command cannot silently skip a prerequisite.

### 11.2 Ordered pass evidence

The shooter-side task progresses through:

```text
IDLE
  -> RECEIVER_READY
  -> PASS_ARMED
  -> PASS_EXECUTED
  -> RECEIPT_CONFIRMED
```

Each transition has a dedicated method and a reason when the evidence is missing. This is stronger than a fixed delay because it represents the receiver's actual readiness and the completed transfer.

### 11.3 `FireShot` gate

`request_fire_shot` accepts a shot only when all of the following are true:

```text
supervisor is ACTIVE
pose_valid
map_locked
pose_age <= 0.30 s
target_valid
target_age <= 0.30 s
ball_present
mechanism_healthy
teammate_safe
```

The first failed condition becomes the decision reason. The same pattern can be extended with rule phase, shooting zone, target class, release feedback, and possession timers.

### 11.4 Versioned, expiring, idempotent messages

`MessageEnvelope` contains:

```text
protocol_version
message_type
task_id
sender_id
sequence
created_at_ns
expires_at_ns
payload
```

`Deduplicator` validates freshness, keys messages by sender/task/type, rejects expired envelopes, and rejects sequence numbers that repeat or move backward. This supports a robust two-robot workflow under retries and temporary link delays.

The physical communication design described in the portfolio uses UART and ESP-NOW for high-level exchange. The embedded boundary remains explicit: the NUC requests a high-level action, while the STM32/CAN layer owns real-time control and safety.

---

## 12. Research Interpretation

### 12.1 Main contribution

The strongest contribution is an evidence-oriented architecture for competition autonomy. Instead of treating localization, perception, communication, and control as independent streams, the project defines the information each consumer needs:

- every pose has frame, timestamp, covariance, sequence, age, and reason;
- every target has validity, age, and geometric context;
- every task has an ordered evidence path;
- every message has identity, sequence, lifetime, and payload;
- every actuator request has a feedback and cancellation path.

This turns system reliability into an observable property that can be tested at interfaces.

### 12.2 Engineering contributions

1. **Native sensor timing:** `CustomMsg.timebase` and per-point `offset_time` remain available to the LIO front end.
2. **Unique TF ownership:** the bridge owns `odom -> base_link`; the map anchor owns `map -> odom`.
3. **Geometric correctness:** full quaternion normalization, sensor-origin to base-origin conversion, twist transformation, and covariance propagation.
4. **Global/local separation:** fixed-map anchoring changes the field reference without resetting local motion.
5. **Explicit recovery:** localization states, preflight readiness, relocalization requests, pause, recovery, fault, and ESTOP are first-class interfaces.
6. **Deterministic coordination:** ordered pass evidence and versioned, expiring, idempotent messages.
7. **Real-time boundary:** high-level ROS 2 reasoning stays separate from STM32/CAN motor and mechanism safety.
8. **Reproducibility:** templates, launch arguments, diagnostics, tests, and run manifests preserve configuration provenance.

### 12.3 Why this matters for robotics research

Many robotics demonstrations show a successful trajectory. This project focuses on the conditions that make a trajectory trustworthy in a changing arena: timing, frames, quality, action permission, acknowledgement, and recovery. Those conditions form a useful bridge between SLAM research and embodied multi-robot systems.

---

## 13. Verification and Reproducibility

### 13.1 Tests executed from the G-drive workspace

The pure Python contract tests were run from `G:\mid360_ws_test_20260819`:

```text
pytest -q \
  src/mid360_localization_contract/test/test_geometry.py \
  src/mid360_localization_contract/test/test_templates.py \
  src/mid360_localization_contract/test/test_tracking.py \
  src/game_supervisor/test/test_supervisor.py

16 passed in 0.16s
```

The tests cover:

- transform composition and inversion;
- sensor-height removal;
- full-quaternion yaw extraction;
- non-unit and zero quaternion behavior;
- private template placeholders and disabled static TF;
- tracking state transitions and map-lock semantics;
- shot evidence and freshness limits;
- ordered pass evidence;
- terminal ESTOP behavior;
- message expiry and duplicate suppression.

The ROS adapter test imports `rclpy`, so it belongs on the Ubuntu/ROS 2 target environment. The current Windows verification reports the dependency requirement explicitly rather than treating an unavailable ROS runtime as a source failure.

### 13.2 ROS 2 build and launch

On the target Ubuntu 22.04 / ROS 2 Humble machine:

```bash
cd ~/robocon_mid360_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

ros2 launch mid360_localization_contract competition_localization.launch.py

ros2 launch mid360_localization_contract robocon_mid360_bringup.launch.py \
  driver_config_path:=/absolute/private/MID360_config.local.json \
  fast_lio_params_file:=/absolute/private/fast_lio_mid360.local.yaml \
  contract_params_file:=/absolute/private/mid360_competition.local.yaml

ros2 launch game_supervisor robocon_competition.launch.py \
  driver_config_path:=/absolute/private/MID360_config.local.json \
  fast_lio_params_file:=/absolute/private/fast_lio_mid360.local.yaml \
  contract_params_file:=/absolute/private/mid360_competition.local.yaml \
  supervisor_params_file:=/absolute/private/supervisor.local.yaml
```

### 13.3 Evidence ladder

```text
synthetic message tests
    -> ROS graph and topic-type preflight
    -> recorded bag timing and calibration checks
    -> fixed-map replay and relocalization
    -> single-robot bench tests
    -> dual-robot task and communication tests
    -> full field operation with run manifest
```

Each level answers a different question. Unit tests establish mathematical and state-machine invariants. Bag replay establishes timing and geometry behavior. Hardware runs establish installation, networking, mechanism, and competition-rhythm behavior.

---

## 14. Failure Modes and Recovery Design

| Failure signal | Detection | Immediate behavior | Recovery path |
| --- | --- | --- | --- |
| malformed LiDAR packet | point count, timebase, offset checks | `input_valid=false` | inspect driver/network and replay a raw bag |
| stale IMU or LiDAR | arrival age | preflight blocked | restore stream and verify QoS |
| wrong source frame | pose bridge frame contract | reject pose | configure exact FAST-LIO2 frame names |
| zero or invalid quaternion | normalization | reject pose | inspect estimator output and calibration |
| excessive pose jump | optional quality threshold | `pose_valid=false` | tune measured threshold or relocalize |
| map anchor absent | tracking state | `map_locked=false` | accept a verified `/initialpose` |
| tracking lost | pose validity/freshness | absolute actions paused | request relocalization and accept new anchor |
| stale target | target age gate | shot request held | reacquire RGB-D/YOLO target |
| teammate message expired | envelope age | task remains pending | heartbeat, retry, or safe return |
| missing mechanism feedback | supervisor safety snapshot | shot/pass blocked | mechanism recovery or operator intervention |
| process exit | launch `OnProcessExit` handler | coherent stack shutdown | systemd/manual restart after diagnosis |
| emergency stop | supervisor `ESTOP` | terminal safe state | external reset and fresh preflight |

The common pattern is “detect -> explain -> bound -> recover”. The system preserves the reason, so field debugging can follow evidence instead of guessing.

---

## 15. Limitations and Next Research Experiments

The public workspace intentionally leaves several deployment facts open. This is a strength for reproducibility because each open item has a named measurement plan.

### 15.1 Measurements to add

- LiDAR-IMU time offset from calibration bags;
- `base_link -> imu_link` and `imu_link -> lidar_mid360` extrinsics;
- end-to-end pose rate and P95 pose age on the NUC;
- position and yaw error against a field reference;
- map registration overlap and relocalization time;
- packet drop and accepted-sequence statistics under competition networking;
- target detection precision, recall, depth error, and observation stability;
- pass completion time and shot completion rate;
- mechanism timeout, recovery, and ESTOP counts.

### 15.2 Recommended experiments

**A. Timing ablation.** Compare native `CustomMsg` per-point timing against a frame-stamp-only conversion using the same bags. Report motion-distortion residuals and pose drift during fast turns.

**B. Frame-contract ablation.** Inject an incorrect child frame, duplicate TF publisher, and unnormalized quaternion. Measure detection time and downstream action suppression.

**C. Map-anchor robustness.** Start from multiple initial poses, interrupt odometry, request relocalization, and evaluate time to regain `TRACKING`.

**D. Supervisor safety study.** Replay stale pose, stale target, teammate timeout, ball absence, and mechanism faults. Report false accepts, correct blocks, recovery latency, and reason coverage.

**E. Dual-robot protocol stress.** Reorder, duplicate, delay, and expire envelopes. Report deduplication accuracy, task completion, and safe fallback behavior.

**F. Sim-to-real comparison.** Use the same target and action contracts in simulation, bag replay, and hardware. Compare pose freshness, target stability, action latency, and recovery statistics.

### 15.3 Publication framing

A strong systems paper can present the work as:

> An evidence-bearing ROS 2 autonomy contract for rule-constrained dual-robot manipulation, combining native LiDAR timing, canonical frame ownership, fixed-map anchoring, semantic target quality, ordered task evidence, and real-time embedded safety boundaries.

The paper should report the exact rule revision, hardware configuration, bag IDs, calibration versions, software commits, run counts, failure definitions, and confidence intervals. The current templates already provide the fields for this evidence package.

---

## 16. Interview Answer Bank

### 16.1 One-minute answer

I worked on a dual-robot ROBOCON Robot Basketball autonomy system. The central challenge was turning heterogeneous and sometimes stale signals into legal, executable actions. The robots used a Livox MID-360 and its internal IMU, with FAST-LIO2 providing continuous local odometry. I designed a ROS 2 contract around that estimator: a packet guard preserved `CustomMsg` point timing, a pose bridge enforced source frames and produced one canonical `odom -> base_link`, and a map anchor maintained `map -> odom` from a verified initial pose or correction. On top of localization, the physical system used RGB-D/YOLO for basket observation, UART/ESP-NOW for dual-robot messages, and STM32/CAN for real-time chassis and mechanism control. The game supervisor encoded ordered pass evidence and a shot gate requiring fresh pose, map lock, target, ball, mechanism, and teammate safety. My main contribution was the integration boundary and the field-debugging workflow, from sensor timing and calibration to task feedback and recovery.

### 16.2 Three-minute answer

The project is a rule-driven autonomy stack rather than a single algorithm demo. In Robot Basketball, a robot can only make progress when geometry, semantics, rules, communication, and mechanism state agree. I organized those dependencies into contracts.

At the sensing layer, the official Livox driver publishes `CustomMsg` and IMU data. The input guard checks point count, `timebase`, per-point `offset_time`, timestamp monotonicity, finite IMU values, and stream freshness. FAST-LIO2 consumes the native timing and produces local odometry. The pose bridge checks source frame names, normalizes the quaternion, converts the tracking-origin pose to the base origin, preserves covariance and timestamp, and owns the single public `odom -> base_link` transform. The fixed-map anchor combines this local pose with `/initialpose` to compute `map -> odom`. A tracking state machine exposes `UNINITIALIZED`, `TRACKING`, `RELOCALIZING`, and `LOST`, so global actions depend on `map_locked` rather than on an old transform.

The physical system adds RGB-D/YOLO target observations and a dual-robot communication layer. The supervisor moves through preflight, waiting, active, pause, recovery, fault, and finish states. A pass requires receiver-ready, armed, executed, and receipt-confirmed evidence in order. A shot requires active match state, fresh pose and target, map lock, ball presence, mechanism health, and teammate safety. Messages carry task identity, sequence, expiry, and sender identity; duplicates and stale retries are rejected. The high-level computer sends requests, while STM32/CAN keeps motor PID, limits, current protection, timeout, and ESTOP deterministic.

The research lesson is that reliable autonomy is an information-flow problem. Every consumer receives a value plus its frame, age, quality, and reason. That made calibration, replay, field debugging, and recovery much more systematic.

### 16.3 Technical question: Why retain `CustomMsg`?

Because a Livox scan is collected over time. `timebase` identifies the first point and each point's `offset_time` identifies its sampling instant. FAST-LIO2 can use that information for motion compensation. A frame-stamped cloud would hide intra-scan motion and can increase distortion during turning or acceleration.

### 16.4 Technical question: Why create a pose bridge?

The upstream estimator's tracking origin and frame names are implementation details. The rest of the robot needs a stable `odom -> base_link` contract with calibrated origin, normalized quaternion, covariance, freshness, and diagnostics. The bridge also gives the system one TF owner, eliminating transform races.

### 16.5 Technical question: Why separate `map -> odom` from `odom -> base_link`?

Local odometry should remain smooth for control. The fixed map supplies a global field reference. Updating `map -> odom` changes the global alignment while preserving the local trajectory. This separation supports initial pose, map corrections, and relocalization without resetting the estimator.

### 16.6 Technical question: What happens when localization is lost?

The pose bridge marks the stream invalid when the source is stale, malformed, or outside configured quality limits. The map anchor moves to `LOST` or `RELOCALIZING`, clears `map_locked`, and publishes a reason. Preflight and the supervisor hold absolute-coordinate actions. After a fresh valid pose and accepted anchor, the state returns to `TRACKING` and the task can resume.

### 16.7 Technical question: Why does a shot need so many gates?

The shot combines geometry, perception, possession, mechanism, and team safety. A fresh pose gives the field relation; a fresh target gives aim geometry; map lock gives a trusted global frame; ball and mechanism signals indicate physical readiness; teammate safety prevents an unsafe dual-robot action. The conjunction is explicit and each failed condition is reported.

### 16.8 Technical question: What did you personally learn?

I learned to treat interfaces as research objects. A successful algorithm output becomes useful only after its frame, timestamp, covariance, freshness, failure reason, and consumer assumptions are defined. I also learned that real-robot progress comes from short evidence loops: synthetic tests, bag replay, calibration, preflight, bench tests, and field runs with versioned manifests.

---

## 17. Memory Card

### The five-step story

1. **Sense:** MID-360 `CustomMsg` plus IMU, with native point timing preserved.
2. **Estimate:** FAST-LIO2 gives local motion; the bridge gives canonical `odom -> base_link`.
3. **Anchor:** `/initialpose` or verified correction gives `map -> odom`; tracking exposes lock state.
4. **Decide:** RGB-D/YOLO, rule profile, teammate protocol, and supervisor combine into an evidence gate.
5. **Execute and recover:** UART/ESP-NOW requests reach STM32/CAN; feedback, timeout, recovery, and ESTOP close the loop.

### The three equations

```text
T_odom_base = T_odom_tracking * inverse(T_base_tracking)
T_map_odom  = T_map_base * inverse(T_odom_base)
FireShot    = ACTIVE & pose_valid & map_locked & fresh_target
              & ball_present & mechanism_healthy & teammate_safe
```

### The four tracking states

```text
UNINITIALIZED -> TRACKING -> LOST
                       \-> RELOCALIZING -> TRACKING
```

### The four pass states

```text
receiver_ready -> pass_armed -> pass_executed -> receipt_confirmed
```

### The six words for debugging

`frame` · `time` · `freshness` · `quality` · `acknowledgement` · `recovery`

### A compact closing sentence

I connected native LiDAR timing, LiDAR-inertial estimation, fixed-map geometry, semantic target evidence, rule supervision, dual-robot communication, and embedded execution into one inspectable ROS 2 loop that can explain every action and recover from every major quality failure.

---

## 18. Source and Evidence Index

### Direct source

```text
G:\mid360_ws_test_20260819\src\mid360_localization_contract\README.md
G:\mid360_ws_test_20260819\src\mid360_localization_contract\INTERFACE_CONTRACT.md
G:\mid360_ws_test_20260819\src\mid360_localization_contract\config\competition.yaml
G:\mid360_ws_test_20260819\src\mid360_localization_contract\config\fast_lio_mid360.template.yaml
G:\mid360_ws_test_20260819\src\mid360_localization_contract\config\map_metadata.template.yaml
G:\mid360_ws_test_20260819\src\mid360_localization_contract\mid360_localization_contract\input_guard.py
G:\mid360_ws_test_20260819\src\mid360_localization_contract\mid360_localization_contract\pose_bridge.py
G:\mid360_ws_test_20260819\src\mid360_localization_contract\mid360_localization_contract\map_odom_anchor.py
G:\mid360_ws_test_20260819\src\mid360_localization_contract\mid360_localization_contract\tracking.py
G:\mid360_ws_test_20260819\src\mid360_localization_contract\mid360_localization_contract\preflight.py
G:\mid360_ws_test_20260819\src\mid360_localization_contract\test\
G:\mid360_ws_test_20260819\src\game_supervisor\README.md
G:\mid360_ws_test_20260819\src\game_supervisor\config\supervisor.yaml
G:\mid360_ws_test_20260819\src\game_supervisor\config\game_profile.template.yaml
G:\mid360_ws_test_20260819\src\game_supervisor\config\run_manifest.template.yaml
G:\mid360_ws_test_20260819\src\game_supervisor\game_supervisor\supervisor.py
G:\mid360_ws_test_20260819\src\game_supervisor\game_supervisor\protocol.py
G:\mid360_ws_test_20260819\src\game_supervisor\test\
G:\mid360_ws_test_20260819\src\vendor_fast_lio\
G:\mid360_ws_test_20260819\src\vendor_livox_ros_driver2\
```

### Archive and portfolio context

```text
G:\MID360_ROBOCON_C_archive_20260825\MID360_ROBOCON_C_archive_20260825.tar.gz
C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\docs\project-deep-dives\robocon-mid360-autonomy-stack.md
C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\docs\project-deep-dives\robocon-mid360-autonomy-stack-detailed-technical-description-en.md
C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\docs\project-deep-dives\robocon-mid360-autonomy-stack-详尽技术说明.md
C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\docs\project-deep-dives\robocon-mid360-lio-algorithms-zh.md
C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\docs\project-deep-dives\robocon-mid360-lio-learning-guide-zh.md
```

### Background references

1. ABU ROBOCON 2025 Robot Basketball Rulebook, revision 2024-08-14.
2. Xu, W. et al. “FAST-LIO2: Fast Direct LiDAR-inertial Odometry.” *IEEE Transactions on Robotics*, 2022.
3. Shan, T. et al. “LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping.” *IROS*, 2020.
4. Livox ROS Driver 2 message and configuration documentation.
5. ROS 2 Humble, `tf2`, `diagnostic_msgs`, and Nav2 interface documentation.

---

## Final Takeaway

This project is best presented as a systems research contribution grounded in real robot integration. The distinctive idea is the evidence-bearing loop: native sensor timing feeds a quality-aware estimator; a unique TF and fixed-map contract provide usable geometry; target, teammate, and mechanism signals become explicit task evidence; versioned protocols and embedded safety boundaries turn decisions into recoverable physical actions. The result is a clear path from LiDAR packet to competition behavior, with each assumption visible and each recovery step testable.
