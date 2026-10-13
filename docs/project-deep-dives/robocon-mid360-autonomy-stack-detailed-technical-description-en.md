# ROBOCON 2025 Robot Basketball Autonomy System

## A dual-robot competition system built with Livox MID-360, FAST-LIO2, RGB-D/YOLO, ROS 2, and STM32 control

> This document explains the system as an engineering and research project. It focuses on the design reasoning, hands-on implementation, cross-disciplinary integration, field debugging, and the path from sensor data to a verified robot action.

---

## 1. Project Overview

### 1.1 One-sentence description

The project is a dual-robot autonomy system for the ABU ROBOCON 2025 Robot Basketball task. Each robot uses a Livox MID-360 and its built-in IMU for LiDAR--inertial localization, an RGB-D camera with a YOLO detector for basket observation and depth measurement, ROS 2 for state and task coordination, and an STM32/CAN execution layer for the chassis and mechanisms. UART and ESP-NOW connect the high-level computer, the two robots, and the real-time controllers.

The complete operational chain is:

```text
competition rules and strategy
        -> task state and action conditions
        -> MID-360/IMU measurements
        -> FAST-LIO2 local motion estimate
        -> fixed-map field pose
        -> RGB-D/YOLO basket observation
        -> distance, heading, and shooting-parameter calculation
        -> dual-robot coordination and acknowledgement
        -> UART/ESP-NOW command transport
        -> STM32/CAN chassis and mechanism control
        -> feedback, diagnosis, recovery, and the next task
```

### 1.2 The operational problem

Robot Basketball combines motion, perception, localization, ball handling, passing, receiving, aiming, and shooting in a time-limited match. A useful system must keep these activities coordinated while the robot accelerates, turns, vibrates, changes its viewpoint, communicates with a teammate, and interacts with mechanisms whose response time varies with load and battery voltage.

The project addresses one concrete systems question:

> How can a pair of competition robots maintain a trustworthy geometric state and a fresh semantic target observation, then convert them into rule-compliant actions that are acknowledged by the real mechanisms?

The question has three coupled layers:

1. **State estimation:** where the robot is, how its motion evolves, and whether the field coordinate is usable;
2. **Task reasoning:** whether passing, receiving, or shooting is currently permitted and which robot should act;
3. **Execution:** whether the command reaches the chassis and mechanisms, whether the action is complete, and how the system returns to a safe state after an interruption.

### 1.3 My responsibility

I led the LiDAR and perception work and owned the integration boundary between localization, navigation, dual-robot communication, and embedded control. My work covered physical sensor installation, timestamp and extrinsic calibration, ROS 2 launch and topic integration, field mapping, FAST-LIO2 adaptation, RGB-D/YOLO tuning, shooting-parameter computation, UART and ESP-NOW communication, STM32 handoff, chassis tests, mechanism tests, and full-system field debugging.

The work required continuous movement between disciplines:

- geometry and state estimation for the LiDAR--IMU chain;
- computer vision and depth statistics for basket observations;
- rigid-body transforms and ballistic modeling for aiming;
- networking and protocol design for the two-robot workflow;
- embedded control and motor feedback for the physical mechanisms;
- competition tactics and timing for the task supervisor.

The integrated system was migrated to the physical competition robots and used in ROBOCON field work. The team received a national second prize in the 24th China University Robot Competition ROBOCON Robot Basketball event and a national second prize in the Robot Basketball Shooting event.

---

## 2. Competition Rules as Engineering Requirements

### 2.1 Why the rulebook shapes the architecture

The ABU ROBOCON 2025 Robot Basketball rulebook provides measurable conditions that directly affect sensing, planning, communication, and actuator design. The project stores the rule revision, source URL, and file hash together with the simulation and competition profiles. This keeps the field model, timers, and task logic tied to one rule version.

| Item | Rule value | Engineering consequence |
| --- | ---: | --- |
| Playing field | 15.0 m x 8.0 m | Map scale, navigation boundary, and initial pose share one geometry |
| Fence | 0.10 m high, 0.05 m wide | Boundary geometry appears in the map and local motion checks |
| Backboard | 1.80 m x 1.05 m | The visual target has a structured physical model |
| Basket height | 2.43 m | Ballistic calculation uses a fixed target height |
| Inner basket diameter | 0.450--0.459 m | Target-center estimation and mechanism repeatability matter |
| Preliminary match | 120 s | Match clock and terminal state are explicit |
| Knockout match | 160 s | Match phase is a separate configuration profile |
| Possession | 20 s | Ball-control and possession timers bound the task sequence |
| Advance | 8 s | The transition to the offensive side has its own deadline |
| Minimum pass distance | 1.0 m | Two-robot position sharing is part of pass preparation |
| Dribble hand height | at least 0.70 m | Dribble evidence comes from mechanism or physical feedback |
| Ordinary scores | 2 and 3 points | Shooting zone and shot type enter the task state |
| Dunk score | 7 points | A dunk requires additional spatial and motion evidence |

The rule text gives a qualitative description of the three-point boundary. The implementation therefore keeps the numeric field geometry and the separately calibrated zone geometry as distinct configuration items. This design makes the origin of every threshold explicit.

### 2.2 Plain-language meaning of rule-driven design

The phrase “rule-driven design” means something simple in this project: first determine what the robot must do in the match, then turn each requirement into a timer, a position check, a sensor condition, an acknowledgement, or a stop path.

- The 8-second advance requirement becomes a dedicated timer in the task supervisor.
- The 1-meter pass requirement becomes a check on the two robots' synchronized positions.
- Ball possession becomes a state with explicit start, expiry, transfer, and confirmation events.
- A shot becomes a combination of field pose, basket observation, ball presence, mechanism health, and teammate safety.
- A referee stop becomes a pause, cancellation, and restart path in the supervisor.

In practical terms, the project translates the printed rulebook into robot actions that sensors can observe, software can decide, actuators can execute, and the team can stop safely.

### 2.3 Rule logic and physical evidence

The software separates three decisions:

1. **Permission:** the task supervisor checks whether the current state permits an action;
2. **Execution:** the embedded layer performs the action and reports its state;
3. **Competition meaning:** the rule logic interprets the physical evidence as a legal pass, dribble, shot, or score.

For example, sending a pass command starts an execution attempt. A receiver acknowledgement and ball-arrival signal move the task to confirmed receipt. A shot command starts a mechanism action; release feedback and match observation determine the shot result.

---

## 3. Design Principles

### 3.1 Layered implementation with a closed feedback loop

The system is divided into layers with explicit state feedback:

```text
sensing       MID-360, internal IMU, RGB-D camera, mechanism sensors
estimation    FAST-LIO2, local odometry, fixed-map localization
perception    YOLO detection, depth measurement, target quality gate
task          match clock, ball state, roles, pass and shot sequence
execution     ROS 2 commands, UART/ESP-NOW, STM32, CAN, motor feedback
safety        freshness, map lock, teammate safety, timeout, cancel, ESTOP
```

The purpose of the layers is practical: when a signal degrades, the effect is visible and bounded. A stale pose blocks field-relative actions, a stale target blocks shooting, and a missing mechanism acknowledgement prevents the next task from starting.

### 3.2 Geometric localization and semantic perception have different jobs

The LiDAR--IMU chain answers “how is the robot moving in the field coordinate system?” The RGB-D/YOLO chain answers “where is the basket, how far away is it, and is the observation stable?”

This division is deliberate:

- LiDAR geometry supplies motion continuity under lighting and texture changes;
- RGB-D/YOLO supplies object identity and target distance;
- the task layer consumes quality-bearing interfaces instead of raw sensor streams.

The separation also creates a clear debugging path. A wrong basket distance points to the target observation or transform chain; a discontinuous field pose points to time, geometry, or localization; an unexecuted shot points to the supervisor or embedded feedback boundary.

### 3.3 High-level autonomy and real-time control are separate

The Linux computer and ROS 2 handle maps, perception, task sequencing, and communication. The STM32 handles motor loops, current limits, mechanism timing, limit switches, watchdogs, and emergency stop. The high-level side sends semantic commands such as `PreparePass`, `PrepareShot`, and `FireShot`; the embedded side converts them into deterministic actuator control and returns status.

This division matches the computational and timing requirements of the two sides. A map matcher and a detector benefit from a general-purpose operating system; a motor current loop and an emergency stop benefit from a bounded microcontroller path.

### 3.4 Every action has a state and feedback

The common action pattern is:

```text
request -> precondition check -> command transport -> acknowledgement
        -> execution feedback -> state transition -> timeout or cancellation
```

This pattern applies to navigation, receiving, passing, collecting, preparing a shot, and firing. It prevents a function return or a published message from being mistaken for physical completion.

### 3.5 Simulation, interfaces, and physical deployment form one development path

The workflow starts with message contracts, transforms, state transitions, and rule timing. It then adds sensor replay, field mapping, mechanism bench tests, single-robot motion, dual-robot coordination, and complete competition sequences. Each stage uses the same action names and quality fields, so an issue discovered in a later stage can be traced back through the same interfaces.

---

## 4. System Architecture

### 4.1 Hardware

| Subsystem | Components and interfaces | Role |
| --- | --- | --- |
| 3-D sensing | Livox MID-360 | Field geometry, obstacles, and scan observations |
| Inertial sensing | MID-360 built-in IMU | High-rate angular velocity and acceleration |
| Semantic sensing | RGB-D camera | Basket detection, depth measurement, target confirmation |
| High-level computer | Linux computer running ROS 2 | Localization, perception, task coordination, logging |
| Inter-robot link | ESP-NOW and ROS 2 messages | Position sharing, roles, acknowledgements |
| Computer-to-MCU link | UART | High-level commands and shooting parameters |
| Embedded controller | STM32 | Chassis, dribble, collection, delivery, and shooting mechanisms |
| Motor bus | CAN | Motor feedback and real-time actuator control |

### 4.2 Software data flow

```text
MID-360 CustomMsg + internal IMU
             |
             v
     input validation and timestamp checks
             |
             v
          FAST-LIO2
             |
             +--> local odometry -------------------+
             |                                      |
             v                                      v
     fixed-map localization                  pose quality/status
             |                                      |
             +--> field-frame pose ----------------+
                                                    |
RGB-D frames -> YOLO -> depth -> target gate ------+
                                                    |
                      pose + target + rule state  |
                                                    v
                      game supervisor / task logic
                         |          |          |
                         |          |          +--> safety, cancel, recovery
                         |          v
                         |      dual-robot protocol
                         v
                    pose and shooting command bridge
                         |
                    UART / ESP-NOW
                         |
                         v
                     STM32 execution
                         |
                    CAN / mechanism feedback
                         |
                         +----------> supervisor state update
```

### 4.3 Coordinate-frame ownership

The external transform chain is:

```text
map -> odom -> base_link -> imu_link -> lidar_mid360
```

- `map -> odom` expresses the fixed-field anchor relative to local odometry;
- `odom -> base_link` is published by one pose adapter;
- `base_link -> imu_link` and `imu_link -> lidar_mid360` encode installation geometry;
- FAST-LIO2 private frames stay behind the adapter boundary.

The single-owner rule avoids transform races and pose jumps. All orientations are normalized quaternions, with yaw computed from the complete quaternion.

---

## 5. Hands-on Engineering and Field Integration

The architecture explains the system structure; this section explains how I built and integrated it on a competition robot.

### 5.1 Sensor installation, wiring, and initial checks

I installed the MID-360, the RGB-D camera, and the computing unit on the physical robot, then checked power, Ethernet, USB, brackets, cable routing, and emergency-stop access. The installation checklist covered:

- unobstructed LiDAR and camera fields of view;
- rigid brackets that preserve the sensor geometry during acceleration and braking;
- consistent sensor-axis and `base_link` conventions;
- separation of data cables from high-current motor wiring;
- a known power-up and node-start sequence;
- a mechanically reachable emergency-stop path.

The first runtime check was deliberately small: start the official Livox driver, inspect the `CustomMsg` stream, inspect the internal IMU, verify message rates and timestamps, and query ROS 2 topic types, QoS, and publisher counts. This gives the localization and perception work a known-good input boundary.

### 5.2 Timestamp and extrinsic calibration

LiDAR, IMU, camera, and chassis each have a coordinate frame and a clock relationship. I first verified the spatial axes with controlled motion:

```text
forward translation -> positive forward displacement
side translation    -> positive lateral displacement
in-place rotation   -> correct yaw direction and sign
```

I then aligned the camera target direction with the field map and checked the LiDAR--IMU installation transform. Timestamp alignment used simultaneous recordings of LiDAR, IMU, camera, and control feedback. Motion changes were compared across streams, and the resulting offset was validated again during physical motion.

The calibration values are treated as shared system parameters because they affect scan deskewing, map registration, target coordinates, and shooting direction at the same time.

### 5.3 ROS 2 launch, topic, and TF debugging

During field integration I used a fixed diagnostic order:

1. `ros2 node list` to confirm the driver, FAST-LIO2, map, perception, supervisor, and bridges;
2. `ros2 topic list` to confirm the end-to-end names;
3. `ros2 topic info -v` to check message types, QoS, and publisher/subscriber counts;
4. `ros2 topic hz` to check LiDAR, IMU, pose, target, and feedback rates;
5. TF inspection to verify a single owner for each transform;
6. timestamp and frame inspection to catch stale data or mixed clocks;
7. log correlation across ROS 2, UART, ESP-NOW, STM32, and CAN.

This order turns a vague symptom into a bounded hypothesis. If the robot reaches a physical location while the shot remains withheld, I check target validity, pose age, map lock, command contents, and mechanism feedback in that order before changing navigation parameters.

### 5.4 Field mapping and localization checks

I built and reviewed the field map with the basket structures, fences, boundaries, and initial robot pose represented in one coordinate reference. During mapping and relocalization I checked:

- scan shape during acceleration and turns;
- the geometric consistency of fences and basket structures;
- map scale against the physical 15 m x 8 m field;
- restart and initial-pose repeatability;
- compatibility between continuous local odometry and map anchoring.

The map version, sensor installation state, and field layout were recorded together. This makes a change in a bracket or field structure visible in the deployment record.

### 5.5 Vision, depth, and shooting-parameter integration

The visual integration path was built incrementally:

```text
single-frame YOLO detection
  -> depth statistics inside the target region
  -> temporal stability
  -> target transform and robot pose
  -> distance and heading error
  -> calibrated shooting parameter
  -> UART command to STM32
  -> mechanism feedback
```

Known-distance basket observations checked depth scale. Controlled robot rotations checked target bearing. Static shots at several distances produced calibration data for the numerical ballistic model. The same interface was then exercised while the robot moved.

### 5.6 Dual-robot and mechanism integration

I integrated the two robots in stages:

1. publish receiver position and `receiver_ready`;
2. check relative distance and message age on the passing robot;
3. send `pass_armed` and verify one-time processing;
4. execute the pass and wait for the receiver acknowledgement;
5. connect the collection mechanism and verify ball-arrival feedback;
6. connect the confirmed receipt state to the shooting sequence.

The debugging record included ESP-NOW messages, ROS 2 messages, UART frames, STM32 status words, and CAN feedback. This allowed each failure to be assigned to transport, protocol, command format, or physical execution.

### 5.7 Field-fault workflow

The practical debugging loop was:

```text
symptom -> testable hypothesis -> smallest measurement
        -> targeted change -> same-scene re-test
```

| Field symptom | First measurements | Engineering action |
| --- | --- | --- |
| Pose jump | timestamp, TF owner, bracket motion, scan shape | correct frame, timing, extrinsic, or map anchor |
| Robot arrives but shot is withheld | target gate, pose age, map lock, ball and mechanism state | restore the missing observation or feedback path |
| Shooting distance varies | depth stability, target transform, ballistic calibration, battery state | recalibrate distance-to-parameter mapping |
| Repeated dual-robot action | sequence, task identifier, ACK history | enforce idempotent task processing |
| Pass sequence pauses | receiver acknowledgement, ball arrival, limits, timeout | connect the physical completion signal and recovery path |
| Chassis response is absent | UART frame, MCU state, CAN and motor enable | trace the command from ROS 2 to the motor feedback |
| Wireless response slows | heartbeat, message age, network load | pause dependent actions and re-establish the session |

### 5.8 Cross-disciplinary collaboration

| Discipline | Shared engineering question | Interface produced |
| --- | --- | --- |
| Mechanical design | Where can the sensors see, and how does the mechanism move? | installation transform, clearance, limits |
| Embedded control | What is the exact frame, timeout, and feedback state? | UART command and STM32/CAN status |
| Computer vision | When is a basket observation usable? | `target_observation` |
| Localization | Which pose frame and timestamp drive field actions? | `localization_odometry`, `pose_status` |
| Networking | How are roles, retries, and duplicates handled? | task protocol with sequence and ACK |
| Competition strategy | Which action follows and what is the fallback? | supervisor states and task transitions |

The key cross-disciplinary skill was converting expert knowledge from each area into shared fields, units, timestamps, and state transitions.

### 5.9 How the engineering capability appears in the deliverables

| Capability | Hands-on work | Resulting deliverable |
| --- | --- | --- |
| Physical integration | installed MID-360, RGB-D camera, computer, power, and data cables | a sensor layout that operates on the physical robot |
| Calibration and measurement | checked axes, LiDAR--IMU/camera extrinsics, timing, and field scale | reusable calibration parameters, maps, and run records |
| Algorithm deployment | integrated FAST-LIO2 and preserved per-point timing | a pose interface that supports competition actions |
| ROS 2 integration | connected launches, topics, QoS, TF, and supervisor states | an end-to-end message path from sensing to tasks |
| Mathematical modeling | computed distance and heading error and calibrated shooting parameters | distance-dependent actuator commands |
| Protocol engineering | designed roles, task IDs, sequences, ACK, expiry, and cancellation | traceable and recoverable pass/receive coordination |
| Embedded integration | defined UART commands and debugged STM32/CAN motor and mechanism feedback | a clear boundary between autonomy and real-time control |
| Field debugging | correlated logs, topics, transforms, serial frames, and CAN feedback | a repeatable symptom-to-fix workflow |
| Team integration | aligned mechanical, electrical, vision, localization, and strategy interfaces | a robot system that the team can maintain at the venue |

The deliverables show both sides of the work: I understand the estimator and perception principles, and I can install, connect, calibrate, debug, and validate them on a real competition robot.

---

## 6. MID-360, Internal IMU, and FAST-LIO2

### 6.1 Why LiDAR--inertial estimation fits the robot

The robot accelerates, turns, brakes, and vibrates while the basket and ball mechanisms change the body dynamics. Wheel odometry alone is sensitive to slip; vision alone is sensitive to occlusion, lighting, and texture. MID-360 supplies spatial geometry, while the built-in IMU supplies high-rate motion propagation. FAST-LIO2 combines these signals into a continuous local estimate that can feed mapping, navigation, and shooting geometry.

The project contribution is the engineering path around the estimator: correct input timing, frame ownership, map anchoring, quality state, and downstream action contracts.

### 6.2 `CustomMsg` and per-point time

The Livox driver publishes `livox_ros_driver2/msg/CustomMsg` with a frame-level `timebase` and per-point `offset_time`. For point `i`:

```text
t_i = timebase + offset_time_i
```

The scan deskewing stage uses `t_i` to associate each point with the interpolated body motion. This matters during turns and acceleration because a scan represents a time interval rather than one instantaneous measurement.

The input guard checks finite coordinates, point count, offset span, timebase validity, LiDAR/IMU freshness, and sequence continuity.

### 6.3 Propagation and geometric correction

The state can be expressed as:

```text
x = (R, p, v, b_g, b_a)
```

The IMU propagates orientation, position, velocity, and bias state at high rate. LiDAR geometry provides scan-to-map residuals. A simplified point-to-plane residual is:

```text
r_i = n_i^T (R p_i + p - q_i)
```

where `p_i` is a point in the sensor frame, `q_i` is a corresponding map point, and `n_i` is a local surface normal. The estimator uses these residuals to correct the propagated state.

The practical conditions are as important as the equations: timebase, point offsets, IMU ordering, sensor extrinsics, frame names, and queue behavior must remain consistent.

### 6.4 Local odometry and fixed-map anchoring

The localization stack uses two related outputs:

- **local odometry** for continuous short-horizon motion;
- **field-frame localization** for basket coordinates, navigation targets, and competition actions.

The field pose is composed as:

```text
T_map_base = T_map_odom * T_odom_base
```

FAST-LIO2 supplies the continuous `T_odom_base`. The map localizer estimates `T_map_odom` from the initial pose and scan-to-map alignment. This keeps global corrections separate from local motion continuity.

### 6.5 Localization quality state

The tracking state is explicit:

```text
UNINITIALIZED -> TRACKING -> RELOCALIZING -> LOST
                       ^             |
                       +-------------+
```

The quality interface reports input freshness, point statistics, source and accepted sequence numbers, pose age, map lock, anchor source, map fitness, scan/map overlap, degeneracy, dynamic-point ratio, resource watermarks, and correction-rejection reasons. A consumer can therefore decide whether a pose is suitable for a field-relative action instead of reading coordinates alone.

---

## 7. RGB-D and YOLO Basket Perception

### 7.1 From detection to a usable target

The perception chain is:

```text
RGB-D image
  -> YOLO basket/backboard detection
  -> depth samples inside the detection region
  -> robust depth statistic and temporal smoothing
  -> target distance and bearing
  -> confidence, stability, and freshness gate
  -> target observation interface
```

The target observation contains confidence, distance, stability, timestamp, target type, and evidence metadata.

### 7.2 Target quality gate

The default target profile uses:

| Condition | Default value | Purpose |
| --- | ---: | --- |
| confidence | `>= 0.70` | suppress weak detections |
| distance | `1.0--10.0 m` | keep depth within the operational range |
| observation age | `<= 0.50 s` | keep the target synchronized with motion |
| temporal stability | required | prefer a stable target over a single frame |

The gate publishes a boolean, a JSON status, and diagnostics with a specific reason such as stale observation, low confidence, out-of-range distance, or unstable target.

### 7.3 Why freshness is a physical quantity

As the robot moves, the basket distance and viewing angle change. The target timestamp therefore participates in the control decision together with pose age. A current target with an old pose, or a current pose with an old target, represents a mismatched geometry. The supervisor waits for a coherent pair.

---

## 8. Pose-to-Shooting Interface

### 8.1 Geometric calculation

Let the robot pose be:

```text
robot = (x_r, y_r, psi_r)
```

and the basket center be:

```text
target = (x_t, y_t, z_t)
```

The planar distance, target bearing, and heading error are:

```text
d_xy = sqrt((x_t - x_r)^2 + (y_t - y_r)^2)
psi_t = atan2(y_t - y_r, x_t - x_r)
e_psi = wrap(psi_t - psi_r)
```

The pose command bridge owns this calculation so that visual perception, navigation, and the embedded controller use the same definition.

### 8.2 Ballistic model and calibration

The shooting module uses a numerical model containing gravity, aerodynamic drag, and Magnus force. In abstract form:

```text
m dv/dt = F_gravity + F_drag + F_magnus
dp/dt = v
```

The model provides an interpretable starting point. Offline measurements then calibrate the effects of wheel friction, release timing, mechanism compression, battery voltage, and ball variation. The resulting parameter is sent with the current distance and heading context to the STM32.

### 8.3 Why model-based calibration is useful

A fixed speed table describes a few tested positions. A calibrated model provides a continuous relation between measured distance and actuator setting, while field tests correct the model for the actual mechanism. The resulting chain is:

```text
pose and target -> distance/heading -> physical model
                -> calibrated parameter -> STM32 execution
```

---

## 9. Dual-Robot Coordination

### 9.1 Roles and task ownership

The two robots have complementary responsibilities:

- the ball-handling robot obtains and transports the ball, satisfies the advance condition, and executes the pass;
- the receiving/shooting robot prepares a legal receiving position, confirms receipt, computes the shot, and operates the shooting mechanism.

Role state is tied to task phase, position, communication freshness, and physical feedback.

### 9.2 Message fields

| Field | Purpose |
| --- | --- |
| `protocol_version` | identify the message schema |
| `task_id` | bind messages to one pass or shot |
| `sender_id`, `receiver_id` | identify the participating robots |
| `sequence` | detect duplicates and stale order |
| `stamp_ns` | evaluate message age |
| `command` | request a task transition |
| `ack` | confirm receipt and processing |
| `status` | report running, complete, cancelled, or faulted |
| `pose` / `relative_pose` | support distance and safety checks |
| `reason` | explain rejection or recovery |

### 9.3 Acknowledgement, idempotence, and expiry

The protocol uses:

1. acknowledgements for reception and processing;
2. idempotent handling for repeated `task_id + sequence` pairs;
3. message expiry for positions and action permissions.

When the wireless link slows, the supervisor pauses actions that require a teammate and re-establishes the session before continuing.

### 9.4 Transport roles

- ROS 2 carries high-level state, diagnostics, and task messages;
- ESP-NOW carries lightweight inter-robot position and coordination signals;
- UART carries command parameters and mechanism requests to the STM32;
- CAN carries motor control and feedback inside the embedded subsystem.

---

## 10. STM32 and Actuator Boundary

### 10.1 Semantic high-level commands

The computer sends versioned commands such as:

```text
prepare_receive
prepare_pass
execute_pass
collect_ball
prepare_shot
fire_shot
abort_task
emergency_stop
```

Pose-derived commands include position, yaw, distance, heading error, and timestamp. The STM32 translates these commands into motor and mechanism behavior according to current limits, mechanism state, and feedback.

### 10.2 Feedback defines completion

The system uses physical feedback for ball presence, mechanism position, release, current, limits, and CAN health. This makes completion time adaptive to battery voltage, load, and mechanical variation. The supervisor advances only after the corresponding feedback arrives.

### 10.3 Embedded safety functions

The STM32 retains:

- motor PID and current limits;
- limit-switch and mechanism interlocks;
- CAN node monitoring;
- command timeout and watchdog behavior;
- emergency-stop input;
- over-current, stall, and thermal protection.

The high-level autonomy stack therefore operates above a deterministic safety boundary.

---

## 11. Competition Supervisor and Finite-State Machines

### 11.1 System state

```text
BOOT -> PREFLIGHT -> WAIT_START -> ACTIVE
                                  |
                                  +-> PAUSED -> RECOVERY
                                  +-> FAULT
                                  +-> ESTOP
                                  +-> FINISHED
```

| State | Meaning |
| --- | --- |
| `BOOT` | processes and hardware are starting |
| `PREFLIGHT` | sensors, transforms, map, communication, and mechanisms are checked |
| `WAIT_START` | the system is ready for the match start signal |
| `ACTIVE` | match tasks may execute |
| `PAUSED` | new tasks are held while the robot remains safe |
| `RECOVERY` | relocalization, reconnection, or action cancellation is running |
| `FAULT` | a condition requires operator handling |
| `ESTOP` | hazardous motion is latched off |
| `FINISHED` | the match clock or referee has ended the run |

### 11.2 Dual-robot task state

```text
IDLE
  -> RECEIVER_READY
  -> PASS_ARMED
  -> PASS_EXECUTED
  -> RECEIPT_CONFIRMED
  -> PREPARING_SHOT
  -> SHOT_EXECUTED
```

Each transition consumes evidence from the other robot, the mechanisms, or the field state. The sequence is therefore auditable from logs.

### 11.3 `preflight_ready`

The preflight signal becomes true when the following conditions are fresh and mutually consistent:

```text
LiDAR input
AND IMU input
AND FAST-LIO2 tracking
AND unique, valid TF chain
AND map lock and initial pose
AND target/task interface
AND dual-robot communication
AND UART/CAN/mechanism health
```

This turns the start-of-match check into an explicit system contract.

---

## 12. Rule, Observation, and Action Matrix

| Action | Required observations | Rule or task condition | Resulting command |
| --- | --- | --- | --- |
| enter offensive side | field pose, ball state, dribble/pass feedback | 8-second advance window | continue or recover |
| prepare pass | teammate pose, safety, fresh link | distance at least 1 m | `PreparePass` |
| execute pass | launcher state, receiver ACK | receiver fully on offensive side | `ExecutePass` |
| confirm receipt | ball presence, mechanism state | ball-control transfer evidence | `CollectBall` / receipt confirmation |
| prepare shot | pose, target, distance, heading, map lock | legal offensive position | `PrepareShot` |
| fire shot | target stability, ball, mechanism, teammate safety | legal possession and phase | `FireShot` |
| recover | quality state, timeout, or cancellation | safety priority | `AbortTask` / recovery routine |

The matrix is the bridge between the paper rulebook and the robot implementation.

---

## 13. Complete Task Timeline

### 13.1 Preflight

```text
BOOT
  -> start MID-360 driver and IMU
  -> validate CustomMsg timing and point structure
  -> start FAST-LIO2
  -> establish local odometry
  -> load versioned field map
  -> set initial pose
  -> verify map -> odom -> base_link
  -> check RGB-D/YOLO, ESP-NOW, UART, CAN, and mechanisms
  -> preflight_ready = true
```

### 13.2 Pass and receive

```text
ball-handling robot: ball confirmed
  -> request receiver_ready
shooting robot: pose and teammate safety checked
  -> receiver_ready
ball-handling robot: distance and advance timer checked
  -> pass_armed
  -> execute_pass
shooting robot: ball arrival and mechanism feedback
  -> receipt_confirmed
```

### 13.3 Aim and shoot

```text
RGB-D/YOLO detects basket
  -> depth and temporal stability
FAST-LIO2/map supplies field pose
  -> distance and heading error
  -> calibrated ballistic parameter
  -> UART command to STM32
  -> mechanism preparation feedback
  -> supervisor safety checks
  -> FireShot
  -> release feedback and result recording
```

### 13.4 Recovery

```text
stale pose       -> pause -> relocalize or operator confirmation
stale target     -> wait for a fresh observation
expired link     -> cancel teammate-dependent action
mechanism timeout -> recovery -> safe position or operator handling
CAN fault        -> fault/ESTOP -> inhibit hazardous motion
```

---

## 14. Pose and Target Quality in Action Decisions

### 14.1 Pose gate

A field-relative command is permitted when:

```text
pose_available
AND pose_valid
AND map_locked
AND 0 <= pose_age_sec <= max_pose_age_sec
```

The bridge reports explicit reasons such as unavailable, invalid, unlocked, or stale.

### 14.2 Shooting interlock

```text
FireShotEnabled =
  active_stage
  AND pose_valid
  AND map_locked
  AND pose_fresh
  AND target_valid
  AND target_fresh
  AND ball_present
  AND mechanism_healthy
  AND teammate_safe
```

This expression gives the team a direct explanation for every accepted or withheld shot.

---

## 15. Simulation-to-Real Migration

### 15.1 Interface-first simulation

The simulation profile checks message fields, units, transforms, state transitions, rule timing, and sensor-flow assumptions. The same topic and action names are used later in replay, bench tests, and physical deployment.

### 15.2 Physical differences that guide testing

The real robot adds scan density and network traffic from the MID-360, actual IMU/LiDAR timing, bracket vibration, RGB-D reflections, wireless interference, mechanism latency, battery variation, and field personnel. These factors define the field test matrix.

### 15.3 Migration sequence

```text
1. sensor input and power/network check
2. FAST-LIO2 local pose
3. field mapping and map freeze
4. initial pose and map anchoring
5. low-speed pose-to-target geometry
6. RGB-D/YOLO gate
7. single-robot UART and STM32 test
8. dual-robot position, ACK, expiry, and cancel test
9. motion before loading a ball
10. collection, pass, and receipt
11. static shooting calibration
12. moving and complete competition sequence
```

### 15.4 Deployment manifest

Each field run is associated with:

```text
rulebook revision
map version
extrinsic version
software commit
FAST-LIO2 profile
camera/YOLO profile
communication profile
STM32 firmware version
robot identifier
run timestamp
```

This record makes changes in the environment and software traceable across field sessions.

---

## 16. Engineering Repository Structure

The public engineering repository reorganizes the system around readable ROS 2 packages, interfaces, configuration profiles, simulation assets, and tests.

| Package | Responsibility |
| --- | --- |
| `vendor_livox_ros_driver2` | Livox driver and `CustomMsg` input boundary |
| `vendor_fast_lio` | FAST-LIO2 estimator and profiles |
| `mid360_localization_contract` | input checks, TF ownership, pose/status/preflight contract |
| `mid360_map_localizer` | initial pose, map anchoring, scan-to-map, relocalization |
| `mid360_map_tools` | point-cloud processing, mapping, map metadata |
| `camera_yolo_adapter` | camera and YOLO bridge |
| `perception_adapter` | confidence, depth, stability, and freshness gate |
| `pose_command_bridge` | pose validity, distance/heading calculation, command serialization |
| `game_supervisor` | match state, dual-robot task state, safety interlocks |
| `mid360_simulation` | field, robot, sensor pattern, and rule profile |

### 16.1 Core ROS 2 interfaces

| Topic | Type | Role |
| --- | --- | --- |
| `/livox/lidar` | `livox_ros_driver2/msg/CustomMsg` | timed raw LiDAR input |
| `/livox/imu` | `sensor_msgs/msg/Imu` | inertial input |
| `/Odometry` | `nav_msgs/msg/Odometry` | selected FAST-LIO2 private output |
| `/mid360/local_odometry` | `nav_msgs/msg/Odometry` | normalized `odom -> base_link` |
| `/mid360/localization_odometry` | `nav_msgs/msg/Odometry` | field-frame pose |
| `/mid360/pose_valid` | `std_msgs/msg/Bool` | pose validity |
| `/mid360/input_valid` | `std_msgs/msg/Bool` | LiDAR/IMU structure and freshness |
| `/mid360/map_locked` | `std_msgs/msg/Bool` | map-anchor state |
| `/mid360/pose_status` | `std_msgs/msg/String` | versioned quality JSON |
| `/mid360/preflight_ready` | `std_msgs/msg/Bool` | start permission |

### 16.2 Quality JSON

`pose_status` carries tracking state, map lock, pose age, anchor source, and a quality object in one versioned message. Diagnostic consumers therefore read a shared state instead of reconstructing health from unrelated topics.

---

## 17. Source-Reading Route

Read the project through the data flow:

```text
rule configuration
  -> input guard
  -> FAST-LIO2 adapter
  -> TF and pose contract
  -> fixed-map localization
  -> visual target gate
  -> pose-to-shot bridge
  -> game supervisor
  -> dual-robot protocol
  -> STM32/CAN boundary
```

The most informative source files are:

- `abu_robocon_2025_rulebook.yaml` and `rulebook_2025.py` for competition timing and geometry;
- `INTERFACE_CONTRACT.md`, `input_validation.py`, `tracking.py`, and `pose_bridge.py` for sensor and pose semantics;
- `target_gate.py` for target freshness and confidence;
- `core.py` in the pose command bridge for quaternion yaw and distance/heading calculation;
- `supervisor.py` for task states and safety decisions;
- simulation worlds, profiles, and tests for field geometry and replayable behavior.

---

## 18. Validation Methodology

### 18.1 Input and transform tests

Synthetic messages exercise finite-point checks, timestamp monotonicity, point-count thresholds, IMU freshness, quaternion normalization, yaw calculation, message versions, units, and frame names.

### 18.2 Localization replay

Recorded or replayed streams check LiDAR/IMU timing, FAST-LIO2 continuity, map loading, initial pose, map anchoring, tracking transitions, pose age, drop counters, and matching quality.

### 18.3 Vision tests

The target gate is tested with fresh high-confidence observations, low-confidence detections, out-of-range depth, stale timestamps, invalid depth, and temporally unstable boxes. Every result maps to a readable reason.

### 18.4 Supervisor and communication tests

| Test | Expected behavior |
| --- | --- |
| normal match start | `BOOT -> PREFLIGHT -> WAIT_START -> ACTIVE` |
| stale pose | recovery and inhibition of field-relative actions |
| stale target | shot request waits for a new observation |
| duplicate message | one task execution |
| out-of-order message | old sequence ignored |
| disconnected teammate | dependent action cancelled safely |
| UART timeout | MCU safety behavior and supervisor fault record |
| CAN loss | hazardous action inhibited |
| emergency stop | `ESTOP` latched |
| match timer expiry | `FINISHED` |

### 18.5 Physical metrics

Field runs record LiDAR/IMU rate and drops, scan timing, local pose continuity, map-lock transitions, relocalization time, target confidence/depth stability, target and pose age, pass acknowledgement latency, mechanism completion time, timeout counts, and per-stage match outcomes.

---

## 19. Engineering Decisions and Trade-offs

### 19.1 Preserve per-point timing

Keeping `timebase` and `offset_time` preserves the information required for motion compensation. The input contract is more explicit, while high-speed geometry remains physically meaningful.

### 19.2 Separate local odometry from field anchoring

The local estimate stays continuous and the map module owns the global anchor. This supports both smooth motion feedback and field-relative actions.

### 19.3 Separate ROS 2 autonomy from STM32 control

High-level tasks remain inspectable and configurable, while motor loops and protection remain deterministic. The UART/CAN protocol becomes an explicit engineering boundary.

### 19.4 Use feedback instead of fixed waiting

Physical completion signals adapt to voltage, load, friction, and mechanism variation. The task supervisor receives an event with a reason and timestamp.

### 19.5 Use shared contracts instead of direct coupling

Versioned pose, target, task, and command interfaces allow the estimator, detector, communication link, and controller to evolve independently while preserving the supervisor semantics.

---

## 20. Scientific Reasoning Through Engineering Work

### 20.1 Turn a vague field symptom into a variable

| Field observation | Testable hypothesis | Controlled comparison | Measurements |
| --- | --- | --- | --- |
| localization changes during motion | scan timing affects deskewing | per-point time preserved/omitted | residual, continuity, tracking state |
| robot reaches a position but waits | a quality gate is false or stale | fresh/stale target or pose | age, reason, state transition |
| shot result changes with distance | calibration map is inaccurate | fixed/calibrated parameter | distance error, release setting, landing point |
| a task repeats after retransmission | protocol lacks idempotence | one/repeated sequence | task ID, sequence, execution count |
| next action starts early | physical completion is unobserved | fixed wait/feedback event | mechanism state, completion time |

### 20.2 Reproducible field practice

My working habits were:

1. preserve logs, parameters, videos, messages, and versions;
2. change one main factor at a time;
3. grow from a single interface test to a single robot, dual robots, and a full match;
4. record successful and unsuccessful runs in the same table;
5. repeat the original scene after a fix and compare the same signals.

These habits connect hands-on debugging with scientific reasoning: a change is meaningful when its mechanism, measurement, and repeated effect are clear.

### 20.3 Transferable engineering pattern

The pattern generalizes to indoor mobile robots, multi-robot transport, logistics manipulators, and aerial teams:

```text
observation quality -> task state -> action permission
                   -> physical execution -> feedback -> recovery
```

The reusable contribution is the organization of heterogeneous sensing, state, communication, and actuation around observable interfaces.

---

## 21. Relation to Recent Robotics Literature

Recent ICRA, IROS, Science Robotics, and RoboCup system papers commonly frame an integrated robot around an operational question, then connect sensing, estimation, planning, control, and field validation. The following works provide technical and writing context for this project:

| Work | Venue | Relevance |
| --- | --- | --- |
| Tightly Coupled 3D Lidar Inertial Odometry and Mapping | ICRA 2019 | LiDAR--IMU estimation background |
| LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry via Smoothing and Mapping | ICRA 2021 | multi-sensor timing and coordinate consistency |
| Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction | ICRA 2023 | continuous-time motion compensation for fast motion |
| LiDAR Inertial Odometry Aided Robust LiDAR Localization System in Changing City Scenes | ICRA 2020 | local odometry plus global map localization |
| An Integrated System for Perception-Driven Autonomy with Modular Robots | Science Robotics 2018 | modular perception-to-action system design |
| Planning and Executing Joint Navigation Tasks in Autonomous Robot Soccer | RoboCup/LNCS | joint navigation and game behavior |
| Play Everywhere: A Temporal Logic based Game Environment Independent Approach for Playing Soccer with Robots | RoboCup 2023/LNCS | explicit game states and temporal behavior |
| RLSS: Real-time Multi-Robot Trajectory Replanning using Linear Spatial Separations | arXiv 2021 | multi-robot safety and replanning context |

### 21.1 How the project is positioned

The project uses established components such as FAST-LIO2, YOLO, ROS 2, ESP-NOW, UART, CAN, and STM32. Its engineering contribution is the system-level integration around a real competition workflow:

1. preserve sensor timing and establish one coordinate chain;
2. propagate pose and target quality into task decisions;
3. connect rule timing to dual-robot states and acknowledgements;
4. separate high-level autonomy from deterministic embedded execution;
5. migrate the same interfaces through simulation, replay, bench tests, and physical competition robots.

This is a systems integration and field reliability contribution, complementary to algorithm-level novelty.

### 21.2 Verified citation metadata

- Ye, H., Chen, Y., and Liu, M. “Tightly Coupled 3D Lidar Inertial Odometry and Mapping.” ICRA 2019. DOI: `10.1109/ICRA.2019.8793511`.
- Shan, T. et al. “LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry via Smoothing and Mapping.” ICRA 2021. DOI: `10.1109/ICRA48506.2021.9561996`.
- Chen, K. et al. “Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction.” ICRA 2023. DOI: `10.1109/ICRA48891.2023.10160508`.
- Daudelin, J. et al. “An Integrated System for Perception-Driven Autonomy with Modular Robots.” Science Robotics 2018. DOI: `10.1126/scirobotics.aat4983`.
- Suriani, V. et al. “Play Everywhere: A Temporal Logic based Game Environment Independent Approach for Playing Soccer with Robots.” RoboCup 2023/LNCS. DOI: `10.1007/978-3-031-55015-7_1`.

---

## 22. How I Explain the Project

### 22.1 Thirty-second version

I led LiDAR and perception for a ROBOCON Robot Basketball system. I integrated a Livox MID-360 with its built-in IMU and FAST-LIO2 for field localization, connected RGB-D/YOLO basket perception to distance and heading calculation, and linked the result to dual-robot communication and STM32 mechanisms. My main work was the hands-on integration: sensor installation and calibration, ROS 2 and TF debugging, shooting-parameter calibration, UART/ESP-NOW/CAN interfaces, and field testing on the physical competition robots.

### 22.2 Two-minute version

The robot has to move, receive, pass, aim, and shoot under a time-limited basketball rulebook. I treated each action as a measurable state transition. MID-360 and the internal IMU feed FAST-LIO2, which provides continuous local motion; a fixed field map supplies the global reference. RGB-D/YOLO provides the basket observation and distance, while the pose bridge computes the current heading error and the shooting model maps distance to a calibrated actuator parameter. A supervisor checks pose freshness, map lock, target stability, ball state, teammate safety, and mechanism feedback before releasing a shot. The high-level ROS 2 side communicates through UART and ESP-NOW, and the STM32/CAN layer performs the real-time motor and mechanism control. I debugged the system from the physical sensor and cable level through topics, transforms, protocols, and motor feedback, then migrated the integrated chain to the competition robots.

### 22.3 Five-minute research discussion

The research value of this project is the organization of heterogeneous observations into a reliable competition loop. Geometric sensing supplies continuity, semantic sensing supplies the target, the rule profile supplies task meaning, the dual-robot protocol supplies coordination consistency, and the STM32 layer supplies deterministic actuation. The important design choices are explicit timing, one TF owner, versioned quality fields, acknowledgement-based task transitions, and recovery paths. These choices let the team explain why a robot acted, why it waited, and which signal must be repaired before continuing.

---

## 23. Terminology Guide

| Ambiguous expression | Preferred expression | Meaning |
| --- | --- | --- |
| internal position index | desired formation position / target position | a geometric position relative to a team reference |
| the robot sees the basket | valid target observation | confidence, depth, stability, and freshness pass |
| command sent | action request transmitted | transport event before physical completion |
| action complete | execution feedback accepted | the mechanism reported completion |
| robot knows where it is | valid field-frame pose | a pose with a defined frame and quality state |
| communication is fine | heartbeat, ACK, sequence, and age satisfy the protocol | measurable link health |
| robust algorithm | maintains the specified task loop under stated conditions | a conditional engineering claim |
| simulation finished | interface and state tests passed | a stage in the deployment path |

Preferred words are **architecture, interface, data contract, state machine, observation, estimate, command, feedback, quality gate, recovery, and reproducible validation**.

---

## 24. Evidence Organization for the Portfolio

### 24.1 Physical competition system

The physical project includes:

- Livox MID-360 and built-in IMU;
- FAST-LIO2 LiDAR--inertial localization;
- RGB-D/YOLO basket perception;
- ROS 2 task and communication coordination;
- UART, ESP-NOW, STM32, and CAN execution;
- dual-robot pass, receive, and shoot workflow;
- simulation-to-real migration and field integration;
- national second prizes in Robot Basketball and Shooting.

### 24.2 Public engineering repository

The public repository organizes reusable interfaces and tests around the same system ideas:

- MID-360 input and timing contract;
- FAST-LIO2 and map-localization interface;
- TF ownership and pose quality;
- RGB-D/YOLO target gate;
- competition supervisor and rule profile;
- dual-robot task protocol;
- Gazebo field and sensor profiles;
- contract tests and diagnostics.

### 24.3 Academic presentation

An academic project description can separate three layers of evidence:

1. **design principle:** quality-bearing observation, state, action, feedback, and recovery;
2. **implementation:** ROS 2 packages, messages, transforms, protocols, and tests;
3. **field result:** physical integration, competition operation, and team awards.

This organization makes the project readable both as a real engineering system and as a research-oriented systems case study.

---

## 25. Final System Summary

The project can be remembered through five statements:

1. The rulebook defines the timing, distance, possession, and scoring conditions.
2. MID-360, the built-in IMU, and FAST-LIO2 provide geometric motion continuity and a field pose.
3. RGB-D/YOLO provides a semantic basket target, distance, and observation quality.
4. ROS 2, the dual-robot protocol, and the supervisor turn those observations into acknowledged tasks.
5. UART, ESP-NOW, STM32, and CAN turn the task into real chassis and mechanism behavior with feedback.

The complete loop is:

```text
rulebook and strategy
  -> match state
  -> sensor measurements
  -> LiDAR--IMU pose and RGB-D/YOLO target
  -> distance, heading, ballistic parameter, and task decision
  -> high-level communication
  -> STM32/CAN execution
  -> acknowledgement, diagnosis, and recovery
  -> next task or match completion
```

The central engineering achievement is the physical integration of estimation, perception, task reasoning, communication, embedded control, and competition practice into one system that can be inspected, calibrated, debugged, and operated on real robots.

---

## References

1. ABU ROBOCON 2025 Robot Basketball Rulebook, revision 2024-08-14. The project rule profile records the source URL and file hash.
2. Ye, H., Chen, Y., and Liu, M. “Tightly Coupled 3D Lidar Inertial Odometry and Mapping.” ICRA 2019. DOI: `10.1109/ICRA.2019.8793511`.
3. Shan, T. et al. “LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry via Smoothing and Mapping.” ICRA 2021. DOI: `10.1109/ICRA48506.2021.9561996`.
4. Chen, K. et al. “Direct LiDAR-Inertial Odometry: Lightweight LIO with Continuous-Time Motion Correction.” ICRA 2023. DOI: `10.1109/ICRA48891.2023.10160508`.
5. Daudelin, J. et al. “An Integrated System for Perception-Driven Autonomy with Modular Robots.” Science Robotics 2018. DOI: `10.1126/scirobotics.aat4983`.
6. Suriani, V. et al. “Play Everywhere: A Temporal Logic based Game Environment Independent Approach for Playing Soccer with Robots.” RoboCup 2023/LNCS. DOI: `10.1007/978-3-031-55015-7_1`.
7. Şenbaşlar, B., Hönig, W., and Ayanian, N. “RLSS: Real-time Multi-Robot Trajectory Replanning using Linear Spatial Separations.” arXiv:2103.07588.
