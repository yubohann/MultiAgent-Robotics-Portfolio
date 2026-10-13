# ROBOCON MID-360 Dual-Robot Autonomy System

## A research-oriented project report for advisors and interviewers

**Platform:** ROS 2 Humble, Gazebo, Livox MID-360, FAST-LIO2, RGB-D/YOLO, STM32/CAN  
**Task context:** dual-robot autonomy for the ABU ROBOCON 2025 Robot Basketball rules

> This report presents the project as a research and systems-engineering effort. The public materials emphasize interfaces, simulation, fault injection, tests, and reproducible experiment scaffolding. Hardware configuration, calibration, and field measurements are retained in the corresponding experiment records.

## 1. Executive Summary

The project studies a practical robotics question: how can a pair of competition robots turn LiDAR-inertial geometry, RGB-D/YOLO semantics, teammate communication, and mechanism feedback into an explainable, recoverable, and repeatable autonomy loop under a time-limited rule set?

My main contribution is the boundary design that connects established components into one reliable system. The system preserves scan start time and per-point sampling offsets from the Livox sensor, uses FAST-LIO2 as a mature LiDAR-inertial front end, and adds data-quality checks, pose conversion, a unified coordinate tree, fixed-map anchoring, state assessment, target confirmation, task arbitration, dual-robot communication, and an STM32/CAN execution boundary. Every action can therefore be traced to its coordinate reference and time, its quality evidence, and its physical completion feedback.

The system is organized into ten independent functional modules with extensive tests, launch configuration, and experiment parameters. Its continuous-integration workflow targets Ubuntu 22.04 and ROS 2 Humble for dependency checks, building, and testing. Gazebo profiles provide a candidate indoor competition field, an open degraded scene, simulated MID-360 data, dual-robot tasks, and a basketball trajectory demonstration. The experiment dispatcher preserves the configuration, logs, and result summary of each run as an auditable evidence trail.

The resulting platform is a strong basis for robotics-systems, embodied-AI, and multi-robot research. It treats an algorithm output as an evidence object carrying a coordinate reference, time, uncertainty, freshness, quality, and recovery meaning. This makes later studies of localization robustness, perception-control coupling, coordination protocols, and sim-to-real transfer measurable within one common vocabulary.

## 2. Project Thesis and Research Question

### One-sentence thesis

This is a simulation-first autonomy system for a rule-driven dual-robot basketball task: MID-360 and IMU provide continuous geometry, a fixed map provides the field frame, RGB-D/YOLO provides basket evidence, and a task supervisor turns these signals into feedback-bearing pass, receive, and shot actions.

### Research question

> When local odometry, global mapping, semantic targets, teammate state, and mechanisms operate at different rates and with different delays, how should a common quality contract decide when an action is permitted and how the system recovers when evidence degrades?

The question has three coupled layers:

1. **State estimation:** estimate pose, velocity, and tracking quality from temporally resolved LiDAR and IMU data;
2. **Task reasoning:** convert rules, roles, and target observations into ordered action evidence;
3. **Execution:** deliver intent to the real-time controller, observe completion, and recover from timeout, cancellation, or interruption.

### Working assumptions

| Assumption | Engineering interpretation | Observable evidence |
| --- | --- | --- |
| Timing can be calibrated | scan time, point offsets, and IMU time are monotonic | sensor diagnostics and replay analysis |
| Geometry is locally constrained | the field map supplies enough surfaces for scan registration | registration error, overlap, and tracking state |
| Frames compose consistently | map, local odometry, body, and sensor frames each have one owner | coordinate tree, interface contract, and extrinsic version |
| Semantics can be confirmed | a target carries confidence, range, stability, and age | target observation and quality reason |
| Tasks can be acknowledged | each action has identity, order, expiry, and completion evidence | action request, feedback, and acknowledgement |
| Execution can be protected | the real-time controller owns PID, limits, current protection, watchdog, and E-stop | controller state and E-stop latch |

## 3. Contributions

### 3.1 Native MID-360 input contract

The official Livox message remains intact until the FAST-LIO2 boundary. A data-quality stage checks point-count consistency, minimum point count, increasing scan time, monotonic point offsets, increasing IMU time, finite inertial values, and LiDAR/IMU freshness. It produces a unified input-quality state and diagnostics with accepted data counts, discarded data counts, point statistics, timing span, and issue reasons.

### 3.2 Canonical pose and TF contract

The FAST-LIO tracking frame is an implementation detail of the upstream estimator. A pose-adaptation layer converts it to a stable local-odometry-to-robot-body relationship, validates source frames, normalizes the full quaternion, transforms tracking origin to body origin, transforms velocities, preserves uncertainty and timestamps, and maintains this coordinate relationship through one owner.

### 3.3 Local continuity and global anchoring

Local LIO remains responsible for smooth motion. The fixed-map module estimates a separate global anchor from the map to local odometry. Initial pose, relocalization, and map correction can then update field alignment while preserving the trajectory used by the control loop. The system exposes four interpretable states: uninitialized, tracking, relocalizing, and lost.

### 3.4 Quality-aware perception and action gates

The RGB-D/YOLO adapter combines detector preprocessing, candidate filtering, image-scale recovery, and depth query into a standard target observation carrying confidence, range, stability, and observation time. A target-quality stage reports a clear reason from confidence, range, stability, and freshness. Global actions are released when the pose is valid, the map is trusted, and the data remains fresh.

### 3.5 Rule, protocol, and feedback supervision

The task supervisor combines match phase, localization quality, basket evidence, ball state, mechanism health, and teammate safety. A pass follows an evidence sequence of receiver readiness, pass preparation, pass execution, and receipt confirmation. A shot requires an active match, fresh pose and target observation, trusted map alignment, ball presence, healthy mechanism state, and teammate safety.

### 3.6 Reproducible experiments and recovery

The experiment dispatcher supports quality, fault, mapping, localization, interlock, competition, and perception studies. Each run uses an isolated communication domain and records configuration, logs, and result summaries. Fault paths cover stale messages, duplicate tasks, localization loss, teammate disconnect, missing mechanism feedback, process exit, and E-stop.

## 4. From Rulebook to System Requirements

The rule source is the **ABU ROBOCON 2025 Robot Basketball** rulebook, revision 2024-08-14. Its numerical requirements are organized as competition parameters and converted into testable timing, spatial, and action decisions.

| Rule quantity | Value | System consequence |
| --- | ---: | --- |
| Field | `15.0 m x 8.0 m` | map scale, boundary, and initial pose share one geometry |
| Fence | `0.10 m` high, `0.05 m` wide | boundary structure enters map and motion checks |
| Backboard | `1.80 m x 1.05 m` | structured physical prior for target perception |
| Basket height | `2.43 m` | fixed target height in the trajectory model |
| Inner diameter | `0.450-0.459 m` | target-center accuracy and shot repeatability |
| Preliminary match | `120 s` | explicit match clock and terminal state |
| Knockout match | `160 s` | separate competition profile |
| Possession | `20 s` | possession timer bounds the task sequence |
| Advance | `8 s` | dedicated offensive-side transition timer |
| Minimum pass distance | `1.0 m` | synchronized robot positions enter pass permission |
| Dribble hand height | `0.70 m` | dribble evidence connects to mechanism feedback |
| Ordinary scores | `2 / 3` | shot zone and action type become parameters |
| Dunk score | `7` | additional spatial and motion evidence |

Rule-driven design translates natural-language requirements into timers, spatial constraints, state conditions, physical feedback, and stop paths. The eight-second advance becomes a task timer; the one-meter pass becomes a two-robot geometry check; possession becomes a state with explicit expiry; scoring type becomes part of shot planning and result logging.

## 5. System Architecture and Module Ownership

```text
Gazebo / physical sensors
        |
        v
Livox point cloud + IMU
        |
        v
data-quality checks -> FAST-LIO2 -> local odometry + registered cloud
                                      |
                                      v
                           mapping + fixed-map localization
                                      |
                                      v
                       global map -> local odometry -> robot body
                                      |
                 +--------------------+--------------------+
                 |                                         |
          RGB-D / YOLO basket observation          pose and action computation
                 |                                         |
                 +--------------------+--------------------+
                                      v
                           competition task supervisor
                                      |
                              structured action request
                                      |
                             UART / ESP-NOW
                                      |
                               STM32 / CAN
                                      |
                           chassis and mechanism feedback
```

The system is organized into ten independent functional modules with clear responsibilities:

| Functional module | Responsibility |
| --- | --- |
| LiDAR and localization interfaces | input quality, pose conversion, coordinate tree, map anchoring, tracking, startup checks |
| Mapping and localization | fixed-map registration, point-cloud processing, occupancy representation, global correction |
| Vision and target quality | YOLO detection, RGB-D depth, target stability, observation freshness |
| Pose and action interfaces | distance and heading, command permission, structured action requests |
| Competition supervision | rule engine, match state machine, action feedback, protocol, recovery |
| Simulation and experiments | Gazebo scenes, robot, sensors, basketball loop, experiment scheduling |
| Upstream algorithm components | standardized integration and version management for FAST-LIO2 and the Livox driver |

The boundary principles are direct: upstream algorithms connect through stable interfaces, each coordinate relationship has one owner, and every high-level action is traceable to input quality and physical feedback.

## 6. MID-360 Sensing and Point-Level Time

### 6.1 Why retain point-level timing

A LiDAR scan unfolds over time. For point `i`, its sampling instant can be represented as

```text
t_i = scan_start_time + point_offset_i
```

During acceleration or rotation, points in one scan correspond to different instantaneous poses. FAST-LIO2 uses IMU propagation to estimate this intra-scan motion and applies geometric residuals for correction. Preserving point-level timing therefore supports deskewing and stable registration during fast maneuvers.

### 6.2 Data-quality checks

```text
point-count consistency
        -> scan-time monotonicity
        -> point-level offset monotonicity
        -> finite coordinates
        -> valid IMU time and values
        -> freshness of both streams
        -> unified input-quality state and diagnostics
```

The quality stage keeps received and accepted counts, discarded counts, point statistics, timing span, and issue reason. Network loss, driver configuration errors, and malformed packets become identifiable at the input boundary.

### 6.3 Timing and calibration study

On the physical platform, a LiDAR-IMU time offset can be estimated from synchronized data containing energetic rotational motion. The offset that minimizes deskewed geometric residuals becomes a versioned calibration value. Static installation measurement and motion validation jointly determine the body-to-IMU-to-LiDAR relationship; its value and provenance enter the experiment record.

## 7. FAST-LIO2 Theory and Integration Scope

### 7.1 State and observation model

The estimator state can be written as

```text
x = (R, p, v, b_g, b_a)
```

where `R` is orientation, `p` position, `v` velocity, and `b_g`, `b_a` gyroscope and accelerometer biases. IMU data drives high-rate propagation, while LiDAR geometry provides correction. For map normal `n_i`, map point `q_i`, and current point `p_i`, a point-to-plane residual is

```text
r_i = n_i^T (R p_i + p - q_i)
```

FAST-LIO2 provides a tightly coupled LiDAR-inertial route with an incremental map structure and direct geometric updates. In this project it is used as a mature external front end. The project contribution surrounds that front end with timing preservation, configuration, coordinate consistency, quality state, map anchoring, and action gating.

### 7.2 Relation to alternative methods

| Method | Characteristic | Role in this project |
| --- | --- | --- |
| LOAM / LeGO-LOAM | feature extraction and structured ground handling | classical geometric reference |
| LIO-SAM | factor-graph smoothing, loop closure, IMU coupling | global-optimization comparison point |
| LVI-SAM / FAST-LIVO2 | deeper visual-LiDAR fusion | future multimodal baseline |
| KISS-ICP | compact LiDAR registration | fast no-IMU smoke baseline |
| FAST-LIO2 | direct point-to-map updates and incremental estimation | front end selected for MID-360 timing and competition motion |

The comparison clarifies design choices. The current system uses FAST-LIO2 as its primary localization front end; the other methods are candidates for controlled baselines and future studies.

## 8. Pose Conversion and Coordinate-Frame Ownership

### 8.1 Rigid-body transforms

All poses use rigid-body `SE(3)` transforms. If the front end provides a tracking-origin pose and calibration provides the transform from the tracking origin to the robot body origin, the bridge computes

```text
T_odom_base = T_odom_tracking * inverse(T_base_tracking)
```

The global field pose is

```text
T_map_base = T_map_odom * T_odom_base
```

The unified coordinate chain is

```text
map -> odom -> base_link -> imu_link -> lidar_mid360
```

### 8.2 Pose-adaptation responsibilities

- validate source odometry and tracking frame names;
- normalize the complete orientation quaternion and detect degenerate attitudes;
- convert the tracking origin to the robot body origin;
- transform linear and angular velocity using rigid-body kinematics;
- retain pose uncertainty and the sensor timestamp;
- publish one canonical local robot pose and one corresponding coordinate relationship;
- expose pose age, continuity, discarded-data statistics, and quality reason.

Planar yaw is recovered from the normalized quaternion:

```text
yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
```

This preserves the full orientation semantics and uses the complete attitude estimate for planar heading.

## 9. Mapping and Occupancy Representation

### 9.1 Registered-cloud map

The registered-cloud mapping stage follows:

```text
point-cloud parsing
  -> numerical, height, and range filtering
  -> scan-level voxel downsampling
  -> radius outlier filtering
  -> map-voxel accumulation
  -> reliable cloud and map outputs
  -> map file and metadata
```

The default geometry and resource limits use a near range of `0.45 m`, a far range of `35.0 m`, scan voxels of `0.08 m`, map voxels of `0.12 m`, and bounded point and map sizes. Metadata carries coordinate frame, resolution, source, calibration version, and software version so localization experiments remain traceable.

### 9.2 Occupancy probability model

The occupancy representation uses log-odds updates:

```text
L_t(c) = L_(t-1)(c) + l_hit / l_free
P(c) = 1 / (1 + exp(-L_t(c)))
```

Ray casting updates free cells along the beam and the occupied cell at the endpoint. Inflation adds chassis and safety margin. Decay and stale reset keep long-unobserved areas interpretable. The system publishes a two-dimensional occupancy map and visualization markers for comparison with the point-cloud map and field boundary.

## 10. Fixed-Map Localization and Local/Global Separation

### 10.1 Fixed-map registration pipeline

The fixed-map localization stage implements:

```text
load fixed map
  -> voxel downsample
  -> map and scan size check
  -> initial pose and fresh local odometry
  -> establish global anchor
  -> register scan against map
  -> convergence and error check
  -> translation and heading jump check
  -> publish global pose correction
```

Representative parameters require at least 500 map points and 80 scan points, use `0.12 m` voxels, a `1.5 m` correspondence radius, at most 30 iterations, a registration-error threshold of `0.40`, a maximum translation jump of `0.80 m`, a maximum heading jump of `25 degrees`, and a `0.30 s` local-pose freshness window. These values balance field scale, real-time operation, and mismatching risk.

### 10.2 Why separate local and global references

Local odometry supplies a smooth signal for control, while the fixed map supplies a stable field reference. Updating the global anchor corrects field alignment and keeps the local body trajectory continuous. An initial pose initializes or resets the global anchor while leaving the local estimator trajectory intact.

### 10.3 Tracking states

```text
UNINITIALIZED -> TRACKING -> LOST
                       \-> RELOCALIZING -> TRACKING
```

Only the tracking state asserts that the global map is trusted. Reduced overlap, degraded registration error, stale pose, or invalid input moves the system to relocalization or lost and propagates a reason to the action layer. A fresh accepted anchor returns the system to stable tracking.

## 11. RGB-D/YOLO Target Quality

### 11.1 Detection pipeline

```text
image acquisition
  -> object detection
  -> candidate filtering and scale recovery
  -> RGB-D depth measurement
  -> temporal stability assessment
  -> standardized target observation
```

The vision module combines object detection, depth measurement, and temporal stability into a single target observation. Runtime, replay, and simulation observations retain their source category so the evidence presented to an advisor remains easy to interpret.

### 11.2 Observation and back-projection

Each target observation contains confidence, range, stability, observation time, target category, image location, and evidence source. With camera intrinsics `(fx, fy, cx, cy)` and depth `Z`, the pinhole back-projection is

```text
X = (u - cx) Z / fx
Y = (v - cy) Z / fy
Z = depth(u, v)
```

The target-quality stage uses a confidence threshold of `0.70`, a valid range of `1.0-10.0 m`, an observation-age window of `0.50 s`, and temporal stability. Its reason explains whether the observation is old, uncertain, out of range, or unstable, allowing the supervisor to wait, reacquire, or recover.

## 12. Pose Commands and Shooting Geometry

For current pose `(x,y,yaw)` and target `(x_t,y_t)`:

```text
distance = sqrt((x_t - x)^2 + (y_t - y)^2)
heading_error = wrap(atan2(y_t-y, x_t-x) - yaw)
```

The pose-command gate releases a global action when

```text
pose available
&& pose quality accepted
&& global map trusted
&& pose inside freshness window
```

The structured action request carries a version, time, target position, range, and heading error, so the same semantic command can cross the computer, communication link, and real-time controller. The high-level computer expresses intent; the controller performs deterministic motor and mechanism execution.

The shot model also uses basket height and launcher calibration. A nominal projectile trajectory is

```text
z(t) = z0 + v0z * t - 0.5 * g * t^2,  g = 9.81 m/s^2
```

In the physical system, range, relative height, launch angle, velocity, and mechanism state form a calibrated parameter map, updated from landing error and mechanism feedback. Gazebo and the physical platform use the same action semantics, giving simulation a direct role in preparing and diagnosing the embodied experiment.

## 13. Rule Engine, Task Supervision, and Action Feedback

### 13.1 Match state machine

```text
boot -> preflight -> wait for match start -> active match
                                            |
                                            +-> pause -> recovery -> active match
                                            +-> fault handling
                                            +-> emergency stop
                                            +-> finished
```

Preflight evaluates sensor streams, localization, coordinate consistency, map, initial pose, target perception, teammate communication, and mechanism health. The active phase releases actions only when spatial, temporal, teammate, and physical conditions agree. Recovery cancels affected tasks, returns the robot to a safe state, and records a reason. Emergency stop is a latched state that requires external reset.

### 13.2 Action lifecycle

The action set covers moving to a target position, preparing to receive, collecting a ball, preparing and executing a pass, preparing and executing a shot, cancelling a task, and emergency stop. Feedback uses accepted, running, succeeded, failed, and cancelled states, with task identity and completion evidence attached to each transition.

### 13.3 Shot interlock

```text
active match
&& pose quality accepted
&& global map trusted
&& pose age <= 0.30 s
&& target observation accepted
&& target age <= 0.30 s
&& ball present
&& mechanism healthy
&& teammate safe
```

The shot permission is a conjunction over information quality, possession, mechanism state, and team safety. A changed condition yields a readable reason and routes the task to hold, cancel, or recovery; physical completion is established through feedback.

## 14. Dual-Robot Protocol and Embedded Boundary

### 14.1 Versioned messages

Each action request carries:

```text
protocol version
action and task identity
sender and sequence number
creation time and expiry
action type and parameters
```

Action feedback adds execution state, reason, and evidence. Dual-robot messages use task identity, sequence, expiry, acknowledgement, and heartbeat to maintain consistency. Retransmission remains idempotent, while expired messages route to safe wait or recovery, so short communication disturbances do not cause repeated execution.

### 14.2 Pass evidence

```text
idle
  -> receiver ready
  -> pass prepared
  -> pass executed
  -> receipt confirmed
```

The sending robot checks ball state, range, and advance timing. The receiving robot declares readiness, waits for physical arrival, and returns receipt confirmation. Transport delivery and physical completion are separate events, so retries remain distinguishable from a second pass.

### 14.3 ROS 2 and STM32/CAN roles

```text
host computer / ROS 2
  localization, perception, task, logging, recovery
        |
        | semantic action + sequence + timeout
        v
STM32 / CAN
  motor PID, limit switches, current protection,
  watchdog, mechanism timing, emergency stop
        |
        v
chassis and shooting mechanisms
        |
        `-> state, completion, fault, timeout feedback
```

ROS 2 is the natural home for maps, perception, strategy, and task composition. STM32/CAN keeps motor and mechanism protection short-cycle and deterministic. This separation supports rapid high-level iteration while preserving predictable limits, watchdog, current protection, and E-stop behavior.

## 15. Gazebo Simulation and Basketball Loop

The simulation includes a candidate indoor competition field and an open degraded scene, together with a high-density LiDAR mode for mapping and a fast interface-check mode. Scene geometry, point-cloud density, RGB-D, and ground truth can be varied independently to create controlled comparisons.

The synthetic mechanism model tracks health, possession, receiving preparation, pass execution, receipt confirmation, shot preparation, release, and emergency-stop state. The basketball demonstration follows:

```text
start -> robot 1 dribble -> move to pass position -> pass in flight
      -> pass success -> robot 2 dribble -> shot preparation
      -> ball in flight -> shot success
```

Each stage associates a rule state, target pose, action feedback, and simulated physical state. The spatial error between the ball and basket is written to the experiment summary, exercising the perception-decision-execution-feedback interface. Simulation and camera-runtime observations remain separate evidence categories.

### Competition and physical evidence

The autonomy stack was integrated with physical competition robots and exercised through sensor bring-up, calibration, communication checks, mechanism tests, and complete task rehearsals. The team received a national second prize in the Robot Basketball event and a national second prize in the Shooting Special Competition. These are team achievements; my individual contribution was the LiDAR/perception integration and the interface work that connected localization, target geometry, task supervision, communication, and embedded execution.

The physical integration cycle included sensor mounting and wiring, LiDAR-IMU-camera calibration, coordinate verification, ROS 2 bring-up, serial and CAN tracing, shooting-parameter calibration, dual-robot message checks, and field-side retesting. Each change was evaluated through a short loop of observation, hypothesis, measurement, focused repair, and same-scene retest.

## 16. Fault Matrix and Recovery

| Event | Detection | Immediate behavior | Recovery entry |
| --- | --- | --- | --- |
| malformed LiDAR data | point count, scan time, and offset checks | mark input quality degraded | inspect driver/network and replay raw data |
| stale LiDAR or IMU | arrival age | hold preflight | restore stream and verify communication |
| wrong coordinate relationship | coordinate contract | hold pose use | correct front-end frame parameters |
| invalid orientation | quaternion normalization | mark pose invalid | inspect estimator and calibration |
| registration jump or poor error | jump/error gate | clear map trust | relocalize |
| missing map anchor | tracking state | hold global actions | accept verified initial pose |
| stale or unstable target | target-quality stage | keep shot pending | reacquire RGB-D target |
| expired teammate message | expiry/heartbeat | cancel dependent action | heartbeat, retry, or safe return |
| missing mechanism feedback | safety snapshot | block pass/shot | mechanism recovery or operator input |
| process exit | lifecycle monitoring | coherent shutdown | diagnose and restart |
| E-stop | emergency-stop latch | terminate motion and mechanisms | external reset and fresh preflight |

The common recovery pattern is:

```text
detect -> explain -> bound -> recover
```

Reasons remain in diagnostics, action feedback, and experiment summaries, giving field debugging a direct evidence path.

## 17. Reproducible Experiments and CI

### 17.1 Experiment dispatcher

The experiment dispatcher assigns an isolated run directory and communication domain to each study and records configuration, logs, map version, calibration version, rule version, and result summary. Fixed-map localization uses a consistent map, calibration, rule, and software version.

Recommended progression:

```text
simulation interface tests
  -> system connectivity and data-type checks
  -> LiDAR/IMU replay
  -> mapping and map-version record
  -> fixed-map localization
  -> dual-robot task protocol
  -> fault injection
  -> physical deployment
```

### 17.2 CI and local checks

The continuous-integration workflow targets Ubuntu 22.04 and ROS 2 Humble. It runs interface tests, metadata checks, dependency installation, system builds, tests, and result aggregation. The project also provides a lightweight structural validation stage for rapid checks during development. This validation confirms consistent package metadata and test entry points across the functional modules.

Full system connectivity, Gazebo, and sensor validation are evaluated on the Ubuntu/ROS 2 target and captured with the corresponding hardware facts in the experiment record.

## 18. Research Experiment Design

### Experiment A: point-timing ablation

Compare input with point-level timing against input represented by a single frame timestamp, using identical LiDAR/IMU data. Report point-to-plane residuals, pose drift during turns, tracking interruptions, and computational cost. The hypothesis is that point-level timing reduces intra-scan motion distortion.

### Experiment B: frame-contract ablation

Inject an incorrect sensor coordinate, duplicate coordinate publisher, and unnormalized attitude. Measure detection latency, downstream action suppression, and error propagation. This quantifies the value of unified coordinate ownership.

### Experiment C: fixed-map relocalization

Start from multiple initial positions, interrupt local odometry, request relocalization, and introduce geometric degradation. Report time to return to stable tracking, registration error, overlap, and rejection rate.

### Experiment D: supervisor safety study

Replay stale pose, stale target, teammate timeout, ball absence, and mechanism faults. Report false-accept rate, correct-block rate, recovery latency, and reason coverage.

### Experiment E: protocol stress test

Reorder, duplicate, delay, and expire dual-robot messages. Report duplicate-detection accuracy, task completion rate, acknowledgement latency, and safe-fallback rate.

### Experiment F: sim-to-real comparison

Keep message semantics, frames, units, action definitions, and evaluator fixed across Gazebo, replay, and the physical platform. Compare pose age, target stability, command latency, mechanism completion time, and recovery count.

These experiments turn field observations into falsifiable variables and provide a clear path to a systems paper, a master's research plan, or an internal laboratory benchmark.

## 19. Personal Contribution and Evidence Boundary

### My technical contribution

- design of MID-360 and IMU data-quality checks, timing semantics, and freshness diagnostics;
- FAST-LIO2 integration, local-odometry adaptation, coordinate conversion, and pose-quality contract;
- fixed-map construction, scan registration, global anchoring, map-trust state, and relocalization;
- RGB-D/YOLO target adaptation, depth ranging, stability assessment, and shot-parameter interface;
- ROS 2 bring-up, communication quality, parameter templates, diagnostics, and contract tests;
- dual-robot task protocol with acknowledgement, ordering, expiry, heartbeat, and cancellation;
- semantic action boundary from the host computer to STM32/CAN and field-side diagnosis;
- sensor installation, cable and communication bring-up, calibration, bench tests, and competition-flow replay.

### Evidence layers

| Evidence | What it establishes |
| --- | --- |
| implementation and configuration | interface, algorithm invocation, thresholds, states, recovery logic |
| unit and contract tests | geometry, message, state-machine, idempotence, and interlock invariants |
| Gazebo simulation | sensor timing, field geometry, basketball loop, synthetic faults |
| bag replay | real message timing, calibration, repeatable input conditions |
| physical experiment record | device, firmware, extrinsics, network, run count, field metrics |
| team competition outcome | operation of the integrated dual-robot system in a competition setting |

FAST-LIO2, the Livox driver, the YOLO detector, and third-party simulation assets are integrated as mature components. My contribution is the contract design, integration, quality propagation, task supervision, test strategy, and field loop around them. This gives upstream methods accurate attribution while making the systems-research contribution explicit.

## 20. Interview Answer Bank and Memory Card

### One-minute answer

I worked on a dual-robot autonomy system for the ROBOCON Robot Basketball task. The central engineering problem was turning signals with different time scales, coordinate references, and reliability into legal, executable, and observable actions. The robots used a Livox MID-360 and its internal IMU with FAST-LIO2 for local motion estimation. I preserved point-level scan timing, built data-quality checks and a canonical body-pose interface, and used fixed-map localization to maintain the global field reference. RGB-D/YOLO supplied basket confidence, depth, and temporal stability. Dual-robot communication supplied task identity, ordering, acknowledgement, and expiry, while STM32/CAN handled deterministic motor and mechanism protection. The supervisor released a shot only when pose freshness, trusted map alignment, valid target observation, ball presence, mechanism health, and teammate safety all agreed. My contribution was the integration and evidence loop that made these components inspectable, cancellable, and recoverable.

### Technical memory card

```text
Sense:    MID-360 + IMU + point-level scan timing
Estimate: FAST-LIO2 + local odometry + quality assessment
Anchor:   fixed-map registration + global anchoring + tracking state
Decide:   RGB-D/YOLO + rule engine + dual-robot task protocol
Execute:  host action -> real-time controller -> feedback and recovery
```

```text
body pose = local tracking pose x calibrated sensor-to-body transform
field pose = global map anchor x local body pose
shot permission = active match x fresh pose x trusted map x valid target
                  x ball present x healthy mechanism x safe teammate
```

### Likely advisor questions

**Why retain point-level timing?** A scan is collected over a time interval. The sampling instant of each point lets the LiDAR-inertial front end estimate intra-scan motion and perform deskewing.

**Why add a pose-adaptation layer?** The estimator's tracking frame is an implementation detail. The task layer needs one calibrated robot-body pose with time and uncertainty attached.

**Why separate local and global references?** Local control receives a smooth trajectory while global field alignment can be corrected independently for initialization and relocalization.

**What happens when localization quality drops?** The supervisor pauses actions that depend on global coordinates, records the reason, and enters relocalization or recovery. A fresh pose and trusted anchor restore execution.

**How is completion established?** Communication delivery, controller execution, and physical completion are separate events. Task identity, ordering, acknowledgement, feedback, and timeout form the completion evidence.

**What is the next study?** Point-timing ablation, relocalization recovery statistics, dual-robot protocol stress, and aligned sim-to-real metrics are the most direct next experiments.

## 21. Future Research Directions

1. **Uncertainty propagation:** map LiDAR/IMU uncertainty, registration error, target-depth variance, and communication latency to an action risk score.
2. **Dynamic-scene localization:** add dynamic-point rejection, semantic masks, or robust kernels and study moving robots and balls in the fixed-map setting.
3. **Cooperative planning:** study role assignment and spatiotemporal planning under communication delay and action-failure probability.
4. **Learning-assisted mechanism control:** learn a range-to-launch residual model inside the real-time controller's safety envelope, preserving interpretable limits and E-stop paths.
5. **Verifiable sim-to-real:** use one evaluator across simulation, replay, bench, and physical runs for pose, target, action, and recovery metrics.
6. **Systems paper:** frame the work as an “evidence-bearing ROS 2 autonomy contract for rule-constrained dual-robot manipulation,” reporting rule revision, hardware, calibration, datasets, run counts, and statistical intervals.

### Conceptual references

- Xu et al., “FAST-LIO2: Fast Direct LiDAR-inertial Odometry,” *IEEE Transactions on Robotics*, 2022: tightly coupled LiDAR-inertial estimation and incremental mapping.
- Shan et al., “LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping,” *IROS*, 2020: factor-graph smoothing and loop-closure comparison.
- Besl and McKay, “A Method for Registration of 3-D Shapes,” *IEEE TPAMI*, 1992: the ICP family underlying scan-to-map registration.
- ROS 2 and `tf2` design principles: explicit coordinate ownership, time-stamped transforms, and composable distributed nodes.
- ABU ROBOCON 2025 Robot Basketball rulebook, revision 2024-08-14: the task constraints used to derive timers, field geometry, and action evidence.

## 22. Research Value and Advisor Takeaway

The research value of the ROBOCON project lies in its system-level coherence. Native sensor timing, LiDAR-inertial estimation, fixed-map geometry, semantic target quality, rule supervision, dual-robot coordination, and embedded safety form one inspectable evidence chain. Each module has a clear theoretical role, and each cross-module transition carries coordinate, time, uncertainty, expiry, and feedback semantics.

This project demonstrates a complete path from published methods to physical robot capability: understanding state estimation and geometric residuals, designing stable robot-software interfaces, propagating perception and localization quality into action permission, constructing controlled simulation and replay experiments, and validating sensing, communication, mechanisms, and competition rhythm on physical robots. These skills transfer directly to SLAM, embodied intelligence, multi-robot coordination, and reliable autonomy, and they provide a strong starting point for deeper algorithmic and systems research under an advisor's guidance.

---

## 23. Extended Technical Appendix for Advisors

### 23.1 Evidence-bearing autonomy chain

```text
sensor sample
    -> timing and frame validation
    -> local LiDAR-inertial state
    -> fixed-map consistency
    -> target geometry with quality
    -> rule and task authorization
    -> bounded host intent
    -> embedded execution
    -> feedback and recovery
```

At each transition, the downstream module receives a value together with its timestamp, frame, numerical validity, freshness, and quality. This makes the first violated precondition visible in a failure trace and turns system integration into an auditable research object.

### 23.2 Timing, frames, and global anchoring

For a scan beginning at `t_start`, point `i` is associated with `t_i=t_start+Delta t_i`. The estimator uses the pose appropriate to that point time when deskewing. A direct ablation compares point-level timing with a single frame timestamp using residual, turn-segment drift, tracking interruptions, target stability, and compute time.

`map` is the global field reference, `odom` is a locally smooth frame that may drift, `base` is the robot body, and `sensor` frames describe hardware extrinsics. Local odometry supplies continuity; fixed-map registration supplies global anchoring. When the anchor changes, global targets and teammate relations are revalidated before authorization.

### 23.3 Target quality and authorization

The detector identifies target class and image support, while RGB-D supplies range and three-dimensional geometry. A target record contains position, range, confidence, valid-depth ratio, temporal stability, and age. The supervisor authorizes an action only when pose, target, task phase, teammate state, mechanism interlocks, and freshness predicates are valid. Authorization permits a bounded action; embedded feedback confirms completion.

### 23.4 Dual-robot protocol invariants

| Invariant | Meaning | Test |
| --- | --- | --- |
| idempotence | retransmission does not execute twice | duplicate injection |
| ordering | an older task cannot overwrite a newer task | sequence reordering |
| expiry | stale intent is rejected or cancelled | controlled delay |
| liveness | heartbeat loss reaches a bounded safe state | timeout injection |

Messages carry identity, sequence or version, timestamp, expiry, action type, parameters, sender, and acknowledgement state. These invariants support controlled research on delayed and unreliable multi-robot coordination.

### 23.5 Experiment matrix and evidence boundary

| Study | Controlled factor | Metrics | Evidence |
| --- | --- | --- | --- |
| point timing | point offsets vs frame time | residual, drift, interruptions | replay/simulation |
| frame ownership | unique authority vs conflict | pose error, suppression, recovery | contract/replay |
| map anchoring | odometry vs fixed-map anchor | drift, registration, relocalization | replay/field |
| target quality | single frame vs temporal gate | stability, command variance, shot outcome | bag/bench |
| supervisor | checks enabled vs removed | false accept, correct block | fault injection |
| protocol | clean vs delay/duplicate/reorder | duplicate execution, completion | simulation/bench |

The national second prize in Robot Basketball and Shooting Special Competition is system-level evidence of effective integrated operation under competition pressure. The research contribution around mature components is the explicit contract for timing, frames, quality, rules, communication, embedded safety, and recovery, together with experiments that can isolate each mechanism.

## 24. Advisor Takeaway

ROBOCON is a physical autonomy study organized around admissible evidence. It connects LiDAR-inertial estimation, global map consistency, target geometry, task rules, communication, and embedded feedback. The project provides a concrete starting point for uncertainty-aware multi-robot autonomy and for experiments that compare cooperative planning and recovery under delayed information.
