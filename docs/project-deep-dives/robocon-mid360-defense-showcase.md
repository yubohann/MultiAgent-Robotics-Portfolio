# ROBOCON MID-360 | Engineering Defense Display

> **Use**: slide-ready English content for insertion into the FAIR Lab interview deck. Each numbered block is designed to become one Reveal.js `<section>`. The display text is written for the audience; speaker notes, layout hints, and evidence fields are marked as production guidance for the editor.
>
> **Scope**: this document presents the physical ROBOCON Robot Basketball system and my hands-on engineering contribution. It complements the AeroGate Graph and HM3D ObjectNav research sections, with separate evaluators, metrics, and claims for each project.

## Production notes | How the current HTML is made

The supplied interview file is a static, offline Reveal.js presentation. Its shipped runtime executes bundled browser JavaScript, with no application framework dependency in the delivered file.

| Layer | Observed implementation | Practical implication for insertion |
| --- | --- | --- |
| Authoring source | `internal-archive-20260903/reveal-sample-source/index-source.html` | Add new content as top-level `<section>` blocks inside `.reveal > .slides`. |
| Presentation runtime | Reveal.js 6.0.1, with the Notes and Zoom plugins | Keep headings, `fragment` usage, and speaker notes compatible with Reveal.js. |
| Styling | Custom CSS after the Simple theme: navy, teal, amber, paper, split grids, matrices, state boards, and fixed footers | Reuse the existing classes and visual hierarchy for a consistent deck. |
| Packaging | Node.js CommonJS script `bundle-standalone.js` | The script reads the source HTML, inlines CSS and JavaScript, converts listed media to Base64 data URIs, checks for unresolved local resources, and writes a standalone HTML file. |
| Delivered artifact | `Bohan_Yu_FAIR_Lab_Interview.html` | The sent file is a self-contained offline artifact with media and libraries embedded. |
| Slide structure | 22 top-level `<section>` elements and a small footer-number synchronizer | Add pages as top-level sections and let the existing script assign their two-digit footer numbers. |
| Asset footprint | Inline PNG, SVG, GIF, and MP4 data URIs | The current artifact is about 39 MB; each additional embedded media file increases the sendable HTML size. |
| Runtime behavior | `Reveal.initialize({hash:true, controls:true, progress:true, width:1280, height:720, ...})` | Keep every new section within the existing 1280 x 720 canvas and preserve keyboard navigation. |

The current file contains three ROBOCON pages (system interface, personal LiDAR/perception, and state quality) plus one adjacent multi-UAV research-environment video page. The pages below add the missing competition context, physical integration sequence, communication and embedded boundary, debugging practice, and team result. They can be inserted after the current ROBOCON system page or used as a replacement block.

### Asset paths already available to the bundler

```text
media/robocon-competition-flag.png
media/robocon-field-debug.png
media/robocon-team-arrival.png
media/robocon-team-trophy.png
```

For a new image, add its relative path and MIME type to the `assets` array in `bundle-standalone.js` before rebuilding. Use relative media paths in the source HTML and keep Windows paths in production notes only.

## Page map

| Page | Slide title | Primary job | Suggested visual |
| --- | --- | --- | --- |
| 01 | ROBOCON Robot Basketball | Establish the physical-system story | Competition flag or team image |
| 02 | My responsibility in the team | Separate personal ownership from team ownership | Role and interface matrix |
| 03 | From the rulebook to robot states | Explain the design in plain language | Rule-to-check table |
| 04 | One competition loop | Show the end-to-end architecture | Layered data-flow diagram |
| 05 | From MID-360 packets to field pose | Explain the estimation chain | Sensor-to-pose pipeline |
| 06 | The FAST-LIO2 integration boundary | Show scientific and implementation understanding | Input/output contract |
| 07 | A pose is useful only with quality | Connect frames, timing, and command release | TF and quality board |
| 08 | From a basket image to a shooting command | Explain semantic perception and geometry | Coordinate sketch and equations |
| 09 | Two robots, one task lifecycle | Explain coordination and acknowledgement | Message sequence |
| 10 | ROS 2 for decisions, STM32 for real-time control | Explain the hardware/software boundary | Boundary diagram |
| 11 | The match supervisor is a finite-state machine | Connect time, possession, and recovery | State transition strip |
| 12 | Building the system on a real robot | Highlight hands-on integration | Six-step field timeline |
| 13 | Field debugging as controlled diagnosis | Demonstrate engineering judgment | Symptom-hypothesis-measurement table |
| 14 | Cross-disciplinary interfaces | Show collaboration through concrete artifacts | Interface matrix |
| 15 | Simulation, bench tests, and physical deployment | Explain migration discipline | Deployment ladder |
| 16 | Competition result and engineering evidence | Close the physical-system case | Arrival and trophy images |
| 17 | What the project taught me about research | Connect engineering to scientific work | Observation-state-action-feedback loop |

## 01 | ROBOCON Robot Basketball

**English display**

### ROBOCON 2025 Robot Basketball

**Turning sensing, coordination, and actuation into a verified competition loop**

The project was a dual-robot system for Robot Basketball. I was responsible for LiDAR and perception, and worked across the interfaces that connected the NUC, ROS 2, the two robots, the embedded controllers, and the physical mechanisms.

```text
MID-360 + internal IMU
        -> LiDAR-inertial field pose
RGB-D + YOLO
        -> basket target and depth
ROS 2 + dual-robot communication
        -> task and safety decision
STM32 + CAN + chassis/mechanisms
        -> physical action and feedback
```

The team received a national second prize in the Robot Basketball competition and a national second prize in the Shooting Special Competition.

**Speaker note**

Open with the system question: how to make a robot act correctly when perception, localization, communication, mechanics, and timing all change during a match. The awards are team results; my contribution is the LiDAR/perception and system-integration path that supported the competition robots.

**Reveal layout**

- Use the existing `hero` layout with a dark navy background.
- Left: title, subtitle, and one-sentence role statement.
- Right: `media/robocon-competition-flag.png` or `media/robocon-team-arrival.png`.
- Keep the system chain as a compact four-line visual.

**Evidence**

- `robocon-mid360-autonomy-stack-详尽技术说明.md`, Sections 1, 5, and 24.
- `robocon-mid360-autonomy-stack-detailed-technical-description-en.md`, Sections 1, 5, and 24.
- Team award images in `internal-archive-20260903/reveal-sample-source/media/`.

## 02 | My responsibility in the team

**English display**

### My responsibility in the team

I joined the university ROBOCON team in the second year of my undergraduate study. After a training and assessment period in simulation, ROS, IMU, and LiDAR, I became one of the official team members selected from more than thirty candidates.

My work sat at the boundary between raw measurements and executable robot behavior:

| Area I owned | Work delivered | Interface exposed to the team |
| --- | --- | --- |
| LiDAR and IMU | MID-360 bring-up, timed point-cloud input, FAST-LIO2 integration | continuous local motion estimate |
| Localization | map alignment, frame ownership, pose quality and freshness | field-frame pose with status |
| Perception | basket detection, RGB-D depth, temporal stability | valid target observation |
| Pose-to-action | distance, heading, and shooting-parameter calculation | navigation or shooting request |
| System integration | ROS 2 topics, TF, launch order, replay and field checks | a traceable command path |

Mechanical design, electrical power, low-level motor control, and mechanism construction were owned by the corresponding teammates. I coordinated with those owners through shared units, timestamps, states, and feedback signals.

**Speaker note**

This page answers the question "what did you personally do?" Emphasize direct work at the sensor, cable, topic, transform, protocol, and physical-robot levels. The project was collaborative; the interfaces made that collaboration executable.

**Reveal layout**

- Use the existing `split wide-left` layout.
- Left: the four-line personal timeline.
- Right: the compact matrix above, with the personal ownership column highlighted in teal.
- Keep ownership wording focused on my interfaces and the team's shared robot result.

## 03 | From the rulebook to robot states

**English display**

### From the rulebook to robot states

The competition rulebook became a set of timers, geometric checks, action evidence, and recovery paths. In plain language: I translated what the robot had to do into conditions that sensors could observe, software could judge, and actuators could confirm.

| Competition requirement | Measurable condition | System consequence |
| --- | --- | --- |
| Preliminary match duration: `120 s` | match clock and terminal state | supervisor stops new tasks when time expires |
| Knockout match duration: `160 s` | competition profile | timer and strategy profile are selected explicitly |
| Possession window: `20 s` | possession start and expiry | pass or shoot task is cancelled or replaced on timeout |
| Advance time limit: `8 s` | task start and deadline | the advance phase receives its own deadline |
| Pass separation: at least `1.0 m` | shared robot poses and pose age | pass release requires a valid separation check |
| Basket height: `2.43 m` | target geometry and launch model | distance and height enter shooting calibration |
| Scoring zones and shot types | active match stage and physical result | the supervisor records the action context |

The important design decision was to treat a command as a request, with completion determined by receiving confirmation, mechanism feedback, release evidence, and the relevant timing and spatial conditions.

**Speaker note**

This is the plain-language explanation of rule-driven design. The program turns rule numbers into checks that determine whether the next physical action may be released.

**Reveal layout**

- Use the existing `matrix` table style.
- Add a teal `CHECK` column accent and an amber time accent.
- Avoid a dense rulebook screenshot; the audience should see how a number changes the robot interface.

## 04 | One competition loop

**English display**

### One competition loop

The system was layered for debugging, but every layer returned a state that the next layer could inspect.

```text
competition profile and strategy
        -> match and task state
        -> MID-360 / IMU / RGB-D measurements
        -> FAST-LIO2 local motion + field-map pose
        -> basket target, depth, and confidence
        -> distance, heading, and task decision
        -> ROS 2 / ESP-NOW / UART messages
        -> STM32 / CAN chassis and mechanism control
        -> feedback, acknowledgement, recovery, and the next task
```

| Layer | Physical or software object | Output used by the next layer |
| --- | --- | --- |
| Sensing | MID-360, internal IMU, RGB-D camera | timestamped measurements |
| Estimation | FAST-LIO2 and map anchor | local and field-frame pose |
| Perception | YOLO detector and depth adapter | target state and confidence |
| Task | supervisor and dual-robot protocol | permitted action and deadline |
| Execution | STM32, CAN, chassis, shooting mechanism | motion and completion feedback |

**Speaker note**

Point out that the diagram is a data-flow contract. The system can explain which measurement produced a command and which feedback allowed the next state. This is what made field debugging tractable.

**Reveal layout**

- Use `split` with the data flow on the left and the five-row table on the right.
- Replace the long text arrow chain with a compact vertical pipeline if the slide becomes crowded.

## 05 | From MID-360 packets to field pose

**English display**

### From MID-360 packets to field pose

The LiDAR path preserved the temporal information needed by a moving platform:

```text
livox_ros_driver2/msg/CustomMsg
  -> timebase and per-point offset_time checks
  -> IMU propagation to each point time
  -> motion compensation and point preparation
  -> FAST-LIO2 direct geometric update
  -> local odometry: odom -> base_link
  -> fixed-map localization: map -> odom
  -> field-frame pose and quality state
```

FAST-LIO2 maintains a state containing orientation, position, velocity, IMU biases, and gravity. IMU samples propagate the state at high rate; LiDAR points supply geometric residuals against the incremental map. The estimator updates the state with an iterated error-state Kalman procedure and maintains local map structure for nearest-neighbour queries.

The engineering point is the boundary: the estimator provides continuous local motion, while the map-localization layer owns the field anchor. A smooth local estimate and a field-relative command therefore use different, explicit interfaces.

**Speaker note**

Describe FAST-LIO2 as an established method that I integrated, configured, checked, and connected to the rest of the robot. The project contribution is the physical interface, verification path, and competition integration around that method.

**Reveal layout**

- Use a horizontal five-stage pipeline with one highlighted `CustomMsg` box.
- Add a small `LOCAL` / `FIELD` two-row label to show the separation between odometry and map anchoring.
- If using an equation, show only the state symbols; keep the slide focused on the data path.

## 06 | The FAST-LIO2 integration boundary

**English display**

### The FAST-LIO2 integration boundary

The integration work was a chain of observable contracts that made every launch and field test inspectable.

| Contract | Check performed | Result passed downstream |
| --- | --- | --- |
| Point timing | `timebase`, `offset_time`, monotonicity, scan span | motion-compensated point stream |
| Point structure | finite values, point count, field names, frame | valid LiDAR update input |
| IMU stream | rate, timestamp freshness, axis convention | high-rate propagation |
| Extrinsics | LiDAR-to-IMU and sensor-to-body calibration | consistent body pose |
| Estimator state | tracking, covariance/quality fields, drop counters | pose status and local odometry |
| Map bridge | map lock, initial pose, single TF owner | field-frame command input |

When a pose changed unexpectedly, I traced the signal in this order: sensor input, timestamp and frame, estimator state, map anchor, command message, and finally physical feedback. That ordering separated a data-contract problem from an estimator problem and from a mechanical response problem.

**Speaker note**

The audience should leave with a concrete sense of what "integration" means: checking packet structure, timing, axes, transforms, and state transitions, then following the same signal into the command path.

**Reveal layout**

- Use the existing `matrix` style with three visual bands: `CHECK`, `INTERFACE`, `OUTPUT`.
- Highlight `timebase`, `frame`, and `map lock` as the three fields most likely to change physical behavior.

## 07 | A pose is useful only with quality

**English display**

### A pose is useful only with quality

The TF chain had one declared ownership path:

```text
map -> odom -> base_link -> imu_link -> lidar_mid360
                         \-> camera_link -> camera_optical_frame
```

A field-relative command was released only when the pose was available, valid, map-locked, and fresh. The shared quality state carried:

```text
tracking_state | map_locked | pose_age | sequence | drop_count | anchor_source
```

```text
PoseReady = pose_available
         AND pose_valid
         AND map_locked
         AND 0 <= pose_age <= max_pose_age
```

The same idea applied to target perception and teammate communication. A value that looks numerically reasonable may belong to an earlier frame, a different coordinate system, or an expired task. The quality fields therefore lived inside both the command interface and the diagnostic record.

**Speaker note**

Use this page to explain why a robot can appear to have a correct number and still make a wrong physical decision. Freshness, frame ownership, and sequence are part of the meaning of a measurement.

**Reveal layout**

- Use `split wide-right`: TF and formula on the left, the four quality fields on the right.
- Reuse the existing `state-board` component with `VALID`, `WAIT`, and `RECOVER` states.
- Reserve the red state for blocked action and keep the visual calm.

## 08 | From a basket image to a shooting command

**English display**

### From a basket image to a shooting command

The target path combined semantic detection with geometric calculation:

```text
RGB-D frame
  -> YOLO basket detection
  -> depth values inside the target region
  -> confidence and temporal-stability gate
  -> camera-frame target point
  -> camera-to-body/map transform
  -> range and heading error
  -> calibrated launch parameter
  -> STM32 shooting request
```

For a target point transformed into the field frame, let `(dx, dy)` be the horizontal displacement from the robot to the target and `robot_yaw` be the robot heading. The range and heading error can be written as:

```text
r       = sqrt(dx^2 + dy^2)
heading = atan2(dy, dx) - robot_yaw
```

The launch setting was calibrated as a function of target range and height difference through a range-dependent model. The detector supplied the semantic target; LiDAR-inertial localization supplied the robot pose; the bridge combined both into an action request with validity and freshness fields.

**Speaker note**

The key separation is geometric versus semantic information. LiDAR/IMU answers how the robot moves in the field; RGB-D/YOLO answers which observed object is the basket and how far it is. The supervisor releases a shot only after both are usable.

**Reveal layout**

- Use `split` with a coordinate sketch on the left and the pipeline/table on the right.
- Keep the two equations large enough to read but subordinate to the input-output story.
- Use the existing `media/robocon-field-debug.png` as a small field-side inset when the geometry sketch has sufficient space.

## 09 | Two robots, one task lifecycle

**English display**

### Two robots, one task lifecycle

The two robots shared task meaning and a complete task lifecycle, with raw velocity commands remaining inside their respective execution layers.

| Task phase | Ball-handling robot | Receiving or shooting robot |
| --- | --- | --- |
| Prepare | confirms ball state and requests teammate readiness | publishes pose and `receiver_ready` |
| Check | verifies distance, pose age, and advance timer | verifies its own pose and safe receiving state |
| Arm | sends `pass_armed` with a task identifier | accepts one task and acknowledges it |
| Execute | sends the pass request to the mechanism | waits for arrival or mechanism evidence |
| Confirm | records the acknowledgement and result | reports `receipt_confirmed` or a recoverable reason |

The protocol carried a role, `task_id`, sequence number, timestamp, state, and acknowledgement. Duplicate messages were handled idempotently; old sequence numbers were ignored; an expired heartbeat cancelled teammate-dependent motion. ESP-NOW handled the peer link, ROS 2 carried high-level state, and UART connected the host-side task to the embedded controller.

**Speaker note**

Transmission and physical completion were separate events. The task lifecycle made the evidence explicit and gave the team a shared way to recover from delay, duplication, or a missing acknowledgement.

**Reveal layout**

- Use a five-row `matrix` with the two robot roles as columns.
- Add a thin sequence strip under the table: `READY -> CHECK -> ARM -> EXECUTE -> CONFIRM`.
- Avoid a generic network diagram; the task lifecycle is the more informative visual.

## 10 | ROS 2 for decisions, STM32 for real-time control

**English display**

### ROS 2 for decisions, STM32 for real-time control

The computer and the embedded controller had different jobs.

```text
NUC / ROS 2
  localization, map, perception, task logic,
  dual-robot coordination, logging, recovery decision
                |
                | semantic command + sequence + timeout
                v
STM32 / CAN
  motor PID, limit switches, current protection,
  watchdog, mechanism timing, emergency stop
                |
                v
chassis and shooting mechanisms
                |
                `-> state, completion, fault, and timeout feedback
```

High-level commands described intent: move to a pose, prepare a pass, receive, aim, shoot, pause, recover, or stop. The STM32 converted that intent into deterministic motor and mechanism behavior and returned an observable state. This boundary kept computation-heavy autonomy on the NUC while preserving short, predictable safety paths in the controller.

**Speaker note**

Explain the division as a real engineering decision. ROS 2 is the right place for maps, perception, and task composition. PID loops, limit protection, watchdogs, and emergency stop need a shorter and more deterministic path.

**Reveal layout**

- Use the existing `system-map` visual: one computer block, one command boundary, one embedded block, one feedback line.
- Label the boundary `intent -> deterministic execution -> feedback`.
- Keep protocol bytes in the speaker notes and show the semantic boundary first.

## 11 | The match supervisor is a finite-state machine

**English display**

### The match supervisor is a finite-state machine

The supervisor connected competition time, sensor quality, task ownership, and physical feedback.

```text
BOOT
  -> PREFLIGHT
  -> WAIT_START
  -> ACTIVE
       -> ADVANCE
       -> PASS / RECEIVE
       -> AIM / SHOOT
       -> RECOVER
  -> FINISHED or ESTOP
```

`PREFLIGHT` checked sensor input, estimator state, map lock, target pipeline, communication heartbeat, UART, CAN, and mechanism readiness. `ACTIVE` released only the task whose spatial, temporal, and teammate conditions were valid. `RECOVER` cancelled the affected action, returned the robot to a safe state, and recorded a reason before a new task was considered.

The same pattern applied to a shot:

```text
target_valid + pose_fresh + ball_present + mechanism_ready
        -> aim
        -> release
        -> mechanism feedback
        -> result record or recovery
```

**Speaker note**

This is where the rulebook, perception, localization, communication, and hardware meet. A state transition is accepted when its evidence is present and the complete condition has been checked.

**Reveal layout**

- Use a horizontal state strip with `PREFLIGHT`, `ACTIVE`, `RECOVER`, and `FINISHED` as the visual anchors.
- Place the shot sub-state as a small inset alongside the main state machine.

## 12 | Building the system on a real robot

**English display**

### Building the system on a real robot

The work moved from individual interfaces to a complete physical loop.

| Stage | Hands-on work | Check before continuing |
| --- | --- | --- |
| 1. Mount and wire | fixed MID-360, IMU, RGB-D camera, NUC, network, USB, and power paths | sensor axes, cable routing, no mechanism occlusion |
| 2. Calibrate | LiDAR-IMU-camera extrinsics, time offsets, body-frame directions | known motion produces the expected frame motion |
| 3. Bring up ROS 2 | launch order, topic types, QoS, TF ownership, message rates | every required publisher and subscriber is present |
| 4. Map and localize | map the field, set initial pose, verify field dimensions | restart returns to a consistent field frame |
| 5. Connect action | basket detection, depth, distance, heading, shooting calibration | target state and pose state agree in the same frame |
| 6. Integrate the match | dual-robot messages, UART, STM32, CAN, mechanisms, recovery | physical feedback closes the task state |

This was direct work on the robot: mounting, wiring, starting nodes, measuring timing, checking transforms, reading serial and CAN feedback, calibrating the shooting path, and repeating the same task after each change.

**Speaker note**

Spend time on the physical verbs: installed, wired, calibrated, launched, measured, traced, adjusted, and retested. They make the engineering contribution visible while preserving the team's subsystem ownership.

**Reveal layout**

- Use an `evidence-row` timeline on the left.
- Use `media/robocon-field-debug.png` on the right with the caption `Field-side integration and debugging`.
- Keep the image large enough to identify a real robot and field context.

## 13 | Field debugging as controlled diagnosis

**English display**

### Field debugging as controlled diagnosis

I treated a field symptom as a testable hypothesis. The debugging record followed one repeatable loop:

```text
phenomenon -> hypothesis -> measurement -> one focused repair -> same-scene retest
```

| Field symptom | First measurements | Repair direction | Retest signal |
| --- | --- | --- | --- |
| Pose changes during a turn | point timing, IMU freshness, TF publisher, bracket motion | correct timing/frame/extrinsic or remap | pose continuity and tracking state |
| Robot reaches a position but waits | pose age, map lock, target validity, ball and mechanism state | complete the missing evidence path | state transition and command reason |
| Shooting result changes with range | depth statistics, target transform, launch calibration, battery state | recalibrate range-to-parameter mapping | release setting and landing result |
| Pass task repeats | `task_id`, sequence, ACK, execution count | enforce idempotence and expiry | one execution per task |
| Pass flow stalls | receive feedback, mechanism limit, UART/CAN state | add the real completion event or recovery branch | confirmed receipt or explicit recovery |
| Chassis command is unresponsive | ROS command, UART bytes, STM32 state, CAN feedback, enable line | trace the boundary one layer at a time | command-to-motor response |

The value of this workflow is a repeatable diagnosis path. It keeps algorithm tuning, interface repair, and hardware repair separate, so a fix can be attributed to the signal it changed.

**Speaker note**

Give one concrete example if asked: a robot waiting after reaching a target led to a check of pose freshness, map lock, target state, and mechanism feedback before any navigation parameter was changed. This shows how the diagnosis avoided blind tuning.

**Reveal layout**

- Use a three-column table: `SYMPTOM`, `MEASURE`, `REPAIR / RETEST`.
- Add the four-step loop as a small top band.
- Avoid a failure collage; the method is the visual point.

## 14 | Cross-disciplinary interfaces

**English display**

### Cross-disciplinary interfaces

The project required a common language across computer science, control, mechanical design, electronics, communication, and match strategy.

| Collaborator | What had to agree | Interface artifact |
| --- | --- | --- |
| Mechanical team | sensor height, field of view, bracket stiffness, mechanism travel | extrinsics, limits, action-complete condition |
| Electrical team | supply, UART framing, CAN IDs, emergency-stop path | command and feedback specification |
| Vision team | class, confidence, depth region, temporal stability | target observation message |
| Localization team | frame names, map version, initial pose, timestamp | pose and quality message |
| Communication team | role, task lifecycle, sequence, ACK, heartbeat | dual-robot task protocol |
| Strategy team | advance, pass, receive, aim, shoot, fallback order | supervisor state profile |

The practical skill was translating a specialist concern into a shared field: a unit, a timestamp, a frame, a state, a reason, or a feedback event. That translation allowed a mechanical adjustment or a perception update to be checked by the same supervisor logic.

**Speaker note**

This page focuses on engineering integration. The interface artifacts allowed different specialists to work on the same robot with shared, documented assumptions.

**Reveal layout**

- Use the existing `matrix` style with the middle column in amber and the artifact column in teal.
- A small caption can read `one robot, multiple owners, explicit interfaces`.

## 15 | Simulation, bench tests, and physical deployment

**English display**

### Simulation, bench tests, and physical deployment

The migration path kept message names, units, transforms, and state semantics stable while the physical conditions became harder.

```text
interface simulation
  -> sensor and transform replay
  -> single-robot bench test
  -> low-speed physical motion
  -> target and shooting calibration
  -> dual-robot communication test
  -> complete competition sequence
```

Each deployment record attached the operating context:

```text
rulebook revision | map version | extrinsic version | software commit
FAST-LIO2 profile | camera/YOLO profile | communication profile
STM32 firmware | robot identifier | run timestamp
```

The physical robot introduced scan timing, chassis vibration, RGB-D reflections, wireless interference, mechanism latency, battery variation, and the human-operated competition field. The staged path made each new source of variation observable before it entered the complete match loop.

**Speaker note**

Stress that the real-robot migration was completed and repeatedly integrated on the competition platform. Simulation and replay prepared and diagnosed the system; physical deployment supplied the competition evidence.

**Reveal layout**

- Use a seven-step horizontal deployment ladder.
- Put the deployment manifest in a compact monospace footer panel.
- Use `media/robocon-team-arrival.png` with its field context visible as a real deployment image.

## 16 | Competition result and engineering evidence

**English display**

### Competition result and engineering evidence

The system reached the setting it was designed for: a physical competition robot operating with real sensing, communication, embedded control, mechanisms, and a time-limited task.

| Evidence layer | What it shows |
| --- | --- |
| Physical integration | the MID-360, RGB-D camera, compute unit, controller, chassis, and mechanisms operated as one system |
| Field operation | sensing, pose, target, task, communication, and feedback were exercised under competition conditions |
| Team outcome | national second prize in Robot Basketball |
| Specialist outcome | national second prize in the Shooting Special Competition |
| Personal engineering record | LiDAR/perception ownership, interface bring-up, calibration, ROS 2/TF diagnosis, protocol tracing, and field retest |

The result is best described at the right level: the awards are team achievements, while my contribution was the engineering path that made the LiDAR and perception interfaces usable inside the complete robot system.

**Speaker note**

Use the two award statements once, clearly. Then return to the evidence chain: hardware mounted, data checked, state released, mechanism confirmed, and the system operated in the competition environment.

**Reveal layout**

- Use `split`: `media/robocon-team-trophy.png` or `media/robocon-team-arrival.png` on the right, evidence table on the left.
- Keep the award wording exact and team-level.
- Report recorded team outcomes and engineering evidence; keep timing, ranking, and per-robot performance fields tied to their source records.

## 17 | What the project taught me about research

**English display**

### What the project taught me about research

The physical robot gave me a way to turn an engineering symptom into a scientific question.

| Engineering observation | Testable variable | Controlled comparison |
| --- | --- | --- |
| motion changes the point-cloud geometry | point timing and deskewing | timed versus timing-stripped input |
| a command waits after navigation | pose/target freshness and state evidence | fresh versus stale input |
| shooting varies with distance | target geometry and launch calibration | fixed parameter versus calibrated parameter |
| a retransmitted task runs twice | sequence and idempotence | one message versus repeated message |
| one robot finishes while the other waits | task ownership and teammate feedback | acknowledgement path present versus absent |

The reusable pattern is:

```text
observation quality
        -> task state
        -> action permission
        -> physical execution
        -> feedback
        -> next observation
```

This is also how I approach multi-robot research: define the state and evaluator first, make the decision interface observable, change one mechanism at a time, and preserve the run record needed to explain the result.

**Speaker note**

Close the ROBOCON case by connecting it to research practice while keeping algorithm novelty claims tied to their actual source. The transferable ability is building an executable interface and using controlled evidence to improve it.

**Reveal layout**

- Use the existing `split wide-left` layout with the table on the left and the observation-state-action loop on the right.
- This page should transition naturally to the AeroGate Graph or HM3D research section.

## Short answers for an oral defense

These answers are optional speaker support. They can be kept in the Markdown source and omitted from the slide deck.

### Why MID-360 and its internal IMU?

The robot needed continuous geometric motion information while accelerating and turning. MID-360 provided the LiDAR stream, and the internal IMU supplied high-rate inertial information for motion propagation. Preserving the driver's timing fields allowed the estimator to compensate for motion during a scan.

### Why FAST-LIO2?

FAST-LIO2 provides a tightly coupled LiDAR-inertial estimation path with direct point-to-map geometric updates and an incremental map structure. It was a suitable established component for a moving competition platform. My work was to integrate it with the driver's message contract, frames, map anchor, quality state, and the downstream command path.

### Why combine geometric localization with RGB-D/YOLO?

The two streams answer different questions. LiDAR/IMU provides motion continuity and field pose; RGB-D/YOLO identifies the basket and estimates target depth. Keeping those roles explicit prevents a detection-box coordinate from being mistaken for a field pose and lets the supervisor check both quality states before a shot.

### What did you personally implement?

I worked on MID-360 and IMU input handling, FAST-LIO2 integration, pose and TF interfaces, map and freshness checks, basket perception and depth handling, distance/heading and shooting-parameter interfaces, ROS 2 bring-up, and the field-side debugging path. I coordinated the resulting interfaces with teammates responsible for mechanics, electrical systems, low-level control, communication, and strategy.

### How did you know a command had completed?

The system separated transmission from completion. A task carried an identifier and sequence; the receiving robot or embedded controller returned an acknowledgement or physical state. Timeouts cancelled dependent actions and moved the supervisor into an explicit recovery state.

### What would you measure in a follow-up study?

I would log synchronized pose error, target error, command age, communication latency, mechanism completion time, and task outcome. Then I would compare one controlled mechanism at a time over repeated runs, keeping the rule profile, map, sensor calibration, and evaluator fixed.

## Terminology for the deck

| Avoid | Use instead |
| --- | --- |
| internal position index | desired formation position, target position, or interface field |
| the robot sees the basket | valid basket observation with confidence, depth, stability, and freshness |
| command sent | action request transmitted |
| action finished | execution feedback accepted |
| robot knows where it is | valid field-frame pose with a defined frame and quality state |
| communication is fine | heartbeat, acknowledgement, sequence, and age satisfy the protocol |
| robust algorithm | the task loop remains operational under the stated conditions |
| rules in code | timed, spatial, and action-evidence checks in the supervisor |

## Insertion checklist

- [ ] Insert each numbered block as a top-level Reveal.js `<section>` alongside the existing AeroGate sections.
- [ ] Keep the four existing ROBOCON pages or replace them with the new block only after checking total speaking time.
- [ ] Reuse `hero`, `split`, `split wide-left`, `split wide-right`, `matrix`, `metric-band`, `state-board`, `system-map`, `evidence-row`, and `footer` classes.
- [ ] Add only the four existing ROBOCON media paths listed at the top unless a new asset is deliberately registered in `bundle-standalone.js`.
- [ ] Keep image `alt` text and a short caption on every visual.
- [ ] Keep the award wording team-level and attach individual ranking or timing claims only when a source record supports them.
- [ ] Keep FAST-LIO2, YOLO, ROS 2, ESP-NOW, UART, CAN, and STM32 described as integrated components and interfaces, with algorithm authorship attributed to the original methods.
- [ ] Rebuild the single-file artifact with Node.js `bundle-standalone.js` only after the source HTML is edited.
- [ ] Open the rebuilt artifact in Chrome and Edge, verify keyboard navigation, check the video remains paused, and confirm that the new sections fit the 1280 x 720 canvas.

## Source documents

- [ROBOCON MID-360 autonomy stack](robocon-mid360-autonomy-stack.md)
- [ROBOCON 2025 detailed Chinese technical description](robocon-mid360-autonomy-stack-详尽技术说明.md)
- [ROBOCON 2025 detailed English technical description](robocon-mid360-autonomy-stack-detailed-technical-description-en.md)
- [FAST-LIO2, Point-LIO, and autonomy algorithm notes](robocon-mid360-lio-algorithms-zh.md)
- The FAIR Lab interview HTML and its Reveal.js standalone bundler are private application artifacts and stay outside this repository.
