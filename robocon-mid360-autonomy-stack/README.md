# Robocon MID-360 Autonomy Stack

**Simulation-first ROS 2 autonomy for a competition basketball robot.**

<p align="center">
  <img src="site/assets/01-scene-gazebo-court.png" alt="2025 ROBOCON style Gazebo basketball court" width="49%">
  <img src="site/assets/02-pointcloud-rviz.png" alt="MID-360 point cloud in RViz" width="49%">
</p>
<p align="center">
  <img src="site/assets/03-pcd-2d-map-rviz.png" alt="2D PCD projection and occupancy map in RViz" width="49%">
  <img src="site/assets/04-rgbd-depth-robot2.png" alt="Robot 2 RGB-D and depth view" width="49%">
</p>
<p align="center">
  <img src="site/assets/05-basketball-shot-process.png" alt="Dual robot basketball shot process" width="82%">
</p>
<p align="center">
  <img src="site/assets/robocon-mid360-basketball-demo.gif" alt="Live two-robot basketball demonstration" width="82%">
</p>

The stack connects a Livox MID-360 simulation, FAST-LIO2 odometry and mapping, fixed-map localization, perception gating, pose command arbitration and a safety-aware competition supervisor in one ROS 2 workspace. Every handoff carries its own validity signal, from per-point LiDAR timing to map lock, target validity and action feedback, and an invalid input is rejected at the gate that owns it.

**Status.** `v0.1.0-simulation-prealpha`. The full chain runs on Gazebo simulation and recorded inputs, with explicit synthetic adapters at the perception and mechanism interfaces. Hardware steps follow the calibration and interlock procedures in the source tree.

## What It Does

- **MID-360 data path.** Livox `CustomMsg`, per-point timing, IMU transport, input validation and freshness diagnostics.
- **FAST-LIO2 integration.** ROS 2 launch profiles for mapping and local odometry with controlled motion scripts.
- **Fixed-map localization.** PCD loading, map metadata, scan matching, `map -> odom` anchoring, tracking states and recovery transitions.
- **Competition control.** A rule-aware supervisor, versioned action requests, perception gates, protocol handling and fault-injection tests.
- **Gazebo environments.** A candidate indoor competition scene, an open-field degradation scene, field geometry, robot model, hoops and a simulated MID-360 sensor.
- **Deterministic experiments.** One-command dispatchers, manifests, deterministic ROS domain isolation, metrics exporters and plotting tools.

## My Role

I led the LiDAR and perception work for the ROBOCON Robot Basketball system and owned the integration boundary between localization, navigation, dual-robot communication, and embedded control. In this workspace that covered:

- **LiDAR and perception.** MID-360 and built-in IMU integration, timestamp and extrinsic checks, RGB-D/YOLO basket observations, and the target-validity gates.
- **Localization and mapping.** FAST-LIO2 integration and configuration, field mapping, fixed-map scan matching, map metadata, and the `map -> odom` anchor.
- **Autonomy stack integration.** ROS 2 launch composition, topic and TF contracts, the pose-to-command path, and the interfaces to the dual-robot link and the STM32 execution layer.
- **Simulation and validation.** Gazebo scenes, robot and sensor models, the deterministic experiment runners, fault injection, and the metric exporters used to check interfaces before field tests.
- **Documentation.** The interface contract, the bilingual READMEs, the upstream notices, and the deployment procedures.

The same interfaces were migrated to the physical competition robots, where I ran the sensor, topic, transform, protocol, and mechanism-level field debugging.

## Pipeline

```text
Gazebo or recorded bags
          |
          v
Livox CustomMsg + IMU  ->  FAST-LIO2  ->  local odometry
                                            |
                                            v
                             registered cloud mapper
                                            |
                              frozen PCD + metadata
                                            |
                                            v
                          scan-to-map localization
                                            |
                                            v
             perception gate -> action arbitration -> competition supervisor
```

The public TF contract is the chain `map -> odom -> base_link -> imu_link -> lidar_mid360`. Each interface is explicit, testable and replaceable, and the control layer keeps localization freshness, map lock, perception validity, action feedback, heartbeat, expiry, cancellation and recovery as first-class state.

## Quick Start

The supported development environment is Ubuntu 22.04 with ROS 2 Humble.

```bash
source /opt/ros/humble/setup.bash
export LIVOX_SDK2_ROOT=/path/to/Livox-SDK2/install
colcon build --symlink-install --cmake-args -DLIVOX_SDK2_ROOT="$LIVOX_SDK2_ROOT"
source install/setup.bash
```

Inspect launch arguments and run the dependency-light validation suite.

```bash
ros2 launch mid360_simulation gazebo_mid360_lio.launch.py --show-args
python3 tools/runners/validate_project.py
python3 tools/runners/run_python_contract_tests.py
```

Run the bounded experiment groups from one command.

```bash
bash tools/runners/run_experiments.sh --dry-run all
bash tools/runners/run_experiments.sh quality
bash tools/runners/run_experiments.sh faults
bash tools/runners/run_experiments.sh rgbd
```

A visible Gazebo and RViz session.

```bash
bash tools/runners/run_experiments.sh gui
```

The dispatcher creates an isolated run directory, records the exact command and ROS domain, and keeps generated evidence outside the public source files.

## Runtime Profiles

| Profile | Purpose |
| --- | --- |
| `30000-ray` | Full-density Gazebo sensor profile for controlled mapping and LIO experiments |
| `2000-ray` | Fast topic and interface smoke profile |
| `indoor_competition_candidate` | Structured indoor scene with field geometry and four hoop assets |
| `open_field_degraded` | Quality-gating and geometric-degradation tests |
| `gazebo_simulation` | Gazebo sensor and physics runtime |
| `bag_replay` | Replayed recorded inputs |

## Verification

CI runs the Python contract tests, validates ROS package manifests, installs ROS dependencies, builds the interface and control packages and executes the ROS test suite. Local runs use the same commands.

```bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

Additional tools export run summaries and figures from retained JSON and CSV data.

```bash
python3 tools/metrics/export_run_metrics.py <run-directory> <output-directory>
python3 tools/plotting/plot_run_metrics.py <metrics.csv> <output-directory>
```

## Repository Layout

```text
src/
  mid360_localization_contract/  Input, frame, tracking, and map contracts
  mid360_map_localizer/          Fixed-map scan matcher
  mid360_map_tools/              Registered-cloud mapper and occupancy tools
  game_supervisor/               Competition state machine and safety gates
  perception_adapter/            Target validity and perception interface
  camera_yolo_adapter/           Detector interface and metric scoring
  pose_command_bridge/           Pose-to-command arbitration interface
  mid360_simulation/             Gazebo worlds, robot, sensor, and runners
  vendor/
    fast_lio/                    FAST-LIO2 source and license notice
    livox_ros_driver2/           Livox ROS 2 driver and license notice
tools/                            Capture, metrics, plotting, and runner utilities
site/                             GitHub Pages portfolio site
.github/workflows/                Continuous integration and Pages deployment
```

Private run archives, generated maps, bags, internal audits and development notebooks stay outside the public source tree, and Git ignores them.

## Documentation

- [MID-360 interface contract](src/mid360_localization_contract/INTERFACE_CONTRACT.md) covers frames, topics, diagnostics, QoS and timing.
- [Portfolio site](site/index.html) presents the architecture and the ordered 2025 ROBOCON views.
- The upstream notices under `src/` record the source repository, revision and scope of each adapted package.

## Attribution

Third-party components remain in their source locations with their original license files and notices. Adapted packages include an `UPSTREAM_NOTICE.md` describing the source repository, revision and scope of changes. Read the notices under `src/` before redistributing a modified build.

## Portfolio

**https://yubohann.github.io/Robocon-mid360-autonomy-stack/**

## License

Team-owned files are released under the repository license in [`LICENSE`](LICENSE). Vendored components retain their own licenses and notices.
