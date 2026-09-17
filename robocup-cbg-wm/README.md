# Uncertainty-Aware Counterfactual Belief-Graph World Model for Rule-Constrained Multi-Robot Tactics

[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-2563EB)](https://docs.ros.org/en/jazzy/)
[![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-E95420)](https://ubuntu.com/)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-Sim2Real-16A34A)](https://isaac-sim.github.io/IsaacLab/)
[![RL](https://img.shields.io/badge/RL-CBG-WM%20CVaR%20MPC-7C3AED)](isaaclab_sim/rl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-111827)](LICENSE)

<p align="center">
  <img src="./assets/readme/overview.png" alt="RoboCup vision robot platform, ROS2 stack, and competition outcomes" width="96%" />
</p>

<p align="center"><strong>Object-centric visual robotics, rule-gated behavior, and replayable multi-agent benchmark runs.</strong></p>

<p align="center">
  <a href="./docs/media/large_scale_50v50_isaaclab_preview.gif">
    <img src="./docs/media/large_scale_50v50_isaaclab_preview.gif" alt="12-second IsaacLab 50v50 replay preview" width="86%" />
  </a>
</p>

<p align="center">
  <a href="./docs/media/large_scale_50v50_isaaclab_replay.mp4">Full IsaacLab replay MP4</a>,
  <a href="./docs/media/最终回放_三视角同步拼接版.gif">Synchronized three-view replay GIF</a>,
  <a href="./docs/media/README.md">Media notes</a>
</p>

<p align="center">
  <img src="./assets/readme/robot_sensor_layout.png" alt="Robot sensor layout, coordinate frames, and actuator interfaces" width="96%" />
</p>

CBG-WM, the Uncertainty-Aware Counterfactual Belief-Graph World Model, is a ROS2 and IsaacLab robotics research project for adversarial multi-agent visual navigation. It combines uncertainty-aware belief tokens, typed object-interaction graph dynamics, a probabilistic ensemble world model, and Flow-proposal CVaR risk MPC with rule-aware action shielding, pushable rigid obstacles, laser-target dwell and range constraints, IsaacLab replay, and a Sim2Real deployment contract. The legacy object-centric SAC Flow policy is retained as the required baseline.

The active implementation is [CBG-WM](./docs/cbg_wm.md), with uncertainty-aware belief tokens, typed object-interaction dynamics, a probabilistic ensemble, and Flow-proposal CVaR MPC. Its code, OOD protocol and ablations are included, and the performance numbers below are evidence from the published legacy SAC Flow baseline run.

The repository is organized as a replayable engineering artifact. The validated main line is a two-robot RoboCup-style adversarial match with a 128-episode stochastic scoring run, strict replay audits, three-view IsaacLab media, and subsequent 1v1 real-robot experiment coverage. A separate 50v50 simulation-stage benchmark is included as a scalable rule-level extension.

Core areas cover multi-agent reinforcement learning, object-centric world models, SAC Flow and PolicyFlow, IsaacLab, ROS2 and Nav2, Sim2Real, visual target interaction, robot safety audits.

## Evidence Snapshot

| Area | Public Evidence | Scope |
| --- | --- | --- |
| 1v1 adversarial robot match | 128-episode scoring run, yellow 49.22%, blue 50.78%, draw 0.00%, zero static and box penetrations, zero robot contacts | Real-robot 1v1 experiments were performed, with public rosbag and statistical hardware tables as the next packaging step |
| IsaacLab replay | Compact synchronized three-view GIF with top view, yellow first-person view and blue first-person view | The replay visualizes the selected audited run, and hardware statistics follow the Sim2Real track |
| Object-centric SAC Flow and PolicyFlow | Training summaries, contract scoring JSON and CSV, strict replay audit and generated figures under `docs/rl_data/` and `docs/figures/` | Current results are project-level evidence with peer review as the next step |
| 50v50 extension | Staged 5v5 to 10v10 to 25v25 to 50v50 rule-level curriculum, 256-game scoring run, IsaacLab tactical replay | Simulation-stage rule-level benchmark covering 100 vehicles |
| Replay | Rule-environment checks, ROS2 dry-run commands, IsaacLab wrapper, capability evidence docs | IsaacLab and ROS2 setup uses the documented platform dependencies |

For a short reviewer-oriented summary, start with [Admissions Project Brief](./docs/admissions_project_brief.md). For the capability list and measured evidence, read [Capabilities and Measured Evidence](./docs/capabilities.md).

## Highlights

- ROS2 Jazzy workspace using `colcon` and `ament_cmake`
- Nav2-based navigation with centralized costmap and controller parameters
- `slam_toolbox` mapping/localization configuration
- AprilTag Tag36h11 visual target detection from `/camera/image_raw`
- ROS2 service based shooter controller
- Competition state machine covering navigation, target search, alignment, opponent-only firing, retry and timeout handling
- IsaacLab two-robot arena scene with falling targets, armor removal, differential-drive motion and collision handling
- Realistic sensor stack with wheel odometry, IMU, 2D lidar, RGB and depth camera frames, ToF and bumper contacts and a fixed laser module
- Rule-accurate laser model with 5-50 cm normal-target range, 20-80 cm recessed-base range, line-of-sight blockers, 0.80 s dwell gate and distance-dependent accuracy
- Recessed base targets behind ground-touching blue armor blockers, with 45-degree normal target placement
- Pushable rigid obstacle boxes whose map poses change in strict replay and IsaacLab playback
- Sim2Real domain randomization and a geometry-aware action shield for safer learned strategy execution
- Collision and stuck recovery through localization-confidence modeling and spin-in-place map rebuild
- Documentation for architecture, migration, Sim2Real, and third-party attribution

## Quick Start

Install the dependency-light rule environment and test tools.

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

Install the optional training and rendering stack for those paths.

```bash
python -m pip install -e ".[training]"
```

```bash
cd crc_robocup_vision_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch rcvrl_bringup competition.launch.py
```

Run the yellow-side elimination launch.

```bash
ros2 launch rcvrl_bringup competition.launch.py team_color:=yellow target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.yellow.yaml
```

Run the blue-side elimination launch.

```bash
ros2 launch rcvrl_bringup competition.launch.py team_color:=blue target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.blue.yaml
```

Run the hardware-free launch smoke test.

```bash
ros2 launch rcvrl_bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

When building from WSL, copy the workspace into a native Linux path such as `~/crc_robocup_vision_ws` first, because ROSIDL generation requires an ASCII path.

Run Python rule-environment checks.

```bash
python -m pip install -r isaaclab_sim/rl/requirements.txt
cd isaaclab_sim/rl
python evaluate_selfplay.py --episodes 8
```

The public rule environment includes its own arena geometry and synthetic target configuration, so rule-environment tests, self-play smoke runs and the 50v50 rule simulator run from the repository alone.

IsaacLab preview on Windows runs through the project wrapper, which keeps Kit user config, logs, pip envs and extension cache under `.isaaclab_runtime/` inside the project.

```powershell
.\scripts\run_isaaclab_project.ps1 -Headless -DemoFlow -Duration 120
```

To stop a previous preview run, inspect this project's processes first.

```powershell
.\scripts\stop_project_isaaclab.ps1 -WhatIfOnly
.\scripts\stop_project_isaaclab.ps1
```

Detailed onboarding guides.

- [Getting Started Guide](./docs/getting_started.md) covers step-by-step Python, ROS2 and IsaacLab setup, a quick demo tutorial and troubleshooting.
- [Capabilities and Measured Evidence](./docs/capabilities.md) lists the validated agent scale, published metrics, distributed-training profile and Sim2Real evidence.

## Target Platform

- Ubuntu 24.04
- ROS2 Jazzy
- OpenCV with ArUco/AprilTag dictionary support
- Nav2
- slam_toolbox

## Portfolio Scope

The ROS2 workspace is the clean submission package, and the retained docs focus on the current ROS2, IsaacLab, object-centric world-model and SAC Flow stack. Sim2Real calibration and validation steps are documented in `docs/sim2real.md`, and a concise rules summary is kept in `docs/rules_summary.md`.

![RoboCup field rule scene](./assets/readme/arena_rule_scene.png?raw=true)

![Robot sensor layout](./assets/readme/robot_sensor_layout.png?raw=true)

![ROS2 runtime graph](./assets/readme/ros2_runtime_graph.png?raw=true)

## Learning Strategy

The reinforcement-learning layer is implemented under `isaaclab_sim/rl/`. The active method is CBG-WM with a PolicyFlow-style tactical actor. Sensor-facing object beliefs feed a typed interaction graph and stochastic ensemble, and short-horizon CVaR MPC ranks joint Flow proposals before the existing action shield executes the first action.

The learned model predicts object-state distributions, rewards, termination and four rule-risk channels, and the scoring contract reports 1, 5 and 10-step prediction, risk calibration, CVaR risk, OOD outcomes and paired push-box and armor counterfactual directions. The previously audited auxiliary-MLP SAC Flow policy is retained as the required baseline. The final replay uses recessed base targets, ground-touching blue armor blockers, 45-degree normal target placement, dynamic pushable boxes and strict replay collision checks.

Publication-style method and experiment figures.

![Project overview](./docs/figures/paper/fig01_project_overview.png)

![Method architecture](./docs/figures/paper/fig02_method_architecture.png)

![Training and results](./docs/figures/paper/fig03_training_and_results.png)

![Ablation and safety audit](./docs/figures/paper/fig04_ablation_and_safety.png)

![Sim2Real replay pipeline](./docs/figures/paper/fig05_sim2real_replay_pipeline.png)

Data-driven GPU training and scoring figures.

![World-model SAC Flow training curve](./docs/figures/rl/rl_training_curve_gpu.svg)

![Self-play strategy contract metrics](./docs/figures/rl/rl_strategy_event_metrics.svg)

![Target and base-rush metrics](./docs/figures/rl/rl_target_base_metrics.svg)

![Pushable box metrics](./docs/figures/rl/rl_box_push_metrics.svg)

Runtime checkpoints, replay traces and policy exports live under `isaaclab_sim/output/` after local training and scoring, and Git ignores them by design. The compact reviewer brief is in `docs/admissions_project_brief.md`.

Final stochastic scoring run for the selected residual scale.

| Episodes | Yellow Win | Blue Win | Draw or Timeout | Static Penetrations | Box Penetrations | Robot Contacts per Game |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 49.22% | 50.78% | 0.00% | 0 | 0 | 0.00 |

Final strict replay audit.

| Episodes | Yellow Win | Blue Win | Draw or Timeout | Hard Violations | Warnings | Own-Target Penalties | Base Wins per Episode |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 37.50% | 62.50% | 0.00% | 0 | 0 | 0.0 | 1.0000 |

## Capability Scope

The public validated multi-agent result is a two-robot yellow-vs-blue adversarial RoboCup-style match. The repository validates object-centric world-model SAC Flow self-play, rule-aware action shielding, pushable boxes, base blockers, laser dwell and range constraints, ROS2 runtime contracts, IsaacLab three-view replay and 1v1 real-robot experiment coverage for this two-agent setting.

The large-scale 50v50 extension is a simulation-stage benchmark with staged rule-level curriculum training, a 256-game scoring run and an IsaacLab tactical replay, and it is published below. The Sim2Real material documents the ROS2 interface contract, calibration order, domain randomization and deployment validation ladder, and 1v1 real-robot trials have been performed. See [Capabilities and Measured Evidence](./docs/capabilities.md) for the capability list and metrics.

## Large-Scale 50v50 Benchmark

The repository also includes a large-scale extension for studying 100-agent adversarial coordination before committing to expensive full-physics training. The benchmark uses two teams of 50 differential-drive vehicles in an `80 m x 50 m` arena with three control zones, static cover, shielded bases, line-of-sight shooting, fire cooldowns, agent elimination, base damage, robot-contact metrics and obstacle-contact metrics.

The accepted long-run baseline uses staged population-based swarm-flow policy search. The 5v5 stage first learns the zone-to-shield-to-base attack loop, then 10v10, 25v25 and 50v50 reuse the last passing checkpoint with stricter HP and shield gates. Candidate team policies are sampled, scored against archive opponents from both yellow and blue sides, promoted through elite weighting, checked against candidate archives, then scored over 256 games. The accepted trace is replayed in IsaacLab with 100 vehicle-shaped actors, visible heading noses, bases, zones, barriers, tactical lanes and a telemetry panel.

Formal 50v50 baseline.

| Final-Stage Training Episodes | Scoring Episodes | Yellow Win | Blue Win | Draw | Yellow Base Damage | Blue Base Damage | Robot Contacts Mean and P95 | Obstacle Contacts |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12000 | 256 | 36.72% | 42.19% | 21.09% | 44.90 | 44.89 | 0.00 / 0.00 | 0.00 |

[50v50 IsaacLab replay MP4](./docs/media/large_scale_50v50_isaaclab_replay.mp4)

![50v50 rule layout](./docs/figures/large_scale_50v50/large_scale_50v50_rule_layout.png)

![50v50 rule-scoring closure](./docs/figures/large_scale_50v50/large_scale_50v50_rule_closure.png)

![50v50 training curve](./docs/figures/large_scale_50v50/large_scale_50v50_training.png)

![50v50 scoring summary](./docs/figures/large_scale_50v50/large_scale_50v50_eval.png)

This is a scalable rule-level training benchmark with IsaacLab tactical replay evidence, published as a simulation-stage 50v50 result.

## Runtime Evidence

The ROS2 runtime is organized around `rcvrl_bringup`, `rcvrl_behavior`, `rcvrl_vision`, `rcvrl_navigation`, `rcvrl_motion`, `rcvrl_shooter`, `rcvrl_description` and `rcvrl_interfaces`. A demo video is available on Bilibili.

[RoboCup VisionRL runtime/demo video](https://www.bilibili.com/video/BV1Pj9ZBKEc8/?spm_id_from=333.1387.list.card_archive.click&vd_source=f79b94dd69d0c8d08ee5c3400b69d46d)

The compact IsaacLab replay below is generated from the audited physical-box trajectory trace. Both robots leave their start zones at `t=0`, attack opponent-side targets only, push rigid obstacle boxes with changing map poses, trigger armor removal after normal-target hits, and finish with a base-target win. The repository keeps a compact synchronized three-view GIF for GitHub display, and full-resolution source videos stay as local generated artifacts.

![最终回放：三视角同步 GIF](./docs/media/最终回放_三视角同步拼接版.gif)

The rendered episode passes strict checks for static-obstacle penetration, pushable-box penetration, target legality, own-target safety, differential-drive step limits and score and armor consistency. The selected 8-episode strict audit reports 37.50% yellow wins, 62.50% blue wins, 0.00% draw and timeout, 0 hard violations and 0 own-target penalties, and the larger stochastic scoring run above measures side balance.

![ROS2 runtime evidence](./assets/readme/ros2_runtime_graph.png?raw=true)

## Replay and Evidence Guide

- `docs/admissions_project_brief.md` gives a concise English reviewer summary of the contribution and the evidence.
- `docs/getting_started.md` covers step-by-step environment setup, a quick demo, the ROS2 dry run, IsaacLab preview and troubleshooting.
- `docs/capabilities.md` lists the validated scope, measured metrics and Sim2Real evidence.
- `docs/architecture.md` covers the system architecture and the ROS2 and IsaacLab component interfaces.
- `docs/reproducibility.md` lists the exact smoke-test, ROS2 dry-run, IsaacLab preview and scoring commands.
- `docs/rules_summary.md` summarizes the public rules used by rule gates and replay checks.
- `docs/sim2real.md` documents sensor calibration, domain randomization and the deployment validation plan.

## Repository Layout

- `config/` holds the public rule, target-layout and scoring contract used by docs.
- `assets/readme/` holds GitHub README preview images.
- `crc_robocup_vision_ws/` is the ROS2 workspace for the competition robot.
- `isaaclab_sim/` holds the IsaacLab arena, rule simulation, and RL training interfaces.
- `docs/` holds architecture, Sim2Real, migration, and result notes.
- `THIRD_PARTY_NOTICES.md` lists dependency and mesh attribution notes.
