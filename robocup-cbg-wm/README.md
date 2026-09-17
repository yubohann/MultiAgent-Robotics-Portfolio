# RoboCup CBG-WM

<p align="center">
  <img src="./assets/readme/overview.png" alt="RoboCup vision robot platform, ROS 2 stack and competition outcomes" width="92%" />
</p>

<p align="center">
  <a href="./docs/media/large_scale_50v50_isaaclab_preview.gif">
    <img src="./docs/media/large_scale_50v50_isaaclab_preview.gif" alt="IsaacLab 50v50 replay preview" width="86%" />
  </a>
</p>

<p align="center">
  <a href="./docs/media/large_scale_50v50_isaaclab_replay.mp4">Full IsaacLab replay MP4</a>,
  <a href="./docs/media/最终回放_三视角同步拼接版.gif">Synchronized three-view replay GIF</a>,
  <a href="./docs/media/README.md">Media notes</a>
</p>

**An uncertainty-aware belief-graph world model for rule-gated multi-robot tactics, built in ROS 2 and IsaacLab.**

CBG-WM is a robotics research artifact for a RoboCup-style adversarial match. Two differential-drive robots share a 3 m by 3 m arena, hunt opponent targets through a 0.80 s laser dwell gate and a 5 cm to 50 cm normal-target range, push rigid red boxes and open base armor plates in order. The hard part is tactics under rules. Detections go stale behind cover, movable boxes change line of sight, armor state decides which base shots are legal, and referee events rewrite object presence. CBG-WM keeps typed belief tokens for every object, predicts interaction dynamics with a stochastic ensemble, and ranks joint Flow proposals by lower-tail risk before a geometry-aware shield executes the first action.

**Status.** `v0.1.0` research artifact. The validated main line is the two-robot match with a 128-episode scoring run, an 8-episode strict replay audit, three-view IsaacLab media and 1v1 real-robot experiment coverage. The 50v50 rule benchmark is a simulation-stage extension.

## Verified Evidence

### Two-Robot Match

The 128-episode stochastic scoring run reports balanced sides and clean contacts.

| Metric | Value |
| --- | ---: |
| Episodes | 128 |
| Yellow win rate | 49.22% |
| Blue win rate | 50.78% |
| Draw rate | 0.00% |
| Static penetrations | 0 |
| Box penetrations | 0 |
| Robot contacts per game | 0.00 |

The strict replay audit replays the selected checkpoint step by step against rule and physics invariants. Eight episodes report 37.50% yellow wins, 62.50% blue wins, zero hard violations, zero own-target penalties and 1.0000 base wins per episode.

### Large-Scale 50v50 Extension

The extension trains a staged curriculum from 5v5 to 10v10 to 25v25 to 50v50 and scores the final stage over 256 games with 50 vehicles per side in an 80 m by 50 m arena. The accepted trace replays in IsaacLab with 100 vehicle-shaped actors.

| Metric | Value |
| --- | ---: |
| Final-stage training episodes | 12000 |
| Scoring episodes | 256 |
| Yellow win rate | 36.72% |
| Blue win rate | 42.19% |
| Draw rate | 21.09% |
| Yellow base damage | 44.90 |
| Blue base damage | 44.89 |
| Robot contacts, mean and P95 | 0.00 and 0.00 |
| Obstacle contacts | 0.00 |

[50v50 IsaacLab replay MP4](./docs/media/large_scale_50v50_isaaclab_replay.mp4)

![50v50 rule layout](./docs/figures/large_scale_50v50/large_scale_50v50_rule_layout.png)

![50v50 training curve](./docs/figures/large_scale_50v50/large_scale_50v50_training.png)

![World-model SAC Flow training curve](./docs/figures/rl/rl_training_curve_gpu.svg)

### Runtime Stack

The ROS 2 Jazzy workspace carries `rcvrl_bringup`, `rcvrl_behavior`, `rcvrl_vision`, `rcvrl_navigation`, `rcvrl_motion`, `rcvrl_shooter`, `rcvrl_description` and `rcvrl_interfaces`. Nav2 owns navigation, `slam_toolbox` owns mapping, `rcvrl_vision` detects AprilTag Tag36h11 targets, and shooter commands pass through ROS 2 services behind an opponent-target safety gate.

[RoboCup VisionRL runtime and demo video](https://www.bilibili.com/video/BV1Pj9ZBKEc8/?spm_id_from=333.1387.list.card_archive.click&vd_source=f79b94dd69d0c8d08ee5c3400b69d46d)

## Inside the Model

- Typed belief tokens carry pose, velocity, attributes, extent, visibility, age and covariance. Occluded objects keep their last belief and accumulate uncertainty.
- A typed interaction graph covers observation, contact, route blocking, base protection, threat, proximity and line of sight, and it stays equivariant under token permutation.
- A stochastic ensemble predicts state deltas, reward, termination and four rule-risk channels, which are robot collision, blocked motion, illegal fire and line-of-sight violation.
- Flow-proposal CVaR MPC ranks joint candidate sequences by lower-tail return, predicted rule risk and ensemble disagreement, then executes the first action through the existing action shield.

The method note is in [CBG-WM](docs/cbg_wm.md).

![Method architecture](./docs/figures/paper/fig02_method_architecture.png)

## Platform

![Robot sensor layout, coordinate frames and actuator interfaces](./assets/readme/robot_sensor_layout.png)

![RoboCup field rule scene](./assets/readme/arena_rule_scene.png)

![ROS 2 runtime graph](./assets/readme/ros2_runtime_graph.png)

## Quick Start

Python 3.10 or newer for the rule environment and tests.

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

Optional training and rendering stack.

```bash
python -m pip install -e ".[training]"
```

ROS 2 workspace on Ubuntu 24.04 with ROS 2 Jazzy.

```bash
cd crc_robocup_vision_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch rcvrl_bringup competition.launch.py
```

Hardware-free launch smoke test.

```bash
ros2 launch rcvrl_bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

The yellow and blue elimination launches pick their side and route file.

```bash
ros2 launch rcvrl_bringup competition.launch.py team_color:=yellow target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.yellow.yaml
ros2 launch rcvrl_bringup competition.launch.py team_color:=blue target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.blue.yaml
```

ROSIDL generation needs an ASCII build path, so WSL users copy the workspace to a native Linux location first.

Rule environment smoke run.

```bash
python -m pip install -r isaaclab_sim/rl/requirements.txt
cd isaaclab_sim/rl
python evaluate_selfplay.py --episodes 8
```

IsaacLab preview on Windows runs through the project wrapper, which keeps Kit config, logs, pip envs and extension cache under `.isaaclab_runtime/` inside the project.

```powershell
.\scripts\run_isaaclab_project.ps1 -Headless -DemoFlow -Duration 120
```

Inspect or stop this project's IsaacLab processes.

```powershell
.\scripts\stop_project_isaaclab.ps1 -WhatIfOnly
.\scripts\stop_project_isaaclab.ps1
```

## Documentation

- [Getting started](docs/getting_started.md) covers Python, ROS 2 and IsaacLab setup, a quick demo and common issues.
- [Capabilities and measured evidence](docs/capabilities.md) lists the validated scope and published metrics.
- [Architecture](docs/architecture.md) covers the runtime graph, packages, topics and the competition state machine.
- [CBG-WM method](docs/cbg_wm.md) covers belief tokens, graph dynamics, training and scoring commands.
- [Replay guide](docs/reproducibility.md) lists the smoke-test, scoring, export and replay commands.
- [Competition rules](docs/rules_summary.md) summarizes the public 2025 rule set used by the rule gates.
- [Sim2Real](docs/sim2real.md) documents calibration, domain randomization and the deployment validation ladder.
- [Admissions project brief](docs/admissions_project_brief.md) is a short reviewer-oriented summary.

## License

Repository-authored code and documentation follow [LICENSE](LICENSE). Third-party dependencies and meshes keep their own terms, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
