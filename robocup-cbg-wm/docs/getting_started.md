# Getting Started Guide

This guide gives a minimal path for running the public repository evidence. The project can be inspected at three levels, Python rule tests, ROS2 dry run, and IsaacLab replay.

## 1. Repository Scope

Main directories.

| Path | Purpose |
| --- | --- |
| `ros_ws/` | ROS2 Jazzy workspace for robot bringup, navigation, vision, behavior, shooter and interfaces |
| `sim/` | IsaacLab scene, replay utilities, rule environment and RL tooling |
| `sim/rl/` | Self-play environments, world-model SAC Flow training, scoring and export scripts |
| `config/` | Public arena, target layout and scoring contracts |
| `docs/rl_data/` | Published training summaries, scoring JSON and CSV data and replay audits |
| `docs/assets/` | Final MP4 and GIF replay media |
| `tests/` | Pytest checks for rule contracts, target layout, strategy logic and Sim2Real configuration |

## 2. Environment Levels

### Level 0 Python-Only Smoke Test

Use this level to check the rule environment and scoring utilities.

Requirements.

- Python 3.10 or newer
- `pip`
- Optional CUDA PyTorch for training, with CPU sufficient for many smoke tests

Commands.

```bash
python -m pip install -r sim/rl/requirements.txt
python -m pytest tests -q
```

Quick rule-environment scoring run.

```bash
cd sim/rl
python evaluate_selfplay.py --episodes 8
```

Expected behavior.

- tests complete and satisfy the rule contracts.
- the scoring run prints match score, winner, target hits, base hits, collision and penetration fields.
- this level runs from the Python package alone.

### Level 1 ROS2 Dry Run

Use this to check that the ROS2 workspace builds and the launch files start on software alone.

Recommended platform.

- Ubuntu 24.04
- ROS2 Jazzy
- `colcon`
- `rosdep`

ROSIDL generation needs a build path with ASCII characters, so a WSL workspace is copied to a native Linux path such as `~/ros_ws` before building.

Commands.

```bash
cd ~/ros_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

Useful launch variants.

```bash
ros2 launch bringup competition.launch.py team_color:=yellow target_file:=$(ros2 pkg prefix navigation)/share/navigation/config/targets.elimination.yellow.yaml
ros2 launch bringup competition.launch.py team_color:=blue target_file:=$(ros2 pkg prefix navigation)/share/navigation/config/targets.elimination.blue.yaml
```

Expected behavior.

- bringup, behavior, navigation, vision and shooter service nodes can be launched.
- shooter can run in dry-run mode.
- the dry run needs software alone.

### Level 2 IsaacLab Replay

Use this to inspect the published replay behavior. IsaacLab and Isaac Sim setup is heavier than the Python rule tests, so start here after Level 0 works.

On Windows, use the project wrapper, which keeps runtime files under `.isaaclab_runtime/` inside the project.

```powershell
.\scripts\run_isaaclab_project.ps1 -Headless -DemoFlow -Duration 120
```

To inspect or stop this project's IsaacLab processes.

```powershell
.\scripts\stop_project_isaaclab.ps1 -WhatIfOnly
.\scripts\stop_project_isaaclab.ps1
```

Published replay media.

```text
docs/assets/最终回放_三视角同步拼接版.gif
```

The compact GitHub checkout carries the three-view GIF, and the individual MP4 source views are generated locally.

## 3. Training and Scoring

The public training and scoring artifacts are already included under `docs/rl_data/`. To regenerate them, use the commands in `docs/reproducibility.md`.

Important generated-output rules.

- local training outputs go under `sim/output/`.
- temporary videos, cache files and debug frames stay local, and selected final evidence is committed.
- public claims point to JSON and CSV metrics, replay audits and MP4 or GIF files.

## 4. First Files to Read

Recommended order.

1. `README.md`
2. `docs/admissions_project_brief.md`
3. `docs/capabilities.md`
4. `docs/reproducibility.md`
5. `docs/cbg_wm.md`
6. `docs/sim2real.md`

## 5. Common Issues

### ROS2 build under WSL

Move the workspace to a native Linux path.

```bash
cp -r /mnt/c/path/to/ros_ws ~/ros_ws
```

Then rebuild from `~/ros_ws`.

### IsaacLab runtime path conflicts

Use `scripts/run_isaaclab_project.ps1`, which sets project-local runtime paths. Inspect running processes with `scripts/stop_project_isaaclab.ps1 -WhatIfOnly` before stopping anything.

### Checking a surprising score

Check the published artifacts alongside the reward signal.

- `docs/capabilities.md`
- `docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json`
- `docs/rl_data/world_model_sacflow_final/strict_replay_summary.json`
- replay MP4 and GIF files under `docs/assets/`

### 50v50 evidence status

The 50v50 result is simulation-stage rule-level evidence with an IsaacLab tactical replay.
