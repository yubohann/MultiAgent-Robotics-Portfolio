# AeroGate Graph

<p align="center">
  <img src="assets/demos/formation-transition.gif" alt="Multi-UAV formation transition through a gate field" width="49%" />
  <img src="assets/demos/multi-uav-dynamic-obstacle-avoidance.gif" alt="Multi-UAV dynamic obstacle avoidance" width="49%" />
</p>

<p align="center"><em>Formation transitions and dynamic multi-UAV obstacle avoidance.</em></p>

[English](README.md) | [简体中文](README.zh-CN.md)

**A reproducible research environment for graph-based multi-UAV formation control, route planning, and dynamic gate navigation.**

AeroGate Graph separates a deterministic fixed-height 2D core from optional PyTorch training
and Isaac Lab replay. This makes route progress, formation error, clearance, and policy
behavior inspectable as separate evidence rather than one opaque demonstration.

The central research question is how a variable-size team can follow a global route while
preserving a deformable formation and respecting collision and gate-channel constraints.
The codebase combines graph observations, virtual-structure slots, A* route planning,
action-level safety shielding, Graph-SAC/Graph-MASAC learning paths, expert pretraining,
DAgger-style imitation, curriculum stages, and optional 3D replay.

The repository consolidates the original gate-graph research workspace into one coherent
project root, including the complete working codebase, retained assets, release
checkpoints, and evaluation artifacts.

## Evidence gallery

The following figures correspond to specific modeling, training, or evaluation claims in
the repository. They are organized as a trace from formation diagnostics to route execution,
learning design, and multi-metric evaluation.

### Formation-transition diagnostics

The four error curves show eight-UAV convergence for line-to-triangle, triangle-to-rectangle,
rectangle-to-diamond, and diamond-to-circle transitions. Each plot keeps per-agent error and
the team mean visible so transient formation cost is not hidden.

<p align="center">
  <img src="assets/formation-control/01_line_to_triangle_formation_error.png" alt="Line to triangle formation error" width="49%" />
  <img src="assets/formation-control/02_triangle_to_rectangle_formation_error.png" alt="Triangle to rectangle formation error" width="49%" />
  <img src="assets/formation-control/03_rectangle_to_diamond_formation_error.png" alt="Rectangle to diamond formation error" width="49%" />
  <img src="assets/formation-control/04_diamond_to_circle_formation_error.png" alt="Diamond to circle formation error" width="49%" />
</p>

### Route and simulation evidence

These panels connect the 2D task to the Isaac Lab adapter: complete per-drone routes,
four-stage formation transitions, fixed-height 3D geometry, and replay views from both the
scene and following-drone perspectives.

<p align="center">
  <img src="assets/formation-control/formation_routes_2d.png" alt="Complete 2D routes for each drone" width="49%" />
  <img src="assets/formation-control/formation_transition_isaaclab_3d_overview.png" alt="Isaac Lab three-dimensional formation transition overview" width="49%" />
  <img src="assets/formation-control/formation_stage_transitions_2d.png" alt="Two-dimensional formation transition stages" width="49%" />
  <img src="assets/formation-control/formation_stage_transitions_isaaclab_3d.png" alt="Isaac Lab three-dimensional formation transition stages" width="49%" />
  <img src="assets/formation-control/isaaclab_replay_stage_grid.png" alt="Isaac Lab replay stage grid" width="49%" />
  <img src="assets/formation-control/isaaclab_follow_view_stage_grid.png" alt="Isaac Lab replay from a following-drone view" width="49%" />
</p>

### Control and learning design

The diagrams make the control contract explicit: virtual-structure control maps team state
to slot targets; graph encoders aggregate typed node and edge information; centralized
critics evaluate joint actions; reward feedback and curriculum stages govern policy updates.

<p align="center">
  <img src="assets/formation-control/virtual_structure_formation_control.png" alt="Virtual-structure formation control pipeline" width="49%" />
  <img src="assets/formation-control/graph_flash_sac_architecture_overview.png" alt="Graph-FlashSAC overall architecture" width="49%" />
  <img src="assets/formation-control/graph_flash_sac_control_architecture.png" alt="Graph-FlashSAC network and training update flow" width="49%" />
  <img src="assets/formation-control/formation_reward_components_compact.png" alt="Compact reward component design" width="49%" />
  <img src="assets/formation-control/formation_reward_components_detailed.png" alt="Detailed reward feedback and policy update flow" width="49%" />
  <img src="assets/formation-control/curriculum_training_schedule.png" alt="Curriculum training schedule" width="49%" />
</p>

### Evaluation evidence

The multi-metric pressure heatmap summarizes how formation quality, safety, and task pressure
vary across evaluated stages. Read it alongside raw rollouts and seeded reports; it is not a
standalone performance claim.

<p align="center">
  <img src="assets/formation-control/multi_metric_pressure_heatmap.png" alt="Multi-metric pressure heatmap" width="76%" />
</p>

## Highlights

- Fixed-height 2D kinematics with velocity, acceleration, and collision limits.
- Static slalom and dynamic moving-gate scenarios with a shared geometry model.
- Variable-team graph observations, formation slots, safety shielding, and A* route planning.
- Graph-SAC and Graph-MASAC implementation paths, with behavior-cloning and DAgger utilities.
- Standalone NumPy environment tests plus optional PyTorch training and Isaac Lab replay adapters.
- Optional local asset and checkpoint packs for IsaacLab replay and checkpoint-driven evaluation.

## Quick Start

### Locked core setup (recommended)

Install [uv](https://docs.astral.sh/uv/) and use the committed lockfile:

~~~powershell
uv sync --extra dev
uv run python -m aerogate info
uv run python -m aerogate smoke --scenario single-static --steps 8
uv run python -m aerogate smoke --scenario multi-static --agents 4 --steps 8
uv run python -m pytest
~~~

Create a portable, deterministic core-environment evidence report:

~~~powershell
uv run python -m aerogate reproduce --scenario multi-static --agents 4 --seeds 3 7 11 --steps 8 --output artifacts/reproducibility/multi-static.json
~~~

The included core environments generate their own gate layouts and require no external dataset for smoke tests or deterministic reproduction checks.

The report records exact rollout diagnostics and core runtime versions. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the protocol, expected scope, and
interpretation limits.

### pip fallback

`pip` remains supported when uv is unavailable. Its version ranges are less exact than the
lockfile, so use it for local exploration rather than a reference environment:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m aerogate info
python -m aerogate smoke --scenario single-static --steps 8
python -m aerogate smoke --scenario multi-static --agents 4 --steps 8
python -m pytest
~~~

Install the training dependencies only when needed:

~~~powershell
python -m pip install -e ".[rl,dev]"
~~~

Isaac Lab integration is intentionally optional. Install Isaac Lab in its supported Python
environment first, then add this repository to PYTHONPATH or install it editable there.

## Project Layout

~~~text
AeroGateGraph/
├── aerogate/                 Public API, CLI, and scenario adapters
├── shared/                   Geometry, dynamics, common configuration, runtime helpers
├── single_gate/              Single-drone environment, policy, training, and replay
├── multi_gate/               Multi-drone environment, formation, planner, safety, and RL
├── gate_density_single/      Single-drone gate-density benchmark tooling
├── gate_density_multi_8/     Eight-drone dynamic-density curriculum tooling
├── single_internal_gate/     Planner baselines, policy arbitration, and safety methods
├── assets/                   Gate and drone scene assets
├── release_bundle/           Reproducible release commands and selected checkpoints
├── evaluation_artifacts/     Retained reports, manifests, and tabular results
├── tests/                    Deterministic regression and smoke tests
└── docs/                     Architecture, setup, and contribution documentation
~~~

See [Architecture](docs/architecture.md), the [research overview](docs/RESEARCH_OVERVIEW.md),
[Quick Start](docs/QUICKSTART.md), and [Contributing](CONTRIBUTING.md) for supported
boundaries and workflows.

## Scope and Safety

The primary environments are 2D, fixed-height research abstractions. Passing their tests
does not validate perception, flight dynamics, actuator behavior, or physical flight safety.
Use the Isaac Lab adapters and an appropriate simulation/flight-validation workflow before
making claims about a physical vehicle.

## License

Copyright (c) 2026 Bohan Yu. All rights reserved. See LICENSE.

The gate and drone assets are distributed under the BSD 3-Clause License from their
upstream project. See THIRD_PARTY_NOTICES.md for the complete attribution and terms.
