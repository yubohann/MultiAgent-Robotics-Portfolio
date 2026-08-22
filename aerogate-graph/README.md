# AeroGate Graph

**A modular 2D drone-racing simulator for graph-based route planning, formation control, and dynamic gate navigation.**

AeroGate Graph is an English, runnable research codebase for fixed-height drone racing.
It provides single-agent and variable-size multi-agent environments, deterministic gate
layouts, dynamic gate-density curricula, graph observations, virtual-structure formation
control, global route planning, safety shields, evaluation scripts, and optional Isaac Lab
visualization adapters.

The repository consolidates the original gate-graph research workspace into one coherent
project root, including the complete working codebase, retained assets, release
checkpoints, and evaluation artifacts.

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
