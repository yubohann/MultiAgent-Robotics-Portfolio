# Contributing to AeroGate Graph

## Development Setup

Use Python 3.10 or later. The core environments need NumPy; training paths also require
PyTorch. Prefer `uv sync --extra dev` to reproduce the locked core dependency set. Keep
Isaac Lab imports confined to the adapter and replay modules so core tests remain runnable
without a simulator installation.

~~~powershell
uv sync --extra dev
uv run python -m pytest
uv run ruff check aerogate shared/core/team_geometry.py multi_gate/env/reward_runtime.py tests
uv run ruff format --check aerogate shared/core/team_geometry.py multi_gate/env/reward_runtime.py tests
uv run python -m aerogate reproduce --scenario multi-static --agents 4 --seeds 3 7 11 --steps 8
~~~

Use `python -m pip install -e ".[dev]"` only when a locked `uv` environment is unavailable.

## Change Boundaries

- Keep gate geometry in shared/core and use it from both single- and multi-agent paths.
- Keep scenario-specific reward, observation, and training concerns inside their owning module.
- Do not add simulator-only imports to the core 2D environment, planner, or unit tests.
- Treat release_bundle/models and USD assets as Git LFS material; do not add them as regular Git blobs.
- Preserve deterministic seeds in benchmark scripts and tests when changing layouts or curriculum logic.

## Pull Requests

Each change should include a focused explanation, tests matching the behavioral risk, and
updated documentation when the public CLI, configuration, or release workflow changes.
