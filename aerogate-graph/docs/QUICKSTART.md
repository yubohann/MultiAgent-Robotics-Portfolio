# Quick Start

AeroGate Graph has two dependency tiers:

- **Core:** NumPy-backed 2D environments, planners, layouts, and tests.
- **Training and visualization:** PyTorch for Graph-SAC and Graph-MASAC; Isaac Lab only for
  the dedicated simulator rendering and replay adapters.

## Core Environment

For a locked reference environment, install `uv` and run:

~~~powershell
uv sync --extra dev
uv run python -m aerogate info
uv run python -m aerogate smoke --scenario single-static --steps 8
uv run python -m aerogate smoke --scenario multi-static --agents 4 --steps 8
uv run python -m aerogate reproduce --scenario multi-static --agents 4 --seeds 3 7 11 --steps 8
~~~

If `uv` is unavailable, the following `pip` workflow remains suitable for local exploration:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m aerogate info
python -m aerogate smoke --scenario single-static --steps 8
python -m aerogate smoke --scenario multi-static --agents 4 --steps 8
~~~

The smoke command steps the real environment with zero actions. It verifies imports,
configuration, shape contracts, and environment state transitions without training.
The reproduction command runs each listed seed twice and compares public rollout diagnostics.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the evidence format and its scope.

## Training Dependencies

~~~powershell
python -m pip install -e ".[rl,dev]"
python single_gate/scripts/train_single.py --help
python multi_gate/scripts/train_multi.py --help
~~~

The training scripts expose task-specific arguments. Start with their help output and use
a dedicated output directory; generated outputs are ignored by Git.

## Release Evaluation

For checkpoint-driven release evaluation, place a local checkpoint pack under a model root and pass it explicitly:

~~~powershell
.\release_bundle\setup_env.ps1 -Python python
.\release_bundle\run_gate_scenarios.ps1 -Scenario single_static -Device cpu -ModelRoot D:\aerogate-models
~~~

Alternatively, set `AEROGATE_MODEL_ROOT` once and omit `-ModelRoot` from subsequent commands.
