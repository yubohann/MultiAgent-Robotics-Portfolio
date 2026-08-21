# AeroGate Graph Release Bundle

This directory contains runtime scripts for checkpoint-driven `aerogate_graph` evaluation. The scenario runner accepts a local model pack without embedding large checkpoints in the source tree.

## Contents

- `setup_env.ps1`: checks Python dependencies and runs the smoke tests.
- `run_gate_scenarios.ps1`: runs single-agent and multi-agent static/dynamic gate scenarios.
- `requirements-runtime.txt`: minimal Python dependencies for the gate-only evaluation scripts.
- `models/`: optional default location for a local checkpoint pack.

Generated outputs are written to `outputs/` when `run_gate_scenarios.ps1` is executed.

## Setup

```powershell
cd <aerogate_graph>
.\release_bundle\setup_env.ps1
```

To select a specific Python executable:

```powershell
.\release_bundle\setup_env.ps1 -Python python
```

To create a local virtual environment under the bundle:

```powershell
.\release_bundle\setup_env.ps1 -CreateVenv
```

## Run Scenarios

```powershell
cd <aerogate_graph>
.\release_bundle\run_gate_scenarios.ps1 -Scenario all -ModelRoot D:\aerogate-models
```

Run one scenario at a time:

```powershell
.\release_bundle\run_gate_scenarios.ps1 -Scenario single_static -ModelRoot D:\aerogate-models
.\release_bundle\run_gate_scenarios.ps1 -Scenario single_dynamic -ModelRoot D:\aerogate-models
.\release_bundle\run_gate_scenarios.ps1 -Scenario multi_static -ModelRoot D:\aerogate-models
.\release_bundle\run_gate_scenarios.ps1 -Scenario multi_dynamic -ModelRoot D:\aerogate-models
```

Useful options:

```powershell
.\release_bundle\run_gate_scenarios.ps1 -Scenario multi_dynamic -Episodes 3 -Seeds 0,1 -Device cuda -ModelRoot D:\aerogate-models
.\release_bundle\run_gate_scenarios.ps1 -Scenario single_dynamic -SingleDynamicGateCount 42 -Device cuda -ModelRoot D:\aerogate-models
```

The model root must contain `single_gate_density`, `multi_static_gate60`, and `multi_dynamic_c4a_24gate` subdirectories with their expected checkpoint files. Set `AEROGATE_MODEL_ROOT` to reuse the same pack across repeated runs.
