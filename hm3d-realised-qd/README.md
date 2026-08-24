# HM3D Realised-QD for Multi-UAV Exploration

<p align="center">
  <img src="assets/demos/hm3d-scene-1.gif" alt="HM3D multi-UAV exploration scene" width="78%" />
</p>

<p align="center"><em>Multi-UAV exploration in an HM3D-derived indoor scene.</em></p>

[English](README.md) | [简体中文](README.zh-CN.md)

Outcome-grounded quality-diversity and reinforcement learning for target-free multi-UAV exploration in HM3D-derived 3D environments.

This project studies cooperative exploration under real four-rotor execution constraints. Agents build local belief from public sparse-range observations, share a bounded candidate pool, execute guarded team plans, and update behavioural diversity only from receipts produced by real execution.

The active task has no manually placed targets, target counts, confirmed-recall objective, or formal RGB-D contract. The primary metric is `Explored-Free-Flight-Volume-AUC_time` under a shared CF2X, communication, safety, and physical-time contract.

## Repository Map

```text
src/aerocity_method/  contracts, adapters, realised-QD, RL, runtime, safety, evaluation
configs/              HM3D protocols and experiment contracts
scripts/              assembly, audits, training, replay, and Isaac launch wrappers
tests/                unit, property, leakage, performance, and integration tests
docs/                 active research plans and execution contracts
```

## Quick Start

```powershell
uv sync --extra dev --extra rl --extra hm3d
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

Isaac/PhysX runs must use `scripts/run_isaac_python.ps1` and an explicitly verified IsaacLab interpreter. Set `AEROCITY_CF2X_USD` for the local CF2X USD asset. HM3D assets, converted meshes, private evaluation data, checkpoints, raw logs, and runtime outputs remain outside the public source boundary.

The current research status and formal P01-P10 boundaries are documented in [docs/README.md](docs/README.md). The Chinese companion is [README.zh-CN.md](README.zh-CN.md).

## Release Boundary

The source tree retains its project-specific release terms. Third-party code and assets retain their own licenses.
