# Environment Setup Summary

This package collects the gate-only minimal replay code prepared under Windows and PowerShell. The pure 2D logic, tests, and CSV and JSON metric handling run on the CPU stack. IsaacLab and Isaac Sim serve the 3D replay and MP4 rendering paths.

## Verified Environment

- OS, Microsoft Windows 11 Pro `10.0.26100`
- Shell, PowerShell
- Python, `python`
- Python version, `Python 3.13.5`
- Project root, `<gate_graph_2d_minimal>`

## Core Dependency Versions

```text
numpy==1.26.4
torch==2.7.0+cu128
matplotlib==3.10.0
pandas==2.2.3
scipy==1.15.3
pytest==8.3.4
gymnasium==1.2.3
networkx==3.4.2
```

## Basic Launch

```powershell
cd <gate_graph_2d_minimal>
python -m pytest tests
```

The 3D replay and video export paths expect a configured NVIDIA Isaac Sim and IsaacLab install with `assets/gate/gate.usd` and `assets/gate/gate.glb` reachable from the project.

## Directory Policy

- The code keeps gate-only tasks.
- Scoring artifacts live under `artifacts/evaluation/`.
- Run outputs default to `outputs/` or the `--output-dir` and `--output-root` flags.
- Intermediate training config files stay outside the artifact set, and deterministic replay uses the entry commands and seeds recorded in `reproducibility.md`.
