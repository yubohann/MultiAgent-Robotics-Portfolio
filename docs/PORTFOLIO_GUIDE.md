# Portfolio Guide

[English](PORTFOLIO_GUIDE.md) | [简体中文](PORTFOLIO_GUIDE.zh-CN.md)

## How to Read This Repository

The root README is an index, not an aggregate claim that every subproject has the same maturity. Open a project README before drawing conclusions about its code, results, data, or hardware status. The machine-readable [project registry](../tools/portfolio_registry.json) defines the curated project set, entry documents, and root-level check keys.

| Evidence label | Meaning |
|---|---|
| Framework | Public architecture, interfaces, documentation, and non-sensitive utilities. It does not imply that a withheld method has been evaluated. |
| Simulation or replay | Evidence generated in the documented simulated or replayed environment. It is not hardware evidence. |
| Hardware | Evidence limited to the hardware setup, protocol, and scope identified in the relevant project documentation. |
| Coursework | Educational implementation or lab artifact. It is not represented as production software or a research benchmark. |

## Project Entry Points

| Goal | Project | First document |
|---|---|---|
| Inspect a benchmark contract and audit boundary | Rivermark | [README](../rivermark/README.md) |
| Inspect visual robotics and replay evidence | RoboCup CBG-WM | [README](../robocup-cbg-wm/README.md) |
| Inspect ROS 2 localization and competition control | Robocon MID-360 Autonomy Stack | [README](../robocon-mid360-autonomy-stack/README.md) |
| Inspect procedural 3D multi-UAV search benchmarking | AeroCityBench | [README](../aerocity-bench/README.md) |
| Inspect outcome-grounded HM3D exploration and realised-QD | HM3D Realised-QD | [README](../hm3d-realised-qd/README.md) |
| Inspect graph-based drone racing, formation, and safety tooling | AeroGate Graph | [Overview](../aerogate-graph/README.md) · [Architecture](../aerogate-graph/docs/architecture.md) |
| Inspect graph-and-sequence fraud-detection training and experiment tooling | FraudGraph ML Engineering | [Overview](../fraudgraph-ml-engineering/README.md) · [Experiment catalog](../fraudgraph-ml-engineering/docs/experiment-catalog.md) |
| Browse compact learning artifacts | Coursework | [Coursework index](../coursework/machine-learning/README.md) |

## Local Checks

Run the root portfolio check after editing entry documents:

```bash
python tools/verify_portfolio.py
python tools/run_portfolio_checks.py
```

The first command validates the registry and curated entry documents. The second runs the dependency-light verification path for each registered core project while keeping training, ROS, Isaac, and external-data workflows separate.

## Change Discipline

- Keep a project's stated evidence level aligned with the files it exposes.
- Do not convert a framework description into a performance claim without a matching result artifact and scope statement.
- Do not move private data, bags, maps, weights, credentials, or restricted competition material into the public tree.
- Treat modifications to experiment parameters, seeds, evaluation order, or reported metrics as experiment changes, not documentation cleanup.
