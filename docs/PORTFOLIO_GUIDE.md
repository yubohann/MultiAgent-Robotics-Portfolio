# Portfolio Guide

## How to Read This Repository

The root README is an index, not an aggregate claim that every subproject has the same maturity. Open a project README before drawing conclusions about its code, results, data, or hardware status.

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
| Inspect graph-based drone racing, formation, and safety tooling | AeroGate Graph | [Overview](../aerogate-graph/README.md) · [Architecture](../aerogate-graph/docs/ARCHITECTURE.md) |
| Inspect graph-and-sequence fraud-detection training and experiment tooling | FraudGraph ML Engineering | [Overview](../fraudgraph-ml-engineering/README.md) · [Experiment catalog](../fraudgraph-ml-engineering/docs/experiment-catalog.md) |
| Browse compact learning artifacts | Coursework | [Coursework index](../coursework/machine-learning/README.md) |

## Local Checks

Run the root portfolio check after editing entry documents:

```bash
python tools/verify_portfolio.py
```

It intentionally checks only the curated entry documents. Each project owns its own build, dependency, simulation, and test commands; this avoids claiming that a single root command validates unrelated runtimes.

## Change Discipline

- Keep a project's stated evidence level aligned with the files it exposes.
- Do not convert a framework description into a performance claim without a matching result artifact and scope statement.
- Do not move private data, bags, maps, weights, credentials, or restricted competition material into the public tree.
- Treat modifications to experiment parameters, seeds, evaluation order, or reported metrics as experiment changes, not documentation cleanup.
