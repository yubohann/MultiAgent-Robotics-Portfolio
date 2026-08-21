# AeroGate Graph

**A method framework for graph-based reinforcement learning in dense, dynamic drone gate traversal.**

Quadrotors that must fly through dense, moving gate layouts face a decision problem that is fundamentally *structural*: which gates can be reached next, in what order, and how the current motion interacts with the surrounding gate field. This project frames that problem as a **graph** — gates become graph nodes, and feasible traversal transitions become edges — and learns policies directly over that structure.

This repository documents the **method framework**: the problem formulation, the system architecture, the learning pipeline, and the engineering principles. The concrete implementation of the research contributions is intentionally withheld until the associated paper is published.

> **Status**: research-stage. Core algorithm implementation released with the paper. See [NOTICE.md](NOTICE.md).

> **Publication status**: the associated paper is under review and has **not
> been published yet**. To protect the work, quantitative results and the core
> implementation are **not shown** in this repository. They will be released
> publicly together with the paper once it is out.

> **Open-source plan**: full open-source (complete implementation + results)
> is planned for the **end of October 2026**.

---

## Problem framework

| Layer | Question the framework addresses |
| --- | --- |
| **Perception of structure** | How should a dense, moving gate field be represented so a policy can reason over traversal order, not just react locally? |
| **Single-agent decision** | How should one drone learn a policy that generalizes across gate count, density, and motion patterns? |
| **Expert guidance** | How can a global route hint accelerate and stabilize learning in high-density layouts? |
| **Multi-agent coordination** | How do multiple drones traverse the same field without conflict, with or without a formation constraint? |
| **Transfer** | How does a policy learned in a planar abstraction transfer to a 3D physical simulation? |

## Method framework

```text
  Gate field (dense / dynamic)
              │
              ▼
  ┌───────────────────────────┐
  │  Structural encoding      │   gates → nodes, feasible transitions → edges
  └───────────────────────────┘
              │
              ▼
  ┌───────────────────────────┐
  │  Graph-structured policy  │   action distribution conditioned on the graph
  └───────────────────────────┘
              │
              ├───────────────►  Expert guidance (global route hint)
              │                              │
              ▼                              ▼
  ┌───────────────────────────┐   ┌───────────────────────────┐
  │  Learning pipeline        │◄──│  Imitation bootstrap       │
  │  (reward, replay,         │   │  (expert-guided rollout)   │
  │   curriculum, selection)  │   └───────────────────────────┘
  └───────────────────────────┘
              │
              ├───────────────►  Multi-agent extension (formation + coordination)
              ▼
  ┌───────────────────────────┐
  │  Evaluation & transfer    │   2D evaluation → 3D physical replay
  └───────────────────────────┘
```

## What is included in this repository

- **The framework documentation** — problem formulation, architecture, pipeline design, and evaluation methodology.
- **Neutral engineering primitives** — planar math, kinematics, and collision utilities under `shared/core/`.
- **The withheld API surface** — `core/` declares the role and interface of each research component; implementations ship with the paper.

## What is intentionally withheld

Per [NOTICE.md](NOTICE.md), the following are **not** distributed in this public repository:

- The graph encoder / message-passing scheme and policy architecture.
- Reward design and shaping terms.
- Training schedule, curriculum rules, and checkpoint-selection logic.
- The expert-guidance mechanism and route-planning internals.
- Experimental configuration values and all quantitative results.

These details stay out of the public repository until the paper is out, so the
repo is safe to share with anyone who wants to understand the framework.

---

## Repository layout

```text
core/                     Withheld research components (API surface + role documentation)
shared/
  core/                   Neutral planar primitives: math, kinematics, collision
tests/                    Test suite for the included neutral primitives
docs/
  methodology.md          Research framework: questions, approach, evaluation
  architecture.md         Component architecture and design rationale
NOTICE.md                 Redaction notice
```

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

The test suite covers the neutral planar primitives and needs no ML runtime.

## Engineering principles

- **Structural first** — encode the environment's structure explicitly instead of flattening it.
- **Guidance over brute force** — use global structure to shape exploration rather than searching blindly.
- **Single to multi** — build the coordination layer on the same structural representation.
- **Validation discipline** — evaluation is a first-class pipeline stage, not an afterthought.
- **Reproducibility** — deterministic seeding and configuration-driven experiments throughout.

---

*Bohan Yu — research-stage project. Core implementation released with the associated paper.*