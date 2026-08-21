# Architecture

This document describes the component architecture and the design rationale.
Implementation details of research components are withheld; the architecture
shows *how the pieces fit together* and *why*.

## System context

```
┌────────────────────────────────────────────────────────────────────┐
│                        AeroGate Graph framework                     │
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Environment  │───▶│   Learning   │───▶│   Evaluation &       │  │
│  │  abstraction  │    │   pipeline   │    │   transfer           │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                  ▲  │                    │               │
│         ▼                  │  ▼                    ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Structure    │    │  Expert       │    │  Multi-agent         │  │
│  │  encoding     │◀──▶│  guidance     │    │  coordination        │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

## Component responsibilities

| Component | Responsibility | Withheld? |
| --- | --- | --- |
| **Environment abstraction** | Planar state, dynamics integration, gate-field dynamics, observation assembly, action interface. | Partial |
| **Structure encoding** | Gates → graph nodes; feasible transitions → edges. | Yes |
| **Graph-structured policy** | Action distribution conditioned on the graph. | Yes |
| **Reward model** | Dense learning signal for efficient, safe traversal. | Yes |
| **Expert guidance** | Global route hints injected during learning. | Yes |
| **Imitation bootstrap** | Expert-guided rollout to initialize the policy. | Yes |
| **Training pipeline** | Replay, curriculum, updates, checkpoint selection, logging. | Yes |
| **Multi-agent coordination** | Team-role assignment, formation coupling, safety shielding. | Yes |
| **Evaluation & transfer** | Validation process, held-out evaluation, 3D physical replay. | Partial |

## Key design decisions

1. **Structure is a first-class input.** Instead of flattening the gate field
   into a feature vector, the framework keeps the graph structure throughout
   the decision process. This is the central design idea and the reason the
   project is named *AeroGate Graph*.

2. **Guidance is decoupled from the learner.** The route hint is produced by
   a separate mechanism and consumed as auxiliary information. This keeps the
   policy capable of improving beyond the guidance quality.

3. **Everything is configurable.** No experimental value lives in code.
   Configurations are explicit and versioned, which is what makes the
   ablation methodology possible.

4. **Safety is a runtime concern, not a training afterthought.** Invalid
   actions are prevented by shielding at both training and evaluation time,
   so reported behaviour never depends on "getting lucky" with the policy.

5. **Single-agent first, multi-agent by extension.** The multi-agent layer
   reuses the single-agent structural representation rather than introducing
   a separate one, which keeps the coordination problem tractable.

## Planar-to-physical transfer

The planar abstraction is the experimental workhorse. A 3D physical replay
stage consumes the same policy interface and replays behaviour in a physical
simulation on representative scenes. The transfer path is:

```
Planar policy → same interface → 3D scene → physical replay → inspection
```

This keeps the simulation gap explicit and measurable instead of implicit.

## Testing strategy

- **Dependency-light tests** — the included test suite exercises neutral
  primitives and needs no ML runtime, so the repository can be validated
  anywhere.
- **Import discipline** — every module is importable and compiles cleanly.
- **Smoke coverage** — environments and configs have smoke-level coverage at
  the framework level; full algorithmic coverage ships with the withheld
  implementation.