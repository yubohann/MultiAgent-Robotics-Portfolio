# Methodology

This document describes the research framework of the project at the level of
method design — the questions, the approach, and the evaluation discipline.
Concrete implementation details and quantitative results are withheld until
the associated paper is published.

## 1. Research questions

1. **Structure matters.** Does an explicitly structural representation of the
   gate field enable a policy that generalizes better than a policy that only
   reacts to local observations?
2. **Guidance accelerates.** Can a global route hint, derived from the
   structure, make learning dramatically more sample-efficient in dense
   layouts where reward is sparse relative to the action space?
3. **Coordination composes.** Can the single-agent solution be extended to a
   team of drones that traverse the same field concurrently, with a
   configurable degree of formation coupling?

## 2. Approach framework

The framework is built from five interacting layers.

### 2.1 Structural encoding

The gate field is encoded as a graph. Nodes carry the essential spatial state
of each gate; edges encode feasible traversal transitions under the dynamics
of the platform. The encoding is deliberately *planning-friendly*: a path
through the field corresponds to a walk over the graph, and the policy can
attend to relevant structure rather than processing a raw coordinate list.

### 2.2 Graph-structured policy

The policy consumes the graph-structured state and produces an action
distribution. Its internal mechanism is designed around the structure: it
aggregates information across the graph before deciding, so decisions about
"which gate to fly toward" and "how to avoid obstacles" are made with
structural awareness.

### 2.3 Expert guidance and imitation

A global route plan provides a reference trajectory that is injected into
learning. An imitation stage bootstraps the policy from expert-guided
rollouts before reinforcement fine-tuning. The guidance channel and the
learning channel are decoupled so that the policy can later exceed the
guidance quality.

### 2.4 Curriculum and training pipeline

Training progresses through controlled stages so that difficulty increases
gradually along axes that matter: gate count, density, and motion. The
pipeline treats reward computation, replay, evaluation, and checkpoint
selection as separate, swappable stages rather than a monolithic loop.

### 2.5 Multi-agent extension

The multi-agent layer reuses the same structural encoding. Each agent holds a
role within the team, and coordination is expressed through the shared graph
plus a formation coupling term. Safety shielding prevents invalid actions
during both training and evaluation.

## 3. Evaluation methodology

Evaluation follows the same discipline across every stage:

- **Configuration-driven**: every experiment is described by an explicit
  configuration; no hard-coded experiment lives in the code.
- **Seed-controlled**: seeds are threaded through environment, policy,
  replay, and trainer for reproducible runs.
- **Selection separated from reporting**: checkpoint selection happens on a
  validation process; the final evaluation is a separate, post-selection
  stage.
- **Ablation by component**: each layer of the framework can be turned off
  independently (guidance, imitation, structure, coordination) so that its
  contribution is measured in isolation.
- **Planar first, physical second**: the planar abstraction is the
  experimental workhorse; a 3D physical replay validates transfer on
  representative scenes.

## 4. Claims and evidence boundaries

- This repository claims a *method framework* and engineering discipline; it
  does **not** claim quantitative results (withheld until publication).
- Dataset-level numbers, scene statistics, and performance metrics are
  reported only in the paper.
- Reproduction of the full pipeline requires the withheld implementation,
  released with the paper.