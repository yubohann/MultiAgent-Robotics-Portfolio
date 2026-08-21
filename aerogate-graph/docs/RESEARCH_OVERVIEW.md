# Research Overview

## Research Focus

AeroGate Graph is a compact research environment for studying how a team of drones can
traverse gate courses while balancing route progress, formation coherence, and clearance.
It intentionally separates the deterministic 2D task from optional learning and rendering
stacks so that each systems question can be inspected without requiring a GPU or Isaac Lab.

The repository is suitable for controlled investigations such as:

- Does graph-structured state help a variable-size team retain formation through static or
  moving gate layouts?
- How do global route plans, local formation slots, and action-level safety corrections
  interact under the same reward and termination contract?
- How do gate density, moving-post speed, team size, and safety margins change completion,
  clearance, and formation error?

These are research questions supported by the codebase, not claims of state-of-the-art
performance or real-world flight readiness.

## Method Stack

The multi-agent task composes several independently inspectable mechanisms:

| Layer | Implementation | Research role |
| --- | --- | --- |
| State and geometry | `shared/core`, static and dynamic gate maps | Fixed-height kinematics, gate posts, collision and clearance contracts |
| Planning | `multi_gate/planners/global_route_planner.py` | Global route waypoints used as a geometric reference |
| Coordination | `multi_gate/formation/virtual_structure.py` | Team slots around a virtual center for variable team sizes |
| Safety | `multi_gate/env/safety_shields.py`, `shared/core/team_geometry.py` | Pairwise, boundary, obstacle, corridor, and gate-channel velocity corrections |
| Observation | `single_gate/env`, `multi_gate/env` | Padded graph features, adjacency, node masks, and action masks |
| Learning | `single_gate/graph_rl`, `multi_gate/graph_rl` | Graph-SAC and centralized-critic Graph-MASAC/FlashSAC paths |
| Evaluation | `aerogate`, task scripts, CI artifacts | Dependency-light smoke rollouts, deterministic reports, and task-specific evaluation drivers |

The graph policy path supports masked active agents, while the centralized multi-agent
critic consumes pooled graph state, joint actions, and the action mask. Optional behavior
cloning and DAgger utilities are kept separate from the core environment so an experiment
can state exactly which training path it used.

## Experiment Design

Use the three public scenario families as a progressive protocol:

1. `single-static` validates one-drone observation, kinematics, reward, and collision paths.
2. `multi-static` isolates variable-team graph observations, formation slots, global planning,
   and safety diagnostics without moving posts.
3. `multi-dynamic` activates the eight-drone moving-gate-density task and its corridor and
   dynamic-clearance checks.

For a comparison, record the scenario name, agent count, configuration revision, random
seeds, episode budget, policy checkpoint, device, and dependency lockfile revision. Keep
the public deterministic report beside result tables; it verifies the lightweight simulator
contract but is not a substitute for learned-policy evaluation.

## Metrics to Report

The task runtimes and evaluation drivers expose ingredients for a transparent scorecard. The
public rollout report includes end-of-rollout clearance for every scenario and adds minimum
pair distance plus mean/maximum slot error for multi-agent scenarios:

| Category | Examples |
| --- | --- |
| Task outcome | team or episode success rate, termination reason, timeout rate, progress distance |
| Safety | gate-post collision rate, inter-agent collision rate, minimum clearance, minimum pair distance |
| Coordination | mean and maximum slot error, lateral-band status, line-collapse diagnostics |
| Efficiency | episode return, action magnitude/smoothness, planner call count, planner latency |
| Dynamic-task behavior | moving-gate speed, gate count/density, corridor completion, dynamic gate clearance |

Report a distribution across explicit seeds rather than a single best episode. When comparing
policies, hold the geometry, team size, horizon, and safety configuration fixed unless the
ablation is specifically about one of those factors.

## Reproduction Boundary

The committed `uv.lock`, deterministic public rollout report, regression tests, and CI
artifact establish a reproducible **core-environment** baseline. The repository does not
claim bitwise-reproducible GPU training, simulator rendering equivalence, physical-drone
safety, perception performance, or a benchmark ranking. Those claims require their own
hardware disclosure, training seed protocol, evaluation budget, and safety validation.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for executable commands and
[ARCHITECTURE.md](ARCHITECTURE.md) for the implementation flow.
