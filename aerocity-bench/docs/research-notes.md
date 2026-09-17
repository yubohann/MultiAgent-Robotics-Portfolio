# Research Notes

A working record of how the AeroCityBench pilot was designed, what has been built and what the current development results are. Every artifact named here carries a `formal_score_eligible` flag, and only frozen formal records count toward benchmark conclusions.

## Research Questions

- **RQ1.** How strongly does spatial coverage predict confirmed-target recall once occlusion, viewpoint and dwell constraints apply?
- **RQ2.** When coverage and confirmed recall diverge, which mechanisms dominate: wrong facade, blocked line of sight, missed dwell window, height miss, or duplicate effort?
- **RQ3.** Do topology, scale, target-process and fleet-degradation shifts change the ranking of methods, and how large is the measured resilience loss beyond raw vehicle count?

## Design Decisions

**2026-07-29. The theme is fixed.** AeroCityBench scores cooperation on hidden 3D targets inside procedural cities with real occlusion, collisions, flight dynamics, communication and energy constraints. Target truth, failure identity, split labels and seeds stay scorer-side. A method that covers a lot of ground still needs a legal observation to score.

**2026-07-31. Task migration to G2I.** The evaluated task moves to `geometry-search-3d`. The public, target-agnostic inspection atlas defines what can be inspected, a mission sector fixes the searchable region, and private targets only decide whether an OBSERVE action produces an anonymous confirmation receipt. The older `exploration-3d` track becomes a retired coverage diagnostic with a separate report line.

**2026-08-01. Mission-sector v2.** The compiler keeps per-cell region metadata while binarizing routes and publishes `cell_assignment_by_drone`. The validator recomputes flight, dwell, climb and return-reserve lower bounds from the public assignment, so a plan with a missing cell, a duplicate assignment or a cross-vehicle reuse is rejected before execution.

**2026-08-02. Statistics and assets.** The ancestor-level statistical protocol is implemented. A power report showed that five calibration ancestors give roughly 0.215 power against a 0.10 confirmed-recall difference, which asks for about 29 independent ancestors, and forbids padding sample size with extra episodes on one layout. On the asset side, the five-asset CC0 mini core closed with 32 USD layers, zero remote references and zero unresolved paths, and twelve development cities passed static geometry, context and full-episode admission review.

**2026-08-03. Calibration and retirement.** The current public scope completed a three-ancestor, four-UAV CF2X and PhysX L1 panel with safe closure, full-fleet return, and an L0 to L1 rank correlation of 1.0. The L2 visual review ran on training, calibration and validation cities at a frozen 960x640 profile. The GPL-3.0 FUEL planner built inside an isolated Focal and ROS Noetic container from a locked upstream commit.

## Current Status

| Workstream | State |
| --- | --- |
| Generator, task schemas, public and private projections | Implemented and tested on CPU |
| G2I inspection atlas and leakage probes | Implemented and tested on CPU |
| Mission sector and capacity certificate | Implemented with independent recomputation |
| Reference baselines and L0 runtime | Implemented and calibrated |
| External method adapters (OR-Tools, MARVEL, ACO3D) | JSONL process boundary with source locks |
| CF2X and PhysX L1 execution | Development panel completed |
| CC0 asset closure and L2 visual review | Completed in development |

## Evidence Discipline

Engineering iterations and reruns are expected. What stays fixed is the public information boundary: the task scope, workload, budget and scoring rules do not change in response to an observed method result. Every record declares its phase, its task-contract state and its evidence scope, and only frozen records marked `formal_score_eligible=true` count as benchmark results.

## Next Steps

Collect the method-independent calibration ancestors needed for the formal comparison matrix, and widen the external-method panel under the same public boundary.
