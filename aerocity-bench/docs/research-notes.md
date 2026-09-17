# Research Notes

A working record of how the AeroCityBench pilot was designed, what has been verified so far, and what stays closed until the gates pass. Every artifact named here carries a `formal_score_eligible` flag, and only frozen formal records will ever count toward benchmark conclusions.

## Research Questions

- **RQ1.** How strongly does spatial coverage predict confirmed-target recall once occlusion, viewpoint and dwell constraints apply?
- **RQ2.** When coverage and confirmed recall diverge, which mechanisms dominate. Wrong facade, blocked line of sight, missed dwell window, height miss, or duplicate effort.
- **RQ3.** Do topology, scale, target-process and fleet-degradation shifts change the ranking of methods, and how large is the measured resilience loss beyond raw vehicle count?

## Design Decisions

**2026-07-29. The theme is fixed.** AeroCityBench scores cooperation on hidden 3D targets inside procedural cities with real occlusion, collisions, flight dynamics, communication and energy constraints. Target truth, failure identity, split labels and seeds stay on the scorer side. A method that covers a lot of ground still needs a legal observation to score.

**2026-07-31. Task migration to G2-I.** The evaluated task moves to `geometry-search-3d`. The public, target-agnostic inspection atlas defines what can be inspected, a mission sector fixes the searchable region, and private targets only decide whether an OBSERVE action produces an anonymous confirmation receipt. The older `exploration-3d` track becomes a retired coverage diagnostic with a separate report line.

**2026-08-01. Mission-sector v2.** The compiler keeps per-cell region metadata while binarizing routes and publishes `cell_assignment_by_drone`. The validator independently recomputes flight, dwell, climb and return-reserve lower bounds from the public assignment. Missing cells, duplicate assignment, cross-vehicle reuse, over-budget plans or hash tampering all fail closed.

**2026-08-02. Statistics and assets.** The ancestor-level statistical protocol is implemented. A power report showed that five calibration ancestors give roughly 0.215 power against a 0.10 confirmed-recall difference, which asks for about 29 independent ancestors, and forbids padding sample size with extra episodes on one layout. On the asset side, the five-asset CC0 mini core closed with 32 USD layers, zero remote references and zero unresolved paths, and twelve development cities passed static geometry, context and full-episode admission review.

**2026-08-03. Calibration and retirement.** The current public scope completed a three-ancestor, four-UAV CF2X and PhysX L1 panel with safe closure, full-fleet return, and an L0 to L1 rank correlation of 1.0. The L2 visual review ran on training, calibration and validation cities at a frozen 960x640 profile. The GPL-3.0 FUEL planner built inside an isolated Focal and ROS Noetic container with a source lock. The legacy v12 and v15 panels were retired, v15 because its public artifact exposed private target-count metadata.

## Validation Gates

| Gate | Workstream | What must hold | State |
| --- | --- | --- | --- |
| A | Searchability and statistics | Non-oracle methods show stable, non-saturated confirmations on medium tasks, and the atlas plus sector leak no target, witness, split or seed information | Fresh A gate required for the current contract |
| B | Real quadrotor CF2X and L1 | Takeoff, hover, sector route, OBSERVE, anonymous receipt, return and reset on the CF2X and PhysX chain across at least three calibration ancestors | v16 panel completed for an older contract hash and retained as calibration evidence |
| C | External methods and statistics | External methods plug in with auditable permissions, licenses and budgets, and the ancestor-level protocol has enough power | Open. OR-Tools L0 smoke and FUEL and MARVEL diagnostics exist, statistics await more ancestors |
| D | Scene and legal closure | Asset licenses, USD dependency closure and L2 visual review pass | L2 batches passed in development, release review still open |
| E | Release and recovery | Clean-source rerun, clean-venv wheel install, hash-bound environment manifest and a recovery drill | Development-verified, formal rerun open |

Until every gate passes, formal test access stays closed, MAPPO and QD plus RL long training stay disabled, and no L0 or private-fixture number may be written as a result.

## Evidence Discipline

The experiment registry in `configs/experiment-governance-v1.json` makes a simple rule machine-checkable. Engineering iterations are fine. Using an observed result to move the public information scope, workload, budget, scoring or exclusion rules is not.

Every record declares its phase, its task-contract state and its evidence scope. The audit recomputes containment on each run: legacy G1-U stays retired, failed-boundary L1 replays stay quarantined, no development record may carry `formal_score_eligible=true`, and the formal main matrix and learning training stay explicitly blocked until the gates clear. The audit reports `CONTAINMENT_PASS_FORMAL_NO_GO` today, which is exactly the intended state for a pilot.

Failure evidence stays in the record. The retired v12 and v15 panels remain visible, with reasons, rather than disappearing.

## Verified Today

- Procedural generation, task schemas, public and private projections, release validation and open-asset licensing checks.
- G2-I inspection-atlas compilation from geometry alone, recursive leakage probes against target, witness, scorer, split and seed fields.
- Scorer-side OBSERVE receipts bound to the real sensor frame, with independent mission-sector recomputation.
- The current-boundary CF2X calibration panel, the L2 visual review batches, the asset closure and the twelve-city admission review, all at development grade.
- A clean-wheel install path and a hash-bound environment manifest, verified in development.

## Next Authorized Step

`RERUN_METHOD_INDEPENDENT_A_GATE_FOR_CURRENT_CONTRACT`. When the A and B gates hold for the frozen contract, the statistical protocol needs roughly 29 independent ancestors before formal comparisons can start.
