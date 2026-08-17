# Limitations

These are the honest boundaries of Rivermark today. They are structural, not cosmetic, and the benchmark is designed so that each one must be resolved with evidence before a corresponding claim is made.

## No statistical dataset yet

The formal dataset index is empty. A handful of development captures cannot support generalization, confidence intervals, or a meaningful train/validation comparison. There are no held-out scenes, seeds, weather conditions, or obstacle layouts.

## Simulation-to-real is unproven

CF2X dynamics and RayCaster LiDAR are simulated. There is no calibrated hardware radar, real-flight log, latency characterization, actuator identification, or sim-to-real transfer study.

## One layout, fixed geometry

The City-Lite scene, public route, camera schedule, and controller are tightly fixed. A policy could memorize this geometry rather than search robustly. Cross-scene generalization requires a second independently contracted layout.

## Conservative collision

Structural collision uses conservative AABB proxies rather than exact building-mesh collision. There is no dedicated impact-response canary yet, so no real-building, damage, aerodynamic, or sim-to-real conclusion can be drawn from contact behavior.

## Incomplete method evaluation

Classical, RL/MARL, QD, VLM/VLN/VLA, and world-model code in the repository is reference material. There are no matched native Isaac rollouts for external frameworks, and no T2 policy loop has produced a score.

## Sensor realism is limited

Synthetic semantics and mesh ray casting do not model lens distortion, rolling shutter, multipath, radar phenomenology, thermal drift, packet loss, or asynchronous hardware clocks. The gates prove consistency, not physical fidelity.

## Sparse task supervision

The public profile has route/state/action traces and learning labels, but no large language corpus, human demonstrations, dense object taxonomy, or verified natural-language grounding benchmark.

## Selection bias risk

Failed captures are retained in quarantine while only passing episodes are admitted. A future release must publish a failure taxonomy and denominator so collection failures are not silently excluded from reported success rates.

## Conditional reproducibility

Native capture depends on external IsaacLab, USD, scene-contract, and private-manifest files. The CPU path is cleanly reproducible; a fully public, source-only native reproduction bundle is not yet published.

## No ecosystem yet

There is no public leaderboard, standardized metric suite across seeds, blind server evaluator, or independent replication by another group.

The near-term goal is honest and bounded: a small, independently validated Isaac development corpus with transparent failure accounting — not parity with mature aerial or robot-learning datasets.
