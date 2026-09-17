# Limitations

This page records the current scope of Rivermark and the development directions that follow from it. Each direction advances through measured evidence.

## Current corpus

The formal dataset index starts empty and fills as cleared episodes pass admission. The live corpus holds a small set of independently validated development captures, which support interface checks and pipeline development. Generalization studies, confidence intervals, and a meaningful train and validation comparison come with a corpus that spans scenes, seeds, weather conditions, and obstacle layouts.

## Simulated dynamics

CF2X dynamics and RayCaster LiDAR run in simulation. Hardware calibration, real-flight logs, latency characterization, actuator identification, and a sim-to-real transfer study mark the hardware path ahead.

## One layout, fixed geometry

The City-Lite scene, public route, camera schedule, and controller run a tightly fixed configuration, so a policy can memorize this geometry. Robust search across scenes comes with a second independently contracted layout.

## Conservative collision

Structural collision uses conservative AABB proxies in place of exact building-mesh collision. Mesh-accurate doorways, concavities, and overhangs come in a later revision, and the dedicated impact-response canary gates real-building, damage, aerodynamic, and sim-to-real conclusions from contact behavior.

## Reference method implementations

Classical, RL and MARL, QD, VLM, VLN, VLA, and world-model code in the repository serves as reference material. Matched native Isaac rollouts for external frameworks and a scored T2 policy loop come next.

## Synthetic sensor model

Synthetic semantics and mesh ray casting cover the sensor pipeline. Lens distortion, rolling shutter, multipath, radar phenomenology, thermal drift, packet loss, and asynchronous hardware clocks belong to the hardware-realism roadmap. The gates check internal consistency across the recorded streams.

## Task supervision profile

The public profile carries route, state, and action traces with learning labels. A large language corpus, human demonstrations, a dense object taxonomy, and a verified natural-language grounding benchmark join later task versions.

## Quarantine accounting

Failed captures go to quarantine while passing episodes enter the dataset. A future release publishes the failure taxonomy and denominator, keeping collection failures visible in reported success rates.

## Native capture dependencies

Native capture relies on external IsaacLab, USD, scene-contract, and private-manifest files. The CPU path replays cleanly today, and a fully public source-only native replay bundle marks the next packaging milestone.

## Ecosystem roadmap

Public leaderboards, a standardized metric suite across seeds, a blind server scorer, and independent replication by another group are the community milestones ahead.

The near-term goal is a small, independently validated Isaac development corpus with transparent failure accounting, growing toward the scale of mature aerial and robot-learning datasets.
