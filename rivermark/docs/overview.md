# Overview

Rivermark is a toolchain for building and auditing a **multi-agent 3D stealth-search benchmark** in Isaac Sim. It runs eight physically simulated CF2X quadrotors inside a procedurally composed city scene, records synchronized multi-sensor data, and binds every capture to reproducible contracts.

The project is organized around three ideas:

- **Reproducibility by construction.** Every scene, protocol, runtime, and source tree is pinned by SHA-256. Episodes are seeded deterministically, and a runtime lock fixes the software stack so a capture can be re-run and compared.
- **Data integrity before release.** Captures are not admitted to the formal dataset until an independent validator checks them and provenance rules are satisfied. Private target information never enters the public projection.
- **One ABI, many methods.** A single observation/action contract lets classical planners, RL/MARL, quality-diversity, and vision-language-action agents target the same evaluation.

## What a capture contains

Each episode is a synchronized multi-agent time series, not a video. For every retained frame it records:

- onboard RGB and depth
- native semantic segmentation (learning labels, not a policy input)
- RayCaster LiDAR ranges
- IMU pose, acceleration, angular velocity, and contact/safety state
- root pose and velocities
- the command written before each simulation step, plus public route/state and explicit team messages
- camera calibration, timestamps, and world/body/camera transform closure

A fixed-world overview camera acts as a route witness: it is rendered and checked at every retained frame, but only a sparse schedule of frames is stored.

## What the benchmark deliberately does not claim

- real flight, hardware radar, or sim-to-real transfer
- calibrated hardware sensors
- execution of external foundation models (OpenVLA, LLaVA, Dreamer, and so on)
- a statistically powered dataset or cross-scene generalization

The current corpus is a small set of independently validated development captures. That is a feature: the honest path is to grow evidence before claiming results, rather than to declare them early.

## Status in one line

The toolchain is complete and CPU-reproducible (412 tests pass on a clean checkout); the formal dataset index is intentionally empty until cleared episodes are admitted.
