# Overview

Rivermark is a toolchain for building and auditing a **multi-agent 3D stealth-search benchmark** in Isaac Sim. It runs eight physically simulated CF2X quadrotors inside a procedurally composed city scene, records synchronized multi-sensor data, and binds every capture to deterministic contracts.

The project follows three ideas.

- **Determinism by construction.** Every scene, protocol, runtime, and source tree is pinned by content identity. Episodes are seeded deterministically, and a runtime lock fixes the software stack so a capture re-runs and compares.
- **Data integrity before release.** Captures enter the formal dataset after an independent validator checks them and the provenance rules pass. Private target information stays on the scorer side.
- **One ABI, many methods.** A single observation and action contract lets classical planners, RL and MARL, quality-diversity, and vision-language-action agents target the same scoring interface.

## What a Capture Contains

Each episode is a synchronized multi-agent time series. Every retained frame records these streams.

- onboard RGB and depth
- native semantic segmentation as learning labels
- RayCaster LiDAR ranges
- IMU pose, acceleration, angular velocity, and contact and safety state
- root pose and velocities
- the command written before each simulation step, plus public route and state and explicit team messages
- camera calibration, timestamps, and the world, body, and camera transform closure

A fixed-world overview camera acts as a route witness. It is rendered and checked at every retained frame, with a sparse schedule of stored frames.

## Scope

Rivermark runs as a physics simulation with synthetic sensors through a CF2X dynamics model and RayCaster LiDAR. The current corpus is a small set of independently validated development captures, and the roadmap covers calibrated hardware, a statistically powered corpus across scenes, and runs with external foundation models such as OpenVLA, LLaVA, and Dreamer.

## Status

The toolchain is complete and deterministic on the CPU path, with a test suite that runs on a clean checkout. The formal dataset index starts empty and fills as cleared episodes pass admission.
