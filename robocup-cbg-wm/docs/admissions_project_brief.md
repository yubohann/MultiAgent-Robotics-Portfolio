# Admissions Project Brief

## Project Summary

Object-Centric World-Model Flow RL for Multi-Agent Robotics is a ROS2 and IsaacLab project built around a RoboCup-style adversarial visual navigation task. The system combines real robot software infrastructure, simulation replay, rule-level reinforcement learning, object-centric state modeling and auditable scoring artifacts.

The main validated scenario is a two-robot yellow-versus-blue match. Each robot must navigate, identify opponent targets, push physical-style red boxes, respect base blockers, satisfy laser range and dwell-time constraints, and avoid illegal hits or penetration. The learning stack uses an object-centric world model with SAC Flow and PolicyFlow-style self-play and rule-aware action shielding.

## Why It Is Relevant for Graduate Applications

This project demonstrates work across several research and engineering layers.

| Layer | Demonstrated Work |
| --- | --- |
| Robotics systems | ROS2 Jazzy workspace, Nav2 integration, launch files, robot descriptions, behavior orchestration and shooter services |
| Simulation | IsaacLab replay scene, differential-drive robot behavior, pushable obstacles, base blockers and target interaction |
| Learning | Object-centric state representation, world-model-assisted SAC Flow and PolicyFlow-style self-play and residual expert behavior |
| Safety and rules | Action shielding, legal target ownership, line of sight, laser dwell time, range gates, collision and penetration audits |
| Scoring | Multi-seed scoring runs, strict replay audit, JSON and CSV metrics, replay videos, GIFs and generated figures |
| Sim2Real | ROS2 deployment contract and 1v1 real-robot experiment coverage |

## Key Public Results

### Two-Robot Main Scenario

The public 128-episode stochastic scoring run reports the following metrics.

| Metric | Value |
| --- | ---: |
| Yellow win rate | 49.22% |
| Blue win rate | 50.78% |
| Draw rate | 0.00% |
| Static penetrations | 0 |
| Box penetrations | 0 |
| Robot contacts per game | 0.00 |

The strict replay audit reports zero hard violations and zero own-target penalties for the selected replay set.

Primary evidence.

- `docs/rl_data/world_model_sacflow_final/training_summary.json`
- `docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json`
- `docs/rl_data/world_model_sacflow_final/strict_replay_summary.json`
- `docs/assets/最终回放_三视角同步拼接版.gif`

### Large-Scale 50v50 Extension

The repository also includes a simulation-stage 50v50 benchmark for scalable rule-level multi-agent strategy. It uses a staged 5v5 to 10v10 to 25v25 to 50v50 curriculum and scores the final stage over 256 games.

This extension is published as simulation-stage evidence with an IsaacLab tactical replay.

Primary evidence.

- `docs/rl_data/large_scale_curriculum/curriculum_summary.json`
- `docs/rl_data/large_scale_50v50/eval_summary.json`
- `docs/assets/large_scale_50v50/`
- `docs/assets/large_scale_50v50_isaaclab_replay.mp4`

## What I Would Emphasize in an Application

The strongest application framing follows.

> I built a replayable ROS2 and IsaacLab multi-agent robotics learning stack that connects rule-aware simulation, object-centric reinforcement learning, strict scoring, replay visualization and real-robot deployment contracts. The project is an end-to-end robotics research artifact with auditable evidence.

Recommended emphasis.

- real robotics system integration from sensors through deployment.
- explicit rule modeling and safety checks.
- scoring discipline with metrics, replay audits and structured failure analysis.
- Sim2Real engineering through ROS2 contracts, with hardware statistics packaging in progress.
- scalability exploration through the 50v50 simulation-stage benchmark.

## Reviewer Reading Path

For a fast review, follow these steps.

1. Read the README top section and watch the first GIF.
2. Open `docs/capabilities.md` for the capability list and measured evidence.
3. Inspect `docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json`.
4. Watch the three-view replay GIFs or MP4s under `docs/assets/`.
5. If interested in scaling, read `docs/rl_data/large_scale_50v50/eval_summary.json` and the figures under `docs/assets/large_scale_50v50/`.

## Suggested Application Description

Short version.

> Built a ROS2 and IsaacLab multi-agent robotic learning system for a RoboCup-style visual adversarial task, combining object-centric world models, SAC Flow and PolicyFlow self-play, rule-aware action shielding, strict replay audits, and Sim2Real deployment contracts.

Long version.

> This project implements an end-to-end robotics learning stack for adversarial visual navigation. I designed the rule environment, ROS2 runtime contracts, IsaacLab replay pipeline, object-centric learning representation, SAC Flow and PolicyFlow-style self-play, scoring scripts, replay audits, figures and documentation. The validated main setting is a two-robot match with a balanced 128-episode scoring run and zero reported penetration or contact violations, and the repository also includes a 50v50 simulation-stage benchmark for scalable multi-agent strategy research.
