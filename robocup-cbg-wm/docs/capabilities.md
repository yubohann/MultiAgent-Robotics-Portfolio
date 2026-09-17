# Capabilities and Measured Evidence

This page lists the validated capabilities of this repository and the measured evidence behind each one.

## 1. Validated Main Scenario

The primary scenario is a RoboCup-style two-robot yellow-versus-blue adversarial match. Real-robot 1v1 experiments were also performed through the ROS2 deployment stack after the IsaacLab and rule-environment evidence package.

Capabilities.

- two differential-drive robots.
- object-centric world-model and SAC Flow or PolicyFlow-style self-play.
- opponent-only target shooting.
- legal target yaw and line-of-sight checks.
- 0.80 s laser dwell gate.
- normal-target and recessed-base shooting range gates.
- pushable red boxes.
- base blockers that open the base target after armor removal.
- collision, penetration, stuck and replay audits.
- IsaacLab three-view MP4 and GIF replay.
- 1v1 real-robot experiment coverage through the ROS2 deployment stack.

Primary evidence files.

```text
docs/rl_data/world_model_sacflow_final/training_summary.json
docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json
docs/rl_data/world_model_sacflow_final/strict_replay_summary.json
docs/media/最终回放_三视角同步拼接版.gif
```

## 2. Two-Robot Measured Results

128-episode multi-seed scoring run.

| Metric | Value |
| --- | ---: |
| Episodes | 128 |
| Yellow win rate | 49.22% |
| Blue win rate | 50.78% |
| Draw rate | 0.00% |
| Mean episode time | 30.8148 s |
| Mean yellow score | 40.8984 |
| Mean blue score | 41.7188 |
| Static penetrations | 0 |
| Box penetrations | 0 |
| Robot contacts per game | 0.00 |
| Repeat target order events | 0 |

8-episode strict replay audit.

| Metric | Value |
| --- | ---: |
| Episodes | 8 |
| Yellow win rate | 37.50% |
| Blue win rate | 62.50% |
| Draw or timeout rate | 0.00% |
| Hard violations | 0 |
| Warnings | 0 |
| Base wins per episode | 1.0000 |
| Own-target penalties per episode | 0.0 |
| Robot contacts per episode | 0.0 |
| Recovery events per episode | 0.0 |

## 3. Large-Scale 50v50 Evidence

The repository also includes a large-scale 50-versus-50 benchmark for scalable strategy research and tactical visualization. The benchmark runs staged rule-level curriculum training, a 256-game scoring run and an IsaacLab tactical replay with 100 vehicle-shaped actors.

Capabilities in the current 50v50 evidence package.

- two teams with 50 vehicles each.
- three control zones.
- base shield opening through zone control.
- legal base damage after shield opening.
- line-of-sight fire and fire cooldowns.
- vehicle elimination.
- score and winner closure.
- a 256-game scoring run.
- an IsaacLab tactical replay with 100 vehicle-shaped actors.

Current 256-game metrics.

| Metric | Value |
| --- | ---: |
| Episodes | 256 |
| Yellow win rate | 36.72% |
| Blue win rate | 42.19% |
| Draw rate | 21.09% |
| Mean yellow base damage | 44.90 |
| Mean blue base damage | 44.89 |
| Mean yellow base open rate | 18.37% |
| Mean blue base open rate | 18.39% |
| Mean robot contacts | 0.00 |
| P95 robot contacts | 0.00 |
| Mean obstacle contacts | 0.00 |

Large-scale evidence files.

```text
docs/rl_data/large_scale_50v50/
docs/rl_data/large_scale_curriculum/
docs/media/large_scale_50v50_isaaclab_replay.mp4
docs/media/large_scale_50v50_replay.mp4
docs/figures/large_scale_50v50/
```

The 50v50 result is a simulation-stage rule-level benchmark with an IsaacLab visual and tactical replay of the accepted trace.

## 4. Capability Matrix

| Capability | Status | Evidence |
| --- | --- | --- |
| Two-agent adversarial match | Validated | 128-episode scoring run and 8-episode strict replay |
| Object-centric world model | Implemented and scored | world-model and SAC Flow training artifacts |
| SAC Flow and PolicyFlow-style actor | Implemented and scored | training config, checkpoint and scoring summaries |
| Rule-aware action shield | Validated in main scenario | zero static and box penetrations |
| Pushable red boxes | Validated in main scenario | push events and changing box poses |
| Base blocker line of sight | Validated in main scenario | strict replay hard violations 0 |
| ROS2 runtime contract | Used for 1v1 real-robot experiments | `crc_robocup_vision_ws/` packages |
| IsaacLab two-robot replay | Validated | final three-view MP4 and GIF |
| 1v1 real-robot experiment | Performed | `docs/sim2real.md` and the ROS2 deployment stack |
| Large-scale 50v50 rule benchmark | Implemented and scored | 256-game scoring run and 50v50 artifacts |
| Large-scale IsaacLab replay | Implemented for simulation-stage 50v50 | `large_scale_50v50_isaaclab_replay.mp4` and generated figures |

## 5. Multi-Agent Scope

The repository uses the term multi-agent in two explicit scopes.

1. Primary scope, a validated two-robot adversarial RoboCup-style match with subsequent 1v1 real-robot experiments.
2. Large-scale extension, a simulation-stage 50-versus-50 rule-level benchmark with IsaacLab replay evidence.

## 6. Distributed Training Support

Supported in the public workflow.

- single-machine training.
- vectorized local environment rollout.
- CUDA actor, critic and world-model updates for the main two-robot policy.
- population-based local policy search for the 50v50 benchmark.
- JSON and CSV scoring output.

## 7. Sim2Real Contract

The Sim2Real contribution is an interface and validation ladder. The 1v1 line has been exercised on real robots after the simulation and replay steps, and the 50v50 line is a simulation-stage extension.

- ROS2 topics, services and actions stay separated from simulation internals.
- Nav2, AprilTag detection, EKF, shooter services and `/cmd_vel` form the deployment contract.
- `docs/sim2real.md` defines calibration order, domain randomization, validation steps and required logs.
- IsaacLab replay checks physical plausibility before real-world deployment.
