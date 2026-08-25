# AeroCityBench Documentation

This directory contains the benchmark contract, research protocol, external-method boundary, and release evidence guidance. The executable source remains in `src/aerocity_bench/`; commands and audit tools remain in `tools/`.

## Start Here

| Question | Document |
|---|---|
| What problem does the benchmark define? | [Theme and evaluator-private truth](AeroCityBench主题与评测器私有真值说明.md) |
| What is the authoritative research contract? | [Authoritative research plan](权威科研执行计划.md) |
| What is the current G2-I execution order? | [G2-I execution and reuse plan](g2-i-execution-and-reuse-plan-2026-07-31.md) |
| How are quadrotor execution and evidence bounded? | [Quadrotor execution contract](四旋翼动力学正式执行合同.md) |
| Which methods and interfaces are admissible? | [External-method input semantics](外部方法输入语义矩阵-20260803.md) |
| What must pass before formal experiments? | [Formal-experiment checklist](正式实验前执行清单与代码落实计划.md) |

## Repository Boundaries

```text
src/aerocity_bench/  installable benchmark package, contracts, generator, evaluator, baselines
configs/             versioned release and experiment configurations
schemas/             JSON schemas for public, private, fault, and release artifacts
tools/               builders, audits, native preflights, and evidence validators
tests/               focused contract, integrity, and quality-gate tests
assets/              provenance registry for approved redistributable assets
external/            isolated upstream-method boundaries and source locks
docs/                research decisions, protocols, and release guidance
```

The `debug/`, `reason/`, `scenario/`, `.wheel_verify_*/`, `dist/`, and local
environment trees are development records or generated outputs. They are not
runtime dependencies of the installable package and are excluded from wheel
assembly where applicable.

## GitHub Metadata

- Repository name: `AeroCityBench`
- Short description: `Open benchmark for 3D multi-UAV search under urban topology, target-process, and fleet-resilience shifts.`
- Suggested topics: `multi-uav`, `drone-search`, `robotics-benchmark`, `3d-search`, `multi-agent-systems`, `isaac-sim`, `procedural-generation`, `reproducible-research`

Use the root [README](../README.md) for the public-facing overview and this
index for technical navigation.
