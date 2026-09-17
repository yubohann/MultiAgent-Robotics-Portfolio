# AeroCityBench Documentation

This directory contains the benchmark contract, research protocol, external-method scope, and release evidence guidance. The executable source remains in `src/aerocity_bench/`. Commands and audit tools remain in `tools/`.

## Start Here

| Topic | Document |
|---|---|
| Benchmark problem statement | [Theme and scorer-private truth](AeroCityBench主题与评测器私有真值说明.md) |
| Authoritative research contract | [Authoritative research plan](权威科研执行计划.md) |
| Current G2-I execution order | [G2-I execution and reuse plan](g2-i-execution-and-reuse-plan-2026-07-31.md) |
| Quadrotor execution and evidence scope | [Quadrotor execution contract](四旋翼动力学正式执行合同.md) |
| Admissible methods and interfaces | [External-method input semantics](外部方法输入语义矩阵-20260803.md) |
| Formal-experiment entry conditions | [Formal-experiment checklist](正式实验前执行清单与代码落实计划.md) |

## Repository Scopes

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
environment trees hold development records or generated outputs. Wheel assembly
includes only the installable package, with these trees excluded where applicable.

## GitHub Metadata

- Repository name, `AeroCityBench`
- Short description, `Open benchmark for 3D multi-UAV search under urban topology, target-process, and fleet-resilience shifts.`
- Suggested topics, `multi-uav`, `drone-search`, `robotics-benchmark`, `3d-search`, `multi-agent-systems`, `isaac-sim`, `procedural-generation`, `reproducible-research`

Use the root [README](../README.md) for the public-facing overview and this
index for technical navigation.
