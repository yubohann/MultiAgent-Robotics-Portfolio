# AeroCityBench Documentation

Start with the benchmark design for the task, scoring contract and splits. The research notes record how the pilot was built and what is verified so far. The reproduction guide covers builds, tests and adapter entry points.

| Start here | Document |
|---|---|
| Benchmark problem, confirmation contract, city generation, splits, metrics | [Benchmark design](benchmark-design.md) |
| Questions, dated design decisions, validation gates, evidence discipline | [Research notes](research-notes.md) |
| Install, focused tests, release builds, native and external paths | [Run guide](run-guide.md) |

## Repository Scopes

```text
src/aerocity_bench/  installable package, contracts, generator, scorer, baselines
configs/             versioned release configurations and the experiment governance registry
schemas/             JSON schemas for public, private, fault and release artifacts
tools/               builders, audits, native preflights and evidence validators
tests/               contract, integrity and quality-gate tests
assets/              provenance registry for approved redistributable assets
external/            isolated upstream-method adapters with source locks
docs/                this documentation set
```

## Governance Evidence Records

Three evidence records are referenced by path from `configs/experiment-governance-v1.json`, so they stay in place while that registry is active. The governance audit validates their presence on every run, and the experiment registry is the single source of truth for what may be promoted to a formal result.

For the public overview, read the root [README](../README.md).
