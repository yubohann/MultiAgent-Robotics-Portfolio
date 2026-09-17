# AeroCityBench Documentation

Start with the benchmark design for the task, scoring contract and splits. The research notes record the questions and dated design decisions. The reproduction guide covers install, tests, builds and adapter entry points.

| Start here | Document |
|---|---|
| Benchmark problem, confirmation contract, city generation, splits, metrics | [Benchmark design](benchmark-design.md) |
| Questions, dated design decisions, current results | [Research notes](research-notes.md) |
| Install, focused tests, release builds, external adapters | [Run guide](run-guide.md) |

## Repository Scopes

```text
src/aerocity_bench/  installable package: core, generation, runtime, atlas, bridge, release
configs/             versioned release configurations and statistical protocols
schemas/             JSON schemas for public, private, fault and release artifacts
tools/               adapters, smoke, calibration, scene, quality, native and release helpers
tests/               unit, pipeline and external-adapter test groups
assets/              provenance registry for approved redistributable assets
external/            isolated upstream-method adapters with source locks
docs/                this documentation set
```

For the public overview, read the root [README](../README.md).
