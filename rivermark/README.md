# Rivermark Benchmark

<p align="center">
  <img src="assets/demos/rivermark-search.gif" alt="Rivermark multi-agent 3D search" width="78%" />
</p>

<p align="center"><em>Multi-agent 3D search in the Rivermark benchmark.</em></p>

[English](README.md) | [简体中文](README.zh-CN.md)

**A toolchain for collecting, auditing, and scoring native Isaac Sim data for multi-agent 3D stealth-search research in the Search3D benchmark, with eight physically simulated CF2X vehicles in a procedural City-Lite scene.**

Rivermark is engineered around three goals, **determinism**, **data integrity**, and **cross-paradigm scoring**. It is a benchmark *infrastructure* that binds every capture to cryptographic contracts, audits each episode before admission, and exposes scoring through a stable, schema-verified interface that classical planning, RL and MARL, QD, and vision-language-action agents can all target.

---

## Design Highlights

- **Determinism by construction.** Every scene, protocol, runtime, and source tree is pinned by SHA-256 contracts. Episodes are seeded deterministically. A runtime lock at profile `citylite-windows-isaacsim-5.1.0.0-local-isaaclab-2.3.2` plus CF2X calibration fixes the software stack, so captures re-run and compare.
- **Formal dataset admission.** Captures enter the formal dataset after provenance checks pass, covering file binding, receipt freshness, split integrity, and lineage approval. A failure ledger and crash-left recovery keep long collection runs auditable.
- **Cross-paradigm scoring.** A single observation and action ABI serves classical planners, RL and MARL, offline RL, and VLA and LeRobot agents, with projection to RLDS, Zarr, and Parquet. The scorer rates search episodes against reference metrics and accepts submissions through a validated, schema-checked interface.
- **Multi-sensor synchronized capture.** Online RGB, depth, semantic labels, RayCaster LiDAR, IMU, contact and safety state, body state, actions, camera extrinsics, and a fixed-world witness, captured for an 8-vehicle fleet in one pass.
- **Contract-first engineering.** 15+ JSON Schemas in `schemas/` define every artifact, a 66-file, 411-test suite in `tests/` runs on CPU, and the supply-chain and asset-provenance modules audit USD assets and dependencies, including CycloneDX SBOM generation.

---

## Repository Layout

```
rivermark/
├── README.md          ← this document
├── code/              ← full source, config, schemas, and CPU test suite
│   ├── src/rivermark_benchmark/   core modules (capture, validation, evaluation, reproducibility)
│   ├── config/                    collection protocols, runtime locks, label ontology
│   ├── schemas/                   JSON Schema contracts for every artifact
│   └── tests/                     CPU-runnable test suite
├── docs/              ← reader-facing design docs (overview, evaluation, reproducibility, governance, limitations)
├── media/             ← rendered overview/composite videos and key frames (8-vehicle fleet)
└── evidence/          ← same-seed repeatability reports and episode manifest examples
```

---

## Quick Start on the CPU Path

Python 3.10+ required. Install the CPU dependencies, then run the researcher smoke check. It creates a small non-formal fixture, verifies it, reads public arrays, and writes a report.

```powershell
cd code
python -m pip install -e ".[cpu-ci]"
$output = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $output
Get-Content "$output\researcher_smoke_report.json"
```

Run the full CPU test suite.

```powershell
python -m unittest discover -s tests -v
```

---

## Determinism Evidence

`evidence/same-seed-repeatability*.json` reports a bounded same-seed experiment, **two independent captures of the same episode seed**, compared by the `rivermark_benchmark.repeatability` analyzer using camera-local, frame-aligned, class and agent ID semantic comparison. The report states bounded same-seed variation under declared tolerances, the measurable determinism story for physics-based simulation. Every binding is pinned by SHA-256, covering the protocol, runtime lock, source tree, receipt, and independent validation.

- `evidence/` holds example episode manifests in public and scorer-private variants.

---

## Media

`media/` contains rendered evidence from train and validation cells.

- **Overview**. `*_overview.mp4` renders the 8-vehicle fleet from above.
- **Composite**. `*_composite.mp4` renders a synchronized multi-camera composite.
- **Key frames**. `*_first/last/25/50/75pct_frame*.png` stills of both views.

Cells `train-cell0/1/2/3/9/11`, `validation-cell13/14/15/16/17/18/19`, and `route-witness-r31`, under protocol `citylite-t1-expert-coverage-v2`.

For a new multi-start native recording batch, use [the route-family video procedure](docs/multistart-native-video-recording.md). It covers both frozen City-Lite route families and encodes native Isaac RGB archives.

---

## Scoring Pipeline

The scorer modules `evaluator.py`, `search_event_evaluator.py`, and `metrics.py` score an episode against reference search metrics and return a validated submission report. Scorer inputs stay scorer-private until admission, and submissions are schema-verified before scoring. Baselines are pluggable through `baseline_harness.py`, with reference methods for classical, learned, MARL, and QD pipelines in `methods.py`, `learned.py`, `marl.py`, `qd_train.py`, and `torch_train.py`.

---

## License

Rivermark-authored source code, schemas, and documentation are licensed under **Apache-2.0** (`code/LICENSE`). NVIDIA Isaac Sim, Rivermark content, CF2X USD, and third-party assets live outside this repository and follow their applicable terms.
