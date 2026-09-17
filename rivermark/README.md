# Rivermark

<p align="center">
  <img src="assets/demos/rivermark-search.gif" alt="Rivermark multi-agent 3D search" width="78%" />
</p>

**A physics-grounded benchmark toolchain for multi-agent 3D stealth search, built on native Isaac Sim captures of eight CF2X quadrotors in a procedural City-Lite scene.**

Cities hide targets. A quadrotor can pass a building and miss a courtyard, lose a target behind an obstacle edge, or cross the visible window too fast to confirm anything. Rivermark records those cases as synchronized multi-sensor episodes. Every step writes the control command first and then advances the simulation, so the causal chain from observation to action stays intact. Scene, protocol, runtime, and source revision are bound to identity contracts, and an episode enters the formal dataset after independent validation passes. Hidden target truth stays on the scorer side, and a search method earns credit through real observations in flight.

**Status.** Protocol `citylite-t1-expert-coverage-v2` is frozen. The 4 train + 4 validation unique-candidate sequence is complete. No further collection binding is permitted under active protocol v2. The native capture path targets Isaac Sim 5.1 with Isaac Lab 2.3.2, and the CPU toolchain verifies from a clean checkout.

## My Role

I designed and built Rivermark end to end. Every section below maps to material in this repository.

- **Benchmark design.** The Search3D task contract, the observation and action ABI, the City-Lite scene contract, the content-identity and runtime-lock scheme, and the frozen `citylite-t1-expert-coverage-v2` collection protocol with its `config/` records and the `schemas/` contract for every artifact.
- **Scene and task implementation.** The procedural City-Lite composition in `src/rivermark_benchmark/citylite_scene/`: route families, safe starts, material and geometry realization, and condition checks.
- **Capture and scoring pipelines.** The native Isaac Sim capture path in `src/rivermark_benchmark/isaac/`, the episode contracts and projections in `data/`, the independent validator and formal-dataset admission in `audit/`, the failure ledger in `runtime/`, and the scorer with public metric definitions in `score/`.
- **Learning baselines.** The pilot method set in `train/` (classical, RL, MARL, quality-diversity, and torch training paths), the SB3 transfer path, and the bounded baseline suite in `config/baseline_suite.cpu_pilot_v1.json` with its provenance-labelled report.
- **Experiment execution.** Collection and capture runs for the frozen cohort, same-seed repeatability captures and geometry scans, and the rendered route-witness and cell videos in `media/` with the reports in `evidence/`.
- **Documentation.** The README set, the `docs/` guides, the schemas, the changelog, and the citation metadata.

## Verified So Far

Development-grade evidence from the frozen cohort.

- Two native captures of episode seed 1751072442 under the same runtime lock passed every declared comparison, with semantic label agreement at 1.0 against a 0.98 floor and onboard RGB frame mean absolute error at 2.24 uint8 levels against an 8.0 ceiling.
- Native geometry scans realized 4 of 4 direct-visible targets on both route pairings.
- The frozen City-Lite contract composes about 20,000 active prims and 276 used USD layers, with 4,807 drivable-surface colliders and 4 task-obstacle colliders.
- The command volume spans 92 m by 92 m by 5.25 m, and the two route families intersect at five points while sharing zero waypoints and segments.

## What It Records

Each episode is a synchronized multi-agent time series for the full eight-vehicle fleet.

- onboard RGB and depth
- native semantic segmentation as learning labels
- RayCaster LiDAR ranges
- IMU, contact, and safety state
- body pose and velocities
- the command written before each simulation step, plus public route state and explicit team messages
- camera calibration, timestamps, and the world, body, and camera transform closure

A fixed-world overview camera acts as a route witness, rendered and checked at every retained frame.

## Determinism and Admission

Every scene, protocol, runtime, and source tree is pinned by content identity. The runtime lock profile `citylite-windows-isaacsim-5.1.0.0-local-isaaclab-2.3.2` fixes the interpreter, package versions, GPU floor, and renderer and physics configuration. A same-seed analyzer compares two captures of one episode seed under predeclared tolerances, frame-aligned by class and agent ID.

An independent validator reopens the raw artifacts and checks stage identity, sensor synchronization, action causality, visual and LiDAR intrusion gates, contacts, route realization, target visibility evidence, provenance, and identity bindings. Cleared episodes enter the formal dataset, and failed artifacts stay visible in a failure ledger with crash-left recovery for long collection runs.

## Scoring

The scorer accepts timestamped confirmation events, each bound to a source observation ID. The scorer owns visibility, matches against hidden targets, and returns a validated report with recall, confirmed AUC, time to first confirmation, false confirmations, collisions, timeout, effort, and failure rate. Metric definitions are versioned and public, and scorer inputs stay private until admission.

## Media and Evidence

The `media` directory holds rendered overview and composite MP4s plus key frames from train cells 0, 1, 2, 3, 9, and 11, validation cells 13 through 19, and route witness r31 under protocol `citylite-t1-expert-coverage-v2`. The `evidence` directory holds same-seed repeatability reports and example episode manifests in public and scorer-private variants.

## Quick Start on the CPU Path

Python 3.10 or newer. Run every command from the repository root.

```powershell
python -m pip install -e ".[cpu-ci]"
$output = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $output
Get-Content "$output\researcher_smoke_report.json"
```

Run the CPU test suite.

```powershell
python -m unittest discover -s tests -v
```

## Source Layout

The repository root is the standalone source package for native Isaac Sim multi-agent 3D stealth-search data collection, validation, and scoring.

- `src/rivermark_benchmark/` is layered by function: `isaac/` (native capture and evidence), `pack/` (pack descriptors, specs, readiness, public manifests), `data/` (contracts, episodes, archives, and projections), `train/` (policy methods and training), `score/` (scoring, evaluation, diagnostics, same-seed analysis), `audit/` (validation, formal dataset admission, and release audit), `runtime/` (simulation runtime, preflight, runtime locks, and failure ledger), and `ops/` (operator and release-media tooling), plus the `citylite_scene/` and `collection_protocol/` contract packages
- `config/` holds collection protocols, runtime locks, the label ontology, and the baseline suite
- `schemas/` holds JSON Schema contracts for every artifact
- `docs/` holds this documentation set plus the API and schema stability, asset policy, and security and integrity policies
- `tests/unit/` holds CPU tests for contracts, integrity, and quality gates
- `tests/integration/` holds native capture, validation, pack, and evidence pipeline tests
- `tools/` holds native video planning and encoding scripts

## Documentation

- [Overview](docs/overview.md), what Rivermark is, what it publishes, and its scope.
- [Task and Scene](docs/task-and-scene.md), the search task and the City-Lite environment.
- [Observation ABI](docs/observation-abi.md), the field-level contract for episode data.
- [Scoring](docs/scoring.md), metric definitions and the submission contract.
- [Native Capture](docs/capture.md), running a native Isaac capture.
- [Video Recording](docs/multistart-native-video-recording.md), planning and encoding native videos from both route families.
- [Admission](docs/validation-and-admission.md), independent validation and formal dataset admission.
- [Determinism](docs/determinism.md), same-seed runs, runtime locks, and clean-room replay.
- [Methods](docs/methods.md), supported method families and evidence rules.
- [Data Access](docs/data-access.md), reading episodes, projections, and the researcher entry path.
- [Governance](docs/governance.md), asset provenance, licensing, and API stability.
- [Limitations](docs/limitations.md), current scope and the development roadmap.

## License

Rivermark-authored source code, schemas, and documentation are licensed under **Apache-2.0**. NVIDIA Isaac Sim, Rivermark content, CF2X USD, and third-party assets live outside this repository and follow their applicable terms. See `LICENSE` for the full text.
