# Changelog

All notable development milestones for AeroCityBench are recorded here.

## [Unreleased] - 2026-09-18

### Changed

- simplified the codebase for the public portfolio: removed the content-addressing and
  evidence-binding layer, the experiment-governance registry and the audit/gate machinery that
  depended on it.
- replaced derived identifiers with explicit IDs: layout IDs from split/index/attempt, site and
  cell IDs from their owners and coordinates, and receipt identity from evaluator-side tokens.
- replaced file-digest provenance with filenames and sizes across asset, adapter and release records.
- simplified release flow to build, validate, export-public; baselines and metrics unchanged.
- restructured documentation into a benchmark design, research notes and a run guide.
- rewrote the ordinary-v3 test module around generation, projections, compilation, scene output,
  release building and baseline evaluation.
- layered the package into `core`, `generation`, `runtime`, `atlas`, `bridge` and `release`
  subpackages, split `tools/` into adapters, smoke, calibration, scene, quality, native and
  release groups, and grouped `tests/` into unit, pipeline and adapters, updating every import,
  path reference and documentation command.
- sized down the demo GIFs and translated the last internal tool output to English.

## [0.2.0.dev0] - 2026-07-30

### Added

- ordinary-v3 procedural city, target-process, public and private projection, and release contracts.
- scorer-private L0 runtime, metrics, reference baselines, adapter contracts, and blind-run
 safeguards.
- CC0 allow-list, provenance, USD dependency-closure, BOM, notice, SBOM, and data-license files.
- resumable visual-review batches with cross-layout evidence binding.
- a four-UAV CF2X and PhysX preflight covering physical execution, camera-rig geometry, observation
 dwell, braking, reset isolation, and deterministic replay.
- v2 execution receipts bound to action, observation, and state, plus an in-memory formal-context
 validator.
- ordinary-paper and research execution plans.

### Changed

- reduced the ordinary-paper scope to one geometry-search main track with four UAVs.
- kept RGB-D and precise L2 replay available, with RGB input optional on the geometry main
 leaderboard.
- defined the 300-second budget as simulated task time, independent of wall-clock sleep time.
- separated fast L0 training and development from mandatory L1 formal geometry scoring and L2 visual
 review.

### Verified

- the CPU test suite and Ruff checks pass.
- two independent native Isaac processes pass all 11 capability checks and produce identical gate
 reports.
- the five-asset CC0 mini legal bundle and a clean wheel installation pass development validation.

### In progress

- formal L1 episode execution receipts and scoring.
- scientific calibration and the formal method matrix.
- external-method reruns and a blind scoring service.
- expanded-core asset QA, PyPI publication, and public v1.0 release.
