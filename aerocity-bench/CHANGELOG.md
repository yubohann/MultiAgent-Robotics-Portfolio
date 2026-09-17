# Changelog

All notable development milestones for AeroCityBench are recorded here. The project remains
pilot-only until the formal release gates documented in the authoritative execution plan pass.

## [Unreleased] - 2026-09-17

### Changed

- restructured documentation into a benchmark design, research notes and a run guide.
- cleaned comments and docstrings across the package, tools and tests.
- sized down the demo GIFs and translated the last internal tool output to English.

## [0.2.0.dev0] - 2026-07-30

### Added

- ordinary-v3 procedural city, target-process, public and private projection, and release contracts.
- scorer-private L0 runtime, metrics, reference baselines, adapter contracts, and blind-run
 safeguards.
- CC0 allow-list, provenance, USD dependency-closure, BOM, notice, SBOM, and data-license gates.
- resumable visual-review batches with cross-layout evidence binding and tamper checks.
- canonical four-UAV Native Isaac L1 capability gate covering physical execution, camera-rig
 geometry, observation dwell, braking, reset isolation, and deterministic replay.
- v2 execution receipts bound to action, observation, and state hashes, plus a trusted formal-context
 validator and capability-only native receipt-set output.
- ordinary-paper and top-tier research execution plans with explicit activation gates.

### Changed

- reduced the ordinary-paper scope to one geometry-search main track with four UAVs.
- kept RGB-D and precise L2 replay available, with RGB input optional on the geometry main
 leaderboard.
- defined the 300-second budget as simulated task time, independent of wall-clock sleep time.
- separated fast L0 training and development from mandatory L1 formal geometry scoring and L2 visual
 review.

### Verified

- 87 Python tests, Ruff, and Git whitespace checks pass.
- two independent Native Isaac processes pass all 11 capability checks and produce byte-identical
 gate reports and dynamic evidence.
- the five-asset CC0 mini legal bundle and clean wheel installation pass development validation.

### In progress

- formal L1 episode execution receipts and scoring.
- scientific calibration and the formal method matrix.
- substantive external-method reruns and a blind scoring service.
- expanded-core asset QA, PyPI publication, and public v1.0 release.
