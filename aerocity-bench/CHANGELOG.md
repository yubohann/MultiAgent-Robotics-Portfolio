# Changelog

All notable development milestones for AeroCityBench are recorded here. The project remains
pilot-only until the formal release gates documented in the authoritative execution plan pass.

## [0.2.0.dev0] - 2026-07-30

### Added

- ordinary-v3 procedural city, target-process, public/private projection, and release contracts;
- evaluator-private L0 runtime, metrics, reference baselines, adapter contracts, and blind-run
  safeguards;
- CC0 allow-list, provenance, USD dependency-closure, BOM, notice, SBOM, and data-license gates;
- resumable visual-review batches with cross-layout evidence binding and tamper checks;
- canonical four-UAV Native Isaac L1 capability gate covering physical execution, camera-rig
  geometry, observation dwell, braking, reset isolation, and deterministic replay;
- v2 execution receipts bound to action/observation/state hashes, plus a trusted formal-context
  validator and capability-only native receipt-set output;
- ordinary-paper and top-tier research execution plans with explicit activation and no-go gates.

### Changed

- reduced the ordinary-paper scope to one geometry-search main track with four UAVs;
- kept RGB-D and precise L2 replay available without requiring RGB input on the geometry main
  leaderboard;
- defined the 300-second budget as simulated task time rather than a wall-clock sleep;
- separated fast L0 training/development from mandatory L1 formal geometry scoring and L2 visual
  review.

### Verified

- 87 Python tests, Ruff, and Git whitespace checks pass;
- two independent Native Isaac processes pass all 11 capability checks and produce byte-identical
  gate reports and dynamic evidence;
- the five-asset CC0 mini legal bundle and clean wheel installation pass development validation.

### Not yet complete

- formal L1 episode execution receipts and evaluator scoring;
- scientific calibration and the formal method matrix;
- substantive external-method reproductions and blind evaluation service;
- expanded-core asset QA, PyPI publication, and public v1.0 release.
