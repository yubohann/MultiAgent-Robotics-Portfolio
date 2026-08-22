# AeroCityBench

[English](README.md) | [简体中文](README.zh-CN.md)

AeroCityBench is an open procedural benchmark for studying coverage-to-search gaps and controlled generalization under target-process and urban-topology shift in physics-grounded 3D multi-UAV search.

Status: `0.2.0.dev0`, pilot-only, formal release `NO-GO`. The ordinary-v3 generator, private L0 evaluator/runtime, contracts, adapters, reference baselines, release validation, five-asset CC0 legal bundle, visual-review tooling, wheel packaging, and a precommitted three-ancestor, three-public-method four-UAV CF2X/PhysX L1 calibration panel are implemented. The nine L1 replays closed safely and the ancestor-weighted L0/L1 rank correlations were 1.0; this is calibration evidence only, not a formal episode score. Substantive external 3-D methods, sufficient independent ancestors, city/legal closure, a clean-source release rerun, formal experiments, and PyPI publication remain incomplete.

The benchmark does not claim to model real post-disaster casualty distributions. It provides evaluator-private, controllable 3D search-target processes attached to actual component surfaces.

**GitHub identity:** `AeroCityBench` — an open benchmark for 3D multi-UAV search under urban topology, target-process, and fleet-resilience shifts. The [documentation index](docs/README.md) maps the benchmark contract, implementation boundaries, and reproducibility workflow to the source tree.

## Development build and validation

Python 3.11 or newer is required. From this repository:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
$assetRoot = (Resolve-Path $env:AEROCITY_ASSET_ROOT)
$outputRoot = Join-Path $env:AEROCITY_OUTPUT_ROOT 'ordinary-v1-mini'
python -m aerocity_bench build --release .\configs\releases\ordinary-v1-mini.json `
  --asset-root $assetRoot `
  --output $outputRoot `
  --allow-uncommitted-development
python -m aerocity_bench validate $outputRoot
```

Set `AEROCITY_ASSET_ROOT` to the verified open-asset bundle and
`AEROCITY_OUTPUT_ROOT` to a writable data directory before running the example. The repository
does not download or redistribute NVIDIA/Nucleus content.

When direct Isaac tools are run from a standalone clone, set
`AEROCITY_ISAACLAB_ROOT` to an IsaacLab checkout containing `source/isaaclab`.
The historical nested development layout is detected automatically.

For local quality checks:

```powershell
python -m pip install -e ".[dev]"
.\tools\run_python_quality_gate.ps1 -Python python
```

Do not use a broad recursive `pytest -q` as the default on this Windows host.
Some integration ranges intentionally launch child Python/Isaac processes and
previously contributed to Windows resource exhaustion.  Run focused test groups
first; execute process-spawning and native Isaac checks as individually
host-guarded jobs with their own receipts. A Python environment that cannot
import `jsonschema`, `pytest`, and `ruff` is not a verified quality-gate
environment, regardless of historical test logs.

Every build is deterministic from the release configuration, generator version, asset registry, and master seed. The output directory must not already exist, which prevents stale files from being mixed into a release.

The `smoke.json` and `v0.2-pilot.json` files are legacy/pre-plan v2 configurations. New ordinary-paper development uses `ordinary-v1-mini.json`.

The installed CLI also provides `list-presets`, `show-preset`, `init-config`, `assets-verify`, `list-baselines`, `run-baseline`, `evaluate`, `probe-isaac`, `native-gate`, `capture-review`, and `capture-review-batch`. Run `aerocity-bench <command> --help` for the exact current syntax. An asset-sync command and PyPI `1.0.0` release remain planned, not implemented publication claims.

## Verified development state (2026-07-31)

- 134 explicitly collected Python tests and Ruff pass under `C:\\Users\\Administrator\\anaconda3\\python.exe`; the ordinary-contract suite is partitioned into deterministic, process/host, and packaging groups so every group has an explicit exit code on this Windows host.
- A six-layout, 16-episode authority build validates with 127 evaluator-private targets and a five-asset CC0 legal gate.
- The L0 centralized oracle confirms 6/6 calibration targets, with zero collisions or clearance interventions and all four UAVs returned home. L0 is never formally score-eligible.
- Earlier v5/v6 Native Isaac processes independently passed all 11/11 capability checks. The latest v10 run also passes 11/11 and emits 144 measured, chain-hashed capability receipts; its native report hash is `2c02416987d3f92abd9b52d12530668dc65396e8517ccd36ef419677efb0ef12`, receipt-list hash is `88d94c12595fa07b5e9234d822982ba1e10ad1ee0dbc457230ab02a90c4478ac`, and `formal_score_eligible=false`. The report is explicitly a `dynamic_cuboid_kinematic_capability_probe`, not evidence that the formal benchmark already uses the project quadrotor.
- The historical Native gate emits a chain-hashed capability receipt set from measured Isaac before/after states. It is explicitly capability-only. CF2X is now the only candidate L1 runtime dependency: its root and relative schema are hash-locked, local-only, and cannot be redistributed or promoted without independent native episode evidence.
- CF2X candidate-controller preflight now includes a 30-second native anti-descent gate: two fresh Isaac processes held the hover state at `z=1.5 m` with zero post-warm-up altitude slope, `2.14e-06 m/s` terminal vertical velocity, and no contact. A separate 30-second lateral hold reached and held a `0.35 m` target under `0.0385 rad` maximum tilt without altitude loss. These are local, non-formal preflight receipts, not real-vehicle calibration, city-collision validation, or an L1 scored episode.
- The four-CF2X public-policy preflight retains both outcomes: v5 failed closed at `54.6 s` when a large facade-yaw turn was coupled to horizontal transit and `uav-00` left the flight bounds. The corrected v6 reached four legal `OBSERVE` actions in `120 s` with no collision or out-of-bounds evidence, but recorded no confirmation or return-home closure. Evidence integrity checks passed in both runs; only v6 passed this limited safety interval, and neither result is score-eligible.
- An internal, evaluator-owned four-CF2X closure fixture now completes `OBSERVE -> private confirmation -> RETURN -> all-home` in three separate development cities. Calibration/train/validation completed in `228.0/200.2/164.8 s` of simulated time with `1,140/1,001/824` control ticks, one confirmation each, no collision/out-of-bounds/deadline miss, and local CPU receipt validation. The new public-only aggregate tool records only commitments and aggregate counters; it does not reveal a target, witness, route, or selected UAV. These are internal preflight fixtures, not public policies, external methods, scored episodes, or formal L1 results.
- The internal single-UAV CF2X vertical-slice fixture now fails closed on its own private evidence chain: each native receipt must bind its action, source observation, measured before/after states, prior receipt, evaluator confirmation, and return-home closure. Focused regression tests reject a deleted receipt, a confirmation attached to a non-`OBSERVE` action, and a claimed successful closure without `RETURN`. This improves preflight evidence integrity only; it remains `formal_score_eligible=false` and does not substitute for a four-UAV formal executor.
- A position-anchored, anisotropic candidate guidance profile was then exercised on one calibration, one train, and one validation development city. All three single-UAV private fixtures confirmed one evaluator-private target and returned home without collision or out-of-bounds evidence in 167.0, 192.6, and 205.8 seconds of simulated time respectively (52.8--65.5 seconds wall clock). A route leg now keeps a fixed transit yaw through a vertical descent, rather than recalculating yaw from residual horizontal error at each control period. These receipts remain local-only preflight evidence: no formal test was read, no camera rendering was enabled, CF2X parameters remain pending audit, and no multi-UAV episode score exists. Each run also requires fresh, distinct public/private `.json` evidence paths so prior receipts cannot be overwritten.
- A 100-layout prepare/recovery stress run passes 100/100 under the v3 batch contract. A cross-layout copied attempt is rejected and only the affected layout is regenerated.
- The v8 Isaac visual-review batch passes calibration/train/validation on the first attempt. Every scene verifies 32 target instances and four start instances across ten RGB/depth/instance-segmentation views. This is development/L2 visual evidence, not formal L1 control or scoring evidence.
- `ordinary-v3.1` adds bounded, collider-backed parapets, entrance canopies, rooftop equipment, and semantic obstacles so review scenes need not be visually bare. Roads, sidewalks, markings, and allow-listed CC0 decorations remain visual-only. Visual-only changes are excluded from `task_geometry_hash`, so they cannot alter hidden targets, starts, or the coarse prior. This has focused static-test evidence only; it is not a completed L1 or formal visual-quality gate.
- `tools/audit_scene_geometry.py` performs one bounded, resumable development-layout static audit at a time. It samples every episode internally but writes only public geometry counts, layout/task hashes, and error categories: target totals, support-site totals, target IDs, coordinates, process labels, witnesses, and raw rejection text are forbidden. One calibration, one train, and one validation layout have passed this receipt format; the required 10--20 layout audit remains incomplete.
- External adapter declarations require a paired upstream URL and full frozen revision, a known license, and an explicit process boundary. The `ExternalProcessPlannerBridge` now provides a bounded JSONL bridge for a real centralized `process` declaration: it recursively removes the two local false audit sentinels before serialization; rejects target/evaluator/private names (including case and separator variants) and non-ASCII object keys; binds request IDs and action rosters; limits response size/time; records bridge latency; and terminates only its owned process tree on failure. It is not a blind-evaluator sandbox; container and ROS declarations still require their own verified runner, and no external method has yet been integrated.
- A wheel installs in a clean venv and contains the preset, schema, capture tool, native-gate tool, and private vertical-slice contract. Its file list is checked to exclude asset directories, USD/mesh files, `5_in_drone`, NVIDIA/Nucleus/Omniverse paths, and investigation artifacts. It has not been uploaded to PyPI.

## Scientific tracks

The quadrotor execution boundary is documented in [docs/四旋翼动力学正式执行合同.md](docs/四旋翼动力学正式执行合同.md). The current native capability probe is explicitly a cuboid probe. The CF2X candidate accepts per-rotor thrust targets, advances actuator states, and applies their geometry-derived allocation as a root-body PhysX wrench; it does not apply per-prop-link external forces. Formal L1 scoring remains blocked until the hash-verified local backend completes native calibration and full episode closure gates.

This repository implements Paper I only: the open generator, evaluator contracts, baseline adapters, controlled benchmark experiments, and release evidence. It contains no proprietary method contribution and must not be calibrated using the later QD/RL method. Method research lives in a separate local-only workspace.

The same CitySpec is intended to support three task views:

- `exploration-3d / G1-U`: target-free 3D exploration with a coarse prior and local occupancy, for frontier and coverage systems;
- `geometry-search-3d / G2-I`: the candidate Paper I primary task, with a target-agnostic public inspection atlas and evaluator-checked range, FoV, line of sight, surface facing, dwell, and source-observation provenance;
- `perception-search-3d`: an optional, separately ranked detector-and-search task.

Pure distance-only target confirmation is forbidden. `G2-I` now has a strict L0/CPU contract: it deterministically derives target-independent roof/facade/entrance/rubble inspection regions, applies public geometric admission, exposes separately hashed coarse-region and full-cell priors, and rejects evaluator-private atlas fields. Inspection credit requires an evaluator-accepted observation satisfying range, FoV, facing, LOS, dwell, freshness, pose-drift, clearance, and safety rules; the primary diagnostic is represented-area weighted. Planner and JSONL process adapters accept a scrubbed G2-I projection, but this is not an external-method closure. A five-ancestor CPU audit passes geometric admission and the paired leakage probe while retaining `formal_score_eligible=false`; exhaustive full-atlas dwell workload still exceeds the 300-second budget. The ordinary-v3 builder and native Isaac gate deliberately still accept only G1-U, so no current geometry-search score, sweep result, or preflight receipt is formal-score eligible. The public atlas never contains targets, counts, support sites, witnesses, target-process labels, or split/seed data.

RGB-D remains an available capability for local 3D mapping, avoidance, and visual policies, but is not mandatory input for every baseline. Geometry training/ranking does not use RGB. `L2` uses RGB-D or instance segmentation only for OBSERVE-triggered visual review or the future perception track. Submissions declare a compatible observation profile, and different profiles are not mixed in one ranking.

The canonical episode budget is 300 seconds of simulated execution time. Isaac capture timeouts are separate host-safety limits and do not mean that development waits 300 wall-clock seconds per episode.

The ordinary-paper plan freezes four UAVs as the only current core setting. Scaling, attrition, VLM/VLA, world-model, and language tracks are deferred until the G2-I public-searchability, scientific, and native-runtime gates pass.

The active research contract is [docs/权威科研执行计划.md](docs/权威科研执行计划.md). The concrete G2-I execution order and reuse boundary are in [docs/g2-i-execution-and-reuse-plan-2026-07-31.md](docs/g2-i-execution-and-reuse-plan-2026-07-31.md). The single operational checklist before formal experiments is [docs/正式实验前执行清单与代码落实计划.md](docs/正式实验前执行清单与代码落实计划.md). Target-model evidence is in [docs/目标模型与论文定位审查_2026-07-29.md](docs/目标模型与论文定位审查_2026-07-29.md); fleet, sensing, baseline, and method-boundary evidence is in [docs/机群传感器基线与方法边界审查_2026-07-29.md](docs/机群传感器基线与方法边界审查_2026-07-29.md). Evidence-backed long-task cost controls are in [docs/高保真长任务成本控制研究_2026-07-31.md](docs/高保真长任务成本控制研究_2026-07-31.md).

## Public and private data

Each authority layout currently writes private scene geometry/USD, a method-visible G1-U task specification with a coarse prior, public episode starts, and integrity hashes. The planned G2-I specification additionally carries a separately versioned target-agnostic inspection atlas. Neither projection contains exact CitySpec geometry, target labels/counts/process, support sites, target coordinates, legal witnesses, split/family labels, generation seeds, or episode seeds.

Targets, target-process assignments, legal witnesses, validity records, and evaluator seeds remain below `evaluator_private/`; starts are public task inputs. A local authority build is a reproducibility/development suite. A formal blind test requires a separately held evaluator partition and service governance that do not yet exist.
