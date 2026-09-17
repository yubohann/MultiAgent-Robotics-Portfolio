# Run Guide

Everything below runs from the `aerocity-bench` directory. Python 3.11 or newer is required.

## Install and focused tests

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_public_boundary.py tests/test_inspection_atlas.py tests/test_ordinary_v3.py -q
```

These three files cover the privacy scope, the inspection atlas and the ordinary-v3 release contracts, and they run on CPU with no simulator.

## Full CPU suite

```powershell
python -m pytest tests -q
```

The suite checks contracts, leakage, generation, schema integrity, host guards and runtime preflights. Tests that need Isaac or a physical asset root report as skipped on a plain CPU host.

## Build a development release

A local build needs a verified open-asset bundle and a writable output directory.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
$assetRoot = (Resolve-Path $env:AEROCITY_ASSET_ROOT)
$outputRoot = Join-Path $env:AEROCITY_OUTPUT_ROOT "ordinary-v1-mini"

python -m aerocity_bench build --release .\configs\releases\ordinary-v1-mini.json `
  --asset-root $assetRoot `
  --output $outputRoot `
  --task-track G2-I `
  --allow-uncommitted-development
python -m aerocity_bench validate $outputRoot
```

Development builds are marked `UNCOMMITTED-DEVELOPMENT`. Official builds refuse a dirty worktree.

## Native and external paths

Native Isaac paths launch child processes and run as bounded, host-guarded jobs on this host.

```powershell
python tools\isaac_native_gate.py --help
python tools\cf2x_l1_fleet_preflight.py --help
```

External method adapters ship with source locks and their own container recipes under `external/`. Each adapter script documents its arguments.

```powershell
python tools\run_ortools_g2i_l0_smoke.py --help
python tools\run_marvel_g2i_l0_smoke.py --help
python tools\run_aco3d_g2i_l0_smoke.py --help
```

## Quality gate

```powershell
powershell -File tools\run_python_quality_gate.ps1
```

## Reading evidence flags

- `formal_score_eligible=true` appears only on frozen formal records. Nothing else counts as a benchmark score.
- `UNCOMMITTED-DEVELOPMENT` marks environment and wheel evidence produced outside a clean frozen commit.
- The governance audit in `tools/audit_experiment_governance.py` recomputes containment, and its report states the current formal status.
