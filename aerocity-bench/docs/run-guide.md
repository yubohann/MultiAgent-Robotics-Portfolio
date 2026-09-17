# Run Guide

Everything below runs from the `aerocity-bench` directory. Python 3.11 or newer is required.

## Install and focused tests

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/unit/test_public_boundary.py tests/pipeline/test_inspection_atlas.py tests/pipeline/test_ordinary_v3.py -q
```

These three files cover the privacy scope, the inspection atlas and the ordinary-v3 release contracts, and they run on CPU with no simulator.

## Full CPU suite

```powershell
python -m pytest tests -q
```

The suite checks contracts, leakage, generation, geometry, metrics, adapters and runtime behaviour. Tests that need Isaac or a physical asset root report as skipped on a plain CPU host.

## Build a development release

A local build needs a verified open-asset bundle and a writable output directory.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
$assetRoot = (Resolve-Path $env:AEROCITY_ASSET_ROOT)
$outputRoot = Join-Path $env:AEROCITY_OUTPUT_ROOT "ordinary-v1-mini"

python -m aerocity_bench build --release .\configs\releases\ordinary-v1-mini.json `
  --asset-root $assetRoot `
  --output $outputRoot `
  --allow-uncommitted-development
python -m aerocity_bench validate $outputRoot
python -m aerocity_bench export-public $outputRoot --output "$outputRoot-public"
```

Development builds are marked `UNCOMMITTED-DEVELOPMENT`. A build with a frozen 40-character commit records the commit in the release index and the legal manifest.

## Run a baseline

```powershell
python -m aerocity_bench list-baselines
python -m aerocity_bench run-baseline $outputRoot --method random-safe --split train `
  --episode-index 0 --output .\output\run-train-0
python -m aerocity_bench evaluate --run .\output\run-train-0 `
  --episode "$outputRoot\splits\train\<layout-id>\evaluator_private\episodes\episode-0000.json" `
  --duration-s 300
```

## External method adapters

External methods run as isolated JSONL processes with their own runtime and source locks under `external/`. Each adapter documents its arguments.

```powershell
python tools\smoke\run_ortools_g2i_l0_smoke.py --help
python tools\smoke\run_marvel_g2i_l0_smoke.py --help
python tools\smoke\run_aco3d_g2i_l0_smoke.py --help
```

## Quality gate

```powershell
powershell -File tools\quality\run_python_quality_gate.ps1
```

## Reading evidence flags

- `formal_score_eligible=true` appears only on frozen formal records. Nothing else counts as a benchmark score.
- `UNCOMMITTED-DEVELOPMENT` marks evidence produced outside a clean frozen commit.
- Calibration reports carry `overall_status=CALIBRATION_ONLY` and aggregate anonymous outcomes only.
