# Data Access

Researchers interact with the benchmark through a CPU-only package, a researcher smoke, a lazy episode reader, and (for a future cleared release) a selective downloader.

## Researcher entry

The fastest path to verify a checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
$out = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $out
python -m rivermark_benchmark.fixture verify "$out\fixture\fixture_manifest.json"
```

The smoke checks the fixture manifest and payload hashes, loader shape/timestamp consistency, a public metric submission, and the absence of private truth. The report records the exact source revision and Python/NumPy versions — deliberately not GPU, VRAM, or Isaac claims.

## Lazy episode reads

A raw capture can be several gigabytes, so the loader reads one modality, frame range, and stride at a time:

```python
from rivermark_benchmark import IsaacCapture

capture = IsaacCapture("/data/rivermark/episode-0001")
for frame in capture.iter_frames("onboard", fields=("rgb",), stride=4):
    train_step(frame.values["rgb"], frame.timestamp_ns)
```

Canonical modality names are `rgb`, `depth`, `semantic`, `lidar`, `radar`, `imu`, `state`, `action`, and `language`. Selections are validated and fail closed on unknown, duplicate, empty, or out-of-range values. A raw capture is not formal data until the admission tool accepts its independent receipt.

## Projections

Three projection formats let researchers move data between ecosystems without copying the whole corpus:

- **Zarr v2** — a dependency-light, bounded-chunk projection of selected concrete NPZ streams, with a CPU parity check between the built-in reader and an independent reader.
- **Parquet** — a development-only projection of the three public streams (state/action, public task, public messages), pinned to PyArrow 25.0.0. It never exports raw sensor modalities or evaluator truth.
- **RLDS-shaped JSONL** — a streaming interchange using the RLDS field names (`observation`, `action`, `reward`, `discount`, `is_first`, `is_last`, `is_terminal`). Timing is explicit: `observation[i] + command[i+1] -> observation[i+1]`, because the native stream has no pre-command observation for command 0. Missing rewards are a hard error; the projector never invents zero rewards.

## Release download (future)

When a cleared release exists, use the signed manifest and selective downloader rather than copying everything:

```powershell
rivermark-release-data verify .\release_manifest.json --require-https
rivermark-release-data download .\release_manifest.json $out\release `
  --split validation --modality state --require-https --dry-run
```

`--dry-run` reports the selected shard paths, sizes, hashes, and total bytes without creating a destination. The transfer is sequential, resumable, hash-verified, and atomic. Until a cleared payload exists, these commands are an interface rehearsal and the formal index stays at `episode_count: 0`.
