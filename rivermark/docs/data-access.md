# Data Access

Researchers interact with the benchmark through a CPU-only package, a researcher smoke, a lazy episode reader, and a selective downloader for a future cleared release.

## Researcher entry

The fastest path to verify a checkout.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
$out = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $out
python -m rivermark_benchmark.fixture verify "$out\fixture\fixture_manifest.json"
```

The smoke checks the fixture manifest and payload hashes, loader shape and timestamp consistency, a public metric submission, and the private-truth separation. The report records the exact source revision and Python and NumPy versions, describing the CPU path.

## Lazy episode reads

A raw capture can span several gigabytes, so the loader reads one modality, frame range, and stride at a time.

```python
from rivermark_benchmark import IsaacCapture

capture = IsaacCapture("/data/rivermark/episode-0001")
for frame in capture.iter_frames("onboard", fields=("rgb",), stride=4):
    train_step(frame.values["rgb"], frame.timestamp_ns)
```

Canonical modality names are `rgb`, `depth`, `semantic`, `lidar`, `radar`, `imu`, `state`, `action`, and `language`. Selections are validated and fail closed on unknown, duplicate, empty, or out-of-range values. A raw capture becomes formal data when the admission tool accepts its independent receipt.

## Projections

Three projection formats move selected data between ecosystems.

- **Zarr v2**. A dependency-light, bounded-chunk projection of selected concrete NPZ streams, with a CPU parity check between the built-in reader and an independent reader.
- **Parquet**. A development-only projection of the three public streams, state and action, public task, and public messages, pinned to PyArrow 25.0.0. Raw sensor modalities and scorer truth stay out of this format.
- **RLDS-shaped JSONL**. A streaming interchange using the RLDS field names, `observation`, `action`, `reward`, `discount`, `is_first`, `is_last`, and `is_terminal`. Timing is explicit, `observation[i] + command[i+1] -> observation[i+1]`, because command 0 runs before the first observation. Missing rewards raise a hard error, and the projector writes measured values only.

## Release Download

When a cleared release exists, the signed manifest and selective downloader move the requested shards.

```powershell
rivermark-release-data verify .\release_manifest.json --require-https
rivermark-release-data download .\release_manifest.json $out\release `
  --split validation --modality state --require-https --dry-run
```

`--dry-run` reports the selected shard paths, sizes, hashes, and total bytes, leaving the destination untouched. The transfer is sequential, resumable, hash-verified, and atomic. Until a cleared payload lands, these commands rehearse the interface and the formal index reads `episode_count: 0`.
