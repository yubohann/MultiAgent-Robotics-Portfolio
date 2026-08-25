# Rivermark Documentation

Rivermark is a benchmark toolchain for multi-agent 3D stealth-search research: eight physically simulated CF2X vehicles search a procedurally generated city while sensors, state, actions, and provenance are recorded under cryptographic contracts.

These docs explain what the benchmark does, how the data is captured and audited, and how to reproduce or extend the pipeline. They are written for researchers, not as an internal status log.

## Where to start

| Document | What it covers |
|---|---|
| [overview.md](overview.md) | What Rivermark is, what it claims, and where its boundaries are |
| [task-and-scene.md](task-and-scene.md) | The search task and the City-Lite environment |
| [observation-abi.md](observation-abi.md) | The field-level contract for episode data |
| [evaluation.md](evaluation.md) | How episodes are scored and how submissions work |
| [capture.md](capture.md) | Running a native Isaac capture |
| [multistart-native-video-recording.md](multistart-native-video-recording.md) | Planning and encoding native videos from both route families |
| [validation-and-admission.md](validation-and-admission.md) | Independent validation and formal dataset admission |
| [reproducibility.md](reproducibility.md) | Same-seed runs, runtime locks, clean-room reproduction |
| [methods.md](methods.md) | Supported method families and what counts as evidence |
| [data-access.md](data-access.md) | Reading episodes, projections, and the researcher entry path |
| [governance.md](governance.md) | Asset provenance, licensing, and API stability |
| [limitations.md](limitations.md) | Honest list of what the benchmark does not yet do |

## Reproducing the CPU path

```powershell
cd code
python -m pip install -e ".[cpu-ci]"
$out = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $out
```

See [data-access.md](data-access.md) for details.
