# Rivermark Documentation

Rivermark is a benchmark toolchain for multi-agent 3D stealth-search research. Eight physically simulated CF2X vehicles search a procedurally generated city while sensors, state, actions, and provenance are recorded under cryptographic contracts.

These documents cover what the benchmark does, how data is captured and audited, and how to extend the pipeline. They serve researchers who run the pipeline, read the data, or extend the benchmark.

## Where to Start

| Document | What it covers |
|---|---|
| [Overview](overview.md) | What Rivermark is, the evidence it publishes, and its scope |
| [Task and Scene](task-and-scene.md) | The search task and the City-Lite environment |
| [Observation ABI](observation-abi.md) | The field-level contract for episode data |
| [Scoring](scoring.md) | How episodes are scored and how submissions work |
| [Native Capture](capture.md) | Running a native Isaac capture |
| [Video Recording](multistart-native-video-recording.md) | Planning and encoding native videos from both route families |
| [Admission](validation-and-admission.md) | Independent validation and formal dataset admission |
| [Determinism](determinism.md) | Same-seed runs, runtime locks, and clean-room replay |
| [Methods](methods.md) | Supported method families and what counts as evidence |
| [Data Access](data-access.md) | Reading episodes, projections, and the researcher entry path |
| [Governance](governance.md) | Asset provenance, licensing, and API stability |
| [Limitations](limitations.md) | Current scope and the development roadmap |

## Repository Scopes

```text
code/src/rivermark_benchmark/  installable package, capture, validation, scoring, admission
code/config/                   collection protocols, runtime locks, label ontology, baseline suite
code/schemas/                  JSON Schema contracts for every artifact
code/tests/                    CPU test suite for contracts, integrity, and quality gates
docs/                          this documentation set
media/                         rendered videos and key frames
evidence/                      repeatability reports and example manifests
```

## Running the CPU Path

```powershell
cd code
python -m pip install -e ".[cpu-ci]"
$out = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $out
```

See [Data Access](data-access.md) for details.
