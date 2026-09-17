# Rivermark Benchmark (source)

Auditable native Isaac Sim tooling for Search3D multi-agent 3D stealth-search data collection, validation, and scoring, with eight physically simulated CF2X vehicles in the procedural City-Lite scene.

This directory is the standalone source package. See `../README.md` for the portfolio-level overview, media, and evidence.

## Status

The t1-expert-coverage-v2 collection cohort is **frozen and complete** under active protocol v2, holding the full 4 train and 4 validation unique-candidate sequence.

## Quick Start on CPU

Python 3.10+.

```powershell
python -m pip install -e ".[cpu-ci]"
$output = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $output
Get-Content "$output\researcher_smoke_report.json"
```

Run the CPU test suite.

```powershell
python -m unittest discover -s tests -v
```

## Layout

- `src/rivermark_benchmark/` holds capture, validation, scoring, determinism, and dataset-admission modules
- `config/` holds collection protocols, runtime locks, the label ontology, and the baseline suite
- `schemas/` holds JSON Schema contracts for every artifact
- `docs/` holds key design documents on API and schema stability, asset policy, integrity, and native capture
- `tests/` holds the CPU-runnable test suite with 412 tests

## License

Rivermark-authored source code, schemas, and documentation are licensed under **Apache-2.0** (`LICENSE`).
