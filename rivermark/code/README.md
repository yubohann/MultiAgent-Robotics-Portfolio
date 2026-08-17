# Rivermark Benchmark (source)

Auditable native Isaac Sim tooling for **multi-agent 3D stealth-search (Search3D)** data collection, validation, and evaluation — eight physically simulated CF2X vehicles in the procedural City-Lite scene.

This directory is the standalone source package. See `../README.md` for the portfolio-level overview, media, and evidence.

## Status

The t1-expert-coverage-v2 collection cohort is **frozen and complete**. **No further collection binding is permitted under active protocol v2.** The 4 train + 4 validation unique-candidate sequence is complete.

## Quick Start (CPU, no Isaac Sim required)

Python 3.10+:

```powershell
python -m pip install -e ".[cpu-ci]"
$output = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $output
Get-Content "$output\researcher_smoke_report.json"
```

Run the CPU test suite:

```powershell
python -m unittest discover -s tests -v
```

## Layout

- `src/rivermark_benchmark/` — capture, validation, evaluation, reproducibility, and dataset-admission modules
- `config/` — collection protocols, runtime locks, label ontology, baseline suite
- `schemas/` — JSON Schema contracts for every artifact
- `docs/` — key design documents (API/schema stability, asset policy, integrity, native capture)
- `tests/` — CPU-runnable test suite (412 tests)

## License

Rivermark-authored source code, schemas, and documentation are licensed under **Apache-2.0** (`LICENSE`).
