# Reproducibility

Reproducibility is the design goal that shapes everything else: scene contracts, runtime locks, same-seed canaries, and clean-room reproduction.

## Contracts, not loose pinning

Every capture binds, by SHA-256:

- the City-Lite scene contract and its composed layers
- the CF2X asset
- the collection protocol and its machine-recomputed split certificate
- the runtime lock (interpreter, package versions, IsaacLab source tree, GPU/driver floor, renderer/physics config)
- the source revision and tracked-tree digest
- the private evaluator manifest commitment

The runtime lock additionally binds `requirements-isaac-capture.lock` by its hash, so editing the requirements file without regenerating the runtime profile is a deliberate mismatch.

## Same-seed evidence

The honest reproducibility claim is *bounded same-seed variation, not bitwise determinism*. Two captures of the same episode seed are compared by an analyzer using camera-local, frame-aligned, class-and-agent-ID semantic comparison, under predeclared tolerances. State, sensor summaries, event output, runtime, peak system commit, and disk growth are all compared.

The first same-seed attempt compared numeric semantic IDs and failed two metrics because IDs can be reassigned across launches; frame-aligned `(class, agent_id)` canonicalization fixed that. That failure history is kept — it explains why the analyzer works the way it does.

## Clean-room reproduction

A second machine and operator must reproduce both the CPU fixture and the public target-free Isaac smoke from a fresh clone before the project can claim P0-D complete. The primary development machine cannot certify itself.

The repository provides a bounded preparation check:

```powershell
rivermark-clean-room-smoke $env:TEMP\rivermark-clean-room-report
```

It refuses a dirty checkout, clones the requested revision with `--no-local --no-hardlinks`, runs the researcher smoke inside that clone, and writes only `clean_room_report.json`. The report carries the clone revision, fixture manifest hash, and bounded child status — no command logs, local paths, or private truth.

## CPU-only entry path

The quickest way to verify a checkout is the researcher smoke, which needs only Python and NumPy:

```powershell
python -m pip install -e ".[cpu-ci]"
$out = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $out
Get-Content "$out\researcher_smoke_report.json"
```

It creates a small non-formal fixture, verifies it, reads public arrays, and writes a report without starting Isaac Sim. The full CPU test suite:

```powershell
python -m unittest discover -s tests -v
```
