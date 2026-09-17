# Determinism

Determinism is the design goal that shapes everything else, from scene contracts and runtime locks to same-seed canaries and clean-room replay.

## Contracts, tightly bound

Every capture binds these items by SHA-256.

- the City-Lite scene contract and its composed layers
- the CF2X asset
- the collection protocol and its machine-recomputed split certificate
- the runtime lock, covering interpreter, package versions, IsaacLab source tree, GPU and driver floor, and renderer and physics config
- the source revision and tracked-tree digest
- the private scorer manifest commitment

The runtime lock additionally binds `requirements-isaac-capture.lock` by its hash, so a requirements edit that skips the runtime profile regeneration produces a deliberate mismatch.

## Same-seed evidence

The determinism claim reads as *bounded same-seed variation*. An analyzer compares two captures of the same episode seed through camera-local, frame-aligned, class and agent ID semantic comparison under predeclared tolerances. The comparison covers state, sensor summaries, event output, runtime, peak system commit, and disk growth.

The first same-seed attempt compared numeric semantic IDs and failed two metrics because launches can reassign IDs. Frame-aligned `(class, agent_id)` canonicalization fixed the comparison, and the failure history stays in the record as the reason the analyzer works this way.

## Clean-room replay

A second machine and operator replay the CPU fixture and the public target-free Isaac smoke from a fresh clone, which marks P0-D complete. The primary development machine relies on that external replay.

The repository provides a bounded preparation check.

```powershell
rivermark-clean-room-smoke $env:TEMP\rivermark-clean-room-report
```

It requires a clean checkout, clones the requested revision with `--no-local --no-hardlinks`, runs the researcher smoke inside that clone, and writes only `clean_room_report.json`. The report carries the clone revision, fixture manifest hash, and bounded child status.

## CPU-only entry path

The researcher smoke gives the quickest checkout verification and needs only Python and NumPy.

```powershell
python -m pip install -e ".[cpu-ci]"
$out = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $out
Get-Content "$out\researcher_smoke_report.json"
```

It creates a small non-formal fixture, verifies it, reads public arrays, and writes a report on the CPU path. The full CPU test suite.

```powershell
python -m unittest discover -s tests -v
```
