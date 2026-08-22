# OR-Tools public-atlas routing baseline

This directory locks the external optimization runtime used by the
`ortools-public-atlas-routing-baseline` development baseline.  The adapter
does not copy OR-Tools code and it is not a claim that OR-Tools supplies a
published 3-D hidden-target-search method.

The method solves a public, sector-constrained vehicle-routing problem over
G2-I inspection cells.  It uses the cells' public poses and represented area,
the public safe-sky flight contract, public starts, and the declared time
budget.  It never receives evaluator-private targets, support sites, witnesses,
the target process, a split label, a private mesh, or rendered depth.

Use a dedicated environment so the benchmark's BSD core has no runtime
dependency on the external solver:

```powershell
py -3.11 -m venv .external-ortools-venv
.\.external-ortools-venv\Scripts\python.exe -m pip install --require-virtualenv -r external\ortools\requirements.txt
```

The exact upstream revision, Apache-2.0 license hash, Python distribution
version, and Windows wheel hash are in `source-lock.json`.  The benchmark
runner starts `tools/ortools_g2i_process_adapter.py` as a separate JSONL
process.  That process only returns high-level waypoints, `OBSERVE`, or
`RETURN`; shared benchmark controllers remain responsible for flight dynamics,
collision handling, dwell validation, confirmation, return, and scoring.

The local L1 launch manifest also pins the SHA-256 of the dedicated virtual
environment's `Scripts/python.exe`.  This prevents Isaac's interpreter from
silently replacing the isolated OR-Tools environment.  Public evidence records
the interpreter hash but never the local path.

The one-time process initialization is capped at 10 seconds and reported
separately; each post-reset action remains subject to the shared 0.15 second
planning deadline.

This is a reproducible external-solver baseline, not a C-gate substitute for a
substantive externally maintained 3-D search method.  It is calibration-only
until the remaining task, physical, statistics, scene, and release gates close.
