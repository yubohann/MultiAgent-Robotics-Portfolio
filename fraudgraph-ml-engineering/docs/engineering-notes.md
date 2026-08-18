# Engineering Notes

## Refactor intent

The source workspace contained full research code beside dated upload packages, environment folders, duplicated snapshots, raw or generated datasets, model weights, figures, caches, and temporary logs. This repository retains the complete active Python implementation while separating source control from runtime state.

## Key changes

- Converted root-level Python modules into the installable `fraud_ml_engineering` package.
- Replaced bare internal imports with relative package imports.
- Moved the retained SplitGNN implementation into `vendor/splitgnn` and documented its provenance.
- Centralized paths in `paths.py`: source data lives in `data/`, graph caches in `data/graphs/`, and generated outputs in `artifacts/`.
- Moved SplitGNN YAML configuration into `configs/splitgnn/` and experiment candidate files into `configs/experiments/`.
- Kept focused protocol scripts, while omitting duplicate upload snapshots, notebooks, archives, generated figures, datasets, caches, weights, and historical run folders.

## Operational contract

The package has no hidden machine-specific root path. A caller may either use the documented local `data/` layout or pass a dataset root explicitly. Generated output defaults to `artifacts/`, which is excluded from version control.

The main command intentionally preserves backward-compatible CLI flags used by the research scripts. Some flags are legacy compatibility options and are normalized by the active deterministic training path.

## Quality checks

The repository includes structural tests that do not require a GPU or benchmark data. Full runtime verification remains dataset- and environment-dependent, so the release checklist records those boundaries rather than treating a static check as equivalent to a completed benchmark run.
