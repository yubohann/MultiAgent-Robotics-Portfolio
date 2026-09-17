# Engineering Notes

## Refactor intent

The source workspace contained full research code beside dated upload packages, environment folders, duplicated snapshots, raw or generated datasets, model weights, figures, caches, and temporary logs. This repository retains the complete active Python implementation while separating source control from runtime state.

## Key changes

- Converted root-level Python modules into the installable `fraud_ml_engineering` package.
- Replaced bare internal imports with relative package imports.
- Moved the retained SplitGNN implementation into `vendor/splitgnn` and documented its provenance.
- Centralized paths in `paths.py`. Source data lives in `data/`, graph caches in `data/graphs/`, and generated outputs in `artifacts/`.
- Moved SplitGNN YAML configuration into `configs/splitgnn/` and experiment candidate files into `configs/experiments/`.
- Kept focused protocol scripts. Duplicate upload snapshots, notebooks, archives, generated figures, datasets, caches, weights, and historical run folders remain in the source workspace.

## Operational contract

The package resolves dataset roots explicitly, either through the documented local `data/` layout or through a caller-supplied path. Generated output defaults to `artifacts/`, outside version control.

The main command preserves backward-compatible CLI flags used by the research scripts. The active deterministic training path normalizes legacy compatibility options.

## Quality checks

The repository includes structural tests for CPU-only environments. Full runtime verification depends on the dataset and environment, and the release checklist records that scope.
