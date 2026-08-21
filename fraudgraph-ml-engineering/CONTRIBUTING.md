# Contributing

FraudGraph ML Engineering is a research engineering portfolio. Contributions should improve reproducibility, measurement quality, or maintainability without hiding negative results.

## Before opening a change

1. Run `python scripts/validate_repository.py`.
2. Run `python -m pytest` when the development dependencies are installed.
3. Run `python -m compileall -q src scripts tests`.
4. Run `python -m build` to confirm the package can be distributed.
5. Install the wheel without dependencies and run `python -m fraud_ml_engineering --help`.
6. Record a manifest for any reported experiment using `scripts/record_run_manifest.py`.
7. Keep datasets, checkpoints, generated reports, and local environment files out of Git.

The `make quality` target runs the validator, tests, and compilation together. The hosted quality workflow additionally builds the wheel and source distribution on Python 3.10 and 3.12.

## Experiment changes

Every new experiment should state its dataset revision, seed policy, split policy, selection metric, and output location. Do not compare a held-out test result selected by test performance. Add an ablation or negative control when a new mechanism is introduced.

## Code changes

Use package-relative imports under `src/fraud_ml_engineering`. Keep repository paths centralized in `paths.py`, preserve the dependency-light CLI help path, and document third-party code or data provenance when adding it.
