# Reproducibility Checklist

Use this checklist for every reported run.

## Environment

- [ ] Python version is between 3.10 and 3.12.
- [ ] CPU or CUDA profile from `requirements/` is recorded.
- [ ] `python scripts/validate_repository.py` passes.
- [ ] `python -m compileall -q src scripts tests` passes.
- [ ] A `manifest.json` records the Git SHA, runtime, dataset, seed, and command.

## Data

- [ ] Dataset provider, revision/date, and license are recorded.
- [ ] Source files are placed under the documented `data/` layout or passed explicitly with a CLI flag.
- [ ] Graph cache generation completed without using an unintended fallback.
- [ ] Class counts and train/validation/test counts are present in the run summary.

## Experiment

- [ ] Seed list is recorded.
- [ ] Label fraction, split policy, and temporal cutoffs are recorded.
- [ ] Model branch flags and fusion variant are recorded.
- [ ] Checkpoint selection metric and fixed precision target are recorded.
- [ ] The held-out test split was not used for tuning.

## Reporting

- [ ] ROC-AUC, PR-AUC, F1, recall, and recall-at-precision are reported where defined.
- [ ] The number of completed rounds and early-stop behavior are reported.
- [ ] At least one negative control or ablation accompanies a mechanism claim.
- [ ] Generated summaries and checkpoints remain under ignored `artifacts/`.
- [ ] The command line and Git commit are preserved with the result.
