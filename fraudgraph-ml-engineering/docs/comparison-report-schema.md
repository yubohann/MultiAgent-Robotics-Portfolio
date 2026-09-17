# Auditable Comparison Report

`scripts/generate_auditable_comparison_report.py` creates a comparison report from explicit result records. Every row comes from a supplied record. Scanned directories, inferred metrics, old summaries, and historical model picks stay out of scope, so the comparison ties directly to evidence a reviewer can inspect.

## Input record

Provide each row as a JSON file passed with a separate `--record` flag. Each record must include these fields.

```json
{
  "dataset": "comp",
  "model": "SplitGNN + Transformer",
  "data_revision": "provider-release-2026-07-31",
  "split_policy": "fixed-public-masks",
  "selection_policy": "validation_only_auc_then_f1macro",
  "seed": 30,
  "metrics": {
    "validation": {"auc": 0.0, "pr_auc": 0.0},
    "test": {"auc": 0.0, "pr_auc": 0.0, "f1_macro": 0.0}
  }
}
```

The numeric values above are schema placeholders. Include any additional numeric metrics that are useful for the comparison.

## Validation rules

- Each record must name the dataset, model, data revision, split policy, validation-only selection policy, seed, validation metrics, and test metrics.
- The selection policy must begin with `validation_only`.
- Within a dataset, all supplied records must share exactly one data revision and one split policy.
- The report covers production runs. Filenames containing `smoke`, `debug`, `probe`, or `stagecheck` mark temporary runs.
- The report preserves input order and presents rows as supplied.

## Generate a report

```powershell
python scripts/generate_auditable_comparison_report.py `
  --record artifacts/records/comp_splitgnn_seed30.json `
  --record artifacts/records/comp_graph_baseline_seed30.json `
  --output_root artifacts/experiments/auditable_comparison
```

The output directory contains `auditable_comparison.json` and `auditable_comparison.md`. Each row names its source record file, so provenance traces back to the input artifact.
