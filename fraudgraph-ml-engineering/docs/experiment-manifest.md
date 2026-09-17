# Experiment Manifest

Every reported result should have a companion `manifest.json`. The manifest is a small, dependency-light provenance record that can be created before training starts. Its fields are reserved for provenance metadata.

## Create a manifest

Record the exact command after the `--` separator.

```powershell
python scripts/record_run_manifest.py `
  --output artifacts/elliptic/mainline_manifest.json `
  --dataset elliptic `
  --seed 42 `
  --config configs/experiments/onchain_main_selection.yaml `
  --note "provider revision and preprocessing decision recorded in the run report" `
  -- python -m fraud_ml_engineering --dataset elliptic --rounds 20 --local_epochs 2 --disable_tb
```

The command writes a provenance record and prints the recorded Git revision for checking before a long run starts.

## Schema

| Field | Purpose |
| --- | --- |
| `schema_version` | Manifest layout version for future compatibility checks. |
| `recorded_at` | UTC timestamp at which the manifest was created. |
| `repository_commit` | Git SHA for the recorded commit, `null` outside a Git checkout. |
| `runtime` | Python version, Python implementation, operating system, and runtime platform. |
| `experiment.dataset` | Dataset adapter identifier used by the planned run. |
| `experiment.seed` | Explicit seed for seeded runs, `null` otherwise. |
| `experiment.command` | Exact recorded command tokens. |
| `experiment.config_path` | Optional configuration reference. |
| `experiment.notes` | Short data-revision or protocol note. Reserve this field for shareable details. |
| `experiment.extra` | Optional JSON-compatible details such as label fraction or split policy. |

## Pairing with a result

Keep the manifest beside the summary, checkpoint, and diagnostics inside the local `artifacts/` directory. Add the result's Git SHA, dataset revision, selection rule, and held-out metrics to the report. A manifest establishes provenance alongside the protocol requirements in [research-protocol.md](research-protocol.md).
