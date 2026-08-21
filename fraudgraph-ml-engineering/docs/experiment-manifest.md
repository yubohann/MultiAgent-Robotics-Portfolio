# Experiment Manifest

Every reported result should have a companion `manifest.json`. The manifest is a small, dependency-light provenance record that can be created before training starts. It does not contain datasets, predictions, checkpoints, credentials, or raw paths unless you explicitly put them in a note.

## Create a manifest

Record the exact command after the `--` separator:

```powershell
python scripts/record_run_manifest.py `
  --output artifacts/elliptic/mainline_manifest.json `
  --dataset elliptic `
  --seed 42 `
  --config configs/experiments/onchain_main_selection.yaml `
  --note "provider revision and preprocessing decision recorded in the run report" `
  -- python -m fraud_ml_engineering --dataset elliptic --rounds 20 --local_epochs 2 --disable_tb
```

The command does not train a model. It writes a provenance record and prints the recorded Git revision so it can be checked before a long run starts.

## Schema

| Field | Purpose |
| --- | --- |
| `schema_version` | Manifest layout version for future compatibility checks. |
| `recorded_at` | UTC timestamp at which the manifest was created. |
| `repository_commit` | Git SHA when available; `null` outside a Git checkout. |
| `runtime` | Python version, Python implementation, and OS/runtime platform. |
| `experiment.dataset` | Dataset adapter identifier used by the planned run. |
| `experiment.seed` | Explicit seed or `null` when the run has no seed. |
| `experiment.command` | Exact recorded command tokens. |
| `experiment.config_path` | Optional configuration reference. |
| `experiment.notes` | Short data-revision or protocol note. Do not include secrets. |
| `experiment.extra` | Optional JSON-compatible details such as label fraction or split policy. |

## Pairing with a result

Keep the manifest beside the summary, checkpoint, and diagnostics under the ignored `artifacts/` directory. Add the result's Git SHA, dataset revision, selection rule, and held-out metrics to the report. A manifest establishes provenance; it does not replace the protocol requirements in [research-protocol.md](research-protocol.md).
