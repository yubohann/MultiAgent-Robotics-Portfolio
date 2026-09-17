# Experiment Catalog

This catalog maps each research question to its executable protocol, controls, and expected evidence. Datasets, checkpoints, and reported scores stay local.

## Before any run

1. Acquire the authorized dataset and follow the layout in [data-and-reproduction.md](data-and-reproduction.md).
2. Install the appropriate CPU or CUDA profile from the root README.
3. Create a provenance sidecar with [record_run_manifest.py](../scripts/record_run_manifest.py) before starting a long job.
4. Keep the manifest, command logs, summaries, diagnostics, and selected checkpoint together inside the local `artifacts/` directory.

The default `reuse` checkpoint mode is useful for interrupted work. Use `--checkpoint_mode fresh` when a new, independent run is required, and keep artifacts from different data revisions separate.

## Research protocols

| Research question | Protocol and scope | Primary controls | Required evidence | Result location |
| --- | --- | --- | --- | --- |
| RQ1. Does graph structure add signal? | `scripts/run_hybrid_mainline_protocol.py` compares full, Transformer-only, and SplitGNN-only branches across `yelp`, `amazon`, and `comp`. | Fixed dataset split, seed, rounds, label fraction, deterministic planner, disabled federated training. | Per-seed summaries, validation-selected checkpoint, held-out test metrics, branch diagnostics, and aggregate JSON and Markdown report. | `artifacts/experiments/mainline_protocol/` by default. |
| RQ2. Which fusion rule is justified? | `scripts/run_hybrid_fusion_ablation.py` runs graph-only, late fusion, graph-dominant residual, and shared-private prototype variants. | Same data, seed set, round budget, device profile, and scoring policy across variants. | Per-variant and per-seed summaries plus `fusion_ablation_summary.json` and `.md`. Selection and comparison use validation metrics. | `artifacts/experiments/fusion_ablation/`. |
| RQ3. What remains under scarce labels? | `scripts/run_hybrid_low_label_mechanism_ablation.py` traverses graph-only and increasingly capable hybrid mechanisms at 10%, 5%, and 1% labeled data. | Fixed label fraction per comparison, seed set, round budget, deterministic planner, and mechanism ladder. | Per-seed records, label-fraction aggregates, uncertainty across seeds, and held-out scoring after validation selection. | `artifacts/experiments/low_label_mechanism_ablation/`. |
| IEEE-CIS engineering acceptance | `scripts/run_ieee_acceptance_matrix.py` stages cache build, one round, four rounds, and the target schedule. | Fixed feature, relation, and sampling profiles, temporal split ratios, cache policy, and resource settings. | Stage stdout and stderr, cache or training summaries, resource observations, and acceptance matrix JSON and Markdown. | `artifacts/experiments/ieee_acceptance_matrix/`. |
| IEEE-CIS candidate selection | `scripts/run_ieee_splitgnn_tuning.py` scores typed candidates from `configs/experiments/ieee_splitgnn_tuning.yaml`. | Same IEEE dataset revision, seed, sampling profile, and candidate-stage scoring policy. | Candidate table, selected configuration, validation-only ranking rule, and separately reported test metrics. | `artifacts/experiments/ieee_splitgnn_tuning/`. |

## Configuration references

| Configuration | Used for | Interpretation |
| --- | --- | --- |
| `configs/experiments/onchain_main_selection.yaml` | On-chain candidate selection. | Defines model capacity, optimization, and selection notes for phishing, Ponzi, and rug-pull adapters. The scope stays within the on-chain family. |
| `configs/experiments/ieee_splitgnn_tuning.yaml` | IEEE-CIS tuning. | Provides typed candidate settings and the validation-only selection policy. |
| `configs/experiments/five_dataset_splitgnn_optimizer.yaml` | Multi-dataset optimizer profiles. | Captures dataset-specific candidates. Comparisons hold within the stated dataset protocol. |
| `configs/splitgnn/*.yaml` | SplitGNN benchmark defaults. | Records dataset-level graph benchmark settings used by the compatible runners. |

## Practical entry points

Start with command help to inspect all switches before scheduling work.

```powershell
python scripts/run_hybrid_mainline_protocol.py --help
python scripts/run_hybrid_fusion_ablation.py --help
python scripts/run_hybrid_low_label_mechanism_ablation.py --help
python scripts/run_ieee_acceptance_matrix.py --help
python scripts/run_ieee_splitgnn_tuning.py --help
```

For a narrowly scoped, smoke-level experiment, use one dataset, one seed, one round, CPU, and `--disable_tb`. This verifies environment and data compatibility.

```powershell
python scripts/run_hybrid_fusion_ablation.py --datasets comp --variants graph_only --seeds 30 --rounds 1 --device cpu --disable_tb --checkpoint_mode fresh
```

For a reportable protocol, retain the default three seeds or explicitly justify another seed set. Preserve the generated summaries and diagnostics for every successful and failed run. The paper-package generator at `scripts/generate_hybrid_paper_package.py` consolidates complete artifact trees. Partial runs stay out of the package.

## Interpretation guardrails

- Validation metrics choose candidates and thresholds. Held-out test metrics follow that choice.
- Report performance results from complete runs with a known data revision.
- Report class balance, temporal or split policy, preprocessing revision, device and software context, and failure cases alongside metrics.
- Compare raw scores only across datasets that share label definitions and sampling protocols.

See [research-protocol.md](research-protocol.md), [reproducibility-checklist.md](reproducibility-checklist.md), and [experiment-manifest.md](experiment-manifest.md) for the governing scoring and provenance rules.
