# Experiment Catalog

This catalog maps each research question to its executable protocol, controls, and expected evidence. It documents workflows rather than benchmark claims: the repository intentionally ships no datasets, checkpoints, or reported scores.

## Before any run

1. Acquire the authorized dataset and follow the layout in [data-and-reproduction.md](data-and-reproduction.md).
2. Install the appropriate CPU or CUDA profile from the root README.
3. Create a provenance sidecar with [record_run_manifest.py](../scripts/record_run_manifest.py) before starting a long job.
4. Keep the manifest, command logs, summaries, diagnostics, and selected checkpoint together below the ignored `artifacts/` directory.

The default `reuse` checkpoint mode is useful for interrupted work. Use `--checkpoint_mode fresh` when a new, independent run is required, and never mix artifacts produced from different data revisions.

## Research protocols

| Research question | Protocol and scope | Primary controls | Required evidence | Result location |
| --- | --- | --- | --- | --- |
| RQ1: Does graph structure add signal? | `scripts/run_hybrid_mainline_protocol.py` compares full, Transformer-only, and SplitGNN-only branches across `yelp`, `amazon`, and `comp`. | Fixed dataset split, seed, rounds, label fraction, deterministic planner, disabled federated training. | Per-seed summaries, validation-selected checkpoint, held-out test metrics, branch diagnostics, and aggregate JSON/Markdown report. | `artifacts/experiments/mainline_protocol/` by default. |
| RQ2: Which fusion rule is justified? | `scripts/run_hybrid_fusion_ablation.py` runs graph-only, late fusion, graph-dominant residual, and shared-private prototype variants. | Same data, seed set, round budget, device profile, and evaluation policy across variants. | Per-variant/per-seed summaries plus `fusion_ablation_summary.json` and `.md`; select or compare with validation metrics, not test ranking. | `artifacts/experiments/fusion_ablation/`. |
| RQ3: What remains under scarce labels? | `scripts/run_hybrid_low_label_mechanism_ablation.py` traverses graph-only and increasingly capable hybrid mechanisms at 10%, 5%, and 1% labeled data. | Fixed label fraction per comparison, seed set, round budget, deterministic planner, and mechanism ladder. | Per-seed records, label-fraction aggregates, uncertainty across seeds, and held-out evaluation after validation selection. | `artifacts/experiments/low_label_mechanism_ablation/`. |
| IEEE-CIS engineering acceptance | `scripts/run_ieee_acceptance_matrix.py` stages cache build, one round, four rounds, and the target schedule. | Fixed feature/relation/sampling profiles, temporal split ratios, cache policy, and resource settings. | Stage stdout/stderr, cache or training summaries, resource observations, and acceptance matrix JSON/Markdown. | `artifacts/experiments/ieee_acceptance_matrix/`. |
| IEEE-CIS candidate selection | `scripts/run_ieee_splitgnn_tuning.py` evaluates typed candidates from `configs/experiments/ieee_splitgnn_tuning.yaml`. | Same IEEE dataset revision, seed, sampling profile, and candidate-stage evaluation policy. | Candidate table, selected configuration, validation-only ranking rule, and separately reported test metrics. | `artifacts/experiments/ieee_splitgnn_tuning/`. |

## Configuration references

| Configuration | Used for | Interpretation |
| --- | --- | --- |
| `configs/experiments/onchain_main_selection.yaml` | On-chain candidate selection. | Defines model capacity, optimization, and selection notes for phishing, Ponzi, and rug-pull adapters. It is not a cross-dataset leaderboard. |
| `configs/experiments/ieee_splitgnn_tuning.yaml` | IEEE-CIS tuning. | Provides typed candidate settings and the validation-only selection policy. |
| `configs/experiments/five_dataset_splitgnn_optimizer.yaml` | Multi-dataset optimizer profiles. | Captures dataset-specific candidates; comparisons remain valid only within the stated dataset protocol. |
| `configs/splitgnn/*.yaml` | SplitGNN benchmark defaults. | Records dataset-level graph benchmark settings used by the compatible runners. |

## Practical entry points

Start with command help to inspect all switches before scheduling work:

```powershell
python scripts/run_hybrid_mainline_protocol.py --help
python scripts/run_hybrid_fusion_ablation.py --help
python scripts/run_hybrid_low_label_mechanism_ablation.py --help
python scripts/run_ieee_acceptance_matrix.py --help
python scripts/run_ieee_splitgnn_tuning.py --help
```

For a narrowly scoped, smoke-level experiment, use one dataset, one seed, one round, CPU, and `--disable_tb`. This verifies environment and data compatibility only; it is not evidence for a research claim. For example:

```powershell
python scripts/run_hybrid_fusion_ablation.py --datasets comp --variants graph_only --seeds 30 --rounds 1 --device cpu --disable_tb --checkpoint_mode fresh
```

For a reportable protocol, retain the default three seeds or explicitly justify another seed set. Preserve the generated summaries and diagnostics for every successful and failed run. The paper-package generator at `scripts/generate_hybrid_paper_package.py` can consolidate complete artifact trees, but it does not turn incomplete or smoke-only runs into evidence.

## Interpretation guardrails

- Validation metrics choose candidates and thresholds. Held-out test metrics are recorded only after that choice.
- A single run, a cached result with unknown data revision, or a smoke test must not be described as a performance result.
- Report class balance, temporal or split policy, preprocessing revision, device/software context, and failure cases alongside metrics.
- Do not compare raw scores across datasets with different label definitions or sampling protocols.

See [research-protocol.md](research-protocol.md), [reproducibility-checklist.md](reproducibility-checklist.md), and [experiment-manifest.md](experiment-manifest.md) for the governing evaluation and provenance rules.
