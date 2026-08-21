# Research Protocol

## Scope

FraudGraph ML Engineering studies fraud classification when transaction records can be represented both as a relational graph and as ordered behavioral sequences. The active model combines a SplitGNN graph branch, relation/event sequence encoders, and a fusion classifier.

The repository is a protocol implementation. It does not claim that one dataset or one checkpoint is universally best.

## Research questions

| Question | Comparison | Evidence required |
| --- | --- | --- |
| RQ1: Does graph structure help under heterophily? | SplitGNN + Transformer vs. sequence-only and graph-only branches | held-out metrics plus branch diagnostics |
| RQ2: Do ordered behaviors add signal? | full fusion vs. graph-only and late-fusion controls | identical splits, seeds, and training budget |
| RQ3: What survives label scarcity? | low-label mechanism ladder at fixed label fractions | validation-selected checkpoints and uncertainty across seeds |

## Data flow

1. An adapter loads an externally sourced dataset and records its provenance.
2. The adapter constructs graph, relation-sequence, and event-sequence views.
3. The model trains on the training partition and selects checkpoints using validation metrics.
4. The evaluator freezes the selected threshold and reports the held-out test partition.
5. Run artifacts preserve arguments, seed, timing, metrics, diagnostics, and checkpoint paths.

## Evaluation rules

- Keep train, validation, and test masks or time windows fixed within an experiment family.
- Select checkpoints and thresholds on validation data only.
- Report ROC-AUC together with PR-AUC, recall at a stated precision target, F1, calibration/threshold details, and class counts.
- Use multiple seeds for claims about mechanism stability; a single seed is a smoke test, not evidence of superiority.
- Preserve the exact dataset revision and preprocessing configuration with each result.
- Treat test metrics as a final report, never as a tuning signal.

## Minimum ablation matrix

| Run | Graph branch | Sequence branch | Purpose |
| --- | --- | --- | --- |
| graph-only | enabled | disabled | structural baseline |
| sequence-only | disabled | enabled | ordered-behavior baseline |
| late-fusion | enabled | enabled | simple multimodal control |
| mainline | enabled | enabled | proposed fusion path |
| low-label ladder | configurable | configurable | mechanism sensitivity under reduced labels |

## Failure modes and limits

The datasets differ in time semantics, label definitions, class imbalance, and licensing. On-chain tasks may require an external negative-address set. GPU kernels and third-party libraries can introduce nondeterminism. A result is not reproducible from a metric alone; the source data, environment, seed, protocol arguments, and artifact manifest are all part of the claim.
