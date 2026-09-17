# Scoring

Scoring happens where the public submission meets the private scorer. The repository ships the public side, the metric definitions, submission schema, and local validator, with hidden truth held separately.

## Event-Based Confirmation

The active event contract is `search-event-submission.v3`. A candidate confirmation must bind a **source observation ID**. The scorer privately owns that observation's agent and timestamp, and accepts a target match when the source observation is scorer-attested as visible for that target. Guessed IDs, cross-agent claims, and stale timing count as false confirmations.

A false-confirmation budget is an eligibility hard gate alongside safety. Eligibility and success are separate, so a silent policy can post zero false positives with zero finds. Every report shows recall, confirmed-AUC, time-to-first-confirm, false confirmations, collisions and near misses, timeout, effort, and failure rate.

## Metrics

`rivermark_benchmark.metrics` defines the versioned Search3D metric and bootstrap summaries over public inputs, keeping target coordinates outside.

```python
from rivermark_benchmark.metrics import bootstrap_summary, score_search_episode

episode = score_search_episode(
    [0.0, 1.0, 2.0], [0, 1, 2], target_count=2, time_budget_s=2.0
)
summary = bootstrap_summary([episode.normalized_confirmed_auc], metric="normalized_confirmed_auc")
```

The public metric code is a scoring and aggregation contract. The private scorer remains the authority on true confirmations for blind splits.

## Submissions

A submission in the `evaluator_submission_v1` schema carries scorer-produced timestamps and cumulative confirmation counts, plus bindings for the dataset index, split, evaluator build, policy revision, checkpoint, and seed. Target coordinates, private truth, and reward traces are rejected.

Validate and score a local submission on the CPU path.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
python -m rivermark_benchmark.evaluator .\submission.json `
  --dataset-version 0.1.0 --split validation `
  --dataset-index-identity <published-index-identity> `
  --output .\submission-report.json
```

The local scorer enforces denial-of-service guards at 64 MiB per submission, 4096 episodes, and 100,000 samples per trace.

## Validation

The design limits truth leakage through private-field rejection, split probing through enforced split binding, replay through duplicate-episode rejection, and metric manipulation through trace checks. Resource caps bound local runs, and detached signatures cover release artifacts.

The local validator and the in-process scorer prototype run on the local machine. A production leaderboard adds an independently operated service, key custody, durable logs, and a published incident policy.
