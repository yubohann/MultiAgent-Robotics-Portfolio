# Evaluation

Scoring happens at the boundary between the public submission and the private evaluator. The repository ships the public side (metric definitions, submission schema, local validator) and keeps the hidden truth separate.

## Event-based confirmation

The active event contract is `search-event-submission.v3`. A candidate confirmation must bind a **source observation ID**. The evaluator privately owns that observation's agent and timestamp, and accepts a target match only when the source observation is evaluator-attested as visible for that target. Guessed IDs, cross-agent claims, and stale timing count as false confirmations.

A false-confirmation budget is an eligibility hard gate, alongside safety. Eligibility and success are different things: a policy that emits no events can have zero false positives while finding nothing. Every report shows recall, confirmed-AUC, time-to-first-confirm, false confirmations, collisions/near misses, timeout, effort, and failure rate.

## Metrics

`rivermark_benchmark.metrics` defines the versioned Search3D metric and bootstrap summaries without accepting target coordinates:

```python
from rivermark_benchmark.metrics import bootstrap_summary, score_search_episode

episode = score_search_episode(
    [0.0, 1.0, 2.0], [0, 1, 2], target_count=2, time_budget_s=2.0
)
summary = bootstrap_summary([episode.normalized_confirmed_auc], metric="normalized_confirmed_auc")
```

The public metric code is a scoring/aggregation contract. The private evaluator remains the authority on true confirmations for blind splits.

## Submissions

A submission (`evaluator_submission_v1`) contains only evaluator-produced timestamps and cumulative confirmation counts, plus bindings for the dataset index, split, evaluator build, policy revision, checkpoint, and seed. Target coordinates, private truth, and reward traces are rejected.

Validate and score a local submission without Isaac:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
python -m rivermark_benchmark.evaluator .\submission.json `
  --dataset-version 0.1.0 --split validation `
  --dataset-index-sha256 <published-index-sha256> `
  --output .\submission-report.json
```

The local evaluator enforces a 64 MiB submission limit, at most 4096 episodes, and at most 100,000 samples per trace as denial-of-service guards.

## Threat boundary

The main risks the design defends against: truth leakage (private fields rejected), split probing (split binding enforced), replay (duplicate episodes rejected), stale provenance (hash binding), metric manipulation (trace checks), resource exhaustion (caps), and result tampering (input hash + detached signature).

One important honesty note: the local validator and the in-process evaluator prototype are engineering controls, not a deployed blind leaderboard service. A real leaderboard needs an independently operated service, key custody, durable logs, and a published incident policy.
