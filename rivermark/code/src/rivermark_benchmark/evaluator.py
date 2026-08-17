"""Legacy development trace scorer and evaluator provenance contract.

Version 1 scores confirmation traces that must already have been produced by
an authoritative evaluator.  It is retained for fixture and metric-regression
compatibility; it is not a policy-submission format and cannot establish a
benchmark result by itself.  New policy evaluations use timestamped candidate
events and server-owned truth through :mod:`search_event_evaluator`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import METRIC_VERSION, MetricError, score_search_episode


SUBMISSION_SCHEMA = "org.rivermark.benchmark.evaluator-submission.v1"
RESULT_SCHEMA = "org.rivermark.benchmark.evaluator-result.v1"
EVALUATOR_VERSION = "1.0.0"
SPLITS = frozenset({"train", "inner_dev", "validation", "blind_test", "ood_test"})
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{7,64}$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")
_PRIVATE_KEY_TOKENS = ("target", "truth", "private", "evaluator", "reward", "ground")
MAX_SUBMISSION_BYTES = 64 * 1024 * 1024
MAX_SUBMISSION_EPISODES = 4096
MAX_TRACE_SAMPLES = 100_000


class EvaluatorSubmissionError(ValueError):
    """Raised when a public evaluator submission is malformed or unsafe."""


@dataclass(frozen=True)
class SubmissionIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class EpisodeScore:
    episode_id: str
    normalized_confirmed_auc: float
    final_recall: float
    false_confirmations: int
    time_to_all_targets_s: float | None
    truncated: bool


@dataclass(frozen=True)
class SubmissionReport:
    schema: str
    status: str
    dataset_version: str | None
    dataset_index_sha256: str | None
    split: str | None
    evaluator_id: str | None
    evaluator_version: str | None
    evaluator_sha256: str | None
    metric_version: str | None
    method_id: str | None
    code_revision: str | None
    checkpoint_sha256: str | None
    seed: int | None
    episode_count: int
    scores: tuple[EpisodeScore, ...]
    issues: tuple[SubmissionIssue, ...]
    submission_sha256: str | None = None

    @property
    def valid(self) -> bool:
        return self.status == "valid" and not self.issues

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = [asdict(score) for score in self.scores]
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


def _issue(issues: list[SubmissionIssue], code: str, path: str, message: str) -> None:
    issues.append(SubmissionIssue(code, path, message))


def _mapping(value: Any, *, path: str, issues: list[SubmissionIssue]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        _issue(issues, "type", path, "must be an object")
        return None
    return value


def _unknown(value: Mapping[str, Any], allowed: frozenset[str], *, path: str, issues: list[SubmissionIssue]) -> None:
    for key in sorted(set(value) - allowed):
        _issue(issues, "unknown_field", f"{path}.{key}", "field is not part of evaluator submission v1")


def _private_keys(
    value: Mapping[str, Any],
    *,
    path: str,
    issues: list[SubmissionIssue],
    public_keys: frozenset[str] = frozenset(),
) -> None:
    for key in value:
        if key in public_keys:
            continue
        lowered = str(key).lower()
        if any(token in lowered for token in _PRIVATE_KEY_TOKENS):
            _issue(issues, "private_field", f"{path}.{key}", "target/evaluator truth and reward fields are forbidden")


def _text(value: Any, *, path: str, pattern: re.Pattern[str], issues: list[SubmissionIssue]) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        _issue(issues, "value", path, "has an invalid identifier or version")


def _sha(value: Any, *, path: str, issues: list[SubmissionIssue]) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _issue(issues, "sha256", path, "must be 64 lowercase hexadecimal characters")


def _validate_trace(value: Any, *, path: str, issues: list[SubmissionIssue]) -> dict[str, Any] | None:
    trace = _mapping(value, path=path, issues=issues)
    if trace is None:
        return None
    allowed = frozenset({"timestamps_s", "confirmed_counts", "target_count", "time_budget_s", "false_confirmations", "truncated"})
    _unknown(trace, allowed, path=path, issues=issues)
    _private_keys(trace, path=path, issues=issues, public_keys=allowed)
    required = ("timestamps_s", "confirmed_counts", "target_count", "time_budget_s")
    for key in required:
        if key not in trace:
            _issue(issues, "required", f"{path}.{key}", "required field is missing")
    timestamps = trace.get("timestamps_s")
    counts = trace.get("confirmed_counts")
    if not isinstance(timestamps, list) or not timestamps:
        _issue(issues, "trace", f"{path}.timestamps_s", "must be a non-empty JSON array")
    elif len(timestamps) > MAX_TRACE_SAMPLES:
        _issue(issues, "resource_budget", f"{path}.timestamps_s", f"must contain at most {MAX_TRACE_SAMPLES} samples")
    if not isinstance(counts, list) or not counts:
        _issue(issues, "trace", f"{path}.confirmed_counts", "must be a non-empty JSON array")
    elif len(counts) > MAX_TRACE_SAMPLES:
        _issue(issues, "resource_budget", f"{path}.confirmed_counts", f"must contain at most {MAX_TRACE_SAMPLES} samples")
    target_count = trace.get("target_count")
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
        _issue(issues, "trace", f"{path}.target_count", "must be a positive integer")
    budget = trace.get("time_budget_s")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget <= 0:
        _issue(issues, "trace", f"{path}.time_budget_s", "must be a positive number")
    false_confirmations = trace.get("false_confirmations", 0)
    if isinstance(false_confirmations, bool) or not isinstance(false_confirmations, int) or false_confirmations < 0:
        _issue(issues, "trace", f"{path}.false_confirmations", "must be a non-negative integer")
    truncated = trace.get("truncated", False)
    if not isinstance(truncated, bool):
        _issue(issues, "trace", f"{path}.truncated", "must be boolean")
    return dict(trace)


def validate_submission(
    payload: Any,
    *,
    expected_dataset_version: str | None = None,
    expected_split: str | None = None,
    expected_dataset_index_sha256: str | None = None,
) -> tuple[SubmissionIssue, ...]:
    """Validate a submission without evaluating or accepting private truth."""

    issues: list[SubmissionIssue] = []
    root = _mapping(payload, path="$", issues=issues)
    if root is None:
        return tuple(issues)
    allowed = frozenset({"schema", "dataset_version", "dataset_index_sha256", "split", "evaluator", "policy", "episodes"})
    _unknown(root, allowed, path="$", issues=issues)
    if root.get("schema") != SUBMISSION_SCHEMA:
        _issue(issues, "schema", "$.schema", f"expected {SUBMISSION_SCHEMA!r}")
    dataset_version = root.get("dataset_version")
    if not isinstance(dataset_version, str) or not _SEMVER.fullmatch(dataset_version):
        _issue(issues, "dataset_version", "$.dataset_version", "must be a semantic version")
    elif expected_dataset_version is not None and dataset_version != expected_dataset_version:
        _issue(issues, "dataset_version", "$.dataset_version", "does not match expected dataset version")
    _sha(root.get("dataset_index_sha256"), path="$.dataset_index_sha256", issues=issues)
    if expected_dataset_index_sha256 is not None and root.get("dataset_index_sha256") != expected_dataset_index_sha256:
        _issue(issues, "dataset_index_sha256", "$.dataset_index_sha256", "does not match expected dataset index hash")
    split = root.get("split")
    if split not in SPLITS:
        _issue(issues, "split", "$.split", "unknown evaluation split")
    elif expected_split is not None and split != expected_split:
        _issue(issues, "split", "$.split", "does not match expected split")
    evaluator = _mapping(root.get("evaluator"), path="$.evaluator", issues=issues)
    if evaluator is not None:
        _unknown(evaluator, frozenset({"evaluator_id", "evaluator_version", "evaluator_sha256", "metric_schema"}), path="$.evaluator", issues=issues)
        _text(evaluator.get("evaluator_id"), path="$.evaluator.evaluator_id", pattern=_ID, issues=issues)
        _text(evaluator.get("evaluator_version"), path="$.evaluator.evaluator_version", pattern=_SEMVER, issues=issues)
        _sha(evaluator.get("evaluator_sha256"), path="$.evaluator.evaluator_sha256", issues=issues)
        if evaluator.get("metric_schema") != METRIC_VERSION:
            _issue(issues, "metric_schema", "$.evaluator.metric_schema", f"expected {METRIC_VERSION!r}")
    policy = _mapping(root.get("policy"), path="$.policy", issues=issues)
    if policy is not None:
        _unknown(policy, frozenset({"method_id", "code_revision", "checkpoint_sha256", "seed"}), path="$.policy", issues=issues)
        _text(policy.get("method_id"), path="$.policy.method_id", pattern=_ID, issues=issues)
        _text(policy.get("code_revision"), path="$.policy.code_revision", pattern=_REVISION, issues=issues)
        _sha(policy.get("checkpoint_sha256"), path="$.policy.checkpoint_sha256", issues=issues)
        seed = policy.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            _issue(issues, "seed", "$.policy.seed", "must be a non-negative integer")
    episodes = root.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        _issue(issues, "episodes", "$.episodes", "must be a non-empty array")
        return tuple(issues)
    if len(episodes) > MAX_SUBMISSION_EPISODES:
        _issue(issues, "resource_budget", "$.episodes", f"must contain at most {MAX_SUBMISSION_EPISODES} episodes")
        return tuple(issues)
    seen: set[str] = set()
    for index, raw_episode in enumerate(episodes):
        path = f"$.episodes[{index}]"
        episode = _mapping(raw_episode, path=path, issues=issues)
        if episode is None:
            continue
        _unknown(episode, frozenset({"episode_id", "split", "trace"}), path=path, issues=issues)
        _private_keys(episode, path=path, issues=issues, public_keys=frozenset({"episode_id", "split", "trace"}))
        episode_id = episode.get("episode_id")
        _text(episode_id, path=f"{path}.episode_id", pattern=_ID, issues=issues)
        if isinstance(episode_id, str):
            if episode_id in seen:
                _issue(issues, "duplicate_episode", f"{path}.episode_id", "episode_id must be unique")
            seen.add(episode_id)
        if episode.get("split") != split:
            _issue(issues, "split", f"{path}.split", "episode split must match submission split")
        _validate_trace(episode.get("trace"), path=f"{path}.trace", issues=issues)
    return tuple(issues)


def evaluate_submission(
    payload: Any,
    *,
    expected_dataset_version: str | None = None,
    expected_split: str | None = None,
    expected_dataset_index_sha256: str | None = None,
    submission_sha256: str | None = None,
) -> SubmissionReport:
    """Validate and score a public confirmation-trace submission."""

    issues = list(
        validate_submission(
            payload,
            expected_dataset_version=expected_dataset_version,
            expected_split=expected_split,
            expected_dataset_index_sha256=expected_dataset_index_sha256,
        )
    )
    root = payload if isinstance(payload, Mapping) else {}
    scores: list[EpisodeScore] = []
    episodes = root.get("episodes", []) if isinstance(root, Mapping) else []
    if not issues and isinstance(episodes, list):
        for index, episode in enumerate(episodes):
            trace = episode["trace"]
            try:
                score = score_search_episode(
                    trace["timestamps_s"],
                    trace["confirmed_counts"],
                    target_count=trace["target_count"],
                    time_budget_s=trace["time_budget_s"],
                    false_confirmations=trace.get("false_confirmations", 0),
                    truncated=trace.get("truncated", False),
                )
            except (KeyError, MetricError) as exc:
                _issue(issues, "metric", f"$.episodes[{index}].trace", str(exc))
                continue
            scores.append(EpisodeScore(episode["episode_id"], score.normalized_confirmed_auc, score.final_recall, score.false_confirmations, score.time_to_all_targets_s, score.truncated))
    return SubmissionReport(
        schema=RESULT_SCHEMA,
        status="valid" if not issues else "invalid",
        dataset_version=root.get("dataset_version") if isinstance(root.get("dataset_version"), str) else None,
        dataset_index_sha256=root.get("dataset_index_sha256") if isinstance(root.get("dataset_index_sha256"), str) else None,
        split=root.get("split") if isinstance(root.get("split"), str) else None,
        evaluator_id=(root.get("evaluator", {}).get("evaluator_id") if isinstance(root.get("evaluator"), Mapping) else None),
        evaluator_version=(root.get("evaluator", {}).get("evaluator_version") if isinstance(root.get("evaluator"), Mapping) else None),
        evaluator_sha256=(root.get("evaluator", {}).get("evaluator_sha256") if isinstance(root.get("evaluator"), Mapping) else None),
        metric_version=(root.get("evaluator", {}).get("metric_schema") if isinstance(root.get("evaluator"), Mapping) else None),
        method_id=(root.get("policy", {}).get("method_id") if isinstance(root.get("policy"), Mapping) else None),
        code_revision=(root.get("policy", {}).get("code_revision") if isinstance(root.get("policy"), Mapping) else None),
        checkpoint_sha256=(root.get("policy", {}).get("checkpoint_sha256") if isinstance(root.get("policy"), Mapping) else None),
        seed=(root.get("policy", {}).get("seed") if isinstance(root.get("policy"), Mapping) and isinstance(root.get("policy", {}).get("seed"), int) else None),
        episode_count=len(episodes) if isinstance(episodes, list) else 0,
        scores=tuple(scores),
        issues=tuple(issues),
        submission_sha256=submission_sha256,
    )


def evaluate_submission_file(
    path: Path,
    *,
    output: Path | None = None,
    expected_dataset_version: str | None = None,
    expected_split: str | None = None,
    expected_dataset_index_sha256: str | None = None,
) -> SubmissionReport:
    """Evaluate a UTF-8 JSON submission and optionally write a JSON report."""

    try:
        raw = path.read_bytes()
        if len(raw) > MAX_SUBMISSION_BYTES:
            raise ValueError(f"submission exceeds the {MAX_SUBMISSION_BYTES} byte limit")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        report = SubmissionReport(
            schema=RESULT_SCHEMA,
            status="invalid",
            dataset_version=None,
            dataset_index_sha256=None,
            split=None,
            evaluator_id=None,
            evaluator_version=None,
            evaluator_sha256=None,
            metric_version=None,
            method_id=None,
            code_revision=None,
            checkpoint_sha256=None,
            seed=None,
            episode_count=0,
            scores=(),
            issues=(SubmissionIssue("input", "$", str(exc)),),
            submission_sha256=None,
        )
    else:
        report = evaluate_submission(
            payload,
            expected_dataset_version=expected_dataset_version,
            expected_split=expected_split,
            expected_dataset_index_sha256=expected_dataset_index_sha256,
            submission_sha256=hashlib.sha256(raw).hexdigest(),
        )
    if output is not None:
        if output.resolve() == path.resolve():
            raise EvaluatorSubmissionError("report output must not overwrite the submission")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-version")
    parser.add_argument("--split")
    parser.add_argument("--dataset-index-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = evaluate_submission_file(
            args.submission,
            output=args.output,
            expected_dataset_version=args.dataset_version,
            expected_split=args.split,
            expected_dataset_index_sha256=args.dataset_index_sha256,
        )
    except (OSError, EvaluatorSubmissionError) as exc:
        print(json.dumps({"schema": RESULT_SCHEMA, "status": "invalid", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
