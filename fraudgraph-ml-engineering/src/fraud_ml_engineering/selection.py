from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

VALIDATION_ONLY_POLICY_NAME = (
    "validation_only_auc_then_f1macro_then_gmean_then_prauc_then_recall_at_precision"
)
DEFAULT_VALIDATION_METRIC_ORDER = (
    "best_valid_auc",
    "best_valid_f1_macro",
    "best_valid_gmean",
    "best_valid_pr_auc",
    "best_valid_recall_at_precision",
)

_NESTED_FALLBACKS: dict[str, tuple[tuple[str, ...], ...]] = {
    "best_valid_auc": (("best_valid_metrics", "auc"),),
    "best_valid_f1_macro": (("best_valid_metrics", "f1_macro"),),
    "best_valid_gmean": (("best_valid_metrics", "gmean"),),
    "best_valid_pr_auc": (("best_valid_metrics", "pr_auc"),),
    "best_valid_recall_at_precision": (("best_valid_metrics", "recall_at_precision"),),
    "best_valid_threshold": (("best_valid_metrics", "threshold"),),
}


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def validation_metric(payload: Mapping[str, Any], metric_name: str, default: float = -1.0) -> float:
    if metric_name in payload:
        return _coerce_float(payload.get(metric_name), default)
    for path in _NESTED_FALLBACKS.get(metric_name, ()):
        current: Any = payload
        missing = False
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                missing = True
                break
            current = current[key]
        if not missing:
            return _coerce_float(current, default)
    return float(default)


def validation_only_rank(
    payload: Mapping[str, Any],
    *,
    metric_order: Sequence[str] = DEFAULT_VALIDATION_METRIC_ORDER,
    completed_first: bool = False,
    best_round_key: str | None = None,
    prefer_lower_best_round: bool = False,
    rounds_ran_key: str | None = None,
    prefer_lower_rounds_ran: bool = False,
) -> tuple[float, ...]:
    rank: list[float] = []
    if completed_first:
        rank.append(1.0 if bool(payload.get("completed", False)) else 0.0)
    for metric_name in metric_order:
        rank.append(validation_metric(payload, metric_name))
    if best_round_key:
        best_round = _coerce_float(payload.get(best_round_key), 10**9)
        rank.append(-best_round if prefer_lower_best_round else best_round)
    if rounds_ran_key:
        rounds_ran = _coerce_float(payload.get(rounds_ran_key), 10**9)
        rank.append(-rounds_ran if prefer_lower_rounds_ran else rounds_ran)
    return tuple(rank)


def select_best_by_validation(
    candidates: Sequence[Mapping[str, Any]],
    *,
    metric_order: Sequence[str] = DEFAULT_VALIDATION_METRIC_ORDER,
    completed_first: bool = False,
    best_round_key: str | None = None,
    prefer_lower_best_round: bool = False,
    rounds_ran_key: str | None = None,
    prefer_lower_rounds_ran: bool = False,
) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("No candidates were provided for validation-only selection.")
    return max(
        candidates,
        key=lambda item: validation_only_rank(
            item,
            metric_order=metric_order,
            completed_first=completed_first,
            best_round_key=best_round_key,
            prefer_lower_best_round=prefer_lower_best_round,
            rounds_ran_key=rounds_ran_key,
            prefer_lower_rounds_ran=prefer_lower_rounds_ran,
        ),
    )
