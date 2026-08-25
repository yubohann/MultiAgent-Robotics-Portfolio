"""Ancestor-weighted L0/L1 method-ranking diagnostics."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from .canonical import content_hash

FIDELITY_REPORT_SCHEMA = "org.aerocity.bench.l0-l1-ranking-audit.v1"


def _aggregate(records: list[dict[str, Any]], expected_level: str) -> dict[tuple[str, str], float]:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if set(record) != {
            "method_id",
            "layout_ancestor",
            "score",
            "execution_level",
            "evidence_hash",
        }:
            raise ValueError("fidelity score record fields differ")
        if record["execution_level"] != expected_level:
            raise ValueError("fidelity record has the wrong execution level")
        method_id = str(record["method_id"])
        ancestor = str(record["layout_ancestor"])
        score = float(record["score"])
        evidence_hash = str(record["evidence_hash"])
        if not method_id or not ancestor or len(evidence_hash) != 64:
            raise ValueError("fidelity record identifiers or evidence hash are invalid")
        if not math.isfinite(score):
            raise ValueError("fidelity score must be finite")
        grouped[(method_id, ancestor)].append(score)
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1], ordered[index][1], abs_tol=1.0e-12
        ):
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for method_id, _ in ordered[index:end]:
            ranks[method_id] = average_rank
        index = end
    return ranks


def _pearson(first: list[float], second: list[float]) -> float:
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second, strict=True)
    )
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    if first_scale <= 1.0e-12 or second_scale <= 1.0e-12:
        return 1.0 if first == second else 0.0
    return numerator / (first_scale * second_scale)


def _kendall(first: dict[str, float], second: dict[str, float]) -> float:
    methods = sorted(first)
    concordant = 0
    discordant = 0
    for index, left in enumerate(methods):
        for right in methods[index + 1 :]:
            first_delta = first[left] - first[right]
            second_delta = second[left] - second[right]
            product = first_delta * second_delta
            concordant += int(product > 0.0)
            discordant += int(product < 0.0)
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else 1.0


def compare_l0_l1_rankings(
    l0_records: list[dict[str, Any]], l1_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare rankings after giving every layout ancestor equal weight."""

    l0 = _aggregate(l0_records, "L0")
    l1 = _aggregate(l1_records, "L1")
    if set(l0) != set(l1):
        raise ValueError("L0 and L1 do not contain identical method/ancestor pairs")
    methods = sorted({method for method, _ in l0})
    ancestors = sorted({ancestor for _, ancestor in l0})
    l0_method = {
        method: statistics.fmean(l0[(method, ancestor)] for ancestor in ancestors)
        for method in methods
    }
    l1_method = {
        method: statistics.fmean(l1[(method, ancestor)] for ancestor in ancestors)
        for method in methods
    }
    l0_ranks = _average_ranks(l0_method)
    l1_ranks = _average_ranks(l1_method)
    enough = len(methods) >= 3 and len(ancestors) >= 3
    report: dict[str, Any] = {
        "schema": FIDELITY_REPORT_SCHEMA,
        "formal_score_eligible": False,
        "status": "MEASURED_NOT_FROZEN" if enough else "INSUFFICIENT_DATA",
        "method_count": len(methods),
        "independent_ancestor_count": len(ancestors),
        "episode_replicates_are_not_independent": True,
        "aggregation_semantics": "episode_mean_within_ancestor_then_equal_weight_across_ancestors",
        "spearman_rank_correlation": (
            round(
                _pearson(
                    [l0_ranks[method] for method in methods],
                    [l1_ranks[method] for method in methods],
                ),
                6,
            )
            if enough
            else None
        ),
        "kendall_rank_correlation": (
            round(_kendall(l0_method, l1_method), 6) if enough else None
        ),
        "method_diagnostics": {
            method: {
                "l0_ancestor_mean": round(l0_method[method], 6),
                "l1_ancestor_mean": round(l1_method[method], 6),
                "paired_delta_l1_minus_l0": round(
                    l1_method[method] - l0_method[method], 6
                ),
                "l0_rank": l0_ranks[method],
                "l1_rank": l1_ranks[method],
            }
            for method in methods
        },
        "contract_freeze_allowed": False,
    }
    report["report_hash"] = content_hash(report)
    return report
