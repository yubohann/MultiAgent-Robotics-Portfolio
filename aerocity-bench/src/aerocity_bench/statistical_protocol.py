"""Ancestor-level statistical planning for the G2-I formal matrix.

Episode rollouts from the same city layout share geometry, the public atlas,
starts, and the target-process realization.  They are useful repeated
measurements, but are not independent samples.  This module therefore only
uses one already-aggregated value per layout ancestor for uncertainty and
sample-size planning.  It is deliberately a planning aid: a calibration
report can never grant formal-score eligibility.
"""

from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist
from typing import Any

from .canonical import content_hash

PROTOCOL_SCHEMA = "org.aerocity.bench.g2-i-statistical-protocol.v1"
SEARCHABILITY_SCHEMA = "org.aerocity.bench.g2-i-l0-searchability-calibration.v1"
REPORT_SCHEMA = "org.aerocity.bench.g2-i-statistical-planning-report.v1"


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must lie in [0, 1]")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _normal_approximation_power(
    *, sample_size: int, effect: float, sample_sd: float, alpha: float
) -> float:
    """Two-sided paired-normal power approximation, retained only for planning."""

    if sample_size < 1 or sample_sd <= 0.0:
        return 0.0
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha / 2.0)
    noncentrality = abs(effect) * math.sqrt(sample_size) / sample_sd
    return (1.0 - normal.cdf(critical - noncentrality)) + normal.cdf(
        -critical - noncentrality
    )


def _required_ancestor_count(
    *, effect: float, sample_sd: float, alpha: float, desired_power: float
) -> int | None:
    if sample_sd <= 0.0 or effect <= 0.0:
        return None
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha / 2.0)
    target = normal.inv_cdf(desired_power)
    return math.ceil(((critical + target) * sample_sd / effect) ** 2)


def _validated_protocol(protocol: object) -> dict[str, Any]:
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported statistical protocol schema")
    expected = {
        "schema",
        "formal_score_eligible",
        "primary_metric",
        "independent_unit",
        "pairing_key",
        "reference_method_id",
        "comparator_method_ids",
        "alpha_two_sided",
        "desired_power",
        "minimum_detectable_effect",
        "minimum_formal_ancestor_count",
        "bootstrap_replicates",
        "bootstrap_seed",
        "multiple_comparison_control",
        "failure_denominator_policy",
        "protocol_hash",
    }
    if set(protocol) != expected:
        raise ValueError("statistical protocol fields differ")
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    if content_hash(payload) != protocol["protocol_hash"]:
        raise ValueError("statistical protocol hash mismatch")
    if protocol["formal_score_eligible"] is not False:
        raise ValueError("a planning protocol cannot be formally score eligible")
    if protocol["primary_metric"] != "mean_final_confirmed_recall":
        raise ValueError("the primary metric must be ancestor-mean final confirmed recall")
    if protocol["independent_unit"] != "layout_ancestor":
        raise ValueError("the independent unit must be layout_ancestor")
    if protocol["pairing_key"] != "layout_hash":
        raise ValueError("the pairing key must be layout_hash")
    reference = protocol["reference_method_id"]
    comparators = protocol["comparator_method_ids"]
    if (
        not isinstance(reference, str)
        or not reference
        or not isinstance(comparators, list)
        or not comparators
        or any(not isinstance(item, str) or not item for item in comparators)
        or len(comparators) != len(set(comparators))
        or reference in comparators
    ):
        raise ValueError("reference and comparator methods must be distinct non-empty IDs")
    alpha = float(protocol["alpha_two_sided"])
    desired_power = float(protocol["desired_power"])
    mde = float(protocol["minimum_detectable_effect"])
    if not 0.0 < alpha < 1.0 or not 0.0 < desired_power < 1.0 or not 0.0 < mde < 1.0:
        raise ValueError("alpha, desired power, and MDE must lie strictly in (0, 1)")
    if int(protocol["minimum_formal_ancestor_count"]) < 3:
        raise ValueError("minimum formal ancestor count must be at least three")
    if int(protocol["bootstrap_replicates"]) < 1000:
        raise ValueError("bootstrap replicates must be at least 1000")
    if not isinstance(protocol["bootstrap_seed"], int):
        raise ValueError("bootstrap seed must be an integer")
    if protocol["multiple_comparison_control"] != "holm_within_primary_family":
        raise ValueError("primary-family multiplicity control must be Holm")
    if protocol["failure_denominator_policy"] != "retain_all_completed_ancestor_rows":
        raise ValueError("failure denominator policy must retain every completed ancestor row")
    return protocol


def _validated_searchability_report(report: object) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema") != SEARCHABILITY_SCHEMA:
        raise ValueError("statistical planning requires a G2-I searchability calibration report")
    if report.get("formal_score_eligible") is not False:
        raise ValueError("formal result reports are not valid statistical-planning inputs")
    expected_hash = report.get("report_hash")
    payload = {key: value for key, value in report.items() if key != "report_hash"}
    if not isinstance(expected_hash, str) or content_hash(payload) != expected_hash:
        raise ValueError("searchability report hash mismatch")
    method_reports = report.get("method_reports")
    if not isinstance(method_reports, list) or not method_reports:
        raise ValueError("searchability report has no method reports")
    return report


def _ancestor_rows(method_report: object, *, method_id: str) -> dict[str, dict[str, Any]]:
    if not isinstance(method_report, dict):
        raise ValueError(f"method report is invalid: {method_id}")
    if method_report.get("requires_private_truth") is not False:
        raise ValueError(f"private-truth method cannot enter the primary comparison: {method_id}")
    rows = method_report.get("ancestors")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"method has no ancestor rows: {method_id}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"method has an invalid ancestor row: {method_id}")
        key = row.get("layout_hash")
        value = row.get("mean_final_confirmed_recall")
        episode_count = row.get("episode_count")
        if (
            not isinstance(key, str)
            or not key
            or key in result
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
            or not isinstance(episode_count, int)
            or episode_count < 1
        ):
            raise ValueError(f"method has an invalid or duplicate ancestor row: {method_id}")
        result[key] = row
    return result


def _bootstrap_interval(
    deltas: list[float], *, replicates: int, seed: int, alpha: float
) -> tuple[float, float]:
    generator = random.Random(seed)
    sample_size = len(deltas)
    means = [
        sum(deltas[generator.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(replicates)
    ]
    return _percentile(means, alpha / 2.0), _percentile(means, 1.0 - alpha / 2.0)


def build_statistical_planning_report(
    protocol: object, searchability_report: object
) -> dict[str, Any]:
    """Plan paired formal comparisons without treating episodes as independent."""

    protocol_node = _validated_protocol(protocol)
    calibration = _validated_searchability_report(searchability_report)
    reports_by_method = {
        str(node.get("method_id")): node
        for node in calibration["method_reports"]
        if isinstance(node, dict)
    }
    reference_id = str(protocol_node["reference_method_id"])
    required_ids = (reference_id, *map(str, protocol_node["comparator_method_ids"]))
    missing = sorted(set(required_ids) - set(reports_by_method))
    if missing:
        raise ValueError(f"statistical protocol methods are absent from calibration: {missing}")

    reference_rows = _ancestor_rows(reports_by_method[reference_id], method_id=reference_id)
    alpha = float(protocol_node["alpha_two_sided"])
    desired_power = float(protocol_node["desired_power"])
    mde = float(protocol_node["minimum_detectable_effect"])
    comparisons: list[dict[str, Any]] = []
    for comparator_id in map(str, protocol_node["comparator_method_ids"]):
        comparator_rows = _ancestor_rows(reports_by_method[comparator_id], method_id=comparator_id)
        if set(comparator_rows) != set(reference_rows):
            raise ValueError(
                "paired methods have different layout-ancestor sets: "
                f"{reference_id}, {comparator_id}"
            )
        ordered_keys = sorted(reference_rows)
        deltas = [
            float(comparator_rows[key]["mean_final_confirmed_recall"])
            - float(reference_rows[key]["mean_final_confirmed_recall"])
            for key in ordered_keys
        ]
        ancestor_count = len(deltas)
        sample_sd = statistics.stdev(deltas) if ancestor_count >= 2 else 0.0
        ci_low, ci_high = _bootstrap_interval(
            deltas,
            replicates=int(protocol_node["bootstrap_replicates"]),
            seed=int(protocol_node["bootstrap_seed"]) + len(comparisons),
            alpha=alpha,
        )
        required = _required_ancestor_count(
            effect=mde,
            sample_sd=sample_sd,
            alpha=alpha,
            desired_power=desired_power,
        )
        required_with_floor = (
            max(int(protocol_node["minimum_formal_ancestor_count"]), required)
            if required is not None
            else None
        )
        comparisons.append(
            {
                "reference_method_id": reference_id,
                "comparator_method_id": comparator_id,
                "layout_ancestor_count": ancestor_count,
                "episode_rows_are_not_independent": True,
                "paired_ancestor_hash": content_hash(ordered_keys),
                "mean_delta_comparator_minus_reference": statistics.fmean(deltas),
                "sample_sd_of_paired_deltas": sample_sd,
                "bootstrap_percentile_ci_95": [ci_low, ci_high],
                "observed_effect_normal_approximation_power": _normal_approximation_power(
                    sample_size=ancestor_count,
                    effect=statistics.fmean(deltas),
                    sample_sd=sample_sd,
                    alpha=alpha,
                ),
                "mde_normal_approximation_power_at_current_ancestors": _normal_approximation_power(
                    sample_size=ancestor_count,
                    effect=mde,
                    sample_sd=sample_sd,
                    alpha=alpha,
                ),
                "estimated_required_ancestors_for_mde": required_with_floor,
                "planning_status": (
                    "INSUFFICIENT_CURRENT_ANCESTORS"
                    if required_with_floor is None or ancestor_count < required_with_floor
                    else "PLANNING_THRESHOLD_MET"
                ),
                "included_failure_rows": {
                    "ancestors_with_any_collision": sum(
                        int(reference_rows[key].get("collision_count", 0)) > 0
                        or int(comparator_rows[key].get("collision_count", 0)) > 0
                        for key in ordered_keys
                    ),
                    "ancestors_with_any_deadline_miss": sum(
                        int(reference_rows[key].get("deadline_misses", 0)) > 0
                        or int(comparator_rows[key].get("deadline_misses", 0)) > 0
                        for key in ordered_keys
                    ),
                    "ancestors_with_any_nonreturn": sum(
                        not bool(reference_rows[key].get("all_returned_home", False))
                        or not bool(comparator_rows[key].get("all_returned_home", False))
                        for key in ordered_keys
                    ),
                },
            }
        )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "overall_status": "CALIBRATION_STATISTICS_ONLY",
        "source_calibration_report_hash": calibration["report_hash"],
        "protocol_hash": protocol_node["protocol_hash"],
        "protocol": {
            "primary_metric": protocol_node["primary_metric"],
            "independent_unit": protocol_node["independent_unit"],
            "pairing_key": protocol_node["pairing_key"],
            "alpha_two_sided": alpha,
            "desired_power": desired_power,
            "minimum_detectable_effect": mde,
            "minimum_formal_ancestor_count": int(protocol_node["minimum_formal_ancestor_count"]),
            "multiple_comparison_control": protocol_node["multiple_comparison_control"],
            "failure_denominator_policy": protocol_node["failure_denominator_policy"],
        },
        "comparisons": comparisons,
        "gate_checks": {
            "all_comparisons_are_ancestor_paired": True,
            "episode_rows_not_treated_as_independent": True,
            "private_oracle_excluded_from_primary_comparisons": True,
            "all_completed_ancestor_failure_rows_retained": True,
            "current_calibration_is_large_enough_for_formal_matrix": all(
                item["planning_status"] == "PLANNING_THRESHOLD_MET" for item in comparisons
            ),
        },
        "next_authorized_step": "COLLECT_PRECOMMITTED_EXTERNAL_L1_ANCESTOR_PANEL",
    }
    report["report_hash"] = content_hash(report)
    return report
