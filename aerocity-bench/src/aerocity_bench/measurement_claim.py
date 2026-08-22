"""Fail-closed analysis for AeroCityBench's coverage measurement hypothesis.

The benchmark claim is not that a particular policy wins.  It is whether
free-space coverage alone predicts evaluator-confirmed recall as well as a
model that also knows the area-weighted, legally observed inspection
footprint.  This module operates on one pre-aggregated row per
``layout_ancestor`` and method, so episode or seed replication cannot inflate
the apparent sample size.

It is deliberately a calibration/formal-analysis tool, never an authority to
mark a run formally eligible.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from typing import Any

from .canonical import content_hash

PROTOCOL_SCHEMA = "org.aerocity.bench.g2-i-measurement-claim-protocol.v1"
PANEL_MANIFEST_SCHEMA = "org.aerocity.bench.g2-i-measurement-claim-panel.v1"
REPORT_SCHEMA = "org.aerocity.bench.g2-i-measurement-claim-report.v1"
_OUTCOME = "mean_final_confirmed_recall"
_COVERAGE_FEATURE = "free_space_coverage_auc"
_INSPECTION_FEATURE = "inspection_footprint_auc"
_TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "collision",
        "controller_failure",
        "deadline_exhausted",
        "out_of_bounds_failure",
        "planner_crash",
        "planner_timeout",
        "reset_failure",
        "return_failure",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _validated_protocol(protocol: object) -> dict[str, Any]:
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported measurement-claim protocol schema")
    expected = {
        "schema",
        "formal_score_eligible",
        "outcome_metric",
        "independent_unit",
        "cross_validation",
        "coverage_only_features",
        "augmented_features",
        "method_fixed_effects",
        "minimum_ancestor_count",
        "bootstrap_replicates",
        "bootstrap_seed",
        "failure_denominator_policy",
        "panel_manifest_schema",
        "protocol_hash",
    }
    if set(protocol) != expected:
        raise ValueError("measurement-claim protocol fields differ")
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    if content_hash(payload) != protocol["protocol_hash"]:
        raise ValueError("measurement-claim protocol hash mismatch")
    if protocol["formal_score_eligible"] is not False:
        raise ValueError("measurement-claim protocol cannot be formally score eligible")
    if protocol["outcome_metric"] != _OUTCOME:
        raise ValueError("measurement claim must use ancestor-mean final confirmed recall")
    if protocol["independent_unit"] != "layout_ancestor":
        raise ValueError("measurement claim independent unit must be layout_ancestor")
    if protocol["cross_validation"] != "leave_one_layout_ancestor_out":
        raise ValueError("measurement claim must leave one layout ancestor out")
    if protocol["coverage_only_features"] != [_COVERAGE_FEATURE]:
        raise ValueError("coverage-only model must use only free-space coverage AUC")
    if protocol["augmented_features"] != [_COVERAGE_FEATURE, _INSPECTION_FEATURE]:
        raise ValueError("augmented model must add only legal inspection-footprint AUC")
    if protocol["method_fixed_effects"] is not True:
        raise ValueError("measurement claim must control for public method identity")
    if int(protocol["minimum_ancestor_count"]) < 3:
        raise ValueError("measurement claim requires at least three independent ancestors")
    if int(protocol["bootstrap_replicates"]) < 1000:
        raise ValueError("measurement claim requires at least 1000 bootstrap replicates")
    if not isinstance(protocol["bootstrap_seed"], int):
        raise ValueError("measurement-claim bootstrap seed must be an integer")
    if protocol["failure_denominator_policy"] != "retain_all_completed_ancestor_rows":
        raise ValueError("measurement claim must retain completed failure rows")
    if protocol["panel_manifest_schema"] != PANEL_MANIFEST_SCHEMA:
        raise ValueError("measurement claim panel-manifest schema differs")
    return protocol


def _validated_panel_manifest(
    manifest: object,
    *,
    protocol: dict[str, Any],
) -> tuple[list[str], list[str], str]:
    """Validate the precommitted denominator before reading outcome rows.

    A complete rectangle inferred from the supplied records is insufficient: a
    failed method/layout pair could have been removed along with the entire
    layout.  The separately hashed panel fixes that denominator before any
    recall, coverage, or receipt result is available.
    """

    if not isinstance(manifest, dict) or manifest.get("schema") != PANEL_MANIFEST_SCHEMA:
        raise ValueError("unsupported measurement-claim panel-manifest schema")
    expected = {
        "schema",
        "formal_score_eligible",
        "purpose",
        "protocol_hash",
        "precommitted_before_execution",
        "layout_ancestors",
        "method_ids",
        "panel_hash",
    }
    if set(manifest) != expected:
        raise ValueError("measurement-claim panel-manifest fields differ")
    payload = {key: value for key, value in manifest.items() if key != "panel_hash"}
    if content_hash(payload) != manifest["panel_hash"]:
        raise ValueError("measurement-claim panel-manifest hash mismatch")
    if manifest["formal_score_eligible"] is not False:
        raise ValueError("measurement-claim panel cannot be formally score eligible")
    if manifest["purpose"] != "precommitted_calibration_measurement_panel":
        raise ValueError("measurement-claim panel purpose differs")
    if manifest["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("measurement-claim panel is bound to another protocol")
    if manifest["precommitted_before_execution"] is not True:
        raise ValueError("measurement-claim panel was not precommitted before execution")

    def identifiers(value: object, *, name: str, minimum: int) -> list[str]:
        if (
            not isinstance(value, list)
            or len(value) < minimum
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))
            or value != sorted(value)
        ):
            raise ValueError(f"measurement-claim panel {name} must be sorted unique identifiers")
        return list(value)

    ancestors = identifiers(manifest["layout_ancestors"], name="layout_ancestors", minimum=3)
    methods = identifiers(manifest["method_ids"], name="method_ids", minimum=2)
    return ancestors, methods, str(manifest["panel_hash"])


def _validated_records(
    records: object,
    *,
    expected_ancestors: list[str],
    expected_methods: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError("measurement claim requires non-empty ancestor records")
    expected = {
        "layout_ancestor",
        "method_id",
        "method_uses_private_truth",
        "episode_count",
        "source_run_report_hashes",
        "failure_included",
        "terminal_status_counts",
        _OUTCOME,
        _COVERAGE_FEATURE,
        _INSPECTION_FEATURE,
    }
    normalized: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError("measurement claim record fields differ")
        ancestor = record["layout_ancestor"]
        method = record["method_id"]
        if (
            not isinstance(ancestor, str)
            or not ancestor
            or not isinstance(method, str)
            or not method
        ):
            raise ValueError("measurement claim ancestor or method identifier is invalid")
        pair = (ancestor, method)
        if pair in seen_pairs:
            raise ValueError("measurement claim has duplicate method/ancestor rows")
        seen_pairs.add(pair)
        if record["method_uses_private_truth"] is not False:
            raise ValueError("private-truth methods cannot enter a measurement claim")
        if not isinstance(record["episode_count"], int) or record["episode_count"] < 1:
            raise ValueError("measurement claim episode count is invalid")
        source_hashes = record["source_run_report_hashes"]
        if (
            not isinstance(source_hashes, list)
            or len(source_hashes) != record["episode_count"]
            or source_hashes != sorted(source_hashes)
            or len(source_hashes) != len(set(source_hashes))
            or any(
                not isinstance(item, str) or not _SHA256.fullmatch(item)
                for item in source_hashes
            )
        ):
            raise ValueError("measurement claim source run-report hashes are invalid")
        if record["failure_included"] is not True:
            raise ValueError("measurement claim cannot drop a completed failure row")
        terminal_status_counts = record["terminal_status_counts"]
        if (
            not isinstance(terminal_status_counts, dict)
            or not terminal_status_counts
            or set(terminal_status_counts) - _TERMINAL_STATUSES
            or any(
                not isinstance(count, int) or count < 1
                for count in terminal_status_counts.values()
            )
            or sum(terminal_status_counts.values()) != record["episode_count"]
        ):
            raise ValueError("measurement claim terminal-status counts are invalid")
        values: dict[str, float] = {}
        for key in (_OUTCOME, _COVERAGE_FEATURE, _INSPECTION_FEATURE):
            value = record[key]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"measurement claim {key} is invalid")
            values[key] = float(value)
            if not 0.0 <= values[key] <= 1.0:
                raise ValueError(f"measurement claim {key} must lie in [0, 1]")
        normalized.append(
            {
                "layout_ancestor": ancestor,
                "method_id": method,
                "episode_count": record["episode_count"],
                "source_run_report_hashes": list(source_hashes),
                "terminal_status_counts": dict(sorted(terminal_status_counts.items())),
                **values,
            }
        )
    expected_pairs = {
        (ancestor, method) for ancestor in expected_ancestors for method in expected_methods
    }
    if seen_pairs != expected_pairs:
        raise ValueError("measurement claim records differ from the precommitted panel")
    return normalized


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small full-rank system using deterministic Gaussian elimination."""

    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1.0e-12:
            raise ValueError("measurement-claim model is rank deficient")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[row][-1] for row in range(size)]


def _design_row(
    record: dict[str, Any],
    *,
    features: list[str],
    method_ids: list[str],
) -> list[float]:
    # The first method is the reference effect; its column is absorbed by the
    # intercept.  Both models receive identical public method controls.
    return [
        1.0,
        *(float(record[feature]) for feature in features),
        *(float(record["method_id"] == method_id) for method_id in method_ids[1:]),
    ]


def _fit_ridge(
    records: list[dict[str, Any]],
    *,
    features: list[str],
    method_ids: list[str],
) -> list[float]:
    rows = [_design_row(record, features=features, method_ids=method_ids) for record in records]
    targets = [float(record[_OUTCOME]) for record in records]
    width = len(rows[0])
    system = [[0.0] * width for _ in range(width)]
    right = [0.0] * width
    for row, target in zip(rows, targets, strict=True):
        for first in range(width):
            right[first] += row[first] * target
            for second in range(width):
                system[first][second] += row[first] * row[second]
    # A fixed tiny ridge penalty makes the comparison well-defined when a
    # calibration panel has nearly collinear coverage signals.  It is applied
    # identically to both models and deliberately not tuned on held-out data.
    for index in range(1, width):
        system[index][index] += 1.0e-8
    return _solve_linear_system(system, right)


def _predict(
    coefficients: list[float],
    record: dict[str, Any],
    *,
    features: list[str],
    method_ids: list[str],
) -> float:
    row = _design_row(record, features=features, method_ids=method_ids)
    prediction = sum(
        weight * value for weight, value in zip(coefficients, row, strict=True)
    )
    return min(1.0, max(0.0, prediction))


def _leave_one_ancestor_out_errors(
    records: list[dict[str, Any]],
    ancestors: list[str],
    method_ids: list[str],
    *,
    features: list[str],
) -> dict[str, float]:
    per_ancestor: dict[str, float] = {}
    for held_out in ancestors:
        train = [record for record in records if record["layout_ancestor"] != held_out]
        test = [record for record in records if record["layout_ancestor"] == held_out]
        coefficients = _fit_ridge(train, features=features, method_ids=method_ids)
        per_ancestor[held_out] = statistics.fmean(
            (
                _predict(coefficients, record, features=features, method_ids=method_ids)
                - float(record[_OUTCOME])
            )
            ** 2
            for record in test
        )
    return per_ancestor


def _bootstrap_mean_interval(
    values: list[float], *, replicates: int, seed: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    count = len(values)
    means = [
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(replicates)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def build_measurement_claim_report(
    protocol: object,
    records: object,
    panel_manifest: object,
) -> dict[str, Any]:
    """Compare coverage-only and coverage-plus-inspection prediction out of sample."""

    protocol_node = _validated_protocol(protocol)
    ancestors, methods, panel_hash = _validated_panel_manifest(
        panel_manifest,
        protocol=protocol_node,
    )
    rows = _validated_records(
        records,
        expected_ancestors=ancestors,
        expected_methods=methods,
    )
    coverage_errors = _leave_one_ancestor_out_errors(
        rows,
        ancestors,
        methods,
        features=[_COVERAGE_FEATURE],
    )
    augmented_errors = _leave_one_ancestor_out_errors(
        rows,
        ancestors,
        methods,
        features=[_COVERAGE_FEATURE, _INSPECTION_FEATURE],
    )
    deltas = [coverage_errors[ancestor] - augmented_errors[ancestor] for ancestor in ancestors]
    ci_low, ci_high = _bootstrap_mean_interval(
        deltas,
        replicates=int(protocol_node["bootstrap_replicates"]),
        seed=int(protocol_node["bootstrap_seed"]),
    )
    coverage_mse = statistics.fmean(coverage_errors.values())
    augmented_mse = statistics.fmean(augmented_errors.values())
    ancestor_count = len(ancestors)
    enough = ancestor_count >= int(protocol_node["minimum_ancestor_count"])
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "overall_status": "CALIBRATION_ANALYSIS_ONLY",
        "protocol_hash": protocol_node["protocol_hash"],
        "source_panel_manifest_hash": panel_hash,
        "source_record_set_hash": content_hash(rows),
        "source_run_report_set_hash": content_hash(
            sorted(
                source_hash
                for record in rows
                for source_hash in record["source_run_report_hashes"]
            )
        ),
        "independent_unit": "layout_ancestor",
        "layout_ancestor_count": ancestor_count,
        "method_count": len(methods),
        "episode_rows_are_not_independent": True,
        "precommitted_method_by_ancestor_panel_complete": True,
        "terminal_status_counts": {
            status: sum(record["terminal_status_counts"].get(status, 0) for record in rows)
            for status in sorted(_TERMINAL_STATUSES)
        },
        "all_completed_failure_rows_retained": True,
        "private_truth_methods_excluded": True,
        "cross_validation": "leave_one_layout_ancestor_out",
        "method_fixed_effects": True,
        "models": {
            "coverage_only": {
                "features": [_COVERAGE_FEATURE],
                "ancestor_equal_oos_mse": coverage_mse,
                "ancestor_equal_oos_rmse": math.sqrt(coverage_mse),
            },
            "coverage_plus_legal_inspection": {
                "features": [_COVERAGE_FEATURE, _INSPECTION_FEATURE],
                "inspection_footprint_semantics": (
                    "area_weighted_credit_after_accepted_observe_freshness_range_fov_"
                    "facing_los_dwell_and_runtime_safety"
                ),
                "ancestor_equal_oos_mse": augmented_mse,
                "ancestor_equal_oos_rmse": math.sqrt(augmented_mse),
            },
        },
        "incremental_prediction": {
            "per_ancestor_mse_reduction_coverage_minus_augmented": {
                ancestor: coverage_errors[ancestor] - augmented_errors[ancestor]
                for ancestor in ancestors
            },
            "mean_ancestor_equal_mse_reduction": statistics.fmean(deltas),
            "bootstrap_percentile_ci_95": [ci_low, ci_high],
            "interpretation": (
                "positive values favor coverage_plus_legal_inspection; the interval is "
                "descriptive unless the frozen formal protocol declares an inferential rule"
            ),
        },
        "gate_checks": {
            "complete_method_by_ancestor_panel": True,
            "precommitted_panel_matches_records": True,
            "minimum_ancestor_count_met": enough,
            "formal_evidence_not_granted_by_this_report": True,
        },
        "next_authorized_step": "FREEZE_FORMAL_PROTOCOL_BEFORE_BLIND_TEST_ACCESS",
    }
    report["report_hash"] = content_hash(report)
    return report
