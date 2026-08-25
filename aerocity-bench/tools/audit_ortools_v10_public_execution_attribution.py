"""Describe a retained zero-confirmation replay without reading hidden truth.

This development-only audit separates public execution evidence from the
unobserved reason that an otherwise legal observation did not confirm a target.
It is intentionally descriptive: it must not select a new route, alter a task,
or turn the retained zero into an exclusion.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

_REPOSITORY = Path(__file__).resolve().parents[1]
_SOURCE = _REPOSITORY / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json  # noqa: E402
from aerocity_bench.cf2x_fleet_preflight_contract import (  # noqa: E402
    assert_public_report_has_no_private_truth,
)
from aerocity_bench.errors import ValidationError  # noqa: E402

CALIBRATION_AUDIT_SCHEMA = "org.aerocity.bench.ortools-v10-grouped-safe-sky-calibration-audit.v1"
SCHEMA = "org.aerocity.bench.ortools-v10-public-execution-attribution.v1"
EXPECTED_ADAPTER_ID = "ortools-public-atlas-routing-v10-grouped-safe-sky"


def _mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        ancestor, separator, raw_path = value.partition("=")
        if not separator or not ancestor or not raw_path or ancestor in result:
            raise ValueError("--public-report must use unique ANCESTOR=PATH values")
        result[ancestor] = Path(raw_path).resolve()
    return result


def _object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{field} must be an object")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{field} must be a non-negative integer")
    return value


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{field} must be finite and non-negative")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValidationError(f"{field} must be finite and non-negative")
    return number


def _audit_rows(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if audit.get("schema") != CALIBRATION_AUDIT_SCHEMA:
        raise ValidationError("unexpected v10 calibration audit schema")
    if audit.get("formal_score_eligible") is not False:
        raise ValidationError("calibration audit must stay development-only")
    if audit.get("adapter_id") != EXPECTED_ADAPTER_ID:
        raise ValidationError("calibration audit uses a different adapter")
    rows = audit.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValidationError("calibration audit lacks rows")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValidationError("calibration audit contains a non-object row")
        ancestor = row.get("layout_ancestor")
        if not isinstance(ancestor, str) or not ancestor or ancestor in result:
            raise ValidationError("calibration audit rows have invalid ancestors")
        result[ancestor] = row
    return result


def _report_row(
    *, ancestor: str, audit_row: dict[str, Any], report_path: Path
) -> dict[str, Any]:
    report = read_json(report_path)
    if not isinstance(report, dict):
        raise ValidationError(f"{ancestor} public report is not an object")
    assert_public_report_has_no_private_truth(report)
    if report.get("formal_score_eligible") is not False:
        raise ValidationError(f"{ancestor} public report is unexpectedly formal")

    execution = _object(report.get("execution"), field=f"{ancestor}.execution")
    progress = _object(report.get("policy_progress"), field=f"{ancestor}.policy_progress")
    final = _object(report.get("final"), field=f"{ancestor}.final")
    timing = _object(report.get("planning_timing"), field=f"{ancestor}.planning_timing")
    adapter = _object(report.get("external_adapter"), field=f"{ancestor}.external_adapter")
    adapter_declaration = _object(adapter.get("declaration"), field=f"{ancestor}.adapter")
    if adapter_declaration.get("adapter_id") != EXPECTED_ADAPTER_ID:
        raise ValidationError(f"{ancestor} public report uses a different adapter")

    assignments = audit_row.get("assigned_cell_count_by_drone")
    if not isinstance(assignments, dict) or not assignments:
        raise ValidationError(f"{ancestor} audit row lacks public cell assignments")
    assigned_counts = {
        str(drone_id): _nonnegative_int(count, field=f"{ancestor}.assigned_cell_count_by_drone")
        for drone_id, count in assignments.items()
    }
    if not assigned_counts or min(assigned_counts.values()) <= 0:
        raise ValidationError(f"{ancestor} has an empty public drone assignment")
    assigned_total = sum(assigned_counts.values())
    if assigned_total != _nonnegative_int(
        audit_row.get("assigned_cell_count"), field=f"{ancestor}.assigned_cell_count"
    ):
        raise ValidationError(f"{ancestor} public assignment counts disagree")

    observation_count = _nonnegative_int(
        execution.get("observation_receipt_count"), field=f"{ancestor}.observation_receipt_count"
    )
    confirmation_count = _nonnegative_int(
        execution.get("confirmed_receipt_count"), field=f"{ancestor}.confirmed_receipt_count"
    )
    if observation_count != _nonnegative_int(
        progress.get("observe_action_count"), field=f"{ancestor}.observe_action_count"
    ):
        raise ValidationError(f"{ancestor} observe actions and receipts disagree")
    if confirmation_count != _nonnegative_int(
        progress.get("confirmation_receipt_count"), field=f"{ancestor}.confirmation_receipt_count"
    ):
        raise ValidationError(f"{ancestor} confirmation summaries disagree")
    if observation_count != _nonnegative_int(
        audit_row.get("observation_receipt_count"), field=f"{ancestor}.audit observations"
    ) or confirmation_count != _nonnegative_int(
        audit_row.get("anonymous_confirmation_receipt_count"),
        field=f"{ancestor}.audit confirmations",
    ):
        raise ValidationError(f"{ancestor} public report differs from calibration audit")

    status = str(progress.get("status", ""))
    safe_completion = final.get("safe_completion") is True
    all_returned_home = final.get("all_returned_home") is True
    collision_detected = final.get("collision_detected") is True
    out_of_bounds_detected = final.get("out_of_bounds_detected") is True
    deadline_miss_tick_count = _nonnegative_int(
        timing.get("deadline_miss_tick_count"), field=f"{ancestor}.deadline_miss_tick_count"
    )
    adapter_failure_count = _nonnegative_int(
        adapter.get("failure_count"), field=f"{ancestor}.adapter_failure_count"
    )
    return {
        "layout_ancestor": ancestor,
        "public_report_sha256": file_hash(report_path),
        "assigned_cell_count": assigned_total,
        "assigned_cell_count_by_drone": dict(sorted(assigned_counts.items())),
        "assignment_imbalance_ratio": max(assigned_counts.values()) / min(assigned_counts.values()),
        "observation_receipt_count": observation_count,
        "observation_receipts_per_assigned_cell": observation_count / assigned_total,
        "anonymous_confirmation_receipt_count": confirmation_count,
        "execution_closed": status == "CALIBRATION_EPISODE_CLOSED",
        "episode_budget_completed": progress.get("episode_budget_completed") is True,
        "safe_completion": safe_completion,
        "all_returned_home": all_returned_home,
        "collision_detected": collision_detected,
        "out_of_bounds_detected": out_of_bounds_detected,
        "deadline_miss_tick_count": deadline_miss_tick_count,
        "adapter_failure_count": adapter_failure_count,
        "control_tick_count": _nonnegative_int(
            execution.get("control_ticks"), field=f"{ancestor}.control_ticks"
        ),
        "simulated_time_s": _finite_nonnegative(
            execution.get("simulated_time_s"), field=f"{ancestor}.simulated_time_s"
        ),
        "policy_call_p99_s": _finite_nonnegative(
            _object(timing.get("policy_call"), field=f"{ancestor}.policy_call").get("p99_s"),
            field=f"{ancestor}.policy_call.p99_s",
        ),
    }


def build(
    *, calibration_audit_path: Path, reports: dict[str, Path], output: Path
) -> dict[str, Any]:
    """Write a public-only, non-causal attribution for retained zero confirmations."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite public attribution: {output}")
    audit = read_json(calibration_audit_path)
    if not isinstance(audit, dict):
        raise ValidationError("calibration audit is not an object")
    audit_rows = _audit_rows(audit)
    if set(reports) != set(audit_rows):
        raise ValidationError("public report ancestors must exactly match the calibration audit")

    rows = [
        _report_row(
            ancestor=ancestor,
            audit_row=audit_rows[ancestor],
            report_path=reports[ancestor],
        )
        for ancestor in sorted(reports)
    ]
    zero_rows = [row for row in rows if row["anonymous_confirmation_receipt_count"] == 0]
    nonzero_rows = [row for row in rows if row["anonymous_confirmation_receipt_count"] > 0]
    if len(zero_rows) != 1 or not nonzero_rows:
        raise ValidationError(
            "attribution requires exactly one retained zero and at least one nonzero replay"
        )
    zero = zero_rows[0]
    public_execution_closed = (
        zero["execution_closed"]
        and zero["episode_budget_completed"]
        and zero["safe_completion"]
        and zero["all_returned_home"]
        and not zero["collision_detected"]
        and not zero["out_of_bounds_detected"]
        and zero["deadline_miss_tick_count"] == 0
        and zero["adapter_failure_count"] == 0
        and zero["observation_receipt_count"] > 0
    )
    if not public_execution_closed:
        raise ValidationError("retained zero includes a public execution failure")

    audit_relative_path = calibration_audit_path.resolve().relative_to(_REPOSITORY).as_posix()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "formal_score_eligible": False,
        "scope": "development-only-public-execution-attribution",
        "calibration_audit_path": audit_relative_path,
        "calibration_audit_sha256": file_hash(calibration_audit_path),
        "adapter_id": EXPECTED_ADAPTER_ID,
        "rows": rows,
        "retained_zero_confirmation": {
            "layout_ancestor": zero["layout_ancestor"],
            "public_execution_closed": True,
            "status": "PUBLIC_EXECUTION_CAUSE_UNIDENTIFIABLE",
            "publicly_observed_nonfailures": [
                "episode budget completed",
                "observation receipts were emitted",
                "no collision or out-of-bounds event",
                "no planning deadline miss or adapter failure",
                "all four vehicles returned home",
            ],
            "comparison_to_nonzero_replays": {
                "assigned_cell_count_range": [
                    min(row["assigned_cell_count"] for row in nonzero_rows),
                    max(row["assigned_cell_count"] for row in nonzero_rows),
                ],
                "observation_receipt_count_range": [
                    min(row["observation_receipt_count"] for row in nonzero_rows),
                    max(row["observation_receipt_count"] for row in nonzero_rows),
                ],
                "zero_assigned_cell_count": zero["assigned_cell_count"],
                "zero_observation_receipt_count": zero["observation_receipt_count"],
                "zero_assignment_imbalance_ratio": zero["assignment_imbalance_ratio"],
                "interpretation": (
                    "The zero-confirmation replay had more assigned cells and observations "
                    "than the nonzero replays, with a larger public workload imbalance. "
                    "These are descriptive differences, not a causal attribution."
                ),
            },
            "not_identifiable_from_public_records": [
                "which legal observations would overlap a hidden confirmation condition",
                "the hidden placement process",
                "the hidden visibility and dwell evidence for unconfirmed observations",
            ],
            "prohibited_response": [
                "do not remove this replay from the denominator",
                "do not retune route, mission budget, confirmation rule, or target process "
                "from this result",
            ],
            "permitted_follow_up": (
                "Run distinct target-agnostic public methods under the unchanged development "
                "contract; treat this retained zero as evidence against claiming a stability gate."
            ),
        },
    }
    payload["attribution_hash"] = content_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-audit", type=Path, required=True)
    parser.add_argument("--public-report", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    attribution = build(
        calibration_audit_path=args.calibration_audit.resolve(),
        reports=_mapping(args.public_report),
        output=args.output.resolve(),
    )
    print(f"ORTOOLS_V10_PUBLIC_EXECUTION_ATTRIBUTION={attribution['attribution_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
