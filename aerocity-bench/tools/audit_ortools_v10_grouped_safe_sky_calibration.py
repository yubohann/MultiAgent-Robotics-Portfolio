"""Audit v10 CF2X calibration evidence using public inputs and public reports only."""

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
from aerocity_bench.cf2x_l1_calibration_aggregate import AGGREGATE_SCHEMA  # noqa: E402

SCHEMA = "org.aerocity.bench.ortools-v10-grouped-safe-sky-calibration-audit.v1"
EXPECTED_ADAPTER_ID = "ortools-public-atlas-routing-v10-grouped-safe-sky"


def _mapping(values: list[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        ancestor, separator, raw_path = value.partition("=")
        if not separator or not ancestor or not raw_path or ancestor in result:
            raise ValueError(f"--{label} must use unique ANCESTOR=PATH values")
        result[ancestor] = Path(raw_path).resolve()
    return result


def build(
    *,
    aggregate_path: Path,
    layouts: dict[str, Path],
    reports: dict[str, Path],
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration audit: {output}")
    aggregate = read_json(aggregate_path)
    if (
        not isinstance(aggregate, dict)
        or aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("result") != "PASS"
        or aggregate.get("adapter", {}).get("adapter_id") != EXPECTED_ADAPTER_ID
    ):
        raise ValueError("aggregate is not a passing v10 grouped-safe-sky calibration")
    aggregate_records = {
        str(record.get("layout_ancestor")): record
        for record in aggregate.get("records", [])
        if isinstance(record, dict)
    }
    if set(layouts) != set(reports) or set(reports) != set(aggregate_records):
        raise ValueError("layouts, public reports, and aggregate ancestors must agree")

    rows: list[dict[str, Any]] = []
    for ancestor in sorted(reports):
        layout = layouts[ancestor]
        report_path = reports[ancestor]
        task_path = layout / "method_public" / "task_spec.json"
        episode_paths = sorted((layout / "method_public" / "episodes").glob("*.json"))
        if not task_path.is_file() or len(episode_paths) != 1:
            raise ValueError("each layout must expose one public task and episode")
        task = read_json(task_path)
        episode = read_json(episode_paths[0])
        report = read_json(report_path)
        if not all(isinstance(value, dict) for value in (task, episode, report)):
            raise ValueError("public calibration inputs must be JSON objects")
        assert_public_report_has_no_private_truth(report)
        assignments = episode.get("mission_sector", {}).get("cell_assignment_by_drone")
        if not isinstance(assignments, dict) or not assignments:
            raise ValueError("public episode lacks a fleet workload assignment")
        execution = report.get("execution")
        final = report.get("final")
        timing = report.get("planning_timing")
        adapter = report.get("external_adapter")
        if not all(isinstance(value, dict) for value in (execution, final, timing, adapter)):
            raise ValueError("public report lacks execution fields")
        if adapter.get("declaration", {}).get("adapter_id") != EXPECTED_ADAPTER_ID:
            raise ValueError("public report uses a different external adapter")
        expected_report_hash = str(aggregate_records[ancestor].get("public_report_file_sha256"))
        if file_hash(report_path) != expected_report_hash:
            raise ValueError("public report differs from aggregate evidence")
        row = {
            "layout_ancestor": ancestor,
            "public_task_sha256": file_hash(task_path),
            "public_episode_sha256": file_hash(episode_paths[0]),
            "assigned_cell_count_by_drone": {
                drone_id: len(cell_ids) for drone_id, cell_ids in sorted(assignments.items())
            },
            "assigned_cell_count": sum(len(cell_ids) for cell_ids in assignments.values()),
            "observe_action_count": int(report["policy_progress"]["observe_action_count"]),
            "observation_receipt_count": int(execution["observation_receipt_count"]),
            "anonymous_confirmation_receipt_count": int(
                execution["confirmed_receipt_count"]
            ),
            "safe_completion": bool(final["safe_completion"]),
            "all_returned_home": bool(final["all_returned_home"]),
            "collision_detected": bool(final["collision_detected"]),
            "out_of_bounds_detected": bool(final["out_of_bounds_detected"]),
            "deadline_miss_tick_count": int(timing["deadline_miss_tick_count"]),
            "adapter_failure_count": int(adapter["failure_count"]),
        }
        rows.append(row)
    nonzero_rows = sum(row["anonymous_confirmation_receipt_count"] > 0 for row in rows)
    safe_rows = sum(row["safe_completion"] for row in rows)
    if not all(math.isfinite(float(row["observation_receipt_count"])) for row in rows):
        raise ValueError("public observation count is not finite")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "formal_score_eligible": False,
        "aggregate_path": aggregate_path.resolve().relative_to(_REPOSITORY).as_posix(),
        "aggregate_sha256": file_hash(aggregate_path),
        "adapter_id": EXPECTED_ADAPTER_ID,
        "rows": rows,
        "summary": {
            "independent_layout_ancestor_count": len(rows),
            "nonzero_confirmation_ancestor_count": nonzero_rows,
            "safe_completion_ancestor_count": safe_rows,
            "total_anonymous_confirmation_receipt_count": sum(
                row["anonymous_confirmation_receipt_count"] for row in rows
            ),
            "total_observation_receipt_count": sum(
                row["observation_receipt_count"] for row in rows
            ),
        },
        "interpretation": {
            "status": "DEVELOPMENT_NONZERO_EVIDENCE_INSUFFICIENT_FOR_STABILITY_GATE",
            "zero_confirmation_is_retained": True,
            "task_contract_changed": False,
            "target_process_changed": False,
            "private_truth_read": False,
            "not_a_formal_method_ranking": True,
            "reason": (
                "Three calibration ancestors are insufficient for the frozen 4-of-5 public-method "
                "stability criterion; one safe, on-time ancestor retained zero anonymous "
                "confirmation."
            ),
        },
    }
    payload["audit_hash"] = content_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--layout", action="append", required=True)
    parser.add_argument("--public-report", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = build(
        aggregate_path=args.aggregate.resolve(),
        layouts=_mapping(args.layout, label="layout"),
        reports=_mapping(args.public_report, label="public-report"),
        output=args.output.resolve(),
    )
    print(f"ORTOOLS_V10_CALIBRATION_AUDIT={audit['audit_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
