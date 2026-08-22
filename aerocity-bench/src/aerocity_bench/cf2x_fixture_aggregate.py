"""Public-only aggregation for internal CF2X closure-fixture receipts.

The individual private-witness fixture reports are deliberately split into a
public summary and a local evaluator-owned private report.  This module only
accepts the public summaries.  It is therefore suitable for recording the
engineering status of development fixtures, but it cannot turn them into
formal L1 scores or validate the private receipt chain by itself.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .canonical import content_hash, read_json, write_json_atomic
from .cf2x_fleet_preflight_contract import (
    PRIVATE_WITNESS_FIXTURE_MODE,
    assert_public_report_has_no_private_truth,
)
from .errors import ValidationError

FIXTURE_LABELS = ("calibration", "train", "validation")
PUBLIC_FIXTURE_SCHEMA = "org.aerocity.bench.cf2x-l1-fleet-preflight.v4"
AGGREGATE_SCHEMA = "org.aerocity.bench.cf2x-private-fixture-aggregate.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValidationError(f"{field} must be a non-negative integer")
    return value


def _positive_finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValidationError(f"{field} must be a finite positive number")
    return result


def _read_closed_fixture(label: str, path: Path) -> dict[str, Any]:
    report = read_json(path)
    if not isinstance(report, dict):
        raise ValidationError(f"{label} public fixture report must be a JSON object")
    report_hash = report.pop("public_report_sha256", None)
    if not isinstance(report_hash, str) or content_hash(report) != report_hash:
        raise ValidationError(f"{label} public fixture report has an invalid content hash")
    if report.get("schema") != PUBLIC_FIXTURE_SCHEMA:
        raise ValidationError(f"{label} public fixture report uses an unsupported schema")
    assert_public_report_has_no_private_truth(report)
    if report.get("execution_mode") != PRIVATE_WITNESS_FIXTURE_MODE:
        raise ValidationError(f"{label} report is not a private-witness fixture")
    if report.get("formal_score_eligible") is not False or report.get(
        "not_a_formal_l1_episode"
    ) is not True:
        raise ValidationError(f"{label} report must explicitly remain non-formal")

    progress = report.get("policy_progress")
    final = report.get("final")
    execution = report.get("execution")
    if not isinstance(progress, dict) or not isinstance(final, dict) or not isinstance(
        execution, dict
    ):
        raise ValidationError(f"{label} report lacks a required public summary")
    if progress.get("status") != "PRIVATE_FIXTURE_CLOSED":
        raise ValidationError(f"{label} private fixture did not close")
    for field in ("observe_action_count", "confirmation_receipt_count", "return_action_count"):
        if _nonnegative_int(progress.get(field), f"{label} {field}") <= 0:
            raise ValidationError(f"{label} private fixture lacks {field}")
    if final != {
        "safe_completion": True,
        "collision_detected": False,
        "out_of_bounds_detected": False,
        "all_returned_home": True,
    }:
        raise ValidationError(f"{label} private fixture safety closure is incomplete")
    control_ticks = _nonnegative_int(execution.get("control_ticks"), f"{label} control_ticks")
    shared_steps = _nonnegative_int(
        execution.get("shared_physx_step_count"), f"{label} shared_physx_step_count"
    )
    if control_ticks <= 0 or shared_steps <= 0:
        raise ValidationError(f"{label} private fixture has no execution evidence")
    return {
        "label": label,
        "public_report_content_sha256": report_hash,
        "private_fixture_commitment": report.get("private_fixture_commitment"),
        "control_ticks": control_ticks,
        "shared_physx_step_count": shared_steps,
        "simulated_time_s": _positive_finite(
            execution.get("simulated_time_s"), f"{label} simulated_time_s"
        ),
        "wall_clock_s": _positive_finite(execution.get("wall_clock_s"), f"{label} wall_clock_s"),
        "observe_action_count": progress["observe_action_count"],
        "confirmation_receipt_count": progress["confirmation_receipt_count"],
        "return_action_count": progress["return_action_count"],
    }


def aggregate_closed_private_fixtures(
    reports_by_label: dict[str, Path],
) -> dict[str, Any]:
    """Build a non-sensitive development-fixture aggregate from public reports."""

    if set(reports_by_label) != set(FIXTURE_LABELS):
        raise ValidationError(
            "fixture aggregate requires exactly calibration, train, and validation reports"
        )
    resolved = {label: path.resolve() for label, path in reports_by_label.items()}
    if len(set(resolved.values())) != len(FIXTURE_LABELS):
        raise ValidationError("fixture aggregate report paths must be distinct")
    summaries = [_read_closed_fixture(label, resolved[label]) for label in FIXTURE_LABELS]
    hashes = [str(item["public_report_content_sha256"]) for item in summaries]
    commitments = [item["private_fixture_commitment"] for item in summaries]
    if len(set(hashes)) != len(hashes):
        raise ValidationError("fixture aggregate public report hashes must be distinct")
    if not all(isinstance(value, str) and _SHA256.fullmatch(value) for value in commitments):
        raise ValidationError("fixture aggregate lacks a valid private commitment")
    if len(set(commitments)) != len(commitments):
        raise ValidationError("fixture aggregate private commitments must be distinct")

    aggregate: dict[str, Any] = {
        "schema": AGGREGATE_SCHEMA,
        "evidence_scope": "cf2x_internal_shared_world_fleet_preflight_aggregate",
        "formal_score_eligible": False,
        "not_a_formal_l1_episode": True,
        "result": "ALL_THREE_DEVELOPMENT_FIXTURES_CLOSED",
        "fixture_count": len(summaries),
        "fixtures": summaries,
        "totals": {
            "control_ticks": sum(int(item["control_ticks"]) for item in summaries),
            "shared_physx_step_count": sum(
                int(item["shared_physx_step_count"]) for item in summaries
            ),
            "observe_action_count": sum(int(item["observe_action_count"]) for item in summaries),
            "confirmation_receipt_count": sum(
                int(item["confirmation_receipt_count"]) for item in summaries
            ),
            "return_action_count": sum(int(item["return_action_count"]) for item in summaries),
        },
        "limitations": [
            "Internal evaluator-owned private-witness fixtures only.",
            "Not a public policy, external method, scored episode, or formal L1 result.",
            "Private target, witness, route, selected-UAV, and evaluator evidence are excluded.",
            "The private receipt chains require separate local CPU validation.",
        ],
    }
    aggregate["aggregate_sha256"] = content_hash(aggregate)
    return aggregate


def write_closed_private_fixture_aggregate(
    reports_by_label: dict[str, Path], output_path: Path
) -> dict[str, Any]:
    """Write a fresh aggregate receipt without overwriting prior evidence."""

    if output_path.suffix.lower() != ".json":
        raise ValidationError("fixture aggregate output must use a .json suffix")
    if output_path.exists():
        raise ValidationError(
            "fixture aggregate output already exists; refusing to overwrite evidence"
        )
    aggregate = aggregate_closed_private_fixtures(reports_by_label)
    write_json_atomic(output_path, aggregate)
    return aggregate
