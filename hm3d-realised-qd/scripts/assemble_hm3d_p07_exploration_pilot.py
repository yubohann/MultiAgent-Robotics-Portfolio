"""Assemble P07 exploration pilot ledgers into a task-validity summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import canonical_sha256, read_json_object, write_json_atomic


def _load_ledgers(paths: tuple[Path, ...]) -> tuple[dict[str, Any], ...]:
    if not paths:
        raise ValueError("at least one episode ledger is required")
    return tuple(read_json_object(path) for path in paths)


def assemble_p07_summary(ledgers: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    reasons: list[str] = []
    if any(row.get("formal_result") is True for row in ledgers):
        reasons.append("PILOT_INPUT_MUST_NOT_PRETEND_FORMAL_RESULT")
    if any(row.get("synthetic") is True or row.get("mock") is True for row in ledgers):
        reasons.append("SYNTHETIC_OR_MOCK_LEDGER_FORBIDDEN")
    statuses = [row.get("status") for row in ledgers]
    valid_statuses = {"TASK_VALID", "TASK_INVALID_OR_UNCALIBRATED", "FAILED"}
    if any(status not in valid_statuses for status in statuses):
        reasons.append("UNKNOWN_EPISODE_STATUS")
    if not any(status == "TASK_VALID" for status in statuses):
        reasons.append("NO_VALID_EXPLORATION_EPISODE")
    if any(row.get("failure_denominator_complete") is not True for row in ledgers):
        reasons.append("INCOMPLETE_FAILURE_DENOMINATOR")
    scene_ids = [row.get("scene_id") for row in ledgers]
    if any(not isinstance(scene_id, str) or not scene_id for scene_id in scene_ids):
        reasons.append("MISSING_SCENE_ID")
    return {
        "schema_version": "hm3d-p07-exploration-pilot-summary-v1",
        "status": "P07_EXPLORATION_TASK_VALID" if not reasons else "P07_NOT_READY",
        "episode_count": len(ledgers),
        "scene_count": len(set(scene_ids)),
        "reasons": reasons,
        "ledger_hash": canonical_sha256(ledgers),
        "formal_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = assemble_p07_summary(_load_ledgers(tuple(args.ledger)))
    write_json_atomic(args.output, summary)
    return 0 if summary["status"] == "P07_EXPLORATION_TASK_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
