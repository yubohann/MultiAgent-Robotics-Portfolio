"""Admit the HM3D mechanism matrix after Exploration has proved task validity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from realised_qd.contracts.io import (  # noqa: E402 -- sys.path bootstrap
    payload_label,
    read_json_object,
    write_json_atomic,
)


def audit_mechanism_admission(
    exploration_summary: dict[str, object], mechanism_matrix: dict[str, object]
) -> dict[str, object]:
    reasons: list[str] = []
    if exploration_summary.get("status") != "EXPLORATION_TASK_VALID":
        reasons.append("EXPLORATION_NOT_VALID")
    chain = mechanism_matrix.get("adjacent_ablation_chain")
    if not isinstance(chain, list) or len(chain) < 4:
        reasons.append("ADJACENT_ABLATION_CHAIN_INCOMPLETE")
    if mechanism_matrix.get("schema_version") != "hm3d-exploration-mechanism-matrix-v4":
        reasons.append("QD_MECHANISM_MATRIX_SCHEMA_OUTDATED")
    if mechanism_matrix.get("qd_controls") != [
        "no_qd",
        "planned_qd",
        "outcome_grounded_realised_qd",
    ]:
        reasons.append("QD_CONTROL_CHAIN_INCOMPLETE")
    if not isinstance(mechanism_matrix.get("qd_admission_rule"), str):
        reasons.append("QD_ADMISSION_RULE_MISSING")
    allowed_statuses = {"NOT_FORMAL_RESULT", "ACTIVE_PROTOCOL_NOT_FORMAL_RESULT"}
    if mechanism_matrix.get("status") not in allowed_statuses:
        reasons.append("MECHANISM_MATRIX_STATUS_UNEXPECTED")
    return {
        "schema_version": "hm3d-mechanism-admission-v1",
        "status": "READY_TO_RUN_MECHANISM_MATRIX" if not reasons else "MECHANISM_NOT_READY",
        "reasons": reasons,
        "exploration_summary_id": payload_label(exploration_summary, prefix="exploration-summary"),
        "mechanism_matrix_id": payload_label(mechanism_matrix, prefix="mechanism-matrix"),
        "formal_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-summary", required=True, type=Path)
    parser.add_argument("--mechanism-matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = audit_mechanism_admission(
        read_json_object(args.exploration_summary),
        read_json_object(args.mechanism_matrix),
    )
    write_json_atomic(args.output, payload)
    return 0 if payload["status"] == "READY_TO_RUN_MECHANISM_MATRIX" else 2


if __name__ == "__main__":
    raise SystemExit(main())
