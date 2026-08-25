"""Freeze HM3D exploration protocol only when P07/P08/evidence gates are complete."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import canonical_sha256, read_json_object, write_json_atomic


def audit_p09_freeze(
    protocol: dict[str, Any],
    p07_summary: dict[str, Any],
    p08_admission: dict[str, Any],
    runtime_evidence: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if protocol.get("status") != "NOT_FORMAL_RESULT":
        reasons.append("PROTOCOL_STATUS_MUST_BE_NOT_FORMAL_RESULT_BEFORE_FREEZE")
    if p07_summary.get("status") != "P07_EXPLORATION_TASK_VALID":
        reasons.append("P07_NOT_COMPLETE")
    if p08_admission.get("status") != "P08_MECHANISM_MATRIX_VALIDATED":
        reasons.append("P08_NOT_COMPLETE")
    if p08_admission.get("synthetic") is not False:
        reasons.append("P08_REAL_RUNTIME_EVIDENCE_REQUIRED")
    if runtime_evidence.get("test_scene_accessed_before_freeze") is True:
        reasons.append("TEST_SCENE_ACCESSED_BEFORE_FREEZE")
    if runtime_evidence.get("failure_denominators", {}).get("complete") is not True:
        reasons.append("INCOMPLETE_FAILURE_DENOMINATORS")
    if runtime_evidence.get("synthetic") is True or runtime_evidence.get("mock") is True:
        reasons.append("SYNTHETIC_OR_MOCK_EVIDENCE_FORBIDDEN")
    status = "P09_PROTOCOL_FROZEN" if not reasons else "P09_FREEZE_REFUSED"
    return {
        "schema_version": "hm3d-p09-protocol-freeze-v1",
        "status": status,
        "reasons": reasons,
        "protocol_hash": canonical_sha256(protocol),
        "p07_summary_hash": canonical_sha256(p07_summary),
        "p08_admission_hash": canonical_sha256(p08_admission),
        "runtime_evidence_hash": canonical_sha256(runtime_evidence),
        "formal_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--p07-summary", required=True, type=Path)
    parser.add_argument("--p08-admission", required=True, type=Path)
    parser.add_argument("--runtime-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = audit_p09_freeze(
        read_json_object(args.protocol),
        read_json_object(args.p07_summary),
        read_json_object(args.p08_admission),
        read_json_object(args.runtime_evidence),
    )
    write_json_atomic(args.output, payload)
    return 0 if payload["status"] == "P09_PROTOCOL_FROZEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
