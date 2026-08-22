"""Admission wrapper for the frozen HM3D formal exploration matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import canonical_sha256, read_json_object, write_json_atomic


def audit_formal_matrix_admission(
    p09_freeze: dict[str, Any],
    baseline_matrix: dict[str, Any],
    metric_registry: dict[str, Any],
    runtime_evidence: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if p09_freeze.get("status") != "P09_PROTOCOL_FROZEN":
        reasons.append("P09_PROTOCOL_NOT_FROZEN")
    if baseline_matrix.get("shared_action_authority") != "public_3d_frontier_team_candidate_set":
        reasons.append("BASELINE_ACTION_AUTHORITY_MISMATCH")
    primary_metrics = metric_registry.get("primary")
    if not isinstance(primary_metrics, list) or not any(
        row.get("id") == "Explored-Free-Flight-Volume-AUC_time"
        for row in primary_metrics
        if isinstance(row, dict)
    ):
        reasons.append("PRIMARY_EXPLORATION_METRIC_NOT_REGISTERED")
    if runtime_evidence.get("test_scene_accessed_before_freeze") is True:
        reasons.append("TEST_SCENE_ACCESS_BEFORE_FREEZE")
    if runtime_evidence.get("failure_denominators", {}).get("complete") is not True:
        reasons.append("INCOMPLETE_FAILURE_DENOMINATORS")
    return {
        "schema_version": "hm3d-formal-exploration-matrix-admission-v1",
        "status": "READY_TO_RUN_FORMAL_MATRIX" if not reasons else "FORMAL_MATRIX_NOT_READY",
        "reasons": reasons,
        "p09_freeze_hash": canonical_sha256(p09_freeze),
        "baseline_matrix_hash": canonical_sha256(baseline_matrix),
        "metric_registry_hash": canonical_sha256(metric_registry),
        "runtime_evidence_hash": canonical_sha256(runtime_evidence),
        "formal_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p09-freeze", required=True, type=Path)
    parser.add_argument("--baseline-matrix", required=True, type=Path)
    parser.add_argument("--metric-registry", required=True, type=Path)
    parser.add_argument("--runtime-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = audit_formal_matrix_admission(
        read_json_object(args.p09_freeze),
        read_json_object(args.baseline_matrix),
        read_json_object(args.metric_registry),
        read_json_object(args.runtime_evidence),
    )
    write_json_atomic(args.output, payload)
    return 0 if payload["status"] == "READY_TO_RUN_FORMAL_MATRIX" else 2


if __name__ == "__main__":
    raise SystemExit(main())
