"""Admission wrapper for training HM3D exploration candidate selectors.

STATUS (2026-08-08): admission audit only.  This script validates that the
protocol / P07 summary / training manifest entitle the project to train a
candidate selector; it does NOT yet train RB-SF-SAC or the QD selector.
The actual RB-SF-SAC training entry (Recurrent Belief-State Shared-Frontier
SAC, QD x RL interface, RFG credit) is an open gap listed in
docs/主方法严格设计_realised_QD_RFG_RB_SF_SAC_2026-08-08.md section 10.
Do not treat READY_TO_TRAIN_SELECTOR as a trained checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import canonical_sha256, read_json_object, write_json_atomic


def audit_training_admission(
    protocol: dict[str, Any],
    p07_summary: dict[str, Any],
    training_manifest: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if protocol.get("task", {}).get("task_interface") != "hm3d-multi-uav-exploration-v1":
        reasons.append("WRONG_TASK_INTERFACE")
    if p07_summary.get("status") != "P07_EXPLORATION_TASK_VALID":
        reasons.append("P07_EXPLORATION_NOT_VALID")
    if training_manifest.get("split") != "train":
        reasons.append("TRAINING_MANIFEST_MUST_BE_TRAIN_SPLIT_ONLY")
    if training_manifest.get("contains_test_scenes") is True:
        reasons.append("TRAINING_MANIFEST_CONTAINS_TEST_SCENES")
    if training_manifest.get("synthetic") is True or training_manifest.get("mock") is True:
        reasons.append("SYNTHETIC_OR_MOCK_TRAINING_MANIFEST_FORBIDDEN")
    if not training_manifest.get("scene_hashes"):
        reasons.append("MISSING_TRAIN_SCENE_HASHES")
    return {
        "schema_version": "hm3d-exploration-selector-training-admission-v1",
        "status": "READY_TO_TRAIN_SELECTOR" if not reasons else "TRAINING_NOT_READY",
        "reasons": reasons,
        "protocol_hash": canonical_sha256(protocol),
        "p07_summary_hash": canonical_sha256(p07_summary),
        "training_manifest_hash": canonical_sha256(training_manifest),
        "formal_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--p07-summary", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = audit_training_admission(
        read_json_object(args.protocol),
        read_json_object(args.p07_summary),
        read_json_object(args.training_manifest),
    )
    write_json_atomic(args.output, payload)
    return 0 if payload["status"] == "READY_TO_TRAIN_SELECTOR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
