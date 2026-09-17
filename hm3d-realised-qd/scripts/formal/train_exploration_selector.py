"""Admission wrapper for training HM3D exploration candidate selectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from realised_qd.contracts.io import (  # noqa: E402 -- sys.path bootstrap
    payload_label,
    read_json_object,
    write_json_atomic,
)


def audit_training_admission(
    protocol: dict[str, Any],
    exploration_summary: dict[str, Any],
    training_manifest: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if protocol.get("task", {}).get("task_interface") != "hm3d-multi-uav-exploration-v1":
        reasons.append("WRONG_TASK_INTERFACE")
    if exploration_summary.get("status") != "EXPLORATION_TASK_VALID":
        reasons.append("EXPLORATION_NOT_VALID")
    if training_manifest.get("split") != "train":
        reasons.append("TRAINING_MANIFEST_MUST_BE_TRAIN_SPLIT_ONLY")
    if training_manifest.get("contains_test_scenes") is True:
        reasons.append("TRAINING_MANIFEST_CONTAINS_TEST_SCENES")
    if training_manifest.get("synthetic") is True or training_manifest.get("mock") is True:
        reasons.append("SYNTHETIC_OR_MOCK_TRAINING_MANIFEST_FORBIDDEN")
    if not training_manifest.get("scene_ids"):
        reasons.append("MISSING_TRAIN_SCENE_IDS")
    return {
        "schema_version": "hm3d-exploration-selector-training-admission-v1",
        "status": "READY_TO_TRAIN_SELECTOR" if not reasons else "TRAINING_NOT_READY",
        "reasons": reasons,
        "protocol_id": payload_label(protocol, prefix="formal-protocol"),
        "exploration_summary_id": payload_label(exploration_summary, prefix="exploration-summary"),
        "training_manifest_id": payload_label(training_manifest, prefix="training-manifest"),
        "formal_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--exploration-summary", required=True, type=Path)
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = audit_training_admission(
        read_json_object(args.protocol),
        read_json_object(args.exploration_summary),
        read_json_object(args.training_manifest),
    )
    write_json_atomic(args.output, payload)
    return 0 if payload["status"] == "READY_TO_TRAIN_SELECTOR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
