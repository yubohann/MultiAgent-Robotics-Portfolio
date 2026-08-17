from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.frame_archive import write_chunked_frame_archive
from rivermark_benchmark.repeatability import (
    REPEATABILITY_REPORT_SCHEMA,
    RepeatabilityError,
    build_repeatability_report,
    main,
)

PROFILE = ROOT / "config" / "isaac_repeatability.citylite_t1_v2.json"
USED_ARTIFACTS = (
    "streams/state_action.npz",
    "sensors/imu.npz",
    "sensors/lidar.npz",
    "sensors/onboard_rgbd.npz",
    "learning_labels/semantic_segmentation.npz",
    "learning_labels/semantic_frame_metadata.jsonl",
    "sensors/overview_rgb.npz",
    "task_outcome.json",
)
AGENTS = 8


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _read_semantic_rows(root: Path) -> list[dict[str, object]]:
    path = root / "learning_labels/semantic_frame_metadata.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_semantic_rows(root: Path, rows: list[dict[str, object]]) -> None:
    path = root / "learning_labels/semantic_frame_metadata.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rebind(root: Path) -> None:
    receipt_path = root / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_hashes"] = {
        relative: {
            "bytes": (root / relative).stat().st_size,
            "sha256": _sha256(root / relative),
        }
        for relative in USED_ARTIFACTS
    }
    _write_json(receipt_path, receipt)
    _write_json(
        root / "independent_validation.json",
        {
            "schema": "org.rivermark.isaac-independent-validation.v1",
            "valid": True,
            "issues": [],
            "capture_receipt_sha256": _sha256(receipt_path),
        },
    )


def _capture(root: Path, *, attempt_id: str) -> Path:
    (root / "streams").mkdir(parents=True)
    (root / "sensors").mkdir()
    (root / "learning_labels").mkdir()
    sensor_times = np.asarray([100, 200], dtype=np.int64)
    state_times = np.asarray([50, 100, 150], dtype=np.int64)
    identity_state = np.tile(
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        (3, AGENTS, 1),
    )
    identity_sensor = identity_state[:2]
    zeros_state = np.zeros((3, AGENTS, 3), dtype=np.float32)
    zeros_sensor = np.zeros((2, AGENTS, 3), dtype=np.float32)
    np.savez_compressed(
        root / "streams/state_action.npz",
        command_time_ns=state_times - 1,
        effective_time_ns=state_times,
        root_pos_w_m=zeros_state,
        root_quat_wxyz=identity_state,
        root_lin_vel_w_mps=zeros_state,
        root_ang_vel_b_radps=zeros_state,
        desired_pos_w_m=zeros_state,
        desired_vel_w_mps=zeros_state,
        target_thrust_n=np.full((3, AGENTS, 4), 0.07, dtype=np.float32),
        applied_thrust_n=np.full((3, AGENTS, 4), 0.07, dtype=np.float32),
    )
    np.savez_compressed(
        root / "sensors/imu.npz",
        timestamps_ns=sensor_times,
        pos_w_m=zeros_sensor,
        quat_wxyz=identity_sensor,
        linear_acceleration_b_mps2=zeros_sensor,
        angular_velocity_b_radps=zeros_sensor,
    )
    np.savez_compressed(
        root / "sensors/lidar.npz",
        timestamps_ns=sensor_times,
        pos_w_m=zeros_sensor,
        quat_wxyz=identity_sensor,
        ranges_m=np.full((2, AGENTS, 16), 10.0, dtype=np.float32),
    )
    write_chunked_frame_archive(
        root / "sensors/onboard_rgbd.npz",
        timestamps_ns=sensor_times,
        inline_fields={},
        frame_fields={
            "rgb": np.zeros((2, AGENTS, 3, 4, 3), dtype=np.uint8),
            "distance_to_image_plane_m": np.full(
                (2, AGENTS, 3, 4, 1), 10.0, dtype=np.float32
            ),
        },
    )
    write_chunked_frame_archive(
        root / "learning_labels/semantic_segmentation.npz",
        timestamps_ns=sensor_times,
        inline_fields={},
        frame_fields={
            "semantic_segmentation": np.zeros((2, AGENTS, 3, 4, 1), dtype=np.int32)
        },
    )
    _write_semantic_rows(
        root,
        [
            {
                "schema": "org.rivermark.isaac-semantic-frame-metadata.v1",
                "frame_index": frame_index,
                "timestamp_ns": int(timestamp_ns),
                "onboard_replicator_info": {
                    "per_camera": [
                        {"id_to_labels": {"0": {"class": "BACKGROUND"}}}
                        for _ in range(AGENTS)
                    ]
                },
                "overview_replicator_info": {
                    "per_camera": [
                        {"id_to_labels": {"0": {"class": "BACKGROUND"}}}
                    ]
                },
            }
            for frame_index, timestamp_ns in enumerate(sensor_times)
        ],
    )
    write_chunked_frame_archive(
        root / "sensors/overview_rgb.npz",
        timestamps_ns=sensor_times,
        inline_fields={
            "camera_pos_w_m": np.zeros((2, 3), dtype=np.float64),
            "camera_quat_wxyz": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1)),
            "target_w_m": np.ones((2, 3), dtype=np.float64),
        },
        frame_fields={
            "rgb": np.zeros((2, 6, 8, 3), dtype=np.uint8),
            "semantic_segmentation": np.zeros((2, 6, 8, 1), dtype=np.int32),
        },
    )
    _write_json(
        root / "task_outcome.json",
        {
            "target_observability": {
                "passed": True,
                "per_target_slot": {
                    f"search_target_slot_{index:03d}": {
                        "visible_frames": 10 + index,
                        "max_pixels": 20 + index,
                    }
                    for index in range(4)
                },
            }
        },
    )
    _write_json(
        root / "capture_receipt.json",
        {
            "schema": "org.rivermark.isaac-swarm-capture.v1",
            "status": "captured",
            "ok": True,
            "source_worktree_dirty": False,
            "source_revision": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "capture_attempt_id": attempt_id,
            "evaluator_manifest_sha256": "c" * 64,
            "agent_count_requested": AGENTS,
            "command": {"capture_stride": 10, "steps": 2, "dt_s": 0.005},
            "condition_request": {"cell_id": "train-citylite-direct-v2"},
            "information_profile": "multisensor_rgbd_lidar_imu_state",
            "modalities": {"rgb": "captured", "rtx_radar": "not_captured"},
            "city_lite_scene": {"environment_id": "RIVERMARK_CITY_LITE_v1"},
            "runtime_live": {"runtime_lock_verified": True},
            "target_visibility_execution_window": {
                "capture_stride": 10,
                "dt_s": 0.005,
            },
            "collection_binding": {
                "protocol_id": "citylite-t1-expert-coverage-v2",
                "protocol_sha256": "d" * 64,
                "cell_id": "train-citylite-direct-v2",
                "split": "train",
                "episode_index": 0,
                "episode_seed": 42,
            },
            "runtime_lock": {"profile_id": "runtime-v1", "sha256": "e" * 64},
            "city_lite_authority": {"contract_sha256": "f" * 64},
            "simulator": {"name": "Isaac Sim", "version": "5.1.0.0"},
            "created_wall_time_ns": 1_000_000_000,
            "finished_wall_time_ns": 3_000_000_000,
            "resource_telemetry": {
                "maxima": {"commit_percent": 50.0, "private_commit_bytes": 1024}
            },
            "capture_storage_budget": {"required_bytes": 4096},
            "artifact_hashes": {},
        },
    )
    _rebind(root)
    return root


def _pair(root: Path) -> tuple[Path, Path]:
    reference = _capture(root / "reference", attempt_id="attempt-reference")
    candidate = root / "candidate"
    shutil.copytree(reference, candidate)
    receipt = json.loads(
        (candidate / "capture_receipt.json").read_text(encoding="utf-8")
    )
    receipt["capture_attempt_id"] = "attempt-candidate"
    _write_json(candidate / "capture_receipt.json", receipt)
    _rebind(candidate)
    return reference, candidate


def test_identical_validated_pair_passes_and_cli_writes_schema_valid_report() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        reference, candidate = _pair(root)
        report = build_repeatability_report(reference, candidate, profile_path=PROFILE)
        assert report["schema"] == REPEATABILITY_REPORT_SCHEMA
        assert report["status"] == "passed"
        assert report["failed_metric_count"] == 0
        assert report["analyzer"]["implementation_sha256"] == _sha256(
            SRC / "rivermark_benchmark" / "repeatability.py"
        )
        output = root / "report.json"
        assert (
            main(
                [
                    str(reference),
                    str(candidate),
                    "--profile",
                    str(PROFILE),
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        written = json.loads(output.read_text(encoding="utf-8"))
        digest_payload = dict(written)
        claimed_digest = digest_payload["report_payload_sha256"]
        digest_payload["report_payload_sha256"] = ""
        assert claimed_digest == hashlib.sha256(
            json.dumps(
                digest_payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        from jsonschema import Draft202012Validator

        schema = json.loads(
            (ROOT / "schemas/isaac_repeatability_report_v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        assert list(Draft202012Validator(schema).iter_errors(written)) == []


def test_state_drift_is_reported_without_weakening_other_gates() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        reference, candidate = _pair(Path(temporary))
        state_path = candidate / "streams/state_action.npz"
        with np.load(state_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        arrays["root_pos_w_m"][1, 2, 0] = 0.2
        np.savez_compressed(state_path, **arrays)
        _rebind(candidate)
        report = build_repeatability_report(reference, candidate, profile_path=PROFILE)
        assert report["status"] == "failed"
        assert report["metrics"]["state_action"]["root_position"]["passed"] is False


def test_rgb_drift_is_detected_frame_by_frame() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        reference, candidate = _pair(Path(temporary))
        archive_path = candidate / "sensors/onboard_rgbd.npz"
        archive_path.unlink()
        times = np.asarray([100, 200], dtype=np.int64)
        write_chunked_frame_archive(
            archive_path,
            timestamps_ns=times,
            inline_fields={},
            frame_fields={
                "rgb": np.full((2, AGENTS, 3, 4, 3), 64, dtype=np.uint8),
                "distance_to_image_plane_m": np.full(
                    (2, AGENTS, 3, 4, 1), 10.0, dtype=np.float32
                ),
            },
        )
        _rebind(candidate)
        report = build_repeatability_report(reference, candidate, profile_path=PROFILE)
        assert report["status"] == "failed"
        assert (
            report["metrics"]["frame_archives"][
                "onboard_rgb_frame_mean_abs_error"
            ]["passed"]
            is False
        )


def test_camera_local_semantic_id_reassignment_preserves_label_agreement() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        reference, candidate = _pair(Path(temporary))
        archive_path = candidate / "learning_labels/semantic_segmentation.npz"
        archive_path.unlink()
        write_chunked_frame_archive(
            archive_path,
            timestamps_ns=np.asarray([100, 200], dtype=np.int64),
            inline_fields={},
            frame_fields={
                "semantic_segmentation": np.full(
                    (2, AGENTS, 3, 4, 1), 7, dtype=np.int32
                )
            },
        )
        rows = _read_semantic_rows(candidate)
        for row in rows:
            for camera in row["onboard_replicator_info"]["per_camera"]:
                camera["id_to_labels"] = {"7": {"class": "BACKGROUND"}}
        _write_semantic_rows(candidate, rows)
        _rebind(candidate)
        report = build_repeatability_report(reference, candidate, profile_path=PROFILE)
        assert report["status"] == "passed"
        assert (
            report["metrics"]["frame_archives"][
                "onboard_semantic_label_frame_agreement"
            ]["value"]
            == 1.0
        )


def test_sparse_overview_semantics_use_timestamp_aligned_metadata() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        reference, candidate = _pair(Path(temporary))
        timestamp = np.asarray([200], dtype=np.int64)
        for root, semantic_id in ((reference, 0), (candidate, 7)):
            archive_path = root / "sensors/overview_rgb.npz"
            archive_path.unlink()
            write_chunked_frame_archive(
                archive_path,
                timestamps_ns=timestamp,
                inline_fields={
                    "camera_pos_w_m": np.zeros((1, 3), dtype=np.float64),
                    "camera_quat_wxyz": np.asarray(
                        [[1.0, 0.0, 0.0, 0.0]], dtype=np.float64
                    ),
                    "target_w_m": np.ones((1, 3), dtype=np.float64),
                },
                frame_fields={
                    "rgb": np.zeros((1, 6, 8, 3), dtype=np.uint8),
                    "semantic_segmentation": np.full(
                        (1, 6, 8, 1), semantic_id, dtype=np.int32
                    ),
                },
            )
        rows = _read_semantic_rows(candidate)
        rows[1]["overview_replicator_info"]["per_camera"][0]["id_to_labels"] = {
            "7": {"class": "BACKGROUND"}
        }
        _write_semantic_rows(candidate, rows)
        _rebind(reference)
        _rebind(candidate)
        report = build_repeatability_report(reference, candidate, profile_path=PROFILE)
        assert report["status"] == "passed"
        assert (
            report["metrics"]["frame_archives"][
                "overview_semantic_label_frame_agreement"
            ]["value"]
            == 1.0
        )


def test_semantic_label_change_and_timestamp_mismatch_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        reference, candidate = _pair(Path(temporary))
        rows = _read_semantic_rows(candidate)
        rows[0]["onboard_replicator_info"]["per_camera"][0]["id_to_labels"][
            "0"
        ]["class"] = "building"
        _write_semantic_rows(candidate, rows)
        _rebind(candidate)
        report = build_repeatability_report(reference, candidate, profile_path=PROFILE)
        assert report["status"] == "failed"
        assert (
            report["metrics"]["frame_archives"][
                "onboard_semantic_label_frame_agreement"
            ]["passed"]
            is False
        )

        rows[0]["onboard_replicator_info"]["per_camera"][0]["id_to_labels"][
            "0"
        ]["class"] = "BACKGROUND"
        rows[0]["timestamp_ns"] = 101
        _write_semantic_rows(candidate, rows)
        _rebind(candidate)
        with pytest.raises(RepeatabilityError, match="no frame-aligned metadata"):
            build_repeatability_report(reference, candidate, profile_path=PROFILE)


def test_binding_mismatch_and_stale_validation_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        reference, candidate = _pair(Path(temporary))
        receipt_path = candidate / "capture_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["collection_binding"]["episode_seed"] = 43
        _write_json(receipt_path, receipt)
        _rebind(candidate)
        with pytest.raises(RepeatabilityError, match="same protocol"):
            build_repeatability_report(reference, candidate, profile_path=PROFILE)

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["collection_binding"]["episode_seed"] = 42
        _write_json(receipt_path, receipt)
        with pytest.raises(RepeatabilityError, match="stale"):
            build_repeatability_report(reference, candidate, profile_path=PROFILE)


def test_same_seed_with_different_capture_configuration_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        reference, candidate = _pair(Path(temporary))
        receipt_path = candidate / "capture_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["command"]["capture_stride"] = 20
        _write_json(receipt_path, receipt)
        _rebind(candidate)
        with pytest.raises(RepeatabilityError, match="captures disagree on command"):
            build_repeatability_report(reference, candidate, profile_path=PROFILE)


def test_output_cannot_mutate_a_bound_capture() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        reference, candidate = _pair(Path(temporary))
        output = candidate / "repeatability.json"
        assert (
            main(
                [
                    str(reference),
                    str(candidate),
                    "--profile",
                    str(PROFILE),
                    "--output",
                    str(output),
                ]
            )
            == 2
        )
        assert not output.exists()


def test_zero_norm_quaternion_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        reference, candidate = _pair(Path(temporary))
        state_path = candidate / "streams/state_action.npz"
        with np.load(state_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        arrays["root_quat_wxyz"][0, 0] = 0.0
        np.savez_compressed(state_path, **arrays)
        _rebind(candidate)
        with pytest.raises(RepeatabilityError, match="zero-norm quaternion"):
            build_repeatability_report(reference, candidate, profile_path=PROFILE)


def test_boolean_or_negative_visible_frame_count_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        reference, candidate = _pair(Path(temporary))
        outcome_path = candidate / "task_outcome.json"
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        outcome["target_observability"]["per_target_slot"][
            "search_target_slot_000"
        ]["visible_frames"] = True
        _write_json(outcome_path, outcome)
        _rebind(candidate)
        with pytest.raises(RepeatabilityError, match="frame count is malformed"):
            build_repeatability_report(reference, candidate, profile_path=PROFILE)

        outcome["target_observability"]["per_target_slot"][
            "search_target_slot_000"
        ]["visible_frames"] = -1
        _write_json(outcome_path, outcome)
        _rebind(candidate)
        with pytest.raises(RepeatabilityError, match="frame count is malformed"):
            build_repeatability_report(reference, candidate, profile_path=PROFILE)
