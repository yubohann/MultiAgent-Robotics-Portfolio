import hashlib
import json
import math
from pathlib import Path

import numpy as np

from rivermark_benchmark.citylite_scene import (
    SCENE_CONTRACT_PAYLOAD_SHA256,
    SCENE_CONTRACT_SHA256,
)
from rivermark_benchmark.frame_archive import write_chunked_frame_archive
from rivermark_benchmark.private_evaluator_manifest import (
    NATIVE_GEOMETRY_SCAN_EVIDENCE_KIND,
    NATIVE_GEOMETRY_SCAN_GENERATOR,
    NATIVE_GEOMETRY_SCAN_SCHEMA,
    NATIVE_GEOMETRY_SCAN_TOOL_PATH,
    native_geometry_scan_sha256,
)
from rivermark_benchmark.target_visibility_diagnosis import (
    TargetVisibilityDiagnosisError,
    _camera_frustum_membership,
    _camera_witness,
    _semantic_ids_by_frame,
    diagnose_failed_target_visibility,
)


def _scan(path: Path) -> None:
    payload = {
        "schema": NATIVE_GEOMETRY_SCAN_SCHEMA,
        "status": "passed",
        "formal": False,
        "generator": NATIVE_GEOMETRY_SCAN_GENERATOR,
        "geometry_evidence_kind": NATIVE_GEOMETRY_SCAN_EVIDENCE_KIND,
        "tool_path": NATIVE_GEOMETRY_SCAN_TOOL_PATH,
        "tool_sha256": "b" * 64,
        "source_revision": "c" * 40,
        "source_tree_sha256": "d" * 64,
        "source_worktree_dirty": False,
        "runtime_lock": {
            "sha256": "e" * 64,
            "profile_id": "isaac-windows-5.1",
            "audit_status": "passed",
        },
        "scene_id": "RIVERMARK_CITY_LITE_v1",
        "scene_contract_sha256": SCENE_CONTRACT_SHA256,
        "scene_content_sha256": SCENE_CONTRACT_PAYLOAD_SHA256,
        "domains": [
            {
                "aabb": {
                    "min": [100.0, 100.0, 0.0],
                    "max": [101.0, 101.0, 1.0],
                    "path": "/World/Test/irrelevant_structure",
                    "source_kind": "test_structure",
                }
            }
        ],
    }
    payload["scan_sha256"] = native_geometry_scan_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _private_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "targets": [
                    {"target_id": "must-not-leak", "position_w_m": [5.0, 0.0, 10.0], "radius_m": 0.3},
                    {"target_id": "must-not-leak-either", "position_w_m": [0.0, 5.0, 10.0], "radius_m": 0.3},
                ]
            }
        ),
        encoding="utf-8",
    )


def _capture(root: Path) -> None:
    spool = root / ".sensor_spool_v1"
    spool.mkdir(parents=True)
    positions = np.asarray([[[0.0, 0.0, 10.0]]], dtype=np.float64)
    quaternions = np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float64)
    np.save(spool / "camera_observed_pos_w_m.npy", positions)
    np.save(spool / "camera_observed_quat_wxyz.npy", quaternions)
    np.save(spool / "semantic.npy", np.zeros((1, 1, 1, 1, 1), dtype=np.int32))
    (root / "capture_progress.json").write_text(
        json.dumps(
            {
                "target_visibility": {
                    "per_target_slot": {
                        "search_target_slot_000": {"visible_frames": 1, "max_pixels": 64},
                        "search_target_slot_001": {"visible_frames": 0, "max_pixels": 0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def _reference_capture(root: Path) -> None:
    sensor_dir = root / "sensors"
    label_dir = root / "learning_labels"
    sensor_dir.mkdir(parents=True)
    label_dir.mkdir()
    positions = np.zeros((1, 8, 3), dtype=np.float64)
    positions[..., 2] = 10.0
    quaternions = np.zeros((1, 8, 4), dtype=np.float64)
    quaternions[..., 0] = 1.0
    np.savez_compressed(
        sensor_dir / "camera_poses.npz",
        camera_observed_pos_w_m=positions,
        camera_observed_quat_wxyz=quaternions,
    )
    rgb = np.zeros((1, 8, 120, 160, 3), dtype=np.uint8)
    rgb[0, 0, 60, 80] = (242, 13, 191)
    depth = np.full((1, 8, 120, 160, 1), 100.0, dtype=np.float32)
    depth[0, 0, 60, 80, 0] = 5.0
    write_chunked_frame_archive(
        sensor_dir / "onboard_rgbd.npz",
        timestamps_ns=np.asarray([1], dtype=np.int64),
        inline_fields={},
        frame_fields={"rgb": rgb, "distance_to_image_plane_m": depth},
    )
    semantic = np.zeros((8, 120, 160, 1), dtype=np.int32)
    semantic[0, 60, 80, 0] = 7
    write_chunked_frame_archive(
        label_dir / "semantic_segmentation.npz",
        timestamps_ns=np.asarray([1], dtype=np.int64),
        inline_fields={},
        frame_fields={"semantic_segmentation": semantic[None, ...]},
    )
    mappings = [{"idToLabels": {}} for _ in range(8)]
    mappings[0] = {"idToLabels": {"7": {"class": "search_target_slot_000"}}}
    frame_row = {
        "schema": "org.rivermark.isaac-semantic-frame-metadata.v1",
        "frame_index": 0,
        "timestamp_ns": 1,
        "onboard_replicator_info": {"per_camera": mappings},
        "overview_replicator_info": {"per_camera": [{"id_to_labels": {"9": {"class": "prop_structure"}}}]},
    }
    (label_dir / "semantic_frame_metadata.jsonl").write_text(
        json.dumps(frame_row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (label_dir / "semantic_metadata.json").write_text(
        json.dumps(
            {
                "schema": "org.rivermark.isaac-semantic-metadata.v2",
                "partition": "learning_labels",
                "policy_visible": False,
                "frame_metadata": {
                    "schema": "org.rivermark.isaac-semantic-frame-metadata.v1",
                    "path": "learning_labels/semantic_frame_metadata.jsonl",
                    "frame_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "capture_receipt.json").write_text("{}", encoding="utf-8")


def _failed_probe_capture(root: Path) -> None:
    _capture(root)
    spool = root / ".sensor_spool_v1"
    rgb = np.zeros((1, 1, 120, 160, 3), dtype=np.uint8)
    rgb[0, 0, 60, 80] = (242, 13, 191)
    depth = np.full((1, 1, 120, 160, 1), 100.0, dtype=np.float32)
    depth[0, 0, 60, 80, 0] = 5.0
    semantic = np.zeros((1, 1, 120, 160, 1), dtype=np.int32)
    np.save(spool / "onboard_rgb.npy", rgb)
    np.save(spool / "depth_m.npy", depth)
    np.save(spool / "semantic.npy", semantic)


def test_recorded_pose_diagnosis_uses_orientation_and_never_leaks_private_truth(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    manifest = tmp_path / "private.json"
    scan = tmp_path / "scan.json"
    _capture(capture)
    _private_manifest(manifest)
    _scan(scan)
    report = diagnose_failed_target_visibility(capture, private_manifest=manifest, geometry_scan=scan)
    first = report["per_target_slot"]["search_target_slot_000"]
    second = report["per_target_slot"]["search_target_slot_001"]
    assert first["recorded_frustum_eligible_samples"] == 1
    assert first["outcome"] == "native_semantic_visible"
    assert second["recorded_frustum_eligible_samples"] == 0
    assert second["outcome"] == "no_recorded_render_fov_witness"
    encoded = json.dumps(report, sort_keys=True)
    assert "must-not-leak" not in encoded
    assert "position_w_m" not in encoded
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == report["private_inputs"]["manifest_sha256"]
    assert report["camera_contract"]["observed_pose_source"] == "unknown_unverified_camera_observed_stream"


def test_diagnosis_marks_usd_render_pose_only_when_receipt_declares_it(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    manifest = tmp_path / "private.json"
    scan = tmp_path / "scan.json"
    _capture(capture)
    _private_manifest(manifest)
    _scan(scan)
    spool_hashes = {}
    for relative in (
        ".sensor_spool_v1/camera_observed_pos_w_m.npy",
        ".sensor_spool_v1/camera_observed_quat_wxyz.npy",
    ):
        path = capture / relative
        spool_hashes[relative] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (capture / "capture_receipt.json").write_text(
        json.dumps(
            {
                "calibration": {
                    "onboard_camera": {
                        "fabric_pose_closure": {
                            "authority": "diagnostic_only_camera_fabric_cache",
                            "acceptance_authority": "render_facing_usd_hierarchy",
                        },
                        "usd_pose_closure": {"max_position_error_m": 0.0},
                    }
                },
                "artifact_hashes": spool_hashes,
            }
        ),
        encoding="utf-8",
    )

    report = diagnose_failed_target_visibility(capture, private_manifest=manifest, geometry_scan=scan)

    assert report["camera_contract"]["observed_pose_source"] == (
        "verified_render_facing_usd_hierarchy_pose_in_isaaclab_world_convention"
    )
    assert report["camera_contract"]["observed_pose_evidence"] == (
        "capture_receipt.usd_pose_closure_and_spool_hash_binding"
    )

    receipt = json.loads((capture / "capture_receipt.json").read_text(encoding="utf-8"))
    receipt["artifact_hashes"][".sensor_spool_v1/camera_observed_pos_w_m.npy"]["sha256"] = "0" * 64
    (capture / "capture_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    rejected = diagnose_failed_target_visibility(capture, private_manifest=manifest, geometry_scan=scan)
    assert rejected["camera_contract"]["observed_pose_source"] == "unknown_unverified_camera_observed_stream"
    assert rejected["camera_contract"]["observed_pose_evidence"] == (
        "matching_capture_receipt_spool_hash_binding_failed"
    )


def test_camera_frustum_is_bound_to_recorded_quaternion_not_ideal_route_yaw() -> None:
    # Identity looks along +X and can see this target. A 90 degree body/camera
    # yaw looks along +Y and cannot. A translation-only envelope misses this.
    assert _camera_witness((0.0, 0.0, 10.0), (1.0, 0.0, 0.0, 0.0), (5.0, 0.0, 10.0))[0]
    quarter_turn = (2.0**-0.5, 0.0, 0.0, 2.0**-0.5)
    assert not _camera_witness((0.0, 0.0, 10.0), quarter_turn, (5.0, 0.0, 10.0))[0]


def test_diagnosis_distinguishes_render_fov_edge_from_conservative_sampling_margin(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture"
    manifest = tmp_path / "private.json"
    scan = tmp_path / "scan.json"
    _capture(capture)
    horizontal_half_fov = math.atan(20.955 / (2.0 * 24.0))
    edge_angle = horizontal_half_fov * 0.96
    manifest.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_id": "must-not-leak",
                        "position_w_m": [5.0, -5.0 * math.tan(edge_angle), 10.0],
                        "radius_m": 0.3,
                    },
                    {"target_id": "must-not-leak-either", "position_w_m": [0.0, 5.0, 10.0], "radius_m": 0.3},
                ]
            }
        ),
        encoding="utf-8",
    )
    _scan(scan)

    full, conservative, _ = _camera_frustum_membership(
        (0.0, 0.0, 10.0), (1.0, 0.0, 0.0, 0.0), (5.0, -5.0 * math.tan(edge_angle), 10.0)
    )
    assert full
    assert not conservative

    report = diagnose_failed_target_visibility(capture, private_manifest=manifest, geometry_scan=scan)
    first = report["per_target_slot"]["search_target_slot_000"]
    assert report["schema"] == "org.rivermark.private-target-visibility-diagnosis.v3"
    assert first["recorded_render_fov_samples"] == 1
    assert first["recorded_conservative_frustum_samples"] == 0
    assert first["recorded_render_fov_edge_samples"] == 1
    assert first["recorded_frustum_eligible_samples"] == 0
    assert first["outcome"] == "native_semantic_visible_only_at_render_fov_edge"


def test_diagnosis_fails_closed_without_observed_camera_pose_stream(tmp_path: Path) -> None:
    manifest = tmp_path / "private.json"
    scan = tmp_path / "scan.json"
    _private_manifest(manifest)
    _scan(scan)
    (tmp_path / "capture").mkdir()
    with np.testing.assert_raises_regex(TargetVisibilityDiagnosisError, "observed onboard camera poses"):
        diagnose_failed_target_visibility(tmp_path / "capture", private_manifest=manifest, geometry_scan=scan)


def test_calibrated_pixel_probe_reports_only_aggregate_residuals(tmp_path: Path) -> None:
    failed = tmp_path / "failed"
    reference = tmp_path / "reference"
    failed_manifest = tmp_path / "failed-private.json"
    reference_manifest = tmp_path / "reference-private.json"
    scan = tmp_path / "scan.json"
    _failed_probe_capture(failed)
    _reference_capture(reference)
    _private_manifest(failed_manifest)
    _private_manifest(reference_manifest)
    _scan(scan)

    report = diagnose_failed_target_visibility(
        failed,
        private_manifest=failed_manifest,
        geometry_scan=scan,
        calibration_reference_capture=reference,
        calibration_reference_private_manifest=reference_manifest,
    )

    probe = report["calibrated_pixel_probe"]
    assert probe["schema"] == "org.rivermark.private-target-pixel-probe.v1"
    assert probe["calibration"]["native_positive_observation_count"] == 1
    target = probe["per_target_slot"]["search_target_slot_000"]
    assert target["joint_color_depth_match_pixels"] >= 1
    encoded = json.dumps(report, sort_keys=True)
    for forbidden in ("must-not-leak", "position_w_m", "[242, 13, 191]"):
        assert forbidden not in encoded


def test_frame_aligned_semantic_mapping_accepts_camera_local_id_reassignment(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    _reference_capture(reference)
    path = reference / "learning_labels/semantic_frame_metadata.jsonl"
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    second = json.loads(json.dumps(first))
    second["frame_index"] = 1
    second["timestamp_ns"] = 2
    labels = second["onboard_replicator_info"]["per_camera"][0]["idToLabels"]
    labels["7"] = {
        "class": "building"
    }
    labels["8"] = {
        "class": "search_target_slot_000"
    }
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in (first, second)),
        encoding="utf-8",
    )
    metadata_path = reference / "learning_labels/semantic_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frame_metadata"]["frame_count"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    mappings = _semantic_ids_by_frame(reference, frame_count=2, target_count=2)

    assert mappings[0]["search_target_slot_000"][0] == (7,)
    assert mappings[1]["search_target_slot_000"][0] == (8,)
