from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark import isaac_validate
from rivermark_benchmark.abi import OBSERVATION_ABI_SCHEMA, observation_abi_sha256
from rivermark_benchmark.isaac_pack import (
    PACK_SPEC_SCHEMA,
    PACK_SPEC_SCHEMA_V2,
    pack_isaac_capture,
)
from rivermark_benchmark.isaac_public_manifest import (
    build_public_scene_manifest,
    public_manifest_sha256,
)
from rivermark_benchmark.isaac_validate import VALIDATION_SCHEMA, IsaacValidationReport
from rivermark_benchmark.video import sha256_file


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _abi() -> dict[str, object]:
    field = {
        "name": "sample",
        "dtype": "float32",
        "shape": ["frame", "agent", 4],
        "units": "m",
        "frame_id": "body",
        "agent_id_field": "agent_id",
        "timestamp_field": "sensor_time_ns",
        "missing": {"policy": "not_applicable", "sentinel": None, "mask_field": None},
        "valid_range": {"min": None, "max": None, "inclusive": True},
        "compression": "npz_deflate",
        "time_semantics": "sensor_sample",
    }
    action = dict(field)
    action["name"] = "command"
    action["time_semantics"] = "command_before_step"
    return {
        "schema": OBSERVATION_ABI_SCHEMA,
        "version": "1.1.0",
        "action_timing": {
            "command_write": "before_simulation_step",
            "simulation_step": "after_command_write",
            "state_update": "after_simulation_step",
            "sensor_read": "after_state_update",
            "storage": "after_sensor_read",
        },
        "coordinate_frames": {
            "handedness": "right",
            "world_up_axis": "+z",
            "world_frame_convention": "x_east_y_north_z_up",
            "body_frame_convention": "flu",
            "camera_optical_frame_convention": "opencv_x_right_y_down_z_forward",
            "length_unit": "m",
            "angle_unit": "rad",
            "quaternion_order": "wxyz",
            "transform_notation": "T_parent_child",
        },
        "calibration": {
            "camera": {
                "status": "recorded",
                "source": "pack-fixture",
                "intrinsics": {"model": "pinhole", "width_px": 8, "height_px": 8, "fx_px": 8.0, "fy_px": 8.0, "cx_px": 4.0, "cy_px": 4.0},
                "extrinsics": {"formula": "T_world_camera = T_world_body * T_body_camera", "quaternion_order": "wxyz"},
                "distortion_model": "none",
                "distortion_coefficients": [],
            },
            "lidar": {"status": "unavailable", "source": "pack-fixture"},
            "imu": {"status": "unavailable", "source": "pack-fixture"},
        },
        "streams": [
            {"stream_id": "state", "modality": "proprioception", "partition": "policy_visible", "encoding": "npz", "fidelity": "simulator_consistent", "fidelity_limitations": ["no_hardware_sensor_noise"], "fields": [field]},
            {"stream_id": "actions", "modality": "high_level_action_history", "partition": "policy_visible", "encoding": "npz", "fidelity": "simulator_consistent", "fidelity_limitations": ["no_actuator_identification"], "fields": [action]},
        ],
    }


def _fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    capture = root / "capture"
    capture.mkdir()
    revision = "0123456789abcdef"
    evaluator = root / "evaluator" / "manifest.json"
    _json(evaluator, {"private": "stored outside candidate"})
    evaluation_sha = sha256_file(evaluator)
    (capture / "capture_receipt.json").write_text(
        json.dumps(
            {
                "schema": "org.rivermark.isaac-swarm-capture.v1",
                "status": "captured",
                "ok": True,
                "task_kind": "search3d",
                "source_revision": revision,
                "source_worktree_dirty": False,
                "evaluator_manifest_sha256": evaluation_sha,
                "capture_backend": {
                    "kind": "isaaclab",
                    "build": "isaac-sim-test",
                    "sensor_physics_smoke_receipt_sha256": _digest("smoke"),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation = root / "validation.json"
    capture_sha = sha256_file(capture / "capture_receipt.json")
    _json(
        validation,
        {
            "schema": VALIDATION_SCHEMA,
            "status": "passed",
            "formal_benchmark_admission": False,
            "capture_receipt_sha256": capture_sha,
            "validator_id": "independent-test-validator",
            "validator_source_sha256": sha256_file(Path(isaac_validate.__file__).resolve()),
            "checks": {
                "online_capture": True,
                "queue_overflow": False,
                "silent_frame_drop": False,
                "timestamp_audit_passed": True,
                "pose_closure_audit_passed": True,
                "action_causality_audit_passed": True,
                "sensor_decode_audit_passed": True,
                "policy_leakage_audit_passed": True,
                "evaluator_manifest_sha256": evaluation_sha,
                "pose_closure_max_error_m": 0.001,
                "pose_closure_threshold_m": 0.01,
            },
            "issues": [],
        },
    )
    _json(capture / "scene.json", {"agent_count": 8, "fresh_stage": True, "legacy_route_or_target_imported": False})
    _json(capture / "task.json", {"schema": "org.rivermark.benchmark.search3d_task.v1", "task": "search"})
    abi_payload = _abi()
    _json(capture / "metadata" / "observation_abi.json", abi_payload)
    timestamps = np.array([1, 2], dtype=np.int64)
    for relative in (
        "streams/state.npz",
        "sensors/rgb.npz",
        "sensors/depth.npz",
        "sensors/semantic.npz",
        "sensors/lidar.npz",
        "sensors/imu.npz",
    ):
        path = capture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, timestamps_ns=timestamps, value=np.arange(2, dtype=np.float32))
    for relative in (
        "streams/high_level_actions.jsonl",
        "streams/public_task_state.jsonl",
        "streams/messages.jsonl",
        "streams/candidates.jsonl",
    ):
        path = capture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"sim_time_ns":1,"value":0}\n{"sim_time_ns":2,"value":1}\n', encoding="utf-8")
    _json(capture / "sensors" / "semantic_labels.json", {"classes": [{"class_id": 1, "name": "search object"}]})

    spec = root / "pack_spec.json"
    stream_specs = [
        ("actions", "high_level_action_history", "streams/high_level_actions.jsonl"),
        ("state", "proprioception", "streams/state.npz"),
        ("task", "public_task_state", "streams/public_task_state.jsonl"),
        ("candidates", "public_task_state", "streams/candidates.jsonl"),
        ("messages", "public_team_messages", "streams/messages.jsonl"),
        ("rgb", "rgb", "sensors/rgb.npz"),
        ("depth", "distance_to_image_plane", "sensors/depth.npz"),
        ("lidar", "lidar", "sensors/lidar.npz"),
        ("imu", "imu", "sensors/imu.npz"),
    ]
    streams = []
    for stream_id, modality, source in stream_specs:
        suffix = Path(source).suffix
        streams.append(
            {
                "stream_id": stream_id,
                "partition": "policy_visible",
                "modality": modality,
                "media_type": "application/x-ndjson" if suffix == ".jsonl" else "application/octet-stream",
                "timestamp_field": "timestamps_ns" if suffix == ".npz" else "sim_time_ns",
                "source": source,
                "path": source,
                **(
                    {"fields": ["timestamps_ns", "value"]}
                    if suffix == ".npz"
                    else ({"sample_count": 1} if suffix == ".json" else {})
                ),
            }
        )
    _json(
        spec,
        {
            "schema": PACK_SPEC_SCHEMA,
            "dataset_version": "0.2.0",
            "episode_id": "isaac-search3d-test-001",
            "split": "train",
            "layout": {
                "layout_id": "layout-test",
                "layout_hash": _digest("layout"),
                "layout_lineage_hash": _digest("layout-lineage"),
                "source": "scene.json",
            },
            "task": {
                "task_id": "multi_uav_search3d",
                "task_variant_id": "search3d-test-v1",
                "information_profile": "multisensor_rgbd_lidar_imu_state",
                "observation_scope": "decentralized_explicit_comm",
                "agent_count": 8,
                "source": "task.json",
            },
            "timebase": {"unit": "ns", "physics_dt_ns": 1, "proprioception_period_ns": 1, "camera_period_ns": 1},
            "coordinate_frames": {
                "handedness": "right",
                "world_up_axis": "+z",
                "world_frame_convention": "x_east_y_north_z_up",
                "body_frame_convention": "flu",
                "camera_optical_frame_convention": "opencv_x_right_y_down_z_forward",
                "length_unit": "m",
                "angle_unit": "rad",
                "quaternion_order": "wxyz",
                "transform_notation": "T_parent_child",
            },
            "observation_abi": {"source": "metadata/observation_abi.json", "path": "metadata/observation_abi.json"},
            "streams": streams,
            "provenance": {
                "route_conditioning": "public_only",
                "observation_generation": "online_runtime",
                "collector_type": "scripted",
                "policy_id": "isaac-search3d-test",
                "code_commit": revision,
                "simulator_build": "isaac-sim-test",
                "scene_asset_license_status": "redistribution_cleared",
            },
            "quality": {"task_success": False, "invalid_reasons": []},
            "lineage_values": {
                "appearance_domain": "appearance-test",
                "dynamics_domain": "dynamics-test",
                "instruction_family": "none",
                "instruction_annotator": "none",
                "asset_lineage": "asset-test",
                "behavior_policy_checkpoint_family": "scripted-test",
            },
            "capture_backend": {
                "build": "isaac-sim-test",
                "sensor_physics_smoke_receipt_sha256": _digest("smoke"),
            },
        },
    )
    return capture, validation, evaluator, spec


def _bind_fixture(capture: Path, validation: Path, binding: dict[str, object]) -> str:
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["collection_binding"] = binding
    _json(receipt_path, receipt)
    capture_sha = sha256_file(receipt_path)
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    validation_payload["capture_receipt_sha256"] = capture_sha
    validation_payload["checks"]["collection_binding_present"] = True
    validation_payload["checks"]["collection_binding_verified"] = True
    _json(validation, validation_payload)
    return capture_sha


def _v2_scene_with_private_diagnostics() -> dict[str, object]:
    return {
        "schema": "org.rivermark.public-isaac-scene.v1",
        "environment_id": "RIVERMARK_CITY_LITE_v1",
        "agent_count": 8,
        "fresh_stage": True,
        "static_scene_authority_verified": True,
        "legacy_route_or_target_imported": False,
        "unresolved_reference_count": 0,
        "private_evaluator_manifest_sha256": "e" * 64,
        "source_scene": r"C:\private\rivermark.usd",
        "scene_contract": {
            "schema": "citylite-contract-v1",
            "gate_status": "pass_city_lite_static_construction",
            "payload_sha256": _digest("contract-payload"),
            "sha256": _digest("contract-file"),
        },
        "rivermark_layer_inventory": {
            "schema": "resolved-layer-inventory-v1",
            "inventory_sha256": _digest("inventory"),
            "local_authority_inventory_sha256": _digest("local-inventory"),
            "rivermarksrc51_external_inventory_sha256": _digest("external-inventory"),
            "local_authority_layer_count": 2,
            "rivermarksrc51_external_layer_count": 3,
            "input_resolved_layer_count": 5,
            "composition_scope": {
                "mode": "selective_references_only",
                "selective_references": [
                    {
                        "source_prim": "/World/City/Rivermark",
                        "destination_prim": "/World/StaticScene/City/Rivermark",
                    },
                    {
                        "source_prim": "/World/CityTaskObstacles",
                        "destination_prim": "/World/StaticScene/CityTaskObstacles",
                    },
                ],
                "whole_final_stage_inventory": False,
            },
        },
    }


class IsaacPackTests(unittest.TestCase):
    def test_backend_smoke_commitment_must_come_from_capture_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            receipt_path = capture / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.pop("capture_backend")
            _json(receipt_path, receipt)
            validation_payload = json.loads(validation.read_text(encoding="utf-8"))
            validation_payload["capture_receipt_sha256"] = sha256_file(receipt_path)
            _json(validation, validation_payload)
            report = IsaacValidationReport(
                capture,
                sha256_file(receipt_path),
                validation_payload["checks"],
                (),
            )
            with patch(
                "rivermark_benchmark.isaac_pack.validate_isaac_capture",
                return_value=report,
            ):
                result = pack_isaac_capture(
                    capture,
                    validation,
                    evaluator,
                    spec,
                    root / "candidate",
                )

            self.assertFalse(result.valid)
            self.assertIn(
                "capture_backend_commitment",
                {issue.code for issue in result.issues},
            )

    def test_pack_spec_cannot_replace_capture_bound_smoke_commitment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["capture_backend"]["sensor_physics_smoke_receipt_sha256"] = _digest(
                "invented"
            )
            _json(spec, payload)
            validation_payload = json.loads(validation.read_text(encoding="utf-8"))
            report = IsaacValidationReport(
                capture,
                sha256_file(capture / "capture_receipt.json"),
                validation_payload["checks"],
                (),
            )
            with patch(
                "rivermark_benchmark.isaac_pack.validate_isaac_capture",
                return_value=report,
            ):
                result = pack_isaac_capture(
                    capture,
                    validation,
                    evaluator,
                    spec,
                    root / "candidate",
                )

            self.assertFalse(result.valid)
            self.assertIn(
                "capture_backend_mismatch",
                {issue.code for issue in result.issues},
            )

    def test_v2_external_abi_requires_capture_receipt_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            external_abi = root / "external_observation_abi.json"
            abi_payload = json.loads(
                (capture / "metadata/observation_abi.json").read_text(encoding="utf-8")
            )
            _json(external_abi, abi_payload)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["schema"] = PACK_SPEC_SCHEMA_V2
            payload["observation_abi"] = {
                "source": external_abi.name,
                "source_scope": "pack_spec",
                "path": "metadata/observation_abi.json",
                "sha256": "0" * 64,
                "capture_receipt_sha256": "0" * 64,
            }
            _json(spec, payload)
            destination = root / "candidate"
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(
                capture,
                sha256_file(capture / "capture_receipt.json"),
                checks,
                (),
            )
            with patch(
                "rivermark_benchmark.isaac_pack.validate_isaac_capture",
                return_value=report,
            ):
                result = pack_isaac_capture(
                    capture,
                    validation,
                    evaluator,
                    spec,
                    destination,
                )

            self.assertFalse(result.valid)
            self.assertIn(
                "observation_abi_capture_binding",
                {issue.code for issue in result.issues},
            )
            self.assertFalse(destination.exists())

    def test_v2_packer_writes_the_same_public_scene_projection_as_the_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            raw_scene = _v2_scene_with_private_diagnostics()
            _json(capture / "scene.json", raw_scene)
            np.savez_compressed(
                capture / "sensors/onboard_rgbd.npz",
                __rivermark_chunked_frame_archive_v1__=np.asarray([1], dtype=np.uint8),
                __rivermark_frame_count__=np.asarray([2], dtype=np.int64),
                timestamps_ns=np.asarray([1, 2], dtype=np.int64),
            )
            external_abi = root / "external_observation_abi.json"
            abi = json.loads((capture / "metadata/observation_abi.json").read_text(encoding="utf-8"))
            _json(external_abi, abi)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["schema"] = PACK_SPEC_SCHEMA_V2
            payload["layout"] = {
                "layout_id": "citylite-v1",
                "layout_hash": public_manifest_sha256(
                    build_public_scene_manifest(raw_scene)
                ),
                "layout_lineage_hash": _digest("contract-payload"),
                "source": "scene.json",
            }
            payload["task"] = {
                "task_id": "multi_uav_search3d",
                "task_variant_id": "search3d-test-v1",
                "information_profile": "multisensor_rgbd_lidar_imu_state",
                "observation_scope": "decentralized_explicit_comm",
                "agent_count": 8,
                "source": "task.json",
            }
            payload["observation_abi"] = {
                "source": external_abi.name,
                "source_scope": "pack_spec",
                "path": "metadata/observation_abi.json",
                "sha256": observation_abi_sha256(abi),
                "capture_receipt_sha256": sha256_file(capture / "capture_receipt.json"),
            }
            payload["streams"] = [
                {
                    "stream_id": "actions",
                    "partition": "policy_visible",
                    "modality": "high_level_action_history",
                    "media_type": "application/x-ndjson",
                    "timestamp_field": "sim_time_ns",
                    "source": "streams/high_level_actions.jsonl",
                    "path": "streams/actions.jsonl",
                },
                {
                    "stream_id": "state",
                    "partition": "policy_visible",
                    "modality": "proprioception",
                    "media_type": "application/x-npz",
                    "timestamp_field": "timestamps_ns",
                    "source": "streams/state.npz",
                    "path": "streams/state.npz",
                    "fields": ["timestamps_ns", "value"],
                },
                {
                    "stream_id": "task",
                    "partition": "policy_visible",
                    "modality": "public_task_state",
                    "media_type": "application/x-ndjson",
                    "timestamp_field": "sim_time_ns",
                    "source": "streams/public_task_state.jsonl",
                    "path": "streams/task.jsonl",
                },
                {
                    "stream_id": "messages",
                    "partition": "policy_visible",
                    "modality": "public_team_messages",
                    "media_type": "application/x-ndjson",
                    "timestamp_field": "sim_time_ns",
                    "source": "streams/messages.jsonl",
                    "path": "streams/messages.jsonl",
                },
                {
                    "stream_id": "rgb",
                    "partition": "policy_visible",
                    "modality": "rgb",
                    "media_type": "application/x-npz",
                    "timestamp_field": "timestamps_ns",
                    "source": "sensors/onboard_rgbd.npz",
                    "path": "streams/onboard_rgbd.npz",
                    "sample_count": 2,
                },
                {
                    "stream_id": "depth",
                    "partition": "policy_visible",
                    "modality": "distance_to_image_plane",
                    "media_type": "application/x-npz",
                    "timestamp_field": "timestamps_ns",
                    "source": "sensors/onboard_rgbd.npz",
                    "path": "streams/onboard_rgbd.npz",
                    "sample_count": 2,
                },
                {
                    "stream_id": "lidar",
                    "partition": "policy_visible",
                    "modality": "lidar",
                    "media_type": "application/x-npz",
                    "timestamp_field": "timestamps_ns",
                    "source": "sensors/lidar.npz",
                    "path": "sensors/lidar.npz",
                    "fields": ["timestamps_ns", "value"],
                },
                {
                    "stream_id": "imu",
                    "partition": "policy_visible",
                    "modality": "imu",
                    "media_type": "application/x-npz",
                    "timestamp_field": "timestamps_ns",
                    "source": "sensors/imu.npz",
                    "path": "sensors/imu.npz",
                    "fields": ["timestamps_ns", "value"],
                },
            ]
            _json(spec, payload)
            candidate_streams = {
                stream["stream_id"]: {
                    "path": stream["source"],
                    "modality": stream["modality"],
                    "timestamp_field": stream["timestamp_field"],
                    **(
                        {"fields": stream["fields"]}
                        if "fields" in stream
                        else {
                            "arrays": {
                                "timestamps_ns": {"shape": [2]},
                            }
                        }
                    ),
                }
                for stream in payload["streams"]
            }
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(
                capture,
                sha256_file(capture / "capture_receipt.json"),
                checks,
                (),
            )
            destination = root / "candidate"
            with (
                patch("rivermark_benchmark.isaac_pack.validate_isaac_capture", return_value=report),
                patch("rivermark_benchmark.isaac_pack.inspect_candidate_pack_streams", return_value=candidate_streams),
                patch("rivermark_benchmark.isaac_pack.validate_candidate_abi_sources", return_value=()),
            ):
                result = pack_isaac_capture(
                    capture,
                    validation,
                    evaluator,
                    spec,
                    destination,
                )

            self.assertTrue(result.valid, result.issues)
            packed_scene = json.loads((destination / "scenes/scene.json").read_text(encoding="utf-8"))
            self.assertEqual(packed_scene, build_public_scene_manifest(raw_scene))
            self.assertEqual(
                sha256_file(destination / "scenes/scene.json"),
                payload["layout"]["layout_hash"],
            )
            self.assertNotIn("private", json.dumps(packed_scene).lower())
            self.assertNotIn("C:\\", json.dumps(packed_scene))

    def test_collection_binding_is_recomputed_and_inherited(self) -> None:
        binding: dict[str, object] = {
            "protocol_id": "citylite-coverage-v1",
            "protocol_sha256": "c" * 64,
            "cell_id": "train-route-0",
            "split": "train",
            "episode_index": 3,
            "episode_seed": 42,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            capture_sha = _bind_fixture(capture, validation, binding)
            protocol_path = root / "collection-protocol.json"
            _json(protocol_path, {"fixture": True})
            destination = root / "candidate"
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(capture, capture_sha, checks, ())
            with (
                patch("rivermark_benchmark.isaac_pack.validate_isaac_capture", return_value=report),
                patch("rivermark_benchmark.isaac_pack.load_collection_protocol", return_value={}),
                patch("rivermark_benchmark.isaac_pack.resolve_collection_binding", return_value=binding),
            ):
                result = pack_isaac_capture(
                    capture,
                    validation,
                    evaluator,
                    spec,
                    destination,
                    protocol_path,
                )
            self.assertTrue(result.valid, result.issues)
            manifest = json.loads((destination / "episode_manifest.json").read_text(encoding="utf-8"))
            receipt = json.loads((destination / "formal_capture_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["collection_binding"], binding)
            self.assertEqual(receipt["collection_binding"], binding)

    def test_collection_binding_tamper_is_rejected(self) -> None:
        expected: dict[str, object] = {
            "protocol_id": "citylite-coverage-v1",
            "protocol_sha256": "c" * 64,
            "cell_id": "train-route-0",
            "split": "train",
            "episode_index": 3,
            "episode_seed": 42,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            capture_sha = _bind_fixture(capture, validation, {**expected, "episode_seed": 43})
            protocol_path = root / "collection-protocol.json"
            _json(protocol_path, {"fixture": True})
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(capture, capture_sha, checks, ())
            with (
                patch("rivermark_benchmark.isaac_pack.validate_isaac_capture", return_value=report),
                patch("rivermark_benchmark.isaac_pack.load_collection_protocol", return_value={}),
                patch("rivermark_benchmark.isaac_pack.resolve_collection_binding", return_value=expected),
            ):
                result = pack_isaac_capture(
                    capture,
                    validation,
                    evaluator,
                    spec,
                    root / "candidate",
                    protocol_path,
                )
            self.assertFalse(result.valid)
            self.assertIn("collection_binding", {issue.code for issue in result.issues})

    def test_collection_binding_requires_independent_attestation(self) -> None:
        binding: dict[str, object] = {
            "protocol_id": "citylite-coverage-v1",
            "protocol_sha256": "c" * 64,
            "cell_id": "train-route-0",
            "split": "train",
            "episode_index": 3,
            "episode_seed": 42,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            capture_sha = _bind_fixture(capture, validation, binding)
            validation_payload = json.loads(validation.read_text(encoding="utf-8"))
            validation_payload["checks"]["collection_binding_verified"] = False
            _json(validation, validation_payload)
            checks = validation_payload["checks"]
            report = IsaacValidationReport(capture, capture_sha, checks, ())
            protocol_path = root / "collection-protocol.json"
            _json(protocol_path, {"fixture": True})
            with patch(
                "rivermark_benchmark.isaac_pack.validate_isaac_capture",
                return_value=report,
            ):
                result = pack_isaac_capture(
                    capture,
                    validation,
                    evaluator,
                    spec,
                    root / "candidate",
                    protocol_path,
                )
            self.assertFalse(result.valid)
            self.assertIn("collection_validation", {issue.code for issue in result.issues})

    def test_projects_closed_world_candidate_and_binds_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture, validation, evaluator, spec = _fixture(Path(temporary))
            destination = Path(temporary) / "candidate"
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(capture, sha256_file(capture / "capture_receipt.json"), checks, ())
            with patch("rivermark_benchmark.isaac_pack.validate_isaac_capture", return_value=report):
                result = pack_isaac_capture(capture, validation, evaluator, spec, destination)
            self.assertTrue(result.valid, result.issues)
            self.assertTrue((destination / "episode_manifest.json").is_file())
            self.assertTrue((destination / "lineage.json").is_file())
            self.assertTrue((destination / "formal_capture_receipt.json").is_file())
            self.assertFalse((destination / "evaluator_private").exists())
            self.assertFalse((destination / "sensors" / "overview_rgb.npz").exists())
            receipt = json.loads((destination / "formal_capture_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["integrity"]["independent_validator_sha256"], sha256_file(validation))
            self.assertEqual(receipt["partitions"]["evaluator_private_server_only"], True)

    def test_legacy_abi_is_rejected_by_formal_packer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            abi_path = capture / "metadata" / "observation_abi.json"
            abi = json.loads(abi_path.read_text(encoding="utf-8"))
            abi["version"] = "1.0.0"
            for stream in abi["streams"]:
                stream.pop("fidelity", None)
                stream.pop("fidelity_limitations", None)
            _json(abi_path, abi)
            destination = root / "candidate"
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(capture, sha256_file(capture / "capture_receipt.json"), checks, ())
            with patch("rivermark_benchmark.isaac_pack.validate_isaac_capture", return_value=report):
                result = pack_isaac_capture(capture, validation, evaluator, spec, destination)
            self.assertFalse(result.valid)
            self.assertIn("observation_abi", {issue.code for issue in result.issues})
            self.assertFalse(destination.exists())

    def test_overview_source_is_rejected_without_leaving_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            (capture / "sensors" / "overview_rgb.npz").write_bytes(b"overview")
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["streams"][0]["source"] = "sensors/overview_rgb.npz"
            spec.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            destination = root / "candidate"
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(capture, sha256_file(capture / "capture_receipt.json"), checks, ())
            with patch("rivermark_benchmark.isaac_pack.validate_isaac_capture", return_value=report):
                result = pack_isaac_capture(capture, validation, evaluator, spec, destination)
            self.assertFalse(result.valid)
            self.assertIn("private_or_overview_source", {issue.code for issue in result.issues})
            self.assertFalse(destination.exists())

    def test_learning_label_source_is_rejected_at_path_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["streams"][0]["source"] = "learning_labels/semantic_labels.json"
            payload["streams"][0].pop("fields", None)
            payload["streams"][0]["sample_count"] = 1
            _json(spec, payload)
            destination = root / "candidate"
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(
                capture,
                sha256_file(capture / "capture_receipt.json"),
                checks,
                (),
            )
            with patch(
                "rivermark_benchmark.isaac_pack.validate_isaac_capture",
                return_value=report,
            ):
                result = pack_isaac_capture(capture, validation, evaluator, spec, destination)
            self.assertFalse(result.valid)
            self.assertIn("private_or_overview_source", {issue.code for issue in result.issues})
            self.assertFalse(destination.exists())

    def test_validation_evaluator_binding_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture, validation, evaluator, spec = _fixture(Path(temporary))
            payload = json.loads(validation.read_text(encoding="utf-8"))
            payload["checks"]["evaluator_manifest_sha256"] = _digest("wrong")
            validation.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            destination = Path(temporary) / "candidate"
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(capture, sha256_file(capture / "capture_receipt.json"), checks, ())
            with patch("rivermark_benchmark.isaac_pack.validate_isaac_capture", return_value=report):
                result = pack_isaac_capture(capture, validation, evaluator, spec, destination)
            self.assertFalse(result.valid)
            self.assertIn("evaluator_binding", {issue.code for issue in result.issues})

    def test_selected_private_truth_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            state_path = capture / "streams" / "state.npz"
            np.savez_compressed(
                state_path,
                timestamps_ns=np.array([1, 2], dtype=np.int64),
                value=np.arange(2, dtype=np.float32),
                target_ids=np.array([7, 8], dtype=np.int64),
            )
            payload = json.loads(spec.read_text(encoding="utf-8"))
            state_stream = next(stream for stream in payload["streams"] if stream["stream_id"] == "state")
            state_stream["fields"].append("target_ids")
            spec.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            destination = root / "candidate"
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(capture, sha256_file(capture / "capture_receipt.json"), checks, ())
            with patch("rivermark_benchmark.isaac_pack.validate_isaac_capture", return_value=report):
                result = pack_isaac_capture(capture, validation, evaluator, spec, destination)
            self.assertFalse(result.valid)
            self.assertIn("policy_truth_leak", {issue.code for issue in result.issues})
            self.assertFalse(destination.exists())

    def test_semantic_learning_labels_cannot_be_projected_as_policy_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, validation, evaluator, spec = _fixture(root)
            payload = json.loads(spec.read_text(encoding="utf-8"))
            payload["streams"].append(
                {
                    "stream_id": "semantic",
                    "partition": "policy_visible",
                    "modality": "semantic_segmentation",
                    "media_type": "application/x-npz",
                    "timestamp_field": "timestamps_ns",
                    "source": "sensors/semantic.npz",
                    "path": "sensors/semantic.npz",
                    "fields": ["timestamps_ns", "value"],
                }
            )
            _json(spec, payload)
            destination = root / "candidate"
            checks = json.loads(validation.read_text(encoding="utf-8"))["checks"]
            report = IsaacValidationReport(
                capture,
                sha256_file(capture / "capture_receipt.json"),
                checks,
                (),
            )
            with patch(
                "rivermark_benchmark.isaac_pack.validate_isaac_capture",
                return_value=report,
            ):
                result = pack_isaac_capture(capture, validation, evaluator, spec, destination)
            self.assertFalse(result.valid)
            self.assertIn("candidate_verification", {issue.code for issue in result.issues})
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
