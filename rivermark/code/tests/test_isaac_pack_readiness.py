from __future__ import annotations

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

from rivermark_benchmark.abi import observation_abi_sha256
from rivermark_benchmark.formal_dataset import sha256_file
from rivermark_benchmark.isaac_pack import PACK_SPEC_SCHEMA, PACK_SPEC_SCHEMA_V2
from rivermark_benchmark.isaac_pack_readiness import audit_isaac_pack_readiness
from rivermark_benchmark.isaac_public_manifest import (
    build_public_scene_manifest,
    public_manifest_sha256,
)
from rivermark_benchmark.policy_projection import (
    PolicyProjectionError,
    PolicySourceInspection,
    inspect_candidate_pack_streams,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _scene() -> dict[str, object]:
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
            "payload_sha256": "1" * 64,
            "sha256": "2" * 64,
        },
        "rivermark_layer_inventory": {
            "schema": "resolved-layer-inventory-v1",
            "inventory_sha256": "3" * 64,
            "local_authority_inventory_sha256": "4" * 64,
            "rivermarksrc51_external_inventory_sha256": "5" * 64,
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


def _public_task() -> dict[str, object]:
    return {
        "schema": "org.rivermark.public-search-task.v1",
        "task_kind": "search3d",
        "task_variant_id": "isaac-eight-agent-public-waypoint-search-v1",
        "agent_count": 8,
        "route_conditioning": "public_only",
    }


def _capture(root: Path) -> tuple[Path, PolicySourceInspection]:
    root.mkdir(parents=True)
    (root / "streams").mkdir()
    (root / "sensors").mkdir()
    _json(root / "scene.json", _scene())
    _json(root / "public_task.json", _public_task())
    binding = {
        "protocol_id": "citylite-t1-expert-coverage-v2",
        "protocol_sha256": "b" * 64,
        "cell_id": "train-citylite-direct-v2",
        "split": "train",
        "episode_index": 1,
        "episode_seed": 42,
    }
    state_times = np.asarray([100, 150, 200], dtype=np.int64)
    np.savez_compressed(
        root / "streams/state_action.npz",
        command_time_ns=state_times - 1,
        effective_time_ns=state_times,
        root_pos_w_m=np.zeros((3, 8, 3), dtype=np.float32),
        root_quat_wxyz=np.zeros((3, 8, 4), dtype=np.float32),
        root_lin_vel_w_mps=np.zeros((3, 8, 3), dtype=np.float32),
        root_ang_vel_b_radps=np.zeros((3, 8, 3), dtype=np.float32),
        desired_pos_w_m=np.zeros((3, 8, 3), dtype=np.float32),
        desired_vel_w_mps=np.zeros((3, 8, 3), dtype=np.float32),
        target_thrust_n=np.ones((3, 8, 4), dtype=np.float32),
        applied_thrust_n=np.ones((3, 8, 4), dtype=np.float32),
    )
    timestamps = np.asarray([100, 200], dtype=np.int64)
    np.savez_compressed(
        root / "streams/public_task.npz",
        timestamps_ns=timestamps,
        waypoint_index=np.zeros((2, 8), dtype=np.int64),
        waypoint_progress=np.zeros((2, 8), dtype=np.float32),
        desired_waypoint_w_m=np.zeros((2, 8, 3), dtype=np.float32),
        distance_to_waypoint_m=np.ones((2, 8), dtype=np.float32),
        waypoint_reached=np.zeros((2, 8), dtype=np.bool_),
        action_mode=np.zeros((2, 8), dtype=np.int8),
        coverage_cell_id=np.zeros((2, 8), dtype=np.int64),
        task_time_s=np.zeros((2, 8), dtype=np.float32),
    )
    np.savez_compressed(
        root / "streams/public_messages.npz",
        timestamps_ns=timestamps,
        sender_agent_id=np.zeros((2, 8), dtype=np.int64),
        message_sequence=np.zeros((2, 8), dtype=np.int64),
        message_waypoint_index=np.zeros((2, 8), dtype=np.int64),
        message_position_w_m=np.zeros((2, 8, 3), dtype=np.float32),
        message_velocity_w_mps=np.zeros((2, 8, 3), dtype=np.float32),
        message_flags=np.ones((2, 8), dtype=np.uint8),
    )
    np.savez_compressed(
        root / "sensors/lidar.npz",
        timestamps_ns=timestamps,
        pos_w_m=np.zeros((2, 8, 3), dtype=np.float32),
        quat_wxyz=np.zeros((2, 8, 4), dtype=np.float32),
        ranges_m=np.zeros((2, 8, 1152), dtype=np.float32),
    )
    np.savez_compressed(
        root / "sensors/imu.npz",
        timestamps_ns=timestamps,
        pos_w_m=np.zeros((2, 8, 3), dtype=np.float32),
        quat_wxyz=np.zeros((2, 8, 4), dtype=np.float32),
        angular_velocity_b_radps=np.zeros((2, 8, 3), dtype=np.float32),
        linear_acceleration_b_mps2=np.zeros((2, 8, 3), dtype=np.float32),
    )
    artifact_paths = (
        "streams/state_action.npz",
        "streams/public_task.npz",
        "streams/public_messages.npz",
        "sensors/lidar.npz",
        "sensors/imu.npz",
    )
    receipt = {
        "schema": "org.rivermark.isaac-swarm-capture.v1",
        "source_revision": "a" * 40,
        "evaluator_manifest_sha256": "e" * 64,
        "collection_binding": binding,
        "condition_request": {"fixture": True},
        "capture_backend": {
            "kind": "isaaclab",
            "build": "isaac-sim-test",
            "sensor_physics_smoke_receipt_sha256": "d" * 64,
        },
        "artifact_hashes": {
            relative: {
                "bytes": (root / relative).stat().st_size,
                "sha256": sha256_file(root / relative),
            }
            for relative in artifact_paths
        },
    }
    _json(root / "capture_receipt.json", receipt)
    from rivermark_benchmark import isaac_validate

    validation = {
        "schema": "org.rivermark.isaac-independent-validation.v1",
        "status": "passed",
        "issues": [],
        "capture_receipt_sha256": sha256_file(root / "capture_receipt.json"),
        "validator_source_sha256": sha256_file(Path(isaac_validate.__file__).resolve()),
        "checks": {
            "evaluator_manifest_sha256": "e" * 64,
            "condition_realization_verified": True,
        },
    }
    _json(root / "independent_validation.json", validation)
    inspection = PolicySourceInspection(
        capture_receipt_sha256=sha256_file(root / "capture_receipt.json"),
        independent_validation_sha256=sha256_file(root / "independent_validation.json"),
        source_revision="a" * 40,
        collection_binding=binding,
        frame_count=2,
        state_sample_count=3,
        source_artifacts=(),
        streams={
            "onboard_rgbd": {
                "path": "sensors/onboard_rgbd.npz",
                "fields": ["distance_to_image_plane_m", "rgb"],
                "arrays": {
                    "rgb": {"dtype": "|u1", "shape": [2, 8, 120, 160, 3]},
                    "distance_to_image_plane_m": {
                        "dtype": "<f4",
                        "shape": [2, 8, 120, 160, 1],
                    },
                },
            },
            "lidar": {
                "path": "sensors/lidar.npz",
                "fields": ["ranges_m"],
                "arrays": {"ranges_m": {"dtype": "<f4", "shape": [2, 8, 1152]}},
            },
            "imu": {
                "path": "sensors/imu.npz",
                "fields": ["angular_velocity_b_radps", "linear_acceleration_b_mps2"],
                "arrays": {
                    "angular_velocity_b_radps": {"dtype": "<f4", "shape": [2, 8, 3]},
                    "linear_acceleration_b_mps2": {"dtype": "<f4", "shape": [2, 8, 3]},
                },
            },
            "state": {
                "path": "streams/state_action.npz",
                "fields": [
                    "root_ang_vel_b_radps",
                    "root_lin_vel_w_mps",
                    "root_pos_w_m",
                    "root_quat_wxyz",
                ],
                "arrays": {
                    "root_ang_vel_b_radps": {"dtype": "<f4", "shape": [3, 8, 3]},
                    "root_lin_vel_w_mps": {"dtype": "<f4", "shape": [3, 8, 3]},
                    "root_pos_w_m": {"dtype": "<f4", "shape": [3, 8, 3]},
                    "root_quat_wxyz": {"dtype": "<f4", "shape": [3, 8, 4]},
                },
            },
        },
    )
    return root, inspection


def _pack_spec(source: str, fields: list[str]) -> dict[str, object]:
    return {
        "schema": PACK_SPEC_SCHEMA,
        "dataset_version": "0.2.0",
        "episode_id": "fixture-episode",
        "split": "train",
        "layout": {},
        "task": {},
        "timebase": {},
        "coordinate_frames": {},
        "observation_abi": {
            "source": "metadata/observation_abi.json",
            "path": "metadata/observation_abi.json",
        },
        "streams": [
            {
                "stream_id": "state",
                "partition": "policy_visible",
                "modality": "proprioception",
                "media_type": "application/x-npz",
                "timestamp_field": "timestamps_ns",
                "source": source,
                "path": "streams/state.npz",
                "fields": fields,
            }
        ],
        "provenance": {"code_commit": "c" * 40},
        "quality": {},
        "lineage_values": {
            "appearance_domain": "a",
            "dynamics_domain": "d",
            "instruction_family": "none",
            "instruction_annotator": "none",
            "asset_lineage": "asset",
            "behavior_policy_checkpoint_family": "scripted",
        },
        "capture_backend": {
            "build": "isaac-sim-test",
            "sensor_physics_smoke_receipt_sha256": "d" * 64,
        },
    }


def _formal_abi(streams: dict[str, object]) -> dict[str, object]:
    result = []
    for stream_id, raw_stream in streams.items():
        stream = dict(raw_stream)
        fields = []
        for field_name, raw_descriptor in stream["arrays"].items():
            descriptor = dict(raw_descriptor)
            shape = list(descriptor["shape"])
            shape[0] = (
                "physics_step" if stream_id in {"actions", "state"} else "sensor_frame"
            )
            fields.append(
                {
                    "name": field_name,
                    "dtype": np.dtype(descriptor["dtype"]).name,
                    "shape": shape,
                    "units": "1",
                    "frame_id": "world",
                    "agent_id_field": None,
                    "timestamp_field": stream["timestamp_field"],
                    "missing": {
                        "policy": "not_applicable",
                        "sentinel": None,
                        "mask_field": None,
                    },
                    "valid_range": {"min": None, "max": None, "inclusive": True},
                    "compression": "npz_deflate",
                    "time_semantics": (
                        "command_before_step"
                        if stream["modality"] == "high_level_action_history"
                        else "sensor_sample"
                    ),
                }
            )
        result.append(
            {
                "stream_id": stream_id,
                "modality": stream["modality"],
                "partition": "policy_visible",
                "encoding": "npz",
                "fidelity": "simulator_consistent",
                "fidelity_limitations": ["unit_test_fixture"],
                "fields": fields,
            }
        )
    return {
        "schema": "org.rivermark.benchmark.observation-abi.v1",
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
            sensor: {"status": "unavailable", "source": "unit-test fixture"}
            for sensor in ("camera", "lidar", "imu")
        },
        "streams": result,
    }


def _v2_pack_spec(
    streams: dict[str, object],
    *,
    abi_name: str,
    abi_sha256: str,
    capture_sha256: str,
) -> dict[str, object]:
    packed_streams = []
    for stream_id, raw_stream in streams.items():
        stream = dict(raw_stream)
        source = stream["path"]
        entry = {
            "stream_id": stream_id,
            "partition": "policy_visible",
            "modality": stream["modality"],
            "media_type": "application/x-npz",
            "timestamp_field": stream["timestamp_field"],
            "source": source,
            "path": (
                "streams/onboard_rgbd.npz"
                if source == "sensors/onboard_rgbd.npz"
                else f"streams/{stream_id}.npz"
            ),
        }
        if source == "sensors/onboard_rgbd.npz":
            entry["sample_count"] = stream["arrays"]["timestamps_ns"]["shape"][0]
        else:
            entry["fields"] = stream["fields"]
        packed_streams.append(entry)
    return {
        "schema": PACK_SPEC_SCHEMA_V2,
        "dataset_version": "0.2.0",
        "episode_id": "fixture-episode",
        "split": "train",
        "layout": {
            "layout_id": "citylite-v1",
            "layout_hash": public_manifest_sha256(
                build_public_scene_manifest(_scene())
            ),
            "layout_lineage_hash": "1" * 64,
            "source": "scene.json",
        },
        "task": {
            "task_id": "multi_uav_search3d",
            "task_variant_id": "isaac-eight-agent-public-waypoint-search-v1",
            "information_profile": "multisensor_rgbd_lidar_imu_state",
            "observation_scope": "decentralized_explicit_comm",
            "agent_count": 8,
            "source": "public_task.json",
        },
        "timebase": {},
        "coordinate_frames": {},
        "observation_abi": {
            "source": abi_name,
            "source_scope": "pack_spec",
            "path": "metadata/observation_abi.json",
            "sha256": abi_sha256,
            "capture_receipt_sha256": capture_sha256,
        },
        "streams": packed_streams,
        "provenance": {"code_commit": "a" * 40},
        "quality": {},
        "lineage_values": {
            "appearance_domain": "a",
            "dynamics_domain": "d",
            "instruction_family": "none",
            "instruction_annotator": "none",
            "asset_lineage": "asset",
            "behavior_policy_checkpoint_family": "scripted",
        },
        "capture_backend": {
            "build": "isaac-sim-test",
            "sensor_physics_smoke_receipt_sha256": "d" * 64,
        },
    }


class IsaacPackReadinessTests(unittest.TestCase):
    def test_external_v2_abi_is_hash_and_capture_bound_without_mutating_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, inspection = _capture(root / "capture")
            np.savez(
                capture / "sensors/onboard_rgbd.npz",
                __rivermark_chunked_frame_archive_v1__=np.asarray([1], dtype=np.uint8),
                __rivermark_frame_count__=np.asarray([2], dtype=np.int64),
                timestamps_ns=np.asarray([100, 200], dtype=np.int64),
            )
            streams = inspect_candidate_pack_streams(capture, inspection=inspection)
            descriptor_root = root / "descriptor"
            abi_path = descriptor_root / "observation_abi.json"
            abi = _formal_abi(streams)
            _json(abi_path, abi)
            spec_path = descriptor_root / "pack_spec.json"
            spec = _v2_pack_spec(
                streams,
                abi_name=abi_path.name,
                abi_sha256=observation_abi_sha256(abi),
                capture_sha256=sha256_file(capture / "capture_receipt.json"),
            )
            _json(spec_path, spec)
            before = sorted(path.relative_to(capture) for path in capture.rglob("*"))
            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                return_value=inspection,
            ):
                report = audit_isaac_pack_readiness(
                    capture,
                    observation_abi=abi_path,
                    pack_spec=spec_path,
                )
            after = sorted(path.relative_to(capture) for path in capture.rglob("*"))

            self.assertEqual(before, after)
            self.assertTrue(report.checks["formal_observation_abi"])
            self.assertTrue(report.checks["pack_spec"])

            spec["observation_abi"]["capture_receipt_sha256"] = "0" * 64
            _json(spec_path, spec)
            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                return_value=inspection,
            ):
                tampered = audit_isaac_pack_readiness(
                    capture,
                    observation_abi=abi_path,
                    pack_spec=spec_path,
                )
            self.assertFalse(tampered.checks["pack_spec"])
            self.assertIn("pack_abi_mismatch", {issue.code for issue in tampered.issues})

            spec["observation_abi"]["capture_receipt_sha256"] = sha256_file(
                capture / "capture_receipt.json"
            )
            depth = next(
                stream for stream in spec["streams"] if stream["stream_id"] == "depth"
            )
            depth["path"] = "streams/duplicated_depth.npz"
            _json(spec_path, spec)
            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                return_value=inspection,
            ):
                duplicated = audit_isaac_pack_readiness(
                    capture,
                    observation_abi=abi_path,
                    pack_spec=spec_path,
                )
            self.assertFalse(duplicated.checks["pack_spec"])
            self.assertIn(
                "chunked_archive_destination",
                {issue.code for issue in duplicated.issues},
            )

    def test_v2_spec_rejects_public_task_leak_and_layout_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, inspection = _capture(root / "capture")
            np.savez(
                capture / "sensors/onboard_rgbd.npz",
                __rivermark_chunked_frame_archive_v1__=np.asarray([1], dtype=np.uint8),
                __rivermark_frame_count__=np.asarray([2], dtype=np.int64),
                timestamps_ns=np.asarray([100, 200], dtype=np.int64),
            )
            streams = inspect_candidate_pack_streams(capture, inspection=inspection)
            descriptor_root = root / "descriptor"
            abi_path = descriptor_root / "observation_abi.json"
            abi = _formal_abi(streams)
            _json(abi_path, abi)
            spec_path = descriptor_root / "pack_spec.json"
            spec = _v2_pack_spec(
                streams,
                abi_name=abi_path.name,
                abi_sha256=observation_abi_sha256(abi),
                capture_sha256=sha256_file(capture / "capture_receipt.json"),
            )
            _json(spec_path, spec)
            leaked_task = _public_task()
            leaked_task["hidden_target_id"] = 4
            _json(capture / "public_task.json", leaked_task)

            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                return_value=inspection,
            ):
                leaked = audit_isaac_pack_readiness(
                    capture,
                    observation_abi=abi_path,
                    pack_spec=spec_path,
                )
            self.assertFalse(leaked.checks["pack_spec"])
            self.assertIn(
                "public_task_projection",
                {issue.code for issue in leaked.issues},
            )

            _json(capture / "public_task.json", _public_task())
            spec["layout"]["layout_hash"] = "0" * 64
            _json(spec_path, spec)
            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                return_value=inspection,
            ):
                tampered = audit_isaac_pack_readiness(
                    capture,
                    observation_abi=abi_path,
                    pack_spec=spec_path,
                )
            self.assertFalse(tampered.checks["pack_spec"])
            self.assertIn("layout_hash", {issue.code for issue in tampered.issues})

    def test_missing_prerequisites_are_explicit_and_do_not_mutate_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            capture, inspection = _capture(Path(temporary) / "capture")
            before = sorted(path.relative_to(capture) for path in capture.rglob("*"))
            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                return_value=inspection,
            ):
                report = audit_isaac_pack_readiness(capture)
            after = sorted(path.relative_to(capture) for path in capture.rglob("*"))

            self.assertEqual(before, after)
            self.assertFalse(report.candidate_pack_ready)
            self.assertFalse(report.supply_chain_release_ready)
            self.assertTrue(report.checks["candidate_stream_contract"])
            self.assertEqual(
                {stream["modality"] for stream in report.candidate_streams.values()},
                {
                    "high_level_action_history",
                    "proprioception",
                    "public_task_state",
                    "public_team_messages",
                    "rgb",
                    "distance_to_image_plane",
                    "lidar",
                    "imu",
                },
            )
            self.assertFalse(report.as_dict()["formal_benchmark_admission"])
            codes = {issue.code for issue in report.issues}
            self.assertTrue(
                {
                    "collection_protocol_missing",
                    "evaluator_manifest_missing",
                    "observation_abi_missing",
                    "pack_spec_missing",
                    "supply_chain_manifest_missing",
                }
                <= codes
            )

    def test_wrong_evaluator_bytes_are_rejected_without_private_path_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, inspection = _capture(root / "capture")
            evaluator = root / "private.json"
            evaluator.write_text("{}\n", encoding="utf-8")
            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                return_value=inspection,
            ):
                report = audit_isaac_pack_readiness(capture, evaluator_manifest=evaluator)

            issue = next(item for item in report.issues if item.code == "evaluator_manifest_mismatch")
            self.assertNotIn(str(evaluator), json.dumps(report.as_dict()))
            self.assertEqual(issue.path, "evaluator_manifest")

    def test_missing_evaluator_path_is_distinguished_from_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, inspection = _capture(root / "capture")
            missing = root / "missing-evaluator.json"
            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                return_value=inspection,
            ):
                report = audit_isaac_pack_readiness(
                    capture,
                    evaluator_manifest=missing,
                )

            codes = {issue.code for issue in report.issues}
            self.assertIn("evaluator_manifest_missing", codes)
            self.assertNotIn("evaluator_manifest_mismatch", codes)
            self.assertNotIn(str(missing), json.dumps(report.as_dict()))

    def test_spec_rejects_source_revision_private_field_and_learning_label_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, inspection = _capture(root / "capture")
            source = capture / "learning_labels/state.npz"
            source.parent.mkdir()
            np.savez_compressed(
                source,
                timestamps_ns=np.asarray([1, 2], dtype=np.int64),
                target_ids=np.asarray([3, 4], dtype=np.int64),
            )
            spec = root / "pack.json"
            _json(spec, _pack_spec("learning_labels/state.npz", ["timestamps_ns", "target_ids"]))
            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                return_value=inspection,
            ):
                report = audit_isaac_pack_readiness(capture, pack_spec=spec)

            codes = {issue.code for issue in report.issues}
            self.assertIn("source_revision_mismatch", codes)
            self.assertIn("private_or_unsafe_source", codes)

    def test_partial_chunked_archive_selection_and_malformed_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, inspection = _capture(root / "capture")
            source = capture / "sensors/onboard_rgbd.npz"
            source.parent.mkdir(exist_ok=True)
            np.savez(
                source,
                __rivermark_chunked_frame_archive_v1__=np.asarray([1], dtype=np.uint8),
                __rivermark_frame_count__=np.asarray([2], dtype=np.int64),
                timestamps_ns=np.asarray([1, 2], dtype=np.int64),
                rgb__frame__000000=np.zeros((8, 2, 2, 3), dtype=np.uint8),
                rgb__frame__000001=np.zeros((8, 2, 2, 3), dtype=np.uint8),
            )
            spec = root / "pack.json"
            _json(
                spec,
                _pack_spec(
                    "sensors/onboard_rgbd.npz",
                    ["timestamps_ns", "rgb__frame__000000"],
                ),
            )
            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                return_value=inspection,
            ):
                report = audit_isaac_pack_readiness(capture, pack_spec=spec)
            self.assertIn(
                "chunked_archive_projection",
                {issue.code for issue in report.issues},
            )

            with patch(
                "rivermark_benchmark.isaac_pack_readiness.inspect_policy_observation_sources",
                side_effect=PolicyProjectionError("missing or unexpected frames"),
            ):
                malformed = audit_isaac_pack_readiness(capture)
            self.assertIn("policy_source_contract", {issue.code for issue in malformed.issues})


if __name__ == "__main__":
    unittest.main()
