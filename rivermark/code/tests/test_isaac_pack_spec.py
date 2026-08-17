from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.isaac_pack import validate_isaac_pack_spec
from rivermark_benchmark.isaac_pack_descriptor import build_isaac_observation_abi
from rivermark_benchmark.isaac_pack_spec import (
    IsaacPackSpecError,
    build_isaac_pack_spec,
    write_pack_spec,
)
from rivermark_benchmark.isaac_public_manifest import (
    build_public_scene_manifest,
    public_manifest_sha256,
)


def _source_streams() -> dict[str, object]:
    def stream(
        path: str,
        modality: str,
        timestamp: str,
        arrays: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        return {
            "path": path,
            "modality": modality,
            "timestamp_field": timestamp,
            "fields": list(arrays),
            "arrays": arrays,
        }

    return {
        "actions": stream(
            "streams/state_action.npz",
            "high_level_action_history",
            "command_time_ns",
            {
                "command_time_ns": {"dtype": "<i8", "shape": [3]},
                "effective_time_ns": {"dtype": "<i8", "shape": [3]},
                "desired_pos_w_m": {"dtype": "<f4", "shape": [3, 8, 3]},
                "desired_vel_w_mps": {"dtype": "<f4", "shape": [3, 8, 3]},
            },
        ),
        "state": stream(
            "streams/state_action.npz",
            "proprioception",
            "effective_time_ns",
            {
                "effective_time_ns": {"dtype": "<i8", "shape": [3]},
                "root_ang_vel_b_radps": {"dtype": "<f4", "shape": [3, 8, 3]},
                "root_lin_vel_w_mps": {"dtype": "<f4", "shape": [3, 8, 3]},
                "root_pos_w_m": {"dtype": "<f4", "shape": [3, 8, 3]},
                "root_quat_wxyz": {"dtype": "<f4", "shape": [3, 8, 4]},
            },
        ),
        "task": stream(
            "streams/public_task.npz",
            "public_task_state",
            "timestamps_ns",
            {
                "timestamps_ns": {"dtype": "<i8", "shape": [2]},
                "task_time_s": {"dtype": "<f4", "shape": [2, 8]},
            },
        ),
        "messages": stream(
            "streams/public_messages.npz",
            "public_team_messages",
            "timestamps_ns",
            {
                "timestamps_ns": {"dtype": "<i8", "shape": [2]},
                "sender_agent_id": {"dtype": "<i8", "shape": [2, 8]},
            },
        ),
        "rgb": stream(
            "sensors/onboard_rgbd.npz",
            "rgb",
            "timestamps_ns",
            {
                "timestamps_ns": {"dtype": "<i8", "shape": [2]},
                "rgb": {"dtype": "|u1", "shape": [2, 8, 120, 160, 3]},
            },
        ),
        "depth": stream(
            "sensors/onboard_rgbd.npz",
            "distance_to_image_plane",
            "timestamps_ns",
            {
                "timestamps_ns": {"dtype": "<i8", "shape": [2]},
                "distance_to_image_plane_m": {
                    "dtype": "<f4",
                    "shape": [2, 8, 120, 160, 1],
                },
            },
        ),
        "lidar": stream(
            "sensors/lidar.npz",
            "lidar",
            "timestamps_ns",
            {
                "timestamps_ns": {"dtype": "<i8", "shape": [2]},
                "ranges_m": {"dtype": "<f4", "shape": [2, 8, 1152]},
            },
        ),
        "imu": stream(
            "sensors/imu.npz",
            "imu",
            "timestamps_ns",
            {
                "timestamps_ns": {"dtype": "<i8", "shape": [2]},
                "angular_velocity_b_radps": {
                    "dtype": "<f4",
                    "shape": [2, 8, 3],
                },
                "linear_acceleration_b_mps2": {
                    "dtype": "<f4",
                    "shape": [2, 8, 3],
                },
            },
        ),
    }


def _abi(streams: dict[str, object]) -> dict[str, object]:
    matrix = [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]
    calibration = {
        "onboard_camera": {
            "intrinsic_matrices": [matrix for _ in range(8)],
            "image_shape_hw": [120, 160],
        },
        "lidar": {"max_distance_m": 100.0},
        "imu": {"implementation": "isaaclab.sensors.Imu", "attachment_frame": "body_flu"},
    }
    return build_isaac_observation_abi(streams, calibration)


def _receipt() -> dict[str, object]:
    return {
        "status": "captured",
        "ok": True,
        "source_worktree_dirty": False,
        "task_kind": "search3d",
        "source_revision": "a" * 40,
        "information_profile": "multisensor_rgbd_lidar_imu_state",
        "command": {"dt_s": 0.005, "capture_stride": 10},
        "collection_binding": {
            "cell_id": "train-citylite-direct-v2",
            "split": "train",
            "episode_index": 1,
        },
        "condition_request": {
            "cell_id": "train-citylite-direct-v2",
            "conditions": {
                "layout": "citylite-v1",
                "visibility_bucket": "direct-visible-v1",
                "dynamics": "cf2x-nominal-v1",
                "route": "fixed-public-route-v1",
            },
        },
        "capture_backend": {
            "kind": "isaaclab",
            "build": "isaaclab:test",
            "sensor_physics_smoke_receipt_sha256": "b" * 64,
        },
    }


def _scene() -> dict[str, object]:
    return {
        "schema": "org.rivermark.public-isaac-scene.v1",
        "environment_id": "RIVERMARK_CITY_LITE_v1",
        "agent_count": 8,
        "fresh_stage": True,
        "static_scene_authority_verified": True,
        "legacy_route_or_target_imported": False,
        "unresolved_reference_count": 0,
        "private_evaluator_manifest_sha256": "9" * 64,
        "source_scene": r"C:\private\rivermark.usd",
        "scene_contract": {
            "schema": "citylite-contract-v1",
            "gate_status": "pass_city_lite_static_construction",
            "payload_sha256": "c" * 64,
            "sha256": "1" * 64,
        },
        "rivermark_layer_inventory": {
            "schema": "resolved-layer-inventory-v1",
            "inventory_sha256": "d" * 64,
            "local_authority_inventory_sha256": "2" * 64,
            "rivermarksrc51_external_inventory_sha256": "3" * 64,
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
        "task_variant_id": "isaac-eight-agent-public-waypoint-search-v1",
        "agent_count": 8,
        "route_conditioning": "public_only",
    }


def test_builder_derives_closed_eight_stream_spec_without_clearance_claim() -> None:
    streams = _source_streams()
    payload = build_isaac_pack_spec(
        capture_receipt=_receipt(),
        scene=_scene(),
        public_task=_public_task(),
        observation_abi=_abi(streams),
        source_streams=streams,
        capture_receipt_sha256="e" * 64,
        observation_abi_source="observation_abi.json",
        dataset_version="0.2.0-dev",
    )

    assert validate_isaac_pack_spec(payload) == ()
    assert payload["provenance"]["scene_asset_license_status"] == "pending"
    assert payload["quality"] == {"task_success": False, "invalid_reasons": []}
    assert payload["timebase"]["physics_dt_ns"] == 5_000_000
    assert payload["timebase"]["camera_period_ns"] == 50_000_000
    by_id = {stream["stream_id"]: stream for stream in payload["streams"]}
    assert set(by_id) == set(streams)
    assert by_id["rgb"]["path"] == by_id["depth"]["path"]
    assert "fields" not in by_id["rgb"]
    assert by_id["rgb"]["sample_count"] == 2
    assert payload["observation_abi"]["capture_receipt_sha256"] == "e" * 64
    assert payload["layout"]["layout_hash"] == public_manifest_sha256(
        build_public_scene_manifest(_scene())
    )
    assert "private" not in json.dumps(payload, sort_keys=True).lower()


def test_builder_rejects_dirty_capture() -> None:
    streams = _source_streams()
    receipt = _receipt()
    receipt["source_worktree_dirty"] = True
    with pytest.raises(IsaacPackSpecError, match="clean source"):
        build_isaac_pack_spec(
            capture_receipt=receipt,
            scene=_scene(),
            public_task=_public_task(),
            observation_abi=_abi(streams),
            source_streams=streams,
            capture_receipt_sha256="e" * 64,
            observation_abi_source="observation_abi.json",
            dataset_version="0.2.0-dev",
        )


def test_builder_rejects_private_task_and_agent_count_mismatch() -> None:
    streams = _source_streams()
    private_task = _public_task()
    private_task["hidden_target_id"] = 7
    with pytest.raises(ValueError, match="forbidden"):
        build_isaac_pack_spec(
            capture_receipt=_receipt(),
            scene=_scene(),
            public_task=private_task,
            observation_abi=_abi(streams),
            source_streams=streams,
            capture_receipt_sha256="e" * 64,
            observation_abi_source="observation_abi.json",
            dataset_version="0.2.0-dev",
        )

    wrong_count = _public_task()
    wrong_count["agent_count"] = 7
    with pytest.raises(IsaacPackSpecError, match="does not match"):
        build_isaac_pack_spec(
            capture_receipt=_receipt(),
            scene=_scene(),
            public_task=wrong_count,
            observation_abi=_abi(streams),
            source_streams=streams,
            capture_receipt_sha256="e" * 64,
            observation_abi_source="observation_abi.json",
            dataset_version="0.2.0-dev",
        )


def test_writer_is_atomic_and_refuses_overwrite() -> None:
    payload = {"schema": "fixture"}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "pack_spec.json"
        with patch(
            "rivermark_benchmark.isaac_pack_spec.pack_spec_for_capture",
            return_value=payload,
        ):
            digest = write_pack_spec(
                root / "capture",
                root / "observation_abi.json",
                output,
                dataset_version="0.2.0-dev",
            )
        assert len(digest) == 64
        assert json.loads(output.read_text(encoding="utf-8")) == payload
        with pytest.raises(IsaacPackSpecError, match="refusing to overwrite"):
            write_pack_spec(
                root / "capture",
                root / "observation_abi.json",
                output,
                dataset_version="0.2.0-dev",
            )
