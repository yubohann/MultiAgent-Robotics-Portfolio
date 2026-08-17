from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.abi import validate_formal_observation_abi
from rivermark_benchmark.isaac_pack_descriptor import (
    IsaacPackDescriptorError,
    build_isaac_observation_abi,
)
from rivermark_benchmark.policy_projection import validate_candidate_abi_sources


def _stream(modality: str, arrays: dict[str, tuple[str, list[int]]]) -> dict[str, object]:
    return {
        "modality": modality,
        "timestamp_field": (
            "command_time_ns"
            if modality == "high_level_action_history"
            else "effective_time_ns"
            if modality == "proprioception"
            else "timestamps_ns"
        ),
        "arrays": {
            name: {"dtype": dtype, "shape": shape}
            for name, (dtype, shape) in arrays.items()
        },
    }


def _streams() -> dict[str, object]:
    return {
        "actions": _stream("high_level_action_history", {
            "command_time_ns": ("<i8", [20]),
            "effective_time_ns": ("<i8", [20]),
            "desired_pos_w_m": ("<f4", [20, 8, 3]),
            "desired_vel_w_mps": ("<f4", [20, 8, 3]),
        }),
        "state": _stream("proprioception", {
            "effective_time_ns": ("<i8", [20]),
            "root_ang_vel_b_radps": ("<f4", [20, 8, 3]),
            "root_lin_vel_w_mps": ("<f4", [20, 8, 3]),
            "root_pos_w_m": ("<f4", [20, 8, 3]),
            "root_quat_wxyz": ("<f4", [20, 8, 4]),
        }),
        "task": _stream("public_task_state", {
            "timestamps_ns": ("<i8", [2]),
            "waypoint_index": ("<i8", [2, 8]),
            "waypoint_progress": ("<f4", [2, 8]),
            "desired_waypoint_w_m": ("<f4", [2, 8, 3]),
            "distance_to_waypoint_m": ("<f4", [2, 8]),
            "waypoint_reached": ("|b1", [2, 8]),
            "action_mode": ("|i1", [2, 8]),
            "coverage_cell_id": ("<i8", [2, 8]),
            "task_time_s": ("<f4", [2, 8]),
        }),
        "messages": _stream("public_team_messages", {
            "timestamps_ns": ("<i8", [2]),
            "sender_agent_id": ("<i8", [2, 8]),
            "message_sequence": ("<i8", [2, 8]),
            "message_waypoint_index": ("<i8", [2, 8]),
            "message_position_w_m": ("<f4", [2, 8, 3]),
            "message_velocity_w_mps": ("<f4", [2, 8, 3]),
            "message_flags": ("|u1", [2, 8]),
        }),
        "rgb": _stream("rgb", {
            "timestamps_ns": ("<i8", [2]),
            "rgb": ("|u1", [2, 8, 120, 160, 3]),
        }),
        "depth": _stream("distance_to_image_plane", {
            "timestamps_ns": ("<i8", [2]),
            "distance_to_image_plane_m": ("<f4", [2, 8, 120, 160, 1]),
        }),
        "lidar": _stream("lidar", {
            "timestamps_ns": ("<i8", [2]),
            "ranges_m": ("<f4", [2, 8, 1152]),
        }),
        "imu": _stream("imu", {
            "timestamps_ns": ("<i8", [2]),
            "angular_velocity_b_radps": ("<f4", [2, 8, 3]),
            "linear_acceleration_b_mps2": ("<f4", [2, 8, 3]),
        }),
    }


def _calibration() -> dict[str, object]:
    matrix = [[183.25, 0.0, 80.0], [0.0, 183.25, 60.0], [0.0, 0.0, 1.0]]
    return {
        "onboard_camera": {
            "image_shape_hw": [120, 160],
            "intrinsic_matrices": [copy.deepcopy(matrix) for _ in range(8)],
        },
        "lidar": {"max_distance_m": 100.0},
        "imu": {"implementation": "fixture", "attachment_frame": "body_flu"},
    }


class IsaacPackDescriptorTests(unittest.TestCase):
    def test_builds_formal_abi_that_matches_all_eight_source_streams(self) -> None:
        streams = _streams()
        payload = build_isaac_observation_abi(streams, _calibration())

        self.assertEqual(validate_formal_observation_abi(payload), ())
        self.assertEqual(validate_candidate_abi_sources(payload, streams), ())
        self.assertEqual({stream["stream_id"] for stream in payload["streams"]}, set(streams))
        task = next(stream for stream in payload["streams"] if stream["stream_id"] == "task")
        action_mode = next(field for field in task["fields"] if field["name"] == "action_mode")
        self.assertEqual(action_mode["dtype"], np.dtype(np.int8).name)
        self.assertEqual(action_mode["shape"], ["sensor_frame", 8])
        rgb = next(stream for stream in payload["streams"] if stream["stream_id"] == "rgb")
        self.assertIn("rolling_shutter", rgb["fidelity_limitations"][0])

    def test_camera_intrinsics_must_be_identical_and_match_source_resolution(self) -> None:
        calibration = _calibration()
        calibration["onboard_camera"]["intrinsic_matrices"][7][0][0] += 1.0
        with self.assertRaisesRegex(IsaacPackDescriptorError, "intrinsic matrices"):
            build_isaac_observation_abi(_streams(), calibration)

        calibration = _calibration()
        calibration["onboard_camera"]["image_shape_hw"] = [72, 96]
        with self.assertRaisesRegex(IsaacPackDescriptorError, "resolution"):
            build_isaac_observation_abi(_streams(), calibration)

    def test_historical_capture_without_imu_calibration_is_explicitly_unavailable(self) -> None:
        calibration = _calibration()
        calibration.pop("imu")

        payload = build_isaac_observation_abi(_streams(), calibration)

        self.assertEqual(payload["calibration"]["imu"]["status"], "unavailable")
        self.assertNotIn("frame_id", payload["calibration"]["imu"])
        self.assertEqual(validate_formal_observation_abi(payload), ())

    def test_malformed_imu_calibration_fails_closed(self) -> None:
        calibration = _calibration()
        calibration["imu"] = "body"

        with self.assertRaisesRegex(IsaacPackDescriptorError, "must be an object"):
            build_isaac_observation_abi(_streams(), calibration)

        calibration = _calibration()
        calibration["imu"] = {}
        with self.assertRaisesRegex(IsaacPackDescriptorError, "requires an implementation"):
            build_isaac_observation_abi(_streams(), calibration)

        calibration = _calibration()
        calibration["imu"]["attachment_frame"] = "world"
        with self.assertRaisesRegex(IsaacPackDescriptorError, "attachment_frame body_flu"):
            build_isaac_observation_abi(_streams(), calibration)

    def test_unknown_or_missing_stream_semantics_fail_closed(self) -> None:
        streams = _streams()
        streams.pop("imu")
        with self.assertRaisesRegex(IsaacPackDescriptorError, "exact eight"):
            build_isaac_observation_abi(streams, _calibration())

        streams = _streams()
        streams["imu"]["arrays"]["temperature_c"] = {
            "dtype": "<f4",
            "shape": [2, 8],
        }
        with self.assertRaisesRegex(IsaacPackDescriptorError, "no reviewed ABI semantics"):
            build_isaac_observation_abi(streams, _calibration())


if __name__ == "__main__":
    unittest.main()
