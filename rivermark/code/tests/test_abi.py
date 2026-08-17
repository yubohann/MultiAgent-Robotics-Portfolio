from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.abi import (
    OBSERVATION_ABI_SCHEMA,
    AbiError,
    assess_observation_abi_compatibility,
    load_observation_abi,
    observation_abi_sha256,
    validate_formal_observation_abi,
    validate_observation_abi,
)


def _field(name: str, *, time_semantics: str = "sensor_sample") -> dict[str, object]:
    return {
        "name": name,
        "dtype": "uint8" if name == "rgb" else "float32",
        "shape": ["frame", "agent", "height", "width", 3] if name == "rgb" else ["frame", "agent", 4],
        "units": "unitless" if name == "rgb" else "m/s",
        "frame_id": "camera_optical" if name == "rgb" else "body",
        "agent_id_field": "agent_id",
        "timestamp_field": "sensor_time_ns",
        "missing": {"policy": "not_applicable", "sentinel": None, "mask_field": None},
        "valid_range": {"min": 0, "max": 255 if name == "rgb" else None, "inclusive": True},
        "compression": "npz_deflate",
        "time_semantics": time_semantics,
    }


def _abi() -> dict[str, object]:
    return {
        "schema": OBSERVATION_ABI_SCHEMA,
        "version": "1.0.0",
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
                "source": "isaac_camera_intrinsics_v1",
                "intrinsics": {"model": "pinhole", "width_px": 96, "height_px": 72, "fx_px": 80.0, "fy_px": 80.0, "cx_px": 48.0, "cy_px": 36.0},
                "extrinsics": {"formula": "T_world_camera = T_world_body * T_body_camera", "quaternion_order": "wxyz"},
                "distortion_model": "none",
                "distortion_coefficients": [],
            },
            "lidar": {"status": "unavailable", "source": "not-captured"},
            "imu": {"status": "unavailable", "source": "not-captured"},
        },
        "streams": [
            {"stream_id": "rgb", "modality": "rgb", "partition": "policy_visible", "encoding": "npz", "fields": [_field("rgb")]},
            {"stream_id": "actions", "modality": "high_level_action_history", "partition": "policy_visible", "encoding": "jsonl", "fields": [_field("velocity", time_semantics="command_before_step")]},
        ],
    }


class ObservationAbiTests(unittest.TestCase):
    def test_valid_abi_is_stable_and_hashable(self) -> None:
        payload = _abi()
        self.assertEqual(validate_observation_abi(payload), ())
        first = observation_abi_sha256(payload)
        changed = copy.deepcopy(payload)
        changed["streams"][0]["fields"][0]["units"] = "pixel"
        self.assertNotEqual(first, observation_abi_sha256(changed))

    def test_signed_int8_action_mode_is_supported(self) -> None:
        payload = _abi()
        payload["streams"][1]["fields"][0]["dtype"] = "int8"

        self.assertEqual(validate_observation_abi(payload), ())
        self.assertEqual(len(observation_abi_sha256(payload)), 64)

    def test_action_causality_and_coordinate_tampering_fail_closed(self) -> None:
        payload = _abi()
        payload["action_timing"]["sensor_read"] = "before_simulation_step"
        payload["coordinate_frames"]["quaternion_order"] = "xyzw"
        codes = {issue.code for issue in validate_observation_abi(payload)}
        self.assertIn("action_causality", codes)
        self.assertIn("coordinate_convention", codes)

    def test_recorded_camera_requires_extrinsics_and_distortion_contract(self) -> None:
        payload = _abi()
        del payload["calibration"]["camera"]["extrinsics"]
        payload["calibration"]["camera"]["distortion_model"] = "unknown"
        codes = {issue.code for issue in validate_observation_abi(payload)}
        self.assertIn("camera_extrinsics", codes)
        self.assertIn("distortion_model", codes)

    def test_unavailable_sensor_cannot_carry_calibration_and_ranges_are_ordered(self) -> None:
        payload = _abi()
        payload["calibration"]["lidar"]["frame_id"] = "lidar"
        payload["streams"][0]["fields"][0]["valid_range"] = {"min": 255, "max": 0, "inclusive": True}
        codes = {issue.code for issue in validate_observation_abi(payload)}
        self.assertIn("calibration_unavailable_payload", codes)
        self.assertIn("valid_range", codes)

    def test_load_rejects_invalid_file_without_exposing_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "abi.json"
            path.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
            with self.assertRaises(AbiError):
                load_observation_abi(path)

    def test_formal_validation_requires_fidelity_but_legacy_reader_does_not(self) -> None:
        payload = _abi()
        self.assertEqual(validate_observation_abi(payload), ())
        self.assertTrue({issue.code for issue in validate_formal_observation_abi(payload)} & {"fidelity_required"})
        payload["version"] = "1.1.0"
        for stream in payload["streams"]:
            stream["fidelity"] = "simulator_consistent"
            stream["fidelity_limitations"] = ["hardware noise and calibration are not represented"]
        self.assertEqual(validate_formal_observation_abi(payload), ())

    def test_fidelity_limitations_must_be_unique_nonempty_strings(self) -> None:
        payload = _abi()
        payload["version"] = "1.1.0"
        payload["streams"][0]["fidelity"] = "simulator_consistent"
        payload["streams"][0]["fidelity_limitations"] = ["", "same", "same"]
        codes = {issue.code for issue in validate_observation_abi(payload)}
        self.assertIn("fidelity_limitation", codes)
        self.assertIn("fidelity_limitation_duplicate", codes)

    def test_legacy_producer_is_development_readable_but_not_formal(self) -> None:
        producer = _abi()
        reader = copy.deepcopy(producer)
        reader["version"] = "1.1.0"
        for stream in reader["streams"]:
            stream["fidelity"] = "simulator_consistent"
            stream["fidelity_limitations"] = ["hardware calibration is not represented"]
        report = assess_observation_abi_compatibility(producer, reader)
        self.assertTrue(report.development_readable)
        self.assertFalse(report.formal_admissible)
        self.assertEqual(report.issues, ())

    def test_newer_producer_and_semantic_change_fail_closed(self) -> None:
        producer = copy.deepcopy(_abi())
        producer["version"] = "1.1.0"
        for stream in producer["streams"]:
            stream["fidelity"] = "simulator_consistent"
            stream["fidelity_limitations"] = ["hardware calibration is not represented"]
        reader = copy.deepcopy(producer)
        reader["version"] = "1.0.0"
        report = assess_observation_abi_compatibility(producer, reader)
        self.assertFalse(report.development_readable)
        self.assertIn("producer_newer_than_reader", report.issues)
        reader = copy.deepcopy(producer)
        reader["coordinate_frames"]["quaternion_order"] = "xyzw"
        report = assess_observation_abi_compatibility(producer, reader)
        self.assertFalse(report.development_readable)
        self.assertIn("semantic_contract_mismatch", report.issues)

    def test_formal_pair_is_admissible_but_major_migration_is_not(self) -> None:
        producer = copy.deepcopy(_abi())
        producer["version"] = "1.1.0"
        for stream in producer["streams"]:
            stream["fidelity"] = "simulator_consistent"
            stream["fidelity_limitations"] = ["hardware calibration is not represented"]
        report = assess_observation_abi_compatibility(producer, copy.deepcopy(producer))
        self.assertTrue(report.development_readable)
        self.assertTrue(report.formal_admissible)
        migrated = copy.deepcopy(producer)
        migrated["version"] = "2.0.0"
        report = assess_observation_abi_compatibility(producer, migrated)
        self.assertFalse(report.development_readable)
        self.assertFalse(report.formal_admissible)
        self.assertIn("major_version_mismatch", report.issues)


if __name__ == "__main__":
    unittest.main()
