from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.formal_dataset import sha256_file
from rivermark_benchmark.parquet_projection import (
    ParquetProjectionError,
    project_development_capture_to_parquet,
    read_development_parquet_table,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _capture_fixture(root: Path, *, formal_admission: bool = False) -> Path:
    root.mkdir(parents=True)
    streams = root / "streams"
    streams.mkdir()
    state = {
        "command_time_ns": np.asarray([0, 100_000_000, 200_000_000], dtype="<i8"),
        "effective_time_ns": np.asarray([1, 100_000_001, 200_000_001], dtype="<i8"),
        "root_pos_w_m": np.arange(18, dtype="<f4").reshape(3, 2, 3),
        "root_quat_wxyz": np.tile(np.asarray([1, 0, 0, 0], dtype="<f4"), (3, 2, 1)),
        "root_lin_vel_w_mps": np.zeros((3, 2, 3), dtype="<f4"),
        "root_ang_vel_b_radps": np.zeros((3, 2, 3), dtype="<f4"),
        "desired_pos_w_m": np.ones((3, 2, 3), dtype="<f4"),
        "desired_vel_w_mps": np.zeros((3, 2, 3), dtype="<f4"),
        "target_thrust_n": np.ones((3, 2, 4), dtype="<f4"),
        "applied_thrust_n": np.ones((3, 2, 4), dtype="<f4"),
    }
    task = {
        "timestamps_ns": np.asarray([100, 200], dtype="<i8"),
        "waypoint_index": np.zeros((2, 2), dtype="<i8"),
        "waypoint_progress": np.zeros((2, 2), dtype="<f4"),
        "desired_waypoint_w_m": np.ones((2, 2, 3), dtype="<f4"),
        "distance_to_waypoint_m": np.ones((2, 2), dtype="<f4"),
        "waypoint_reached": np.zeros((2, 2), dtype=bool),
        "action_mode": np.zeros((2, 2), dtype="i1"),
        "coverage_cell_id": np.zeros((2, 2), dtype="<i8"),
        "task_time_s": np.zeros((2, 2), dtype="<f4"),
    }
    messages = {
        "timestamps_ns": np.asarray([100, 200], dtype="<i8"),
        "sender_agent_id": np.zeros((2, 2), dtype="<i8"),
        "message_sequence": np.zeros((2, 2), dtype="<i8"),
        "message_waypoint_index": np.zeros((2, 2), dtype="<i8"),
        "message_position_w_m": np.zeros((2, 2, 3), dtype="<f4"),
        "message_velocity_w_mps": np.zeros((2, 2, 3), dtype="<f4"),
        "message_flags": np.zeros((2, 2), dtype="u1"),
    }
    for name, values in (
        ("state_action.npz", state),
        ("public_task.npz", task),
        ("public_messages.npz", messages),
    ):
        np.savez_compressed(streams / name, **values)

    artifact_hashes = {
        f"streams/{name}": {"bytes": (streams / name).stat().st_size, "sha256": sha256_file(streams / name)}
        for name in ("state_action.npz", "public_task.npz", "public_messages.npz")
    }
    receipt = {
        "schema": "org.rivermark.isaac-swarm-capture.v1",
        "status": "captured",
        "ok": True,
        "source_worktree_dirty": False,
        "source_revision": "a" * 40,
        "capture_attempt_id": "attempt-parquet-fixture",
        "claim_boundary": {"formal_benchmark_admission": formal_admission},
        "collection_binding": {
            "protocol_id": "citylite-test-v1",
            "protocol_sha256": "b" * 64,
            "cell_id": "train-test",
            "split": "train",
            "episode_index": 1,
            "episode_seed": 42,
            "private_evaluator_path": "C:/private/hidden-targets.json",
        },
        "physics": {"same_world_agent_count": 2, "physics_steps": 3, "sensor_samples": 2},
        "artifact_hashes": artifact_hashes,
    }
    receipt_path = root / "capture_receipt.json"
    _write_json(receipt_path, receipt)
    _write_json(
        root / "independent_validation.json",
        {
            "schema": "org.rivermark.isaac-independent-validation.v1",
            "status": "passed",
            "issues": [],
            "capture_receipt_sha256": sha256_file(receipt_path),
        },
    )
    return root


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional parquet dependency is not installed")
class ParquetProjectionTests(unittest.TestCase):
    def test_projection_has_reader_parity_and_filters_private_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = _capture_fixture(root / "capture")
            output = root / "parquet"
            result = project_development_capture_to_parquet(capture, output, row_group_size=4)

            self.assertEqual(result.table_paths, ("metadata.parquet", "state_action.parquet", "public_task.parquet", "public_messages.parquet"))
            manifest = json.loads((output / "projection_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["formal_benchmark_admission"])
            self.assertEqual(manifest["collection_binding"], {
                "protocol_id": "citylite-test-v1",
                "protocol_sha256": "b" * 64,
                "cell_id": "train-test",
                "split": "train",
                "episode_index": 1,
                "episode_seed": 42,
            })
            serialized = json.dumps(manifest, sort_keys=True)
            self.assertNotIn("private_evaluator_path", serialized)
            self.assertNotIn("hidden-targets", serialized)
            self.assertEqual(read_development_parquet_table(output, "state_action.parquet").num_rows, 6)
            self.assertEqual(read_development_parquet_table(output, "public_task.parquet").num_rows, 4)
            self.assertEqual(read_development_parquet_table(output, "public_messages.parquet").num_rows, 4)
            self.assertEqual(read_development_parquet_table(output, "metadata.parquet").num_rows, 1)

    def test_formal_admission_and_source_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal_capture = _capture_fixture(root / "formal", formal_admission=True)
            with self.assertRaisesRegex(ParquetProjectionError, "formal_benchmark_admission=false"):
                project_development_capture_to_parquet(formal_capture, root / "formal-output")
            tampered_capture = _capture_fixture(root / "tampered")
            with (tampered_capture / "streams" / "public_task.npz").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ParquetProjectionError, "does not match its capture receipt binding"):
                project_development_capture_to_parquet(tampered_capture, root / "tampered-output")
            self.assertFalse((root / "tampered-output").exists())

    def test_unsafe_reader_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ParquetProjectionError):
            read_development_parquet_table(Path(temporary), "../metadata.parquet")


if __name__ == "__main__":
    unittest.main()
