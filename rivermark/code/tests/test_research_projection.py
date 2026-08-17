from __future__ import annotations

import json
import importlib.util
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

from rivermark_benchmark.formal_dataset import sha256_file
from rivermark_benchmark.research_projection import (
    ProjectionError,
    project_episode_to_zarr,
    read_zarr_array,
    read_zarr_array_independent,
    read_zarr_array_external,
)
from tests.test_formal_dataset import _candidate, _write_json


def _append_npz_stream(candidate: Path, stream_id: str, arrays: dict[str, np.ndarray]) -> Path:
    payload_path = candidate / "streams" / f"{stream_id}.npz"
    np.savez_compressed(payload_path, **arrays)
    manifest_path = candidate / "episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["streams"].append(
        {
            "stream_id": stream_id,
            "partition": "policy_visible",
            "modality": "proprioception",
            "media_type": "application/x-npz",
            "sample_count": 1,
            "timestamp_field": "sensor_time_ns",
            "path": f"streams/{stream_id}.npz",
            "sha256": sha256_file(payload_path),
        }
    )
    _write_json(manifest_path, manifest)
    receipt_path = candidate / "formal_capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["episode_manifest_sha256"] = sha256_file(manifest_path)
    _write_json(receipt_path, receipt)
    return payload_path


class ResearchProjectionTests(unittest.TestCase):
    def test_verified_npz_stream_projects_to_standard_zarr_and_readers_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = _candidate(root / "captures", episode_id="formal-episode-zarr", use_template_stream=True)
            payload_path = candidate / "streams" / "demo.npz"
            np.savez_compressed(
                payload_path,
                sensor_time_ns=np.asarray([10, 20, 30], dtype=np.int64),
                value=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            )
            manifest_path = candidate / "episode_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["streams"].append(
                {
                    "stream_id": "demo_npz",
                    "partition": "policy_visible",
                    "modality": "rgb",
                    "media_type": "application/x-npz",
                    "sample_count": 3,
                    "timestamp_field": "sensor_time_ns",
                    "path": "streams/demo.npz",
                    "sha256": sha256_file(payload_path),
                }
            )
            _write_json(manifest_path, manifest)
            receipt_path = candidate / "formal_capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["episode_manifest_sha256"] = sha256_file(manifest_path)
            _write_json(receipt_path, receipt)

            output = root / "zarr"
            result = project_episode_to_zarr(candidate, output, stream_ids=["demo_npz"])
            self.assertEqual(set(result.array_paths), {"demo_npz/sensor_time_ns", "demo_npz/value"})
            first = read_zarr_array(output, "demo_npz/value")
            second = read_zarr_array_independent(output, "demo_npz/value")
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first, np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
            receipt = json.loads((output / "projection_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema"], "org.rivermark.benchmark.zarr-projection.v1")
            self.assertEqual(receipt["episode_manifest_sha256"], sha256_file(manifest_path))
            record = next(item for item in receipt["arrays"] if item["path"] == "demo_npz/value")
            self.assertEqual(record["chunk_sha256"], record["chunk_sha256s"][0])

    def test_unsupported_stream_encoding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate, _ = _candidate(Path(temporary) / "captures", episode_id="formal-episode-zarr-json")
            with self.assertRaises(ProjectionError):
                project_episode_to_zarr(candidate, Path(temporary) / "zarr", stream_ids=["state"])

    def test_reserved_metadata_array_name_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate, _ = _candidate(Path(temporary) / "captures", episode_id="formal-episode-zarr-reserved")
            _append_npz_stream(candidate, "reserved", {".zgroup": np.asarray([1], dtype=np.int32)})
            with self.assertRaisesRegex(ProjectionError, "reserved Zarr path"):
                project_episode_to_zarr(candidate, Path(temporary) / "zarr", stream_ids=["reserved"])

    def test_empty_array_and_object_array_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty_candidate, _ = _candidate(root / "captures", episode_id="formal-episode-zarr-empty")
            _append_npz_stream(empty_candidate, "empty", {"values": np.empty((0,), dtype=np.float32)})
            with self.assertRaisesRegex(ProjectionError, "non-empty"):
                project_episode_to_zarr(empty_candidate, root / "empty-zarr", stream_ids=["empty"])

            object_candidate, _ = _candidate(root / "captures", episode_id="formal-episode-zarr-object")
            _append_npz_stream(object_candidate, "object", {"values": np.asarray([{"x": 1}], dtype=object)})
            with self.assertRaises(ProjectionError):
                project_episode_to_zarr(object_candidate, root / "object-zarr", stream_ids=["object"])

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = _candidate(root / "captures", episode_id="formal-episode-zarr-existing")
            payload_path = _append_npz_stream(candidate, "existing", {"values": np.asarray([1, 2], dtype=np.int32)})
            output = root / "zarr"
            project_episode_to_zarr(candidate, output, stream_ids=["existing"])
            before = (output / "projection_manifest.json").read_bytes()
            with self.assertRaises(ProjectionError):
                project_episode_to_zarr(candidate, output, stream_ids=["existing"])
            self.assertEqual(before, (output / "projection_manifest.json").read_bytes())
            self.assertTrue(payload_path.is_file())

    def test_bounded_first_axis_chunks_have_reader_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = _candidate(root / "captures", episode_id="formal-episode-zarr-chunked")
            payload_path = _append_npz_stream(
                candidate,
                "chunked",
                {"value": np.arange(24, dtype=np.float32).reshape(6, 4)},
            )
            output = root / "zarr"
            project_episode_to_zarr(
                candidate,
                output,
                stream_ids=["chunked"],
                max_chunk_bytes=32,
            )
            self.assertTrue(payload_path.is_file())
            metadata = json.loads((output / "chunked" / "value" / ".zarray").read_text(encoding="utf-8"))
            self.assertEqual(metadata["chunks"], [2, 4])
            self.assertEqual(len(list((output / "chunked" / "value").glob("[0-9]*"))), 3)
            first = read_zarr_array(output, "chunked/value")
            second = read_zarr_array_independent(output, "chunked/value")
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first, np.arange(24, dtype=np.float32).reshape(6, 4))
            manifest = json.loads((output / "projection_manifest.json").read_text(encoding="utf-8"))
            record = next(item for item in manifest["arrays"] if item["path"] == "chunked/value")
            self.assertEqual(record["chunks"], [2, 4])
            self.assertEqual(len(record["chunk_sha256s"]), 3)

    def test_npz_projection_streams_members_without_numpy_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = _candidate(root / "captures", episode_id="formal-episode-zarr-stream")
            _append_npz_stream(
                candidate,
                "streamed",
                {"value": np.arange(24, dtype=np.float32).reshape(6, 4)},
            )
            output = root / "zarr"
            with patch("rivermark_benchmark.research_projection.np.load", side_effect=AssertionError("projection must stream NPY members")):
                project_episode_to_zarr(
                    candidate,
                    output,
                    stream_ids=["streamed"],
                    max_chunk_bytes=32,
                )
            np.testing.assert_array_equal(
                read_zarr_array(output, "streamed/value"),
                np.arange(24, dtype=np.float32).reshape(6, 4),
            )
            self.assertEqual(list(output.rglob("*.npy")), [])

    def test_chunk_budget_rejects_a_row_larger_than_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = _candidate(root / "captures", episode_id="formal-episode-zarr-budget")
            _append_npz_stream(candidate, "budget", {"value": np.zeros((2, 4), dtype=np.float32)})
            with self.assertRaisesRegex(ProjectionError, "exceeding max_chunk_bytes"):
                project_episode_to_zarr(
                    candidate,
                    root / "zarr",
                    stream_ids=["budget"],
                    max_chunk_bytes=8,
                )

    def test_source_member_budget_fails_before_projection_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = _candidate(root / "captures", episode_id="formal-episode-zarr-source-budget")
            _append_npz_stream(
                candidate,
                "source_budget",
                {"value": np.arange(24, dtype=np.float32).reshape(6, 4)},
            )
            output = root / "zarr"
            with self.assertRaisesRegex(ProjectionError, "max_source_member_bytes"):
                project_episode_to_zarr(
                    candidate,
                    output,
                    stream_ids=["source_budget"],
                    max_chunk_bytes=32,
                    max_source_member_bytes=1,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.rglob("*.source-*")), [])

    @unittest.skipUnless(importlib.util.find_spec("zarr"), "optional zarr dependency is not installed")
    def test_external_zarr_reader_matches_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate, _ = _candidate(root / "captures", episode_id="formal-episode-zarr-external")
            _append_npz_stream(candidate, "external", {"value": np.arange(12, dtype=np.int16).reshape(3, 4)})
            output = root / "zarr"
            project_episode_to_zarr(
                candidate,
                output,
                stream_ids=["external"],
                max_chunk_bytes=16,
            )
            np.testing.assert_array_equal(
                read_zarr_array_external(output, "external/value"),
                np.arange(12, dtype=np.int16).reshape(3, 4),
            )


if __name__ == "__main__":
    unittest.main()
