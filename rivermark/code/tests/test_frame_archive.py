from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.frame_archive import (
    ChunkedFrameArchive,
    FrameArchiveError,
    FrameSpool,
    is_chunked_frame_archive,
    oversized_legacy_frame_members,
    write_chunked_frame_archive,
)


class ChunkedFrameArchiveTests(unittest.TestCase):
    def _timestamps(self) -> np.ndarray:
        return np.array([10, 20, 30], dtype=np.int64)

    def _frames(self) -> np.ndarray:
        return np.arange(3 * 2 * 4 * 3, dtype=np.uint8).reshape(3, 2, 4, 3)

    def test_round_trip_reads_one_frame_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frames.npz"
            timestamps = self._timestamps()
            frames = self._frames()
            pose = np.arange(9, dtype=np.float32).reshape(3, 3)
            write_chunked_frame_archive(
                path,
                timestamps_ns=timestamps,
                inline_fields={"camera_pos_w_m": pose},
                frame_fields={"rgb": frames},
            )
            self.assertTrue(is_chunked_frame_archive(path))
            with np.load(path, allow_pickle=False) as raw:
                self.assertNotIn("rgb", raw.files)
                self.assertEqual(sum(name.startswith("rgb__frame__") for name in raw.files), len(timestamps))
            with ChunkedFrameArchive(path) as archive:
                self.assertEqual(archive.fields, {"timestamps_ns", "camera_pos_w_m", "rgb"})
                self.assertEqual(archive.descriptor("rgb").shape, frames.shape)
                np.testing.assert_array_equal(archive.timestamps_ns, timestamps)
                np.testing.assert_array_equal(archive.array("camera_pos_w_m"), pose)
                for index in range(len(timestamps)):
                    np.testing.assert_array_equal(archive.frame("rgb", index), frames[index])

    def test_missing_frame_member_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frames.npz"
            write_chunked_frame_archive(
                path,
                timestamps_ns=self._timestamps(),
                inline_fields={},
                frame_fields={"rgb": self._frames()},
            )
            replacement = Path(temporary) / "replacement.npz"
            with zipfile.ZipFile(path) as source, zipfile.ZipFile(replacement, "w") as destination:
                for member in source.infolist():
                    if member.filename == "rgb__frame__000001.npy":
                        continue
                    destination.writestr(member, source.read(member.filename))
            replacement.replace(path)
            with self.assertRaises(FrameArchiveError):
                ChunkedFrameArchive(path)

    def test_spool_grows_on_demand_and_releases_only_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "spool"
            spool = FrameSpool(root, frame_capacity=32)
            expected = []
            for index, timestamp in enumerate(self._timestamps()):
                value = np.full((2, 3), index, dtype=np.float32)
                expected.append(value)
                spool.append(int(timestamp), {"depth": value})
                if index == 0:
                    mapped = np.load(root / "depth.npy", mmap_mode="r", allow_pickle=False)
                    try:
                        self.assertEqual(mapped.shape[0], 8)
                    finally:
                        mapped._mmap.close()
            self.assertEqual(spool.frame_count, 3)
            self.assertTrue((root / "depth.npy").is_file())
            np.testing.assert_array_equal(spool.timestamps(), self._timestamps())
            np.testing.assert_array_equal(spool.values("depth"), np.stack(expected))
            spool.discard_after_success()
            self.assertFalse(root.exists())

    def test_spool_releases_only_archived_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "spool"
            spool = FrameSpool(root, frame_capacity=3)
            for index, timestamp in enumerate(self._timestamps()):
                spool.append(
                    int(timestamp),
                    {
                        "rgb": np.full((2, 3, 3), index, dtype=np.uint8),
                        "depth": np.full((2, 3, 1), index, dtype=np.float32),
                    },
                )
            archive = Path(temporary) / "rgb.npz"
            write_chunked_frame_archive(
                archive,
                timestamps_ns=spool.timestamps(),
                inline_fields={},
                frame_fields={"rgb": spool.values("rgb")},
            )
            spool.discard_fields_after_archive(("rgb",))
            self.assertFalse((root / "rgb.npy").exists())
            self.assertTrue((root / "depth.npy").exists())
            np.testing.assert_array_equal(spool.values("depth")[2], np.full((2, 3, 1), 2, dtype=np.float32))
            with self.assertRaises(FrameArchiveError):
                spool.values("rgb")
            with ChunkedFrameArchive(archive) as restored:
                np.testing.assert_array_equal(restored.frame("rgb", 2), np.full((2, 3, 3), 2, dtype=np.uint8))
            spool.discard_after_success()

    def test_spool_growth_failure_preserves_the_previous_complete_mapping(self) -> None:
        """A full-volume error while preparing one field must not promote another."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "spool"
            spool = FrameSpool(root, frame_capacity=16)
            for index in range(8):
                spool.append(
                    index,
                    {
                        "rgb": np.full((2, 3), index, dtype=np.uint8),
                        "depth": np.full((2, 3), index, dtype=np.float32),
                    },
                )

            original_open_memmap = np.lib.format.open_memmap

            def fail_only_depth_growth(filename: str | Path, *args: object, **kwargs: object) -> np.memmap:
                if Path(filename).name == ".depth.grow.npy":
                    raise OSError(28, "No space left on device")
                return original_open_memmap(filename, *args, **kwargs)

            with patch(
                "rivermark_benchmark.frame_archive.np.lib.format.open_memmap",
                side_effect=fail_only_depth_growth,
            ):
                with self.assertRaises(OSError):
                    spool.append(
                        8,
                        {
                            "rgb": np.full((2, 3), 8, dtype=np.uint8),
                            "depth": np.full((2, 3), 8, dtype=np.float32),
                        },
                    )

            self.assertEqual(spool.frame_count, 8)
            self.assertEqual(spool._capacity, 8)
            self.assertEqual(np.load(root / "timestamps_ns.npy", allow_pickle=False).shape, (8,))
            self.assertEqual(np.load(root / "rgb.npy", allow_pickle=False).shape, (8, 2, 3))
            self.assertEqual(np.load(root / "depth.npy", allow_pickle=False).shape, (8, 2, 3))
            self.assertEqual(tuple(spool.timestamps()), tuple(range(8)))
            np.testing.assert_array_equal(spool.values("rgb")[-1], np.full((2, 3), 7, dtype=np.uint8))
            np.testing.assert_array_equal(spool.values("depth")[-1], np.full((2, 3), 7, dtype=np.float32))
            self.assertEqual(tuple(root.glob(".*.grow.npy")), ())

            spool.append(
                8,
                {
                    "rgb": np.full((2, 3), 8, dtype=np.uint8),
                    "depth": np.full((2, 3), 8, dtype=np.float32),
                },
            )
            self.assertEqual(spool.frame_count, 9)
            spool.close()

    def test_spool_growth_promotion_failure_rolls_back_every_mapping(self) -> None:
        """A rename fault after one promoted field must be recoverable."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "spool"
            spool = FrameSpool(root, frame_capacity=16)
            for index in range(8):
                spool.append(
                    index,
                    {
                        "rgb": np.full((2, 3), index, dtype=np.uint8),
                        "depth": np.full((2, 3), index, dtype=np.float32),
                    },
                )

            original_replace = Path.replace

            def fail_depth_promotion(self: Path, target: str | Path) -> Path:
                if self.name == ".depth.grow.npy" and Path(target).name == "depth.npy":
                    raise OSError(28, "No space left on device")
                return original_replace(self, target)

            with patch("pathlib.Path.replace", new=fail_depth_promotion):
                with self.assertRaises(OSError):
                    spool.append(
                        8,
                        {
                            "rgb": np.full((2, 3), 8, dtype=np.uint8),
                            "depth": np.full((2, 3), 8, dtype=np.float32),
                        },
                    )

            self.assertEqual(spool.frame_count, 8)
            self.assertEqual(spool._capacity, 8)
            self.assertEqual(tuple(spool.timestamps()), tuple(range(8)))
            np.testing.assert_array_equal(spool.values("rgb")[-1], np.full((2, 3), 7, dtype=np.uint8))
            np.testing.assert_array_equal(spool.values("depth")[-1], np.full((2, 3), 7, dtype=np.float32))
            self.assertEqual(tuple(root.glob(".*.grow.npy")), ())
            self.assertEqual(tuple(root.glob(".*.rollback.npy")), ())

            spool.append(
                8,
                {
                    "rgb": np.full((2, 3), 8, dtype=np.uint8),
                    "depth": np.full((2, 3), 8, dtype=np.float32),
                },
            )
            self.assertEqual(spool.frame_count, 9)
            spool.close()

    def test_oversized_legacy_member_is_detected_without_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.npz"
            np.savez_compressed(path, rgb=np.zeros((1024, 1024, 3), dtype=np.uint8))
            with patch(
                "rivermark_benchmark.frame_archive.LEGACY_FRAME_MEMBER_MAX_UNCOMPRESSED_BYTES",
                1024,
            ):
                oversized = oversized_legacy_frame_members(path, ("rgb",))
            self.assertGreater(oversized["rgb"], 3 * 1024 * 1024)
