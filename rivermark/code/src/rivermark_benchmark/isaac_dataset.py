"""Lazy access to native Isaac capture payloads.

The capture directory is an evidence bundle, not a Python object to load in
one shot.  This module reads JSON metadata eagerly, opens chunked frame
archives lazily, and yields one selected frame at a time.  It is intended for
research scripts that need a small window or a decimated modality without
copying a multi-gigabyte episode into RAM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .frame_archive import (
    ChunkedFrameArchive,
    FrameArchiveError,
    is_chunked_frame_archive,
    oversized_legacy_frame_members,
)


_CHUNKED_MODALITIES = {
    "overview": Path("sensors/overview_rgb.npz"),
    "onboard": Path("sensors/onboard_rgbd.npz"),
    "semantic": Path("learning_labels/semantic_segmentation.npz"),
}
_ARRAY_MODALITIES = {
    "camera_poses": Path("sensors/camera_poses.npz"),
    "lidar": Path("sensors/lidar.npz"),
    "imu": Path("sensors/imu.npz"),
    "contact": Path("sensors/contact.npz"),
    "state_action": Path("streams/state_action.npz"),
    "public_task": Path("streams/public_task.npz"),
    "public_messages": Path("streams/public_messages.npz"),
}
_CHUNKED_FALLBACK_FIELDS = {
    "overview": ("rgb", "distance_to_image_plane_m", "semantic_segmentation"),
    "onboard": ("rgb", "distance_to_image_plane_m"),
    "semantic": ("semantic_segmentation",),
}


@dataclass(frozen=True)
class FrameRecord:
    """One timestamped sensor frame returned by :meth:`IsaacCapture.iter_frames`."""

    index: int
    timestamp_ns: int
    values: Mapping[str, np.ndarray]


class IsaacCapture:
    """Metadata-checked, lazy reader for a native Isaac capture directory.

    By default a capture must have a successful capture receipt and an
    independent validator receipt.  ``require_validated=False`` is useful for
    diagnosing a failed development run, but must not be used for benchmark
    training or publication.
    """

    def __init__(
        self,
        root: Path,
        *,
        require_validated: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"capture directory is missing: {self.root}")
        self.receipt = self._read_json("capture_receipt.json")
        if require_validated:
            if self.receipt.get("status") != "captured" or self.receipt.get("ok") is not True:
                raise ValueError("capture receipt is not a successful Isaac capture")
            validation_path = self.root / "independent_validation.json"
            if not validation_path.is_file():
                raise FileNotFoundError(
                    "independent_validation.json is required for validated dataset access"
                )
            self.validation = self._read_json("independent_validation.json")
            if self.validation.get("status") != "passed":
                raise ValueError("independent Isaac validation did not pass")
        else:
            validation_path = self.root / "independent_validation.json"
            self.validation = (
                self._read_json("independent_validation.json")
                if validation_path.is_file()
                else None
            )
        self._timestamps_cache: np.ndarray | None = None

    def _read_json(self, relative_path: str) -> dict[str, Any]:
        path = self.root / relative_path
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"capture metadata is missing: {path}") from None
        if not isinstance(value, dict):
            raise ValueError(f"capture metadata must be a JSON object: {path}")
        return value

    @property
    def modalities(self) -> tuple[str, ...]:
        """Return available payload names without opening large arrays."""

        names: list[str] = []
        for name, relative in (*_CHUNKED_MODALITIES.items(), *_ARRAY_MODALITIES.items()):
            if (self.root / relative).is_file():
                names.append(name)
        return tuple(names)

    @property
    def timestamps_ns(self) -> np.ndarray:
        """Return the small timestamp vector; sensor frames remain lazy."""

        if self._timestamps_cache is None:
            path = self._require_path("overview", _CHUNKED_MODALITIES)
            if is_chunked_frame_archive(path):
                with ChunkedFrameArchive(path) as archive:
                    self._timestamps_cache = np.asarray(archive.timestamps_ns, dtype=np.int64).copy()
            else:
                with np.load(path, allow_pickle=False) as payload:
                    if "timestamps_ns" not in payload.files:
                        raise FrameArchiveError(f"overview archive has no timestamps_ns: {path}")
                    self._timestamps_cache = np.asarray(payload["timestamps_ns"], dtype=np.int64).copy()
        return self._timestamps_cache

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def _require_path(
        self,
        modality: str,
        table: Mapping[str, Path] | None = None,
    ) -> Path:
        paths = table or {**_CHUNKED_MODALITIES, **_ARRAY_MODALITIES}
        try:
            relative = paths[modality]
        except KeyError as error:
            raise KeyError(f"unknown Isaac capture modality: {modality!r}") from error
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(f"payload for {modality!r} is missing: {path}")
        return path

    @staticmethod
    def _normalize_window(
        frame_count: int,
        start: int,
        stop: int | None,
        stride: int,
    ) -> range:
        if stride < 1:
            raise ValueError("stride must be positive")
        if start < 0 or start > frame_count:
            raise IndexError(start)
        end = frame_count if stop is None else stop
        if end < start or end > frame_count:
            raise IndexError(end)
        return range(start, end, stride)

    def _iter_chunked(
        self,
        path: Path,
        *,
        modality: str,
        fields: Sequence[str] | None,
        start: int,
        stop: int | None,
        stride: int,
    ) -> Iterator[FrameRecord]:
        if not is_chunked_frame_archive(path):
            yield from self._iter_array(
                path.stem,
                path,
                fields=fields or _CHUNKED_FALLBACK_FIELDS[modality],
                start=start,
                stop=stop,
                stride=stride,
            )
            return
        with ChunkedFrameArchive(path) as archive:
            selected = tuple(fields) if fields is not None else tuple(sorted(archive.frame_fields))
            unknown = set(selected) - archive.fields
            if unknown:
                raise KeyError(f"unknown fields in {path.name}: {sorted(unknown)}")
            for index in self._normalize_window(archive.frame_count, start, stop, stride):
                values: dict[str, np.ndarray] = {}
                for field in selected:
                    if field in archive.frame_fields:
                        values[field] = archive.frame(field, index)
                    else:
                        values[field] = archive.array(field)[index]
                yield FrameRecord(index, int(archive.timestamps_ns[index]), values)

    def _iter_array(
        self,
        modality: str,
        path: Path,
        *,
        fields: Sequence[str] | None,
        start: int,
        stop: int | None,
        stride: int,
    ) -> Iterator[FrameRecord]:
        oversized = oversized_legacy_frame_members(path, tuple(fields or ()))
        if oversized:
            raise FrameArchiveError(
                f"legacy payload {path} exceeds bounded lazy-read limits: {sorted(oversized)}; "
                "re-capture with the chunked Isaac archive format"
            )
        with np.load(path, allow_pickle=False) as payload:
            if "timestamps_ns" not in payload.files:
                raise FrameArchiveError(f"payload has no timestamps_ns: {path}")
            timestamps = payload["timestamps_ns"]
            selected = tuple(fields) if fields is not None else tuple(
                name for name in payload.files if name != "timestamps_ns"
            )
            unknown = set(selected) - set(payload.files)
            if unknown:
                raise KeyError(f"unknown fields in {path.name}: {sorted(unknown)}")
            for index in self._normalize_window(int(timestamps.shape[0]), start, stop, stride):
                yield FrameRecord(
                    index,
                    int(timestamps[index]),
                    {field: payload[field][index] for field in selected},
                )

    def iter_frames(
        self,
        modality: str,
        *,
        fields: Sequence[str] | None = None,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
    ) -> Iterator[FrameRecord]:
        """Yield selected frames without materializing a sequence-sized copy.

        ``fields`` defaults to frame fields for chunked camera archives and to
        all non-timestamp arrays for small legacy NPZ payloads.  The yielded
        arrays are valid until the next iteration; callers that retain them
        should explicitly copy only those frames they need.
        """

        if modality in _CHUNKED_MODALITIES:
            yield from self._iter_chunked(
                self._require_path(modality, _CHUNKED_MODALITIES),
                modality=modality,
                fields=fields,
                start=start,
                stop=stop,
                stride=stride,
            )
            return
        if modality in _ARRAY_MODALITIES:
            yield from self._iter_array(
                modality,
                self._require_path(modality, _ARRAY_MODALITIES),
                fields=fields,
                start=start,
                stop=stop,
                stride=stride,
            )
            return
        raise KeyError(f"unknown Isaac capture modality: {modality!r}")

    def read_frame(
        self,
        modality: str,
        frame_index: int,
        *,
        fields: Sequence[str] | None = None,
    ) -> FrameRecord:
        """Read exactly one frame and close its archive immediately."""

        try:
            return next(
                self.iter_frames(
                    modality,
                    fields=fields,
                    start=frame_index,
                    stop=frame_index + 1,
                )
            )
        except StopIteration as error:
            raise IndexError(frame_index) from error


__all__ = ["FrameRecord", "IsaacCapture"]
