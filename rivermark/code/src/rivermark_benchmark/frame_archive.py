"""Bounded-memory frame storage for large Isaac sensor streams.

Each frame is a separate compressed member of an NPZ-compatible ZIP archive.
That keeps capture, validation, and video encoding from materializing an
entire RGB-D sequence in private process memory.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_FORMAT_FIELD = "__rivermark_chunked_frame_archive_v1__"
_FRAME_COUNT_FIELD = "__rivermark_frame_count__"
_FRAME_SEPARATOR = "__frame__"
_RESERVED_FIELDS = {_FORMAT_FIELD, _FRAME_COUNT_FIELD}
LEGACY_FRAME_MEMBER_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_STORED_FRAME_MEMBER_MIN_BYTES = 1 * 1024 * 1024
_INITIAL_SPOOL_CAPACITY = 8


class FrameArchiveError(ValueError):
    """Raised when a chunked frame archive is malformed."""


def _member_name(field: str) -> str:
    return f"{field}.npy"


def _frame_member_name(field: str, frame_index: int) -> str:
    return f"{field}{_FRAME_SEPARATOR}{frame_index:06d}.npy"


def _write_array_member(archive: zipfile.ZipFile, name: str, value: np.ndarray) -> None:
    array = np.asanyarray(value)
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    # Deflating multi-megabyte RGB-D frames spends several minutes of CPU and
    # creates a long-lived finalization window.  Stored members are still valid
    # NPZ entries and keep the writer strictly one-frame-at-a-time; reserve
    # compression for small metadata where it is cheap and useful.
    info.compress_type = (
        zipfile.ZIP_STORED
        if array.nbytes >= _STORED_FRAME_MEMBER_MIN_BYTES
        else zipfile.ZIP_DEFLATED
    )
    info.external_attr = 0o600 << 16
    with archive.open(info, mode="w", force_zip64=True) as member:
        # ``write_array`` uses bounded buffered writes for a ZipExtFile. In
        # particular it does not concatenate a memmap-backed frame sequence.
        np.lib.format.write_array(member, array, allow_pickle=False)


def write_chunked_frame_archive(
    path: Path,
    *,
    timestamps_ns: np.ndarray,
    inline_fields: Mapping[str, np.ndarray],
    frame_fields: Mapping[str, np.ndarray],
) -> None:
    """Write a fail-closed, one-member-per-frame NPZ archive atomically.

    ``inline_fields`` are small arrays such as timestamps and camera poses.
    Every field in ``frame_fields`` must have the timestamp count on axis 0;
    each frame is read and compressed independently.
    """

    timestamps = np.asarray(timestamps_ns)
    if timestamps.dtype != np.int64 or timestamps.ndim != 1 or len(timestamps) <= 0:
        raise FrameArchiveError("timestamps_ns must be a non-empty int64 [T] array")
    if len(np.unique(timestamps)) != len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise FrameArchiveError("timestamps_ns must be strictly increasing")
    field_names = set(inline_fields) | set(frame_fields)
    if not field_names or field_names & _RESERVED_FIELDS:
        raise FrameArchiveError("frame archive fields are missing or reserved")
    if "timestamps_ns" in field_names:
        raise FrameArchiveError("timestamps_ns is written by the archive writer")
    for name, value in inline_fields.items():
        array = np.asanyarray(value)
        if array.dtype.hasobject or array.ndim < 1 or array.shape[0] != len(timestamps):
            raise FrameArchiveError(f"inline field {name!r} must have leading frame axis {len(timestamps)}")
    for name, value in frame_fields.items():
        array = np.asanyarray(value)
        if array.dtype.hasobject or array.ndim < 2 or array.shape[0] != len(timestamps):
            raise FrameArchiveError(f"frame field {name!r} must have leading frame axis {len(timestamps)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".partial", dir=path.parent
    )
    # ZipFile reopens the staged path itself; retaining this descriptor would
    # fail on Windows and leak one handle per capture artifact.
    import os

    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            _write_array_member(archive, _member_name(_FORMAT_FIELD), np.asarray([1], dtype=np.uint8))
            _write_array_member(archive, _member_name(_FRAME_COUNT_FIELD), np.asarray([len(timestamps)], dtype=np.int64))
            _write_array_member(archive, _member_name("timestamps_ns"), timestamps)
            for name in sorted(inline_fields):
                _write_array_member(archive, _member_name(name), np.asanyarray(inline_fields[name]))
            for name in sorted(frame_fields):
                values = np.asanyarray(frame_fields[name])
                for frame_index in range(len(timestamps)):
                    _write_array_member(archive, _frame_member_name(name, frame_index), values[frame_index])
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def is_chunked_frame_archive(path: Path) -> bool:
    """Return whether ``path`` is a Rivermark v1 chunked frame archive."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            return _FORMAT_FIELD in archive.files and int(archive[_FORMAT_FIELD][0]) == 1
    except (OSError, ValueError, EOFError, IndexError):
        return False


def oversized_legacy_frame_members(path: Path, fields: tuple[str, ...]) -> dict[str, int]:
    """Return legacy NPZ members that would violate the bounded-memory gate."""

    if is_chunked_frame_archive(path):
        return {}
    try:
        with zipfile.ZipFile(path) as archive:
            oversized: dict[str, int] = {}
            for field in fields:
                try:
                    info = archive.getinfo(_member_name(field))
                except KeyError:
                    continue
                if info.file_size > LEGACY_FRAME_MEMBER_MAX_UNCOMPRESSED_BYTES:
                    oversized[field] = info.file_size
            return oversized
    except (OSError, zipfile.BadZipFile):
        return {}


@dataclass(frozen=True)
class FrameDescriptor:
    dtype: np.dtype[Any]
    shape: tuple[int, ...]


class ChunkedFrameArchive:
    """Read a v1 archive one frame at a time without a sequence-sized copy."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._archive = np.load(self.path, allow_pickle=False)
        try:
            if _FORMAT_FIELD not in self._archive.files or int(self._archive[_FORMAT_FIELD][0]) != 1:
                raise FrameArchiveError(f"not a Rivermark chunked frame archive: {self.path}")
            count = self._archive[_FRAME_COUNT_FIELD]
            if count.dtype != np.int64 or count.shape != (1,) or int(count[0]) <= 0:
                raise FrameArchiveError("chunked archive frame count is invalid")
            self.frame_count = int(count[0])
            self.timestamps_ns = self._archive["timestamps_ns"]
            if (
                self.timestamps_ns.dtype != np.int64
                or self.timestamps_ns.shape != (self.frame_count,)
                or np.any(np.diff(self.timestamps_ns) <= 0)
            ):
                raise FrameArchiveError("chunked archive timestamps are invalid")
            self._inline_fields: set[str] = set()
            self._frame_fields: set[str] = set()
            for name in self._archive.files:
                if name in _RESERVED_FIELDS or name == "timestamps_ns":
                    continue
                if _FRAME_SEPARATOR in name:
                    field, _, index_text = name.partition(_FRAME_SEPARATOR)
                    if not field or not index_text.isdecimal() or len(index_text) != 6:
                        raise FrameArchiveError(f"invalid chunked archive member {name!r}")
                    self._frame_fields.add(field)
                else:
                    self._inline_fields.add(name)
            if not self._frame_fields:
                raise FrameArchiveError("chunked archive has no per-frame fields")
            self._descriptors: dict[str, FrameDescriptor] = {}
            for field in self._frame_fields:
                expected = {_frame_member_name(field, index)[:-4] for index in range(self.frame_count)}
                actual = {
                    name
                    for name in self._archive.files
                    if name.startswith(f"{field}{_FRAME_SEPARATOR}")
                }
                if actual != expected:
                    raise FrameArchiveError(f"chunked archive field {field!r} has missing or unexpected frames")
                first = self._archive[_frame_member_name(field, 0)[:-4]]
                if first.dtype.hasobject or first.ndim < 1:
                    raise FrameArchiveError(f"chunked archive frame field {field!r} is invalid")
                self._descriptors[field] = FrameDescriptor(first.dtype, (self.frame_count, *first.shape))
        except BaseException:
            self._archive.close()
            raise

    @property
    def fields(self) -> set[str]:
        return {"timestamps_ns", *self._inline_fields, *self._frame_fields}

    @property
    def frame_fields(self) -> set[str]:
        return set(self._frame_fields)

    def descriptor(self, field: str) -> FrameDescriptor:
        if field in self._descriptors:
            return self._descriptors[field]
        if field not in self._inline_fields and field != "timestamps_ns":
            raise FrameArchiveError(f"unknown archive field {field!r}")
        value = self._archive[field]
        return FrameDescriptor(value.dtype, tuple(value.shape))

    def array(self, field: str) -> np.ndarray:
        if field in self._frame_fields:
            raise FrameArchiveError(f"per-frame field {field!r} must be read with frame()")
        if field not in self._inline_fields and field != "timestamps_ns":
            raise FrameArchiveError(f"unknown archive field {field!r}")
        return self._archive[field]

    def frame(self, field: str, frame_index: int) -> np.ndarray:
        if field not in self._frame_fields:
            raise FrameArchiveError(f"unknown per-frame field {field!r}")
        if frame_index < 0 or frame_index >= self.frame_count:
            raise IndexError(frame_index)
        value = self._archive[_frame_member_name(field, frame_index)[:-4]]
        descriptor = self._descriptors[field]
        if value.dtype != descriptor.dtype or value.shape != descriptor.shape[1:]:
            raise FrameArchiveError(f"frame {frame_index} of {field!r} disagrees with frame 0")
        return value

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> "ChunkedFrameArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class FrameSpool:
    """Dynamically growing disk-backed sample storage for one Isaac capture.

    Isaac RGB-D frames are large enough that preallocating the declared capture
    length can consume tens of GiB before the first route-witness checkpoint.
    Start with a small mapping and grow only when a frame arrives.  The spool
    still enforces the declared upper bound, while keeping a short smoke or a
    fail-closed capture from reserving the full run on disk.
    """

    def __init__(self, root: Path, *, frame_capacity: int):
        if frame_capacity <= 0:
            raise ValueError("frame_capacity must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.frame_capacity = frame_capacity
        self._capacity = min(frame_capacity, _INITIAL_SPOOL_CAPACITY)
        self.frame_count = 0
        self._arrays: dict[str, np.memmap] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._dtypes: dict[str, np.dtype[Any]] = {}
        self._timestamps: np.memmap | None = np.lib.format.open_memmap(
            self.root / "timestamps_ns.npy", mode="w+", dtype=np.int64, shape=(self._capacity,)
        )

    @staticmethod
    def _close_mapping(value: np.memmap) -> None:
        value.flush()
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()

    def _grow(self) -> None:
        if self._capacity >= self.frame_capacity:
            raise FrameArchiveError("sensor spool exceeds its declared frame capacity")
        new_capacity = min(self.frame_capacity, max(self._capacity * 2, self._capacity + 1))

        mappings: list[tuple[str, np.memmap]] = []
        if self._timestamps is not None:
            mappings.append(("timestamps_ns", self._timestamps))
        mappings.extend(self._arrays.items())

        # Disk-full faults used to promote timestamp/RGB growth before a later
        # field failed.  Stage every replacement first so an allocation or copy
        # fault leaves the previous complete multi-modal mapping untouched.
        staged: list[tuple[str, np.memmap, Path, Path]] = []
        try:
            for name, old in mappings:
                path = self.root / f"{name}.npy"
                temporary = self.root / f".{name}.grow.npy"
                temporary.unlink(missing_ok=True)
                staged.append((name, old, path, temporary))
                new = np.lib.format.open_memmap(
                    temporary,
                    mode="w+",
                    dtype=old.dtype,
                    shape=(new_capacity, *old.shape[1:]),
                )
                try:
                    if self.frame_count:
                        new[: self.frame_count] = old[: self.frame_count]
                    new.flush()
                finally:
                    self._close_mapping(new)
        except BaseException:
            for _, _, _, temporary in staged:
                temporary.unlink(missing_ok=True)
            raise

        backups: list[tuple[str, Path, Path, np.memmap]] = []
        reopened_new: list[tuple[str, np.memmap]] = []
        closed_names: set[str] = set()
        try:
            # Move every old mapping out of the way before promoting a new
            # generation.  If a later rename fails (for example an antivirus
            # handle races on Windows), the backups let us restore the exact
            # pre-growth directory instead of leaving closed mappings pointing
            # at a mixture of old and new files.
            for name, old, path, temporary in staged:
                self._close_mapping(old)
                closed_names.add(name)
                backup = self.root / f".{name}.rollback.npy"
                backup.unlink(missing_ok=True)
                path.replace(backup)
                backups.append((name, path, backup, old))
            for name, path, temporary, _old in (
                (name, path, temporary, old) for name, old, path, temporary in staged
            ):
                temporary.replace(path)
            for name, old, path, _temporary in staged:
                reopened = np.lib.format.open_memmap(
                    path,
                    mode="r+",
                    dtype=old.dtype,
                    shape=(new_capacity, *old.shape[1:]),
                )
                reopened_new.append((name, reopened))
                if name == "timestamps_ns":
                    self._timestamps = reopened
                else:
                    self._arrays[name] = reopened
        except BaseException:
            for _name, mapping in reopened_new:
                self._close_mapping(mapping)
            for name, old, _path, _temporary in staged:
                if name not in closed_names:
                    self._close_mapping(old)
            # Remove any promoted new generation before restoring backups.
            for _name, path, _backup, _old in backups:
                if path.is_file():
                    path.unlink()
            for _name, path, backup, _old in reversed(backups):
                if backup.is_file():
                    backup.replace(path)
            for _name, _old, path, _temporary in staged:
                _temporary.unlink(missing_ok=True)
                restored = np.lib.format.open_memmap(
                    path,
                    mode="r+",
                    dtype=_old.dtype,
                    shape=_old.shape,
                )
                if _name == "timestamps_ns":
                    self._timestamps = restored
                else:
                    self._arrays[_name] = restored
            raise
        finally:
            # Successful growth no longer needs rollback copies.  Cleanup is
            # deliberately outside the transaction: a transient unlink race
            # must not make a valid, fully promoted generation roll back.
            if not reopened_new:
                # The exception path restores the old files below; leave any
                # rollback copy that could not be removed for diagnosis.
                pass
            else:
                for _name, _path, backup, _old in backups:
                    backup.unlink(missing_ok=True)
        self._capacity = new_capacity

    def append(self, timestamp_ns: int, fields: Mapping[str, Any]) -> None:
        if self.frame_count >= self.frame_capacity:
            raise FrameArchiveError("sensor spool exceeds its declared frame capacity")
        if self.frame_count >= self._capacity:
            self._grow()
        if not fields:
            raise FrameArchiveError("sensor spool frame has no fields")
        normalized = {name: np.ascontiguousarray(value) for name, value in fields.items()}
        if self._arrays and set(normalized) != set(self._arrays):
            raise FrameArchiveError("sensor spool fields changed after the first frame")
        for name, value in normalized.items():
            if value.dtype.hasobject or value.ndim < 1:
                raise FrameArchiveError(f"sensor spool field {name!r} is not a numeric array")
            if name not in self._arrays:
                self._shapes[name] = tuple(value.shape)
                self._dtypes[name] = value.dtype
                self._arrays[name] = np.lib.format.open_memmap(
                    self.root / f"{name}.npy",
                    mode="w+",
                    dtype=value.dtype,
                    shape=(self._capacity, *value.shape),
                )
            if value.dtype != self._dtypes[name] or value.shape != self._shapes[name]:
                raise FrameArchiveError(f"sensor spool field {name!r} changed dtype or shape")
            self._arrays[name][self.frame_count] = value
        self._timestamps[self.frame_count] = int(timestamp_ns)
        self.frame_count += 1

    def timestamps(self) -> np.ndarray:
        if self.frame_count <= 0 or self._timestamps is None:
            raise FrameArchiveError("sensor spool has no frames")
        return self._timestamps[: self.frame_count]

    def values(self, field: str) -> np.ndarray:
        if field not in self._arrays:
            raise FrameArchiveError(f"unknown sensor spool field {field!r}")
        return self._arrays[field][: self.frame_count]

    def discard_fields_after_archive(self, fields: tuple[str, ...]) -> None:
        """Release spool fields only after their replacement archive is durable.

        A capture finalizer calls this only after ``write_chunked_frame_archive``
        returns successfully.  Keeping unrelated fields makes a subsequent
        finalization failure inspectable without retaining a second full copy of
        already archived RGB-D or overview frames.
        """

        for field in fields:
            try:
                value = self._arrays.pop(field)
            except KeyError as error:
                raise FrameArchiveError(f"unknown sensor spool field {field!r}") from error
            self._close_mapping(value)
            (self.root / f"{field}.npy").unlink(missing_ok=False)
            self._shapes.pop(field, None)
            self._dtypes.pop(field, None)

    def close(self) -> None:
        for value in self._arrays.values():
            self._close_mapping(value)
        self._arrays.clear()
        if self._timestamps is not None:
            self._close_mapping(self._timestamps)
            self._timestamps = None

    def discard_after_success(self) -> None:
        self.close()
        shutil.rmtree(self.root)
