"""Dependency-light Zarr v2 projection for verified Rivermark episodes.

The native capture format is deliberately retained as evidence.  This module
projects selected NPZ streams into ordinary Zarr v2 arrays so researchers can
open only the modalities they need with a standard chunked-array interface.
Projection is fail-closed: the source episode must already pass the formal
candidate/release verifier, and unsupported stream encodings are reported
instead of being silently omitted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .formal_dataset import (
    CandidateIntegrityReport,
    _verify_release_episode,
    sha256_file,
    verify_candidate_episode,
)
from .schema import is_safe_relative_path


ZARR_PROJECTION_SCHEMA = "org.rivermark.benchmark.zarr-projection.v1"
ZARR_FORMAT = 2
_SUPPORTED_DTYPE_KINDS = frozenset({"b", "i", "u", "f", "c", "S"})
_RESERVED_ZARR_PARTS = frozenset({".zgroup", ".zattrs", ".zarray", "projection_manifest.json"})
_NPY_COPY_BLOCK_BYTES = 8 * 1024 * 1024


class ProjectionError(ValueError):
    """Raised when a verified episode cannot be projected safely."""


@dataclass(frozen=True)
class ZarrProjectionResult:
    output_root: Path
    episode_id: str
    array_paths: tuple[str, ...]
    projection_receipt_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _safe_relative(value: object) -> str:
    if not is_safe_relative_path(value):
        raise ProjectionError(f"unsafe projection path: {value!r}")
    canonical = PurePosixPath(str(value).replace("\\", "/")).as_posix()
    if any(part in {"", ".", ".."} for part in PurePosixPath(canonical).parts):
        raise ProjectionError(f"unsafe projection path: {value!r}")
    return canonical


def _safe_zarr_path(value: object) -> str:
    """Validate an array path without allowing Zarr metadata collisions."""

    canonical = _safe_relative(value)
    parts = PurePosixPath(canonical).parts
    if any(part.startswith(".") or part in _RESERVED_ZARR_PARTS for part in parts):
        raise ProjectionError(f"reserved Zarr path: {value!r}")
    return canonical


def _contained_file(root: Path, relative: object) -> Path:
    canonical = _safe_relative(relative)
    candidate = (root / canonical).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ProjectionError(f"missing source stream: {canonical}")
    return candidate


def _dtype_json(dtype: np.dtype[Any]) -> str:
    if dtype.kind not in _SUPPORTED_DTYPE_KINDS:
        raise ProjectionError(f"unsupported NumPy dtype for Zarr projection: {dtype.str}")
    normalized = dtype.newbyteorder("<")
    return normalized.str


def _normalise_array(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value))
    if array.ndim == 0:
        array = array.reshape((1,))
    if not array.shape or any(dimension <= 0 for dimension in array.shape):
        raise ProjectionError("Zarr v2 projection requires non-empty arrays")
    _dtype_json(array.dtype)
    return array


def _chunk_key(value: int | Sequence[int]) -> str:
    """Return a Zarr v2 chunk key while retaining the old ndim helper form."""

    if isinstance(value, int):
        coordinates = (0,) * max(1, value)
    else:
        coordinates = tuple(value)
        if not coordinates:
            coordinates = (0,)
    return ".".join(str(int(coordinate)) for coordinate in coordinates)


def _chunk_shape(shape: Sequence[int], dtype: np.dtype[Any], max_chunk_bytes: int | None) -> tuple[int, ...]:
    """Choose a first-axis chunk bounded by the requested byte budget."""

    normalized_shape = tuple(int(dimension) for dimension in shape)
    if max_chunk_bytes is None:
        return normalized_shape
    if isinstance(max_chunk_bytes, bool) or not isinstance(max_chunk_bytes, int) or max_chunk_bytes <= 0:
        raise ProjectionError("max_chunk_bytes must be a positive integer")
    row_elements = math.prod(normalized_shape[1:]) if len(normalized_shape) > 1 else 1
    row_bytes = row_elements * max(1, dtype.itemsize)
    if row_bytes > max_chunk_bytes:
        raise ProjectionError(
            f"array row requires {row_bytes} bytes, exceeding max_chunk_bytes={max_chunk_bytes}"
        )
    first_axis = max(1, min(normalized_shape[0], max_chunk_bytes // row_bytes))
    return (first_axis, *normalized_shape[1:])


def _write_array(
    root: Path,
    relative: str,
    value: np.ndarray,
    *,
    max_chunk_bytes: int | None = None,
) -> tuple[str, ...]:
    relative = _safe_zarr_path(relative)
    array = _normalise_array(value)
    array_root = root / relative
    array_root.mkdir(parents=True, exist_ok=False)
    shape = list(array.shape)
    chunks = list(_chunk_shape(shape, array.dtype, max_chunk_bytes))
    metadata = {
        "zarr_format": ZARR_FORMAT,
        "shape": shape,
        "chunks": chunks,
        "dtype": _dtype_json(array.dtype),
        "compressor": None,
        "fill_value": None,
        "order": "C",
        "filters": None,
    }
    _write_json(array_root / ".zarray", metadata)
    chunk_paths: list[str] = []
    # The projection intentionally chunks along time/first axis.  This keeps
    # each write bounded while preserving the source array's remaining shape.
    for first in range(0, shape[0], chunks[0]):
        stop = min(shape[0], first + chunks[0])
        selection = (slice(first, stop),) + (slice(None),) * (array.ndim - 1)
        coordinates = (first // chunks[0],) + (0,) * (array.ndim - 1)
        chunk_name = _chunk_key(coordinates)
        chunk_path = array_root / chunk_name
        source_chunk = np.ascontiguousarray(array[selection]).astype(array.dtype.newbyteorder("<"), copy=False)
        # Zarr v2 stores edge chunks at the declared chunk shape; pad the
        # unused tail so standard readers can decode the chunk unambiguously.
        chunk = np.zeros(tuple(chunks), dtype=array.dtype.newbyteorder("<"))
        actual_selection = (slice(0, stop - first),) + (slice(None),) * (array.ndim - 1)
        chunk[actual_selection] = source_chunk
        chunk_path.write_bytes(chunk.tobytes(order="C"))
        chunk_paths.append(f"{relative}/{chunk_name}")
    return tuple(chunk_paths)


def _load_source_manifest(episode_root: Path) -> tuple[dict[str, Any], str, str]:
    root = episode_root.resolve()
    admission_path = root / "admission.json"
    if admission_path.is_file():
        manifest, _, issues = _verify_release_episode(root)
        if issues or manifest is None:
            raise ProjectionError("release episode failed integrity verification: " + "; ".join(issue.code for issue in issues))
        manifest_path = root / "episode_manifest.json"
        return dict(manifest), sha256_file(manifest_path), sha256_file(root / "formal_capture_receipt.json")
    report: CandidateIntegrityReport = verify_candidate_episode(root, require_trusted_receipt=False)
    if not report.valid or report.manifest is None or report.manifest_sha256 is None or report.receipt_sha256 is None:
        raise ProjectionError("candidate failed integrity verification: " + "; ".join(issue.code for issue in report.issues))
    return dict(report.manifest), report.manifest_sha256, report.receipt_sha256


def _selected_streams(manifest: Mapping[str, Any], stream_ids: Iterable[str] | None) -> list[Mapping[str, Any]]:
    streams = manifest.get("streams")
    if not isinstance(streams, list):
        raise ProjectionError("episode manifest has no stream list")
    requested: set[str] | None = None
    if stream_ids is not None:
        requested = set()
        for stream_id in stream_ids:
            if not isinstance(stream_id, str) or not stream_id:
                raise ProjectionError("requested stream ids must be non-empty strings")
            requested.add(stream_id)
    selected: list[Mapping[str, Any]] = []
    for index, raw in enumerate(streams):
        if not isinstance(raw, Mapping):
            raise ProjectionError(f"malformed stream at index {index}")
        stream_id = raw.get("stream_id")
        if not isinstance(stream_id, str):
            raise ProjectionError(f"stream {index} has no stable stream_id")
        if requested is None or stream_id in requested:
            selected.append(raw)
    if requested is not None:
        found = {str(stream.get("stream_id")) for stream in selected}
        missing = sorted(requested - found)
        if missing:
            raise ProjectionError(f"requested streams are absent: {missing}")
    if not selected:
        raise ProjectionError("no streams selected for projection")
    return selected


def _iter_array_outputs(
    episode_root: Path,
    stream: Mapping[str, Any],
    *,
    temporary_root: Path | None = None,
    max_source_member_bytes: int | None = None,
) -> Iterable[tuple[str, np.ndarray]]:
    stream_id = stream.get("stream_id")
    path = stream.get("path")
    if not isinstance(stream_id, str) or not isinstance(path, str):
        raise ProjectionError(f"stream {stream_id!r} must use a concrete path for Zarr projection")
    if "path_template" in stream:
        raise ProjectionError(f"stream {stream_id!r} uses a path template; select concrete per-agent shards first")
    source = _contained_file(episode_root, path)
    if source.suffix.lower() != ".npz":
        raise ProjectionError(f"stream {stream_id!r} uses unsupported encoding {source.suffix or '<none>'}; Zarr projection currently accepts NPZ only")
    if (
        max_source_member_bytes is not None
        and (isinstance(max_source_member_bytes, bool) or not isinstance(max_source_member_bytes, int) or max_source_member_bytes <= 0)
    ):
        raise ProjectionError("max_source_member_bytes must be a positive integer")
    temporary_directory = tempfile.TemporaryDirectory(
        prefix=f".{source.stem}.source-",
        dir=str(temporary_root) if temporary_root is not None else None,
    )
    emitted = False
    try:
        try:
            archive = zipfile.ZipFile(source, mode="r")
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise ProjectionError(f"cannot decode NPZ stream {stream_id!r}: {exc}") from exc
        with archive:
            members = [info for info in archive.infolist() if info.filename.endswith(".npy")]
            for info in members:
                name = info.filename[:-4]
                if not name:
                    raise ProjectionError(f"stream {stream_id!r} contains an empty array name")
                if max_source_member_bytes is not None and info.file_size > max_source_member_bytes:
                    raise ProjectionError(
                        f"stream {stream_id!r} array {name!r} requires {info.file_size} temporary bytes, "
                        f"exceeding max_source_member_bytes={max_source_member_bytes}"
                    )
                temporary_path = Path(temporary_directory.name) / f"{len(name)}-{len(info.filename)}.npy"
                projected: np.ndarray | np.memmap | None = None
                array: np.ndarray | None = None
                try:
                    # NPZ compression is decoded directly to a temporary NPY
                    # file.  open_memmap then lets the writer read only one
                    # first-axis chunk, avoiding a sequence-sized heap copy.
                    with archive.open(info, mode="r") as member, temporary_path.open("wb") as output:
                        shutil.copyfileobj(member, output, length=_NPY_COPY_BLOCK_BYTES)
                    try:
                        projected = np.lib.format.open_memmap(
                            temporary_path,
                            mode="r",
                            max_header_size=10_000,
                        )
                    except (OSError, ValueError, EOFError) as exc:
                        raise ProjectionError(f"stream {stream_id!r} array {name!r} has an invalid NPY payload: {exc}") from exc
                    if not projected.flags.c_contiguous:
                        raise ProjectionError(f"stream {stream_id!r} array {name!r} uses unsupported Fortran order")
                    array = _normalise_array(projected)
                    output_name = _safe_zarr_path(f"{stream_id}/{name}")
                    emitted = True
                    yield output_name, array
                except (OSError, ValueError, TypeError, zipfile.BadZipFile, ProjectionError) as exc:
                    if isinstance(exc, ProjectionError):
                        raise
                    raise ProjectionError(f"stream {stream_id!r} array {name!r}: {exc}") from exc
                finally:
                    # Drop the ndarray view before closing the mmap on Windows.
                    array = None
                    mapping = getattr(projected, "_mmap", None)
                    if mapping is not None:
                        mapping.close()
                    projected = None
                    temporary_path.unlink(missing_ok=True)
        if not emitted:
            raise ProjectionError(f"NPZ stream {stream_id!r} contains no arrays")
    finally:
        temporary_directory.cleanup()


def project_episode_to_zarr(
    episode_root: Path,
    output_root: Path,
    *,
    stream_ids: Sequence[str] | None = None,
    max_chunk_bytes: int | None = None,
    max_source_member_bytes: int | None = None,
) -> ZarrProjectionResult:
    """Project selected verified NPZ streams to a standard Zarr v2 directory.

    ``max_chunk_bytes`` bounds each emitted first-axis chunk.  ``None`` keeps
    the historical one-chunk layout for compatibility; callers projecting
    large streams should pass an explicit budget.  Compressed NPZ members are
    decoded to temporary NPY files and memory-mapped, so the source array is
    never materialized as one heap allocation.  Temporary disk usage is
    bounded by ``max_source_member_bytes`` when supplied and is cleaned on
    success or failure.
    """

    source_root = episode_root.resolve()
    destination = output_root.resolve()
    if destination.exists():
        raise ProjectionError(f"projection output already exists: {destination}")
    manifest, manifest_hash, receipt_hash = _load_source_manifest(source_root)
    episode_id = manifest.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ProjectionError("manifest has no episode_id")
    abi_ref = manifest.get("observation_abi")
    if not isinstance(abi_ref, Mapping) or not isinstance(abi_ref.get("sha256"), str):
        raise ProjectionError("manifest has no observation ABI binding")
    selected = _selected_streams(manifest, stream_ids)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{destination.name}.{uuid.uuid4().hex}.", dir=destination.parent))
    try:
        _write_json(staging / ".zgroup", {"zarr_format": ZARR_FORMAT})
        _write_json(staging / ".zattrs", {
            "schema": ZARR_PROJECTION_SCHEMA,
            "episode_id": episode_id,
            "episode_manifest_sha256": manifest_hash,
            "formal_capture_receipt_sha256": receipt_hash,
            "observation_abi_sha256": abi_ref["sha256"],
            "source_encoding": "npz",
            "array_reader_contract": "zarr_v2_chunked_little_endian",
        })
        array_records: list[dict[str, Any]] = []
        names: list[str] = []
        for stream in selected:
            for name, array in _iter_array_outputs(
                source_root,
                stream,
                temporary_root=staging,
                max_source_member_bytes=max_source_member_bytes,
            ):
                if name in names:
                    raise ProjectionError("selected streams produce duplicate Zarr array paths")
                chunk_paths = _write_array(staging, name, array, max_chunk_bytes=max_chunk_bytes)
                names.append(name)
                metadata = json.loads((staging / name / ".zarray").read_text(encoding="utf-8"))
                record = {
                    "path": name,
                    "shape": list(array.shape),
                    "chunks": metadata["chunks"],
                    "dtype": _dtype_json(array.dtype),
                    "zarray_sha256": sha256_file(staging / name / ".zarray"),
                    "chunk_sha256s": [sha256_file(staging / path) for path in chunk_paths],
                }
                if len(chunk_paths) == 1:
                    # Preserve the v1 manifest field for the historical
                    # single-chunk layout while exposing the multi-chunk list.
                    record["chunk_sha256"] = record["chunk_sha256s"][0]
                array_records.append(record)
                # Release the memmap-backed view before requesting the next
                # source member; this is required for deterministic cleanup on
                # Windows.
                array = None  # type: ignore[assignment]
        if not array_records:
            raise ProjectionError("selected streams produced no arrays")
        manifest_payload = {
            "schema": ZARR_PROJECTION_SCHEMA,
            "zarr_format": ZARR_FORMAT,
            "episode_id": episode_id,
            "episode_manifest_sha256": manifest_hash,
            "formal_capture_receipt_sha256": receipt_hash,
            "observation_abi_sha256": abi_ref["sha256"],
            "arrays": array_records,
        }
        _write_json(staging / "projection_manifest.json", manifest_payload)
        os.replace(staging, destination)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    receipt_path = destination / "projection_manifest.json"
    return ZarrProjectionResult(destination, episode_id, tuple(names), sha256_file(receipt_path))


def _read_metadata(root: Path, relative: str) -> Mapping[str, Any]:
    relative = _safe_zarr_path(relative)
    array_root = (root / relative).resolve()
    if not array_root.is_relative_to(root.resolve()) or not array_root.is_dir():
        raise ProjectionError(f"missing Zarr array: {relative}")
    path = array_root / ".zarray"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectionError(f"invalid Zarr metadata for {relative}: {exc}") from exc
    if not isinstance(metadata, Mapping) or metadata.get("zarr_format") != ZARR_FORMAT:
        raise ProjectionError(f"unsupported Zarr metadata for {relative}")
    return metadata


def _array_layout(metadata: Mapping[str, Any], relative: str) -> tuple[tuple[int, ...], tuple[int, ...], np.dtype[Any]]:
    shape_raw = metadata.get("shape")
    chunks_raw = metadata.get("chunks")
    dtype_raw = metadata.get("dtype")
    if (
        not isinstance(shape_raw, list)
        or not isinstance(chunks_raw, list)
        or len(shape_raw) != len(chunks_raw)
        or not isinstance(dtype_raw, str)
        or not shape_raw
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape_raw + chunks_raw)
        or any(chunk > dimension for chunk, dimension in zip(chunks_raw, shape_raw))
    ):
        raise ProjectionError(f"invalid Zarr array layout for {relative}")
    try:
        dtype = np.dtype(dtype_raw)
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"invalid Zarr dtype for {relative}: {dtype_raw!r}") from exc
    _dtype_json(dtype)
    return tuple(shape_raw), tuple(chunks_raw), dtype


def _read_chunked_array(root: Path, relative: str, metadata: Mapping[str, Any]) -> np.ndarray:
    shape, chunks, dtype = _array_layout(metadata, relative)
    array_root = (root / _safe_zarr_path(relative)).resolve()
    if not array_root.is_relative_to(root.resolve()):
        raise ProjectionError(f"unsafe Zarr array path: {relative}")
    output = np.empty(shape, dtype=dtype)
    ranges = [range((dimension + chunk - 1) // chunk) for dimension, chunk in zip(shape, chunks)]
    for coordinates in itertools.product(*ranges):
        offsets = tuple(coordinate * chunk for coordinate, chunk in zip(coordinates, chunks))
        actual_shape = tuple(min(chunk, dimension - offset) for dimension, chunk, offset in zip(shape, chunks, offsets))
        path = array_root / _chunk_key(coordinates)
        if not path.is_file():
            raise ProjectionError(f"missing Zarr chunk for {relative}: {_chunk_key(coordinates)}")
        payload = path.read_bytes()
        expected = math.prod(chunks) * dtype.itemsize
        if len(payload) != expected:
            raise ProjectionError(f"Zarr chunk size mismatch for {relative}: {_chunk_key(coordinates)}")
        chunk = np.frombuffer(payload, dtype=dtype, count=math.prod(chunks)).reshape(chunks)
        selection = tuple(slice(offset, offset + size) for offset, size in zip(offsets, actual_shape))
        output[selection] = chunk[tuple(slice(0, size) for size in actual_shape)]
    return output


def read_zarr_array(root: Path, relative: str) -> np.ndarray:
    """Read one Zarr v2 array with strict metadata and chunk checks."""

    base = root.resolve()
    if not base.is_dir():
        raise ProjectionError(f"missing Zarr root: {root}")
    metadata = _read_metadata(base, relative)
    if metadata.get("compressor") is not None or metadata.get("filters") is not None:
        raise ProjectionError(f"compressed Zarr arrays are outside the projection contract: {relative}")
    return _read_chunked_array(base, relative, metadata)


def read_zarr_array_independent(root: Path, relative: str) -> np.ndarray:
    """Independent parity reader for the restricted Zarr v2 projection."""

    base = root.resolve()
    if not base.is_dir():
        raise ProjectionError(f"missing Zarr root: {root}")
    relative = _safe_zarr_path(relative)
    metadata_path = (base / relative / ".zarray").resolve()
    if not metadata_path.is_relative_to(base) or not metadata_path.is_file():
        raise ProjectionError(f"missing Zarr metadata for {relative}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectionError(f"invalid Zarr metadata for {relative}: {exc}") from exc
    if not isinstance(metadata, Mapping) or metadata.get("zarr_format") != 2 or metadata.get("compressor") is not None or metadata.get("filters") is not None:
        raise ProjectionError(f"unsupported Zarr features for {relative}")
    shape_raw = metadata.get("shape")
    chunks_raw = metadata.get("chunks")
    dtype_value = metadata.get("dtype")
    if (
        not isinstance(shape_raw, list)
        or not isinstance(chunks_raw, list)
        or len(shape_raw) != len(chunks_raw)
        or not isinstance(dtype_value, str)
        or not shape_raw
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape_raw + chunks_raw)
        or any(chunk > dimension for chunk, dimension in zip(chunks_raw, shape_raw))
    ):
        raise ProjectionError(f"invalid Zarr layout for {relative}")
    try:
        dtype = np.dtype(dtype_value)
        _dtype_json(dtype)
    except (TypeError, ValueError, ProjectionError) as exc:
        raise ProjectionError(f"invalid Zarr dtype for {relative}: {dtype_value!r}") from exc
    shape = tuple(shape_raw)
    chunks = tuple(chunks_raw)
    array_root = (base / relative).resolve()
    if not array_root.is_relative_to(base) or not array_root.is_dir():
        raise ProjectionError(f"missing Zarr array for {relative}")
    output = np.empty(shape, dtype=dtype)
    grid = [range((dimension + chunk - 1) // chunk) for dimension, chunk in zip(shape, chunks)]
    for coordinates in itertools.product(*grid):
        offsets = tuple(coordinate * chunk for coordinate, chunk in zip(coordinates, chunks))
        extents = tuple(min(chunk, dimension - offset) for dimension, chunk, offset in zip(shape, chunks, offsets))
        data_path = array_root / _chunk_key(coordinates)
        if not data_path.is_file():
            raise ProjectionError(f"missing Zarr chunk for {relative}: {_chunk_key(coordinates)}")
        payload = data_path.read_bytes()
        expected = math.prod(chunks) * dtype.itemsize
        if len(payload) != expected:
            raise ProjectionError(f"invalid chunk length for {relative}: {_chunk_key(coordinates)}")
        chunk_array = np.frombuffer(payload, dtype=dtype, count=math.prod(chunks)).reshape(chunks)
        output[tuple(slice(offset, offset + extent) for offset, extent in zip(offsets, extents))] = chunk_array[
            tuple(slice(0, extent) for extent in extents)
        ]
    return output


def read_zarr_array_external(root: Path, relative: str) -> np.ndarray:
    """Read a projected array through the optional standard ``zarr`` package."""

    try:
        import zarr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ProjectionError("external Zarr parity requires the optional 'zarr' dependency") from exc
    array_root = (root.resolve() / _safe_zarr_path(relative)).resolve()
    if not array_root.is_relative_to(root.resolve()) or not array_root.is_dir():
        raise ProjectionError(f"missing Zarr array: {relative}")
    try:
        return np.asarray(zarr.open(str(array_root), mode="r"))
    except (OSError, ValueError, TypeError) as exc:
        raise ProjectionError(f"external Zarr reader failed for {relative}: {exc}") from exc


__all__ = [
    "ProjectionError",
    "ZarrProjectionResult",
    "ZARR_PROJECTION_SCHEMA",
    "project_episode_to_zarr",
    "read_zarr_array",
    "read_zarr_array_independent",
    "read_zarr_array_external",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--stream-id", action="append", default=[])
    parser.add_argument("--max-chunk-bytes", type=int, default=None)
    parser.add_argument("--max-source-member-bytes", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        result = project_episode_to_zarr(
            args.episode_root,
            args.output_root,
            stream_ids=args.stream_id or None,
            max_chunk_bytes=args.max_chunk_bytes,
            max_source_member_bytes=args.max_source_member_bytes,
        )
    except (OSError, ProjectionError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({
        "status": "projected",
        "output_root": str(result.output_root),
        "episode_id": result.episode_id,
        "array_paths": list(result.array_paths),
        "projection_manifest_sha256": result.projection_receipt_sha256,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
