"""Bounded Parquet projection for validated, development-only Isaac captures.

The formal packer intentionally requires redistribution-cleared assets.  A
local Isaac capture can still be useful to researchers before that decision,
but it must not be relabelled as a formal episode.  This module therefore
projects only the small public state/action/task/message streams, binds them
to the raw capture and independent-validation hashes, and writes a manifest
whose development-only boundary is machine-readable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .formal_dataset import sha256_file
from .schema import is_safe_relative_path


DEVELOPMENT_PARQUET_SCHEMA = "org.rivermark.benchmark.development-parquet-projection.v1"
_CAPTURE_SCHEMA = "org.rivermark.isaac-swarm-capture.v1"
_VALIDATION_SCHEMAS = frozenset(
    {
        "org.rivermark.isaac-independent-validation.v1",
        "org.rivermark.isaac-state-only-transfer-independent-validation.v1",
    }
)
_STATE_PATH = "streams/state_action.npz"
_TASK_PATH = "streams/public_task.npz"
_MESSAGES_PATH = "streams/public_messages.npz"
_DEFAULT_MAX_SOURCE_MEMBER_BYTES = 64 * 1024 * 1024
_DEFAULT_ROW_GROUP_SIZE = 4096

_STATE_FIELDS = {
    "command_time_ns",
    "effective_time_ns",
    "root_pos_w_m",
    "root_quat_wxyz",
    "root_lin_vel_w_mps",
    "root_ang_vel_b_radps",
    "desired_pos_w_m",
    "desired_vel_w_mps",
    "target_thrust_n",
    "applied_thrust_n",
}
_TASK_FIELDS = {
    "timestamps_ns",
    "waypoint_index",
    "waypoint_progress",
    "desired_waypoint_w_m",
    "distance_to_waypoint_m",
    "waypoint_reached",
    "action_mode",
    "coverage_cell_id",
    "task_time_s",
}
_MESSAGE_FIELDS = {
    "timestamps_ns",
    "sender_agent_id",
    "message_sequence",
    "message_waypoint_index",
    "message_position_w_m",
    "message_velocity_w_mps",
    "message_flags",
}
_PUBLIC_BINDING_FIELDS = (
    "protocol_id",
    "protocol_sha256",
    "cell_id",
    "split",
    "episode_index",
    "episode_seed",
)


class ParquetProjectionError(ValueError):
    """Raised when a development projection cannot be proved safe."""


@dataclass(frozen=True)
class ParquetProjectionResult:
    output_root: Path
    capture_receipt_sha256: str
    independent_validation_sha256: str
    projection_manifest_sha256: str
    table_paths: tuple[str, ...]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_bytes(value))


def _arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ParquetProjectionError(
            "development Parquet projection requires the optional 'parquet' extra (pyarrow==25.0.0)"
        ) from exc
    return pa, pq


def _contained_file(root: Path, relative: str) -> Path:
    if not is_safe_relative_path(relative):
        raise ParquetProjectionError(f"unsafe source path: {relative!r}")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ParquetProjectionError(f"missing source stream: {relative}")
    return candidate


def _source_hash(receipt: Mapping[str, Any], root: Path, relative: str) -> dict[str, Any]:
    source = _contained_file(root, relative)
    expected = receipt.get("artifact_hashes", {}).get(relative)
    if not isinstance(expected, Mapping) or not isinstance(expected.get("sha256"), str):
        raise ParquetProjectionError(f"{relative} is not bound by capture artifact_hashes")
    actual = sha256_file(source)
    if actual != expected["sha256"] or int(expected.get("bytes", -1)) != source.stat().st_size:
        raise ParquetProjectionError(f"{relative} does not match its capture receipt binding")
    return {"path": relative, "bytes": source.stat().st_size, "sha256": actual}


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParquetProjectionError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ParquetProjectionError(f"{label} must be a JSON object")
    return value


def _public_collection_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the collection fields allowed in a public projection."""

    binding = receipt.get("collection_binding")
    if not isinstance(binding, Mapping):
        raise ParquetProjectionError("capture receipt has no collection binding")
    result: dict[str, Any] = {}
    for field in _PUBLIC_BINDING_FIELDS:
        value = binding.get(field)
        if field in {"episode_index", "episode_seed"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ParquetProjectionError(f"collection binding {field} must be an integer")
        elif not isinstance(value, str) or not value:
            raise ParquetProjectionError(f"collection binding {field} must be a non-empty string")
        result[field] = value
    return result


def _verify_capture_boundary(root: Path) -> tuple[dict[str, Any], str, str, dict[str, dict[str, Any]]]:
    receipt_path = root / "capture_receipt.json"
    validation_path = root / "independent_validation.json"
    receipt = dict(_load_json(receipt_path, "capture_receipt.json"))
    validation = _load_json(validation_path, "independent_validation.json")
    if receipt.get("schema") != _CAPTURE_SCHEMA or receipt.get("status") != "captured" or receipt.get("ok") is not True:
        raise ParquetProjectionError("development projection requires a successful native capture receipt")
    if receipt.get("source_worktree_dirty") is not False:
        raise ParquetProjectionError("development projection requires a clean capture source revision")
    boundary = receipt.get("claim_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("formal_benchmark_admission") is not False:
        raise ParquetProjectionError("development projection requires formal_benchmark_admission=false")
    if validation.get("schema") not in _VALIDATION_SCHEMAS or validation.get("status") != "passed" or validation.get("issues") != []:
        raise ParquetProjectionError("development projection requires a passing independent validation receipt")
    receipt_hash = sha256_file(receipt_path)
    if validation.get("capture_receipt_sha256") != receipt_hash:
        raise ParquetProjectionError("independent validation is not bound to this capture receipt")
    validation_hash = sha256_file(validation_path)
    paths = (_STATE_PATH, _TASK_PATH, _MESSAGES_PATH)
    artifacts = {relative: _source_hash(receipt, root, relative) for relative in paths}
    artifacts["capture_receipt.json"] = {"path": "capture_receipt.json", "bytes": receipt_path.stat().st_size, "sha256": receipt_hash}
    artifacts["independent_validation.json"] = {
        "path": "independent_validation.json",
        "bytes": validation_path.stat().st_size,
        "sha256": validation_hash,
    }
    return receipt, receipt_hash, validation_hash, artifacts


def _bounded_npz(path: Path, *, fields: set[str], max_source_member_bytes: int) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = {info.filename[:-4]: info for info in archive.infolist() if info.filename.endswith(".npy")}
            if set(members) != fields:
                raise ParquetProjectionError(
                    f"{path.name} fields differ from the capture ABI: expected {sorted(fields)}, got {sorted(members)}"
                )
            oversized = [name for name, info in members.items() if info.file_size > max_source_member_bytes]
            if oversized:
                raise ParquetProjectionError(
                    f"{path.name} contains unbounded members {sorted(oversized)}; "
                    f"member limit is {max_source_member_bytes} bytes"
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ParquetProjectionError(f"cannot inspect {path.name}: {exc}") from exc
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {field: np.asarray(archive[field]) for field in sorted(fields)}
    except (OSError, ValueError, EOFError) as exc:
        raise ParquetProjectionError(f"cannot decode {path.name}: {exc}") from exc


def _require_shape(array: np.ndarray, *, name: str, dtype: np.dtype[Any], shape: tuple[int, ...]) -> None:
    if array.dtype != dtype or array.shape != shape or not np.all(np.isfinite(array.astype(np.float64, copy=False))):
        raise ParquetProjectionError(f"{name} must be finite {dtype} {shape}, got {array.dtype} {array.shape}")


def _flatten_vector_fields(
    columns: dict[str, np.ndarray],
    values: Mapping[str, np.ndarray],
    field: str,
    suffixes: Sequence[str],
) -> None:
    array = values[field]
    for index, suffix in enumerate(suffixes):
        columns[f"{field}_{suffix}"] = array[:, :, index].reshape(-1)


def _state_table(values: Mapping[str, np.ndarray], *, agent_count: int, row_group_size: int, writer_factory: Any) -> int:
    steps = len(values["command_time_ns"])
    if steps <= 0:
        raise ParquetProjectionError("state/action stream is empty")
    for name, array in values.items():
        if name in {"command_time_ns", "effective_time_ns"}:
            _require_shape(array, name=name, dtype=np.dtype("<i8"), shape=(steps,))
        elif name in {"root_pos_w_m", "root_lin_vel_w_mps", "root_ang_vel_b_radps", "desired_pos_w_m", "desired_vel_w_mps"}:
            _require_shape(array, name=name, dtype=np.dtype("<f4"), shape=(steps, agent_count, 3))
        elif name == "root_quat_wxyz":
            _require_shape(array, name=name, dtype=np.dtype("<f4"), shape=(steps, agent_count, 4))
        else:
            _require_shape(array, name=name, dtype=np.dtype("<f4"), shape=(steps, agent_count, 4))
    if np.any(values["command_time_ns"] >= values["effective_time_ns"]):
        raise ParquetProjectionError("state/action command_time_ns must precede effective_time_ns")

    pa, _ = _arrow()
    schema: Any | None = None
    rows = 0
    agent_ids = np.arange(agent_count, dtype=np.int32)
    for start in range(0, steps, max(1, row_group_size // agent_count)):
        stop = min(steps, start + max(1, row_group_size // agent_count))
        columns: dict[str, np.ndarray] = {
            "step_index": np.repeat(np.arange(start, stop, dtype=np.int64), agent_count),
            "agent_id": np.tile(agent_ids, stop - start),
            "command_time_ns": np.repeat(values["command_time_ns"][start:stop], agent_count),
            "effective_time_ns": np.repeat(values["effective_time_ns"][start:stop], agent_count),
        }
        for field, suffixes in (
            ("root_pos_w_m", ("x", "y", "z")),
            ("root_quat_wxyz", ("w", "x", "y", "z")),
            ("root_lin_vel_w_mps", ("x", "y", "z")),
            ("root_ang_vel_b_radps", ("x", "y", "z")),
            ("desired_pos_w_m", ("x", "y", "z")),
            ("desired_vel_w_mps", ("x", "y", "z")),
            ("target_thrust_n", ("rotor_0", "rotor_1", "rotor_2", "rotor_3")),
            ("applied_thrust_n", ("rotor_0", "rotor_1", "rotor_2", "rotor_3")),
        ):
            for index, suffix in enumerate(suffixes):
                columns[f"{field}_{suffix}"] = values[field][start:stop, :, index].reshape(-1)
        table = pa.Table.from_pydict(columns, schema=schema)
        if schema is None:
            schema = table.schema
            writer_factory.set_schema(schema)
        writer_factory.write_table(table, row_group_size=len(table))
        rows += len(table)
    return rows


def _task_table(values: Mapping[str, np.ndarray], *, agent_count: int, row_group_size: int, writer_factory: Any) -> int:
    samples = len(values["timestamps_ns"])
    for name, array in values.items():
        expected_tail = {
            "timestamps_ns": (),
            "waypoint_index": (agent_count,),
            "waypoint_progress": (agent_count,),
            "desired_waypoint_w_m": (agent_count, 3),
            "distance_to_waypoint_m": (agent_count,),
            "waypoint_reached": (agent_count,),
            "action_mode": (agent_count,),
            "coverage_cell_id": (agent_count,),
            "task_time_s": (agent_count,),
        }[name]
        dtype = np.dtype("<i8") if name in {"timestamps_ns", "waypoint_index", "coverage_cell_id"} else np.dtype("<f4")
        if name == "waypoint_reached":
            dtype = np.dtype("bool")
        if name == "action_mode":
            dtype = np.dtype("i1")
        _require_shape(array, name=name, dtype=dtype, shape=(samples, *expected_tail))
    pa, _ = _arrow()
    schema: Any | None = None
    rows = 0
    for start in range(0, samples, max(1, row_group_size // agent_count)):
        stop = min(samples, start + max(1, row_group_size // agent_count))
        columns: dict[str, np.ndarray] = {
            "sample_index": np.repeat(np.arange(start, stop, dtype=np.int64), agent_count),
            "agent_id": np.tile(np.arange(agent_count, dtype=np.int32), stop - start),
            "timestamp_ns": np.repeat(values["timestamps_ns"][start:stop], agent_count),
        }
        for field in ("waypoint_index", "waypoint_progress", "distance_to_waypoint_m", "waypoint_reached", "action_mode", "coverage_cell_id", "task_time_s"):
            columns[field] = values[field][start:stop].reshape(-1)
        for index, suffix in enumerate(("x", "y", "z")):
            columns[f"desired_waypoint_w_m_{suffix}"] = values["desired_waypoint_w_m"][start:stop, :, index].reshape(-1)
        table = pa.Table.from_pydict(columns, schema=schema)
        if schema is None:
            schema = table.schema
            writer_factory.set_schema(schema)
        writer_factory.write_table(table, row_group_size=len(table))
        rows += len(table)
    return rows


def _messages_table(values: Mapping[str, np.ndarray], *, agent_count: int, row_group_size: int, writer_factory: Any) -> int:
    samples = len(values["timestamps_ns"])
    specs = {
        "timestamps_ns": (np.dtype("<i8"), ()),
        "sender_agent_id": (np.dtype("<i8"), (agent_count,)),
        "message_sequence": (np.dtype("<i8"), (agent_count,)),
        "message_waypoint_index": (np.dtype("<i8"), (agent_count,)),
        "message_position_w_m": (np.dtype("<f4"), (agent_count, 3)),
        "message_velocity_w_mps": (np.dtype("<f4"), (agent_count, 3)),
        "message_flags": (np.dtype("u1"), (agent_count,)),
    }
    for name, (dtype, tail) in specs.items():
        _require_shape(values[name], name=name, dtype=dtype, shape=(samples, *tail))
    pa, _ = _arrow()
    schema: Any | None = None
    rows = 0
    for start in range(0, samples, max(1, row_group_size // agent_count)):
        stop = min(samples, start + max(1, row_group_size // agent_count))
        columns: dict[str, np.ndarray] = {
            "sample_index": np.repeat(np.arange(start, stop, dtype=np.int64), agent_count),
            "agent_id": np.tile(np.arange(agent_count, dtype=np.int32), stop - start),
            "timestamp_ns": np.repeat(values["timestamps_ns"][start:stop], agent_count),
            "sender_agent_id": values["sender_agent_id"][start:stop].reshape(-1),
            "message_sequence": values["message_sequence"][start:stop].reshape(-1),
            "message_waypoint_index": values["message_waypoint_index"][start:stop].reshape(-1),
            "message_flags": values["message_flags"][start:stop].reshape(-1),
        }
        for index, suffix in enumerate(("x", "y", "z")):
            columns[f"message_position_w_m_{suffix}"] = values["message_position_w_m"][start:stop, :, index].reshape(-1)
            columns[f"message_velocity_w_mps_{suffix}"] = values["message_velocity_w_mps"][start:stop, :, index].reshape(-1)
        table = pa.Table.from_pydict(columns, schema=schema)
        if schema is None:
            schema = table.schema
            writer_factory.set_schema(schema)
        writer_factory.write_table(table, row_group_size=len(table))
        rows += len(table)
    return rows


class _WriterFactory:
    """Delay ParquetWriter construction until the first validated row group."""

    def __init__(self, path: Path, pq: Any, *, row_group_size: int):
        self.path = path
        self.pq = pq
        self.row_group_size = row_group_size
        self.writer: Any | None = None

    def set_schema(self, schema: Any) -> None:
        if self.writer is not None:
            return
        self.writer = self.pq.ParquetWriter(
            str(self.path), schema, version="2.6", compression="zstd", use_dictionary=False
        )

    def write_table(self, table: Any, *, row_group_size: int) -> None:
        if self.writer is None:
            raise ParquetProjectionError("Parquet writer schema was not initialized")
        self.writer.write_table(table, row_group_size=row_group_size)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _write_table(path: Path, values: Mapping[str, np.ndarray], *, kind: str, agent_count: int, row_group_size: int) -> int:
    _, pq = _arrow()
    writer = _WriterFactory(path, pq, row_group_size=row_group_size)
    try:
        if kind == "state_action":
            rows = _state_table(values, agent_count=agent_count, row_group_size=row_group_size, writer_factory=writer)
        elif kind == "public_task":
            rows = _task_table(values, agent_count=agent_count, row_group_size=row_group_size, writer_factory=writer)
        elif kind == "public_messages":
            rows = _messages_table(values, agent_count=agent_count, row_group_size=row_group_size, writer_factory=writer)
        else:  # pragma: no cover - internal dispatch
            raise AssertionError(kind)
    finally:
        writer.close()
    return rows


def _metadata_table(path: Path, receipt: Mapping[str, Any], receipt_hash: str, validation_hash: str) -> int:
    pa, pq = _arrow()
    physics = receipt.get("physics") if isinstance(receipt.get("physics"), Mapping) else {}
    binding = _public_collection_binding(receipt)
    metadata = {
        "schema": [DEVELOPMENT_PARQUET_SCHEMA],
        "formal_benchmark_admission": [False],
        "capture_attempt_id": [str(receipt.get("capture_attempt_id", ""))],
        "capture_receipt_sha256": [receipt_hash],
        "independent_validation_sha256": [validation_hash],
        "source_revision": [str(receipt.get("source_revision", ""))],
        "collection_protocol_id": [binding["protocol_id"]],
        "collection_cell_id": [binding["cell_id"]],
        "split": [binding["split"]],
        "episode_index": [binding["episode_index"]],
        "episode_seed": [binding["episode_seed"]],
        "agent_count": [int(physics.get("same_world_agent_count", 0))],
        "physics_steps": [int(physics.get("physics_steps", 0))],
        "sensor_samples": [int(physics.get("sensor_samples", 0))],
        "action_timing": ["command_before_step"],
        "state_timing": ["state_after_step"],
        "coordinate_frames": ["right-handed; +Z up; world x_east_y_north_z_up; body FLU; quaternion wxyz"],
        "source_encoding": ["native Isaac chunked NPZ"],
        "parquet_compression": ["zstd"],
    }
    table = pa.Table.from_pydict(metadata)
    pq.write_table(table, str(path), version="2.6", compression="zstd", use_dictionary=False)
    return table.num_rows


def _table_record(root: Path, relative: str, rows: int) -> dict[str, Any]:
    path = root / relative
    _, pq = _arrow()
    parquet_file = pq.ParquetFile(str(path))
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(rows),
        "columns": [field.name for field in parquet_file.schema_arrow],
        "row_groups": int(parquet_file.metadata.num_row_groups),
    }


def project_development_capture_to_parquet(
    capture_root: Path,
    output_root: Path,
    *,
    max_source_member_bytes: int = _DEFAULT_MAX_SOURCE_MEMBER_BYTES,
    row_group_size: int = _DEFAULT_ROW_GROUP_SIZE,
) -> ParquetProjectionResult:
    """Write small public Parquet tables from a validated development capture.

    The source capture and the output directory must be distinct.  The output
    contains no RGB, depth, semantic, LiDAR, evaluator, or private-target
    payload; those remain in the external evidence directory.
    """

    if isinstance(max_source_member_bytes, bool) or not isinstance(max_source_member_bytes, int) or max_source_member_bytes <= 0:
        raise ParquetProjectionError("max_source_member_bytes must be a positive integer")
    if isinstance(row_group_size, bool) or not isinstance(row_group_size, int) or row_group_size <= 0:
        raise ParquetProjectionError("row_group_size must be a positive integer")
    capture = capture_root.resolve()
    destination = output_root.resolve()
    if capture == destination or destination.is_relative_to(capture):
        raise ParquetProjectionError("projection output must not be inside the source capture")
    if destination.exists():
        raise ParquetProjectionError(f"projection output already exists: {destination}")
    receipt, receipt_hash, validation_hash, artifacts = _verify_capture_boundary(capture)
    physics = receipt.get("physics")
    if not isinstance(physics, Mapping) or not isinstance(physics.get("same_world_agent_count"), int):
        raise ParquetProjectionError("capture receipt has no agent count")
    agent_count = int(physics["same_world_agent_count"])
    if agent_count <= 0:
        raise ParquetProjectionError("capture agent count must be positive")
    state = _bounded_npz(_contained_file(capture, _STATE_PATH), fields=_STATE_FIELDS, max_source_member_bytes=max_source_member_bytes)
    task = _bounded_npz(_contained_file(capture, _TASK_PATH), fields=_TASK_FIELDS, max_source_member_bytes=max_source_member_bytes)
    messages = _bounded_npz(_contained_file(capture, _MESSAGES_PATH), fields=_MESSAGE_FIELDS, max_source_member_bytes=max_source_member_bytes)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        table_rows = {
            "metadata.parquet": _metadata_table(staging / "metadata.parquet", receipt, receipt_hash, validation_hash),
            "state_action.parquet": _write_table(staging / "state_action.parquet", state, kind="state_action", agent_count=agent_count, row_group_size=row_group_size),
            "public_task.parquet": _write_table(staging / "public_task.parquet", task, kind="public_task", agent_count=agent_count, row_group_size=row_group_size),
            "public_messages.parquet": _write_table(staging / "public_messages.parquet", messages, kind="public_messages", agent_count=agent_count, row_group_size=row_group_size),
        }
        manifest = {
            "schema": DEVELOPMENT_PARQUET_SCHEMA,
            "status": "projected",
            "development_only": True,
            "formal_benchmark_admission": False,
            "source_capture_receipt_sha256": receipt_hash,
            "independent_validation_sha256": validation_hash,
            "source_revision": receipt["source_revision"],
            "capture_attempt_id": receipt["capture_attempt_id"],
            "collection_binding": _public_collection_binding(receipt),
            "source_artifacts": list(artifacts.values()),
            "tables": [_table_record(staging, relative, rows) for relative, rows in table_rows.items()],
            "parquet": {
                "format_version": "2.6",
                "engine": "pyarrow",
                "engine_version": "25.0.0",
                "compression": "zstd",
                "row_group_size": row_group_size,
            },
            "omitted_modalities": ["rgb", "distance_to_image_plane", "semantic_segmentation", "multi_mesh_raycaster_lidar", "imu", "contact"],
            "claim_boundary": "development evidence only; not a formal episode or public release payload",
        }
        _write_json(staging / "projection_manifest.json", manifest)
        os.replace(staging, destination)
        staging = None  # type: ignore[assignment]
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    manifest_path = destination / "projection_manifest.json"
    return ParquetProjectionResult(
        output_root=destination,
        capture_receipt_sha256=receipt_hash,
        independent_validation_sha256=validation_hash,
        projection_manifest_sha256=sha256_file(manifest_path),
        table_paths=tuple(table_rows),
    )


def read_development_parquet_table(root: Path, relative: str) -> Any:
    """Read one projection table through the standard PyArrow reader."""

    if not is_safe_relative_path(relative) or not relative.endswith(".parquet"):
        raise ParquetProjectionError(f"unsafe Parquet table path: {relative!r}")
    path = (root.resolve() / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ParquetProjectionError(f"missing Parquet table: {relative}")
    _, pq = _arrow()
    try:
        return pq.read_table(str(path))
    except (OSError, ValueError, RuntimeError) as exc:
        raise ParquetProjectionError(f"cannot read Parquet table {relative}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--max-source-member-bytes", type=int, default=_DEFAULT_MAX_SOURCE_MEMBER_BYTES)
    parser.add_argument("--row-group-size", type=int, default=_DEFAULT_ROW_GROUP_SIZE)
    args = parser.parse_args(argv)
    try:
        result = project_development_capture_to_parquet(
            args.capture_root,
            args.output_root,
            max_source_member_bytes=args.max_source_member_bytes,
            row_group_size=args.row_group_size,
        )
    except (OSError, ParquetProjectionError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({
        "status": "projected",
        "development_only": True,
        "formal_benchmark_admission": False,
        "output_root": str(result.output_root),
        "capture_receipt_sha256": result.capture_receipt_sha256,
        "independent_validation_sha256": result.independent_validation_sha256,
        "projection_manifest_sha256": result.projection_manifest_sha256,
        "table_paths": list(result.table_paths),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEVELOPMENT_PARQUET_SCHEMA",
    "ParquetProjectionError",
    "ParquetProjectionResult",
    "project_development_capture_to_parquet",
    "read_development_parquet_table",
]
