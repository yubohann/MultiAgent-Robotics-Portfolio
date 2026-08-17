"""LeRobot v3.0 state/action projection for validated development captures.

The native capture remains the source of truth.  This module consumes the
bounded development Parquet projection, writes one LeRobot episode per agent,
and preserves the eight-agent relationship in a separate group manifest.
Visual and asynchronous sensor streams are deliberately out of scope until a
reader-verified media projection exists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .formal_dataset import sha256_file
from .parquet_projection import DEVELOPMENT_PARQUET_SCHEMA
from .schema import is_safe_relative_path

LEROBOT_PROJECTION_SCHEMA = "org.rivermark.benchmark.development-lerobot-projection.v1"
LEROBOT_FORMAT_VERSION = "v3.0"
LEROBOT_SOURCE_VERSION = "0.6.1"
_SOURCE_STATE_PATH = "state_action.parquet"
_SOURCE_METADATA_PATH = "metadata.parquet"
_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_OUTPUT_FILES = (
    "data/chunk-000/file-000.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "meta/info.json",
    "meta/tasks.parquet",
)
_CLAIM_BOUNDARY = (
    "development-only LeRobot-compatible state/action projection; "
    "not a formal episode, multimodal payload, or native Isaac execution receipt"
)

_SOURCE_VECTOR_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("root_pos_w_m", ("x", "y", "z")),
    ("root_quat_wxyz", ("w", "x", "y", "z")),
    ("root_lin_vel_w_mps", ("x", "y", "z")),
    ("root_ang_vel_b_radps", ("x", "y", "z")),
    ("desired_pos_w_m", ("x", "y", "z")),
    ("desired_vel_w_mps", ("x", "y", "z")),
    ("target_thrust_n", ("rotor_0", "rotor_1", "rotor_2", "rotor_3")),
    ("applied_thrust_n", ("rotor_0", "rotor_1", "rotor_2", "rotor_3")),
)
_SOURCE_COLUMNS = (
    "step_index",
    "agent_id",
    "command_time_ns",
    "effective_time_ns",
    *(f"{field}_{suffix}" for field, suffixes in _SOURCE_VECTOR_FIELDS for suffix in suffixes),
)


class LeRobotProjectionError(ValueError):
    """Raised when a LeRobot projection cannot be proved consistent."""


@dataclass(frozen=True)
class LeRobotProjectionResult:
    output_root: Path
    fleet_episode_id: str
    agent_episode_count: int
    frame_count: int
    fps: int
    source_projection_manifest_sha256: str
    group_manifest_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LeRobotProjectionError(
            "LeRobot projection requires the optional 'parquet' extra (pyarrow==25.0.0)"
        ) from exc
    return pa, pq


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LeRobotProjectionError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise LeRobotProjectionError(f"{label} must be a JSON object")
    return value


def _contained_file(root: Path, relative: str) -> Path:
    if not is_safe_relative_path(relative):
        raise LeRobotProjectionError(f"unsafe relative path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise LeRobotProjectionError(f"missing file: {relative}")
    return path


def _source_table_records(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    tables = manifest.get("tables")
    if not isinstance(tables, list):
        raise LeRobotProjectionError("source projection manifest has no tables list")
    records: dict[str, Mapping[str, Any]] = {}
    for record in tables:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise LeRobotProjectionError("source projection contains an invalid table record")
        relative = record["path"]
        if relative in records:
            raise LeRobotProjectionError(f"source projection contains duplicate table path: {relative}")
        records[relative] = record
    return records


def _verify_file_record(root: Path, record: Mapping[str, Any], relative: str) -> Path:
    path = _contained_file(root, relative)
    if record.get("path") != relative:
        raise LeRobotProjectionError(f"file record path mismatch for {relative}")
    if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
        raise LeRobotProjectionError(f"file does not match its manifest binding: {relative}")
    return path


def _source_boundary(root: Path) -> tuple[Mapping[str, Any], str, Mapping[str, Any], Path, int, int]:
    manifest_path = _contained_file(root, "projection_manifest.json")
    manifest = _load_json(manifest_path, "source projection_manifest.json")
    if (
        manifest.get("schema") != DEVELOPMENT_PARQUET_SCHEMA
        or manifest.get("status") != "projected"
        or manifest.get("development_only") is not True
        or manifest.get("formal_benchmark_admission") is not False
    ):
        raise LeRobotProjectionError("source must be a development-only Parquet projection")
    tables = _source_table_records(manifest)
    for relative in (_SOURCE_METADATA_PATH, _SOURCE_STATE_PATH):
        if relative not in tables:
            raise LeRobotProjectionError(f"source projection does not bind {relative}")
        _verify_file_record(root, tables[relative], relative)

    pa, pq = _arrow()
    metadata = pq.read_table(str(root / _SOURCE_METADATA_PATH))
    if metadata.num_rows != 1:
        raise LeRobotProjectionError("source metadata table must contain exactly one row")
    metadata_row = metadata.to_pylist()[0]
    agent_count = metadata_row.get("agent_count")
    if isinstance(agent_count, bool) or not isinstance(agent_count, int) or agent_count <= 0:
        raise LeRobotProjectionError("source metadata has no positive agent_count")
    if metadata_row.get("action_timing") != "command_before_step":
        raise LeRobotProjectionError("source action timing is not command_before_step")
    state_path = root / _SOURCE_STATE_PATH
    state_file = pq.ParquetFile(str(state_path))
    if tuple(field.name for field in state_file.schema_arrow) != _SOURCE_COLUMNS:
        raise LeRobotProjectionError("source state/action columns do not match the frozen projection ABI")
    row_count = int(state_file.metadata.num_rows)
    if row_count < 2 * agent_count or row_count % agent_count:
        raise LeRobotProjectionError("source state/action rows do not form complete agent steps")
    steps = row_count // agent_count
    if tables[_SOURCE_STATE_PATH].get("rows") != row_count:
        raise LeRobotProjectionError("source state/action row count differs from its manifest")
    if metadata.schema.metadata and b"pandas" in metadata.schema.metadata:
        _ = pa  # Keep the optional import visibly tied to this schema read.
    return manifest, sha256_file(manifest_path), metadata_row, state_path, agent_count, steps


def _column_numpy(table: Any, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    column = table.column(name)
    if column.null_count:
        raise LeRobotProjectionError(f"source column contains nulls: {name}")
    try:
        values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise LeRobotProjectionError(f"cannot decode source column {name}: {exc}") from exc
    return values


def _agent_source_table(state_path: Path, agent_id: int, steps: int) -> dict[str, np.ndarray]:
    _, pq = _arrow()
    table = pq.read_table(str(state_path), columns=list(_SOURCE_COLUMNS), filters=[("agent_id", "=", agent_id)])
    if table.num_rows != steps:
        raise LeRobotProjectionError(f"agent {agent_id} has {table.num_rows} rows, expected {steps}")
    result: dict[str, np.ndarray] = {
        "step_index": _column_numpy(table, "step_index", np.dtype("<i8")),
        "agent_id": _column_numpy(table, "agent_id", np.dtype("<i8")),
        "command_time_ns": _column_numpy(table, "command_time_ns", np.dtype("<i8")),
        "effective_time_ns": _column_numpy(table, "effective_time_ns", np.dtype("<i8")),
    }
    for field, suffixes in _SOURCE_VECTOR_FIELDS:
        result[field] = np.column_stack(
            [_column_numpy(table, f"{field}_{suffix}", np.dtype("<f4")) for suffix in suffixes]
        ).astype("<f4", copy=False)
    if not np.array_equal(result["step_index"], np.arange(steps, dtype="<i8")):
        raise LeRobotProjectionError(f"agent {agent_id} source step indices are not contiguous")
    if not np.all(result["agent_id"] == agent_id):
        raise LeRobotProjectionError(f"agent {agent_id} filter returned another agent")
    if np.any(result["command_time_ns"] >= result["effective_time_ns"]):
        raise LeRobotProjectionError(f"agent {agent_id} command time does not precede effective time")
    if np.any(np.diff(result["command_time_ns"]) <= 0) or np.any(np.diff(result["effective_time_ns"]) <= 0):
        raise LeRobotProjectionError(f"agent {agent_id} native timestamps are not strictly increasing")
    for field, _ in _SOURCE_VECTOR_FIELDS:
        if not np.all(np.isfinite(result[field])):
            raise LeRobotProjectionError(f"agent {agent_id} source field is not finite: {field}")
    return result


def _resolve_fps(effective_time_ns: np.ndarray, requested_fps: int | None, tolerance_s: float) -> int:
    if len(effective_time_ns) < 2:
        raise LeRobotProjectionError("at least two native timestamps are required")
    deltas_s = np.diff(effective_time_ns.astype(np.float64)) / 1_000_000_000.0
    median_delta = float(np.median(deltas_s))
    inferred = round(1.0 / median_delta) if median_delta > 0 else 0
    fps = inferred if requested_fps is None else requested_fps
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 1000:
        raise LeRobotProjectionError(f"fps must be an integer in [1, 1000], got {fps!r}")
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise LeRobotProjectionError("timestamp_tolerance_s must be finite and non-negative")
    expected_delta = 1.0 / fps
    if np.any(np.abs(deltas_s - expected_delta) > tolerance_s):
        raise LeRobotProjectionError(
            f"native effective timestamps do not match {fps} fps within {tolerance_s} seconds"
        )
    return fps


def _features() -> dict[str, dict[str, Any]]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [13],
            "names": [
                "pos_w_x_m",
                "pos_w_y_m",
                "pos_w_z_m",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
                "lin_vel_w_x_mps",
                "lin_vel_w_y_mps",
                "lin_vel_w_z_mps",
                "ang_vel_b_x_radps",
                "ang_vel_b_y_radps",
                "ang_vel_b_z_radps",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [4],
            "names": ["target_thrust_rotor_0_n", "target_thrust_rotor_1_n", "target_thrust_rotor_2_n", "target_thrust_rotor_3_n"],
        },
        "rivermark.applied_thrust_n": {
            "dtype": "float32",
            "shape": [4],
            "names": ["rotor_0", "rotor_1", "rotor_2", "rotor_3"],
        },
        "rivermark.desired_state": {
            "dtype": "float32",
            "shape": [6],
            "names": ["pos_w_x_m", "pos_w_y_m", "pos_w_z_m", "vel_w_x_mps", "vel_w_y_mps", "vel_w_z_mps"],
        },
        "rivermark.command_time_ns": {"dtype": "int64", "shape": [1], "names": None},
        "rivermark.effective_time_ns": {"dtype": "int64", "shape": [1], "names": None},
        "rivermark.source_step_index": {"dtype": "int64", "shape": [1], "names": None},
        "rivermark.agent_id": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _fixed_list(pa: Any, values: np.ndarray, width: int) -> Any:
    flat = np.asarray(values, dtype="<f4").reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), width)


def _agent_lerobot_table(pa: Any, source: Mapping[str, np.ndarray], agent_id: int, fps: int, steps: int) -> Any:
    global_start = agent_id * steps
    observation = np.concatenate(
        (
            source["root_pos_w_m"],
            source["root_quat_wxyz"],
            source["root_lin_vel_w_mps"],
            source["root_ang_vel_b_radps"],
        ),
        axis=1,
    )
    desired = np.concatenate((source["desired_pos_w_m"], source["desired_vel_w_mps"]), axis=1)
    arrays = [
        _fixed_list(pa, observation, 13),
        _fixed_list(pa, source["target_thrust_n"], 4),
        _fixed_list(pa, source["applied_thrust_n"], 4),
        _fixed_list(pa, desired, 6),
        pa.array(source["command_time_ns"], type=pa.int64()),
        pa.array(source["effective_time_ns"], type=pa.int64()),
        pa.array(source["step_index"], type=pa.int64()),
        pa.array(np.full(steps, agent_id, dtype="<i8"), type=pa.int64()),
        pa.array(np.arange(steps, dtype="<f4") / np.float32(fps), type=pa.float32()),
        pa.array(np.arange(steps, dtype="<i8"), type=pa.int64()),
        pa.array(np.full(steps, agent_id, dtype="<i8"), type=pa.int64()),
        pa.array(np.arange(global_start, global_start + steps, dtype="<i8"), type=pa.int64()),
        pa.array(np.zeros(steps, dtype="<i8"), type=pa.int64()),
    ]
    return pa.Table.from_arrays(arrays, names=list(_features()))


def _write_tasks(path: Path, task_description: str, pa: Any, pq: Any) -> None:
    table = pa.Table.from_arrays(
        [pa.array([0], type=pa.int64()), pa.array([task_description], type=pa.string())],
        names=["task_index", "task"],
    )
    pandas_metadata = {
        "index_columns": ["task"],
        "column_indexes": [
            {
                "name": None,
                "field_name": None,
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": {"encoding": "UTF-8"},
            }
        ],
        "columns": [
            {
                "name": "task_index",
                "field_name": "task_index",
                "pandas_type": "int64",
                "numpy_type": "int64",
                "metadata": None,
            },
            {
                "name": "task",
                "field_name": "task",
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": None,
            },
        ],
        "creator": {"library": "pyarrow", "version": pa.__version__},
        "pandas_version": "2.0.0",
    }
    table = table.replace_schema_metadata({b"pandas": json.dumps(pandas_metadata).encode("utf-8")})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path), compression="snappy", use_dictionary=True)


def _write_episodes(path: Path, agent_count: int, steps: int, task_description: str, pa: Any, pq: Any) -> None:
    table = pa.Table.from_arrays(
        [
            pa.array(np.arange(agent_count, dtype="<i8"), type=pa.int64()),
            pa.array([[task_description] for _ in range(agent_count)], type=pa.list_(pa.string())),
            pa.array(np.full(agent_count, steps, dtype="<i8"), type=pa.int64()),
            pa.array(np.zeros(agent_count, dtype="<i8"), type=pa.int64()),
            pa.array(np.zeros(agent_count, dtype="<i8"), type=pa.int64()),
            pa.array(np.arange(agent_count, dtype="<i8") * steps, type=pa.int64()),
            pa.array((np.arange(agent_count, dtype="<i8") + 1) * steps, type=pa.int64()),
        ],
        names=[
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path), compression="snappy", use_dictionary=True)


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    path = _contained_file(root, relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def project_development_parquet_to_lerobot(
    source_root: Path,
    output_root: Path,
    *,
    fps: int | None = None,
    timestamp_tolerance_s: float = 1e-4,
    task_description: str = "Follow the frozen public Rivermark City-Lite route.",
) -> LeRobotProjectionResult:
    """Project one validated fleet capture into LeRobot v3.0 state/action files."""

    source = Path(source_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if not task_description.strip():
        raise LeRobotProjectionError("task_description must be non-empty")
    if source == destination or destination.is_relative_to(source):
        raise LeRobotProjectionError("LeRobot output must not be inside the source projection")
    if destination.exists():
        raise LeRobotProjectionError(f"LeRobot output already exists: {destination}")
    source_manifest, source_manifest_hash, _metadata, state_path, agent_count, steps = _source_boundary(source)
    first = _agent_source_table(state_path, 0, steps)
    resolved_fps = _resolve_fps(first["effective_time_ns"], fps, timestamp_tolerance_s)
    reference_command_time = first["command_time_ns"]
    reference_effective_time = first["effective_time_ns"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    pa, pq = _arrow()
    writer: Any | None = None
    try:
        data_path = staging / _OUTPUT_FILES[0]
        data_path.parent.mkdir(parents=True, exist_ok=True)
        for agent_id in range(agent_count):
            values = first if agent_id == 0 else _agent_source_table(state_path, agent_id, steps)
            if not np.array_equal(values["command_time_ns"], reference_command_time) or not np.array_equal(
                values["effective_time_ns"], reference_effective_time
            ):
                raise LeRobotProjectionError("agent native state timelines are not synchronized")
            table = _agent_lerobot_table(pa, values, agent_id, resolved_fps, steps)
            if writer is None:
                writer = pq.ParquetWriter(str(data_path), table.schema, compression="snappy", use_dictionary=True)
            writer.write_table(table, row_group_size=steps)
        if writer is not None:
            writer.close()
            writer = None

        _write_episodes(staging / _OUTPUT_FILES[1], agent_count, steps, task_description.strip(), pa, pq)
        collection = source_manifest.get("collection_binding")
        split = collection.get("split") if isinstance(collection, Mapping) else "train"
        if not isinstance(split, str) or not split:
            split = "train"
        info = {
            "codebase_version": LEROBOT_FORMAT_VERSION,
            "fps": resolved_fps,
            "features": _features(),
            "total_episodes": agent_count,
            "total_frames": agent_count * steps,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "data_path": _DEFAULT_DATA_PATH,
            "video_path": None,
            "robot_type": "cf2x_swarm_agent",
            "splits": {split: f"0:{agent_count}"},
        }
        _write_json(staging / _OUTPUT_FILES[2], info)
        _write_tasks(staging / _OUTPUT_FILES[3], task_description.strip(), pa, pq)

        fleet_episode_id = str(source_manifest.get("capture_attempt_id", "")).strip()
        if not fleet_episode_id:
            raise LeRobotProjectionError("source projection has no capture_attempt_id")
        group_manifest = {
            "schema": LEROBOT_PROJECTION_SCHEMA,
            "status": "projected",
            "development_only": True,
            "formal_benchmark_admission": False,
            "claim_boundary": _CLAIM_BOUNDARY,
            "fleet_episode_id": fleet_episode_id,
            "agent_episode_count": agent_count,
            "frames_per_agent": steps,
            "frame_count": agent_count * steps,
            "source_projection_manifest_sha256": source_manifest_hash,
            "source_capture_receipt_sha256": source_manifest.get("source_capture_receipt_sha256"),
            "independent_validation_sha256": source_manifest.get("independent_validation_sha256"),
            "source_revision": source_manifest.get("source_revision"),
            "collection_binding": collection,
            "lerobot": {
                "format_version": LEROBOT_FORMAT_VERSION,
                "source_version_reviewed": LEROBOT_SOURCE_VERSION,
                "fps": resolved_fps,
                "standard_timestamp": "frame_index / fps; synthetic compatibility index only",
                "native_time_fields": ["rivermark.command_time_ns", "rivermark.effective_time_ns"],
                "data_compression": "snappy",
            },
            "agent_episodes": [
                {"agent_id": agent_id, "episode_index": agent_id, "dataset_from_index": agent_id * steps, "dataset_to_index": (agent_id + 1) * steps}
                for agent_id in range(agent_count)
            ],
            "included_modalities": ["state", "desired_state", "target_thrust", "applied_thrust", "native_timestamps"],
            "omitted_modalities": ["rgb", "depth", "semantic_segmentation", "lidar", "imu", "contact", "messages"],
            "files": [_file_record(staging, relative) for relative in _OUTPUT_FILES],
        }
        _write_json(staging / "meta" / "rivermark_group_manifest.json", group_manifest)
        os.replace(staging, destination)
        staging = None  # type: ignore[assignment]
    except Exception:
        if writer is not None:
            writer.close()
        raise
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    report = verify_lerobot_projection(destination, source_root=source)
    group_manifest_path = destination / "meta" / "rivermark_group_manifest.json"
    return LeRobotProjectionResult(
        output_root=destination,
        fleet_episode_id=str(report["fleet_episode_id"]),
        agent_episode_count=int(report["agent_episode_count"]),
        frame_count=int(report["frame_count"]),
        fps=int(report["fps"]),
        source_projection_manifest_sha256=source_manifest_hash,
        group_manifest_sha256=sha256_file(group_manifest_path),
    )


def verify_lerobot_projection(root: Path, *, source_root: Path | None = None) -> dict[str, Any]:
    """Verify the LeRobot disk layout, fleet grouping, native time, and hashes."""

    destination = Path(root).expanduser().resolve()
    manifest_path = _contained_file(destination, "meta/rivermark_group_manifest.json")
    manifest = _load_json(manifest_path, "rivermark_group_manifest.json")
    if (
        manifest.get("schema") != LEROBOT_PROJECTION_SCHEMA
        or manifest.get("status") != "projected"
        or manifest.get("development_only") is not True
        or manifest.get("formal_benchmark_admission") is not False
        or manifest.get("claim_boundary") != _CLAIM_BOUNDARY
    ):
        raise LeRobotProjectionError("LeRobot group manifest boundary is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or [record.get("path") for record in files if isinstance(record, Mapping)] != list(
        _OUTPUT_FILES
    ):
        raise LeRobotProjectionError("LeRobot group manifest has an unexpected file set")
    for relative, record in zip(_OUTPUT_FILES, files, strict=True):
        if not isinstance(record, Mapping):
            raise LeRobotProjectionError("LeRobot group manifest contains an invalid file record")
        _verify_file_record(destination, record, relative)
    if source_root is not None:
        source_manifest = _contained_file(Path(source_root).expanduser().resolve(), "projection_manifest.json")
        if manifest.get("source_projection_manifest_sha256") != sha256_file(source_manifest):
            raise LeRobotProjectionError("LeRobot projection is not bound to the supplied source projection")

    info = _load_json(destination / "meta" / "info.json", "LeRobot meta/info.json")
    agent_count = manifest.get("agent_episode_count")
    steps = manifest.get("frames_per_agent")
    frame_count = manifest.get("frame_count")
    fps = info.get("fps")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (agent_count, steps, fps)):
        raise LeRobotProjectionError("LeRobot counts and fps must be positive integers")
    if frame_count != agent_count * steps:
        raise LeRobotProjectionError("LeRobot fleet frame count is inconsistent")
    if (
        info.get("codebase_version") != LEROBOT_FORMAT_VERSION
        or info.get("features") != _features()
        or info.get("total_episodes") != agent_count
        or info.get("total_frames") != frame_count
        or info.get("total_tasks") != 1
        or info.get("video_path") is not None
    ):
        raise LeRobotProjectionError("LeRobot meta/info.json does not match the frozen state/action profile")

    _pa, pq = _arrow()
    task_table = pq.read_table(str(destination / "meta" / "tasks.parquet"))
    pandas_metadata = task_table.schema.metadata or {}
    if task_table.num_rows != 1 or task_table.column_names != ["task_index", "task"] or b"pandas" not in pandas_metadata:
        raise LeRobotProjectionError("LeRobot tasks table is not pandas-index compatible")
    try:
        task_index_metadata = json.loads(pandas_metadata[b"pandas"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LeRobotProjectionError(f"LeRobot tasks pandas metadata is invalid: {exc}") from exc
    if task_index_metadata.get("index_columns") != ["task"] or task_table.column("task_index")[0].as_py() != 0:
        raise LeRobotProjectionError("LeRobot task index metadata is inconsistent")

    episodes = pq.read_table(str(destination / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))
    if episodes.num_rows != agent_count:
        raise LeRobotProjectionError("LeRobot episode metadata count is inconsistent")
    episode_rows = episodes.to_pylist()
    mapping = manifest.get("agent_episodes")
    if not isinstance(mapping, list) or len(mapping) != agent_count:
        raise LeRobotProjectionError("LeRobot fleet group mapping is incomplete")
    for agent_id, (episode, group) in enumerate(zip(episode_rows, mapping, strict=True)):
        expected_from = agent_id * steps
        expected_to = (agent_id + 1) * steps
        if (
            episode.get("episode_index") != agent_id
            or episode.get("length") != steps
            or episode.get("dataset_from_index") != expected_from
            or episode.get("dataset_to_index") != expected_to
            or not isinstance(group, Mapping)
            or group.get("agent_id") != agent_id
            or group.get("episode_index") != agent_id
            or group.get("dataset_from_index") != expected_from
            or group.get("dataset_to_index") != expected_to
        ):
            raise LeRobotProjectionError(f"LeRobot episode/group mapping is invalid for agent {agent_id}")

    data_path = destination / "data" / "chunk-000" / "file-000.parquet"
    data_file = pq.ParquetFile(str(data_path))
    if data_file.metadata.num_rows != frame_count or data_file.metadata.num_row_groups != agent_count:
        raise LeRobotProjectionError("LeRobot data file must have one complete row group per agent")
    expected_columns = list(_features())
    if [field.name for field in data_file.schema_arrow] != expected_columns:
        raise LeRobotProjectionError("LeRobot data columns do not match meta/info.json")
    reference_native_time: tuple[np.ndarray, np.ndarray] | None = None
    for agent_id in range(agent_count):
        table = data_file.read_row_group(agent_id)
        if table.num_rows != steps:
            raise LeRobotProjectionError(f"LeRobot row group {agent_id} has an invalid length")
        agent_ids = _column_numpy(table, "rivermark.agent_id", np.dtype("<i8"))
        episode_ids = _column_numpy(table, "episode_index", np.dtype("<i8"))
        frame_indices = _column_numpy(table, "frame_index", np.dtype("<i8"))
        global_indices = _column_numpy(table, "index", np.dtype("<i8"))
        command_time = _column_numpy(table, "rivermark.command_time_ns", np.dtype("<i8"))
        effective_time = _column_numpy(table, "rivermark.effective_time_ns", np.dtype("<i8"))
        standard_time = _column_numpy(table, "timestamp", np.dtype("<f4"))
        if (
            not np.all(agent_ids == agent_id)
            or not np.all(episode_ids == agent_id)
            or not np.array_equal(frame_indices, np.arange(steps, dtype="<i8"))
            or not np.array_equal(global_indices, np.arange(agent_id * steps, (agent_id + 1) * steps, dtype="<i8"))
            or not np.allclose(standard_time, np.arange(steps, dtype="<f4") / np.float32(fps), rtol=0, atol=1e-7)
            or np.any(command_time >= effective_time)
        ):
            raise LeRobotProjectionError(f"LeRobot frame semantics are invalid for agent {agent_id}")
        if reference_native_time is None:
            reference_native_time = (command_time, effective_time)
        elif not np.array_equal(command_time, reference_native_time[0]) or not np.array_equal(
            effective_time, reference_native_time[1]
        ):
            raise LeRobotProjectionError("LeRobot agent timelines do not reconstruct a synchronized fleet")
        for name in ("observation.state", "action", "rivermark.applied_thrust_n", "rivermark.desired_state"):
            values = np.asarray(table.column(name).to_pylist(), dtype=np.float32)
            if values.shape != (steps, _features()[name]["shape"][0]) or not np.all(np.isfinite(values)):
                raise LeRobotProjectionError(f"LeRobot feature is malformed: {name}")
    return {
        "schema": LEROBOT_PROJECTION_SCHEMA,
        "status": "valid",
        "fleet_episode_id": manifest.get("fleet_episode_id"),
        "agent_episode_count": agent_count,
        "frame_count": frame_count,
        "fps": fps,
        "source_projection_manifest_sha256": manifest.get("source_projection_manifest_sha256"),
        "group_manifest_sha256": sha256_file(manifest_path),
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _integer_value(value: Any, label: str) -> int:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int):
        raise LeRobotProjectionError(f"upstream LeRobot reader returned a non-integer {label}")
    return value


def verify_with_upstream_lerobot(root: Path, upstream_source: Path) -> dict[str, Any]:
    """Read the projection with the reviewed local LeRobot source tree.

    This check is intentionally opt-in.  Rivermark does not vendor LeRobot or
    its heavyweight dataset dependencies, and a structural projection check is
    not a substitute for loading through the upstream reader.
    """

    report = verify_lerobot_projection(root)
    source = Path(upstream_source).expanduser().resolve()
    package_root = source / "src"
    reader_module = package_root / "lerobot" / "datasets" / "lerobot_dataset.py"
    if not reader_module.is_file():
        raise LeRobotProjectionError(
            "upstream_source must contain src/lerobot/datasets/lerobot_dataset.py from LeRobot 0.6.1"
        )
    existing = sys.modules.get("lerobot")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if not isinstance(existing_file, str) or not Path(existing_file).resolve().is_relative_to(package_root):
            raise LeRobotProjectionError("another LeRobot package is already imported; run upstream verification in a clean process")
    inserted = False
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
        inserted = True
    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except (ImportError, OSError) as exc:
            raise LeRobotProjectionError(
                "upstream LeRobot reader is unavailable; use Python >=3.12 with the LeRobot 0.6.1 'dataset' extra "
                "(including datasets, pandas, pyarrow>=21, torch, and a supported decoder)"
            ) from exc
        destination = Path(root).expanduser().resolve()
        dataset = LeRobotDataset(
            repo_id="rivermark/development-state-action",
            root=destination,
            revision=LEROBOT_FORMAT_VERSION,
            download_videos=False,
        )
        expected_frames = int(report["frame_count"])
        expected_agents = int(report["agent_episode_count"])
        frames_per_agent = expected_frames // expected_agents
        if len(dataset) != expected_frames or dataset.num_episodes != expected_agents:
            raise LeRobotProjectionError("upstream LeRobot reader count differs from the fleet group manifest")
        for agent_id in range(expected_agents):
            item = dataset.get_raw_item(agent_id * frames_per_agent)
            if not isinstance(item, Mapping):
                raise LeRobotProjectionError("upstream LeRobot reader returned a non-mapping frame")
            required = {"observation.state", "action", "rivermark.command_time_ns", "rivermark.effective_time_ns"}
            if not required.issubset(item):
                raise LeRobotProjectionError("upstream LeRobot reader omitted a required Rivermark state/action field")
            if _integer_value(item["episode_index"], "episode_index") != agent_id:
                raise LeRobotProjectionError(f"upstream LeRobot reader remapped agent episode {agent_id}")
            if _integer_value(item["rivermark.agent_id"], "rivermark.agent_id") != agent_id:
                raise LeRobotProjectionError(f"upstream LeRobot reader remapped native agent {agent_id}")
        return {
            **report,
            "upstream_reader": {
                "status": "passed",
                "source_version_reviewed": LEROBOT_SOURCE_VERSION,
                "format_version": LEROBOT_FORMAT_VERSION,
            },
        }
    finally:
        if inserted:
            try:
                sys.path.remove(str(package_root))
            except ValueError:  # pragma: no cover - defensive cleanup
                pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    project = subparsers.add_parser("project", help="project a development Parquet capture")
    project.add_argument("source_root", type=Path)
    project.add_argument("output_root", type=Path)
    project.add_argument("--fps", type=int)
    project.add_argument("--timestamp-tolerance-s", type=float, default=1e-4)
    project.add_argument("--task-description", default="Follow the frozen public Rivermark City-Lite route.")
    verify = subparsers.add_parser("verify", help="verify an existing LeRobot projection")
    verify.add_argument("root", type=Path)
    verify.add_argument("--source-root", type=Path)
    upstream_verify = subparsers.add_parser("upstream-verify", help="read through a local LeRobot 0.6.1 source tree")
    upstream_verify.add_argument("root", type=Path)
    upstream_verify.add_argument("--upstream-source", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "project":
            result = project_development_parquet_to_lerobot(
                args.source_root,
                args.output_root,
                fps=args.fps,
                timestamp_tolerance_s=args.timestamp_tolerance_s,
                task_description=args.task_description,
            )
            report = {
                "status": "projected",
                "output_root": str(result.output_root),
                "fleet_episode_id": result.fleet_episode_id,
                "agent_episode_count": result.agent_episode_count,
                "frame_count": result.frame_count,
                "fps": result.fps,
                "source_projection_manifest_sha256": result.source_projection_manifest_sha256,
                "group_manifest_sha256": result.group_manifest_sha256,
                "claim_boundary": _CLAIM_BOUNDARY,
            }
        elif args.command == "verify":
            report = verify_lerobot_projection(args.root, source_root=args.source_root)
        else:
            report = verify_with_upstream_lerobot(args.root, args.upstream_source)
    except (OSError, LeRobotProjectionError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LEROBOT_FORMAT_VERSION",
    "LEROBOT_PROJECTION_SCHEMA",
    "LEROBOT_SOURCE_VERSION",
    "LeRobotProjectionError",
    "LeRobotProjectionResult",
    "project_development_parquet_to_lerobot",
    "verify_lerobot_projection",
    "verify_with_upstream_lerobot",
]
