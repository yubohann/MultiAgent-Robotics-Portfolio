"""Compose synchronized audit views from one completed HM3D PhysX replay.

The input MP4s must all be trace-driven renderings produced by
``record_hm3d_physx_trace_replay.py`` from the same engineering-smoke source
record. This utility only combines decoded pixels. It never changes a PhysX
trace or supplies observations to the exploration system.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-record", required=True, type=Path)
    parser.add_argument("--global-video", required=True, type=Path)
    parser.add_argument("--uav0-video", required=True, type=Path)
    parser.add_argument("--uav1-video", required=True, type=Path)
    parser.add_argument("--uav2-video", required=True, type=Path)
    parser.add_argument("--uav3-video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _source_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "P07_EXECUTION_SMOKE_COMPLETE"
        or payload.get("synthetic") is not False
        or payload.get("record_purpose") != "engineering_smoke"
        or payload.get("execution_profile", {}).get("evidence_class")
        != "real_isaac_physx_cf2x"
    ):
        raise ValueError("mosaic source is not a completed real engineering-smoke PhysX record")
    return {
        "scene_id": payload["scene_id"],
        "controller_id": payload["controller_id"],
        "elapsed_physics_s": float(payload["elapsed_physics_s"]),
        "decision_count": int(payload["decision_count"]),
        "runtime_record_sha256": payload["runtime_record_sha256"],
        "file_sha256": _sha256(path),
    }


def _playback_timestamp_s(frame_index: int, frame_count: int, horizon_s: float) -> float:
    """Mirror the recorder's inclusive endpoint mapping exactly."""

    if frame_count <= 0 or not 0 <= frame_index < frame_count:
        raise ValueError("invalid replay frame index or frame count")
    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("invalid replay horizon")
    if frame_count == 1:
        return 0.0
    return horizon_s * frame_index / float(frame_count - 1)


def _canonical_mapping_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(rows, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_frame_time_mapping(
    mapping: Any,
    *,
    frame_count: int,
    horizon_s: float,
    source_trace_timestamps: tuple[float, ...],
) -> dict[str, Any]:
    """Validate that one replay manifest really samples this source trace."""

    payload = _require_dict(mapping, "replay frame-time mapping")
    if payload.get("schema_version") != "hm3d-physx-replay-frame-time-mapping-v1":
        raise ValueError("replay manifest has an unsupported frame-time mapping schema")
    if payload.get("sampling") != "zero_order_hold_final_sample_at_duplicate_boundary_v1":
        raise ValueError("replay manifest has an unsupported trace sampling rule")
    if int(payload.get("output_frame_count", -1)) != frame_count:
        raise ValueError("replay manifest frame-time mapping count disagrees with the video")
    if int(payload.get("source_trace_frame_count", -1)) != len(source_trace_timestamps):
        raise ValueError("replay manifest source trace count disagrees with the record")
    window = payload.get("playback_window_s")
    if (
        not isinstance(window, list | tuple)
        or len(window) != 2
        or abs(float(window[0])) > 1.0e-9
        or abs(float(window[1]) - horizon_s) > 1.0e-9
    ):
        raise ValueError("replay manifest frame-time mapping has a wrong playback window")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != frame_count:
        raise ValueError("replay manifest frame-time mapping rows are incomplete")
    for output_frame_index, row in enumerate(rows):
        row_dict = _require_dict(row, "replay frame-time mapping row")
        if int(row_dict.get("output_frame_index", -1)) != output_frame_index:
            raise ValueError("replay manifest frame-time mapping has non-sequential output frames")
        playback_timestamp_s = float(row_dict.get("playback_timestamp_s"))
        expected_timestamp_s = _playback_timestamp_s(
            output_frame_index, frame_count, horizon_s
        )
        if abs(playback_timestamp_s - expected_timestamp_s) > 1.0e-9:
            raise ValueError("replay manifest frame-time mapping has a mismatched playback timestamp")
        source_index = int(row_dict.get("source_trace_frame_index", -1))
        expected_source_index = min(
            len(source_trace_timestamps) - 1,
            bisect_right(source_trace_timestamps, playback_timestamp_s + 1.0e-9) - 1,
        )
        if source_index != expected_source_index:
            raise ValueError("replay manifest frame-time mapping selects the wrong source sample")
        if abs(float(row_dict.get("source_trace_timestamp_s")) - source_trace_timestamps[source_index]) > 1.0e-9:
            raise ValueError("replay manifest frame-time mapping has a wrong source timestamp")
    expected_digest = _canonical_mapping_sha256(rows)
    if payload.get("rows_sha256") != expected_digest:
        raise ValueError("replay manifest frame-time mapping digest does not match its rows")
    return {
        "schema_version": payload["schema_version"],
        "sampling": payload["sampling"],
        "output_frame_count": frame_count,
        "source_trace_frame_count": len(source_trace_timestamps),
        "playback_window_s": [0.0, horizon_s],
        "rows_sha256": expected_digest,
        "rows": rows,
    }


def _load_and_validate_replay_manifest(
    *,
    label: str,
    video_path: Path,
    source: dict[str, Any],
    agent_order: tuple[str, ...],
    source_trace_timestamps: tuple[float, ...],
) -> dict[str, Any]:
    """Reject a video whose sidecar is not bound to this trace and view role."""

    manifest_path = _require_file(video_path.with_suffix(".manifest.json"), f"{label} replay manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "hm3d-physx-trace-replay-v2":
        raise ValueError(f"{label} replay must use hm3d-physx-trace-replay-v2")
    if payload.get("status") != "PHYSX_TRACE_REPLAY_COMPLETE":
        raise ValueError(f"{label} replay manifest is incomplete")
    source_record = _require_dict(payload.get("source_record"), f"{label} replay source record")
    expected_source_pairs = {
        "file_sha256": source["file_sha256"],
        "runtime_record_sha256": source["runtime_record_sha256"],
        "scene_id": source["scene_id"],
        "controller_id": source["controller_id"],
    }
    for key, expected in expected_source_pairs.items():
        if source_record.get(key) != expected:
            raise ValueError(f"{label} replay is bound to a different {key}")
    if abs(float(source_record.get("trace_horizon_s")) - float(source["elapsed_physics_s"])) > 1.0e-9:
        raise ValueError(f"{label} replay has a different trace horizon")
    if tuple(source_record.get("agent_order", ())) != agent_order:
        raise ValueError(f"{label} replay has a different agent order")
    if int(source_record.get("trace_frame_count", -1)) != len(source_trace_timestamps):
        raise ValueError(f"{label} replay has a different source trace length")
    video = _require_dict(payload.get("video"), f"{label} replay video")
    if Path(str(video.get("path", ""))).expanduser().resolve() != video_path:
        raise ValueError(f"{label} replay manifest path does not match its supplied video")
    actual_video_hash = _sha256(video_path)
    if video.get("sha256") != actual_video_hash:
        raise ValueError(f"{label} replay video hash does not match its manifest")
    frames = int(video.get("frames", -1))
    fps = float(video.get("fps", 0.0))
    if frames <= 0 or fps <= 0.0:
        raise ValueError(f"{label} replay has invalid frame metadata")
    mapping = _validate_frame_time_mapping(
        video.get("frame_time_mapping"),
        frame_count=frames,
        horizon_s=float(source["elapsed_physics_s"]),
        source_trace_timestamps=source_trace_timestamps,
    )
    view = _require_dict(payload.get("view"), f"{label} replay view")
    camera_binding = _require_dict(payload.get("camera_audit_binding"), f"{label} camera audit binding")
    for key, expected in (
        ("source_record_file_sha256", source["file_sha256"]),
        ("source_record_runtime_sha256", source["runtime_record_sha256"]),
        ("source_trace_frame_count", len(source_trace_timestamps)),
        ("camera_pose_source", "replay_frame_time_mapping_v1"),
    ):
        if camera_binding.get(key) != expected:
            raise ValueError(f"{label} camera audit binding is not attached to this trace")
    if camera_binding.get("view_mode") != view.get("mode"):
        raise ValueError(f"{label} camera audit view mode disagrees with its replay view")
    if label == "global":
        if view.get("mode") != "global" or view.get("fpv_agent") is not None or view.get("follow_agent") is not None:
            raise ValueError("global input must be the global review view")
    else:
        expected_agent = int(label.removeprefix("uav").removesuffix("_follow"))
        if view.get("mode") != "follow" or int(view.get("follow_agent", -1)) != expected_agent:
            raise ValueError(f"{label} input must be follow view for UAV {expected_agent}")
        certificate = _require_dict(
            camera_binding.get("follow_clearance_certificate"),
            f"{label} follow camera certificate",
        )
        if certificate.get("status") != "FOLLOW_CAMERA_CLEARANCE_CERTIFIED":
            raise ValueError(f"{label} follow camera is not clearance-certified")
        if certificate.get("agent_id") != agent_order[expected_agent]:
            raise ValueError(f"{label} follow camera certificate refers to a different UAV")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "video_sha256": actual_video_hash,
        "frames": frames,
        "fps": fps,
        "mapping": mapping,
        "view": view,
        "camera_audit_binding": camera_binding,
    }


def _load_trace_timeline(
    path: Path,
) -> tuple[
    list[dict[str, Any]],
    tuple[float, ...],
    tuple[str, ...],
    tuple[float, float, float, float, float, float],
]:
    """Load only realised visual-audit samples for the dynamic 3-D trace panel."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("trace source omits bootstrap")
    segments: list[tuple[float, Any]] = [(0.0, bootstrap.get("physics_visualization_trace"))]
    previous_elapsed_s = float(bootstrap["elapsed_physics_s"])
    for decision in payload.get("decisions", []):
        if not isinstance(decision, dict):
            raise ValueError("trace source has a malformed decision")
        calibration = decision.get("execution_calibration")
        segments.append(
            (previous_elapsed_s, calibration.get("physics_visualization_trace") if isinstance(calibration, dict) else None)
        )
        previous_elapsed_s = float(decision["elapsed_physics_s"])
    terminal_tail = payload.get("execution", {}).get("terminal_budget_tail")
    if isinstance(terminal_tail, dict):
        segments.append(
            (float(terminal_tail["executed_from_episode_s"]), terminal_tail.get("physics_visualization_trace"))
        )

    rows: list[dict[str, Any]] = []
    agent_order: tuple[str, ...] | None = None
    for offset_s, trace in segments:
        if not isinstance(trace, dict) or trace.get("purpose") != "engineering_visual_audit_only":
            raise ValueError("trace source has a missing or invalid audit-only visual trace segment")
        samples = trace.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("trace source has an empty visual trace segment")
        for sample in samples:
            raw_agents = sample.get("agents") if isinstance(sample, dict) else None
            if not isinstance(raw_agents, list) or not raw_agents:
                raise ValueError("trace source has a malformed agent sample")
            states: dict[str, dict[str, float | bool | tuple[float, float, float]]] = {}
            order = []
            for raw_agent in raw_agents:
                if not isinstance(raw_agent, dict):
                    raise ValueError("trace source has a malformed agent state")
                agent_id = raw_agent.get("agent_id")
                raw_position = raw_agent.get("position_m")
                if not isinstance(agent_id, str) or not isinstance(raw_position, list | tuple) or len(raw_position) != 3:
                    raise ValueError("trace source has an invalid agent identity or position")
                position = tuple(float(value) for value in raw_position)
                if not all(value == value and abs(value) < float("inf") for value in position):
                    raise ValueError("trace source has a non-finite agent position")
                states[agent_id] = {
                    "position_m": position,
                    "linear_speed_mps": float(raw_agent.get("linear_speed_mps", 0.0)),
                    "reservation_waiting": bool(raw_agent.get("reservation_waiting")),
                }
                order.append(agent_id)
            order_tuple = tuple(order)
            if agent_order is None:
                agent_order = order_tuple
            if order_tuple != agent_order:
                raise ValueError("trace source changes the agent order")
            rows.append({"timestamp_s": offset_s + float(sample["physics_timestamp_s"]), "states": states})
    if agent_order is None or len(agent_order) != 4:
        raise ValueError("trajectory mosaic requires exactly four traced UAVs")
    rows.sort(key=lambda row: float(row["timestamp_s"]))
    timestamps = tuple(float(row["timestamp_s"]) for row in rows)
    all_positions = [
        state["position_m"]
        for row in rows
        for state in row["states"].values()
    ]
    lower = tuple(min(position[axis] for position in all_positions) for axis in range(3))
    upper = tuple(max(position[axis] for position in all_positions) for axis in range(3))
    centre = tuple((low + high) * 0.5 for low, high in zip(lower, upper, strict=True))
    # Equal-scale coordinates make the vertical contribution visible and avoid
    # turning a narrow XY spread into a deceptively flat trace.
    half_span_m = max(
        0.75,
        max(high - low for low, high in zip(lower, upper, strict=True)) * 0.5 + 0.35,
    )
    bounds_min = tuple(value - half_span_m for value in centre)
    bounds_max = tuple(value + half_span_m for value in centre)
    return rows, timestamps, agent_order, (*bounds_min, *bounds_max)


def _draw_trajectory_panel(
    frame: Any,
    source: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    timestamps: tuple[float, ...],
    agent_order: tuple[str, ...],
    xyz_bounds: tuple[float, float, float, float, float, float],
    timestamp_s: float,
) -> None:
    import cv2

    frame[:] = (252, 253, 254)
    height, width = frame.shape[:2]
    current_index = max(0, min(len(rows) - 1, bisect_right(timestamps, timestamp_s) - 1))
    current_row = rows[current_index]
    cv2.putText(
        frame,
        "3D TRAJECTORIES",
        (30, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (38, 45, 52),
        2,
    )
    left, top, right, bottom = 28, 64, width - 28, height - 32
    cv2.rectangle(frame, (left, top), (right, bottom), (215, 220, 224), 1, cv2.LINE_AA)
    lower_x, lower_y, lower_z, upper_x, upper_y, upper_z = xyz_bounds
    centre = (
        (lower_x + upper_x) * 0.5,
        (lower_y + upper_y) * 0.5,
        (lower_z + upper_z) * 0.5,
    )
    half_span = max((upper_x - lower_x) * 0.5, 1.0e-9)
    azimuth_rad = math.radians(-48.0)
    elevation_rad = math.radians(27.0)
    plot_centre = ((left + right) * 0.5, (top + bottom) * 0.54)
    scale = min(right - left, bottom - top) * 0.66

    def project(position: tuple[float, float, float]) -> tuple[int, int]:
        x = (position[0] - centre[0]) / half_span
        y = (position[1] - centre[1]) / half_span
        z = (position[2] - centre[2]) / half_span
        screen_x = math.cos(azimuth_rad) * x + math.sin(azimuth_rad) * y
        horizontal_depth = -math.sin(azimuth_rad) * x + math.cos(azimuth_rad) * y
        screen_y = -math.sin(elevation_rad) * horizontal_depth + math.cos(elevation_rad) * z
        depth = math.cos(elevation_rad) * horizontal_depth + math.sin(elevation_rad) * z
        perspective = 1.0 / (2.75 + 0.16 * depth)
        return (
            int(round(plot_centre[0] + screen_x * scale * perspective)),
            int(round(plot_centre[1] - screen_y * scale * perspective)),
        )

    def draw_segment(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        cv2.line(frame, project(start), project(end), color, thickness, cv2.LINE_AA)

    cube = tuple(
        (x, y, z)
        for z in (lower_z, upper_z)
        for y in (lower_y, upper_y)
        for x in (lower_x, upper_x)
    )
    cube_edges = (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    )
    for start_index, end_index in cube_edges:
        draw_segment(cube[start_index], cube[end_index], (190, 198, 204))
    for fraction in (0.2, 0.4, 0.6, 0.8):
        x = lower_x + (upper_x - lower_x) * fraction
        y = lower_y + (upper_y - lower_y) * fraction
        draw_segment((x, lower_y, lower_z), (x, upper_y, lower_z), (224, 228, 232))
        draw_segment((lower_x, y, lower_z), (upper_x, y, lower_z), (224, 228, 232))
    origin = (lower_x, lower_y, lower_z)
    axis_ends = ((upper_x, lower_y, lower_z), (lower_x, upper_y, lower_z), (lower_x, lower_y, upper_z))
    axis_names = ("X", "Y", "Z")
    for name, end in zip(axis_names, axis_ends, strict=True):
        draw_segment(origin, end, (92, 104, 114), 2)
        label = project(end)
        cv2.putText(
            frame,
            name,
            (label[0] + 6, label[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (60, 72, 82),
            1,
            cv2.LINE_AA,
        )

    colors_bgr = ((234, 142, 45), (71, 178, 89), (70, 92, 211), (45, 161, 226))
    for agent_index, agent_id in enumerate(agent_order):
        color = colors_bgr[agent_index]
        trail = [
            project(row["states"][agent_id]["position_m"])
            for row in rows[: current_index + 1]
        ]
        for start, end in zip(trail, trail[1:]):
            cv2.line(frame, start, end, color, 4, cv2.LINE_AA)
        state = current_row["states"][agent_id]
        marker = project(state["position_m"])
        start_marker = trail[0]
        cv2.circle(frame, start_marker, 6, color, 1, cv2.LINE_AA)
        cv2.circle(frame, marker, 10, color, -1, cv2.LINE_AA)
        cv2.circle(frame, marker, 12, (255, 255, 255), 2, cv2.LINE_AA)
        label_y = 82 + agent_index * 23
        label_x = right - 145
        cv2.line(frame, (label_x, label_y - 5), (label_x + 20, label_y - 5), color, 4, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"UAV {agent_index}",
            (label_x + 26, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (50, 59, 67),
            1,
            cv2.LINE_AA,
        )


def main() -> int:
    import cv2
    import numpy as np

    args = _parse_args()
    trace_record = _require_file(args.trace_record, "PhysX trace record")
    source = _source_summary(trace_record)
    trace_rows, trace_timestamps, agent_order, xyz_bounds = _load_trace_timeline(trace_record)
    names_and_paths = (
        ("global", _require_file(args.global_video, "global replay")),
        ("uav0_follow", _require_file(args.uav0_video, "UAV0 replay")),
        ("uav1_follow", _require_file(args.uav1_video, "UAV1 replay")),
        ("uav2_follow", _require_file(args.uav2_video, "UAV2 replay")),
        ("uav3_follow", _require_file(args.uav3_video, "UAV3 replay")),
    )
    replay_audits = {
        name: _load_and_validate_replay_manifest(
            label=name,
            video_path=path,
            source=source,
            agent_order=agent_order,
            source_trace_timestamps=trace_timestamps,
        )
        for name, path in names_and_paths
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {output}")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must have an .mp4 suffix")

    captures = [(name, path, cv2.VideoCapture(str(path))) for name, path in names_and_paths]
    try:
        metadata = []
        for name, path, capture in captures:
            if not capture.isOpened():
                raise RuntimeError(f"OpenCV cannot open {name} input: {path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_count <= 0 or fps <= 0.0 or width <= 0 or height <= 0:
                raise RuntimeError(f"invalid {name} input metadata")
            audit = replay_audits[name]
            if frame_count != audit["frames"] or abs(fps - audit["fps"]) > 1.0e-6:
                raise RuntimeError(
                    f"{name} decoded metadata disagrees with its validated replay manifest"
                )
            metadata.append(
                {"name": name, "path": str(path), "frames": frame_count, "fps": fps,
                 "width": width, "height": height, "sha256": audit["video_sha256"],
                 "replay_manifest_path": audit["manifest_path"],
                 "replay_manifest_sha256": audit["manifest_sha256"],
                 "frame_time_mapping_sha256": audit["mapping"]["rows_sha256"],
                 "view": audit["view"]}
            )
        reference = metadata[0]
        if any(
            item["frames"] != reference["frames"]
            or abs(item["fps"] - reference["fps"]) > 1.0e-6
            or item["width"] != reference["width"]
            or item["height"] != reference["height"]
            for item in metadata[1:]
        ):
            raise RuntimeError("all replay views must have exactly matching frame, FPS, and dimensions")
        width, height, fps, frame_count = (
            reference["width"],
            reference["height"],
            reference["fps"],
            reference["frames"],
        )
        reference_mapping = replay_audits["global"]["mapping"]
        if any(
            replay_audits[item["name"]]["mapping"]["rows_sha256"]
            != reference_mapping["rows_sha256"]
            for item in metadata[1:]
        ):
            raise RuntimeError("all replay views must have the same validated frame-time mapping")
        temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
        writer = cv2.VideoWriter(
            str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3, height * 2)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not initialise the mosaic MP4 writer")
        try:
            for frame_index in range(frame_count):
                decoded = []
                for name, _, capture in captures:
                    ok, frame = capture.read()
                    if not ok or frame is None or frame.shape[:2] != (height, width):
                        raise RuntimeError(f"{name} failed to decode frame {frame_index}")
                    decoded.append(frame)
                trajectory_panel = np.empty((height, width, 3), dtype=np.uint8)
                _draw_trajectory_panel(
                    trajectory_panel,
                    source,
                    rows=trace_rows,
                    timestamps=trace_timestamps,
                    agent_order=agent_order,
                    xyz_bounds=xyz_bounds,
                    timestamp_s=float(
                        reference_mapping["rows"][frame_index]["playback_timestamp_s"]
                    ),
                )
                mosaic = np.vstack((np.hstack((decoded[0], decoded[1], decoded[2])),
                                    np.hstack((decoded[3], decoded[4], trajectory_panel))))
                writer.write(mosaic)
        finally:
            writer.release()
        decoded = cv2.VideoCapture(str(temporary))
        decoded_count = 0
        means: list[float] = []
        while True:
            ok, frame = decoded.read()
            if not ok:
                break
            decoded_count += 1
            means.append(float(frame.mean()))
        decoded.release()
        if decoded_count != frame_count or not means or min(means) <= 1.0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("mosaic MP4 failed decode read-back verification")
        os.replace(temporary, output)
    finally:
        for _, _, capture in captures:
            capture.release()

    manifest = {
        "schema_version": "hm3d-physx-trace-mosaic-v2",
        "status": "PHYSX_TRACE_MOSAIC_COMPLETE",
        "formal_result": False,
        "render_role": "human_audit_only",
        "source_record": {"path": str(trace_record), **source},
        "inputs": metadata,
        "frame_time_mapping": reference_mapping,
        "input_compatibility": {
            "status": "SOURCE_AND_CAMERA_AUDIT_BINDINGS_VALIDATED",
            "source_record_file_sha256": source["file_sha256"],
            "runtime_record_sha256": source["runtime_record_sha256"],
            "frame_time_mapping_sha256": reference_mapping["rows_sha256"],
            "validated_input_count": len(metadata),
        },
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "frames": frame_count,
            "fps": fps,
            "width": width * 3,
            "height": height * 2,
            "decoded_mean_min": min(means),
            "decoded_mean_max": max(means),
            "decoder": "opencv-readback-verified",
        },
        "layout": "top: global, UAV0 follow, UAV1 follow; bottom: UAV2 follow, UAV3 follow, dynamic actual-XYZ audit trace",
        "caveat": (
            "The mosaic combines trace-driven audit views. It is excluded from candidate "
            "selection, control, sensing, rewards, training, QD, OGFR, and formal metrics."
        ),
    }
    manifest_path = output.with_suffix(".manifest.json")
    _write_json(manifest_path, manifest)
    print(json.dumps({"video": str(output), "manifest": str(manifest_path), "frames": frame_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
