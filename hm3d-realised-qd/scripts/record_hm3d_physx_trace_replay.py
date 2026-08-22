"""Render an auditable HM3D replay from one real CF2X/PhysX trace.

This tool is intentionally separate from the target-free exploration runtime.
It reads only the optional post-step audit telemetry emitted by
``run_hm3d_p07_exploration_episode.py --visualization-trace-hz`` and never
creates observations, alters a physical state, or influences a result.  The
mesh render is a human-inspection aid; all vehicle positions and yaw headings
come from the already-realised PhysX trace.

Run with the configured Isaac Lab interpreter, for example::

    python scripts/record_hm3d_physx_trace_replay.py \
      --trace-record E:/asset/.../00803_physx_visual_source.json \
      --scene-usd E:/asset/hm3d_video_00803/hm3d_00803.usd \
      --output E:/asset/.../00803_global.mp4
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
DRONE_ROOT = ROOT.parents[1]
DRONE_ASSET = DRONE_ROOT / "assets" / "new" / "cf2x.usd"
LOCATOR_COLORS = (
    (0.27, 0.69, 1.00),
    (0.27, 0.91, 0.53),
    (1.00, 0.36, 0.36),
    (0.91, 0.65, 0.23),
    (0.93, 0.33, 0.83),
    (0.29, 0.85, 0.91),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # AppLauncher calls parse_known_args() while registering its own flags.
    # Keep these as post-parse requirements so ``--help`` can remain useful
    # without a pretend trace record or USD path.
    parser.add_argument("--trace-record", type=Path)
    parser.add_argument("--scene-usd", type=Path)
    parser.add_argument("--cf2x-usd", type=Path, default=DRONE_ASSET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--global-focal-length-mm",
        type=float,
        default=36.0,
        help=(
            "focal length used only by the global review camera; a tighter default "
            "keeps the complete scene readable while making four visual UAVs visible"
        ),
    )
    parser.add_argument(
        "--global-camera-framing",
        choices=("trace_local", "scene_exterior"),
        default="trace_local",
        help=(
            "trace_local orbits the realised trace bounds. scene_exterior uses a stable elevated "
            "view outside the composed scene envelope while keeping the realised trace as its focus"
        ),
    )
    parser.add_argument(
        "--visual-locator-radius-m",
        type=float,
        default=0.035,
        help=(
            "radius of the coloured visual-only beacon placed above each true-scale CF2X; "
            "it has no physics or collision role"
        ),
    )
    parser.add_argument(
        "--visual-locator-height-m",
        type=float,
        default=0.10,
        help="height of each visual-only UAV beacon in metres",
    )
    parser.add_argument(
        "--visual-rotor-hz",
        type=float,
        default=12.0,
        help=(
            "visual-only rotor phase frequency used when the source trace has no motor telemetry; "
            "this never changes a replayed pose or any physical evidence"
        ),
    )
    parser.add_argument(
        "--show-vehicles-in-fpv",
        action="store_true",
        help=(
            "show visual UAV bodies and locators in FPV captures. The default hides them "
            "so FPV is an environment-only replay view."
        ),
    )
    parser.add_argument(
        "--view-mode",
        choices=("global", "fpv", "follow"),
        default="global",
        help=(
            "global is a bounds-derived review orbit; fpv is a replay camera; "
            "follow is a fixed-offset third-person camera."
        ),
    )
    parser.add_argument(
        "--fpv-agent",
        type=int,
        help="zero-based agent ID for --view-mode fpv",
    )
    parser.add_argument(
        "--follow-agent",
        type=int,
        help="zero-based agent ID for --view-mode follow",
    )
    parser.add_argument(
        "--follow-camera-radius-m",
        type=float,
        default=0.16,
        help=(
            "maximum distance from the realised UAV root pose for a follow-camera eye or "
            "target ray. Follow rendering is refused unless the recorded static-collision "
            "clearance certificate leaves this radius plus the safety margin free; that "
            "does not prove render-mesh line-of-sight."
        ),
    )
    parser.add_argument(
        "--follow-camera-safety-margin-m",
        type=float,
        default=0.02,
        help="additional static-clearance reserve required around a follow-camera ray",
    )
    parser.add_argument(
        "--scene-review-mode",
        choices=("opaque", "ghost", "trajectory_only"),
        default="opaque",
        help=(
            "opaque renders the source scan normally. ghost replaces source scan materials "
            "with a semi-transparent audit-only material so scan-shell fragments cannot hide "
            "the replayed vehicles. trajectory_only hides the source scan and adds a neutral "
            "visual-only backdrop for an unobstructed trajectory review; neither mode is sensor imagery."
        ),
    )
    parser.add_argument(
        "--ghost-scene-opacity",
        type=float,
        default=0.18,
        help=(
            "opacity for --scene-review-mode ghost, constrained to (0, 1); this changes only "
            "the audit render material and never the source trace or simulator."
        ),
    )
    parser.add_argument(
        "--trajectory-overview",
        type=Path,
        help=(
            "optional PNG path for a white-background XYZ overview of the actual replayed "
            "root trajectories; it is an audit artifact, not a formal metric plot"
        ),
    )
    parser.add_argument(
        "--progress",
        type=Path,
        help="optional JSON file updated after every recording initialisation step",
    )
    parser.add_argument("--overwrite", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    # This command always captures an RTX-rendered camera stream.  Leaving the
    # AppLauncher camera flag at its generic default selects the non-rendering
    # headless Kit experience, which can make CaptureExtension report progress
    # while creating no usable image sequence.
    parser.set_defaults(enable_cameras=True)
    args = parser.parse_args()
    missing = [
        option
        for option, value in (
            ("--trace-record", args.trace_record),
            ("--scene-usd", args.scene_usd),
            ("--output", args.output),
        )
        if value is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    return args


# Keep pure trace-validation and plotting helpers importable by unit tests. Isaac
# is launched only by ``main`` after the command line has been accepted.
ARGS: argparse.Namespace | None = None
SIMULATION_APP: Any | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_input(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _progress(status: str, **details: Any) -> None:
    """Persist progress before an Isaac/RTX operation that may terminate Kit."""

    if ARGS is None or ARGS.progress is None:
        return
    path = ARGS.progress.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"status": status, **details})
    print(f"[HM3D PhysX trace replay] progress={status}", flush=True)


@dataclass(frozen=True)
class ReplayAgentState:
    position_m: tuple[float, float, float]
    yaw_deg: float
    linear_speed_mps: float
    reservation_waiting: bool
    transit_completed: bool
    failed: bool


@dataclass(frozen=True)
class ReplayFrame:
    timestamp_s: float
    states_by_agent: dict[str, ReplayAgentState]
    minimum_inter_agent_distance_m: float


@dataclass(frozen=True)
class FollowCameraClearanceEvidence:
    """Static-clearance lower bounds already measured for the realised root trace."""

    required_root_clearance_m: float
    minimum_root_clearance_by_agent_m: dict[str, float]
    source_phase_minimum_clearance_m: float


@dataclass(frozen=True)
class PhysxTraceReplay:
    scene_id: str
    controller_id: str
    source_record_file_sha256: str
    source_record_runtime_sha256: str
    horizon_s: float
    agent_order: tuple[str, ...]
    frames: tuple[ReplayFrame, ...]
    follow_camera_clearance: FollowCameraClearanceEvidence | None

    def frame_index_at(self, timestamp_s: float) -> int:
        """Return the source trace sample used by the zero-order replay sampler."""

        if timestamp_s <= self.frames[0].timestamp_s:
            return 0
        timestamps = tuple(frame.timestamp_s for frame in self.frames)
        # ``bisect_right`` deliberately selects the final source sample at a
        # duplicated decision-boundary timestamp. That is the same state the
        # historic loop selected and gives every output frame one auditably
        # defined realised PhysX source sample.
        return min(len(self.frames) - 1, bisect_right(timestamps, timestamp_s + 1.0e-9) - 1)

    def frame_at(self, timestamp_s: float) -> ReplayFrame:
        return self.frames[self.frame_index_at(timestamp_s)]


def _finite_positive_float(raw: Any, label: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a finite positive scalar")
    return value


def _follow_camera_clearance_evidence(
    payload: dict[str, Any], agent_order: tuple[str, ...]
) -> FollowCameraClearanceEvidence | None:
    """Return a conservative camera certificate or ``None`` when evidence is incomplete.

    A follow camera is visual-only, but an arbitrary third-person offset can
    leave an indoor room through a wall.  The executor already records exact
    static-mesh clearance at every realised root trace pose.  The camera is
    therefore allowed only inside a ball around that certified root pose.
    """

    execution_payload = payload.get("execution")
    terminal_tail = (
        execution_payload.get("terminal_budget_tail")
        if isinstance(execution_payload, dict)
        else None
    )
    phase_minima: list[float] = []
    for label, raw_phase in (
        ("bootstrap", payload.get("bootstrap")),
        ("terminal budget tail", terminal_tail),
    ):
        if not isinstance(raw_phase, dict):
            return None
        execution = raw_phase.get("execution")
        if not isinstance(execution, dict):
            return None
        try:
            phase_minima.append(
                _finite_positive_float(execution.get("minimum_clearance_m"), label)
            )
        except (TypeError, ValueError):
            return None

    per_agent: dict[str, list[float]] = {agent_id: list(phase_minima) for agent_id in agent_order}
    required_clearances: list[float] = []
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        return None
    for decision in decisions:
        if not isinstance(decision, dict):
            return None
        calibration = decision.get("execution_calibration")
        if not isinstance(calibration, dict):
            return None
        static_trace = calibration.get("static_trace_clearance")
        if not isinstance(static_trace, dict) or static_trace.get(
            "static_clearance_contract_passed"
        ) is not True:
            return None
        if static_trace.get("method") != "exact_same_static_collision_mesh_at_each_physics_trace_pose_v1":
            return None
        try:
            required_clearances.append(
                _finite_positive_float(
                    static_trace.get("static_clearance_contract_required_m"),
                    "static trace clearance requirement",
                )
            )
        except (TypeError, ValueError):
            return None
        agents = calibration.get("agents")
        if not isinstance(agents, list) or len(agents) != len(agent_order):
            return None
        seen: set[str] = set()
        for agent in agents:
            if not isinstance(agent, dict):
                return None
            agent_id = agent.get("agent_id")
            if not isinstance(agent_id, str) or agent_id not in per_agent or agent_id in seen:
                return None
            seen.add(agent_id)
            try:
                per_agent[agent_id].append(
                    _finite_positive_float(
                        agent.get("minimum_static_mesh_clearance_m"),
                        f"{agent_id} static clearance",
                    )
                )
            except (TypeError, ValueError):
                return None
        if seen != set(agent_order):
            return None
    return FollowCameraClearanceEvidence(
        required_root_clearance_m=min(required_clearances),
        minimum_root_clearance_by_agent_m={
            agent_id: min(values) for agent_id, values in per_agent.items()
        },
        source_phase_minimum_clearance_m=min(phase_minima),
    )


def _certify_follow_camera(
    replay: PhysxTraceReplay,
    *,
    agent_index: int,
    camera_radius_m: float,
    safety_margin_m: float,
) -> dict[str, object]:
    """Refuse a follow pose outside the recorded static-collision free-space ball.

    HM3D's render mesh and its collision representation need not be identical.
    This certificate constrains the review camera relative to collision geometry;
    it cannot certify that scan fragments will not visually occlude the camera.
    """

    radius = _finite_positive_float(camera_radius_m, "follow camera radius")
    margin = float(safety_margin_m)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("follow camera safety margin must be finite and non-negative")
    if not 0 <= agent_index < len(replay.agent_order):
        raise ValueError("follow camera agent index is outside the trace")
    evidence = replay.follow_camera_clearance
    if evidence is None:
        raise ValueError(
            "follow rendering requires exact realised static-clearance evidence; "
            "use global replay when the record does not contain it"
        )
    agent_id = replay.agent_order[agent_index]
    root_clearance = evidence.minimum_root_clearance_by_agent_m[agent_id]
    if radius + margin > root_clearance + 1.0e-9:
        raise ValueError(
            "follow camera radius plus safety margin exceeds the recorded root clearance: "
            f"{radius + margin:.3f} m > {root_clearance:.3f} m for {agent_id}"
        )
    return {
        "status": "FOLLOW_CAMERA_CLEARANCE_CERTIFIED",
        "agent_id": agent_id,
        "camera_radius_m": radius,
        "safety_margin_m": margin,
        "minimum_realised_root_clearance_m": root_clearance,
        "residual_static_clearance_lower_bound_m": root_clearance - radius,
        "required_root_clearance_m": evidence.required_root_clearance_m,
        "source_phase_minimum_clearance_m": evidence.source_phase_minimum_clearance_m,
        "method": (
            "camera_eye_and_target_ray_within_certified_static_collision_root_clearance_ball_v1; "
            "each rendered pose uses a realised PhysX root trace sample; render-mesh visibility "
            "is not certified"
        ),
    }


def _capture_frame_count(horizon_s: float, fps: int) -> int:
    """Return a video frame count whose playback duration is the requested horizon."""

    horizon = _finite_positive_float(horizon_s, "trace horizon")
    if fps <= 0:
        raise ValueError("capture FPS must be positive")
    return max(1, int(round(horizon * fps)))


def _playback_timestamp_s(frame_index: int, frame_count: int, horizon_s: float) -> float:
    """Map the first and last output frame inclusively onto the source trace window."""

    horizon = _finite_positive_float(horizon_s, "trace horizon")
    if frame_count <= 0:
        raise ValueError("capture frame count must be positive")
    if not 0 <= frame_index < frame_count:
        raise ValueError("capture frame index is outside the configured output range")
    if frame_count == 1:
        return 0.0
    return horizon * frame_index / float(frame_count - 1)


def _frame_time_mapping(
    replay: PhysxTraceReplay,
    *,
    frame_count: int,
) -> dict[str, Any]:
    """Record the exact output-frame to realised-trace sample relation.

    The capture extension consumes Kit frames, not a simulation clock.  The
    emitted mapping proves which already-recorded PhysX sample drove every
    visual frame and lets the compositor refuse a superficially similar video
    with a different time base.
    """

    rows: list[dict[str, float | int]] = []
    for output_frame_index in range(frame_count):
        playback_timestamp_s = _playback_timestamp_s(
            output_frame_index, frame_count, replay.horizon_s
        )
        source_frame_index = replay.frame_index_at(playback_timestamp_s)
        source_frame = replay.frames[source_frame_index]
        rows.append(
            {
                "output_frame_index": output_frame_index,
                "playback_timestamp_s": playback_timestamp_s,
                "source_trace_frame_index": source_frame_index,
                "source_trace_timestamp_s": source_frame.timestamp_s,
            }
        )
    canonical = json.dumps(rows, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return {
        "schema_version": "hm3d-physx-replay-frame-time-mapping-v1",
        "sampling": "zero_order_hold_final_sample_at_duplicate_boundary_v1",
        "output_frame_count": frame_count,
        "source_trace_frame_count": len(replay.frames),
        "playback_window_s": [0.0, replay.horizon_s],
        "rows": rows,
        "rows_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
    }


def _camera_audit_binding(
    replay: PhysxTraceReplay,
    *,
    view_mode: str,
    fpv_agent: int | None,
    follow_agent: int | None,
    focal_length_mm: float,
    global_camera_framing: str,
    trace_review_bounds_min: tuple[float, float, float],
    trace_review_bounds_max: tuple[float, float, float],
    follow_clearance_certificate: dict[str, object] | None,
    scene_review_mode: str,
) -> dict[str, Any]:
    """Bind view settings to the exact record that supplied camera poses."""

    return {
        "schema_version": "hm3d-physx-camera-audit-binding-v1",
        "source_record_file_sha256": replay.source_record_file_sha256,
        "source_record_runtime_sha256": replay.source_record_runtime_sha256,
        "source_trace_frame_count": len(replay.frames),
        "camera_pose_source": "replay_frame_time_mapping_v1",
        "view_mode": view_mode,
        "fpv_agent": fpv_agent,
        "follow_agent": follow_agent,
        "focal_length_mm": focal_length_mm,
        "global_camera_framing": global_camera_framing,
        "global_trace_review_bounds_min_m": trace_review_bounds_min,
        "global_trace_review_bounds_max_m": trace_review_bounds_max,
        "follow_clearance_certificate": follow_clearance_certificate,
        "scene_review_mode": scene_review_mode,
        "limits": (
            "camera inspection only; collision-clearance evidence does not prove render-mesh "
            "line-of-sight. Global local bounds improve framing but do not repair open Matterport "
            "scan shells. Ghost mode is a non-sensor visual aid; trajectory_only mode intentionally "
            "omits architectural geometry and therefore cannot establish visual line-of-sight."
        ),
    }


def _as_finite_triplet(raw: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(raw, list | tuple) or len(raw) != 3:
        raise ValueError(f"{label} must be a finite three-vector")
    values = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must be finite")
    return values


def _yaw_deg_from_quaternion_wxyz(raw: Any) -> float:
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError("visualization trace quaternion must be wxyz")
    w, x, y, z = (float(value) for value in raw)
    if not all(math.isfinite(value) for value in (w, x, y, z)):
        raise ValueError("visualization trace quaternion must be finite")
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _trace_segment(
    raw: Any,
    *,
    offset_s: float,
    expected_order: tuple[str, ...] | None,
) -> tuple[tuple[ReplayFrame, ...], tuple[str, ...]]:
    if not isinstance(raw, dict) or raw.get("purpose") != "engineering_visual_audit_only":
        raise ValueError("record has no engineering-only PhysX visualization trace")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("visualization trace has no samples")
    rows: list[ReplayFrame] = []
    discovered_order = expected_order
    previous_timestamp = -math.inf
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("visualization trace sample is malformed")
        local_timestamp_s = float(sample.get("physics_timestamp_s"))
        if not math.isfinite(local_timestamp_s) or local_timestamp_s < previous_timestamp:
            raise ValueError("visualization trace timestamps are invalid")
        previous_timestamp = local_timestamp_s
        raw_agents = sample.get("agents")
        if not isinstance(raw_agents, list) or not raw_agents:
            raise ValueError("visualization trace sample has no agent states")
        states: dict[str, ReplayAgentState] = {}
        order = []
        for raw_agent in raw_agents:
            if not isinstance(raw_agent, dict):
                raise ValueError("visualization trace agent state is malformed")
            agent_id = raw_agent.get("agent_id")
            if not isinstance(agent_id, str) or not agent_id:
                raise ValueError("visualization trace agent ID is invalid")
            position = _as_finite_triplet(raw_agent.get("position_m"), "agent position")
            speed = float(raw_agent.get("linear_speed_mps"))
            if not math.isfinite(speed) or speed < 0.0:
                raise ValueError("visualization trace speed is invalid")
            states[agent_id] = ReplayAgentState(
                position_m=position,
                yaw_deg=_yaw_deg_from_quaternion_wxyz(raw_agent.get("quaternion_wxyz")),
                linear_speed_mps=speed,
                reservation_waiting=bool(raw_agent.get("reservation_waiting")),
                transit_completed=bool(raw_agent.get("transit_completed")),
                failed=bool(raw_agent.get("failed")),
            )
            order.append(agent_id)
        order_tuple = tuple(order)
        if discovered_order is None:
            discovered_order = order_tuple
        if order_tuple != discovered_order:
            raise ValueError("visualization trace agent order changes within a run")
        minimum_distance = float(sample.get("minimum_inter_agent_distance_m"))
        if not math.isfinite(minimum_distance) or minimum_distance <= 0.0:
            raise ValueError("visualization trace minimum separation is invalid")
        rows.append(
            ReplayFrame(
                timestamp_s=offset_s + local_timestamp_s,
                states_by_agent=states,
                minimum_inter_agent_distance_m=minimum_distance,
            )
        )
    assert discovered_order is not None
    return tuple(rows), discovered_order


def _load_physx_trace(record_path: Path) -> PhysxTraceReplay:
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if payload.get("status") != "P07_EXECUTION_SMOKE_COMPLETE":
        raise ValueError("only completed real PhysX records can be replayed")
    if payload.get("synthetic") is not False:
        raise ValueError("visual replay source must explicitly be non-synthetic")
    if payload.get("record_purpose") != "engineering_smoke":
        raise ValueError("visual replay requires an engineering-smoke source record")
    if payload.get("execution_profile", {}).get("evidence_class") != "real_isaac_physx_cf2x":
        raise ValueError("visual replay source is not real CF2X/PhysX evidence")
    scene_id = payload.get("scene_id")
    controller_id = payload.get("controller_id")
    horizon_s = float(payload.get("elapsed_physics_s"))
    runtime_hash = payload.get("runtime_record_sha256")
    if (
        not isinstance(scene_id, str)
        or not isinstance(controller_id, str)
        or not isinstance(runtime_hash, str)
        or not math.isfinite(horizon_s)
        or horizon_s <= 0.0
    ):
        raise ValueError("visual replay source record is missing identity fields")
    rows: list[ReplayFrame] = []
    order: tuple[str, ...] | None = None
    bootstrap = payload.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("visual replay source record omits bootstrap evidence")
    segment, order = _trace_segment(
        bootstrap.get("physics_visualization_trace"), offset_s=0.0, expected_order=order
    )
    rows.extend(segment)
    previous_elapsed_s = float(bootstrap.get("elapsed_physics_s"))
    for decision in payload.get("decisions", []):
        if not isinstance(decision, dict):
            raise ValueError("visual replay decision is malformed")
        calibration = decision.get("execution_calibration")
        segment, order = _trace_segment(
            calibration.get("physics_visualization_trace") if isinstance(calibration, dict) else None,
            offset_s=previous_elapsed_s,
            expected_order=order,
        )
        rows.extend(segment)
        previous_elapsed_s = float(decision.get("elapsed_physics_s"))
    execution_payload = payload.get("execution")
    tail = execution_payload.get("terminal_budget_tail") if isinstance(execution_payload, dict) else None
    if isinstance(tail, dict) and tail.get("physics_visualization_trace") is not None:
        tail_start_s = float(tail.get("executed_from_episode_s"))
        segment, order = _trace_segment(
            tail.get("physics_visualization_trace"),
            offset_s=tail_start_s,
            expected_order=order,
        )
        rows.extend(segment)
    if order is None or len(order) != 4:
        raise ValueError("visual replay requires exactly four traced CF2X agents")
    rows.sort(key=lambda row: row.timestamp_s)
    if not rows or rows[0].timestamp_s > 1.0e-6 or rows[-1].timestamp_s < horizon_s - 0.25:
        raise ValueError("visualization trace does not cover the requested physical horizon")
    return PhysxTraceReplay(
        scene_id=scene_id,
        controller_id=controller_id,
        source_record_file_sha256=sha256(record_path),
        source_record_runtime_sha256=runtime_hash,
        horizon_s=horizon_s,
        agent_order=order,
        frames=tuple(rows),
        follow_camera_clearance=_follow_camera_clearance_evidence(payload, order),
    )


def _trace_local_bounds(
    replay: PhysxTraceReplay,
    *,
    minimum_span_m: float = 2.0,
    padding_m: float = 0.60,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build a compact, equal-scale review volume from realised root positions.

    The global review camera must not use the whole HM3D mesh extent: Matterport
    scan shells can be tens of metres wide while the realised four-UAV trace is
    local.  This is a framing volume only, never a flight-space or clearance
    certificate.
    """

    if minimum_span_m <= 0.0 or padding_m < 0.0:
        raise ValueError("trace review bounds require positive span and non-negative padding")
    positions = [
        state.position_m
        for frame in replay.frames
        for state in frame.states_by_agent.values()
    ]
    if not positions:
        raise ValueError("cannot derive review bounds from an empty replay trace")
    lower = tuple(min(position[axis] for position in positions) for axis in range(3))
    upper = tuple(max(position[axis] for position in positions) for axis in range(3))
    centre = tuple((low + high) * 0.5 for low, high in zip(lower, upper, strict=True))
    maximum_span = max(high - low for low, high in zip(lower, upper, strict=True))
    half_span = max(minimum_span_m, maximum_span) * 0.5 + padding_m
    return (
        tuple(value - half_span for value in centre),
        tuple(value + half_span for value in centre),
    )


def _write_trajectory_overview(
    replay: PhysxTraceReplay,
    *,
    output: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Write a white-background XYZ audit overview of realised root trajectories."""

    if output.suffix.lower() != ".png":
        raise ValueError("trajectory overview output must have a .png suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"trajectory overview exists; add --overwrite: {output}")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    bounds_min, bounds_max = _trace_local_bounds(replay)
    figure = plt.figure(figsize=(10.0, 7.6), dpi=160, facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("white")
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        pane.set_edgecolor((0.72, 0.72, 0.72, 1.0))
    axis.grid(True, color="#c8c8c8", linewidth=0.65, alpha=0.85)
    for agent_index, agent_id in enumerate(replay.agent_order):
        positions = [frame.states_by_agent[agent_id].position_m for frame in replay.frames]
        xs, ys, zs = zip(*positions, strict=True)
        colour = LOCATOR_COLORS[agent_index % len(LOCATOR_COLORS)]
        axis.plot(xs, ys, zs, color=colour, linewidth=2.0, label=f"UAV {agent_index}")
        axis.scatter(xs[0], ys[0], zs[0], color=colour, marker="o", s=26, depthshade=False)
        axis.scatter(xs[-1], ys[-1], zs[-1], color=colour, marker="X", s=34, depthshade=False)
    axis.set_xlim(bounds_min[0], bounds_max[0])
    axis.set_ylim(bounds_min[1], bounds_max[1])
    axis.set_zlim(bounds_min[2], bounds_max[2])
    axis.set_box_aspect(tuple(high - low for low, high in zip(bounds_min, bounds_max, strict=True)))
    axis.view_init(elev=25.0, azim=-54.0)
    axis.set_xlabel("X (m)", labelpad=8)
    axis.set_ylabel("Y (m)", labelpad=8)
    axis.set_zlabel("Z (m)", labelpad=8)
    axis.set_title(
        "CF2X/PhysX trajectories | "
        f"{replay.frames[0].timestamp_s:.2f}-{replay.frames[-1].timestamp_s:.2f} s"
    )
    axis.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#a0a0a0")
    figure.tight_layout()
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
    try:
        figure.savefig(temporary, dpi=160, facecolor="white")
        os.replace(temporary, output)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "sha256": sha256(output),
        "source": "realised_cf2x_physx_root_trace",
        "trace_window_s": [replay.frames[0].timestamp_s, replay.frames[-1].timestamp_s],
        "trace_samples": len(replay.frames),
        "agent_count": len(replay.agent_order),
        "white_background": True,
        "xyz_axes_unit": "m",
        "grid": True,
        "equal_scale_bounds_min_m": bounds_min,
        "equal_scale_bounds_max_m": bounds_max,
        "colours_rgb": LOCATOR_COLORS[: len(replay.agent_order)],
        "role": "human_audit_only_not_a_formal_metric_plot",
    }


def _frame_integrity_diagnostics(
    video_path: Path,
    *,
    expected_frames: int,
) -> dict[str, Any]:
    """Decode every output frame and flag near-uniform render failures.

    RGB content cannot prove that a global camera is unobstructed by geometry.
    This diagnostic only detects blank, almost uniform, or malformed rendered
    frames, while the follow view has its separate static-clearance certificate.
    """

    import cv2

    decoder = cv2.VideoCapture(str(video_path))
    if not decoder.isOpened():
        raise RuntimeError(f"OpenCV cannot read back output video: {video_path}")
    decoded = 0
    means: list[float] = []
    content_stddevs: list[float] = []
    near_uniform_indices: list[int] = []
    try:
        while True:
            ok, frame = decoder.read()
            if not ok:
                break
            decoded += 1
            means.append(float(frame.mean()))
            standard_deviation = float(frame.std())
            content_stddevs.append(standard_deviation)
            if standard_deviation < 3.0:
                near_uniform_indices.append(decoded - 1)
    finally:
        decoder.release()
    if decoded != expected_frames or not means:
        raise RuntimeError(
            "output video frame read-back is incomplete: "
            f"decoded={decoded}, expected={expected_frames}"
        )
    return {
        "decoded_frames": decoded,
        "expected_frames": expected_frames,
        "mean_luminance_min": min(means),
        "mean_luminance_max": max(means),
        "content_stddev_min": min(content_stddevs),
        "content_stddev_max": max(content_stddevs),
        "near_uniform_frame_count": len(near_uniform_indices),
        "near_uniform_frame_indices_sample": near_uniform_indices[:20],
        "status": (
            "FRAME_CONTENT_HEURISTIC_PASS"
            if not near_uniform_indices
            else "FRAME_CONTENT_HEURISTIC_FAILED"
        ),
        "limitation": (
            "This detects blank or near-uniform frames only; it is not a geometry-level "
            "proof that a global review camera is unobstructed."
        ),
    }


def _define_xform(
    stage: Any,
    path: str,
    translation: tuple[float, float, float],
    yaw_deg: float = 0.0,
) -> Any:
    from pxr import Gf, UsdGeom

    prim = UsdGeom.Xform.Define(stage, path)
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xformable.AddRotateZOp().Set(float(yaw_deg))
    return prim


def _set_pose(prim: Any, position: tuple[float, float, float], yaw_deg: float) -> None:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    if len(ops) != 2:
        raise RuntimeError(f"unexpected transform schema at {prim.GetPath()}")
    ops[0].Set(Gf.Vec3d(*position))
    ops[1].Set(float(yaw_deg))


def _set_visible(prim: Any, visible: bool) -> None:
    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def _define_visual_locator(
    *,
    stage: Any,
    agent_id: int,
    position: tuple[float, float, float],
    radius_m: float,
    height_m: float,
) -> Any:
    """Create a non-physical beacon so a true-scale CF2X stays visible in a large mesh."""

    from pxr import Gf, UsdGeom

    root = _define_xform(stage, f"/World/VisualLocator_{agent_id}", position)
    cylinder = UsdGeom.Cylinder.Define(stage, f"/World/VisualLocator_{agent_id}/Beacon")
    cylinder.CreateRadiusAttr(radius_m * 0.32)
    cylinder.CreateHeightAttr(height_m)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    colour = Gf.Vec3f(*LOCATOR_COLORS[agent_id % len(LOCATOR_COLORS)])
    cylinder.CreateDisplayColorAttr().Set([colour])
    cylinder_xform = UsdGeom.Xformable(cylinder)
    cylinder_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, height_m * 0.5 + 0.08))
    cap = UsdGeom.Sphere.Define(stage, f"/World/VisualLocator_{agent_id}/Cap")
    cap.CreateRadiusAttr(radius_m)
    cap.CreateDisplayColorAttr().Set([colour])
    cap_xform = UsdGeom.Xformable(cap)
    cap_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, height_m + 0.08))
    return root.GetPrim()


def _set_camera(
    camera: Any,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> None:
    from pxr import Gf, UsdGeom

    eye_v = Gf.Vec3d(*eye)
    target_v = Gf.Vec3d(*target)
    if (target_v - eye_v).GetLength() < 1.0e-9:
        return
    # A USD camera looks along local -Z.  ``SetLookAt`` returns a view matrix,
    # so its inverse is the camera's world transform.  This avoids the former
    # hand-built quaternion path, whose roll correction could point a valid
    # camera away from its target in Isaac Sim 5.1.
    view_matrix = Gf.Matrix4d().SetLookAt(eye_v, target_v, Gf.Vec3d(0.0, 0.0, 1.0))
    xformable = UsdGeom.Xformable(camera)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(view_matrix.GetInverse())


def _global_camera_pose(
    frame_fraction: float,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Orbit an exact composed USD extent, never a guessed fixed scene centre."""

    centre = tuple(
        (lower + upper) * 0.5 for lower, upper in zip(bounds_min, bounds_max, strict=True)
    )
    extent = tuple(upper - lower for lower, upper in zip(bounds_min, bounds_max, strict=True))
    horizontal_radius = max(extent[0], extent[1], 1.0) * 1.10
    angle = math.tau * (0.08 + frame_fraction * 0.28)
    eye = (
        centre[0] + horizontal_radius * math.cos(angle),
        centre[1] + horizontal_radius * math.sin(angle),
        centre[2] + max(extent[2] * 0.42, horizontal_radius * 0.28, 2.0),
    )
    return eye, centre


def _global_exterior_camera_pose(
    frame_fraction: float,
    *,
    scene_bounds_min: tuple[float, float, float],
    scene_bounds_max: tuple[float, float, float],
    trace_bounds_min: tuple[float, float, float],
    trace_bounds_max: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Frame the trace from above the HM3D envelope instead of inside scan shells."""

    target = tuple(
        (lower + upper) * 0.5
        for lower, upper in zip(trace_bounds_min, trace_bounds_max, strict=True)
    )
    scene_horizontal_span = max(
        scene_bounds_max[0] - scene_bounds_min[0],
        scene_bounds_max[1] - scene_bounds_min[1],
        1.0,
    )
    # A restrained orbit avoids the wall-crossing behaviour of the prior
    # local review camera while still conveying the room-scale geometry.
    angle = math.radians(-132.0) + frame_fraction * math.radians(8.0)
    horizontal_radius = scene_horizontal_span * 0.92
    eye = (
        target[0] + horizontal_radius * math.cos(angle),
        target[1] + horizontal_radius * math.sin(angle),
        scene_bounds_max[2] + scene_horizontal_span * 0.48,
    )
    return eye, target


def _fpv_camera_pose(
    position: tuple[float, float, float], yaw_deg: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return a short-forward FPV camera, offset from the visual CF2X body."""

    yaw_rad = math.radians(yaw_deg)
    forward = (math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
    eye = (
        position[0] + forward[0] * 0.09,
        position[1] + forward[1] * 0.09,
        position[2] + 0.035,
    )
    target = (
        eye[0] + forward[0] * 3.0,
        eye[1] + forward[1] * 3.0,
        eye[2] - 0.08,
    )
    return eye, target


def _follow_camera_pose(
    position: tuple[float, float, float],
    yaw_deg: float,
    *,
    camera_radius_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return a clearance-certified, close third-person follow camera.

    Both eye and target lie inside ``camera_radius_m`` of the realised root
    pose.  The segment between them therefore remains inside the same convex
    ball and can be certified from the executor's static root-clearance trace.
    """

    radius = _finite_positive_float(camera_radius_m, "follow camera radius")
    yaw_rad = math.radians(yaw_deg)
    forward = (math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
    lateral = (-forward[1], forward[0], 0.0)
    # A former 1.08 m chase offset could cross a Matterport wall.  This
    # normalized local pose stays inside the root-clearance sphere instead.
    raw_eye_direction = (
        -forward[0] + lateral[0] * 0.35,
        -forward[1] + lateral[1] * 0.35,
        0.30,
    )
    direction_norm = math.sqrt(sum(value * value for value in raw_eye_direction))
    assert direction_norm > 0.0
    eye = (
        position[0] + radius * raw_eye_direction[0] / direction_norm,
        position[1] + radius * raw_eye_direction[1] / direction_norm,
        position[2] + radius * raw_eye_direction[2] / direction_norm,
    )
    target = (
        position[0] + forward[0] * radius * 0.30,
        position[1] + forward[1] * radius * 0.30,
        position[2] + radius * 0.12,
    )
    return eye, target


def _scene_bounds(scene_prim: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute source-scene bounds after USD reference composition."""

    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        (UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy),
        useExtentsHint=True,
    )
    world_box = cache.ComputeWorldBound(scene_prim)
    aligned = world_box.ComputeAlignedRange()
    lower = aligned.GetMin()
    upper = aligned.GetMax()
    bounds_min = (float(lower[0]), float(lower[1]), float(lower[2]))
    bounds_max = (float(upper[0]), float(upper[1]), float(upper[2]))
    if any(not math.isfinite(value) for value in (*bounds_min, *bounds_max)) or any(
        upper_value <= lower_value
        for lower_value, upper_value in zip(bounds_min, bounds_max, strict=True)
    ):
        raise RuntimeError(f"invalid composed HM3D bounds: {bounds_min} to {bounds_max}")
    return bounds_min, bounds_max


def _apply_ghost_scene_material(
    stage: Any,
    *,
    scene_prim: Any,
    opacity: float,
) -> dict[str, Any]:
    """Replace render materials on scan meshes with an audit-only translucent material.

    Matterport render meshes contain open shells and small non-manifold scan
    fragments.  A close collision-clearance-certified review camera can still
    be visually covered by those render-only triangles.  This function acts
    only on the replay stage and never opens a physics scene or changes the
    source record.  Its resulting video is explicitly marked non-sensor.
    """

    if not math.isfinite(opacity) or not 0.0 < opacity < 1.0:
        raise ValueError("--ghost-scene-opacity must be finite and strictly between zero and one")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    material = UsdShade.Material.Define(stage, "/World/AuditGhostScanMaterial")
    # Use USD Preview Surface rather than reusing source OmniPBR inputs.  The
    # source scan has per-face material subsets; a mesh-level *strong* binding
    # must override those subsets before opacity becomes visible in RTX.
    shader = UsdShade.Shader.Define(stage, "/World/AuditGhostScanMaterial/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.72, 0.80, 0.88))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    mesh_count = 0
    for prim in Usd.PrimRange(scene_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                UsdShade.Tokens.strongerThanDescendants,
            )
            mesh_count += 1
    if mesh_count == 0:
        raise RuntimeError("ghost-scene review found no render meshes below the HM3D reference")
    return {
        "mode": "ghost",
        "mesh_count": mesh_count,
        "material": str(material.GetPath()),
        "opacity": opacity,
        "visual_only": True,
        "not_sensor_imagery": True,
        "reason": "render_mesh_scan_fragment_occlusion_review_aid",
    }


def _configure_trajectory_only_scene(
    stage: Any,
    *,
    scene_prim: Any,
    trace_review_bounds_min: tuple[float, float, float],
    trace_review_bounds_max: tuple[float, float, float],
) -> dict[str, Any]:
    """Hide scan render meshes and add a non-physical neutral review backdrop.

    Matterport scan shells are useful scene context but can visually obscure a
    clearance-certified camera.  This creates a deliberately geometry-free
    visual review mode.  It has no Physics APIs and is never loaded by the
    exploration runtime.
    """

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    mesh_count = sum(1 for prim in Usd.PrimRange(scene_prim) if prim.IsA(UsdGeom.Mesh))
    if mesh_count == 0:
        raise RuntimeError("trajectory-only review found no render meshes below the HM3D reference")
    _set_visible(scene_prim, False)

    spans = tuple(
        upper - lower
        for lower, upper in zip(trace_review_bounds_min, trace_review_bounds_max, strict=True)
    )
    centre = tuple(
        (lower + upper) * 0.5
        for lower, upper in zip(trace_review_bounds_min, trace_review_bounds_max, strict=True)
    )
    radius_m = max(10.0, max(spans) * 4.0 + 4.0)
    backdrop = UsdGeom.Sphere.Define(stage, "/World/AuditTrajectoryOnlyBackdrop")
    backdrop.CreateRadiusAttr(float(radius_m))
    backdrop.CreateDoubleSidedAttr(True)
    UsdGeom.Xformable(backdrop).AddTranslateOp().Set(Gf.Vec3d(*centre))
    material = UsdShade.Material.Define(stage, "/World/AuditTrajectoryOnlyBackdropMaterial")
    shader = UsdShade.Shader.Define(
        stage, "/World/AuditTrajectoryOnlyBackdropMaterial/PreviewSurface"
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    neutral = Gf.Vec3f(0.98, 0.985, 0.99)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(neutral)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(neutral)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(backdrop.GetPrim()).Bind(material)

    # A sparse world-space grid gives the unrestricted team review a stable
    # depth reference without adding captions or screen-space annotations.
    # It is placed below every realised root pose, so it cannot hide the
    # recorded CF2X models or their growing trace curves.
    horizontal_span = max(spans[0], spans[1], 1.0)
    raw_step_m = horizontal_span / 8.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step_m))
    grid_step_m = next(
        candidate * magnitude
        for candidate in (1.0, 2.0, 5.0, 10.0)
        if candidate * magnitude >= raw_step_m
    )
    grid_half_span_m = math.ceil((horizontal_span * 0.5 + grid_step_m) / grid_step_m) * grid_step_m
    grid_origin_x = round(centre[0] / grid_step_m) * grid_step_m
    grid_origin_y = round(centre[1] / grid_step_m) * grid_step_m
    floor_z = trace_review_bounds_min[2] - max(0.20, grid_step_m * 0.18)
    grid_points: list[Any] = []
    grid_counts: list[int] = []
    grid_line_count = int(round((2.0 * grid_half_span_m) / grid_step_m)) + 1
    for line_index in range(grid_line_count):
        offset = -grid_half_span_m + line_index * grid_step_m
        grid_points.extend(
            (
                Gf.Vec3f(grid_origin_x - grid_half_span_m, grid_origin_y + offset, floor_z),
                Gf.Vec3f(grid_origin_x + grid_half_span_m, grid_origin_y + offset, floor_z),
                Gf.Vec3f(grid_origin_x + offset, grid_origin_y - grid_half_span_m, floor_z),
                Gf.Vec3f(grid_origin_x + offset, grid_origin_y + grid_half_span_m, floor_z),
            )
        )
        grid_counts.extend((2, 2))
    grid = UsdGeom.BasisCurves.Define(stage, "/World/TrajectoryReviewGrid")
    grid.CreateTypeAttr(UsdGeom.Tokens.linear)
    grid.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    grid.CreateCurveVertexCountsAttr(grid_counts)
    grid.CreatePointsAttr(grid_points)
    grid.CreateWidthsAttr([0.014])
    grid.CreateDisplayColorAttr().Set([Gf.Vec3f(0.30, 0.36, 0.43)])

    axis_length_m = min(grid_half_span_m * 0.72, max(1.0, horizontal_span * 0.36))
    axis_origin = (grid_origin_x, grid_origin_y, floor_z + 0.006)
    axis_specs = (
        ("X", (0.86, 0.20, 0.23), (axis_length_m, 0.0, 0.0)),
        ("Y", (0.16, 0.64, 0.32), (0.0, axis_length_m, 0.0)),
        ("Z", (0.22, 0.44, 0.88), (0.0, 0.0, axis_length_m)),
    )
    for axis_name, colour, delta in axis_specs:
        curve = UsdGeom.BasisCurves.Define(stage, f"/World/TrajectoryReviewAxis{axis_name}")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr(
            [
                Gf.Vec3f(*axis_origin),
                Gf.Vec3f(
                    axis_origin[0] + delta[0],
                    axis_origin[1] + delta[1],
                    axis_origin[2] + delta[2],
                ),
            ]
        )
        curve.CreateWidthsAttr([0.030])
        curve.CreateDisplayColorAttr().Set([Gf.Vec3f(*colour)])
    return {
        "mode": "trajectory_only",
        "source_scan_hidden": True,
        "hidden_mesh_count": mesh_count,
        "backdrop": {
            "path": str(backdrop.GetPath()),
            "radius_m": radius_m,
            "centre_m": centre,
            "material": str(material.GetPath()),
            "physics_api_attached": False,
        },
        "floor_grid": {
            "path": str(grid.GetPath()),
            "plane_z_m": floor_z,
            "step_m": grid_step_m,
            "half_span_m": grid_half_span_m,
            "line_count_per_axis": grid_line_count,
        },
        "axes": {
            "origin_m": axis_origin,
            "length_m": axis_length_m,
            "paths": [f"/World/TrajectoryReviewAxis{name}" for name, _, _ in axis_specs],
        },
        "visual_only": True,
        "not_sensor_imagery": True,
        "reason": "unobstructed_replay_pose_and_visual_rotor_review",
    }


def _bind_visual_rotors(stage: Any, drones: list[Any]) -> list[tuple[Any, int, tuple[float, float, float]]]:
    """Bind the visible CF2X propeller prims to a source-safe replay phase."""

    from pxr import UsdGeom

    bound: list[tuple[Any, int, tuple[float, float, float]]] = []
    propeller_directions = (1, -1, 1, -1)
    for drone in drones:
        drone_path = str(drone.GetPath())
        for propeller_index, direction in enumerate(propeller_directions, start=1):
            propeller = stage.GetPrimAtPath(
                # The referenced asset's default prim is /crazyflie, so its
                # children compose directly below each /World/UAV_N instance.
                f"{drone_path}/m{propeller_index}_prop"
            )
            if not propeller.IsValid():
                raise RuntimeError(
                    f"CF2X visual asset is missing m{propeller_index}_prop below {drone_path}"
                )
            rotation_ops = [
                operation
                for operation in UsdGeom.Xformable(propeller).GetOrderedXformOps()
                if operation.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ
            ]
            if len(rotation_ops) != 1:
                raise RuntimeError(
                    f"CF2X visual propeller has an unexpected rotation schema: {propeller.GetPath()}"
                )
            rotation = rotation_ops[0]
            base = rotation.Get()
            if base is None:
                base_xyz = (0.0, 0.0, 0.0)
            else:
                base_xyz = tuple(float(value) for value in base)
            bound.append((rotation, direction, base_xyz))
    return bound


def _apply_review_vehicle_materials(stage: Any, drones: list[Any]) -> dict[str, Any]:
    """Give replay-only CF2X meshes enough contrast for close human review.

    The source trace has no camera or motor telemetry, and the imported HM3D
    scans include very dark interior surfaces.  The referenced CF2X asset is
    therefore hard to inspect in a clearance-certified close-follow view.
    These USD bindings live only on the temporary replay stage: they neither
    modify the source asset nor attach physics APIs.  Rotor meshes retain a
    darker material so their replayed phase is visually observable.
    """

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    material_root = "/World/ReplayReviewMaterials"
    rotor_tokens = tuple(f"/m{index}_prop" for index in range(1, 5))
    audit_agents: list[dict[str, Any]] = []

    def define_material(
        name: str,
        *,
        diffuse: tuple[float, float, float],
        emissive: tuple[float, float, float],
    ) -> Any:
        material = UsdShade.Material.Define(stage, f"{material_root}/{name}")
        shader = UsdShade.Shader.Define(stage, f"{material_root}/{name}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.58)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    for agent_index, drone in enumerate(drones):
        source_colour = LOCATOR_COLORS[agent_index % len(LOCATOR_COLORS)]
        body_colour = tuple(0.42 + 0.58 * value for value in source_colour)
        body_emissive = tuple(0.10 + 0.14 * value for value in source_colour)
        body = define_material(
            f"UAV{agent_index}Body",
            diffuse=body_colour,
            emissive=body_emissive,
        )
        rotor = define_material(
            f"UAV{agent_index}Rotor",
            diffuse=(0.055, 0.065, 0.080),
            emissive=(0.008, 0.010, 0.014),
        )
        body_mesh_count = 0
        rotor_mesh_count = 0
        for prim in Usd.PrimRange(drone):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            material = rotor if any(token in str(prim.GetPath()) for token in rotor_tokens) else body
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            if material is rotor:
                rotor_mesh_count += 1
            else:
                body_mesh_count += 1
        if body_mesh_count == 0 or rotor_mesh_count == 0:
            raise RuntimeError(
                "review material binding could not locate both CF2X body and rotor meshes: "
                f"agent={agent_index}, body_meshes={body_mesh_count}, rotor_meshes={rotor_mesh_count}"
            )
        audit_agents.append(
            {
                "agent_index": agent_index,
                "body_material": str(body.GetPath()),
                "rotor_material": str(rotor.GetPath()),
                "body_mesh_count": body_mesh_count,
                "rotor_mesh_count": rotor_mesh_count,
                "body_colour_rgb": body_colour,
            }
        )
    return {
        "role": "temporary_replay_stage_visual_aid_only",
        "source_cf2x_usd_modified": False,
        "physics_api_attached": False,
        "agents": audit_agents,
    }


def _set_visual_rotor_phase(
    rotors: list[tuple[Any, int, tuple[float, float, float]]],
    *,
    timestamp_s: float,
    frequency_hz: float,
) -> None:
    """Animate mesh-only rotors without claiming an unavailable motor signal."""

    from pxr import Gf

    phase_deg = math.fmod(timestamp_s * frequency_hz * 360.0, 360.0)
    for rotation, direction, base_xyz in rotors:
        rotation.Set(
            Gf.Vec3f(
                base_xyz[0],
                base_xyz[1],
                base_xyz[2] + direction * phase_deg,
            )
        )


def _define_visual_trails(
    stage: Any,
    replay: PhysxTraceReplay,
    *,
    visible_agent_indices: tuple[int, ...],
) -> list[tuple[Any, tuple[tuple[float, float, float], ...]]]:
    """Create thin, trace-driven trajectory curves for unobstructed follow views."""

    from pxr import Gf, UsdGeom

    trails: list[tuple[Any, tuple[tuple[float, float, float], ...]]] = []
    for agent_index in visible_agent_indices:
        agent_id = replay.agent_order[agent_index]
        positions = tuple(
            frame.states_by_agent[agent_id].position_m for frame in replay.frames
        )
        if len(positions) < 2:
            raise RuntimeError("visual trail requires at least two realised trace samples")
        curve = UsdGeom.BasisCurves.Define(stage, f"/World/ReplayTrail_{agent_index}")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateBasisAttr(UsdGeom.Tokens.bezier)
        curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curve.CreateCurveVertexCountsAttr([2])
        start = Gf.Vec3f(*positions[0])
        curve.CreatePointsAttr([start, start])
        # A single width is constant by USD's default interpolation; Isaac
        # Sim 5.1 does not expose a CreateWidthsInterpolationAttr helper.
        curve.CreateWidthsAttr([0.014])
        curve.CreateDisplayColorAttr().Set([Gf.Vec3f(*LOCATOR_COLORS[agent_index])])
        trails.append((curve, positions))
    return trails


def _set_visual_trails(
    trails: list[tuple[Any, tuple[tuple[float, float, float], ...]]],
    *,
    source_frame_index: int,
) -> None:
    """Reveal only the realised portion of each visual trajectory curve."""

    from pxr import Gf

    for curve, positions in trails:
        final_index = min(max(source_frame_index, 0), len(positions) - 1)
        visible_positions = positions[: final_index + 1]
        if len(visible_positions) == 1:
            visible_positions = (visible_positions[0], visible_positions[0])
        curve.CreateCurveVertexCountsAttr().Set([len(visible_positions)])
        curve.CreatePointsAttr().Set([Gf.Vec3f(*position) for position in visible_positions])


def _reencode_capture_frames(
    *,
    output: Path,
    expected_frames: int,
    fps: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Replace Isaac 5.1's sometimes unreadable MP4 with an OpenCV-decoded MP4.

    CaptureExtension is retained for renderer access, but its bundled H.264
    encoder can produce a file that Windows/OpenCV cannot decode.  The PNG
    sequence is the authoritative rendered output; this function only packs
    it into a portable MP4 container and checks that result immediately.
    """

    import cv2

    frames_dir = output.parent / f"{output.stem}_frames"
    frame_paths = sorted(frames_dir.glob("*.png"))
    if len(frame_paths) != expected_frames:
        raise RuntimeError(
            "capture produced "
            f"{len(frame_paths)} PNG frames; expected {expected_frames}: {frames_dir}"
        )
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None or first.shape[:2] != (height, width):
        raise RuntimeError("capture's first PNG is unreadable or has the wrong dimensions")
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.reencode.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialise the portable MP4 encoder")
    try:
        for frame_index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[:2] != (height, width):
                raise RuntimeError(f"capture frame is unreadable or malformed: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()
    diagnostics = _frame_integrity_diagnostics(temporary, expected_frames=expected_frames)
    if diagnostics["near_uniform_frame_count"]:
        temporary.unlink(missing_ok=True)
        # CaptureExtension writes its provisional movie before the portable
        # PNG-backed re-encode is checked.  A rejected replay must not remain
        # at the requested delivery path.
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "re-encoded MP4 failed frame-content validation: "
            f"near_uniform_frames={diagnostics['near_uniform_frame_count']}/"
            f"{expected_frames}, stddev_min={diagnostics['content_stddev_min']:.3f}"
        )
    if diagnostics["mean_luminance_min"] <= 1.0:
        temporary.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "re-encoded MP4 failed read-back: "
            f"decoded={diagnostics['decoded_frames']}, expected={expected_frames}, "
            f"minimum_luminance={diagnostics['mean_luminance_min']}"
        )
    os.replace(temporary, output)
    return {
        "frames_dir": str(frames_dir),
        "captured_png_frames": len(frame_paths),
        "decoded_mp4_frames": diagnostics["decoded_frames"],
        "decoded_mean_min": diagnostics["mean_luminance_min"],
        "decoded_mean_max": diagnostics["mean_luminance_max"],
        "frame_integrity": diagnostics,
        "encoder": "opencv-mp4v-readback-verified",
    }


def main() -> int:
    global ARGS, SIMULATION_APP
    ARGS = parse_args()
    print("[HM3D PhysX trace replay] entering recorder", flush=True)
    _progress("entered_main")

    trace_record = _assert_input(ARGS.trace_record, "PhysX trace record")
    replay = _load_physx_trace(trace_record)
    scene_usd = _assert_input(ARGS.scene_usd, "scene USD")
    cf2x_usd = _assert_input(ARGS.cf2x_usd, "CF2X USD")
    if ARGS.fps <= 0 or ARGS.width <= 0 or ARGS.height <= 0:
        raise ValueError("fps and frame dimensions must be positive")
    if ARGS.visual_locator_radius_m <= 0.0 or ARGS.visual_locator_height_m <= 0.0:
        raise ValueError("visual locator dimensions must be positive")
    if not math.isfinite(ARGS.visual_rotor_hz) or ARGS.visual_rotor_hz <= 0.0:
        raise ValueError("--visual-rotor-hz must be finite and positive")
    frame_count = _capture_frame_count(replay.horizon_s, ARGS.fps)
    frame_time_mapping = _frame_time_mapping(replay, frame_count=frame_count)
    agent_count = len(replay.agent_order)
    if ARGS.view_mode == "fpv" and (
        ARGS.fpv_agent is None or not 0 <= ARGS.fpv_agent < agent_count
    ):
        raise ValueError("--view-mode fpv requires --fpv-agent within the traced agent range")
    if ARGS.view_mode == "follow" and (
        ARGS.follow_agent is None or not 0 <= ARGS.follow_agent < agent_count
    ):
        raise ValueError(
            "--view-mode follow requires --follow-agent within the traced agent range"
        )
    if ARGS.view_mode != "fpv" and ARGS.fpv_agent is not None:
        raise ValueError("--fpv-agent is only valid with --view-mode fpv")
    if ARGS.view_mode != "follow" and ARGS.follow_agent is not None:
        raise ValueError("--follow-agent is only valid with --view-mode follow")
    follow_camera_audit: dict[str, object] | None = None
    if ARGS.view_mode == "follow":
        assert ARGS.follow_agent is not None
        follow_camera_audit = _certify_follow_camera(
            replay,
            agent_index=ARGS.follow_agent,
            camera_radius_m=ARGS.follow_camera_radius_m,
            safety_margin_m=ARGS.follow_camera_safety_margin_m,
        )
    output = ARGS.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not ARGS.overwrite:
        raise FileExistsError(f"output already exists; add --overwrite: {output}")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must have an .mp4 suffix")
    overview_path = (
        ARGS.trajectory_overview.expanduser().resolve()
        if ARGS.trajectory_overview is not None
        else None
    )
    trajectory_overview: dict[str, Any] | None = None
    if overview_path is not None:
        trajectory_overview = _write_trajectory_overview(
            replay, output=overview_path, overwrite=ARGS.overwrite
        )
    trace_review_bounds_min, trace_review_bounds_max = _trace_local_bounds(replay)

    _progress(
        "inputs_validated",
        trace_record=str(trace_record),
        source_record_file_sha256=replay.source_record_file_sha256,
        scene_usd=str(scene_usd),
        cf2x_usd=str(cf2x_usd),
        trace_horizon_s=replay.horizon_s,
        frame_time_mapping_sha256=frame_time_mapping["rows_sha256"],
        output_frame_count=frame_count,
        trace_review_bounds_min_m=trace_review_bounds_min,
        trace_review_bounds_max_m=trace_review_bounds_max,
        follow_camera_clearance_audit=follow_camera_audit,
        trajectory_overview=trajectory_overview,
    )
    app_launcher = AppLauncher(ARGS)
    SIMULATION_APP = app_launcher.app
    print("[HM3D PhysX trace replay] Isaac Sim started", flush=True)
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    context = omni.usd.get_context()
    _progress("usd_context_acquired")
    context.new_stage()
    _progress("new_stage_created")
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    _progress("world_created")

    scene_prim = _define_xform(stage, "/World/HM3D", (0.0, 0.0, 0.0))
    scene_prim.GetPrim().GetReferences().AddReference(str(scene_usd))
    _progress("scene_reference_added")
    # The imported official mesh contains materials and therefore needs a
    # moderate dome light instead of unbounded exposure.
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    # Keep the scan readable without washing out its light materials.  The
    # previous 1200/1800 pair made the review orbit nearly white on RTX 4090.
    dome.CreateIntensityAttr(450.0)
    dome.CreateColorAttr((1.0, 1.0, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(650.0)
    key.CreateAngleAttr(0.5)
    key.AddRotateXYZOp().Set((35.0, -20.0, 20.0))
    _progress("lights_created")

    # Let USD finish composing before reading the real scene bounds.  This is
    # intentionally performed before drone prims are added, so the visual CF2X
    # models cannot enlarge or otherwise corrupt the scene extent.
    for _ in range(12):
        SIMULATION_APP.update()
    bounds_min, bounds_max = _scene_bounds(scene_prim.GetPrim())
    _progress("scene_bounds_measured", bounds_min_m=bounds_min, bounds_max_m=bounds_max)
    if ARGS.scene_review_mode == "ghost":
        scene_review = _apply_ghost_scene_material(
            stage,
            scene_prim=scene_prim.GetPrim(),
            opacity=ARGS.ghost_scene_opacity,
        )
    elif ARGS.scene_review_mode == "trajectory_only":
        # Bounds and trace-coordinate registration below must still see the
        # authored HM3D hierarchy before its visuals are hidden.
        scene_review = {
            "mode": "trajectory_only",
            "pending_after_coordinate_registration": True,
            "visual_only": True,
            "not_sensor_imagery": True,
        }
    else:
        scene_review = {
            "mode": "opaque",
            "visual_only": True,
            "not_sensor_imagery": False,
            "reason": "normal_source_scan_render",
        }
    # This is a minimal registration check, not a collision re-evaluation: a
    # visual USD with a different world origin would otherwise make a real
    # trace look plausible while actually being rendered in the wrong building.
    coordinate_padding_m = 0.50
    invalid_trace_positions: list[tuple[str, tuple[float, float, float]]] = []
    for frame in replay.frames:
        for agent_id, state in frame.states_by_agent.items():
            if any(
                value < lower - coordinate_padding_m or value > upper + coordinate_padding_m
                for value, lower, upper in zip(
                    state.position_m, bounds_min, bounds_max, strict=True
                )
            ):
                invalid_trace_positions.append((agent_id, state.position_m))
                break
        if invalid_trace_positions:
            break
    if invalid_trace_positions:
        raise RuntimeError(
            "visual USD coordinate bounds do not contain the real PhysX trace; "
            f"first offending state={invalid_trace_positions[0]}"
        )
    if ARGS.scene_review_mode == "trajectory_only":
        scene_review = _configure_trajectory_only_scene(
            stage,
            scene_prim=scene_prim.GetPrim(),
            trace_review_bounds_min=trace_review_bounds_min,
            trace_review_bounds_max=trace_review_bounds_max,
        )
    _progress("scene_review_configured", scene_review=scene_review)

    initial_frame = replay.frame_at(0.0)
    drones: list[Any] = []
    locators: list[Any] = []
    for agent_index, agent_id in enumerate(replay.agent_order):
        initial_state = initial_frame.states_by_agent[agent_id]
        prim = _define_xform(
            stage,
            f"/World/UAV_{agent_index}",
            initial_state.position_m,
            initial_state.yaw_deg,
        )
        prim.GetPrim().GetReferences().AddReference(str(cf2x_usd))
        drones.append(prim.GetPrim())
        locators.append(
            _define_visual_locator(
                stage=stage,
                agent_id=agent_index,
                position=initial_state.position_m,
                radius_m=ARGS.visual_locator_radius_m,
                height_m=ARGS.visual_locator_height_m,
            )
        )
    review_vehicle_materials = _apply_review_vehicle_materials(stage, drones)
    # The source record intentionally contains vehicle poses, not motor RPM.
    # Animate only the visible mesh propellers with a fixed, explicit visual
    # phase so the human replay reads as a flying CF2X without inventing a
    # control or telemetry signal.
    visual_rotors = _bind_visual_rotors(stage, drones)
    if ARGS.scene_review_mode != "trajectory_only":
        trail_agent_indices: tuple[int, ...] = ()
    elif ARGS.view_mode == "global":
        trail_agent_indices = tuple(range(agent_count))
    elif ARGS.view_mode == "follow":
        assert ARGS.follow_agent is not None
        trail_agent_indices = (ARGS.follow_agent,)
    else:
        trail_agent_indices = ()
    visual_trails = _define_visual_trails(
        stage,
        replay,
        visible_agent_indices=trail_agent_indices,
    )
    if ARGS.view_mode == "fpv" and not ARGS.show_vehicles_in_fpv:
        for prim in [*drones, *locators]:
            _set_visible(prim, False)
    elif ARGS.view_mode == "follow":
        # A follow camera is an inspection view of one vehicle.  Keeping the
        # other five beacons in the scene makes the camera look as if it is
        # following several vehicles at once and can obscure the mesh.
        assert ARGS.follow_agent is not None
        for agent_index, prim in enumerate(drones):
            _set_visible(prim, agent_index == ARGS.follow_agent)
        # Follow footage shows the actual CF2X mesh only. The coloured beacon
        # is useful in the global room-scale camera but can occlude a UAV that
        # is only 11 cm wide when the camera is nearby.
        for locator in locators:
            _set_visible(locator, False)
    _progress(
        "drone_references_added",
        agent_count=len(drones),
        agent_order=list(replay.agent_order),
        visual_locators=True,
        review_vehicle_materials=review_vehicle_materials,
    )

    camera = UsdGeom.Camera.Define(stage, "/World/ReviewCamera")
    # FPV keeps a broad 18 mm view.  The global view frames the realised trace
    # volume rather than the complete building shell, so four true-scale UAVs
    # remain legible in an audit capture.
    focal_length = ARGS.global_focal_length_mm if ARGS.view_mode == "global" else 24.0
    if focal_length <= 0.0:
        raise ValueError("--global-focal-length-mm must be positive")
    camera_audit_binding = _camera_audit_binding(
        replay,
        view_mode=ARGS.view_mode,
        fpv_agent=ARGS.fpv_agent,
        follow_agent=ARGS.follow_agent,
        focal_length_mm=focal_length,
        global_camera_framing=ARGS.global_camera_framing,
        trace_review_bounds_min=trace_review_bounds_min,
        trace_review_bounds_max=trace_review_bounds_max,
        follow_clearance_certificate=follow_camera_audit,
        scene_review_mode=ARGS.scene_review_mode,
    )
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(36.0)
    # Third-person review places the camera within a certified local sphere of
    # a true-scale 11 cm CF2X.  The USD default near plane clips the vehicle;
    # this changes rendering only, never the flight-space or physics contract.
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10_000.0))
    active_agent = ARGS.fpv_agent if ARGS.view_mode == "fpv" else ARGS.follow_agent
    active_state = initial_frame.states_by_agent[replay.agent_order[active_agent or 0]]
    initial_position = active_state.position_m
    initial_yaw = active_state.yaw_deg
    if ARGS.view_mode == "fpv":
        eye, target = _fpv_camera_pose(initial_position, initial_yaw)
    elif ARGS.view_mode == "follow":
        eye, target = _follow_camera_pose(
            initial_position,
            initial_yaw,
            camera_radius_m=ARGS.follow_camera_radius_m,
        )
    elif ARGS.global_camera_framing == "scene_exterior":
        eye, target = _global_exterior_camera_pose(
            0.0,
            scene_bounds_min=bounds_min,
            scene_bounds_max=bounds_max,
            trace_bounds_min=trace_review_bounds_min,
            trace_bounds_max=trace_review_bounds_max,
        )
    else:
        eye, target = _global_camera_pose(
            0.0, trace_review_bounds_min, trace_review_bounds_max
        )
    _set_camera(camera.GetPrim(), eye, target)
    _progress("review_camera_created")

    # The viewport capture API is deliberately used here instead of Replicator.
    # It is the path already used by the historical IsaacLab video scripts and
    # avoids a second RTX pipeline initialisation on this Windows installation.
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.kit.capture.viewport")
    _progress("capture_extension_enabled")
    for _ in range(8):
        SIMULATION_APP.update()
    # Isaac Sim 5.1 exposes movie capture through the extension singleton.
    # The former get_active_viewport_capture() helper used by old recordings
    # was removed, so using it silently deferred failure until the HM3D mesh
    # had already been fully loaded and materialized.
    from omni.kit.capture.viewport import (
        CaptureExtension,
        CaptureOptions,
        CaptureRangeType,
        CaptureRenderPreset,
    )

    capture = CaptureExtension.get_instance()
    if capture is None:
        raise RuntimeError("omni.kit.capture.viewport did not create a capture instance")
    capture.options = CaptureOptions(
        camera=str(camera.GetPath()),
        range_type=CaptureRangeType.FRAMES,
        fps=ARGS.fps,
        start_frame=0,
        end_frame=frame_count - 1,
        res_width=ARGS.width,
        res_height=ARGS.height,
        render_preset=CaptureRenderPreset.RAY_TRACE,
        spp_per_iteration=1,
        output_folder=str(output.parent),
        file_name=output.stem,
        file_type=".mp4",
        overwrite_existing_frames=True,
        real_time_settle_latency_frames=1,
        animation_fps=ARGS.fps,
    )

    def set_visual_state(frame: int) -> None:
        timestamp_s = _playback_timestamp_s(frame, frame_count, replay.horizon_s)
        replay_frame = replay.frame_at(timestamp_s)
        for agent_index, (agent_id, prim, locator) in enumerate(
            zip(replay.agent_order, drones, locators, strict=True)
        ):
            state = replay_frame.states_by_agent[agent_id]
            _set_pose(prim, state.position_m, state.yaw_deg)
            _set_pose(locator, state.position_m, state.yaw_deg)
        _set_visual_rotor_phase(
            visual_rotors,
            timestamp_s=timestamp_s,
            frequency_hz=ARGS.visual_rotor_hz,
        )
        _set_visual_trails(
            visual_trails,
            source_frame_index=replay.frame_index_at(timestamp_s),
        )
        if ARGS.view_mode == "fpv":
            assert ARGS.fpv_agent is not None
            state = replay_frame.states_by_agent[replay.agent_order[ARGS.fpv_agent]]
            eye, target = _fpv_camera_pose(state.position_m, state.yaw_deg)
        elif ARGS.view_mode == "follow":
            assert ARGS.follow_agent is not None
            state = replay_frame.states_by_agent[replay.agent_order[ARGS.follow_agent]]
            eye, target = _follow_camera_pose(
                state.position_m,
                state.yaw_deg,
                camera_radius_m=ARGS.follow_camera_radius_m,
            )
        elif ARGS.global_camera_framing == "scene_exterior":
            eye, target = _global_exterior_camera_pose(
                timestamp_s / max(replay.horizon_s, 1.0e-9),
                scene_bounds_min=bounds_min,
                scene_bounds_max=bounds_max,
                trace_bounds_min=trace_review_bounds_min,
                trace_bounds_max=trace_review_bounds_max,
            )
        else:
            eye, target = _global_camera_pose(
                timestamp_s / max(replay.horizon_s, 1.0e-9),
                trace_review_bounds_min,
                trace_review_bounds_max,
            )
        _set_camera(camera.GetPrim(), eye, target)

    captured_frame = 0
    set_visual_state(captured_frame)

    def forward_one_frame(_: float) -> bool:
        nonlocal captured_frame
        captured_frame += 1
        if captured_frame >= frame_count:
            return False
        set_visual_state(captured_frame)
        if captured_frame % max(1, ARGS.fps * 5) == 0:
            print(f"[HM3D PhysX trace replay] recorded {captured_frame}/{frame_count} frames", flush=True)
            _progress("capturing", frame=captured_frame, total_frames=frame_count)
        return True

    capture.forward_one_frame_fn = forward_one_frame
    if not capture.start():
        raise RuntimeError("Isaac Sim viewport capture refused the configured movie")
    _progress("capture_started", output=str(output), total_frames=frame_count)

    # CaptureExtension runs on Kit's update event.  It owns image sequencing
    # and MP4 encoding; the loop only advances Kit until that work is complete.
    update_count = 0
    max_updates = max(1000, frame_count * 16)
    while SIMULATION_APP.is_running() and not capture.done:
        SIMULATION_APP.update()
        update_count += 1
        if update_count > max_updates:
            capture.cancel()
            raise RuntimeError(f"capture exceeded {max_updates} Kit updates without completion")
    if not capture.done:
        raise RuntimeError("Isaac Sim application exited before movie capture completed")
    _progress("capture_stopped")

    encoding = _reencode_capture_frames(
        output=output,
        expected_frames=frame_count,
        fps=ARGS.fps,
        width=ARGS.width,
        height=ARGS.height,
    )
    manifest_output = output.with_suffix(".manifest.json")
    _write_json(
        manifest_output,
        {
            "schema_version": "hm3d-physx-trace-replay-v2",
            "status": "PHYSX_TRACE_REPLAY_COMPLETE",
            "formal_result": False,
            "trained_policy_rollout": False,
            "render_role": "human_audit_only",
            "source_record": {
                "path": str(trace_record),
                "file_sha256": replay.source_record_file_sha256,
                "runtime_record_sha256": replay.source_record_runtime_sha256,
                "scene_id": replay.scene_id,
                "controller_id": replay.controller_id,
                "trace_horizon_s": replay.horizon_s,
                "agent_order": list(replay.agent_order),
                "trace_frame_count": len(replay.frames),
            },
            "scene": {
                "scene_id": replay.scene_id,
                "scene_usd": str(scene_usd),
                "scene_usd_sha256": sha256(scene_usd),
                "review": scene_review,
            },
            "vehicle": {"asset": str(cf2x_usd), "sha256": sha256(cf2x_usd)},
            "review_vehicle_materials": review_vehicle_materials,
            "visual_locator": {
                "enabled": True,
                "visible_in_fpv": bool(ARGS.view_mode != "fpv" or ARGS.show_vehicles_in_fpv),
                "role": "visual_only_nonphysical_uav_locator",
                "radius_m": ARGS.visual_locator_radius_m,
                "height_m": ARGS.visual_locator_height_m,
                "colours_rgb": LOCATOR_COLORS[:agent_count],
            },
            "visual_rotors": {
                "enabled": True,
                "prim_paths": ["/crazyflie/m1_prop", "/crazyflie/m2_prop", "/crazyflie/m3_prop", "/crazyflie/m4_prop"],
                "rotation_directions": ["ccw", "cw", "ccw", "cw"],
                "phase_frequency_hz": ARGS.visual_rotor_hz,
                "telemetry_source": "not_recorded_fixed_visual_phase",
                "role": "visual_only_nonphysical_animation",
            },
            "visual_trails": {
                "enabled": bool(visual_trails),
                "agent_indices": list(trail_agent_indices),
                "role": "visual_only_realised_trace_history",
            },
            "video": {
                "path": str(output),
                "sha256": sha256(output),
                "fps": ARGS.fps,
                "frames": frame_count,
                "width": ARGS.width,
                "height": ARGS.height,
                "frame_time_mapping": frame_time_mapping,
                **encoding,
            },
            "agent_count": agent_count,
            "view": {
                "mode": ARGS.view_mode,
                "global_camera_framing": ARGS.global_camera_framing,
                "fpv_agent": ARGS.fpv_agent,
                "follow_agent": ARGS.follow_agent,
                "focal_length_mm": focal_length,
                "composed_scene_bounds_min_m": bounds_min,
                "composed_scene_bounds_max_m": bounds_max,
                "trace_review_bounds_min_m": trace_review_bounds_min,
                "trace_review_bounds_max_m": trace_review_bounds_max,
            },
            "camera_audit_binding": camera_audit_binding,
            "caveat": (
                "The mesh and camera are audit aids only. Vehicle translations and yaw headings "
                "are replayed from a completed real CF2X/PhysX trace; this artifact is not a "
                "formal performance result and is excluded from candidate selection, control, "
                "sensing, rewards, training, QD, and OGFR."
            ),
        },
    )
    _progress("completed", output=str(output), manifest=str(manifest_output))
    print(
        json.dumps({"video": str(output), "manifest": str(manifest_output), "frames": frame_count})
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        _progress(
            "failed",
            exception_type=type(error).__name__,
            exception=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if SIMULATION_APP is not None:
            SIMULATION_APP.close()
