"""Shared 2D dynamic gate-density geometry for training, eval, and replay."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import numpy as np

UNIFIED_GATE_BOTTOM_HEIGHT_M = 0.0
UNIFIED_GATE_TOP_HEIGHT_M = 8.0
UNIFIED_GATE_CENTER_HEIGHT_M = 0.5 * (UNIFIED_GATE_BOTTOM_HEIGHT_M + UNIFIED_GATE_TOP_HEIGHT_M)
TRAINING_DRONE_SPEED_AXIS_MPS = (1.15, 1.45, 1.75, 2.05, 2.40, 2.75, 3.10, 3.50)
TRAINING_DRONE_ACCEL_AXIS_MPS2 = (0.75, 1.00, 1.20, 1.40, 1.65, 1.90, 2.15, 2.45)
TRAINING_DRONE_STAGE_SPEED_SCHEDULE_MPS = (1.15, 1.15, 1.45, 1.75, 2.05, 2.40, 2.75, 3.10, 3.50, 3.50, 3.50)
TRAINING_DRONE_STAGE_ACCEL_SCHEDULE_MPS2 = (0.75, 0.75, 1.00, 1.20, 1.40, 1.65, 1.90, 2.15, 2.45, 2.45, 2.45)
EVAL_DRONE_SPEED_AXIS_MPS = (0.80, 1.25, 1.70, 2.15, 2.60, 3.00, 3.25, 3.50)
MAX_DRONE_COMMAND_SPEED_MPS = max(max(TRAINING_DRONE_SPEED_AXIS_MPS), max(EVAL_DRONE_SPEED_AXIS_MPS))
MAX_DRONE_COMMAND_ACCEL_MPS2 = max(TRAINING_DRONE_ACCEL_AXIS_MPS2)


@dataclass(frozen=True)
class DynamicGateDensity2DConfig:
    scene_mode: str = "dynamic_gate_density_8d_v1"
    team_sizes: tuple[int, ...] = (8,)
    world_x_bounds_m: tuple[float, float] = (-32.0, 32.0)
    world_y_bounds_m: tuple[float, float] = (-12.0, 12.0)
    start_x_m: float = -27.0
    goal_x_m: float = 27.0
    fixed_height_m: float = UNIFIED_GATE_CENTER_HEIGHT_M
    gate_count: int = 0
    moving_gate_speed_mps: float = 0.0
    moving_gate_amplitude_m: float = 0.0
    gate_region_x_m: tuple[float, float] = (-12.5, 24.0)
    gate_lane_y_m: tuple[float, float, float] = (0.0, -7.2, 7.2)
    moving_clip_x_m: tuple[float, float] = (-25.5, 25.5)
    moving_clip_y_m: tuple[float, float] = (-9.25, 9.25)
    gate_yaw_range_deg: tuple[float, float] = (-5.0, 5.0)
    max_gate_count: int = 60
    max_moving_gate_speed_mps: float = 2.0
    gate_half_width_m: float = 2.40
    gate_post_radius_m: float = 0.32
    drone_radius_m: float = 0.35
    gate_opening_bottom_height_m: float = UNIFIED_GATE_BOTTOM_HEIGHT_M
    gate_opening_top_height_m: float = UNIFIED_GATE_TOP_HEIGHT_M
    gate_center_height_m: float = UNIFIED_GATE_CENTER_HEIGHT_M
    gate_top_clearance_margin_m: float = 0.30
    max_allowed_flight_height_m: float = UNIFIED_GATE_CENTER_HEIGHT_M
    min_allowed_flight_height_m: float = UNIFIED_GATE_CENTER_HEIGHT_M
    corridor_through_required: bool = True
    corridor_half_width_m: float = 11.20
    corridor_x_margin_m: float = 0.60
    side_bypass_policy: str = "terminal_failure_inside_gate_region_outside_corridor"
    height_escape_policy: str = "fixed_2d_plane_equals_gate_geometric_center"
    gate_gate_clearance_margin_m: float = 0.12
    non_overlap_iterations: int = 28
    dynamic_motion_policy: str = "centerline_lateral_lissajous_antiphase_open_close"
    collision_policy: str = "terminal_crash_on_live_or_swept_gate_post_contact"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DroneSpeedGradientStage:
    """Forward-speed curriculum knobs for the 2D gate-density line."""

    stage_index: int
    max_command_speed_mps: float
    max_accel_mps2: float


@dataclass(frozen=True)
class DynamicGate2D:
    base_center_xy: tuple[float, float]
    yaw_rad: float
    motion_phase: float
    motion_mode: str
    lane_index: int
    column_index: int


def default_dynamic_gate_density_config() -> DynamicGateDensity2DConfig:
    return DynamicGateDensity2DConfig()


def training_drone_speed_gradient() -> tuple[DroneSpeedGradientStage, ...]:
    if len(TRAINING_DRONE_SPEED_AXIS_MPS) != len(TRAINING_DRONE_ACCEL_AXIS_MPS2):
        raise RuntimeError("training drone speed and acceleration axes must have the same length")
    return tuple(
        DroneSpeedGradientStage(
            stage_index=idx,
            max_command_speed_mps=float(speed),
            max_accel_mps2=float(TRAINING_DRONE_ACCEL_AXIS_MPS2[idx]),
        )
        for idx, speed in enumerate(TRAINING_DRONE_SPEED_AXIS_MPS)
    )


def speed_gradient_for_stage(stage_index: int) -> DroneSpeedGradientStage:
    if len(TRAINING_DRONE_STAGE_SPEED_SCHEDULE_MPS) != len(TRAINING_DRONE_STAGE_ACCEL_SCHEDULE_MPS2):
        raise RuntimeError("stage drone speed and acceleration schedules must have the same length")
    idx = max(0, min(int(stage_index), len(TRAINING_DRONE_STAGE_SPEED_SCHEDULE_MPS) - 1))
    return DroneSpeedGradientStage(
        stage_index=idx,
        max_command_speed_mps=float(TRAINING_DRONE_STAGE_SPEED_SCHEDULE_MPS[idx]),
        max_accel_mps2=float(TRAINING_DRONE_STAGE_ACCEL_SCHEDULE_MPS2[idx]),
    )


def eval_drone_speed_axis_mps() -> tuple[float, ...]:
    return tuple(float(value) for value in EVAL_DRONE_SPEED_AXIS_MPS)


def drone_accel_limit_for_speed_mps2(speed_mps: float, override_mps2: float | None = None) -> float:
    """Return the unified acceleration limit paired with a drone speed cap."""

    if override_mps2 is not None:
        return float(override_mps2)
    if len(TRAINING_DRONE_SPEED_AXIS_MPS) != len(TRAINING_DRONE_ACCEL_AXIS_MPS2):
        raise RuntimeError("training drone speed and acceleration axes must have the same length")
    speed_axis = np.asarray(TRAINING_DRONE_SPEED_AXIS_MPS, dtype=np.float64)
    accel_axis = np.asarray(TRAINING_DRONE_ACCEL_AXIS_MPS2, dtype=np.float64)
    speed = float(np.clip(float(speed_mps), float(speed_axis[0]), float(speed_axis[-1])))
    return float(np.interp(speed, speed_axis, accel_axis))


def resolve_moving_gate_speed_hz(*, amplitude_m: float, speed_mps: float, config: DynamicGateDensity2DConfig) -> float:
    if amplitude_m <= 1.0e-6 or speed_mps <= 1.0e-6:
        return 0.0
    capped_speed = min(float(speed_mps), float(config.max_moving_gate_speed_mps))
    return float(capped_speed / (2.0 * math.pi * max(float(amplitude_m), 1.0e-6)))


def generate_gate_layout(
    *,
    gate_count: int,
    seed: int,
    config: DynamicGateDensity2DConfig | None = None,
    static_layout: bool | None = None,
) -> list[DynamicGate2D]:
    """Generate a formation-facing, tight-opening dynamic gate layout.

    Low counts use only the center lane so the curriculum first learns to pass
    moving gates.  Higher counts add side lanes and increasing column density.
    """

    cfg = config or default_dynamic_gate_density_config()
    count = int(max(0, min(int(cfg.max_gate_count), int(gate_count))))
    if count <= 0:
        return []
    if static_layout is None:
        static_layout = (
            float(getattr(cfg, "moving_gate_speed_mps", 0.0) or 0.0) <= 1.0e-6
            or float(getattr(cfg, "moving_gate_amplitude_m", 0.0) or 0.0) <= 1.0e-6
        )
    rng = random.Random(int(seed) + 28117)
    if count <= 12:
        # Sparse curricula first teach the team to pass route-facing gates on
        # the main corridor before side-lane clutter is introduced.
        lane_order = (0,)
        column_count = count
        selected_column_indices = tuple(range(column_count))
    else:
        # Spread dense layouts over three lanes so the 60-gate case stays readable.
        # Increasing gate_count should add gates, not rescale earlier columns.
        lane_order = (0, 1, 2)
        column_count = int(math.ceil(count / len(lane_order)))
        max_dense_columns = int(math.ceil(int(cfg.max_gate_count) / len(lane_order)))
        selected_column_indices = tuple(sorted(_nested_dense_column_order(max_dense_columns)[:column_count]))
        column_count = max_dense_columns
    xs = np.linspace(cfg.gate_region_x_m[0], cfg.gate_region_x_m[1], max(column_count, 1), dtype=np.float32)
    gates: list[DynamicGate2D] = []
    yaw_min_deg = float(cfg.gate_yaw_range_deg[0])
    yaw_max_deg = float(cfg.gate_yaw_range_deg[1])
    for column_index in selected_column_indices:
        active_lane_order = lane_order
        layer_x_offset = 0.0
        layer_y_offset = 0.0
        layer_lane_scale = 1.0
        if bool(static_layout):
            layer_rng = random.Random(int(seed) + 104_729 + int(column_index) * 7_919)
            active_lane_order = tuple(layer_rng.sample(tuple(lane_order), k=len(tuple(lane_order))))
            layer_x_offset = layer_rng.uniform(-0.20, 0.20) if len(lane_order) == 1 else layer_rng.uniform(-0.35, 0.35)
            layer_y_offset = layer_rng.uniform(-2.40, 2.40) if len(lane_order) == 1 else layer_rng.uniform(-1.15, 1.15)
            layer_lane_scale = 1.0 if len(lane_order) == 1 else layer_rng.uniform(0.86, 1.10)
        for lane_index in active_lane_order:
            if len(gates) >= count:
                break
            if bool(static_layout):
                gate_rng = random.Random(int(seed) + 130_363 + int(column_index) * 8_191 + int(lane_index) * 379)
                lane_base_y = float(cfg.gate_lane_y_m[lane_index])
                base_y = layer_y_offset if len(lane_order) == 1 else lane_base_y * layer_lane_scale + layer_y_offset
                jitter_x = gate_rng.uniform(-0.42, 0.42) if count <= 12 else gate_rng.uniform(-0.32, 0.32)
                jitter_y = gate_rng.uniform(-0.70, 0.70) if len(lane_order) == 1 else gate_rng.uniform(-0.48, 0.48)
                lane_x_offset = gate_rng.uniform(-0.24, 0.24) if lane_index == 0 else gate_rng.uniform(-0.52, 0.52)
                yaw = math.radians(gate_rng.uniform(yaw_min_deg, yaw_max_deg))
            else:
                base_y = float(cfg.gate_lane_y_m[lane_index])
                jitter_x = rng.uniform(-0.18, 0.18) if count <= 12 else rng.uniform(-0.08, 0.08)
                jitter_y = rng.uniform(-0.12, 0.12) if lane_index == 0 else rng.uniform(-0.18, 0.18)
                lane_x_offset = 0.0 if lane_index == 0 else (0.38 if lane_index == 1 else -0.38)
                yaw = math.radians(rng.uniform(yaw_min_deg, yaw_max_deg))
            if lane_index == 0:
                mode = "antiphase" if column_index % 2 else "lateral"
            else:
                mode = "lissajous"
            center_x = float(
                np.clip(
                    float(xs[column_index]) + float(layer_x_offset) + float(lane_x_offset) + float(jitter_x),
                    float(cfg.moving_clip_x_m[0]),
                    float(cfg.moving_clip_x_m[1]),
                )
            )
            center_y = float(
                np.clip(
                    float(base_y) + float(jitter_y),
                    float(cfg.moving_clip_y_m[0]),
                    float(cfg.moving_clip_y_m[1]),
                )
            )
            gates.append(
                DynamicGate2D(
                    base_center_xy=(center_x, center_y),
                    yaw_rad=float(yaw),
                    motion_phase=float(0.71 * column_index + 0.43 * lane_index + rng.uniform(-0.1, 0.1)),
                    motion_mode=mode,
                    lane_index=int(lane_index),
                    column_index=int(column_index),
                )
            )
    return gates


def _nested_dense_column_order(column_count: int) -> tuple[int, ...]:
    """Return a deterministic farthest-first column order for nested curricula."""

    count = int(max(column_count, 0))
    if count <= 0:
        return ()
    selected: list[int] = []
    remaining = set(range(count))
    anchors = (0, count - 1, (count - 1) // 2)
    for anchor in anchors:
        if anchor in remaining:
            selected.append(anchor)
            remaining.remove(anchor)
    while remaining:
        best = max(
            remaining,
            key=lambda idx: (
                min(abs(idx - chosen) for chosen in selected),
                -abs(idx - (count - 1) * 0.5),
                -idx,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return tuple(selected)


def _rotation(yaw_rad: float) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def gate_posts_by_gate(
    gates: list[DynamicGate2D],
    centers_xy: np.ndarray,
    *,
    config: DynamicGateDensity2DConfig | None = None,
) -> np.ndarray:
    cfg = config or default_dynamic_gate_density_config()
    if len(gates) == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)
    posts: list[list[np.ndarray]] = []
    for gate, center in zip(gates, centers_xy, strict=True):
        axis = _rotation(gate.yaw_rad) @ np.asarray([0.0, cfg.gate_half_width_m], dtype=np.float32)
        posts.append([np.asarray(center, dtype=np.float32) + axis, np.asarray(center, dtype=np.float32) - axis])
    return np.asarray(posts, dtype=np.float32)


def gate_posts(
    gates: list[DynamicGate2D],
    centers_xy: np.ndarray,
    *,
    config: DynamicGateDensity2DConfig | None = None,
) -> np.ndarray:
    posts_by_gate = gate_posts_by_gate(gates, centers_xy, config=config)
    if posts_by_gate.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return posts_by_gate.reshape((-1, 2))


def _project_gate_non_overlap(
    gates: list[DynamicGate2D],
    centers_xy: np.ndarray,
    *,
    config: DynamicGateDensity2DConfig,
) -> np.ndarray:
    if len(gates) <= 1:
        return centers_xy.astype(np.float32, copy=True)
    centers = centers_xy.astype(np.float32, copy=True)
    min_distance = 2.0 * float(config.gate_post_radius_m) + float(config.gate_gate_clearance_margin_m)
    for _ in range(int(config.non_overlap_iterations)):
        changed = False
        posts = gate_posts_by_gate(gates, centers, config=config)
        for i in range(len(gates)):
            for j in range(i + 1, len(gates)):
                for post_i in posts[i]:
                    for post_j in posts[j]:
                        delta = post_i - post_j
                        dist = float(np.linalg.norm(delta))
                        if dist >= min_distance:
                            continue
                        direction = delta / dist if dist > 1.0e-6 else np.asarray([1.0, 0.0], dtype=np.float32)
                        correction = 0.5 * (min_distance - dist) * direction
                        centers[i] += correction
                        centers[j] -= correction
                        changed = True
        centers[:, 0] = np.clip(centers[:, 0], config.moving_clip_x_m[0], config.moving_clip_x_m[1])
        centers[:, 1] = np.clip(centers[:, 1], config.moving_clip_y_m[0], config.moving_clip_y_m[1])
        if not changed:
            break
    return centers


def live_gate_centers(
    gates: list[DynamicGate2D],
    *,
    t_sec: float,
    amplitude_m: float,
    speed_mps: float,
    config: DynamicGateDensity2DConfig | None = None,
    project_non_overlap: bool = True,
) -> np.ndarray:
    cfg = config or default_dynamic_gate_density_config()
    if not gates:
        return np.zeros((0, 2), dtype=np.float32)
    hz = resolve_moving_gate_speed_hz(amplitude_m=amplitude_m, speed_mps=speed_mps, config=cfg)
    if hz <= 0.0 or amplitude_m <= 0.0:
        return np.asarray([gate.base_center_xy for gate in gates], dtype=np.float32)
    omega_t = 2.0 * math.pi * hz * float(t_sec)
    centers: list[np.ndarray] = []
    for gate in gates:
        base = np.asarray(gate.base_center_xy, dtype=np.float32)
        phase = omega_t + float(gate.motion_phase)
        if hz <= 0.0 or amplitude_m <= 0.0:
            delta = np.zeros(2, dtype=np.float32)
        elif gate.motion_mode == "lateral":
            delta = np.asarray([0.0, amplitude_m * math.sin(phase)], dtype=np.float32)
        elif gate.motion_mode == "lissajous":
            side_scale = 0.72 if gate.lane_index != 0 else 1.0
            delta = np.asarray(
                [
                    0.34 * amplitude_m * math.sin(0.73 * phase),
                    side_scale * amplitude_m * math.sin(phase),
                ],
                dtype=np.float32,
            )
        else:
            sign = -1.0 if gate.column_index % 2 else 1.0
            delta = np.asarray(
                [
                    0.22 * amplitude_m * math.sin(phase + 0.5 * math.pi),
                    sign * amplitude_m * math.sin(phase),
                ],
                dtype=np.float32,
            )
        center = base + delta
        center[0] = float(np.clip(center[0], cfg.moving_clip_x_m[0], cfg.moving_clip_x_m[1]))
        center[1] = float(np.clip(center[1], cfg.moving_clip_y_m[0], cfg.moving_clip_y_m[1]))
        centers.append(center)
    result = np.asarray(centers, dtype=np.float32)
    if project_non_overlap:
        result = _project_gate_non_overlap(gates, result, config=cfg)
    return result


def live_gate_velocities(
    gates: list[DynamicGate2D],
    *,
    t_sec: float,
    dt_s: float,
    amplitude_m: float,
    speed_mps: float,
    config: DynamicGateDensity2DConfig | None = None,
) -> np.ndarray:
    if not gates:
        return np.zeros((0, 2), dtype=np.float32)
    cfg = config or default_dynamic_gate_density_config()
    centers_a = live_gate_centers(
        gates,
        t_sec=t_sec,
        amplitude_m=amplitude_m,
        speed_mps=speed_mps,
        config=cfg,
    )
    centers_b = live_gate_centers(
        gates,
        t_sec=t_sec + float(dt_s),
        amplitude_m=amplitude_m,
        speed_mps=speed_mps,
        config=cfg,
    )
    return (centers_b - centers_a) / max(float(dt_s), 1.0e-6)


def post_clearance(
    positions_xy: np.ndarray,
    posts_xy: np.ndarray,
    *,
    config: DynamicGateDensity2DConfig | None = None,
) -> float:
    cfg = config or default_dynamic_gate_density_config()
    if len(posts_xy) == 0:
        return float("inf")
    best = float("inf")
    combined_radius = float(cfg.gate_post_radius_m) + float(cfg.drone_radius_m)
    for pos in np.asarray(positions_xy, dtype=np.float32):
        distances = np.linalg.norm(np.asarray(posts_xy, dtype=np.float32) - pos, axis=1) - combined_radius
        best = min(best, float(np.min(distances)))
    return best


def swept_post_clearance(
    prev_positions_xy: np.ndarray,
    next_positions_xy: np.ndarray,
    start_posts_xy: np.ndarray,
    end_posts_xy: np.ndarray,
    *,
    config: DynamicGateDensity2DConfig | None = None,
) -> float:
    cfg = config or default_dynamic_gate_density_config()
    if len(start_posts_xy) == 0 or len(end_posts_xy) == 0:
        return float("inf")
    combined_radius = float(cfg.gate_post_radius_m) + float(cfg.drone_radius_m)
    best = float("inf")
    prev = np.asarray(prev_positions_xy, dtype=np.float32)
    nxt = np.asarray(next_positions_xy, dtype=np.float32)
    start_posts = np.asarray(start_posts_xy, dtype=np.float32)
    end_posts = np.asarray(end_posts_xy, dtype=np.float32)
    for agent_start, agent_end in zip(prev, nxt, strict=True):
        for post_start, post_end in zip(start_posts, end_posts, strict=True):
            rel0 = agent_start - post_start
            rel1 = agent_end - post_end
            vel = rel1 - rel0
            vv = float(np.dot(vel, vel))
            alpha = 0.0 if vv <= 1.0e-12 else float(np.clip(-float(np.dot(rel0, vel)) / vv, 0.0, 1.0))
            closest = rel0 + alpha * vel
            best = min(best, float(np.linalg.norm(closest)) - combined_radius)
    return best


def gate_gate_clearance_stats(
    gates: list[DynamicGate2D],
    centers_xy: np.ndarray,
    *,
    config: DynamicGateDensity2DConfig | None = None,
) -> dict[str, float | int]:
    cfg = config or default_dynamic_gate_density_config()
    posts_by_gate = gate_posts_by_gate(gates, centers_xy, config=cfg)
    if len(posts_by_gate) <= 1:
        return {
            "gate_gate_min_clearance_m": float("inf"),
            "gate_gate_overlap_pair_count": 0,
            "gate_gate_checked_post_pair_count": 0,
        }
    min_clearance = float("inf")
    overlap_count = 0
    checked = 0
    for i in range(len(posts_by_gate)):
        for j in range(i + 1, len(posts_by_gate)):
            for post_i in posts_by_gate[i]:
                for post_j in posts_by_gate[j]:
                    checked += 1
                    clearance = float(np.linalg.norm(post_i - post_j)) - 2.0 * float(cfg.gate_post_radius_m)
                    min_clearance = min(min_clearance, clearance)
                    if clearance < 0.0:
                        overlap_count += 1
    return {
        "gate_gate_min_clearance_m": float(min_clearance),
        "gate_gate_overlap_pair_count": int(overlap_count),
        "gate_gate_checked_post_pair_count": int(checked),
    }


def resolved_corridor_half_width_m(config: DynamicGateDensity2DConfig | None = None) -> float:
    cfg = config or default_dynamic_gate_density_config()
    configured = float(getattr(cfg, "corridor_half_width_m", 0.0) or 0.0)
    if configured > 0.0:
        return configured
    lane_extent = max(abs(float(value)) for value in cfg.gate_lane_y_m)
    return float(lane_extent + cfg.gate_half_width_m + cfg.drone_radius_m + 0.75)


def validate_height_and_corridor_invariants(
    *,
    config: DynamicGateDensity2DConfig | None = None,
) -> dict[str, object]:
    """Validate the paper hard constraints that prevent over-gate or side bypass.

    The simulator line is planar, so a "height escape" is a configuration
    error: the fixed 2D flight plane must lie inside the visual gate opening
    and below the gate top minus a margin.  The corridor width must also be
    tighter than the world Y bounds, otherwise side bypass cannot be detected.
    """

    cfg = config or default_dynamic_gate_density_config()
    failures: list[str] = []
    fixed_height = float(cfg.fixed_height_m)
    opening_bottom = float(cfg.gate_opening_bottom_height_m)
    opening_top = float(cfg.gate_opening_top_height_m)
    gate_center = float(getattr(cfg, "gate_center_height_m", 0.5 * (opening_bottom + opening_top)))
    top_margin = float(cfg.gate_top_clearance_margin_m)
    max_allowed = float(cfg.max_allowed_flight_height_m)
    min_allowed = float(cfg.min_allowed_flight_height_m)
    drone_radius = float(cfg.drone_radius_m)
    corridor_half_width = resolved_corridor_half_width_m(cfg)
    world_y_abs = min(abs(float(cfg.world_y_bounds_m[0])), abs(float(cfg.world_y_bounds_m[1])))

    if opening_top <= opening_bottom:
        failures.append(f"invalid_gate_opening_height: bottom={opening_bottom:.3f}, top={opening_top:.3f}")
    expected_center = 0.5 * (opening_bottom + opening_top)
    if abs(gate_center - expected_center) > 1.0e-5:
        failures.append(f"gate_center_not_geometric_center: center={gate_center:.3f}, expected={expected_center:.3f}")
    if abs(fixed_height - gate_center) > 1.0e-5:
        failures.append(f"fixed_height_not_gate_center: fixed={fixed_height:.3f}, center={gate_center:.3f}")
    if fixed_height < opening_bottom or fixed_height > (opening_top - top_margin):
        failures.append(
            "fixed_height_not_inside_gate_opening: "
            f"fixed={fixed_height:.3f}, bottom={opening_bottom:.3f}, "
            f"top_minus_margin={opening_top - top_margin:.3f}"
        )
    if fixed_height - drone_radius < opening_bottom:
        failures.append(
            f"drone_shell_below_gate_bottom: fixed={fixed_height:.3f}, radius={drone_radius:.3f}, "
            f"bottom={opening_bottom:.3f}"
        )
    if fixed_height + drone_radius > (opening_top - top_margin):
        failures.append(
            f"drone_shell_exceeds_gate_top_margin: fixed={fixed_height:.3f}, radius={drone_radius:.3f}, "
            f"top_minus_margin={opening_top - top_margin:.3f}"
        )
    if min_allowed > fixed_height or fixed_height > max_allowed:
        failures.append(
            f"fixed_height_outside_allowed_band: fixed={fixed_height:.3f}, "
            f"allowed=[{min_allowed:.3f}, {max_allowed:.3f}]"
        )
    if max_allowed > (opening_top - top_margin):
        failures.append(
            f"max_allowed_height_exceeds_gate_top_margin: max_allowed={max_allowed:.3f}, "
            f"top_minus_margin={opening_top - top_margin:.3f}"
        )
    if corridor_half_width <= 0.0:
        failures.append(f"invalid_corridor_half_width={corridor_half_width:.3f}")
    if corridor_half_width >= (world_y_abs - float(cfg.drone_radius_m)):
        failures.append(
            f"corridor_too_wide_to_detect_side_bypass: half_width={corridor_half_width:.3f}, "
            f"world_y_abs={world_y_abs:.3f}, drone_radius={cfg.drone_radius_m:.3f}"
        )
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "fixed_height_m": fixed_height,
        "gate_opening_bottom_height_m": opening_bottom,
        "gate_opening_top_height_m": opening_top,
        "gate_center_height_m": gate_center,
        "gate_top_clearance_margin_m": top_margin,
        "drone_shell_top_m": float(fixed_height + drone_radius),
        "drone_shell_bottom_m": float(fixed_height - drone_radius),
        "drone_top_clearance_to_gate_top_m": float(opening_top - (fixed_height + drone_radius)),
        "drone_bottom_clearance_to_gate_bottom_m": float((fixed_height - drone_radius) - opening_bottom),
        "max_allowed_flight_height_m": max_allowed,
        "min_allowed_flight_height_m": min_allowed,
        "corridor_half_width_m": corridor_half_width,
        "corridor_through_required": bool(cfg.corridor_through_required),
        "side_bypass_policy": str(cfg.side_bypass_policy),
        "height_escape_policy": str(cfg.height_escape_policy),
    }


def corridor_region_status(
    positions_xy: np.ndarray,
    *,
    config: DynamicGateDensity2DConfig | None = None,
) -> dict[str, object]:
    """Return whether any active agent bypasses the obstacle corridor laterally."""

    cfg = config or default_dynamic_gate_density_config()
    positions = np.asarray(positions_xy, dtype=np.float32)
    if positions.size == 0:
        return {
            "inside_gate_region": False,
            "side_bypass_failure": False,
            "corridor_half_width_m": resolved_corridor_half_width_m(cfg),
            "max_abs_y_inside_region_m": 0.0,
        }
    margin = max(float(cfg.corridor_x_margin_m), 0.0)
    x_min = float(cfg.gate_region_x_m[0]) - margin
    x_max = float(cfg.gate_region_x_m[1]) + margin
    corridor_half_width = resolved_corridor_half_width_m(cfg)
    inside_mask = (positions[:, 0] >= x_min) & (positions[:, 0] <= x_max)
    inside = bool(np.any(inside_mask))
    if inside:
        max_abs_y = float(np.max(np.abs(positions[inside_mask, 1])))
    else:
        max_abs_y = 0.0
    return {
        "inside_gate_region": inside,
        "side_bypass_failure": bool(inside and max_abs_y > corridor_half_width),
        "corridor_half_width_m": float(corridor_half_width),
        "max_abs_y_inside_region_m": float(max_abs_y),
    }


def center_has_completed_corridor(
    center_xy: tuple[float, float] | np.ndarray,
    *,
    config: DynamicGateDensity2DConfig | None = None,
) -> bool:
    cfg = config or default_dynamic_gate_density_config()
    center = np.asarray(center_xy, dtype=np.float32)
    if center.shape != (2,):
        return False
    margin = max(float(cfg.corridor_x_margin_m), 0.0)
    return bool(
        float(center[0]) >= float(cfg.gate_region_x_m[1]) + margin
        and abs(float(center[1])) <= resolved_corridor_half_width_m(cfg)
    )


def validate_dynamic_gate_density_geometry(
    *,
    gate_count: int,
    speed_mps: float,
    amplitude_m: float,
    seed: int,
    config: DynamicGateDensity2DConfig | None = None,
    sample_times_s: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, object]:
    """Run geometry-only sanity checks before training/eval/replay.

    This catches fake dynamic scenes early: missing gates, frozen live centers,
    gate-gate post overlap, or a collision query that does not report contact.
    """

    cfg = config or default_dynamic_gate_density_config()
    invariant_report = validate_height_and_corridor_invariants(config=cfg)
    failures: list[str] = list(invariant_report["failures"])
    expected_count = int(max(0, min(int(cfg.max_gate_count), int(gate_count))))
    dynamic_motion_required = expected_count > 0 and float(speed_mps) > 0.0 and float(amplitude_m) > 0.0
    gates = generate_gate_layout(
        gate_count=gate_count,
        seed=seed,
        config=cfg,
        static_layout=not bool(dynamic_motion_required),
    )
    if len(gates) != expected_count:
        failures.append(f"gate_count_mismatch: expected {expected_count}, got {len(gates)}")
    if expected_count <= 0:
        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "gate_count": int(len(gates)),
            "expected_gate_count": expected_count,
            "dynamic_motion_required": False,
            "max_center_motion_m": 0.0,
            "max_velocity_mps": 0.0,
            "min_gate_gate_clearance_m": None,
            "gate_gate_overlap_pair_count": 0,
            "synthetic_live_collision_clearance_m": None,
            "synthetic_swept_collision_clearance_m": None,
            "height_and_corridor": invariant_report,
        }

    sampled_centers: list[np.ndarray] = []
    sampled_velocities: list[np.ndarray] = []
    min_gate_gate_clearance = float("inf")
    gate_gate_overlap_pair_count = 0
    for t_sec in sample_times_s:
        centers = live_gate_centers(
            gates,
            t_sec=float(t_sec),
            amplitude_m=float(amplitude_m),
            speed_mps=float(speed_mps),
            config=cfg,
        )
        velocities = live_gate_velocities(
            gates,
            t_sec=float(t_sec),
            dt_s=0.05,
            amplitude_m=float(amplitude_m),
            speed_mps=float(speed_mps),
            config=cfg,
        )
        if not np.all(np.isfinite(centers)):
            failures.append(f"non_finite_live_centers_at_t={t_sec}")
        if not np.all(np.isfinite(velocities)):
            failures.append(f"non_finite_live_velocities_at_t={t_sec}")
        stats = gate_gate_clearance_stats(gates, centers, config=cfg)
        min_gate_gate_clearance = min(
            min_gate_gate_clearance,
            float(stats["gate_gate_min_clearance_m"]),
        )
        gate_gate_overlap_pair_count += int(stats["gate_gate_overlap_pair_count"])
        sampled_centers.append(centers)
        sampled_velocities.append(velocities)

    if gate_gate_overlap_pair_count > 0:
        failures.append(f"gate_gate_overlap_pair_count={gate_gate_overlap_pair_count}")

    base_centers = sampled_centers[0]
    center_motion = [
        float(np.max(np.linalg.norm(centers - base_centers, axis=1)))
        for centers in sampled_centers[1:]
        if centers.size > 0
    ]
    max_center_motion = max(center_motion) if center_motion else 0.0
    sampled_speed_maxima = [
        float(np.max(np.linalg.norm(velocities, axis=1))) for velocities in sampled_velocities if velocities.size > 0
    ]
    max_velocity = max(sampled_speed_maxima, default=0.0)
    min_required_motion = max(0.05, 0.10 * float(amplitude_m))
    if dynamic_motion_required and max_center_motion < min_required_motion:
        failures.append(
            f"frozen_dynamic_gates: max_center_motion_m={max_center_motion:.4f} < required {min_required_motion:.4f}"
        )
    if dynamic_motion_required and max_velocity <= 1.0e-4:
        failures.append(f"zero_live_gate_velocity: max_velocity_mps={max_velocity:.6f}")

    posts_t0 = gate_posts(gates, sampled_centers[0], config=cfg)
    synthetic_live_clearance = None
    synthetic_swept_clearance = None
    if posts_t0.size > 0:
        contact_position = posts_t0[:1].copy()
        synthetic_live_clearance = post_clearance(contact_position, posts_t0, config=cfg)
        synthetic_swept_clearance = swept_post_clearance(
            contact_position,
            contact_position,
            posts_t0,
            posts_t0,
            config=cfg,
        )
        if synthetic_live_clearance > 0.0:
            failures.append(f"live_collision_not_detected: clearance={synthetic_live_clearance:.6f}")
        if synthetic_swept_clearance > 0.0:
            failures.append(f"swept_collision_not_detected: clearance={synthetic_swept_clearance:.6f}")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "gate_count": int(len(gates)),
        "expected_gate_count": expected_count,
        "dynamic_motion_required": bool(dynamic_motion_required),
        "max_center_motion_m": float(max_center_motion),
        "max_velocity_mps": float(max_velocity),
        "min_gate_gate_clearance_m": (
            None if not math.isfinite(min_gate_gate_clearance) else float(min_gate_gate_clearance)
        ),
        "gate_gate_overlap_pair_count": int(gate_gate_overlap_pair_count),
        "synthetic_live_collision_clearance_m": (
            None if synthetic_live_clearance is None else float(synthetic_live_clearance)
        ),
        "synthetic_swept_collision_clearance_m": (
            None if synthetic_swept_clearance is None else float(synthetic_swept_clearance)
        ),
        "height_and_corridor": invariant_report,
    }


def assert_dynamic_gate_density_geometry_sane(**kwargs: object) -> dict[str, object]:
    report = validate_dynamic_gate_density_geometry(**kwargs)
    if not bool(report.get("passed")):
        failures = ", ".join(str(item) for item in report.get("failures", []))
        raise AssertionError(f"Dynamic gate-density geometry sanity failed: {failures}")
    return report
