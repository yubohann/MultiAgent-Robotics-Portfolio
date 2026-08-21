"""Gate layout and moving-gate geometry helpers for single-drone density studies."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random


WORLD_X_BOUNDS_M = (-10.0, 10.0)
WORLD_Y_BOUNDS_M = (-4.0, 4.0)
GATE_BOTTOM_HEIGHT_M = 0.0
GATE_TOP_HEIGHT_M = 8.0
GATE_CENTER_HEIGHT_M = 0.5 * (GATE_BOTTOM_HEIGHT_M + GATE_TOP_HEIGHT_M)
GATE_NATIVE_VISUAL_HEIGHT_M = 4.2
GATE_VISUAL_SCALE_Z = GATE_TOP_HEIGHT_M / GATE_NATIVE_VISUAL_HEIGHT_M
START_XYZ = (-9.0, 0.0, GATE_CENTER_HEIGHT_M)
GOAL_XYZ = (9.0, 0.0, GATE_CENTER_HEIGHT_M)
GATE_REGION_X = (-6.5, 6.5)
GATE_REGION_Y = (-3.0, 3.0)
GATE_HALF_WIDTH_M = 1.05
GATE_POST_RADIUS_M = 0.32
MAX_MOVING_GATE_SPEED_MPS = 2.0
GATE_GATE_CLEARANCE_MARGIN_M = 0.12
GATE_GATE_FRAME_CLEARANCE_MARGIN_M = 0.06
GATE_VISUAL_FRAME_HALF_DEPTH_M = 0.42
DYNAMIC_GATE_NON_OVERLAP_ITERATIONS = 24
GATE_LAYOUT_VERSION = "irregular_centerline_v2"
ALLOWED_GATE_LAYOUT_VERSIONS = (
    "irregular_centerline_v2",
    "irregular_centerline_v3_heldout",
    "irregular_centerline_v4_stress_s_curve",
    "irregular_centerline_v5_dynamic_s_curve",
    "irregular_centerline_v6_large_motion_dynamic",
    "irregular_centerline_v7_large_arena_dynamic",
)


@dataclass(frozen=True)
class GateDensityLayoutProfile:
    world_x_bounds_m: tuple[float, float]
    world_y_bounds_m: tuple[float, float]
    start_xyz: tuple[float, float, float]
    goal_xyz: tuple[float, float, float]
    gate_region_x_m: tuple[float, float]
    gate_region_y_m: tuple[float, float]
    moving_clip_x_m: tuple[float, float]
    moving_clip_y_m: tuple[float, float]
    training_render_policy: str
    obstacle_dynamics_policy: str
    collision_policy: str
    distribution_policy: str


def _layout_profile(layout_version: str) -> GateDensityLayoutProfile:
    """Resolve geometry/rule knobs for each layout without mixing tables."""

    version = str(layout_version or GATE_LAYOUT_VERSION)
    base = GateDensityLayoutProfile(
        world_x_bounds_m=WORLD_X_BOUNDS_M,
        world_y_bounds_m=WORLD_Y_BOUNDS_M,
        start_xyz=START_XYZ,
        goal_xyz=GOAL_XYZ,
        gate_region_x_m=GATE_REGION_X,
        gate_region_y_m=GATE_REGION_Y,
        moving_clip_x_m=(GATE_REGION_X[0] + 0.20, GATE_REGION_X[1] - 0.20),
        moving_clip_y_m=(-1.95, 1.95),
        training_render_policy="no_camera_render; update dynamic gate centers analytically per env step",
        obstacle_dynamics_policy="kinematic_immovable_obstacles; contacts never topple gates",
        collision_policy="terminal_crash_on_contact; collision terminates episode immediately",
        distribution_policy="legacy_irregular_centerline",
    )
    if version == "irregular_centerline_v4_stress_s_curve":
        return replace(base, world_y_bounds_m=(-2.4, 2.4), distribution_policy="stress_s_curve")
    if version == "irregular_centerline_v5_dynamic_s_curve":
        return replace(base, world_y_bounds_m=(-2.8, 2.8), distribution_policy="historical_dynamic_s_curve")
    if version == "irregular_centerline_v6_large_motion_dynamic":
        return replace(
            base,
            world_y_bounds_m=(-2.7, 2.7),
            moving_clip_y_m=(-2.25, 2.25),
            distribution_policy="large_motion_staggered_lanes",
        )
    if version == "irregular_centerline_v7_large_arena_dynamic":
        return GateDensityLayoutProfile(
            world_x_bounds_m=(-30.0, 30.0),
            world_y_bounds_m=(-10.0, 10.0),
            start_xyz=(-27.0, 0.0, GATE_CENTER_HEIGHT_M),
            goal_xyz=(27.0, 0.0, GATE_CENTER_HEIGHT_M),
            gate_region_x_m=(-24.0, 24.0),
            gate_region_y_m=(-8.0, 8.0),
            moving_clip_x_m=(-25.5, 25.5),
            moving_clip_y_m=(-8.75, 8.75),
            training_render_policy="no_camera_render; vector envs update live gate centers in math only",
            obstacle_dynamics_policy=(
                "kinematic_immovable_obstacles; no toppling regardless of contact; "
                "gate_gate_overlap_resolved_before_collision_query"
            ),
            collision_policy="terminal_crash_on_contact; collision is a hard failure and immediate episode end",
            distribution_policy="large_arena_collision_safe_uniform_columns_and_lanes",
        )
    return base


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return max(float(bounds[0]), min(float(bounds[1]), float(value)))


def _resolve_moving_gate_speed_hz(
    *,
    amplitude_m: float,
    speed_hz: float,
    speed_mps: float,
) -> float:
    """Resolve moving-gate frequency, supporting a physical m/s speed cap."""

    resolved_speed_mps = float(speed_mps)
    if resolved_speed_mps > 0.0:
        if resolved_speed_mps > MAX_MOVING_GATE_SPEED_MPS:
            raise ValueError(
                f"moving gate speed_mps must be <= {MAX_MOVING_GATE_SPEED_MPS}; got {resolved_speed_mps}"
            )
        return float(resolved_speed_mps / (2.0 * math.pi * max(float(amplitude_m), 1.0e-6)))
    return float(speed_hz)


def _sample_gate_yaw_minus5_to_5_rad(rng: random.Random) -> float:
    """Sample the formation-facing obstacle yaw band: random -5 to +5 degrees."""

    return math.radians(float(rng.uniform(-5.0, 5.0)))


def _distance_point_to_segment_local(
    point_xy: tuple[float, float],
    seg_start_xy: tuple[float, float],
    seg_end_xy: tuple[float, float],
) -> float:
    px, py = float(point_xy[0]), float(point_xy[1])
    ax, ay = float(seg_start_xy[0]), float(seg_start_xy[1])
    bx, by = float(seg_end_xy[0]), float(seg_end_xy[1])
    abx = bx - ax
    aby = by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq <= 1.0e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_sq))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _moving_points_min_distance(
    a0_xy: tuple[float, float],
    a1_xy: tuple[float, float],
    b0_xy: tuple[float, float],
    b1_xy: tuple[float, float],
) -> float:
    """Minimum distance between two linearly moving planar points over one step."""

    rx = float(a0_xy[0]) - float(b0_xy[0])
    ry = float(a0_xy[1]) - float(b0_xy[1])
    vx = (float(a1_xy[0]) - float(a0_xy[0])) - (float(b1_xy[0]) - float(b0_xy[0]))
    vy = (float(a1_xy[1]) - float(a0_xy[1])) - (float(b1_xy[1]) - float(b0_xy[1]))
    vv = vx * vx + vy * vy
    if vv <= 1.0e-12:
        return math.hypot(rx, ry)
    alpha = max(0.0, min(1.0, -((rx * vx) + (ry * vy)) / vv))
    return math.hypot(rx + alpha * vx, ry + alpha * vy)


def _gate_post_centers_by_gate(
    gate_centers_xy: tuple[tuple[float, float], ...],
    gate_yaws: tuple[float, ...],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    posts_by_gate: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for gate_idx, center_xy in enumerate(gate_centers_xy):
        yaw_rad = gate_yaws[gate_idx] if gate_idx < len(gate_yaws) else 0.0
        side_xy = (-math.sin(yaw_rad), math.cos(yaw_rad))
        posts = []
        for sign in (1.0, -1.0):
            posts.append(
                (
                    float(center_xy[0] + sign * GATE_HALF_WIDTH_M * side_xy[0]),
                    float(center_xy[1] + sign * GATE_HALF_WIDTH_M * side_xy[1]),
                )
            )
        posts_by_gate.append((posts[0], posts[1]))
    return tuple(posts_by_gate)


def _gate_gate_clearance_stats(
    gate_centers_xy: tuple[tuple[float, float], ...],
    gate_yaws: tuple[float, ...],
) -> dict[str, float | int]:
    """Measure physical post-to-post clearance between different gates."""

    posts_by_gate = _gate_post_centers_by_gate(gate_centers_xy, gate_yaws)
    min_clearance = float("inf")
    overlap_count = 0
    checked_pairs = 0
    for gate_i in range(len(posts_by_gate)):
        for gate_j in range(gate_i + 1, len(posts_by_gate)):
            for post_i in posts_by_gate[gate_i]:
                for post_j in posts_by_gate[gate_j]:
                    checked_pairs += 1
                    distance_m = math.hypot(float(post_i[0]) - float(post_j[0]), float(post_i[1]) - float(post_j[1]))
                    clearance_m = distance_m - 2.0 * GATE_POST_RADIUS_M
                    min_clearance = min(min_clearance, clearance_m)
                    if clearance_m < 0.0:
                        overlap_count += 1
    if not math.isfinite(min_clearance):
        min_clearance = float("inf")
    return {
        "gate_gate_min_clearance_m": float(min_clearance),
        "gate_gate_overlap_pair_count": int(overlap_count),
        "gate_gate_checked_post_pair_count": int(checked_pairs),
    }


def _gate_frame_axes(yaw_rad: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return gate normal and side axes used by the thin visual-frame proxy."""

    normal_xy = (math.cos(float(yaw_rad)), math.sin(float(yaw_rad)))
    side_xy = (-math.sin(float(yaw_rad)), math.cos(float(yaw_rad)))
    return normal_xy, side_xy


def _gate_frame_pair_clearance_axis(
    center_a_xy: tuple[float, float],
    yaw_a_rad: float,
    center_b_xy: tuple[float, float],
    yaw_b_rad: float,
) -> tuple[float, tuple[float, float]]:
    """SAT clearance for two thin oriented gate-frame proxies."""

    normal_a, side_a = _gate_frame_axes(yaw_a_rad)
    normal_b, side_b = _gate_frame_axes(yaw_b_rad)
    axes = (normal_a, side_a, normal_b, side_b)
    delta_xy = (
        float(center_b_xy[0]) - float(center_a_xy[0]),
        float(center_b_xy[1]) - float(center_a_xy[1]),
    )
    half_side_m = GATE_HALF_WIDTH_M + GATE_POST_RADIUS_M
    half_depth_m = GATE_VISUAL_FRAME_HALF_DEPTH_M
    best_clearance = float("-inf")
    best_axis = axes[0]
    for axis_xy in axes:
        axis_norm = math.hypot(float(axis_xy[0]), float(axis_xy[1]))
        if axis_norm <= 1.0e-9:
            continue
        ux = float(axis_xy[0]) / axis_norm
        uy = float(axis_xy[1]) / axis_norm
        radius_a = (
            half_depth_m * abs(ux * normal_a[0] + uy * normal_a[1])
            + half_side_m * abs(ux * side_a[0] + uy * side_a[1])
        )
        radius_b = (
            half_depth_m * abs(ux * normal_b[0] + uy * normal_b[1])
            + half_side_m * abs(ux * side_b[0] + uy * side_b[1])
        )
        center_gap = abs(delta_xy[0] * ux + delta_xy[1] * uy)
        clearance = float(center_gap - radius_a - radius_b)
        if clearance > best_clearance:
            sign = 1.0 if (delta_xy[0] * ux + delta_xy[1] * uy) >= 0.0 else -1.0
            best_clearance = clearance
            best_axis = (sign * ux, sign * uy)
    return float(best_clearance), best_axis


def _gate_gate_frame_clearance_stats(
    gate_centers_xy: tuple[tuple[float, float], ...],
    gate_yaws: tuple[float, ...],
) -> dict[str, float | int]:
    """Measure visual frame-to-frame overlap with a thin oriented-rectangle proxy."""

    min_clearance = float("inf")
    overlap_count = 0
    checked_pairs = 0
    for gate_i in range(len(gate_centers_xy)):
        yaw_i = gate_yaws[gate_i] if gate_i < len(gate_yaws) else 0.0
        for gate_j in range(gate_i + 1, len(gate_centers_xy)):
            yaw_j = gate_yaws[gate_j] if gate_j < len(gate_yaws) else 0.0
            clearance_m, _axis = _gate_frame_pair_clearance_axis(
                gate_centers_xy[gate_i],
                yaw_i,
                gate_centers_xy[gate_j],
                yaw_j,
            )
            checked_pairs += 1
            min_clearance = min(min_clearance, clearance_m)
            if clearance_m < 0.0:
                overlap_count += 1
    if not math.isfinite(min_clearance):
        min_clearance = float("inf")
    return {
        "gate_gate_frame_min_clearance_m": float(min_clearance),
        "gate_gate_frame_overlap_pair_count": int(overlap_count),
        "gate_gate_frame_checked_pair_count": int(checked_pairs),
    }


def _enforce_gate_non_overlap(
    gate_centers_xy: tuple[tuple[float, float], ...],
    gate_yaws: tuple[float, ...],
    *,
    clip_x_m: tuple[float, float],
    clip_y_m: tuple[float, float],
    min_clearance_m: float = GATE_GATE_CLEARANCE_MARGIN_M,
    iterations: int = DYNAMIC_GATE_NON_OVERLAP_ITERATIONS,
) -> tuple[tuple[float, float], ...]:
    """Project moving gate centers so their post collision disks do not overlap."""

    if len(gate_centers_xy) <= 1:
        return gate_centers_xy
    centers = [[float(x), float(y)] for x, y in gate_centers_xy]
    required_gap_m = 2.0 * GATE_POST_RADIUS_M + max(float(min_clearance_m), 0.0)
    for _ in range(max(int(iterations), 0)):
        moved_any = False
        posts_by_gate = _gate_post_centers_by_gate(tuple((x, y) for x, y in centers), gate_yaws)
        for gate_i in range(len(centers)):
            for gate_j in range(gate_i + 1, len(centers)):
                closest_distance = float("inf")
                closest_delta = (0.0, 0.0)
                for post_i in posts_by_gate[gate_i]:
                    for post_j in posts_by_gate[gate_j]:
                        dx = float(post_j[0]) - float(post_i[0])
                        dy = float(post_j[1]) - float(post_i[1])
                        distance_m = math.hypot(dx, dy)
                        if distance_m < closest_distance:
                            closest_distance = distance_m
                            closest_delta = (dx, dy)
                if closest_distance >= required_gap_m:
                    continue
                if closest_distance <= 1.0e-9:
                    dx = centers[gate_j][0] - centers[gate_i][0]
                    dy = centers[gate_j][1] - centers[gate_i][1]
                    norm = math.hypot(dx, dy)
                    if norm <= 1.0e-9:
                        angle = 2.399963229728653 * (gate_i + 1) + 0.917 * (gate_j + 1)
                        ux, uy = math.cos(angle), math.sin(angle)
                    else:
                        ux, uy = dx / norm, dy / norm
                else:
                    ux, uy = closest_delta[0] / closest_distance, closest_delta[1] / closest_distance
                correction_m = 0.5 * (required_gap_m - closest_distance)
                centers[gate_i][0] = _clip(centers[gate_i][0] - correction_m * ux, clip_x_m)
                centers[gate_i][1] = _clip(centers[gate_i][1] - correction_m * uy, clip_y_m)
                centers[gate_j][0] = _clip(centers[gate_j][0] + correction_m * ux, clip_x_m)
                centers[gate_j][1] = _clip(centers[gate_j][1] + correction_m * uy, clip_y_m)
                moved_any = True
        for gate_i in range(len(centers)):
            yaw_i = gate_yaws[gate_i] if gate_i < len(gate_yaws) else 0.0
            for gate_j in range(gate_i + 1, len(centers)):
                yaw_j = gate_yaws[gate_j] if gate_j < len(gate_yaws) else 0.0
                clearance_m, axis_xy = _gate_frame_pair_clearance_axis(
                    (centers[gate_i][0], centers[gate_i][1]),
                    yaw_i,
                    (centers[gate_j][0], centers[gate_j][1]),
                    yaw_j,
                )
                if clearance_m >= GATE_GATE_FRAME_CLEARANCE_MARGIN_M:
                    continue
                correction_m = 0.5 * (GATE_GATE_FRAME_CLEARANCE_MARGIN_M - float(clearance_m))
                ux, uy = axis_xy
                centers[gate_i][0] = _clip(centers[gate_i][0] - correction_m * ux, clip_x_m)
                centers[gate_i][1] = _clip(centers[gate_i][1] - correction_m * uy, clip_y_m)
                centers[gate_j][0] = _clip(centers[gate_j][0] + correction_m * ux, clip_x_m)
                centers[gate_j][1] = _clip(centers[gate_j][1] + correction_m * uy, clip_y_m)
                moved_any = True
        if not moved_any:
            break
    return tuple((float(x), float(y)) for x, y in centers)


def _moving_gate_swept_clearance_m(
    *,
    drone_start_xy: tuple[float, float],
    drone_end_xy: tuple[float, float],
    gate_centers_start_xy: tuple[tuple[float, float], ...],
    gate_centers_end_xy: tuple[tuple[float, float], ...],
    gate_yaws: tuple[float, ...],
    drone_radius_m: float,
) -> float:
    """Continuous clearance against posts while both the drone and gates move."""

    start_posts = _gate_post_centers(gate_centers_start_xy, gate_yaws)
    end_posts = _gate_post_centers(gate_centers_end_xy, gate_yaws)
    count = min(len(start_posts), len(end_posts))
    if count <= 0:
        return float("inf")
    min_distance_m = float("inf")
    for idx in range(count):
        min_distance_m = min(
            min_distance_m,
            _moving_points_min_distance(drone_start_xy, drone_end_xy, start_posts[idx], end_posts[idx]),
        )
    return float(min_distance_m - GATE_POST_RADIUS_M - float(drone_radius_m))


def _generate_gate_layout(
    *,
    gate_count: int,
    seed: int,
    random_yaw: bool,
    layout_version: str = GATE_LAYOUT_VERSION,
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
    if gate_count <= 0:
        return tuple(), tuple()
    rng = random.Random(int(seed))
    layout_version = str(layout_version or GATE_LAYOUT_VERSION)
    profile = _layout_profile(layout_version)
    if layout_version == "irregular_centerline_v4_stress_s_curve":
        # Held-out S-curve layout for dense static stress tests.
        x_margin_m = 0.25
        usable_x_min = GATE_REGION_X[0] + x_margin_m
        usable_x_max = GATE_REGION_X[1] - x_margin_m
        x_step = (usable_x_max - usable_x_min) / max(gate_count - 1, 1)
        centers: list[tuple[float, float]] = []
        yaws: list[float] = []
        for idx in range(gate_count):
            x_base = usable_x_min + idx * x_step
            x = _clip(x_base + rng.uniform(-0.10, 0.10), (usable_x_min, usable_x_max))
            phase = idx / max(gate_count - 1, 1)
            s_curve = 1.18 * math.sin(2.25 * math.pi * phase + 0.45 * int(seed))
            pinch = 0.26 * ((-1.0) ** idx)
            y = _clip(s_curve + pinch + rng.uniform(-0.07, 0.07), (-1.65, 1.65))
            centers.append((float(x), float(y)))
            if random_yaw:
                base_yaw = math.pi / 2.0 if idx % 2 == 0 else -math.pi / 2.0
                yaw = base_yaw + 0.36 * math.sin(1.7 * idx + seed) + rng.uniform(-0.22, 0.22)
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            else:
                yaw = 0.0
            yaws.append(float(yaw))
        return tuple(centers), tuple(yaws)
    if layout_version == "irregular_centerline_v5_dynamic_s_curve":
        # Small-motion dynamic S-curve layout for 0..14 gate studies.
        x_margin_m = 0.50
        usable_x_min = GATE_REGION_X[0] + x_margin_m
        usable_x_max = GATE_REGION_X[1] - x_margin_m
        x_step = (usable_x_max - usable_x_min) / max(gate_count - 1, 1)
        centers: list[tuple[float, float]] = []
        yaws: list[float] = []
        for idx in range(gate_count):
            x_base = usable_x_min + idx * x_step
            x_jitter = 0.28 if gate_count <= 6 else 0.16
            x = _clip(x_base + rng.uniform(-x_jitter, x_jitter), (usable_x_min, usable_x_max))
            phase = idx / max(gate_count - 1, 1)
            s_curve = 1.05 * math.sin(2.05 * math.pi * phase + 0.31 * int(seed))
            center_bias = 0.24 * math.sin(1.63 * idx + 0.71 * int(seed))
            density_pull = 0.18 * ((-1.0) ** (idx + seed)) if gate_count >= 8 else 0.0
            y = _clip(s_curve + center_bias + density_pull + rng.uniform(-0.10, 0.10), (-1.75, 1.75))
            centers.append((float(x), float(y)))
            if random_yaw:
                base_yaw = 0.70 * math.pi if idx % 2 == 0 else -0.70 * math.pi
                if gate_count <= 4:
                    base_yaw *= 0.55
                yaw = base_yaw + 0.28 * math.sin(1.9 * idx + 0.5 * seed) + rng.uniform(-0.24, 0.24)
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            else:
                yaw = 0.0
            yaws.append(float(yaw))
        return tuple(centers), tuple(yaws)
    if layout_version == "irregular_centerline_v6_large_motion_dynamic":
        # Large-motion dynamic layout with staggered lanes and side-bypass pressure.
        x_margin_m = 0.55
        usable_x_min = GATE_REGION_X[0] + x_margin_m
        usable_x_max = GATE_REGION_X[1] - x_margin_m
        x_step = (usable_x_max - usable_x_min) / max(gate_count - 1, 1)
        lane_pattern = (0.0, -1.20, 1.20, -0.55, 0.55, -1.55, 1.55, -0.90, 0.90)
        centers: list[tuple[float, float]] = []
        yaws: list[float] = []
        for idx in range(gate_count):
            x_base = usable_x_min + idx * x_step
            x_jitter = 0.16 if gate_count <= 8 else 0.08
            x = _clip(x_base + rng.uniform(-x_jitter, x_jitter), (usable_x_min, usable_x_max))
            phase = idx / max(gate_count - 1, 1)
            lane_bias = lane_pattern[(idx + 2 * int(seed)) % len(lane_pattern)]
            center_wave = 0.24 * math.sin(2.20 * math.pi * phase + 0.37 * int(seed))
            local_bias = 0.16 * math.sin(1.41 * idx + 0.63 * int(seed))
            y = _clip(lane_bias + center_wave + local_bias + rng.uniform(-0.07, 0.07), (-1.65, 1.65))
            centers.append((float(x), float(y)))
            if random_yaw:
                # Keep the yaw continuous but biased toward cross-corridor
                # Gates, so moving posts actively open and close passages.
                base_yaw = 0.62 * math.pi if idx % 2 == 0 else -0.62 * math.pi
                yaw = base_yaw + 0.24 * math.sin(1.8 * idx + 0.4 * seed) + rng.uniform(-0.20, 0.20)
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            else:
                yaw = 0.0
            yaws.append(float(yaw))
        return tuple(centers), tuple(yaws)
    if layout_version == "irregular_centerline_v7_large_arena_dynamic":
        # Large-arena dynamic layout for up to 60 gates.
        x_min, x_max = profile.gate_region_x_m
        y_min, y_max = (-7.20, 7.20)
        row_count = max(3, min(6, int(math.ceil(gate_count / 10.0))))
        col_count = int(math.ceil(gate_count / row_count))
        lane_values = [0.0] if row_count == 1 else [
            y_min + row_idx * (y_max - y_min) / max(row_count - 1, 1)
            for row_idx in range(row_count)
        ]
        centers: list[tuple[float, float]] = []
        yaws: list[float] = []
        for idx in range(gate_count):
            col = idx // row_count
            row = (idx + 2 * int(seed) + (idx // max(row_count, 1))) % row_count
            col_phase = col / max(col_count - 1, 1)
            x_base = x_min + col_phase * (x_max - x_min)
            lane = lane_values[row]
            x_spacing = (x_max - x_min) / max(col_count - 1, 1)
            y_spacing = (y_max - y_min) / max(row_count - 1, 1)
            x = _clip(
                x_base + rng.uniform(-0.22, 0.22) * max(x_spacing, 0.5),
                (x_min, x_max),
            )
            y = _clip(
                lane
                + 0.22 * y_spacing * math.sin(0.91 * idx + 0.37 * int(seed))
                + rng.uniform(-0.10, 0.10) * max(y_spacing, 0.5),
                (y_min, y_max),
            )
            centers.append((float(x), float(y)))
            yaw = _sample_gate_yaw_minus5_to_5_rad(rng) if random_yaw else 0.0
            yaws.append(float(yaw))
        centers = list(
            _enforce_gate_non_overlap(
                tuple(centers),
                tuple(yaws),
                clip_x_m=profile.gate_region_x_m,
                clip_y_m=(y_min, y_max),
            )
        )
        return tuple(centers), tuple(yaws)
    if layout_version == "irregular_centerline_v3_heldout":
        # Held-out centerline layout with near-transverse yaw pressure.
        x_margin_m = 0.40
        usable_x_min = GATE_REGION_X[0] + x_margin_m
        usable_x_max = GATE_REGION_X[1] - x_margin_m
        x_step = (usable_x_max - usable_x_min) / max(gate_count - 1, 1)
        y_pattern = (0.0, 0.42, -0.42, 0.88, -0.88, 0.18, -0.18, 1.22, -1.22)
        centers: list[tuple[float, float]] = []
        yaws: list[float] = []
        for idx in range(gate_count):
            x_base = usable_x_min + idx * x_step
            # Reduce x jitter at higher density so gates form a narrow slalom.
            jitter_scale = 0.30 if gate_count <= 6 else 0.18
            x = _clip(x_base + rng.uniform(-jitter_scale, jitter_scale), (usable_x_min, usable_x_max))
            y_base = y_pattern[(idx + 2 * int(seed)) % len(y_pattern)]
            centerline_wave = 0.28 * math.sin(1.31 * idx + 0.77 * int(seed))
            y = _clip(y_base + centerline_wave + rng.uniform(-0.12, 0.12), (-2.15, 2.15))
            centers.append((float(x), float(y)))
            if random_yaw:
                # Bias some gates toward cross-corridor orientations.
                base_yaw = (math.pi / 2.0) if (idx + seed) % 2 == 0 else (-math.pi / 2.0)
                yaw = base_yaw + rng.uniform(-0.62, 0.62)
                if idx % 3 == 1:
                    yaw += rng.uniform(-0.95, 0.95)
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            else:
                yaw = 0.0
            yaws.append(float(yaw))
        return tuple(centers), tuple(yaws)
    if layout_version != "irregular_centerline_v2":
        raise ValueError(f"Unsupported gate layout version: {layout_version}")
    # Irregular centerline layout with near-center and side blockers.
    x_margin_m = 0.55
    usable_x_min = GATE_REGION_X[0] + x_margin_m
    usable_x_max = GATE_REGION_X[1] - x_margin_m
    x_step = (usable_x_max - usable_x_min) / max(gate_count - 1, 1)
    y_pattern = (0.0, -2.25, 2.25, -0.75, 0.75, -1.55, 1.55)
    centers: list[tuple[float, float]] = []
    yaws: list[float] = []
    for idx in range(gate_count):
        x_base = usable_x_min + idx * x_step
        x = _clip(x_base + rng.uniform(-0.38, 0.38), (usable_x_min, usable_x_max))
        y_base = y_pattern[(idx + int(seed)) % len(y_pattern)]
        centerline_pull = 0.55 * math.sin(0.83 * idx + 1.37 * int(seed))
        y = _clip(y_base + centerline_pull + rng.uniform(-0.22, 0.22), GATE_REGION_Y)
        centers.append((float(x), float(y)))
        yaws.append(float(rng.uniform(-math.pi, math.pi) if random_yaw else 0.0))
    return tuple(centers), tuple(yaws)


def _gate_post_centers(
    gate_centers_xy: tuple[tuple[float, float], ...],
    gate_yaws: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    return tuple(post for gate_posts in _gate_post_centers_by_gate(gate_centers_xy, gate_yaws) for post in gate_posts)


def _moving_gate_centers(
    *,
    base_centers_xy: tuple[tuple[float, float], ...],
    gate_yaws: tuple[float, ...] | None = None,
    seed: int,
    t_sec: float,
    enabled: bool,
    amplitude_m: float,
    speed_hz: float,
    layout_version: str = GATE_LAYOUT_VERSION,
) -> tuple[tuple[float, float], ...]:
    if not enabled or amplitude_m <= 0.0 or not base_centers_xy:
        if gate_yaws is not None and layout_version in {
            "irregular_centerline_v6_large_motion_dynamic",
            "irregular_centerline_v7_large_arena_dynamic",
        }:
            profile = _layout_profile(layout_version)
            return _enforce_gate_non_overlap(
                base_centers_xy,
                tuple(gate_yaws),
                clip_x_m=profile.gate_region_x_m,
                clip_y_m=profile.gate_region_y_m,
            )
        return base_centers_xy
    moved: list[tuple[float, float]] = []
    layout_version = str(layout_version or GATE_LAYOUT_VERSION)
    profile = _layout_profile(layout_version)
    for idx, (base_x, base_y) in enumerate(base_centers_xy):
        phase = 0.73 * int(seed) + 1.11 * idx
        omega_t = 2.0 * math.pi * float(speed_hz) * float(t_sec)
        if layout_version in {
            "irregular_centerline_v6_large_motion_dynamic",
            "irregular_centerline_v7_large_arena_dynamic",
        }:
            # Mix three motion families: lateral sweep, diagonal drift, and
            # anti-phase motion that opens and closes passages for replanning.
            mode = idx % 3
            if mode == 0:
                dy = float(amplitude_m) * math.sin(omega_t + phase)
                dx = 0.10 * float(amplitude_m) * math.sin(0.43 * omega_t + phase + 0.35)
            elif mode == 1:
                dy = 0.86 * float(amplitude_m) * math.sin(omega_t + phase)
                dx = 0.42 * float(amplitude_m) * math.sin(0.67 * omega_t + phase + 0.90)
            else:
                pair_phase = 0.53 * int(seed) + 0.71 * (idx // 2)
                anti_phase = 1.0 if idx % 2 == 0 else -1.0
                dy = anti_phase * float(amplitude_m) * math.sin(omega_t + pair_phase)
                dx = 0.36 * float(amplitude_m) * math.sin(0.59 * omega_t + pair_phase + 1.20)
            moved.append(
                (
                    float(_clip(base_x + dx, profile.moving_clip_x_m)),
                    float(_clip(base_y + dy, profile.moving_clip_y_m)),
                )
            )
        else:
            # Keep the v5 small-motion profile for comparable results.
            dy = amplitude_m * math.sin(omega_t + phase)
            dx = 0.25 * amplitude_m * math.sin(0.61 * omega_t + phase + 0.9)
            moved.append((float(_clip(base_x + dx, GATE_REGION_X)), float(_clip(base_y + dy, (-1.95, 1.95)))))
    moved_tuple = tuple(moved)
    if gate_yaws is not None and layout_version in {
        "irregular_centerline_v6_large_motion_dynamic",
        "irregular_centerline_v7_large_arena_dynamic",
    }:
        moved_tuple = _enforce_gate_non_overlap(
            moved_tuple,
            tuple(gate_yaws),
            clip_x_m=profile.moving_clip_x_m,
            clip_y_m=profile.moving_clip_y_m,
        )
    return moved_tuple

