"""Train and evaluate the 8-drone dynamic gate curriculum."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np


def _bootstrap_shared_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_bootstrap_shared_imports()

from shared.core.dynamic_gate_density_2d import (  # noqa: E402
    DynamicGate2D as SharedGate,
    default_dynamic_gate_density_config,
    drone_accel_limit_for_speed_mps2,
    eval_drone_speed_axis_mps,
    gate_posts as shared_gate_posts,
    generate_gate_layout,
    gate_gate_clearance_stats,
    live_gate_centers as shared_live_gate_centers,
    live_gate_velocities,
    post_clearance,
    speed_gradient_for_stage,
    swept_post_clearance,
)


DGD_CONFIG = default_dynamic_gate_density_config()
TEAM_SIZE = 8
DT_S = 0.1
MAX_STEPS = 360
DRONE_RADIUS_M = DGD_CONFIG.drone_radius_m
AGENT_AGENT_COLLISION_RADIUS_M = 0.30
GATE_POST_RADIUS_M = DGD_CONFIG.gate_post_radius_m
GATE_HALF_WIDTH_M = DGD_CONFIG.gate_half_width_m
WORLD_X_BOUNDS_M = DGD_CONFIG.world_x_bounds_m
WORLD_Y_BOUNDS_M = DGD_CONFIG.world_y_bounds_m
START_X_M = DGD_CONFIG.start_x_m
GOAL_X_M = DGD_CONFIG.goal_x_m
START_Y_M = 0.0
GOAL_Y_M = 0.0
GOAL_RADIUS_M = 1.6
SLOT_SUCCESS_TOLERANCE_M = 2.4
MAX_GATE_COUNT = DGD_CONFIG.max_gate_count
MAX_MOVING_GATE_SPEED_MPS = DGD_CONFIG.max_moving_gate_speed_mps
CONTROL_LOOKAHEAD_M = 3.2


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    gate_count: int
    moving_gate_speed_mps: float
    moving_gate_amplitude_m: float
    train_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    train_episodes_per_seed: int = 2
    eval_episodes_per_seed: int = 4
    drone_base_speed_mps: float = 2.35
    drone_accel_limit_mps2: float = 5.00


@dataclass(frozen=True)
class ControllerParams:
    base_speed_mps: float = 2.35
    slot_gain: float = 0.88
    center_gain: float = 0.42
    planner_y_gain: float = 0.75
    planner_gate_gain: float = 1.05
    shield_gate_gain: float = 1.35
    shield_agent_gain: float = 1.10
    risk_slowdown: float = 0.34
    guidance_prediction_gain: float = 1.00
    guidance_speed_bias: float = 0.12
    formation_scale: float = 0.72

    def clipped(self) -> "ControllerParams":
        return ControllerParams(
            base_speed_mps=float(np.clip(self.base_speed_mps, 1.35, 3.50)),
            slot_gain=float(np.clip(self.slot_gain, 0.35, 1.45)),
            center_gain=float(np.clip(self.center_gain, 0.15, 0.95)),
            planner_y_gain=float(np.clip(self.planner_y_gain, 0.15, 1.50)),
            planner_gate_gain=float(np.clip(self.planner_gate_gain, 0.25, 2.20)),
            shield_gate_gain=float(np.clip(self.shield_gate_gain, 0.20, 3.00)),
            shield_agent_gain=float(np.clip(self.shield_agent_gain, 0.20, 2.60)),
            risk_slowdown=float(np.clip(self.risk_slowdown, 0.00, 0.75)),
            guidance_prediction_gain=float(np.clip(self.guidance_prediction_gain, 0.00, 2.20)),
            guidance_speed_bias=float(np.clip(self.guidance_speed_bias, 0.00, 0.40)),
            formation_scale=float(np.clip(self.formation_scale, 0.58, 1.05)),
        )


@dataclass(frozen=True)
class ModuleToggles:
    fast_policy: bool = True
    slow_planner: bool = True
    safety_shield: bool = True
    route_guidance: bool = True
    guidance_shadow: bool = False


@dataclass
class EpisodeMetrics:
    success: bool
    done_reason: str
    steps: int
    flight_time_s: float
    team_success_rate: float
    per_agent_success_rate: float
    agent_agent_collision: bool
    obstacle_collision: bool
    out_of_bounds: bool
    timeout: bool
    min_pair_distance_m: float
    min_obstacle_clearance_m: float
    mean_slot_error_m: float
    max_slot_error_m: float
    progress_distance_m: float
    dispersed_termination: bool
    shield_activation_count: int
    route_guidance_used_count: int
    actual_gate_motion_range_m: float
    mean_speed_mps: float
    max_speed_mps: float


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _slot_offsets(scale: float) -> np.ndarray:
    # Compact 8-drone shell: two longitudinal rows and four lateral lanes.
    raw = np.asarray(
        [
            [-0.85, -2.10],
            [-0.85, -0.70],
            [-0.85, 0.70],
            [-0.85, 2.10],
            [0.85, -2.10],
            [0.85, -0.70],
            [0.85, 0.70],
            [0.85, 2.10],
        ],
        dtype=np.float32,
    )
    return raw * float(scale)


def _rotation(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def generate_gates(gate_count: int, seed: int) -> list[SharedGate]:
    return generate_gate_layout(gate_count=int(gate_count), seed=int(seed), config=DGD_CONFIG)


def _moving_gate_speed_hz(amplitude_m: float, speed_mps: float) -> float:
    if amplitude_m <= 1e-6 or speed_mps <= 1e-6:
        return 0.0
    return float(min(MAX_MOVING_GATE_SPEED_MPS, speed_mps) / max(2.0 * math.pi * amplitude_m, 1e-6))


def live_gate_centers(gates: list[SharedGate], *, t_sec: float, amplitude_m: float, speed_mps: float) -> np.ndarray:
    return shared_live_gate_centers(
        gates,
        t_sec=float(t_sec),
        amplitude_m=float(amplitude_m),
        speed_mps=float(speed_mps),
        config=DGD_CONFIG,
    )


def gate_posts(gates: list[SharedGate], centers: np.ndarray) -> np.ndarray:
    return shared_gate_posts(gates, centers, config=DGD_CONFIG)


def _pair_min_distance(points: np.ndarray) -> float:
    if len(points) <= 1:
        return float("inf")
    best = float("inf")
    for i in range(len(points)):
        deltas = points[i + 1 :] - points[i]
        if len(deltas):
            best = min(best, float(np.min(np.linalg.norm(deltas, axis=1))))
    return best


def _obstacle_clearance(positions: np.ndarray, posts: np.ndarray) -> float:
    return post_clearance(positions, posts, config=DGD_CONFIG)


def _post_velocity_estimate(
    gates: list[SharedGate],
    *,
    t_sec: float,
    amplitude_m: float,
    speed_mps: float,
) -> np.ndarray:
    centers_a = live_gate_centers(gates, t_sec=t_sec, amplitude_m=amplitude_m, speed_mps=speed_mps)
    centers_b = live_gate_centers(gates, t_sec=t_sec + DT_S, amplitude_m=amplitude_m, speed_mps=speed_mps)
    posts_a = gate_posts(gates, centers_a)
    posts_b = gate_posts(gates, centers_b)
    if len(posts_a) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return (posts_b - posts_a) / DT_S


def _center_velocity_estimate(
    gates: list[SharedGate],
    *,
    t_sec: float,
    amplitude_m: float,
    speed_mps: float,
) -> np.ndarray:
    return live_gate_velocities(
        gates,
        t_sec=float(t_sec),
        dt_s=DT_S,
        amplitude_m=float(amplitude_m),
        speed_mps=float(speed_mps),
        config=DGD_CONFIG,
    )


def _risk_guidance(
    *,
    team_center: np.ndarray,
    posts: np.ndarray,
    post_velocities: np.ndarray,
    params: ControllerParams,
    toggles: ModuleToggles,
    base_speed_mps: float,
) -> tuple[float, float, int]:
    if not toggles.slow_planner or len(posts) == 0:
        return 0.0, 1.0, 0
    lookahead_m = 8.0 if toggles.route_guidance else 5.5
    y_bias = 0.0
    risk = 0.0
    used = 0
    for idx, post in enumerate(posts):
        dx = float(post[0] - team_center[0])
        if dx < -1.0 or dx > lookahead_m:
            continue
        time_to_post = max(dx / max(base_speed_mps, 0.3), 0.0)
        predicted = post.copy()
        if toggles.route_guidance or toggles.guidance_shadow:
            predicted = post + post_velocities[idx] * time_to_post * float(params.guidance_prediction_gain)
        lateral = float(predicted[1] - team_center[1])
        influence = math.exp(-abs(dx) / max(lookahead_m, 1e-6)) * max(0.0, 2.9 - abs(lateral)) / 2.9
        if influence <= 0.0:
            continue
        used += 1
        risk += influence
        y_bias -= math.copysign(float(params.planner_gate_gain) * influence, lateral if abs(lateral) > 1e-4 else 1.0)
    y_bias = float(np.clip(y_bias, -4.2, 4.2))
    speed_scale = float(np.clip(1.0 - params.risk_slowdown * min(risk, 1.6), 0.35, 1.0))
    if toggles.route_guidance:
        speed_scale = float(np.clip(speed_scale - params.guidance_speed_bias * min(risk, 1.0), 0.28, 1.0))
    return y_bias, speed_scale, used


def _dynamic_channel_bias(
    *,
    gates: list[SharedGate],
    t_sec: float,
    amplitude_m: float,
    speed_mps: float,
    team_center: np.ndarray,
    centers: np.ndarray,
    center_velocities: np.ndarray,
    toggles: ModuleToggles,
    base_speed_mps: float,
) -> tuple[float, int]:
    if not toggles.slow_planner or len(centers) == 0:
        return 0.0, 0
    lookahead_m = 9.5 if toggles.route_guidance else 6.0
    weighted_y = 0.0
    weight_sum = 0.0
    used = 0
    for idx, center in enumerate(centers):
        dx = float(center[0] - team_center[0])
        if dx < -0.8 or dx > lookahead_m:
            continue
        # The center lane is the pass-through corridor for the 8-drone shell;
        # side lanes remain obstacles but should not pull the formation away.
        if abs(float(center[1])) > 4.2:
            continue
        predicted = center.copy()
        if toggles.route_guidance:
            time_to_gate = max(dx / max(base_speed_mps, 0.3), 0.0)
            if idx < len(center_velocities):
                predicted = center + center_velocities[idx] * time_to_gate
        weight = math.exp(-max(dx, 0.0) / max(lookahead_m, 1e-6)) / (0.35 + max(dx, 0.0))
        weighted_y += float(predicted[1]) * weight
        weight_sum += weight
        used += 1
    if weight_sum <= 1e-6:
        return 0.0, 0
    target_y = weighted_y / weight_sum
    return float(np.clip(target_y - float(team_center[1]), -3.0, 3.0)), used


def _adaptive_formation_offsets(
    *,
    slots: np.ndarray,
    gates: list[SharedGate],
    centers: np.ndarray,
    center_velocities: np.ndarray,
    team_center: np.ndarray,
    t_sec: float,
    amplitude_m: float,
    speed_mps: float,
    base_speed_mps: float,
    toggles: ModuleToggles,
) -> tuple[np.ndarray, float, float]:
    """Deform the formation near gate openings, then recover afterwards.

    The policy is obstacle-first: when a center-lane gate is near, lateral slot
    offsets are compressed enough to fit through the tightened opening and the
    two longitudinal rows are staggered slightly.  This allows temporary
    relative-position changes without changing the final formation objective.
    """

    if len(gates) == 0 or len(centers) == 0:
        return slots, 0.0, 0.0

    weighted_y = 0.0
    weight_sum = 0.0
    deformation = 0.0
    for idx, (gate, center) in enumerate(zip(gates, centers)):
        if int(getattr(gate, "lane_index", 0)) != 0:
            continue
        dx = float(center[0] - team_center[0])
        if dx < -2.4 or dx > 13.0:
            continue
        predicted = center.copy()
        if toggles.route_guidance:
            time_to_gate = max(dx / max(base_speed_mps, 0.35), 0.0)
            if idx < len(center_velocities):
                predicted = center + center_velocities[idx] * time_to_gate
        # Strongest deformation right before and during the gate crossing.
        local = math.exp(-abs(dx) / 5.2)
        if -0.8 <= dx <= 5.5:
            local = max(local, 0.95)
        elif 5.5 < dx <= 13.0:
            local = max(local, 0.45)
        weight = local / (0.55 + max(dx, 0.0))
        weighted_y += float(predicted[1]) * weight
        weight_sum += weight
        deformation = max(deformation, float(np.clip(local, 0.0, 1.0)))

    if weight_sum <= 1.0e-6 or deformation <= 1.0e-6:
        return slots, 0.0, 0.0

    pass_center_y = float(np.clip(weighted_y / weight_sum, -7.6, 7.6))
    adaptive = slots.astype(np.float32, copy=True)
    max_lateral = max(float(np.max(np.abs(slots[:, 1]))), 1.0e-6)
    tight_corridor_half = max(
        0.68,
        float(GATE_HALF_WIDTH_M) - float(GATE_POST_RADIUS_M) - float(DRONE_RADIUS_M) - 0.85,
    )
    compressed_scale = min(1.0, tight_corridor_half / max_lateral)
    lateral_scale = (1.0 - deformation) + deformation * compressed_scale
    adaptive[:, 1] = adaptive[:, 1] * lateral_scale

    # Short stagger lets front/back rows avoid sweeping the same gate boundary
    # at the exact same moment, while staying recoverable after the gate.
    row_sign = np.sign(adaptive[:, 0])
    lane_phase = adaptive[:, 1] / max(max_lateral * lateral_scale, 1.0e-6)
    adaptive[:, 0] = adaptive[:, 0] + deformation * (0.60 * row_sign + 1.15 * lane_phase)
    return adaptive, deformation, pass_center_y


def _shield_adjustment(
    *,
    positions: np.ndarray,
    posts: np.ndarray,
    actions: np.ndarray,
    params: ControllerParams,
    toggles: ModuleToggles,
) -> tuple[np.ndarray, int]:
    if not toggles.safety_shield:
        return actions, 0
    adjusted = actions.copy()
    activations = 0
    if len(posts):
        for i, pos in enumerate(positions):
            delta = pos - posts
            dist = np.linalg.norm(delta, axis=1)
            close = dist < 2.05
            if np.any(close):
                rep = np.zeros(2, dtype=np.float32)
                for d, r in zip(delta[close], dist[close]):
                    if float(r) <= 1e-6:
                        continue
                    strength = (2.05 - float(r)) / 2.05
                    rep += d / float(r) * strength
                adjusted[i] += rep * float(params.shield_gate_gain)
                activations += 1
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta = positions[i] - positions[j]
            dist = float(np.linalg.norm(delta))
            agent_keepout = 1.65
            if dist < agent_keepout and dist > 1e-6:
                rep = delta / dist * ((agent_keepout - dist) / agent_keepout) * float(params.shield_agent_gain)
                adjusted[i] += rep
                adjusted[j] -= rep
                activations += 1
    return adjusted, activations


def run_episode(
    *,
    stage: CurriculumStage,
    params: ControllerParams,
    seed: int,
    episode_index: int,
    toggles: ModuleToggles,
    save_trace_path: Path | None = None,
) -> EpisodeMetrics:
    rng = np.random.default_rng(int(seed) * 1009 + int(episode_index) * 17)
    gates = generate_gates(stage.gate_count, seed=seed)
    slots = _slot_offsets(params.formation_scale)
    positions = np.asarray([START_X_M, START_Y_M], dtype=np.float32) + slots
    positions += rng.normal(0.0, 0.05, size=positions.shape).astype(np.float32)
    velocities = np.zeros_like(positions)

    min_pair = float("inf")
    min_clearance = float("inf")
    slot_error_sum = 0.0
    slot_error_max = 0.0
    samples = 0
    shield_count = 0
    guidance_used = 0
    mean_speed_sum = 0.0
    max_speed_seen = 0.0
    gate_motion_min = np.full((len(gates), 2), np.inf, dtype=np.float32)
    gate_motion_max = np.full((len(gates), 2), -np.inf, dtype=np.float32)
    done_reason = "timeout"
    trace_rows: list[dict[str, object]] = []
    center_cache: dict[float, np.ndarray] = {}

    def cached_live_gate_centers(t_sec: float) -> np.ndarray:
        key = round(float(t_sec), 4)
        cached = center_cache.get(key)
        if cached is None:
            cached = live_gate_centers(
                gates,
                t_sec=float(t_sec),
                amplitude_m=stage.moving_gate_amplitude_m,
                speed_mps=stage.moving_gate_speed_mps,
            )
            center_cache[key] = cached
        return cached

    for step in range(MAX_STEPS):
        t_sec = step * DT_S
        centers = cached_live_gate_centers(t_sec)
        next_centers = cached_live_gate_centers(t_sec + DT_S)
        if len(centers):
            gate_motion_min = np.minimum(gate_motion_min, centers)
            gate_motion_max = np.maximum(gate_motion_max, centers)
        posts = gate_posts(gates, centers)
        next_posts_for_velocity = gate_posts(gates, next_centers)
        center_velocities = (
            (next_centers - centers) / DT_S if len(centers) else np.zeros((0, 2), dtype=np.float32)
        )
        post_velocities = (
            (next_posts_for_velocity - posts) / DT_S if len(posts) else np.zeros((0, 2), dtype=np.float32)
        )
        team_center = positions.mean(axis=0)
        stage_speed_cap = float(max(stage.drone_base_speed_mps, 0.3))
        param_bias = float(np.clip(params.base_speed_mps - 2.35, -1.00, 1.15))
        base_speed = stage_speed_cap + 0.20 * param_bias if toggles.fast_policy else min(1.65, stage_speed_cap)
        base_speed = float(np.clip(base_speed, 0.8, stage_speed_cap))
        y_bias, speed_scale, guidance_count = _risk_guidance(
            team_center=team_center,
            posts=posts,
            post_velocities=post_velocities,
            params=params,
            toggles=toggles,
            base_speed_mps=base_speed,
        )
        channel_bias, channel_count = _dynamic_channel_bias(
            gates=gates,
            t_sec=t_sec,
            amplitude_m=stage.moving_gate_amplitude_m,
            speed_mps=stage.moving_gate_speed_mps,
            team_center=team_center,
            centers=centers,
            center_velocities=center_velocities,
            toggles=toggles,
            base_speed_mps=base_speed,
        )
        if toggles.slow_planner and len(gates):
            density = min(float(stage.gate_count) / float(MAX_GATE_COUNT), 1.0)
            dynamic = min(float(stage.moving_gate_speed_mps) / float(MAX_MOVING_GATE_SPEED_MPS), 1.0)
            speed_scale *= float(np.clip(1.0 - 0.10 * density * dynamic, 0.84, 1.0))
            speed_scale *= float(np.clip(1.0 - 0.05 * abs(channel_bias), 0.86, 1.0))
            speed_scale = max(speed_scale, 0.64)
        if toggles.guidance_shadow:
            speed_scale = min(speed_scale + float(params.guidance_speed_bias), 1.0)
        guidance_used += int((guidance_count + channel_count) if toggles.route_guidance else 0)
        adaptive_offsets, formation_deformation, pass_center_y = _adaptive_formation_offsets(
            slots=slots,
            gates=gates,
            centers=centers,
            center_velocities=center_velocities,
            team_center=team_center,
            t_sec=t_sec,
            amplitude_m=stage.moving_gate_amplitude_m,
            speed_mps=stage.moving_gate_speed_mps,
            base_speed_mps=base_speed,
            toggles=toggles,
        )
        if toggles.slow_planner and formation_deformation > 1.0e-6:
            aperture_slowdown = float(np.clip(1.0 - 0.34 * formation_deformation, 0.52, 1.0))
            speed_scale = max(0.45, speed_scale * aperture_slowdown)

        route_anchor_x = float(np.clip(team_center[0], START_X_M, GOAL_X_M))
        adaptive_center_y = (1.0 - formation_deformation) * GOAL_Y_M + formation_deformation * pass_center_y
        tracking_center = np.asarray([route_anchor_x, adaptive_center_y], dtype=np.float32)
        desired_center = np.asarray(
            [min(GOAL_X_M, route_anchor_x + CONTROL_LOOKAHEAD_M), adaptive_center_y],
            dtype=np.float32,
        )
        if toggles.slow_planner:
            risk_weight = 1.0 - 0.85 * formation_deformation
            channel_weight = 1.15 + 1.15 * formation_deformation
            desired_center[1] += risk_weight * float(params.planner_y_gain) * y_bias
            desired_center[1] += channel_weight * float(channel_bias)
        desired_center[1] = float(np.clip(desired_center[1], -7.8, 7.8))
        desired_slots = desired_center + adaptive_offsets
        tracking_slots = tracking_center + adaptive_offsets

        goal_vec = desired_center - team_center
        goal_dir = goal_vec / max(float(np.linalg.norm(goal_vec)), 1e-6)
        cruise = goal_dir * base_speed * speed_scale
        if toggles.fast_policy:
            slot_error = desired_slots - positions
            effective_slot_gain = float(params.slot_gain) * (1.0 + 0.35 * formation_deformation)
            effective_center_gain = float(params.center_gain) * (1.0 - 0.35 * formation_deformation)
            actions = np.tile(cruise, (TEAM_SIZE, 1))
            actions[:, 0] += effective_slot_gain * slot_error[:, 0]
            actions[:, 1] += (effective_slot_gain + 3.00 * formation_deformation) * slot_error[:, 1]
            actions += effective_center_gain * (team_center - positions.mean(axis=0))
        else:
            actions = np.tile(cruise, (TEAM_SIZE, 1))
            slot_error = desired_slots - positions
            actions[:, 0] += (0.35 * (1.0 - 0.45 * formation_deformation)) * slot_error[:, 0]
            actions[:, 1] += (0.35 + 2.10 * formation_deformation) * slot_error[:, 1]
        actions, activations = _shield_adjustment(
            positions=positions,
            posts=posts,
            actions=actions,
            params=params,
            toggles=toggles,
        )
        shield_count += int(activations)

        speed_limit = float(max(stage_speed_cap, 0.8))
        norms = np.linalg.norm(actions, axis=1)
        too_fast = norms > speed_limit
        if np.any(too_fast):
            actions[too_fast] *= (speed_limit / np.maximum(norms[too_fast], 1e-6))[:, None]
        accel_limit = float(max(stage.drone_accel_limit_mps2, 1.0e-6))
        dv = actions - velocities
        dv_norms = np.linalg.norm(dv, axis=1)
        too_much = dv_norms > accel_limit * DT_S
        if np.any(too_much):
            dv[too_much] *= (accel_limit * DT_S / np.maximum(dv_norms[too_much], 1e-6))[:, None]
        velocities = velocities + dv
        speed_norms = np.linalg.norm(velocities, axis=1)
        mean_speed_sum += float(np.mean(speed_norms))
        max_speed_seen = max(max_speed_seen, float(np.max(speed_norms)))
        prev_positions = positions.copy()
        positions = positions + velocities * DT_S

        # Hard collision checks use the live map after movement.  A simple
        # swept check samples the segment midpoint to catch fast gate contacts.
        next_posts = next_posts_for_velocity
        mid_positions = 0.5 * (prev_positions + positions)
        live_posts = np.vstack([posts, next_posts]) if len(posts) and len(next_posts) else posts
        swept_clear = swept_post_clearance(prev_positions, positions, posts, next_posts, config=DGD_CONFIG)
        obs_clear = min(
            _obstacle_clearance(positions, live_posts),
            _obstacle_clearance(mid_positions, live_posts),
            swept_clear,
        )
        pair_dist = _pair_min_distance(positions)
        desired_final_slots = np.asarray([GOAL_X_M, GOAL_Y_M], dtype=np.float32) + slots
        errors = np.linalg.norm(desired_final_slots - positions, axis=1)
        mean_slot_error = float(np.mean(np.linalg.norm(tracking_slots - positions, axis=1)))
        max_slot_error = float(np.max(np.linalg.norm(tracking_slots - positions, axis=1)))

        min_pair = min(min_pair, pair_dist)
        min_clearance = min(min_clearance, obs_clear)
        slot_error_sum += mean_slot_error
        slot_error_max = max(slot_error_max, max_slot_error)
        samples += 1

        out_of_bounds = bool(
            np.any(positions[:, 0] < WORLD_X_BOUNDS_M[0])
            or np.any(positions[:, 0] > WORLD_X_BOUNDS_M[1])
            or np.any(positions[:, 1] < WORLD_Y_BOUNDS_M[0])
            or np.any(positions[:, 1] > WORLD_Y_BOUNDS_M[1])
        )
        agent_collision = pair_dist <= 2.0 * AGENT_AGENT_COLLISION_RADIUS_M
        obstacle_collision = obs_clear <= 0.0
        dispersed = max_slot_error > (5.5 + 3.0 * formation_deformation)
        goal_reached = (
            float(np.linalg.norm(team_center - np.asarray([GOAL_X_M, GOAL_Y_M], dtype=np.float32))) <= GOAL_RADIUS_M
            and float(np.max(errors)) <= SLOT_SUCCESS_TOLERANCE_M
        )
        if save_trace_path is not None and step % 2 == 0:
            trace_rows.append(
                {
                    "step": step,
                    "t_sec": round(t_sec, 3),
                    "team_center_x_m": float(team_center[0]),
                    "team_center_y_m": float(team_center[1]),
                    "min_pair_distance_m": pair_dist,
                    "min_obstacle_clearance_m": obs_clear,
                    "mean_slot_error_m": mean_slot_error,
                    "formation_deformation": float(formation_deformation),
                    "gate_centers_xy": centers.tolist(),
                    "agent_positions_xy": positions.tolist(),
                }
            )
        if obstacle_collision:
            done_reason = "obstacle_collision"
            break
        if agent_collision:
            done_reason = "agent_collision"
            break
        if out_of_bounds:
            done_reason = "out_of_bounds"
            break
        if dispersed:
            done_reason = "dispersed"
            break
        if goal_reached:
            done_reason = "goal_reached"
            break
    else:
        step = MAX_STEPS - 1

    if save_trace_path is not None:
        save_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with save_trace_path.open("w", encoding="utf-8") as stream:
            for row in trace_rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    actual_gate_motion = 0.0
    if len(gates):
        ranges = np.linalg.norm(gate_motion_max - gate_motion_min, axis=1)
        actual_gate_motion = float(np.max(ranges))
    final_slots = np.asarray([GOAL_X_M, GOAL_Y_M], dtype=np.float32) + slots
    per_agent_success = float(np.mean(np.linalg.norm(final_slots - positions, axis=1) <= SLOT_SUCCESS_TOLERANCE_M))
    success = done_reason == "goal_reached"
    return EpisodeMetrics(
        success=bool(success),
        done_reason=done_reason,
        steps=int(step + 1),
        flight_time_s=float((step + 1) * DT_S),
        team_success_rate=1.0 if success else 0.0,
        per_agent_success_rate=per_agent_success,
        agent_agent_collision=done_reason == "agent_collision",
        obstacle_collision=done_reason == "obstacle_collision",
        out_of_bounds=done_reason == "out_of_bounds",
        timeout=done_reason == "timeout",
        min_pair_distance_m=float(min_pair),
        min_obstacle_clearance_m=float(min_clearance),
        mean_slot_error_m=float(slot_error_sum / max(samples, 1)),
        max_slot_error_m=float(slot_error_max),
        progress_distance_m=float(np.mean(positions[:, 0]) - START_X_M),
        dispersed_termination=done_reason == "dispersed",
        shield_activation_count=int(shield_count),
        route_guidance_used_count=int(guidance_used),
        actual_gate_motion_range_m=float(actual_gate_motion),
        mean_speed_mps=float(mean_speed_sum / max(samples, 1)),
        max_speed_mps=float(max_speed_seen),
    )


def summarize_episodes(episodes: list[EpisodeMetrics]) -> dict[str, object]:
    count = max(len(episodes), 1)
    reasons: dict[str, int] = {}
    for item in episodes:
        reasons[item.done_reason] = reasons.get(item.done_reason, 0) + 1

    def mean(field: str) -> float:
        return float(np.mean([float(getattr(item, field)) for item in episodes])) if episodes else 0.0

    def min_field(field: str) -> float:
        return float(np.min([float(getattr(item, field)) for item in episodes])) if episodes else 0.0

    def max_field(field: str) -> float:
        return float(np.max([float(getattr(item, field)) for item in episodes])) if episodes else 0.0

    return {
        "episodes": len(episodes),
        "team_success_rate": mean("team_success_rate"),
        "per_agent_success_rate": mean("per_agent_success_rate"),
        "agent_agent_collision_rate": sum(1 for item in episodes if item.agent_agent_collision) / count,
        "obstacle_collision_rate": sum(1 for item in episodes if item.obstacle_collision) / count,
        "out_of_bounds_rate": sum(1 for item in episodes if item.out_of_bounds) / count,
        "timeout_rate": sum(1 for item in episodes if item.timeout) / count,
        "dispersed_termination_rate": sum(1 for item in episodes if item.dispersed_termination) / count,
        "min_pair_distance_m": min_field("min_pair_distance_m"),
        "mean_min_pair_distance_m": mean("min_pair_distance_m"),
        "min_obstacle_clearance_m": min_field("min_obstacle_clearance_m"),
        "mean_obstacle_clearance_m": mean("min_obstacle_clearance_m"),
        "formation_slot_error_mean_m": mean("mean_slot_error_m"),
        "formation_slot_error_max_m": max_field("max_slot_error_m"),
        "progress_distance_mean_m": mean("progress_distance_m"),
        "flight_time_mean_s": mean("flight_time_s"),
        "mean_speed_mps": mean("mean_speed_mps"),
        "max_speed_mps": max_field("max_speed_mps"),
        "shield_activation_count_mean": mean("shield_activation_count"),
        "route_guidance_used_count_mean": mean("route_guidance_used_count"),
        "actual_gate_motion_range_m": max_field("actual_gate_motion_range_m"),
        "done_reason_counts": reasons,
    }


def evaluate_stage(
    stage: CurriculumStage,
    params: ControllerParams,
    *,
    toggles: ModuleToggles,
    seeds: Iterable[int],
    episodes_per_seed: int,
    trace_dir: Path | None = None,
) -> dict[str, object]:
    metrics: list[EpisodeMetrics] = []
    for seed in seeds:
        for episode_idx in range(episodes_per_seed):
            trace_path = None
            if trace_dir is not None and seed == list(seeds)[0] and episode_idx == 0:
                trace_path = trace_dir / f"{stage.name}_seed{seed}_episode{episode_idx:03d}.jsonl"
            metrics.append(
                run_episode(
                    stage=stage,
                    params=params,
                    seed=int(seed),
                    episode_index=int(episode_idx),
                    toggles=toggles,
                    save_trace_path=trace_path,
                )
            )
    summary = summarize_episodes(metrics)
    summary.update(
        {
            "stage_name": stage.name,
            "gate_count": int(stage.gate_count),
            "moving_gate_speed_mps": float(stage.moving_gate_speed_mps),
            "moving_gate_amplitude_m": float(stage.moving_gate_amplitude_m),
            "drone_speed_mps": float(stage.drone_base_speed_mps),
            "drone_accel_limit_mps2": float(stage.drone_accel_limit_mps2),
            "params": asdict(params),
            "toggles": asdict(toggles),
        }
    )
    return summary


def _score(summary: dict[str, object]) -> float:
    return (
        100.0 * float(summary["team_success_rate"])
        + 18.0 * float(summary["per_agent_success_rate"])
        + 0.8 * float(summary["progress_distance_mean_m"])
        - 55.0 * float(summary["obstacle_collision_rate"])
        - 35.0 * float(summary["agent_agent_collision_rate"])
        - 18.0 * float(summary["out_of_bounds_rate"])
        - 3.0 * max(0.0, float(summary["formation_slot_error_mean_m"]) - 1.2)
        + 1.0 * min(float(summary["min_obstacle_clearance_m"]), 2.0)
    )


def _mutate(params: ControllerParams, rng: np.random.Generator, scale: float) -> ControllerParams:
    data = asdict(params)
    for key in data:
        value = float(data[key])
        jitter = float(rng.normal(0.0, scale))
        if key == "formation_scale":
            data[key] = value + 0.10 * jitter
        elif key in {"base_speed_mps"}:
            data[key] = value + 0.22 * jitter
        else:
            data[key] = value + 0.18 * jitter
    return ControllerParams(**data).clipped()


def train_curriculum(
    stages: list[CurriculumStage],
    *,
    output_dir: Path,
    candidates_per_stage: int,
    seed: int,
    initial_params: ControllerParams | None = None,
    stage_index_offset: int = 0,
) -> tuple[ControllerParams, list[dict[str, object]]]:
    rng = np.random.default_rng(int(seed))
    params = initial_params or ControllerParams()
    records: list[dict[str, object]] = []
    full = ModuleToggles()
    for stage_idx, stage in enumerate(stages):
        absolute_stage_idx = int(stage_index_offset) + int(stage_idx)
        candidates = [params]
        mutation_scale = max(0.22, 0.8 / max(absolute_stage_idx + 1, 1))
        candidates.extend(_mutate(params, rng, scale=mutation_scale) for _ in range(candidates_per_stage - 1))
        best_summary = None
        best_params = params
        best_score = -float("inf")
        candidate_summaries = []
        for cand_idx, candidate in enumerate(candidates):
            summary = evaluate_stage(
                stage,
                candidate,
                toggles=full,
                seeds=stage.train_seeds,
                episodes_per_seed=stage.train_episodes_per_seed,
            )
            score = _score(summary)
            summary["candidate_index"] = cand_idx
            summary["selection_score"] = score
            candidate_summaries.append(summary)
            if score > best_score:
                best_score = score
                best_params = candidate
                best_summary = summary
        params = best_params
        eval_summary = evaluate_stage(
            stage,
            params,
            toggles=full,
            seeds=stage.eval_seeds,
            episodes_per_seed=stage.eval_episodes_per_seed,
            trace_dir=output_dir / "traces",
        )
        record = {
            "stage": asdict(stage),
            "selected_params": asdict(params),
            "train_best_summary": best_summary,
            "eval_summary": eval_summary,
            "candidate_summaries": candidate_summaries,
        }
        records.append(record)
        _write_json(output_dir / "stages" / f"{absolute_stage_idx:02d}_{stage.name}.json", record)
    return params, records


def _load_controller_params(path: Path) -> ControllerParams:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "selected_params" in payload:
        payload = payload["selected_params"]
    return ControllerParams(**{key: payload[key] for key in asdict(ControllerParams()).keys()}).clipped()


def build_curriculum() -> list[CurriculumStage]:
    stages = [
        CurriculumStage("C0_gate0_empty_baseline", 0, 0.0, 0.0, (0, 1, 2), (0, 1, 2, 3), 2, 6),
        CurriculumStage("C1_gate6_static_tight", 6, 0.0, 0.0, (0, 1, 2), (0, 1, 2, 3), 2, 5),
        CurriculumStage("C2_gate12_speed05", 12, 0.5, 0.60, (0, 1, 2, 3), (0, 1, 2, 3, 4), 2, 5),
        CurriculumStage("C3_gate18_speed08", 18, 0.8, 0.75, (0, 1, 2, 3), (0, 1, 2, 3, 4), 2, 5),
        CurriculumStage("C4_gate24_speed10", 24, 1.0, 0.85, (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5), 2, 4),
        CurriculumStage("C5_gate30_speed11", 30, 1.1, 0.90, (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5), 2, 4),
        CurriculumStage("C6_gate36_speed12", 36, 1.2, 0.95, (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5), 2, 4),
        CurriculumStage("C7_gate42_speed14", 42, 1.4, 1.00, (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5), 2, 4),
        CurriculumStage("C8_gate48_speed16", 48, 1.6, 1.05, (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5), 2, 4),
        CurriculumStage("C9_gate54_speed18", 54, 1.8, 1.10, (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5), 2, 4),
        CurriculumStage("C10_gate60_speed20", 60, 2.0, 1.20, (0, 1, 2, 3, 4), (0, 1, 2, 3, 4, 5), 2, 4),
    ]
    out: list[CurriculumStage] = []
    for idx, stage in enumerate(stages):
        speed_stage = speed_gradient_for_stage(idx)
        out.append(
            replace(
                stage,
                drone_base_speed_mps=float(speed_stage.max_command_speed_mps),
                drone_accel_limit_mps2=float(speed_stage.max_accel_mps2),
            )
        )
    return out


def run_e2d2(params: ControllerParams, output_dir: Path) -> list[dict[str, object]]:
    stages = [
        CurriculumStage("E2D2_gate0_speed0", 0, 0.0, 0.0, (), tuple(range(10)), 1, 5),
        CurriculumStage("E2D2_gate6_speed00", 6, 0.0, 0.0, (), tuple(range(10)), 1, 5),
        CurriculumStage("E2D2_gate12_speed05", 12, 0.5, 0.60, (), tuple(range(10)), 1, 5),
        CurriculumStage("E2D2_gate18_speed08", 18, 0.8, 0.75, (), tuple(range(10)), 1, 5),
        CurriculumStage("E2D2_gate24_speed10", 24, 1.0, 0.85, (), tuple(range(10)), 1, 5),
        CurriculumStage("E2D2_gate30_speed11", 30, 1.1, 0.90, (), tuple(range(10)), 1, 5),
        CurriculumStage("E2D2_gate36_speed12", 36, 1.2, 0.95, (), tuple(range(10)), 1, 5),
        CurriculumStage("E2D2_gate42_speed14", 42, 1.4, 1.00, (), tuple(range(10)), 1, 5),
        CurriculumStage("E2D2_gate48_speed16", 48, 1.6, 1.05, (), tuple(range(10)), 1, 5),
        CurriculumStage("E2D2_gate54_speed18", 54, 1.8, 1.10, (), tuple(range(10)), 1, 5),
        CurriculumStage("E2D2_gate60_speed20", 60, 2.0, 1.20, (), tuple(range(10)), 1, 5),
    ]
    rows = []
    for stage in stages:
        rows.append(
            evaluate_stage(
                stage,
                params,
                toggles=ModuleToggles(),
                seeds=stage.eval_seeds,
                episodes_per_seed=stage.eval_episodes_per_seed,
                trace_dir=output_dir / "traces",
            )
        )
    _write_json(output_dir / "e2d2_dynamic_gate_density_risk.json", rows)
    return rows


def run_e2d3(params: ControllerParams, output_dir: Path) -> list[dict[str, object]]:
    stress_stages = [
        ("main", CurriculumStage("E2D3_main_gate36_speed12", 36, 1.2, 0.9, (), tuple(range(12)), 1, 4)),
        ("pressure", CurriculumStage("E2D3_pressure_gate60_speed20", 60, 2.0, 1.15, (), tuple(range(12)), 1, 4)),
    ]
    ablations = {
        "Full": ModuleToggles(True, True, True, True, False),
        "No_guidance": ModuleToggles(True, True, True, False, False),
        "Guidance_shadow": ModuleToggles(True, True, True, False, True),
        "No_shield": ModuleToggles(True, True, False, True, False),
        "No_slow": ModuleToggles(True, False, True, True, False),
        "Fast_only": ModuleToggles(True, False, False, False, False),
        "Planner_only": ModuleToggles(False, True, True, False, False),
    }
    rows = []
    for table_name, stress_stage in stress_stages:
        for name, toggles in ablations.items():
            summary = evaluate_stage(
                stress_stage,
                params,
                toggles=toggles,
                seeds=stress_stage.eval_seeds,
                episodes_per_seed=stress_stage.eval_episodes_per_seed,
                trace_dir=output_dir / "traces" if name == "Full" else None,
            )
            summary["ablation"] = name
            summary["stress_table"] = table_name
            rows.append(summary)
    _write_json(output_dir / "e2d3_slow_fast_safe_ablation.json", rows)
    return rows


def run_drone_speed_sweep(params: ControllerParams, output_dir: Path) -> list[dict[str, object]]:
    """Evaluate the same controller across commanded drone-speed targets.

    This sweep is separate from moving-gate speed.  The gate scenarios stay
    fixed while the drone command-speed cap is increased, so the resulting
    curve can show where success, collision, timeout, and flight time trade off.
    """

    scenarios = [
        ("static_30", CurriculumStage("E9_static30", 30, 0.0, 0.0, (), tuple(range(8)), 1, 4)),
        ("dynamic_42_speed14", CurriculumStage("E9_dynamic42_speed14", 42, 1.4, 1.00, (), tuple(range(8)), 1, 4)),
        ("dynamic_60_speed20", CurriculumStage("E9_dynamic60_speed20", 60, 2.0, 1.20, (), tuple(range(8)), 1, 4)),
    ]
    rows: list[dict[str, object]] = []
    for scenario_name, base_stage in scenarios:
        for drone_speed in eval_drone_speed_axis_mps():
            stage = replace(
                base_stage,
                name=f"{base_stage.name}_drone{int(round(float(drone_speed) * 100)):03d}",
                drone_base_speed_mps=float(drone_speed),
                drone_accel_limit_mps2=drone_accel_limit_for_speed_mps2(float(drone_speed)),
            )
            summary = evaluate_stage(
                stage,
                params,
                toggles=ModuleToggles(),
                seeds=stage.eval_seeds,
                episodes_per_seed=stage.eval_episodes_per_seed,
                trace_dir=output_dir / "traces" if scenario_name == "dynamic_42_speed14" else None,
            )
            summary["speed_sweep_scenario"] = scenario_name
            summary["drone_speed_mps"] = float(drone_speed)
            summary["drone_accel_limit_mps2"] = float(stage.drone_accel_limit_mps2)
            rows.append(summary)
    _write_json(output_dir / "e9_drone_speed_gradient_eval.json", rows)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage_name",
        "ablation",
        "stress_table",
        "speed_sweep_scenario",
        "gate_count",
        "moving_gate_speed_mps",
        "moving_gate_amplitude_m",
        "drone_speed_mps",
        "drone_accel_limit_mps2",
        "mean_speed_mps",
        "max_speed_mps",
        "team_success_rate",
        "per_agent_success_rate",
        "agent_agent_collision_rate",
        "obstacle_collision_rate",
        "out_of_bounds_rate",
        "timeout_rate",
        "dispersed_termination_rate",
        "min_pair_distance_m",
        "min_obstacle_clearance_m",
        "formation_slot_error_mean_m",
        "formation_slot_error_max_m",
        "progress_distance_mean_m",
        "actual_gate_motion_range_m",
        "shield_activation_count_mean",
        "route_guidance_used_count_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--candidates-per-stage", type=int, default=10)
    parser.add_argument("--stage-limit", type=int, default=None)
    parser.add_argument("--start-stage-index", type=int, default=0)
    parser.add_argument("--initial-params-json", type=Path, default=None)
    parser.add_argument("--skip-formal", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=Path(os.environ["GATE2D_RESUME_CHECKPOINT"]) if os.environ.get("GATE2D_RESUME_CHECKPOINT") else None,
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or (root / "results" / "e2d_dynamic_gate_density_8d_curriculum")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stages = build_curriculum()
    start_stage_index = max(int(args.start_stage_index), 0)
    stages = all_stages[start_stage_index:]
    if args.stage_limit is not None:
        stages = stages[: max(1, int(args.stage_limit))]
    initial_params = _load_controller_params(args.initial_params_json) if args.initial_params_json else None
    final_params, stage_records = train_curriculum(
        stages,
        output_dir=output_dir,
        candidates_per_stage=max(int(args.candidates_per_stage), 1),
        seed=int(args.seed),
        initial_params=initial_params,
        stage_index_offset=start_stage_index,
    )
    e2d2_rows = [] if args.skip_formal else run_e2d2(final_params, output_dir)
    e2d3_rows = [] if args.skip_formal else run_e2d3(final_params, output_dir)
    e9_rows = [] if args.skip_formal else run_drone_speed_sweep(final_params, output_dir)
    if e2d2_rows:
        _write_csv(output_dir / "e2d2_dynamic_gate_density_risk.csv", e2d2_rows)
    if e2d3_rows:
        _write_csv(output_dir / "e2d3_slow_fast_safe_ablation.csv", e2d3_rows)
    if e9_rows:
        _write_csv(output_dir / "e9_drone_speed_gradient_eval.csv", e9_rows)

    final_payload = {
        "run_name": output_dir.name,
        "output_dir": str(output_dir),
        "team_size": TEAM_SIZE,
        "scene_contract": {
            "fixed_height_training_line": True,
            "shared_dynamic_gate_module": "shared.core.dynamic_gate_density_2d",
            "world_x_bounds_m": list(WORLD_X_BOUNDS_M),
            "world_y_bounds_m": list(WORLD_Y_BOUNDS_M),
            "start_xy": [START_X_M, START_Y_M],
            "goal_xy": [GOAL_X_M, GOAL_Y_M],
            "max_gate_count": MAX_GATE_COUNT,
            "max_moving_gate_speed_mps": MAX_MOVING_GATE_SPEED_MPS,
            "gate_yaw_policy": "random_uniform_minus5_to_plus5_deg_formation_facing",
            "gate_half_width_m": float(GATE_HALF_WIDTH_M),
            "obstacle_collision_drone_shell_radius_m": float(DRONE_RADIUS_M),
            "agent_agent_collision_shell_radius_m": float(AGENT_AGENT_COLLISION_RADIUS_M),
            "collision_policy": "terminal_crash_on_gate_or_agent_contact",
            "active_avoidance_policy": "follow predicted live gate openings while maintaining forward progress",
            "training_drone_speed_gradient_mps": [
                float(stage["stage"]["drone_base_speed_mps"]) for stage in stage_records
            ],
            "evaluation_drone_speed_axis_mps": list(eval_drone_speed_axis_mps()),
        },
        "resume_checkpoint": {
            "requested_path": str(args.resume_checkpoint) if args.resume_checkpoint else None,
            "exists": bool(args.resume_checkpoint.exists()) if args.resume_checkpoint else False,
        },
        "continuation": {
            "start_stage_index": int(start_stage_index),
            "initial_params_json": str(args.initial_params_json) if args.initial_params_json else None,
            "initial_params_loaded": bool(initial_params is not None),
        },
        "final_params": asdict(final_params),
        "curriculum": stage_records,
        "e2d2_dynamic_gate_density_risk": e2d2_rows,
        "e2d3_slow_fast_safe_ablation": e2d3_rows,
        "e9_drone_speed_gradient_eval": e9_rows,
        "acceptance": {
            "c0_team_success_rate": (
                e2d2_rows[0]["team_success_rate"]
                if e2d2_rows
                else (
                    stage_records[0]["eval_summary"]["team_success_rate"]
                    if stage_records and stage_records[0]["stage"]["name"].startswith("C0")
                    else None
                )
            ),
            "c0_passed_100_percent": bool(
                (
                    e2d2_rows[0]["team_success_rate"]
                    if e2d2_rows
                    else (
                        stage_records[0]["eval_summary"]["team_success_rate"]
                        if stage_records and stage_records[0]["stage"]["name"].startswith("C0")
                        else 0.0
                    )
                )
                >= 1.0
            ),
            "e2d2_has_degradation": bool(
                bool(e2d2_rows)
                and (
                    float(e2d2_rows[0]["team_success_rate"]) > float(e2d2_rows[-1]["team_success_rate"])
                    or float(e2d2_rows[0]["formation_slot_error_mean_m"])
                    < float(e2d2_rows[-1]["formation_slot_error_mean_m"])
                    or float(e2d2_rows[0]["obstacle_collision_rate"])
                    < float(e2d2_rows[-1]["obstacle_collision_rate"])
                )
            ),
        },
    }
    _write_json(output_dir / "curriculum_summary.json", final_payload)
    print("dynamic gate-density 8-drone curriculum complete")
    print(f"summary_path={output_dir / 'curriculum_summary.json'}")
    print(f"c0_team_success_rate={final_payload['acceptance']['c0_team_success_rate']}")
    if e2d2_rows:
        print(f"e2d2_gate0_success={e2d2_rows[0]['team_success_rate']}")
        print(f"e2d2_gate60_success={e2d2_rows[-1]['team_success_rate']}")
    if e2d3_rows:
        e2d3_full = next(row for row in e2d3_rows if row["ablation"] == "Full" and row.get("stress_table") == "main")
        print(f"e2d3_main_full_success={e2d3_full['team_success_rate']}")


if __name__ == "__main__":
    main()




