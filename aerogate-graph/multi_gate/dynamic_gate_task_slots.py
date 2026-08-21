"""Dynamic-gate slot targets shared by expert rollout and actor observations."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from shared.core.dynamic_gate_density_2d import live_gate_centers


def dynamic_gate_task_first_slots(
    *,
    desired_slots_xy: np.ndarray,
    positions_xy: np.ndarray | None = None,
    center_xy: np.ndarray | Sequence[float],
    gates: Sequence[Any],
    centers_xy: np.ndarray,
    gate_velocities_xy: np.ndarray | None,
    gate_cfg: Any,
    gate_time_s: float,
    feedforward_speed_mps: float,
    start_x_m: float,
    goal_x_m: float,
    drone_radius_m: float,
) -> tuple[np.ndarray | None, float]:
    """Return gate-pass slots when the team is close enough to a dynamic gate."""

    desired_slots = np.asarray(desired_slots_xy, dtype=np.float32)
    if desired_slots.ndim != 2 or desired_slots.shape[0] <= 1 or desired_slots.shape[1] != 2:
        return None, 0.0
    gate_list = list(gates or [])
    centers = np.asarray(centers_xy, dtype=np.float32)
    if not gate_list or centers.size == 0:
        return None, 0.0

    def _gate_local_ratio(dx_m: float, *, strong_behind_m: float = -6.0) -> float:
        local_ratio = float(np.exp(-abs(float(dx_m)) / 7.2))
        if strong_behind_m <= float(dx_m) <= 6.5:
            local_ratio = max(local_ratio, 0.95)
        elif 6.5 < float(dx_m) <= 16.0:
            local_ratio = max(local_ratio, 0.78)
        elif 16.0 < float(dx_m) <= 28.0:
            local_ratio = max(local_ratio, 0.50)
        if float(dx_m) < strong_behind_m:
            local_ratio *= 0.35
        return float(np.clip(local_ratio, 0.0, 1.0))

    center = np.asarray(center_xy, dtype=np.float32).reshape(2)
    center_x = float(center[0])
    gate_progress_reference_x = center_x
    candidates: list[tuple[float, int, np.ndarray]] = []
    for gate_idx, gate in enumerate(gate_list):
        if int(getattr(gate, "lane_index", 0)) != 0 or gate_idx >= len(centers):
            continue
        gate_center = np.asarray(centers[gate_idx], dtype=np.float32)
        dx_center = float(gate_center[0] - gate_progress_reference_x)
        if -7.0 <= dx_center <= 28.0:
            candidates.append((dx_center, gate_idx, gate_center))
    if not candidates:
        return None, 0.0

    weighted_x = 0.0
    weighted_y = 0.0
    weight_sum = 0.0
    task_ratio = 0.0
    arrival_speed_mps = max(float(feedforward_speed_mps) * 0.72, 0.50)
    gate_amplitude_m = float(getattr(gate_cfg, "moving_gate_amplitude_m", 0.0) or 0.0)
    gate_speed_mps = float(getattr(gate_cfg, "moving_gate_speed_mps", 0.0) or 0.0)
    velocities = np.zeros_like(centers, dtype=np.float32) if gate_velocities_xy is None else np.asarray(gate_velocities_xy, dtype=np.float32)
    for dx, gate_idx, gate_center in candidates:
        predicted = np.asarray(gate_center, dtype=np.float32)
        time_to_gate = max(float(dx) / arrival_speed_mps, 0.0)
        if gate_cfg is not None:
            predicted_centers = live_gate_centers(
                gate_list,
                t_sec=float(gate_time_s) + time_to_gate,
                amplitude_m=gate_amplitude_m,
                speed_mps=gate_speed_mps,
                config=gate_cfg,
            )
            if gate_idx < len(predicted_centers):
                predicted = np.asarray(predicted_centers[gate_idx], dtype=np.float32)
        elif gate_idx < len(velocities):
            predicted = predicted + velocities[gate_idx] * float(time_to_gate)

        local = _gate_local_ratio(float(dx), strong_behind_m=-5.0)
        weight = local / (1.15 + max(float(dx), 0.0) ** 1.25)
        weighted_x += float(predicted[0]) * weight
        weighted_y += float(predicted[1]) * weight
        weight_sum += weight
        task_ratio = max(task_ratio, float(np.clip(local, 0.0, 1.0)))
    if weight_sum <= 1.0e-6 or task_ratio <= 1.0e-6:
        return None, 0.0

    pass_center_x = float(np.clip(weighted_x / weight_sum, float(start_x_m), float(goal_x_m)))
    pass_center_y = float(np.clip(weighted_y / weight_sum, -7.6, 7.6))

    num_agents = int(desired_slots.shape[0])
    desired_center = np.mean(desired_slots[:num_agents], axis=0)
    local_offsets = desired_slots[:num_agents] - desired_center.reshape(1, 2)
    max_lateral = max(float(np.max(np.abs(local_offsets[:, 1]))), 1.0e-6)
    gate_half_width_m = float(getattr(gate_cfg, "gate_half_width_m", 2.4))
    gate_post_radius_m = float(getattr(gate_cfg, "gate_post_radius_m", 0.32))
    clear_half_width_m = max(0.95, gate_half_width_m - gate_post_radius_m - float(drone_radius_m) - 0.18)
    tight_corridor_half = max(1.05, min(1.42, clear_half_width_m))
    compressed_scale = min(1.0, tight_corridor_half / max_lateral)
    lateral_scale = (1.0 - task_ratio) + task_ratio * compressed_scale
    adaptive_offsets = local_offsets.astype(np.float32, copy=True)
    adaptive_offsets[:, 1] *= float(lateral_scale)

    used_gate_pass_slots = False
    if num_agents >= 3:
        assignment_offsets = local_offsets
        if positions_xy is not None:
            positions = np.asarray(positions_xy, dtype=np.float32)
            if positions.shape[0] >= num_agents and positions.shape[1] == 2:
                position_center = np.mean(positions[:num_agents], axis=0)
                assignment_offsets = positions[:num_agents] - position_center.reshape(1, 2)
        base_lane_size = num_agents // 3
        remainder = num_agents % 3
        if remainder == 0:
            lane_sizes = [base_lane_size, base_lane_size, base_lane_size]
        elif remainder == 1:
            lane_sizes = [base_lane_size, base_lane_size + 1, base_lane_size]
        else:
            lane_sizes = [base_lane_size + 1, base_lane_size, base_lane_size + 1]
        lane_y_span = float(min(1.08, max(0.74 * tight_corridor_half, 1.00)))
        lane_y = np.asarray((-lane_y_span, 0.0, lane_y_span), dtype=np.float32)
        gate_offsets = np.zeros_like(adaptive_offsets, dtype=np.float32)
        lateral_order = np.argsort(assignment_offsets[:, 1])
        lane_groups: list[np.ndarray] = []
        start = 0
        for lane_size in lane_sizes:
            stop = min(start + int(lane_size), num_agents)
            lane_groups.append(lateral_order[start:stop])
            start = stop
        for lane_idx, lane_group in enumerate(lane_groups):
            if len(lane_group) == 0:
                continue
            longitudinal_order = sorted(
                (int(agent_idx) for agent_idx in lane_group),
                key=lambda agent_idx: float(assignment_offsets[agent_idx, 0]),
            )
            lane_count = max(len(longitudinal_order), 1)
            lane_spacing = 2.42 if num_agents <= 9 else 1.70
            x_offsets = (
                (np.arange(lane_count, dtype=np.float32) - 0.5 * float(lane_count - 1))
                * float(lane_spacing)
            )
            x_offsets += float((-0.35, 0.35, -0.35)[lane_idx])
            x_offsets = np.clip(x_offsets, -3.05, 3.05)
            for slot_idx, agent_idx in enumerate(longitudinal_order):
                gate_offsets[agent_idx, 0] = x_offsets[slot_idx]
                gate_offsets[agent_idx, 1] = lane_y[lane_idx]
        adaptive_offsets[:num_agents] = gate_offsets[:num_agents]
        used_gate_pass_slots = True

    if not used_gate_pass_slots:
        row_sign = np.sign(adaptive_offsets[:, 0])
        lane_phase = adaptive_offsets[:, 1] / max(max_lateral * lateral_scale, 1.0e-6)
        lateral_order_for_stagger = np.argsort(adaptive_offsets[:, 1])
        stagger_phase = np.zeros((num_agents,), dtype=np.float32)
        if num_agents > 1:
            stagger_phase[lateral_order_for_stagger] = np.linspace(-1.0, 1.0, num_agents, dtype=np.float32)
        adaptive_offsets[:, 0] += task_ratio * (
            0.30 * row_sign
            + 0.50 * lane_phase
            + 0.35 * stagger_phase
        )
        adaptive_offsets[:, 0] = np.clip(adaptive_offsets[:, 0], -2.85, 2.85)

    route_anchor_x = float(np.clip(center[0], float(start_x_m), float(goal_x_m)))
    through_offset_m = float(0.45 + 0.45 * task_ratio)
    min_lookahead_m = float(1.4 if task_ratio >= 0.75 else 2.4)
    max_lookahead_m = float(3.8 if task_ratio >= 0.75 else 5.5)
    target_x = float(
        np.clip(
            pass_center_x + through_offset_m,
            route_anchor_x + min_lookahead_m,
            route_anchor_x + max_lookahead_m,
        )
    )
    adaptive_center = np.asarray([min(float(goal_x_m), target_x), pass_center_y], dtype=np.float32)
    gate_slots = desired_slots.copy()
    gate_slots[:num_agents] = adaptive_center.reshape(1, 2) + adaptive_offsets
    blended = desired_slots.copy()
    blended[:num_agents] = (
        (1.0 - task_ratio) * desired_slots[:num_agents]
        + task_ratio * gate_slots[:num_agents]
    )
    return blended.astype(np.float32, copy=False), float(task_ratio)

