"""Velocity-space safety corrections for the multi-agent gate environment."""

from __future__ import annotations

import math

import numpy as np


def _default_action_shield_info(self) -> dict[str, object]:
    return {
        "enabled": bool(getattr(self.env_config, "action_safety_shield_enabled", False)),
        "active": False,
        "mean_intervention_norm": 0.0,
        "max_pair_closeness": 0.0,
        "max_boundary_closeness": 0.0,
        "max_obstacle_closeness": 0.0,
        "max_guidance_closeness": 0.0,
        "max_gate_channel_closeness": 0.0,
        "max_post_gate_cruise_boost": 0.0,
    }


def _apply_action_safety_shield(
    self,
    action_xy: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    resolved_action = np.asarray(action_xy, dtype=np.float32).reshape(self._num_agents, 2)
    diagnostics = self._default_action_shield_info()
    if not bool(diagnostics["enabled"]) or resolved_action.size == 0:
        return resolved_action.copy(), diagnostics

    positions_xy = self._active_positions_xy()
    if positions_xy.shape[0] != self._num_agents:
        return resolved_action.copy(), diagnostics

    virtual_center_xy = self._virtual_center_xy()
    desired_slots_xy, dynamic_gate_task_ratio = self._actor_observation_desired_slots_xy(virtual_center_xy)
    dynamic_gate_task_threshold = float(getattr(self.env_config, "formation_line_collapse_task_ratio", 0.70) or 0.70)
    if dynamic_gate_task_ratio < dynamic_gate_task_threshold:
        desired_slots_xy = self._desired_slots[: self._num_agents].copy()
    heading_xy = np.asarray(self._current_guidance_heading(virtual_center_xy), dtype=np.float32)
    heading_norm = float(np.linalg.norm(heading_xy))
    if heading_norm <= 1.0e-6:
        heading_xy = np.asarray((1.0, 0.0), dtype=np.float32)
    else:
        heading_xy = heading_xy / heading_norm

    commanded_velocities_xy = np.asarray(
        [self.action_to_velocity_command(agent_action) for agent_action in resolved_action],
        dtype=np.float32,
    )
    commanded_velocities_xy, max_post_gate_cruise_boost = self._apply_post_gate_cruise_velocity_floor(
        positions_xy=positions_xy,
        commanded_velocities_xy=commanded_velocities_xy,
        heading_xy=heading_xy,
        virtual_center_xy=virtual_center_xy,
    )
    shielded_velocities_xy, max_pair_closeness = self._apply_pairwise_velocity_shield(
        positions_xy=positions_xy,
        desired_slots_xy=desired_slots_xy,
        virtual_center_xy=virtual_center_xy,
        heading_xy=heading_xy,
        commanded_velocities_xy=commanded_velocities_xy,
    )
    shielded_velocities_xy, max_boundary_closeness = self._apply_boundary_velocity_shield(
        positions_xy=positions_xy,
        commanded_velocities_xy=shielded_velocities_xy,
    )
    shielded_velocities_xy, max_guidance_closeness = self._apply_guidance_corridor_velocity_shield(
        virtual_center_xy=virtual_center_xy,
        commanded_velocities_xy=shielded_velocities_xy,
    )
    shielded_velocities_xy, max_gate_channel_closeness = self._apply_dynamic_gate_channel_velocity_shield(
        positions_xy=positions_xy,
        commanded_velocities_xy=shielded_velocities_xy,
        heading_xy=heading_xy,
    )
    shielded_velocities_xy, max_obstacle_closeness = self._apply_obstacle_velocity_shield(
        positions_xy=positions_xy,
        commanded_velocities_xy=shielded_velocities_xy,
    )

    shielded_action = np.asarray(
        [self.desired_velocity_to_action(velocity_xy) for velocity_xy in shielded_velocities_xy],
        dtype=np.float32,
    )
    intervention_norm = (
        float(np.linalg.norm(shielded_action - resolved_action, axis=1).mean()) if self._num_agents > 0 else 0.0
    )
    diagnostics.update(
        {
            "active": bool(
                intervention_norm > 1.0e-6
                or max_pair_closeness > 0.0
                or max_boundary_closeness > 0.0
                or max_obstacle_closeness > 0.0
                or max_guidance_closeness > 0.0
                or max_gate_channel_closeness > 0.0
                or max_post_gate_cruise_boost > 0.0
            ),
            "mean_intervention_norm": float(intervention_norm),
            "max_pair_closeness": float(max_pair_closeness),
            "max_boundary_closeness": float(max_boundary_closeness),
            "max_obstacle_closeness": float(max_obstacle_closeness),
            "max_guidance_closeness": float(max_guidance_closeness),
            "max_gate_channel_closeness": float(max_gate_channel_closeness),
            "max_post_gate_cruise_boost": float(max_post_gate_cruise_boost),
        }
    )
    return shielded_action, diagnostics


def _apply_post_gate_cruise_velocity_floor(
    self,
    *,
    positions_xy: np.ndarray,
    commanded_velocities_xy: np.ndarray,
    heading_xy: np.ndarray,
    virtual_center_xy: tuple[float, float],
) -> tuple[np.ndarray, float]:
    if (
        not self._dynamic_gate_enabled
        or not self._dynamic_gates
        or not bool(getattr(self.env_config, "action_safety_shield_post_gate_cruise_enabled", False))
        or positions_xy.size == 0
        or commanded_velocities_xy.size == 0
    ):
        return commanded_velocities_xy.copy(), 0.0

    min_forward_mps = max(
        float(getattr(self.env_config, "action_safety_shield_post_gate_cruise_min_forward_mps", 0.0) or 0.0),
        0.0,
    )
    if min_forward_mps <= 0.0:
        return commanded_velocities_xy.copy(), 0.0

    heading = np.asarray(heading_xy, dtype=np.float32).reshape(2)
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= 1.0e-6:
        return commanded_velocities_xy.copy(), 0.0
    heading = heading / heading_norm

    gate_behind_m = max(
        float(getattr(self.env_config, "action_safety_shield_post_gate_cruise_gate_behind_m", 0.0) or 0.0),
        0.0,
    )
    gate_posts_xy = self._dynamic_gate_posts_xy(next_frame=False)
    if gate_posts_xy.size == 0:
        return commanded_velocities_xy.copy(), 0.0
    relative_gate_to_agents = gate_posts_xy[:, None, :] - positions_xy[None, :, :]
    max_gate_forward_m = float(np.max(relative_gate_to_agents @ heading))
    if max_gate_forward_m > -gate_behind_m:
        return commanded_velocities_xy.copy(), 0.0

    goal_margin_m = max(
        float(getattr(self.env_config, "action_safety_shield_post_gate_cruise_goal_margin_m", 0.0) or 0.0),
        0.0,
    )
    if self._goal_distance(virtual_center_xy) <= float(self.env_config.goal_radius_m) + goal_margin_m:
        return commanded_velocities_xy.copy(), 0.0

    min_pair_required_m = max(
        float(getattr(self.env_config, "action_safety_shield_post_gate_cruise_min_pair_distance_m", 0.0) or 0.0),
        0.0,
    )
    pair_collision, min_pair_distance_m = self._pairwise_collision_stats(positions_xy)
    if pair_collision or min_pair_distance_m < min_pair_required_m:
        return commanded_velocities_xy.copy(), 0.0

    min_clearance_required_m = max(
        float(getattr(self.env_config, "action_safety_shield_post_gate_cruise_min_clearance_m", 0.0) or 0.0),
        0.0,
    )
    if self._min_clearance(positions_xy) < min_clearance_required_m:
        return commanded_velocities_xy.copy(), 0.0

    forward_limit_mps = max(self._resolved_forward_command_speed_mps(), 1.0e-6)
    floor_mps = min(min_forward_mps, forward_limit_mps)
    shielded = commanded_velocities_xy.copy()
    forward_components = shielded @ heading
    boost_mps = np.maximum(floor_mps - forward_components, 0.0)
    if not np.any(boost_mps > 1.0e-6):
        return shielded, 0.0
    shielded += heading.reshape(1, 2) * boost_mps.reshape(-1, 1)
    return shielded, float(np.max(boost_mps) / forward_limit_mps)


def _apply_pairwise_velocity_shield(
    self,
    *,
    positions_xy: np.ndarray,
    desired_slots_xy: np.ndarray,
    virtual_center_xy: tuple[float, float],
    heading_xy: np.ndarray,
    commanded_velocities_xy: np.ndarray,
) -> tuple[np.ndarray, float]:
    safe_distance_m = float(self.env_config.inter_agent_safe_distance_m)
    activation_margin_m = max(
        float(getattr(self.env_config, "action_safety_shield_separation_margin_m", 0.0) or 0.0),
        0.0,
    )
    activation_distance_m = safe_distance_m + activation_margin_m
    if positions_xy.shape[0] <= 1 or activation_distance_m <= 0.0:
        return commanded_velocities_xy.copy(), 0.0

    brake_scale = max(float(getattr(self.env_config, "action_safety_shield_brake_scale", 0.0) or 0.0), 0.0)
    repulsion_scale = max(
        float(getattr(self.env_config, "action_safety_shield_repulsion_scale", 0.0) or 0.0),
        0.0,
    )
    outward_bias_scale = max(
        float(getattr(self.env_config, "action_safety_shield_outward_slot_bias_scale", 0.0) or 0.0),
        0.0,
    )
    pair_closing_brake_only = bool(getattr(self.env_config, "action_safety_shield_pair_closing_brake_only", False))
    pair_time_horizon_s = max(
        float(getattr(self.env_config, "action_safety_shield_pair_time_horizon_s", 0.0) or 0.0),
        0.0,
    )
    priority_team_size_limit = max(
        int(getattr(self.env_config, "action_safety_shield_priority_team_size_limit", 3) or 3),
        1,
    )
    lateral_normal_xy = np.asarray((-float(heading_xy[1]), float(heading_xy[0])), dtype=np.float32)

    shielded = commanded_velocities_xy.copy()
    max_pair_closeness = 0.0
    activation_span_m = max(activation_margin_m, 1.0e-6)
    virtual_center = np.asarray(virtual_center_xy, dtype=np.float32)
    desired_lateral_offsets = np.asarray(
        [np.dot(desired_slots_xy[idx] - virtual_center, lateral_normal_xy) for idx in range(self._num_agents)],
        dtype=np.float32,
    )
    for idx in range(self._num_agents):
        for jdx in range(idx + 1, self._num_agents):
            delta_xy = positions_xy[idx] - positions_xy[jdx]
            distance_m = float(np.linalg.norm(delta_xy))
            relative_velocity_xy = shielded[idx] - shielded[jdx]
            predicted_delta_xy = delta_xy + relative_velocity_xy * max(float(self.env_config.dt_s), pair_time_horizon_s)
            predicted_distance_m = float(np.linalg.norm(predicted_delta_xy))
            risk_distance_m = min(distance_m, predicted_distance_m)
            if risk_distance_m >= activation_distance_m:
                continue
            closeness = float(np.clip((activation_distance_m - risk_distance_m) / activation_span_m, 0.0, 1.0))
            if risk_distance_m <= safe_distance_m:
                closeness = 1.0
            if closeness <= 0.0:
                continue
            max_pair_closeness = max(max_pair_closeness, closeness)

            if distance_m <= 1.0e-6:
                pair_axis_xy = lateral_normal_xy
            else:
                pair_axis_xy = np.asarray(delta_xy / distance_m, dtype=np.float32)
            if self._dynamic_gate_enabled and self._dynamic_gates:
                longitudinal_delta = float(np.dot(delta_xy, heading_xy))
                if abs(longitudinal_delta) <= 1.0e-5:
                    desired_delta = desired_slots_xy[idx] - desired_slots_xy[jdx]
                    longitudinal_delta = float(np.dot(desired_delta, heading_xy))
                longitudinal_sign = 1.0 if longitudinal_delta >= 0.0 else -1.0
                longitudinal_axis_xy = np.asarray(heading_xy * longitudinal_sign, dtype=np.float32)
                blended_axis = 0.55 * pair_axis_xy + 0.45 * longitudinal_axis_xy
                blended_norm = float(np.linalg.norm(blended_axis))
                if blended_norm > 1.0e-6:
                    pair_axis_xy = np.asarray(blended_axis / blended_norm, dtype=np.float32)
            if repulsion_scale > 0.0:
                repulsion_velocity_xy = pair_axis_xy * float(repulsion_scale * closeness)
                shielded[idx] += repulsion_velocity_xy
                shielded[jdx] -= repulsion_velocity_xy

            if brake_scale > 0.0:
                brake_ratio = min(float(brake_scale * closeness), 1.0)
                if pair_closing_brake_only:
                    relative_speed_mps = float(np.dot(shielded[idx] - shielded[jdx], pair_axis_xy))
                    closing_speed_mps = max(0.0, -relative_speed_mps)
                    closing_correction_xy = 0.5 * closing_speed_mps * brake_ratio * pair_axis_xy
                    shielded[idx] += closing_correction_xy
                    shielded[jdx] -= closing_correction_xy
                else:
                    forward_speed_idx = max(0.0, float(np.dot(shielded[idx], heading_xy)))
                    forward_speed_jdx = max(0.0, float(np.dot(shielded[jdx], heading_xy)))
                    shielded[idx] -= heading_xy * forward_speed_idx * brake_ratio
                    shielded[jdx] -= heading_xy * forward_speed_jdx * brake_ratio

            if outward_bias_scale > 0.0 and self._num_agents <= priority_team_size_limit:
                for agent_idx in (idx, jdx):
                    lateral_offset = float(desired_lateral_offsets[agent_idx])
                    if abs(lateral_offset) <= 1.0e-5:
                        continue
                    lateral_sign = 1.0 if lateral_offset > 0.0 else -1.0
                    shielded[agent_idx] += lateral_normal_xy * float(outward_bias_scale * closeness * lateral_sign)

    return shielded, float(max_pair_closeness)


def _apply_boundary_velocity_shield(
    self,
    *,
    positions_xy: np.ndarray,
    commanded_velocities_xy: np.ndarray,
) -> tuple[np.ndarray, float]:
    boundary_margin_m = max(
        float(getattr(self.env_config, "action_safety_shield_boundary_margin_m", 0.0) or 0.0),
        0.0,
    )
    brake_scale = max(
        float(getattr(self.env_config, "action_safety_shield_boundary_brake_scale", 0.0) or 0.0),
        0.0,
    )
    inward_scale = max(
        float(getattr(self.env_config, "action_safety_shield_boundary_inward_scale", 0.0) or 0.0),
        0.0,
    )
    if boundary_margin_m <= 0.0 or positions_xy.size == 0:
        return commanded_velocities_xy.copy(), 0.0

    shielded = commanded_velocities_xy.copy()
    x_min, x_max = self.env_config.world_x_bounds_m
    y_min, y_max = self.env_config.world_y_bounds_m
    max_boundary_closeness = 0.0
    for idx, position_xy in enumerate(positions_xy):
        lower_x_clearance = float(position_xy[0]) - float(x_min) - float(self.env_config.drone_radius_m)
        upper_x_clearance = float(x_max) - float(position_xy[0]) - float(self.env_config.drone_radius_m)
        lower_y_clearance = float(position_xy[1]) - float(y_min) - float(self.env_config.drone_radius_m)
        upper_y_clearance = float(y_max) - float(position_xy[1]) - float(self.env_config.drone_radius_m)

        if lower_x_clearance < boundary_margin_m:
            closeness = float(np.clip((boundary_margin_m - lower_x_clearance) / boundary_margin_m, 0.0, 1.0))
            max_boundary_closeness = max(max_boundary_closeness, closeness)
            if shielded[idx, 0] < 0.0:
                shielded[idx, 0] *= max(0.0, 1.0 - brake_scale * closeness)
            shielded[idx, 0] += inward_scale * closeness
        if upper_x_clearance < boundary_margin_m:
            closeness = float(np.clip((boundary_margin_m - upper_x_clearance) / boundary_margin_m, 0.0, 1.0))
            max_boundary_closeness = max(max_boundary_closeness, closeness)
            if shielded[idx, 0] > 0.0:
                shielded[idx, 0] *= max(0.0, 1.0 - brake_scale * closeness)
            shielded[idx, 0] -= inward_scale * closeness
        if lower_y_clearance < boundary_margin_m:
            closeness = float(np.clip((boundary_margin_m - lower_y_clearance) / boundary_margin_m, 0.0, 1.0))
            max_boundary_closeness = max(max_boundary_closeness, closeness)
            if shielded[idx, 1] < 0.0:
                shielded[idx, 1] *= max(0.0, 1.0 - brake_scale * closeness)
            shielded[idx, 1] += inward_scale * closeness
        if upper_y_clearance < boundary_margin_m:
            closeness = float(np.clip((boundary_margin_m - upper_y_clearance) / boundary_margin_m, 0.0, 1.0))
            max_boundary_closeness = max(max_boundary_closeness, closeness)
            if shielded[idx, 1] > 0.0:
                shielded[idx, 1] *= max(0.0, 1.0 - brake_scale * closeness)
            shielded[idx, 1] -= inward_scale * closeness
    return shielded, float(max_boundary_closeness)


def _apply_guidance_corridor_velocity_shield(
    self,
    *,
    virtual_center_xy: tuple[float, float],
    commanded_velocities_xy: np.ndarray,
) -> tuple[np.ndarray, float]:
    corridor_margin_m = max(
        float(getattr(self.env_config, "action_safety_shield_guidance_margin_m", 0.0) or 0.0),
        0.0,
    )
    inward_scale = max(
        float(getattr(self.env_config, "action_safety_shield_guidance_inward_scale", 0.0) or 0.0),
        0.0,
    )
    if corridor_margin_m <= 0.0 or inward_scale <= 0.0 or commanded_velocities_xy.size == 0:
        return commanded_velocities_xy.copy(), 0.0

    segment_start_xy, segment_end_xy = self._active_path_segment_xy()
    segment_dx = float(segment_end_xy[0] - segment_start_xy[0])
    segment_dy = float(segment_end_xy[1] - segment_start_xy[1])
    segment_norm = math.hypot(segment_dx, segment_dy)
    if segment_norm <= 1.0e-6:
        return commanded_velocities_xy.copy(), 0.0

    normal_xy = np.asarray((-segment_dy / segment_norm, segment_dx / segment_norm), dtype=np.float32)
    relative_xy = np.asarray(
        (
            float(virtual_center_xy[0] - segment_start_xy[0]),
            float(virtual_center_xy[1] - segment_start_xy[1]),
        ),
        dtype=np.float32,
    )
    signed_lateral_m = float(np.dot(relative_xy, normal_xy))
    lateral_abs_m = abs(signed_lateral_m)
    if lateral_abs_m <= corridor_margin_m:
        return commanded_velocities_xy.copy(), 0.0

    closeness = float(np.clip((lateral_abs_m - corridor_margin_m) / corridor_margin_m, 0.0, 1.0))
    correction_sign = -1.0 if signed_lateral_m > 0.0 else 1.0
    correction_velocity_xy = normal_xy * float(correction_sign * inward_scale * closeness)
    shielded = commanded_velocities_xy.copy()
    shielded += correction_velocity_xy.reshape(1, 2)
    return shielded, closeness


def _apply_dynamic_gate_channel_velocity_shield(
    self,
    *,
    positions_xy: np.ndarray,
    commanded_velocities_xy: np.ndarray,
    heading_xy: np.ndarray,
) -> tuple[np.ndarray, float]:
    if (
        not self._dynamic_gate_enabled
        or not self._dynamic_gates
        or not bool(getattr(self.env_config, "action_safety_shield_gate_channel_enabled", False))
        or positions_xy.size == 0
        or commanded_velocities_xy.size == 0
    ):
        return commanded_velocities_xy.copy(), 0.0

    lookahead_m = max(
        float(getattr(self.env_config, "action_safety_shield_gate_channel_lookahead_m", 0.0) or 0.0),
        0.0,
    )
    behind_m = max(
        float(getattr(self.env_config, "action_safety_shield_gate_channel_behind_m", 0.0) or 0.0),
        0.0,
    )
    lateral_gain = max(
        float(getattr(self.env_config, "action_safety_shield_gate_channel_lateral_gain", 0.0) or 0.0),
        0.0,
    )
    max_lateral_mps = max(
        float(getattr(self.env_config, "action_safety_shield_gate_channel_max_lateral_mps", 0.0) or 0.0),
        0.0,
    )
    slowdown_scale = max(
        float(getattr(self.env_config, "action_safety_shield_gate_channel_slowdown_scale", 0.0) or 0.0),
        0.0,
    )
    if lookahead_m <= 0.0 or (lateral_gain <= 0.0 and slowdown_scale <= 0.0):
        return commanded_velocities_xy.copy(), 0.0

    centers_xy = self._dynamic_gate_centers_xy(next_frame=False)
    velocities_xy = self._dynamic_gate_velocities_xy()
    if centers_xy.size == 0:
        return commanded_velocities_xy.copy(), 0.0

    heading = np.asarray(heading_xy, dtype=np.float32)
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= 1.0e-6:
        heading = np.asarray((1.0, 0.0), dtype=np.float32)
    else:
        heading = heading / heading_norm
    lateral_axis = np.asarray((-float(heading[1]), float(heading[0])), dtype=np.float32)
    team_center = np.asarray(np.mean(positions_xy, axis=0), dtype=np.float32)
    forward_components = np.maximum(commanded_velocities_xy @ heading, 0.0)
    base_speed_mps = max(float(np.mean(forward_components)), 0.35)

    weighted_lateral = 0.0
    weight_sum = 0.0
    max_closeness = 0.0
    for gate_idx, (gate, center_xy) in enumerate(zip(self._dynamic_gates, centers_xy, strict=False)):
        if int(getattr(gate, "lane_index", 0)) != 0:
            continue
        relative_xy = np.asarray(center_xy, dtype=np.float32) - team_center
        dx_m = float(np.dot(relative_xy, heading))
        if dx_m < -behind_m or dx_m > lookahead_m:
            continue
        velocity_xy = (
            np.asarray(velocities_xy[gate_idx], dtype=np.float32)
            if gate_idx < len(velocities_xy)
            else np.zeros((2,), dtype=np.float32)
        )
        time_to_gate_s = max(dx_m / base_speed_mps, 0.0)
        predicted_center_xy = np.asarray(center_xy, dtype=np.float32) + velocity_xy * float(time_to_gate_s)
        predicted_relative_xy = predicted_center_xy - team_center
        predicted_lateral_m = float(np.dot(predicted_relative_xy, lateral_axis))

        local = math.exp(-abs(dx_m) / 5.2)
        if -0.8 <= dx_m <= 5.5:
            local = max(local, 0.95)
        elif 5.5 < dx_m <= lookahead_m:
            local = max(local, 0.45)
        weight = float(local / (0.55 + max(dx_m, 0.0)))
        weighted_lateral += predicted_lateral_m * weight
        weight_sum += weight
        max_closeness = max(max_closeness, float(np.clip(local, 0.0, 1.0)))

    if weight_sum <= 1.0e-6 or max_closeness <= 1.0e-6:
        return commanded_velocities_xy.copy(), 0.0

    target_lateral_m = float(np.clip(weighted_lateral / weight_sum, -7.6, 7.6))
    lateral_command_mps = float(lateral_gain * max_closeness * target_lateral_m)
    if max_lateral_mps > 0.0:
        lateral_command_mps = float(np.clip(lateral_command_mps, -max_lateral_mps, max_lateral_mps))

    shielded = commanded_velocities_xy.copy()
    if abs(lateral_command_mps) > 1.0e-6:
        shielded += lateral_axis.reshape(1, 2) * lateral_command_mps
    gate_half_width_m = float(getattr(self._dynamic_gate_config, "gate_half_width_m", 2.4))
    gate_post_radius_m = float(getattr(self._dynamic_gate_config, "gate_post_radius_m", 0.32))
    clear_half_width_m = max(
        0.95,
        gate_half_width_m - gate_post_radius_m - float(self.env_config.drone_radius_m) - 0.22,
    )
    aperture_soft_half_m = max(0.92, min(clear_half_width_m - 0.18, 1.36))
    aperture_span_m = max(clear_half_width_m - aperture_soft_half_m, 1.0e-6)
    aperture_gain = max(0.72, lateral_gain * 1.35)
    gate_center_lateral_xy = team_center + lateral_axis * float(target_lateral_m)
    for agent_idx, position_xy in enumerate(np.asarray(positions_xy, dtype=np.float32)):
        agent_lateral_m = float(np.dot(position_xy - gate_center_lateral_xy, lateral_axis))
        lateral_excess_m = abs(agent_lateral_m) - aperture_soft_half_m
        if lateral_excess_m <= 0.0:
            continue
        edge_closeness = float(np.clip(lateral_excess_m / aperture_span_m, 0.0, 1.0))
        correction_sign = -1.0 if agent_lateral_m > 0.0 else 1.0
        inward_velocity_mps = float(correction_sign * aperture_gain * max_closeness * edge_closeness)
        if max_lateral_mps > 0.0:
            inward_velocity_mps = float(np.clip(inward_velocity_mps, -max_lateral_mps, max_lateral_mps))
        shielded[agent_idx] += lateral_axis * inward_velocity_mps
        forward_speed_mps = max(0.0, float(np.dot(shielded[agent_idx], heading)))
        edge_slowdown = float(np.clip(0.22 * max_closeness * edge_closeness, 0.0, 0.42))
        shielded[agent_idx] -= heading * float(forward_speed_mps * edge_slowdown)
    if slowdown_scale > 0.0:
        slowdown = float(np.clip(slowdown_scale * max_closeness, 0.0, 0.65))
        forward = shielded @ heading
        positive_forward = np.maximum(forward, 0.0)
        shielded -= heading.reshape(1, 2) * (positive_forward * slowdown).reshape(-1, 1)
    return shielded, float(max_closeness)


def _apply_obstacle_velocity_shield(
    self,
    *,
    positions_xy: np.ndarray,
    commanded_velocities_xy: np.ndarray,
) -> tuple[np.ndarray, float]:
    margin_m = max(
        float(getattr(self.env_config, "action_safety_shield_obstacle_margin_m", 0.0) or 0.0),
        0.0,
    )
    brake_scale = max(
        float(getattr(self.env_config, "action_safety_shield_obstacle_brake_scale", 0.0) or 0.0),
        0.0,
    )
    repulsion_scale = max(
        float(getattr(self.env_config, "action_safety_shield_obstacle_repulsion_scale", 0.0) or 0.0),
        0.0,
    )
    horizon_s = max(
        float(getattr(self.env_config, "action_safety_shield_obstacle_time_horizon_s", 0.0) or 0.0),
        0.0,
    )
    if (
        margin_m <= 0.0
        or (brake_scale <= 0.0 and repulsion_scale <= 0.0)
        or positions_xy.size == 0
        or commanded_velocities_xy.size == 0
    ):
        return commanded_velocities_xy.copy(), 0.0

    obstacles = tuple(self._active_obstacle_map().obstacles)
    if not obstacles:
        return commanded_velocities_xy.copy(), 0.0

    shielded = commanded_velocities_xy.copy()
    max_closeness = 0.0
    horizon = max(float(self.env_config.dt_s), horizon_s)
    drone_radius_m = float(self.env_config.drone_radius_m)
    for agent_idx, position_xy in enumerate(np.asarray(positions_xy, dtype=np.float32)):
        commanded_velocity_xy = np.asarray(shielded[agent_idx], dtype=np.float32)
        for obstacle in obstacles:
            obstacle_xy = np.asarray(obstacle.center_xy, dtype=np.float32)
            obstacle_velocity_xy = np.asarray(getattr(obstacle, "velocity_xy", (0.0, 0.0)), dtype=np.float32)
            combined_radius_m = drone_radius_m + float(obstacle.collision_radius_m)
            relative_xy = position_xy - obstacle_xy
            relative_velocity_xy = commanded_velocity_xy - obstacle_velocity_xy
            predicted_relative_xy = relative_xy + relative_velocity_xy * float(horizon)

            distance_now_m = float(np.linalg.norm(relative_xy))
            distance_predicted_m = float(np.linalg.norm(predicted_relative_xy))
            risk_distance_m = min(distance_now_m, distance_predicted_m)

            clearance_m = risk_distance_m - combined_radius_m
            if clearance_m >= margin_m:
                continue
            closeness = float(np.clip((margin_m - clearance_m) / max(margin_m, 1.0e-6), 0.0, 1.0))
            if closeness <= 0.0:
                continue

            axis_reference_xy = relative_xy if distance_now_m > 1.0e-6 else predicted_relative_xy
            axis_norm = float(np.linalg.norm(axis_reference_xy))
            if axis_norm <= 1.0e-6:
                velocity_norm = float(np.linalg.norm(relative_velocity_xy))
                if velocity_norm <= 1.0e-6:
                    away_axis_xy = np.asarray((0.0, 1.0), dtype=np.float32)
                else:
                    away_axis_xy = np.asarray(-relative_velocity_xy / velocity_norm, dtype=np.float32)
            else:
                away_axis_xy = np.asarray(axis_reference_xy / axis_norm, dtype=np.float32)
            if str(getattr(obstacle, "species", "")) == "dynamic_gate_post":
                heading_xy = np.asarray(self.current_heading_xy(), dtype=np.float32)
                heading_norm = float(np.linalg.norm(heading_xy))
                if heading_norm <= 1.0e-6:
                    heading_xy = np.asarray((1.0, 0.0), dtype=np.float32)
                else:
                    heading_xy = heading_xy / heading_norm
                lateral_axis_xy = np.asarray((-float(heading_xy[1]), float(heading_xy[0])), dtype=np.float32)
                lateral_component = float(np.dot(away_axis_xy, lateral_axis_xy))
                if abs(lateral_component) <= 1.0e-4:
                    continue
                away_axis_xy = np.asarray(lateral_axis_xy * lateral_component, dtype=np.float32)

            max_closeness = max(max_closeness, closeness)
            if repulsion_scale > 0.0:
                shielded[agent_idx] += away_axis_xy * float(repulsion_scale * closeness)
            if brake_scale > 0.0:
                closing_speed_mps = max(0.0, -float(np.dot(relative_velocity_xy, away_axis_xy)))
                if closing_speed_mps > 0.0:
                    shielded[agent_idx] += away_axis_xy * float(brake_scale * closing_speed_mps * closeness)
            commanded_velocity_xy = np.asarray(shielded[agent_idx], dtype=np.float32)

    return shielded, float(max_closeness)
