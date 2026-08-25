"""Per-step diagnostics, safety checks, and reward-facing environment state."""

from __future__ import annotations

import math

import numpy as np

from shared.core.dynamic_gate_density_2d import (
    center_has_completed_corridor,
    corridor_region_status,
    gate_gate_clearance_stats,
    post_clearance,
    swept_post_clearance,
    validate_height_and_corridor_invariants,
)
from shared.core.team_geometry import boundary_proximity_deficit_m, pairwise_separation_stats


def _build_info(
    self,
    *,
    done_reason: str | None,
    reward_terms: dict[str, float] | None,
) -> dict[str, object]:
    positions = self._active_positions_xy()
    virtual_center = self._virtual_center_xy()
    slot_anchor_xy = self._slot_anchor_center_xy(virtual_center)
    observation_desired_slots, dynamic_gate_task_ratio = self._actor_observation_desired_slots_xy(virtual_center)
    min_pair_distance = self._pairwise_collision_stats(positions)[1]
    current_goal_distance = self._goal_distance(virtual_center)
    initial_goal_distance = max(float(self._initial_goal_distance), 1.0e-6)
    goal_distance_improvement = max(0.0, initial_goal_distance - float(current_goal_distance))
    height_report = validate_height_and_corridor_invariants(config=self._dynamic_gate_config)
    corridor_status = corridor_region_status(positions, config=self._dynamic_gate_config)
    formation_shape_status = self._formation_shape_status(
        positions_xy=positions,
        center_xy=virtual_center,
        dynamic_gate_task_ratio=dynamic_gate_task_ratio,
        corridor_status=corridor_status,
    )
    mean_slot_error_m, max_slot_error_m = self._slot_error_stats(
        positions,
        self._desired_slots[: self._num_agents],
    )
    live_gate_centers_xy = self._dynamic_gate_centers_xy(next_frame=False)
    live_gate_posts_xy = self._dynamic_gate_posts_xy(next_frame=False)
    info = {
        "snapshot": self.snapshot(),
        "step_count": int(self._step_count),
        "num_agents": self._num_agents,
        "path_waypoints": self._plan.waypoints_xy,
        "path_index": self._path_index,
        "desired_slots": self._desired_slots[: self._num_agents].copy(),
        "actor_observation_desired_slots": observation_desired_slots.copy(),
        "actor_observation_dynamic_gate_task_ratio": float(dynamic_gate_task_ratio),
        "agent_positions_xy": positions.copy(),
        "agent_velocities_xy": self._active_velocities_xy().copy(),
        "lookahead_heading_xy": self._current_guidance_heading(virtual_center),
        "fixed_height_m": self.env_config.fixed_height_m,
        "gate_bottom_height_m": float(getattr(self._dynamic_gate_config, "gate_opening_bottom_height_m", 0.0)),
        "gate_top_height_m": float(getattr(self._dynamic_gate_config, "gate_opening_top_height_m", 0.0)),
        "gate_center_height_m": float(
            getattr(self._dynamic_gate_config, "gate_center_height_m", self.env_config.fixed_height_m)
        ),
        "height_contract_passed": bool(height_report.get("passed", False)),
        "height_escape_failure": bool(self._last_height_escape_failure),
        "side_bypass_failure": bool(self._last_side_bypass_failure),
        "corridor_miss_failure": bool(self._last_corridor_miss_failure),
        "corridor_region_status": dict(corridor_status),
        "corridor_completed": center_has_completed_corridor(virtual_center, config=self._dynamic_gate_config),
        "formation_shape_status": dict(formation_shape_status),
        "formation_shape_active": bool(formation_shape_status.get("active", False)),
        "formation_lateral_band_count": int(formation_shape_status.get("lateral_band_count", 0) or 0),
        "formation_required_lateral_bands": int(formation_shape_status.get("required_lateral_bands", 0) or 0),
        "formation_lateral_span_m": float(formation_shape_status.get("lateral_span_m", 0.0) or 0.0),
        "formation_line_collapse_score": float(formation_shape_status.get("formation_line_collapse_score", 0.0) or 0.0),
        "formation_line_collapse_failure": bool(
            self._last_formation_line_collapse_failure
            or formation_shape_status.get("formation_line_collapse_failure", False)
        ),
        "slot_anchor_xy": slot_anchor_xy,
    }
    info.update(
        _build_gate_diagnostics_info(
            self,
            positions=positions,
            live_gate_centers_xy=live_gate_centers_xy,
            live_gate_posts_xy=live_gate_posts_xy,
            min_pair_distance=min_pair_distance,
            mean_slot_error_m=mean_slot_error_m,
            max_slot_error_m=max_slot_error_m,
        )
    )
    info.update(
        _build_goal_progress_info(
            self,
            current_goal_distance=current_goal_distance,
            goal_distance_improvement=goal_distance_improvement,
            initial_goal_distance=initial_goal_distance,
        )
    )
    info.update(_build_guidance_info(self, virtual_center))
    info.update(
        _build_episode_info(self, done_reason=done_reason, reward_terms=reward_terms, virtual_center=virtual_center)
    )
    return info


def _build_gate_diagnostics_info(
    self,
    *,
    positions: np.ndarray,
    live_gate_centers_xy: np.ndarray,
    live_gate_posts_xy: np.ndarray,
    min_pair_distance: float,
    mean_slot_error_m: float,
    max_slot_error_m: float,
) -> dict[str, object]:
    """Build collision, clearance, and moving-gate fields for one info snapshot."""

    return {
        "min_clearance_m": self._min_clearance(positions),
        "dynamic_gate_enabled": bool(self._dynamic_gate_enabled),
        "dynamic_gate_collision": bool(self._last_dynamic_gate_collision),
        "dynamic_gate_count": int(len(self._dynamic_gates)),
        "moving_gate_speed_mps": float(getattr(self._dynamic_gate_config, "moving_gate_speed_mps", 0.0) or 0.0),
        "moving_gate_amplitude_m": float(getattr(self._dynamic_gate_config, "moving_gate_amplitude_m", 0.0) or 0.0),
        "actual_gate_motion_range_m": self._dynamic_gate_motion_range_m(),
        "live_gate_centers_xy": live_gate_centers_xy.copy(),
        "live_gate_post_positions_xy": live_gate_posts_xy.copy(),
        "dynamic_gate_gate_clearance": gate_gate_clearance_stats(
            self._dynamic_gates,
            live_gate_centers_xy,
            config=self._dynamic_gate_config,
        ),
        "min_pair_distance_m": min_pair_distance,
        "mean_slot_error_m": mean_slot_error_m,
        "max_slot_error_m": max_slot_error_m,
        "boundary_proximity_deficit_m": self._boundary_proximity_deficit_m(positions),
    }


def _build_goal_progress_info(
    self,
    *,
    current_goal_distance: float,
    goal_distance_improvement: float,
    initial_goal_distance: float,
) -> dict[str, object]:
    """Build goal and termination configuration fields for one info snapshot."""

    return {
        "initial_goal_distance_m": float(self._initial_goal_distance),
        "goal_xy": tuple(float(value) for value in self._goal_xy),
        "goal_radius_m": float(self.env_config.goal_radius_m),
        "goal_distance_m": float(current_goal_distance),
        "goal_distance_improvement_m": float(goal_distance_improvement),
        "goal_progress_ratio": float(np.clip(goal_distance_improvement / initial_goal_distance, 0.0, 1.0)),
        "goal_termination_enabled": bool(getattr(self.env_config, "goal_termination_enabled", True)),
        "preparation_hold_mode": bool(getattr(self.env_config, "preparation_hold_mode", False)),
        "timeout_counts_as_success": bool(getattr(self.env_config, "timeout_counts_as_success", False)),
        "goal_zone_clearance_m": self._goal_zone_clearance_m(),
    }


def _build_guidance_info(self, virtual_center: tuple[float, float]) -> dict[str, object]:
    """Build planner and guidance diagnostics without changing their public schema."""

    return {
        "route_plan_guidance": self._route_plan_guidance_summary(virtual_center),
        "route_guidance": self._route_guidance_summary(virtual_center),
        "route_guidance_visible": self._route_guidance_visible(),
        "guidance_shadow_mode": bool(getattr(self.multi_config.reasoning, "guidance_shadow_mode", False)),
        "route_guidance_meta": dict(self._route_guidance_meta),
        "route_guidance_source": self._route_guidance_meta.get("source"),
        "guidance_latency_ms": self._route_guidance_meta.get("latency_ms"),
        "guidance_cache_hit": self._route_guidance_meta.get("cache_hit"),
        "guidance_tracking_error_m": self._guidance_tracking_error_m(virtual_center),
        "route_guidance_tracking_error_m": self._route_guidance_tracking_error_m(virtual_center),
    }


def _build_episode_info(
    self,
    *,
    done_reason: str | None,
    reward_terms: dict[str, float] | None,
    virtual_center: tuple[float, float],
) -> dict[str, object]:
    """Build action-shield, planner-latency, and step-result fields."""

    planner_call_count = int(self._planner_call_count_episode)
    planner_latency_ms_total = float(self._planner_latency_ms_episode)
    return {
        "action_safety_shield": dict(self._last_action_shield_info),
        "planner_call_count": planner_call_count,
        "planner_latency_ms_total": planner_latency_ms_total,
        "planner_latency_ms_mean": (
            float(planner_latency_ms_total / planner_call_count) if planner_call_count > 0 else 0.0
        ),
        "done_reason": done_reason,
        "reward_terms": reward_terms or {},
        "virtual_center_xy": virtual_center,
    }


def _route_plan_guidance_summary(
    self,
    center_xy: tuple[float, float],
) -> dict[str, float] | None:
    if not bool(getattr(self.multi_config.reasoning, "global_planner_enabled", False)):
        return None
    path_waypoints = self._plan.waypoints_xy
    target_index = min(self._path_index, len(path_waypoints) - 1)
    target_xy = path_waypoints[target_index]
    rel_x = (float(target_xy[0]) - center_xy[0]) / 50.0
    rel_y = (float(target_xy[1]) - center_xy[1]) / 50.0
    heading_x, heading_y = self._current_guidance_heading(center_xy)
    distance = math.hypot(float(target_xy[0]) - center_xy[0], float(target_xy[1]) - center_xy[1])
    return {
        "target_rel_x": float(np.clip(rel_x, -1.0, 1.0)),
        "target_rel_y": float(np.clip(rel_y, -1.0, 1.0)),
        "heading_x": float(heading_x),
        "heading_y": float(heading_y),
        "distance_norm": float(np.clip(distance / 40.0, 0.0, 1.0)),
        "path_progress_norm": float(np.clip(self._path_index / max(len(path_waypoints) - 1, 1), 0.0, 1.0)),
        "path_index_norm": float(np.clip(target_index / max(len(path_waypoints) - 1, 1), 0.0, 1.0)),
        "speed_scale": float(np.clip(distance / max(self._resolved_max_command_speed_mps() * 20.0, 1e-6), 0.0, 1.0)),
        "confidence": 1.0,
    }


def _route_guidance_summary(
    self,
    center_xy: tuple[float, float],
) -> dict[str, float] | None:
    del center_xy
    if not self._guidance_runtime_active():
        return None
    return None if self._route_guidance_state is None else dict(self._route_guidance_state)


def _guidance_tracking_error_m(self, center_xy: tuple[float, float]) -> float | None:
    segment_start_xy, segment_end_xy = self._active_path_segment_xy()
    segment_dx = float(segment_end_xy[0] - segment_start_xy[0])
    segment_dy = float(segment_end_xy[1] - segment_start_xy[1])
    segment_norm = math.hypot(segment_dx, segment_dy)
    if segment_norm <= 1.0e-6:
        return 0.0
    relative_dx = float(center_xy[0] - segment_start_xy[0])
    relative_dy = float(center_xy[1] - segment_start_xy[1])
    lateral_distance = abs(relative_dx * segment_dy - relative_dy * segment_dx) / segment_norm
    return float(lateral_distance)


def _route_guidance_tracking_error_m(self, center_xy: tuple[float, float]) -> float | None:
    guidance = self._route_guidance_summary(center_xy)
    if guidance is None:
        return None
    target_rel_x = float(guidance.get("target_rel_x", 0.0)) * 50.0
    target_rel_y = float(guidance.get("target_rel_y", 0.0)) * 50.0
    return float(math.hypot(target_rel_x, target_rel_y))


def _boundary_proximity_deficit_m(self, positions_xy: np.ndarray) -> float:
    return boundary_proximity_deficit_m(
        positions_xy,
        world_x_bounds_m=self.env_config.world_x_bounds_m,
        world_y_bounds_m=self.env_config.world_y_bounds_m,
        agent_radius_m=self.env_config.drone_radius_m,
        soft_margin_m=self.env_config.boundary_soft_margin_m,
    )


def _any_gate_post_collision(
    self,
    previous_positions: np.ndarray,
    current_positions: np.ndarray,
    *,
    dynamic_gate_start_posts_xy: np.ndarray | None = None,
    dynamic_gate_end_posts_xy: np.ndarray | None = None,
) -> bool:
    self._last_dynamic_gate_collision = False
    for agent_idx in range(self._num_agents):
        if self.obstacle_map.segment_collides(
            (float(previous_positions[agent_idx, 0]), float(previous_positions[agent_idx, 1])),
            (float(current_positions[agent_idx, 0]), float(current_positions[agent_idx, 1])),
            drone_radius_m=self.env_config.drone_radius_m,
        ):
            return True
    if self._dynamic_gate_enabled and self._dynamic_gates:
        # Align swept dynamic-gate collision with the same interval as the
        # drone kinematics: previous drone positions at t, current positions
        # at t+dt, gate posts at t and t+dt. Using next_frame after
        # _step_count has already advanced shifts the gate one frame ahead
        # and makes eval/replay disagree.
        start_posts_xy = (
            np.asarray(dynamic_gate_start_posts_xy, dtype=np.float32)
            if dynamic_gate_start_posts_xy is not None
            else self._dynamic_gate_posts_xy(next_frame=False)
        )
        end_posts_xy = (
            np.asarray(dynamic_gate_end_posts_xy, dtype=np.float32)
            if dynamic_gate_end_posts_xy is not None
            else self._dynamic_gate_posts_xy(next_frame=True)
        )
        live_clearance = post_clearance(
            current_positions,
            end_posts_xy,
            config=self._dynamic_gate_config,
        )
        swept_clearance = swept_post_clearance(
            previous_positions,
            current_positions,
            start_posts_xy,
            end_posts_xy,
            config=self._dynamic_gate_config,
        )
        self._last_dynamic_gate_collision = bool(live_clearance <= 0.0 or swept_clearance <= 0.0)
        if self._last_dynamic_gate_collision:
            return True
    return False


def _any_out_of_bounds(self, positions_xy: np.ndarray) -> bool:
    x_min, x_max = self.env_config.world_x_bounds_m
    y_min, y_max = self.env_config.world_y_bounds_m
    for position in positions_xy:
        if not (x_min <= float(position[0]) <= x_max and y_min <= float(position[1]) <= y_max):
            return True
    return False


def _min_clearance(self, positions_xy: np.ndarray) -> float:
    if positions_xy.size == 0:
        return float("inf")
    static_clearance = min(
        self.obstacle_map.min_signed_distance(
            (float(position[0]), float(position[1])),
            drone_radius_m=self.env_config.drone_radius_m,
        )
        for position in positions_xy
    )
    if not self._dynamic_gate_enabled or not self._dynamic_gates:
        return static_clearance
    dynamic_clearance = post_clearance(
        positions_xy,
        self._dynamic_gate_posts_xy(next_frame=False),
        config=self._dynamic_gate_config,
    )
    return min(float(static_clearance), float(dynamic_clearance))


def _pairwise_collision_stats(self, positions_xy: np.ndarray) -> tuple[bool, float]:
    stats = pairwise_separation_stats(
        positions_xy,
        safe_distance_m=self.env_config.inter_agent_safe_distance_m,
    )
    return stats.collision, stats.min_distance_m


def _formation_shape_status(
    self,
    *,
    positions_xy: np.ndarray,
    center_xy: tuple[float, float],
    dynamic_gate_task_ratio: float,
    corridor_status: dict[str, object] | None = None,
) -> dict[str, object]:
    positions = np.asarray(positions_xy, dtype=np.float32)
    num_agents = int(positions.shape[0]) if positions.ndim == 2 else 0
    required_bands = int(getattr(self.env_config, "formation_line_collapse_min_lateral_bands", 0) or 0)
    if required_bands <= 0 or num_agents <= 0:
        return self._default_formation_shape_status()
    required_bands = max(1, min(required_bands, num_agents))
    task_ratio = float(dynamic_gate_task_ratio)
    corridor_inside = bool((corridor_status or {}).get("inside_gate_region", False))
    active = bool(
        self._dynamic_gate_enabled
        and self._dynamic_gates
        and (
            task_ratio >= float(getattr(self.env_config, "formation_line_collapse_task_ratio", 0.70) or 0.70)
            or corridor_inside
        )
    )
    if positions.ndim != 2 or positions.shape[1] != 2:
        status = self._default_formation_shape_status()
        status.update(
            {
                "checked": True,
                "active": active,
                "required_lateral_bands": required_bands,
                "dynamic_gate_task_ratio": task_ratio,
            }
        )
        return status

    heading = np.asarray(self._current_guidance_heading(center_xy), dtype=np.float32)
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= 1.0e-6:
        heading = np.asarray((1.0, 0.0), dtype=np.float32)
    else:
        heading = heading / heading_norm
    lateral_axis = np.asarray((-float(heading[1]), float(heading[0])), dtype=np.float32)
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    local_x = centered @ heading
    local_y = centered @ lateral_axis
    band_width = max(float(getattr(self.env_config, "formation_line_collapse_band_width_m", 0.50) or 0.50), 1.0e-6)
    lateral_band_count = self._count_lateral_bands(local_y, band_width_m=band_width)
    lateral_span = float(np.max(local_y) - np.min(local_y)) if local_y.size else 0.0
    longitudinal_span = float(np.max(local_x) - np.min(local_x)) if local_x.size else 0.0
    missing_bands = max(0, required_bands - int(lateral_band_count)) if active else 0
    collapse_score = float(missing_bands)
    return {
        "checked": True,
        "active": active,
        "lateral_band_count": int(lateral_band_count),
        "required_lateral_bands": int(required_bands),
        "lateral_span_m": lateral_span,
        "longitudinal_span_m": longitudinal_span,
        "band_width_m": float(band_width),
        "dynamic_gate_task_ratio": task_ratio,
        "formation_line_collapse_score": collapse_score,
        "formation_line_collapse_failure": bool(active and missing_bands > 0),
    }
