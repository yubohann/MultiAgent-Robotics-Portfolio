"""Graph-observation assembly and dynamic-gate formation targets."""

from __future__ import annotations

import math

import numpy as np

from multi_gate.dynamic_gate_task_slots import dynamic_gate_task_first_slots
from multi_gate.env.observation_multi import build_multi_graph_observation


def _actor_observation_desired_slots_xy(
    self,
    center_xy: tuple[float, float],
) -> tuple[np.ndarray, float]:
    base_slots = self._desired_slots[: self._num_agents].copy()
    task_slots, task_ratio = self._dynamic_gate_task_slots_for_observation(
        center_xy=center_xy,
        base_slots_xy=base_slots,
    )
    if task_slots is None:
        return base_slots, 0.0
    return task_slots[: self._num_agents].astype(np.float32, copy=False), float(task_ratio)


def _dynamic_gate_task_slots_for_observation(
    self,
    *,
    center_xy: tuple[float, float],
    base_slots_xy: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    if not self._dynamic_gate_enabled or not self._dynamic_gates or self._num_agents <= 1 or base_slots_xy.size == 0:
        return None, 0.0
    centers_xy = self._dynamic_gate_centers_xy(next_frame=False)
    if centers_xy.size == 0:
        return None, 0.0
    gate_velocities_xy = self._dynamic_gate_velocities_xy()
    gate_time_s = float(self._dynamic_gate_time_s(next_frame=False))
    feedforward_speed_mps = self._dynamic_gate_observation_feedforward_speed_mps(
        center_xy=center_xy,
        positions_xy=self._active_positions_xy(),
    )
    return dynamic_gate_task_first_slots(
        desired_slots_xy=base_slots_xy,
        positions_xy=self._active_positions_xy(),
        center_xy=np.asarray(center_xy, dtype=np.float32),
        gates=self._dynamic_gates,
        centers_xy=centers_xy,
        gate_velocities_xy=gate_velocities_xy,
        gate_cfg=self._dynamic_gate_config,
        gate_time_s=gate_time_s,
        feedforward_speed_mps=feedforward_speed_mps,
        start_x_m=float(self.env_config.start_x_m),
        goal_x_m=float(self.env_config.goal_x_m),
        drone_radius_m=float(self.env_config.drone_radius_m),
    )


def _dynamic_gate_observation_feedforward_speed_mps(
    self,
    *,
    center_xy: tuple[float, float],
    positions_xy: np.ndarray,
) -> float:
    max_speed = float(self.env_config.max_command_speed_mps)
    num_agents = int(self._num_agents)
    size_ratio = 0.74 if num_agents <= 4 else 0.79 if num_agents <= 16 else 0.72
    path_index = min(int(self._path_index), len(self._plan.waypoints_xy) - 1)
    target_xy = self._plan.waypoints_xy[path_index]
    target_distance_m = math.hypot(float(target_xy[0]) - center_xy[0], float(target_xy[1]) - center_xy[1])
    goal_distance_m = self._goal_distance(center_xy)
    min_clearance_m = self._min_clearance(positions_xy) if positions_xy.size else float("inf")
    desired_speed = min(
        max_speed * size_ratio,
        1.8 + 0.045 * goal_distance_m + 0.22 * min(target_distance_m, 8.0),
    )
    if goal_distance_m <= 10.0:
        desired_speed = min(desired_speed, 1.2 + 0.35 * goal_distance_m)
    if 8 < num_agents <= 16 and int(self._path_index) >= len(self._plan.waypoints_xy) - 3 and min_clearance_m >= 2.0:
        desired_speed = min(max_speed * 0.98, desired_speed * 1.35)
    if min_clearance_m < 0.75:
        desired_speed *= 0.58
    elif min_clearance_m < 1.5:
        desired_speed *= 0.76
    elif min_clearance_m < 2.5:
        desired_speed *= 0.9
    if getattr(self, "_step_count", 0) < 10 and num_agents >= 24:
        desired_speed *= 0.7

    min_pair_distance = self._pairwise_collision_stats(positions_xy)[1] if positions_xy.size else float("inf")
    base_pair_distance = float(self.env_config.inter_agent_safe_distance_m)
    if num_agents <= 3:
        pair_warning_distance = base_pair_distance * 1.90
    elif num_agents <= 8:
        pair_warning_distance = max(base_pair_distance * 2.50, 1.80)
    else:
        pair_warning_distance = base_pair_distance * 1.45
    pair_proximity_ratio = 0.0
    if np.isfinite(min_pair_distance):
        pair_proximity_ratio = float(
            np.clip(
                (pair_warning_distance - float(min_pair_distance)) / max(pair_warning_distance, 1.0e-6),
                0.0,
                1.0,
            )
        )
    if pair_proximity_ratio > 0.0:
        if num_agents <= 3:
            desired_speed *= float(np.clip(1.0 - 0.75 * pair_proximity_ratio, 0.28, 1.0))
        elif num_agents <= 8:
            desired_speed *= float(np.clip(1.0 - 0.55 * pair_proximity_ratio, 0.40, 1.0))
        else:
            desired_speed *= float(np.clip(1.0 - 0.35 * pair_proximity_ratio, 0.55, 1.0))
    return max(0.8, float(desired_speed))


def _build_observation(self) -> dict[str, np.ndarray]:
    positions = self._active_positions_xy()
    virtual_center = self._virtual_center_xy()
    goal_distance = self._goal_distance(virtual_center)
    initial_goal_distance = math.hypot(
        self._goal_xy[0] - self._start_center_xy[0],
        self._goal_xy[1] - self._start_center_xy[1],
    )
    progress_ratio = 1.0 - goal_distance / max(initial_goal_distance, 1e-6)
    observation_desired_slots, _dynamic_gate_task_ratio = self._actor_observation_desired_slots_xy(virtual_center)
    observation = build_multi_graph_observation(
        agent_positions_xy=positions,
        agent_velocities_xy=self._active_velocities_xy(),
        desired_slots_xy=observation_desired_slots,
        virtual_center_xy=virtual_center,
        lookahead_waypoints_xy=self._lookahead_waypoints(),
        lookahead_heading_xy=self._current_guidance_heading(virtual_center),
        obstacle_map=self._active_obstacle_map(),
        env_config=self.env_config,
        observation_config=self.observation_config,
        max_agents_soft=self.max_agents_soft,
        progress_ratio=float(np.clip(progress_ratio, 0.0, 1.0)),
        min_clearance_m=self._min_clearance(positions),
        route_plan_guidance=self._route_plan_guidance_summary(virtual_center),
        route_guidance=self._route_guidance_summary(virtual_center),
    )
    return observation.as_dict()
