"""Multi-agent 2D gate environment with global route planning and formation slots."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from multi_gate.configs.experiment_config import (
    MULTI_EXPERIMENT_CONFIG,
    MultiExperimentConfig,
    MultiFormationConfig,
    MultiGateEnvConfig,
    MultiGraphObservationConfig,
    MultiPlannerConfig,
    is_dynamic_gate_density_scene_mode,
    is_exp3_empty_scene_mode,
)
from multi_gate.env import dynamic_gate_runtime as _dynamic_gate_runtime
from multi_gate.env import guidance_runtime as _guidance_runtime
from multi_gate.env import observation_runtime as _observation_runtime
from multi_gate.env import reward_runtime as _reward_runtime
from multi_gate.env import safety_shields as _safety_shields
from multi_gate.formation.virtual_structure import VirtualStructure2D
from multi_gate.guidance import RouteGuidanceEngine, build_guidance_engine_from_reasoning
from multi_gate.planners.global_route_planner import GlobalRoutePlan2D, GlobalRoutePlanner2D
from multi_gate.rewards.multi_agent_rewards import (
    compute_multi_agent_reward,
    evaluate_multi_agent_termination,
)
from shared.core.collision_2d import GateObstacleMap2D
from shared.core.dynamic_gate_density_2d import (
    DynamicGate2D,
    center_has_completed_corridor,
    corridor_region_status,
    validate_height_and_corridor_invariants,
)
from shared.core.kinematics_2d import (
    Kinematics2DConfig,
    Kinematics2DUpdater,
    KinematicState2D,
    PlanarVelocityCommand2D,
)
from shared.core.team_geometry import count_lateral_bands, slot_error_stats


@dataclass(frozen=True)
class MultiGateSnapshot:
    """Compact state summary exposed through env info and tests."""

    num_agents: int
    virtual_center_xy: tuple[float, float]
    mean_slot_error_m: float
    max_slot_error_m: float
    goal_distance_m: float
    path_index: int


class MultiGate2DEnv:
    """Variable-size multi-agent environment for 2D gate traversal."""

    def __init__(
        self,
        *,
        multi_config: MultiExperimentConfig | None = None,
        env_config: MultiGateEnvConfig | None = None,
        observation_config: MultiGraphObservationConfig | None = None,
        formation_config: MultiFormationConfig | None = None,
        planner_config: MultiPlannerConfig | None = None,
        obstacle_map: GateObstacleMap2D | None = None,
        virtual_structure: VirtualStructure2D | None = None,
        global_planner: GlobalRoutePlanner2D | None = None,
        guidance_engine: RouteGuidanceEngine | None = None,
    ) -> None:
        self.multi_config = multi_config or MULTI_EXPERIMENT_CONFIG
        self.env_config = env_config or self.multi_config.environment
        self.observation_config = observation_config or self.multi_config.observation
        self.formation_config = formation_config or self.multi_config.formation
        self.planner_config = planner_config or self.multi_config.planner
        self.max_agents_soft = self.multi_config.max_agents_soft
        resolved_scene_mode = str(getattr(self.multi_config.scene, "scene_mode", "")).strip().lower()
        self._dynamic_gate_enabled = is_dynamic_gate_density_scene_mode(resolved_scene_mode)
        self._dynamic_gate_config = self.multi_config.dynamic_gate_density
        self._dynamic_gates: list[DynamicGate2D] = []
        self._last_dynamic_gate_collision = False
        self._dynamic_gate_cache_step: int | None = None
        self._dynamic_gate_centers_cache: dict[bool, np.ndarray] = {}
        self._dynamic_gate_posts_cache: dict[bool, np.ndarray] = {}
        self._dynamic_gate_velocities_cache: np.ndarray | None = None
        self._dynamic_gate_obstacle_map_cache: GateObstacleMap2D | None = None
        self._active_obstacle_map_cache: GateObstacleMap2D | None = None
        self._last_height_escape_failure = False
        self._last_side_bypass_failure = False
        self._last_corridor_miss_failure = False
        self._last_formation_line_collapse_failure = False
        self._last_formation_shape_status = self._default_formation_shape_status()
        if obstacle_map is not None:
            self.obstacle_map = obstacle_map
        elif is_exp3_empty_scene_mode(resolved_scene_mode) or self._dynamic_gate_enabled:
            self.obstacle_map = GateObstacleMap2D.empty()
        else:
            self.obstacle_map = GateObstacleMap2D.from_gate(
                gate_post_radius_scale=self.env_config.gate_post_radius_scale,
            )
        self.virtual_structure = virtual_structure or VirtualStructure2D(self.formation_config)
        self.global_planner = global_planner or GlobalRoutePlanner2D(
            obstacle_map=self.obstacle_map,
            env_config=self.env_config,
            planner_config=self.planner_config,
        )
        self._kinematics = Kinematics2DUpdater(
            Kinematics2DConfig(
                dt_s=self.env_config.dt_s,
                max_speed_mps=self.env_config.max_command_speed_mps,
                max_speed_x_mps=self.env_config.max_command_forward_speed_mps,
                max_speed_y_mps=self.env_config.max_command_lateral_speed_mps,
                max_accel_mps2=self.env_config.max_accel_mps2,
                max_accel_x_mps2=self.env_config.max_forward_accel_mps2,
                max_accel_y_mps2=self.env_config.max_lateral_accel_mps2,
                align_yaw_to_velocity=True,
            )
        )
        self._rng = np.random.default_rng(0)
        self._num_agents = self.multi_config.default_agents
        self._states: list[KinematicState2D] = []
        self._goal_xy = (self.env_config.goal_x_m, 0.0)
        self._start_center_xy = (self.env_config.start_x_m, 0.0)
        self._plan = GlobalRoutePlan2D(
            waypoints_xy=((self._start_center_xy[0], self._start_center_xy[1]), self._goal_xy)
        )
        self._path_index = 1
        self._desired_slots = np.zeros((self.max_agents_soft, 2), dtype=np.float32)
        self._previous_action = np.zeros((self.max_agents_soft, 2), dtype=np.float32)
        self._last_action_shield_info = self._default_action_shield_info()
        self._previous_mean_slot_error = 0.0
        self._previous_goal_distance = 0.0
        self._initial_goal_distance = 0.0
        self._route_morph_active_index: int | None = None
        self._route_morph_phase_index: int | None = None
        self._step_count = 0
        self._planner_call_count_episode = 0
        self._planner_latency_ms_episode = 0.0
        self._episode_done = False
        self._owns_guidance_engine = False
        self._guidance_engine = guidance_engine
        if self._guidance_engine is None:
            self._guidance_engine = build_guidance_engine_from_reasoning(self.multi_config.reasoning)
            self._owns_guidance_engine = self._guidance_engine is not None
        self._route_guidance_state: dict[str, float] | None = None
        self._route_guidance_meta: dict[str, object] = self._default_route_guidance_meta(source="disabled")
        self._guidance_query_interval_steps = self._resolve_guidance_query_interval_steps()
        self._guidance_session_key = f"{self.__class__.__name__}:{id(self)}"

    @property
    def action_shape(self) -> tuple[int, int]:
        return (self.max_agents_soft, 2)

    @property
    def observation_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "node_features": (
                self.observation_config.max_nodes,
                self.observation_config.node_feature_dim,
            ),
            "adjacency": (
                self.observation_config.max_nodes,
                self.observation_config.max_nodes,
            ),
            "node_mask": (self.observation_config.max_nodes,),
            "action_mask": (self.max_agents_soft,),
        }

    def close(self) -> None:
        if self._owns_guidance_engine and self._guidance_engine is not None:
            self._guidance_engine.shutdown()
            self._guidance_engine = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup only
        try:
            self.close()
        except Exception:
            pass

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sample_random_action(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=self.action_shape).astype(np.float32)

    def active_positions_xy(self) -> np.ndarray:
        """Return active agent positions as an `(N, 2)` array."""

        return self._active_positions_xy().copy()

    def active_velocities_xy(self) -> np.ndarray:
        """Return active agent velocities as an `(N, 2)` array."""

        return self._active_velocities_xy().copy()

    def desired_slots_xy(self) -> np.ndarray:
        """Return desired slot positions for the active team."""

        return self._desired_slots[: self._num_agents].copy()

    def path_waypoints(self) -> tuple[tuple[float, float], ...]:
        """Return the current global route centerline path."""

        return self._plan.waypoints_xy

    def current_heading_xy(self) -> tuple[float, float]:
        """Return the current lookahead heading used by the team."""

        return self._current_guidance_heading(self._virtual_center_xy())

    def reset(
        self,
        *,
        seed: int | None = None,
        num_agents: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        if seed is not None:
            self.seed(seed)
        self._episode_done = False
        self._step_count = 0
        self._planner_call_count_episode = 0
        self._planner_latency_ms_episode = 0.0
        self._num_agents = self._resolve_num_agents(num_agents)
        self._reset_dynamic_gate_layout(seed=seed)
        self._path_index = 1
        self._route_morph_active_index = None
        self._route_morph_phase_index = None
        self._sync_virtual_structure_route_shape()
        self._previous_action = np.zeros((self.max_agents_soft, 2), dtype=np.float32)
        self._last_action_shield_info = self._default_action_shield_info()
        self._last_dynamic_gate_collision = False
        self._last_height_escape_failure = False
        self._last_side_bypass_failure = False
        self._last_corridor_miss_failure = False
        self._last_formation_line_collapse_failure = False
        self._last_formation_shape_status = self._default_formation_shape_status()

        formation_summary = self.virtual_structure.summary(self._num_agents)
        boundary_margin = self.env_config.drone_radius_m + self.planner_config.safety_margin_m + 0.5
        y_padding = formation_summary.lateral_half_span_m + boundary_margin
        configured_path = self._configured_path_waypoints()
        if configured_path:
            start_y = float(configured_path[0][1])
            goal_y = float(configured_path[-1][1])
            preferred_start_x = float(configured_path[0][0])
            preferred_goal_x = float(configured_path[-1][0])
        else:
            preferred_start_x = float(self.env_config.start_x_m)
            preferred_goal_x = float(self.env_config.goal_x_m)
        fixed_start_goal_y = None if configured_path else self._fixed_team_start_goal_y(self._num_agents)
        if configured_path:
            start_y = float(start_y)
            goal_y = float(goal_y)
        elif fixed_start_goal_y is None:
            start_y = float(self._rng.uniform(*self._clamped_y_range(self.env_config.start_y_range_m, y_padding)))
            goal_y = float(self._rng.uniform(*self._clamped_y_range(self.env_config.goal_y_range_m, y_padding)))
        else:
            start_y = self._clamp_fixed_y(float(fixed_start_goal_y[0]), y_padding)
            goal_y = self._clamp_fixed_y(float(fixed_start_goal_y[1]), y_padding)
        start_x = max(
            preferred_start_x,
            self.env_config.world_x_bounds_m[0] + formation_summary.trailing_length_m + boundary_margin,
        )
        self._start_center_xy = (
            (float(start_x), float(start_y)) if configured_path else self._clamp_center_xy((start_x, start_y))
        )
        if bool(getattr(self.env_config, "preparation_hold_mode", False)):
            sampled_goal_xy = self._start_center_xy
        elif configured_path:
            sampled_goal_xy = (float(preferred_goal_x), float(goal_y))
        else:
            sampled_goal_xy = self._clamp_center_xy((preferred_goal_x, goal_y))
        self._goal_xy = (
            sampled_goal_xy
            if configured_path
            else self._resolve_safe_goal_center(
                start_center_xy=self._start_center_xy,
                preferred_goal_xy=sampled_goal_xy,
                num_agents=self._num_agents,
            )
        )
        self._start_center_xy, self._plan, initial_slots = self._resolve_safe_start_setup(
            start_center_xy=self._start_center_xy,
            goal_xy=self._goal_xy,
            num_agents=self._num_agents,
        )
        self._states = [
            KinematicState2D(
                x_m=float(position[0]),
                y_m=float(position[1]),
                vx_mps=0.0,
                vy_mps=0.0,
                yaw_rad=0.0,
                t_sec=0.0,
            )
            for position in initial_slots
        ]
        self._path_index = 1 if len(self._plan.waypoints_xy) > 1 else 0
        self._route_morph_active_index = None
        self._route_morph_phase_index = None
        self._sync_virtual_structure_route_shape()
        initial_virtual_center = self._virtual_center_xy()
        self._desired_slots = self._compute_desired_slots(self._slot_anchor_center_xy(initial_virtual_center))
        self._previous_mean_slot_error = self._slot_error_stats(
            self._active_positions_xy(),
            self._desired_slots[: self._num_agents],
        )[0]
        self._previous_goal_distance = self._goal_distance(initial_virtual_center)
        self._initial_goal_distance = self._previous_goal_distance
        self._route_guidance_state = None
        self._route_guidance_meta = self._default_route_guidance_meta(source="reset")
        self._refresh_route_guidance(initial_virtual_center, force=True)

        observation = self._build_observation()
        info = self._build_info(done_reason=None, reward_terms=None)
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        if self._episode_done:
            raise RuntimeError("Episode already ended. Call reset() before step().")

        padded_action = self._normalize_action(action)
        active_action = padded_action[: self._num_agents]
        active_action, self._last_action_shield_info = self._apply_action_safety_shield(active_action)

        previous_positions = self._active_positions_xy().copy()
        previous_dynamic_gate_posts_xy = (
            self._dynamic_gate_posts_xy(next_frame=False).copy()
            if self._dynamic_gate_enabled and self._dynamic_gates
            else None
        )
        previous_goal_distance = self._goal_distance(self._virtual_center_xy())
        previous_slot_error = self._slot_error_stats(
            previous_positions,
            self._desired_slots[: self._num_agents],
        )[0]

        next_states: list[KinematicState2D] = []
        for agent_idx, state in enumerate(self._states):
            agent_action = np.clip(active_action[agent_idx], -1.0, 1.0)
            commanded_velocity = self.action_to_velocity_command(agent_action)
            command = PlanarVelocityCommand2D(
                vx_cmd_mps=float(commanded_velocity[0]),
                vy_cmd_mps=float(commanded_velocity[1]),
            )
            next_states.append(self._kinematics.step(state, command))
        self._states = next_states
        self._step_count += 1

        current_positions = self._active_positions_xy()
        current_dynamic_gate_posts_xy = (
            self._dynamic_gate_posts_xy(next_frame=False).copy()
            if self._dynamic_gate_enabled and self._dynamic_gates
            else None
        )
        gate_post_collision = self._any_gate_post_collision(
            previous_positions,
            current_positions,
            dynamic_gate_start_posts_xy=previous_dynamic_gate_posts_xy,
            dynamic_gate_end_posts_xy=current_dynamic_gate_posts_xy,
        )
        agent_collision, min_pair_distance = self._pairwise_collision_stats(current_positions)
        out_of_bounds = self._any_out_of_bounds(current_positions)
        virtual_center = self._virtual_center_xy()
        self._update_path_index(virtual_center)
        slot_anchor_center = self._slot_anchor_center_xy(virtual_center)
        self._desired_slots = self._compute_desired_slots(slot_anchor_center)
        current_slot_error, current_max_slot_error = self._slot_error_stats(
            current_positions,
            self._desired_slots[: self._num_agents],
        )
        current_goal_distance = self._goal_distance(virtual_center)
        current_guidance_tracking_error = self._guidance_tracking_error_m(virtual_center)
        self._refresh_route_guidance(virtual_center)
        boundary_proximity_deficit = self._boundary_proximity_deficit_m(current_positions)
        min_clearance = self._min_clearance(current_positions)
        goal_termination_enabled = bool(getattr(self.env_config, "goal_termination_enabled", True))
        goal_requires_slot_tolerance = bool(getattr(self.env_config, "goal_requires_slot_tolerance", True))
        reached_goal = (
            goal_termination_enabled
            and current_goal_distance <= self.env_config.goal_radius_m
            and (not goal_requires_slot_tolerance or current_slot_error <= self.formation_config.goal_slot_tolerance_m)
            and min_clearance > 0.0
        )
        height_report = validate_height_and_corridor_invariants(config=self._dynamic_gate_config)
        corridor_status = corridor_region_status(current_positions, config=self._dynamic_gate_config)
        corridor_required = bool(
            self._dynamic_gate_enabled
            and self._dynamic_gates
            and getattr(self._dynamic_gate_config, "corridor_through_required", False)
        )
        self._last_height_escape_failure = bool(
            self._dynamic_gate_enabled and not bool(height_report.get("passed", False))
        )
        self._last_side_bypass_failure = bool(
            corridor_required and bool(corridor_status.get("side_bypass_failure", False))
        )
        self._last_corridor_miss_failure = bool(
            corridor_required
            and reached_goal
            and not center_has_completed_corridor(virtual_center, config=self._dynamic_gate_config)
        )
        _observation_desired_slots, dynamic_gate_task_ratio = self._actor_observation_desired_slots_xy(virtual_center)
        self._last_formation_shape_status = self._formation_shape_status(
            positions_xy=current_positions,
            center_xy=virtual_center,
            dynamic_gate_task_ratio=dynamic_gate_task_ratio,
            corridor_status=corridor_status,
        )
        self._last_formation_line_collapse_failure = bool(
            self._last_formation_shape_status.get("formation_line_collapse_failure", False)
        )

        termination = evaluate_multi_agent_termination(
            gate_post_collision=gate_post_collision,
            agent_collision=agent_collision,
            reached_goal=reached_goal,
            out_of_bounds=out_of_bounds,
            step_count=self._step_count,
            config=self.env_config,
            height_escape_failure=self._last_height_escape_failure,
            side_bypass_failure=self._last_side_bypass_failure,
            corridor_miss_failure=self._last_corridor_miss_failure,
            formation_line_collapse_failure=self._last_formation_line_collapse_failure,
        )
        reward, reward_terms = compute_multi_agent_reward(
            previous_goal_distance_m=previous_goal_distance,
            current_goal_distance_m=current_goal_distance,
            previous_mean_slot_error_m=previous_slot_error,
            current_mean_slot_error_m=current_slot_error,
            current_max_slot_error_m=current_max_slot_error,
            current_guidance_tracking_error_m=current_guidance_tracking_error,
            boundary_proximity_deficit_m=boundary_proximity_deficit,
            min_clearance_m=min_clearance,
            min_pair_distance_m=min_pair_distance,
            formation_line_collapse_score=float(
                self._last_formation_shape_status.get("formation_line_collapse_score", 0.0) or 0.0
            ),
            action=active_action,
            previous_action=self._previous_action[: self._num_agents],
            termination=termination,
            config=self.env_config,
        )
        self._previous_action[: self._num_agents] = active_action
        self._previous_mean_slot_error = current_slot_error
        self._previous_goal_distance = current_goal_distance
        self._episode_done = termination.terminated or termination.truncated

        observation = self._build_observation()
        info = self._build_info(done_reason=termination.reason, reward_terms=reward_terms)
        return observation, reward, termination.terminated, termination.truncated, info

    def snapshot(self) -> MultiGateSnapshot:
        mean_slot_error_m, max_slot_error_m = self._slot_error_stats(
            self._active_positions_xy(),
            self._desired_slots[: self._num_agents],
        )
        return MultiGateSnapshot(
            num_agents=self._num_agents,
            virtual_center_xy=self._virtual_center_xy(),
            mean_slot_error_m=mean_slot_error_m,
            max_slot_error_m=max_slot_error_m,
            goal_distance_m=self._goal_distance(self._virtual_center_xy()),
            path_index=self._path_index,
        )

    def _guidance_runtime_active(self, *args, **kwargs):
        return _guidance_runtime._guidance_runtime_active(self, *args, **kwargs)

    def _route_guidance_visible(self, *args, **kwargs):
        return _guidance_runtime._route_guidance_visible(self, *args, **kwargs)

    def _default_route_guidance_meta(self, *args, **kwargs):
        return _guidance_runtime._default_route_guidance_meta(self, *args, **kwargs)

    def _resolve_guidance_query_interval_steps(self, *args, **kwargs):
        return _guidance_runtime._resolve_guidance_query_interval_steps(self, *args, **kwargs)

    def _heuristic_route_guidance_summary(self, *args, **kwargs):
        return _guidance_runtime._heuristic_route_guidance_summary(self, *args, **kwargs)

    def _build_guidance_query_payload(self, *args, **kwargs):
        return _guidance_runtime._build_guidance_query_payload(self, *args, **kwargs)

    def _refresh_route_guidance(self, *args, **kwargs):
        return _guidance_runtime._refresh_route_guidance(self, *args, **kwargs)

    def _resolve_num_agents(self, num_agents: int | None) -> int:
        resolved = int(self.multi_config.default_agents if num_agents is None else num_agents)
        min_agents = int(self.multi_config.min_agents)
        max_agents = int(self.multi_config.max_agents_soft)
        if resolved < min_agents or resolved > max_agents:
            raise ValueError(f"num_agents must be within [{min_agents}, {max_agents}], got {resolved}")
        return resolved

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action_np = np.asarray(action, dtype=np.float32)
        expected_shape = (self._num_agents, 2)
        if action_np.shape == expected_shape:
            padded = np.zeros(self.action_shape, dtype=np.float32)
            padded[: self._num_agents] = action_np
            return padded
        if action_np.shape == self.action_shape:
            return action_np
        raise ValueError(f"Expected action shape {expected_shape} or {self.action_shape}, got {action_np.shape}")

    def _active_positions_xy(self) -> np.ndarray:
        return np.asarray([(state.x_m, state.y_m) for state in self._states], dtype=np.float32)

    def _active_velocities_xy(self) -> np.ndarray:
        return np.asarray([(state.vx_mps, state.vy_mps) for state in self._states], dtype=np.float32)

    def _reset_dynamic_gate_layout(self, *args, **kwargs):
        return _dynamic_gate_runtime._reset_dynamic_gate_layout(self, *args, **kwargs)

    def _clear_dynamic_gate_runtime_cache(self, *args, **kwargs):
        return _dynamic_gate_runtime._clear_dynamic_gate_runtime_cache(self, *args, **kwargs)

    def _ensure_dynamic_gate_runtime_cache(self, *args, **kwargs):
        return _dynamic_gate_runtime._ensure_dynamic_gate_runtime_cache(self, *args, **kwargs)

    def _dynamic_gate_time_s(self, *args, **kwargs):
        return _dynamic_gate_runtime._dynamic_gate_time_s(self, *args, **kwargs)

    def _dynamic_gate_centers_xy(self, *args, **kwargs):
        return _dynamic_gate_runtime._dynamic_gate_centers_xy(self, *args, **kwargs)

    def _dynamic_gate_posts_xy(self, *args, **kwargs):
        return _dynamic_gate_runtime._dynamic_gate_posts_xy(self, *args, **kwargs)

    def _dynamic_gate_velocities_xy(self, *args, **kwargs):
        return _dynamic_gate_runtime._dynamic_gate_velocities_xy(self, *args, **kwargs)

    def _dynamic_gate_obstacle_map(self, *args, **kwargs):
        return _dynamic_gate_runtime._dynamic_gate_obstacle_map(self, *args, **kwargs)

    def _active_obstacle_map(self, *args, **kwargs):
        return _dynamic_gate_runtime._active_obstacle_map(self, *args, **kwargs)

    def _dynamic_gate_motion_range_m(self, *args, **kwargs):
        return _dynamic_gate_runtime._dynamic_gate_motion_range_m(self, *args, **kwargs)

    def _virtual_center_xy(self) -> tuple[float, float]:
        positions = self._active_positions_xy()
        if positions.size == 0:
            return self._start_center_xy
        return (float(np.mean(positions[:, 0])), float(np.mean(positions[:, 1])))

    def _goal_distance(self, center_xy: tuple[float, float]) -> float:
        goal_xy = self._plan.waypoints_xy[-1]
        return math.hypot(goal_xy[0] - center_xy[0], goal_xy[1] - center_xy[1])

    def _resolved_forward_command_speed_mps(self) -> float:
        return float(
            self.env_config.max_command_speed_mps
            if self.env_config.max_command_forward_speed_mps is None
            else self.env_config.max_command_forward_speed_mps
        )

    def _resolved_lateral_command_speed_mps(self) -> float:
        return float(
            self.env_config.max_command_speed_mps
            if self.env_config.max_command_lateral_speed_mps is None
            else self.env_config.max_command_lateral_speed_mps
        )

    def _resolved_max_command_speed_mps(self) -> float:
        return float(
            max(
                self.env_config.max_command_speed_mps,
                self._resolved_forward_command_speed_mps(),
                self._resolved_lateral_command_speed_mps(),
            )
        )

    def action_to_velocity_command(
        self,
        action_xy: np.ndarray | tuple[float, float],
    ) -> np.ndarray:
        """Map one normalized action into the axis-limited planar velocity command."""

        action = np.asarray(action_xy, dtype=np.float32).reshape(2)
        forward_speed_limit_mps = self._resolved_forward_command_speed_mps()
        lateral_speed_limit_mps = self._resolved_lateral_command_speed_mps()
        return np.asarray(
            (
                float(np.clip(action[0], -1.0, 1.0) * forward_speed_limit_mps),
                float(np.clip(action[1], -1.0, 1.0) * lateral_speed_limit_mps),
            ),
            dtype=np.float32,
        )

    def desired_velocity_to_action(
        self,
        desired_velocity_xy: np.ndarray | tuple[float, float],
    ) -> np.ndarray:
        """Project one desired planar velocity into the env's normalized action space."""

        desired_velocity = np.asarray(desired_velocity_xy, dtype=np.float32).reshape(2)
        forward_speed_limit_mps = max(self._resolved_forward_command_speed_mps(), 1.0e-6)
        lateral_speed_limit_mps = max(self._resolved_lateral_command_speed_mps(), 1.0e-6)
        commanded_velocity = np.asarray(
            (
                float(np.clip(desired_velocity[0], -forward_speed_limit_mps, forward_speed_limit_mps)),
                float(np.clip(desired_velocity[1], -lateral_speed_limit_mps, lateral_speed_limit_mps)),
            ),
            dtype=np.float32,
        )
        if (
            self.env_config.max_command_forward_speed_mps is None
            and self.env_config.max_command_lateral_speed_mps is None
        ):
            speed = float(np.linalg.norm(commanded_velocity))
            max_speed = max(float(self.env_config.max_command_speed_mps), 1.0e-6)
            if speed > max_speed:
                commanded_velocity = commanded_velocity / speed * max_speed
        action = np.asarray(
            (
                float(commanded_velocity[0] / forward_speed_limit_mps),
                float(commanded_velocity[1] / lateral_speed_limit_mps),
            ),
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0)

    def _default_action_shield_info(self, *args, **kwargs):
        return _safety_shields._default_action_shield_info(self, *args, **kwargs)

    def _apply_action_safety_shield(self, *args, **kwargs):
        return _safety_shields._apply_action_safety_shield(self, *args, **kwargs)

    def _apply_post_gate_cruise_velocity_floor(self, *args, **kwargs):
        return _safety_shields._apply_post_gate_cruise_velocity_floor(self, *args, **kwargs)

    def _apply_pairwise_velocity_shield(self, *args, **kwargs):
        return _safety_shields._apply_pairwise_velocity_shield(self, *args, **kwargs)

    def _apply_boundary_velocity_shield(self, *args, **kwargs):
        return _safety_shields._apply_boundary_velocity_shield(self, *args, **kwargs)

    def _apply_guidance_corridor_velocity_shield(self, *args, **kwargs):
        return _safety_shields._apply_guidance_corridor_velocity_shield(self, *args, **kwargs)

    def _apply_dynamic_gate_channel_velocity_shield(self, *args, **kwargs):
        return _safety_shields._apply_dynamic_gate_channel_velocity_shield(self, *args, **kwargs)

    def _apply_obstacle_velocity_shield(self, *args, **kwargs):
        return _safety_shields._apply_obstacle_velocity_shield(self, *args, **kwargs)

    def _planner_inflation_radius(self, num_agents: int) -> float:
        self._sync_virtual_structure_route_shape()
        summary = self.virtual_structure.summary(num_agents)
        footprint_radius = math.hypot(summary.trailing_length_m, summary.lateral_half_span_m)
        tracking_buffer = min(
            1.4,
            0.5 * self.formation_config.goal_slot_tolerance_m + 0.1 * summary.row_count,
        )
        fixed_safety_radius = self.env_config.drone_radius_m + self.planner_config.safety_margin_m
        return footprint_radius + fixed_safety_radius + tracking_buffer

    def _update_path_index(self, center_xy: tuple[float, float]) -> None:
        if len(self._plan.waypoints_xy) <= 1:
            self._path_index = 0
            self._route_morph_active_index = None
            self._route_morph_phase_index = None
            self._sync_virtual_structure_route_shape()
            return
        previous_path_index = int(self._path_index)
        reach_tolerance = self._path_reach_tolerance_m(self._num_agents)
        while self._path_index < len(self._plan.waypoints_xy) - 1:
            target_xy = self._plan.waypoints_xy[self._path_index]
            distance = math.hypot(center_xy[0] - target_xy[0], center_xy[1] - target_xy[1])
            if distance > reach_tolerance and not self._has_passed_waypoint(
                center_xy, self._path_index, reach_tolerance
            ):
                break
            self._path_index += 1
        if int(self._path_index) != previous_path_index:
            morph_index = int(self._path_index) - 2
            route_morph_paths = tuple(getattr(self.formation_config, "bootstrap_route_morph_paths_xy", ()) or ())
            self._route_morph_active_index = morph_index if 0 <= morph_index < len(route_morph_paths) else None
            self._route_morph_phase_index = 1 if self._route_morph_active_index is not None else None
            self._sync_virtual_structure_route_shape()

    def _compute_desired_slots(self, center_xy: tuple[float, float]) -> np.ndarray:
        self._sync_virtual_structure_route_shape()
        morph_slots = self._active_morph_target_slots_xy()
        if morph_slots.size:
            padded = np.zeros((self.max_agents_soft, 2), dtype=np.float32)
            padded[: min(self._num_agents, morph_slots.shape[0])] = morph_slots[: self._num_agents]
            return padded
        heading_xy = self._current_guidance_heading(center_xy)
        slots = self.virtual_structure.slot_world_positions(
            center_xy=center_xy,
            heading_xy=heading_xy,
            num_agents=self._num_agents,
        )
        padded = np.zeros((self.max_agents_soft, 2), dtype=np.float32)
        padded[: self._num_agents] = slots
        return padded

    def _active_morph_target_slots_xy(self) -> np.ndarray:
        route_morph_paths = tuple(getattr(self.formation_config, "bootstrap_route_morph_paths_xy", ()) or ())
        if self._route_morph_active_index is None or self._route_morph_phase_index is None:
            return np.zeros((0, 2), dtype=np.float32)
        morph_index = int(self._route_morph_active_index)
        if morph_index < 0 or morph_index >= len(route_morph_paths):
            self._route_morph_active_index = None
            self._route_morph_phase_index = None
            return np.zeros((0, 2), dtype=np.float32)
        paths = tuple(route_morph_paths[morph_index])
        if not paths:
            self._route_morph_active_index = None
            self._route_morph_phase_index = None
            return np.zeros((0, 2), dtype=np.float32)
        phase_index = int(self._route_morph_phase_index)
        targets = self._morph_targets_for_phase(paths, phase_index)
        if targets.size and self._states:
            mean_error, _max_error = self._slot_error_stats(self._active_positions_xy(), targets[: self._num_agents])
            phase_tolerance = max(0.65, float(self.formation_config.goal_slot_tolerance_m))
            if mean_error <= phase_tolerance:
                next_phase_index = phase_index + 1
                if any(len(path) > next_phase_index for path in paths[: self._num_agents]):
                    self._route_morph_phase_index = next_phase_index
                    targets = self._morph_targets_for_phase(paths, next_phase_index)
                else:
                    self._route_morph_active_index = None
                    self._route_morph_phase_index = None
                    return np.zeros((0, 2), dtype=np.float32)
        return targets

    def _morph_targets_for_phase(
        self,
        paths: tuple[tuple[tuple[float, float], ...], ...],
        phase_index: int,
    ) -> np.ndarray:
        targets: list[tuple[float, float]] = []
        for path in paths[: self._num_agents]:
            if not path:
                continue
            resolved_index = max(0, min(int(phase_index), len(path) - 1))
            targets.append((float(path[resolved_index][0]), float(path[resolved_index][1])))
        return np.asarray(targets, dtype=np.float32)

    def _configured_path_waypoints(self) -> tuple[tuple[float, float], ...]:
        raw_waypoints = tuple(getattr(self.env_config, "path_waypoints_xy", ()) or ())
        resolved: list[tuple[float, float]] = []
        for point in raw_waypoints:
            if len(point) != 2:
                continue
            resolved.append((float(point[0]), float(point[1])))
        return tuple(resolved)

    def _configured_route_plan(
        self,
        *,
        fallback_start_xy: tuple[float, float],
        fallback_goal_xy: tuple[float, float],
    ) -> GlobalRoutePlan2D | None:
        waypoints = self._configured_path_waypoints()
        if len(waypoints) < 2:
            return None
        return GlobalRoutePlan2D(waypoints_xy=waypoints)

    def _active_bootstrap_shape_name(self) -> str | None:
        route_shapes = tuple(
            str(shape) for shape in (getattr(self.formation_config, "bootstrap_route_shape_names", ()) or ())
        )
        if route_shapes:
            segment_index = max(0, min(int(self._path_index) - 1, len(route_shapes) - 1))
            return route_shapes[segment_index]
        initial_shape = getattr(self.formation_config, "bootstrap_initial_shape_name", None)
        shape_name = initial_shape or getattr(self.formation_config, "bootstrap_shape_name", None)
        return None if shape_name is None else str(shape_name)

    def _active_bootstrap_slot_permutation(self) -> tuple[int, ...]:
        route_permutations = tuple(getattr(self.formation_config, "bootstrap_route_slot_permutations", ()) or ())
        if route_permutations:
            segment_index = max(0, min(int(self._path_index) - 1, len(route_permutations) - 1))
            return tuple(int(idx) for idx in route_permutations[segment_index])
        return tuple(int(idx) for idx in (getattr(self.formation_config, "bootstrap_slot_permutation", ()) or ()))

    def _morph_phase_targets_xy(self) -> np.ndarray:
        route_morph_paths = tuple(getattr(self.formation_config, "bootstrap_route_morph_paths_xy", ()) or ())
        if self._route_morph_active_index is None:
            return np.zeros((0, 2), dtype=np.float32)
        morph_index = int(self._route_morph_active_index)
        if morph_index < 0 or morph_index >= len(route_morph_paths):
            return np.zeros((0, 2), dtype=np.float32)
        phase_index = 1 if self._route_morph_phase_index is None else int(self._route_morph_phase_index)
        return self._morph_targets_for_phase(tuple(route_morph_paths[morph_index]), phase_index)

    def _sync_virtual_structure_route_shape(self) -> None:
        active_shape = self._active_bootstrap_shape_name()
        active_permutation = self._active_bootstrap_slot_permutation()
        current_shape = getattr(self.virtual_structure.config, "bootstrap_shape_name", None)
        current_permutation = tuple(getattr(self.virtual_structure.config, "bootstrap_slot_permutation", ()) or ())
        if current_shape == active_shape and current_permutation == active_permutation:
            return
        self.virtual_structure.config = replace(
            self.formation_config,
            bootstrap_shape_name=active_shape,
            bootstrap_slot_permutation=active_permutation,
        )

    def _slot_anchor_center_xy(self, center_xy: tuple[float, float]) -> tuple[float, float]:
        path_anchor_xy = self._path_projection_anchor_xy(center_xy)
        anchor_blend = float(np.clip(getattr(self.env_config, "slot_anchor_blend", 1.0), 0.0, 1.0))
        if anchor_blend <= 0.0:
            return (float(center_xy[0]), float(center_xy[1]))
        if anchor_blend >= 1.0:
            return path_anchor_xy
        return (
            float((1.0 - anchor_blend) * center_xy[0] + anchor_blend * path_anchor_xy[0]),
            float((1.0 - anchor_blend) * center_xy[1] + anchor_blend * path_anchor_xy[1]),
        )

    def _path_projection_anchor_xy(self, center_xy: tuple[float, float]) -> tuple[float, float]:
        segment_start_xy, segment_end_xy = self._active_path_segment_xy()
        segment_dx = float(segment_end_xy[0] - segment_start_xy[0])
        segment_dy = float(segment_end_xy[1] - segment_start_xy[1])
        segment_norm_sq = segment_dx * segment_dx + segment_dy * segment_dy
        if segment_norm_sq <= 1.0e-6:
            return (float(segment_start_xy[0]), float(segment_start_xy[1]))
        relative_dx = float(center_xy[0] - segment_start_xy[0])
        relative_dy = float(center_xy[1] - segment_start_xy[1])
        projection = (relative_dx * segment_dx + relative_dy * segment_dy) / segment_norm_sq
        projection_t = float(np.clip(projection, 0.0, 1.0))
        return (
            float(segment_start_xy[0] + projection_t * segment_dx),
            float(segment_start_xy[1] + projection_t * segment_dy),
        )

    def _active_path_segment_xy(self) -> tuple[tuple[float, float], tuple[float, float]]:
        if len(self._plan.waypoints_xy) <= 1:
            return (self._start_center_xy, self._goal_xy)
        target_index = max(1, min(self._path_index, len(self._plan.waypoints_xy) - 1))
        return (self._plan.waypoints_xy[target_index - 1], self._plan.waypoints_xy[target_index])

    def _resolve_safe_start_setup(
        self,
        *,
        start_center_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        num_agents: int,
    ) -> tuple[tuple[float, float], GlobalRoutePlan2D, np.ndarray]:
        inflation = self._planner_inflation_radius(num_agents)
        min_required_clearance = max(0.3, self.env_config.drone_radius_m * 0.85)
        min_required_swept_clearance = self._required_start_swept_clearance(num_agents)
        best_score = float("-inf")
        best_result: tuple[tuple[float, float], GlobalRoutePlan2D, np.ndarray] | None = None

        for candidate_center_xy in self._candidate_start_centers(start_center_xy):
            plan = self._configured_route_plan(
                fallback_start_xy=candidate_center_xy,
                fallback_goal_xy=goal_xy,
            ) or self._timed_global_plan(
                start_xy=candidate_center_xy,
                goal_xy=goal_xy,
                inflation_radius_m=inflation,
            )
            heading_xy = plan.heading_at(0)
            self._sync_virtual_structure_route_shape()
            slots_xy = self.virtual_structure.slot_world_positions(
                center_xy=candidate_center_xy,
                heading_xy=heading_xy,
                num_agents=num_agents,
            )
            shift_xy = self._formation_boundary_shift(slots_xy)
            if abs(shift_xy[0]) > 1e-6 or abs(shift_xy[1]) > 1e-6:
                candidate_center_xy = (
                    candidate_center_xy[0] + shift_xy[0],
                    candidate_center_xy[1] + shift_xy[1],
                )
                plan = self._configured_route_plan(
                    fallback_start_xy=candidate_center_xy,
                    fallback_goal_xy=goal_xy,
                ) or self._timed_global_plan(
                    start_xy=candidate_center_xy,
                    goal_xy=goal_xy,
                    inflation_radius_m=inflation,
                )
                heading_xy = plan.heading_at(0)
                self._sync_virtual_structure_route_shape()
                slots_xy = self.virtual_structure.slot_world_positions(
                    center_xy=candidate_center_xy,
                    heading_xy=heading_xy,
                    num_agents=num_agents,
                )
            min_clearance = self._min_clearance(slots_xy)
            swept_clearance = self._formation_swept_clearance(
                start_center_xy=candidate_center_xy,
                plan=plan,
                num_agents=num_agents,
            )
            shift_cost = math.hypot(
                candidate_center_xy[0] - start_center_xy[0],
                candidate_center_xy[1] - start_center_xy[1],
            )
            score = min(min_clearance, swept_clearance) - 0.05 * shift_cost
            if score > best_score:
                best_score = score
                best_result = (candidate_center_xy, plan, slots_xy)
            if min_clearance >= min_required_clearance and swept_clearance >= min_required_swept_clearance:
                return (candidate_center_xy, plan, slots_xy)

        if best_result is None:
            raise RuntimeError("Failed to construct an initial multi-agent start state")
        return best_result

    def _timed_global_plan(
        self,
        *,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        inflation_radius_m: float,
    ) -> GlobalRoutePlan2D:
        started = time.perf_counter()
        try:
            return self.global_planner.plan(
                start_xy=start_xy,
                goal_xy=goal_xy,
                inflation_radius_m=inflation_radius_m,
            )
        finally:
            self._planner_call_count_episode += 1
            self._planner_latency_ms_episode += float((time.perf_counter() - started) * 1000.0)

    def _candidate_start_centers(
        self,
        preferred_center_xy: tuple[float, float],
    ) -> list[tuple[float, float]]:
        resolution = max(1.0, float(self.planner_config.grid_resolution_m))
        candidate_offsets: list[tuple[int, int]] = [(0, 0)]
        for radius in range(1, 6):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate_offsets.append((dx, dy))

        candidates: list[tuple[float, float]] = []
        seen: set[tuple[int, int]] = set()
        for dx_steps, dy_steps in candidate_offsets:
            candidate_xy = (
                preferred_center_xy[0] + dx_steps * resolution,
                preferred_center_xy[1] + dy_steps * resolution,
            )
            clamped_xy = self._clamp_center_xy(candidate_xy)
            key = (
                int(round(clamped_xy[0] * 1000.0)),
                int(round(clamped_xy[1] * 1000.0)),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(clamped_xy)
        return candidates

    def _required_start_swept_clearance(self, num_agents: int) -> float:
        if num_agents <= 8:
            return max(0.3, self.env_config.drone_radius_m * 0.85)
        if num_agents <= 16:
            return 0.45
        return 0.75

    def _formation_swept_clearance(
        self,
        *,
        start_center_xy: tuple[float, float],
        plan: GlobalRoutePlan2D,
        num_agents: int,
    ) -> float:
        """Estimate near-start clearance for the whole formation footprint."""

        horizon_m = min(
            18.0,
            max(8.0, 1.2 * self.virtual_structure.summary(num_agents).trailing_length_m),
        )
        step_m = max(0.75, min(1.5, 0.5 * self.planner_config.grid_resolution_m))
        current_xy = np.asarray(start_center_xy, dtype=np.float32)
        path_index = 1 if len(plan.waypoints_xy) > 1 else 0
        min_clearance = float("inf")
        travelled_m = 0.0
        while travelled_m <= horizon_m + 1e-6:
            heading_xy, path_index = self._plan_heading_from_point(
                center_xy=(float(current_xy[0]), float(current_xy[1])),
                plan=plan,
                path_index=path_index,
            )
            slots_xy = self.virtual_structure.slot_world_positions(
                center_xy=(float(current_xy[0]), float(current_xy[1])),
                heading_xy=heading_xy,
                num_agents=num_agents,
            )
            min_clearance = min(min_clearance, self._min_clearance(slots_xy))
            current_xy = current_xy + np.asarray(heading_xy, dtype=np.float32) * float(step_m)
            travelled_m += step_m
        return float(min_clearance)

    @staticmethod
    def _plan_heading_from_point(
        *,
        center_xy: tuple[float, float],
        plan: GlobalRoutePlan2D,
        path_index: int,
    ) -> tuple[tuple[float, float], int]:
        if len(plan.waypoints_xy) <= 1:
            return ((1.0, 0.0), 0)
        resolved_index = max(1, min(int(path_index), len(plan.waypoints_xy) - 1))
        target_xy = plan.waypoints_xy[resolved_index]
        dx = target_xy[0] - center_xy[0]
        dy = target_xy[1] - center_xy[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-6 and resolved_index < len(plan.waypoints_xy) - 1:
            resolved_index += 1
            target_xy = plan.waypoints_xy[resolved_index]
            dx = target_xy[0] - center_xy[0]
            dy = target_xy[1] - center_xy[1]
            norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            return (plan.heading_at(resolved_index), resolved_index)
        return ((dx / norm, dy / norm), resolved_index)

    def _clamp_center_xy(self, center_xy: tuple[float, float]) -> tuple[float, float]:
        summary = self.virtual_structure.summary(self._num_agents)
        boundary_margin = self.env_config.drone_radius_m + self.planner_config.safety_margin_m + 0.5
        x_min, x_max = self.env_config.world_x_bounds_m
        y_min, y_max = self.env_config.world_y_bounds_m
        min_x = x_min + summary.trailing_length_m + boundary_margin
        max_x = x_max - summary.trailing_length_m - boundary_margin
        min_y = y_min + summary.lateral_half_span_m + boundary_margin
        max_y = y_max - summary.lateral_half_span_m - boundary_margin
        if min_x > max_x:
            clamped_x = (x_min + x_max) / 2.0
        else:
            clamped_x = min(max(center_xy[0], min_x), max_x)
        if min_y > max_y:
            clamped_y = (y_min + y_max) / 2.0
        else:
            clamped_y = min(max(center_xy[1], min_y), max_y)
        return (float(clamped_x), float(clamped_y))

    def _clamped_y_range(
        self,
        y_range: tuple[float, float],
        padding: float,
    ) -> tuple[float, float]:
        y_min, y_max = self.env_config.world_y_bounds_m
        low = max(float(y_range[0]), y_min + padding)
        high = min(float(y_range[1]), y_max - padding)
        if low > high:
            midpoint = (y_min + y_max) / 2.0
            return (midpoint, midpoint)
        return (low, high)

    def _clamp_fixed_y(self, y_value: float, padding: float) -> float:
        low, high = self._clamped_y_range((float(y_value), float(y_value)), padding)
        return float(low if low == high else min(max(float(y_value), low), high))

    def _fixed_team_start_goal_y(self, num_agents: int) -> tuple[float, float] | None:
        presets = tuple(getattr(self.env_config, "fixed_team_start_goal_y_m", ()) or ())
        if not presets:
            return None
        for preset in presets:
            if len(preset) != 3:
                continue
            preset_num_agents, start_y, goal_y = preset
            if int(preset_num_agents) == int(num_agents):
                return (float(start_y), float(goal_y))
        return None

    def _resolve_safe_goal_center(
        self,
        *,
        start_center_xy: tuple[float, float],
        preferred_goal_xy: tuple[float, float],
        num_agents: int,
    ) -> tuple[float, float]:
        required_clearance_m = max(self.formation_config.goal_slot_tolerance_m + 1.0, 2.6)
        best_goal_xy = preferred_goal_xy
        best_score = float("-inf")
        for candidate_goal_xy in self._candidate_start_centers(preferred_goal_xy):
            heading_xy = self._candidate_heading_xy(start_center_xy, candidate_goal_xy)
            self._sync_virtual_structure_route_shape()
            slots_xy = self.virtual_structure.slot_world_positions(
                center_xy=candidate_goal_xy,
                heading_xy=heading_xy,
                num_agents=num_agents,
            )
            shift_xy = self._formation_boundary_shift(slots_xy)
            if abs(shift_xy[0]) > 1.0e-6 or abs(shift_xy[1]) > 1.0e-6:
                candidate_goal_xy = self._clamp_center_xy(
                    (
                        candidate_goal_xy[0] + shift_xy[0],
                        candidate_goal_xy[1] + shift_xy[1],
                    )
                )
                heading_xy = self._candidate_heading_xy(start_center_xy, candidate_goal_xy)
                slots_xy = self.virtual_structure.slot_world_positions(
                    center_xy=candidate_goal_xy,
                    heading_xy=heading_xy,
                    num_agents=num_agents,
                )
            min_clearance_m = self._min_clearance(slots_xy)
            score = float(min_clearance_m) - 0.03 * math.hypot(
                candidate_goal_xy[0] - preferred_goal_xy[0],
                candidate_goal_xy[1] - preferred_goal_xy[1],
            )
            if min_clearance_m >= required_clearance_m:
                return candidate_goal_xy
            if score > best_score:
                best_score = score
                best_goal_xy = candidate_goal_xy
        return best_goal_xy

    @staticmethod
    def _candidate_heading_xy(
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> tuple[float, float]:
        dx = float(goal_xy[0] - start_xy[0])
        dy = float(goal_xy[1] - start_xy[1])
        norm = math.hypot(dx, dy)
        if norm <= 1.0e-6:
            return (1.0, 0.0)
        return (dx / norm, dy / norm)

    def _goal_zone_clearance_m(self) -> float:
        if len(self._plan.waypoints_xy) > 1:
            heading_xy = self._plan.heading_at(len(self._plan.waypoints_xy) - 1)
        else:
            heading_xy = self._candidate_heading_xy(self._start_center_xy, self._goal_xy)
        goal_slots_xy = self.virtual_structure.slot_world_positions(
            center_xy=self._goal_xy,
            heading_xy=heading_xy,
            num_agents=self._num_agents,
        )
        min_clearance_m = self._min_clearance(goal_slots_xy)
        return float(min_clearance_m - self.formation_config.goal_slot_tolerance_m)

    def _formation_boundary_shift(self, slots_xy: np.ndarray) -> tuple[float, float]:
        if slots_xy.size == 0:
            return (0.0, 0.0)
        x_min, x_max = self.env_config.world_x_bounds_m
        y_min, y_max = self.env_config.world_y_bounds_m
        margin = self.env_config.drone_radius_m + 0.2
        min_x = float(np.min(slots_xy[:, 0]))
        max_x = float(np.max(slots_xy[:, 0]))
        min_y = float(np.min(slots_xy[:, 1]))
        max_y = float(np.max(slots_xy[:, 1]))
        dx = 0.0
        dy = 0.0
        if min_x < x_min + margin:
            dx += (x_min + margin) - min_x
        if max_x > x_max - margin:
            dx -= max_x - (x_max - margin)
        if min_y < y_min + margin:
            dy += (y_min + margin) - min_y
        if max_y > y_max - margin:
            dy -= max_y - (y_max - margin)
        return (dx, dy)

    def _lookahead_waypoints(self) -> list[tuple[float, float]]:
        end_index = min(
            len(self._plan.waypoints_xy),
            self._path_index + self.observation_config.lookahead_waypoint_count,
        )
        return list(self._plan.waypoints_xy[self._path_index : end_index])

    def _path_reach_tolerance_m(self, num_agents: int) -> float:
        if self._configured_path_waypoints():
            return max(0.5, float(self.env_config.goal_radius_m))
        summary = self.virtual_structure.summary(num_agents)
        footprint_radius = math.hypot(summary.trailing_length_m, summary.lateral_half_span_m)
        extra_margin = 0.2 * footprint_radius + 0.5 * self.planner_config.grid_resolution_m
        return max(
            self.env_config.goal_radius_m,
            min(self.env_config.goal_radius_m + extra_margin, self.env_config.goal_radius_m + 2.4),
        )

    def _has_passed_waypoint(
        self,
        center_xy: tuple[float, float],
        waypoint_index: int,
        reach_tolerance_m: float,
    ) -> bool:
        if waypoint_index <= 0 or waypoint_index >= len(self._plan.waypoints_xy):
            return False
        previous_xy = self._plan.waypoints_xy[waypoint_index - 1]
        target_xy = self._plan.waypoints_xy[waypoint_index]
        segment_dx = target_xy[0] - previous_xy[0]
        segment_dy = target_xy[1] - previous_xy[1]
        segment_norm = math.hypot(segment_dx, segment_dy)
        if segment_norm <= 1e-6:
            return False
        direction_xy = (segment_dx / segment_norm, segment_dy / segment_norm)
        relative_xy = (center_xy[0] - target_xy[0], center_xy[1] - target_xy[1])
        longitudinal = relative_xy[0] * direction_xy[0] + relative_xy[1] * direction_xy[1]
        lateral = abs(relative_xy[0] * direction_xy[1] - relative_xy[1] * direction_xy[0])
        return longitudinal >= 0.0 and lateral <= reach_tolerance_m

    def _current_guidance_heading(self, center_xy: tuple[float, float]) -> tuple[float, float]:
        if len(self._plan.waypoints_xy) <= 1:
            return (1.0, 0.0)
        target_xy = self._plan.waypoints_xy[self._path_index]
        dx = target_xy[0] - center_xy[0]
        dy = target_xy[1] - center_xy[1]
        if self._path_index >= len(self._plan.waypoints_xy) - 1:
            terminal_heading = self._plan.heading_at(self._path_index)
            longitudinal_error = dx * terminal_heading[0] + dy * terminal_heading[1]
            if longitudinal_error <= 0.0:
                return terminal_heading
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            return self._plan.heading_at(self._path_index)
        return (dx / norm, dy / norm)

    def _actor_observation_desired_slots_xy(self, *args, **kwargs):
        return _observation_runtime._actor_observation_desired_slots_xy(self, *args, **kwargs)

    def _dynamic_gate_task_slots_for_observation(self, *args, **kwargs):
        return _observation_runtime._dynamic_gate_task_slots_for_observation(self, *args, **kwargs)

    def _dynamic_gate_observation_feedforward_speed_mps(self, *args, **kwargs):
        return _observation_runtime._dynamic_gate_observation_feedforward_speed_mps(self, *args, **kwargs)

    def _build_observation(self, *args, **kwargs):
        return _observation_runtime._build_observation(self, *args, **kwargs)

    def _build_info(self, *args, **kwargs):
        return _reward_runtime._build_info(self, *args, **kwargs)

    def _route_plan_guidance_summary(self, *args, **kwargs):
        return _reward_runtime._route_plan_guidance_summary(self, *args, **kwargs)

    def _route_guidance_summary(self, *args, **kwargs):
        return _reward_runtime._route_guidance_summary(self, *args, **kwargs)

    def _guidance_tracking_error_m(self, *args, **kwargs):
        return _reward_runtime._guidance_tracking_error_m(self, *args, **kwargs)

    def _route_guidance_tracking_error_m(self, *args, **kwargs):
        return _reward_runtime._route_guidance_tracking_error_m(self, *args, **kwargs)

    def _boundary_proximity_deficit_m(self, *args, **kwargs):
        return _reward_runtime._boundary_proximity_deficit_m(self, *args, **kwargs)

    def _any_gate_post_collision(self, *args, **kwargs):
        return _reward_runtime._any_gate_post_collision(self, *args, **kwargs)

    def _any_out_of_bounds(self, *args, **kwargs):
        return _reward_runtime._any_out_of_bounds(self, *args, **kwargs)

    def _min_clearance(self, *args, **kwargs):
        return _reward_runtime._min_clearance(self, *args, **kwargs)

    def _pairwise_collision_stats(self, *args, **kwargs):
        return _reward_runtime._pairwise_collision_stats(self, *args, **kwargs)

    @staticmethod
    def _default_formation_shape_status() -> dict[str, object]:
        return {
            "checked": False,
            "active": False,
            "lateral_band_count": 0,
            "required_lateral_bands": 0,
            "lateral_span_m": 0.0,
            "longitudinal_span_m": 0.0,
            "band_width_m": 0.0,
            "dynamic_gate_task_ratio": 0.0,
            "formation_line_collapse_score": 0.0,
            "formation_line_collapse_failure": False,
        }

    def _formation_shape_status(self, *args, **kwargs):
        return _reward_runtime._formation_shape_status(self, *args, **kwargs)

    @staticmethod
    def _count_lateral_bands(local_lateral_positions_m: np.ndarray, *, band_width_m: float) -> int:
        return count_lateral_bands(local_lateral_positions_m, band_width_m=band_width_m)

    @staticmethod
    def _slot_error_stats(agent_positions_xy: np.ndarray, desired_slots_xy: np.ndarray) -> tuple[float, float]:
        return slot_error_stats(agent_positions_xy, desired_slots_xy)

    @staticmethod
    def _mean_slot_error(agent_positions_xy: np.ndarray, desired_slots_xy: np.ndarray) -> float:
        return slot_error_stats(agent_positions_xy, desired_slots_xy)[0]
