"""Single-agent 2D gate environment for the isolated graph-RL experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from single_gate.configs.experiment_config import (
    SINGLE_EXPERIMENT_CONFIG,
    SingleGateEnvConfig,
    SingleGraphObservationConfig,
)
from single_gate.env.observation_single import build_single_graph_observation
from single_gate.rewards.single_agent_rewards import (
    compute_single_agent_reward,
    evaluate_single_agent_termination,
)
from shared.core.collision_2d import GateObstacleMap2D
from shared.core.kinematics_2d import (
    KinematicState2D,
    Kinematics2DConfig,
    Kinematics2DUpdater,
    PlanarVelocityCommand2D,
)


@dataclass(frozen=True)
class SingleGateState:
    """Minimal state snapshot exposed through env info and tests."""

    position_xy: tuple[float, float]
    velocity_xy: tuple[float, float]
    yaw_rad: float
    goal_xy: tuple[float, float]
    step_count: int


class SingleGate2DEnv:
    """A lightweight 2D gate traversal task with fixed flight height."""

    def __init__(
        self,
        *,
        env_config: SingleGateEnvConfig | None = None,
        observation_config: SingleGraphObservationConfig | None = None,
        obstacle_map: GateObstacleMap2D | None = None,
    ) -> None:
        self.env_config = env_config or SINGLE_EXPERIMENT_CONFIG.environment
        self.observation_config = observation_config or SINGLE_EXPERIMENT_CONFIG.observation
        self.obstacle_map = obstacle_map if obstacle_map is not None else GateObstacleMap2D.from_gate()
        self._kinematics = Kinematics2DUpdater(
            Kinematics2DConfig(
                dt_s=self.env_config.dt_s,
                max_speed_mps=self.env_config.max_command_speed_mps,
                max_accel_mps2=self.env_config.max_accel_mps2,
                align_yaw_to_velocity=True,
            )
        )
        self._rng = np.random.default_rng(0)
        self._state = KinematicState2D(x_m=self.env_config.start_x_m, y_m=0.0)
        self._goal_xy = (self.env_config.goal_x_m, 0.0)
        self._initial_goal_distance_m = 1.0
        self._previous_action = np.zeros((2,), dtype=np.float32)
        self._step_count = 0
        self._episode_done = False

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (2,)

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
        }

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sample_random_action(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

    def current_state(self) -> SingleGateState:
        return SingleGateState(
            position_xy=(float(self._state.x_m), float(self._state.y_m)),
            velocity_xy=(float(self._state.vx_mps), float(self._state.vy_mps)),
            yaw_rad=float(self._state.yaw_rad),
            goal_xy=self._goal_xy,
            step_count=self._step_count,
        )

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        if seed is not None:
            self.seed(seed)
        self._step_count = 0
        self._episode_done = False
        self._previous_action = np.zeros((2,), dtype=np.float32)
        sampled_start_y = float(self._rng.uniform(*self.env_config.start_y_range_m))
        start_y = self._resolve_safe_endpoint_y(
            x_m=self.env_config.start_x_m,
            y_range_m=self.env_config.start_y_range_m,
            preferred_y=sampled_start_y,
            required_clearance_m=max(2.0, self.env_config.drone_radius_m + 1.0),
        )
        sampled_goal_y = float(self._rng.uniform(*self.env_config.goal_y_range_m))
        goal_y = self._resolve_safe_endpoint_y(
            x_m=self.env_config.goal_x_m,
            y_range_m=self.env_config.goal_y_range_m,
            preferred_y=sampled_goal_y,
            required_clearance_m=max(self.env_config.goal_radius_m + 1.8, 3.6),
        )
        self._state = KinematicState2D(
            x_m=self.env_config.start_x_m,
            y_m=start_y,
            vx_mps=0.0,
            vy_mps=0.0,
            yaw_rad=0.0,
            t_sec=0.0,
        )
        self._goal_xy = (self.env_config.goal_x_m, goal_y)
        self._initial_goal_distance_m = self._goal_distance(
            (self._state.x_m, self._state.y_m),
            self._goal_xy,
        )
        observation = self._build_observation()
        info = self._info_dict(done_reason=None, reward_terms=None)
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        if self._episode_done:
            raise RuntimeError("Episode already ended. Call reset() before step().")

        action_np = np.asarray(action, dtype=np.float32).reshape(self.action_shape)
        clipped_action = np.clip(action_np, -1.0, 1.0)

        previous_position = (float(self._state.x_m), float(self._state.y_m))
        previous_goal_distance = self._goal_distance(previous_position, self._goal_xy)

        command = PlanarVelocityCommand2D(
            vx_cmd_mps=float(clipped_action[0] * self.env_config.max_command_speed_mps),
            vy_cmd_mps=float(clipped_action[1] * self.env_config.max_command_speed_mps),
        )
        next_state = self._kinematics.step(self._state, command)
        next_position = (float(next_state.x_m), float(next_state.y_m))

        segment_collision = self.obstacle_map.segment_collides(
            previous_position,
            next_position,
            drone_radius_m=self.env_config.drone_radius_m,
        )
        signed_clearance_m = self.obstacle_map.min_signed_distance(
            next_position,
            drone_radius_m=self.env_config.drone_radius_m,
        )
        collided = bool(segment_collision or signed_clearance_m <= 0.0)
        current_goal_distance = self._goal_distance(next_position, self._goal_xy)
        reached_goal = current_goal_distance <= self.env_config.goal_radius_m
        out_of_bounds = self._out_of_bounds(next_position)

        self._state = next_state
        self._step_count += 1

        termination = evaluate_single_agent_termination(
            collided=collided,
            reached_goal=reached_goal,
            out_of_bounds=out_of_bounds,
            step_count=self._step_count,
            config=self.env_config,
        )
        reward, reward_terms = compute_single_agent_reward(
            previous_goal_distance_m=previous_goal_distance,
            current_goal_distance_m=current_goal_distance,
            signed_clearance_m=signed_clearance_m,
            action=clipped_action,
            previous_action=self._previous_action,
            termination=termination,
            config=self.env_config,
        )
        self._previous_action = clipped_action
        self._episode_done = termination.terminated or termination.truncated

        observation = self._build_observation()
        info = self._info_dict(done_reason=termination.reason, reward_terms=reward_terms)
        return observation, reward, termination.terminated, termination.truncated, info

    def _build_observation(self) -> dict[str, np.ndarray]:
        position = (float(self._state.x_m), float(self._state.y_m))
        goal_distance = self._goal_distance(position, self._goal_xy)
        progress_ratio = 1.0 - (goal_distance / max(self._initial_goal_distance_m, 1e-6))
        observation = build_single_graph_observation(
            position_xy=position,
            velocity_xy=(float(self._state.vx_mps), float(self._state.vy_mps)),
            goal_xy=self._goal_xy,
            obstacle_map=self.obstacle_map,
            env_config=self.env_config,
            observation_config=self.observation_config,
            progress_ratio=float(np.clip(progress_ratio, 0.0, 1.0)),
            initial_goal_distance_m=self._initial_goal_distance_m,
        )
        return observation.as_dict()

    def _goal_distance(self, position_xy: tuple[float, float], goal_xy: tuple[float, float]) -> float:
        return math.hypot(goal_xy[0] - position_xy[0], goal_xy[1] - position_xy[1])

    def _out_of_bounds(self, position_xy: tuple[float, float]) -> bool:
        x_min, x_max = self.env_config.world_x_bounds_m
        y_min, y_max = self.env_config.world_y_bounds_m
        return not (x_min <= position_xy[0] <= x_max and y_min <= position_xy[1] <= y_max)

    def _resolve_safe_endpoint_y(
        self,
        *,
        x_m: float,
        y_range_m: tuple[float, float],
        preferred_y: float,
        required_clearance_m: float,
    ) -> float:
        low = float(min(y_range_m))
        high = float(max(y_range_m))
        clamped_preferred_y = min(max(float(preferred_y), low), high)
        resolution_m = 0.5
        span_steps = max(int(math.ceil((high - low) / resolution_m)), 0)
        candidate_values = {round(low + step * resolution_m, 4) for step in range(span_steps + 1)}
        candidate_values.add(round(high, 4))
        candidate_values.add(round(clamped_preferred_y, 4))
        candidates = sorted(candidate_values, key=lambda value: abs(float(value) - clamped_preferred_y))

        best_y = clamped_preferred_y
        best_score = float("-inf")
        for candidate_y in candidates:
            clearance_m = self.obstacle_map.min_signed_distance(
                (float(x_m), float(candidate_y)),
                drone_radius_m=self.env_config.drone_radius_m,
            )
            score = float(clearance_m) - 0.04 * abs(float(candidate_y) - clamped_preferred_y)
            if clearance_m >= float(required_clearance_m):
                return float(candidate_y)
            if score > best_score:
                best_score = score
                best_y = float(candidate_y)
        return best_y

    def _goal_zone_clearance_m(self) -> float:
        goal_center_clearance_m = self.obstacle_map.min_signed_distance(
            self._goal_xy,
            drone_radius_m=self.env_config.drone_radius_m,
        )
        return float(goal_center_clearance_m - self.env_config.goal_radius_m)

    def _info_dict(
        self,
        *,
        done_reason: str | None,
        reward_terms: dict[str, float] | None,
    ) -> dict[str, object]:
        position = (float(self._state.x_m), float(self._state.y_m))
        return {
            "state": self.current_state(),
            "fixed_height_m": self.env_config.fixed_height_m,
            "goal_distance_m": self._goal_distance(position, self._goal_xy),
            "signed_clearance_m": self.obstacle_map.min_signed_distance(
                position,
                drone_radius_m=self.env_config.drone_radius_m,
            ),
            "goal_zone_clearance_m": self._goal_zone_clearance_m(),
            "done_reason": done_reason,
            "reward_terms": reward_terms or {},
        }

