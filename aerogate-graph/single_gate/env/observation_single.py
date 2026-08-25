"""Graph observation builder for the single-agent 2D gate task."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from single_gate.configs.experiment_config import SingleGateEnvConfig, SingleGraphObservationConfig
from shared.core.collision_2d import GateObstacleMap2D, GatePostObstacle2D


@dataclass(frozen=True)
class SingleGraphObservation:
    """Fixed-size graph tensors consumed by the Graph-FlashSAC agent."""

    node_features: np.ndarray
    adjacency: np.ndarray
    node_mask: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "node_features": self.node_features.astype(np.float32, copy=False),
            "adjacency": self.adjacency.astype(np.float32, copy=False),
            "node_mask": self.node_mask.astype(np.float32, copy=False),
        }


def _empty_observation(config: SingleGraphObservationConfig) -> SingleGraphObservation:
    return SingleGraphObservation(
        node_features=np.zeros((config.max_nodes, config.node_feature_dim), dtype=np.float32),
        adjacency=np.zeros((config.max_nodes, config.max_nodes), dtype=np.float32),
        node_mask=np.zeros((config.max_nodes,), dtype=np.float32),
    )


def _node_feature(
    *,
    type_one_hot: tuple[float, float, float, float],
    rel_x: float,
    rel_y: float,
    vel_x: float,
    vel_y: float,
    radius: float,
    progress_ratio: float,
    clearance_m: float,
    normalized_distance: float,
) -> np.ndarray:
    return np.array(
        [
            type_one_hot[0],
            type_one_hot[1],
            type_one_hot[2],
            type_one_hot[3],
            rel_x,
            rel_y,
            vel_x,
            vel_y,
            radius,
            progress_ratio,
            clearance_m,
            normalized_distance,
        ],
        dtype=np.float32,
    )


def _waypoint_positions(
    position_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    count: int,
) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    waypoints = []
    for idx in range(1, count + 1):
        alpha = idx / float(count + 1)
        x = (1.0 - alpha) * position_xy[0] + alpha * goal_xy[0]
        y = (1.0 - alpha) * position_xy[1] + alpha * goal_xy[1]
        waypoints.append((x, y))
    return waypoints


def _nearest_obstacles(
    obstacle_map: GateObstacleMap2D,
    position_xy: tuple[float, float],
    sensor_range_m: float,
    limit: int,
) -> list[GatePostObstacle2D]:
    candidates = list(obstacle_map.query_local(position_xy, sensor_range_m))
    candidates.sort(
        key=lambda obstacle: math.hypot(
            position_xy[0] - obstacle.center_xy[0],
            position_xy[1] - obstacle.center_xy[1],
        )
    )
    return candidates[:limit]


def build_single_graph_observation(
    *,
    position_xy: tuple[float, float],
    velocity_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    obstacle_map: GateObstacleMap2D,
    env_config: SingleGateEnvConfig,
    observation_config: SingleGraphObservationConfig,
    progress_ratio: float,
    initial_goal_distance_m: float,
) -> SingleGraphObservation:
    """Construct a fixed-size graph around the current agent state."""

    observation = _empty_observation(observation_config)
    current_goal_distance = math.hypot(goal_xy[0] - position_xy[0], goal_xy[1] - position_xy[1])
    normalized_goal_distance = current_goal_distance / max(initial_goal_distance_m, 1e-6)
    current_clearance = obstacle_map.min_signed_distance(position_xy, drone_radius_m=env_config.drone_radius_m)
    if not math.isfinite(current_clearance):
        current_clearance = float(observation_config.sensor_range_m)

    active_positions: list[tuple[float, float]] = []
    active_indices: list[int] = []

    def _set_node(index: int, feature: np.ndarray, world_position: tuple[float, float]) -> None:
        observation.node_features[index] = feature
        observation.node_mask[index] = 1.0
        active_indices.append(index)
        active_positions.append(world_position)

    _set_node(
        0,
        _node_feature(
            type_one_hot=(1.0, 0.0, 0.0, 0.0),
            rel_x=0.0,
            rel_y=0.0,
            vel_x=velocity_xy[0],
            vel_y=velocity_xy[1],
            radius=env_config.drone_radius_m,
            progress_ratio=progress_ratio,
            clearance_m=current_clearance,
            normalized_distance=normalized_goal_distance,
        ),
        position_xy,
    )

    goal_rel_x = goal_xy[0] - position_xy[0]
    goal_rel_y = goal_xy[1] - position_xy[1]
    _set_node(
        1,
        _node_feature(
            type_one_hot=(0.0, 1.0, 0.0, 0.0),
            rel_x=goal_rel_x,
            rel_y=goal_rel_y,
            vel_x=0.0,
            vel_y=0.0,
            radius=env_config.goal_radius_m,
            progress_ratio=progress_ratio,
            clearance_m=0.0,
            normalized_distance=normalized_goal_distance,
        ),
        goal_xy,
    )

    waypoints = _waypoint_positions(position_xy, goal_xy, observation_config.lookahead_waypoint_count)
    heading_den = max(current_goal_distance, 1e-6)
    heading_xy = (goal_rel_x / heading_den, goal_rel_y / heading_den)
    for offset, waypoint in enumerate(waypoints, start=2):
        waypoint_distance = math.hypot(waypoint[0] - position_xy[0], waypoint[1] - position_xy[1])
        _set_node(
            offset,
            _node_feature(
                type_one_hot=(0.0, 0.0, 1.0, 0.0),
                rel_x=waypoint[0] - position_xy[0],
                rel_y=waypoint[1] - position_xy[1],
                vel_x=heading_xy[0],
                vel_y=heading_xy[1],
                radius=0.0,
                progress_ratio=progress_ratio,
                clearance_m=0.0,
                normalized_distance=waypoint_distance / max(initial_goal_distance_m, 1e-6),
            ),
            waypoint,
        )

    obstacle_start = 2 + observation_config.lookahead_waypoint_count
    nearest_obstacles = _nearest_obstacles(
        obstacle_map,
        position_xy,
        observation_config.sensor_range_m,
        observation_config.nearest_obstacle_count,
    )
    for offset, obstacle in enumerate(nearest_obstacles, start=obstacle_start):
        obstacle_distance = math.hypot(
            obstacle.center_xy[0] - position_xy[0],
            obstacle.center_xy[1] - position_xy[1],
        )
        signed_clearance = obstacle_distance - obstacle.collision_radius_m - env_config.drone_radius_m
        _set_node(
            offset,
            _node_feature(
                type_one_hot=(0.0, 0.0, 0.0, 1.0),
                rel_x=obstacle.center_xy[0] - position_xy[0],
                rel_y=obstacle.center_xy[1] - position_xy[1],
                vel_x=0.0,
                vel_y=0.0,
                radius=obstacle.collision_radius_m,
                progress_ratio=progress_ratio,
                clearance_m=signed_clearance,
                normalized_distance=obstacle_distance / max(observation_config.sensor_range_m, 1e-6),
            ),
            obstacle.center_xy,
        )

    for i, index_i in enumerate(active_indices):
        pos_i = active_positions[i]
        for j, index_j in enumerate(active_indices):
            pos_j = active_positions[j]
            if index_i == index_j:
                observation.adjacency[index_i, index_j] = 1.0
                continue
            distance = math.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
            weight = math.exp(-distance / max(observation_config.adjacency_distance_scale_m, 1e-6))
            observation.adjacency[index_i, index_j] = float(weight)

    return observation

