"""Graph observation builder for the multi-agent 2D gate task."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from multi_gate.configs.experiment_config import MultiGateEnvConfig, MultiGraphObservationConfig
from shared.core.collision_2d import GateObstacleMap2D, GatePostObstacle2D


@dataclass(frozen=True)
class MultiGraphObservation:
    """Fixed-size graph tensors consumed by the Graph-FlashSAC agent."""

    node_features: np.ndarray
    adjacency: np.ndarray
    node_mask: np.ndarray
    action_mask: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "node_features": self.node_features.astype(np.float32, copy=False),
            "adjacency": self.adjacency.astype(np.float32, copy=False),
            "node_mask": self.node_mask.astype(np.float32, copy=False),
            "action_mask": self.action_mask.astype(np.float32, copy=False),
        }


def build_multi_graph_observation(
    *,
    agent_positions_xy: np.ndarray,
    agent_velocities_xy: np.ndarray,
    desired_slots_xy: np.ndarray,
    virtual_center_xy: tuple[float, float],
    lookahead_waypoints_xy: list[tuple[float, float]],
    lookahead_heading_xy: tuple[float, float],
    obstacle_map: GateObstacleMap2D,
    env_config: MultiGateEnvConfig,
    observation_config: MultiGraphObservationConfig,
    max_agents_soft: int,
    progress_ratio: float,
    min_clearance_m: float,
    route_plan_guidance: Mapping[str, float] | None = None,
    route_guidance: Mapping[str, float] | None = None,
) -> MultiGraphObservation:
    """Construct a fixed-size graph for the active team and planner context."""

    max_nodes = observation_config.max_nodes
    node_features = np.zeros((max_nodes, observation_config.node_feature_dim), dtype=np.float32)
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    node_mask = np.zeros((max_nodes,), dtype=np.float32)
    action_mask = np.zeros((max_agents_soft,), dtype=np.float32)

    world_positions: dict[int, tuple[float, float]] = {}
    scale_xy = 50.0
    speed_scale = max(env_config.max_command_speed_mps, 1e-6)
    num_agents = int(agent_positions_xy.shape[0])
    graph_agent_capacity = min(
        int(max_agents_soft),
        int(observation_config.max_agents_for_nodes or max_agents_soft),
    )
    graph_agent_count = min(num_agents, graph_agent_capacity)
    action_mask[:num_agents] = 1.0

    feature_dim = observation_config.node_feature_dim

    def _pad_feature(feature: list[float]) -> np.ndarray:
        feature_array = np.asarray(feature, dtype=np.float32)
        if feature_array.shape[0] >= feature_dim:
            return feature_array[:feature_dim]
        padded = np.zeros((feature_dim,), dtype=np.float32)
        padded[: feature_array.shape[0]] = feature_array
        return padded

    def _set_node(index: int, world_position: tuple[float, float], feature: list[float]) -> None:
        node_features[index] = _pad_feature(feature)
        node_mask[index] = 1.0
        world_positions[index] = world_position

    heading_x, heading_y = lookahead_heading_xy
    for agent_idx in range(graph_agent_count):
        pos = agent_positions_xy[agent_idx]
        vel = agent_velocities_xy[agent_idx]
        slot = desired_slots_xy[agent_idx]
        slot_error = slot - pos
        clearance = obstacle_map.min_signed_distance(
            (float(pos[0]), float(pos[1])),
            drone_radius_m=env_config.drone_radius_m,
        )
        _set_node(
            agent_idx,
            (float(pos[0]), float(pos[1])),
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                (float(pos[0]) - virtual_center_xy[0]) / scale_xy,
                (float(pos[1]) - virtual_center_xy[1]) / scale_xy,
                float(vel[0]) / speed_scale,
                float(vel[1]) / speed_scale,
                float(slot_error[0]) / 20.0,
                float(slot_error[1]) / 20.0,
                env_config.drone_radius_m / 10.0,
                float(np.clip(clearance / 10.0, -1.0, 1.0)),
                float(progress_ratio),
                float(num_agents / max(max_agents_soft, 1)),
                1.0,
            ],
        )

    slot_start = graph_agent_capacity
    for slot_idx in range(graph_agent_count):
        slot = desired_slots_xy[slot_idx]
        _set_node(
            slot_start + slot_idx,
            (float(slot[0]), float(slot[1])),
            [
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                (float(slot[0]) - virtual_center_xy[0]) / scale_xy,
                (float(slot[1]) - virtual_center_xy[1]) / scale_xy,
                0.0,
                0.0,
                heading_x,
                heading_y,
                0.0,
                float(slot_idx / max(num_agents, 1)),
                float(progress_ratio),
                float(num_agents / max(max_agents_soft, 1)),
                1.0,
            ],
        )

    waypoint_start = graph_agent_capacity * 2
    for offset, waypoint in enumerate(lookahead_waypoints_xy[: observation_config.lookahead_waypoint_count]):
        _set_node(
            waypoint_start + offset,
            waypoint,
            [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                (float(waypoint[0]) - virtual_center_xy[0]) / scale_xy,
                (float(waypoint[1]) - virtual_center_xy[1]) / scale_xy,
                0.0,
                0.0,
                heading_x,
                heading_y,
                0.0,
                float(offset / max(observation_config.lookahead_waypoint_count, 1)),
                float(progress_ratio),
                float(num_agents / max(max_agents_soft, 1)),
                1.0,
            ],
        )

    obstacle_start = waypoint_start + observation_config.lookahead_waypoint_count
    obstacles = list(obstacle_map.query_local(virtual_center_xy, 28.0))
    heading_norm = math.hypot(float(heading_x), float(heading_y))
    if heading_norm <= 1.0e-6:
        heading_unit = (1.0, 0.0)
    else:
        heading_unit = (float(heading_x) / heading_norm, float(heading_y) / heading_norm)

    def _dynamic_obstacle_velocity_xy(obstacle: GatePostObstacle2D) -> tuple[float, float]:
        velocity_xy = getattr(obstacle, "velocity_xy", (0.0, 0.0))
        return (float(velocity_xy[0]), float(velocity_xy[1]))

    def _obstacle_sort_key(obstacle: GatePostObstacle2D) -> tuple[float, float, float, float]:
        dx = float(obstacle.center_xy[0] - virtual_center_xy[0])
        dy = float(obstacle.center_xy[1] - virtual_center_xy[1])
        distance = math.hypot(dx, dy)
        forward = dx * heading_unit[0] + dy * heading_unit[1]
        lateral = abs(-heading_unit[1] * dx + heading_unit[0] * dy)
        if obstacle.species == "dynamic_gate_post":
            behind_penalty = max(0.0, -forward - 3.0) * 10.0
            route_risk = behind_penalty + 0.08 * max(forward, 0.0) + 0.45 * lateral
            return (0.0, route_risk, max(forward, 0.0), distance)
        return (1.0, distance, max(forward, 0.0), lateral)

    obstacles.sort(key=_obstacle_sort_key)
    for offset, obstacle in enumerate(obstacles[: observation_config.nearest_obstacle_count]):
        rel_x = float(obstacle.center_xy[0] - virtual_center_xy[0]) / scale_xy
        rel_y = float(obstacle.center_xy[1] - virtual_center_xy[1]) / scale_xy
        distance = math.hypot(
            obstacle.center_xy[0] - virtual_center_xy[0],
            obstacle.center_xy[1] - virtual_center_xy[1],
        )
        obstacle_velocity_xy = _dynamic_obstacle_velocity_xy(obstacle)
        is_dynamic_gate_post = obstacle.species == "dynamic_gate_post"
        prediction_horizon_s = 0.75 if is_dynamic_gate_post else 0.0
        future_x = float(obstacle.center_xy[0]) + obstacle_velocity_xy[0] * prediction_horizon_s
        future_y = float(obstacle.center_xy[1]) + obstacle_velocity_xy[1] * prediction_horizon_s
        nearest_agent_clearance = float("inf")
        for agent_position in agent_positions_xy[:num_agents]:
            nearest_agent_clearance = min(
                nearest_agent_clearance,
                math.hypot(
                    float(obstacle.center_xy[0]) - float(agent_position[0]),
                    float(obstacle.center_xy[1]) - float(agent_position[1]),
                )
                - float(obstacle.collision_radius_m)
                - float(env_config.drone_radius_m),
            )
        if not math.isfinite(nearest_agent_clearance):
            nearest_agent_clearance = distance - float(obstacle.collision_radius_m) - float(env_config.drone_radius_m)
        clearance_feature = (
            float(np.clip(nearest_agent_clearance / 10.0, -1.0, 1.0))
            if is_dynamic_gate_post
            else float(np.clip(distance / 40.0, 0.0, 1.0))
        )
        _set_node(
            obstacle_start + offset,
            obstacle.center_xy,
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                rel_x,
                rel_y,
                float(obstacle_velocity_xy[0]) / speed_scale if is_dynamic_gate_post else 0.0,
                float(obstacle_velocity_xy[1]) / speed_scale if is_dynamic_gate_post else 0.0,
                (future_x - virtual_center_xy[0]) / scale_xy if is_dynamic_gate_post else 0.0,
                (future_y - virtual_center_xy[1]) / scale_xy if is_dynamic_gate_post else 0.0,
                float(obstacle.collision_radius_m / 10.0),
                clearance_feature,
                float(progress_ratio),
                float(num_agents / max(max_agents_soft, 1)),
                1.0,
            ],
        )

    guidance_start = obstacle_start + observation_config.nearest_obstacle_count
    guidance_count = 0
    if observation_config.guidance_node_count > 0 and route_plan_guidance is not None:
        slow_target_rel_x = float(route_plan_guidance.get("target_rel_x", 0.0))
        slow_target_rel_y = float(route_plan_guidance.get("target_rel_y", 0.0))
        _set_node(
            guidance_start,
            (
                virtual_center_xy[0] + slow_target_rel_x * scale_xy,
                virtual_center_xy[1] + slow_target_rel_y * scale_xy,
            ),
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                slow_target_rel_x,
                slow_target_rel_y,
                float(route_plan_guidance.get("heading_x", 0.0)),
                float(route_plan_guidance.get("heading_y", 0.0)),
                float(route_plan_guidance.get("distance_norm", 0.0)),
                float(route_plan_guidance.get("path_progress_norm", 0.0)),
                float(route_plan_guidance.get("path_index_norm", 0.0)),
                float(route_plan_guidance.get("speed_scale", 0.0)),
                float(progress_ratio),
                float(num_agents / max(max_agents_soft, 1)),
                float(route_plan_guidance.get("confidence", 1.0)),
            ],
        )
        guidance_count += 1

    if observation_config.guidance_node_count > guidance_count and route_guidance is not None:
        guidance_target_rel_x = float(route_guidance.get("target_rel_x", 0.0))
        guidance_target_rel_y = float(route_guidance.get("target_rel_y", 0.0))
        _set_node(
            guidance_start + guidance_count,
            (
                virtual_center_xy[0] + guidance_target_rel_x * scale_xy,
                virtual_center_xy[1] + guidance_target_rel_y * scale_xy,
            ),
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                guidance_target_rel_x,
                guidance_target_rel_y,
                float(route_guidance.get("heading_x", 0.0)),
                float(route_guidance.get("heading_y", 0.0)),
                float(route_guidance.get("risk_level", 0.0)),
                float(route_guidance.get("formation_compactness", 0.0)),
                float(route_guidance.get("speed_scale", 0.0)),
                float(route_guidance.get("mode_code", 0.0)),
                float(progress_ratio),
                float(num_agents / max(max_agents_soft, 1)),
                float(route_guidance.get("confidence", 1.0)),
            ],
        )
        guidance_count += 1

    global_index = guidance_start + observation_config.guidance_node_count
    _set_node(
        global_index,
        virtual_center_xy,
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            heading_x,
            heading_y,
            0.0,
            float(np.clip(min_clearance_m / 10.0, -1.0, 1.0)),
            float(progress_ratio),
            float(num_agents / max(max_agents_soft, 1)),
            1.0,
        ],
    )

    active_indices = [idx for idx, value in enumerate(node_mask.tolist()) if value > 0.5]
    for idx_i in active_indices:
        pos_i = world_positions[idx_i]
        adjacency[idx_i, idx_i] = 1.0
        for idx_j in active_indices:
            if idx_i == idx_j:
                continue
            pos_j = world_positions[idx_j]
            distance = math.hypot(pos_i[0] - pos_j[0], pos_i[1] - pos_j[1])
            adjacency[idx_i, idx_j] = float(
                math.exp(-distance / max(observation_config.adjacency_distance_scale_m, 1e-6))
            )

    return MultiGraphObservation(
        node_features=node_features,
        adjacency=adjacency,
        node_mask=node_mask,
        action_mask=action_mask,
    )
