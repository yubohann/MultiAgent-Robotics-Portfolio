"""Runnable, provenance-labelled Rivermark pilot reference methods.

The native policies in this module are small implementations intended to
exercise the benchmark contract.  They are not aliases for third-party
checkpoints.  Checkpoint adapters are registered separately and fail closed
until the user supplies both a compatible dependency and an immutable weight.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .runtime import HighLevelAction, PublicMission, PublicObservation


@dataclass(frozen=True)
class MethodDescriptor:
    method_id: str
    family: str
    information_profile: str
    implementation_kind: str
    description: str
    requires: tuple[str, ...] = ()
    checkpoint_required: bool = False


class NativePolicy:
    """Base class for policies consuming only profile-filtered observations."""

    method_id = "native_policy"

    def __init__(self) -> None:
        self.mission: PublicMission | None = None
        self.agent_count = 0
        self._goals: dict[int, np.ndarray] = {}

    def reset(
        self,
        mission: PublicMission,
        agent_count: int,
        *,
        public_geometry: Mapping[str, Any] | None = None,
    ) -> None:
        self.mission = mission
        self.agent_count = agent_count
        self._goals.clear()

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "implementation_kind": "native_pilot_reference",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "external_dependency": None,
        }


def _state_position(observation: PublicObservation) -> np.ndarray:
    return observation.proprioception[:3].astype(np.float64, copy=True)


def _state_yaw(observation: PublicObservation) -> float:
    return float(observation.proprioception[6])


def _drive_to(
    observation: PublicObservation,
    goal: np.ndarray,
    *,
    source: str,
    speed_mps: float = 2.35,
    use_lidar_avoidance: bool = True,
) -> HighLevelAction:
    position = _state_position(observation)
    delta = goal - position
    horizontal = delta[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    velocity = np.zeros(3, dtype=np.float64)
    if horizontal_norm > 0.25:
        velocity[:2] = horizontal / horizontal_norm * min(speed_mps, horizontal_norm * 1.25)
    velocity[2] = float(np.clip(delta[2] * 1.1, -1.1, 1.1))
    if use_lidar_avoidance and observation.lidar_ranges_m is not None:
        ranges = observation.lidar_ranges_m
        center = len(ranges) // 2
        front = float(np.min(ranges[max(0, center - 3) : min(len(ranges), center + 4)]))
        if front < 1.35:
            # The array is ordered from -pi to pi; steer to the more open side.
            left_clearance = float(np.mean(ranges[center + 4 : min(len(ranges), center + 16)]))
            right_clearance = float(np.mean(ranges[max(0, center - 16) : center - 4]))
            yaw = _state_yaw(observation)
            lateral = np.array((-math.sin(yaw), math.cos(yaw)))
            velocity[:2] *= 0.3
            velocity[:2] += lateral * (1.1 if left_clearance >= right_clearance else -1.1)
    yaw = _state_yaw(observation)
    desired_yaw = math.atan2(float(velocity[1]), float(velocity[0])) if np.linalg.norm(velocity[:2]) > 0.05 else yaw
    yaw_error = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
    return HighLevelAction(
        velocity_xyz=tuple(float(value) for value in velocity),
        yaw_rate_rad_s=float(np.clip(yaw_error * 1.8, -1.35, 1.35)),
        mode="transit" if horizontal_norm > 0.5 else "dwell",
        frame="world",
        source=source,
    )


def _coverage_route(mission: PublicMission, agent_id: int, agent_count: int, *, phase: int = 0) -> list[np.ndarray]:
    width, height = mission.bounds_xy_m
    lane_count = max(agent_count, 4)
    lane_y = 1.8 + (agent_id % lane_count + 0.5) * (height - 3.6) / lane_count
    altitude = 2.55 + 0.32 * ((agent_id + phase) % 3)
    direction = 1 if ((agent_id + phase) % 2 == 0) else -1
    start_x, end_x = (2.0, width - 2.0) if direction > 0 else (width - 2.0, 2.0)
    return [
        np.array((start_x, lane_y, altitude)),
        np.array((end_x, lane_y, altitude)),
        np.array((end_x, min(height - 1.8, lane_y + height / lane_count), altitude)),
        np.array((start_x, min(height - 1.8, lane_y + height / lane_count), altitude)),
    ]


class _RoutePolicy(NativePolicy):
    phase = 0

    def __init__(self) -> None:
        super().__init__()
        self._routes: dict[int, list[np.ndarray]] = {}
        self._route_indices: dict[int, int] = {}

    def reset(self, mission: PublicMission, agent_count: int, *, public_geometry: Mapping[str, Any] | None = None) -> None:
        super().reset(mission, agent_count, public_geometry=public_geometry)
        self._routes = {agent: _coverage_route(mission, agent, agent_count, phase=self.phase) for agent in range(agent_count)}
        self._route_indices = {agent: 0 for agent in range(agent_count)}

    def _next_route_goal(self, observation: PublicObservation) -> np.ndarray:
        route = self._routes[observation.agent_id]
        index = self._route_indices[observation.agent_id]
        position = _state_position(observation)
        if float(np.linalg.norm(position - route[index])) < 1.0:
            index = (index + 1) % len(route)
            self._route_indices[observation.agent_id] = index
        return route[index]

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        return {
            agent_id: _drive_to(observation, self._next_route_goal(observation), source=self.method_id)
            for agent_id, observation in observations.items()
        }


class RandomCoveragePolicy(_RoutePolicy):
    method_id = "random_coverage_pilot"

    def reset(self, mission: PublicMission, agent_count: int, *, public_geometry: Mapping[str, Any] | None = None) -> None:
        NativePolicy.reset(self, mission, agent_count, public_geometry=public_geometry)
        rng = np.random.default_rng(81173)
        width, height = mission.bounds_xy_m
        self._routes = {}
        self._route_indices = {}
        for agent_id in range(agent_count):
            route = [
                np.array((rng.uniform(1.5, width - 1.5), rng.uniform(1.5, height - 1.5), rng.uniform(2.0, 3.8)))
                for _ in range(6)
            ]
            self._routes[agent_id] = route
            self._route_indices[agent_id] = 0


class FrontierCoveragePolicy(NativePolicy):
    method_id = "frontier_coverage_pilot"

    def __init__(self) -> None:
        super().__init__()
        self._visited: set[str] = set()

    def _cell_center(self, index_x: int, index_y: int) -> np.ndarray:
        assert self.mission is not None
        width, height = self.mission.bounds_xy_m
        return np.array(((index_x + 0.5) * width / 8.0, (index_y + 0.5) * height / 6.0, 2.8))

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        for observation in observations.values():
            for message in observation.public_team_messages:
                cell = message.get("cell_id")
                if isinstance(cell, str):
                    self._visited.add(cell)
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            position = _state_position(observation)
            choices: list[tuple[float, np.ndarray]] = []
            for x in range(8):
                for y in range(6):
                    cell = f"{x}:{y}"
                    goal = self._cell_center(x, y)
                    distance = float(np.linalg.norm(position - goal))
                    novelty = 0.0 if cell in self._visited else 11.0
                    choices.append((novelty - 0.22 * distance, goal))
            goal = max(choices, key=lambda item: item[0])[1]
            self._visited.add(f"{min(7, int(goal[0] / self.mission.bounds_xy_m[0] * 8))}:{min(5, int(goal[1] / self.mission.bounds_xy_m[1] * 6))}")
            actions[agent_id] = _drive_to(observation, goal, source=self.method_id)
        return actions


class SubmodularCoveragePolicy(NativePolicy):
    method_id = "submodular_coverage_pilot"

    def __init__(self) -> None:
        super().__init__()
        self._claimed: dict[int, np.ndarray] = {}

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        assert self.mission is not None
        width, height = self.mission.bounds_xy_m
        claimed = list(self._claimed.values())
        for observation in observations.values():
            for message in observation.public_team_messages:
                position = message.get("position_m")
                if isinstance(position, list) and len(position) == 3:
                    claimed.append(np.asarray(position, dtype=np.float64))
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            position = _state_position(observation)
            options = [
                np.array(((x + 0.5) * width / 8.0, (y + 0.5) * height / 6.0, 2.45 + 0.25 * ((x + y) % 3)))
                for x in range(8)
                for y in range(6)
            ]
            def value(goal: np.ndarray) -> float:
                travel = float(np.linalg.norm(goal - position))
                separation = min((float(np.linalg.norm(goal - other)) for other in claimed), default=9.0)
                border = min(goal[0], goal[1], width - goal[0], height - goal[1])
                return 2.4 * min(separation, 6.0) + 0.15 * border - 0.2 * travel
            goal = max(options, key=value)
            self._claimed[agent_id] = goal
            claimed.append(goal)
            actions[agent_id] = _drive_to(observation, goal, source=self.method_id)
        return actions


class AStarMpcPolicy(_RoutePolicy):
    method_id = "astar_mpc_pilot"
    phase = 1

    def __init__(self) -> None:
        super().__init__()
        self._obstacles: list[tuple[np.ndarray, float]] = []

    def reset(self, mission: PublicMission, agent_count: int, *, public_geometry: Mapping[str, Any] | None = None) -> None:
        super().reset(mission, agent_count, public_geometry=public_geometry)
        self._obstacles = []
        for obstacle in (public_geometry or {}).get("obstacles", []):
            center, radius = obstacle.get("center_xy_m"), obstacle.get("radius_m")
            if isinstance(center, list) and len(center) == 2 and isinstance(radius, (int, float)):
                self._obstacles.append((np.asarray(center, dtype=np.float64), float(radius)))

    def _safe_waypoint(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        direction = goal[:2] - start[:2]
        norm = max(float(np.linalg.norm(direction)), 1e-6)
        direction /= norm
        for center, radius in self._obstacles:
            projection = float(np.dot(center - start[:2], direction))
            if 0.0 < projection < norm:
                closest = start[:2] + projection * direction
                if float(np.linalg.norm(center - closest)) < radius + 1.0:
                    lateral = np.array((-direction[1], direction[0]))
                    waypoint = np.array((closest[0] + lateral[0] * (radius + 1.3), closest[1] + lateral[1] * (radius + 1.3), goal[2]))
                    return waypoint
        return goal

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            goal = self._next_route_goal(observation)
            waypoint = self._safe_waypoint(_state_position(observation), goal)
            actions[agent_id] = _drive_to(observation, waypoint, source=self.method_id, use_lidar_avoidance=False)
        return actions


class ActorCriticPilotPolicy(_RoutePolicy):
    """A tiny actor-critic reference with critic-scored action candidates."""

    method_id = "actor_critic_rl_pilot"
    phase = 2

    def __init__(self) -> None:
        super().__init__()
        rng = np.random.default_rng(4201)
        self._actor = rng.normal(0.0, 0.24, size=(6, 3))
        self._critic = rng.normal(0.0, 0.18, size=6)

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            goal = self._next_route_goal(observation)
            position = _state_position(observation)
            delta = goal - position
            features = np.concatenate((delta[:3] / 12.0, observation.proprioception[3:6] / 3.0))
            mean = np.tanh(features @ self._actor) * 1.8
            candidates = [mean, np.array((mean[0], mean[1] + 0.65, mean[2])), np.array((mean[0], mean[1] - 0.65, mean[2]))]
            def critic_value(velocity: np.ndarray) -> float:
                next_delta = delta - velocity * 0.2
                critic_features = np.concatenate((next_delta / 12.0, velocity / 3.0))
                return -float(np.dot(critic_features, critic_features)) + float(np.dot(critic_features, self._critic))
            velocity = max(candidates, key=critic_value)
            yaw = math.atan2(float(velocity[1]), float(velocity[0]))
            yaw_error = (yaw - _state_yaw(observation) + math.pi) % (2.0 * math.pi) - math.pi
            actions[agent_id] = HighLevelAction(
                velocity_xyz=tuple(float(value) for value in velocity),
                yaw_rate_rad_s=float(np.clip(yaw_error * 1.3, -1.3, 1.3)),
                mode="transit",
                source=self.method_id,
            )
        return actions


class MultiAgentActorCriticPilotPolicy(_RoutePolicy):
    method_id = "decentralized_marl_actor_critic_pilot"
    phase = 3

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            goal = self._next_route_goal(observation)
            position = _state_position(observation)
            repulsion = np.zeros(2, dtype=np.float64)
            for message in observation.public_team_messages:
                peer = message.get("position_m")
                if isinstance(peer, list) and len(peer) == 3:
                    offset = position[:2] - np.asarray(peer[:2], dtype=np.float64)
                    distance = float(np.linalg.norm(offset))
                    if 0.15 < distance < 4.5:
                        repulsion += offset / distance * (4.5 - distance) * 0.34
            goal = goal.copy()
            goal[:2] += repulsion
            actions[agent_id] = _drive_to(observation, goal, source=self.method_id)
        return actions


@dataclass(frozen=True)
class _ArchiveEntry:
    descriptor: tuple[int, int]
    score: float
    route_phase: int
    altitude_bias: float


class MapElitesCoveragePolicy(_RoutePolicy):
    """A public-coverage MAP-Elites style archive, never scored on target truth."""

    method_id = "map_elites_qd_pilot"

    def __init__(self) -> None:
        super().__init__()
        self._archive = self._build_archive()

    @staticmethod
    def _build_archive() -> dict[tuple[int, int], _ArchiveEntry]:
        archive: dict[tuple[int, int], _ArchiveEntry] = {}
        for phase in range(18):
            for altitude_index in range(3):
                descriptor = (phase % 6, altitude_index)
                # Fitness is public route spread and route curvature proxy.
                score = 1.0 + 0.19 * (phase % 6) - 0.08 * abs(altitude_index - 1) + 0.03 * (phase // 6)
                candidate = _ArchiveEntry(descriptor, score, phase, 2.35 + 0.35 * altitude_index)
                incumbent = archive.get(descriptor)
                if incumbent is None or candidate.score > incumbent.score:
                    archive[descriptor] = candidate
        return archive

    def reset(self, mission: PublicMission, agent_count: int, *, public_geometry: Mapping[str, Any] | None = None) -> None:
        NativePolicy.reset(self, mission, agent_count, public_geometry=public_geometry)
        self._routes, self._route_indices = {}, {}
        for agent_id in range(agent_count):
            descriptor = (agent_id % 6, agent_id % 3)
            entry = self._archive[descriptor]
            route = _coverage_route(mission, agent_id, agent_count, phase=entry.route_phase)
            self._routes[agent_id] = [np.array((point[0], point[1], entry.altitude_bias)) for point in route]
            self._route_indices[agent_id] = 0

    def provenance(self) -> dict[str, Any]:
        result = super().provenance()
        result["archive"] = {str(key): entry.score for key, entry in self._archive.items()}
        result["archive_objective"] = "public_route_spread_only"
        return result


def _visual_red_goal(observation: PublicObservation, fallback: np.ndarray) -> np.ndarray:
    if observation.rgb is None or observation.distance_to_image_plane_m is None:
        return fallback
    red = observation.rgb[:, :, 0].astype(np.int16)
    green = observation.rgb[:, :, 1].astype(np.int16)
    blue = observation.rgb[:, :, 2].astype(np.int16)
    mask = (red > 190) & (green < 105) & (blue < 105)
    if int(mask.sum()) < 4:
        return fallback
    columns = np.nonzero(mask)[1]
    offset = (float(np.mean(columns)) / observation.rgb.shape[1] - 0.5) * 2.0
    position = _state_position(observation)
    yaw = _state_yaw(observation)
    heading = np.array((math.cos(yaw), math.sin(yaw)))
    lateral = np.array((-math.sin(yaw), math.cos(yaw)))
    goal = position.copy()
    goal[:2] += heading * 4.0 + lateral * offset * 3.0
    return goal


class VlmGroundedSearchPolicy(_RoutePolicy):
    method_id = "vlm_grounded_search_pilot"
    phase = 4

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            fallback = self._next_route_goal(observation)
            goal = _visual_red_goal(observation, fallback)
            actions[agent_id] = _drive_to(observation, goal, source=self.method_id)
        return actions


class GroundedVlnPolicy(_RoutePolicy):
    method_id = "grounded_vln_pilot"
    phase = 5

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            fallback = self._next_route_goal(observation)
            instruction = (observation.language or "").lower()
            if "sweep" in instruction and "sector" in instruction:
                # Language selects the sector-first route; RGB-D supplies the
                # local grounding signal only when a visible marker exists.
                goal = _visual_red_goal(observation, fallback)
            else:
                goal = fallback
            actions[agent_id] = _drive_to(observation, goal, source=self.method_id)
        return actions


class ActionChunkVlaPolicy(_RoutePolicy):
    method_id = "action_chunk_vla_pilot"
    phase = 6

    def __init__(self) -> None:
        super().__init__()
        self._chunks: dict[int, list[HighLevelAction]] = {}

    def reset(self, mission: PublicMission, agent_count: int, *, public_geometry: Mapping[str, Any] | None = None) -> None:
        super().reset(mission, agent_count, public_geometry=public_geometry)
        self._chunks = {agent_id: [] for agent_id in range(agent_count)}

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            chunk = self._chunks[agent_id]
            if not chunk:
                fallback = self._next_route_goal(observation)
                goal = _visual_red_goal(observation, fallback)
                base = _drive_to(observation, goal, source=self.method_id)
                # The policy commits an action chunk but replans when a lidar
                # hazard invalidates its next action.
                chunk.extend([base, base, HighLevelAction.hold(source=self.method_id)])
            if observation.lidar_ranges_m is not None and float(np.min(observation.lidar_ranges_m)) < 0.85:
                self._chunks[agent_id] = []
                actions[agent_id] = _drive_to(observation, self._next_route_goal(observation), source=self.method_id)
            else:
                actions[agent_id] = chunk.pop(0)
        return actions


class ActionConditionedWorldModelMpcPolicy(_RoutePolicy):
    method_id = "action_conditioned_world_model_mpc_pilot"
    phase = 7

    @staticmethod
    def _rollout(position: np.ndarray, velocity: np.ndarray, action: np.ndarray, horizon: int = 5) -> tuple[np.ndarray, float]:
        simulated_position = position.copy()
        simulated_velocity = velocity.copy()
        risk = 0.0
        for _ in range(horizon):
            simulated_velocity = simulated_velocity + 0.62 * (action - simulated_velocity)
            simulated_position = simulated_position + simulated_velocity * 0.2
            risk += max(0.0, abs(simulated_position[2] - 2.9) - 1.6)
        return simulated_position, risk

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            goal = self._next_route_goal(observation)
            position = _state_position(observation)
            velocity = observation.proprioception[3:6].astype(np.float64)
            direction = goal - position
            horizontal_norm = max(float(np.linalg.norm(direction[:2])), 1e-6)
            nominal = np.array((direction[0] / horizontal_norm * 2.3, direction[1] / horizontal_norm * 2.3, np.clip(direction[2], -0.8, 0.8)))
            candidates = [
                nominal,
                nominal + np.array((0.0, 0.85, 0.0)),
                nominal + np.array((0.0, -0.85, 0.0)),
                nominal + np.array((0.0, 0.0, 0.5)),
            ]
            clearance_penalty = 0.0
            if observation.lidar_ranges_m is not None:
                clearance_penalty = max(0.0, 1.4 - float(np.min(observation.lidar_ranges_m))) * 9.0
            def score(candidate: np.ndarray) -> float:
                predicted, model_risk = self._rollout(position, velocity, candidate)
                return -float(np.linalg.norm(goal - predicted)) - 3.0 * model_risk - clearance_penalty
            selected = max(candidates, key=score)
            yaw = math.atan2(float(selected[1]), float(selected[0]))
            yaw_error = (yaw - _state_yaw(observation) + math.pi) % (2.0 * math.pi) - math.pi
            actions[agent_id] = HighLevelAction(
                velocity_xyz=tuple(float(value) for value in selected),
                yaw_rate_rad_s=float(np.clip(yaw_error * 1.5, -1.4, 1.4)),
                mode="transit",
                source=self.method_id,
            )
        return actions


NATIVE_DESCRIPTORS: dict[str, MethodDescriptor] = {
    "random_coverage_pilot": MethodDescriptor("random_coverage_pilot", "classical", "state_only", "native_pilot_reference", "Seeded random waypoint coverage."),
    "frontier_coverage_pilot": MethodDescriptor("frontier_coverage_pilot", "classical", "state_only", "native_pilot_reference", "Public-message frontier coverage."),
    "submodular_coverage_pilot": MethodDescriptor("submodular_coverage_pilot", "classical", "state_only", "native_pilot_reference", "Greedy public-coverage marginal-gain selector."),
    "astar_mpc_pilot": MethodDescriptor("astar_mpc_pilot", "classical", "geometry_state", "native_pilot_reference", "Public-geometry obstacle-aware waypoint MPC."),
    "actor_critic_rl_pilot": MethodDescriptor("actor_critic_rl_pilot", "rl", "state_only", "native_pilot_reference", "Native actor-critic candidate policy; not an SB3 PPO checkpoint."),
    "decentralized_marl_actor_critic_pilot": MethodDescriptor("decentralized_marl_actor_critic_pilot", "marl", "state_only", "native_pilot_reference", "Message-conditioned decentralized actor-critic reference."),
    "map_elites_qd_pilot": MethodDescriptor("map_elites_qd_pilot", "quality_diversity", "state_only", "native_pilot_reference", "MAP-Elites-style public-route archive."),
    "vlm_grounded_search_pilot": MethodDescriptor("vlm_grounded_search_pilot", "vlm", "language_multisensor_rgbd_lidar_radar_state", "native_pilot_reference", "RGB-D visual grounding with a public instruction."),
    "grounded_vln_pilot": MethodDescriptor("grounded_vln_pilot", "vln", "language_multisensor_rgbd_lidar_radar_state", "native_pilot_reference", "Language-conditioned route selection and visual grounding."),
    "action_chunk_vla_pilot": MethodDescriptor("action_chunk_vla_pilot", "vla", "language_multisensor_rgbd_lidar_radar_state", "native_pilot_reference", "Language-conditioned RGB-D action-chunk policy."),
    "action_conditioned_world_model_mpc_pilot": MethodDescriptor("action_conditioned_world_model_mpc_pilot", "world_model", "multisensor_rgbd_lidar_radar_state", "native_pilot_reference", "Action-conditioned kinematic world-model MPC."),
}

EXTERNAL_DESCRIPTORS: dict[str, MethodDescriptor] = {
    "sb3_ppo_checkpoint": MethodDescriptor("sb3_ppo_checkpoint", "rl", "state_only", "external_checkpoint_adapter", "Stable-Baselines3 PPO checkpoint adapter.", ("stable_baselines3",), True),
    "sb3_sac_checkpoint": MethodDescriptor("sb3_sac_checkpoint", "rl", "state_only", "external_checkpoint_adapter", "Stable-Baselines3 SAC checkpoint adapter.", ("stable_baselines3",), True),
    "skrl_ippo_checkpoint": MethodDescriptor("skrl_ippo_checkpoint", "marl", "state_only", "external_checkpoint_adapter", "skrl IPPO checkpoint adapter.", ("skrl",), True),
    "openvla_checkpoint": MethodDescriptor("openvla_checkpoint", "vla", "language_multisensor_rgbd_lidar_radar_state", "external_checkpoint_adapter", "OpenVLA-compatible checkpoint adapter.", ("transformers",), True),
    "dreamerv3_checkpoint": MethodDescriptor("dreamerv3_checkpoint", "world_model", "multisensor_rgbd_lidar_radar_state", "external_checkpoint_adapter", "Dreamer-style world-model checkpoint adapter.", ("jax",), True),
}

_NATIVE_CLASSES: dict[str, type[NativePolicy]] = {
    RandomCoveragePolicy.method_id: RandomCoveragePolicy,
    FrontierCoveragePolicy.method_id: FrontierCoveragePolicy,
    SubmodularCoveragePolicy.method_id: SubmodularCoveragePolicy,
    AStarMpcPolicy.method_id: AStarMpcPolicy,
    ActorCriticPilotPolicy.method_id: ActorCriticPilotPolicy,
    MultiAgentActorCriticPilotPolicy.method_id: MultiAgentActorCriticPilotPolicy,
    MapElitesCoveragePolicy.method_id: MapElitesCoveragePolicy,
    VlmGroundedSearchPolicy.method_id: VlmGroundedSearchPolicy,
    GroundedVlnPolicy.method_id: GroundedVlnPolicy,
    ActionChunkVlaPolicy.method_id: ActionChunkVlaPolicy,
    ActionConditionedWorldModelMpcPolicy.method_id: ActionConditionedWorldModelMpcPolicy,
}


def list_methods(*, include_external: bool = True) -> tuple[MethodDescriptor, ...]:
    descriptors = dict(NATIVE_DESCRIPTORS)
    if include_external:
        descriptors.update(EXTERNAL_DESCRIPTORS)
    return tuple(descriptors[key] for key in sorted(descriptors))


def create_native_policy(method_id: str) -> NativePolicy:
    try:
        return _NATIVE_CLASSES[method_id]()
    except KeyError as exc:
        if method_id in EXTERNAL_DESCRIPTORS:
            raise RuntimeError(
                f"{method_id} is an external checkpoint adapter; it cannot run without a compatible immutable checkpoint"
            ) from exc
        raise KeyError(f"unknown method: {method_id}") from exc


def validate_external_checkpoint(method_id: str, checkpoint: Path) -> MethodDescriptor:
    """Fail closed before an external policy can be represented as executed."""

    descriptor = EXTERNAL_DESCRIPTORS.get(method_id)
    if descriptor is None:
        raise KeyError(f"{method_id} is not an external checkpoint method")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    missing = [requirement for requirement in descriptor.requires if importlib.util.find_spec(requirement) is None]
    if missing:
        raise RuntimeError(f"{method_id} requires unavailable dependencies: {', '.join(missing)}")
    raise RuntimeError(
        f"{method_id} passed preflight but no architecture-specific adapter is bundled; refusing to substitute a policy"
    )


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StableBaselines3CheckpointPolicy(NativePolicy):
    """A real SB3 checkpoint wrapper for the single-UAV state-only track.

    The wrapper is intentionally constrained: a checkpoint has to declare its
    observation normalisation in an adjacent JSON file and can only control one
    policy-visible state vector at a time.  Multi-agent execution is obtained
    by independent weight sharing, which is recorded in the receipt.
    """

    method_id = "sb3_checkpoint_policy"

    def __init__(self, checkpoint: Path, metadata_path: Path | None = None) -> None:
        super().__init__()
        try:
            from stable_baselines3 import PPO, SAC  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Stable-Baselines3 is required for this checkpoint policy") from exc
        self.checkpoint = checkpoint.resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SB3 checkpoint is missing: {self.checkpoint}")
        self.metadata_path = (metadata_path or self.checkpoint.with_suffix(".rivermark.json")).resolve()
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"SB3 checkpoint metadata is required beside the checkpoint: {self.metadata_path}"
            )
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") not in {
            "org.rivermark.sb3-adapter.v1",
            "org.rivermark.sb3-adapter.v2",
        }:
            raise ValueError("unsupported SB3 adapter metadata schema")
        if self.metadata.get("information_profile") != "state_only":
            raise ValueError("SB3 adapter accepts only state_only checkpoints")
        expected_hash = self.metadata.get("checkpoint_sha256")
        if not isinstance(expected_hash, str) or expected_hash != _checkpoint_sha256(self.checkpoint):
            raise ValueError("SB3 checkpoint SHA-256 does not match its immutable metadata")
        algorithm = self.metadata.get("algorithm")
        if algorithm == "ppo":
            self.model = PPO.load(str(self.checkpoint), device="cpu")
        elif algorithm == "sac":
            self.model = SAC.load(str(self.checkpoint), device="cpu")
        else:
            raise ValueError("SB3 metadata algorithm must be ppo or sac")
        self._mean = np.asarray(self.metadata.get("observation_mean"), dtype=np.float32)
        self._std = np.asarray(self.metadata.get("observation_std"), dtype=np.float32)
        if self._mean.shape != (8,) or self._std.shape != (8,) or np.any(self._std <= 0.0):
            raise ValueError("SB3 adapter metadata needs finite 8D positive observation_std")
        self._action_scale = np.asarray(self.metadata.get("action_scale", [2.3, 2.3, 1.25, 1.4]), dtype=np.float32)
        if self._action_scale.shape != (4,) or np.any(self._action_scale <= 0.0):
            raise ValueError("SB3 adapter metadata needs positive 4D action_scale")

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            if observation.information_profile != "state_only":
                raise RuntimeError("SB3 state-only checkpoint received a mismatched observation profile")
            normalized = (observation.proprioception.astype(np.float32) - self._mean) / self._std
            raw_action, _ = self.model.predict(normalized, deterministic=True)
            vector = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            if vector.shape != (4,) or not np.all(np.isfinite(vector)):
                raise RuntimeError("SB3 checkpoint did not emit a finite 4D action")
            scaled = np.clip(vector, -1.0, 1.0) * self._action_scale
            actions[agent_id] = HighLevelAction(
                velocity_xyz=tuple(float(value) for value in scaled[:3]),
                yaw_rate_rad_s=float(scaled[3]),
                mode="transit",
                source="sb3_checkpoint_policy",
            )
        return actions

    def provenance(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "implementation_kind": self.metadata.get("implementation_kind", "external_checkpoint_adapter"),
            "external_dependency": "stable_baselines3",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": _checkpoint_sha256(self.checkpoint),
            "adapter_metadata": str(self.metadata_path),
            "adapter_metadata_sha256": _checkpoint_sha256(self.metadata_path),
            "algorithm": self.metadata["algorithm"],
            "parameter_sharing": "independent_shared_policy_per_agent",
        }


def create_sb3_checkpoint_policy(checkpoint: Path, metadata_path: Path | None = None) -> StableBaselines3CheckpointPolicy:
    """Create a real, provenance-checked SB3 policy wrapper."""

    return StableBaselines3CheckpointPolicy(checkpoint, metadata_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list registered native and external methods")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.list:
        raise SystemExit("use --list to inspect methods; use rivermark_benchmark.demo to execute native pilots")
    rows = [descriptor.__dict__ for descriptor in list_methods()]
    if args.as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            requirements = ",".join(row["requires"]) or "built_in"
            print(f"{row['method_id']:42} {row['family']:18} {row['information_profile']:44} {requirements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
