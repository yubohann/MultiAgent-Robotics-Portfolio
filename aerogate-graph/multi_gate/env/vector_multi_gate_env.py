"""Synchronous vector wrapper for the multi-agent 2D gate task."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from multi_gate.configs.experiment_config import (
    MULTI_EXPERIMENT_CONFIG,
    MultiExperimentConfig,
    MultiGateEnvConfig,
    MultiFormationConfig,
    MultiGraphObservationConfig,
    MultiPlannerConfig,
    is_dynamic_gate_density_scene_mode,
    is_exp3_empty_scene_mode,
)
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.env.multi_gate_kinematic_3d_env import MultiGateKinematic3DEnv
from multi_gate.formation.virtual_structure import VirtualStructure2D
from multi_gate.guidance import build_guidance_engine_from_reasoning
from multi_gate.planners.global_route_planner import GlobalRoutePlanner2D
from shared.core.collision_2d import GateObstacleMap2D
from shared.runtime.vector_training_utils import (
    done_indices,
    normalize_optional_int_sequence,
    replace_observation_rows,
    resolve_seeds,
    stack_observations,
)


@dataclass(frozen=True)
class VectorMultiResetResult:
    indices: np.ndarray
    observations: dict[str, np.ndarray]
    infos: list[dict[str, object]]


class VectorMultiGate2DEnv:
    """Run many independent multi-agent 2D gates through one sync API."""

    def __init__(
        self,
        *,
        num_envs: int,
        multi_config: MultiExperimentConfig | None = None,
        env_config: MultiGateEnvConfig | None = None,
        observation_config: MultiGraphObservationConfig | None = None,
        formation_config: MultiFormationConfig | None = None,
        planner_config: MultiPlannerConfig | None = None,
        env_cls: type[MultiGate2DEnv] = MultiGate2DEnv,
    ) -> None:
        self.num_envs = max(int(num_envs), 1)
        self.multi_config = multi_config or MULTI_EXPERIMENT_CONFIG
        self.env_config = env_config or self.multi_config.environment
        self.observation_config = observation_config or self.multi_config.observation
        self.formation_config = formation_config or self.multi_config.formation
        self.planner_config = planner_config or self.multi_config.planner
        self.env_cls = env_cls
        scene_mode = str(getattr(self.multi_config.scene, "scene_mode", "")).strip().lower()
        if is_exp3_empty_scene_mode(scene_mode) or is_dynamic_gate_density_scene_mode(scene_mode):
            self.obstacle_map = GateObstacleMap2D.empty()
        else:
            self.obstacle_map = GateObstacleMap2D.from_gate(
                gate_post_radius_scale=self.env_config.gate_post_radius_scale,
            )
        self.virtual_structure = VirtualStructure2D(self.formation_config)
        self.global_planner = GlobalRoutePlanner2D(
            obstacle_map=self.obstacle_map,
            env_config=self.env_config,
            planner_config=self.planner_config,
        )
        self.guidance_engine = build_guidance_engine_from_reasoning(self.multi_config.reasoning)
        self.envs = [
            self.env_cls(
                multi_config=self.multi_config,
                env_config=self.env_config,
                observation_config=self.observation_config,
                formation_config=self.formation_config,
                planner_config=self.planner_config,
                obstacle_map=self.obstacle_map,
                virtual_structure=self.virtual_structure,
                global_planner=self.global_planner,
                guidance_engine=self.guidance_engine,
            )
            for _ in range(self.num_envs)
        ]

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self.num_envs,) + self.envs[0].action_shape

    @property
    def observation_shapes(self) -> dict[str, tuple[int, ...]]:
        return self.envs[0].observation_shapes

    def sample_random_action(self) -> np.ndarray:
        return np.stack([env.sample_random_action() for env in self.envs], axis=0).astype(np.float32, copy=False)

    def reset(
        self,
        *,
        seed: int | None = None,
        num_agents: int | Sequence[int] | np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
        seeds = resolve_seeds(seed, self.num_envs)
        agent_counts = normalize_optional_int_sequence(num_agents, self.num_envs)
        observations = []
        infos = []
        for env, env_seed, env_agents in zip(self.envs, seeds, agent_counts):
            observation, info = env.reset(seed=env_seed, num_agents=env_agents)
            observations.append(observation)
            infos.append(info)
        return stack_observations(observations), infos

    def reset_done(
        self,
        done_mask: np.ndarray | list[bool],
        *,
        seed: int | None = None,
        num_agents: int | Sequence[int] | np.ndarray | None = None,
    ) -> VectorMultiResetResult:
        indices = done_indices(done_mask)
        if indices.size == 0:
            empty = {
                name: np.zeros((0,) + shape, dtype=np.float32)
                for name, shape in self.observation_shapes.items()
            }
            return VectorMultiResetResult(indices=indices, observations=empty, infos=[])
        seeds = resolve_seeds(seed, int(indices.size))
        agent_counts = normalize_optional_int_sequence(num_agents, int(indices.size))
        observations = []
        infos = []
        for env_idx, env_seed, env_agents in zip(indices.tolist(), seeds, agent_counts):
            observation, info = self.envs[int(env_idx)].reset(seed=env_seed, num_agents=env_agents)
            observations.append(observation)
            infos.append(info)
        return VectorMultiResetResult(
            indices=indices,
            observations=stack_observations(observations),
            infos=infos,
        )

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
        action_batch = np.asarray(actions, dtype=np.float32)
        if action_batch.shape != self.action_shape:
            raise ValueError(f"Expected action batch shape {self.action_shape}, got {action_batch.shape}")
        observations = []
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        infos: list[dict[str, object]] = []
        for env_idx, env in enumerate(self.envs):
            observation, reward, env_terminated, env_truncated, info = env.step(action_batch[env_idx])
            observations.append(observation)
            rewards[env_idx] = float(reward)
            terminated[env_idx] = bool(env_terminated)
            truncated[env_idx] = bool(env_truncated)
            infos.append(info)
        return stack_observations(observations), rewards, terminated, truncated, infos

    def replace_done_observations(
        self,
        observations: dict[str, np.ndarray],
        reset_result: VectorMultiResetResult,
    ) -> dict[str, np.ndarray]:
        return replace_observation_rows(observations, reset_result.indices, reset_result.observations)

    def close(self) -> None:
        if self.guidance_engine is not None:
            self.guidance_engine.shutdown()
            self.guidance_engine = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup only
        try:
            self.close()
        except Exception:
            pass


VectorMultiGateEnv = VectorMultiGate2DEnv
SupportedMultiEnvClass = MultiGate2DEnv | MultiGateKinematic3DEnv

