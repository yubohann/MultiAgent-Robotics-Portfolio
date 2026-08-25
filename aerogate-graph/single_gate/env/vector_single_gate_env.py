"""Synchronous vector wrapper for the single-agent 2D gate task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from single_gate.configs.experiment_config import (
    SINGLE_EXPERIMENT_CONFIG,
    SingleGateEnvConfig,
    SingleGraphObservationConfig,
)
from single_gate.env.single_gate_env import SingleGate2DEnv
from shared.core.collision_2d import GateObstacleMap2D
from shared.runtime.vector_training_utils import (
    done_indices,
    replace_observation_rows,
    resolve_seeds,
    stack_observations,
)


@dataclass(frozen=True)
class VectorSingleResetResult:
    indices: np.ndarray
    observations: dict[str, np.ndarray]
    infos: list[dict[str, object]]


class VectorSingleGate2DEnv:
    """Run many independent single-agent 2D gates through one sync API."""

    def __init__(
        self,
        *,
        num_envs: int,
        env_config: SingleGateEnvConfig | None = None,
        observation_config: SingleGraphObservationConfig | None = None,
    ) -> None:
        self.num_envs = max(int(num_envs), 1)
        self.env_config = env_config or SINGLE_EXPERIMENT_CONFIG.environment
        self.observation_config = observation_config or SINGLE_EXPERIMENT_CONFIG.observation
        self.obstacle_map = GateObstacleMap2D.from_gate()
        self.envs = [
            SingleGate2DEnv(
                env_config=self.env_config,
                observation_config=self.observation_config,
                obstacle_map=self.obstacle_map,
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

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
        seeds = resolve_seeds(seed, self.num_envs)
        observations = []
        infos = []
        for env, env_seed in zip(self.envs, seeds):
            observation, info = env.reset(seed=env_seed)
            observations.append(observation)
            infos.append(info)
        return stack_observations(observations), infos

    def reset_done(
        self,
        done_mask: np.ndarray | list[bool],
        *,
        seed: int | None = None,
    ) -> VectorSingleResetResult:
        indices = done_indices(done_mask)
        if indices.size == 0:
            empty = {
                name: np.zeros((0,) + shape, dtype=np.float32)
                for name, shape in self.observation_shapes.items()
            }
            return VectorSingleResetResult(indices=indices, observations=empty, infos=[])
        seeds = resolve_seeds(seed, int(indices.size))
        observations = []
        infos = []
        for env_idx, env_seed in zip(indices.tolist(), seeds):
            observation, info = self.envs[int(env_idx)].reset(seed=env_seed)
            observations.append(observation)
            infos.append(info)
        return VectorSingleResetResult(
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
        reset_result: VectorSingleResetResult,
    ) -> dict[str, np.ndarray]:
        return replace_observation_rows(observations, reset_result.indices, reset_result.observations)

