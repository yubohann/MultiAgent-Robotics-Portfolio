"""Replay buffer for fixed-size graph observations."""

from __future__ import annotations

import numpy as np
import torch


class GraphReplayBuffer:
    """A replay buffer specialized for graph-shaped observations."""

    def __init__(
        self,
        *,
        capacity: int,
        obs_shapes: dict[str, tuple[int, ...]],
        action_dim: int,
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._index = 0
        self._size = 0
        self._obs = {
            name: np.zeros((self.capacity,) + shape, dtype=np.float32)
            for name, shape in obs_shapes.items()
        }
        self._next_obs = {
            name: np.zeros((self.capacity,) + shape, dtype=np.float32)
            for name, shape in obs_shapes.items()
        }
        self._actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self._dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        for name, array in observation.items():
            self._obs[name][self._index] = np.asarray(array, dtype=np.float32)
        for name, array in next_observation.items():
            self._next_obs[name][self._index] = np.asarray(array, dtype=np.float32)
        self._actions[self._index] = np.asarray(action, dtype=np.float32)
        self._rewards[self._index, 0] = float(reward)
        self._dones[self._index, 0] = float(done)
        self._index = (self._index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: dict[str, np.ndarray],
        done: np.ndarray,
    ) -> None:
        batch_size = int(np.asarray(action).shape[0])
        for batch_idx in range(batch_size):
            self.add(
                {name: np.asarray(values[batch_idx], dtype=np.float32) for name, values in observation.items()},
                np.asarray(action[batch_idx], dtype=np.float32),
                float(np.asarray(reward[batch_idx]).reshape(())),
                {name: np.asarray(values[batch_idx], dtype=np.float32) for name, values in next_observation.items()},
                bool(np.asarray(done[batch_idx]).reshape(())),
            )

    def sample(self, batch_size: int, device: torch.device) -> dict[str, object]:
        if self._size < batch_size:
            raise ValueError(f"Not enough samples in replay buffer: {self._size} < {batch_size}")
        indices = self._rng.integers(0, self._size, size=(batch_size,))
        return {
            "obs": {
                name: torch.as_tensor(values[indices], dtype=torch.float32, device=device)
                for name, values in self._obs.items()
            },
            "next_obs": {
                name: torch.as_tensor(values[indices], dtype=torch.float32, device=device)
                for name, values in self._next_obs.items()
            },
            "actions": torch.as_tensor(self._actions[indices], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self._rewards[indices], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self._dones[indices], dtype=torch.float32, device=device),
        }

