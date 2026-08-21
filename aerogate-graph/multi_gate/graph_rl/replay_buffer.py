"""Replay buffer for fixed-size multi-agent graph observations."""

from __future__ import annotations

import numpy as np
import torch


class MultiGraphReplayBuffer:
    """Replay buffer specialized for graph observations and joint actions."""

    def __init__(
        self,
        *,
        capacity: int,
        obs_shapes: dict[str, tuple[int, ...]],
        joint_action_shape: tuple[int, ...],
        seed: int = 0,
        failure_replay_ratio: float = 0.0,
        enable_failure_replay: bool = False,
    ) -> None:
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._index = 0
        self._size = 0
        self.failure_replay_ratio = float(np.clip(failure_replay_ratio, 0.0, 1.0))
        self.enable_failure_replay = bool(enable_failure_replay)
        self._obs = {
            name: np.zeros((self.capacity,) + shape, dtype=np.float32)
            for name, shape in obs_shapes.items()
        }
        self._next_obs = {
            name: np.zeros((self.capacity,) + shape, dtype=np.float32)
            for name, shape in obs_shapes.items()
        }
        self._actions = np.zeros((self.capacity,) + joint_action_shape, dtype=np.float32)
        self._rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self._dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self._failure_tags = np.zeros((self.capacity, 1), dtype=np.float32)
        self._safety_costs = np.zeros((self.capacity, 1), dtype=np.float32)
        self._failure_reasons = np.asarray([""] * self.capacity, dtype=object)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
        failure_tag: bool = False,
        safety_cost: float = 0.0,
        failure_reason: str = "",
    ) -> None:
        for name, array in observation.items():
            self._obs[name][self._index] = np.asarray(array, dtype=np.float32)
        for name, array in next_observation.items():
            self._next_obs[name][self._index] = np.asarray(array, dtype=np.float32)
        self._actions[self._index] = np.asarray(action, dtype=np.float32)
        self._rewards[self._index, 0] = float(reward)
        self._dones[self._index, 0] = float(done)
        self._failure_tags[self._index, 0] = float(bool(failure_tag))
        self._safety_costs[self._index, 0] = max(float(safety_cost), 0.0)
        self._failure_reasons[self._index] = str(failure_reason or "")
        self._index = (self._index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        next_observation: dict[str, np.ndarray],
        done: np.ndarray,
        *,
        failure_tag: np.ndarray | None = None,
        safety_cost: np.ndarray | None = None,
        failure_reason: np.ndarray | list[str] | None = None,
    ) -> None:
        batch_size = int(np.asarray(action).shape[0])
        resolved_failure_tag = np.zeros((batch_size,), dtype=np.float32) if failure_tag is None else np.asarray(failure_tag)
        resolved_safety_cost = np.zeros((batch_size,), dtype=np.float32) if safety_cost is None else np.asarray(safety_cost)
        if failure_reason is None:
            resolved_failure_reason = [""] * batch_size
        else:
            resolved_failure_reason = [str(item) for item in np.asarray(failure_reason, dtype=object).reshape(-1).tolist()]
        for batch_idx in range(batch_size):
            self.add(
                {name: np.asarray(values[batch_idx], dtype=np.float32) for name, values in observation.items()},
                np.asarray(action[batch_idx], dtype=np.float32),
                float(np.asarray(reward[batch_idx]).reshape(())),
                {name: np.asarray(values[batch_idx], dtype=np.float32) for name, values in next_observation.items()},
                bool(np.asarray(done[batch_idx]).reshape(())),
                failure_tag=bool(np.asarray(resolved_failure_tag[batch_idx]).reshape(())),
                safety_cost=float(np.asarray(resolved_safety_cost[batch_idx]).reshape(())),
                failure_reason=resolved_failure_reason[batch_idx],
            )

    def sample(self, batch_size: int, device: torch.device) -> dict[str, object]:
        if self._size < batch_size:
            raise ValueError(f"Not enough samples in replay buffer: {self._size} < {batch_size}")
        indices = self._sample_indices(batch_size)
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
            "failure_mask": torch.as_tensor(self._failure_tags[indices], dtype=torch.float32, device=device),
            "safety_costs": torch.as_tensor(self._safety_costs[indices], dtype=torch.float32, device=device),
        }

    def stats(self) -> dict[str, object]:
        """Return replay diagnostics for training summaries."""

        active_failure_tags = self._failure_tags[: self._size, 0]
        active_reasons = self._failure_reasons[: self._size]
        reason_counts: dict[str, int] = {}
        for reason, tag in zip(active_reasons.tolist(), active_failure_tags.tolist()):
            if float(tag) <= 0.5:
                continue
            resolved_reason = str(reason or "risk")
            reason_counts[resolved_reason] = reason_counts.get(resolved_reason, 0) + 1
        return {
            "size": int(self._size),
            "capacity": int(self.capacity),
            "failure_replay_enabled": bool(self.enable_failure_replay),
            "failure_replay_ratio": float(self.failure_replay_ratio),
            "failure_buffer_size": int(active_failure_tags.sum()),
            "failure_reason_counts": reason_counts,
            "mean_safety_cost": float(self._safety_costs[: self._size].mean()) if self._size else 0.0,
        }

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if not self.enable_failure_replay or self.failure_replay_ratio <= 0.0:
            return self._rng.integers(0, self._size, size=(batch_size,))

        active = np.arange(self._size)
        failure_indices = active[self._failure_tags[: self._size, 0] > 0.5]
        if failure_indices.size == 0:
            return self._rng.integers(0, self._size, size=(batch_size,))

        failure_count = min(
            int(round(batch_size * self.failure_replay_ratio)),
            int(failure_indices.size),
            int(batch_size),
        )
        normal_count = int(batch_size) - failure_count
        sampled_failure = self._rng.choice(failure_indices, size=failure_count, replace=failure_indices.size < failure_count)

        normal_indices = active[self._failure_tags[: self._size, 0] <= 0.5]
        if normal_indices.size == 0:
            normal_indices = active
        sampled_normal = self._rng.choice(normal_indices, size=normal_count, replace=normal_indices.size < normal_count)
        indices = np.concatenate([sampled_failure, sampled_normal]).astype(np.int64, copy=False)
        self._rng.shuffle(indices)
        return indices

