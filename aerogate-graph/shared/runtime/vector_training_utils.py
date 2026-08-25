"""Shared helpers for synchronous vectorized aerogate_graph training."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def stack_observations(observations: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-env observations into one batch-first observation dictionary."""

    if not observations:
        raise ValueError("At least one observation is required to stack.")
    names = tuple(observations[0].keys())
    return {
        name: np.stack(
            [np.asarray(observation[name], dtype=np.float32) for observation in observations],
            axis=0,
        ).astype(np.float32, copy=False)
        for name in names
    }


def replace_observation_rows(
    observations: dict[str, np.ndarray],
    indices: Sequence[int] | np.ndarray,
    replacement: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Replace selected batch rows in-place and return the observation batch."""

    resolved_indices = np.asarray(indices, dtype=np.int64)
    if resolved_indices.size == 0:
        return observations
    for name, values in observations.items():
        values[resolved_indices] = np.asarray(replacement[name], dtype=np.float32)
    return observations


def done_indices(done_mask: Sequence[bool] | np.ndarray) -> np.ndarray:
    """Return the indices of finished environments."""

    return np.flatnonzero(np.asarray(done_mask, dtype=bool))


def resolve_seeds(base_seed: int | None, count: int) -> list[int | None]:
    """Expand one optional base seed into deterministic per-env seeds."""

    if base_seed is None:
        return [None] * int(count)
    start = int(base_seed)
    return [start + idx for idx in range(int(count))]


def normalize_optional_int_sequence(
    values: int | Sequence[int] | np.ndarray | None,
    count: int,
) -> list[int | None]:
    """Broadcast or validate one optional integer input across environments."""

    resolved_count = int(count)
    if values is None:
        return [None] * resolved_count
    if isinstance(values, np.ndarray):
        flat = values.reshape(-1).tolist()
        if len(flat) == 1:
            return [int(flat[0])] * resolved_count
        if len(flat) != resolved_count:
            raise ValueError(f"Expected {resolved_count} values, got {len(flat)}")
        return [int(value) for value in flat]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        flat = list(values)
        if len(flat) != resolved_count:
            raise ValueError(f"Expected {resolved_count} values, got {len(flat)}")
        return [int(value) for value in flat]
    return [int(values)] * resolved_count


def resolve_updates_per_collect(num_envs: int, updates_per_step: int) -> int:
    """Convert per-transition update intent into per-collect repeats."""

    return max(int(num_envs), 1) * max(int(updates_per_step), 0)


def should_checkpoint_now(
    *,
    transitions_collected: int,
    next_checkpoint_transition: int | None,
) -> bool:
    """Return whether the current collector pass crossed a checkpoint boundary."""

    return bool(next_checkpoint_transition is not None and transitions_collected >= next_checkpoint_transition)

