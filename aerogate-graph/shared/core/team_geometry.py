"""Small, deterministic geometry metrics shared by multi-agent runtime code.

The functions in this module deliberately operate on plain ``(N, 2)`` arrays.  Keeping
these diagnostics independent from an environment instance makes their safety semantics
easy to test and reuse in evaluation code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TeamSeparationStats:
    """Pairwise separation diagnostics for a team at one simulation instant."""

    collision: bool
    min_distance_m: float
    pair_count: int


def boundary_proximity_deficit_m(
    positions_xy: np.ndarray,
    *,
    world_x_bounds_m: tuple[float, float],
    world_y_bounds_m: tuple[float, float],
    agent_radius_m: float,
    soft_margin_m: float,
) -> float:
    """Return the amount by which the nearest usable boundary clearance misses a margin.

    A result of zero means every agent is at least ``soft_margin_m`` inside the usable
    world bounds.  A positive result also grows outside the bounds, which lets callers
    use the same continuous diagnostic before and after a termination condition.
    """

    positions = _as_positions_xy(positions_xy)
    if positions.size == 0:
        return 0.0

    x_min, x_max = (float(value) for value in world_x_bounds_m)
    y_min, y_max = (float(value) for value in world_y_bounds_m)
    radius_m = float(agent_radius_m)
    margin_m = max(float(soft_margin_m), 0.0)
    min_clearance_m = float("inf")
    for position_xy in positions:
        lower_x_clearance = float(position_xy[0]) - x_min - radius_m
        upper_x_clearance = x_max - float(position_xy[0]) - radius_m
        lower_y_clearance = float(position_xy[1]) - y_min - radius_m
        upper_y_clearance = y_max - float(position_xy[1]) - radius_m
        min_clearance_m = min(
            min_clearance_m,
            lower_x_clearance,
            upper_x_clearance,
            lower_y_clearance,
            upper_y_clearance,
        )
    return max(0.0, margin_m - min_clearance_m)


def pairwise_separation_stats(
    positions_xy: np.ndarray,
    *,
    safe_distance_m: float,
) -> TeamSeparationStats:
    """Measure the closest pair and whether it violates an inclusive safety threshold."""

    positions = _as_positions_xy(positions_xy)
    num_agents = int(positions.shape[0])
    if num_agents <= 1:
        return TeamSeparationStats(collision=False, min_distance_m=float("inf"), pair_count=0)

    min_distance_m = float("inf")
    collision = False
    pair_count = 0
    threshold_m = float(safe_distance_m)
    for first_index in range(num_agents):
        for second_index in range(first_index + 1, num_agents):
            pair_count += 1
            distance_m = math.hypot(
                float(positions[first_index, 0] - positions[second_index, 0]),
                float(positions[first_index, 1] - positions[second_index, 1]),
            )
            min_distance_m = min(min_distance_m, distance_m)
            collision = collision or distance_m <= threshold_m
    return TeamSeparationStats(
        collision=collision,
        min_distance_m=float(min_distance_m),
        pair_count=pair_count,
    )


def count_lateral_bands(local_lateral_positions_m: np.ndarray, *, band_width_m: float) -> int:
    """Count stable lateral groups using the running center of each sorted band."""

    values = np.sort(np.asarray(local_lateral_positions_m, dtype=np.float32).reshape(-1))
    if values.size == 0:
        return 0

    band_count = 1
    band_sum = float(values[0])
    band_size = 1
    band_center = band_sum / band_size
    for value in values[1:]:
        value_m = float(value)
        if abs(value_m - band_center) > float(band_width_m):
            band_count += 1
            band_sum = value_m
            band_size = 1
        else:
            band_sum += value_m
            band_size += 1
        band_center = band_sum / max(band_size, 1)
    return int(band_count)


def slot_error_stats(agent_positions_xy: np.ndarray, desired_slots_xy: np.ndarray) -> tuple[float, float]:
    """Return mean and maximum agent-to-slot distances for one formation instant."""

    if agent_positions_xy.size == 0:
        return (0.0, 0.0)
    deltas = desired_slots_xy - agent_positions_xy
    distances = np.linalg.norm(deltas, axis=1)
    return (float(np.mean(distances)), float(np.max(distances)))


def _as_positions_xy(positions_xy: np.ndarray) -> np.ndarray:
    """Normalize position input while rejecting shapes that hide caller errors."""

    positions = np.asarray(positions_xy, dtype=np.float32)
    if positions.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_xy must have shape (N, 2)")
    return positions
