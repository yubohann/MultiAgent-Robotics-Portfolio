"""Small 2D math helpers used by the isolated experiment line."""

from __future__ import annotations

import math
from typing import Tuple


Point2D = Tuple[float, float]
Vector2D = Tuple[float, float]


def vector_norm(vec: Vector2D) -> float:
    """Return the Euclidean norm of a 2D vector."""

    return math.hypot(vec[0], vec[1])


def clip_vector_norm(vec: Vector2D, max_norm: float) -> Vector2D:
    """Clip a 2D vector magnitude while preserving direction."""

    max_norm = max(0.0, float(max_norm))
    norm = vector_norm(vec)
    if norm <= max_norm or norm <= 1e-12:
        return (float(vec[0]), float(vec[1]))
    scale = max_norm / norm
    return (float(vec[0]) * scale, float(vec[1]) * scale)


def yaw_from_velocity(velocity: Vector2D, fallback_yaw_rad: float = 0.0) -> float:
    """Convert a planar velocity vector into a heading angle."""

    if vector_norm(velocity) <= 1e-12:
        return float(fallback_yaw_rad)
    return math.atan2(float(velocity[1]), float(velocity[0]))


def subtract_points(a: Point2D, b: Point2D) -> Vector2D:
    """Return vector a - b in the XY plane."""

    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

