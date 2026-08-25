"""Shared deterministic public range-ray patterns.

Every ranked method must use the same sensor entitlement.  The 26-ray pattern
retains the public range-outcome semantics (no camera pixels and no evaluator
truth) while giving the shared free-space belief enough corridor connectivity to
navigate through rooms and toward frontiers.
"""

from __future__ import annotations

from collections.abc import Sequence

Point3 = tuple[float, float, float]

LEGACY_SIX_AXIS_PATTERN = "six-axis-range-rays"
DENSE_26_RAY_PATTERN = "dense-public-range-grid-26"

_SIX_AXIS_DIRECTIONS: tuple[Point3, ...] = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)

_HALF_SQRT2 = 0.7071067811865476
_THIRD_SQRT3 = 0.5773502691896258

_PLANAR_DIAGONAL_DIRECTIONS: tuple[Point3, ...] = (
    (_HALF_SQRT2, _HALF_SQRT2, 0.0),
    (_HALF_SQRT2, -_HALF_SQRT2, 0.0),
    (-_HALF_SQRT2, _HALF_SQRT2, 0.0),
    (-_HALF_SQRT2, -_HALF_SQRT2, 0.0),
    (_HALF_SQRT2, 0.0, _HALF_SQRT2),
    (_HALF_SQRT2, 0.0, -_HALF_SQRT2),
    (-_HALF_SQRT2, 0.0, _HALF_SQRT2),
    (-_HALF_SQRT2, 0.0, -_HALF_SQRT2),
    (0.0, _HALF_SQRT2, _HALF_SQRT2),
    (0.0, _HALF_SQRT2, -_HALF_SQRT2),
    (0.0, -_HALF_SQRT2, _HALF_SQRT2),
    (0.0, -_HALF_SQRT2, -_HALF_SQRT2),
)

_SPATIAL_DIAGONAL_DIRECTIONS: tuple[Point3, ...] = (
    (_THIRD_SQRT3, _THIRD_SQRT3, _THIRD_SQRT3),
    (_THIRD_SQRT3, _THIRD_SQRT3, -_THIRD_SQRT3),
    (_THIRD_SQRT3, -_THIRD_SQRT3, _THIRD_SQRT3),
    (_THIRD_SQRT3, -_THIRD_SQRT3, -_THIRD_SQRT3),
    (-_THIRD_SQRT3, _THIRD_SQRT3, _THIRD_SQRT3),
    (-_THIRD_SQRT3, _THIRD_SQRT3, -_THIRD_SQRT3),
    (-_THIRD_SQRT3, -_THIRD_SQRT3, _THIRD_SQRT3),
    (-_THIRD_SQRT3, -_THIRD_SQRT3, -_THIRD_SQRT3),
)

_DENSE_26_DIRECTIONS: tuple[Point3, ...] = (
    _SIX_AXIS_DIRECTIONS
    + _PLANAR_DIAGONAL_DIRECTIONS
    + _SPATIAL_DIAGONAL_DIRECTIONS
)

_PATTERNS: dict[str, tuple[Point3, ...]] = {
    LEGACY_SIX_AXIS_PATTERN: _SIX_AXIS_DIRECTIONS,
    DENSE_26_RAY_PATTERN: _DENSE_26_DIRECTIONS,
}


def public_range_direction_count(pattern: str) -> int:
    """Return the ray count encoded by a public range pattern."""

    if pattern not in _PATTERNS:
        raise ValueError(f"unsupported public range-ray pattern: {pattern}")
    return len(_PATTERNS[pattern])


def resolve_public_range_directions(pattern: str) -> tuple[Point3, ...]:
    """Return the deterministic unit directions for a public range pattern."""

    if pattern not in _PATTERNS:
        raise ValueError(f"unsupported public range-ray pattern: {pattern}")
    return _PATTERNS[pattern]


def validate_public_range_directions(directions: Sequence[Sequence[float]]) -> None:
    """Validate a contract-declared ray-direction list."""

    if not isinstance(directions, Sequence) or isinstance(directions, (str, bytes)):
        raise ValueError("ray_directions must be a list")
    resolved = tuple(directions)
    if not resolved:
        raise ValueError("ray_directions must not be empty")
    canonical = tuple(tuple(float(value) for value in raw) for raw in resolved)
    if len(canonical) != len(set(canonical)):
        raise ValueError("ray_directions must be unique")
    for index, raw in enumerate(resolved):
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
            raise ValueError(f"ray_directions[{index}] must be a length-3 vector")
        values = canonical[index]
        norm_squared = sum(value * value for value in values)
        if not 0.999 < norm_squared < 1.001:
            raise ValueError(f"ray_directions[{index}] must be a unit vector")


__all__ = [
    "DENSE_26_RAY_PATTERN",
    "LEGACY_SIX_AXIS_PATTERN",
    "public_range_direction_count",
    "resolve_public_range_directions",
    "validate_public_range_directions",
]
