"""Axis-aligned bounding-box primitives and flight/command volumes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _nonnegative(value: Any, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if number < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _vec3(value: Sequence[Any], *, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three coordinates")
    return tuple(
        _finite_number(component, label=f"{label}[{axis}]")
        for axis, component in enumerate(value)
    )  # type: ignore[return-value]

@dataclass(frozen=True)
class AABB:
    """A world-frame axis-aligned box used by route and LiDAR contracts."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    source_prim: str = ""
    category: str = ""

    def __post_init__(self) -> None:
        minimum = _vec3(self.minimum, label="AABB.minimum")
        maximum = _vec3(self.maximum, label="AABB.maximum")
        if any(minimum[axis] > maximum[axis] for axis in range(3)):
            raise ValueError("AABB minimum must not exceed maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "source_prim", str(self.source_prim))
        object.__setattr__(self, "category", str(self.category))

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
        }
        if self.source_prim:
            payload["source_prim"] = self.source_prim
        if self.category:
            payload["category"] = self.category
        return payload

    def expanded(self, clearance_m: float) -> AABB:
        clearance = _nonnegative(clearance_m, label="clearance_m")
        return AABB(
            tuple(value - clearance for value in self.minimum),
            tuple(value + clearance for value in self.maximum),
            source_prim=self.source_prim,
            category=self.category,
        )

    def contains(self, point: Sequence[Any], *, margin_m: float = 0.0) -> bool:
        """Return whether a point is inside after shrinking by ``margin_m``."""

        xyz = _vec3(point, label="point")
        margin = _nonnegative(margin_m, label="margin_m")
        return all(
            self.minimum[axis] + margin
            <= xyz[axis]
            <= self.maximum[axis] - margin
            for axis in range(3)
        )


def coerce_aabb(value: AABB | Mapping[str, Any]) -> AABB:
    if isinstance(value, AABB):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("AABB must be an AABB or object")
    minimum = value.get("minimum", value.get("min"))
    maximum = value.get("maximum", value.get("max"))
    if not isinstance(minimum, Sequence) or not isinstance(maximum, Sequence):
        raise ValueError("AABB object requires minimum/maximum coordinates")
    return AABB(
        _vec3(minimum, label="AABB.minimum"),
        _vec3(maximum, label="AABB.maximum"),
        source_prim=str(value.get("source_prim", value.get("path", ""))),
        category=str(value.get("category", value.get("kind", ""))),
    )

# The scoring volume is broader than the physical and commanded center-point
# envelopes. All coordinates are measured world-frame metres with +Z up.
FORMAL_SCORING_VOLUME_W_M = AABB((-46.0, -48.0, 0.0), (46.0, 44.0, 19.0))
# This is the unshrunk physical-envelope volume, not the commanded-center
# volume below. Its lower boundary must contain the upstream runtime starts
# after both the 0.08 m CF2X body radius and the permitted post-reset settling
# tolerance are applied: min(start_z) - 0.08 - 0.05 = 8.927 m.
CITY_LITE_FLIGHT_VOLUME_W_M = AABB((-46.0, -48.0, 8.9), (46.0, 44.0, 15.0))
CITY_LITE_COMMAND_VOLUME_W_M = AABB(
    (-46.0, -48.0, 9.0),
    (46.0, 44.0, 14.25),
)


# Target-free safe spawn anchors (public geometry only, no target/evaluator/seed info).
TARGET_FREE_SAFE_STARTS_W_M: tuple[tuple[float, float, float], ...] = (
    (-40.0, -12.0, 9.081),
    (-4.0, -32.0, 9.847),
    (0.0, -42.0, 10.024),
    (40.0, -12.0, 9.336),
    (-40.0, 38.0, 9.157),
    (-10.0, -2.0, 9.177),
    (0.0, 38.0, 9.057),
    (40.0, -2.0, 9.362),
)
