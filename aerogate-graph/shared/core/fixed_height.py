"""Fixed-height helpers for the 2D drone experiment line."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class FixedHeightConfig:
    """Definition of the invariant flight height used in the 2D experiments."""

    flight_height_m: float = 4.0
    world_origin_z_m: float = 0.0
    tolerance_m: float = 1e-6


def fixed_height_z(config: FixedHeightConfig) -> float:
    """Return the world-space Z coordinate enforced by the config."""

    return float(config.world_origin_z_m + config.flight_height_m)


def planar_xy_to_fixed_xyz(x_m: float, y_m: float, config: FixedHeightConfig) -> Point3D:
    """Promote an XY location into a fixed-height XYZ pose."""

    return (float(x_m), float(y_m), fixed_height_z(config))


def enforce_fixed_height(position_xyz: Point3D, config: FixedHeightConfig) -> Point3D:
    """Project any 3D point onto the configured flight height."""

    return (float(position_xyz[0]), float(position_xyz[1]), fixed_height_z(config))


def is_at_fixed_height(position_xyz: Point3D, config: FixedHeightConfig) -> bool:
    """Check whether a 3D point lies on the configured height plane."""

    return abs(float(position_xyz[2]) - fixed_height_z(config)) <= float(config.tolerance_m)

