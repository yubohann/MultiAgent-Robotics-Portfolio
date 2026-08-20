"""Pure validation helpers for Livox CustomMsg point fields."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol


class LivoxPointLike(Protocol):
    """The subset of CustomPoint fields checked before estimation."""

    offset_time: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class PointPacketStatistics:
    point_count: int
    finite_point_count: int
    non_finite_point_count: int
    offsets_monotonic: bool
    offset_span_ns: int


def inspect_custom_points(points: Iterable[LivoxPointLike]) -> PointPacketStatistics:
    """Inspect packet-local point validity without changing the sensor message."""

    point_count = 0
    finite_point_count = 0
    non_finite_point_count = 0
    offsets_monotonic = True
    previous_offset: int | None = None
    last_offset = 0

    for point in points:
        point_count += 1
        offset = int(point.offset_time)
        if previous_offset is not None and offset < previous_offset:
            offsets_monotonic = False
        previous_offset = offset
        last_offset = offset

        if all(math.isfinite(value) for value in (point.x, point.y, point.z)):
            finite_point_count += 1
        else:
            non_finite_point_count += 1

    return PointPacketStatistics(
        point_count=point_count,
        finite_point_count=finite_point_count,
        non_finite_point_count=non_finite_point_count,
        offsets_monotonic=offsets_monotonic,
        offset_span_ns=max(0, last_offset),
    )
