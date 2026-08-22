"""Exact-result cache for repeated static PhysX ray queries.

The P07 route guard asks the same immutable HM3D collision stage about many
identical segments while it screens public candidate routes.  Caching those
queries is a runtime optimisation only: every cache miss is delegated to
PhysX with its original arguments, and cached hits return that exact response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class RaycastClosestQuery(Protocol):
    """Minimum interface provided by Isaac Sim's scene-query object."""

    def raycast_closest(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        distance: float,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RaycastCacheStats:
    """Telemetry proving how much repeated static-world work was avoided."""

    physics_raycast_calls: int
    raycast_cache_hits: int
    cached_unique_rays: int


class MemoizedRaycastClosestQuery:
    """Cache exact ``raycast_closest`` calls against one immutable scene."""

    def __init__(self, query: RaycastClosestQuery) -> None:
        self._query = query
        self._cache: dict[
            tuple[tuple[float, float, float], tuple[float, float, float], float], dict[str, Any]
        ] = {}
        self._physics_raycast_calls = 0
        self._raycast_cache_hits = 0

    @staticmethod
    def _point(point: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(point) != 3:
            raise ValueError("ray origin and direction must have exactly three coordinates")
        return tuple(float(value) for value in point)  # type: ignore[return-value]

    def raycast_closest(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        distance: float,
    ) -> dict[str, Any]:
        key = (self._point(origin), self._point(direction), float(distance))
        if key in self._cache:
            self._raycast_cache_hits += 1
            return dict(self._cache[key])
        response = dict(self._query.raycast_closest(*key))
        self._cache[key] = response
        self._physics_raycast_calls += 1
        return dict(response)

    @property
    def stats(self) -> RaycastCacheStats:
        return RaycastCacheStats(
            physics_raycast_calls=self._physics_raycast_calls,
            raycast_cache_hits=self._raycast_cache_hits,
            cached_unique_rays=len(self._cache),
        )
