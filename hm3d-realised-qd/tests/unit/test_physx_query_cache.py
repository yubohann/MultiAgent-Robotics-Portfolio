from __future__ import annotations

from aerocity_method.runtime.physx_query_cache import MemoizedRaycastClosestQuery


class _StaticQuery:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, float, float], tuple[float, float, float], float]] = []

    def raycast_closest(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        distance: float,
    ) -> dict[str, object]:
        self.calls.append((origin, direction, distance))
        return {"hit": False, "distance": distance}


def test_memoized_raycast_preserves_arguments_and_caches_only_exact_repeats() -> None:
    raw = _StaticQuery()
    query = MemoizedRaycastClosestQuery(raw)

    first = query.raycast_closest((1, 2, 3), (0, 0, 1), 4.0)
    first["hit"] = True
    second = query.raycast_closest((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 4.0)
    third = query.raycast_closest((1.0, 2.0, 3.0), (0.0, 1.0, 0.0), 4.0)

    assert raw.calls == [
        ((1.0, 2.0, 3.0), (0.0, 0.0, 1.0), 4.0),
        ((1.0, 2.0, 3.0), (0.0, 1.0, 0.0), 4.0),
    ]
    assert second == {"hit": False, "distance": 4.0}
    assert third == {"hit": False, "distance": 4.0}
    assert query.stats.physics_raycast_calls == 2
    assert query.stats.raycast_cache_hits == 1
    assert query.stats.cached_unique_rays == 2


def test_memoized_raycast_caches_an_empty_physx_response() -> None:
    class _EmptyQuery:
        def __init__(self) -> None:
            self.calls = 0

        def raycast_closest(
            self,
            origin: tuple[float, float, float],
            direction: tuple[float, float, float],
            distance: float,
        ) -> dict[str, object]:
            del origin, direction, distance
            self.calls += 1
            return {}

    raw = _EmptyQuery()
    query = MemoizedRaycastClosestQuery(raw)
    assert query.raycast_closest((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0) == {}
    assert query.raycast_closest((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0) == {}
    assert raw.calls == 1
    assert query.stats.raycast_cache_hits == 1
