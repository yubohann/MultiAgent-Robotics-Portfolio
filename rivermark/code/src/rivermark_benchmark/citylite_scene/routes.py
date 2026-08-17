"""Public-route families, segment geometry checks, and route contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .aabb import (
    AABB,
    CITY_LITE_COMMAND_VOLUME_W_M,
    TARGET_FREE_SAFE_STARTS_W_M,
    _nonnegative,
    _vec3,
    coerce_aabb,
)
from .constants import (
    _EPSILON,
    AGENT_COUNT,
    ENVIRONMENT_ID,
    MINIMUM_ROUTE_WAYPOINTS,
    ROUTE_CLEARANCE_M,
    ROUTE_CONDITIONING,
    ROUTE_CONTRACT_SCHEMA,
    ROUTE_GENERATION,
)
from .scene import CityLiteRouteError, aabb_geometry_sha256, canonical_payload_sha256

# These routes are target-free and were selected against the City-Lite static
# structural/task-obstacle AABB contract.  Their first waypoint is also the
# spawn anchor, so a capture cannot silently translate a small gray-box route
# into this scene.
PUBLIC_ROUTES_W_M: tuple[tuple[tuple[float, float, float], ...], ...] = (
    ((-40.0, -12.0, 9.081), (-40.0, -12.0, 11.0), (-40.0, -2.0, 11.0), (-40.0, 8.0, 11.0), (-30.0, 8.0, 11.0), (-30.0, 18.0, 11.0)),
    ((-4.0, -32.0, 9.847), (-4.0, -32.0, 11.0), (6.0, -32.0, 11.0), (16.0, -32.0, 11.0), (16.0, -22.0, 11.0), (16.0, -12.0, 11.0)),
    ((0.0, -42.0, 10.024), (0.0, -42.0, 13.0), (0.0, -32.0, 13.0), (10.0, -32.0, 13.0), (20.0, -32.0, 13.0), (20.0, -22.0, 13.0)),
    ((40.0, -12.0, 9.336), (40.0, -12.0, 11.0), (40.0, -22.0, 11.0), (30.0, -22.0, 11.0), (30.0, -32.0, 11.0), (40.0, -32.0, 11.0)),
    ((-40.0, 38.0, 9.157), (-40.0, 38.0, 13.0), (-40.0, 28.0, 13.0), (-30.0, 28.0, 13.0), (-30.0, 38.0, 13.0), (-20.0, 38.0, 13.0)),
    ((-10.0, -2.0, 9.177), (-10.0, -2.0, 11.0), (-10.0, 8.0, 11.0), (0.0, 8.0, 11.0), (0.0, 18.0, 11.0), (10.0, 18.0, 11.0)),
    ((0.0, 38.0, 9.057), (0.0, 38.0, 11.0), (0.0, 28.0, 11.0), (10.0, 28.0, 11.0), (20.0, 28.0, 11.0), (20.0, 38.0, 11.0)),
    ((40.0, -2.0, 9.362), (40.0, -2.0, 13.0), (40.0, 8.0, 13.0), (30.0, 8.0, 13.0), (30.0, 18.0, 13.0), (40.0, 18.0, 13.0)),
)

# The pilot protocol uses two physically different route families in the same
# City-Lite layout.  Family A remains byte-for-byte compatible with the
# original public route.  Family B was constructed without target/evaluator
# input and validated against the same native 13-AABB City-Lite geometry used
# by the latest clean capture.  A native acceptance run is still required
# before family B can become episode evidence.
CITY_LITE_ROUTE_FAMILY_A_ID = "citylite-route-family-a-v1"
CITY_LITE_ROUTE_FAMILY_B_ID = "citylite-route-family-b-v1"
CITY_LITE_START_ANCHOR_A_ID = "citylite-start-anchor-a-v1"
CITY_LITE_START_ANCHOR_B_ID = "citylite-start-anchor-b-v1"
CITY_LITE_TARGET_REGION_A_ID = "citylite-target-region-a-v1"
CITY_LITE_TARGET_REGION_B_ID = "citylite-target-region-b-v1"

PUBLIC_ROUTES_B_W_M: tuple[tuple[tuple[float, float, float], ...], ...] = (
    ((40.0, 12.0, 9.081), (40.0, 12.0, 11.0), (40.0, 2.0, 11.0), (40.0, -8.0, 11.0), (30.0, -8.0, 11.0), (30.0, -18.0, 11.0)),
    ((4.0, 32.0, 9.847), (4.0, 32.0, 11.0), (-6.0, 32.0, 11.0), (-16.0, 32.0, 11.0), (-16.0, 22.0, 11.0), (-16.0, 12.0, 11.0)),
    ((0.0, 42.0, 10.024), (0.0, 42.0, 13.0), (0.0, 32.0, 13.0), (-10.0, 32.0, 13.0), (-20.0, 32.0, 13.0), (-20.0, 22.0, 13.0)),
    ((-40.0, 12.0, 9.336), (-40.0, 12.0, 11.0), (-40.0, 22.0, 11.0), (-30.0, 22.0, 11.0), (-30.0, 32.0, 11.0), (-40.0, 32.0, 11.0)),
    ((40.0, -38.0, 9.157), (40.0, -38.0, 13.0), (40.0, -28.0, 13.0), (30.0, -28.0, 13.0), (30.0, -38.0, 13.0), (20.0, -38.0, 13.0)),
    ((10.0, 2.0, 9.177), (10.0, 2.0, 11.0), (10.0, -8.0, 11.0), (0.0, -8.0, 11.0), (0.0, -18.0, 11.0), (-10.0, -18.0, 11.0)),
    ((0.0, -38.0, 9.057), (0.0, -38.0, 11.0), (0.0, -28.0, 11.0), (10.0, -28.0, 11.0), (10.0, -18.0, 11.0), (20.0, -18.0, 11.0)),
    ((-40.0, 2.0, 9.362), (-40.0, 2.0, 13.0), (-30.0, 2.0, 13.0), (-20.0, 2.0, 13.0), (-10.0, 2.0, 13.0), (0.0, 2.0, 13.0)),
)

PUBLIC_ROUTE_FAMILIES_W_M = {
    CITY_LITE_ROUTE_FAMILY_A_ID: PUBLIC_ROUTES_W_M,
    CITY_LITE_ROUTE_FAMILY_B_ID: PUBLIC_ROUTES_B_W_M,
}
START_ANCHOR_IDS_BY_ROUTE_FAMILY = {
    CITY_LITE_ROUTE_FAMILY_A_ID: CITY_LITE_START_ANCHOR_A_ID,
    CITY_LITE_ROUTE_FAMILY_B_ID: CITY_LITE_START_ANCHOR_B_ID,
}
TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M = {
    route_family_id: tuple(route[0] for route in routes)
    for route_family_id, routes in PUBLIC_ROUTE_FAMILIES_W_M.items()
}
TARGET_REGIONS_W_M = {
    CITY_LITE_TARGET_REGION_A_ID: AABB((-46.0, -48.0, 9.0), (-2.0, 44.0, 14.25)),
    CITY_LITE_TARGET_REGION_B_ID: AABB((2.0, -48.0, 9.0), (46.0, 44.0, 14.25)),
}

def resolve_public_route_family(
    route_family_id: str,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """Return one frozen public route family or fail closed."""

    try:
        return PUBLIC_ROUTE_FAMILIES_W_M[route_family_id]
    except KeyError as exc:
        raise CityLiteRouteError(f"unknown City-Lite route family: {route_family_id}") from exc

@dataclass(frozen=True)
class RouteValidationReport:
    agent_count: int
    waypoint_count_per_agent: int
    segment_count: int
    clearance_m: float
    aabb_count: int
    aabb_geometry_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_count": self.agent_count,
            "waypoint_count_per_agent": self.waypoint_count_per_agent,
            "segment_count": self.segment_count,
            "clearance_m": self.clearance_m,
            "aabb_count": self.aabb_count,
            "aabb_geometry_sha256": self.aabb_geometry_sha256,
        }

def segment_intersects_aabb(
    start: Sequence[Any],
    end: Sequence[Any],
    aabb: AABB | Mapping[str, Any],
    *,
    clearance_m: float = 0.0,
) -> bool:
    """Conservative slab test; touching the expanded AABB is an intersection."""

    first = _vec3(start, label="segment start")
    second = _vec3(end, label="segment end")
    box = coerce_aabb(aabb).expanded(clearance_m)
    low, high = 0.0, 1.0
    for axis in range(3):
        delta = second[axis] - first[axis]
        if abs(delta) <= _EPSILON:
            if not box.minimum[axis] <= first[axis] <= box.maximum[axis]:
                return False
            continue
        left = (box.minimum[axis] - first[axis]) / delta
        right = (box.maximum[axis] - first[axis]) / delta
        low = max(low, min(left, right))
        high = min(high, max(left, right))
        if low > high + _EPSILON:
            return False
    return True


def segment_has_clearance(
    start: Sequence[Any],
    end: Sequence[Any],
    aabbs: Sequence[AABB | Mapping[str, Any]],
    *,
    clearance_m: float = ROUTE_CLEARANCE_M,
) -> bool:
    clearance = _nonnegative(clearance_m, label="clearance_m")
    return not any(
        segment_intersects_aabb(start, end, value, clearance_m=clearance)
        for value in aabbs
    )

def validate_public_routes(
    routes_w_m: Sequence[Sequence[Sequence[Any]]],
    obstacle_aabbs: Sequence[AABB | Mapping[str, Any]],
    *,
    clearance_m: float = ROUTE_CLEARANCE_M,
    expected_starts_w_m: Sequence[Sequence[Any]] = TARGET_FREE_SAFE_STARTS_W_M,
) -> RouteValidationReport:
    """Validate eight target-free routes against one shared static AABB set."""

    if isinstance(routes_w_m, (str, bytes)) or len(routes_w_m) != AGENT_COUNT:
        raise CityLiteRouteError(f"routes must contain exactly {AGENT_COUNT} agents")
    if not obstacle_aabbs:
        raise CityLiteRouteError(
            "route validation requires nonempty structural/task-obstacle AABBs"
        )
    try:
        boxes = tuple(coerce_aabb(value) for value in obstacle_aabbs)
        clearance = _nonnegative(clearance_m, label="clearance_m")
    except ValueError as exc:
        raise CityLiteRouteError(str(exc)) from exc
    if not math.isclose(clearance, ROUTE_CLEARANCE_M, rel_tol=0.0, abs_tol=_EPSILON):
        raise CityLiteRouteError(
            f"route clearance must remain frozen at {ROUTE_CLEARANCE_M} m"
        )

    if isinstance(expected_starts_w_m, (str, bytes)) or len(expected_starts_w_m) != AGENT_COUNT:
        raise CityLiteRouteError(f"expected starts must contain exactly {AGENT_COUNT} agents")
    try:
        expected_starts = tuple(
            _vec3(point, label=f"expected_starts[{agent_id}]")
            for agent_id, point in enumerate(expected_starts_w_m)
        )
    except ValueError as exc:
        raise CityLiteRouteError(str(exc)) from exc

    normalized: list[tuple[tuple[float, float, float], ...]] = []
    waypoint_count: int | None = None
    for agent_id, route in enumerate(routes_w_m):
        if isinstance(route, (str, bytes)) or len(route) < MINIMUM_ROUTE_WAYPOINTS:
            raise CityLiteRouteError(
                f"agent {agent_id} route requires at least {MINIMUM_ROUTE_WAYPOINTS} waypoints"
            )
        try:
            points = tuple(
                _vec3(point, label=f"routes[{agent_id}][{waypoint_id}]")
                for waypoint_id, point in enumerate(route)
            )
        except ValueError as exc:
            raise CityLiteRouteError(str(exc)) from exc
        if waypoint_count is None:
            waypoint_count = len(points)
        elif len(points) != waypoint_count:
            raise CityLiteRouteError("all routes must have the same waypoint count")
        expected_start = expected_starts[agent_id]
        if any(
            not math.isclose(points[0][axis], expected_start[axis], rel_tol=0.0, abs_tol=1.0e-6)
            for axis in range(3)
        ):
            raise CityLiteRouteError(
                f"agent {agent_id} route does not start at its target-free safe anchor"
            )
        for waypoint_id, point in enumerate(points):
            if not CITY_LITE_COMMAND_VOLUME_W_M.contains(point):
                raise CityLiteRouteError(
                    f"agent {agent_id} waypoint {waypoint_id} is outside the command volume"
                )
        for segment_id, (start, end) in enumerate(zip(points, points[1:])):
            if math.dist(start, end) <= _EPSILON:
                raise CityLiteRouteError(
                    f"agent {agent_id} segment {segment_id} has zero length"
                )
            for box_id, box in enumerate(boxes):
                if segment_intersects_aabb(
                    start,
                    end,
                    box,
                    clearance_m=clearance,
                ):
                    source = box.source_prim or f"AABB[{box_id}]"
                    raise CityLiteRouteError(
                        f"agent {agent_id} segment {segment_id} violates "
                        f"{clearance} m clearance from {source}"
                    )
        normalized.append(points)

    assert waypoint_count is not None
    return RouteValidationReport(
        agent_count=AGENT_COUNT,
        waypoint_count_per_agent=waypoint_count,
        segment_count=AGENT_COUNT * (waypoint_count - 1),
        clearance_m=clearance,
        aabb_count=len(boxes),
        aabb_geometry_sha256=aabb_geometry_sha256(boxes),
    )

def make_public_route_contract(
    obstacle_aabbs: Sequence[AABB | Mapping[str, Any]],
    *,
    route_family_id: str | None = None,
    routes_w_m: Sequence[Sequence[Sequence[Any]]] | None = None,
) -> dict[str, Any]:
    if not obstacle_aabbs:
        raise CityLiteRouteError("route contract requires nonempty AABB geometry")
    boxes = tuple(coerce_aabb(value) for value in obstacle_aabbs)
    selected_routes = (
        resolve_public_route_family(route_family_id)
        if route_family_id is not None
        else PUBLIC_ROUTES_W_M
    )
    if routes_w_m is not None and canonical_payload_sha256(routes_w_m) != canonical_payload_sha256(selected_routes):
        raise CityLiteRouteError("route contract routes do not match the selected route family")
    selected_starts = tuple(route[0] for route in selected_routes)
    payload = {
        "schema": ROUTE_CONTRACT_SCHEMA,
        "environment_id": ENVIRONMENT_ID,
        "route_generation": ROUTE_GENERATION,
        "route_conditioning": ROUTE_CONDITIONING,
        "agent_count": AGENT_COUNT,
        "target_or_evaluator_consumed": False,
        "legacy_mission_route_consumed": False,
        "clearance_m": ROUTE_CLEARANCE_M,
        "command_volume_w_m": CITY_LITE_COMMAND_VOLUME_W_M.as_dict(),
        "target_free_safe_starts_sha256": canonical_payload_sha256(
            selected_starts
        ),
        "aabb_count": len(boxes),
        "aabb_geometry_sha256": aabb_geometry_sha256(boxes),
    }
    if route_family_id is not None:
        payload.update(
            {
                "route_family_id": route_family_id,
                "routes_sha256": canonical_payload_sha256(selected_routes),
                "start_anchor_id": START_ANCHOR_IDS_BY_ROUTE_FAMILY[route_family_id],
            }
        )
    return payload

def _forbidden_route_metadata_keys(value: Any) -> list[str]:
    forbidden = {
        "evaluator_seed",
        "evaluator_truth",
        "legacy_route_points",
        "mission_route",
        "reference_route",
        "target",
        "target_positions",
        "target_positions_w_m",
        "target_xyz",
        "target_xyz_m",
        "targets",
    }
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in forbidden:
                found.append(str(key))
            found.extend(_forbidden_route_metadata_keys(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            found.extend(_forbidden_route_metadata_keys(nested))
    return found

def validate_public_route_contract(
    contract: Mapping[str, Any],
    routes_w_m: Sequence[Sequence[Sequence[Any]]],
    obstacle_aabbs: Sequence[AABB | Mapping[str, Any]],
) -> RouteValidationReport:
    """Validate metadata plus geometry without accepting target/evaluator input."""

    if not isinstance(contract, Mapping):
        raise CityLiteRouteError("public route contract must be an object")
    route_family_id = contract.get("route_family_id")
    if route_family_id is None:
        selected_routes = PUBLIC_ROUTES_W_M
        selected_starts = TARGET_FREE_SAFE_STARTS_W_M
    elif isinstance(route_family_id, str):
        selected_routes = resolve_public_route_family(route_family_id)
        selected_starts = TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M[route_family_id]
        if contract.get("routes_sha256") != canonical_payload_sha256(selected_routes):
            raise CityLiteRouteError("public route contract route-family hash is stale")
        if contract.get("start_anchor_id") != START_ANCHOR_IDS_BY_ROUTE_FAMILY[route_family_id]:
            raise CityLiteRouteError("public route contract start anchor is stale")
        if canonical_payload_sha256(routes_w_m) != canonical_payload_sha256(selected_routes):
            raise CityLiteRouteError("public routes do not match their declared route family")
    else:
        raise CityLiteRouteError("public route contract route_family_id must be a string")
    expected_fields = {
        "schema": ROUTE_CONTRACT_SCHEMA,
        "environment_id": ENVIRONMENT_ID,
        "route_generation": ROUTE_GENERATION,
        "route_conditioning": ROUTE_CONDITIONING,
        "agent_count": AGENT_COUNT,
        "target_or_evaluator_consumed": False,
        "legacy_mission_route_consumed": False,
        "clearance_m": ROUTE_CLEARANCE_M,
        "command_volume_w_m": CITY_LITE_COMMAND_VOLUME_W_M.as_dict(),
        "target_free_safe_starts_sha256": canonical_payload_sha256(
            selected_starts
        ),
    }
    for key, expected in expected_fields.items():
        if contract.get(key) != expected:
            raise CityLiteRouteError(f"invalid public route contract field: {key}")
    forbidden_keys = _forbidden_route_metadata_keys(contract)
    if forbidden_keys:
        raise CityLiteRouteError(
            "public route contract contains target/evaluator/legacy route keys: "
            + ", ".join(sorted(set(forbidden_keys)))
        )

    if not obstacle_aabbs:
        raise CityLiteRouteError("public route contract requires nonempty AABB geometry")
    boxes = tuple(coerce_aabb(value) for value in obstacle_aabbs)
    if contract.get("aabb_count") != len(boxes):
        raise CityLiteRouteError("public route contract AABB count is stale")
    geometry_sha256 = aabb_geometry_sha256(boxes)
    if contract.get("aabb_geometry_sha256") != geometry_sha256:
        raise CityLiteRouteError("public route contract AABB geometry hash is stale")
    return validate_public_routes(
        routes_w_m,
        boxes,
        clearance_m=ROUTE_CLEARANCE_M,
        expected_starts_w_m=selected_starts,
    )
