"""Run an external OR-Tools public-atlas inspection-routing baseline.

The process accepts only the benchmark JSONL public projection.  OR-Tools is
used for a bounded, sector-constrained routing problem; all physics and private
target confirmation remain in the parent benchmark executor.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "org.aerocity.bench.external-planner-request.v1"
RESPONSE_SCHEMA = "org.aerocity.bench.external-planner-response.v1"
UPSTREAM_URL = "https://github.com/google/or-tools.git"
UPSTREAM_COMMIT = "98c165af62df62b3056c2ee0fca66b24e79097cb"
UPSTREAM_LICENSE = "Apache-2.0"
ORTOOLS_VERSION = "9.15.6755"
_FORBIDDEN_FRAGMENTS = (
    "target",
    "support",
    "witness",
    "evaluator",
    "private",
    "split_label",
)


def _reject_non_public(value: object, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str) or not key.isascii():
                raise ValueError(f"public input has a non-ASCII key at {path}")
            normalized = key.lower().replace("-", "_")
            if normalized in {
                "target_count_public",
                "target_process_public",
                "formal_split_label_public",
            } and nested is False:
                continue
            if any(fragment in normalized for fragment in _FORBIDDEN_FRAGMENTS):
                raise ValueError(f"public input contains forbidden field at {path}.{key}")
            _reject_non_public(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_non_public(nested, f"{path}[{index}]")


def _verify_upstream_source(path: Path) -> None:
    """Bind an optional process launch to the exact locked OR-Tools checkout."""

    if not path.is_dir():
        raise ValueError("OR-Tools upstream source directory is unavailable")
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("OR-Tools upstream source is not a readable Git checkout") from exc
    if head != UPSTREAM_COMMIT:
        raise ValueError("OR-Tools upstream Git revision differs from the source lock")
    if dirty:
        raise ValueError("OR-Tools upstream source checkout must be clean")


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _pose(
    position: tuple[float, float, float], yaw_deg: float, pitch_deg: float = 0.0
) -> dict[str, Any]:
    return {
        "position": [round(value, 6) for value in position],
        "yaw_deg": round(yaw_deg, 6),
        "pitch_deg": round(pitch_deg, 6),
        "roll_deg": 0.0,
    }


@dataclass(frozen=True)
class PublicAABB:
    """One coarse public building envelope used only for route safety."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


def _public_coarse_colliders(public_task_spec: dict[str, Any]) -> tuple[PublicAABB, ...]:
    prior = public_task_spec.get("coarse_prior")
    if not isinstance(prior, dict):
        return ()
    buildings = prior.get("buildings", [])
    if not isinstance(buildings, list):
        raise ValueError("public coarse prior buildings are invalid")
    colliders: list[PublicAABB] = []
    for index, building in enumerate(buildings):
        if not isinstance(building, dict):
            raise ValueError("public coarse-prior building is invalid")
        center_xy = building.get("center_xy")
        size_xy = building.get("size_xy")
        height_m = building.get("height_m")
        if (
            not isinstance(center_xy, list)
            or not isinstance(size_xy, list)
            or len(center_xy) != 2
            or len(size_xy) != 2
        ):
            raise ValueError(f"public coarse-prior building {index} lacks XY bounds")
        try:
            center_x, center_y = (float(value) for value in center_xy)
            size_x, size_y = (float(value) for value in size_xy)
            height = float(height_m)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"public coarse-prior building {index} is non-numeric") from exc
        values = (center_x, center_y, size_x, size_y, height)
        if not all(math.isfinite(value) for value in values) or min(size_x, size_y, height) <= 0.0:
            raise ValueError(f"public coarse-prior building {index} has invalid bounds")
        colliders.append(
            PublicAABB(
                minimum=(center_x - size_x / 2.0, center_y - size_y / 2.0, 0.0),
                maximum=(center_x + size_x / 2.0, center_y + size_y / 2.0, height),
            )
        )
    return tuple(colliders)


def _segment_intersects_expanded_box(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    box: PublicAABB,
    margin_m: float,
) -> bool:
    """Conservative slab test against one public building envelope."""

    if not math.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError("direct scan margin is invalid")
    lower, upper = 0.0, 1.0
    for origin, destination, minimum, maximum in zip(
        start, end, box.minimum, box.maximum, strict=True
    ):
        delta = destination - origin
        minimum -= margin_m
        maximum += margin_m
        if abs(delta) <= 1.0e-12:
            if origin < minimum or origin > maximum:
                return False
            continue
        first = (minimum - origin) / delta
        second = (maximum - origin) / delta
        if first > second:
            first, second = second, first
        lower = max(lower, first)
        upper = min(upper, second)
        if lower > upper:
            return False
    return True


@dataclass(frozen=True)
class PublicCell:
    cell_id: str
    position: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    represented_area_m2: float
    # A mission-sector capacity certificate permits direct scan motion inside
    # one public inspection region only when the public coarse geometry also
    # proves that the segment clears every building envelope.
    region_id: str = ""


@dataclass
class RouteState:
    ordered_cell_ids: list[str]
    cell_id: str | None = None
    phase: str = "select"
    observe_ticks_remaining: int = 0
    previous_region_id: str | None = None


@dataclass
class ORToolsInspectionPlanner:
    duration_s: float
    control_period_s: float
    dwell_s: float
    safe_sky_altitude_m: float
    return_reserve_s: float
    horizontal_speed_mps: float
    vertical_speed_mps: float
    starts: dict[str, tuple[float, float, float]]
    cells: dict[str, PublicCell]
    assignments: dict[str, tuple[str, ...]]
    coarse_colliders: tuple[PublicAABB, ...] = ()
    direct_scan_clearance_m: float = 0.0
    routes: dict[str, RouteState] = field(default_factory=dict)
    completed: dict[str, set[str]] = field(default_factory=dict)
    max_observe_linear_speed_mps: float = 0.25
    max_observe_angular_speed_deg_s: float = 8.0

    @classmethod
    def from_public_reset(
        cls, public_episode: dict[str, Any], public_task_spec: dict[str, Any]
    ) -> ORToolsInspectionPlanner:
        _reject_non_public(public_episode)
        _reject_non_public(public_task_spec)
        if public_task_spec.get("task_track") != "G2-I":
            raise ValueError("OR-Tools adapter requires a public G2-I task")
        atlas = public_task_spec.get("inspection_atlas")
        sector = public_episode.get("mission_sector")
        contract = public_task_spec.get("execution_contract")
        transit = public_task_spec.get("public_transit_contract")
        if not all(isinstance(value, dict) for value in (atlas, sector, contract, transit)):
            raise ValueError("OR-Tools reset lacks a public atlas, sector, or contract")
        if (
            sector.get("truth_independent") is not True
            or sector.get("frozen_before_sampling") is not True
        ):
            raise ValueError("OR-Tools adapter requires a frozen target-independent sector")
        if sector.get("atlas_hash") != atlas.get("atlas_hash"):
            raise ValueError("public mission sector is not bound to the public atlas")

        starts: dict[str, tuple[float, float, float]] = {}
        for item in public_episode.get("starts", []):
            if not isinstance(item, dict):
                raise ValueError("public start record is invalid")
            drone_id = str(item.get("drone_id", ""))
            position = item.get("position")
            if not drone_id or not isinstance(position, list) or len(position) != 3:
                raise ValueError("public start record lacks a pose")
            starts[drone_id] = tuple(float(value) for value in position)  # type: ignore[assignment]
        if not starts:
            raise ValueError("OR-Tools adapter requires a non-empty fleet")

        selected = {str(value) for value in sector.get("selected_cell_ids", [])}
        if not selected:
            raise ValueError("public mission sector has no selected cells")
        cells: dict[str, PublicCell] = {}
        for region_index, region in enumerate(atlas.get("regions", [])):
            if not isinstance(region, dict):
                raise ValueError("public atlas region is invalid")
            region_id = str(region.get("region_id", f"region-{region_index:04d}"))
            if not region_id:
                raise ValueError("public atlas region lacks an ID")
            for item in region.get("cells", []):
                if not isinstance(item, dict):
                    raise ValueError("public atlas cell is invalid")
                cell_id = str(item.get("cell_id", ""))
                if cell_id not in selected:
                    continue
                pose = item.get("pose")
                position = pose.get("position") if isinstance(pose, dict) else None
                if not isinstance(position, list) or len(position) != 3:
                    raise ValueError("public inspection cell lacks a pose")
                cells[cell_id] = PublicCell(
                    cell_id=cell_id,
                    position=tuple(float(value) for value in position),  # type: ignore[arg-type]
                    yaw_deg=float(pose.get("yaw_deg", 0.0)),
                    pitch_deg=float(pose.get("pitch_deg", 0.0)),
                    represented_area_m2=float(item.get("represented_area_m2", 0.0)),
                    region_id=region_id,
                )
        if set(cells) != selected or any(
            cell.represented_area_m2 <= 0.0 for cell in cells.values()
        ):
            raise ValueError("public sector does not resolve to valid positive-area cells")

        raw_assignments = sector.get("cell_assignment_by_drone")
        if not isinstance(raw_assignments, dict) or set(raw_assignments) != set(starts):
            raise ValueError("public sector lacks one workload assignment per drone")
        assignments = {
            drone_id: tuple(str(cell_id) for cell_id in raw_assignments[drone_id])
            for drone_id in sorted(starts)
        }
        flattened = [cell_id for values in assignments.values() for cell_id in values]
        if set(flattened) != selected or len(flattened) != len(set(flattened)):
            raise ValueError("public workload assignment is not a cell partition")

        episode = contract.get("episode")
        observe = contract.get("observe")
        vehicle = contract.get("vehicle")
        capacity_certificate = sector.get("capacity_certificate")
        if not all(
            isinstance(value, dict)
            for value in (episode, observe, vehicle, capacity_certificate)
        ):
            raise ValueError("public execution contract is incomplete")
        values = {
            "duration_s": float(episode.get("duration_s", 0.0)),
            "control_period_s": float(contract.get("control_period_s", 0.0)),
            "dwell_s": float(observe.get("continuous_dwell_s", 0.0)),
            "max_observe_linear_speed_mps": float(observe.get("max_linear_speed_mps", 0.0)),
            "max_observe_angular_speed_deg_s": float(
                observe.get("max_angular_speed_deg_s", 0.0)
            ),
            "return_reserve_s": float(capacity_certificate.get("return_reserve_s", 0.0)),
            "safe_sky_altitude_m": float(transit.get("safe_sky_altitude_m", 0.0)),
            # The sector was admitted with these public, frozen transit rates.
            # Vehicle speeds in the task contract are upper bounds; using them
            # here silently overstates the work that fits before return.
            "horizontal_speed_mps": float(
                capacity_certificate.get("horizontal_speed_mps", 0.0)
            ),
            "vertical_speed_mps": float(capacity_certificate.get("vertical_speed_mps", 0.0)),
        }
        if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
            raise ValueError("public timing or transit contract is invalid")
        vehicle_horizontal_cap = float(vehicle.get("horizontal_speed_mps", 0.0))
        vehicle_vertical_cap = float(vehicle.get("vertical_speed_mps", 0.0))
        direct_scan_clearance_m = float(vehicle.get("radius_m", 0.0)) + float(
            vehicle.get("minimum_clearance_m", 0.0)
        )
        if (
            not math.isfinite(vehicle_horizontal_cap)
            or not math.isfinite(vehicle_vertical_cap)
            or values["horizontal_speed_mps"] > vehicle_horizontal_cap
            or values["vertical_speed_mps"] > vehicle_vertical_cap
        ):
            raise ValueError("public capacity-certificate speed exceeds vehicle contract cap")
        if not math.isfinite(direct_scan_clearance_m) or direct_scan_clearance_m < 0.0:
            raise ValueError("public vehicle direct-scan clearance is invalid")
        maximum = public_task_spec.get("flight_bounds", {}).get("maximum")
        if (
            not isinstance(maximum, list)
            or len(maximum) != 3
            or values["safe_sky_altitude_m"] >= float(maximum[2])
        ):
            raise ValueError("public safe-sky altitude is outside flight bounds")
        planner = cls(
            starts=starts,
            cells=cells,
            assignments=assignments,
            coarse_colliders=_public_coarse_colliders(public_task_spec),
            direct_scan_clearance_m=direct_scan_clearance_m,
            routes={},
            completed={drone_id: set() for drone_id in starts},
            **values,
        )
        planner.routes = {
            drone_id: RouteState(planner._solve_sector_route(drone_id))
            for drone_id in sorted(starts)
        }
        return planner

    def _transit_time_s(
        self, origin: tuple[float, float, float], destination: tuple[float, float, float]
    ) -> float:
        climb = abs(self.safe_sky_altitude_m - origin[2]) / self.vertical_speed_mps
        horizontal = math.hypot(destination[0] - origin[0], destination[1] - origin[1])
        horizontal /= self.horizontal_speed_mps
        descend = abs(self.safe_sky_altitude_m - destination[2]) / self.vertical_speed_mps
        return climb + horizontal + descend

    def _direct_scan_time_s(
        self, origin: tuple[float, float, float], destination: tuple[float, float, float]
    ) -> float:
        """Match the public grouped-region capacity certificate exactly."""

        horizontal = math.hypot(destination[0] - origin[0], destination[1] - origin[1])
        vertical = abs(destination[2] - origin[2])
        return max(horizontal / self.horizontal_speed_mps, vertical / self.vertical_speed_mps)

    def _direct_scan_is_publicly_clear(
        self, origin: tuple[float, float, float], destination: tuple[float, float, float]
    ) -> bool:
        return not any(
            _segment_intersects_expanded_box(
                origin, destination, collider, self.direct_scan_clearance_m
            )
            for collider in self.coarse_colliders
        )

    def _solve_sector_route(self, drone_id: str) -> list[str]:
        """Use the locked external vehicle-routing solver over public cells only."""

        try:
            import ortools
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        except ImportError as exc:  # pragma: no cover - exercised in integration.
            raise RuntimeError(
                f"OR-Tools {ORTOOLS_VERSION} is required in the external planner environment"
            ) from exc
        if str(ortools.__version__) != ORTOOLS_VERSION:
            raise RuntimeError(
                f"OR-Tools version drift: expected {ORTOOLS_VERSION}, got {ortools.__version__}"
            )
        cells = [self.cells[cell_id] for cell_id in self.assignments[drone_id]]
        if not cells:
            return []
        points = [self.starts[drone_id], *(cell.position for cell in cells)]
        manager = pywrapcp.RoutingIndexManager(len(points), 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        dwell_ms = int(round(self.dwell_s * 1000.0))

        def time_cost(from_index: int, to_index: int) -> int:
            source = manager.IndexToNode(from_index)
            destination = manager.IndexToNode(to_index)
            source_cell = cells[source - 1] if source else None
            destination_cell = cells[destination - 1] if destination else None
            source_position = self.starts[drone_id] if source_cell is None else source_cell.position
            destination_position = (
                self.starts[drone_id] if destination_cell is None else destination_cell.position
            )
            if (
                source_cell is not None
                and destination_cell is not None
                and source_cell.region_id == destination_cell.region_id
                and self._direct_scan_is_publicly_clear(source_position, destination_position)
            ):
                travel_s = self._direct_scan_time_s(source_position, destination_position)
            else:
                travel_s = self._transit_time_s(source_position, destination_position)
            travel_ms = int(
                round(travel_s * 1000.0)
            )
            return travel_ms + (dwell_ms if source else 0)

        callback = routing.RegisterTransitCallback(time_cost)
        routing.SetArcCostEvaluatorOfAllVehicles(callback)
        maximum_ms = int(round((self.duration_s - self.return_reserve_s) * 1000.0))
        routing.AddDimension(callback, 0, maximum_ms, True, "time")
        maximum_area = max(cell.represented_area_m2 for cell in cells)
        for node, cell in enumerate(cells, start=1):
            penalty = max(1, int(round(100_000.0 * cell.represented_area_m2 / maximum_area)))
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
        parameters = pywrapcp.DefaultRoutingSearchParameters()
        parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        parameters.time_limit.FromMilliseconds(250)
        assignment = routing.SolveWithParameters(parameters)
        if assignment is None:
            return []
        route: list[str] = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node:
                route.append(cells[node - 1].cell_id)
            index = assignment.Value(routing.NextVar(index))
        return route

    @staticmethod
    def _observation_pose(observation: dict[str, Any]) -> tuple[float, float, float]:
        state = observation.get("self_state")
        pose = state.get("pose") if isinstance(state, dict) else None
        position = pose.get("position") if isinstance(pose, dict) else None
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("public action lacks a self pose")
        return tuple(float(value) for value in position)  # type: ignore[return-value]

    @staticmethod
    def _observation_yaw(observation: dict[str, Any]) -> float:
        state = observation.get("self_state")
        pose = state.get("pose") if isinstance(state, dict) else None
        return float(pose.get("yaw_deg", 0.0)) if isinstance(pose, dict) else 0.0

    def _observation_is_settled(self, observation: dict[str, Any]) -> bool:
        """Use only public self-state before starting an evaluator dwell."""

        state = observation.get("self_state")
        velocity = state.get("linear_velocity_world_mps") if isinstance(state, dict) else None
        angular_speed = state.get("angular_speed_deg_s") if isinstance(state, dict) else None
        if not isinstance(velocity, list) or len(velocity) != 3:
            raise ValueError("public observation lacks a linear velocity")
        try:
            linear_speed = math.sqrt(sum(float(component) ** 2 for component in velocity))
            angular_speed_deg_s = float(angular_speed)
        except (TypeError, ValueError) as exc:
            raise ValueError("public observation has an invalid velocity") from exc
        if not math.isfinite(linear_speed) or not math.isfinite(angular_speed_deg_s):
            raise ValueError("public observation has a non-finite velocity")
        return (
            linear_speed <= self.max_observe_linear_speed_mps
            and abs(angular_speed_deg_s) <= self.max_observe_angular_speed_deg_s
        )

    @staticmethod
    def _at(position: tuple[float, float, float], goal: tuple[float, float, float]) -> bool:
        return _distance(position, goal) <= 0.35

    def _return_lower_bound_s(self, drone_id: str, position: tuple[float, float, float]) -> float:
        return self._transit_time_s(position, self.starts[drone_id])

    def _route_action(
        self, drone_id: str, observation: dict[str, Any], route: RouteState
    ) -> dict[str, Any]:
        position = self._observation_pose(observation)
        yaw = self._observation_yaw(observation)
        home = self.starts[drone_id]
        is_return = route.phase.startswith("return")
        target = home if is_return else self.cells[str(route.cell_id)].position
        target_yaw = yaw if is_return else self.cells[str(route.cell_id)].yaw_deg
        target_pitch = 0.0 if is_return else self.cells[str(route.cell_id)].pitch_deg
        if route.phase in {"ascend", "return-ascend"}:
            goal = (position[0], position[1], self.safe_sky_altitude_m)
            if self._at(position, goal):
                route.phase = "return-transit" if is_return else "transit"
            else:
                return {"kind": "WAYPOINT", "waypoint": _pose(goal, yaw)}
        if route.phase in {"transit", "return-transit"}:
            goal = (target[0], target[1], self.safe_sky_altitude_m)
            if self._at(position, goal):
                route.phase = "return-descend" if is_return else "descend"
            else:
                return {"kind": "WAYPOINT", "waypoint": _pose(goal, target_yaw)}
        if route.phase in {"descend", "direct", "return-descend"}:
            if self._at(position, target):
                if is_return:
                    route.phase = "returned"
                else:
                    # Arriving inside the position tolerance does not imply a
                    # CF2X has stopped.  Hold first, then consult only the
                    # public velocity fields before starting an OBSERVE dwell.
                    route.phase = "settle"
                    return {"kind": "HOVER"}
            else:
                return {
                    "kind": "WAYPOINT",
                    # Waypoints command CF2X position and yaw only.  The
                    # atlas pitch is a public bounded-camera command, not an
                    # impossible body-hover attitude.
                    "waypoint": _pose(target, target_yaw),
                    "sensor_pitch_deg": target_pitch,
                }
        if route.phase == "settle":
            if self._observation_is_settled(observation):
                route.phase = "observe"
                route.observe_ticks_remaining = max(
                    # The first accepted observation starts the dwell
                    # interval.  It does not itself contribute a full
                    # control period, so the endpoint needs one extra sample.
                    1, math.ceil(self.dwell_s / self.control_period_s) + 1,
                )
            # Issue a final HOVER even when the packet has settled.  The next
            # packet then begins the dwell from a verified stationary source.
            return {"kind": "HOVER"}
        if route.phase == "observe":
            route.observe_ticks_remaining -= 1
            if route.observe_ticks_remaining <= 0:
                completed_cell_id = str(route.cell_id)
                self.completed[drone_id].add(completed_cell_id)
                route.previous_region_id = self.cells[completed_cell_id].region_id
                route.cell_id = None
                route.phase = "select"
            return {"kind": "OBSERVE", "source_observation_id": str(observation["observation_id"])}
        if route.phase == "returned":
            return {"kind": "RETURN"}
        raise RuntimeError(f"invalid public route phase: {route.phase}")

    def action(self, drone_id: str, observation: dict[str, Any]) -> dict[str, Any]:
        if drone_id not in self.routes:
            raise ValueError("observation has an unknown drone")
        route = self.routes[drone_id]
        position = self._observation_pose(observation)
        timestamp_s = float(observation.get("timestamp_s", 0.0))
        if (
            timestamp_s + self._return_lower_bound_s(drone_id, position) + self.return_reserve_s
            >= self.duration_s
            and not route.phase.startswith("return")
            and route.phase != "returned"
        ):
            route.cell_id = None
            route.phase = "return-ascend"
        if route.phase == "select":
            while route.ordered_cell_ids and route.ordered_cell_ids[0] in self.completed[drone_id]:
                route.ordered_cell_ids.pop(0)
            if route.ordered_cell_ids:
                route.cell_id = route.ordered_cell_ids.pop(0)
                next_region_id = self.cells[str(route.cell_id)].region_id
                route.phase = (
                    "direct"
                    if (
                        route.previous_region_id == next_region_id
                        and self._direct_scan_is_publicly_clear(
                            position, self.cells[str(route.cell_id)].position
                        )
                    )
                    else "ascend"
                )
            else:
                route.phase = "return-ascend"
        return self._route_action(drone_id, observation, route)


def _response(request_id: object, **payload: Any) -> str:
    return json.dumps(
        {"schema": RESPONSE_SCHEMA, "request_id": request_id, **payload},
        ensure_ascii=True,
        separators=(",", ":"),
    )


def serve(lines: Iterable[str] | None = None) -> None:
    planner: ORToolsInspectionPlanner | None = None
    for line in sys.stdin if lines is None else lines:
        request: object = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
                raise ValueError("request schema differs")
            request_id = request.get("request_id")
            if request.get("kind") == "reset":
                planner = ORToolsInspectionPlanner.from_public_reset(
                    request["public_episode"], request["public_task_spec"]
                )
                output = _response(request_id, status="ok")
            elif request.get("kind") == "act":
                if planner is None:
                    raise ValueError("act arrived before reset")
                observations = request.get("observations")
                if not isinstance(observations, dict) or set(observations) != set(planner.starts):
                    raise ValueError("active observations differ from the public fleet")
                actions = {
                    drone_id: planner.action(drone_id, observation)
                    for drone_id, observation in sorted(observations.items())
                    if isinstance(observation, dict)
                }
                if set(actions) != set(planner.starts):
                    raise ValueError("an observation was not an object")
                output = _response(request_id, status="ok", actions=actions)
            else:
                raise ValueError("request kind is unsupported")
        except Exception as exc:
            request_id = request.get("request_id") if isinstance(request, dict) else None
            output = _response(request_id, status=f"error:{type(exc).__name__}")
        print(output, flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--upstream-source", type=Path)
    args = parser.parse_args(argv)
    if args.upstream_source is not None:
        _verify_upstream_source(args.upstream_source)
    if args.version:
        print(ORTOOLS_VERSION)
        return
    serve()


if __name__ == "__main__":
    main()
