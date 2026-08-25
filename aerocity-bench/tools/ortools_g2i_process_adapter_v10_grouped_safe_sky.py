"""Run the versioned, grouped-safe-sky OR-Tools G2-I routing baseline.

This adapter is a repair of the v9 *implementation*, not of the G2-I task
contract.  Version 9 charged a full climb, horizontal safe-sky transfer, and
descent for every pair of cells.  The frozen public capacity certificate uses
that transfer only when moving between scan groups or around a top-down roof
observation; consecutive facade cells in one public region are a local scan.

The original v9 source and all of its L1 receipts remain immutable evidence.
This module loads it only after verifying its recorded digest, reuses its
public-boundary validation and JSONL ABI, and replaces the route compilation
and local-transition state machine.  It never reads targets, support sites,
evaluator witnesses, or private split fields.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

ADAPTER_ID = "ortools-public-atlas-routing-v10-grouped-safe-sky"
LEGACY_ADAPTER_SHA256 = "e5dbda94ebebb7166b68440a1c6308f09b039d241473f9ba53b96c0e81127230"
GROUPED_ROUTE_MODEL = "public-fixed-assignment-grouped-safe-sky-route-v1"


def _legacy_path() -> Path:
    return Path(__file__).with_name("ortools_g2i_process_adapter.py")


def _load_legacy_adapter() -> ModuleType:
    """Load the frozen v9 ABI implementation after checking its source hash."""

    path = _legacy_path()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != LEGACY_ADAPTER_SHA256:
        raise RuntimeError("the frozen v9 OR-Tools adapter source digest differs")
    spec = importlib.util.spec_from_file_location("_aerocity_ortools_v9_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the frozen v9 OR-Tools adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_adapter()
REQUEST_SCHEMA = _legacy.REQUEST_SCHEMA
RESPONSE_SCHEMA = _legacy.RESPONSE_SCHEMA
UPSTREAM_URL = _legacy.UPSTREAM_URL
UPSTREAM_COMMIT = _legacy.UPSTREAM_COMMIT
UPSTREAM_LICENSE = _legacy.UPSTREAM_LICENSE
ORTOOLS_VERSION = _legacy.ORTOOLS_VERSION
PublicCell = _legacy.PublicCell
RouteState = _legacy.RouteState


class GroupedSafeSkyORToolsPlanner(_legacy.ORToolsInspectionPlanner):
    """Public-only planner whose route model matches the public certificate."""

    @classmethod
    def from_public_reset(
        cls, public_episode: dict[str, Any], public_task_spec: dict[str, Any]
    ) -> GroupedSafeSkyORToolsPlanner:
        # The inherited constructor validates the public boundary before it
        # resolves the atlas.  Dynamic dispatch invokes this class's route
        # compiler, so v9's all-safe-sky objective is never used here.
        planner = super().from_public_reset(public_episode, public_task_spec)
        planner.last_completed_cell_by_drone = {
            drone_id: None for drone_id in planner.starts
        }
        return planner

    @staticmethod
    def _is_top_down(cell: Any) -> bool:
        return float(cell.pitch_deg) <= -60.0

    def _grouped_assignment(self, drone_id: str) -> list[list[str]]:
        """Split the fixed public assignment using certificate scan groups.

        A facade group consists of consecutive cells in one public region.
        A top-down cell is always a group of one, and also forces the following
        facade cell to enter through safe sky.  This is exactly the grouping
        condition in ``_mission_route_lower_bound_s``.
        """

        groups: list[list[str]] = []
        current: list[str] = []
        current_region: str | None = None
        previous_was_top_down = False
        for cell_id in self.assignments[drone_id]:
            cell = self.cells[cell_id]
            top_down = self._is_top_down(cell)
            starts_new_group = (
                not current
                or top_down
                or previous_was_top_down
                or cell.region_id != current_region
            )
            if starts_new_group and current:
                groups.append(current)
                current = []
            current.append(cell_id)
            if top_down:
                groups.append(current)
                current = []
                current_region = None
            else:
                current_region = cell.region_id
            previous_was_top_down = top_down
        if current:
            groups.append(current)
        if [cell_id for group in groups for cell_id in group] != list(
            self.assignments[drone_id]
        ):
            raise RuntimeError("grouped public assignment lost or reordered a cell")
        return groups

    def _motion_time_s(
        self, origin: tuple[float, float, float], destination: tuple[float, float, float]
    ) -> float:
        horizontal = math.hypot(destination[0] - origin[0], destination[1] - origin[1])
        vertical = abs(destination[2] - origin[2])
        return max(
            horizontal / self.horizontal_speed_mps,
            vertical / self.vertical_speed_mps,
        )

    def _local_order_cost_s(
        self,
        origin: tuple[float, float, float],
        ordered_cell_ids: list[str],
    ) -> float:
        current = origin
        total = 0.0
        for cell_id in ordered_cell_ids:
            position = self.cells[cell_id].position
            total += self._motion_time_s(current, position)
            current = position
        return total

    def _solve_local_group(
        self,
        *,
        origin: tuple[float, float, float],
        group: list[str],
    ) -> list[str]:
        """Optimize a mandatory facade-group visit order with locked OR-Tools.

        The group is deliberately mandatory.  In v9, optional-node penalties
        were lower than a falsely inflated all-safe-sky edge, so an empty route
        was a solver optimum.  Here an optimizer proposal is accepted only if
        it beats the certificate-compatible canonical order; otherwise the
        canonical public ordering is retained.
        """

        if len(group) <= 1:
            return list(group)
        try:
            import ortools
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        except ImportError as exc:  # pragma: no cover - integration environment only.
            raise RuntimeError(
                f"OR-Tools {ORTOOLS_VERSION} is required in the external planner environment"
            ) from exc
        if str(ortools.__version__) != ORTOOLS_VERSION:
            raise RuntimeError(
                f"OR-Tools version drift: expected {ORTOOLS_VERSION}, got {ortools.__version__}"
            )
        points = [origin, *(self.cells[cell_id].position for cell_id in group)]
        manager = pywrapcp.RoutingIndexManager(len(points), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def cost(from_index: int, to_index: int) -> int:
            source = manager.IndexToNode(from_index)
            destination = manager.IndexToNode(to_index)
            return int(round(1000.0 * self._motion_time_s(points[source], points[destination])))

        callback = routing.RegisterTransitCallback(cost)
        routing.SetArcCostEvaluatorOfAllVehicles(callback)
        parameters = pywrapcp.DefaultRoutingSearchParameters()
        parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        parameters.time_limit.FromMilliseconds(100)
        solution = routing.SolveWithParameters(parameters)
        if solution is None:
            # The group is a small mandatory TSP.  A solver failure must not
            # silently discard public work; retain the independently certified
            # canonical order and let the parent record planner failures only
            # for an actual process-level error.
            return list(group)
        proposed: list[str] = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node:
                proposed.append(group[node - 1])
            index = solution.Value(routing.NextVar(index))
        if set(proposed) != set(group) or len(proposed) != len(group):
            raise RuntimeError("OR-Tools local group route is not a cell permutation")
        canonical_cost = self._local_order_cost_s(origin, list(group))
        proposed_cost = self._local_order_cost_s(origin, proposed)
        return proposed if proposed_cost <= canonical_cost + 1.0e-9 else list(group)

    def _solve_sector_route(self, drone_id: str) -> list[str]:
        """Return all assigned cells under the certificate-compatible topology."""

        if not hasattr(self, "direct_successors_by_drone"):
            self.direct_successors_by_drone: dict[str, dict[str, str]] = {}
        route: list[str] = []
        direct_successors: dict[str, str] = {}
        current = self.starts[drone_id]
        for group in self._grouped_assignment(drone_id):
            ordered_group = self._solve_local_group(origin=current, group=group)
            for first, second in zip(ordered_group, ordered_group[1:], strict=False):
                direct_successors[first] = second
            route.extend(ordered_group)
            current = self.cells[ordered_group[-1]].position
        assigned = list(self.assignments[drone_id])
        if set(route) != set(assigned) or len(route) != len(assigned):
            raise RuntimeError("grouped OR-Tools route is not a complete public assignment")
        self.direct_successors_by_drone[drone_id] = direct_successors
        return route

    def _route_action(
        self, drone_id: str, observation: dict[str, Any], route: Any
    ) -> dict[str, Any]:
        if route.phase == "local-transit":
            target = self.cells[str(route.cell_id)]
            position = self._observation_pose(observation)
            if self._at(position, target.position):
                route.phase = "settle"
                return {"kind": "HOVER"}
            return {
                "kind": "WAYPOINT",
                "waypoint": _legacy._pose(target.position, target.yaw_deg),
                "sensor_pitch_deg": target.pitch_deg,
            }
        completed_cell_id = str(route.cell_id) if route.phase == "observe" else None
        result = super()._route_action(drone_id, observation, route)
        if completed_cell_id is not None and route.phase == "select":
            self.last_completed_cell_by_drone[drone_id] = completed_cell_id
        return result

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
                previous = self.last_completed_cell_by_drone[drone_id]
                successors = self.direct_successors_by_drone.get(drone_id, {})
                route.phase = (
                    "local-transit"
                    if previous is not None and successors.get(previous) == route.cell_id
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
    planner: GroupedSafeSkyORToolsPlanner | None = None
    for line in sys.stdin if lines is None else lines:
        request: object = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
                raise ValueError("request schema differs")
            request_id = request.get("request_id")
            if request.get("kind") == "reset":
                planner = GroupedSafeSkyORToolsPlanner.from_public_reset(
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
        _legacy._verify_upstream_source(args.upstream_source)
    if args.version:
        print(ORTOOLS_VERSION)
        return
    serve()


if __name__ == "__main__":
    main()
