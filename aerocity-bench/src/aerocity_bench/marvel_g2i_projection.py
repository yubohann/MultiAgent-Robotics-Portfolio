"""Public G2-I to MARVEL graph projection for an external-process diagnostic.

MARVEL was trained for 2-D frontier exploration.  This module deliberately
does not relabel it as a native 3-D hidden-target method.  It converts only
the public inspection atlas and public mission sector into the fixed-shape
graph expected by the frozen upstream policy.  The upstream policy selects a
public inspection cell; this module expands that decision into a conservative
public safe-sky route and an ``OBSERVE`` dwell sequence.

No evaluator-private episode field, target field, support site, witness, or
confirmation identity is accepted by this projection.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .public_boundary import assert_public_fields

_FORBIDDEN_FRAGMENTS = (
    "target",
    "support",
    "witness",
    "evaluator",
    "private",
    "split_label",
)
_MAX_NODES = 360
_NEIGHBOR_SLOTS = 25
_HEADING_BINS = 36
_HEADING_CHOICES = 3


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _bearing_bin(
    origin: tuple[float, float, float], destination: tuple[float, float, float]
) -> int:
    angle = math.degrees(math.atan2(destination[1] - origin[1], destination[0] - origin[0]))
    return int((angle % 360.0) / 360.0 * _HEADING_BINS) % _HEADING_BINS


def _one_hot(index: int) -> list[float]:
    values = [0.0] * _HEADING_BINS
    values[index % _HEADING_BINS] = 1.0
    return values


def _pose(position: tuple[float, float, float], yaw_deg: float) -> dict[str, Any]:
    return {
        "position": [round(value, 6) for value in position],
        "yaw_deg": round(yaw_deg, 6),
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
    }


def _require_public(value: object, *, path: str = "$") -> None:
    """Reject fields that an external method must never receive."""

    # Reuse the benchmark-wide rule.  In particular, the three explicit
    # ``*_public: false`` sentinels document withheld facts and are not leaks.
    try:
        assert_public_fields(value, path=path)
    except ValueError as exc:
        raise ValueError(f"MARVEL projection received a non-public field: {exc}") from exc


@dataclass(frozen=True)
class PublicInspectionCell:
    cell_id: str
    position: tuple[float, float, float]
    yaw_deg: float
    represented_area_m2: float


@dataclass(frozen=True)
class MarvelGraphInput:
    """Numerical input in MARVEL's original fixed-shape tensor layout."""

    node_inputs: list[list[float]]
    node_padding_mask: list[int]
    edge_mask: list[list[int]]
    current_edge: list[int]
    edge_padding_mask: list[int]
    frontier_distribution: list[list[float]]
    headings_visited: list[list[float]]
    neighbor_best_headings: list[list[list[float]]]
    candidate_cell_ids: list[str]


@dataclass
class _RouteState:
    cell_id: str | None = None
    phase: str = "select"
    observe_ticks_remaining: int = 0


@dataclass
class MarvelG2IProjection:
    """Stateful public projection and route executor used by the adapter process."""

    duration_s: float
    control_period_s: float
    dwell_s: float
    safe_sky_altitude_m: float
    return_reserve_s: float
    horizontal_speed_mps: float
    vertical_speed_mps: float
    starts: dict[str, tuple[float, float, float]]
    cells: dict[str, PublicInspectionCell]
    assignments: dict[str, tuple[str, ...]]
    completed: dict[str, set[str]] = field(default_factory=dict)
    routes: dict[str, _RouteState] = field(default_factory=dict)

    @classmethod
    def from_public_reset(
        cls, public_episode: dict[str, Any], public_task_spec: dict[str, Any]
    ) -> MarvelG2IProjection:
        _require_public(public_episode)
        _require_public(public_task_spec)
        if public_task_spec.get("task_track") != "G2-I":
            raise ValueError("MARVEL diagnostic requires a public G2-I task")
        atlas = public_task_spec.get("inspection_atlas")
        sector = public_episode.get("mission_sector")
        contract = public_task_spec.get("execution_contract")
        transit = public_task_spec.get("public_transit_contract")
        if not all(isinstance(item, dict) for item in (atlas, sector, contract, transit)):
            raise ValueError("MARVEL diagnostic reset lacks public atlas, sector, or contract")
        if (
            sector.get("truth_independent") is not True
            or sector.get("frozen_before_sampling") is not True
        ):
            raise ValueError(
                "MARVEL diagnostic requires a frozen target-independent mission sector"
            )
        if sector.get("atlas_hash") != atlas.get("atlas_hash"):
            raise ValueError("MARVEL diagnostic mission sector is not bound to the public atlas")

        starts: dict[str, tuple[float, float, float]] = {}
        for item in public_episode.get("starts", []):
            if not isinstance(item, dict):
                raise ValueError("MARVEL diagnostic start record is invalid")
            position = item.get("position")
            drone_id = str(item.get("drone_id", ""))
            if not drone_id or not isinstance(position, list) or len(position) != 3:
                raise ValueError("MARVEL diagnostic start record lacks a public pose")
            starts[drone_id] = tuple(float(value) for value in position)  # type: ignore[assignment]
        if not starts:
            raise ValueError("MARVEL diagnostic requires a non-empty public fleet")

        selected = {str(value) for value in sector.get("selected_cell_ids", [])}
        if not selected:
            raise ValueError("MARVEL diagnostic mission sector has no public cells")
        cells: dict[str, PublicInspectionCell] = {}
        for region in atlas.get("regions", []):
            if not isinstance(region, dict):
                raise ValueError("MARVEL diagnostic atlas region is invalid")
            for item in region.get("cells", []):
                if not isinstance(item, dict):
                    raise ValueError("MARVEL diagnostic atlas cell is invalid")
                cell_id = str(item.get("cell_id", ""))
                if cell_id not in selected:
                    continue
                pose = item.get("pose")
                position = pose.get("position") if isinstance(pose, dict) else None
                if not isinstance(position, list) or len(position) != 3:
                    raise ValueError("MARVEL diagnostic cell lacks a public pose")
                cells[cell_id] = PublicInspectionCell(
                    cell_id=cell_id,
                    position=tuple(float(value) for value in position),  # type: ignore[arg-type]
                    yaw_deg=float(pose.get("yaw_deg", 0.0)),
                    represented_area_m2=float(item.get("represented_area_m2", 0.0)),
                )
        if set(cells) != selected or any(
            cell.represented_area_m2 <= 0.0 for cell in cells.values()
        ):
            raise ValueError(
                "MARVEL diagnostic sector does not resolve to valid public atlas cells"
            )

        raw_assignments = sector.get("cell_assignment_by_drone")
        if not isinstance(raw_assignments, dict) or set(raw_assignments) != set(starts):
            raise ValueError("MARVEL diagnostic requires one public workload assignment per drone")
        assignments = {
            drone_id: tuple(str(cell_id) for cell_id in raw_assignments[drone_id])
            for drone_id in sorted(starts)
        }
        flattened = [cell_id for values in assignments.values() for cell_id in values]
        if set(flattened) != selected or len(flattened) != len(set(flattened)):
            raise ValueError("MARVEL diagnostic public workload assignment is not a partition")

        episode = contract.get("episode")
        observe = contract.get("observe")
        vehicle = contract.get("vehicle")
        if not all(isinstance(item, dict) for item in (episode, observe, vehicle)):
            raise ValueError("MARVEL diagnostic execution contract is incomplete")
        duration_s = float(episode.get("duration_s", 0.0))
        control_period_s = float(contract.get("control_period_s", 0.0))
        dwell_s = float(observe.get("continuous_dwell_s", 0.0))
        reserve_s = float(sector.get("capacity_certificate", {}).get("return_reserve_s", 0.0))
        safe_sky = float(transit.get("safe_sky_altitude_m", 0.0))
        horizontal_speed_mps = float(vehicle.get("horizontal_speed_mps", 0.0))
        vertical_speed_mps = float(vehicle.get("vertical_speed_mps", 0.0))
        if min(
            duration_s,
            control_period_s,
            dwell_s,
            reserve_s,
            safe_sky,
            horizontal_speed_mps,
            vertical_speed_mps,
        ) <= 0.0:
            raise ValueError("MARVEL diagnostic public timing or safe-sky contract is invalid")
        if safe_sky >= float(public_task_spec["flight_bounds"]["maximum"][2]):
            raise ValueError("MARVEL diagnostic safe-sky altitude is outside public flight bounds")
        return cls(
            duration_s=duration_s,
            control_period_s=control_period_s,
            dwell_s=dwell_s,
            safe_sky_altitude_m=safe_sky,
            return_reserve_s=reserve_s,
            horizontal_speed_mps=horizontal_speed_mps,
            vertical_speed_mps=vertical_speed_mps,
            starts=starts,
            cells=cells,
            assignments=assignments,
            completed={drone_id: set() for drone_id in starts},
            routes={drone_id: _RouteState() for drone_id in starts},
        )

    @staticmethod
    def _observation_pose(observation: dict[str, Any]) -> tuple[float, float, float]:
        state = observation.get("self_state")
        pose = state.get("pose") if isinstance(state, dict) else None
        position = pose.get("position") if isinstance(pose, dict) else None
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("MARVEL diagnostic action lacks a public self pose")
        return tuple(float(value) for value in position)  # type: ignore[return-value]

    @staticmethod
    def _observation_yaw(observation: dict[str, Any]) -> float:
        state = observation.get("self_state")
        pose = state.get("pose") if isinstance(state, dict) else None
        return float(pose.get("yaw_deg", 0.0)) if isinstance(pose, dict) else 0.0

    def _available_cells(self, drone_id: str) -> list[PublicInspectionCell]:
        return [
            self.cells[cell_id]
            for cell_id in self.assignments[drone_id]
            if cell_id not in self.completed[drone_id]
        ]

    def graph_input(self, drone_id: str, observation: dict[str, Any]) -> MarvelGraphInput:
        """Project one public agent state into MARVEL's 360-node input ABI."""

        origin = self._observation_pose(observation)
        candidates = sorted(
            self._available_cells(drone_id),
            key=lambda cell: (_distance(origin, cell.position), cell.position, cell.yaw_deg),
        )[: _MAX_NODES - 1]
        max_area = max((cell.represented_area_m2 for cell in candidates), default=1.0)
        node_inputs = [[0.0] * 6 for _ in range(_MAX_NODES)]
        node_padding_mask = [1] * _MAX_NODES
        edge_mask = [[1] * _MAX_NODES for _ in range(_MAX_NODES)]
        frontier_distribution = [[0.0] * _HEADING_BINS for _ in range(_MAX_NODES)]
        headings_visited = [[0.0] * _HEADING_BINS for _ in range(_MAX_NODES)]
        node_padding_mask[0] = 0
        for index in range(len(candidates) + 1):
            for neighbor in range(len(candidates) + 1):
                edge_mask[index][neighbor] = 0
        for index, cell in enumerate(candidates, start=1):
            bearing = _bearing_bin(origin, cell.position)
            node_padding_mask[index] = 0
            node_inputs[index] = [
                (cell.position[0] - origin[0]) / 30.0,
                (cell.position[1] - origin[1]) / 30.0,
                min(1.0, cell.represented_area_m2 / max_area),
                0.0,
                0.0,
                bearing / _HEADING_BINS,
            ]
            frontier_distribution[0][bearing] += min(1.0, cell.represented_area_m2 / max_area)
        neighbors = candidates[: _NEIGHBOR_SLOTS - 1]
        current_edge = [0] + list(range(1, len(neighbors) + 1))
        current_edge.extend([0] * (_NEIGHBOR_SLOTS - len(current_edge)))
        edge_padding_mask = [1] + [0] * len(neighbors)
        edge_padding_mask.extend([1] * (_NEIGHBOR_SLOTS - len(edge_padding_mask)))
        neighbor_best_headings: list[list[list[float]]] = []
        for cell in [None, *neighbors]:
            if cell is None:
                heading = 0
            else:
                heading = _bearing_bin(origin, cell.position)
            neighbor_best_headings.append(
                [_one_hot(heading - 1), _one_hot(heading), _one_hot(heading + 1)]
            )
        neighbor_best_headings.extend(
            [[[0.0] * _HEADING_BINS for _ in range(_HEADING_CHOICES)]
            for _ in range(_NEIGHBOR_SLOTS - len(neighbor_best_headings))]
        )
        return MarvelGraphInput(
            node_inputs=node_inputs,
            node_padding_mask=node_padding_mask,
            edge_mask=edge_mask,
            current_edge=current_edge,
            edge_padding_mask=edge_padding_mask,
            frontier_distribution=frontier_distribution,
            headings_visited=headings_visited,
            neighbor_best_headings=neighbor_best_headings,
            candidate_cell_ids=[cell.cell_id for cell in neighbors],
        )

    @staticmethod
    def _at(position: tuple[float, float, float], goal: tuple[float, float, float]) -> bool:
        return _distance(position, goal) <= 0.35

    def _public_return_lower_bound_s(
        self, drone_id: str, position: tuple[float, float, float]
    ) -> float:
        """Return via the public safe sky without inspecting private truth."""

        home = self.starts[drone_id]
        climb_s = abs(self.safe_sky_altitude_m - position[2]) / self.vertical_speed_mps
        transit_s = math.hypot(position[0] - home[0], position[1] - home[1])
        transit_s /= self.horizontal_speed_mps
        descend_s = abs(self.safe_sky_altitude_m - home[2]) / self.vertical_speed_mps
        return climb_s + transit_s + descend_s

    def _route_action(
        self, drone_id: str, observation: dict[str, Any], route: _RouteState
    ) -> dict[str, Any]:
        position = self._observation_pose(observation)
        yaw = self._observation_yaw(observation)
        home = self.starts[drone_id]
        is_return = route.phase.startswith("return")
        target = home if is_return else self.cells[str(route.cell_id)].position
        target_yaw = yaw if is_return else self.cells[str(route.cell_id)].yaw_deg
        if route.phase in {"select", "ascend", "return-ascend"}:
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
        if route.phase in {"descend", "return-descend"}:
            if self._at(position, target):
                if is_return:
                    route.phase = "returned"
                else:
                    route.phase = "observe"
                    route.observe_ticks_remaining = max(
                        1, math.ceil(self.dwell_s / self.control_period_s)
                    )
            else:
                return {"kind": "WAYPOINT", "waypoint": _pose(target, target_yaw)}
        if route.phase == "observe":
            route.observe_ticks_remaining -= 1
            if route.observe_ticks_remaining <= 0:
                self.completed[drone_id].add(str(route.cell_id))
                route.cell_id = None
                route.phase = "select"
            return {
                "kind": "OBSERVE",
                "source_observation_id": str(observation["observation_id"]),
            }
        if route.phase == "returned":
            return {"kind": "RETURN"}
        raise RuntimeError(f"MARVEL diagnostic route state is invalid: {route.phase}")

    def action(
        self,
        drone_id: str,
        observation: dict[str, Any],
        choose_slot: Callable[[MarvelGraphInput], int],
    ) -> dict[str, Any]:
        if drone_id not in self.routes:
            raise ValueError("MARVEL diagnostic observation has an unknown drone")
        timestamp_s = float(observation.get("timestamp_s", 0.0))
        route = self.routes[drone_id]
        position = self._observation_pose(observation)
        return_deadline_s = (
            timestamp_s
            + self._public_return_lower_bound_s(drone_id, position)
            + self.return_reserve_s
        )
        if (
            return_deadline_s >= self.duration_s
            and not route.phase.startswith("return")
            and route.phase != "returned"
        ):
            route.cell_id = None
            route.phase = "return-ascend"
        if route.phase == "select":
            graph = self.graph_input(drone_id, observation)
            if not graph.candidate_cell_ids:
                route.phase = "return-ascend"
            else:
                slot = int(choose_slot(graph))
                if not 0 <= slot < len(graph.candidate_cell_ids):
                    raise ValueError("MARVEL policy selected a masked or absent public candidate")
                route.cell_id = graph.candidate_cell_ids[slot]
                route.phase = "ascend"
        return self._route_action(drone_id, observation, route)
