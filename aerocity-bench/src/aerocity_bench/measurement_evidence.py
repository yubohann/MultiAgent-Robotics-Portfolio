"""Public, replay-derived measurement evidence for G2-I calibration runs.

This module deliberately has no evaluator-private target dependency.  It
reconstructs the two explanatory variables used by :mod:`measurement_claim`
from measured L1 vehicle states plus receipt-bound public observations:

* free-space coverage is the union of measured CF2X 2 m voxels; and
* inspection footprint is public-atlas area credited only after an accepted,
  safe OBSERVE action satisfies range, FoV, facing, LOS, and dwell.

The same class is used while an L1 replay is running and while its protected
evidence is independently re-aggregated.  That prevents a report summary from
becoming an unverified, hand-entered statistical input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .canonical import content_hash
from .contracts import ObservationPacket
from .geometry import (
    Vec3,
    colliders_from_city,
    distance,
    in_field_of_view,
    line_of_sight,
    sensor_pose,
    surface_facing,
)
from .inspection_atlas import validate_public_inspection_atlas, validate_public_mission_sector

EVIDENCE_SCHEMA = "org.aerocity.bench.g2-i-l1-measurement-evidence.v1"
FREE_SPACE_SEMANTICS = "L1_measured_physx_voxel_proxy_2m"
INSPECTION_SEMANTICS = (
    "area_weighted_public_atlas_cell_after_accepted_observe_runtime_safety_"
    "range_fov_facing_los_and_continuous_dwell"
)
_RESOLUTION_M = 2.0


def validate_measurement_evidence_snapshot(
    snapshot: object,
    *,
    measured_state_trace: object,
    input_bindings_hash: object,
) -> None:
    """Validate the hash-bound, target-independent measurement payload.

    This is intentionally structural: reconstructing visibility and coverage
    requires the CitySpec and public task, which the measurement aggregator
    performs.  The fleet report validator still needs to reject truncated or
    tampered evidence before a report can be accepted anywhere else.
    """

    if not isinstance(snapshot, dict) or snapshot.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("measurement evidence snapshot schema is unsupported")
    expected_fields = {
        "schema",
        "coverage_semantics",
        "inspection_footprint_semantics",
        "coverage_resolution_m",
        "input_bindings_hash",
        "coverage_trace",
        "inspection_coverage_trace",
        "inspection_cell_count_trace",
        "coverage_denominators",
        "measured_state_trace_hash",
    }
    if set(snapshot) != expected_fields:
        raise ValueError("measurement evidence snapshot fields differ")
    if snapshot["coverage_semantics"] != FREE_SPACE_SEMANTICS:
        raise ValueError("measurement evidence coverage semantics differ")
    if snapshot["inspection_footprint_semantics"] != INSPECTION_SEMANTICS:
        raise ValueError("measurement evidence inspection semantics differ")
    resolution = snapshot["coverage_resolution_m"]
    if not isinstance(resolution, (int, float)) or isinstance(resolution, bool):
        raise ValueError("measurement evidence resolution is invalid")
    if not math.isclose(float(resolution), _RESOLUTION_M, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("measurement evidence resolution differs")
    if not isinstance(input_bindings_hash, str) or snapshot["input_bindings_hash"] != (
        input_bindings_hash
    ):
        raise ValueError("measurement evidence input binding hash differs")
    if not isinstance(measured_state_trace, list) or not measured_state_trace:
        raise ValueError("measurement evidence state trace is empty")
    if snapshot["measured_state_trace_hash"] != content_hash(measured_state_trace):
        raise ValueError("measurement evidence state trace hash mismatch")

    denominators = snapshot["coverage_denominators"]
    if not isinstance(denominators, dict) or set(denominators) != {
        "coverage_2d_cells",
        "coverage_3d_free_cells",
        "inspection_atlas_cells",
        "inspection_atlas_area_m2",
    }:
        raise ValueError("measurement evidence denominators are incomplete")
    for field in ("coverage_2d_cells", "coverage_3d_free_cells", "inspection_atlas_cells"):
        value = denominators[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"measurement evidence denominator {field} is invalid")
    area = denominators["inspection_atlas_area_m2"]
    if (
        not isinstance(area, (int, float))
        or isinstance(area, bool)
        or not math.isfinite(float(area))
        or float(area) <= 0.0
    ):
        raise ValueError("measurement evidence inspection area denominator is invalid")

    coverage = snapshot["coverage_trace"]
    footprint = snapshot["inspection_coverage_trace"]
    cell_count = snapshot["inspection_cell_count_trace"]
    if (
        not isinstance(coverage, list)
        or not isinstance(footprint, list)
        or not isinstance(cell_count, list)
    ):
        raise ValueError("measurement evidence traces are malformed")
    if not (len(coverage) == len(footprint) == len(cell_count) == len(measured_state_trace)):
        raise ValueError("measurement evidence trace lengths differ")
    previous_time = -math.inf
    for index, (coverage_row, footprint_row, count_row) in enumerate(
        zip(coverage, footprint, cell_count, strict=True)
    ):
        if (
            not isinstance(coverage_row, list) or len(coverage_row) != 3
            or not isinstance(footprint_row, list) or len(footprint_row) != 2
            or not isinstance(count_row, list) or len(count_row) != 2
        ):
            raise ValueError(f"measurement evidence trace row {index} is malformed")
        timestamp = float(coverage_row[0])
        if not math.isfinite(timestamp) or timestamp < 0.0 or timestamp <= previous_time:
            raise ValueError("measurement evidence timestamps are not strictly increasing")
        previous_time = timestamp
        if float(footprint_row[0]) != timestamp or float(count_row[0]) != timestamp:
            raise ValueError("measurement evidence trace timestamps disagree")
        for value in coverage_row[1:]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("measurement evidence voxel counts are invalid")
        area_value = footprint_row[1]
        count_value = count_row[1]
        if (
            not isinstance(area_value, (int, float))
            or isinstance(area_value, bool)
            or not math.isfinite(float(area_value))
            or float(area_value) < 0.0
        ):
            raise ValueError("measurement evidence footprint values are invalid")
        if not isinstance(count_value, int) or isinstance(count_value, bool) or count_value < 0:
            raise ValueError("measurement evidence cell counts are invalid")


@dataclass(frozen=True)
class _AtlasCell:
    cell_id: str
    surface_point: Vec3
    surface_normal: Vec3
    represented_area_m2: float


@dataclass
class _Dwell:
    started_at_s: float
    last_seen_at_s: float
    initial_position: Vec3


class L1MeasurementEvidence:
    """Accumulate target-independent evidence from measured L1 execution."""

    def __init__(
        self,
        *,
        city: dict[str, Any],
        task_spec: dict[str, Any],
        public_episode: dict[str, Any],
    ) -> None:
        if task_spec.get("task_track") != "G2-I":
            raise ValueError("L1 measurement evidence requires a G2-I public task")
        execution = task_spec.get("execution_contract")
        if not isinstance(execution, dict):
            raise ValueError("G2-I task lacks its public execution contract")
        atlas = task_spec.get("inspection_atlas")
        if not isinstance(atlas, dict):
            raise ValueError("L1 measurement evidence requires the full public atlas")
        validate_public_inspection_atlas(atlas)
        starts = public_episode.get("starts")
        sector = public_episode.get("mission_sector")
        if not isinstance(starts, list) or not isinstance(sector, dict):
            raise ValueError("G2-I evidence requires public starts and mission sector")
        validate_public_mission_sector(sector, atlas, starts, execution)
        if public_episode.get("mission_sector_hash") != sector.get("sector_hash"):
            raise ValueError("G2-I evidence mission-sector hash differs")

        self.city = city
        self.execution = execution
        self._colliders = colliders_from_city(city)
        self._cells = self._load_cells(atlas, sector)
        self._visited_voxels: set[tuple[int, int, int]] = set()
        self._visited_cells: set[str] = set()
        self._dwell: dict[tuple[str, str], _Dwell] = {}
        self.coverage_trace: list[list[float | int]] = []
        self.inspection_coverage_trace: list[list[float]] = []
        self.inspection_cell_count_trace: list[list[float | int]] = []
        self.coverage_denominators = self._coverage_denominators()
        self.coverage_denominators["inspection_atlas_cells"] = len(self._cells)
        self.coverage_denominators["inspection_atlas_area_m2"] = round(
            sum(cell.represented_area_m2 for cell in self._cells.values()), 6
        )

    def _load_cells(self, atlas: dict[str, Any], sector: dict[str, Any]) -> dict[str, _AtlasCell]:
        selected = {str(value) for value in sector["selected_cell_ids"]}
        cells: dict[str, _AtlasCell] = {}
        for region in atlas["regions"]:
            for raw in region["cells"]:
                cell_id = str(raw["cell_id"])
                if cell_id not in selected:
                    continue
                if cell_id in cells:
                    raise ValueError("public atlas repeats a selected cell ID")
                point = tuple(float(value) for value in raw["surface_point"])
                normal = tuple(float(value) for value in raw["surface_normal"])
                if len(point) != 3 or len(normal) != 3:
                    raise ValueError("public atlas surface geometry is malformed")
                represented_area = float(raw["represented_area_m2"])
                if not math.isfinite(represented_area) or represented_area <= 0.0:
                    raise ValueError("public atlas represented area must be positive")
                cells[cell_id] = _AtlasCell(
                    cell_id=cell_id,
                    surface_point=point,  # type: ignore[arg-type]
                    surface_normal=normal,  # type: ignore[arg-type]
                    represented_area_m2=represented_area,
                )
        if set(cells) != selected:
            raise ValueError("public mission sector does not resolve against its atlas")
        return cells

    def _coverage_denominators(self) -> dict[str, int]:
        bounds = self.city.get("flight_bounds")
        if not isinstance(bounds, dict):
            raise ValueError("city lacks flight bounds")
        axes = [
            range(
                math.ceil(float(low) / _RESOLUTION_M),
                math.floor(float(high) / _RESOLUTION_M) + 1,
            )
            for low, high in zip(bounds["minimum"], bounds["maximum"], strict=True)
        ]
        if len(axes) != 3 or any(len(axis) <= 0 for axis in axes):
            raise ValueError("city coverage grid is empty")
        occupied: set[tuple[int, int, int]] = set()
        radius = float(self.execution["vehicle"]["radius_m"])
        for collider in self._colliders:
            expanded = collider.expanded(radius)
            index_ranges = [
                range(
                    max(axis.start, math.ceil(expanded.minimum[index] / _RESOLUTION_M)),
                    min(axis.stop - 1, math.floor(expanded.maximum[index] / _RESOLUTION_M)) + 1,
                )
                for index, axis in enumerate(axes)
            ]
            for x_index in index_ranges[0]:
                for y_index in index_ranges[1]:
                    for z_index in index_ranges[2]:
                        point = (
                            x_index * _RESOLUTION_M,
                            y_index * _RESOLUTION_M,
                            z_index * _RESOLUTION_M,
                        )
                        if expanded.contains(point):
                            occupied.add((x_index, y_index, z_index))
        total_2d = len(axes[0]) * len(axes[1])
        total_3d = total_2d * len(axes[2])
        free = total_3d - len(occupied)
        if free <= 0:
            raise ValueError("city has no free coverage voxels")
        return {"coverage_2d_cells": total_2d, "coverage_3d_free_cells": free}

    def _reset_dwell(self, drone_id: str, *, except_cells: set[str] | None = None) -> None:
        retained = except_cells or set()
        for key in list(self._dwell):
            if key[0] == drone_id and key[1] not in retained:
                del self._dwell[key]

    def end_observe(self, drone_id: str) -> None:
        self._reset_dwell(drone_id)

    def _cell_visible(self, observation: ObservationPacket, cell: _AtlasCell) -> bool:
        vehicle = self.execution["vehicle"]
        body_margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
        if any(
            collider.expanded(body_margin).contains(observation.pose.position)
            for collider in self._colliders
        ):
            return False
        rig = self.execution["sensor_rig"]
        camera_pose = sensor_pose(
            observation.pose,
            rig["translation_body_m"],
            sensor_pitch_deg=(
                observation.sensor_pitch_deg if rig["gimbal_mode"] == "bounded" else None
            ),
        )
        observe = self.execution["observe"]
        if distance(camera_pose.position, cell.surface_point) > float(observe["max_range_m"]):
            return False
        in_view, _, _ = in_field_of_view(
            camera_pose,
            cell.surface_point,
            float(observe["horizontal_fov_deg"]),
            float(observe["vertical_fov_deg"]),
        )
        if not in_view:
            return False
        facing, _ = surface_facing(
            camera_pose.position,
            cell.surface_point,
            cell.surface_normal,
            float(observe["surface_facing_min_cosine"]),
        )
        if not facing:
            return False
        visible, _ = line_of_sight(camera_pose.position, cell.surface_point, self._colliders)
        return visible

    def record_observe(
        self,
        observation: ObservationPacket,
        *,
        evaluator_accepted: bool,
        runtime_safe: bool,
    ) -> None:
        """Update public inspection credit from one receipt-bound OBSERVE action."""

        drone_id = observation.drone_id
        if not evaluator_accepted or not runtime_safe:
            self._reset_dwell(drone_id)
            return
        eligible = {
            cell_id
            for cell_id, cell in self._cells.items()
            if cell_id not in self._visited_cells and self._cell_visible(observation, cell)
        }
        self._reset_dwell(drone_id, except_cells=eligible)
        observe = self.execution["observe"]
        control_period = float(self.execution["control_period_s"])
        for cell_id in sorted(eligible):
            key = (drone_id, cell_id)
            dwell = self._dwell.get(key)
            if (
                dwell is None
                or observation.timestamp_s - dwell.last_seen_at_s > control_period * 1.6
                or distance(dwell.initial_position, observation.pose.position)
                > float(observe["max_pose_drift_m"])
            ):
                dwell = _Dwell(
                    started_at_s=observation.timestamp_s,
                    last_seen_at_s=observation.timestamp_s,
                    initial_position=observation.pose.position,
                )
                self._dwell[key] = dwell
            else:
                dwell.last_seen_at_s = observation.timestamp_s
            if dwell.last_seen_at_s - dwell.started_at_s + 1.0e-9 >= float(
                observe["continuous_dwell_s"]
            ):
                self._visited_cells.add(cell_id)
                self._dwell.pop(key, None)

    def record_measured_positions(
        self,
        task_time_s: float,
        positions_by_drone: dict[str, tuple[float, float, float]],
        *,
        safe_drone_ids: set[str],
    ) -> None:
        """Record measured post-PhysX positions after the tick safety decision."""

        if not math.isfinite(task_time_s) or task_time_s < 0.0:
            raise ValueError("measurement evidence timestamp is invalid")
        if not positions_by_drone or not safe_drone_ids <= set(positions_by_drone):
            raise ValueError("measurement evidence roster is invalid")
        for drone_id in sorted(safe_drone_ids):
            position = positions_by_drone[drone_id]
            if len(position) != 3 or not all(math.isfinite(value) for value in position):
                raise ValueError("measured CF2X position is invalid")
            self._visited_voxels.add(
                tuple(round(float(value) / _RESOLUTION_M) for value in position)
            )
        visited_3d = len(self._visited_voxels)
        visited_2d = len({(cell[0], cell[1]) for cell in self._visited_voxels})
        if self.coverage_trace and task_time_s <= float(self.coverage_trace[-1][0]):
            raise ValueError("measurement evidence timestamps must increase")
        area = sum(self._cells[cell_id].represented_area_m2 for cell_id in self._visited_cells)
        self.coverage_trace.append([round(task_time_s, 9), visited_2d, visited_3d])
        self.inspection_coverage_trace.append([round(task_time_s, 9), round(area, 6)])
        self.inspection_cell_count_trace.append([round(task_time_s, 9), len(self._visited_cells)])

    def snapshot(
        self,
        *,
        measured_state_trace: list[dict[str, Any]],
        input_bindings_hash: str,
    ) -> dict[str, Any]:
        """Return the hash-bound raw source used by the ancestor aggregator."""

        if not self.coverage_trace:
            raise ValueError("L1 measurement evidence has no measured state trace")
        return {
            "schema": EVIDENCE_SCHEMA,
            "coverage_semantics": FREE_SPACE_SEMANTICS,
            "inspection_footprint_semantics": INSPECTION_SEMANTICS,
            "coverage_resolution_m": _RESOLUTION_M,
            "input_bindings_hash": input_bindings_hash,
            "coverage_trace": self.coverage_trace,
            "inspection_coverage_trace": self.inspection_coverage_trace,
            "inspection_cell_count_trace": self.inspection_cell_count_trace,
            "coverage_denominators": self.coverage_denominators,
            "measured_state_trace_hash": content_hash(measured_state_trace),
        }
