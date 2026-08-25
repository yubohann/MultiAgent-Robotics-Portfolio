"""Deterministic L0 fleet runtime for training, debugging, and contract tests.

L0 is never a formal leaderboard backend.  Formal ordinary-v3 scores require
the L1 Isaac runtime and must carry an L1 execution receipt.
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .canonical import content_hash, derived_seed
from .contracts import (
    ActionPacket,
    BudgetLedger,
    ExecutionReceipt,
    FailureRecord,
    MessagePacket,
    ObservationPacket,
    ObservationReceipt,
    Pose3D,
)
from .evaluator import PrivateEvaluator
from .geometry import (
    Vec3,
    colliders_from_city,
    distance,
    in_field_of_view,
    line_of_sight,
    minimum_segment_clearance,
    segment_intersection_fraction,
    segment_segment_distance,
    sensor_pose,
    surface_facing,
)
from .inspection_atlas import (
    validate_public_inspection_atlas,
    validate_public_mission_sector,
)
from .ordinary_config import OrdinaryReleaseConfig
from .planning_cadence import PlanningCadenceController
from .public_boundary import validate_public_episode


@dataclass
class _DroneState:
    drone_id: str
    pose: Pose3D
    home: Pose3D
    sensor_pitch_deg: float = 0.0
    velocity: Vec3 = (0.0, 0.0, 0.0)
    angular_speed_deg_s: float = 0.0
    energy_remaining_j: float = 0.0
    terminal: bool = False
    terminal_reason: str | None = None
    observation_sequence: int = 0
    action_sequence: int = 0
    consecutive_deadline_misses: int = 0
    out_of_bounds_actions: int = 0
    visited_voxels: set[tuple[int, int, int]] = field(default_factory=set)


@dataclass(frozen=True)
class _PublicAtlasCell:
    cell_id: str
    pose: Pose3D
    surface_point: Vec3
    surface_normal: Vec3
    represented_area_m2: float
    lateral_tolerance_m: float
    vertical_tolerance_m: float


@dataclass
class _PublicAtlasDwellState:
    started_at_s: float
    last_seen_at_s: float
    initial_position: Vec3


@dataclass(frozen=True)
class RuntimeStep:
    observations: dict[str, ObservationPacket]
    confirmations: tuple[dict[str, Any], ...]
    execution_receipts: tuple[ExecutionReceipt, ...]
    failures: tuple[FailureRecord, ...]
    task_time_s: float
    done: bool


def _wrap_angle_deg(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _shortest_angle_delta_deg(start: float, end: float) -> float:
    return _wrap_angle_deg(float(end) - float(start))


class L0FleetRuntime:
    """Approximate kinematics with exact public execution semantics."""

    execution_level = "L0"
    formal_score_eligible = False
    coverage_resolution_m = 2.0

    def __init__(
        self,
        config: OrdinaryReleaseConfig,
        city: dict[str, Any],
        private_episode: dict[str, Any],
        *,
        receipt_secret: bytes = b"development-only-evaluator-secret",
        public_task_spec: dict[str, Any] | None = None,
        public_episode: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        # The authority runtime owns an immutable-in-practice snapshot.  A
        # policy, adapter, or caller must never be able to change the task
        # that a live evaluator is scoring by mutating a dictionary it passed
        # during setup.
        self.city = copy.deepcopy(city)
        self.private_episode = copy.deepcopy(private_episode)
        self.public_task_spec = copy.deepcopy(public_task_spec)
        self.public_episode = copy.deepcopy(public_episode)
        self.episode_id = str(self.private_episode["episode_id"])
        self._validate_public_episode_binding()
        self.evaluator = PrivateEvaluator(
            config, self.city, self.private_episode, receipt_secret=receipt_secret
        )
        self.task_time_s = 0.0
        self._step_index = 0
        self._colliders = colliders_from_city(self.city)
        self._states: dict[str, _DroneState] = {}
        energy_budget = float(config.raw["execution_contract"]["vehicle"]["energy_budget_j"])
        for start in self.private_episode["starts"]:
            pose = Pose3D(
                position=tuple(float(value) for value in start["position"]),
                yaw_deg=float(start["yaw_deg"]),
            )
            state = _DroneState(
                drone_id=str(start["drone_id"]),
                pose=pose,
                home=pose,
                energy_remaining_j=energy_budget,
            )
            self._assert_start_clear(state)
            self._states[state.drone_id] = state
        self._pending_messages: deque[tuple[float, str, MessagePacket]] = deque()
        self._seen_message_ids: set[str] = set()
        self._delivered_messages: dict[str, list[MessagePacket]] = {
            drone_id: [] for drone_id in self._states
        }
        self.ledger = BudgetLedger()
        self.execution_receipts: list[ExecutionReceipt] = []
        self._last_receipt_hash_by_drone: dict[str, str] = {}
        self.failures: list[FailureRecord] = []
        self.confirmation_log: list[dict[str, Any]] = []
        self.coverage_trace: list[tuple[float, int, int]] = []
        self.coverage_denominators = self._coverage_grid_denominators()
        self._public_atlas_cells = self._load_public_atlas_cells(public_task_spec)
        self._visited_public_atlas_cells: set[str] = set()
        self._public_atlas_dwell: dict[tuple[str, str], _PublicAtlasDwellState] = {}
        self.inspection_coverage_trace: list[tuple[float, float]] = []
        self.inspection_cell_count_trace: list[tuple[float, int]] = []
        self.coverage_denominators["inspection_atlas_cells"] = len(self._public_atlas_cells)
        self.coverage_denominators["inspection_atlas_area_m2"] = round(
            sum(cell.represented_area_m2 for cell in self._public_atlas_cells.values()), 6
        )
        self._latest_observations = self._make_observations()

    def _load_public_atlas_cells(
        self,
        public_task_spec: dict[str, Any] | None,
    ) -> dict[str, _PublicAtlasCell]:
        """Load target-independent cells used by the strict public coverage diagnostic."""

        if not isinstance(public_task_spec, dict) or public_task_spec.get("task_track") != "G2-I":
            return {}
        atlas = public_task_spec.get("inspection_atlas")
        if not isinstance(atlas, dict):
            raise ValueError("G2-I task spec lacks its public inspection atlas")
        validate_public_inspection_atlas(atlas)
        selected_cell_ids: set[str] | None = None
        mission_sector = self.public_episode.get("mission_sector")
        if mission_sector is not None:
            if not isinstance(mission_sector, dict):
                raise ValueError("G2-I episode mission sector must be an object")
            validate_public_mission_sector(
                mission_sector,
                atlas,
                self.public_episode["starts"],
                self.config.raw["execution_contract"],
            )
            if self.public_episode.get("mission_sector_hash") != mission_sector.get(
                "sector_hash"
            ):
                raise ValueError("G2-I public episode mission-sector binding differs")
            selected_cell_ids = {
                str(value) for value in mission_sector["selected_cell_ids"]
            }
        atlas_observe = atlas["observation_contract"]
        active_observe = self.config.raw["execution_contract"]["observe"]
        observation_bindings = {
            "maximum_range_m": "max_range_m",
            "horizontal_fov_deg": "horizontal_fov_deg",
            "vertical_fov_deg": "vertical_fov_deg",
            "continuous_dwell_s": "continuous_dwell_s",
            "minimum_clearance_m": None,
        }
        for atlas_key, active_key in observation_bindings.items():
            active_value = (
                self.config.raw["execution_contract"]["vehicle"]["minimum_clearance_m"]
                if active_key is None
                else active_observe[active_key]
            )
            if not math.isclose(
                float(atlas_observe[atlas_key]), float(active_value), abs_tol=1.0e-6
            ):
                raise ValueError("G2-I atlas observation contract differs from active runtime")
        cells: dict[str, _PublicAtlasCell] = {}
        for region in atlas.get("regions", []):
            for cell in region.get("cells", []):
                if (
                    selected_cell_ids is not None
                    and str(cell.get("cell_id", "")) not in selected_cell_ids
                ):
                    continue
                pose_data = cell.get("pose")
                envelope = cell.get("pose_envelope")
                if not isinstance(pose_data, dict) or not isinstance(envelope, dict):
                    raise ValueError("G2-I atlas cell is missing public pose envelope")
                pose = Pose3D.from_dict(pose_data)
                lateral_tolerance = float(envelope.get("lateral_tolerance_m", 0.0))
                vertical_tolerance = float(envelope.get("vertical_tolerance_m", 0.0))
                nominal_standoff = float(envelope.get("nominal_standoff_m", 0.0))
                represented_area = float(cell.get("represented_area_m2", 0.0))
                if min(
                    lateral_tolerance,
                    vertical_tolerance,
                    nominal_standoff,
                    represented_area,
                ) <= 0.0:
                    raise ValueError("G2-I atlas cell pose envelope must be positive")
                cell_id = str(cell.get("cell_id", ""))
                if not cell_id or cell_id in cells:
                    raise ValueError("G2-I atlas cell IDs must be unique")
                normal_x, normal_y, normal_z = (
                    float(value) for value in cell["surface_normal"]
                )
                surface_normal: Vec3 = (normal_x, normal_y, normal_z)
                surface_x, surface_y, surface_z = (
                    float(value) for value in cell["surface_point"]
                )
                surface_point: Vec3 = (surface_x, surface_y, surface_z)
                cells[cell_id] = _PublicAtlasCell(
                    cell_id=cell_id,
                    pose=pose,
                    surface_point=surface_point,
                    surface_normal=surface_normal,
                    represented_area_m2=represented_area,
                    lateral_tolerance_m=lateral_tolerance,
                    vertical_tolerance_m=vertical_tolerance,
                )
        if selected_cell_ids is not None and set(cells) != selected_cell_ids:
            raise ValueError("G2-I runtime mission sector does not resolve exactly")
        return cells

    def _validate_public_episode_binding(self) -> None:
        """Bind evaluator-private state to the exact episode seen by the method.

        The L0 runtime is authority-side because it owns the private evaluator.
        It must nevertheless never use private-only state to select the public
        inspection workload.  Shared identity, roster, and mission-sector
        fields are compared before reset so a mismatched projection cannot be
        silently scored.
        """

        public_task_declares_g2_i = (
            isinstance(self.public_task_spec, dict)
            and self.public_task_spec.get("task_track") == "G2-I"
        )
        private_episode_declares_g2_i = "mission_sector" in self.private_episode
        is_g2_i = public_task_declares_g2_i or private_episode_declares_g2_i
        if not is_g2_i:
            return
        if not public_task_declares_g2_i:
            raise ValueError("G2-I runtime requires the method-visible public task spec")
        if not isinstance(self.public_episode, dict):
            raise ValueError("G2-I runtime requires the method-visible public episode")
        validate_public_episode(self.public_episode, self.public_task_spec)
        for binding_field in ("episode_id", "layout_id", "fleet_profile", "starts"):
            if binding_field not in self.private_episode:
                raise ValueError(f"G2-I private episode lacks {binding_field}")
            public_value = content_hash(self.public_episode[binding_field])
            private_value = content_hash(self.private_episode[binding_field])
            if public_value != private_value:
                raise ValueError(f"G2-I public episode binding differs for {binding_field}")
        public_sector = self.public_episode.get("mission_sector")
        private_sector = self.private_episode.get("mission_sector")
        if (public_sector is None) != (private_sector is None):
            raise ValueError("G2-I public/private mission-sector presence differs")
        if public_sector is not None and (
            content_hash(public_sector) != content_hash(private_sector)
        ):
            raise ValueError("G2-I public/private mission-sector binding differs")

    def _reset_public_atlas_dwell(
        self, drone_id: str, *, except_cells: set[str] | None = None
    ) -> None:
        retained = except_cells or set()
        for key in list(self._public_atlas_dwell):
            if key[0] == drone_id and key[1] not in retained:
                del self._public_atlas_dwell[key]

    def _public_cell_visible(
        self, observation: ObservationPacket, cell: _PublicAtlasCell
    ) -> bool:
        # A public cell is an area-bearing inspection domain, not a mandatory
        # waypoint.  Baselines are allowed to refine the nominal pose using
        # public local occupancy, and a valid observation may therefore move
        # outside the nominal pose envelope while still inspecting the same
        # surface footprint.  Credit is constrained by the actual sensor
        # geometry below (range, FoV, facing, LOS) plus dwell and safety checks
        # in _record_public_atlas_observation.
        contract = self.config.raw["execution_contract"]
        body_margin = float(contract["vehicle"]["radius_m"]) + float(
            contract["vehicle"]["minimum_clearance_m"]
        )
        if any(
            collider.expanded(body_margin).contains(observation.pose.position)
            for collider in self._colliders
        ):
            return False
        rig = contract["sensor_rig"]
        camera_pose = sensor_pose(
            observation.pose,
            rig["translation_body_m"],
            sensor_pitch_deg=(
                observation.sensor_pitch_deg if rig["gimbal_mode"] == "bounded" else None
            ),
        )
        observe = contract["observe"]
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

    def _record_public_atlas_observation(
        self, observation: ObservationPacket, receipt: ObservationReceipt
    ) -> None:
        drone_id = observation.drone_id
        if not receipt.accepted:
            self._reset_public_atlas_dwell(drone_id)
            return
        eligible = {
            cell_id
            for cell_id, cell in self._public_atlas_cells.items()
            if cell_id not in self._visited_public_atlas_cells
            and self._public_cell_visible(observation, cell)
        }
        self._reset_public_atlas_dwell(drone_id, except_cells=eligible)
        observe = self.config.raw["execution_contract"]["observe"]
        control_period = float(self.config.raw["execution_contract"]["control_period_s"])
        for cell_id in sorted(eligible):
            key = (drone_id, cell_id)
            state = self._public_atlas_dwell.get(key)
            if (
                state is None
                or observation.timestamp_s - state.last_seen_at_s > control_period * 1.6
                or distance(state.initial_position, observation.pose.position)
                > float(observe["max_pose_drift_m"])
            ):
                state = _PublicAtlasDwellState(
                    started_at_s=observation.timestamp_s,
                    last_seen_at_s=observation.timestamp_s,
                    initial_position=observation.pose.position,
                )
                self._public_atlas_dwell[key] = state
            else:
                state.last_seen_at_s = observation.timestamp_s
            if (
                state.last_seen_at_s - state.started_at_s + 1.0e-9
                >= float(observe["continuous_dwell_s"])
            ):
                self._visited_public_atlas_cells.add(cell_id)
                self._public_atlas_dwell.pop(key, None)

    def _coverage_grid_denominators(self) -> dict[str, int]:
        bounds = self.city["flight_bounds"]
        resolution = self.coverage_resolution_m
        axes = [
            range(math.ceil(float(low) / resolution), math.floor(float(high) / resolution) + 1)
            for low, high in zip(bounds["minimum"], bounds["maximum"], strict=True)
        ]
        total_2d = len(axes[0]) * len(axes[1])
        total_3d = total_2d * len(axes[2])
        occupied: set[tuple[int, int, int]] = set()
        radius = float(self.config.raw["execution_contract"]["vehicle"]["radius_m"])
        for collider in self._colliders:
            expanded = collider.expanded(radius)
            index_ranges = [
                range(
                    max(axes[index].start, math.ceil(expanded.minimum[index] / resolution)),
                    min(axes[index].stop - 1, math.floor(expanded.maximum[index] / resolution)) + 1,
                )
                for index in range(3)
            ]
            for x_index in index_ranges[0]:
                for y_index in index_ranges[1]:
                    for z_index in index_ranges[2]:
                        point = (
                            x_index * resolution,
                            y_index * resolution,
                            z_index * resolution,
                        )
                        if expanded.contains(point):
                            occupied.add((x_index, y_index, z_index))
        return {
            "coverage_2d_cells": total_2d,
            "coverage_3d_free_cells": total_3d - len(occupied),
        }

    def _assert_start_clear(self, state: _DroneState) -> None:
        radius = float(self.config.raw["execution_contract"]["vehicle"]["radius_m"])
        if any(
            collider.expanded(radius).contains(state.pose.position) for collider in self._colliders
        ):
            raise ValueError(f"start pose intersects a collider: {state.drone_id}")

    def _local_occupancy(
        self, position: Vec3
    ) -> tuple[Vec3, float, float, tuple[tuple[int, int, int], ...]]:
        cells: set[tuple[int, int, int]] = set()
        radius = 14.0
        resolution = 2.0
        origin = tuple(round(value / resolution) * resolution for value in position)
        index_limit = math.ceil(radius / resolution)
        for collider in self._colliders:
            if collider.point_distance(position) > radius:
                continue
            ranges = []
            for axis in range(3):
                low = math.ceil(
                    (collider.minimum[axis] - origin[axis]) / resolution - 0.5
                )
                high = math.floor(
                    (collider.maximum[axis] - origin[axis]) / resolution + 0.5
                )
                ranges.append(
                    range(max(-index_limit, low), min(index_limit, high) + 1)
                )
            for x_index in ranges[0]:
                for y_index in ranges[1]:
                    for z_index in ranges[2]:
                        center = (
                            origin[0] + x_index * resolution,
                            origin[1] + y_index * resolution,
                            origin[2] + z_index * resolution,
                        )
                        if distance(position, center) <= radius + math.sqrt(3.0) * resolution / 2.0:
                            cells.add((x_index, y_index, z_index))
        return origin, resolution, radius, tuple(sorted(cells))  # type: ignore[return-value]

    def _make_observations(self) -> dict[str, ObservationPacket]:
        observations: dict[str, ObservationPacket] = {}
        communication_range = float(
            self.config.raw["execution_contract"]["communication"]["range_m"]
        )
        for state in self._states.values():
            if state.terminal:
                continue
            teammate_public = [
                {
                    "drone_id": other.drone_id,
                    "position": list(other.pose.position),
                    "health": "terminal" if other.terminal else "nominal",
                }
                for other in sorted(self._states.values(), key=lambda item: item.drone_id)
                if other.drone_id != state.drone_id
                and distance(state.pose.position, other.pose.position) <= communication_range
            ]
            observation_payload = [
                self.episode_id,
                state.drone_id,
                state.observation_sequence,
                self.task_time_s,
            ]
            observation_id = f"obs-{content_hash(observation_payload)[:20]}"
            occupancy_origin, occupancy_resolution, occupancy_radius, occupancy_cells = (
                self._local_occupancy(state.pose.position)
            )
            observations[state.drone_id] = ObservationPacket(
                episode_id=self.episode_id,
                observation_id=observation_id,
                drone_id=state.drone_id,
                sequence=state.observation_sequence,
                timestamp_s=self.task_time_s,
                pose=state.pose,
                linear_velocity_world_mps=state.velocity,
                angular_speed_deg_s=state.angular_speed_deg_s,
                energy_remaining_j=max(0.0, state.energy_remaining_j),
                local_occupancy=occupancy_cells,
                local_occupancy_origin_world_m=occupancy_origin,
                local_occupancy_resolution_m=occupancy_resolution,
                local_occupancy_radius_m=occupancy_radius,
                teammate_states=tuple(teammate_public),
                received_messages=tuple(self._delivered_messages[state.drone_id]),
                health="terminal" if state.terminal else "nominal",
                sensor_pitch_deg=state.sensor_pitch_deg,
            )
            state.observation_sequence += 1
            self._delivered_messages[state.drone_id].clear()
        return observations

    def reset(self) -> dict[str, ObservationPacket]:
        return dict(self._latest_observations)

    def public_inspection_state(self) -> dict[str, Any]:
        """Return target-free inspection history for public planners and RL wrappers."""

        ordered_ids = sorted(self._public_atlas_cells)
        visited_ids = sorted(self._visited_public_atlas_cells)
        visited_area = sum(
            self._public_atlas_cells[cell_id].represented_area_m2
            for cell_id in visited_ids
        )
        total_area = float(self.coverage_denominators["inspection_atlas_area_m2"])
        return {
            "schema": "org.aerocity.bench.public-inspection-state.v1",
            "cell_ids": ordered_ids,
            "visited_cell_ids": visited_ids,
            "visited_cell_mask": [
                cell_id in self._visited_public_atlas_cells for cell_id in ordered_ids
            ],
            "visited_area_m2": round(visited_area, 6),
            "total_area_m2": total_area,
            "area_fraction": (0.0 if total_area <= 0.0 else visited_area / total_area),
        }

    @staticmethod
    def _receipt_state_payload(state: _DroneState) -> dict[str, Any]:
        """Project only execution-authority state into the per-agent receipt chain."""

        return {
            "drone_id": state.drone_id,
            "pose": state.pose.to_dict(),
            "sensor_pitch_deg": state.sensor_pitch_deg,
            "linear_velocity_world_mps": list(state.velocity),
            "angular_speed_deg_s": state.angular_speed_deg_s,
            "energy_remaining_j": state.energy_remaining_j,
            "terminal": state.terminal,
            "terminal_reason": state.terminal_reason,
            "action_sequence": state.action_sequence,
        }

    def _in_bounds(self, point: Vec3) -> bool:
        bounds = self.city["flight_bounds"]
        return all(
            float(low) <= value <= float(high)
            for value, low, high in zip(point, bounds["minimum"], bounds["maximum"], strict=True)
        )

    def _collision(self, start: Vec3, end: Vec3, state: _DroneState) -> tuple[bool, str | None]:
        vehicle = self.config.raw["execution_contract"]["vehicle"]
        margin = float(vehicle["radius_m"])
        for collider in self._colliders:
            if segment_intersection_fraction(start, end, collider.expanded(margin)) is not None:
                return True, collider.collider_id
        return False, None

    def _requested_destination(self, state: _DroneState, action: ActionPacket) -> Pose3D:
        period = float(self.config.raw["execution_contract"]["control_period_s"])
        vehicle = self.config.raw["execution_contract"]["vehicle"]
        if action.kind in {"HOVER", "OBSERVE"}:
            return state.pose
        target = state.home if action.kind == "RETURN" else action.waypoint
        if action.kind == "VELOCITY":
            velocity = action.velocity_body_mps or (0.0, 0.0, 0.0)
            yaw = math.radians(state.pose.yaw_deg)
            world_velocity = (
                velocity[0] * math.cos(yaw) - velocity[1] * math.sin(yaw),
                velocity[0] * math.sin(yaw) + velocity[1] * math.cos(yaw),
                velocity[2],
            )
            target_position = tuple(
                current + delta * period
                for current, delta in zip(state.pose.position, world_velocity, strict=True)
            )
            return Pose3D(
                target_position,
                _wrap_angle_deg(state.pose.yaw_deg + action.yaw_rate_deg_s * period),
            )
        if target is None:
            return state.pose
        delta = tuple(
            desired - current
            for desired, current in zip(target.position, state.pose.position, strict=True)
        )
        horizontal_distance = math.hypot(delta[0], delta[1])
        horizontal_limit = float(vehicle["horizontal_speed_mps"]) * period
        vertical_limit = float(vehicle["vertical_speed_mps"]) * period
        horizontal_scale = (
            min(1.0, horizontal_limit / horizontal_distance)
            if horizontal_distance > 1.0e-9
            else 0.0
        )
        move = (
            delta[0] * horizontal_scale,
            delta[1] * horizontal_scale,
            max(-vertical_limit, min(vertical_limit, delta[2])),
        )
        yaw_delta = _shortest_angle_delta_deg(state.pose.yaw_deg, target.yaw_deg)
        yaw_limit = float(vehicle["yaw_rate_deg_s"]) * period
        yaw = _wrap_angle_deg(
            state.pose.yaw_deg + max(-yaw_limit, min(yaw_limit, yaw_delta))
        )
        return Pose3D(
            tuple(
                current + displacement
                for current, displacement in zip(state.pose.position, move, strict=True)
            ),
            yaw,
        )

    def _requested_sensor_pitch(self, state: _DroneState, action: ActionPacket) -> float:
        """Advance the public bounded camera gimbal independently of body attitude."""

        rig = self.config.raw["execution_contract"]["sensor_rig"]
        if rig["gimbal_mode"] == "fixed":
            if action.sensor_pitch_deg is not None:
                raise ValueError("fixed sensor rig cannot accept a gimbal pitch command")
            return state.pose.pitch_deg
        if action.kind == "OBSERVE" and action.sensor_pitch_deg is not None:
            raise ValueError("OBSERVE cannot move the bounded sensor gimbal")
        target = (
            state.sensor_pitch_deg
            if action.sensor_pitch_deg is None
            else float(action.sensor_pitch_deg)
        )
        lower, upper = (float(value) for value in rig["pitch_limits_deg"])
        if not lower <= target <= upper:
            raise ValueError("gimbal pitch command lies outside public contract limits")
        period = float(self.config.raw["execution_contract"]["control_period_s"])
        maximum_delta = float(rig["max_pitch_rate_deg_s"]) * period
        delta = max(-maximum_delta, min(maximum_delta, target - state.sensor_pitch_deg))
        return max(lower, min(upper, state.sensor_pitch_deg + delta))

    def _deliver_messages(self) -> None:
        communication = self.config.raw["execution_contract"]["communication"]
        rng = random.Random(
            derived_seed(self.private_episode["episode_seed"], "comm", self._step_index)
        )
        remaining: deque[tuple[float, str, MessagePacket]] = deque()
        while self._pending_messages:
            delivery_time, destination, message = self._pending_messages.popleft()
            if delivery_time > self.task_time_s:
                remaining.append((delivery_time, destination, message))
                continue
            if message.expires_at_s < self.task_time_s:
                self.ledger.communication_bytes_dropped += message.payload_bytes
                self.ledger.communication_packets_dropped += 1
                self.ledger.stale_messages_rejected += 1
                continue
            source_position = self._states[message.source_drone_id].pose.position
            destination_position = self._states[destination].pose.position
            if distance(source_position, destination_position) > float(communication["range_m"]):
                self.ledger.communication_bytes_dropped += message.payload_bytes
                self.ledger.communication_packets_dropped += 1
                continue
            if rng.random() < float(communication["drop_probability"]):
                self.ledger.communication_bytes_dropped += message.payload_bytes
                self.ledger.communication_packets_dropped += 1
                continue
            self._delivered_messages[destination].append(message)
            self.ledger.communication_bytes_delivered += message.payload_bytes
            self.ledger.communication_packets_delivered += 1
        self._pending_messages = remaining

    def _queue_messages(self, action: ActionPacket) -> None:
        communication = self.config.raw["execution_contract"]["communication"]
        payload_limit = int(communication["payload_bytes"])
        latency = float(communication["latency_s"])
        ttl = float(communication["ttl_s"])
        period = float(self.config.raw["execution_contract"]["control_period_s"])
        remaining_bandwidth = int(float(communication["bandwidth_bytes_s"]) * period)
        for message in action.messages:
            destinations = tuple(dict.fromkeys(message.destination_drone_ids))
            transmission_bytes = message.payload_bytes * len(destinations)
            invalid_identity = (
                message.source_drone_id != action.drone_id
                or message.message_id in self._seen_message_ids
                or len(destinations) != len(message.destination_drone_ids)
            )
            invalid_time = (
                abs(float(message.created_at_s) - float(action.issued_at_s)) > 1.0e-9
                or float(message.expires_at_s) - float(message.created_at_s) > ttl + 1.0e-9
                or float(message.expires_at_s) <= self.task_time_s
            )
            over_budget = (
                message.payload_bytes > payload_limit or transmission_bytes > remaining_bandwidth
            )
            if invalid_identity or invalid_time or over_budget:
                self.ledger.communication_bytes_dropped += transmission_bytes
                self.ledger.communication_packets_dropped += len(destinations)
                if message.message_id in self._seen_message_ids:
                    self.ledger.duplicate_messages_rejected += 1
                if invalid_time:
                    self.ledger.stale_messages_rejected += 1
                if over_budget:
                    self.ledger.bandwidth_messages_rejected += 1
                continue
            self._seen_message_ids.add(message.message_id)
            remaining_bandwidth -= transmission_bytes
            self.ledger.communication_bytes_sent += transmission_bytes
            self.ledger.communication_packets_sent += len(destinations)
            for destination in destinations:
                if destination not in self._states:
                    self.ledger.communication_bytes_dropped += message.payload_bytes
                    self.ledger.communication_packets_dropped += 1
                    continue
                self._pending_messages.append((self.task_time_s + latency, destination, message))

    def _terminate(self, state: _DroneState, reason: str, detail: str) -> FailureRecord:
        state.terminal = True
        state.terminal_reason = reason
        state.velocity = (0.0, 0.0, 0.0)
        failure = FailureRecord(
            episode_id=self.episode_id,
            drone_id=state.drone_id,
            task_time_s=self.task_time_s,
            category=reason,
            detail=detail,
            terminal=True,
        )
        self.failures.append(failure)
        return failure

    def step(
        self,
        actions: dict[str, ActionPacket],
        *,
        planning_latencies_s: dict[str, float] | None = None,
        planner_invoked_by_drone: dict[str, bool] | None = None,
    ) -> RuntimeStep:
        planning_latencies_s = planning_latencies_s or {}
        planner_invoked_by_drone = planner_invoked_by_drone or {
            drone_id: True for drone_id in actions
        }
        expected = {drone_id for drone_id, state in self._states.items() if not state.terminal}
        if set(actions) != expected:
            raise ValueError(
                f"actions must exactly match active drones: expected={sorted(expected)}"
            )
        period = float(self.config.raw["execution_contract"]["control_period_s"])
        deadline = float(self.config.raw["execution_contract"]["planning_deadline_s"])
        clock = self.config.raw["execution_contract"]["clock"]
        vehicle = self.config.raw["execution_contract"]["vehicle"]
        safety = self.config.raw["execution_contract"]["safety"]
        step_start = self.task_time_s
        if set(planning_latencies_s) - expected:
            raise ValueError("planning latency contains an inactive or unknown drone")
        if set(planner_invoked_by_drone) != expected or any(
            not isinstance(value, bool) for value in planner_invoked_by_drone.values()
        ):
            raise ValueError("planner invocation flags must cover exactly the active drones")
        decisions: dict[str, dict[str, Any]] = {}
        for drone_id in sorted(expected):
            state = self._states[drone_id]
            action = actions[drone_id]
            source_observation = self._latest_observations[drone_id]
            if (
                action.sequence != state.action_sequence
                or action.sequence != source_observation.sequence
                or action.drone_id != drone_id
                or action.episode_id != self.episode_id
            ):
                raise ValueError(f"action identity/sequence mismatch for {drone_id}")
            if abs(float(action.issued_at_s) - source_observation.timestamp_s) > 1.0e-9:
                raise ValueError(
                    f"action timestamp is not bound to the latest observation: {drone_id}"
                )
            latency = float(planning_latencies_s.get(drone_id, 0.0))
            if not math.isfinite(latency) or latency < 0:
                raise ValueError(f"planning latency must be finite and non-negative: {drone_id}")
            deadline_miss = latency > deadline
            executed_kind = "HOVER" if deadline_miss else action.kind
            status = str(clock["overrun_policy"]) if deadline_miss else "executed"
            intervention = deadline_miss
            effective_action = ActionPacket(
                episode_id=action.episode_id,
                drone_id=action.drone_id,
                sequence=action.sequence,
                issued_at_s=action.issued_at_s,
                kind=("HOVER" if executed_kind == "OBSERVE" else executed_kind),  # type: ignore[arg-type]
                waypoint=action.waypoint if executed_kind == "WAYPOINT" else None,
                velocity_body_mps=(
                    action.velocity_body_mps if executed_kind == "VELOCITY" else None
                ),
                yaw_rate_deg_s=action.yaw_rate_deg_s,
                sensor_pitch_deg=(action.sensor_pitch_deg if not deadline_miss else None),
                source_observation_id=None,
            )
            requested = self._requested_destination(state, effective_action)
            requested_sensor_pitch = self._requested_sensor_pitch(state, effective_action)
            out_of_bounds = not self._in_bounds(requested.position)
            if out_of_bounds:
                requested = state.pose
                executed_kind = "HOVER"
                status = str(safety["out_of_bounds_policy"])
                intervention = True
            collision, collision_id = self._collision(
                state.pose.position, requested.position, state
            )
            center_clearance, clearance_id = minimum_segment_clearance(
                state.pose.position, requested.position, self._colliders
            )
            body_clearance = max(0.0, center_clearance - float(vehicle["radius_m"]))
            clearance_intervention = False
            if not collision and body_clearance + 1.0e-9 < float(vehicle["minimum_clearance_m"]):
                requested = state.pose
                executed_kind = "HOVER"
                status = f"minimum_clearance_intervention:{clearance_id}"
                intervention = True
                clearance_intervention = True
                center_clearance, _ = minimum_segment_clearance(
                    state.pose.position, state.pose.position, self._colliders
                )
                body_clearance = max(0.0, center_clearance - float(vehicle["radius_m"]))
            decisions[drone_id] = {
                "action": action,
                "source_observation": source_observation,
                "action_packet_hash": content_hash(action.to_dict()),
                "source_observation_hash": content_hash(source_observation.to_dict()),
                "state_before_hash": content_hash(self._receipt_state_payload(state)),
                "latency": latency,
                "deadline_miss": deadline_miss,
                "executed_kind": executed_kind,
                "status": status,
                "safety_intervention": intervention,
                "out_of_bounds": out_of_bounds,
                "collision": collision,
                "collision_id": collision_id,
                "requested": requested,
                "requested_sensor_pitch_deg": requested_sensor_pitch,
                "minimum_clearance_m": body_clearance,
                "clearance_intervention": clearance_intervention,
            }
        pair_collisions: dict[str, str] = {}
        active_ids = sorted(expected)
        for first_index, first_id in enumerate(active_ids):
            first_state = self._states[first_id]
            first_decision = decisions[first_id]
            if first_decision["collision"]:
                continue
            for second_id in active_ids[first_index + 1 :]:
                second_state = self._states[second_id]
                second_decision = decisions[second_id]
                if second_decision["collision"]:
                    continue
                separation = segment_segment_distance(
                    first_state.pose.position,
                    first_decision["requested"].position,
                    second_state.pose.position,
                    second_decision["requested"].position,
                )
                if separation < 2.0 * float(vehicle["radius_m"]):
                    pair_collisions[first_id] = second_id
                    pair_collisions[second_id] = first_id
        max_latency = max((float(value) for value in planning_latencies_s.values()), default=0.0)
        self.task_time_s += period + max(0.0, max_latency - deadline)
        step_confirmations: list[dict[str, Any]] = []
        step_receipts: list[ExecutionReceipt] = []
        step_failures: list[FailureRecord] = []
        for drone_id in sorted(expected):
            state = self._states[drone_id]
            decision = decisions[drone_id]
            action = decision["action"]
            state.action_sequence += 1
            latency = float(decision["latency"])
            deadline_miss = bool(decision["deadline_miss"])
            self.ledger.planning_time_s += latency
            if deadline_miss:
                state.consecutive_deadline_misses += 1
                self.ledger.deadline_misses += 1
            else:
                state.consecutive_deadline_misses = 0
            executed_kind = str(decision["executed_kind"])
            status = str(decision["status"])
            safety_intervention = bool(decision["safety_intervention"])
            out_of_bounds = bool(decision["out_of_bounds"])
            collision = bool(decision["collision"] or drone_id in pair_collisions)
            collision_id = decision["collision_id"] or pair_collisions.get(drone_id)
            minimum_clearance = float(decision["minimum_clearance_m"])
            confirmation_ids: list[str] = []
            source_observation = decision["source_observation"]
            if deadline_miss:
                self.evaluator.end_observe(drone_id, source_observation.timestamp_s)
            if safety_intervention:
                self.ledger.safety_interventions += 1
            if decision["clearance_intervention"]:
                self.ledger.clearance_interventions += 1
            if out_of_bounds:
                state.out_of_bounds_actions += 1
                self.ledger.out_of_bounds_actions += 1
            if action.kind == "OBSERVE" and not deadline_miss:
                observation_receipt, confirmations = self.evaluator.process(
                    source_observation, action
                )
                if collision or out_of_bounds or safety_intervention:
                    self._reset_public_atlas_dwell(drone_id)
                else:
                    self._record_public_atlas_observation(
                        source_observation, observation_receipt
                    )
                for confirmation in confirmations:
                    public = confirmation.to_dict()
                    step_confirmations.append(public)
                    self.confirmation_log.append(public)
                    confirmation_ids.append(confirmation.confirmation_id)
            else:
                self.evaluator.end_observe(drone_id, source_observation.timestamp_s)
                self._reset_public_atlas_dwell(drone_id)
            requested = decision["requested"]
            moved = distance(state.pose.position, requested.position)
            if collision:
                moved = 0.0
                requested = state.pose
                self.ledger.collisions += 1
                step_failures.append(
                    self._terminate(state, "collision", f"collision with {collision_id}")
                )
                status = "agent_terminal_collision"
            else:
                state.velocity = tuple(
                    (end - start) / period
                    for start, end in zip(state.pose.position, requested.position, strict=True)
                )
                state.angular_speed_deg_s = abs(
                    _shortest_angle_delta_deg(state.pose.yaw_deg, requested.yaw_deg)
                ) / period
                state.pose = requested
            state.sensor_pitch_deg = float(decision["requested_sensor_pitch_deg"])
            hover_time = period if moved <= 1.0e-9 else 0.0
            energy_used = moved * float(vehicle["energy_per_meter_j"]) + hover_time * float(
                vehicle["hover_power_w"]
            )
            state.energy_remaining_j -= energy_used
            self.ledger.path_distance_m += moved
            self.ledger.energy_used_j += energy_used
            if self.ledger.minimum_clearance_m is None:
                self.ledger.minimum_clearance_m = minimum_clearance
            else:
                self.ledger.minimum_clearance_m = min(
                    self.ledger.minimum_clearance_m, minimum_clearance
                )
            if state.energy_remaining_j <= 0 and not state.terminal:
                step_failures.append(
                    self._terminate(state, "energy_exhausted", "energy budget reached zero")
                )
            if (
                state.consecutive_deadline_misses >= int(clock["max_consecutive_deadline_misses"])
                and not state.terminal
            ):
                step_failures.append(
                    self._terminate(
                        state,
                        "deadline_failure",
                        "too many consecutive planning deadline misses",
                    )
                )
            if (
                state.out_of_bounds_actions >= int(safety["max_out_of_bounds_actions"])
                and not state.terminal
            ):
                step_failures.append(
                    self._terminate(
                        state,
                        "out_of_bounds_failure",
                        "too many rejected out-of-bounds actions",
                    )
                )
            if not deadline_miss and not collision and not out_of_bounds:
                self._queue_messages(action)
            previous_receipt_hash = self._last_receipt_hash_by_drone.get(drone_id)
            receipt = ExecutionReceipt(
                episode_id=self.episode_id,
                drone_id=drone_id,
                action_sequence=action.sequence,
                task_time_start_s=step_start,
                task_time_end_s=self.task_time_s,
                planning_latency_s=latency,
                action_requested=action.kind,
                action_executed=executed_kind,
                status=status,
                distance_m=moved,
                energy_used_j=energy_used,
                minimum_clearance_m=minimum_clearance,
                collision=collision,
                out_of_bounds=out_of_bounds,
                safety_intervention=safety_intervention,
                deadline_miss=deadline_miss,
                execution_level="L0",
                action_packet_hash=str(decision["action_packet_hash"]),
                source_observation_id=source_observation.observation_id,
                source_observation_hash=str(decision["source_observation_hash"]),
                state_before_hash=str(decision["state_before_hash"]),
                state_after_hash=content_hash(self._receipt_state_payload(state)),
                previous_receipt_hash=previous_receipt_hash,
                confirmation_ids=tuple(confirmation_ids),
                planner_invoked=planner_invoked_by_drone[drone_id],
            )
            self.execution_receipts.append(receipt)
            self._last_receipt_hash_by_drone[drone_id] = str(
                receipt.to_dict()["receipt_hash"]
            )
            step_receipts.append(receipt)
        self._step_index += 1
        self._deliver_messages()
        duration = float(self.config.raw["execution_contract"]["episode"]["duration_s"])
        done = self.task_time_s >= duration or all(
            state.terminal for state in self._states.values()
        )
        for state in self._states.values():
            state.visited_voxels.add(tuple(round(value / 2.0) for value in state.pose.position))
        visited_3d = len(set().union(*(state.visited_voxels for state in self._states.values())))
        visited_2d = len(
            {(cell[0], cell[1]) for state in self._states.values() for cell in state.visited_voxels}
        )
        self.coverage_trace.append((self.task_time_s, visited_2d, visited_3d))
        self.inspection_coverage_trace.append(
            (
                self.task_time_s,
                round(
                    sum(
                        self._public_atlas_cells[cell_id].represented_area_m2
                        for cell_id in self._visited_public_atlas_cells
                    ),
                    6,
                ),
            )
        )
        self.inspection_cell_count_trace.append(
            (self.task_time_s, len(self._visited_public_atlas_cells))
        )
        self._latest_observations = self._make_observations()
        return RuntimeStep(
            observations=dict(self._latest_observations),
            confirmations=tuple(step_confirmations),
            execution_receipts=tuple(step_receipts),
            failures=tuple(step_failures),
            task_time_s=self.task_time_s,
            done=done,
        )

    def run_policy(
        self,
        policy: Callable[[dict[str, ObservationPacket]], dict[str, ActionPacket]],
        *,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        observations = self.reset()
        cadence = PlanningCadenceController.from_execution_contract(
            self.config.raw["execution_contract"]
        )
        limit = max_steps or math.ceil(
            float(self.config.raw["execution_contract"]["episode"]["duration_s"])
            / float(self.config.raw["execution_contract"]["control_period_s"])
        )
        wall_start = time.perf_counter()
        deadline = float(self.config.raw["execution_contract"]["planning_deadline_s"])
        episode = self.config.raw["execution_contract"]["episode"]
        return_event_s = float(episode["duration_s"]) - float(episode["return_reserve_s"])
        return_event_emitted = False
        for control_tick in range(limit):
            if not return_event_emitted and self.task_time_s >= return_event_s:
                cadence.request_event("return_reserve_entry")
                return_event_emitted = True
            due_reasons = cadence.due_reasons(
                control_tick=control_tick,
                active_drone_ids=tuple(observations),
            )
            planner_invoked = bool(due_reasons)
            elapsed = 0.0
            if planner_invoked:
                planning_start = time.perf_counter()
                try:
                    actions = policy(observations)
                except Exception as exc:
                    for state in self._states.values():
                        if not state.terminal:
                            self._terminate(
                                state,
                                "method_failure",
                                f"{type(exc).__name__}: {exc}",
                            )
                    break
                elapsed = time.perf_counter() - planning_start
                if elapsed <= deadline:
                    cadence.approve(actions)
                else:
                    cadence.reject_planning_attempt()
            else:
                actions = cadence.held_actions(observations)
            result = self.step(
                actions,
                planning_latencies_s={drone_id: elapsed for drone_id in actions},
                planner_invoked_by_drone={
                    drone_id: planner_invoked for drone_id in actions
                },
            )
            if result.confirmations:
                cadence.request_event("anonymous_confirmation")
            if any(receipt.safety_intervention for receipt in result.execution_receipts):
                cadence.request_event("safety_intervention")
            observations = result.observations
            if result.done:
                break
        return self.result(wall_clock_s=time.perf_counter() - wall_start)

    def result(self, *, wall_clock_s: float | None = None) -> dict[str, Any]:
        receipt_records = [receipt.to_dict() for receipt in self.execution_receipts]
        return {
            "schema": "org.aerocity.bench.run-result.v1",
            "episode_id": self.episode_id,
            "execution_level": self.execution_level,
            "formal_score_eligible": self.formal_score_eligible,
            "task_time_s": self.task_time_s,
            "wall_clock_s": wall_clock_s,
            "confirmations": list(self.confirmation_log),
            "execution_receipts": receipt_records,
            "execution_receipt_set_hash": content_hash(receipt_records),
            "execution_contract_hash": content_hash(self.config.raw["execution_contract"]),
            "public_task_spec_hash": (
                content_hash(self.public_task_spec) if self.public_task_spec is not None else None
            ),
            "public_episode_hash": (
                content_hash(self.public_episode) if self.public_episode is not None else None
            ),
            "failures": [failure.to_dict() for failure in self.failures],
            "budget_ledger": self.ledger.to_dict(),
            "coverage_trace": [list(item) for item in self.coverage_trace],
            "inspection_coverage_trace": [list(item) for item in self.inspection_coverage_trace],
            "inspection_cell_count_trace": [
                list(item) for item in self.inspection_cell_count_trace
            ],
            "coverage_denominators": dict(self.coverage_denominators),
            "coverage_semantics": "L0_path_voxel_proxy",
            "evaluator_private_audit": self.evaluator.private_audit_snapshot(),
            "returned_home": {
                drone_id: (
                    not state.terminal
                    and distance(state.pose.position, state.home.position)
                    <= float(self.config.raw["execution_contract"]["vehicle"]["home_radius_m"])
                )
                for drone_id, state in sorted(self._states.items())
            },
        }
