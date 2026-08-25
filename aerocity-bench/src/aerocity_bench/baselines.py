"""Reference closed-loop policies for interface and difficulty calibration.

These implementations are benchmark-owned references, not reimplementations of
FUEL, RACER, FALCON, MARVEL, or any other upstream project.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Protocol

from .adapters import arbitrate_public_fleet_actions
from .canonical import derived_seed
from .contracts import ActionPacket, ObservationPacket, Pose3D
from .geometry import (
    AABB,
    distance,
    minimum_segment_clearance,
    pose_looking_at,
    segment_aabb_clearance,
    segment_intersects_expanded_aabb,
)
from .inspection_atlas import (
    ATLAS_PRIOR_COARSE,
    validate_inspection_atlas_projection,
    validate_public_mission_sector,
)
from .ordinary_config import OrdinaryReleaseConfig
from .public_boundary import validate_public_episode, validate_public_task_spec


class FleetPolicy(Protocol):
    method_id: str
    role: str

    def __call__(self, observations: dict[str, ObservationPacket]) -> dict[str, ActionPacket]: ...


@dataclass(frozen=True)
class BaselineDescriptor:
    method_id: str
    display_name: str
    category: str
    role: str
    observation_profile: str
    requires_private_truth: bool
    substantive_method: bool


BASELINES = {
    "random-safe": BaselineDescriptor(
        "random-safe", "Random Safe", "lower_bound", "diagnostic", "G1", False, False
    ),
    "sweep-2d": BaselineDescriptor(
        "sweep-2d", "Fixed-altitude 2D Sweep", "coverage", "diagnostic", "G1", False, False
    ),
    "sweep-3d": BaselineDescriptor(
        "sweep-3d", "Volumetric 3D Sweep", "coverage", "diagnostic", "G1", False, False
    ),
    "nearest-frontier": BaselineDescriptor(
        "nearest-frontier", "Nearest Frontier", "planning", "substantive", "G1", False, True
    ),
    "information-frontier": BaselineDescriptor(
        "information-frontier",
        "Information-gain Frontier",
        "planning",
        "substantive",
        "G1",
        False,
        True,
    ),
    "decentralized-auction": BaselineDescriptor(
        "decentralized-auction",
        "Decentralized Spatial Auction",
        "coordination",
        "substantive",
        "G1",
        False,
        True,
    ),
    "atlas-surface-inspector": BaselineDescriptor(
        "atlas-surface-inspector",
        "Budgeted Public Atlas Inspector",
        "inspection",
        "substantive",
        "G2-I",
        False,
        True,
    ),
    "atlas-region-greedy": BaselineDescriptor(
        "atlas-region-greedy",
        "Route-Budgeted Nearest-Region Atlas Inspector",
        "inspection",
        "substantive",
        "G2-I",
        False,
        True,
    ),
    "atlas-coarse-region-inspector": BaselineDescriptor(
        "atlas-coarse-region-inspector",
        "Coarse Region-Bounds Inspector",
        "inspection_ablation",
        "diagnostic",
        "G2-I-coarse",
        False,
        False,
    ),
    "centralized-oracle": BaselineDescriptor(
        "centralized-oracle",
        "Centralized Witness Oracle",
        "upper_bound",
        "diagnostic",
        "O1-private",
        True,
        False,
    ),
}


def baseline_descriptors() -> list[dict[str, Any]]:
    return [descriptor.__dict__ for descriptor in BASELINES.values()]


def _legal_position(
    point: tuple[float, float, float], colliders: list[AABB], margin: float
) -> bool:
    return all(collider.point_distance(point) >= margin for collider in colliders)


def _prior_colliders(task_spec: dict[str, Any]) -> list[AABB]:
    colliders = []
    for building in task_spec["coarse_prior"]["buildings"]:
        center_x, center_y = (float(value) for value in building["center_xy"])
        size_x, size_y = (float(value) for value in building["size_xy"])
        height = float(building["height_m"])
        colliders.append(
            AABB.from_center_size(
                str(building["prior_id"]),
                (center_x, center_y, height / 2.0),
                (size_x, size_y, height),
                "coarse_prior_building",
            )
        )
    return colliders


def _axis_values(low: float, high: float, spacing: float) -> list[float]:
    values = []
    cursor = low + spacing
    while cursor <= high - spacing:
        values.append(cursor)
        cursor += spacing
    return values


def _grid_points(
    task_spec: dict[str, Any],
    altitudes: tuple[float, ...],
    spacing: float,
    clearance: float,
) -> list[Pose3D]:
    bounds = task_spec["flight_bounds"]
    x_values = _axis_values(float(bounds["minimum"][0]), float(bounds["maximum"][0]), spacing)
    y_values = _axis_values(float(bounds["minimum"][1]), float(bounds["maximum"][1]), spacing)
    colliders = _prior_colliders(task_spec)
    routes: list[Pose3D] = []
    for altitude_index, altitude in enumerate(altitudes):
        for row_index, y_value in enumerate(y_values):
            row = x_values if row_index % 2 == 0 else list(reversed(x_values))
            for x_value in row:
                point = (x_value, y_value, altitude)
                if _legal_position(point, colliders, clearance):
                    yaw = 0.0 if row_index % 2 == 0 else 180.0
                    routes.append(Pose3D(point, yaw, pitch_deg=-12.0 if altitude_index else 0.0))
    return routes


@dataclass(frozen=True)
class _ScanGroup:
    group_id: str
    center: tuple[float, float, float]
    poses: tuple[Pose3D, ...]
    represented_area_m2: float
    vertical_span_m: float


def _centered_axis_values(low: float, high: float, spacing: float) -> list[float]:
    """Cover an interval without encoding a global world-coordinate lattice."""

    if high < low:
        return []
    span = high - low
    count = max(1, math.ceil(span / spacing))
    return [low + span * (index + 0.5) / count for index in range(count)]


def _prior_facade_scan_groups(
    task_spec: dict[str, Any],
    *,
    horizontal_spacing_m: float,
    vertical_spacing_m: float,
    stand_off_m: float,
    fixed_altitude_m: float | None = None,
) -> list[_ScanGroup]:
    """Generate target-agnostic facade observations from the public coarse prior."""

    bounds = task_spec["flight_bounds"]
    minimum_z = float(bounds["minimum"][2]) + 0.4
    maximum_z = float(bounds["maximum"][2]) - 1.2
    groups: list[_ScanGroup] = []
    for building in task_spec["coarse_prior"]["buildings"]:
        building_id = str(building["prior_id"])
        center_x, center_y = (float(value) for value in building["center_xy"])
        size_x, size_y = (float(value) for value in building["size_xy"])
        height = float(building["height_m"])
        x_low, x_high = center_x - size_x / 2.0, center_x + size_x / 2.0
        y_low, y_high = center_y - size_y / 2.0, center_y + size_y / 2.0
        if fixed_altitude_m is None:
            altitudes = _centered_axis_values(
                minimum_z,
                min(maximum_z, max(minimum_z, height - 0.55)),
                vertical_spacing_m,
            )
        elif minimum_z <= fixed_altitude_m <= min(maximum_z, height + 1.5):
            altitudes = [fixed_altitude_m]
        else:
            altitudes = []
        faces = (
            (
                "south",
                _centered_axis_values(x_low + 0.35, x_high - 0.35, horizontal_spacing_m),
                (0.0, -1.0, 0.0),
                "y",
                y_low,
                size_x,
            ),
            (
                "north",
                _centered_axis_values(x_low + 0.35, x_high - 0.35, horizontal_spacing_m),
                (0.0, 1.0, 0.0),
                "y",
                y_high,
                size_x,
            ),
            (
                "west",
                _centered_axis_values(y_low + 0.35, y_high - 0.35, horizontal_spacing_m),
                (-1.0, 0.0, 0.0),
                "x",
                x_low,
                size_y,
            ),
            (
                "east",
                _centered_axis_values(y_low + 0.35, y_high - 0.35, horizontal_spacing_m),
                (1.0, 0.0, 0.0),
                "x",
                x_high,
                size_y,
            ),
        )
        for face_name, along_values, normal, fixed_axis, fixed_value, face_width in faces:
            poses: list[Pose3D] = []
            for altitude_index, altitude in enumerate(altitudes):
                ordered_along = (
                    along_values if altitude_index % 2 == 0 else list(reversed(along_values))
                )
                for along in ordered_along:
                    surface = (
                        (fixed_value, along, altitude)
                        if fixed_axis == "x"
                        else (along, fixed_value, altitude)
                    )
                    position = tuple(
                        surface[axis] + normal[axis] * stand_off_m for axis in range(3)
                    )
                    if all(
                        float(low) <= value <= float(high)
                        for value, low, high in zip(
                            position,
                            bounds["minimum"],
                            bounds["maximum"],
                            strict=True,
                        )
                    ):
                        poses.append(pose_looking_at(position, surface))  # type: ignore[arg-type]
            if not poses:
                continue
            groups.append(
                _ScanGroup(
                    group_id=f"{building_id}/{face_name}",
                    center=(
                        sum(pose.position[0] for pose in poses) / len(poses),
                        sum(pose.position[1] for pose in poses) / len(poses),
                        sum(pose.position[2] for pose in poses) / len(poses),
                    ),
                    poses=tuple(poses),
                    represented_area_m2=face_width * max(1.0, height),
                    vertical_span_m=(
                        max(p.position[2] for p in poses)
                        - min(p.position[2] for p in poses)
                    ),
                )
            )
    return groups


def _public_transit_altitude(
    task_spec: dict[str, Any], config: OrdinaryReleaseConfig, lane_index: int = 0
) -> float:
    bounds = task_spec["flight_bounds"]
    vehicle = config.raw["execution_contract"]["vehicle"]
    body_margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    maximum = float(bounds["maximum"][2]) - body_margin
    transit_contract = task_spec.get("public_transit_contract")
    if not isinstance(transit_contract, dict):
        raise ValueError("public task spec lacks its aggregate safe-sky transit contract")
    safe_sky = float(transit_contract.get("safe_sky_altitude_m", math.nan))
    if not math.isfinite(safe_sky):
        raise ValueError("public safe-sky transit altitude must be finite")
    # The public height envelope is a certified aggregate, not exact collision
    # truth.  It prevents an unnecessarily near-ceiling transit that makes a
    # 300-second baseline infeasible solely because the flight volume is tall.
    lane_spacing = 2.0 * float(vehicle["radius_m"]) + 0.25
    requested = safe_sky + lane_index * lane_spacing
    if requested > maximum + 1.0e-9:
        raise ValueError("safe-sky fleet lane does not fit inside public flight bounds")
    return requested


def _group_route(
    groups: list[_ScanGroup],
    *,
    start: tuple[float, float, float],
    transit_altitude_m: float,
    approach_offset_m: float,
    flight_bounds: dict[str, list[float]],
    body_margin_m: float,
) -> tuple[list[Pose3D], set[int]]:
    """Compile safe-sky transfers and mark only actual scan poses as observable."""

    route: list[Pose3D] = [Pose3D((start[0], start[1], transit_altitude_m), 0.0)]
    observe_indices: set[int] = set()

    def append(pose: Pose3D, *, observable: bool = False) -> None:
        # A zero-length approach leg is not a useful waypoint.  More
        # importantly, leaving it in the route makes a native policy settle
        # twice at the same pose before it may issue OBSERVE.
        if route and distance(route[-1].position, pose.position) <= 1.0e-9:
            # Preserve a coincident non-observable staging pose immediately
            # before an observation pose.  The policy uses it to arrive and
            # settle before commanding the facade-facing yaw at zero
            # translation; merging them would request a large yaw turn during
            # the final descent.
            if observable and len(route) - 1 not in observe_indices:
                route.append(pose)
                observe_indices.add(len(route) - 1)
                return
            if observable:
                observe_indices.add(len(route) - 1)
            return
        route.append(pose)
        if observable:
            observe_indices.add(len(route) - 1)

    def bounded(point: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(
            max(float(low) + body_margin_m, min(float(high) - body_margin_m, value))
            for value, low, high in zip(
                point,
                flight_bounds["minimum"],
                flight_bounds["maximum"],
                strict=True,
            )
        )  # type: ignore[return-value]

    for group in groups:
        first = group.poses[0]
        last = group.poses[-1]
        first_yaw = math.radians(first.yaw_deg)
        last_yaw = math.radians(last.yaw_deg)
        # Scan poses face inward, so their negative forward direction is away from the facade.
        first_approach = bounded((
            first.position[0] - math.cos(first_yaw) * approach_offset_m,
            first.position[1] - math.sin(first_yaw) * approach_offset_m,
            first.position[2],
        ))
        last_approach = bounded((
            last.position[0] - math.cos(last_yaw) * approach_offset_m,
            last.position[1] - math.sin(last_yaw) * approach_offset_m,
            last.position[2],
        ))
        append(Pose3D((first_approach[0], first_approach[1], transit_altitude_m), first.yaw_deg))
        append(Pose3D(first_approach, first.yaw_deg, first.pitch_deg))
        for pose in group.poses:
            append(pose, observable=True)
        append(Pose3D(last_approach, last.yaw_deg, last.pitch_deg))
        append(Pose3D((last_approach[0], last_approach[1], transit_altitude_m), last.yaw_deg))
    return route, observe_indices


def _route_respects_public_prior(
    route: list[Pose3D],
    *,
    start: tuple[float, float, float],
    home: tuple[float, float, float],
    task_spec: dict[str, Any],
    body_margin_m: float,
) -> bool:
    """Screen a diagnostic route using only its G1 coarse prior.

    The coarse prior is deliberately imperfect, so this is not a collision
    proof.  It does reject a route whose straight vertical/horizontal transfer
    already enters a method-visible building envelope.  The native executor
    still uses measured local occupancy and PhysX contacts for execution.
    """

    if body_margin_m <= 0.0:
        raise ValueError("public-route body margin must be positive")
    positions = (start, *(pose.position for pose in route), home)
    colliders = _prior_colliders(task_spec)
    for first, second in zip(positions[:-1], positions[1:], strict=True):
        clearance, _ = minimum_segment_clearance(first, second, colliders)
        if clearance + 1.0e-9 < body_margin_m:
            return False
    return True


def _limited_scan_group(group: _ScanGroup, maximum_scan_poses: int | None) -> _ScanGroup:
    if maximum_scan_poses is None:
        return group
    if maximum_scan_poses < 1:
        raise ValueError("maximum_scan_poses must be positive")
    poses = group.poses[:maximum_scan_poses]
    if not poses:
        raise ValueError("scan group unexpectedly has no poses")
    return _ScanGroup(
        group_id=group.group_id,
        center=(
            sum(pose.position[0] for pose in poses) / len(poses),
            sum(pose.position[1] for pose in poses) / len(poses),
            sum(pose.position[2] for pose in poses) / len(poses),
        ),
        poses=poses,
        represented_area_m2=group.represented_area_m2 * len(poses) / len(group.poses),
        vertical_span_m=(max(p.position[2] for p in poses) - min(p.position[2] for p in poses)),
    )


def _assign_groups(
    groups: list[_ScanGroup],
    start_positions: dict[str, tuple[float, float, float]],
    *,
    information_gain: bool = False,
) -> dict[str, list[_ScanGroup]]:
    """Balanced public assignment followed by a deterministic nearest-neighbour tour."""

    assigned = {drone_id: [] for drone_id in sorted(start_positions)}
    remaining = list(groups)
    current = dict(start_positions)
    building_visits = {
        group.group_id.rsplit("/", 1)[0]: 0
        for group in groups
    }
    while remaining:
        drone_id = min(assigned, key=lambda key: (len(assigned[key]), key))
        minimum_visits = min(
            building_visits[group.group_id.rsplit("/", 1)[0]] for group in remaining
        )
        eligible = [
            group
            for group in remaining
            if building_visits[group.group_id.rsplit("/", 1)[0]] == minimum_visits
        ]

        def score(group: _ScanGroup, owner: str = drone_id) -> tuple[float, float, str]:
            travel = distance(current[owner], group.center)
            gain = group.represented_area_m2 + 8.0 * group.vertical_span_m
            return ((-gain if information_gain else 0.0), travel, group.group_id)

        selected = min(eligible, key=score)
        assigned[drone_id].append(selected)
        current[drone_id] = selected.center
        building_visits[selected.group_id.rsplit("/", 1)[0]] += 1
        remaining.remove(selected)
    return assigned


def _anisotropic_motion_lower_bound_s(
    positions: tuple[tuple[float, float, float], ...],
    *,
    horizontal_speed_mps: float,
    vertical_speed_mps: float,
) -> float:
    """Return the speed-cap lower bound for an ordered public route.

    This is deliberately optimistic.  It excludes controller settling,
    obstacle avoidance, attitude changes, and planner compute.  Consequently a
    route that exceeds the episode even here is objectively unusable, while a
    route that fits remains only a candidate for native execution.
    """

    if len(positions) < 2:
        return 0.0
    if horizontal_speed_mps <= 0.0 or vertical_speed_mps <= 0.0:
        raise ValueError("route-audit speed limits must be positive")
    total = 0.0
    for start, end in zip(positions[:-1], positions[1:], strict=True):
        horizontal_time_s = math.hypot(end[0] - start[0], end[1] - start[1]) / horizontal_speed_mps
        vertical_time_s = abs(end[2] - start[2]) / vertical_speed_mps
        # Horizontal and vertical references may be executed concurrently.  A
        # sum would be a conservative route estimate, but not a kinematic
        # lower bound and could incorrectly reject a feasible public method.
        total += max(horizontal_time_s, vertical_time_s)
    return total


class _RoutePolicy:
    role = "reference"

    def __init__(
        self,
        descriptor: BaselineDescriptor,
        config: OrdinaryReleaseConfig,
        routes: dict[str, list[Pose3D]],
        *,
        observe_indices: dict[str, set[int]] | None = None,
        homes: dict[str, Pose3D] | None = None,
        transit_altitudes: dict[str, float] | None = None,
        flight_bounds: dict[str, list[float]] | None = None,
    ) -> None:
        self.method_id = descriptor.method_id
        self.descriptor = descriptor
        self.config = config
        self.routes = routes
        self.indices = {drone_id: 0 for drone_id in routes}
        self.observe_indices = observe_indices or {
            drone_id: set(range(len(route))) for drone_id, route in routes.items()
        }
        self.homes = homes or {}
        self.transit_altitudes = transit_altitudes or {}
        self.flight_bounds = flight_bounds
        self.observe_remaining = {drone_id: 0 for drone_id in routes}
        self.refined_scan_indices: dict[str, set[int]] = {
            drone_id: set() for drone_id in routes
        }
        self.return_phases = {drone_id: "search" for drone_id in routes}
        self.return_retreat_poses: dict[str, Pose3D | None] = {
            drone_id: None for drone_id in routes
        }
        self.previous_positions: dict[str, tuple[float, float, float] | None] = {
            drone_id: None for drone_id in routes
        }
        self.public_selection_contract: dict[str, Any] | None = None
        vehicle = config.raw["execution_contract"]["vehicle"]
        self.execution_horizontal_speed_mps = float(vehicle["horizontal_speed_mps"])
        self.execution_vertical_speed_mps = float(vehicle["vertical_speed_mps"])
        dwell = float(config.raw["execution_contract"]["observe"]["continuous_dwell_s"])
        period = float(config.raw["execution_contract"]["control_period_s"])
        self.observe_steps = math.ceil(dwell / period) + 1

    def _pose_ready(
        self,
        target: Pose3D,
        observation: ObservationPacket,
        *,
        require_sensor_pitch: bool = False,
    ) -> bool:
        observe = self.config.raw["execution_contract"]["observe"]
        position_tolerance = min(0.05, float(observe["max_pose_drift_m"]) * 0.5)
        yaw_error = abs(((target.yaw_deg - observation.pose.yaw_deg + 180.0) % 360.0) - 180.0)
        # Inspection poses specify camera pitch. The CF2X body remains level
        # while the bounded gimbal reaches this public angle. Transit and return
        # do not wait for gimbal motion before the vehicle can reach home.
        pitch_ready = (
            not require_sensor_pitch
            or abs(target.pitch_deg - float(observation.sensor_pitch_deg)) <= 0.5
        )
        linear_speed = math.sqrt(
            sum(value * value for value in observation.linear_velocity_world_mps)
        )
        return (
            distance(observation.pose.position, target.position) <= position_tolerance
            and yaw_error <= 0.5
            and pitch_ready
            and linear_speed <= float(observe["max_linear_speed_mps"])
            and observation.angular_speed_deg_s <= float(observe["max_angular_speed_deg_s"])
        )

    def _next_pose(self, drone_id: str, observation: ObservationPacket) -> Pose3D | None:
        route = self.routes[drone_id]
        while self.indices[drone_id] < len(route):
            index = self.indices[drone_id]
            target = route[index]
            if (
                index in self.observe_indices[drone_id]
                and index not in self.refined_scan_indices[drone_id]
                and distance(observation.pose.position, target.position)
                <= float(observation.local_occupancy_radius_m)
            ):
                # The public occupancy map is local to the current vehicle
                # pose.  Refining a distant cell both cannot use that map and
                # permanently prevented the old policy from refining it after
                # arrival.  Repeating the full voxel scan on every subsequent
                # tick then exceeded the planner deadline.  Refine exactly once
                # when the public cell enters the sensed neighborhood.
                target = self._refine_scan_pose(observation, target)
                route[index] = target
                self.refined_scan_indices[drone_id].add(index)
            if index not in self.observe_indices[drone_id]:
                # A transit waypoint specifies geometry only.  Keeping the
                # measured attitude prevents a large facade-yaw correction
                # from being coupled with horizontal acceleration.  A separate
                # coincident observation pose below performs that rotation only
                # after the vehicle has settled at its public scan position.
                target = Pose3D(
                    target.position,
                    observation.pose.yaw_deg,
                    observation.pose.pitch_deg,
                    observation.pose.roll_deg,
                )
            if not self._pose_ready(
                target,
                observation,
                require_sensor_pitch=index in self.observe_indices[drone_id],
            ):
                return target
            if index in self.observe_indices[drone_id]:
                if self.observe_remaining[drone_id] == 0:
                    self.observe_remaining[drone_id] = self.observe_steps
                return target
            self.indices[drone_id] += 1
        return None

    def _return_triggered(self, drone_id: str, observation: ObservationPacket) -> bool:
        if drone_id not in self.homes or drone_id not in self.transit_altitudes:
            return False
        execution = self.config.raw["execution_contract"]
        episode = execution["episode"]
        transit_z = self.transit_altitudes[drone_id]
        home = self.homes[drone_id].position
        current = observation.pose.position
        vertical_distance = abs(transit_z - current[2]) + abs(transit_z - home[2])
        horizontal_distance = math.hypot(current[0] - home[0], current[1] - home[1])
        required = (
            vertical_distance / self.execution_vertical_speed_mps
            + horizontal_distance / self.execution_horizontal_speed_mps
            + 3.0
        )
        remaining = float(episode["duration_s"]) - observation.timestamp_s
        return remaining <= max(
            float(episode["return_reserve_s"]) + 12.0,
            required * 1.25 + 8.0,
        )

    @staticmethod
    def _local_occupancy_boxes(observation: ObservationPacket) -> list[AABB]:
        origin = observation.local_occupancy_origin_world_m
        resolution = observation.local_occupancy_resolution_m
        return [
            AABB.from_center_size(
                f"local-{cell[0]}-{cell[1]}-{cell[2]}",
                tuple(origin[axis] + cell[axis] * resolution for axis in range(3)),
                (resolution, resolution, resolution),
                "local_occupancy",
            )
            for cell in observation.local_occupancy
        ]

    def _refine_scan_pose(
        self, observation: ObservationPacket, target: Pose3D
    ) -> Pose3D:
        """Correct a facade scan pose using only method-visible G1 occupancy."""

        if not observation.local_occupancy:
            return target
        if distance(observation.pose.position, target.position) > float(
            observation.local_occupancy_radius_m
        ):
            return target
        forward = (
            math.cos(math.radians(target.yaw_deg)),
            math.sin(math.radians(target.yaw_deg)),
        )
        axis = 0 if abs(forward[0]) >= abs(forward[1]) else 1
        lateral_axis = 1 - axis
        sign = 1.0 if forward[axis] >= 0.0 else -1.0
        occupied = self._local_occupancy_boxes(observation)
        candidates: list[tuple[float, float]] = []
        for box in occupied:
            if not (
                box.minimum[lateral_axis] - 0.25
                <= target.position[lateral_axis]
                <= box.maximum[lateral_axis] + 0.25
                and box.minimum[2] - 0.25
                <= target.position[2]
                <= box.maximum[2] + 0.25
            ):
                continue
            surface = box.minimum[axis] if sign > 0.0 else box.maximum[axis]
            forward_distance = (surface - target.position[axis]) * sign
            if 0.0 <= forward_distance <= 6.0:
                candidates.append((forward_distance, surface))
        vehicle = self.config.raw["execution_contract"]["vehicle"]
        center_clearance = max(
            1.5,
            float(vehicle["radius_m"])
            + float(vehicle["minimum_clearance_m"])
            + 0.08,
        )
        refined = list(target.position)
        surface_point = list(target.position)
        if candidates:
            _, surface = min(candidates)
            refined[axis] = surface - sign * center_clearance
            surface_point[axis] = surface
            if abs(refined[axis] - target.position[axis]) > 3.5:
                refined = list(target.position)
                surface_point[axis] = target.position[axis] + sign * center_clearance
        else:
            surface_point[axis] = target.position[axis] + sign * center_clearance
        base = Pose3D(tuple(refined), target.yaw_deg, target.pitch_deg)  # type: ignore[arg-type]
        return self._locally_safe_scan_pose(
            observation,
            base,
            tuple(surface_point),  # type: ignore[arg-type]
            occupied,
        )

    def _locally_safe_scan_pose(
        self,
        observation: ObservationPacket,
        base: Pose3D,
        surface_point: tuple[float, float, float],
        occupied: list[AABB],
    ) -> Pose3D:
        """Find a nearby safe view without using targets or evaluator witnesses."""

        vehicle = self.config.raw["execution_contract"]["vehicle"]
        required = (
            float(vehicle["radius_m"])
            + float(vehicle["minimum_clearance_m"])
            + 0.05
        )
        yaw = math.radians(base.yaw_deg)
        forward = (math.cos(yaw), math.sin(yaw))
        lateral = (-forward[1], forward[0])

        def corridor_boxes(
            start: tuple[float, float, float],
            end: tuple[float, float, float],
            margin: float,
        ) -> list[AABB]:
            """Conservatively filter local voxels before exact clearance checks."""

            lower = tuple(min(start[index], end[index]) - margin for index in range(3))
            upper = tuple(max(start[index], end[index]) + margin for index in range(3))
            return [
                box
                for box in occupied
                if all(
                    box.maximum[index] >= lower[index]
                    and box.minimum[index] <= upper[index]
                    for index in range(3)
                )
            ]

        def is_safe(
            candidate_position: tuple[float, float, float],
            candidate_occupied: list[AABB],
        ) -> bool:
            # This predicate only needs to prove that every box clears the
            # threshold.  Computing the exact global minimum evaluates every
            # expensive segment/AABB pair even after one blocking voxel has
            # already made the candidate unusable.
            for box in candidate_occupied:
                if box.point_distance(candidate_position) + 1.0e-9 < required:
                    return False
            for box in candidate_occupied:
                if not segment_intersects_expanded_aabb(
                    observation.pose.position,
                    candidate_position,
                    box,
                    required,
                ):
                    continue
                if (
                    segment_aabb_clearance(
                        observation.pose.position, candidate_position, box
                    )
                    + 1.0e-9
                    < required
                ):
                    return False
            return True

        # This is exactly the first (lowest-displacement) member of the full
        # search below.  Most atlas poses are already locally safe, so checking
        # it first avoids evaluating 288 equivalent public candidates at every
        # newly observed cell while preserving the same safety predicate.
        base_candidate = pose_looking_at(base.position, surface_point)
        base_occupied = corridor_boxes(
            observation.pose.position, base_candidate.position, required
        )
        if is_safe(base_candidate.position, base_occupied):
            return base_candidate

        # Every fallback endpoint is within this distance of ``base``.  If a
        # voxel is farther than that displacement plus the required clearance
        # from the base segment, triangle inequality proves it cannot affect
        # any fallback endpoint or its route from the current pose.  This is a
        # conservative exact broad phase; all retained voxels still use the
        # original point and segment clearance tests below.
        maximum_endpoint_displacement_m = math.sqrt(0.75**2 + 1.2**2 + 1.5**2)
        relevant_occupied = corridor_boxes(
            observation.pose.position,
            base.position,
            required + maximum_endpoint_displacement_m + 1.0e-9,
        )

        candidates: list[
            tuple[tuple[float, float, float, float], Pose3D, tuple[float, float, float]]
        ] = []
        lateral_offsets = (0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2)
        vertical_offsets = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5)
        outward_offsets = (0.0, 0.25, 0.5, 0.75)
        for outward in outward_offsets:
            for vertical in vertical_offsets:
                for sideways in lateral_offsets:
                    raw = (
                        base.position[0] - forward[0] * outward + lateral[0] * sideways,
                        base.position[1] - forward[1] * outward + lateral[1] * sideways,
                        base.position[2] + vertical,
                    )
                    candidate_position = self._bounded_position(raw)
                    if distance(raw, candidate_position) > 1.0e-9:
                        continue
                    candidate = pose_looking_at(candidate_position, surface_point)
                    displacement = distance(base.position, candidate_position)
                    score = (displacement, outward, abs(sideways), vertical)
                    candidates.append((score, candidate, candidate_position))
        # The previous implementation evaluated exact segment clearance for all
        # 288 candidates and selected the minimum score afterwards.  Sorting the
        # same finite candidate set first is equivalent to that argmin, while
        # allowing the first safe candidate to terminate the expensive checks.
        candidates.sort(key=lambda item: item[0])
        for _, candidate, candidate_position in candidates:
            if is_safe(candidate_position, relevant_occupied):
                return candidate
        return base

    def _bounded_position(
        self, point: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if self.flight_bounds is None:
            return point
        vehicle = self.config.raw["execution_contract"]["vehicle"]
        margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
        return tuple(
            max(float(low) + margin, min(float(high) - margin, value))
            for value, low, high in zip(
                point,
                self.flight_bounds["minimum"],
                self.flight_bounds["maximum"],
                strict=True,
            )
        )  # type: ignore[return-value]

    def _best_retreat_pose(self, observation: ObservationPacket) -> Pose3D:
        current = observation.pose.position
        boxes = self._local_occupancy_boxes(observation)
        vehicle = self.config.raw["execution_contract"]["vehicle"]
        required = (
            float(vehicle["radius_m"])
            + float(vehicle["minimum_clearance_m"])
            + 0.05
        )
        preferred = math.radians(observation.pose.yaw_deg + 180.0)
        angles = [preferred + index * math.pi / 4.0 for index in range(8)]
        candidates = [
            self._bounded_position(
                (
                    current[0] + 3.0 * math.cos(angle),
                    current[1] + 3.0 * math.sin(angle),
                    current[2],
                )
            )
            for angle in angles
        ]

        scored_candidates: list[
            tuple[tuple[float, float], tuple[float, float, float]]
        ] = []
        for candidate in candidates:
            endpoint_clearance = min(
                (box.point_distance(candidate) for box in boxes), default=math.inf
            )
            if endpoint_clearance + 1.0e-9 < required:
                continue
            segment_clearance, _ = minimum_segment_clearance(current, candidate, boxes)
            candidate_clearance = min(endpoint_clearance, segment_clearance)
            if candidate_clearance + 1.0e-9 < required:
                continue
            scored_candidates.append(
                ((candidate_clearance, distance(current, candidate)), candidate)
            )
        if not scored_candidates:
            return Pose3D(current, observation.pose.yaw_deg)
        selected = max(scored_candidates, key=lambda item: item[0])[1]
        # Public CF2X waypoints command position and yaw.  Roll and pitch are
        # controller-derived attitudes, so freezing an in-motion pitch here
        # creates an unreachable return-completion condition after the vehicle
        # settles at the retreat point.
        return Pose3D(selected, observation.pose.yaw_deg)

    def _return_action(self, drone_id: str, observation: ObservationPacket) -> ActionPacket:
        home = self.homes[drone_id]
        transit_z = self.transit_altitudes[drone_id]
        phase = self.return_phases[drone_id]
        if phase == "search":
            self.return_retreat_poses[drone_id] = self._best_retreat_pose(observation)
            self.return_phases[drone_id] = "retreat"
            phase = "retreat"
        if phase == "retreat":
            target = self.return_retreat_poses[drone_id]
            assert target is not None
            if self._pose_ready(target, observation):
                self.return_phases[drone_id] = "ascend"
                phase = "ascend"
            else:
                return self._waypoint_action(observation, target)
        if phase == "ascend":
            target = Pose3D(
                (observation.pose.position[0], observation.pose.position[1], transit_z),
                home.yaw_deg,
            )
            if self._pose_ready(target, observation):
                self.return_phases[drone_id] = "cruise"
                phase = "cruise"
            else:
                return self._waypoint_action(observation, target)
        if phase == "cruise":
            target = Pose3D((home.position[0], home.position[1], transit_z), home.yaw_deg)
            if self._pose_ready(target, observation):
                self.return_phases[drone_id] = "descend"
                phase = "descend"
            else:
                return self._waypoint_action(observation, target)
        if phase == "descend":
            # RETURN remains a distinct public action, but its endpoint is
            # explicit.  The native adapter must never infer a direct-home
            # coordinate that is absent from the receipt-bound ActionPacket.
            if self._pose_ready(home, observation):
                self.return_phases[drone_id] = "home"
                return ActionPacket(
                    observation.episode_id,
                    drone_id,
                    observation.sequence,
                    observation.timestamp_s,
                    "HOVER",
                )
            return ActionPacket(
                observation.episode_id,
                drone_id,
                observation.sequence,
                observation.timestamp_s,
                "RETURN",
                waypoint=home,
            )
        if phase == "home":
            return ActionPacket(
                observation.episode_id,
                drone_id,
                observation.sequence,
                observation.timestamp_s,
                "HOVER",
            )
        raise RuntimeError(f"unknown return phase for {drone_id}: {phase}")

    @staticmethod
    def _waypoint_action(observation: ObservationPacket, target: Pose3D) -> ActionPacket:
        return ActionPacket(
            observation.episode_id,
            observation.drone_id,
            observation.sequence,
            observation.timestamp_s,
            "WAYPOINT",
            waypoint=target,
            sensor_pitch_deg=target.pitch_deg,
        )

    def _local_occupancy_blocks_scan(
        self, observation: ObservationPacket, target: Pose3D
    ) -> bool:
        if not observation.local_occupancy:
            return False
        period = float(self.config.raw["execution_contract"]["control_period_s"])
        vehicle = self.config.raw["execution_contract"]["vehicle"]
        current = observation.pose.position
        # A local voxel observation cannot justify skipping a distant route
        # leg.  Treating it as global collision truth previously caused the
        # reference policy to collapse a high-to-low transfer into a sequence
        # of unrelated low-altitude waypoints before the CF2X arrived.
        if distance(current, target.position) > float(observation.local_occupancy_radius_m):
            return False
        delta = tuple(
            desired - value for desired, value in zip(target.position, current, strict=True)
        )
        horizontal_distance = math.hypot(delta[0], delta[1])
        horizontal_limit = float(vehicle["horizontal_speed_mps"]) * period
        horizontal_scale = (
            min(1.0, horizontal_limit / horizontal_distance)
            if horizontal_distance > 1.0e-9
            else 0.0
        )
        vertical_limit = float(vehicle["vertical_speed_mps"]) * period
        next_position = (
            current[0] + delta[0] * horizontal_scale,
            current[1] + delta[1] * horizontal_scale,
            current[2] + max(-vertical_limit, min(vertical_limit, delta[2])),
        )
        occupied = self._local_occupancy_boxes(observation)
        clearance, _ = min(
            (
                (box.point_distance(next_position), box.collider_id)
                for box in occupied
            ),
            default=(math.inf, None),
        )
        # Cells already conservatively cover their full voxel. A small numerical margin is
        # enough here; minimum-clearance enforcement remains the execution backend's job.
        return clearance <= 0.25

    def _arbitrate_teammate_trajectories(
        self,
        actions: dict[str, ActionPacket],
        observations: dict[str, ObservationPacket],
    ) -> dict[str, ActionPacket]:
        return arbitrate_public_fleet_actions(
            actions,
            observations,
            vehicle_radius_m=float(self.config.raw["execution_contract"]["vehicle"]["radius_m"]),
        )

    def __call__(self, observations: dict[str, ObservationPacket]) -> dict[str, ActionPacket]:
        actions: dict[str, ActionPacket] = {}
        for drone_id, observation in observations.items():
            if observation.health == "terminal":
                continue
            if self.return_phases[drone_id] != "search" or self._return_triggered(
                drone_id, observation
            ):
                self.observe_remaining[drone_id] = 0
                actions[drone_id] = self._return_action(drone_id, observation)
                continue
            target = self._next_pose(drone_id, observation)
            if target is None:
                if drone_id in self.homes:
                    actions[drone_id] = self._return_action(drone_id, observation)
                else:
                    actions[drone_id] = ActionPacket(
                        observation.episode_id,
                        drone_id,
                        observation.sequence,
                        observation.timestamp_s,
                        "HOVER",
                    )
                continue
            if self.observe_remaining[drone_id] > 0:
                self.observe_remaining[drone_id] -= 1
                actions[drone_id] = ActionPacket(
                    observation.episode_id,
                    drone_id,
                    observation.sequence,
                    observation.timestamp_s,
                    "OBSERVE",
                    source_observation_id=observation.observation_id,
                )
                if self.observe_remaining[drone_id] == 0:
                    self.indices[drone_id] += 1
                continue
            if (
                self.indices[drone_id] in self.observe_indices[drone_id]
                and self.indices[drone_id] not in self.refined_scan_indices[drone_id]
                and self._local_occupancy_blocks_scan(observation, target)
            ):
                self.indices[drone_id] += 1
                self.observe_remaining[drone_id] = 0
                target = self._next_pose(drone_id, observation)
                if target is None:
                    actions[drone_id] = self._return_action(drone_id, observation)
                    continue
            # A reference policy cannot infer a failed waypoint from a few
            # unchanged observations. The native executor may deliberately
            # hover on a planning deadline; skipping here turns a continuous
            # route into discontinuous low-altitude commands. Keep the target
            # until it is reached or the executor reports terminal failure.
            self.previous_positions[drone_id] = observation.pose.position
            actions[drone_id] = self._waypoint_action(observation, target)
        return self._arbitrate_teammate_trajectories(actions, observations)

    def route_budget_audit(
        self,
        *,
        horizontal_speed_mps: float,
        vertical_speed_mps: float,
    ) -> dict[str, Any]:
        """Audit the complete public route before an expensive native run.

        The audit charges the compiled search route, the discrete observation
        dwell used by this policy, a safe-sky return to the public home pose,
        and the frozen return reserve.  It does not inspect targets, evaluator
        witnesses, or any private CitySpec field.
        """

        if horizontal_speed_mps <= 0.0 or vertical_speed_mps <= 0.0:
            raise ValueError("route-audit speed limits must be positive")
        self.execution_horizontal_speed_mps = float(horizontal_speed_mps)
        self.execution_vertical_speed_mps = float(vertical_speed_mps)

        execution = self.config.raw["execution_contract"]
        episode = execution["episode"]
        period = float(execution["control_period_s"])
        # ``observe_steps`` includes the initial OBSERVE sample and is the
        # actual integer-time behavior of this reference policy, not merely
        # the continuous dwell value in the task declaration.
        observation_dwell_s = self.observe_steps * period
        duration_s = float(episode["duration_s"])
        return_reserve_s = float(episode["return_reserve_s"])
        by_drone: dict[str, dict[str, float | int | str]] = {}
        for drone_id in sorted(self.routes):
            route = self.routes[drone_id]
            home = self.homes.get(drone_id)
            if home is None or drone_id not in self.transit_altitudes:
                raise ValueError("route budget audit requires public homes and transit altitudes")
            search_positions = (home.position, *(pose.position for pose in route))
            search_motion_s = _anisotropic_motion_lower_bound_s(
                search_positions,
                horizontal_speed_mps=horizontal_speed_mps,
                vertical_speed_mps=vertical_speed_mps,
            )
            last_position = route[-1].position if route else home.position
            transit_z = self.transit_altitudes[drone_id]
            return_positions = (
                last_position,
                (last_position[0], last_position[1], transit_z),
                (home.position[0], home.position[1], transit_z),
                home.position,
            )
            return_motion_s = _anisotropic_motion_lower_bound_s(
                return_positions,
                horizontal_speed_mps=horizontal_speed_mps,
                vertical_speed_mps=vertical_speed_mps,
            )
            observe_pose_count = len(self.observe_indices[drone_id])
            observe_dwell_s = observe_pose_count * observation_dwell_s
            required_s = search_motion_s + observe_dwell_s + return_motion_s + return_reserve_s
            by_drone[drone_id] = {
                "route_point_count": len(route),
                "observe_pose_count": observe_pose_count,
                "search_motion_lower_bound_s": round(search_motion_s, 9),
                "observe_dwell_lower_bound_s": round(observe_dwell_s, 9),
                "return_motion_lower_bound_s": round(return_motion_s, 9),
                "return_reserve_s": round(return_reserve_s, 9),
                "total_required_lower_bound_s": round(required_s, 9),
                "status": "LOWER_BOUND_FITS"
                if required_s <= duration_s + 1.0e-9
                else "BUDGET_INFEASIBLE",
            }
        status = (
            "BUDGET_INFEASIBLE"
            if any(item["status"] == "BUDGET_INFEASIBLE" for item in by_drone.values())
            else "LOWER_BOUND_FITS"
        )
        return {
            "schema": "org.aerocity.bench.baseline-route-budget-audit.v1",
            "method_id": self.method_id,
            "model": (
                "ordered_public_waypoint_route_with_discrete_observe_and_safe_sky_return"
            ),
            "kinematic_lower_bound_only": True,
            "horizontal_speed_mps": float(horizontal_speed_mps),
            "vertical_speed_mps": float(vertical_speed_mps),
            "episode_duration_s": duration_s,
            "by_drone": by_drone,
            "status": status,
        }


class _FrontierPolicy(_RoutePolicy):
    def __init__(
        self,
        descriptor: BaselineDescriptor,
        config: OrdinaryReleaseConfig,
        task_spec: dict[str, Any],
        drone_ids: list[str],
        start_positions: dict[str, tuple[float, float, float]],
        *,
        information_gain: bool,
    ) -> None:
        maximum_height = float(task_spec["flight_bounds"]["maximum"][2])
        altitude = min(18.0, maximum_height * 0.45)
        candidates = _grid_points(
            task_spec,
            (6.0, altitude, min(maximum_height - 4.0, 34.0)),
            11.0,
            1.3,
        )
        routes: dict[str, list[Pose3D]] = {drone_id: [] for drone_id in drone_ids}
        for candidate in candidates:
            owner = min(
                drone_ids,
                key=lambda drone_id: distance(start_positions[drone_id], candidate.position),
            )
            routes[owner].append(candidate)
        super().__init__(descriptor, config, routes)
        self.information_gain = information_gain
        self.visited: list[tuple[float, float, float]] = []

    def _next_pose(self, drone_id: str, observation: ObservationPacket) -> Pose3D | None:
        candidates = self.routes[drone_id]
        if not candidates:
            return None
        remaining = [
            candidate
            for candidate in candidates
            if all(distance(candidate.position, visited) > 4.0 for visited in self.visited)
        ]
        if not remaining:
            self.visited.clear()
            remaining = candidates
        if self.information_gain:

            def score(candidate: Pose3D) -> tuple[float, float]:
                novelty = min(
                    (distance(candidate.position, visited) for visited in self.visited),
                    default=100.0,
                )
                travel = distance(observation.pose.position, candidate.position)
                vertical_bonus = abs(candidate.position[2] - observation.pose.position[2]) * 0.25
                return novelty + vertical_bonus - 0.15 * travel, -travel

            target = max(remaining, key=score)
        else:
            target = min(
                remaining,
                key=lambda candidate: distance(observation.pose.position, candidate.position),
            )
        if self._pose_ready(target, observation, require_sensor_pitch=True):
            self.visited.append(target.position)
            if self.observe_remaining[drone_id] == 0:
                self.observe_remaining[drone_id] = self.observe_steps
        return target


class _OraclePolicy(_RoutePolicy):
    """Finite private-truth route with non-observing safe-transit waypoints."""

    def __init__(
        self,
        descriptor: BaselineDescriptor,
        config: OrdinaryReleaseConfig,
        routes: dict[str, list[Pose3D]],
        observe_indices: dict[str, set[int]],
    ) -> None:
        super().__init__(descriptor, config, routes)
        self.observe_indices = observe_indices

    def _next_pose(self, drone_id: str, observation: ObservationPacket) -> Pose3D | None:
        route = self.routes[drone_id]
        while self.indices[drone_id] < len(route):
            index = self.indices[drone_id]
            target = route[index]
            if not self._pose_ready(
                target,
                observation,
                require_sensor_pitch=index in self.observe_indices[drone_id],
            ):
                return target
            if index in self.observe_indices[drone_id]:
                if self.observe_remaining[drone_id] == 0:
                    self.observe_remaining[drone_id] = self.observe_steps
                return target
            self.indices[drone_id] += 1
        return None


def _partition_route(route: list[Pose3D], drone_ids: list[str]) -> dict[str, list[Pose3D]]:
    partitions = {drone_id: [] for drone_id in drone_ids}
    if not drone_ids:
        return partitions
    ordered = sorted(drone_ids)
    for pose in route:
        angle = math.atan2(pose.position[1], pose.position[0])
        normalized = (angle + math.pi) / (2.0 * math.pi)
        owner = ordered[min(len(ordered) - 1, int(normalized * len(ordered)))]
        partitions[owner].append(pose)
    return partitions


def _partition_route_by_nearest_start(
    route: list[Pose3D], start_positions: dict[str, tuple[float, float, float]]
) -> dict[str, list[Pose3D]]:
    partitions: dict[str, list[Pose3D]] = {drone_id: [] for drone_id in start_positions}
    for pose in route:
        owner = min(
            start_positions,
            key=lambda drone_id: distance(start_positions[drone_id], pose.position),
        )
        partitions[owner].append(pose)
    return partitions


def _surface_scan_policy(
    descriptor: BaselineDescriptor,
    config: OrdinaryReleaseConfig,
    task_spec: dict[str, Any],
    start_positions: dict[str, tuple[float, float, float]],
    start_yaws: dict[str, float],
    *,
    horizontal_spacing_m: float,
    vertical_spacing_m: float,
    stand_off_m: float,
    fixed_altitude_m: float | None = None,
    information_gain: bool = False,
    randomize_seed: int | None = None,
    maximum_groups_per_drone: int | None = None,
    maximum_scan_poses_per_group: int | None = None,
    screen_route_against_public_prior: bool = False,
    groups_override: list[_ScanGroup] | None = None,
) -> _RoutePolicy:
    groups = list(groups_override) if groups_override is not None else _prior_facade_scan_groups(
        task_spec,
        horizontal_spacing_m=horizontal_spacing_m,
        vertical_spacing_m=vertical_spacing_m,
        stand_off_m=stand_off_m,
        fixed_altitude_m=fixed_altitude_m,
    )
    if randomize_seed is not None:
        random.Random(randomize_seed).shuffle(groups)
        assigned = {drone_id: [] for drone_id in sorted(start_positions)}
        for index, group in enumerate(groups):
            assigned[sorted(assigned)[index % len(assigned)]].append(group)
    else:
        assigned = _assign_groups(
            groups,
            start_positions,
            information_gain=information_gain,
        )
    if maximum_groups_per_drone is not None:
        if maximum_groups_per_drone < 1:
            raise ValueError("maximum_groups_per_drone must be positive")
        # ``sweep-3d`` is a budgeted diagnostic baseline.  It must not silently
        # compile an exhaustive route whose lower bound exceeds the task.  The
        # unselected public groups remain useful for an explicit coverage audit,
        # but are not claimed to have been executed by this policy.
        assigned = {
            drone_id: groups_for_drone[:maximum_groups_per_drone]
            for drone_id, groups_for_drone in assigned.items()
        }
    homes = {
        drone_id: Pose3D(start_positions[drone_id], start_yaws[drone_id])
        for drone_id in start_positions
    }
    body_margin_m = float(config.raw["execution_contract"]["vehicle"]["radius_m"]) + float(
        config.raw["execution_contract"]["vehicle"]["minimum_clearance_m"]
    )
    routes: dict[str, list[Pose3D]] = {}
    observe_indices: dict[str, set[int]] = {}
    transit_altitudes: dict[str, float] = {}
    skipped_public_groups: dict[str, list[str]] = {}
    for lane_index, drone_id in enumerate(sorted(start_positions)):
        transit_altitude = _public_transit_altitude(task_spec, config, lane_index)
        selected_groups = [
            _limited_scan_group(group, maximum_scan_poses_per_group)
            for group in assigned[drone_id]
        ]
        if screen_route_against_public_prior:
            # A diagnostic route may contain multiple public groups.  Its former
            # outward approach waypoint could sit inside a neighboring coarse-
            # prior building column.  A direct descent to the stand-off scan
            # pose avoids that unobservable collision trap.
            safe_group: _ScanGroup | None = None
            for candidate in selected_groups:
                candidate_route, _ = _group_route(
                    [candidate],
                    start=start_positions[drone_id],
                    transit_altitude_m=transit_altitude,
                    approach_offset_m=0.0,
                    flight_bounds=task_spec["flight_bounds"],
                    body_margin_m=body_margin_m,
                )
                if _route_respects_public_prior(
                    candidate_route,
                    start=start_positions[drone_id],
                    home=homes[drone_id].position,
                    task_spec=task_spec,
                    body_margin_m=body_margin_m,
                ):
                    safe_group = candidate
                    break
            if safe_group is None:
                skipped_public_groups[drone_id] = [
                    group.group_id for group in selected_groups
                ]
                selected_groups = []
            else:
                selected_groups = [safe_group]
            approach_offset_m = 0.0
        else:
            approach_offset_m = 2.0
        route, observing = _group_route(
            selected_groups,
            start=start_positions[drone_id],
            transit_altitude_m=transit_altitude,
            approach_offset_m=approach_offset_m,
            flight_bounds=task_spec["flight_bounds"],
            body_margin_m=body_margin_m,
        )
        routes[drone_id] = route
        observe_indices[drone_id] = observing
        transit_altitudes[drone_id] = transit_altitude
    policy = _RoutePolicy(
        descriptor,
        config,
        routes,
        observe_indices=observe_indices,
        homes=homes,
        transit_altitudes=transit_altitudes,
        flight_bounds=task_spec["flight_bounds"],
    )
    if screen_route_against_public_prior:
        policy.public_selection_contract = {
            "schema": "org.aerocity.bench.public-route-screening.v1",
            "target_independent": True,
            "skipped_public_groups": skipped_public_groups,
            "screening_is_not_a_collision_or_completion_claim": True,
        }
    return policy


def _atlas_scan_groups(
    task_spec: dict[str, Any],
    *,
    maximum_regions: int,
    maximum_cells_per_region: int,
    mission_sector: dict[str, Any] | None = None,
) -> list[_ScanGroup]:
    """Turn only public G2-I atlas cells into a bounded L0 route.

    This is intentionally a small calibration bracket.  It does not claim to
    exhaust the atlas; the cap is a public, target-independent method budget
    that lets us test whether the new search object is reachable before an
    expensive native replay or RL run.
    """

    if maximum_regions < 1 or maximum_cells_per_region < 1:
        raise ValueError("atlas inspector caps must be positive")
    atlas = task_spec.get("inspection_atlas")
    if not isinstance(atlas, dict) or task_spec.get("task_track") != "G2-I":
        raise ValueError("atlas-surface-inspector requires a G2-I task spec")
    regions = atlas.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("G2-I task spec has no public inspection regions")
    selected_region_ids = (
        {str(value) for value in mission_sector["selected_region_ids"]}
        if mission_sector is not None
        else None
    )
    selected_cell_ids = (
        {str(value) for value in mission_sector["selected_cell_ids"]}
        if mission_sector is not None
        else None
    )
    selected = sorted(
        (
            region
            for region in regions
            if isinstance(region, dict)
            and (
                selected_region_ids is None
                or str(region.get("region_id", "")) in selected_region_ids
            )
        ),
        key=lambda region: (str(region.get("region_class")), str(region.get("region_id"))),
    )[:maximum_regions]
    groups: list[_ScanGroup] = []
    for region in selected:
        cells = region.get("cells")
        if not isinstance(cells, list) or not cells:
            continue
        if selected_cell_ids is not None:
            cells = [
                cell
                for cell in cells
                if str(cell.get("cell_id", "")) in selected_cell_ids
            ]
            if not cells:
                continue
        # Evenly retain cells across a public region.  The stride is derived
        # from the public cell count and cap, never from private targets.
        stride = max(1, math.ceil(len(cells) / maximum_cells_per_region))
        chosen = cells[::stride][:maximum_cells_per_region]
        poses: list[Pose3D] = []
        for cell in chosen:
            pose = cell.get("pose")
            if not isinstance(pose, dict):
                raise ValueError("G2-I atlas cell lacks a public pose")
            poses.append(
                Pose3D(
                    tuple(float(value) for value in pose["position"]),
                    float(pose["yaw_deg"]),
                    float(pose["pitch_deg"]),
                )
            )
        if poses:
            bounds = region.get("bounds", {})
            minimum = tuple(float(value) for value in bounds["minimum"])
            maximum = tuple(float(value) for value in bounds["maximum"])
            groups.append(
                _ScanGroup(
                    group_id=f"atlas/{region['region_id']}",
                    center=tuple(
                        (lo + hi) / 2.0 for lo, hi in zip(minimum, maximum, strict=True)
                    ),
                    poses=tuple(poses),
                    represented_area_m2=float(region["represented_area_m2"]),
                    vertical_span_m=max(0.0, maximum[2] - minimum[2]),
                )
            )
    if not groups:
        raise ValueError("G2-I atlas has no usable public inspection cells")
    return groups


def _coarse_region_scan_groups(
    task_spec: dict[str, Any],
    *,
    selected_region_ids: set[str] | None = None,
    horizontal_spacing_m: float = 3.0,
    vertical_spacing_m: float = 3.0,
    stand_off_m: float = 2.65,
) -> list[_ScanGroup]:
    """Compile viewpoints from the coarse G2-I region projection only.

    The projection deliberately has no cell, surface point, normal, or transit
    graph fields.  This planner therefore reconstructs a conservative perimeter
    scan from public bounds and structural labels; it is an ablation, not a
    replacement for the full-cell canonical contract.
    """

    projection = task_spec.get("inspection_atlas_projection")
    if not isinstance(projection, dict) or projection.get("prior_level") != ATLAS_PRIOR_COARSE:
        raise ValueError("coarse region inspector requires the coarse G2-I projection")
    validate_inspection_atlas_projection(projection)
    flight_bounds = task_spec["flight_bounds"]
    flight_min = tuple(float(value) for value in flight_bounds["minimum"])
    flight_max = tuple(float(value) for value in flight_bounds["maximum"])
    vehicle = task_spec["execution_contract"]["vehicle"]
    body_margin = (
        float(vehicle["radius_m"])
        + float(vehicle["minimum_clearance_m"])
        + float(projection["sampling_policy"]["flight_bound_buffer_m"])
    )
    groups: list[_ScanGroup] = []

    def bounded(point: tuple[float, float, float]) -> tuple[float, float, float] | None:
        if all(
            flight_min[index] + body_margin <= value <= flight_max[index] - body_margin
            for index, value in enumerate(point)
        ):
            return point
        return None

    def add_group(
        region: dict[str, Any],
        region_suffix: str,
        poses: list[Pose3D],
        center: tuple[float, float, float],
        vertical_span_m: float,
    ) -> None:
        if poses:
            groups.append(
                _ScanGroup(
                    group_id=f"coarse/{region['region_id']}/{region_suffix}",
                    center=center,
                    poses=tuple(poses),
                    represented_area_m2=float(region["represented_area_m2"]),
                    vertical_span_m=max(0.0, vertical_span_m),
                )
            )

    for region in projection["regions"]:
        region_id = str(region["region_id"])
        if selected_region_ids is not None and region_id not in selected_region_ids:
            continue
        lower = tuple(float(value) for value in region["bounds"]["minimum"])
        upper = tuple(float(value) for value in region["bounds"]["maximum"])
        x_values = _centered_axis_values(lower[0], upper[0], horizontal_spacing_m)
        y_values = _centered_axis_values(lower[1], upper[1], horizontal_spacing_m)
        z_low = max(flight_min[2] + body_margin, lower[2] + 0.55)
        z_high = min(flight_max[2] - body_margin, upper[2] - 0.55)
        z_values = _centered_axis_values(z_low, z_high, vertical_spacing_m)
        if not z_values:
            z_values = [
                max(
                    flight_min[2] + body_margin,
                    min(
                        flight_max[2] - body_margin,
                        (lower[2] + upper[2]) / 2.0,
                    ),
                )
            ]
        faces = (
            ("south", (0.0, -1.0, 0.0), "y", lower[1], x_values, z_values),
            ("north", (0.0, 1.0, 0.0), "y", upper[1], x_values, z_values),
            ("west", (-1.0, 0.0, 0.0), "x", lower[0], y_values, z_values),
            ("east", (1.0, 0.0, 0.0), "x", upper[0], y_values, z_values),
        )
        for face_name, normal, fixed_axis, fixed_value, along_values, altitudes in faces:
            poses: list[Pose3D] = []
            for altitude in altitudes:
                for along in along_values:
                    surface = (
                        (fixed_value, along, altitude)
                        if fixed_axis == "x"
                        else (along, fixed_value, altitude)
                    )
                    position = tuple(
                        surface[index] + normal[index] * stand_off_m for index in range(3)
                    )
                    candidate = bounded(position)
                    if candidate is not None:
                        look_at = tuple(surface)
                        poses.append(pose_looking_at(candidate, look_at))
            center = tuple((low + high) / 2.0 for low, high in zip(lower, upper, strict=True))
            add_group(region, face_name, poses, center, upper[2] - lower[2])

        # Roof/entrance/rubble bounds do not expose a public normal.  A small
        # top-down sample remains honest about that missing information and
        # prevents the coarse ablation from silently using full-cell geometry.
        if str(region["region_class"]) in {"roof", "entrance", "rubble"}:
            roof_z = min(flight_max[2] - body_margin, upper[2] + stand_off_m)
            poses = []
            for y_value in _centered_axis_values(lower[1], upper[1], horizontal_spacing_m):
                for x_value in _centered_axis_values(lower[0], upper[0], horizontal_spacing_m):
                    candidate = bounded((x_value, y_value, roof_z))
                    if candidate is not None:
                        poses.append(pose_looking_at(candidate, (x_value, y_value, upper[2])))
            center = tuple((low + high) / 2.0 for low, high in zip(lower, upper, strict=True))
            add_group(region, "top", poses, center, upper[2] - lower[2])
    if not groups:
        raise ValueError("coarse G2-I projection has no executable public regions")
    return groups


def _coarse_region_policy(
    descriptor: BaselineDescriptor,
    config: OrdinaryReleaseConfig,
    task_spec: dict[str, Any],
    public_episode: dict[str, Any],
) -> _RoutePolicy:
    selected = public_episode.get("coarse_region_ids")
    selected_ids = {str(value) for value in selected} if isinstance(selected, list) else None
    start_positions = {
        str(item["drone_id"]): tuple(float(value) for value in item["position"])
        for item in public_episode["starts"]
    }
    start_yaws = {
        str(item["drone_id"]): float(item["yaw_deg"])
        for item in public_episode["starts"]
    }
    groups = _coarse_region_scan_groups(task_spec, selected_region_ids=selected_ids)
    return _surface_scan_policy(
        descriptor,
        config,
        task_spec,
        start_positions,
        start_yaws,
        horizontal_spacing_m=3.0,
        vertical_spacing_m=3.0,
        stand_off_m=2.65,
        maximum_groups_per_drone=1,
        maximum_scan_poses_per_group=16,
        screen_route_against_public_prior=True,
        groups_override=groups,
    )


def _budgeted_atlas_region_policy(
    descriptor: BaselineDescriptor,
    config: OrdinaryReleaseConfig,
    task_spec: dict[str, Any],
    start_positions: dict[str, tuple[float, float, float]],
    start_yaws: dict[str, float],
    mission_sector: dict[str, Any] | None = None,
) -> _RoutePolicy:
    """Budget a target-independent, multi-region public inspection route.

    Region allocation uses a deterministic balanced nearest-neighbour assignment
    from public starts. For each public region breadth, a binary search retains
    the largest evenly spaced cell subset whose optimistic motion, discrete dwell,
    return, and reserve lower bound fits the 300-second task. The selected route
    maximizes target-independent OBSERVE opportunities; it never reads private
    targets or confirmation outcomes, and a fitting lower bound is not a native
    guarantee.
    """

    atlas = task_spec.get("inspection_atlas")
    if not isinstance(atlas, dict):
        raise ValueError("atlas-region-greedy requires the full G2-I atlas")
    selected_region_ids = (
        {str(value) for value in mission_sector["selected_region_ids"]}
        if mission_sector is not None
        else None
    )
    selected_cell_ids = (
        {str(value) for value in mission_sector["selected_cell_ids"]}
        if mission_sector is not None
        else None
    )
    region_count = (
        len(selected_region_ids)
        if selected_region_ids is not None
        else len(atlas.get("regions", []))
    )
    if region_count < 1 or not start_positions:
        raise ValueError("atlas-region-greedy requires public regions and starts")
    regions = [
        region
        for region in atlas.get("regions", [])
        if selected_region_ids is None
        or str(region.get("region_id", "")) in selected_region_ids
    ]
    maximum_cell_count = max(
        sum(
            selected_cell_ids is None or str(cell.get("cell_id", "")) in selected_cell_ids
            for cell in region.get("cells", [])
        )
        for region in regions
        if isinstance(region, dict)
    )
    if maximum_cell_count < 1:
        raise ValueError("atlas-region-greedy has no public cells")
    horizontal_speed_mps = 1.5
    vertical_speed_mps = 1.0
    selected_policy: _RoutePolicy | None = None
    selected_audit: dict[str, Any] | None = None
    selected_cap = 0
    selected_group_cap = 0
    selected_score = (-1, -1, -1)
    maximum_groups_per_drone = math.ceil(region_count / len(start_positions))
    for group_cap in range(1, maximum_groups_per_drone + 1):
        lower = 1
        upper = maximum_cell_count
        while lower <= upper:
            cap = (lower + upper) // 2
            groups = _atlas_scan_groups(
                task_spec,
                maximum_regions=region_count,
                maximum_cells_per_region=cap,
                mission_sector=mission_sector,
            )
            candidate = _surface_scan_policy(
                descriptor,
                config,
                task_spec,
                start_positions,
                start_yaws,
                horizontal_spacing_m=2.0,
                vertical_spacing_m=2.0,
                stand_off_m=2.65,
                information_gain=False,
                maximum_groups_per_drone=group_cap,
                groups_override=groups,
            )
            audit = candidate.route_budget_audit(
                horizontal_speed_mps=horizontal_speed_mps,
                vertical_speed_mps=vertical_speed_mps,
            )
            if audit["status"] == "LOWER_BOUND_FITS":
                observe_count = sum(
                    len(indices) for indices in candidate.observe_indices.values()
                )
                # This method is the breadth counterpart to the single-region
                # surface inspector. Prefer visiting more public regions, then
                # use remaining budget for additional cells inside each region.
                score = (group_cap, observe_count, cap)
                if score > selected_score:
                    selected_policy = candidate
                    selected_audit = audit
                    selected_cap = cap
                    selected_group_cap = group_cap
                    selected_score = score
                lower = cap + 1
            else:
                upper = cap - 1
    if selected_policy is None or selected_audit is None:
        raise ValueError("no atlas-region-greedy route fits the public task lower bound")
    selected_policy.public_selection_contract = {
        "schema": "org.aerocity.bench.atlas-region-selection-public.v1",
        "target_independent": True,
        "selection_objective": (
            "maximize_public_region_breadth_then_observe_cells_under_budget"
        ),
        "one_public_region_per_drone": selected_group_cap == 1,
        "candidate_region_count": region_count,
        "maximum_regions_per_drone": selected_group_cap,
        "maximum_evenly_spaced_cells_per_region": selected_cap,
        "selected_observe_pose_count": sum(
            len(indices) for indices in selected_policy.observe_indices.values()
        ),
        "route_budget_audit": selected_audit,
        "not_a_solvability_or_native_completion_claim": True,
    }
    return selected_policy


def _budgeted_atlas_surface_policy(
    descriptor: BaselineDescriptor,
    config: OrdinaryReleaseConfig,
    task_spec: dict[str, Any],
    start_positions: dict[str, tuple[float, float, float]],
    start_yaws: dict[str, float],
    mission_sector: dict[str, Any] | None = None,
) -> _RoutePolicy:
    """Deeply inspect one public region per vehicle under the task budget."""

    atlas = task_spec.get("inspection_atlas")
    if not isinstance(atlas, dict):
        raise ValueError("atlas-surface-inspector requires the full G2-I atlas")
    selected_region_ids = (
        {str(value) for value in mission_sector["selected_region_ids"]}
        if mission_sector is not None
        else None
    )
    region_count = (
        len(selected_region_ids)
        if selected_region_ids is not None
        else len(atlas.get("regions", []))
    )
    if region_count < 1 or not start_positions:
        raise ValueError("atlas-surface-inspector requires public regions and starts")
    selected_cell_ids = (
        {str(value) for value in mission_sector["selected_cell_ids"]}
        if mission_sector is not None
        else None
    )
    regions = [
        region
        for region in atlas.get("regions", [])
        if selected_region_ids is None
        or str(region.get("region_id", "")) in selected_region_ids
    ]
    maximum_cell_count = max(
        sum(
            selected_cell_ids is None or str(cell.get("cell_id", "")) in selected_cell_ids
            for cell in region.get("cells", [])
        )
        for region in regions
        if isinstance(region, dict)
    )
    if maximum_cell_count < 1:
        raise ValueError("atlas-surface-inspector has no public cells")
    selected_policy: _RoutePolicy | None = None
    selected_audit: dict[str, Any] | None = None
    selected_cell_cap = 0
    lower, upper = 1, maximum_cell_count
    while lower <= upper:
        cell_cap = (lower + upper) // 2
        groups = _atlas_scan_groups(
            task_spec,
            maximum_regions=region_count,
            maximum_cells_per_region=cell_cap,
            mission_sector=mission_sector,
        )
        candidate = _surface_scan_policy(
            descriptor,
            config,
            task_spec,
            start_positions,
            start_yaws,
            horizontal_spacing_m=2.0,
            vertical_spacing_m=2.0,
            stand_off_m=2.65,
            information_gain=False,
            maximum_groups_per_drone=1,
            groups_override=groups,
        )
        audit = candidate.route_budget_audit(
            horizontal_speed_mps=1.5,
            vertical_speed_mps=1.0,
        )
        if audit["status"] == "LOWER_BOUND_FITS":
            selected_policy = candidate
            selected_audit = audit
            selected_cell_cap = cell_cap
            lower = cell_cap + 1
        else:
            upper = cell_cap - 1
    if selected_policy is None or selected_audit is None:
        raise ValueError("no atlas-surface-inspector route fits the public task lower bound")
    selected_observe_count = sum(
        len(indices) for indices in selected_policy.observe_indices.values()
    )
    selected_policy.public_selection_contract = {
        "schema": "org.aerocity.bench.atlas-surface-selection-public.v1",
        "target_independent": True,
        "selection_objective": "one_public_region_per_drone_then_maximize_surface_cells",
        "candidate_region_count": region_count,
        "maximum_regions_per_drone": 1,
        "maximum_evenly_spaced_cells_per_region": selected_cell_cap,
        "selected_observe_pose_count": selected_observe_count,
        "route_budget_audit": selected_audit,
        "not_a_solvability_or_native_completion_claim": True,
    }
    return selected_policy


def create_baseline(
    method_id: str,
    config: OrdinaryReleaseConfig,
    task_spec: dict[str, Any],
    public_episode: dict[str, Any],
    *,
    private_episode: dict[str, Any] | None = None,
) -> FleetPolicy:
    if method_id not in BASELINES:
        raise ValueError(f"unknown baseline: {method_id}")
    descriptor = BASELINES[method_id]
    if descriptor.requires_private_truth and private_episode is None:
        raise ValueError(f"{method_id} is a private diagnostic and needs authority truth")
    # The policy constructor is a method-facing boundary.  Validate before
    # looking at starts or the inspection sector so malformed public data
    # cannot reach a reference policy and only fail later in the evaluator.
    if task_spec.get("task_track") == "G2-I":
        validate_public_task_spec(task_spec)
        validate_public_episode(public_episode, task_spec)
    drone_ids = [str(item["drone_id"]) for item in public_episode["starts"]]
    start_positions = {
        str(item["drone_id"]): tuple(float(value) for value in item["position"])
        for item in public_episode["starts"]
    }
    start_yaws = {
        str(item["drone_id"]): float(item["yaw_deg"]) for item in public_episode["starts"]
    }
    mission_sector = public_episode.get("mission_sector")
    if mission_sector is not None:
        atlas = task_spec.get("inspection_atlas")
        if not isinstance(mission_sector, dict) or not isinstance(atlas, dict):
            raise ValueError("public mission sector requires the full G2-I atlas")
        validate_public_mission_sector(
            mission_sector,
            atlas,
            public_episode["starts"],
            config.raw["execution_contract"],
        )
        if public_episode.get("mission_sector_hash") != mission_sector.get("sector_hash"):
            raise ValueError("public episode mission-sector hash differs")
    if method_id == "sweep-2d":
        return _surface_scan_policy(
            descriptor,
            config,
            task_spec,
            start_positions,
            start_yaws,
            horizontal_spacing_m=2.4,
            vertical_spacing_m=99.0,
            stand_off_m=2.65,
            fixed_altitude_m=3.0,
        )
    if method_id == "sweep-3d":
        return _surface_scan_policy(
            descriptor,
            config,
            task_spec,
            start_positions,
            start_yaws,
            horizontal_spacing_m=7.5,
            vertical_spacing_m=10.0,
            stand_off_m=2.65,
            maximum_groups_per_drone=1,
            maximum_scan_poses_per_group=1,
            screen_route_against_public_prior=True,
        )
    if method_id in {"nearest-frontier", "information-frontier"}:
        return _surface_scan_policy(
            descriptor,
            config,
            task_spec,
            start_positions,
            start_yaws,
            horizontal_spacing_m=2.0,
            vertical_spacing_m=2.4,
            stand_off_m=2.65,
            information_gain=(method_id == "information-frontier"),
        )
    if method_id == "decentralized-auction":
        return _surface_scan_policy(
            descriptor,
            config,
            task_spec,
            start_positions,
            start_yaws,
            horizontal_spacing_m=1.8,
            vertical_spacing_m=2.2,
            stand_off_m=2.65,
        )
    if method_id == "atlas-surface-inspector":
        return _budgeted_atlas_surface_policy(
            descriptor,
            config,
            task_spec,
            start_positions,
            start_yaws,
            mission_sector,
        )
    if method_id == "atlas-region-greedy":
        return _budgeted_atlas_region_policy(
            descriptor,
            config,
            task_spec,
            start_positions,
            start_yaws,
            mission_sector,
        )
    if method_id == "atlas-coarse-region-inspector":
        if mission_sector is not None:
            raise ValueError(
                "coarse region inspector must not receive full-cell mission-sector data"
            )
        return _coarse_region_policy(descriptor, config, task_spec, public_episode)
    if method_id == "centralized-oracle":
        assert private_episode is not None
        routes: dict[str, list[Pose3D]] = {drone_id: [] for drone_id in drone_ids}
        observe_indices: dict[str, set[int]] = {drone_id: set() for drone_id in drone_ids}
        homes = {
            str(item["drone_id"]): Pose3D(
                tuple(float(value) for value in item["position"]),
                float(item["yaw_deg"]),
            )
            for item in public_episode["starts"]
        }
        current_positions = {drone_id: pose.position for drone_id, pose in homes.items()}
        vehicle = config.raw["execution_contract"]["vehicle"]
        body_margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
        maximum_transit_z = float(task_spec["flight_bounds"]["maximum"][2]) - body_margin
        lane_spacing = 2.0 * float(vehicle["radius_m"]) + 0.25
        lane_offsets = {
            drone_id: lane_spacing * index for index, drone_id in enumerate(sorted(drone_ids))
        }
        episode_contract = config.raw["execution_contract"]["episode"]
        vehicle_contract = config.raw["execution_contract"]["vehicle"]
        control_period_s = float(config.raw["execution_contract"]["control_period_s"])
        observe_contract = config.raw["execution_contract"]["observe"]
        observe_steps = math.ceil(
            float(observe_contract["continuous_dwell_s"]) / control_period_s
        ) + 1
        duration_s = float(episode_contract["duration_s"])
        reserve_s = float(episode_contract["return_reserve_s"])
        horizontal_speed_mps = float(vehicle_contract["horizontal_speed_mps"])
        vertical_speed_mps = float(vehicle_contract["vertical_speed_mps"])

        def route_fits_budget(
            drone_id: str,
            candidate_route: list[Pose3D],
            candidate_observe_count: int,
            transit_z: float,
        ) -> bool:
            home = homes[drone_id]
            search_positions = (
                home.position,
                *(pose.position for pose in candidate_route),
            )
            search_motion_s = _anisotropic_motion_lower_bound_s(
                search_positions,
                horizontal_speed_mps=horizontal_speed_mps,
                vertical_speed_mps=vertical_speed_mps,
            )
            last_position = (
                candidate_route[-1].position if candidate_route else home.position
            )
            return_positions = (
                last_position,
                (last_position[0], last_position[1], transit_z),
                (home.position[0], home.position[1], transit_z),
                home.position,
            )
            return_motion_s = _anisotropic_motion_lower_bound_s(
                return_positions,
                horizontal_speed_mps=horizontal_speed_mps,
                vertical_speed_mps=vertical_speed_mps,
            )
            required_s = (
                search_motion_s
                + candidate_observe_count * observe_steps * control_period_s
                + return_motion_s
                + reserve_s
            )
            return required_s <= duration_s + 1.0e-9

        # The oracle is a private feasibility upper bound.  It may use target
        # witnesses, but it must still obey the declared task clock.  Greedily
        # admit the fastest witness that leaves a measured return reserve; do
        # not append an impossible route and call it an oracle failure.
        ordered_targets = sorted(
            private_episode["targets"],
            key=lambda target: (
                min(
                    float(item["reachability_proof"]["travel_time_upper_bound_s"])
                    for item in target["legal_witnesses"]
                ),
                str(target.get("target_id", "")),
            ),
        )
        transit_altitudes: dict[str, float] = {}
        for target in ordered_targets:
            witnesses = sorted(
                target["legal_witnesses"],
                key=lambda item: (
                    float(item["reachability_proof"]["travel_time_upper_bound_s"]),
                    float(item["target_distance_m"]),
                    str(item.get("witness_id", "")),
                ),
            )
            admitted = False
            for witness in witnesses:
                proof = witness["reachability_proof"]
                drone_id = str(proof["start_drone_id"])
                witness_pose = Pose3D.from_dict(witness["pose"])
                transit_z = transit_altitudes.get(drone_id)
                if transit_z is None:
                    transit_z = min(
                        maximum_transit_z,
                        float(proof["transit_altitude_m"]) + lane_offsets[drone_id],
                    )
                current = current_positions[drone_id]
                candidate_route = [
                    *routes[drone_id],
                    Pose3D((current[0], current[1], transit_z), witness_pose.yaw_deg),
                    Pose3D(
                        (witness_pose.position[0], witness_pose.position[1], transit_z),
                        witness_pose.yaw_deg,
                    ),
                    witness_pose,
                ]
                candidate_observe_count = len(observe_indices[drone_id]) + 1
                if not route_fits_budget(
                    drone_id,
                    candidate_route,
                    candidate_observe_count,
                    transit_z,
                ):
                    continue
                routes[drone_id] = candidate_route
                observe_indices[drone_id].add(len(candidate_route) - 1)
                current_positions[drone_id] = witness_pose.position
                transit_altitudes[drone_id] = transit_z
                admitted = True
                break
            if not admitted:
                # A private upper bound is allowed to leave a target
                # unconfirmed when the declared physical budget cannot fit it.
                continue
        for drone_id, route in routes.items():
            if not route:
                continue
            home = homes[drone_id]
            current = current_positions[drone_id]
            transit_z = transit_altitudes[drone_id]
            route.extend(
                (
                    Pose3D((current[0], current[1], transit_z), home.yaw_deg),
                    Pose3D((home.position[0], home.position[1], transit_z), home.yaw_deg),
                    home,
                )
            )
        return _OraclePolicy(descriptor, config, routes, observe_indices)
    return _surface_scan_policy(
        descriptor,
        config,
        task_spec,
        start_positions,
        start_yaws,
        horizontal_spacing_m=3.4,
        vertical_spacing_m=4.8,
        stand_off_m=2.65,
        randomize_seed=derived_seed(task_spec["layout_id"], method_id),
    )
