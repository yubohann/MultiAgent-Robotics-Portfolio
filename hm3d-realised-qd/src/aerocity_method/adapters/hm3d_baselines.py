"""Target-free joint candidates for the HM3D P07 weak-baseline matrix.

This module is deliberately a clean-room implementation.  It borrows the
useful *protocol* lesson from ``md_qd_swarm`` -- all methods must face the
same public state, action authority, guard and outcome contract -- without
importing City-Lite routes, controllers, targets or performance evidence.

The module does not execute a flight and cannot produce a P07 result.  It
builds a common set of high-level multi-UAV waypoint candidates.  A separate
Isaac/CF2X runtime must authorize and execute one selected candidate before a
real execution outcome or exploration score can exist.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from aerocity_method.contracts.io import (
    canonical_sha256,
    finite_number,
    require_identifier,
    require_sha256,
)
from aerocity_method.contracts.hm3d_public_schema import (
    PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION,
    PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
)
from aerocity_method.contracts.models import (
    CandidateFragmentManifest,
    FragmentInstance,
    FragmentTypeSignature,
    PublicMethodContext,
)
from aerocity_method.contracts.privacy import walk_public_payload
from aerocity_method.evaluation.hm3d_safety import (
    TimedPolyline,
    TimedStationary,
    assess_route_tube_separation,
)
from aerocity_method.runtime.hm3d_trajectory import (
    maximum_rest_to_rest_distance_m,
    minimum_rest_to_rest_duration_s,
)
from aerocity_method.runtime.hm3d_realised_qd import HM3D_CANDIDATE_INTENT_SPEC
from aerocity_method.runtime.hm3d_team_collaboration import (
    audit_translation_invariant_team_trajectories,
)

Point3 = tuple[float, float, float]
BASELINE_STRATEGIES = frozenset({"random", "frontier_3d", "auction"})
TRANSIT_TIMING_SCHEMA_VERSION = "hm3d-kinematic-transit-timing-v4"
PUBLIC_CANDIDATE_POOL_SOURCE = PUBLIC_CANDIDATE_POOL_SCHEMA_VERSION
# Route-access quality credit for the shared candidate authority. It is bounded
# so a long route is preferred only when its public gain is otherwise close.
PUBLIC_ROUTE_CONTINUITY_BONUS_MIN_M = 2.0
PUBLIC_ROUTE_CONTINUITY_BONUS_RAMP_M = 5.0
PUBLIC_ROUTE_CONTINUITY_BONUS_MAX = 0.10
# A region-access view is a committed public-route prefix into an under-explored
# region, not a local observation.  The frontier selector uses a bounded
# progression credit so a legal 3 m+ access route is not always erased by a
# slightly higher-gain micro-observation that stops immediately.
PUBLIC_REGION_ACCESS_CREDIT_REFERENCE_M = 3.0
PUBLIC_REGION_ACCESS_CREDIT_MAX = 0.50
# Execution-mileage preference weight for the transparent gain selector.
# 0.02 / metre gives a 16 m team route ~0.32 credit, the same order as one
# cluster-gain unit, so a completeable long access route can beat a myopic
# short viewpoint when observed gains are close.  Frozen protocol constant.
PUBLIC_EXECUTION_MILEAGE_PREFERENCE_WEIGHT = 0.02


def _point3(values: Sequence[float], name: str) -> Point3:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    return tuple(finite_number(value, f"{name}[{index}]") for index, value in enumerate(values))  # type: ignore[return-value]


def _distance(left: Point3, right: Point3) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _path_length_m(path_m: Sequence[Point3]) -> float:
    path = tuple(path_m)
    if len(path) < 2:
        raise ValueError("transit path requires at least two points")
    return sum(_distance(start, end) for start, end in zip(path[:-1], path[1:], strict=True))


def _manifest_planned_route_length_m(manifest: CandidateFragmentManifest) -> float:
    """Total planned team transit length, used by the mileage preference."""
    return sum(
        _path_length_m(fragment.path)
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    )


def _segment_boundary_duration_s(
    distance_m: float,
    *,
    start_speed_mps: float,
    end_speed_mps: float,
    cruise_speed_mps: float,
    max_accel_mps2: float,
) -> float:
    """Return time for one line segment with bounded endpoint speeds."""

    if distance_m <= 1.0e-9:
        return 0.0
    start_speed = min(max(0.0, start_speed_mps), cruise_speed_mps)
    end_speed = min(max(0.0, end_speed_mps), cruise_speed_mps)
    if start_speed == end_speed:
        if start_speed <= 1.0e-12:
            return minimum_rest_to_rest_duration_s(
                distance_m,
                cruise_speed_mps=cruise_speed_mps,
                max_accel_mps2=max_accel_mps2,
            )
        return distance_m / start_speed
    if start_speed < end_speed:
        acceleration_distance_m = (
            end_speed**2 - start_speed**2
        ) / (2.0 * max_accel_mps2)
        if distance_m <= acceleration_distance_m:
            peak_speed_mps = math.sqrt(
                start_speed**2 + 2.0 * max_accel_mps2 * distance_m
            )
            return (peak_speed_mps - start_speed) / max_accel_mps2
        return (
            (end_speed - start_speed) / max_accel_mps2
            + (distance_m - acceleration_distance_m) / end_speed
        )
    deceleration_distance_m = (
        start_speed**2 - end_speed**2
    ) / (2.0 * max_accel_mps2)
    if distance_m <= deceleration_distance_m:
        return minimum_rest_to_rest_duration_s(
            distance_m,
            cruise_speed_mps=cruise_speed_mps,
            max_accel_mps2=max_accel_mps2,
        )
    return (
        (distance_m - deceleration_distance_m) / start_speed
        + (start_speed - end_speed) / max_accel_mps2
    )


@dataclass(frozen=True, slots=True)
class ConservativeTransitTimingModel:
    """Executor-aligned timing plus outcome-validated terminal and turn margins.

    The previous model divided distance by a short-route average speed. That
    folded acceleration, braking and settling into every metre and made longer
    candidates artificially slow. This contract uses the same rest-to-rest
    triangular/trapezoidal limits as the executor, then adds one terminal
    convergence margin and an additional margin for each intermediate
    waypoint. The executor requires a settled stop at every waypoint, but a
    terminal convergence delay must not be charged twice on a two-segment
    route.
    """

    calibration_id: str
    cruise_speed_mps: float
    max_accel_mps2: float
    terminal_tracking_margin_s: float
    intermediate_waypoint_settle_margin_s: float = 0.0
    # The calibration artifact currently contains completed routes with at
    # most this many line segments.  Longer public polylines are still legal,
    # but their unobserved controller residual must be reserved explicitly.
    calibrated_max_segment_count: int = 2
    uncovered_segment_reserve_s: float = 0.0
    intermediate_waypoint_requires_settle: bool = True
    continuous_waypoint_speed_mps: float = 0.35
    schema_version: str = TRANSIT_TIMING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRANSIT_TIMING_SCHEMA_VERSION:
            raise ValueError("transit timing model schema version mismatch")
        require_identifier(self.calibration_id, "calibration_id")
        speed = finite_number(self.cruise_speed_mps, "cruise_speed_mps")
        acceleration = finite_number(self.max_accel_mps2, "max_accel_mps2")
        terminal_margin = finite_number(
            self.terminal_tracking_margin_s, "terminal_tracking_margin_s"
        )
        intermediate_margin = finite_number(
            self.intermediate_waypoint_settle_margin_s,
            "intermediate_waypoint_settle_margin_s",
        )
        if isinstance(self.calibrated_max_segment_count, bool) or not isinstance(
            self.calibrated_max_segment_count, int
        ):
            raise TypeError("calibrated_max_segment_count must be an integer")
        calibrated_max_segments = self.calibrated_max_segment_count
        uncovered_reserve = finite_number(
            self.uncovered_segment_reserve_s,
            "uncovered_segment_reserve_s",
        )
        if speed <= 0.0:
            raise ValueError("cruise_speed_mps must be positive")
        if acceleration <= 0.0:
            raise ValueError("max_accel_mps2 must be positive")
        if terminal_margin < 0.0:
            raise ValueError("terminal_tracking_margin_s must be non-negative")
        if intermediate_margin < 0.0:
            raise ValueError("intermediate_waypoint_settle_margin_s must be non-negative")
        if calibrated_max_segments < 1:
            raise ValueError("calibrated_max_segment_count must be positive")
        if uncovered_reserve < 0.0:
            raise ValueError("uncovered_segment_reserve_s must be non-negative")
        if not isinstance(self.intermediate_waypoint_requires_settle, bool):
            raise ValueError("intermediate_waypoint_requires_settle must be boolean")
        continuous_speed = finite_number(
            self.continuous_waypoint_speed_mps,
            "continuous_waypoint_speed_mps",
        )
        if continuous_speed <= 0.0:
            raise ValueError("continuous_waypoint_speed_mps must be positive")
        if continuous_speed > speed + 1.0e-9:
            raise ValueError("continuous_waypoint_speed_mps cannot exceed cruise_speed_mps")
        object.__setattr__(self, "cruise_speed_mps", speed)
        object.__setattr__(self, "max_accel_mps2", acceleration)
        object.__setattr__(self, "terminal_tracking_margin_s", terminal_margin)
        object.__setattr__(self, "intermediate_waypoint_settle_margin_s", intermediate_margin)
        object.__setattr__(self, "calibrated_max_segment_count", calibrated_max_segments)
        object.__setattr__(self, "uncovered_segment_reserve_s", uncovered_reserve)
        object.__setattr__(
            self,
            "intermediate_waypoint_requires_settle",
            self.intermediate_waypoint_requires_settle,
        )
        object.__setattr__(self, "continuous_waypoint_speed_mps", continuous_speed)

    def motion_seconds_for_distance(self, distance_m: float) -> float:
        return minimum_rest_to_rest_duration_s(
            distance_m,
            cruise_speed_mps=self.cruise_speed_mps,
            max_accel_mps2=self.max_accel_mps2,
        )

    def maximum_direct_path_length_m(self, transit_duration_s: float) -> float:
        duration = finite_number(transit_duration_s, "transit_duration_s")
        if not self.intermediate_waypoint_requires_settle:
            speed = self.continuous_waypoint_speed_mps
            accel_time_s = speed / self.max_accel_mps2
            motion_duration_s = max(
                0.0,
                duration - self.terminal_tracking_margin_s - 2.0 * accel_time_s,
            )
            return speed * motion_duration_s + speed**2 / self.max_accel_mps2
        return maximum_rest_to_rest_distance_m(
            max(0.0, duration - self.terminal_tracking_margin_s),
            cruise_speed_mps=self.cruise_speed_mps,
            max_accel_mps2=self.max_accel_mps2,
        )

    def estimate_seconds(self, path_m: Sequence[Point3]) -> float:
        """Return a conservative planned duration for a guarded 3D polyline."""

        path = tuple(path_m)
        _path_length_m(path)
        if not self.intermediate_waypoint_requires_settle:
            if all(
                _distance(path[0], point) <= 1.0e-9
                for point in path[1:]
            ):
                # An explicit hold has no traversed segment, but the executor
                # still converges to the terminal waypoint before observing.
                return self.terminal_tracking_margin_s
            return self.continuous_polyline_seconds(path)
        segment_count = len(path) - 1
        intermediate_waypoint_count = len(path) - 2
        uncovered_segment_count = max(0, segment_count - self.calibrated_max_segment_count)
        return (
            sum(
                self.motion_seconds_for_distance(_distance(start, end))
                for start, end in zip(path[:-1], path[1:], strict=True)
            )
            + self.terminal_tracking_margin_s
            + intermediate_waypoint_count * self.intermediate_waypoint_settle_margin_s
            + uncovered_segment_count * self.uncovered_segment_reserve_s
        )

    def continuous_polyline_seconds(self, path_m: Sequence[Point3]) -> float:
        """Return the planned time for a route flown through intermediate corners.

        The first segment accelerates from rest to the pass-through speed, all
        intermediate segments keep that speed, and the final segment decelerates
        to the terminal settle condition. This is the timing counterpart of the
        executor's waypoint pass-through contract.
        """

        path = tuple(path_m)
        if len(path) < 2:
            raise ValueError("transit path requires at least two points")
        segment_lengths = tuple(
            _distance(left, right) for left, right in zip(path[:-1], path[1:], strict=True)
        )
        if any(length <= 0.0 for length in segment_lengths):
            raise ValueError("transit path contains a zero-length segment")
        pass_speed_mps = min(
            self.continuous_waypoint_speed_mps,
            math.sqrt(2.0 * self.max_accel_mps2 * min(segment_lengths)),
        )
        total_s = 0.0
        for index, segment_length_m in enumerate(segment_lengths):
            first = index == 0
            final = index == len(segment_lengths) - 1
            start_speed_mps = 0.0 if first else pass_speed_mps
            end_speed_mps = 0.0 if final else pass_speed_mps
            total_s += _segment_boundary_duration_s(
                segment_length_m,
                start_speed_mps=start_speed_mps,
                end_speed_mps=end_speed_mps,
                cruise_speed_mps=self.cruise_speed_mps,
                max_accel_mps2=self.max_accel_mps2,
            )
        uncovered_segment_count = max(
            0, len(segment_lengths) - self.calibrated_max_segment_count
        )
        return (
            total_s
            + self.terminal_tracking_margin_s
            + uncovered_segment_count * self.uncovered_segment_reserve_s
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "cruise_speed_mps": self.cruise_speed_mps,
            "max_accel_mps2": self.max_accel_mps2,
            "terminal_tracking_margin_s": self.terminal_tracking_margin_s,
            "intermediate_waypoint_settle_margin_s": self.intermediate_waypoint_settle_margin_s,
            "calibrated_max_segment_count": self.calibrated_max_segment_count,
            "uncovered_segment_reserve_s": self.uncovered_segment_reserve_s,
            "intermediate_waypoint_requires_settle": (
                self.intermediate_waypoint_requires_settle
            ),
            "continuous_waypoint_speed_mps": self.continuous_waypoint_speed_mps,
            "formula": (
                (
                    "sum(rest_to_rest_triangular_or_trapezoidal_segment_time) + "
                    "terminal_tracking_margin_s + intermediate_waypoint_count * "
                    "intermediate_waypoint_settle_margin_s + "
                    "max(0, segment_count - calibrated_max_segment_count) * "
                    "uncovered_segment_reserve_s"
                )
                if self.intermediate_waypoint_requires_settle
                else (
                    "first_acceleration + intermediate_constant_speed_segments + "
                    "final_deceleration + terminal_tracking_margin_s + "
                    "max(0, segment_count - calibrated_max_segment_count) * "
                    "uncovered_segment_reserve_s"
                )
            ),
            "executor_alignment": (
                "shared_reference_speed_and_acceleration_limits; "
                "uncovered-route-segment reserve is an admission margin, not a "
                "controller command; intermediate waypoints use pass-through "
                "speed and only the terminal waypoint settles"
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConservativeTransitTimingModel:
        if not isinstance(payload, Mapping):
            raise TypeError("transit timing model payload must be a mapping")
        # v4 artifacts must bind the route-length extrapolation contract. Do
        # not silently restore the old zero-reserve behavior when a stale
        # artifact is loaded into the active runner.
        if "calibrated_max_segment_count" not in payload:
            raise ValueError("transit timing model omits calibrated_max_segment_count")
        if "uncovered_segment_reserve_s" not in payload:
            raise ValueError("transit timing model omits uncovered_segment_reserve_s")
        return cls(
            calibration_id=payload["calibration_id"],
            cruise_speed_mps=payload["cruise_speed_mps"],
            max_accel_mps2=payload["max_accel_mps2"],
            terminal_tracking_margin_s=payload["terminal_tracking_margin_s"],
            intermediate_waypoint_settle_margin_s=payload["intermediate_waypoint_settle_margin_s"],
            calibrated_max_segment_count=payload["calibrated_max_segment_count"],
            uncovered_segment_reserve_s=payload["uncovered_segment_reserve_s"],
            intermediate_waypoint_requires_settle=payload.get(
                "intermediate_waypoint_requires_settle", True
            ),
            continuous_waypoint_speed_mps=payload.get(
                "continuous_waypoint_speed_mps", 0.35
            ),
            schema_version=payload.get("schema_version", ""),
        )


@dataclass(frozen=True, slots=True)
class PublicAgentPose:
    """One method-visible vehicle state at a high-level decision boundary."""

    agent_id: str
    position_m: Point3
    remaining_energy_fraction: float
    communication_degree: int

    def __post_init__(self) -> None:
        require_identifier(self.agent_id, "agent_id")
        object.__setattr__(self, "position_m", _point3(self.position_m, "position_m"))
        energy = finite_number(self.remaining_energy_fraction, "remaining_energy_fraction")
        if not 0.0 <= energy <= 1.0:
            raise ValueError("remaining_energy_fraction must be in [0, 1]")
        if (
            not isinstance(self.communication_degree, int)
            or isinstance(self.communication_degree, bool)
            or self.communication_degree < 0
        ):
            raise ValueError("communication_degree must be a non-negative integer")
        object.__setattr__(self, "remaining_energy_fraction", energy)

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "position_m": self.position_m,
            "remaining_energy_fraction": self.remaining_energy_fraction,
            "communication_degree": self.communication_degree,
        }


@dataclass(frozen=True, slots=True)
class PublicFrontier:
    """A candidate view location extracted from public sparse-range mapping."""

    frontier_id: str
    position_m: Point3
    information_gain: float
    traversal_risk: float
    source_agent_id: str | None = None
    # A outcome-backed return is a recovery action, not a public information-
    # gain frontier. It remains in the common candidate authority so every
    # selector receives the same safe way to leave a locally unsupported pose.
    task_kind: str = "explore"
    exclusive_agent_id: str | None = None
    # Observation poses are the primary frontier actions. Route-progress and
    # region-access rows are public, guarded alternatives that keep a corridor
    # or vertical access route when the endpoint-only observation is short.
    viewpoint_kind: str = "observation"
    # A full route produced from the current shared sparse belief, keyed by
    # public agent ID.  This is an action proposal rather than a static-map
    # clearance certificate: the runtime guard still checks every segment
    # after selection.  Keeping it here prevents candidate construction from
    # discarding a public access route and silently recomputing a different
    # endpoint-only route later in the same decision.
    access_paths_m: tuple[tuple[str, tuple[Point3, ...]], ...] = ()
    # Extractor-local cluster provenance is public and audit-only. It is not a
    # stable task identity: extraction order can change as public outcomes
    # arrive.
    frontier_cluster_id: str = ""
    # These values are derived from the current public frontier cluster. They
    # associate a newly extracted frontier with a short-lived, outcome-backed
    # task reservation without retaining a stale manifest or private geometry.
    task_anchor_m: Point3 | None = None
    task_normal_unit: Point3 | None = None

    def __post_init__(self) -> None:
        require_identifier(self.frontier_id, "frontier_id")
        position = _point3(self.position_m, "frontier_position_m")
        object.__setattr__(self, "position_m", position)
        gain = finite_number(self.information_gain, "information_gain")
        risk = finite_number(self.traversal_risk, "traversal_risk")
        if gain < 0.0 or not 0.0 <= risk <= 1.0:
            raise ValueError("frontier gain must be non-negative and risk must be in [0, 1]")
        if self.source_agent_id is not None:
            require_identifier(self.source_agent_id, "source_agent_id")
        if self.task_kind not in {"explore", "backtrack"}:
            raise ValueError("public frontier task_kind must be explore or backtrack")
        if self.exclusive_agent_id is not None:
            require_identifier(self.exclusive_agent_id, "exclusive_agent_id")
        if self.task_kind == "backtrack" and self.exclusive_agent_id is None:
            raise ValueError("outcome-backed backtrack needs an exclusive agent")
        if self.task_kind == "explore" and self.exclusive_agent_id is not None:
            raise ValueError("ordinary exploration frontier cannot be exclusive")
        viewpoint_kind = self.viewpoint_kind
        # Keep existing outcome-backtrack call sites source-compatible while
        # serializing their distinct action semantics explicitly.
        if self.task_kind == "backtrack" and viewpoint_kind == "observation":
            viewpoint_kind = "outcome_backtrack"
        if self.task_kind == "explore" and viewpoint_kind not in {
            "observation",
            "route_progress",
            "region_access",
        }:
            raise ValueError("ordinary exploration frontier has an invalid viewpoint kind")
        if self.task_kind == "backtrack" and viewpoint_kind not in {
            "outcome_backtrack",
            # A current-public-map route can be used as a one-agent geometric
            # escape when the fleet is already inside the planning envelope.
            # It is still a backtrack/recovery action and never an exploration
            # frontier; the runtime joint guard remains the authority.
            "collision_avoidance_recovery",
        }:
            raise ValueError(
                "backtrack frontier must be a outcome-backed frontier using "
                "outcome_backtrack or collision_avoidance_recovery viewpoint kind"
            )
        access_paths: list[tuple[str, tuple[Point3, ...]]] = []
        seen_access_agents: set[str] = set()
        for agent_id, path_m in self.access_paths_m:
            require_identifier(agent_id, "public access path agent_id")
            if agent_id in seen_access_agents:
                raise ValueError("public frontier has duplicate access-path agent")
            seen_access_agents.add(agent_id)
            path = tuple(_point3(point, "public_access_path_m") for point in path_m)
            if len(path) < 2:
                raise ValueError("public access path needs at least one segment")
            if math.dist(path[-1], self.position_m) > PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M:
                raise ValueError("public access path endpoint must match frontier position")
            access_paths.append((agent_id, path))
        cluster_id = self.frontier_cluster_id
        if cluster_id:
            require_identifier(cluster_id, "frontier_cluster_id")
        task_anchor = (
            position
            if self.task_anchor_m is None
            else _point3(self.task_anchor_m, "frontier_task_anchor_m")
        )
        task_normal: Point3 | None = None
        if self.task_normal_unit is not None:
            normal = _point3(self.task_normal_unit, "frontier_task_normal_unit")
            normal_norm = math.sqrt(sum(component * component for component in normal))
            if normal_norm <= 1.0e-12:
                raise ValueError("frontier task normal must be non-zero")
            task_normal = tuple(component / normal_norm for component in normal)  # type: ignore[assignment]
        object.__setattr__(self, "information_gain", gain)
        object.__setattr__(self, "traversal_risk", risk)
        object.__setattr__(self, "viewpoint_kind", viewpoint_kind)
        object.__setattr__(self, "access_paths_m", tuple(sorted(access_paths)))
        object.__setattr__(self, "frontier_cluster_id", cluster_id)
        object.__setattr__(self, "task_anchor_m", task_anchor)
        object.__setattr__(self, "task_normal_unit", task_normal)

    def access_path_for_agent(self, agent_id: str) -> tuple[Point3, ...] | None:
        """Return the current-belief access route for one agent, if supplied."""

        require_identifier(agent_id, "public access-path query agent_id")
        for owner_agent_id, path_m in self.access_paths_m:
            if owner_agent_id == agent_id:
                return path_m
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "frontier_id": self.frontier_id,
            "position_m": self.position_m,
            "information_gain": self.information_gain,
            "traversal_risk": self.traversal_risk,
            "source_agent_id": self.source_agent_id,
            "task_kind": self.task_kind,
            "exclusive_agent_id": self.exclusive_agent_id,
            "viewpoint_kind": self.viewpoint_kind,
            "access_paths_m": [
                {"agent_id": agent_id, "path_m": path_m}
                for agent_id, path_m in self.access_paths_m
            ],
            "frontier_cluster_id": self.frontier_cluster_id,
            "task_anchor_m": self.task_anchor_m,
            "task_normal_unit": self.task_normal_unit,
        }


@dataclass(frozen=True, slots=True)
class PublicTaskReservation:
    """A public, outcome-backed task association retained across decisions.

    The reservation owns no stale action. At every later decision the runner
    extracts fresh frontiers from current public outcomes, matches their public
    anchor/normal to this reservation, regenerates an access route from the
    current pose, and submits it to the unchanged static and joint guards.
    """

    agent_id: str
    source_decision_id: str
    source_manifest_hash: str
    source_transit_outcome_sha256: str
    source_public_path_sha256: str
    completed_position_m: Point3
    terminal_heading_unit: Point3
    source_viewpoint_kind: str = "observation"
    task_anchor_m: Point3 | None = None
    task_normal_unit: Point3 | None = None
    source_frontier_cluster_id: str = ""

    def __post_init__(self) -> None:
        require_identifier(self.agent_id, "task reservation agent_id")
        require_identifier(self.source_decision_id, "task reservation source decision_id")
        require_sha256(self.source_manifest_hash, "task reservation source manifest hash")
        require_sha256(
            self.source_transit_outcome_sha256,
            "task reservation source transit outcome hash",
        )
        require_sha256(self.source_public_path_sha256, "task reservation source public path hash")
        if self.source_viewpoint_kind not in {
            "observation",
            "route_progress",
            "region_access",
        }:
            raise ValueError("task reservation must originate from a public exploration route")
        completed_position = _point3(
            self.completed_position_m,
            "task_reservation_completed_position_m",
        )
        heading = _point3(self.terminal_heading_unit, "task_reservation_terminal_heading_unit")
        heading_norm = math.sqrt(sum(component * component for component in heading))
        if heading_norm <= 1.0e-12:
            raise ValueError("task reservation terminal heading must be non-zero")
        task_anchor = (
            completed_position
            if self.task_anchor_m is None
            else _point3(self.task_anchor_m, "task_reservation_anchor_m")
        )
        task_normal: Point3 | None = None
        if self.task_normal_unit is not None:
            normal = _point3(self.task_normal_unit, "task_reservation_normal_unit")
            normal_norm = math.sqrt(sum(component * component for component in normal))
            if normal_norm <= 1.0e-12:
                raise ValueError("task reservation normal must be non-zero")
            task_normal = tuple(component / normal_norm for component in normal)  # type: ignore[assignment]
        if self.source_frontier_cluster_id:
            require_identifier(
                self.source_frontier_cluster_id,
                "task reservation source frontier_cluster_id",
            )
        object.__setattr__(self, "completed_position_m", completed_position)
        object.__setattr__(
            self,
            "terminal_heading_unit",
            tuple(component / heading_norm for component in heading),
        )
        object.__setattr__(self, "task_anchor_m", task_anchor)
        object.__setattr__(self, "task_normal_unit", task_normal)

    @classmethod
    def from_completed_public_exploration_transit(
        cls,
        *,
        agent_id: str,
        source_decision_id: str,
        source_manifest_hash: str,
        source_transit_outcome_sha256: str,
        public_path_m: Sequence[Point3],
        task_anchor_m: Point3 | None = None,
        task_normal_unit: Point3 | None = None,
        source_frontier_cluster_id: str = "",
        source_viewpoint_kind: str = "observation",
    ) -> PublicTaskReservation:
        """Create a reservation only after a completed guarded public route."""

        path = tuple(_point3(point, "completed_public_transit_path_m") for point in public_path_m)
        if len(path) < 2 or math.dist(path[0], path[-1]) <= PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M:
            raise ValueError("completed public exploration route aliases its settled endpoint")
        heading = _terminal_path_heading(path)
        if heading is None:
            raise ValueError("completed public exploration route has no non-alias heading")
        return cls(
            agent_id=agent_id,
            source_decision_id=source_decision_id,
            source_manifest_hash=source_manifest_hash,
            source_transit_outcome_sha256=source_transit_outcome_sha256,
            source_public_path_sha256=canonical_sha256(path),
            completed_position_m=path[-1],
            terminal_heading_unit=heading,
            source_viewpoint_kind=source_viewpoint_kind,
            task_anchor_m=task_anchor_m,
            task_normal_unit=task_normal_unit,
            source_frontier_cluster_id=source_frontier_cluster_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "source_decision_id": self.source_decision_id,
            "source_manifest_hash": self.source_manifest_hash,
            "source_transit_outcome_sha256": self.source_transit_outcome_sha256,
            "source_public_path_sha256": self.source_public_path_sha256,
            "completed_position_m": self.completed_position_m,
            "terminal_heading_unit": self.terminal_heading_unit,
            "source_viewpoint_kind": self.source_viewpoint_kind,
            "task_anchor_m": self.task_anchor_m,
            "task_normal_unit": self.task_normal_unit,
            "source_frontier_cluster_id": self.source_frontier_cluster_id,
        }


@dataclass(frozen=True, slots=True)
class PublicSearchState:
    """Strictly public input shared by every ranked P07 method.

    ``frontiers`` may only originate from the frozen sparse-range observation
    schedule.  The state contains no target identity, target distance, full
    mesh or evaluator ESDF.  The runtime safety guard remains evaluator-side;
    it receives a candidate after selection, not as a policy feature.
    """

    context: PublicMethodContext
    agents: tuple[PublicAgentPose, ...]
    frontiers: tuple[PublicFrontier, ...]
    decision_start_s: float
    decision_duration_s: float
    transit_timing_model: ConservativeTransitTimingModel
    observe_dwell_s: float
    # Isolated unit fixtures retain the development contract default; every
    # real P07 worker passes the hash-bound communication-contract value.
    communication_range_m: float = 10.0
    # A public execution outcome may retain a short-lived task association for
    # the next decision. It never bypasses fresh frontier extraction, routing,
    # static clearance or joint safety admission.
    task_reservations: tuple[PublicTaskReservation, ...] = ()

    def __post_init__(self) -> None:
        agents = tuple(sorted(self.agents, key=lambda item: item.agent_id))
        if not agents:
            raise ValueError("public search state needs at least one agent")
        agent_ids = tuple(item.agent_id for item in agents)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("public search state has duplicate agent IDs")
        context_ids = tuple(agent_id for agent_id, _ in self.context.agent_features)
        if agent_ids != context_ids:
            raise ValueError("public agent states must match PublicMethodContext exactly")
        frontiers = tuple(sorted(self.frontiers, key=lambda item: item.frontier_id))
        if len(frontiers) < len(agents):
            raise ValueError("public search state needs at least one frontier per agent")
        if len({item.frontier_id for item in frontiers}) != len(frontiers):
            raise ValueError("public search state has duplicate frontier IDs")
        sources = {frontier.source_agent_id for frontier in frontiers}
        if not {source for source in sources if source is not None} <= set(agent_ids):
            raise ValueError("public frontier source must match a public agent")
        access_path_agents = {
            agent_id
            for frontier in frontiers
            for agent_id, _path_m in frontier.access_paths_m
        }
        if not access_path_agents <= set(agent_ids):
            raise ValueError("public frontier access path must match a public agent")
        agent_positions = {agent.agent_id: agent.position_m for agent in agents}
        for frontier in frontiers:
            for agent_id, path_m in frontier.access_paths_m:
                if (
                    math.dist(path_m[0], agent_positions[agent_id])
                    > PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M
                ):
                    raise ValueError(
                        "public frontier access path must start at the current public agent pose"
                    )
        reservations = tuple(sorted(self.task_reservations, key=lambda item: item.agent_id))
        if len({reservation.agent_id for reservation in reservations}) != len(reservations):
            raise ValueError("public search state has duplicate task reservations")
        if not {reservation.agent_id for reservation in reservations} <= set(agent_ids):
            raise ValueError("public task reservation must match a public agent")
        start = finite_number(self.decision_start_s, "decision_start_s")
        duration = finite_number(self.decision_duration_s, "decision_duration_s")
        dwell = finite_number(self.observe_dwell_s, "observe_dwell_s")
        communication_range = finite_number(self.communication_range_m, "communication_range_m")
        if start < 0.0 or duration <= 0.0 or dwell <= 0.0:
            raise ValueError("decision timing and dwell must be positive")
        if communication_range <= 0.0:
            raise ValueError("communication_range_m must be positive")
        if not isinstance(self.transit_timing_model, ConservativeTransitTimingModel):
            raise TypeError("transit_timing_model must be a ConservativeTransitTimingModel")
        if dwell > duration:
            raise ValueError("observe dwell cannot exceed decision duration")
        object.__setattr__(self, "agents", agents)
        object.__setattr__(self, "frontiers", frontiers)
        object.__setattr__(self, "decision_start_s", start)
        object.__setattr__(self, "decision_duration_s", duration)
        object.__setattr__(self, "observe_dwell_s", dwell)
        object.__setattr__(self, "communication_range_m", communication_range)
        object.__setattr__(self, "task_reservations", reservations)
        walk_public_payload(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "context": self.context.to_dict(),
            "agents": [agent.to_dict() for agent in self.agents],
            "frontiers": [frontier.to_dict() for frontier in self.frontiers],
            "decision_start_s": self.decision_start_s,
            "decision_duration_s": self.decision_duration_s,
            "transit_timing_model": self.transit_timing_model.to_dict(),
            "observe_dwell_s": self.observe_dwell_s,
            "communication_range_m": self.communication_range_m,
        }
        if self.task_reservations:
            payload["task_reservations"] = [
                reservation.to_dict() for reservation in self.task_reservations
            ]
        return payload


@dataclass(frozen=True, slots=True)
class GuardedPath:
    """A runtime guard result; its path is safe to expose after guarding."""

    legal: bool
    path_m: tuple[Point3, ...]
    rewritten: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        path = tuple(_point3(point, "guarded_path_m") for point in self.path_m)
        if len(path) < 2:
            raise ValueError("guarded transit path requires at least two points")
        if not isinstance(self.legal, bool) or not isinstance(self.rewritten, bool):
            raise ValueError("guard result flags must be boolean")
        if self.reason:
            require_identifier(self.reason, "guard reason")
        object.__setattr__(self, "path_m", path)


PathGuard = Callable[[str, tuple[Point3, ...]], GuardedPath]
JointManifestGuard = Callable[[CandidateFragmentManifest], str | None]
_HOLD_ASSIGNMENT = -1
# Candidate generation must not feed a joint guard routes that are already
# known to occupy the same physical tube. The runtime guard remains the
# admission authority; this pre-filter only avoids spending the whole bounded
# pool on obvious simultaneous-route collisions. It matches the physical
# CF2X minimum separation in hm3d_cf2x_execution.
_PUBLIC_ROUTE_TUBE_SEPARATION_M = 0.50


def _assignment_route_tube_separation_m(
    guarded_edges: Mapping[tuple[int, int], Any],
    assignment: tuple[int, ...],
    agents: Sequence[Any],
) -> float:
    """Return the minimum spatial tube separation of one team assignment."""

    routes: list[TimedPolyline | TimedStationary] = []
    for agent_index, frontier_index in enumerate(assignment):
        agent = agents[agent_index]
        path = tuple(guarded_edges[(agent_index, frontier_index)].path_m)
        if frontier_index == _HOLD_ASSIGNMENT or len(path) < 2 or _path_length_m(path) <= 1.0e-9:
            routes.append(
                TimedStationary(agent.agent_id, path[0], 0.0, 1.0)
            )
        else:
            routes.append(
                TimedPolyline(agent.agent_id, path, 0.0, 1.0)
            )
    if len(routes) < 2:
        return 0.0
    return float(
        assess_route_tube_separation(
            routes,
            minimum_separation_m=_PUBLIC_ROUTE_TUBE_SEPARATION_M,
        ).minimum_route_separation_m
    )


def _manifest_route_tube_separation_m(
    manifest: CandidateFragmentManifest,
) -> float:
    """Return the same pre-filter separation evidence from a built manifest."""

    routes: list[TimedPolyline | TimedStationary] = []
    for fragment in manifest.fragments:
        if fragment.type_signature.fragment_type != "transit":
            continue
        features = dict(fragment.type_signature.public_features)
        path = tuple(fragment.path)
        if (
            features.get("assignment_role") == "hold"
            or len(path) < 2
            or _path_length_m(path) <= 1.0e-9
        ):
            routes.append(TimedStationary(fragment.agent_id, path[0], 0.0, 1.0))
        else:
            routes.append(TimedPolyline(fragment.agent_id, path, 0.0, 1.0))
    if len(routes) < 2:
        return 0.0
    return float(
        assess_route_tube_separation(
            routes,
            minimum_separation_m=_PUBLIC_ROUTE_TUBE_SEPARATION_M,
        ).minimum_route_separation_m
    )


def _assignment_joint_prefilter(
    guarded_edges: Mapping[tuple[int, int], Any],
    assignment: tuple[int, ...],
    agents: Sequence[Any],
) -> tuple[str | None, float, float]:
    """Return the shared joint-guard reason that would reject this assignment.

    The runtime joint guard remains the admission authority.  This pre-filter
    only mirrors its frozen route-tube, endpoint and translated-trajectory
    contracts so the bounded public pool does not fill its slots with
    assignments that can never be admitted.
    """

    routes: list[TimedPolyline | TimedStationary] = []
    paths_by_agent: dict[str, tuple[Point3, ...]] = {}
    roles_by_agent: dict[str, str] = {}
    endpoints: list[Point3] = []
    active_count = 0
    for agent_index, frontier_index in enumerate(assignment):
        agent = agents[agent_index]
        path = tuple(guarded_edges[(agent_index, frontier_index)].path_m)
        hold = frontier_index == _HOLD_ASSIGNMENT or _path_length_m(path) <= 1.0e-9
        if hold:
            routes.append(TimedStationary(agent.agent_id, path[0], 0.0, 1.0))
            roles_by_agent[agent.agent_id] = "hold"
            paths_by_agent[agent.agent_id] = (path[0], path[0])
        else:
            active_count += 1
            routes.append(TimedPolyline(agent.agent_id, path, 0.0, 1.0))
            roles_by_agent[agent.agent_id] = "explore"
            paths_by_agent[agent.agent_id] = path
        endpoints.append(path[-1])
    if active_count < 2:
        return None, 0.0, 0.0
    tube_separation = float(
        assess_route_tube_separation(
            routes,
            minimum_separation_m=_PUBLIC_ROUTE_TUBE_SEPARATION_M,
        ).minimum_route_separation_m
    )
    endpoint_separation = min(
        math.dist(left, right)
        for left_index, left in enumerate(endpoints)
        for right in endpoints[left_index + 1 :]
    )
    if tube_separation + 1.0e-9 < _PUBLIC_ROUTE_TUBE_SEPARATION_M:
        return "route_tube_separation", tube_separation, endpoint_separation
    if (
        endpoint_separation + 1.0e-9
        < _PUBLIC_ROUTE_EXTREME_ENDPOINT_SEPARATION_M
    ):
        return "planned_endpoint_separation_margin", tube_separation, endpoint_separation
    diversity = audit_translation_invariant_team_trajectories(
        paths_by_agent,
        roles_by_agent=roles_by_agent,
        scope="candidate_pool_joint_prefilter",
    )
    if diversity.has_translated_duplicate:
        return "translated_explorer_trajectory_copy", tube_separation, endpoint_separation
    return None, tube_separation, endpoint_separation


# This remains the outcome-backed recovery and route-prefix floor. It is not
# an eligibility threshold for ordinary observation frontiers: a short, legal
# doorway or vertical observation can carry genuine public information gain.
MINIMUM_MEANINGFUL_EXPLORATION_PATH_M = 0.50
# A public route whose *endpoint* is inside the executor's settled-position
# tolerance is an alias of the current command, not an exploration target.
# This is deliberately an endpoint identity test, not a universal route-length
# target or an incentive to fly farther. It must remain aligned with the v6
# CF2X waypoint settle tolerance through the runner regression test.
PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M = 0.03
# A reservation only associates frontiers extracted from public outcomes. It
# is deliberately a local identity radius, not a target path-length setting.
PUBLIC_TASK_RESERVATION_ASSOCIATION_RADIUS_M = 1.00
# Opposite frontier normals close to the same wall are distinct tasks. A
# missing normal keeps association possible for compatibility with simple
# public fixtures and outcome sources that cannot estimate one.
PUBLIC_TASK_RESERVATION_MIN_NORMAL_ALIGNMENT = 0.0
# Replanning away from a still-revalidated public task needs a frozen material
# public-gain advantage. This is a soft selection term only; routing and all
# safety authorities remain unchanged.
PUBLIC_TASK_RESERVATION_SWITCH_MARGIN_GAIN = 0.20
# This is the same opportunity definition used by the P07 outcome summary.
# It is a selection tie-break only, never a route-length target or safety rule.
PUBLIC_VERTICAL_OPPORTUNITY_THRESHOLD_M = 0.50
_MINIMUM_TEAM_ASSIGNMENT_SEARCH_BUDGET = 128
_TEAM_ASSIGNMENT_SEARCH_BUDGET_PER_OUTPUT = 32
# A traffic reservation is an executor-enforced earliest departure, not an
# idealized timing hint. The executor also waits for the predecessor's measured
# settled completion before releasing the delayed vehicle.
TRAFFIC_RESERVATION_RELEASE_MARGIN_S = 0.25
_TRAFFIC_RESERVATION_ASSIGNMENT_LIMIT_MULTIPLIER = 2
_TRAFFIC_RESERVATION_CHAIN_VARIANTS_PER_ASSIGNMENT = 4

# A pool that meets the numeric feasibility floor can still be degenerate:
# four "moving" rows may share one long route while the other vehicles move
# only a few centimetres. This is a shared task-validity criterion, not a
# selector preference or safety relaxation.
PUBLIC_TEAM_LONG_ROUTE_MIN_AGENT_PATH_M = 1.0
PUBLIC_TEAM_LONG_ROUTE_MIN_ACTIVE_AGENTS = 2
PUBLIC_TEAM_LONG_ROUTE_MIN_TEAM_PATH_M = 4.0
# Route-extreme ranking preference only. The runtime joint guard remains the
# admission authority; this value mirrors the frozen 0.95 m endpoint margin so
# the bounded enumerator does not spend its slots on four long routes that can
# never pass the endpoint separation contract.
_PUBLIC_ROUTE_EXTREME_ENDPOINT_SEPARATION_M = 0.95


def _initial_path_heading(path_m: Sequence[Point3]) -> Point3 | None:
    path = tuple(path_m)
    for start, end in zip(path, path[1:], strict=False):
        vector = tuple(end[axis] - start[axis] for axis in range(3))
        norm = math.sqrt(sum(component * component for component in vector))
        if norm > PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M:
            return tuple(component / norm for component in vector)  # type: ignore[return-value]
    return None


def _terminal_path_heading(path_m: Sequence[Point3]) -> Point3 | None:
    path = tuple(path_m)
    for start, end in zip(reversed(path[:-1]), reversed(path[1:]), strict=False):
        vector = tuple(end[axis] - start[axis] for axis in range(3))
        norm = math.sqrt(sum(component * component for component in vector))
        if norm > PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M:
            return tuple(component / norm for component in vector)  # type: ignore[return-value]
    return None


def is_non_alias_exploration_path(path_m: Sequence[Point3]) -> bool:
    """Accept a real endpoint change without imposing a route-length target.

    The guard may snap a request to the current settled point. Such a command
    cannot be credited as an observation transit. A short path to another
    public endpoint remains legal and is evaluated normally by the guard,
    joint safety, execution outcome, and no-gain cooldown.
    """

    path = tuple(path_m)
    if len(path) < 2:
        return False
    return math.dist(path[0], path[-1]) > PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M


def _task_reservation_for_agent(
    state: PublicSearchState,
    agent_id: str,
) -> PublicTaskReservation | None:
    for reservation in state.task_reservations:
        if reservation.agent_id == agent_id:
            return reservation
    return None


def _current_public_access_path(
    agent: PublicAgentPose,
    frontier: PublicFrontier | None,
) -> tuple[tuple[Point3, ...], bool]:
    """Use an access route only when it is anchored at this decision state.

    ``PublicFrontier`` routes are built from the current public belief by the
    runner.  The state check here is intentionally repeated at the common
    candidate boundary: a stale path from a previous robot pose must never be
    replayed as though it were a current plan.  Falling back to the endpoint
    request lets the runtime's public router revalidate it instead.
    """

    if frontier is None:
        return (agent.position_m, agent.position_m), False
    access_path = frontier.access_path_for_agent(agent.agent_id)
    if access_path is None:
        return (agent.position_m, frontier.position_m), False
    if (
        math.dist(access_path[0], agent.position_m) > PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M
        or math.dist(access_path[-1], frontier.position_m) > PUBLIC_ENDPOINT_ALIAS_TOLERANCE_M
    ):
        return (agent.position_m, frontier.position_m), False
    return access_path, True


def _guarded_public_path(
    state: PublicSearchState,
    guard: PathGuard,
    agent: PublicAgentPose,
    frontier: PublicFrontier | None,
) -> tuple[GuardedPath, bool]:
    """Guard the current access route shared by every selector."""

    requested_path, access_path_revalidated = _current_public_access_path(agent, frontier)
    return guard(agent.agent_id, requested_path), access_path_revalidated


def _public_gain_proxy(frontier: PublicFrontier | None) -> float:
    if frontier is None or frontier.task_kind != "explore":
        return 0.0
    gain = frontier.information_gain * (1.0 - frontier.traversal_risk)
    # A bounded route-access credit prevents equal-gain micro-observation views
    # from erasing a committed corridor/vertical route. It is a common candidate
    # quality hint, not a safety authority or a selector-only bonus.
    longest_access_m = 0.0
    for _agent_id, path in frontier.access_paths_m:
        if len(path) < 2:
            continue
        length_m = sum(
            math.dist(left, right)
            for left, right in zip(path, path[1:], strict=False)
        )
        longest_access_m = max(longest_access_m, length_m)
    if longest_access_m >= PUBLIC_ROUTE_CONTINUITY_BONUS_MIN_M:
        bonus = min(
            PUBLIC_ROUTE_CONTINUITY_BONUS_MAX,
            PUBLIC_ROUTE_CONTINUITY_BONUS_MAX
            * (longest_access_m - PUBLIC_ROUTE_CONTINUITY_BONUS_MIN_M)
            / PUBLIC_ROUTE_CONTINUITY_BONUS_RAMP_M,
        )
        return gain * (1.0 + bonus)
    return gain


def _frontier_cluster_key(frontier: PublicFrontier | None) -> str:
    """Return the public information unit represented by one frontier view."""

    if frontier is None or frontier.task_kind != "explore":
        return ""
    # Older callers may not yet provide extractor cluster provenance.  Their
    # frontier ID is still a stable public unit and must not collapse unrelated
    # views into one artificial cluster.
    return frontier.frontier_cluster_id or frontier.frontier_id


def _unique_cluster_gain_from_transit_features(
    transit_features: Sequence[Mapping[str, Any]],
) -> tuple[float, int]:
    """Count a public frontier cluster once, retaining its best legal view."""

    gains_by_cluster: dict[str, float] = {}
    for features in transit_features:
        if (
            features.get("assignment_role") != "explore"
            or features.get("task_kind") != "explore"
        ):
            continue
        cluster_key = str(features.get("frontier_cluster_id") or features.get("frontier_id") or "")
        if not cluster_key:
            raise ValueError("exploration transit is missing public frontier provenance")
        gain = finite_number(
            features.get("expected_public_gain_proxy", 0.0),
            "expected_public_gain_proxy",
        )
        gains_by_cluster[cluster_key] = max(gains_by_cluster.get(cluster_key, 0.0), gain)
    return sum(gains_by_cluster.values()), len(gains_by_cluster)


def task_reservation_matches_frontier(
    reservation: PublicTaskReservation | None,
    frontier: PublicFrontier | None,
 ) -> tuple[bool, float, float]:
    """Return public task association, anchor distance and normal alignment."""

    if reservation is None or frontier is None or frontier.task_kind != "explore":
        return False, 0.0, 0.0
    anchor_distance = math.dist(reservation.task_anchor_m, frontier.task_anchor_m)
    normal_alignment = 1.0
    if reservation.task_normal_unit is not None and frontier.task_normal_unit is not None:
        normal_alignment = min(
            1.0,
            max(
                -1.0,
                sum(
                    reservation.task_normal_unit[axis] * frontier.task_normal_unit[axis]
                    for axis in range(3)
                ),
            ),
        )
    return (
        anchor_distance <= PUBLIC_TASK_RESERVATION_ASSOCIATION_RADIUS_M + 1.0e-9
        and normal_alignment >= PUBLIC_TASK_RESERVATION_MIN_NORMAL_ALIGNMENT,
        anchor_distance,
        normal_alignment,
    )


def _task_reservation_features(
    reservation: PublicTaskReservation | None,
    frontier: PublicFrontier | None,
    guarded_path_m: Sequence[Point3],
) -> tuple[bool, float, float, float, float]:
    """Return task match, association evidence, heading and switch cost.

    Holds and outcome backtracks deliberately receive no reservation privilege.
    A normal candidate has already been freshly routed and guarded before this
    helper ranks it. A matched task continues only when its first path heading
    is not a reversal; a new task remains selectable after a fixed material
    public-gain margin rather than being permanently locked out.
    """

    matched, anchor_distance, normal_alignment = task_reservation_matches_frontier(
        reservation,
        frontier,
    )
    if reservation is None or frontier is None or frontier.task_kind != "explore":
        return False, anchor_distance, normal_alignment, 0.0, 0.0
    heading = _initial_path_heading(guarded_path_m)
    if heading is None:
        return matched, anchor_distance, normal_alignment, 0.0, 0.0
    heading_alignment = min(
        1.0,
        max(
            -1.0,
            sum(
                reservation.terminal_heading_unit[axis] * heading[axis]
                for axis in range(3)
            ),
        ),
    )
    switch_cost = PUBLIC_TASK_RESERVATION_SWITCH_MARGIN_GAIN * (
        (1.0 - heading_alignment) / 2.0
        + (0.0 if matched else 1.0)
    )
    return matched, anchor_distance, normal_alignment, heading_alignment, switch_cost


def identity_path_guard(agent_id: str, path_m: tuple[Point3, ...]) -> GuardedPath:
    """Test-only pass-through guard; never use this function for P07."""

    require_identifier(agent_id, "agent_id")
    return GuardedPath(legal=True, path_m=path_m)


def _cyclic_assignments(
    agent_count: int, frontier_count: int, limit: int
) -> tuple[tuple[int, ...], ...]:
    """Produce deterministic distinct frontier assignments without truth labels."""

    if limit < 1:
        raise ValueError("candidate_limit must be positive")
    assignments: list[tuple[int, ...]] = []
    for offset in range(frontier_count):
        assignment = tuple(
            (offset + agent_index) % frontier_count for agent_index in range(agent_count)
        )
        if len(set(assignment)) == agent_count:
            assignments.append(assignment)
        if len(assignments) >= limit:
            break
    if not assignments:
        raise ValueError("could not build a distinct public frontier assignment")
    return tuple(assignments)


def _shared_assignments(state: PublicSearchState, limit: int) -> tuple[tuple[int, ...], ...]:
    """Build method-neutral team assignments over every delivered public frontier.

    ``source_agent_id`` records which vehicle produced a sparse-range outcome;
    it is provenance, not task ownership.  Once that outcome is admitted to the
    public map, every selector must be able to assign any vehicle to its
    frontier.  Candidate guards, timing and team separation decide whether the
    resulting route is legal.
    """

    if limit < 1:
        raise ValueError("candidate_limit must be positive")
    if len(state.frontiers) < len(state.agents):
        raise ValueError("shared frontier assignment needs one frontier per agent")
    assignments: list[tuple[int, ...]] = []

    def is_assignable(agent_index: int, frontier_index: int) -> bool:
        exclusive_agent_id = state.frontiers[frontier_index].exclusive_agent_id
        return exclusive_agent_id is None or exclusive_agent_id == state.agents[agent_index].agent_id

    def append_once(assignment: tuple[int, ...]) -> None:
        if assignment not in assignments and len(assignments) < limit:
            assignments.append(assignment)

    def greedy_distinct(objective: Callable[[int, int], float]) -> tuple[int, ...]:
        unused = set(range(len(state.frontiers)))
        selected: list[int] = []
        for agent_index in range(len(state.agents)):
            eligible = tuple(index for index in unused if is_assignable(agent_index, index))
            if not eligible:
                raise ValueError("shared frontier assignment has no legal owner edge")
            frontier_index = max(
                eligible,
                key=lambda index: (objective(agent_index, index), -index),
            )
            selected.append(frontier_index)
            unused.remove(frontier_index)
        return tuple(selected)

    def vertical_delta(agent_index: int, frontier_index: int) -> float:
        return (
            state.frontiers[frontier_index].position_m[2] - state.agents[agent_index].position_m[2]
        )

    def path_distance(agent_index: int, frontier_index: int) -> float:
        return _distance(
            state.agents[agent_index].position_m,
            state.frontiers[frontier_index].position_m,
        )

    nearest = greedy_distinct(lambda agent, frontier: -path_distance(agent, frontier))
    gain_per_distance = greedy_distinct(
        lambda agent, frontier: (
            state.frontiers[frontier].information_gain
            * (1.0 - state.frontiers[frontier].traversal_risk)
            / max(path_distance(agent, frontier), 0.25)
        )
    )
    up = greedy_distinct(vertical_delta)
    down = greedy_distinct(lambda agent, frontier: -vertical_delta(agent, frontier))
    level = greedy_distinct(lambda agent, frontier: -abs(vertical_delta(agent, frontier)))
    high_gain = greedy_distinct(
        lambda _agent, frontier: (
            state.frontiers[frontier].information_gain
            * (1.0 - state.frontiers[frontier].traversal_risk)
        )
    )

    append_once(nearest)
    append_once(gain_per_distance)
    append_once(up)
    append_once(down)
    append_once(level)
    append_once(high_gain)
    append_once(tuple(up[index] if index % 2 == 0 else down[index] for index in range(len(up))))

    for assignment in _cyclic_assignments(
        len(state.agents), len(state.frontiers), max(limit, len(state.frontiers))
    ):
        if not all(
            is_assignable(agent_index, frontier_index)
            for agent_index, frontier_index in enumerate(assignment)
        ):
            continue
        append_once(assignment)
        if len(assignments) >= limit:
            break
    if not assignments:
        raise ValueError("could not build shared public frontier assignments")
    return tuple(assignments)


def _duration_for_path(
    path_m: tuple[Point3, ...], timing_model: ConservativeTransitTimingModel
) -> float:
    return timing_model.estimate_seconds(path_m)


def outcome_calibrated_path_length_budget_m(
    *,
    decision_duration_s: float,
    observe_dwell_s: float,
    transit_timing_model: ConservativeTransitTimingModel,
) -> float:
    """Return the reachable path length while preserving the sensing dwell.

    The calibrated tracking margin is part of transit time. Outcome timestamp
    tolerance is audit slack, not physical flight time, so it is not deducted.
    """

    duration = finite_number(decision_duration_s, "decision_duration_s")
    dwell = finite_number(observe_dwell_s, "observe_dwell_s")
    if duration <= 0.0 or dwell <= 0.0:
        raise ValueError("decision duration and observation dwell must be positive")
    return transit_timing_model.maximum_direct_path_length_m(duration - dwell)


def _path_length(path_m: Sequence[Point3]) -> float:
    return sum(math.dist(start, end) for start, end in zip(path_m, path_m[1:], strict=False))


def _public_candidate_intent(
    paths_m: Sequence[Sequence[Point3]],
    endpoints: Sequence[Point3],
    *,
    spatial_reference_m: float,
) -> tuple[float, float, float]:
    """Describe a *publicly intended* mode without treating it as realised.

    These values serve only to make the candidate emitter cover different
    possibilities and to index historical outcome evidence.  They must never
    be used as QD archive coordinates: guard rewriting, dynamics and sensing
    can make the realised descriptor different.
    """

    total_length_m = 0.0
    vertical_length_m = 0.0
    path_lengths: list[float] = []
    for path in paths_m:
        points = tuple(path)
        path_length_m = _path_length(points)
        path_lengths.append(path_length_m)
        total_length_m += path_length_m
        vertical_length_m += sum(
            abs(end[2] - start[2]) for start, end in zip(points, points[1:], strict=False)
        )
    vertical_motion_intent = (
        0.0 if total_length_m <= 1.0e-12 else vertical_length_m / total_length_m
    )

    reference = finite_number(spatial_reference_m, "spatial_reference_m")
    if reference <= 0.0:
        raise ValueError("spatial_reference_m must be positive")
    pair_distances = [
        math.dist(left, right)
        for index, left in enumerate(endpoints)
        for right in endpoints[index + 1 :]
    ]
    mean_pair_distance_m = 0.0 if not pair_distances else sum(pair_distances) / len(pair_distances)
    endpoint_dispersion_intent = min(1.0, max(0.0, mean_pair_distance_m / reference))

    # A formation that sends every vehicle in the same direction can have
    # balanced path lengths and still observe the same corridor.  Directional
    # complementarity instead measures the public planned displacement vectors
    # before execution; its only authority is to retrieve outcome-backed
    # history.  The realised archive uses observation complementarity below.
    directions: list[Point3] = []
    for path in paths_m:
        start, end = path[0], path[-1]
        vector = tuple(end[axis] - start[axis] for axis in range(3))
        norm = math.sqrt(sum(value * value for value in vector))
        direction = (0.0, 0.0, 0.0) if norm <= 1.0e-12 else tuple(value / norm for value in vector)
        directions.append(direction)
    directional_distances = [
        (1.0 - sum(left[axis] * right[axis] for axis in range(3))) / 2.0
        for index, left in enumerate(directions)
        for right in directions[index + 1 :]
    ]
    directional_complementarity_intent = (
        0.0
        if not directional_distances
        else sum(directional_distances) / len(directional_distances)
    )
    return (
        min(1.0, max(0.0, vertical_motion_intent)),
        endpoint_dispersion_intent,
        min(1.0, max(0.0, directional_complementarity_intent)),
    )


def _manifest_for_assignment(
    state: PublicSearchState,
    assignment: tuple[int, ...],
    guard: PathGuard,
    *,
    candidate_index: int,
    hold_reason_overrides: Mapping[str, str] | None = None,
    traffic_reservation_delays_s: Mapping[str, float] | None = None,
    traffic_reservation_predecessors: Mapping[str, str] | None = None,
    collision_avoidance_recovery_agent_id: str | None = None,
) -> CandidateFragmentManifest:
    fragments: list[FragmentInstance] = []
    endpoints: list[Point3] = []
    guarded_paths: list[tuple[Point3, ...]] = []
    total_cost = 0.0
    gains_by_cluster: dict[str, float] = {}
    feasible = True
    admission_reasons: list[str] = []
    duration_limit = state.decision_duration_s
    delays = dict(traffic_reservation_delays_s or {})
    predecessors = dict(traffic_reservation_predecessors or {})
    known_agent_ids = {agent.agent_id for agent in state.agents}
    if set(delays) - known_agent_ids or set(predecessors) - known_agent_ids:
        raise ValueError("traffic reservation references an unknown public agent")
    if set(delays) != set(predecessors):
        raise ValueError("traffic reservation delay and predecessor keys must match")
    if collision_avoidance_recovery_agent_id is not None:
        require_identifier(
            collision_avoidance_recovery_agent_id,
            "collision-avoidance recovery agent ID",
        )
        if collision_avoidance_recovery_agent_id not in known_agent_ids:
            raise ValueError("collision-avoidance recovery references an unknown public agent")
    for agent_id, delay_s in delays.items():
        delay = finite_number(delay_s, f"traffic reservation delay for {agent_id}")
        predecessor = predecessors[agent_id]
        if delay <= 0.0:
            raise ValueError("traffic reservation delay must be positive")
        if predecessor not in known_agent_ids or predecessor == agent_id:
            raise ValueError("traffic reservation predecessor must be another public agent")
    for agent_index, (agent, frontier_index) in enumerate(
        zip(state.agents, assignment, strict=True)
    ):
        holding = frontier_index == _HOLD_ASSIGNMENT
        # A stationary fragment is never presented as a relay task merely
        # because matching ran out of distinct public viewpoints.  The current
        # common candidate authority does not allocate explicit relay jobs;
        # this label keeps that limitation auditable in every manifest.
        hold_reason = ""
        if holding:
            hold_reason = (
                "no_reachable_viewpoint"
                if hold_reason_overrides is None
                else hold_reason_overrides.get(agent.agent_id, "no_reachable_viewpoint")
            )
            if hold_reason not in {
                "no_reachable_viewpoint",
                "collision_avoidance",
                "collision_avoidance_recovery",
                "waiting_for_team_completion",
            }:
                raise ValueError(f"unsupported public hold reason: {hold_reason}")
        frontier = None if holding else state.frontiers[frontier_index]
        reservation = _task_reservation_for_agent(state, agent.agent_id)
        if collision_avoidance_recovery_agent_id is not None:
            if agent.agent_id == collision_avoidance_recovery_agent_id:
                if (
                    holding
                    or frontier is None
                    or frontier.task_kind != "backtrack"
                    or frontier.exclusive_agent_id
                    != collision_avoidance_recovery_agent_id
                ):
                    raise ValueError(
                        "collision-avoidance recovery must use an owned backtrack route"
                    )
            else:
                if not holding:
                    raise ValueError(
                        "collision-avoidance recovery requires every nonrecovering agent to hold"
                    )
                if hold_reason != "collision_avoidance_recovery":
                    raise ValueError(
                        "collision-avoidance recovery requires an explicit recovery hold reason"
                    )
        guarded, access_path_revalidated = _guarded_public_path(
            state,
            guard,
            agent,
            frontier,
        )
        travel_s = _duration_for_path(guarded.path_m, state.transit_timing_model)
        reservation_delay_s = float(delays.get(agent.agent_id, 0.0))
        reservation_predecessor = predecessors.get(agent.agent_id, "")
        if holding and reservation_delay_s > 0.0:
            raise ValueError("a stationary hold cannot be used as a traffic reservation route")
        transit_start_s = state.decision_start_s + reservation_delay_s
        arrival_s = transit_start_s + travel_s
        minimum_observation_end_s = arrival_s + state.observe_dwell_s
        decision_end_s = state.decision_start_s + duration_limit
        within_window = minimum_observation_end_s <= decision_end_s + 1.0e-9
        legal = guarded.legal and within_window
        feasible = feasible and legal
        if not guarded.legal:
            admission_reasons.append(guarded.reason or "static_path_rejected")
        if not within_window:
            admission_reasons.append("decision_window_exceeded")
        # Planned observation duration is the minimum valid dwell. The executor
        # owns the actual completion timestamp and immediately returns once the
        # whole team has completed transit plus dwell; the remaining episode
        # budget is a deadline, never an instruction to hover until it expires.
        observation_end_s = minimum_observation_end_s
        endpoint = guarded.path_m[-1]
        endpoints.append(endpoint)
        guarded_paths.append(guarded.path_m)
        # Delay consumes common physical budget and hover energy. It is part
        # of the public effort hint, not a hidden planner-side correction.
        total_cost += travel_s + reservation_delay_s
        expected_public_gain_proxy = _public_gain_proxy(frontier)
        cluster_key = _frontier_cluster_key(frontier)
        if cluster_key:
            gains_by_cluster[cluster_key] = max(
                gains_by_cluster.get(cluster_key, 0.0), expected_public_gain_proxy
            )
        role = "hold" if holding else frontier.task_kind
        viewpoint_kind = "hold" if holding else frontier.viewpoint_kind
        (
            task_reservation_matched,
            task_reservation_anchor_distance_m,
            task_reservation_normal_alignment,
            task_reservation_heading_alignment,
            task_reservation_switch_cost,
        ) = _task_reservation_features(reservation, frontier, guarded.path_m)
        task_reservation_forward_compatible = (
            task_reservation_matched and task_reservation_heading_alignment >= 0.0
        )
        predicted_physical_makespan_s = observation_end_s - state.decision_start_s
        common_features = (
            ("frontier_rank", frontier_index),
            ("frontier_id", "" if frontier is None else frontier.frontier_id),
            ("viewpoint_kind", viewpoint_kind),
            ("guard_rewritten", guarded.rewritten),
            ("public_access_path_revalidated", access_path_revalidated),
            (
                "public_access_path_sha256",
                ""
                if not access_path_revalidated or frontier is None
                else canonical_sha256(frontier.access_path_for_agent(agent.agent_id)),
            ),
            (
                "frontier_cluster_id",
                "" if frontier is None else frontier.frontier_cluster_id,
            ),
            ("vertical_delta_m", endpoint[2] - agent.position_m[2]),
            ("assignment_role", role),
            ("task_kind", role),
            ("hold_reason", hold_reason),
            ("traffic_reservation_delay_s", reservation_delay_s),
            ("traffic_reservation_predecessor_agent_id", reservation_predecessor),
            (
                "safety_recovery_kind",
                (
                    "collision_avoidance_recovery"
                    if collision_avoidance_recovery_agent_id is not None
                    else ""
                ),
            ),
            ("safety_recovery_agent_id", collision_avoidance_recovery_agent_id or ""),
        )
        # Emit the task-association evidence for every normal frontier decision,
        # including the first one. This makes reservation preservation and
        # release auditable without an unrecorded scoring rule.
        common_features += (
            ("task_reservation_active", reservation is not None),
            ("task_reservation_matched", task_reservation_matched),
            ("task_reservation_forward_compatible", task_reservation_forward_compatible),
            ("task_reservation_anchor_distance_m", task_reservation_anchor_distance_m),
            ("task_reservation_normal_alignment", task_reservation_normal_alignment),
            ("task_reservation_heading_alignment", task_reservation_heading_alignment),
            ("task_reservation_switch_cost", task_reservation_switch_cost),
            ("expected_public_gain_proxy", expected_public_gain_proxy),
            ("predicted_physical_makespan_s", predicted_physical_makespan_s),
            (
                "task_reservation_source_decision_id",
                "" if reservation is None else reservation.source_decision_id,
            ),
            (
                "task_reservation_source_manifest_hash",
                "" if reservation is None else reservation.source_manifest_hash,
            ),
            (
                "task_reservation_source_public_path_sha256",
                "" if reservation is None else reservation.source_public_path_sha256,
            ),
            (
                "task_reservation_source_transit_outcome_sha256",
                "" if reservation is None else reservation.source_transit_outcome_sha256,
            ),
            (
                "task_reservation_source_frontier_cluster_id",
                "" if reservation is None else reservation.source_frontier_cluster_id,
            ),
        )
        fragments.append(
            FragmentInstance(
                instance_fragment_id=f"candidate{candidate_index}-agent{agent_index}-transit",
                type_signature=FragmentTypeSignature("transit", common_features),
                episode_id=state.context.episode_id,
                decision_id=state.context.decision_id,
                agent_id=agent.agent_id,
                planned_start=transit_start_s,
                planned_end=arrival_s,
                path=guarded.path_m,
                pose_mode="guarded_waypoint",
                context_bucket=(
                    "hm3d-collision-avoidance-recovery"
                    if collision_avoidance_recovery_agent_id is not None
                    else (
                        "hm3d-public-hold"
                        if holding
                        else (
                            "hm3d-outcome-backed-backtrack"
                            if role == "backtrack"
                            else "hm3d-public-frontier"
                        )
                    )
                ),
                guard_rewritten=guarded.rewritten,
            )
        )
        fragments.append(
            FragmentInstance(
                instance_fragment_id=f"candidate{candidate_index}-agent{agent_index}-observe",
                type_signature=FragmentTypeSignature(
                    "observation",
                    (
                        ("frontier_rank", frontier_index),
                        ("viewpoint_kind", viewpoint_kind),
                        ("assignment_role", role),
                        ("task_kind", role),
                        ("hold_reason", hold_reason),
                        ("traffic_reservation_delay_s", reservation_delay_s),
                        ("traffic_reservation_predecessor_agent_id", reservation_predecessor),
                        (
                            "safety_recovery_kind",
                            (
                                "collision_avoidance_recovery"
                                if collision_avoidance_recovery_agent_id is not None
                                else ""
                            ),
                        ),
                        (
                            "safety_recovery_agent_id",
                            collision_avoidance_recovery_agent_id or "",
                        ),
                        ("task_reservation_active", reservation is not None),
                        ("task_reservation_matched", task_reservation_matched),
                        (
                            "task_reservation_forward_compatible",
                            task_reservation_forward_compatible,
                        ),
                        (
                            "task_reservation_anchor_distance_m",
                            task_reservation_anchor_distance_m,
                        ),
                        (
                            "task_reservation_normal_alignment",
                            task_reservation_normal_alignment,
                        ),
                        (
                            "task_reservation_heading_alignment",
                            task_reservation_heading_alignment,
                        ),
                        ("task_reservation_switch_cost", task_reservation_switch_cost),
                        ("expected_public_gain_proxy", expected_public_gain_proxy),
                        ("predicted_physical_makespan_s", predicted_physical_makespan_s),
                        (
                            "task_reservation_source_decision_id",
                            "" if reservation is None else reservation.source_decision_id,
                        ),
                        (
                            "task_reservation_source_manifest_hash",
                            "" if reservation is None else reservation.source_manifest_hash,
                        ),
                        (
                            "task_reservation_source_public_path_sha256",
                            "" if reservation is None else reservation.source_public_path_sha256,
                        ),
                        (
                            "task_reservation_source_transit_outcome_sha256",
                            "" if reservation is None else reservation.source_transit_outcome_sha256,
                        ),
                        (
                            "task_reservation_source_frontier_cluster_id",
                            "" if reservation is None else reservation.source_frontier_cluster_id,
                        ),
                    ),
                ),
                episode_id=state.context.episode_id,
                decision_id=state.context.decision_id,
                agent_id=agent.agent_id,
                planned_start=arrival_s,
                planned_end=observation_end_s,
                path=(endpoint,),
                pose_mode="dwell",
                context_bucket=(
                    "hm3d-collision-avoidance-recovery"
                    if collision_avoidance_recovery_agent_id is not None
                    else (
                        "hm3d-public-hold"
                        if holding
                        else (
                            "hm3d-outcome-backed-backtrack"
                            if role == "backtrack"
                            else "hm3d-public-frontier"
                        )
                    )
                ),
                guard_rewritten=guarded.rewritten,
            )
        )
    descriptor = _public_candidate_intent(
        guarded_paths,
        endpoints,
        spatial_reference_m=state.communication_range_m,
    )
    return CandidateFragmentManifest(
        candidate_id=f"hm3d-public-candidate-{candidate_index}",
        context_hash=state.context.digest,
        fragments=tuple(fragments),
        planned_descriptor=descriptor,
        feasible=feasible,
        quality_hint=sum(gains_by_cluster.values()),
        cost_hint=total_cost,
        source=PUBLIC_CANDIDATE_POOL_SOURCE,
        admission_reasons=tuple(admission_reasons),
    )


def _collision_avoidance_fallback_assignments(
    assignments: Sequence[tuple[int, ...]],
    *,
    candidate_limit: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Derive bounded one-agent safety fallbacks from normal joint assignments.

    Individual reachability cannot establish that two CF2X routes can occupy a
    narrow corridor concurrently.  The joint guard is the authority on that
    question.  If it rejects the entire normal pool, holding one otherwise
    movable vehicle is a valid collision-avoidance action; labelling it as
    ``no_reachable_viewpoint`` would be false.  Do not construct these rows
    until the caller has established that the primary pool lacks enough safe
    candidates, so a safety fallback cannot dilute a healthy four-agent pool.
    """

    if candidate_limit < 1:
        raise ValueError("collision-avoidance fallback candidate limit must be positive")
    rows: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    fallback_limit = max(candidate_limit * 4, 8)
    for assignment in assignments:
        for agent_index, frontier_index in enumerate(assignment):
            if frontier_index == _HOLD_ASSIGNMENT:
                continue
            fallback = list(assignment)
            fallback[agent_index] = _HOLD_ASSIGNMENT
            fallback_tuple = tuple(fallback)
            if fallback_tuple in seen:
                continue
            # A fallback must still command at least one real task.  An
            # all-hold team is neither an exploration action nor a useful
            # candidate for a selector comparison.
            if all(index == _HOLD_ASSIGNMENT for index in fallback_tuple):
                continue
            seen.add(fallback_tuple)
            rows.append((fallback_tuple, (agent_index,)))
            if len(rows) >= fallback_limit:
                return tuple(rows)
    return tuple(rows)


def _outcome_backtrack_conflict_recovery_assignments(
    state: PublicSearchState,
    assignments: Sequence[tuple[int, ...]],
    guard: PathGuard,
    *,
    candidate_limit: int,
) -> tuple[tuple[int, ...], ...]:
    """Offer an owned outcome reversal only after joint traffic deadlock.

    A stationary collision-avoidance hold can itself occupy the only safe
    corridor.  A completed own-agent path provides a stronger alternative:
    its reverse endpoint and geometry have execution outcome authority, and
    the regular path guard still validates the current connector.  This stage
    never makes recovery a normal gain-seeking action; it is reached only when
    the maximal-participation pool and its stationary safety fallbacks both
    failed the common joint guard.
    """

    if candidate_limit < 1:
        raise ValueError("outcome-backtrack recovery candidate limit must be positive")
    rows: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    fallback_limit = max(candidate_limit * 4, 8)
    backtracks_by_agent: dict[int, tuple[int, ...]] = {}
    for agent_index, agent in enumerate(state.agents):
        legal: list[int] = []
        for frontier_index, frontier in enumerate(state.frontiers):
            if (
                frontier.task_kind != "backtrack"
                or frontier.exclusive_agent_id != agent.agent_id
            ):
                continue
            guarded, _ = _guarded_public_path(state, guard, agent, frontier)
            within_window = (
                state.decision_start_s
                + _duration_for_path(guarded.path_m, state.transit_timing_model)
                + state.observe_dwell_s
                <= state.decision_start_s + state.decision_duration_s + 1.0e-9
            )
            if (
                guarded.legal
                and within_window
                and _path_length(guarded.path_m) + 1.0e-9
                >= MINIMUM_MEANINGFUL_EXPLORATION_PATH_M
            ):
                legal.append(frontier_index)
        if legal:
            backtracks_by_agent[agent_index] = tuple(legal)

    for assignment in assignments:
        occupied = set(index for index in assignment if index != _HOLD_ASSIGNMENT)
        for agent_index, backtrack_indices in backtracks_by_agent.items():
            for frontier_index in backtrack_indices:
                if frontier_index in occupied and assignment[agent_index] != frontier_index:
                    continue
                recovery = list(assignment)
                recovery[agent_index] = frontier_index
                recovery_tuple = tuple(recovery)
                if recovery_tuple in seen:
                    continue
                seen.add(recovery_tuple)
                rows.append(recovery_tuple)
                if len(rows) >= fallback_limit:
                    return tuple(rows)
    return tuple(rows)


def _collision_avoidance_envelope_recovery_assignments(
    state: PublicSearchState,
    guard: PathGuard,
    *,
    candidate_limit: int,
) -> tuple[tuple[tuple[int, ...], str], ...]:
    """Offer one-agent outcome-backed exits after ordinary admission is exhausted.

    The caller's joint guard is still the authority for the physical and
    planning envelopes.  This generator merely makes the narrow recovery
    shape explicit: one owner follows a prior safe route in reverse while all
    other agents remain stationary.  It cannot become a gain-seeking team
    candidate because it is reached only after normal, reservation, hold and
    ordinary backtrack variants have all been rejected.
    """

    if candidate_limit < 1:
        raise ValueError("collision-avoidance recovery candidate limit must be positive")
    rows: list[tuple[tuple[int, ...], str]] = []
    for agent_index, agent in enumerate(state.agents):
        for frontier_index, frontier in enumerate(state.frontiers):
            if frontier.task_kind != "backtrack" or frontier.exclusive_agent_id != agent.agent_id:
                continue
            guarded, _ = _guarded_public_path(state, guard, agent, frontier)
            within_window = (
                state.decision_start_s
                + _duration_for_path(guarded.path_m, state.transit_timing_model)
                + state.observe_dwell_s
                <= state.decision_start_s + state.decision_duration_s + 1.0e-9
            )
            if (
                not guarded.legal
                or not within_window
                or _path_length(guarded.path_m) + 1.0e-9
                < MINIMUM_MEANINGFUL_EXPLORATION_PATH_M
            ):
                continue
            assignment = [_HOLD_ASSIGNMENT for _ in state.agents]
            assignment[agent_index] = frontier_index
            rows.append((tuple(assignment), agent.agent_id))
            if len(rows) >= max(candidate_limit * 2, 4):
                return tuple(rows)
    return tuple(rows)


def _nonconverging_recovery_path(
    path_m: Sequence[Point3],
    *,
    stationary_positions_m: Sequence[Point3],
) -> bool:
    """Return whether every segment is non-converging to every stationary UAV.

    For a straight segment, the dot product between the relative position and
    displacement is the derivative of half the squared separation.  Requiring
    it to be non-negative on every segment proves that the moving vehicle does
    not reduce distance to any stationary neighbour.  The runtime guard still
    checks the sampled continuous path against the physical separation.
    """

    path = tuple(path_m)
    if len(path) < 2 or not stationary_positions_m:
        return False
    for segment_start, segment_end in zip(path[:-1], path[1:], strict=True):
        displacement = tuple(
            segment_end[axis] - segment_start[axis] for axis in range(3)
        )
        if math.sqrt(sum(value * value for value in displacement)) <= 1.0e-9:
            continue
        for stationary in stationary_positions_m:
            relative = tuple(
                segment_start[axis] - stationary[axis] for axis in range(3)
            )
            if sum(relative[axis] * displacement[axis] for axis in range(3)) < -1.0e-9:
                return False
    return True


def _collision_avoidance_geometric_recovery_candidates(
    state: PublicSearchState,
    guard: PathGuard,
    *,
    candidate_limit: int,
) -> tuple[tuple[PublicSearchState, tuple[int, ...], str], ...]:
    """Build explicit one-agent escapes from the current public route graph.

    Outcome-backed reverse paths are unavailable at cold start and can be too
    short after a later decision.  In that case, reuse a freshly generated
    public access route only as a recovery candidate when its guarded geometry
    is monotone away from every other current pose.  The returned temporary
    state contains synthetic ``backtrack`` frontiers so the normal manifest
    schema records the recovery role without granting public gain.
    """

    if candidate_limit < 1:
        raise ValueError("geometric recovery candidate limit must be positive")
    synthetic: list[PublicFrontier] = []
    rows: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    maximum_rows = max(candidate_limit * 2, 4)
    for agent_index, agent in enumerate(state.agents):
        stationary = tuple(
            other.position_m
            for other_index, other in enumerate(state.agents)
            if other_index != agent_index
        )
        for frontier in state.frontiers:
            if frontier.task_kind != "explore":
                continue
            guarded, _ = _guarded_public_path(state, guard, agent, frontier)
            if not guarded.legal or not is_non_alias_exploration_path(guarded.path_m):
                continue
            if not _nonconverging_recovery_path(
                guarded.path_m,
                stationary_positions_m=stationary,
            ):
                continue
            initial_distances = tuple(
                math.dist(agent.position_m, other) for other in stationary
            )
            endpoint = guarded.path_m[-1]
            endpoint_distances = tuple(math.dist(endpoint, other) for other in stationary)
            if not any(
                endpoint_distance > initial_distance + 1.0e-6
                for endpoint_distance, initial_distance in zip(
                    endpoint_distances, initial_distances, strict=True
                )
            ):
                continue
            key = (agent.agent_id, frontier.frontier_id)
            if key in seen:
                continue
            seen.add(key)
            access_path = frontier.access_path_for_agent(agent.agent_id)
            synthetic.append(
                PublicFrontier(
                    frontier_id=(
                        f"collision-avoidance-recovery-{agent.agent_id}-"
                        f"{frontier.frontier_id}"
                    ),
                    position_m=frontier.position_m,
                    information_gain=0.0,
                    traversal_risk=0.0,
                    source_agent_id=agent.agent_id,
                    task_kind="backtrack",
                    exclusive_agent_id=agent.agent_id,
                    viewpoint_kind="collision_avoidance_recovery",
                    access_paths_m=(
                        ()
                        if access_path is None
                        else ((agent.agent_id, access_path),)
                    ),
                )
            )
            # PublicSearchState sorts frontiers by ID in __post_init__. Keep
            # the stable synthetic ID here and resolve its final tuple index
            # only after constructing the sorted recovery state.
            rows.append((synthetic[-1].frontier_id, agent.agent_id))
            if len(rows) >= maximum_rows:
                break
        if len(rows) >= maximum_rows:
            break
    if not rows:
        return ()
    recovery_state = replace(state, frontiers=(*state.frontiers, *synthetic))
    frontier_index_by_id = {
        frontier.frontier_id: index
        for index, frontier in enumerate(recovery_state.frontiers)
    }
    result: list[tuple[PublicSearchState, tuple[int, ...], str]] = []
    for synthetic_frontier_id, agent_id in rows:
        try:
            frontier_index = frontier_index_by_id[synthetic_frontier_id]
        except KeyError as exc:
            raise RuntimeError(
                "synthetic collision-avoidance recovery frontier was lost after sorting"
            ) from exc
        agent_index = next(
            index
            for index, agent in enumerate(recovery_state.agents)
            if agent.agent_id == agent_id
        )
        assignment = tuple(
            frontier_index if index == agent_index else _HOLD_ASSIGNMENT
            for index in range(len(recovery_state.agents))
        )
        result.append((recovery_state, assignment, agent_id))
    return tuple(result)


def _traffic_reservation_variants(
    state: PublicSearchState,
    assignments: Sequence[tuple[int, ...]],
    guard: PathGuard,
    *,
    candidate_limit: int,
) -> tuple[tuple[dict[str, float], dict[str, str]], ...]:
    """Offer bounded, common delayed-departure alternatives for bottlenecks.

    This routine does not infer a hidden corridor graph. Every pair is derived
    from the same guarded public paths exposed to every selector. A later joint
    guard must still prove full scheduled separation, while the executor makes
    the predecessor's *measured* settled arrival a second release condition.

    A single delayed agent behind one predecessor is enough for a two-vehicle
    crossing, but a corridor used by three or more vehicles needs a serial
    departure chain. The chain metadata keeps every delayed agent's immediate
    predecessor, so the executor only waits for that one settled arrival while
    the joint guard can verify the transitive schedule through predecessor
    edges.
    """

    if candidate_limit < 1:
        raise ValueError("traffic reservation candidate limit must be positive")
    variants: list[tuple[dict[str, float], dict[str, str]]] = []
    seen: set[tuple[tuple[str, float], tuple[str, str]]] = set()

    def _append_variant(
        delays: dict[str, float], predecessors: dict[str, str]
    ) -> None:
        signature = (tuple(sorted(delays.items())), tuple(sorted(predecessors.items())))
        if signature in seen:
            return
        seen.add(signature)
        variants.append((delays, predecessors))

    assignment_limit = min(
        len(assignments),
        max(candidate_limit * _TRAFFIC_RESERVATION_ASSIGNMENT_LIMIT_MULTIPLIER, 4),
    )
    for assignment in assignments[:assignment_limit]:
        active: list[tuple[str, float]] = []
        for agent, frontier_index in zip(state.agents, assignment, strict=True):
            if frontier_index == _HOLD_ASSIGNMENT:
                continue
            frontier = state.frontiers[frontier_index]
            if frontier.task_kind == "backtrack":
                # Outcome reversal is already a recovery protocol. Do not
                # stack a second traffic protocol on it.
                continue
            guarded, _ = _guarded_public_path(state, guard, agent, frontier)
            if guarded.legal:
                active.append(
                    (
                        agent.agent_id,
                        _duration_for_path(guarded.path_m, state.transit_timing_model),
                    )
                )
        for delayed_agent_id, delayed_duration_s in active:
            for predecessor_agent_id, predecessor_duration_s in active:
                if delayed_agent_id == predecessor_agent_id:
                    continue
                delay_s = predecessor_duration_s + TRAFFIC_RESERVATION_RELEASE_MARGIN_S
                if delay_s + delayed_duration_s + state.observe_dwell_s > (
                    state.decision_duration_s + 1.0e-9
                ):
                    continue
                delays = {delayed_agent_id: delay_s}
                predecessors = {delayed_agent_id: predecessor_agent_id}
                _append_variant(delays, predecessors)
        if len(active) < 3:
            continue
        duration_by_agent = dict(active)
        orders = (
            tuple(active),
            tuple(reversed(active)),
            tuple(sorted(active, key=lambda row: (row[1], row[0]))),
            tuple(sorted(active, key=lambda row: (-row[1], row[0]))),
        )
        for order in orders[: _TRAFFIC_RESERVATION_CHAIN_VARIANTS_PER_ASSIGNMENT]:
            delays: dict[str, float] = {}
            predecessors: dict[str, str] = {}
            release_end_s = duration_by_agent[order[0][0]]
            chain_valid = True
            for index in range(1, len(order)):
                agent_id, _ = order[index]
                delay_s = release_end_s + TRAFFIC_RESERVATION_RELEASE_MARGIN_S
                if (
                    delay_s
                    + duration_by_agent[agent_id]
                    + state.observe_dwell_s
                    > state.decision_duration_s + 1.0e-9
                ):
                    chain_valid = False
                    break
                delays[agent_id] = delay_s
                predecessors[agent_id] = order[index - 1][0]
                release_end_s = delay_s + duration_by_agent[agent_id]
            if chain_valid and delays:
                _append_variant(delays, predecessors)
    return tuple(variants)


def _manifest_transit_path_lengths(
    manifest: CandidateFragmentManifest,
) -> dict[str, float]:
    """Return guarded transit distance per agent for one shared candidate."""
    lengths: dict[str, float] = {}
    for fragment in manifest.fragments:
        if fragment.type_signature.fragment_type != "transit":
            continue
        length = sum(
            math.dist(left, right)
            for left, right in zip(fragment.path, fragment.path[1:], strict=False)
        )
        lengths[fragment.agent_id] = lengths.get(fragment.agent_id, 0.0) + length
    return lengths


def _has_meaningful_multi_agent_routes(
    manifest: CandidateFragmentManifest,
    *,
    min_agent_path_m: float = PUBLIC_TEAM_LONG_ROUTE_MIN_AGENT_PATH_M,
    min_active_agents: int = PUBLIC_TEAM_LONG_ROUTE_MIN_ACTIVE_AGENTS,
    min_team_path_m: float = PUBLIC_TEAM_LONG_ROUTE_MIN_TEAM_PATH_M,
) -> bool:
    lengths = tuple(_manifest_transit_path_lengths(manifest).values())
    if not lengths:
        return False
    active = sum(length >= min_agent_path_m - 1.0e-9 for length in lengths)
    return (
        active >= min_active_agents
        and sum(lengths) >= min_team_path_m - 1.0e-9
    )


def _waiting_hold_overrides_for_partial_route(
    state: PublicSearchState,
    guard: PathGuard,
    assignment: tuple[int, ...],
) -> dict[str, str] | None:
    """Return team-completion hold labels for a jointly useful partial route.

    A partial-active route is a deliberate resource allocation, not a missing
    viewpoint.  This helper labels the held agents only when the moving agents
    already form a meaningful multi-agent route.
    """

    if _HOLD_ASSIGNMENT not in assignment:
        return None
    lengths: list[float] = []
    for agent, frontier_index in zip(state.agents, assignment, strict=True):
        if frontier_index == _HOLD_ASSIGNMENT:
            continue
        frontier = state.frontiers[frontier_index]
        guarded, _ = _guarded_public_path(state, guard, agent, frontier)
        lengths.append(_path_length(guarded.path_m))
    if (
        len(lengths) < PUBLIC_TEAM_LONG_ROUTE_MIN_ACTIVE_AGENTS
        or any(
            length < PUBLIC_TEAM_LONG_ROUTE_MIN_AGENT_PATH_M - 1.0e-9
            for length in lengths
        )
        or sum(lengths) < PUBLIC_TEAM_LONG_ROUTE_MIN_TEAM_PATH_M - 1.0e-9
    ):
        return None
    return {
        state.agents[agent_index].agent_id: "waiting_for_team_completion"
        for agent_index, frontier_index in enumerate(assignment)
        if frontier_index == _HOLD_ASSIGNMENT
    }


def _feasibility_first_assignments(
    state: PublicSearchState,
    guard: PathGuard,
    *,
    candidate_limit: int,
    include_route_extreme: bool = True,
    include_partial_route_extreme: bool = False,
    require_joint_prefilter: bool = False,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Build a bounded, diverse set of assignments from individually legal edges.

    A bounded list of geometric assignments can be entirely infeasible even
    when a valid matching exists elsewhere in the public frontier graph.  The
    shared action authority therefore checks every public robot--frontier edge
    first and includes a guarded stationary observation edge.  It then retains
    public utility and descriptor extremes plus deterministic legal matchings.

    Materialising every distinct team permutation is not acceptable: with four
    vehicles and F frontiers it grows as P(F, 4), even though the public output
    contains only ``candidate_limit`` rows.  The fixed search budget is shared
    by every selector and depends only on that output limit, never on a method
    score or evaluator truth.
    """

    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    assignment_limit = max(
        _MINIMUM_TEAM_ASSIGNMENT_SEARCH_BUDGET,
        candidate_limit * _TEAM_ASSIGNMENT_SEARCH_BUDGET_PER_OUTPUT,
    )

    guarded_edges: dict[tuple[int, int], GuardedPath] = {}
    choices: list[tuple[int, ...]] = []
    for agent_index, agent in enumerate(state.agents):
        legal_exploration: list[int] = []
        legal_backtrack: list[int] = []
        for frontier_index, frontier in enumerate(state.frontiers):
            if (
                frontier.exclusive_agent_id is not None
                and frontier.exclusive_agent_id != agent.agent_id
            ):
                continue
            direct_lower_bound = (agent.position_m, frontier.position_m)
            if (
                state.decision_start_s
                + _duration_for_path(direct_lower_bound, state.transit_timing_model)
                + state.observe_dwell_s
                > state.decision_start_s + state.decision_duration_s + 1.0e-9
            ):
                # Any guarded route is at least as long as the direct Euclidean
                # segment.  Rejecting an impossible lower bound avoids an
                # expensive routed collision query without removing a feasible
                # edge from the shared candidate graph.
                continue
            guarded, _ = _guarded_public_path(state, guard, agent, frontier)
            guarded_edges[(agent_index, frontier_index)] = guarded
            within_window = (
                state.decision_start_s
                + _duration_for_path(guarded.path_m, state.transit_timing_model)
                + state.observe_dwell_s
                <= state.decision_start_s + state.decision_duration_s + 1.0e-9
            )
            # A guard may snap a nominal frontier request onto the current
            # settled point. That is not an exploration action and cannot
            # inherit a remote frontier's gain. Ordinary observation routes
            # otherwise retain short legal doorway/vertical moves; the 0.50 m
            # floor remains only for outcome-backed recovery routes.
            eligible_progress = (
                is_non_alias_exploration_path(guarded.path_m)
                if frontier.task_kind == "explore"
                else _path_length(guarded.path_m) + 1.0e-9
                >= MINIMUM_MEANINGFUL_EXPLORATION_PATH_M
            )
            if guarded.legal and within_window and eligible_progress:
                if frontier.task_kind == "explore":
                    legal_exploration.append(frontier_index)
                else:
                    legal_backtrack.append(frontier_index)
        # Recovery has no planning gain and may only rescue an agent that has
        # no ordinary, meaningful exploration edge in the current public map.
        # This keeps it from displacing normal exploration or becoming a
        # selector-specific shortcut.
        legal = legal_exploration if legal_exploration else legal_backtrack
        hold = guard(agent.agent_id, (agent.position_m, agent.position_m))
        guarded_edges[(agent_index, _HOLD_ASSIGNMENT)] = hold
        hold_within_window = (
            state.decision_start_s
            + _duration_for_path(hold.path_m, state.transit_timing_model)
            + state.observe_dwell_s
            <= state.decision_start_s + state.decision_duration_s + 1.0e-9
        )
        if not legal and hold.legal and hold_within_window:
            legal.append(_HOLD_ASSIGNMENT)
        if not legal:
            raise ValueError(f"public guard exposes no movement or hold edge for {agent.agent_id}")
        choices.append(tuple(legal))

    assignments: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def append_if_legal(assignment: tuple[int, ...], *, limit: int) -> None:
        if len(assignments) >= limit or assignment in seen:
            return
        if len(assignment) != len(state.agents):
            return
        explored = tuple(index for index in assignment if index != _HOLD_ASSIGNMENT)
        if not explored or len(explored) != len(set(explored)):
            return
        if any(index not in choices[agent] for agent, index in enumerate(assignment)):
            return
        if require_joint_prefilter:
            prefilter_reason, _, _ = _assignment_joint_prefilter(
                guarded_edges,
                assignment,
                state.agents,
            )
            if prefilter_reason is not None:
                return
        seen.add(assignment)
        assignments.append(assignment)

    def edge_distance(agent_index: int, frontier_index: int) -> float:
        return _path_length(guarded_edges[(agent_index, frontier_index)].path_m)

    def edge_gain(frontier_index: int) -> float:
        frontier = state.frontiers[frontier_index]
        return _public_gain_proxy(frontier)

    def task_reservation_switch_cost(agent_index: int, frontier_index: int) -> float:
        if frontier_index == _HOLD_ASSIGNMENT:
            return 0.0
        reservation = _task_reservation_for_agent(state, state.agents[agent_index].agent_id)
        return _task_reservation_features(
            reservation,
            state.frontiers[frontier_index],
            guarded_edges[(agent_index, frontier_index)].path_m,
        )[4]

    def task_reservation_matched(agent_index: int, frontier_index: int) -> bool:
        if frontier_index == _HOLD_ASSIGNMENT:
            return False
        reservation = _task_reservation_for_agent(state, state.agents[agent_index].agent_id)
        return _task_reservation_features(
            reservation,
            state.frontiers[frontier_index],
            guarded_edges[(agent_index, frontier_index)].path_m,
        )[0]

    def observation_priority(frontier_index: int) -> int:
        """Keep every continuously supported exploration view in one tier.

        ``route_progress`` and ``region_access`` are public-map route
        alternatives, not synthetic recovery actions. Treating them as a
        fallback after complete observations made the matcher discard the only
        continuous route through a corridor whenever four short observation
        poses existed. All three view kinds therefore participate in the same
        exploration tier; the route guard and joint guard still decide physical
        admissibility.
        """

        if frontier_index == _HOLD_ASSIGNMENT:
            return 0
        frontier = state.frontiers[frontier_index]
        return int(
            frontier.task_kind == "explore"
            and frontier.viewpoint_kind in {
                "observation",
                "route_progress",
                "region_access",
            }
        )

    def vertical_delta(agent_index: int, frontier_index: int) -> float:
        return (
            guarded_edges[(agent_index, frontier_index)].path_m[-1][2]
            - state.agents[agent_index].position_m[2]
        )

    def greedy_legal(objective: Callable[[int, int], float]) -> tuple[int, ...]:
        used: set[int] = set()
        selected: list[int] = []
        for agent_index, legal_choices in enumerate(choices):
            available = tuple(
                index for index in legal_choices if index != _HOLD_ASSIGNMENT and index not in used
            )
            if not available:
                selected.append(_HOLD_ASSIGNMENT)
                continue
            frontier_index = max(
                available,
                key=lambda index: (
                    observation_priority(index),
                    task_reservation_matched(agent_index, index),
                    objective(agent_index, index),
                    -index,
                ),
            )
            selected.append(frontier_index)
            used.add(frontier_index)
        return tuple(selected)

    objectives_list: list[Callable[[int, int], float]] = [
        lambda agent, frontier: -edge_distance(agent, frontier),
    ]
    if include_route_extreme:
        # Keep a route-length extreme in the bounded matching search.  This
        # is only a candidate-pool objective: the common guard, joint safety
        # certificate and physical deadline remain the admission authority.
        objectives_list.append(lambda agent, frontier: edge_distance(agent, frontier))
    objectives_list.extend(
        [
            lambda agent, frontier: (
                edge_gain(frontier) / max(edge_distance(agent, frontier), 0.25)
                - task_reservation_switch_cost(agent, frontier)
            ),
            lambda agent, frontier: (
                edge_gain(frontier) / max(edge_distance(agent, frontier), 0.25)
            ),
            vertical_delta,
            lambda agent, frontier: -vertical_delta(agent, frontier),
            lambda agent, frontier: -abs(vertical_delta(agent, frontier)),
            lambda _agent, frontier: edge_gain(frontier),
            lambda agent, frontier: (
                vertical_delta(agent, frontier)
                if agent % 2 == 0
                else -vertical_delta(agent, frontier)
            ),
        ]
    )
    objectives = tuple(objectives_list)
    seed_limit = assignment_limit // 2
    for objective in objectives:
        base = greedy_legal(objective)
        append_if_legal(base, limit=seed_limit)

    ordered_choices = tuple(
        tuple(
            sorted(
                legal_choices,
                key=lambda frontier: (
                    frontier == _HOLD_ASSIGNMENT,
                    -observation_priority(frontier),
                    0 if frontier == _HOLD_ASSIGNMENT else -int(
                        task_reservation_matched(agent_index, frontier)
                    ),
                    0.0
                    if frontier == _HOLD_ASSIGNMENT
                    else (
                        -edge_gain(frontier) / max(edge_distance(agent_index, frontier), 0.25)
                        + task_reservation_switch_cost(agent_index, frontier)
                    ),
                    frontier,
                ),
            )
        )
        for agent_index, legal_choices in enumerate(choices)
    )

    # Reserve bounded-search slots for route-length extremes.  Without this
    # reservation, the gain-density-first DFS can consume the entire
    # deterministic budget before it reaches a legal long-route matching.
    normal_assignment_limit = max(
        1, assignment_limit - (1 if include_route_extreme else 0)
    )

    def extend(prefix: tuple[int, ...], used_frontiers: frozenset[int]) -> None:
        if len(assignments) >= normal_assignment_limit:
            return
        agent_index = len(prefix)
        if agent_index == len(state.agents):
            append_if_legal(prefix, limit=normal_assignment_limit)
            return
        for frontier_index in ordered_choices[agent_index]:
            if frontier_index != _HOLD_ASSIGNMENT and frontier_index in used_frontiers:
                continue
            extend(
                prefix + (frontier_index,),
                (
                    used_frontiers
                    if frontier_index == _HOLD_ASSIGNMENT
                    else used_frontiers | {frontier_index}
                ),
            )
            if len(assignments) >= normal_assignment_limit:
                return

    extend((), frozenset())
    if not assignments:
        # A long receding-horizon episode can reach a decision where no
        # non-trivial team matching survives the joint guard (e.g. agents
        # dispersed into separate rooms with tube-conflicting routes).  A
        # crash here would terminate the whole episode and discard every
        # collected receipt.  Fall back to the weakest legal team action:
        # one explorer per best legal edge with the rest holding, then a
        # shared stationary hold.  These rows are still ranked and joint
        # guarded like any other candidate.
        fallback_assignments: list[tuple[int, ...]] = []
        for agent_index, ordered in enumerate(ordered_choices):
            non_hold = tuple(
                index for index in ordered if index != _HOLD_ASSIGNMENT
            )
            if not non_hold:
                continue
            best = non_hold[0]
            fallback = tuple(
                best if index == agent_index else _HOLD_ASSIGNMENT
                for index in range(len(state.agents))
            )
            if fallback not in fallback_assignments:
                fallback_assignments.append(fallback)
        assignments.extend(fallback_assignments[:2])
        if not assignments:
            assignments.append(tuple(_HOLD_ASSIGNMENT for _ in state.agents))
    if not assignments:
        raise ValueError("public feasible-edge graph has no non-trivial team matching")

    route_extreme_assignments: list[tuple[int, ...]] = []
    partial_route_extreme_assignments: list[tuple[int, ...]] = []
    if include_route_extreme:
        # Search the same individually guarded edge graph with a bounded
        # per-agent K-choice product.  This is deliberately independent of any
        # selector score or evaluator truth.  It prevents a public long-route
        # Pareto extreme from disappearing behind the gain-density DFS prefix
        # and, unlike an unbounded DFS, can compare endpoint separation before
        # a route reaches the joint guard.
        route_extreme_per_agent_k = 6
        route_extreme_pool_count = max(4, candidate_limit * 2)
        route_choices_by_agent: list[tuple[int, ...]] = []
        for agent_index, legal_choices in enumerate(choices):
            non_hold = tuple(
                frontier_index
                for frontier_index in legal_choices
                if frontier_index != _HOLD_ASSIGNMENT
            )
            deduplicated: list[int] = []
            seen_clusters: set[str] = set()
            for frontier_index in sorted(
                non_hold,
                key=lambda frontier_index: (
                    edge_distance(agent_index, frontier_index),
                    observation_priority(frontier_index),
                    -frontier_index,
                ),
                reverse=True,
            ):
                cluster_key = _frontier_cluster_key(state.frontiers[frontier_index])
                if cluster_key:
                    if cluster_key in seen_clusters:
                        continue
                    seen_clusters.add(cluster_key)
                deduplicated.append(frontier_index)
                if len(deduplicated) >= route_extreme_per_agent_k:
                    break
            route_choices_by_agent.append(tuple(deduplicated) + (_HOLD_ASSIGNMENT,))

        if all(route_choices_by_agent):
            route_combos: list[tuple[tuple[float, ...], tuple[int, ...]]] = []
            route_tube_separation_by_assignment: dict[
                tuple[int, ...], float
            ] = {}
            joint_prefilter_reason_by_assignment: dict[
                tuple[int, ...], str | None
            ] = {}
            for assignment in itertools.product(*route_choices_by_agent):
                explored = tuple(
                    frontier_index
                    for frontier_index in assignment
                    if frontier_index != _HOLD_ASSIGNMENT
                )
                if not explored or len(explored) != len(set(explored)):
                    continue
                lengths = tuple(
                    0.0
                    if frontier_index == _HOLD_ASSIGNMENT
                    else edge_distance(agent_index, frontier_index)
                    for agent_index, frontier_index in enumerate(assignment)
                )
                active_count = sum(
                    frontier_index != _HOLD_ASSIGNMENT
                    for frontier_index in assignment
                )
                prefilter_reason, minimum_route_tube_separation, minimum_endpoint_separation = (
                    _assignment_joint_prefilter(
                        guarded_edges,
                        assignment,
                        state.agents,
                    )
                    if active_count >= 2
                    else (None, 0.0, 0.0)
                )
                route_tube_separation_by_assignment[assignment] = (
                    minimum_route_tube_separation
                )
                joint_prefilter_reason_by_assignment[assignment] = (
                    prefilter_reason
                )
                cluster_count = len(
                    {
                        cluster_key
                        for frontier_index in explored
                        for cluster_key in (_frontier_cluster_key(state.frontiers[frontier_index]),)
                        if cluster_key
                    }
                )
                observation_count = sum(
                    observation_priority(frontier_index)
                    for frontier_index in explored
                )
                route_combos.append(
                    (
                        (
                            sum(lengths),
                            float(active_count),
                            minimum_endpoint_separation,
                            float(cluster_count),
                            float(observation_count),
                            *tuple(-float(index) for index in assignment),
                        ),
                        assignment,
                    )
                )
            endpoint_separated_combos = tuple(
                row
                for row in route_combos
                if row[0][2]
                >= _PUBLIC_ROUTE_EXTREME_ENDPOINT_SEPARATION_M - 1.0e-9
            )
            joint_prefilter_separated_combos = tuple(
                row
                for row in route_combos
                if joint_prefilter_reason_by_assignment[row[1]] is None
            )
            route_extreme_candidates = (
                joint_prefilter_separated_combos
                if require_joint_prefilter
                else (endpoint_separated_combos or route_combos)
            )
            for score, assignment in sorted(
                route_extreme_candidates,
                key=lambda row: row[0],
                reverse=True,
            )[:route_extreme_pool_count]:
                route_extreme_assignments.append(assignment)
                if assignment not in seen:
                    seen.add(assignment)
                    assignments.append(assignment)
            if include_partial_route_extreme:
                partial_active_quotas = {2: 4, 3: 4}
                partial_route_candidates: list[
                    tuple[tuple[float, ...], tuple[int, ...]]
                ] = []
                for active_count, quota in sorted(
                    partial_active_quotas.items(), reverse=True
                ):
                    if active_count >= len(state.agents):
                        continue
                    if require_joint_prefilter:
                        bucket_rows = tuple(
                            row
                            for row in route_combos
                            if int(row[0][1]) == active_count
                            and joint_prefilter_reason_by_assignment[row[1]] is None
                        )
                    else:
                        bucket_rows = tuple(
                            row
                            for row in route_combos
                            if int(row[0][1]) == active_count
                            and row[0][2]
                            >= _PUBLIC_ROUTE_EXTREME_ENDPOINT_SEPARATION_M - 1.0e-9
                        )
                    window_rows: list[
                        tuple[tuple[float, ...], tuple[int, ...]]
                    ] = []
                    for score, assignment in bucket_rows:
                        lengths = tuple(
                            0.0
                            if frontier_index == _HOLD_ASSIGNMENT
                            else edge_distance(agent_index, frontier_index)
                            for agent_index, frontier_index in enumerate(assignment)
                        )
                        if not all(
                            length
                            >= PUBLIC_TEAM_LONG_ROUTE_MIN_AGENT_PATH_M - 1.0e-9
                            for length in lengths
                            if length > 0.0
                        ):
                            continue
                        durations = tuple(
                            0.0
                            if frontier_index == _HOLD_ASSIGNMENT
                            else _duration_for_path(
                                guarded_edges[(agent_index, frontier_index)].path_m,
                                state.transit_timing_model,
                            )
                            for agent_index, frontier_index in enumerate(assignment)
                        )
                        if (
                            max(durations) + state.observe_dwell_s
                            > state.decision_duration_s + 1.0e-9
                        ):
                            continue
                        window_rows.append((score, assignment))
                    for score, assignment in sorted(
                        window_rows,
                        key=lambda row: (
                            row[0][2],
                            row[0][0],
                            row[0][3],
                            row[0][4],
                            row[0][5:],
                        ),
                        reverse=True,
                    )[:quota]:
                        partial_route_candidates.append((score, assignment))
                for score, assignment in partial_route_candidates:
                    if assignment in seen:
                        continue
                    seen.add(assignment)
                    assignments.append(assignment)
                    partial_route_extreme_assignments.append(assignment)
                    route_extreme_assignments.insert(0, assignment)
            route_extreme_assignments = [
                assignment
                for assignment in route_extreme_assignments
                if assignment in assignments
            ]

    def assignment_metrics(assignment: tuple[int, ...]) -> tuple[float, ...]:
        paths = tuple(
            guarded_edges[(agent_index, frontier_index)].path_m
            for agent_index, frontier_index in enumerate(assignment)
        )
        endpoints = tuple(path[-1] for path in paths)
        descriptor = _public_candidate_intent(
            paths,
            endpoints,
            spatial_reference_m=state.communication_range_m,
        )
        cluster_gains: dict[str, float] = {}
        for frontier_index in assignment:
            if frontier_index == _HOLD_ASSIGNMENT:
                continue
            frontier = state.frontiers[frontier_index]
            cluster_key = _frontier_cluster_key(frontier)
            if cluster_key:
                cluster_gains[cluster_key] = max(
                    cluster_gains.get(cluster_key, 0.0), _public_gain_proxy(frontier)
                )
        quality = sum(cluster_gains.values())
        unique_cluster_count = len(cluster_gains)
        cost = sum(_duration_for_path(path, state.transit_timing_model) for path in paths)
        task_reservation_switch_cost_total = sum(
            task_reservation_switch_cost(agent_index, frontier_index)
            for agent_index, frontier_index in enumerate(assignment)
        )
        hold_count = sum(index == _HOLD_ASSIGNMENT for index in assignment)
        observation_count = sum(
            observation_priority(index) for index in assignment if index != _HOLD_ASSIGNMENT
        )
        # Negative count preserves ascending ``min`` sorting for primary
        # exploration-view assignments while keeping a required route prefix
        # visible when the public matching graph has too few distinct views.
        return (
            float(hold_count),
            float(-observation_count),
            float(-unique_cluster_count),
            *descriptor,
            quality,
            cost,
            task_reservation_switch_cost_total,
        )

    metrics = {assignment: assignment_metrics(assignment) for assignment in assignments}
    ordered: list[tuple[int, ...]] = []

    def append_once(assignment: tuple[int, ...]) -> None:
        if assignment not in ordered:
            ordered.append(assignment)

    # A hold is a safety/fallback action, not a synthetic QD diversity mode.
    # Keep the largest feasible active fleet first, then provide descriptor
    # contrast only within that same participation level.  This guarantees an
    # all-four-active option survives whenever four compatible jobs exist.
    minimum_hold_count = min(int(metrics[row][0]) for row in assignments)
    maximum_exploration_view_count = max(
        -int(metrics[row][1])
        for row in assignments
        if int(metrics[row][0]) == minimum_hold_count
    )
    maximum_unique_cluster_count = max(
        -int(metrics[row][2])
        for row in assignments
        if (
            int(metrics[row][0]) == minimum_hold_count
            and -int(metrics[row][1]) == maximum_exploration_view_count
        )
    )
    primary_assignments = [
        row
        for row in assignments
        if int(metrics[row][0]) == minimum_hold_count
        and -int(metrics[row][1]) == maximum_exploration_view_count
        and -int(metrics[row][2]) == maximum_unique_cluster_count
    ]
    append_once(
        max(
            primary_assignments,
            key=lambda row: (metrics[row][6] - metrics[row][8], -metrics[row][7], row),
        )
    )
    for descriptor_axis in (3, 4, 5):
        append_once(min(primary_assignments, key=lambda row: (metrics[row][descriptor_axis], row)))
        append_once(max(primary_assignments, key=lambda row: (metrics[row][descriptor_axis], row)))
    for assignment in sorted(
        assignments,
        key=lambda row: (
            metrics[row][0],
            metrics[row][1],
            metrics[row][2],
            -(metrics[row][6] - metrics[row][8]),
            metrics[row][7],
            row,
        ),
    ):
        append_once(assignment)
    primary_set = frozenset(primary_assignments)
    primary_ordered_list = [assignment for assignment in ordered if assignment in primary_set]
    # Route extremes are common action-authority rows, not frontier_3d or RL
    # hints.  Insert them after the existing primary row so candidate_limit=1
    # retains legacy single-row behavior, while every normal comparison pool
    # (limit >= 2) receives the same long-route options before admission is
    # filled by short gain-density rows.
    if include_route_extreme and route_extreme_assignments:
        insertion_index = 1 if primary_ordered_list else 0
        for route_assignment in route_extreme_assignments[
            : max(1, candidate_limit - 1)
        ]:
            if route_assignment in primary_ordered_list:
                primary_ordered_list.remove(route_assignment)
            primary_ordered_list.insert(insertion_index, route_assignment)
            insertion_index += 1
    primary_ordered = tuple(primary_ordered_list)
    if not primary_ordered:
        raise RuntimeError("public matching omitted its primary exploration tier")
    return tuple(ordered), primary_ordered, tuple(partial_route_extreme_assignments)


def build_public_candidate_pool(
    state: PublicSearchState,
    guard: PathGuard,
    *,
    candidate_limit: int = 8,
    joint_guard: JointManifestGuard | None = None,
    minimum_feasible_candidates: int = 1,
    minimum_multi_agent_route_candidates: int = 0,
    include_route_extreme: bool = True,
    require_joint_prefilter: bool = False,
) -> tuple[CandidateFragmentManifest, ...]:
    """Build the shared P07 action authority before any baseline ranks it.

    ``minimum_feasible_candidates`` is a protocol guard, not a selection
    preference.  A shared pool with one legal row makes every ranked method
    select that row, so it cannot support a task-validity comparison.  The
    caller must therefore request at least two legal choices for a P07 pilot
    that intends to compare selectors.
    """

    if not callable(guard):
        raise ValueError("guard must be callable")
    if (
        not isinstance(minimum_feasible_candidates, int)
        or isinstance(minimum_feasible_candidates, bool)
        or minimum_feasible_candidates < 1
    ):
        raise ValueError("minimum_feasible_candidates must be a positive integer")
    if minimum_feasible_candidates > candidate_limit:
        raise ValueError("minimum_feasible_candidates cannot exceed candidate_limit")
    if (
        not isinstance(minimum_multi_agent_route_candidates, int)
        or isinstance(minimum_multi_agent_route_candidates, bool)
        or minimum_multi_agent_route_candidates < 0
    ):
        raise ValueError(
            "minimum_multi_agent_route_candidates must be a non-negative integer"
        )
    assignments, primary_assignments, _ = _feasibility_first_assignments(
        state,
        guard,
        candidate_limit=candidate_limit,
        include_route_extreme=include_route_extreme,
        require_joint_prefilter=require_joint_prefilter,
    )
    admitted: list[CandidateFragmentManifest] = []
    rejected: list[CandidateFragmentManifest] = []
    admitted_by_assignment: dict[tuple[int, ...], CandidateFragmentManifest] = {}
    long_route_assignments: set[tuple[int, ...]] = set()
    evaluated_variants: set[
        tuple[
            int,
            tuple[int, ...],
            tuple[tuple[str, float], ...],
            tuple[tuple[str, str], ...],
            str,
        ]
    ] = set()

    def evaluate(
        assignment: tuple[int, ...],
        *,
        candidate_index: int,
        hold_reason_overrides: Mapping[str, str] | None = None,
        traffic_reservation_delays_s: Mapping[str, float] | None = None,
        traffic_reservation_predecessors: Mapping[str, str] | None = None,
        collision_avoidance_recovery_agent_id: str | None = None,
        evaluation_state: PublicSearchState = state,
    ) -> None:
        delays = tuple(sorted((traffic_reservation_delays_s or {}).items()))
        predecessors = tuple(sorted((traffic_reservation_predecessors or {}).items()))
        variant_key = (
            len(evaluation_state.frontiers),
            assignment,
            delays,
            predecessors,
            collision_avoidance_recovery_agent_id or "",
            tuple(sorted((hold_reason_overrides or {}).items())),
        )
        if variant_key in evaluated_variants:
            return
        evaluated_variants.add(variant_key)
        manifest = _manifest_for_assignment(
            evaluation_state,
            assignment,
            guard,
            candidate_index=candidate_index,
            hold_reason_overrides=hold_reason_overrides,
            traffic_reservation_delays_s=traffic_reservation_delays_s,
            traffic_reservation_predecessors=traffic_reservation_predecessors,
            collision_avoidance_recovery_agent_id=collision_avoidance_recovery_agent_id,
        )
        if _has_meaningful_multi_agent_routes(manifest):
            long_route_assignments.add(assignment)
        reason = None if joint_guard is None or not manifest.feasible else joint_guard(manifest)
        if reason is not None:
            require_identifier(reason, "joint candidate admission reason")
            rejected.append(
                replace(
                    manifest,
                    feasible=False,
                    admission_reasons=manifest.admission_reasons + (reason,),
                )
            )
            return
        if manifest.feasible:
            stale = admitted_by_assignment.get(assignment)
            if stale is not None and stale in admitted:
                admitted.remove(stale)
            admitted_by_assignment[assignment] = manifest
            admitted.append(manifest)
        else:
            rejected.append(manifest)

    assignment_indices = {assignment: index for index, assignment in enumerate(assignments)}
    # Complete observations and public route prefixes are both normal
    # exploration actions. Keep the maximal-participation tier first, then use
    # other public variants only when it cannot provide strategy headroom.
    for assignment in primary_assignments:
        evaluate(
            assignment,
            candidate_index=assignment_indices[assignment],
            hold_reason_overrides=_waiting_hold_overrides_for_partial_route(
                state,
                guard,
                assignment,
            ),
        )
        if len(admitted) >= candidate_limit:
            break

    # Intent-diversity enhancement.  The candidate-intent richness audit that
    # QD strategies enforce rejects pools whose feasible candidates collapse
    # onto a few planned-descriptor cells, which can happen at a constrained
    # first decision even though more feasible assignments exist.  Evaluate the
    # remaining feasible assignments and keep a coverage-prioritised subset:
    # every distinct intent cell is retained with its best-quality row, and the
    # remaining budget fills with the highest-quality rows.  This changes only
    # which legal rows appear in the shared action authority; it cannot relax
    # the guard, joint safety, timing or diversity contracts.
    if joint_guard is not None:
        cells = tuple(
            HM3D_CANDIDATE_INTENT_SPEC.cell(tuple(candidate.planned_descriptor))
            for candidate in admitted
            if candidate.feasible
        )
        axis_bins = tuple(len({cell[axis] for cell in cells}) for axis in range(3))
        joint_cells = len(set(cells))
        if len(cells) >= minimum_feasible_candidates and (
            min(axis_bins) < 2 or joint_cells < 6
        ):
            for assignment in primary_assignments:
                if len(admitted) >= candidate_limit * 2:
                    break
                if assignment in admitted_by_assignment:
                    continue
                evaluate(
                    assignment,
                    candidate_index=assignment_indices.get(assignment, len(admitted)),
                )
            diverse_rows = [row for row in admitted if row.feasible]
            if len(diverse_rows) > candidate_limit:
                cell_of_row = {
                    id(row): HM3D_CANDIDATE_INTENT_SPEC.cell(tuple(row.planned_descriptor))
                    for row in diverse_rows
                }
                by_cell: dict[tuple[int, ...], list[CandidateFragmentManifest]] = {}
                for row in diverse_rows:
                    by_cell.setdefault(cell_of_row[id(row)], []).append(row)
                keep: list[CandidateFragmentManifest] = []
                for group in by_cell.values():
                    group.sort(key=lambda row: (-row.quality_hint, row.manifest_hash))
                    keep.append(group[0])
                remaining_budget = candidate_limit - len(keep)
                extras = sorted(
                    (row for group in by_cell.values() for row in group[1:]),
                    key=lambda row: (-row.quality_hint, row.manifest_hash),
                )
                keep.extend(extras[: max(0, remaining_budget)])
                admitted[:] = [row for row in admitted if row in keep]

    if (
        joint_guard is not None
        and minimum_multi_agent_route_candidates > 0
        and sum(
            _has_meaningful_multi_agent_routes(manifest) for manifest in admitted
        )
        < minimum_multi_agent_route_candidates
    ):
        _, _, partial_route_assignments = _feasibility_first_assignments(
            state,
            guard,
            candidate_limit=candidate_limit,
            include_route_extreme=True,
            include_partial_route_extreme=True,
            require_joint_prefilter=require_joint_prefilter,
        )
        partial_index = len(assignments) + candidate_limit * 6
        for assignment in partial_route_assignments:
            hold_reason_overrides = {
                state.agents[agent_index].agent_id: "waiting_for_team_completion"
                for agent_index, frontier_index in enumerate(assignment)
                if frontier_index == _HOLD_ASSIGNMENT
            }
            evaluate(
                assignment,
                candidate_index=partial_index,
                hold_reason_overrides=hold_reason_overrides,
            )
            partial_index += 1
            if (
                sum(
                    _has_meaningful_multi_agent_routes(manifest)
                    for manifest in admitted
                )
                >= minimum_multi_agent_route_candidates
            ):
                break

    # Numeric feasibility can be satisfied without any candidate that gives two
    # vehicles meaningful routes. When that happens, bounded delayed-departure
    # variants are the normal joint-safety rescue, not a selector hint.
    if (
        joint_guard is not None
        and minimum_multi_agent_route_candidates > 0
        and long_route_assignments
        and sum(
            _has_meaningful_multi_agent_routes(manifest) for manifest in admitted
        )
        < minimum_multi_agent_route_candidates
    ):
        reservation_index = len(assignments) + candidate_limit * 16
        for assignment in tuple(long_route_assignments):
            for delays, predecessors in _traffic_reservation_variants(
                state,
                (assignment,),
                guard,
                candidate_limit=candidate_limit,
            ):
                evaluate(
                    assignment,
                    candidate_index=reservation_index,
                    traffic_reservation_delays_s=delays,
                    traffic_reservation_predecessors=predecessors,
                )
                reservation_index += 1
                if (
                    sum(
                        _has_meaningful_multi_agent_routes(manifest)
                        for manifest in admitted
                    )
                    >= minimum_multi_agent_route_candidates
                ):
                    break
            if (
                sum(
                    _has_meaningful_multi_agent_routes(manifest)
                    for manifest in admitted
                )
                >= minimum_multi_agent_route_candidates
            ):
                break

    if len(admitted) < minimum_feasible_candidates:
        primary_set = frozenset(primary_assignments)
        for assignment in assignments:
            if assignment in primary_set:
                continue
            evaluate(assignment, candidate_index=assignment_indices[assignment])
            if len(admitted) >= candidate_limit:
                break

    # If every immediate joint plan is unsafe, allow one route to reserve a
    # later departure behind a real predecessor. This preserves multi-agent
    # work in a narrow corridor when the scheduled route certificate passes.
    # It is generated before stationary holding, so safety does not needlessly
    # turn a solvable traffic conflict into an idle vehicle.
    if len(admitted) < minimum_feasible_candidates and joint_guard is not None:
        reservation_index = len(assignments) + candidate_limit * 8
        for assignment in assignments[
            : max(candidate_limit * _TRAFFIC_RESERVATION_ASSIGNMENT_LIMIT_MULTIPLIER, 4)
        ]:
            for delays, predecessors in _traffic_reservation_variants(
                state,
                (assignment,),
                guard,
                candidate_limit=candidate_limit,
            ):
                evaluate(
                    assignment,
                    candidate_index=reservation_index,
                    traffic_reservation_delays_s=delays,
                    traffic_reservation_predecessors=predecessors,
                )
                reservation_index += 1
                if len(admitted) >= candidate_limit:
                    break
            if len(admitted) >= candidate_limit:
                break

    # Only a joint-route conflict can trigger this second stage.  It preserves
    # normal maximal-participation candidates whenever they are safe, while
    # avoiding a hard episode failure when a narrow corridor cannot carry all
    # individually legal paths at once.
    if len(admitted) < minimum_feasible_candidates and joint_guard is not None:
        for fallback_index, (assignment, held_agent_indices) in enumerate(
            _collision_avoidance_fallback_assignments(
                assignments,
                candidate_limit=candidate_limit,
            ),
            start=len(assignments),
        ):
            hold_reason_overrides = {
                state.agents[agent_index].agent_id: "collision_avoidance"
                for agent_index in held_agent_indices
            }
            evaluate(
                assignment,
                candidate_index=fallback_index,
                hold_reason_overrides=hold_reason_overrides,
            )
            if len(admitted) >= candidate_limit:
                break

    # A hold at a contested endpoint may still violate the same separation
    # envelope.  At that point an own outcome-backed reversal is the only
    # permitted yielding move: it is a real trajectory, not an invented route
    # or an unlabelled stationary action.
    if len(admitted) < minimum_feasible_candidates and joint_guard is not None:
        for recovery_index, assignment in enumerate(
            _outcome_backtrack_conflict_recovery_assignments(
                state,
                assignments,
                guard,
                candidate_limit=candidate_limit,
            ),
            start=len(assignments) + candidate_limit * 4,
        ):
            evaluate(assignment, candidate_index=recovery_index)
            if len(admitted) >= candidate_limit:
                break
    # The ordinary outcome backtrack may still be rejected because all routes
    # start slightly inside the *planning* envelope.  Only then expose the
    # one-moving-agent recovery shape.  Its stricter physical, stationarity,
    # non-convergence and endpoint-restoration contract is evaluated by the
    # runtime joint guard; the metadata keeps it out of exploration/QD/OGFR.
    if len(admitted) < minimum_feasible_candidates and joint_guard is not None:
        for recovery_index, (assignment, recovery_agent_id) in enumerate(
            _collision_avoidance_envelope_recovery_assignments(
                state,
                guard,
                candidate_limit=candidate_limit,
            ),
            start=len(assignments) + candidate_limit * 8,
        ):
            hold_reason_overrides = {
                agent.agent_id: "collision_avoidance_recovery"
                for agent in state.agents
                if agent.agent_id != recovery_agent_id
            }
            evaluate(
                assignment,
                candidate_index=recovery_index,
                hold_reason_overrides=hold_reason_overrides,
                collision_avoidance_recovery_agent_id=recovery_agent_id,
            )
            if len(admitted) >= candidate_limit:
                break
    # Cold-start or short-outcome states may not have a usable reverse route.
    # Reuse only a freshly guarded public route whose geometry monotonically
    # increases separation from every stationary neighbour.  The temporary
    # state marks the route as a recovery backtrack, so it cannot earn public
    # information gain or enter QD/OGFR replay.
    if len(admitted) < minimum_feasible_candidates and joint_guard is not None:
        for recovery_index, (recovery_state, assignment, recovery_agent_id) in enumerate(
            _collision_avoidance_geometric_recovery_candidates(
                state,
                guard,
                candidate_limit=candidate_limit,
            ),
            start=len(assignments) + candidate_limit * 12,
        ):
            hold_reason_overrides = {
                agent.agent_id: "collision_avoidance_recovery"
                for agent in recovery_state.agents
                if agent.agent_id != recovery_agent_id
            }
            evaluate(
                assignment,
                candidate_index=recovery_index,
                hold_reason_overrides=hold_reason_overrides,
                collision_avoidance_recovery_agent_id=recovery_agent_id,
                evaluation_state=recovery_state,
            )
            if len(admitted) >= candidate_limit:
                break
    if minimum_multi_agent_route_candidates > 0:
        long_admitted = [
            manifest
            for manifest in admitted
            if _has_meaningful_multi_agent_routes(manifest)
        ]
        if long_admitted:
            admitted = long_admitted + [
                manifest
                for manifest in admitted
                if not _has_meaningful_multi_agent_routes(manifest)
            ]
    pool = tuple((admitted + rejected)[:candidate_limit])
    feasible_count = sum(manifest.feasible for manifest in pool)
    if feasible_count == 0:
        diagnostics = "; ".join(
            f"{manifest.candidate_id}={','.join(manifest.admission_reasons) or 'rejected'}"
            for manifest in (admitted + rejected)
        )
        raise ValueError(f"runtime guard rejected every public candidate: {diagnostics}")
    if feasible_count < minimum_feasible_candidates:
        diagnostics = "; ".join(
            f"{manifest.candidate_id}={','.join(manifest.admission_reasons) or 'admitted'}"
            for manifest in pool
        )
        raise ValueError(
            "public candidate pool lacks strategy headroom: "
            f"feasible={feasible_count}, required={minimum_feasible_candidates}; {diagnostics}"
        )
    for manifest in pool:
        walk_public_payload(manifest.to_dict())
    return pool


def _auction_score(manifest: CandidateFragmentManifest) -> float:
    """Score a joint assignment with dimensionless public gain and effort.

    Transit fragments execute concurrently.  Subtracting their raw durations
    from information gain made a four-agent assignment look four times more
    expensive than a one-agent assignment and systematically selected one
    explorer plus three holds.  Normalize aggregate effort by the common team
    decision budget instead; this retains an explicit travel penalty without
    charging parallel flight as sequential wall time.
    """

    transits = tuple(
        fragment
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    )
    observations = tuple(
        fragment
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "observation"
    )
    if not transits or not observations:
        return float("-inf")
    decision_start_s = min(fragment.planned_start for fragment in transits)
    decision_end_s = max(fragment.planned_end for fragment in observations)
    team_budget_s = len(transits) * max(decision_end_s - decision_start_s, 1.0e-9)
    normalized_effort = manifest.cost_hint / team_budget_s
    return manifest.quality_hint - normalized_effort


def _frontier_3d_score(manifest: CandidateFragmentManifest) -> float:
    """Rank public team information gain without rewarding a shorter replan.

    Every candidate has already passed the common physical deadline, static
    guard and synchronized fleet-safety checks.  Dividing a local gain proxy
    by its own makespan therefore creates a myopic incentive to stop, observe,
    and select another nearby frontier instead of completing a valid public
    access route.  The fixed-horizon episode metric accounts for physical time;
    this high-level selector ranks expected public information, adds a bounded
    region-access progression credit, and subtracts only the outcome-grounded
    cost of abandoning a still-revalidated public task.
    """

    observations = tuple(
        fragment
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "observation"
    )
    if not observations:
        return float("-inf")
    transit_features = tuple(
        dict(fragment.type_signature.public_features)
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    )
    expected_gain_proxy, _ = _unique_cluster_gain_from_transit_features(transit_features)
    switch_cost = sum(
        float(features.get("task_reservation_switch_cost", 0.0))
        for features in transit_features
    )
    return (
        expected_gain_proxy
        - switch_cost
        + _frontier_3d_region_access_credit(manifest)
    )


def _frontier_3d_region_access_credit(manifest: CandidateFragmentManifest) -> float:
    """Credit legal region-access progress without changing safety or quality.

    Only ``region_access`` transit fragments receive this score term.  The
    credit ramps from zero to its bound over the frozen reference distance, so
    an accidental 3 mm prefix cannot masquerade as a region access route.
    """

    longest_access_m = 0.0
    for fragment in manifest.fragments:
        if fragment.type_signature.fragment_type != "transit":
            continue
        features = dict(fragment.type_signature.public_features)
        if features.get("viewpoint_kind") != "region_access":
            continue
        if len(fragment.path) < 2:
            continue
        longest_access_m = max(longest_access_m, _path_length_m(fragment.path))
    if longest_access_m <= 0.0:
        return 0.0
    return min(
        PUBLIC_REGION_ACCESS_CREDIT_MAX,
        PUBLIC_REGION_ACCESS_CREDIT_MAX
        * longest_access_m
        / PUBLIC_REGION_ACCESS_CREDIT_REFERENCE_M,
    )


def _candidate_semantic_hash(manifest: CandidateFragmentManifest) -> str:
    """Hash candidate meaning without generated row or fragment identifiers."""

    fragments = [
        {
            "agent_id": fragment.agent_id,
            "fragment_type": fragment.type_signature.fragment_type,
            "public_features": list(fragment.type_signature.public_features),
            "planned_start": fragment.planned_start,
            "planned_end": fragment.planned_end,
            "path": list(fragment.path),
            "pose_mode": fragment.pose_mode,
            "context_bucket": fragment.context_bucket,
            "guard_rewritten": fragment.guard_rewritten,
        }
        for fragment in manifest.fragments
    ]
    return canonical_sha256(
        {
            "context_hash": manifest.context_hash,
            "fragments": sorted(fragments, key=canonical_sha256),
            "planned_descriptor": list(manifest.planned_descriptor),
            "feasible": manifest.feasible,
            "quality_hint": manifest.quality_hint,
            "cost_hint": manifest.cost_hint,
            "admission_reasons": list(manifest.admission_reasons),
        }
    )


def _public_semantic_tie_key(
    manifest: CandidateFragmentManifest,
) -> tuple[float, float, float, str]:
    """Resolve equal public utilities without generated candidate numbering."""

    return (
        manifest.planned_descriptor[1],
        manifest.planned_descriptor[2],
        -manifest.cost_hint,
        _candidate_semantic_hash(manifest),
    )


def _vertical_access_count(manifest: CandidateFragmentManifest) -> int:
    """Count public exploration edges that cross the registered height band."""

    count = 0
    for fragment in manifest.fragments:
        if fragment.type_signature.fragment_type != "transit":
            continue
        features = dict(fragment.type_signature.public_features)
        if (
            features.get("assignment_role") == "explore"
            and features.get("task_kind") == "explore"
            and abs(float(features.get("vertical_delta_m", 0.0)))
            >= PUBLIC_VERTICAL_OPPORTUNITY_THRESHOLD_M - 1.0e-9
        ):
            count += 1
    return count


def _frontier_3d_selection_key(
    manifest: CandidateFragmentManifest,
    *,
    prioritize_vertical: bool = False,
) -> tuple[int, float, int, float, float, float, str]:
    transit_features = tuple(
        dict(fragment.type_signature.public_features)
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    )
    _, cluster_count = _unique_cluster_gain_from_transit_features(transit_features)
    return (
        _vertical_access_count(manifest) if prioritize_vertical else 0,
        _frontier_3d_score(manifest),
        cluster_count,
        *_public_semantic_tie_key(manifest),
    )


def _forward_reservation_count(manifest: CandidateFragmentManifest) -> int:
    """Count outcome-backed, direction-compatible exploration assignments.

    This is intentionally a selection hint, not a safety authority.  The
    candidate has already been routed and guarded; the count only expresses
    whether a still-valid public task can be continued without an immediate
    reversal.  Holds and outcome backtracks never receive continuity credit.
    """

    return sum(
        bool(features.get("task_reservation_forward_compatible"))
        and bool(features.get("task_reservation_active"))
        and features.get("assignment_role") == "explore"
        for features in (
            dict(fragment.type_signature.public_features)
            for fragment in manifest.fragments
            if fragment.type_signature.fragment_type == "transit"
        )
    )


def _select_frontier_3d_candidate(
    legal: Sequence[CandidateFragmentManifest],
) -> CandidateFragmentManifest:
    """Select a public frontier candidate with bounded task hysteresis.

    Replanning can otherwise reverse a vehicle merely because a newly emitted
    micro-frontier has a slightly higher local gain.  We first compute the
    ordinary public score for every legal row.  A forward-compatible
    reservation may win only inside the frozen material-gain margin; a larger
    new-task advantage still wins.  No path length, hidden map state, or
    physical safety check is changed here.
    """

    if not legal:
        raise ValueError("frontier selection requires at least one legal candidate")
    scores = {manifest.manifest_hash: _frontier_3d_score(manifest) for manifest in legal}
    best_score = max(scores.values())
    continuity_floor = best_score - PUBLIC_TASK_RESERVATION_SWITCH_MARGIN_GAIN
    continuity_candidates = tuple(
        manifest
        for manifest in legal
        if scores[manifest.manifest_hash] >= continuity_floor - 1.0e-12
        and _forward_reservation_count(manifest) > 0
    )
    candidates = continuity_candidates or tuple(legal)
    equivalent_candidates = tuple(
        manifest
        for manifest in candidates
        if scores[manifest.manifest_hash] >= continuity_floor - 1.0e-12
    )
    vertical_equivalent = tuple(
        manifest for manifest in equivalent_candidates if _vertical_access_count(manifest) > 0
    )
    if vertical_equivalent:
        candidates = vertical_equivalent
    # A legal cross-height public edge is valuable when the ordinary public
    # score is already within the frozen continuity margin.  Giving it
    # priority only inside this subset prevents an unproductive vertical
    # detour from beating a materially better public assignment.
    return max(
        candidates,
        key=lambda manifest: _frontier_3d_selection_key(
            manifest,
            prioritize_vertical=bool(vertical_equivalent),
        ),
    )


@dataclass(frozen=True, slots=True)
class BaselineSelection:
    strategy: str
    selected_manifest_hash: str
    selected_candidate_id: str
    scores: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.strategy not in BASELINE_STRATEGIES:
            raise ValueError("unsupported weak-baseline strategy")
        require_identifier(self.selected_candidate_id, "selected_candidate_id")
        if len(self.selected_manifest_hash) != 64:
            raise ValueError("selected_manifest_hash must be a SHA-256 digest")
        if not self.scores:
            raise ValueError("baseline selection needs candidate scores")

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "selected_manifest_hash": self.selected_manifest_hash,
            "selected_candidate_id": self.selected_candidate_id,
            "scores": list(self.scores),
        }


def select_public_baseline(
    strategy: str,
    pool: Sequence[CandidateFragmentManifest],
    *,
    random_key: int = 0,
) -> tuple[CandidateFragmentManifest, BaselineSelection]:
    """Choose from a common candidate pool without exposing evaluator truth."""

    if strategy not in BASELINE_STRATEGIES:
        raise ValueError(f"unsupported weak-baseline strategy {strategy!r}")
    rows = tuple(pool)
    if not rows:
        raise ValueError("candidate pool cannot be empty")
    legal = tuple(manifest for manifest in rows if manifest.feasible)
    if not legal:
        raise ValueError("candidate pool has no legal candidate")
    if strategy == "random":
        generator = random.Random(random_key)
        selected = legal[generator.randrange(len(legal))]
        scores = tuple((manifest.candidate_id, 1.0) for manifest in legal)
    elif strategy == "frontier_3d":
        scores = tuple((manifest.candidate_id, _frontier_3d_score(manifest)) for manifest in legal)
        selected = _select_frontier_3d_candidate(legal)
    elif strategy == "auction":
        scores = tuple((manifest.candidate_id, _auction_score(manifest)) for manifest in legal)
        selected = max(
            legal,
            key=lambda manifest: (_auction_score(manifest), *_public_semantic_tie_key(manifest)),
        )
    selection = BaselineSelection(
        strategy=strategy,
        selected_manifest_hash=selected.manifest_hash,
        selected_candidate_id=selected.candidate_id,
        scores=tuple(sorted(scores)),
    )
    walk_public_payload(selection.to_dict())
    return selected, selection


def fixed_altitude_frontiers(
    frontiers: Sequence[PublicFrontier], *, altitude_m: float
) -> tuple[PublicFrontier, ...]:
    """Construct the pre-registered fixed-height control input.

    The same guard subsequently decides which projected waypoints are legal.
    An empty legal pool is a valid constrained-control outcome; callers must
    record it rather than silently reverting to free height.
    """

    altitude = finite_number(altitude_m, "altitude_m")
    return tuple(
        PublicFrontier(
            frontier_id=frontier.frontier_id,
            position_m=(frontier.position_m[0], frontier.position_m[1], altitude),
            information_gain=frontier.information_gain,
            traversal_risk=frontier.traversal_risk,
            source_agent_id=frontier.source_agent_id,
            task_kind=frontier.task_kind,
            exclusive_agent_id=frontier.exclusive_agent_id,
            viewpoint_kind=frontier.viewpoint_kind,
            # Projection changes the endpoint, so reusing the old 3D access
            # route would violate the current-belief anchoring contract.
            access_paths_m=(),
            frontier_cluster_id=frontier.frontier_cluster_id,
            task_anchor_m=frontier.task_anchor_m,
            task_normal_unit=frontier.task_normal_unit,
        )
        for frontier in frontiers
    )


def public_candidate_pool_hash(pool: Sequence[CandidateFragmentManifest]) -> str:
    """Stable identity proving every baseline received the same candidate authority."""

    return canonical_sha256([manifest.to_dict() for manifest in pool])


__all__ = [
    "BASELINE_STRATEGIES",
    "BaselineSelection",
    "ConservativeTransitTimingModel",
    "GuardedPath",
    "JointManifestGuard",
    "PathGuard",
    "PublicAgentPose",
    "PublicFrontier",
    "PublicSearchState",
    "PublicTaskReservation",
    "PUBLIC_TASK_RESERVATION_ASSOCIATION_RADIUS_M",
    "PUBLIC_TASK_RESERVATION_SCHEMA_VERSION",
    "PUBLIC_TASK_RESERVATION_SWITCH_MARGIN_GAIN",
    "PUBLIC_VERTICAL_OPPORTUNITY_THRESHOLD_M",
    "TRANSIT_TIMING_SCHEMA_VERSION",
    "build_public_candidate_pool",
    "fixed_altitude_frontiers",
    "identity_path_guard",
    "public_candidate_pool_hash",
    "select_public_baseline",
    "task_reservation_matches_frontier",
]
