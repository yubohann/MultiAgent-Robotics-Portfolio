"""Lockstep PhysX execution for isolated four-CF2X HM3D clusters."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from aerocity_method.adapters.hm3d_execution import FragmentExecutionSample
from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.models import CandidateFragmentManifest, FragmentInstance
from aerocity_method.runtime.communication import (
    RelayGraphSnapshot,
    RelayMessage,
    RelayMessageQueue,
    build_range_los_relay_graph,
)
from aerocity_method.runtime.hm3d_belief import (
    PublicRangeObservationFrameOutcome,
    PublicRangeRayOutcome,
)
from aerocity_method.runtime.hm3d_cf2x_bitcraze_lee import BitcrazeLeeTracker
from aerocity_method.runtime.hm3d_cf2x_bitcraze_mellinger import BitcrazeMellingerTracker
from aerocity_method.runtime.hm3d_cf2x_execution import (
    BITCRAZE_LEE_CONTROLLER_ID,
    BITCRAZE_MELLINGER_CONTROLLER_ID,
    BITCRAZE_MELLINGER_OFFICIAL_CONTROL_RATE_HZ,
    CF2X_DEFAULT_CONTROLLER_ID,
    CF2X_EXECUTION_BACKEND_ID,
    CF2X_EXECUTION_EVIDENCE_CLASS,
    CF2X_MAX_FEEDBACK_ACCELERATION_MPS2,
    CF2X_MAXIMUM_TILT_RAD,
    CONTACT_HARD_FAIL_N,
    FLIGHT_CLEARANCE_M,
    HOVER_THRUST_PER_ROTOR_N,
    WAYPOINT_SETTLE_POSITION_TOLERANCE_M,
    _bounded_rotor_thrust,
    _clear_static_collision_los,
    _controller_tracking_profile,
    _energy_increment_j,
    _euler_xyz_from_quaternion_wxyz,
    _first_static_scene_hit,
    _minimum_observation_dwell_completed,
    _minimum_time_line_reference_with_boundary_speeds,
    _observation_failure_reason,
    _observation_source_identity,
    _path_length_m,
    _rate_limited_yaw_reference_deg,
    _route_corner_speed_mps,
    _scheduled_observation_completed,
    _sparse_range_sampling_phase,
    _waypoint_reached,
    _yaw_from_delta,
)
from aerocity_method.runtime.range_sensing import DENSE_26_RAY_PATTERN
from aerocity_method.runtime.range_sensing import (
    resolve_public_range_directions,
)
from aerocity_method.runtime.hm3d_multicluster import HM3DClusterLayout
from aerocity_method.runtime.hm3d_team_collaboration import (
    audit_translation_invariant_team_trajectories,
)

Point3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class VectorizedClusterExecutionResult:
    """One independently auditable cluster result from a shared PhysX step loop."""

    manifest_hash: str
    token_hash: str
    samples: tuple[FragmentExecutionSample, ...]
    engineering_diagnostics: dict[str, object]
    public_range_frames: tuple[PublicRangeObservationFrameOutcome, ...]
    public_range_outcomes: tuple[PublicRangeRayOutcome, ...]
    public_map_sender_ids: tuple[str, ...]
    final_root_positions_m: tuple[Point3, ...]

    def precomputed_backend(self) -> PrecomputedClusterExecutionBackend:
        return PrecomputedClusterExecutionBackend(self)


@dataclass(slots=True)
class PrecomputedClusterExecutionBackend:
    """Expose a batch result through the existing outcome-ledger interface."""

    result: VectorizedClusterExecutionResult
    backend_id: str = CF2X_EXECUTION_BACKEND_ID
    evidence_class: str = CF2X_EXECUTION_EVIDENCE_CLASS
    engineering_diagnostics: dict[str, object] = field(init=False)
    public_range_frames: tuple[PublicRangeObservationFrameOutcome, ...] = field(init=False)
    public_range_outcomes: tuple[PublicRangeRayOutcome, ...] = field(init=False)
    public_map_sender_ids: tuple[str, ...] = field(init=False)
    final_root_positions_m: tuple[Point3, ...] = field(init=False)
    last_execution_samples: tuple[FragmentExecutionSample, ...] = field(init=False)
    _consumed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.engineering_diagnostics = self.result.engineering_diagnostics
        self.public_range_frames = self.result.public_range_frames
        self.public_range_outcomes = self.result.public_range_outcomes
        self.public_map_sender_ids = self.result.public_map_sender_ids
        self.final_root_positions_m = self.result.final_root_positions_m
        self.last_execution_samples = self.result.samples

    def execute_manifest(
        self, manifest: CandidateFragmentManifest, token: Any
    ) -> tuple[FragmentExecutionSample, ...]:
        if self._consumed:
            raise RuntimeError("precomputed cluster execution result was already consumed")
        if manifest.manifest_hash != self.result.manifest_hash:
            raise ValueError("precomputed batch result manifest mismatch")
        if token.digest != self.result.token_hash:
            raise ValueError("precomputed batch result token mismatch")
        self._consumed = True
        return self.result.samples


@dataclass(slots=True)
class _ClusterRuntime:
    manifest: CandidateFragmentManifest
    token: Any
    routes: list[tuple[FragmentInstance, FragmentInstance]]
    horizon_s: float
    execution_deadline_s: float
    transit_traces: list[list[Point3]]
    observation_traces: list[list[Point3]]
    energy_j: list[float]
    connected_at_all_samples: list[bool]
    transit_end_s: list[float | None]
    observation_start_s: list[float | None]
    observation_end_s: list[float | None]
    transit_contact: list[bool]
    transit_oob: list[bool]
    observation_contact: list[bool]
    observation_oob: list[bool]
    failed: list[bool]
    waypoint_index: list[int]
    segment_start_s: list[float]
    segment_start_m: list[Point3]
    segment_initial_speed_mps: list[float]
    route_corner_speed_mps: list[float]
    first_collision_step: list[int | None]
    first_collision_position: list[Point3 | None]
    first_collision_waypoint: list[Point3 | None]
    waypoint_transitions: list[list[dict[str, object]]]
    maximum_contact_force_n: list[float]
    maximum_linear_speed_mps: list[float]
    maximum_linear_acceleration_mps2: list[float]
    last_sensor_timestamp_s: list[float | None]
    last_sensor_source_id: list[str | None]
    sensor_frames_by_agent: list[int]
    range_frames_by_phase: dict[str, int]
    public_range_frames: list[PublicRangeObservationFrameOutcome]
    public_range_outcomes: list[PublicRangeRayOutcome]
    source_observation_ids_by_agent: dict[str, list[str]]
    message_queue: RelayMessageQueue
    relay_measurement_count: int = 0
    relay_fully_connected_count: int = 0
    relay_direct_link_count_sum: int = 0
    relay_component_count_max: int = 0
    relay_maximum_hops_max: int = 0
    current_disconnect_started_s: float | None = None
    longest_disconnected_duration_s: float = 0.0
    partition_event_count: int = 0
    reconnection_count: int = 0
    previous_relay_connected: bool | None = None
    last_relay_graph: RelayGraphSnapshot | None = None
    last_communication_measurement_s: float | None = None


def _routes(
    manifest: CandidateFragmentManifest, agent_order: tuple[str, ...]
) -> list[tuple[FragmentInstance, FragmentInstance]]:
    by_agent: dict[str, list[FragmentInstance]] = defaultdict(list)
    for fragment in manifest.fragments:
        by_agent[fragment.agent_id].append(fragment)
    if tuple(sorted(by_agent)) != agent_order:
        raise ValueError("manifest agents do not match the four-CF2X cluster")
    rows: list[tuple[FragmentInstance, FragmentInstance]] = []
    for agent_id in agent_order:
        fragments = sorted(by_agent[agent_id], key=lambda row: row.planned_start)
        kinds = [row.type_signature.fragment_type for row in fragments]
        if len(fragments) != 2 or kinds != ["transit", "observation"]:
            raise ValueError("each vectorized cluster requires one transit and observation per UAV")
        rows.append((fragments[0], fragments[1]))
    return rows


@dataclass(slots=True)
class IsaacCF2XVectorizedExecutionBackend:
    """Execute multiple isolated team manifests in one shared PhysX step loop."""

    sim: Any
    robot: Any
    contact: Any
    scene_query: Any
    static_clearance_oracle: Any
    layout: HM3DClusterLayout
    bounds_min_m: Point3
    bounds_max_m: Point3
    arrival_tolerance_m: float
    communication_max_range_m: float = 10.0
    communication_base_latency_s: float = 0.05
    communication_per_hop_latency_s: float = 0.02
    communication_loss_probability: float = 0.0
    communication_update_hz: float = 10.0
    sparse_range_update_hz: float = 10.0
    sparse_range_directions: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: resolve_public_range_directions(DENSE_26_RAY_PATTERN)
    )
    sparse_range_max_m: float = 20.0
    communication_message_ttl_s: float = 0.5
    minimum_observation_dwell_s: float = 1.0
    event_driven_action_completion: bool = True
    calibration_timeout_probe_s: float | None = None
    controller_id: str = CF2X_DEFAULT_CONTROLLER_ID
    agent_order: tuple[str, ...] = ("uav0", "uav1", "uav2", "uav3")

    def _world_graph(self, positions_w: list[Point3]) -> RelayGraphSnapshot:
        return build_range_los_relay_graph(
            positions_w,
            maximum_range_m=self.communication_max_range_m,
            line_of_sight_clear=lambda source, target: _clear_static_collision_los(
                self.scene_query, source, target
            ),
        )

    def _new_runtime(self, manifest: CandidateFragmentManifest, token: Any) -> _ClusterRuntime:
        routes = _routes(manifest, self.agent_order)
        horizon_s = float(token.duration)
        if not math.isfinite(horizon_s) or horizon_s <= 0.0:
            raise ValueError("vectorized execution token duration must be positive")
        execution_deadline_s = (
            horizon_s
            if self.calibration_timeout_probe_s is None
            else float(self.calibration_timeout_probe_s)
        )
        if (
            not math.isfinite(execution_deadline_s)
            or execution_deadline_s <= 0.0
            or execution_deadline_s > horizon_s
        ):
            raise ValueError(
                "vectorized execution deadline must be positive and not exceed the token"
            )
        if self.calibration_timeout_probe_s is not None and not execution_deadline_s < horizon_s:
            raise ValueError("calibration timeout probe must end before the token duration")
        return _ClusterRuntime(
            manifest=manifest,
            token=token,
            routes=routes,
            horizon_s=horizon_s,
            execution_deadline_s=execution_deadline_s,
            transit_traces=[[tuple(transit.path[0])] for transit, _ in routes],
            observation_traces=[[] for _ in routes],
            energy_j=[0.0 for _ in routes],
            connected_at_all_samples=[True for _ in routes],
            transit_end_s=[None for _ in routes],
            observation_start_s=[None for _ in routes],
            observation_end_s=[None for _ in routes],
            transit_contact=[False for _ in routes],
            transit_oob=[False for _ in routes],
            observation_contact=[False for _ in routes],
            observation_oob=[False for _ in routes],
            failed=[False for _ in routes],
            waypoint_index=[1 for _ in routes],
            segment_start_s=[0.0 for _ in routes],
            segment_start_m=[tuple(transit.path[0]) for transit, _ in routes],
            segment_initial_speed_mps=[0.0 for _ in routes],
            route_corner_speed_mps=[
                _route_corner_speed_mps(transit.path) for transit, _ in routes
            ],
            first_collision_step=[None for _ in routes],
            first_collision_position=[None for _ in routes],
            first_collision_waypoint=[None for _ in routes],
            waypoint_transitions=[[] for _ in routes],
            maximum_contact_force_n=[0.0 for _ in routes],
            maximum_linear_speed_mps=[0.0 for _ in routes],
            maximum_linear_acceleration_mps2=[0.0 for _ in routes],
            last_sensor_timestamp_s=[None for _ in routes],
            last_sensor_source_id=[None for _ in routes],
            sensor_frames_by_agent=[0 for _ in routes],
            range_frames_by_phase={"transit": 0, "dwell": 0},
            public_range_frames=[],
            public_range_outcomes=[],
            source_observation_ids_by_agent={agent_id: [] for agent_id in self.agent_order},
            message_queue=RelayMessageQueue(
                self.agent_order,
                self.communication_base_latency_s,
                self.communication_per_hop_latency_s,
                self.communication_loss_probability,
            ),
        )

    def _record_graph(
        self,
        runtime: _ClusterRuntime,
        timestamp_s: float,
        positions_w: list[Point3],
        communication_interval_s: float,
    ) -> RelayGraphSnapshot:
        graph = self._world_graph(positions_w)
        runtime.relay_measurement_count += 1
        runtime.relay_fully_connected_count += int(graph.fully_relay_connected)
        runtime.relay_direct_link_count_sum += graph.direct_link_count
        runtime.relay_component_count_max = max(
            runtime.relay_component_count_max, len(graph.components)
        )
        runtime.relay_maximum_hops_max = max(
            runtime.relay_maximum_hops_max, graph.maximum_relay_hops or 0
        )
        if graph.fully_relay_connected:
            if runtime.current_disconnect_started_s is not None:
                runtime.longest_disconnected_duration_s = max(
                    runtime.longest_disconnected_duration_s,
                    timestamp_s - runtime.current_disconnect_started_s,
                )
                runtime.current_disconnect_started_s = None
            if runtime.previous_relay_connected is False:
                runtime.reconnection_count += 1
        else:
            if runtime.current_disconnect_started_s is None:
                runtime.current_disconnect_started_s = max(
                    0.0, timestamp_s - communication_interval_s
                )
            if runtime.previous_relay_connected is True:
                runtime.partition_event_count += 1
        runtime.previous_relay_connected = graph.fully_relay_connected
        runtime.last_relay_graph = graph
        runtime.last_communication_measurement_s = timestamp_s
        for index, connected in enumerate(graph.agent_relay_reachable_to_all):
            runtime.connected_at_all_samples[index] = (
                runtime.connected_at_all_samples[index] and connected
            )
        return graph

    def execute_manifests(
        self,
        manifests: tuple[CandidateFragmentManifest, ...],
        tokens: tuple[Any, ...],
    ) -> tuple[VectorizedClusterExecutionResult, ...]:
        import torch

        execution_wall_started = time.perf_counter()
        wall_seconds = {
            "batch_setup": 0.0,
            "controller_command": 0.0,
            "physics_step": 0.0,
            "state_and_contact_readback": 0.0,
            "cluster_accounting_and_sparse_range": 0.0,
            "final_communication": 0.0,
            "cluster_outcome_finalization": 0.0,
        }
        phase_started = time.perf_counter()
        if len(manifests) != self.layout.cluster_count or len(tokens) != self.layout.cluster_count:
            raise ValueError("batch request count must equal the multi-cluster layout")
        if int(self.robot.num_instances) != self.layout.total_agent_count:
            raise ValueError("vectorized robot instance count does not match the cluster layout")
        if self.agent_order != tuple(f"uav{index}" for index in range(self.layout.fleet_size)):
            raise ValueError("vectorized agent order must match the formal four-CF2X fleet")
        _controller_tracking_profile(self.controller_id, physics_dt_s=float(self.sim.cfg.dt))
        for name, value in (
            ("communication_update_hz", self.communication_update_hz),
            ("sparse_range_update_hz", self.sparse_range_update_hz),
            ("minimum_observation_dwell_s", self.minimum_observation_dwell_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        runtimes = [
            self._new_runtime(manifest, token)
            for manifest, token in zip(manifests, tokens, strict=True)
        ]
        horizon_s = runtimes[0].horizon_s
        if any(not math.isclose(row.horizon_s, horizon_s) for row in runtimes[1:]):
            raise ValueError("lockstep clusters must use the same physical decision duration")
        execution_deadline_s = runtimes[0].execution_deadline_s
        if any(
            not math.isclose(row.execution_deadline_s, execution_deadline_s) for row in runtimes[1:]
        ):
            raise ValueError("lockstep clusters must use the same execution deadline")
        dt_s = float(self.sim.cfg.dt)
        controller_mass_kg = (HOVER_THRUST_PER_ROTOR_N * float(self.robot.num_thrusters)) / 9.81
        if self.controller_id == BITCRAZE_LEE_CONTROLLER_ID:
            controller = BitcrazeLeeTracker(
                mass_kg=controller_mass_kg,
                dt_s=dt_s,
                maximum_feedback_acceleration_mps2=CF2X_MAX_FEEDBACK_ACCELERATION_MPS2,
                maximum_tilt_rad=CF2X_MAXIMUM_TILT_RAD,
            )
        elif self.controller_id == BITCRAZE_MELLINGER_CONTROLLER_ID:
            controller = BitcrazeMellingerTracker(
                mass_kg=controller_mass_kg,
                dt_s=1.0 / BITCRAZE_MELLINGER_OFFICIAL_CONTROL_RATE_HZ,
            )
        else:
            controller = None
        maximum_steps = max(1, math.floor(execution_deadline_s / dt_s))
        final_physics_timestamp_s = maximum_steps * dt_s
        communication_interval_s = 1.0 / self.communication_update_hz

        initial_world = [
            tuple(float(value) for value in row)
            for row in self.robot.data.root_pos_w.detach().cpu().tolist()
        ]
        for cluster_id, runtime in enumerate(runtimes):
            self._record_graph(
                runtime,
                0.0,
                list(initial_world[self.layout.cluster_slice(cluster_id)]),
                communication_interval_s,
            )

        positions_w = initial_world
        _, _, initial_yaw_rad = _euler_xyz_from_quaternion_wxyz(self.robot.data.root_quat_w)
        heading_references_deg = torch.rad2deg(initial_yaw_rad).detach().cpu().tolist()
        previous_linear_velocity = self.robot.data.root_lin_vel_w.detach().clone()
        wall_seconds["batch_setup"] += time.perf_counter() - phase_started
        completed_steps = maximum_steps
        for step in range(1, maximum_steps + 1):
            phase_started = time.perf_counter()
            reference_position_world: list[Point3] = []
            reference_velocity_world: list[Point3] = []
            reference_acceleration_world: list[Point3] = []
            headings: list[float] = []
            control_timestamp_s = (step - 1) * dt_s
            for cluster_id, runtime in enumerate(runtimes):
                for index, (transit, observe) in enumerate(runtime.routes):
                    destination_local = (
                        transit.path[runtime.waypoint_index[index]]
                        if runtime.transit_end_s[index] is None
                        else observe.path[0]
                    )
                    if runtime.transit_end_s[index] is None:
                        terminal_segment = (
                            runtime.waypoint_index[index] + 1 >= len(transit.path)
                        )
                        terminal_speed_mps = (
                            0.0
                            if terminal_segment
                            else runtime.route_corner_speed_mps[index]
                        )
                        reference = _minimum_time_line_reference_with_boundary_speeds(
                            runtime.segment_start_m[index],
                            tuple(destination_local),
                            max(
                                0.0,
                                control_timestamp_s - runtime.segment_start_s[index],
                            ),
                            initial_speed_mps=runtime.segment_initial_speed_mps[index],
                            terminal_speed_mps=terminal_speed_mps,
                        )
                        reference_position_world.append(
                            self.layout.to_world(cluster_id, reference.position_m)
                        )
                        reference_velocity_world.append(reference.velocity_mps)
                        reference_acceleration_world.append(reference.acceleration_mps2)
                    else:
                        reference_position_world.append(
                            self.layout.to_world(cluster_id, tuple(destination_local))
                        )
                        reference_velocity_world.append((0.0, 0.0, 0.0))
                        reference_acceleration_world.append((0.0, 0.0, 0.0))
                    flat_index = self.layout.flat_agent_index(cluster_id, index)
                    horizontal_delta_m = math.hypot(
                        destination_local[0] - runtime.transit_traces[index][-1][0],
                        destination_local[1] - runtime.transit_traces[index][-1][1],
                    )
                    target_heading_deg = (
                        _yaw_from_delta(runtime.transit_traces[index][-1], destination_local)
                        if horizontal_delta_m > 1.0e-9
                        else heading_references_deg[flat_index]
                    )
                    heading_references_deg[flat_index] = _rate_limited_yaw_reference_deg(
                        heading_references_deg[flat_index],
                        target_heading_deg,
                        dt_s,
                    )
                    headings.append(heading_references_deg[flat_index])
            thrust = _bounded_rotor_thrust(
                self.robot,
                torch.tensor(
                    reference_position_world,
                    device=self.robot.device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    reference_velocity_world,
                    device=self.robot.device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    reference_acceleration_world,
                    device=self.robot.device,
                    dtype=torch.float32,
                ),
                headings,
                controller=controller,
                dt_s=dt_s,
            )
            self.robot.set_thrust_target(thrust)
            self.robot.write_data_to_sim()
            wall_seconds["controller_command"] += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            self.sim.step(render=False)
            wall_seconds["physics_step"] += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            self.robot.update(dt_s)
            self.contact.update(dt_s, force_recompute=True)
            timestamp_s = step * dt_s
            positions_w = [
                tuple(float(value) for value in row)
                for row in self.robot.data.root_pos_w.detach().cpu().tolist()
            ]
            local_by_cluster = [
                self.layout.local_team_from_flat_world(cluster_id, positions_w)
                for cluster_id in range(self.layout.cluster_count)
            ]
            linear_velocity = self.robot.data.root_lin_vel_w
            linear_speeds = torch.linalg.norm(linear_velocity, dim=1)
            linear_accelerations = torch.linalg.norm(
                (linear_velocity - previous_linear_velocity) / dt_s, dim=1
            )
            previous_linear_velocity = linear_velocity.detach().clone()
            contact_forces = (
                torch.linalg.norm(self.contact.data.net_forces_w, dim=-1).max(dim=1).values
            )
            thrust_rows = thrust.detach().cpu().tolist()
            wall_seconds["state_and_contact_readback"] += time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            for cluster_id, runtime in enumerate(runtimes):
                world_rows = list(positions_w[self.layout.cluster_slice(cluster_id)])
                if (
                    runtime.last_communication_measurement_s is None
                    or timestamp_s - runtime.last_communication_measurement_s + 1.0e-12
                    >= communication_interval_s
                ):
                    self._record_graph(runtime, timestamp_s, world_rows, communication_interval_s)
                local_rows = local_by_cluster[cluster_id]
                for agent_index, position_local in enumerate(local_rows):
                    flat_index = self.layout.flat_agent_index(cluster_id, agent_index)
                    runtime.energy_j[agent_index] += _energy_increment_j(
                        thrust_rows[flat_index], dt_s
                    )
                    runtime.maximum_linear_speed_mps[agent_index] = max(
                        runtime.maximum_linear_speed_mps[agent_index],
                        float(linear_speeds[flat_index].item()),
                    )
                    runtime.maximum_linear_acceleration_mps2[agent_index] = max(
                        runtime.maximum_linear_acceleration_mps2[agent_index],
                        float(linear_accelerations[flat_index].item()),
                    )
                    out_of_bounds = any(
                        position_local[axis] < self.bounds_min_m[axis] - 1.0e-6
                        or position_local[axis] > self.bounds_max_m[axis] + 1.0e-6
                        for axis in range(3)
                    )
                    force_n = float(contact_forces[flat_index].item())
                    collided = force_n > CONTACT_HARD_FAIL_N
                    runtime.maximum_contact_force_n[agent_index] = max(
                        runtime.maximum_contact_force_n[agent_index], force_n
                    )
                    transit, observe = runtime.routes[agent_index]
                    destination_local = (
                        transit.path[runtime.waypoint_index[agent_index]]
                        if runtime.transit_end_s[agent_index] is None
                        else observe.path[0]
                    )
                    if collided and runtime.first_collision_step[agent_index] is None:
                        runtime.first_collision_step[agent_index] = step
                        runtime.first_collision_position[agent_index] = position_local
                        runtime.first_collision_waypoint[agent_index] = tuple(destination_local)
                    if runtime.transit_end_s[agent_index] is None:
                        runtime.transit_traces[agent_index].append(position_local)
                        runtime.transit_contact[agent_index] |= collided
                        runtime.transit_oob[agent_index] |= out_of_bounds
                        if collided or out_of_bounds:
                            runtime.failed[agent_index] = True
                            continue
                        waypoint = transit.path[runtime.waypoint_index[agent_index]]
                        error_m = math.dist(position_local, waypoint)
                        speed_mps = float(linear_speeds[flat_index].item())
                        terminal_segment = (
                            runtime.waypoint_index[agent_index] + 1
                            >= len(transit.path)
                        )
                        waypoint_requires_settle = terminal_segment
                        if _waypoint_reached(
                            error_m=error_m,
                            speed_mps=speed_mps,
                            requires_settle=waypoint_requires_settle,
                            arrival_tolerance_m=self.arrival_tolerance_m,
                        ):
                            runtime.waypoint_transitions[agent_index].append(
                                {
                                    "waypoint_index": runtime.waypoint_index[agent_index],
                                    "timestamp_s": timestamp_s,
                                    "position_m": position_local,
                                "error_m": error_m,
                                "speed_mps": speed_mps,
                                "stop_required": waypoint_requires_settle,
                                "position_tolerance_m": (
                                    WAYPOINT_SETTLE_POSITION_TOLERANCE_M
                                    if waypoint_requires_settle
                                    else self.arrival_tolerance_m
                                ),
                                }
                            )
                            if not terminal_segment:
                                runtime.waypoint_index[agent_index] += 1
                                runtime.segment_start_s[agent_index] = timestamp_s
                                runtime.segment_start_m[agent_index] = position_local
                                runtime.segment_initial_speed_mps[agent_index] = min(
                                    max(speed_mps, 0.0),
                                    runtime.route_corner_speed_mps[agent_index],
                                )
                            else:
                                runtime.transit_end_s[agent_index] = timestamp_s
                                runtime.observation_start_s[agent_index] = timestamp_s
                                runtime.observation_traces[agent_index].append(position_local)
                    elif runtime.observation_end_s[agent_index] is None:
                        runtime.observation_traces[agent_index].append(position_local)
                        runtime.observation_contact[agent_index] |= collided
                        runtime.observation_oob[agent_index] |= out_of_bounds
                        if collided or out_of_bounds:
                            runtime.failed[agent_index] = True
                            continue
                        observation_completed = (
                            _minimum_observation_dwell_completed(
                                timestamp_s=timestamp_s,
                                actual_start_s=runtime.observation_start_s[agent_index],
                                minimum_dwell_s=self.minimum_observation_dwell_s,
                            )
                            if self.event_driven_action_completion
                            else _scheduled_observation_completed(
                                timestamp_s=timestamp_s,
                                planned_end_s=observe.planned_end,
                                actual_start_s=runtime.observation_start_s[agent_index],
                                minimum_dwell_s=self.minimum_observation_dwell_s,
                                final_physics_timestamp_s=final_physics_timestamp_s,
                            )
                        )
                        if observation_completed:
                            runtime.observation_end_s[agent_index] = timestamp_s

                for agent_index, position_local in enumerate(local_rows):
                    sampling_phase = _sparse_range_sampling_phase(
                        transit_completed=runtime.transit_end_s[agent_index] is not None,
                        observation_completed=runtime.observation_end_s[agent_index] is not None,
                        failed=runtime.failed[agent_index],
                        reservation_waiting=False,
                        team_awaiting=not all(
                            runtime.failed[other]
                            or runtime.observation_end_s[other] is not None
                            for other in range(len(local_rows))
                        ),
                    )
                    if sampling_phase is None:
                        continue
                    previous_s = runtime.last_sensor_timestamp_s[agent_index]
                    if (
                        previous_s is not None
                        and timestamp_s - previous_s + 1.0e-12 < 1.0 / self.sparse_range_update_hz
                    ):
                        continue
                    transit, observe = runtime.routes[agent_index]
                    source_id = (
                        f"range-{runtime.manifest.manifest_hash[:12]}-{observe.agent_id}"
                        f"-{runtime.sensor_frames_by_agent[agent_index]:04d}"
                    )
                    position_world = positions_w[
                        self.layout.flat_agent_index(cluster_id, agent_index)
                    ]
                    valid_ray_count = 0
                    for direction_index, direction in enumerate(self.sparse_range_directions):
                        target_world_ray = tuple(
                            position_world[axis] + direction[axis] * self.sparse_range_max_m
                            for axis in range(3)
                        )
                        hit = _first_static_scene_hit(
                            self.scene_query,
                            position_world,
                            target_world_ray,
                            endpoint_margin_m=0.0,
                        )
                        hit_occupied = bool(hit is not None and hit.get("hit", False))
                        distance_m = (
                            float(hit.get("distance", self.sparse_range_max_m))
                            if hit_occupied
                            else self.sparse_range_max_m
                        )
                        if not math.isfinite(distance_m) or distance_m <= 0.02:
                            continue
                        valid_ray_count += 1
                        endpoint_local = tuple(
                            position_local[axis] + direction[axis] * distance_m for axis in range(3)
                        )
                        runtime.public_range_outcomes.append(
                            PublicRangeRayOutcome(
                                observation_id=f"{source_id}-ray{direction_index}",
                                agent_id=observe.agent_id,
                                timestamp_s=timestamp_s,
                                origin_m=position_local,
                                endpoint_m=endpoint_local,
                                hit_occupied=hit_occupied,
                            )
                        )
                    if valid_ray_count == 0:
                        continue
                    runtime.public_range_frames.append(
                        PublicRangeObservationFrameOutcome(
                            observation_frame_id=source_id,
                            agent_id=observe.agent_id,
                            timestamp_s=timestamp_s,
                            sensor_position_m=position_local,
                            ray_count=valid_ray_count,
                        )
                    )
                    runtime.source_observation_ids_by_agent[observe.agent_id].append(source_id)
                    runtime.last_sensor_timestamp_s[agent_index] = timestamp_s
                    runtime.last_sensor_source_id[agent_index] = source_id
                    runtime.sensor_frames_by_agent[agent_index] += 1
                    runtime.range_frames_by_phase[sampling_phase] += 1
            wall_seconds["cluster_accounting_and_sparse_range"] += (
                time.perf_counter() - phase_started
            )
            if self.event_driven_action_completion and all(
                all(
                    failed or observation_end is not None
                    for failed, observation_end in zip(
                        runtime.failed, runtime.observation_end_s, strict=True
                    )
                )
                for runtime in runtimes
            ):
                completed_steps = step
                break

        final_timestamp_s = completed_steps * dt_s
        phase_started = time.perf_counter()
        for cluster_id, runtime in enumerate(runtimes):
            final_world_rows = list(positions_w[self.layout.cluster_slice(cluster_id)])
            if runtime.last_communication_measurement_s != final_timestamp_s:
                self._record_graph(
                    runtime, final_timestamp_s, final_world_rows, communication_interval_s
                )
            if runtime.current_disconnect_started_s is not None:
                runtime.longest_disconnected_duration_s = max(
                    runtime.longest_disconnected_duration_s,
                    final_timestamp_s - runtime.current_disconnect_started_s,
                )
        wall_seconds["final_communication"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        results = tuple(
            self._finalize_cluster(
                cluster_id,
                runtime,
                self.layout.local_team_from_flat_world(cluster_id, positions_w),
                communication_interval_s,
                final_timestamp_s=final_timestamp_s,
            )
            for cluster_id, runtime in enumerate(runtimes)
        )
        wall_seconds["cluster_outcome_finalization"] += time.perf_counter() - phase_started
        execute_total_s = time.perf_counter() - execution_wall_started
        timing = {
            "schema_version": "isaac-cf2x-vectorized-wall-clock-v1",
            "shared_across_clusters": True,
            "cluster_count": self.layout.cluster_count,
            "fleet_size_per_cluster": self.layout.fleet_size,
            "total_agent_count": self.layout.total_agent_count,
            "physics_step_count": completed_steps,
            "physics_dt_s": dt_s,
            "physical_horizon_s": final_timestamp_s,
            "phase_seconds": wall_seconds,
            "execute_manifests_total_s": execute_total_s,
            "wall_seconds_per_physical_second": execute_total_s / final_timestamp_s,
            "cluster_decisions_per_wall_hour": (
                3600.0 * self.layout.cluster_count / execute_total_s
            ),
            "agent_decisions_per_wall_hour": (
                3600.0 * self.layout.total_agent_count / execute_total_s
            ),
            "claim_limit": (
                "Shared backend execution timing only. Process launch, Isaac application "
                "startup, stage construction, asset loading, and JSON serialization are excluded."
            ),
        }
        for result in results:
            result.engineering_diagnostics["shared_wall_clock_timing_s"] = timing
        return results

    def _finalize_cluster(
        self,
        cluster_id: int,
        runtime: _ClusterRuntime,
        final_positions_local: tuple[Point3, ...],
        communication_interval_s: float,
        *,
        final_timestamp_s: float,
    ) -> VectorizedClusterExecutionResult:
        if runtime.last_relay_graph is None:
            raise RuntimeError("vectorized cluster produced no relay telemetry")
        clearance_by_agent: dict[str, dict[str, object]] = {}
        minimum_clearances: list[float] = []
        for index, (transit, _) in enumerate(runtime.routes):
            trace = tuple(runtime.transit_traces[index] + runtime.observation_traces[index])
            distances = self.static_clearance_oracle.exact_static_distances_m(trace)
            minimum_index = min(range(len(distances)), key=distances.__getitem__)
            minimum_m = distances[minimum_index]
            violated = minimum_m + 1.0e-12 < FLIGHT_CLEARANCE_M
            minimum_clearances.append(minimum_m)
            clearance_by_agent[transit.agent_id] = {
                "minimum_static_mesh_clearance_m": minimum_m,
                "minimum_clearance_position_m": trace[minimum_index],
                "trace_pose_count": len(trace),
                "static_clearance_contract_required_m": FLIGHT_CLEARANCE_M,
                "static_clearance_contract_violation": violated,
            }

        samples: list[FragmentExecutionSample] = []
        for index, (transit, observe) in enumerate(runtime.routes):
            transit_trace = tuple(runtime.transit_traces[index])
            transit_completed = runtime.transit_end_s[index] is not None
            common = {
                "static_clearance_contract_violation": bool(
                    clearance_by_agent[transit.agent_id]["static_clearance_contract_violation"]
                ),
                "minimum_clearance_m": minimum_clearances[index],
                "communication_connected_at_every_telemetry_tick": (
                    runtime.connected_at_all_samples[index]
                ),
            }
            samples.append(
                FragmentExecutionSample(
                    planned_fragment_hash=transit.digest,
                    executed=True,
                    actual_start_s=transit.planned_start,
                    actual_end_s=(runtime.transit_end_s[index] or runtime.execution_deadline_s),
                    command_path_m=transit.path,
                    actual_path_m=transit_trace,
                    execution_trace_hash=canonical_sha256(transit_trace),
                    collision=runtime.transit_contact[index],
                    out_of_bounds=runtime.transit_oob[index],
                    energy_used_j=runtime.energy_j[index],
                    failure_reason=(
                        ""
                        if transit_completed
                        else "collision"
                        if runtime.transit_contact[index]
                        else "out_of_bounds"
                        if runtime.transit_oob[index]
                        else "transit_timeout"
                    ),
                    **common,
                )
            )
            observation_trace = tuple(runtime.observation_traces[index])
            observation_completed = runtime.observation_end_s[index] is not None
            if (
                observation_trace
                or runtime.observation_contact[index]
                or runtime.observation_oob[index]
            ):
                source_id, source_episode_id, source_agent_id = _observation_source_identity(
                    runtime.last_sensor_source_id[index],
                    episode_id=observe.episode_id,
                    agent_id=observe.agent_id,
                )
                verified = (
                    observation_completed
                    and source_id is not None
                    and not runtime.observation_contact[index]
                    and not runtime.observation_oob[index]
                )
                samples.append(
                    FragmentExecutionSample(
                        planned_fragment_hash=observe.digest,
                        executed=True,
                        actual_start_s=runtime.observation_start_s[index] or observe.planned_start,
                        actual_end_s=(
                            runtime.observation_end_s[index] or runtime.execution_deadline_s
                        ),
                        command_path_m=observe.path,
                        actual_path_m=observation_trace or (observe.path[0],),
                        execution_trace_hash=canonical_sha256(observation_trace),
                        collision=runtime.observation_contact[index],
                        out_of_bounds=runtime.observation_oob[index],
                        energy_used_j=0.0,
                        source_observation_id=source_id,
                        source_observation_episode_id=source_episode_id,
                        source_observation_agent_id=source_agent_id,
                        range_ok=verified,
                        fov_ok=verified,
                        los_ok=verified,
                        orientation_ok=verified,
                        dwell_ok=verified,
                        failure_reason=_observation_failure_reason(
                            completed=observation_completed,
                            collided=runtime.observation_contact[index],
                            out_of_bounds=runtime.observation_oob[index],
                            source_id=source_id,
                        ),
                        **common,
                    )
                )
            else:
                samples.append(
                    FragmentExecutionSample(
                        planned_fragment_hash=observe.digest,
                        executed=False,
                        actual_start_s=runtime.execution_deadline_s,
                        actual_end_s=runtime.execution_deadline_s,
                        execution_trace_hash=canonical_sha256(observation_trace),
                        failure_reason="observation_not_reached",
                    )
                )

        fusion_agent_id = self.agent_order[0]
        senders = tuple(
            agent_id
            for agent_id in self.agent_order
            if runtime.source_observation_ids_by_agent[agent_id]
        )
        for sender_id in senders:
            runtime.message_queue.publish(
                RelayMessage(
                    message_id=(f"map-segment-{runtime.manifest.manifest_hash[:12]}-{sender_id}"),
                    sender_id=sender_id,
                    source_timestamp_s=runtime.execution_deadline_s,
                    payload_digest=canonical_sha256(
                        {
                            "sender_id": sender_id,
                            "source_observation_ids": (
                                runtime.source_observation_ids_by_agent[sender_id]
                            ),
                        }
                    ),
                    time_to_live_s=self.communication_message_ttl_s,
                )
            )
        boundary_s = (
            runtime.execution_deadline_s
            + self.communication_base_latency_s
            + self.communication_per_hop_latency_s * max(1, len(self.agent_order) - 1)
        )
        runtime.message_queue.advance(timestamp_s=boundary_s, graph=runtime.last_relay_graph)
        runtime.message_queue.finalize_episode(timestamp_s=boundary_s)
        public_map_sender_ids = {
            fusion_agent_id for sender_id in senders if sender_id == fusion_agent_id
        }
        public_map_sender_ids.update(
            outcome.sender_id
            for outcome in runtime.message_queue.outcomes
            if outcome.status == "DELIVERED" and outcome.receiver_id == fusion_agent_id
        )
        expected_outcomes = len(senders) * (len(self.agent_order) - 1)
        resolved_outcomes = len(runtime.message_queue.outcomes)
        if resolved_outcomes != expected_outcomes:
            raise RuntimeError("vectorized relay message denominator mismatch")
        delivered_ages_s = [
            row.delivery.age_seconds
            for row in runtime.message_queue.outcomes
            if row.status == "DELIVERED" and row.delivery is not None
        ]
        static_trace_clearance = {
            "method": "exact_same_static_collision_mesh_at_each_physics_trace_pose_v1",
            "scope": (
                "Root-position samples at the physics integration cadence; this is distinct "
                "from the continuous planned-centreline admission certificate."
            ),
            "vehicle_self_collider_excluded": True,
            "per_agent": clearance_by_agent,
            "minimum_static_mesh_clearance_m": min(minimum_clearances),
            "static_clearance_contract_required_m": FLIGHT_CLEARANCE_M,
            "static_clearance_contract_passed": all(
                not bool(row["static_clearance_contract_violation"])
                for row in clearance_by_agent.values()
            ),
        }
        communication = {
            "model": "range_los_undirected_relay_graph_v1",
            "maximum_range_m": self.communication_max_range_m,
            "measurement_count": runtime.relay_measurement_count,
            "fully_relay_connected_count": runtime.relay_fully_connected_count,
            "fully_relay_connected_fraction": (
                runtime.relay_fully_connected_count / runtime.relay_measurement_count
            ),
            "telemetry_update_hz": self.communication_update_hz,
            "telemetry_sample_interval_s": communication_interval_s,
            "telemetry_sampling_claim": (
                "contract-rate range/LOS snapshots; no unmeasured continuous-link claim"
            ),
            "relay_telemetry_sample_count": runtime.relay_measurement_count,
            "relay_connected_telemetry_sample_count": runtime.relay_fully_connected_count,
            "relay_connected_telemetry_sample_fraction": (
                runtime.relay_fully_connected_count / runtime.relay_measurement_count
            ),
            "longest_sampled_disconnected_duration_s": (runtime.longest_disconnected_duration_s),
            "partition_event_count": runtime.partition_event_count,
            "reconnection_count": runtime.reconnection_count,
            "mean_direct_link_count": (
                runtime.relay_direct_link_count_sum / runtime.relay_measurement_count
            ),
            "maximum_component_count": runtime.relay_component_count_max,
            "maximum_relay_hops": runtime.relay_maximum_hops_max,
            "final_graph": runtime.last_relay_graph.to_dict(),
            "claim_limit": (
                "Decision-boundary aggregate public-map delivery only. No RF propagation, "
                "bandwidth, or connectivity between telemetry samples is represented."
            ),
        }
        delivery = {
            "model": "range-los-relay-decision-boundary-delta-v2",
            "aggregation": "one_delta_per_sender_per_decision",
            "fusion_agent_id": fusion_agent_id,
            "public_map_sender_ids": sorted(public_map_sender_ids),
            "public_map_delta_count": len(senders),
            "base_latency_s": self.communication_base_latency_s,
            "per_hop_latency_s": self.communication_per_hop_latency_s,
            "loss_probability": self.communication_loss_probability,
            "outcome_counts_after_close": {
                status: sum(row.status == status for row in runtime.message_queue.outcomes)
                for status in ("DELIVERED", "DROPPED", "EXPIRED")
            },
            "expected_recipient_outcomes": expected_outcomes,
            "resolved_recipient_outcomes": resolved_outcomes,
            "maximum_delivery_age_s": max(delivered_ages_s, default=0.0),
        }
        roles_by_agent = {
            transit.agent_id: str(
                dict(transit.type_signature.public_features).get("assignment_role", "explore")
            )
            for transit, _ in runtime.routes
        }
        team_trajectory_diversity = audit_translation_invariant_team_trajectories(
            {
                transit.agent_id: runtime.transit_traces[index]
                for index, (transit, _) in enumerate(runtime.routes)
            },
            roles_by_agent=roles_by_agent,
            scope="realised_physx",
        )
        diagnostics = {
            "schema_version": "isaac-cf2x-vectorized-cluster-diagnostics-v1",
            "cluster_id": cluster_id,
            "backend_id": CF2X_EXECUTION_BACKEND_ID,
            "evidence_class": CF2X_EXECUTION_EVIDENCE_CLASS,
            "token_authorization_duration_s": runtime.token.duration,
            "execution_deadline_s": runtime.execution_deadline_s,
            "execution_elapsed_physics_s": final_timestamp_s,
            "calibration_only_timeout_probe": (self.calibration_timeout_probe_s is not None),
            "contact_hard_fail_n": CONTACT_HARD_FAIL_N,
            "action_completion_mode": (
                "event_driven_all_cluster_routes_completed_plus_minimum_dwell"
                if self.event_driven_action_completion
                else "legacy_planned_fragment_boundary_lockstep"
            ),
            "controller_tracking": _controller_tracking_profile(
                self.controller_id,
                physics_dt_s=float(self.sim.cfg.dt),
            ),
            "team_trajectory_diversity": team_trajectory_diversity.to_dict(),
            "static_trace_clearance": static_trace_clearance,
            "communication": communication,
            "message_delivery": delivery,
            "sparse_range_outcomes": {
                "profile_id": "sparse-range-3d-vfov90",
                "update_hz": self.sparse_range_update_hz,
                "source_observation_frame_count": len(runtime.public_range_frames),
                "frames_by_agent": dict(
                    zip(self.agent_order, runtime.sensor_frames_by_agent, strict=True)
                ),
                "frames_by_phase": dict(runtime.range_frames_by_phase),
                "outcome_hash": canonical_sha256(
                    [row.to_dict() for row in runtime.public_range_frames]
                ),
                "ray_outcome_count": len(runtime.public_range_outcomes),
                "ray_outcome_hash": canonical_sha256(
                    [row.to_dict() for row in runtime.public_range_outcomes]
                ),
            },
            "agents": [
                {
                    "agent_id": self.agent_order[index],
                    "command_path_m": runtime.routes[index][0].path,
                    "initial_planned_position_m": runtime.transit_traces[index][0],
                    "first_simulated_position_m": (
                        runtime.transit_traces[index][1]
                        if len(runtime.transit_traces[index]) > 1
                        else None
                    ),
                    "last_transit_position_m": runtime.transit_traces[index][-1],
                    "first_collision_step": runtime.first_collision_step[index],
                    "first_collision_position_m": runtime.first_collision_position[index],
                    "first_collision_commanded_waypoint_m": (
                        runtime.first_collision_waypoint[index]
                    ),
                    "maximum_contact_force_n": runtime.maximum_contact_force_n[index],
                    "maximum_linear_speed_mps": (runtime.maximum_linear_speed_mps[index]),
                    "maximum_linear_acceleration_mps2": (
                        runtime.maximum_linear_acceleration_mps2[index]
                    ),
                    "transit_collision": runtime.transit_contact[index],
                    "transit_out_of_bounds": runtime.transit_oob[index],
                    "observation_collision": runtime.observation_contact[index],
                    "observation_out_of_bounds": runtime.observation_oob[index],
                    "minimum_static_mesh_clearance_m": minimum_clearances[index],
                    "minimum_clearance_position_m": clearance_by_agent[
                        self.agent_order[index]
                    ]["minimum_clearance_position_m"],
                    "static_clearance_contract_required_m": FLIGHT_CLEARANCE_M,
                    "static_clearance_contract_violation": bool(
                        clearance_by_agent[self.agent_order[index]][
                            "static_clearance_contract_violation"
                        ]
                    ),
                    "waypoint_transitions": runtime.waypoint_transitions[index],
                    "transit_completed": runtime.transit_end_s[index] is not None,
                    "transit_completed_at_s": runtime.transit_end_s[index],
                    "transit_attempted": True,
                    "transit_attempt_actual_end_s": runtime.transit_end_s[index]
                    or runtime.execution_deadline_s,
                    "transit_execution_deadline_s": runtime.execution_deadline_s,
                    "transit_failure_reason": (
                        None
                        if runtime.transit_end_s[index] is not None
                        else "collision"
                        if runtime.transit_contact[index]
                        else "out_of_bounds"
                        if runtime.transit_oob[index]
                        else "transit_timeout"
                    ),
                    "observation_started_at_s": runtime.observation_start_s[index],
                    "observation_completed_at_s": runtime.observation_end_s[index],
                    "realized_transit_path_length_m": _path_length_m(
                        tuple(runtime.transit_traces[index])
                    ),
                    "next_unreached_waypoint_index": runtime.waypoint_index[index],
                }
                for index in range(len(self.agent_order))
            ],
        }
        return VectorizedClusterExecutionResult(
            manifest_hash=runtime.manifest.manifest_hash,
            token_hash=runtime.token.digest,
            samples=tuple(samples),
            engineering_diagnostics=diagnostics,
            public_range_frames=tuple(runtime.public_range_frames),
            public_range_outcomes=tuple(runtime.public_range_outcomes),
            public_map_sender_ids=tuple(sorted(public_map_sender_ids)),
            final_root_positions_m=final_positions_local,
        )


__all__ = [
    "IsaacCF2XVectorizedExecutionBackend",
    "PrecomputedClusterExecutionBackend",
    "VectorizedClusterExecutionResult",
]
