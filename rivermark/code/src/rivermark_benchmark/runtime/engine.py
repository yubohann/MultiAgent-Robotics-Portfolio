"""Deterministic, closed-loop multi-UAV pilot runtime.

This module is deliberately small enough to run without Isaac Sim, but it is
not a post-processing mock.  Every policy action advances a fixed controller,
then generates RGB-D, semantic, LiDAR, radar, IMU, state, and communication
packets at that simulated timestamp.  It is a kinematic engineering pilot;
the Isaac Lab bridge is a separate, fail-closed backend.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ..schema import INFORMATION_PROFILE_MODALITIES
from .config import PilotRuntimeConfig
from .controller import FixedVelocityYawController
from .datatypes import (
    _EPS,
    CandidateEvent,
    CylinderObstacle,
    DroneState,
    EvaluationReport,
    HighLevelAction,
    PublicMission,
    PublicObservation,
    RuntimeFrame,
    SafetyEvent,
    SensorPacket,
    _HiddenTarget,
)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

class PilotSwarmRuntime:
    """A closed-loop, deterministic multi-UAV search runtime.

    The evaluator-private target array stays in this object and is only
    consulted by :meth:`evaluate` after a rollout.  Policies receive a
    :class:`PublicObservation` filtered by an information profile.
    """

    backend_id = "rivermark-kinematic-pilot-v1"

    def __init__(
        self,
        config: PilotRuntimeConfig | None = None,
        *,
        information_profile: str = "multisensor_rgbd_lidar_radar_state",
    ) -> None:
        if information_profile not in INFORMATION_PROFILE_MODALITIES:
            raise ValueError(f"unknown information profile: {information_profile}")
        self.config = config or PilotRuntimeConfig()
        self.information_profile = information_profile
        self.controller = FixedVelocityYawController(
            max_speed_mps=self.config.max_speed_mps,
            max_yaw_rate_rad_s=self.config.max_yaw_rate_rad_s,
        )
        self._obstacles = self._build_obstacles()
        self._mission = self._build_public_mission()
        self._states: dict[int, DroneState] = {}
        self._actions: dict[int, HighLevelAction] = {}
        self._previous_velocities: dict[int, np.ndarray] = {}
        self._action_history: dict[int, list[tuple[float, float, float, float]]] = {}
        self._messages: tuple[Mapping[str, Any], ...] = ()
        self._hidden_targets: tuple[_HiddenTarget, ...] = ()
        self._candidate_tracks: dict[tuple[int, int, int], tuple[int, int, np.ndarray, float]] = {}
        self._confirmed_candidates: list[CandidateEvent] = []
        self._safety_events: list[SafetyEvent] = []
        self._time_ns = 0
        self._step_index = 0
        self._last_frame: RuntimeFrame | None = None

    @property
    def mission(self) -> PublicMission:
        return self._mission

    @property
    def public_geometry(self) -> dict[str, Any]:
        return self._mission.public_geometry(self._obstacles)

    @property
    def sim_time_ns(self) -> int:
        return self._time_ns

    @property
    def done(self) -> bool:
        return self._step_index >= self.config.max_steps

    def reset(self) -> Mapping[int, PublicObservation]:
        self._time_ns = 0
        self._step_index = 0
        self._hidden_targets = self._build_hidden_targets()
        self._candidate_tracks.clear()
        self._confirmed_candidates.clear()
        self._safety_events.clear()
        self._states.clear()
        self._actions.clear()
        self._previous_velocities.clear()
        self._action_history.clear()
        height = (self.config.min_altitude_m + self.config.max_altitude_m) * 0.5
        ys = np.linspace(2.0, self.config.world_size_xy_m[1] - 2.0, self.config.agent_count)
        for agent_id, y in enumerate(ys):
            state = DroneState(
                agent_id=agent_id,
                position_m=np.array((1.75, float(y), height), dtype=np.float64),
                velocity_mps=np.zeros(3, dtype=np.float64),
                yaw_rad=0.0,
            )
            self._states[agent_id] = state
            self._actions[agent_id] = HighLevelAction.hold(source="reset")
            self._previous_velocities[agent_id] = np.zeros(3, dtype=np.float64)
            self._action_history[agent_id] = []
        self._messages = self._make_messages({})
        packets = self._capture_sensors()
        self._last_frame = self._make_frame(packets, (), ())
        return self._public_observations(packets)

    def current_frame(self) -> RuntimeFrame:
        if self._last_frame is None:
            self.reset()
        assert self._last_frame is not None
        return self._last_frame

    def step(self, actions: Mapping[int, HighLevelAction]) -> tuple[Mapping[int, PublicObservation], RuntimeFrame]:
        if not self._states:
            raise RuntimeError("reset must be called before step")
        if self.done:
            raise RuntimeError("episode has already reached its time limit")
        applied_actions = {
            agent_id: actions.get(agent_id, HighLevelAction.hold(source="implicit_hold"))
            for agent_id in self._states
        }
        for agent_id, action in applied_actions.items():
            if not isinstance(action, HighLevelAction):
                raise TypeError(f"action for agent {agent_id} is not a HighLevelAction")

        safety_events: list[SafetyEvent] = []
        velocity_targets: dict[int, np.ndarray] = {}
        proposed: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}
        for agent_id, state in self._states.items():
            velocity, yaw_rate = self.controller.track(state, applied_actions[agent_id], self.config.dt_s)
            position = state.position_m + velocity * self.config.dt_s
            yaw = _wrap_angle(state.yaw_rad + yaw_rate * self.config.dt_s)
            position, velocity, events = self._apply_static_safety(agent_id, state.position_m, position, velocity)
            safety_events.extend(events)
            proposed[agent_id] = (position, velocity, yaw, yaw_rate)
            velocity_targets[agent_id] = self.controller.desired_world_velocity(applied_actions[agent_id], state.yaw_rad)

        # Resolve drone-drone conflicts symmetrically before publishing state.
        agent_ids = sorted(proposed)
        for left_index, left_id in enumerate(agent_ids):
            for right_id in agent_ids[left_index + 1 :]:
                left_pos, left_vel, left_yaw, left_rate = proposed[left_id]
                right_pos, right_vel, right_yaw, right_rate = proposed[right_id]
                if float(np.linalg.norm(left_pos - right_pos)) < self.config.drone_radius_m * 2.2:
                    proposed[left_id] = (
                        self._states[left_id].position_m.copy(),
                        np.zeros(3, dtype=np.float64),
                        left_yaw,
                        0.0,
                    )
                    proposed[right_id] = (
                        self._states[right_id].position_m.copy(),
                        np.zeros(3, dtype=np.float64),
                        right_yaw,
                        0.0,
                    )
                    stamp = self._time_ns + int(self.config.dt_s * 1_000_000_000)
                    safety_events.extend(
                        (
                            SafetyEvent(left_id, stamp, "inter_uav_separation_guard"),
                            SafetyEvent(right_id, stamp, "inter_uav_separation_guard"),
                        )
                    )

        for agent_id, (position, velocity, yaw, yaw_rate) in proposed.items():
            self._previous_velocities[agent_id] = self._states[agent_id].velocity_mps.copy()
            self._states[agent_id] = DroneState(agent_id, position, velocity, yaw, yaw_rate)
            action = applied_actions[agent_id]
            self._actions[agent_id] = action
            self._action_history[agent_id].append(
                (float(action.velocity_xyz[0]), float(action.velocity_xyz[1]), float(action.velocity_xyz[2]), float(action.yaw_rate_rad_s))
            )
            self._action_history[agent_id] = self._action_history[agent_id][-8:]

        self._time_ns += int(round(self.config.dt_s * 1_000_000_000))
        self._step_index += 1
        packets = self._capture_sensors()
        sensor_candidates = [
            candidate
            for agent_id, packet in packets.items()
            for candidate in self._rgbd_candidates(self._states[agent_id], packet)
        ]
        confirmed = self._confirm_candidates(sensor_candidates)
        self._messages = self._make_messages({event.agent_id: event for event in confirmed})
        self._safety_events.extend(safety_events)
        frame = self._make_frame(packets, tuple(confirmed), tuple(safety_events), velocity_targets)
        self._last_frame = frame
        return self._public_observations(packets), frame

    def evaluate(self) -> EvaluationReport:
        """Score sensor-derived candidates against private truth after rollout."""

        matched_targets: set[int] = set()
        true_confirmations: list[CandidateEvent] = []
        false_count = 0
        for candidate in sorted(self._confirmed_candidates, key=lambda item: item.sensor_time_ns):
            estimate = np.asarray(candidate.estimated_xyz_m)
            eligible = [
                (index, float(np.linalg.norm(estimate - np.asarray(target.position_m))))
                for index, target in enumerate(self._hidden_targets)
                if index not in matched_targets
            ]
            if eligible and min(eligible, key=lambda item: item[1])[1] <= self.config.candidate_match_radius_m:
                index, _ = min(eligible, key=lambda item: item[1])
                matched_targets.add(index)
                true_confirmations.append(candidate)
            else:
                false_count += 1
        target_count = len(self._hidden_targets)
        confirmed_count = len(true_confirmations)
        precision = confirmed_count / len(self._confirmed_candidates) if self._confirmed_candidates else 1.0
        budget_ns = max(1, self.config.max_steps * int(round(self.config.dt_s * 1_000_000_000)))
        area = 0.0
        recalled = 0
        previous_time = 0
        for candidate in true_confirmations:
            elapsed = min(budget_ns, candidate.sensor_time_ns)
            area += (elapsed - previous_time) * recalled
            recalled += 1
            previous_time = elapsed
        area += (budget_ns - previous_time) * recalled
        normalized_auc = area / (budget_ns * target_count) if target_count else 0.0
        first_latency = (
            true_confirmations[0].sensor_time_ns / 1_000_000_000 if true_confirmations else None
        )
        truth_digest = _sha256_json([target.position_m for target in self._hidden_targets])
        return EvaluationReport(
            confirmed_count=confirmed_count,
            target_count=target_count,
            confirmation_precision=precision,
            false_confirmation_count=false_count,
            normalized_confirmed_auc=normalized_auc,
            first_confirmation_latency_s=first_latency,
            collision_count=sum(1 for event in self._safety_events if event.kind == "inter_uav_separation_guard") // 2,
            evaluator_truth_sha256=truth_digest,
        )

    def public_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_id,
            "action_abi": {
                "fields": ["vx", "vy", "dz", "yaw_rate"],
                "frames": ["world", "body"],
                "modes": ["transit", "dwell", "hold", "return"],
            },
            "sensor_modalities": [
                "rgb",
                "distance_to_image_plane",
                "semantic_segmentation",
                "lidar",
                "radar",
                "imu",
                "proprioception",
                "public_team_messages",
            ],
            "radar_kind": "deterministic_kinematic_fmcw_proxy",
            "formal_release_eligible": False,
            "limitations": [
                "not Isaac Sim",
                "not hardware",
                "radar is a kinematic proxy rather than a calibrated sensor",
            ],
        }

    def _build_obstacles(self) -> tuple[CylinderObstacle, ...]:
        return (
            CylinderObstacle("warehouse_a", (10.0, 6.0), 2.25, 4.2),
            CylinderObstacle("warehouse_b", (18.5, 16.5), 2.7, 4.5),
            CylinderObstacle("tower_c", (23.5, 7.5), 1.35, 5.0),
            CylinderObstacle("tree_cluster_d", (13.0, 19.5), 1.7, 3.2),
        )

    def _build_public_mission(self) -> PublicMission:
        width, height = self.config.world_size_xy_m
        regions = (
            {"region_id": "southwest", "bounds_xy_m": [0.0, 0.0, width / 2.0, height / 2.0]},
            {"region_id": "southeast", "bounds_xy_m": [width / 2.0, 0.0, width, height / 2.0]},
            {"region_id": "northwest", "bounds_xy_m": [0.0, height / 2.0, width / 2.0, height]},
            {"region_id": "northeast", "bounds_xy_m": [width / 2.0, height / 2.0, width, height]},
        )
        return PublicMission(
            bounds_xy_m=self.config.world_size_xy_m,
            regions=regions,
            instruction="Sweep the assigned public sector, keep separation, and report only sensor-supported candidate markers.",
            time_budget_s=self.config.dt_s * self.config.max_steps,
            target_count_disclosed=8,
        )

    def _build_hidden_targets(self) -> tuple[_HiddenTarget, ...]:
        # The seed never crosses the policy boundary.  Fixed candidate sites
        # make paired method comparisons deterministic while still private.
        sites = np.asarray(
            (
                (5.5, 4.0), (8.0, 14.5), (14.0, 3.3), (16.5, 11.0),
                (21.5, 4.2), (26.5, 12.0), (22.0, 20.0), (29.0, 18.0),
            ),
            dtype=np.float64,
        )
        rng = np.random.default_rng(self.config.seed)
        targets: list[_HiddenTarget] = []
        for x, y in sites:
            jitter = rng.uniform(-0.45, 0.45, size=2)
            targets.append(_HiddenTarget((float(x + jitter[0]), float(y + jitter[1]), 0.35)))
        return tuple(targets)

    def _apply_static_safety(
        self,
        agent_id: int,
        old_position: np.ndarray,
        proposed_position: np.ndarray,
        proposed_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[SafetyEvent]]:
        events: list[SafetyEvent] = []
        width, height = self.config.world_size_xy_m
        position = proposed_position.copy()
        velocity = proposed_velocity.copy()
        stamp = self._time_ns + int(round(self.config.dt_s * 1_000_000_000))
        bounded = np.array(
            (
                np.clip(position[0], self.config.drone_radius_m, width - self.config.drone_radius_m),
                np.clip(position[1], self.config.drone_radius_m, height - self.config.drone_radius_m),
                np.clip(position[2], self.config.min_altitude_m, self.config.max_altitude_m),
            )
        )
        if not np.allclose(position, bounded):
            position = bounded
            velocity *= 0.0
            events.append(SafetyEvent(agent_id, stamp, "world_or_altitude_guard"))
        if any(obstacle.contains(position, self.config.drone_radius_m) for obstacle in self._obstacles):
            position = old_position.copy()
            velocity *= 0.0
            events.append(SafetyEvent(agent_id, stamp, "obstacle_guard"))
        return position, velocity, events

    def _capture_sensors(self) -> dict[int, SensorPacket]:
        return {agent_id: self._sensor_packet(state) for agent_id, state in self._states.items()}

    def _sensor_packet(self, state: DroneState) -> SensorPacket:
        rgb, depth, semantic = self._render_camera(state)
        lidar = self._render_lidar(state)
        radar = self._render_radar(state)
        previous = self._previous_velocities[state.agent_id]
        acceleration_world = (state.velocity_mps - previous) / self.config.dt_s
        c, s = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
        acceleration_body = np.array((c * acceleration_world[0] + s * acceleration_world[1], -s * acceleration_world[0] + c * acceleration_world[1], acceleration_world[2]))
        imu = np.concatenate((acceleration_body, np.array((0.0, 0.0, state.yaw_rate_rad_s))))
        extrinsics = np.array((0.12, 0.0, 0.04, 1.0, 0.0, 0.0, 0.0), dtype=np.float32)
        return SensorPacket(
            agent_id=state.agent_id,
            sensor_time_ns=self._time_ns,
            rgb=rgb,
            distance_to_image_plane_m=depth,
            semantic_segmentation=semantic,
            lidar_ranges_m=lidar,
            radar_detections=radar,
            imu=imu.astype(np.float32),
            camera_extrinsics_body=extrinsics,
        )

    def _camera_basis(self, state: DroneState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c_yaw, s_yaw = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
        c_pitch, s_pitch = math.cos(self.config.camera_pitch_rad), math.sin(self.config.camera_pitch_rad)
        forward = np.array((c_yaw * c_pitch, s_yaw * c_pitch, s_pitch), dtype=np.float64)
        right = np.array((-s_yaw, c_yaw, 0.0), dtype=np.float64)
        up = np.cross(forward, right)
        return forward, right, up / max(_EPS, float(np.linalg.norm(up)))

    def _project(self, state: DroneState, point: np.ndarray) -> tuple[float, float, float] | None:
        forward, right, up = self._camera_basis(state)
        relative = point - state.position_m
        forward_depth = float(np.dot(relative, forward))
        if forward_depth <= 0.15:
            return None
        x = float(np.dot(relative, right) / forward_depth)
        y = float(np.dot(relative, up) / forward_depth)
        tan_h = math.tan(math.radians(self.config.camera_hfov_deg) / 2.0)
        tan_v = math.tan(math.radians(self.config.camera_vfov_deg) / 2.0)
        u = self.config.camera_width * (0.5 + 0.5 * x / tan_h)
        v = self.config.camera_height * (0.5 - 0.5 * y / tan_v)
        if not (-8.0 <= u < self.config.camera_width + 8.0 and -8.0 <= v < self.config.camera_height + 8.0):
            return None
        return u, v, forward_depth

    def _render_camera(self, state: DroneState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = self.config.camera_height, self.config.camera_width
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[: height // 2] = (112, 169, 202)
        rgb[height // 2 :] = (74, 117, 70)
        depth = np.full((height, width), self.config.lidar_max_range_m, dtype=np.float32)
        semantic = np.zeros((height, width), dtype=np.uint8)
        entities: list[tuple[np.ndarray, float, tuple[int, int, int], int]] = []
        for obstacle in self._obstacles:
            entities.append(
                (
                    np.array((obstacle.center_xy_m[0], obstacle.center_xy_m[1], min(obstacle.height_m, state.position_m[2]))),
                    obstacle.radius_m,
                    (86, 86, 92),
                    1,
                )
            )
        for peer_id, peer in self._states.items():
            if peer_id != state.agent_id:
                entities.append((peer.position_m, self.config.drone_radius_m * 1.4, (52, 114, 220), 3))
        for target in self._hidden_targets:
            target_position = np.asarray(target.position_m)
            if self._line_of_sight(state.position_m, target_position):
                entities.append((target_position, 0.32, (238, 44, 42), 2))
        for point, radius_m, color, label in entities:
            projection = self._project(state, point)
            if projection is None:
                continue
            u, v, forward_depth = projection
            radius_px = max(2, int(round(radius_m * width * 0.95 / max(forward_depth, 0.25))))
            x0, x1 = max(0, int(u) - radius_px), min(width, int(u) + radius_px + 1)
            y0, y1 = max(0, int(v) - radius_px), min(height, int(v) + radius_px + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px**2
            local_depth = depth[y0:y1, x0:x1]
            closer = mask & (forward_depth < local_depth)
            if not np.any(closer):
                continue
            local_depth[closer] = forward_depth
            local_rgb = rgb[y0:y1, x0:x1]
            local_semantic = semantic[y0:y1, x0:x1]
            local_rgb[closer] = color
            local_semantic[closer] = label
        return rgb, depth, semantic

    def _line_of_sight(self, start: np.ndarray, end: np.ndarray) -> bool:
        delta = end - start
        distance = float(np.linalg.norm(delta[:2]))
        if distance <= _EPS:
            return True
        steps = max(2, int(math.ceil(distance / 0.25)))
        for fraction in np.linspace(0.0, 1.0, steps, endpoint=False)[1:]:
            point = start + delta * fraction
            if any(obstacle.contains(point, 0.05) for obstacle in self._obstacles):
                return False
        return True

    def _render_lidar(self, state: DroneState) -> np.ndarray:
        angles = np.linspace(-math.pi, math.pi, self.config.lidar_beams, endpoint=False)
        return np.asarray(
            [self._ray_distance(state, state.yaw_rad + float(angle)) for angle in angles],
            dtype=np.float32,
        )

    def _ray_distance(self, state: DroneState, angle: float) -> float:
        origin = state.position_m[:2]
        direction = np.array((math.cos(angle), math.sin(angle)), dtype=np.float64)
        distances: list[float] = [self.config.lidar_max_range_m]
        width, height = self.config.world_size_xy_m
        for axis, bound in ((0, 0.0), (0, width), (1, 0.0), (1, height)):
            if abs(direction[axis]) <= _EPS:
                continue
            distance = (bound - origin[axis]) / direction[axis]
            if distance > 0.0:
                other = origin[1 - axis] + distance * direction[1 - axis]
                limit = height if axis == 0 else width
                if 0.0 <= other <= limit:
                    distances.append(float(distance))
        for obstacle in self._obstacles:
            center = np.asarray(obstacle.center_xy_m)
            offset = origin - center
            b = 2.0 * float(np.dot(offset, direction))
            c = float(np.dot(offset, offset)) - (obstacle.radius_m + self.config.drone_radius_m) ** 2
            discriminant = b * b - 4.0 * c
            if discriminant >= 0.0:
                root = math.sqrt(discriminant)
                for value in ((-b - root) / 2.0, (-b + root) / 2.0):
                    if value > 0.0:
                        distances.append(value)
        for peer_id, peer in self._states.items():
            if peer_id == state.agent_id:
                continue
            center = peer.position_m[:2]
            offset = origin - center
            b = 2.0 * float(np.dot(offset, direction))
            c = float(np.dot(offset, offset)) - (self.config.drone_radius_m * 2.0) ** 2
            discriminant = b * b - 4.0 * c
            if discriminant >= 0.0:
                value = (-b - math.sqrt(discriminant)) / 2.0
                if value > 0.0:
                    distances.append(value)
        return float(np.clip(min(distances), 0.0, self.config.lidar_max_range_m))

    def _render_radar(self, state: DroneState) -> np.ndarray:
        rows: list[tuple[float, float, float, float]] = []
        forward, right, _ = self._camera_basis(state)
        for target in self._hidden_targets:
            relative = np.asarray(target.position_m) - state.position_m
            range_m = float(np.linalg.norm(relative))
            if range_m > self.config.radar_max_range_m or not self._line_of_sight(state.position_m, np.asarray(target.position_m)):
                continue
            bearing = math.atan2(float(np.dot(relative, right)), float(np.dot(relative, forward)))
            if abs(bearing) > math.radians(68.0):
                continue
            radial_velocity = -float(np.dot(state.velocity_mps, relative / max(range_m, _EPS)))
            rows.append((range_m, bearing, radial_velocity, 0.9))
        # Static obstacle returns are deliberately indistinguishable from a
        # target except for RCS and kinematics; no hidden identity leaks.
        for obstacle in self._obstacles:
            point = np.array((obstacle.center_xy_m[0], obstacle.center_xy_m[1], min(obstacle.height_m, state.position_m[2])))
            relative = point - state.position_m
            range_m = float(np.linalg.norm(relative))
            if range_m <= self.config.radar_max_range_m:
                bearing = math.atan2(float(np.dot(relative, right)), float(np.dot(relative, forward)))
                if abs(bearing) <= math.radians(68.0):
                    rows.append((range_m, bearing, -float(np.dot(state.velocity_mps, relative / max(range_m, _EPS))), 0.35))
        rows.sort(key=lambda row: row[0])
        return np.asarray(rows, dtype=np.float32).reshape((-1, 4)) if rows else np.empty((0, 4), dtype=np.float32)

    def _rgbd_candidates(self, state: DroneState, packet: SensorPacket) -> list[CandidateEvent]:
        red = packet.rgb[:, :, 0].astype(np.int16)
        green = packet.rgb[:, :, 1].astype(np.int16)
        blue = packet.rgb[:, :, 2].astype(np.int16)
        mask = (red > 190) & (green < 105) & (blue < 105) & (packet.distance_to_image_plane_m < self.config.lidar_max_range_m)
        components = _connected_components(mask)
        forward, right, up = self._camera_basis(state)
        tan_h = math.tan(math.radians(self.config.camera_hfov_deg) / 2.0)
        tan_v = math.tan(math.radians(self.config.camera_vfov_deg) / 2.0)
        candidates: list[CandidateEvent] = []
        for component in components:
            if len(component) < 4:
                continue
            rows = np.asarray([item[0] for item in component], dtype=np.intp)
            cols = np.asarray([item[1] for item in component], dtype=np.intp)
            depth = float(np.median(packet.distance_to_image_plane_m[rows, cols]))
            if not math.isfinite(depth) or depth <= 0.1:
                continue
            u, v = float(np.mean(cols)), float(np.mean(rows))
            horizontal = ((u / self.config.camera_width) * 2.0 - 1.0) * tan_h
            vertical = (1.0 - (v / self.config.camera_height) * 2.0) * tan_v
            estimate = state.position_m + depth * (forward + horizontal * right + vertical * up)
            confidence = float(min(0.99, 0.55 + len(component) / 36.0))
            candidates.append(
                CandidateEvent(
                    agent_id=state.agent_id,
                    sensor_time_ns=packet.sensor_time_ns,
                    estimated_xyz_m=tuple(float(value) for value in estimate),
                    confidence=confidence,
                )
            )
        return candidates

    def _confirm_candidates(self, candidates: Iterable[CandidateEvent]) -> list[CandidateEvent]:
        confirmed: list[CandidateEvent] = []
        maximum_gap_ns = int(round(self.config.dt_s * 1_000_000_000 * 2.5))
        for candidate in candidates:
            point = np.asarray(candidate.estimated_xyz_m)
            key = tuple(np.rint(point / 1.2).astype(int))
            prior = self._candidate_tracks.get(key)
            if prior is None or candidate.sensor_time_ns - prior[1] > maximum_gap_ns:
                count, previous_time, mean, confidence = 1, candidate.sensor_time_ns, point, candidate.confidence
            else:
                count = prior[0] + 1
                previous_time = candidate.sensor_time_ns
                mean = (prior[2] * prior[0] + point) / count
                confidence = max(prior[3], candidate.confidence)
            self._candidate_tracks[key] = (count, previous_time, mean, confidence)
            if count == self.config.candidate_min_frames:
                event = CandidateEvent(
                    agent_id=candidate.agent_id,
                    sensor_time_ns=candidate.sensor_time_ns,
                    estimated_xyz_m=tuple(float(value) for value in mean),
                    confidence=float(confidence),
                )
                self._confirmed_candidates.append(event)
                confirmed.append(event)
        return confirmed

    def _make_messages(self, candidates: Mapping[int, CandidateEvent]) -> tuple[Mapping[str, Any], ...]:
        messages: list[Mapping[str, Any]] = []
        for agent_id, state in sorted(self._states.items()):
            message: dict[str, Any] = {
                "agent_id": agent_id,
                "position_m": [round(float(value), 3) for value in state.position_m],
                "velocity_mps": [round(float(value), 3) for value in state.velocity_mps],
                "cell_id": self._cell_id(state.position_m),
                "sim_time_ns": self._time_ns,
            }
            if agent_id in candidates:
                message["sensor_candidate_xyz_m"] = [round(value, 3) for value in candidates[agent_id].estimated_xyz_m]
                message["sensor_candidate_confidence"] = round(candidates[agent_id].confidence, 3)
            messages.append(message)
        return tuple(messages)

    def _cell_id(self, position: np.ndarray) -> str:
        return f"{min(7, int(position[0] / self.config.world_size_xy_m[0] * 8))}:{min(5, int(position[1] / self.config.world_size_xy_m[1] * 6))}"

    def _make_frame(
        self,
        packets: Mapping[int, SensorPacket],
        candidate_events: tuple[CandidateEvent, ...],
        safety_events: tuple[SafetyEvent, ...],
        velocity_targets: Mapping[int, np.ndarray] | None = None,
    ) -> RuntimeFrame:
        return RuntimeFrame(
            sim_time_ns=self._time_ns,
            step_index=self._step_index,
            states={agent_id: state.copy() for agent_id, state in self._states.items()},
            actions=dict(self._actions),
            low_level_velocity_targets_mps={
                agent_id: value.copy()
                for agent_id, value in (velocity_targets or {agent_id: np.zeros(3) for agent_id in self._states}).items()
            },
            sensor_packets=dict(packets),
            candidate_events=candidate_events,
            safety_events=safety_events,
        )

    def _public_observations(self, packets: Mapping[int, SensorPacket]) -> Mapping[int, PublicObservation]:
        modalities = INFORMATION_PROFILE_MODALITIES[self.information_profile]
        task_state = {
            "time_remaining_s": max(0.0, self._mission.time_budget_s - self._time_ns / 1_000_000_000),
            "time_budget_s": self._mission.time_budget_s,
            "target_count_disclosed": self._mission.target_count_disclosed,
            "step_index": self._step_index,
        }
        observations: dict[int, PublicObservation] = {}
        for agent_id, state in self._states.items():
            packet = packets[agent_id]
            proprioception = np.concatenate((state.position_m, state.velocity_mps, np.array((state.yaw_rad, state.yaw_rate_rad_s)))).astype(np.float32)
            observations[agent_id] = PublicObservation(
                agent_id=agent_id,
                sim_time_ns=self._time_ns,
                information_profile=self.information_profile,
                proprioception=proprioception,
                public_task_state=dict(task_state),
                public_team_messages=tuple(dict(message) for message in self._messages if message["agent_id"] != agent_id),
                high_level_action_history=tuple(self._action_history[agent_id]),
                public_geometry=self.public_geometry if "public_geometry" in modalities else None,
                rgb=packet.rgb.copy() if "rgb" in modalities else None,
                distance_to_image_plane_m=packet.distance_to_image_plane_m.copy() if "distance_to_image_plane" in modalities else None,
                lidar_ranges_m=packet.lidar_ranges_m.copy() if "lidar" in modalities else None,
                radar_detections=packet.radar_detections.copy() if "radar" in modalities else None,
                imu=packet.imu.copy() if "imu" in modalities else None,
                language=self._mission.instruction if "language" in modalities else None,
            )
        return observations

def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 8-connected components without a vision dependency."""

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for row, col in zip(*np.nonzero(mask)):
        if visited[row, col]:
            continue
        visited[row, col] = True
        stack = [(int(row), int(col))]
        component: list[tuple[int, int]] = []
        while stack:
            current_row, current_col = stack.pop()
            component.append((current_row, current_col))
            for row_delta in (-1, 0, 1):
                for col_delta in (-1, 0, 1):
                    if row_delta == 0 and col_delta == 0:
                        continue
                    next_row, next_col = current_row + row_delta, current_col + col_delta
                    if 0 <= next_row < height and 0 <= next_col < width and mask[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
        components.append(component)
    return components

def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi
