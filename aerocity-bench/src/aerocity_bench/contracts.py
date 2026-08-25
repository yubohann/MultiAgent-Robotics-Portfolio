"""Versioned public packets shared by runtimes, adapters, and evaluators.

The packet layer deliberately contains no target coordinates, support-site IDs,
split labels, or evaluator witnesses.  Private truth is represented only in the
evaluator module and is never accepted from a method process.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .canonical import content_hash

ACTION_KINDS = frozenset({"HOVER", "WAYPOINT", "VELOCITY", "OBSERVE", "RETURN"})
TERMINAL_REASONS = frozenset(
    {
        "episode_timeout",
        "collision",
        "energy_exhausted",
        "deadline_failure",
        "out_of_bounds_failure",
        "method_failure",
        "completed",
    }
)


def _vec3(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a three-vector")
    return float(value[0]), float(value[1]), float(value[2])


@dataclass(frozen=True)
class Pose3D:
    position: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vec3(self.position, "position"))
        for value, name in (
            (self.yaw_deg, "yaw_deg"),
            (self.pitch_deg, "pitch_deg"),
            (self.roll_deg, "roll_deg"),
        ):
            number = float(value)
            if not -360.0 <= number <= 360.0:
                raise ValueError(f"{name} lies outside [-360, 360]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "yaw_deg": float(self.yaw_deg),
            "pitch_deg": float(self.pitch_deg),
            "roll_deg": float(self.roll_deg),
        }

    @classmethod
    def from_dict(cls, node: dict[str, Any]) -> Pose3D:
        return cls(
            position=_vec3(node["position"], "position"),
            yaw_deg=float(node["yaw_deg"]),
            pitch_deg=float(node.get("pitch_deg", 0.0)),
            roll_deg=float(node.get("roll_deg", 0.0)),
        )


@dataclass(frozen=True)
class MessagePacket:
    message_id: str
    source_drone_id: str
    destination_drone_ids: tuple[str, ...]
    created_at_s: float
    expires_at_s: float
    payload: bytes

    def __post_init__(self) -> None:
        if not self.message_id or not self.source_drone_id:
            raise ValueError("message identifiers cannot be empty")
        if not self.destination_drone_ids:
            raise ValueError("a message needs at least one destination")
        if float(self.expires_at_s) <= float(self.created_at_s):
            raise ValueError("message expiration must follow creation")
        if not isinstance(self.payload, bytes):
            raise ValueError("message payload must be bytes")

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "source_drone_id": self.source_drone_id,
            "destination_drone_ids": list(self.destination_drone_ids),
            "created_at_s": self.created_at_s,
            "expires_at_s": self.expires_at_s,
            "payload_hex": self.payload.hex(),
        }


@dataclass(frozen=True)
class ActionPacket:
    episode_id: str
    drone_id: str
    sequence: int
    issued_at_s: float
    kind: Literal["HOVER", "WAYPOINT", "VELOCITY", "OBSERVE", "RETURN"]
    waypoint: Pose3D | None = None
    velocity_body_mps: tuple[float, float, float] | None = None
    yaw_rate_deg_s: float = 0.0
    # A bounded inspection gimbal is independent of the CF2X body attitude.
    # Its bounds and rate are enforced by the active execution contract.
    sensor_pitch_deg: float | None = None
    source_observation_id: str | None = None
    messages: tuple[MessagePacket, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.drone_id:
            raise ValueError("action episode_id and drone_id cannot be empty")
        if int(self.sequence) < 0 or float(self.issued_at_s) < 0:
            raise ValueError("action sequence and time must be non-negative")
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind: {self.kind}")
        if self.kind == "WAYPOINT" and self.waypoint is None:
            raise ValueError("WAYPOINT requires a waypoint pose")
        if self.kind == "VELOCITY" and self.velocity_body_mps is None:
            raise ValueError("VELOCITY requires a velocity vector")
        if self.velocity_body_mps is not None:
            object.__setattr__(
                self,
                "velocity_body_mps",
                _vec3(self.velocity_body_mps, "velocity_body_mps"),
            )
        if self.sensor_pitch_deg is not None:
            sensor_pitch = float(self.sensor_pitch_deg)
            if not math.isfinite(sensor_pitch) or not -360.0 <= sensor_pitch <= 360.0:
                raise ValueError("sensor_pitch_deg lies outside finite [-360, 360]")
            object.__setattr__(self, "sensor_pitch_deg", sensor_pitch)
        if self.kind == "OBSERVE" and not self.source_observation_id:
            raise ValueError("OBSERVE must bind a source_observation_id")
        if self.kind != "OBSERVE" and self.source_observation_id is not None:
            raise ValueError("only OBSERVE may bind a source observation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "org.aerocity.bench.action-packet.v1",
            "episode_id": self.episode_id,
            "drone_id": self.drone_id,
            "sequence": self.sequence,
            "issued_at_s": self.issued_at_s,
            "kind": self.kind,
            "waypoint": self.waypoint.to_dict() if self.waypoint else None,
            "velocity_body_mps": (list(self.velocity_body_mps) if self.velocity_body_mps else None),
            "yaw_rate_deg_s": self.yaw_rate_deg_s,
            "sensor_pitch_deg": self.sensor_pitch_deg,
            "source_observation_id": self.source_observation_id,
            "messages": [message.to_dict() for message in self.messages],
        }


@dataclass(frozen=True)
class ObservationPacket:
    episode_id: str
    observation_id: str
    drone_id: str
    sequence: int
    timestamp_s: float
    pose: Pose3D
    linear_velocity_world_mps: tuple[float, float, float]
    angular_speed_deg_s: float
    energy_remaining_j: float
    local_occupancy: tuple[tuple[int, int, int], ...] = field(default_factory=tuple)
    local_occupancy_origin_world_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_occupancy_resolution_m: float = 2.0
    local_occupancy_radius_m: float = 14.0
    teammate_states: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    received_messages: tuple[MessagePacket, ...] = field(default_factory=tuple)
    health: Literal["nominal", "terminal"] = "nominal"
    # ``pose`` is always the measured vehicle body pose. A bounded gimbal,
    # when declared by the task contract, reports its measured pitch here.
    sensor_pitch_deg: float | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or not self.observation_id or not self.drone_id:
            raise ValueError("observation identifiers cannot be empty")
        if int(self.sequence) < 0 or float(self.timestamp_s) < 0:
            raise ValueError("observation sequence and time must be non-negative")
        object.__setattr__(
            self,
            "linear_velocity_world_mps",
            _vec3(self.linear_velocity_world_mps, "linear_velocity_world_mps"),
        )
        object.__setattr__(
            self,
            "local_occupancy_origin_world_m",
            _vec3(self.local_occupancy_origin_world_m, "local_occupancy_origin_world_m"),
        )
        if self.local_occupancy_resolution_m <= 0.0 or self.local_occupancy_radius_m <= 0.0:
            raise ValueError("local occupancy resolution and radius must be positive")
        for cell in self.local_occupancy:
            if len(cell) != 3 or any(not isinstance(value, int) for value in cell):
                raise ValueError("local occupancy cells must be integer three-vectors")
        if float(self.angular_speed_deg_s) < 0 or float(self.energy_remaining_j) < 0:
            raise ValueError("speed and energy cannot be negative")
        sensor_pitch = self.pose.pitch_deg if self.sensor_pitch_deg is None else float(
            self.sensor_pitch_deg
        )
        if not math.isfinite(sensor_pitch) or not -360.0 <= sensor_pitch <= 360.0:
            raise ValueError("sensor_pitch_deg lies outside finite [-360, 360]")
        object.__setattr__(self, "sensor_pitch_deg", sensor_pitch)
        if self.health not in {"nominal", "terminal"}:
            raise ValueError("unknown public health state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "org.aerocity.bench.observation-packet.v2",
            "episode_id": self.episode_id,
            "observation_id": self.observation_id,
            "drone_id": self.drone_id,
            "sequence": self.sequence,
            "timestamp_s": self.timestamp_s,
            "pose": self.pose.to_dict(),
            "linear_velocity_world_mps": list(self.linear_velocity_world_mps),
            "angular_speed_deg_s": self.angular_speed_deg_s,
            "energy_remaining_j": self.energy_remaining_j,
            "local_occupancy": [list(cell) for cell in self.local_occupancy],
            "local_occupancy_origin_world_m": list(self.local_occupancy_origin_world_m),
            "local_occupancy_resolution_m": self.local_occupancy_resolution_m,
            "local_occupancy_radius_m": self.local_occupancy_radius_m,
            "teammate_states": list(self.teammate_states),
            "received_messages": [message.to_dict() for message in self.received_messages],
            "health": self.health,
            "sensor_pitch_deg": self.sensor_pitch_deg,
        }


@dataclass(frozen=True)
class ObservationReceipt:
    observation_id: str
    drone_id: str
    timestamp_s: float
    accepted: bool
    reason: str
    receipt_hash: str

    @classmethod
    def create(
        cls,
        observation_id: str,
        drone_id: str,
        timestamp_s: float,
        accepted: bool,
        reason: str,
    ) -> ObservationReceipt:
        payload = {
            "observation_id": observation_id,
            "drone_id": drone_id,
            "timestamp_s": timestamp_s,
            "accepted": accepted,
            "reason": reason,
        }
        return cls(**payload, receipt_hash=content_hash(payload))


@dataclass(frozen=True)
class ConfirmationReceipt:
    confirmation_id: str
    anonymous_target_handle: str
    drone_id: str
    confirmed_at_s: float
    source_observation_id: str
    receipt_token: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "org.aerocity.bench.confirmation-receipt.v1",
            **asdict(self),
        }


@dataclass(frozen=True)
class ExecutionReceipt:
    episode_id: str
    drone_id: str
    action_sequence: int
    task_time_start_s: float
    task_time_end_s: float
    planning_latency_s: float
    action_requested: str
    action_executed: str
    status: str
    distance_m: float
    energy_used_j: float
    minimum_clearance_m: float | None
    collision: bool
    out_of_bounds: bool
    safety_intervention: bool
    deadline_miss: bool
    execution_level: Literal["L0", "L1", "L2"]
    action_packet_hash: str
    source_observation_id: str
    source_observation_hash: str
    state_before_hash: str
    state_after_hash: str
    previous_receipt_hash: str | None
    confirmation_ids: tuple[str, ...] = field(default_factory=tuple)
    planner_invoked: bool = True

    def __post_init__(self) -> None:
        if float(self.task_time_end_s) < float(self.task_time_start_s):
            raise ValueError("execution receipt time runs backwards")
        if self.execution_level not in {"L0", "L1", "L2"}:
            raise ValueError("unknown execution level")
        numeric = (
            self.task_time_start_s,
            self.task_time_end_s,
            self.planning_latency_s,
            self.distance_m,
            self.energy_used_j,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in numeric):
            raise ValueError("execution receipt numeric fields must be finite and non-negative")
        if self.minimum_clearance_m is not None and (
            not math.isfinite(float(self.minimum_clearance_m))
            or float(self.minimum_clearance_m) < 0.0
        ):
            raise ValueError("execution receipt clearance must be finite and non-negative")
        if not self.source_observation_id:
            raise ValueError("execution receipt must bind a source observation")
        hashes = (
            self.action_packet_hash,
            self.source_observation_hash,
            self.state_before_hash,
            self.state_after_hash,
        )
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("execution receipt provenance hash is invalid")
        if self.previous_receipt_hash is not None and (
            len(self.previous_receipt_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.previous_receipt_hash)
        ):
            raise ValueError("execution receipt previous hash is invalid")
        if len(set(self.confirmation_ids)) != len(self.confirmation_ids):
            raise ValueError("execution receipt contains duplicate confirmation IDs")
        if not isinstance(self.planner_invoked, bool):
            raise ValueError("execution receipt planner_invoked must be boolean")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "org.aerocity.bench.execution-receipt.v3",
            **asdict(self),
            "confirmation_ids": list(self.confirmation_ids),
        }
        payload["receipt_hash"] = content_hash(payload)
        return payload


@dataclass(frozen=True)
class FailureRecord:
    episode_id: str
    drone_id: str | None
    task_time_s: float
    category: str
    detail: str
    terminal: bool

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "org.aerocity.bench.failure-record.v1", **asdict(self)}


@dataclass
class BudgetLedger:
    path_distance_m: float = 0.0
    energy_used_j: float = 0.0
    planning_time_s: float = 0.0
    communication_bytes_sent: int = 0
    communication_bytes_delivered: int = 0
    communication_bytes_dropped: int = 0
    communication_packets_sent: int = 0
    communication_packets_delivered: int = 0
    communication_packets_dropped: int = 0
    stale_messages_rejected: int = 0
    duplicate_messages_rejected: int = 0
    bandwidth_messages_rejected: int = 0
    deadline_misses: int = 0
    collisions: int = 0
    out_of_bounds_actions: int = 0
    safety_interventions: int = 0
    clearance_interventions: int = 0
    minimum_clearance_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
