"""ROS-neutral wire codec and optional ROS 2 availability boundary."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from typing import Any

from .contracts import ActionPacket, MessagePacket, ObservationPacket, Pose3D

ROS_TOPICS = {
    "observation": "/aerocity/{drone_id}/observation",
    "action": "/aerocity/{drone_id}/action",
    "confirmation": "/aerocity/confirmations",
    "clock": "/clock",
}


def ros2_available() -> bool:
    return importlib.util.find_spec("rclpy") is not None


@dataclass(frozen=True)
class ROS2BridgeConfig:
    namespace: str = "/aerocity"
    qos_depth: int = 10
    use_sim_time: bool = True
    serialization: str = "canonical-json-v1"

    def __post_init__(self) -> None:
        if not self.namespace.startswith("/"):
            raise ValueError("ROS namespace must be absolute")
        if self.qos_depth < 1:
            raise ValueError("ROS QoS depth must be positive")
        if self.serialization != "canonical-json-v1":
            raise ValueError("unsupported ROS bridge serialization")


class ROSWireCodec:
    @staticmethod
    def observation(packet: ObservationPacket) -> bytes:
        return json.dumps(
            packet.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")

    @staticmethod
    def action(data: bytes, observation: ObservationPacket) -> ActionPacket:
        node = json.loads(data.decode("utf-8"))
        allowed = {
            "kind",
            "waypoint",
            "velocity_body_mps",
            "yaw_rate_deg_s",
            "source_observation_id",
            "messages",
        }
        if set(node) - allowed:
            raise ValueError("ROS action contains fields outside the canonical contract")
        kind = str(node["kind"])
        messages = []
        expected_message_fields = {
            "message_id",
            "destination_drone_ids",
            "expires_at_s",
            "payload_hex",
        }
        for record in node.get("messages", []):
            if not isinstance(record, dict) or set(record) != expected_message_fields:
                raise ValueError("ROS message fields differ from the canonical contract")
            destinations = record["destination_drone_ids"]
            if not isinstance(destinations, list):
                raise ValueError("ROS message destinations must be a list")
            try:
                payload = bytes.fromhex(str(record["payload_hex"]))
            except ValueError as exc:
                raise ValueError("ROS message payload_hex is invalid") from exc
            messages.append(
                MessagePacket(
                    message_id=str(record["message_id"]),
                    source_drone_id=observation.drone_id,
                    destination_drone_ids=tuple(str(value) for value in destinations),
                    created_at_s=observation.timestamp_s,
                    expires_at_s=float(record["expires_at_s"]),
                    payload=payload,
                )
            )
        return ActionPacket(
            episode_id=observation.episode_id,
            drone_id=observation.drone_id,
            sequence=observation.sequence,
            issued_at_s=observation.timestamp_s,
            kind=kind,  # type: ignore[arg-type]
            waypoint=(Pose3D.from_dict(node["waypoint"]) if node.get("waypoint") else None),
            velocity_body_mps=(
                tuple(float(value) for value in node["velocity_body_mps"])
                if node.get("velocity_body_mps") is not None
                else None
            ),  # type: ignore[arg-type]
            yaw_rate_deg_s=float(node.get("yaw_rate_deg_s", 0.0)),
            source_observation_id=(
                str(node.get("source_observation_id") or observation.observation_id)
                if kind == "OBSERVE"
                else None
            ),
            messages=tuple(messages),
        )


class ROS2Bridge:
    """Optional process boundary; actual nodes are created only when rclpy exists."""

    def __init__(self, config: ROS2BridgeConfig | None = None) -> None:
        self.config = config or ROS2BridgeConfig()
        if not ros2_available():
            raise RuntimeError(
                "ROS 2 bridge requires rclpy; use the canonical JSON codec in a separate "
                "container when the benchmark Python environment has no ROS installation"
            )

    def contract(self) -> dict[str, Any]:
        return {
            "schema": "org.aerocity.bench.ros2-bridge-contract.v1",
            "namespace": self.config.namespace,
            "qos_depth": self.config.qos_depth,
            "use_sim_time": self.config.use_sim_time,
            "topics": ROS_TOPICS,
            "wire_serialization": self.config.serialization,
            "target_truth_exposed": False,
        }
