"""Report whether the live ROS graph satisfies the MID-360 competition contract."""

from __future__ import annotations

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool


class Mid360Preflight(Node):
    def __init__(self) -> None:
        super().__init__("mid360_preflight")
        self.declare_parameter("lidar_topic", "/livox/lidar")
        self.declare_parameter("imu_topic", "/livox/imu")
        self.declare_parameter("source_odom_topic", "/Odometry")
        self.declare_parameter("canonical_odom_topic", "/mid360/local_odometry")
        self.declare_parameter("input_valid_topic", "/mid360/input_valid")
        self.declare_parameter("pose_valid_topic", "/mid360/pose_valid")
        self.declare_parameter("map_locked_topic", "/mid360/map_locked")
        self.declare_parameter("preflight_ready_topic", "/mid360/preflight_ready")
        self.declare_parameter("diagnostic_topic", "/mid360/preflight_diagnostics")
        self.declare_parameter("require_map_locked", True)
        self.declare_parameter("startup_grace_sec", 10.0)
        self.declare_parameter("max_odom_silence_sec", 0.30)
        self.declare_parameter("status_publish_hz", 2.0)

        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.imu_topic = str(self.get_parameter("imu_topic").value)
        self.source_odom_topic = str(self.get_parameter("source_odom_topic").value)
        self.canonical_odom_topic = str(self.get_parameter("canonical_odom_topic").value)
        self.require_map_locked = bool(self.get_parameter("require_map_locked").value)
        self.startup_grace_sec = float(self.get_parameter("startup_grace_sec").value)
        self.max_odom_silence_sec = float(self.get_parameter("max_odom_silence_sec").value)
        self.started_at = time.monotonic()
        self.last_odom_arrival: float | None = None
        self.flags: dict[str, tuple[bool | None, float | None]] = {
            "input_valid": (None, None),
            "pose_valid": (None, None),
            "map_locked": (None, None),
        }

        self.ready_publisher = self.create_publisher(
            Bool, str(self.get_parameter("preflight_ready_topic").value), 10
        )
        self.diagnostic_publisher = self.create_publisher(
            DiagnosticArray, str(self.get_parameter("diagnostic_topic").value), 10
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("input_valid_topic").value),
            lambda message: self._flag_callback("input_valid", message),
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("pose_valid_topic").value),
            lambda message: self._flag_callback("pose_valid", message),
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("map_locked_topic").value),
            lambda message: self._flag_callback("map_locked", message),
            10,
        )
        self.create_subscription(Odometry, self.canonical_odom_topic, self._odom_callback, 10)

        publish_hz = float(self.get_parameter("status_publish_hz").value)
        if publish_hz <= 0.0:
            raise ValueError("status_publish_hz must be positive.")
        self.create_timer(1.0 / publish_hz, self._publish_status)

    def _flag_callback(self, name: str, message: Bool) -> None:
        self.flags[name] = (message.data, time.monotonic())

    def _odom_callback(self, _message: Odometry) -> None:
        self.last_odom_arrival = time.monotonic()

    def _topic_type_state(self) -> tuple[bool, list[KeyValue], list[str]]:
        graph_types = dict(self.get_topic_names_and_types())
        requirements = [
            ("lidar", self.lidar_topic, "livox_ros_driver2/msg/CustomMsg"),
            ("imu", self.imu_topic, "sensor_msgs/msg/Imu"),
            ("source_odom", self.source_odom_topic, "nav_msgs/msg/Odometry"),
            ("canonical_odom", self.canonical_odom_topic, "nav_msgs/msg/Odometry"),
        ]
        values: list[KeyValue] = []
        failures: list[str] = []
        for name, topic, expected_type in requirements:
            actual_types = graph_types.get(topic, [])
            value = ",".join(actual_types) if actual_types else "missing"
            values.append(KeyValue(key=f"topic.{name}", value=value))
            if expected_type not in actual_types:
                failures.append(f"{topic} must publish {expected_type}")
        return not failures, values, failures

    def _flag_state(self, name: str, now: float) -> tuple[bool, str]:
        value, received_at = self.flags[name]
        if received_at is None:
            return False, f"{name} has not been received"
        if now - received_at > self.max_odom_silence_sec:
            return False, f"{name} is stale"
        if not value:
            return False, f"{name} is false"
        return True, f"{name} is true"

    def _publish_status(self) -> None:
        now = time.monotonic()
        in_grace_period = now - self.started_at < self.startup_grace_sec
        type_ok, topic_values, type_failures = self._topic_type_state()
        input_ok, input_reason = self._flag_state("input_valid", now)
        pose_ok, pose_reason = self._flag_state("pose_valid", now)
        map_ok, map_reason = self._flag_state("map_locked", now)
        odom_ok = (
            self.last_odom_arrival is not None
            and now - self.last_odom_arrival <= self.max_odom_silence_sec
        )

        readiness_items = [type_ok, input_ok, pose_ok, odom_ok]
        if self.require_map_locked:
            readiness_items.append(map_ok)
        ready = all(readiness_items)
        self.ready_publisher.publish(Bool(data=ready))

        failures = list(type_failures)
        if not input_ok:
            failures.append(input_reason)
        if not pose_ok:
            failures.append(pose_reason)
        if not odom_ok:
            failures.append("canonical odometry is stale or absent")
        if self.require_map_locked and not map_ok:
            failures.append(map_reason)

        status = DiagnosticStatus()
        status.name = f"{self.get_name()}/competition_contract"
        status.hardware_id = "livox-mid360"
        if ready:
            status.level = DiagnosticStatus.OK
            status.message = "Competition localization contract is ready."
        elif in_grace_period:
            status.level = DiagnosticStatus.WARN
            status.message = "Waiting for the ROS graph during startup grace period."
        else:
            status.level = DiagnosticStatus.ERROR
            status.message = "; ".join(failures)
        status.values = topic_values + [
            KeyValue(key="input_valid", value=input_reason),
            KeyValue(key="pose_valid", value=pose_reason),
            KeyValue(key="map_locked", value=map_reason),
            KeyValue(key="canonical_odom_fresh", value=str(odom_ok).lower()),
            KeyValue(key="require_map_locked", value=str(self.require_map_locked).lower()),
            KeyValue(key="preflight_ready", value=str(ready).lower()),
        ]
        diagnostic = DiagnosticArray()
        diagnostic.header.stamp = self.get_clock().now().to_msg()
        diagnostic.status = [status]
        self.diagnostic_publisher.publish(diagnostic)


def main() -> None:
    rclpy.init()
    node = Mid360Preflight()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
