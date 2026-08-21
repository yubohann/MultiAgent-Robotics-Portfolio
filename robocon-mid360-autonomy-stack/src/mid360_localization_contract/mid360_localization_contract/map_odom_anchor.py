"""Maintain map-to-odom from a manual initial pose or a verified external correction."""

from __future__ import annotations

import json
import time

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster

from .geometry import compose_transform, invert_transform, normalize_quaternion
from .tracking import TrackingState, TrackingStateMachine


class Mid360MapOdomAnchor(Node):
    def __init__(self) -> None:
        super().__init__("mid360_map_odom_anchor")
        self.declare_parameter("odom_topic", "/mid360/local_odometry")
        self.declare_parameter("initialpose_topic", "/initialpose")
        self.declare_parameter("correction_topic", "/mid360/map_to_odom_correction")
        self.declare_parameter("localized_odom_topic", "/mid360/localization_odometry")
        self.declare_parameter("map_locked_topic", "/mid360/map_locked")
        self.declare_parameter("diagnostic_topic", "/mid360/localization_diagnostics")
        self.declare_parameter("pose_status_topic", "/mid360/pose_status")
        self.declare_parameter("pose_diagnostic_topic", "/mid360/pose_diagnostics")
        self.declare_parameter("input_diagnostic_topic", "/mid360/input_diagnostics")
        self.declare_parameter("map_diagnostic_topic", "/mid360/map_localization_diagnostics")
        self.declare_parameter("relocalization_request_topic", "/mid360/relocalization_request")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("require_pose_valid", True)
        self.declare_parameter("pose_valid_topic", "/mid360/pose_valid")
        self.declare_parameter("max_odom_silence_sec", 1.0)
        self.declare_parameter("tf_publish_hz", 20.0)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.require_pose_valid = bool(self.get_parameter("require_pose_valid").value)
        self.max_odom_silence_sec = float(self.get_parameter("max_odom_silence_sec").value)
        self.latest_odom: Odometry | None = None
        self.latest_odom_arrival: float | None = None
        self.pose_valid = False
        self.map_to_odom = None
        self.anchor_source = "none"
        self.anchor_set_at: float | None = None
        self.tracking = TrackingStateMachine()
        self.last_status = "Waiting for /initialpose or an external map-to-odom correction."
        self.quality_fields: dict[str, str] = {
            "degeneracy_score": "unknown",
            "dynamic_point_ratio": "unknown",
            "scan_map_overlap": "unknown",
            "map_update_allowed": "unknown",
            "map_update_veto_reason": "unknown",
            "protection_level": "unknown",
            "uncertainty_bound_m": "unknown",
            "resource_cpu_percent": "unknown",
            "resource_memory_percent": "unknown",
            "input_qos_ok": "unknown",
            "lidar_drop_count": "unknown",
            "imu_drop_count": "unknown",
            "last_point_count": "unknown",
            "last_offset_span_ns": "unknown",
            "last_fitness": "unknown",
            "last_scan_points": "unknown",
        }

        self.localization_publisher = self.create_publisher(
            Odometry, str(self.get_parameter("localized_odom_topic").value), 10
        )
        self.locked_publisher = self.create_publisher(
            Bool, str(self.get_parameter("map_locked_topic").value), 10
        )
        self.pose_status_publisher = self.create_publisher(
            String, str(self.get_parameter("pose_status_topic").value), 10
        )
        self.diagnostic_publisher = self.create_publisher(
            DiagnosticArray, str(self.get_parameter("diagnostic_topic").value), 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value), self._odom_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter("initialpose_topic").value),
            self._initialpose_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("correction_topic").value),
            self._correction_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("pose_valid_topic").value),
            self._pose_valid_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("relocalization_request_topic").value),
            self._relocalization_request_callback,
            10,
        )
        self.create_subscription(
            DiagnosticArray,
            str(self.get_parameter("pose_diagnostic_topic").value),
            self._diagnostic_array_callback,
            10,
        )
        self.create_subscription(
            DiagnosticArray,
            str(self.get_parameter("input_diagnostic_topic").value),
            self._diagnostic_array_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("map_diagnostic_topic").value),
            self._map_diagnostic_callback,
            10,
        )

        publish_hz = float(self.get_parameter("tf_publish_hz").value)
        if publish_hz <= 0.0:
            raise ValueError("tf_publish_hz must be positive.")
        self.create_timer(1.0 / publish_hz, self._publish_anchor_status)

    def _odom_callback(self, message: Odometry) -> None:
        if message.header.frame_id != self.odom_frame or message.child_frame_id != self.base_frame:
            self.last_status = "Rejected odometry with an unexpected frame contract."
            return
        self.latest_odom = message
        self.latest_odom_arrival = time.monotonic()
        if self.map_to_odom is not None:
            self._publish_localized_odometry(message)

    def _pose_valid_callback(self, message: Bool) -> None:
        self.pose_valid = message.data

    def _relocalization_request_callback(self, message: Bool) -> None:
        if message.data:
            self.tracking.request_relocalization()
            self.last_status = "Relocalization requested; waiting for a new initial pose or verified correction."

    def _diagnostic_array_callback(self, message: DiagnosticArray) -> None:
        for status in message.status:
            for item in status.values:
                if item.key in self.quality_fields:
                    self.quality_fields[item.key] = item.value
            if status.name.endswith("streams"):
                self.quality_fields["input_qos_ok"] = "unknown"

    def _map_diagnostic_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.quality_fields["map_update_veto_reason"] = "malformed_map_diagnostic"
            return
        if "last_fitness" in payload:
            self.quality_fields["last_fitness"] = str(payload["last_fitness"])
        if "last_scan_points" in payload:
            self.quality_fields["last_scan_points"] = str(payload["last_scan_points"])
        status = str(payload.get("status", "unknown"))
        self.quality_fields["map_update_allowed"] = str(status == "TRACKING").lower()
        self.quality_fields["map_update_veto_reason"] = "unknown" if status == "TRACKING" else status

    def _initialpose_callback(self, message: PoseWithCovarianceStamped) -> None:
        if message.header.frame_id and message.header.frame_id != self.map_frame:
            self.last_status = "Rejected initial pose with a frame other than map."
            return
        if not self._ready_for_anchor():
            return
        try:
            map_to_base = self._pose_message_to_transform(message.pose.pose)
            odom_to_base = self._odom_to_transform(self.latest_odom)
        except ValueError as error:
            self.last_status = f"Rejected initial pose: {error}"
            return
        self.map_to_odom = compose_transform(map_to_base, invert_transform(odom_to_base))
        self.anchor_source = "initialpose"
        self.anchor_set_at = time.monotonic()
        self.tracking.set_provisional_anchor()
        self.last_status = "Initial pose stored; waiting for a verified scan-to-map correction."
        self._publish_localized_odometry(self.latest_odom)

    def _correction_callback(self, message: Odometry) -> None:
        if message.header.frame_id != self.map_frame or message.child_frame_id != self.odom_frame:
            self.last_status = "Rejected map-to-odom correction with an unexpected frame contract."
            return
        try:
            self.map_to_odom = self._odom_to_transform(message)
            self.anchor_source = "external_correction"
            self.anchor_set_at = time.monotonic()
            self.tracking.accept_anchor()
        except ValueError as error:
            self.last_status = f"Rejected map-to-odom correction: {error}"
            return
        self.last_status = "map-to-odom anchor updated from external correction."
        if self.latest_odom is not None:
            self._publish_localized_odometry(self.latest_odom)

    def _ready_for_anchor(self) -> bool:
        if self.latest_odom is None or self.latest_odom_arrival is None:
            self.last_status = "Cannot set initial pose before receiving odometry."
            return False
        if time.monotonic() - self.latest_odom_arrival > self.max_odom_silence_sec:
            self.last_status = "Cannot set initial pose because odometry is stale."
            return False
        if self.require_pose_valid and not self.pose_valid:
            self.last_status = "Cannot set initial pose while pose_valid is false."
            return False
        return True

    def _odom_to_transform(self, message: Odometry):
        return self._pose_message_to_transform(message.pose.pose)

    @staticmethod
    def _pose_message_to_transform(pose):
        translation = (pose.position.x, pose.position.y, pose.position.z)
        rotation = normalize_quaternion((
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ))
        return translation, rotation

    def _publish_localized_odometry(self, odom: Odometry) -> None:
        map_to_base = compose_transform(self.map_to_odom, self._odom_to_transform(odom))
        translation, rotation = map_to_base
        output = Odometry()
        output.header.stamp = odom.header.stamp
        output.header.frame_id = self.map_frame
        output.child_frame_id = self.base_frame
        output.pose.pose.position.x = translation[0]
        output.pose.pose.position.y = translation[1]
        output.pose.pose.position.z = translation[2]
        output.pose.pose.orientation.x = rotation[0]
        output.pose.pose.orientation.y = rotation[1]
        output.pose.pose.orientation.z = rotation[2]
        output.pose.pose.orientation.w = rotation[3]
        output.pose.covariance = odom.pose.covariance
        output.twist = odom.twist
        self.localization_publisher.publish(output)

    def _publish_anchor_status(self) -> None:
        now = time.monotonic()
        odom_fresh = (
            self.latest_odom_arrival is not None
            and now - self.latest_odom_arrival <= self.max_odom_silence_sec
        )
        self.tracking.anchor_exists = self.map_to_odom is not None
        self.tracking.update(self.pose_valid, odom_fresh)
        locked = self.tracking.map_locked
        self.locked_publisher.publish(Bool(data=locked))
        if locked:
            translation, rotation = self.map_to_odom
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = self.map_frame
            transform.child_frame_id = self.odom_frame
            transform.transform.translation.x = translation[0]
            transform.transform.translation.y = translation[1]
            transform.transform.translation.z = translation[2]
            transform.transform.rotation.x = rotation[0]
            transform.transform.rotation.y = rotation[1]
            transform.transform.rotation.z = rotation[2]
            transform.transform.rotation.w = rotation[3]
            self.tf_broadcaster.sendTransform(transform)

        status = DiagnosticStatus()
        status.name = f"{self.get_name()}/map_anchor"
        status.hardware_id = "mid360-localization"
        status.level = DiagnosticStatus.OK if locked else DiagnosticStatus.WARN
        status.message = self.last_status
        status.values = [
            KeyValue(key="map_locked", value=str(locked).lower()),
            KeyValue(key="tracking_state", value=self.tracking.state.value),
            KeyValue(key="anchor_source", value=self.anchor_source),
            KeyValue(key="anchor_age_sec", value=str(max(0.0, now - self.anchor_set_at) if self.anchor_set_at is not None else "unknown")),
            KeyValue(key="pose_valid", value=str(self.pose_valid).lower()),
            KeyValue(key="odom_fresh", value=str(odom_fresh).lower()),
            KeyValue(key="relocalization_requested", value=str(self.tracking.relocalization_requested).lower()),
            KeyValue(key="map_frame", value=self.map_frame),
            KeyValue(key="odom_frame", value=self.odom_frame),
        ]
        status.values.extend(
            KeyValue(key=key, value=value) for key, value in sorted(self.quality_fields.items())
        )
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self.diagnostic_publisher.publish(message)
        payload = {
            "version": 2,
            "tracking_state": self.tracking.state.value,
            "pose_valid": bool(self.pose_valid and odom_fresh),
            "map_locked": bool(locked),
            "anchor_source": self.anchor_source,
            "anchor_age_sec": (max(0.0, now - self.anchor_set_at) if self.anchor_set_at is not None else None),
            "pose_age_sec": (max(0.0, now - self.latest_odom_arrival) if self.latest_odom_arrival is not None else None),
            "reason": self.last_status,
            "quality": dict(self.quality_fields),
        }
        self.pose_status_publisher.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main() -> None:
    rclpy.init()
    node = Mid360MapOdomAnchor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
