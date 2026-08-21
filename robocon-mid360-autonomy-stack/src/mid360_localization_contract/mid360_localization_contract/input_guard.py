"""Validate the Livox CustomMsg and IMU streams before they reach competition logic."""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from livox_ros_driver2.msg import CustomMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from .input_validation import inspect_custom_points


class Mid360InputGuard(Node):
    def __init__(self) -> None:
        super().__init__("mid360_input_guard")
        self.declare_parameter("lidar_topic", "/livox/lidar")
        self.declare_parameter("imu_topic", "/livox/imu")
        self.declare_parameter("input_valid_topic", "/mid360/input_valid")
        self.declare_parameter("diagnostic_topic", "/mid360/input_diagnostics")
        self.declare_parameter("minimum_points", 1)
        self.declare_parameter("max_lidar_silence_sec", 1.0)
        self.declare_parameter("max_imu_silence_sec", 1.0)
        self.declare_parameter("status_publish_hz", 10.0)

        self.minimum_points = int(self.get_parameter("minimum_points").value)
        self.max_lidar_silence_sec = float(self.get_parameter("max_lidar_silence_sec").value)
        self.max_imu_silence_sec = float(self.get_parameter("max_imu_silence_sec").value)

        self.last_lidar_arrival: float | None = None
        self.last_imu_arrival: float | None = None
        self.last_timebase: int | None = None
        self.last_lidar_ok = False
        self.last_imu_ok = False
        self.last_lidar_issue = "No LiDAR packet received."
        self.last_imu_issue = "No IMU packet received."
        self.last_point_count = 0
        self.last_finite_point_count = 0
        self.last_non_finite_point_count = 0
        self.last_offset_span_ns = 0
        self.lidar_received_sequence = 0
        self.lidar_accepted_sequence = 0
        self.lidar_drop_count = 0
        self.imu_received_sequence = 0
        self.imu_accepted_sequence = 0
        self.imu_drop_count = 0
        self.last_lidar_stamp_ns: int | None = None
        self.last_imu_stamp_ns: int | None = None
        self._last_status_signature: tuple[object, ...] | None = None

        self.valid_publisher = self.create_publisher(
            Bool, str(self.get_parameter("input_valid_topic").value), 10
        )
        self.diagnostic_publisher = self.create_publisher(
            DiagnosticArray, str(self.get_parameter("diagnostic_topic").value), 10
        )
        self.create_subscription(
            CustomMsg,
            str(self.get_parameter("lidar_topic").value),
            self._lidar_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter("imu_topic").value),
            self._imu_callback,
            qos_profile_sensor_data,
        )

        publish_hz = float(self.get_parameter("status_publish_hz").value)
        if publish_hz <= 0.0:
            raise ValueError("status_publish_hz must be positive.")
        self.create_timer(1.0 / publish_hz, lambda: self._publish_status(force=True))

    def _lidar_callback(self, message: CustomMsg) -> None:
        now = time.monotonic()
        self.lidar_received_sequence += 1
        issues: list[str] = []
        point_count = len(message.points)
        self.last_point_count = point_count

        if message.point_num != point_count:
            issues.append("point_num does not match the received point array length")
        if point_count < self.minimum_points:
            issues.append("received point count is below minimum_points")
        if message.timebase == 0:
            issues.append("timebase is zero")
        if self.last_timebase is not None and message.timebase <= self.last_timebase:
            issues.append("timebase did not increase")

        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        if stamp_ns <= 0:
            issues.append("LiDAR header stamp is zero")
        elif self.last_lidar_stamp_ns is not None and stamp_ns <= self.last_lidar_stamp_ns:
            issues.append("LiDAR header stamp did not increase")

        point_statistics = inspect_custom_points(message.points)
        self.last_finite_point_count = point_statistics.finite_point_count
        self.last_non_finite_point_count = point_statistics.non_finite_point_count
        if not point_statistics.offsets_monotonic:
            issues.append("offset_time is not monotonically nondecreasing")
        if point_statistics.non_finite_point_count:
            issues.append("point coordinates contain non-finite values")
        self.last_offset_span_ns = point_statistics.offset_span_ns

        if message.timebase > 0 and (self.last_timebase is None or message.timebase > self.last_timebase):
            self.last_timebase = int(message.timebase)
        self.last_lidar_arrival = now
        self.last_lidar_ok = not issues
        if self.last_lidar_ok:
            self.lidar_accepted_sequence += 1
            self.last_lidar_stamp_ns = stamp_ns
        else:
            self.lidar_drop_count += 1
        self.last_lidar_issue = "; ".join(issues) if issues else "LiDAR packet is structurally valid."
        self._publish_status(force=False)

    def _imu_callback(self, message: Imu) -> None:
        now = time.monotonic()
        self.imu_received_sequence += 1
        values = (
            message.angular_velocity.x,
            message.angular_velocity.y,
            message.angular_velocity.z,
            message.linear_acceleration.x,
            message.linear_acceleration.y,
            message.linear_acceleration.z,
        )
        issues: list[str] = []
        if message.header.stamp.sec == 0 and message.header.stamp.nanosec == 0:
            issues.append("IMU header stamp is zero")
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        if stamp_ns > 0 and self.last_imu_stamp_ns is not None and stamp_ns <= self.last_imu_stamp_ns:
            issues.append("IMU header stamp did not increase")
        if not all(math.isfinite(value) for value in values):
            issues.append("IMU contains a non-finite value")

        self.last_imu_arrival = now
        self.last_imu_ok = not issues
        if self.last_imu_ok:
            self.imu_accepted_sequence += 1
            self.last_imu_stamp_ns = stamp_ns
        else:
            self.imu_drop_count += 1
        self.last_imu_issue = "; ".join(issues) if issues else "IMU packet is structurally valid."
        self._publish_status(force=False)

    def _stream_is_fresh(self, last_arrival: float | None, timeout: float, now: float) -> bool:
        return last_arrival is not None and now - last_arrival <= timeout

    def _publish_status(self, *, force: bool = True) -> None:
        now = time.monotonic()
        lidar_fresh = self._stream_is_fresh(self.last_lidar_arrival, self.max_lidar_silence_sec, now)
        imu_fresh = self._stream_is_fresh(self.last_imu_arrival, self.max_imu_silence_sec, now)
        valid = self.last_lidar_ok and self.last_imu_ok and lidar_fresh and imu_fresh

        # Sensor callbacks can arrive hundreds of times per second. Publish a
        # transition immediately, but let the timer carry changing counters
        # during steady state so diagnostics do not add a second high-rate
        # workload to the simulator.
        signature = (
            valid,
            self.last_lidar_ok,
            self.last_imu_ok,
            lidar_fresh,
            imu_fresh,
            self.last_lidar_issue,
            self.last_imu_issue,
        )
        if not force and signature == self._last_status_signature:
            return
        self._last_status_signature = signature

        self.valid_publisher.publish(Bool(data=valid))
        status = DiagnosticStatus()
        status.name = f"{self.get_name()}/streams"
        status.hardware_id = "livox-mid360"
        status.level = DiagnosticStatus.OK if valid else DiagnosticStatus.ERROR
        status.message = "LiDAR and IMU streams are valid." if valid else "LiDAR or IMU stream is invalid."
        status.values = [
            KeyValue(key="lidar_ok", value=str(self.last_lidar_ok).lower()),
            KeyValue(key="lidar_fresh", value=str(lidar_fresh).lower()),
            KeyValue(key="lidar_issue", value=self.last_lidar_issue),
            KeyValue(key="last_point_count", value=str(self.last_point_count)),
            KeyValue(key="last_finite_point_count", value=str(self.last_finite_point_count)),
            KeyValue(key="last_non_finite_point_count", value=str(self.last_non_finite_point_count)),
            KeyValue(key="last_offset_span_ns", value=str(self.last_offset_span_ns)),
            KeyValue(key="last_timebase", value=str(self.last_timebase) if self.last_timebase is not None else "unknown"),
            KeyValue(key="lidar_received_sequence", value=str(self.lidar_received_sequence)),
            KeyValue(key="lidar_accepted_sequence", value=str(self.lidar_accepted_sequence)),
            KeyValue(key="lidar_drop_count", value=str(self.lidar_drop_count)),
            KeyValue(key="imu_ok", value=str(self.last_imu_ok).lower()),
            KeyValue(key="imu_fresh", value=str(imu_fresh).lower()),
            KeyValue(key="imu_issue", value=self.last_imu_issue),
            KeyValue(key="imu_received_sequence", value=str(self.imu_received_sequence)),
            KeyValue(key="imu_accepted_sequence", value=str(self.imu_accepted_sequence)),
            KeyValue(key="imu_drop_count", value=str(self.imu_drop_count)),
        ]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self.diagnostic_publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = Mid360InputGuard()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
