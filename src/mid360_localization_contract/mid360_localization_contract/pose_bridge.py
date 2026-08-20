"""Convert a FAST-LIO tracking pose into the single odom-to-base_link contract."""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

from .geometry import (
    compose_transform,
    cross,
    invert_transform,
    normalize_quaternion,
    quaternion_from_rpy,
    rotate_vector,
    vector_norm,
    vector_scale,
)


class Mid360PoseBridge(Node):
    def __init__(self) -> None:
        super().__init__("mid360_pose_bridge")
        self.declare_parameter("source_odom_topic", "/Odometry")
        self.declare_parameter("output_odom_topic", "/mid360/local_odometry")
        self.declare_parameter("pose_valid_topic", "/mid360/pose_valid")
        self.declare_parameter("diagnostic_topic", "/mid360/pose_diagnostics")
        self.declare_parameter("source_odom_frame", "TBD")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("source_tracking_frame", "imu_link")
        self.declare_parameter("strict_source_frames", True)
        self.declare_parameter("base_to_tracking_translation_m", [0.0, 0.0, 0.0])
        self.declare_parameter("base_to_tracking_rpy_rad", [0.0, 0.0, 0.0])
        self.declare_parameter("max_pose_silence_sec", 1.0)
        self.declare_parameter("max_position_step_m", 0.0)
        self.declare_parameter("max_linear_speed_mps", 0.0)
        self.declare_parameter("max_yaw_rate_radps", 0.0)
        self.declare_parameter("max_abs_z_m", 0.0)
        self.declare_parameter("status_publish_hz", 10.0)

        self.source_odom_frame = str(self.get_parameter("source_odom_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.source_tracking_frame = str(self.get_parameter("source_tracking_frame").value)
        self.strict_source_frames = bool(self.get_parameter("strict_source_frames").value)
        translation = tuple(float(value) for value in self.get_parameter("base_to_tracking_translation_m").value)
        rpy = tuple(float(value) for value in self.get_parameter("base_to_tracking_rpy_rad").value)
        if len(translation) != 3 or len(rpy) != 3:
            raise ValueError("The base-to-tracking translation and RPY parameters require three values.")
        self.base_to_tracking = (translation, quaternion_from_rpy(*rpy))

        self.max_pose_silence_sec = float(self.get_parameter("max_pose_silence_sec").value)
        self.max_position_step_m = float(self.get_parameter("max_position_step_m").value)
        self.max_linear_speed_mps = float(self.get_parameter("max_linear_speed_mps").value)
        self.max_yaw_rate_radps = float(self.get_parameter("max_yaw_rate_radps").value)
        self.max_abs_z_m = float(self.get_parameter("max_abs_z_m").value)
        self.last_pose_arrival: float | None = None
        self.last_base_position: tuple[float, float, float] | None = None
        self.last_status = "No FAST-LIO odometry received."
        self.last_valid = False
        self.last_hard_failure = True
        self.received_sequence = 0
        self.accepted_sequence = 0
        self.drop_count = 0
        self.last_source_stamp_ns: int | None = None

        self.odom_publisher = self.create_publisher(
            Odometry, str(self.get_parameter("output_odom_topic").value), 10
        )
        self.valid_publisher = self.create_publisher(
            Bool, str(self.get_parameter("pose_valid_topic").value), 10
        )
        self.diagnostic_publisher = self.create_publisher(
            DiagnosticArray, str(self.get_parameter("diagnostic_topic").value), 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            Odometry,
            str(self.get_parameter("source_odom_topic").value),
            self._odom_callback,
            qos_profile_sensor_data,
        )

        publish_hz = float(self.get_parameter("status_publish_hz").value)
        if publish_hz <= 0.0:
            raise ValueError("status_publish_hz must be positive.")
        self.create_timer(1.0 / publish_hz, self._publish_status)

    def _odom_callback(self, message: Odometry) -> None:
        self.received_sequence += 1
        self.last_pose_arrival = time.monotonic()
        hard_issues = self._frame_issues(message)
        source_position = (
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            message.pose.pose.position.z,
        )
        try:
            source_rotation = normalize_quaternion((
                message.pose.pose.orientation.x,
                message.pose.pose.orientation.y,
                message.pose.pose.orientation.z,
                message.pose.pose.orientation.w,
            ))
        except ValueError as error:
            hard_issues.append(str(error))
            source_rotation = None

        if hard_issues or source_rotation is None:
            self.last_valid = False
            self.last_hard_failure = True
            self.drop_count += 1
            self.last_status = "; ".join(hard_issues)
            self._publish_status()
            return

        odom_to_tracking = (source_position, source_rotation)
        odom_to_base = compose_transform(odom_to_tracking, invert_transform(self.base_to_tracking))
        base_position, base_rotation = odom_to_base

        soft_issues = self._quality_issues(message, base_position)
        self.last_valid = not soft_issues
        self.last_hard_failure = False
        self.last_status = "; ".join(soft_issues) if soft_issues else "Pose is valid."
        if self.last_valid:
            self.accepted_sequence += 1
            self.last_base_position = base_position
        else:
            self.drop_count += 1

        output = Odometry()
        output.header.stamp = message.header.stamp
        output.header.frame_id = self.odom_frame
        output.child_frame_id = self.base_frame
        output.pose.pose.position.x = base_position[0]
        output.pose.pose.position.y = base_position[1]
        output.pose.pose.position.z = base_position[2]
        output.pose.pose.orientation.x = base_rotation[0]
        output.pose.pose.orientation.y = base_rotation[1]
        output.pose.pose.orientation.z = base_rotation[2]
        output.pose.pose.orientation.w = base_rotation[3]
        output.pose.covariance = message.pose.covariance
        self._transform_twist(message, output)
        output.twist.covariance = message.twist.covariance
        self.odom_publisher.publish(output)
        self._broadcast_transform(output)
        self._publish_status()

    def _frame_issues(self, message: Odometry) -> list[str]:
        if not self.strict_source_frames:
            return []
        issues: list[str] = []
        if message.header.frame_id != self.source_odom_frame:
            issues.append(
                f"source odom frame is '{message.header.frame_id}', expected '{self.source_odom_frame}'"
            )
        if message.child_frame_id != self.source_tracking_frame:
            issues.append(
                f"source child frame is '{message.child_frame_id}', expected '{self.source_tracking_frame}'"
            )
        return issues

    def _quality_issues(self, message: Odometry, base_position: tuple[float, float, float]) -> list[str]:
        issues: list[str] = []
        if self.max_position_step_m > 0.0 and self.last_base_position is not None:
            if vector_norm(tuple(base_position[index] - self.last_base_position[index] for index in range(3))) > self.max_position_step_m:
                issues.append("position step exceeds max_position_step_m")
        linear_speed = vector_norm((
            message.twist.twist.linear.x,
            message.twist.twist.linear.y,
            message.twist.twist.linear.z,
        ))
        if self.max_linear_speed_mps > 0.0 and linear_speed > self.max_linear_speed_mps:
            issues.append("linear speed exceeds max_linear_speed_mps")
        if self.max_yaw_rate_radps > 0.0 and abs(message.twist.twist.angular.z) > self.max_yaw_rate_radps:
            issues.append("yaw rate exceeds max_yaw_rate_radps")
        if self.max_abs_z_m > 0.0 and abs(base_position[2]) > self.max_abs_z_m:
            issues.append("base height exceeds max_abs_z_m")
        return issues

    def _transform_twist(self, source: Odometry, output: Odometry) -> None:
        base_to_tracking_translation, base_to_tracking_rotation = self.base_to_tracking
        source_linear = (
            source.twist.twist.linear.x,
            source.twist.twist.linear.y,
            source.twist.twist.linear.z,
        )
        source_angular = (
            source.twist.twist.angular.x,
            source.twist.twist.angular.y,
            source.twist.twist.angular.z,
        )
        angular_in_base = rotate_vector(base_to_tracking_rotation, source_angular)
        velocity_at_tracking_in_base = rotate_vector(base_to_tracking_rotation, source_linear)
        linear_at_base = tuple(
            velocity_at_tracking_in_base[index] - cross(angular_in_base, base_to_tracking_translation)[index]
            for index in range(3)
        )
        output.twist.twist.linear.x = linear_at_base[0]
        output.twist.twist.linear.y = linear_at_base[1]
        output.twist.twist.linear.z = linear_at_base[2]
        output.twist.twist.angular.x = angular_in_base[0]
        output.twist.twist.angular.y = angular_in_base[1]
        output.twist.twist.angular.z = angular_in_base[2]

    def _broadcast_transform(self, odometry: Odometry) -> None:
        transform = TransformStamped()
        transform.header = odometry.header
        transform.child_frame_id = odometry.child_frame_id
        transform.transform.translation.x = odometry.pose.pose.position.x
        transform.transform.translation.y = odometry.pose.pose.position.y
        transform.transform.translation.z = odometry.pose.pose.position.z
        transform.transform.rotation = odometry.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def _publish_status(self) -> None:
        fresh = (
            self.last_pose_arrival is not None
            and time.monotonic() - self.last_pose_arrival <= self.max_pose_silence_sec
        )
        valid = self.last_valid and fresh and not self.last_hard_failure
        self.valid_publisher.publish(Bool(data=valid))

        status = DiagnosticStatus()
        status.name = f"{self.get_name()}/pose"
        status.hardware_id = "fast-lio2"
        status.level = DiagnosticStatus.OK if valid else DiagnosticStatus.ERROR if self.last_hard_failure or not fresh else DiagnosticStatus.WARN
        status.message = self.last_status if fresh else "FAST-LIO odometry is stale."
        status.values = [
            KeyValue(key="pose_valid", value=str(valid).lower()),
            KeyValue(key="pose_fresh", value=str(fresh).lower()),
            KeyValue(key="source_odom_frame", value=self.source_odom_frame),
            KeyValue(key="odom_frame", value=self.odom_frame),
            KeyValue(key="base_frame", value=self.base_frame),
            KeyValue(key="source_tracking_frame", value=self.source_tracking_frame),
            KeyValue(key="pose_age_sec", value=str(self._pose_age_sec())),
            KeyValue(key="received_sequence", value=str(self.received_sequence)),
            KeyValue(key="accepted_sequence", value=str(self.accepted_sequence)),
            KeyValue(key="drop_count", value=str(self.drop_count)),
            KeyValue(key="frame_ok", value=str(not self.last_hard_failure).lower()),
            KeyValue(key="quality_reason", value=self.last_status),
        ]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self.diagnostic_publisher.publish(message)

    def _pose_age_sec(self) -> float:
        if self.last_pose_arrival is None:
            return float("inf")
        return max(0.0, time.monotonic() - self.last_pose_arrival)


def main() -> None:
    rclpy.init()
    node = Mid360PoseBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
