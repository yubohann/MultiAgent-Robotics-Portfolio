"""Pure calculations and safety gates for pose-derived control commands."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PoseCommand:
    x: float
    y: float
    yaw: float
    distance: float
    heading_error: float


def pose_command_gate(
    *,
    pose_available: bool,
    pose_valid: bool,
    map_locked: bool,
    pose_age_sec: float,
    max_pose_age_sec: float,
) -> tuple[bool, str]:
    """Return whether a global pose may drive a control command."""
    if not pose_available:
        return False, "pose_unavailable"
    if not pose_valid:
        return False, "pose_invalid"
    if not map_locked:
        return False, "map_unlocked"
    if pose_age_sec < 0.0 or pose_age_sec > max_pose_age_sec:
        return False, "pose_stale"
    return True, "ready"


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw from a ROS quaternion, never from quaternion.z alone."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        raise ValueError("orientation quaternion is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def make_pose_command(x: float, y: float, yaw: float, target_x: float, target_y: float) -> PoseCommand:
    dx = target_x - x
    dy = target_y - y
    distance = math.hypot(dx, dy)
    target_heading = math.atan2(dy, dx)
    return PoseCommand(x, y, yaw, distance, wrap_angle(target_heading - yaw))


def format_command(command: PoseCommand, stamp_ns: int) -> str:
    """Serialize a versioned command; hardware writers may wrap this payload."""
    return (
        f"{{\"version\":1,\"type\":\"pose_target\",\"stamp_ns\":{stamp_ns},"
        f"\"x\":{command.x:.4f},\"y\":{command.y:.4f},\"yaw\":{command.yaw:.6f},"
        f"\"distance\":{command.distance:.4f},\"heading_error\":{command.heading_error:.6f}}}"
    )
