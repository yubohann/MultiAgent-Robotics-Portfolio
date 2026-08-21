import math

from robocon_pose_command_bridge.core import (
    format_command,
    make_pose_command,
    pose_command_gate,
    quaternion_to_yaw,
)


def test_quaternion_yaw_uses_full_rotation():
    yaw = quaternion_to_yaw(0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0))
    assert math.isclose(yaw, math.pi / 4.0, abs_tol=1e-9)


def test_pose_command_wraps_heading_error():
    command = make_pose_command(0.0, 0.0, math.pi - 0.1, -1.0, -0.1)
    assert abs(command.heading_error) < 0.3


def test_command_is_versioned_json():
    payload = format_command(make_pose_command(1.0, 2.0, 0.0, 4.0, 15.0), 42)
    assert '"version":1' in payload
    assert '"type":"pose_target"' in payload


def test_command_gate_rejects_unlocked_map():
    allowed, reason = pose_command_gate(
        pose_available=True,
        pose_valid=True,
        map_locked=False,
        pose_age_sec=0.01,
        max_pose_age_sec=0.30,
    )
    assert not allowed
    assert reason == "map_unlocked"


def test_command_gate_rejects_stale_pose():
    allowed, reason = pose_command_gate(
        pose_available=True,
        pose_valid=True,
        map_locked=True,
        pose_age_sec=0.31,
        max_pose_age_sec=0.30,
    )
    assert not allowed
    assert reason == "pose_stale"


def test_command_gate_accepts_fresh_valid_global_pose():
    allowed, reason = pose_command_gate(
        pose_available=True,
        pose_valid=True,
        map_locked=True,
        pose_age_sec=0.05,
        max_pose_age_sec=0.30,
    )
    assert allowed
    assert reason == "ready"
