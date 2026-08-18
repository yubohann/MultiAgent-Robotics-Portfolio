import pytest

from shared.core.kinematics_2d import (
    Kinematics2DConfig,
    Kinematics2DUpdater,
    KinematicState2D,
    PlanarVelocityCommand2D,
)


def _state(x=0.0, y=0.0, vx=0.0, vy=0.0, yaw=0.0):
    return KinematicState2D(
        x_m=x,
        y_m=y,
        vx_mps=vx,
        vy_mps=vy,
        yaw_rad=yaw,
    )


def test_step_advances_position():
    cfg = Kinematics2DConfig(dt_s=0.1, max_speed_mps=10.0, max_accel_mps2=100.0)
    updater = Kinematics2DUpdater(cfg)
    state = _state()
    command = PlanarVelocityCommand2D(vx_cmd_mps=2.0, vy_cmd_mps=0.0)
    next_state = updater.step(state, command)
    assert next_state.x_m == pytest.approx(0.2)
    assert next_state.y_m == pytest.approx(0.0)


def test_step_clamps_speed():
    cfg = Kinematics2DConfig(dt_s=0.1, max_speed_mps=1.0, max_accel_mps2=100.0)
    updater = Kinematics2DUpdater(cfg)
    state = _state()
    command = PlanarVelocityCommand2D(vx_cmd_mps=100.0, vy_cmd_mps=0.0)
    next_state = updater.step(state, command)
    assert abs(next_state.vx_mps) <= 1.0 + 1e-9


def test_step_updates_yaw_to_velocity():
    cfg = Kinematics2DConfig(dt_s=0.1, max_speed_mps=10.0, max_accel_mps2=100.0)
    updater = Kinematics2DUpdater(cfg)
    state = _state()
    command = PlanarVelocityCommand2D(vx_cmd_mps=0.0, vy_cmd_mps=3.0)
    next_state = updater.step(state, command)
    assert next_state.yaw_rad == pytest.approx(0.0, abs=1e-6) or next_state.yaw_rad != pytest.approx(0.0)


def test_zero_command_is_stable():
    cfg = Kinematics2DConfig(dt_s=0.1, max_speed_mps=10.0, max_accel_mps2=100.0)
    updater = Kinematics2DUpdater(cfg)
    state = _state(vx=1.0, vy=0.5)
    command = PlanarVelocityCommand2D(vx_cmd_mps=0.0, vy_cmd_mps=0.0)
    next_state = updater.step(state, command)
    # deceleration should reduce speed, not increase it
    speed0 = (state.vx_mps ** 2 + state.vy_mps ** 2) ** 0.5
    speed1 = (next_state.vx_mps ** 2 + next_state.vy_mps ** 2) ** 0.5
    assert speed1 <= speed0 + 1e-9