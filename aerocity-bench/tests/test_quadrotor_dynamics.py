from __future__ import annotations

import math
from pathlib import Path

import pytest

from aerocity_bench.quadrotor_dynamics import (
    FlightCommand,
    FlightState,
    QuadrotorDynamicsSpec,
    allocation_matrix,
    candidate_controller_spec,
    controller_step,
    hover_rotor_speed_for_mass,
    motor_step,
    project_asset_spec,
    rotor_thrust_wrench,
    rotor_wrench,
)


def test_project_asset_is_explicitly_not_formal_until_parameter_sources_are_unified() -> None:
    spec = project_asset_spec()
    assert spec.provenance_status == "parameter_audit_pending"
    assert spec.formal_score_eligible is False
    assert spec.model_id == "cf2x-local-runtime-candidate-v1"
    assert spec.mass_kg == pytest.approx(0.0282)
    assert spec.inertia_diag_kg_m2 == pytest.approx((1.6572e-5, 1.6656e-5, 2.9262e-5))
    assert spec.radial_arm_lengths_m == pytest.approx((math.sqrt(2.0) * 0.031,) * 4)
    assert spec.rotor_joint_names == ("m1_joint", "m2_joint", "m3_joint", "m4_joint")
    assert spec.rotor_spin_directions == (1, -1, 1, -1)
    assert spec.fingerprint_payload()["provenance_status"] == "parameter_audit_pending"


def test_symmetric_hover_has_weight_support_and_zero_reaction_torque() -> None:
    spec = project_asset_spec()
    hover = spec.hover_rotor_speed_rad_s
    total_thrust, roll_torque, pitch_torque, yaw_torque = rotor_wrench(spec, (hover,) * 4)
    assert total_thrust == pytest.approx(spec.mass_kg * spec.gravity_mps2)
    assert roll_torque == pytest.approx(0.0, abs=1.0e-12)
    assert pitch_torque == pytest.approx(0.0, abs=1.0e-12)
    assert yaw_torque == pytest.approx(0.0, abs=1.0e-12)


def test_rotor_differential_changes_attitude_wrench_without_changing_contract() -> None:
    spec = project_asset_spec()
    hover = spec.hover_rotor_speed_rad_s
    wrench = rotor_wrench(spec, (hover * 1.05, hover * 1.05, hover * 0.95, hover * 0.95))
    assert abs(wrench[1]) > 0.0
    assert abs(wrench[2]) < 1.0e-12
    assert abs(wrench[3]) < 1.0e-12


def test_runtime_mass_hover_helper_does_not_promote_the_pending_spec() -> None:
    spec = project_asset_spec()
    runtime_hover = hover_rotor_speed_for_mass(spec, 0.0282)
    assert runtime_hover == pytest.approx(spec.hover_rotor_speed_rad_s)
    assert spec.formal_score_eligible is False


def test_shared_controller_keeps_hover_symmetric_without_promoting_the_model() -> None:
    dynamics = project_asset_spec()
    controller = candidate_controller_spec()
    state = FlightState(
        position_w_m=(0.0, 0.0, 1.5),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity_w_mps=(0.0, 0.0, 0.0),
        angular_velocity_w_rad_s=(0.0, 0.0, 0.0),
    )
    output = controller_step(
        dynamics,
        controller,
        state,
        FlightCommand((0.0, 0.0, 1.5), (0.0, 0.0, 0.0), 0.0),
        mass_kg=0.0282,
    )
    assert output.rotor_references_rad_s == pytest.approx(
        (hover_rotor_speed_for_mass(dynamics, 0.0282),) * 4
    )
    assert output.allocated_wrench[0] == pytest.approx(0.0282 * dynamics.gravity_mps2)
    assert output.allocated_wrench[1:] == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-12)
    assert controller.formal_score_eligible is False


def test_shared_controller_maps_public_lateral_target_to_bounded_pitch_wrench() -> None:
    dynamics = project_asset_spec()
    controller = candidate_controller_spec()
    state = FlightState(
        position_w_m=(0.0, 0.0, 1.5),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity_w_mps=(0.0, 0.0, 0.0),
        angular_velocity_w_rad_s=(0.0, 0.0, 0.0),
    )
    output = controller_step(
        dynamics,
        controller,
        state,
        FlightCommand((100.0, 0.0, 1.5), (0.0, 0.0, 0.0), 0.0),
        mass_kg=0.0282,
    )
    assert output.desired_acceleration_w_mps2[0] == pytest.approx(
        controller.max_acceleration_mps2
    )
    assert output.desired_acceleration_w_mps2[1:] == pytest.approx((0.0, 0.0))
    assert abs(output.allocated_wrench[1]) < 1.0e-10
    assert output.allocated_wrench[2] > 0.0
    assert abs(output.allocated_wrench[3]) < 1.0e-10
    assert all(
        0.0 <= speed <= dynamics.max_rotor_speed_rad_s
        for speed in output.rotor_references_rad_s
    )


def test_shared_controller_maps_public_yaw_target_to_bounded_yaw_wrench() -> None:
    dynamics = project_asset_spec()
    controller = candidate_controller_spec()
    state = FlightState(
        position_w_m=(0.0, 0.0, 1.5),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity_w_mps=(0.0, 0.0, 0.0),
        angular_velocity_w_rad_s=(0.0, 0.0, 0.0),
    )
    output = controller_step(
        dynamics,
        controller,
        state,
        FlightCommand((0.0, 0.0, 1.5), (0.0, 0.0, 0.0), math.pi / 2.0),
        mass_kg=0.0282,
    )
    assert abs(output.allocated_wrench[1]) < 1.0e-10
    assert abs(output.allocated_wrench[2]) < 1.0e-10
    assert output.allocated_wrench[3] > 0.0
    assert all(
        0.0 <= speed <= dynamics.max_rotor_speed_rad_s
        for speed in output.rotor_references_rad_s
    )


def test_motor_response_is_rate_limited_and_converges() -> None:
    spec = project_asset_spec()
    current = (0.0,) * 4
    reference = (spec.max_rotor_speed_rad_s,) * 4
    first = motor_step(spec, current, reference)
    rate_limited_step = spec.motor_rate_limits_rad_s2[0] * spec.physics_dt_s
    first_order_step = (
        spec.max_rotor_speed_rad_s
        * spec.physics_dt_s
        / spec.motor_time_constants_s[0]
    )
    assert first[0] == pytest.approx(min(rate_limited_step, first_order_step))
    assert max(first) < spec.max_rotor_speed_rad_s
    state = first
    for _ in range(500):
        state = motor_step(spec, state, reference)
    assert state == pytest.approx(reference, abs=1.0e-6)


def test_motor_step_does_not_overshoot_when_physics_step_exceeds_time_constant() -> None:
    spec = project_asset_spec()
    target = (1500.0,) * 4
    rising = motor_step(spec, (1000.0,) * 4, target)
    falling = motor_step(spec, (2000.0,) * 4, target)
    assert rising == pytest.approx((1208.3333333333333,) * 4)
    assert falling == pytest.approx((1791.6666666666667,) * 4)
    assert all(1000.0 <= value <= 1500.0 for value in rising)
    assert all(1500.0 <= value <= 2000.0 for value in falling)


def test_invalid_singular_geometry_is_rejected() -> None:
    spec = project_asset_spec()
    bad = QuadrotorDynamicsSpec(
        **{
            **spec.__dict__,
            "rotor_positions_body_m": ((0.1, 0.0, 0.0),) * 4,
        }
    )
    with pytest.raises(ValueError, match="singular"):
        bad.validate()


def test_allocation_matrix_is_four_by_four_and_non_singular() -> None:
    spec = project_asset_spec()
    matrix = allocation_matrix(spec)
    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)
    assert math.isclose(matrix[0][0], 1.0)


def test_native_thrust_units_produce_the_same_wrench_as_derived_rotor_speeds() -> None:
    spec = project_asset_spec()
    speeds = tuple(spec.hover_rotor_speed_rad_s * factor for factor in (0.9, 1.0, 1.1, 1.0))
    thrusts = tuple(spec.thrust_coeff_n_per_rad2 * speed * speed for speed in speeds)
    assert rotor_thrust_wrench(spec, thrusts) == pytest.approx(rotor_wrench(spec, speeds))


def test_legacy_native_gate_is_permanently_labeled_as_cuboid_capability_probe() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / "isaac_native_gate.py").read_text(encoding="utf-8")
    assert '"dynamic_cuboid_kinematic_capability_probe"' in source
    assert '"formal_score_eligible": False' in source
    assert '"status": "not_connected"' in source


def test_quadrotor_preflight_uses_cf2x_multirotor_not_kinematic_state_writes() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / "quadrotor_physics_preflight.py").read_text(encoding="utf-8")
    assert "verify_local_cf2x_asset" in source
    assert "build_cf2x_multirotor_cfg" in source
    assert "robot.set_thrust_target" in source
    assert "permanent_wrench_composer.set_forces_and_torques" not in source
    assert "controller_step(" in source
    assert "open-loop-pitch-pulse" in source
    assert "open-loop-drop" in source
    assert "shared-lateral-hold" in source
    assert "shared-long-hold" in source
    assert "shared-long-lateral-hold" in source
    assert "shared-altitude-hold" in source
    assert "shared-yaw-hold" in source
    assert "episode_reset_before_flight_loop" in source
    assert "applied_rotor_thrust_n" in source
    assert "contact_sensor" in source
    assert "runtime_quality" in source
    assert "max_tilt_angle_rad" in source
    assert "profile_maneuver_satisfied" in source
    assert "profile_attitude_response_satisfied" in source
    assert "profile_position_response_satisfied" in source
    assert "profile_yaw_response_satisfied" in source
    assert "profile_hold_stability_satisfied" in source
    assert "profile_long_hold_stability_satisfied" in source
    assert "profile_long_lateral_response_satisfied" in source
    assert "final_position_target_error_m" in source
    assert "long_horizon_hover_checks" in source
    assert "long_horizon_hover_metrics" in source
    assert "profile_braking_response_satisfied" in source
    assert "profile_contact_response_satisfied" in source
    assert "_CONTACT_EVIDENCE_MIN_FORCE_N = 0.05" in source
    assert "per_rotor_thrust_actuator_applied" in source
    assert "derived_geometry_allocation_to_root_body_physx" in source
    assert "prop_link_forces_applied_directly" in source
    assert "max_horizontal_displacement_m >= 0.01" in source
    assert "final_horizontal_displacement_m <= 0.08" in source
    assert "max_altitude_above_initial_m >= 0.01" in source
    assert "max_yaw_progress_rad >= 0.02" in source
    assert "repeat-resets" not in source
    flight_loop = source.split("for step in range(args.steps):", maxsplit=1)[1]
    assert "robot.write_root_pose_to_sim" not in flight_loop
    assert "robot.write_root_velocity_to_sim" not in flight_loop
    assert "set_linear_velocity" not in source
    assert "set_angular_velocity" not in source
    assert "FIVE_IN_DRONE" not in source
    assert "5_in_drone" not in source
    assert '"formal_score_eligible": False' in source


def test_vertical_slice_uses_the_same_cf2x_multirotor_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "tools" / "quadrotor_l1_vertical_slice.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--cf2x-usd", type=Path, required=True)' in source
    assert "verify_local_cf2x_asset" in source
    assert "build_cf2x_multirotor_cfg" in source
    assert "Multirotor" in source
    assert 'SOURCE_ROOT / "isaaclab_contrib"' in source
    assert "robot.set_thrust_target" in source
    assert "Multirotor.data.applied_thrust" in source
    assert "permanent_wrench_composer.set_forces_and_torques" not in source
    assert "FIVE_IN_DRONE" not in source
    assert "5_in_drone" not in source
