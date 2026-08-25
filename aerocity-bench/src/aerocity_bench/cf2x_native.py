"""Native-only CF2X multirotor construction helpers.

The imports that require IsaacLab are deliberately local.  Importing this
module in ordinary tests is safe; calling the builder is only valid after the
Isaac application has been created.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .cf2x_contract import CF2X_THRUSTER_BODY_NAMES, VerifiedCF2XAsset
from .quadrotor_dynamics import QuadrotorDynamicsSpec, allocation_matrix


def cf2x_allocation_matrix(spec: QuadrotorDynamicsSpec) -> list[list[float]]:
    """Return the IsaacLab 6x4 force/torque allocation for the reviewed USD."""

    matrix = allocation_matrix(spec)
    return [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        list(matrix[0]),
        list(matrix[1]),
        list(matrix[2]),
        list(matrix[3]),
    ]


def cf2x_max_thrust_per_rotor_n(spec: QuadrotorDynamicsSpec) -> float:
    return spec.thrust_coeff_n_per_rad2 * spec.max_rotor_speed_rad_s**2


def cf2x_thrust_constant_n_per_rps2(spec: QuadrotorDynamicsSpec) -> float:
    """Convert the public candidate coefficient from rad/s to Isaac rps."""

    return spec.thrust_coeff_n_per_rad2 * (2.0 * math.pi) ** 2


def cf2x_hover_rps(spec: QuadrotorDynamicsSpec, mass_kg: float | None = None) -> float:
    mass = spec.mass_kg if mass_kg is None else float(mass_kg)
    if mass <= 0.0:
        raise ValueError("mass_kg must be positive")
    return spec.hover_rotor_speed_rad_s / (2.0 * math.pi) * math.sqrt(mass / spec.mass_kg)


def validate_cf2x_runtime_masses_kg(
    masses_kg: Iterable[float],
    *,
    expected_total_mass_kg: float,
    absolute_tolerance_kg: float = 1.0e-6,
) -> tuple[tuple[float, ...], float]:
    """Validate native PhysX body masses against the reviewed USD total.

    ``Multirotor`` currently replaces the base ``ArticulationData`` during
    initialization, so its optional ``data.default_mass`` cache may be empty.
    The authoritative native value is therefore the articulation PhysX view,
    not a best-effort fallback to the static contract.  A mismatch stops the
    run before the controller can use a mass inconsistent with the USD.
    """

    values = tuple(float(value) for value in masses_kg)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("CF2X native body masses must be finite and positive")
    expected = float(expected_total_mass_kg)
    tolerance = float(absolute_tolerance_kg)
    if not math.isfinite(expected) or expected <= 0.0:
        raise ValueError("CF2X expected total mass must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("CF2X mass tolerance must be finite and non-negative")
    total = sum(values)
    if abs(total - expected) > tolerance:
        raise ValueError(
            "CF2X native PhysX mass differs from the reviewed USD contract: "
            f"measured={total:.9f} kg expected={expected:.9f} kg tolerance={tolerance:.9f} kg"
        )
    return values, total


def read_verified_cf2x_runtime_mass_kg(
    robot: Any,
    *,
    expected_total_mass_kg: float,
    absolute_tolerance_kg: float = 1.0e-6,
) -> tuple[tuple[float, ...], float]:
    """Read and validate the native PhysX body masses for one CF2X instance."""

    try:
        tensor = robot.root_physx_view.get_masses()
        values = tensor[0].detach().cpu().reshape(-1).tolist()
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("CF2X native articulation did not expose PhysX body masses") from exc
    return validate_cf2x_runtime_masses_kg(
        values,
        expected_total_mass_kg=expected_total_mass_kg,
        absolute_tolerance_kg=absolute_tolerance_kg,
    )


def build_cf2x_multirotor_cfg(
    asset: VerifiedCF2XAsset,
    dynamics: QuadrotorDynamicsSpec,
    *,
    dt_s: float,
    prim_path: str,
    position_w_m: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
) -> Any:
    """Build the only supported native CF2X execution backend.

    ``Multirotor`` advances four independent ``ThrusterCfg`` actuator states
    from per-rotor thrust targets.  It then maps those applied thrusts through
    the geometry-derived allocation matrix and applies the resulting 6D wrench
    to the root body through PhysX.  It does not apply four external forces at
    the individual prop-link bodies.

    No caller may replace this boundary with direct root-state writes, joint
    velocity targets, a caller-authored raw body wrench, or an alternate USD
    path.
    """

    dynamics.validate()
    if tuple(dynamics.rotor_joint_names) != ("m1_joint", "m2_joint", "m3_joint", "m4_joint"):
        raise ValueError("CF2X dynamics rotor joint order is inconsistent with the USD contract")
    if not prim_path.startswith("/World/"):
        raise ValueError("CF2X native prim path must be rooted below /World")
    if dt_s <= 0.0:
        raise ValueError("native CF2X timestep must be positive")

    import isaaclab.sim as sim_utils
    from isaaclab_contrib.actuators import ThrusterCfg
    from isaaclab_contrib.assets import MultirotorCfg

    maximum_thrust = cf2x_max_thrust_per_rotor_n(dynamics)
    thrust_constant = cf2x_thrust_constant_n_per_rps2(dynamics)
    hover_rps = cf2x_hover_rps(dynamics)
    tau_inc = min(dynamics.motor_time_constants_s)
    tau_dec = min(dynamics.motor_time_constants_s)
    yaw_ratio = dynamics.drag_coeff_nm_per_rad2 / dynamics.thrust_coeff_n_per_rad2
    maximum_thrust_rate_n_s = (
        2.0
        * dynamics.thrust_coeff_n_per_rad2
        * dynamics.max_rotor_speed_rad_s
        * min(dynamics.motor_rate_limits_rad_s2)
    )
    return MultirotorCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset.usd_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.02,
                angular_damping=0.02,
                max_linear_velocity=25.0,
                max_angular_velocity=25.0,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            copy_from_source=False,
        ),
        init_state=MultirotorCfg.InitialStateCfg(
            pos=position_w_m,
            rot=orientation_wxyz,
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            rps={name: hover_rps for name in CF2X_THRUSTER_BODY_NAMES},
        ),
        actuators={
            "thrusters": ThrusterCfg(
                dt=dt_s,
                thrust_range=(0.0, maximum_thrust),
                max_thrust_rate=maximum_thrust_rate_n_s,
                thrust_const_range=(thrust_constant, thrust_constant),
                tau_inc_range=(tau_inc, tau_inc),
                tau_dec_range=(tau_dec, tau_dec),
                torque_to_thrust_ratio=yaw_ratio,
                thruster_names_expr=list(CF2X_THRUSTER_BODY_NAMES),
            )
        },
        allocation_matrix=cf2x_allocation_matrix(dynamics),
        rotor_directions=list(dynamics.rotor_spin_directions),
    )
