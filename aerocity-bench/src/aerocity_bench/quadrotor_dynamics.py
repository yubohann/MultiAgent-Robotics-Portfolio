"""Pure-Python contract for the AeroCityBench quadrotor execution model.

This module intentionally has no Isaac/torch dependency.  It is the single
place where the formal executor and CPU contract tests agree on the rotor
layout, thrust model, motor response, and model provenance.  A model is not
formal-score eligible merely because it contains four rotors: its parameter
provenance must be explicitly promoted to ``frozen_and_verified``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

Vec3 = tuple[float, float, float]
Wrench = tuple[float, float, float, float]
Quaternion = tuple[float, float, float, float]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vec3(values: Iterable[float], name: str) -> Vec3:
    result = tuple(_finite(value, name) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class QuadrotorDynamicsSpec:
    """Physical and actuator parameters used by one native quadrotor.

    ``rotor_positions_body_m`` are measured from the body origin.  The
    allocation matrix is derived from those positions instead of accepting a
    second, potentially inconsistent arm-length parameter.
    """

    model_id: str
    provenance_status: str
    mass_kg: float
    inertia_diag_kg_m2: Vec3
    rotor_joint_names: tuple[str, str, str, str]
    rotor_positions_body_m: tuple[Vec3, Vec3, Vec3, Vec3]
    rotor_spin_directions: tuple[int, int, int, int]
    thrust_coeff_n_per_rad2: float
    drag_coeff_nm_per_rad2: float
    max_rotor_speed_rad_s: float
    motor_time_constants_s: tuple[float, float, float, float]
    motor_rate_limits_rad_s2: tuple[float, float, float, float]
    gravity_mps2: float = 9.81
    physics_dt_s: float = 1.0 / 60.0

    def validate(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.provenance_status not in {"parameter_audit_pending", "frozen_and_verified"}:
            raise ValueError("unsupported quadrotor provenance_status")
        positive = {
            "mass_kg": self.mass_kg,
            "thrust_coeff_n_per_rad2": self.thrust_coeff_n_per_rad2,
            "drag_coeff_nm_per_rad2": self.drag_coeff_nm_per_rad2,
            "max_rotor_speed_rad_s": self.max_rotor_speed_rad_s,
            "gravity_mps2": self.gravity_mps2,
            "physics_dt_s": self.physics_dt_s,
        }
        for name, value in positive.items():
            if _finite(value, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if any(_finite(value, "inertia_diag_kg_m2") <= 0.0 for value in self.inertia_diag_kg_m2):
            raise ValueError("all diagonal inertia values must be positive")
        if len(self.rotor_positions_body_m) != 4:
            raise ValueError("exactly four rotor positions are required")
        if len(self.rotor_joint_names) != 4 or len(set(self.rotor_joint_names)) != 4:
            raise ValueError("exactly four unique rotor joint names are required")
        for index, position in enumerate(self.rotor_positions_body_m):
            _vec3(position, f"rotor_positions_body_m[{index}]")
        if not all(
            abs(position[2] - self.rotor_positions_body_m[0][2]) <= 1.0e-5
            for position in self.rotor_positions_body_m
        ):
            raise ValueError("rotor heights must share one body-frame plane")
        if len(set(self.rotor_spin_directions)) != 2 or set(self.rotor_spin_directions) != {-1, 1}:
            raise ValueError("rotor spin directions must contain both +1 and -1")
        if any(direction not in {-1, 1} for direction in self.rotor_spin_directions):
            raise ValueError("rotor spin directions must be +1 or -1")
        for index, value in enumerate(self.motor_time_constants_s):
            if _finite(value, f"motor_time_constants_s[{index}]") <= 0.0:
                raise ValueError("motor time constants must be positive")
        for index, value in enumerate(self.motor_rate_limits_rad_s2):
            if _finite(value, f"motor_rate_limits_rad_s2[{index}") <= 0.0:
                raise ValueError("motor rate limits must be positive")

        # The X geometry must be non-degenerate.  This catches the historical
        # bug where a controller used an unrelated 0.035 m arm length.
        matrix = allocation_matrix(self)
        if abs(_determinant4(matrix)) <= 1.0e-12:
            raise ValueError("rotor geometry produces a singular allocation matrix")

    @property
    def formal_score_eligible(self) -> bool:
        return self.provenance_status == "frozen_and_verified"

    @property
    def radial_arm_lengths_m(self) -> tuple[float, float, float, float]:
        return tuple(
            math.hypot(position[0], position[1])
            for position in self.rotor_positions_body_m
        )

    @property
    def hover_rotor_speed_rad_s(self) -> float:
        return math.sqrt(self.mass_kg * self.gravity_mps2 / (4.0 * self.thrust_coeff_n_per_rad2))

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "schema": "org.aerocity.bench.quadrotor-dynamics-spec.v1",
            "model_id": self.model_id,
            "provenance_status": self.provenance_status,
            "mass_kg": self.mass_kg,
            "inertia_diag_kg_m2": list(self.inertia_diag_kg_m2),
            "rotor_joint_names": list(self.rotor_joint_names),
            "rotor_positions_body_m": [list(position) for position in self.rotor_positions_body_m],
            "rotor_spin_directions": list(self.rotor_spin_directions),
            "thrust_coeff_n_per_rad2": self.thrust_coeff_n_per_rad2,
            "drag_coeff_nm_per_rad2": self.drag_coeff_nm_per_rad2,
            "max_rotor_speed_rad_s": self.max_rotor_speed_rad_s,
            "motor_time_constants_s": list(self.motor_time_constants_s),
            "motor_rate_limits_rad_s2": list(self.motor_rate_limits_rad_s2),
            "gravity_mps2": self.gravity_mps2,
            "physics_dt_s": self.physics_dt_s,
        }


@dataclass(frozen=True)
class QuadrotorControllerSpec:
    """Shared high-level flight-controller parameters for the L1 executor.

    Baseline and external methods provide task-level targets only.  This
    controller converts them into a body wrench and four rotor references,
    so a planner cannot receive an unearned kinematic-control advantage.
    Its values remain an explicit candidate until native calibration freezes
    their source and operating envelope.
    """

    provenance_status: str
    position_gain_s2: float
    velocity_gain_s: float
    attitude_gain_nm_per_rad: float
    angular_rate_gain_nm_s_per_rad: float
    max_acceleration_mps2: float
    max_torque_nm: float

    def validate(self) -> None:
        if self.provenance_status not in {"parameter_audit_pending", "frozen_and_verified"}:
            raise ValueError("unsupported controller provenance_status")
        values = {
            "position_gain_s2": self.position_gain_s2,
            "velocity_gain_s": self.velocity_gain_s,
            "attitude_gain_nm_per_rad": self.attitude_gain_nm_per_rad,
            "angular_rate_gain_nm_s_per_rad": self.angular_rate_gain_nm_s_per_rad,
            "max_acceleration_mps2": self.max_acceleration_mps2,
            "max_torque_nm": self.max_torque_nm,
        }
        for name, value in values.items():
            if _finite(value, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    @property
    def formal_score_eligible(self) -> bool:
        return self.provenance_status == "frozen_and_verified"

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "schema": "org.aerocity.bench.quadrotor-controller-spec.v1",
            "provenance_status": self.provenance_status,
            "position_gain_s2": self.position_gain_s2,
            "velocity_gain_s": self.velocity_gain_s,
            "attitude_gain_nm_per_rad": self.attitude_gain_nm_per_rad,
            "angular_rate_gain_nm_s_per_rad": self.angular_rate_gain_nm_s_per_rad,
            "max_acceleration_mps2": self.max_acceleration_mps2,
            "max_torque_nm": self.max_torque_nm,
        }


@dataclass(frozen=True)
class FlightState:
    """Measured rigid-body state supplied by PhysX, expressed in world axes."""

    position_w_m: Vec3
    orientation_wxyz: Quaternion
    linear_velocity_w_mps: Vec3
    angular_velocity_w_rad_s: Vec3

    def validate(self) -> None:
        _vec3(self.position_w_m, "position_w_m")
        _vec3(self.linear_velocity_w_mps, "linear_velocity_w_mps")
        _vec3(self.angular_velocity_w_rad_s, "angular_velocity_w_rad_s")
        if len(self.orientation_wxyz) != 4:
            raise ValueError("orientation_wxyz must contain exactly four values")
        norm_sq = sum(
            _finite(value, "orientation_wxyz") * _finite(value, "orientation_wxyz")
            for value in self.orientation_wxyz
        )
        if norm_sq <= 1.0e-12:
            raise ValueError("orientation_wxyz must be non-zero")


@dataclass(frozen=True)
class FlightCommand:
    """Public controller target derived from a method action.

    ``target_position_w_m`` may be held at the current public waypoint while
    ``target_velocity_w_mps`` carries a velocity action.  No private target
    coordinate or evaluator state belongs in this command.
    """

    target_position_w_m: Vec3
    target_velocity_w_mps: Vec3
    target_yaw_rad: float
    target_yaw_rate_rad_s: float = 0.0

    def validate(self) -> None:
        _vec3(self.target_position_w_m, "target_position_w_m")
        _vec3(self.target_velocity_w_mps, "target_velocity_w_mps")
        _finite(self.target_yaw_rad, "target_yaw_rad")
        _finite(self.target_yaw_rate_rad_s, "target_yaw_rate_rad_s")


@dataclass(frozen=True)
class RotorControlOutput:
    """A bounded candidate motor command and its requested/realized wrenches."""

    rotor_references_rad_s: tuple[float, float, float, float]
    requested_wrench: Wrench
    allocated_wrench: Wrench
    desired_acceleration_w_mps2: Vec3
    position_error_w_m: Vec3
    velocity_error_w_mps: Vec3

    def to_dict(self) -> dict[str, object]:
        return {
            "rotor_references_rad_s": list(self.rotor_references_rad_s),
            "requested_wrench": list(self.requested_wrench),
            "allocated_wrench": list(self.allocated_wrench),
            "desired_acceleration_w_mps2": list(self.desired_acceleration_w_mps2),
            "position_error_w_m": list(self.position_error_w_m),
            "velocity_error_w_mps": list(self.velocity_error_w_mps),
        }


def project_asset_spec() -> QuadrotorDynamicsSpec:
    """Return the reviewed local CF2X static geometry without promoting it.

    ``cf2x.usd`` supplies the mass, inertia and rotor locations below.  Its
    USD layer does *not* supply a validated thrust curve, motor response, or
    aerodynamic winding convention.  Those provisional fields are an
    independently documented compatibility candidate, not a claim that the
    local USD is physically calibrated.  Native score eligibility remains
    false until the five CF2X gates are closed.
    """

    spec = QuadrotorDynamicsSpec(
        model_id="cf2x-local-runtime-candidate-v1",
        provenance_status="parameter_audit_pending",
        # The reviewed USD authors a 0.025 kg body and four 0.0008 kg prop
        # links.  The reported mass is the sum of those five authored masses.
        mass_kg=0.0282,
        inertia_diag_kg_m2=(1.6572e-5, 1.6656e-5, 2.9262e-5),
        rotor_joint_names=("m1_joint", "m2_joint", "m3_joint", "m4_joint"),
        rotor_positions_body_m=(
            (0.031, -0.031, 0.021),
            (-0.031, -0.031, 0.021),
            (-0.031, 0.031, 0.021),
            (0.031, 0.031, 0.021),
        ),
        # The alternating winding is a candidate allocation convention.  It
        # must be cross-checked with the native Multirotor allocation matrix
        # and a yaw-response sweep before it can be frozen.
        rotor_spin_directions=(1, -1, 1, -1),
        # Candidate values from a separately audited Crazyflie compatibility
        # profile.  They are deliberately not promoted as properties of this
        # USD asset and are never sufficient for a formal benchmark score.
        thrust_coeff_n_per_rad2=2.88e-8,
        drag_coeff_nm_per_rad2=7.24e-10,
        max_rotor_speed_rad_s=2315.0,
        motor_time_constants_s=(0.04, 0.04, 0.04, 0.04),
        motor_rate_limits_rad_s2=(100000.0, 100000.0, 100000.0, 100000.0),
    )
    spec.validate()
    return spec


def candidate_controller_spec() -> QuadrotorControllerSpec:
    """Return the shared controller candidate without granting formal status.

    These gains are deliberately isolated from benchmark method adapters.  They
    are useful for native preflight and controller regression tests, but their
    provenance must be frozen with a measured vehicle parameter manifest before
    any formal L1 score may use them.
    """

    controller = QuadrotorControllerSpec(
        provenance_status="parameter_audit_pending",
        # Conservative values for preflight diagnostics only.  The formal L1
        # executor uses the shared native thrust controller after its own
        # allocation, contact and maneuver gates have passed.
        position_gain_s2=1.6,
        velocity_gain_s=1.2,
        attitude_gain_nm_per_rad=0.0018,
        angular_rate_gain_nm_s_per_rad=0.00065,
        max_acceleration_mps2=1.5,
        max_torque_nm=0.003,
    )
    controller.validate()
    return controller


def allocation_matrix(spec: QuadrotorDynamicsSpec) -> tuple[tuple[float, ...], ...]:
    """Return the wrench-from-thrust matrix for the body-frame X quad."""

    return (
        (1.0, 1.0, 1.0, 1.0),
        tuple(position[1] for position in spec.rotor_positions_body_m),
        tuple(-position[0] for position in spec.rotor_positions_body_m),
        tuple(
            direction * spec.drag_coeff_nm_per_rad2 / spec.thrust_coeff_n_per_rad2
            for direction in spec.rotor_spin_directions
        ),
    )


def hover_rotor_speed_for_mass(spec: QuadrotorDynamicsSpec, mass_kg: float) -> float:
    """Return symmetric hover speed for an independently measured total mass.

    This supports non-formal runtime audits of a USD articulation whose
    multi-body mass may differ from a source URDF's body-only mass.  Calling
    this function does not promote the dynamics specification to formal use.
    """

    spec.validate()
    mass = _finite(mass_kg, "mass_kg")
    if mass <= 0.0:
        raise ValueError("mass_kg must be positive")
    return math.sqrt(mass * spec.gravity_mps2 / (4.0 * spec.thrust_coeff_n_per_rad2))


def rotor_wrench(spec: QuadrotorDynamicsSpec, rotor_speeds_rad_s: Iterable[float]) -> Wrench:
    """Compute body-frame total thrust and torques from four rotor speeds."""

    spec.validate()
    speeds = tuple(_finite(value, "rotor speed") for value in rotor_speeds_rad_s)
    if len(speeds) != 4:
        raise ValueError("exactly four rotor speeds are required")
    if any(speed < 0.0 or speed > spec.max_rotor_speed_rad_s + 1.0e-9 for speed in speeds):
        raise ValueError("rotor speed lies outside the frozen actuator range")
    thrusts = tuple(spec.thrust_coeff_n_per_rad2 * speed * speed for speed in speeds)
    return rotor_thrust_wrench(spec, thrusts)


def rotor_thrust_wrench(spec: QuadrotorDynamicsSpec, rotor_thrusts_n: Iterable[float]) -> Wrench:
    """Compute a body wrench from four per-rotor thrust commands in N.

    Native CF2X execution sends these four values to ``Multirotor`` as actuator
    targets.  The native Thruster model advances the four actuator states and
    ``Multirotor`` maps its applied values through the geometry-derived
    allocation matrix to a root-body PhysX wrench.  Keeping this conversion
    here lets receipts distinguish requested motor references from the thrust
    actually applied by the native actuator model.
    """

    spec.validate()
    thrusts = tuple(_finite(value, "rotor thrust") for value in rotor_thrusts_n)
    if len(thrusts) != 4:
        raise ValueError("exactly four rotor thrusts are required")
    maximum_thrust = (
        spec.thrust_coeff_n_per_rad2 * spec.max_rotor_speed_rad_s * spec.max_rotor_speed_rad_s
    )
    if any(thrust < 0.0 or thrust > maximum_thrust + 1.0e-9 for thrust in thrusts):
        raise ValueError("rotor thrust lies outside the candidate actuator range")
    matrix = allocation_matrix(spec)
    return tuple(sum(row[index] * thrusts[index] for index in range(4)) for row in matrix)  # type: ignore[return-value]


def motor_step(
    spec: QuadrotorDynamicsSpec,
    current_rad_s: Iterable[float],
    reference_rad_s: Iterable[float],
    *,
    dt_s: float | None = None,
) -> tuple[float, float, float, float]:
    """Advance four first-order motor states with rate and speed saturation."""

    spec.validate()
    current = tuple(_finite(value, "current motor speed") for value in current_rad_s)
    reference = tuple(_finite(value, "reference motor speed") for value in reference_rad_s)
    if len(current) != 4 or len(reference) != 4:
        raise ValueError("motor states and references must contain four values")
    dt = spec.physics_dt_s if dt_s is None else _finite(dt_s, "dt_s")
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")
    output: list[float] = []
    for index, (now, target) in enumerate(zip(current, reference, strict=True)):
        now = min(spec.max_rotor_speed_rad_s, max(0.0, now))
        target = min(spec.max_rotor_speed_rad_s, max(0.0, target))
        raw_rate = (target - now) / spec.motor_time_constants_s[index]
        max_rate = spec.motor_rate_limits_rad_s2[index]
        rate = min(max_rate, max(-max_rate, raw_rate))
        # ``physics_dt_s`` can be much larger than a motor time constant.
        # Clamp the Euler update to the interval between the old state and the
        # reference, otherwise a rate-limited step can overshoot the target
        # and alternate between two motor states on successive PhysX frames.
        proposed = now + dt * rate
        if target >= now:
            next_speed = min(target, proposed)
        else:
            next_speed = max(target, proposed)
        output.append(min(spec.max_rotor_speed_rad_s, max(0.0, next_speed)))
    return tuple(output)  # type: ignore[return-value]


def controller_step(
    dynamics: QuadrotorDynamicsSpec,
    controller: QuadrotorControllerSpec,
    state: FlightState,
    command: FlightCommand,
    *,
    mass_kg: float | None = None,
) -> RotorControlOutput:
    """Map a public flight command to a bounded four-rotor reference.

    The caller supplies a measured PhysX state.  ``mass_kg`` exists for the
    current audit period because the USD articulation mass differs from the
    candidate URDF source mass.  Supplying a runtime mass does not promote the
    dynamics or controller contract to formal use.
    """

    dynamics.validate()
    controller.validate()
    state.validate()
    command.validate()
    mass = dynamics.mass_kg if mass_kg is None else _finite(mass_kg, "mass_kg")
    if mass <= 0.0:
        raise ValueError("mass_kg must be positive")

    position_error = tuple(
        target - actual
        for target, actual in zip(command.target_position_w_m, state.position_w_m, strict=True)
    )
    velocity_error = tuple(
        target - actual
        for target, actual in zip(
            command.target_velocity_w_mps, state.linear_velocity_w_mps, strict=True
        )
    )
    desired_acceleration = _clamp_vector_norm(
        tuple(
            controller.position_gain_s2 * position
            + controller.velocity_gain_s * velocity
            for position, velocity in zip(position_error, velocity_error, strict=True)
        ),
        controller.max_acceleration_mps2,
    )
    requested_force_w = (
        mass * desired_acceleration[0],
        mass * desired_acceleration[1],
        mass * (desired_acceleration[2] + dynamics.gravity_mps2),
    )
    desired_roll, desired_pitch, desired_yaw = _desired_euler_from_force(
        requested_force_w, command.target_yaw_rad
    )
    current_roll, current_pitch, current_yaw = _euler_from_quaternion(
        state.orientation_wxyz
    )
    angular_velocity_b = _world_to_body(
        state.angular_velocity_w_rad_s, state.orientation_wxyz
    )
    attitude_error = (
        _wrap_angle_rad(desired_roll - current_roll),
        _wrap_angle_rad(desired_pitch - current_pitch),
        _wrap_angle_rad(desired_yaw - current_yaw),
    )
    angular_rate_error = (
        -angular_velocity_b[0],
        -angular_velocity_b[1],
        command.target_yaw_rate_rad_s - angular_velocity_b[2],
    )
    requested_torques = tuple(
        _clamp(
            controller.attitude_gain_nm_per_rad * attitude
            + controller.angular_rate_gain_nm_s_per_rad * rate,
            -controller.max_torque_nm,
            controller.max_torque_nm,
        )
        for attitude, rate in zip(attitude_error, angular_rate_error, strict=True)
    )
    current_b3_w = _body_to_world((0.0, 0.0, 1.0), state.orientation_wxyz)
    requested_thrust = _clamp(
        _dot(requested_force_w, current_b3_w),
        0.0,
        4.0
        * dynamics.thrust_coeff_n_per_rad2
        * dynamics.max_rotor_speed_rad_s
        * dynamics.max_rotor_speed_rad_s,
    )
    requested_wrench: Wrench = (requested_thrust, *requested_torques)
    rotor_thrusts = _solve_linear4(allocation_matrix(dynamics), requested_wrench)
    maximum_thrust = (
        dynamics.thrust_coeff_n_per_rad2
        * dynamics.max_rotor_speed_rad_s
        * dynamics.max_rotor_speed_rad_s
    )
    rotor_references = tuple(
        math.sqrt(
            _clamp(thrust, 0.0, maximum_thrust) / dynamics.thrust_coeff_n_per_rad2
        )
        for thrust in rotor_thrusts
    )
    allocated_wrench = rotor_wrench(dynamics, rotor_references)
    return RotorControlOutput(
        rotor_references_rad_s=rotor_references,
        requested_wrench=requested_wrench,
        allocated_wrench=allocated_wrench,
        desired_acceleration_w_mps2=desired_acceleration,
        position_error_w_m=position_error,
        velocity_error_w_mps=velocity_error,
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _clamp_vector_norm(vector: Vec3, maximum_norm: float) -> Vec3:
    maximum = _finite(maximum_norm, "maximum_norm")
    if maximum <= 0.0:
        raise ValueError("maximum_norm must be positive")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= maximum or norm <= 1.0e-12:
        return vector
    scale = maximum / norm
    return tuple(scale * value for value in vector)  # type: ignore[return-value]


def _dot(first: Vec3, second: Vec3) -> float:
    return sum(left * right for left, right in zip(first, second, strict=True))


def _cross(first: Vec3, second: Vec3) -> Vec3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _normalize(vector: Vec3, name: str) -> Vec3:
    norm = math.sqrt(_dot(vector, vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must be non-zero")
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


def _wrap_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _rotation_matrix_from_quaternion(quaternion: Quaternion) -> tuple[Vec3, Vec3, Vec3]:
    if len(quaternion) != 4:
        raise ValueError("quaternion must contain four values")
    w, x, y, z = (_finite(value, "quaternion") for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _body_to_world(vector_b: Vec3, orientation_wxyz: Quaternion) -> Vec3:
    matrix = _rotation_matrix_from_quaternion(orientation_wxyz)
    return tuple(
        sum(matrix[row][column] * vector_b[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _world_to_body(vector_w: Vec3, orientation_wxyz: Quaternion) -> Vec3:
    matrix = _rotation_matrix_from_quaternion(orientation_wxyz)
    return tuple(
        sum(matrix[row][column] * vector_w[row] for row in range(3))
        for column in range(3)
    )  # type: ignore[return-value]


def _euler_from_quaternion(orientation_wxyz: Quaternion) -> tuple[float, float, float]:
    matrix = _rotation_matrix_from_quaternion(orientation_wxyz)
    pitch = math.asin(_clamp(-matrix[2][0], -1.0, 1.0))
    roll = math.atan2(matrix[2][1], matrix[2][2])
    yaw = math.atan2(matrix[1][0], matrix[0][0])
    return roll, pitch, yaw


def _desired_euler_from_force(force_w: Vec3, yaw_rad: float) -> tuple[float, float, float]:
    b3 = _normalize(force_w, "requested_force_w")
    yaw = _finite(yaw_rad, "yaw_rad")
    heading = (math.cos(yaw), math.sin(yaw), 0.0)
    b2 = _normalize(_cross(b3, heading), "requested_force_w and yaw heading")
    b1 = _cross(b2, b3)
    # Rotation columns are b1, b2, b3.  This is the ZYX Euler extraction
    # convention used by the measured IsaacLab wxyz orientation.
    pitch = math.asin(_clamp(-b1[2], -1.0, 1.0))
    roll = math.atan2(b2[2], b3[2])
    desired_yaw = math.atan2(b1[1], b1[0])
    return roll, pitch, desired_yaw


def _solve_linear4(
    matrix: tuple[tuple[float, ...], ...], right_hand_side: Wrench
) -> tuple[float, float, float, float]:
    values = [
        [float(matrix[row][column]) for column in range(4)] + [float(right_hand_side[row])]
        for row in range(4)
    ]
    for pivot in range(4):
        candidate = max(range(pivot, 4), key=lambda row: abs(values[row][pivot]))
        if abs(values[candidate][pivot]) <= 1.0e-12:
            raise ValueError("allocation matrix is singular")
        if candidate != pivot:
            values[pivot], values[candidate] = values[candidate], values[pivot]
        pivot_value = values[pivot][pivot]
        for column in range(pivot, 5):
            values[pivot][column] /= pivot_value
        for row in range(4):
            if row == pivot:
                continue
            factor = values[row][pivot]
            for column in range(pivot, 5):
                values[row][column] -= factor * values[pivot][column]
    return tuple(values[row][4] for row in range(4))  # type: ignore[return-value]


def _determinant4(matrix: tuple[tuple[float, ...], ...]) -> float:
    values = [list(row) for row in matrix]
    determinant = 1.0
    for pivot in range(4):
        candidate = max(range(pivot, 4), key=lambda row: abs(values[row][pivot]))
        if abs(values[candidate][pivot]) <= 1.0e-15:
            return 0.0
        if candidate != pivot:
            values[pivot], values[candidate] = values[candidate], values[pivot]
            determinant *= -1.0
        pivot_value = values[pivot][pivot]
        determinant *= pivot_value
        for row in range(pivot + 1, 4):
            factor = values[row][pivot] / pivot_value
            for column in range(pivot + 1, 4):
                values[row][column] -= factor * values[pivot][column]
    return determinant
