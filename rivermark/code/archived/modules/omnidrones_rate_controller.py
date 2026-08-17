"""CPU reference for the audited OmniDrones rate-controller math.

This module is intentionally not an Isaac controller and is not imported by
the City-Lite capture path.  It gives a dependency-light way to test the
mathematics and units used by the upstream Crazyflie rate-controller example
before an explicitly calibrated T2 policy adapter is considered.

The upstream controller consumes world-frame angular velocity in a 13D state,
rotates it into the body frame, and returns normalized rotor commands.  The
Rivermark runtime instead accepts physical per-rotor thrust targets in newtons.
Both representations are returned so an integration cannot silently exchange
one for the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

OMNIDRONES_RATE_CONTROLLER_SCHEMA = "org.rivermark.omnidrones-rate-controller.v1"
OMNIDRONES_REFERENCE_ACTION_ABI = "omnidrones-normalized-body-rate-collective-thrust.v1"


class OmniDronesRateControllerError(ValueError):
    """Raised when a rate-controller input cannot be interpreted safely."""


def _finite_array(value: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise OmniDronesRateControllerError(
            f"{label} must be a finite numeric array"
        ) from exc
    if result.shape != shape:
        raise OmniDronesRateControllerError(
            f"{label} must have shape {shape}, got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise OmniDronesRateControllerError(f"{label} must contain only finite values")
    return result.copy()


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _batch_array(value: Any, *, width: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise OmniDronesRateControllerError(
            f"{label} must be a finite numeric array"
        ) from exc
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] != width:
        raise OmniDronesRateControllerError(
            f"{label} must have shape [N, {width}], got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise OmniDronesRateControllerError(f"{label} must contain only finite values")
    return result.copy()


def _quaternion_to_rotation_matrix_wxyz(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Match OmniDrones' WXYZ quaternion-to-world-rotation implementation."""

    w, x, y, z = (quaternion_wxyz[:, index] for index in range(4))
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    return np.stack(
        (
            1.0 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1.0 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1.0 - (txx + tyy),
        ),
        axis=-1,
    ).reshape((-1, 3, 3))


def _body_rate_from_world_wxyz(
    quaternion_wxyz: np.ndarray, world_angular_velocity_radps: np.ndarray
) -> np.ndarray:
    """Rotate world angular velocity into the body frame using WXYZ quaternions."""

    rotation = _quaternion_to_rotation_matrix_wxyz(quaternion_wxyz)
    return np.einsum("nji,nj->ni", rotation, world_angular_velocity_radps)


@dataclass(frozen=True)
class OmniDronesRateControllerProfile:
    """Explicit physical parameters needed by the upstream rate-controller math.

    This profile is deliberately a caller-owned calibration object.  The
    snapshot profile below represents OmniDrones' own Crazyflie YAML, not the
    active Rivermark CF2X asset.  A future native adapter must build a separate
    profile from a locked Isaac/USD measurement before it can execute.
    """

    rotor_angles_rad: np.ndarray
    arm_lengths_m: np.ndarray
    directions: np.ndarray
    force_constants_n_per_radps2: np.ndarray
    moment_constants_nm_per_radps2: np.ndarray
    max_rotation_velocities_radps: np.ndarray
    inertia_kg_m2: np.ndarray

    def __post_init__(self) -> None:
        values = {
            "rotor_angles_rad": _finite_array(
                self.rotor_angles_rad, shape=(4,), label="rotor_angles_rad"
            ),
            "arm_lengths_m": _finite_array(
                self.arm_lengths_m, shape=(4,), label="arm_lengths_m"
            ),
            "directions": _finite_array(
                self.directions, shape=(4,), label="directions"
            ),
            "force_constants_n_per_radps2": _finite_array(
                self.force_constants_n_per_radps2,
                shape=(4,),
                label="force_constants_n_per_radps2",
            ),
            "moment_constants_nm_per_radps2": _finite_array(
                self.moment_constants_nm_per_radps2,
                shape=(4,),
                label="moment_constants_nm_per_radps2",
            ),
            "max_rotation_velocities_radps": _finite_array(
                self.max_rotation_velocities_radps,
                shape=(4,),
                label="max_rotation_velocities_radps",
            ),
            "inertia_kg_m2": _finite_array(
                self.inertia_kg_m2, shape=(3,), label="inertia_kg_m2"
            ),
        }
        if np.any(values["arm_lengths_m"] <= 0.0):
            raise OmniDronesRateControllerError("arm_lengths_m must be positive")
        if np.any(values["force_constants_n_per_radps2"] <= 0.0):
            raise OmniDronesRateControllerError("force constants must be positive")
        if np.any(values["moment_constants_nm_per_radps2"] <= 0.0):
            raise OmniDronesRateControllerError("moment constants must be positive")
        if np.any(values["max_rotation_velocities_radps"] <= 0.0):
            raise OmniDronesRateControllerError(
                "maximum rotation velocities must be positive"
            )
        if np.any(values["inertia_kg_m2"] <= 0.0):
            raise OmniDronesRateControllerError("inertia values must be positive")
        if not np.all(np.isin(values["directions"], (-1.0, 1.0))):
            raise OmniDronesRateControllerError("directions must contain only -1 or 1")
        for field, value in values.items():
            object.__setattr__(self, field, _readonly(value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OmniDronesRateControllerProfile:
        required = (
            "rotor_angles_rad",
            "arm_lengths_m",
            "directions",
            "force_constants_n_per_radps2",
            "moment_constants_nm_per_radps2",
            "max_rotation_velocities_radps",
            "inertia_kg_m2",
        )
        missing = [field for field in required if field not in value]
        if missing:
            raise OmniDronesRateControllerError(f"profile is missing fields: {missing}")
        return cls(**{field: value[field] for field in required})

    @property
    def max_thrust_per_rotor_n(self) -> np.ndarray:
        return _readonly(
            np.square(self.max_rotation_velocities_radps)
            * self.force_constants_n_per_radps2
        )

    @property
    def mixer(self) -> np.ndarray:
        """Return OmniDrones' torque/collective-to-rotor-force mixing matrix."""

        allocation = np.stack(
            (
                np.sin(self.rotor_angles_rad) * self.arm_lengths_m,
                -np.cos(self.rotor_angles_rad) * self.arm_lengths_m,
                -self.directions
                * self.moment_constants_nm_per_radps2
                / self.force_constants_n_per_radps2,
                np.ones(4, dtype=np.float64),
            )
        )
        inertia = np.diag(np.concatenate((self.inertia_kg_m2, (1.0,))))
        try:
            result = allocation.T @ np.linalg.inv(allocation @ allocation.T) @ inertia
        except np.linalg.LinAlgError as exc:
            raise OmniDronesRateControllerError("rotor mixer is singular") from exc
        if not np.all(np.isfinite(result)):
            raise OmniDronesRateControllerError("rotor mixer is not finite")
        return _readonly(result)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "rotor_angles_rad": self.rotor_angles_rad.tolist(),
            "arm_lengths_m": self.arm_lengths_m.tolist(),
            "directions": self.directions.tolist(),
            "force_constants_n_per_radps2": self.force_constants_n_per_radps2.tolist(),
            "moment_constants_nm_per_radps2": self.moment_constants_nm_per_radps2.tolist(),
            "max_rotation_velocities_radps": self.max_rotation_velocities_radps.tolist(),
            "inertia_kg_m2": self.inertia_kg_m2.tolist(),
        }


@dataclass(frozen=True)
class OmniDronesRateControllerOutput:
    """Exact normalized command plus the rotor-force interpretation in newtons."""

    normalized_rotor_command: np.ndarray
    requested_rotor_thrust_n: np.ndarray
    clipped_rotor_thrust_n: np.ndarray
    body_angular_velocity_radps: np.ndarray

    def __post_init__(self) -> None:
        count = int(np.asarray(self.normalized_rotor_command).shape[0])
        if count < 1:
            raise OmniDronesRateControllerError(
                "controller output must contain at least one row"
            )
        object.__setattr__(
            self,
            "normalized_rotor_command",
            _readonly(
                _finite_array(
                    self.normalized_rotor_command,
                    shape=(count, 4),
                    label="normalized_rotor_command",
                )
            ),
        )
        object.__setattr__(
            self,
            "requested_rotor_thrust_n",
            _readonly(
                _finite_array(
                    self.requested_rotor_thrust_n,
                    shape=(count, 4),
                    label="requested_rotor_thrust_n",
                )
            ),
        )
        object.__setattr__(
            self,
            "clipped_rotor_thrust_n",
            _readonly(
                _finite_array(
                    self.clipped_rotor_thrust_n,
                    shape=(count, 4),
                    label="clipped_rotor_thrust_n",
                )
            ),
        )
        object.__setattr__(
            self,
            "body_angular_velocity_radps",
            _readonly(
                _finite_array(
                    self.body_angular_velocity_radps,
                    shape=(count, 3),
                    label="body_angular_velocity_radps",
                )
            ),
        )


def omnidrones_crazyflie_snapshot_profile() -> OmniDronesRateControllerProfile:
    """Return the reviewed values in OmniDrones' ``crazyflie.yaml`` snapshot."""

    return OmniDronesRateControllerProfile(
        rotor_angles_rad=np.asarray((0.78539816, 2.35619449, 3.92699082, 5.49778714)),
        arm_lengths_m=np.full(4, 0.043),
        directions=np.asarray((-1.0, 1.0, -1.0, 1.0)),
        force_constants_n_per_radps2=np.full(4, 2.88e-8),
        moment_constants_nm_per_radps2=np.full(4, 7.24e-10),
        max_rotation_velocities_radps=np.full(4, 2315.0),
        inertia_kg_m2=np.asarray((1.4e-5, 1.4e-5, 2.17e-5)),
    )


def compute_omnidrones_rate_controller(
    profile: OmniDronesRateControllerProfile,
    *,
    quaternion_wxyz: Any,
    world_angular_velocity_radps: Any,
    target_body_rate_radps: Any,
    target_collective_thrust_n: Any,
) -> OmniDronesRateControllerOutput:
    """Reproduce OmniDrones ``RateController.forward`` with NumPy.

    ``target_collective_thrust_n`` is a total force, not per-rotor force.  The
    unbounded normalized output has the exact upstream meaning.  The clipped
    force is included only to show the command a physical actuator can accept;
    callers must not substitute it for the upstream normalized action without
    recording that policy/runtime boundary.
    """

    quaternion = _batch_array(quaternion_wxyz, width=4, label="quaternion_wxyz")
    count = quaternion.shape[0]
    norm = np.linalg.norm(quaternion, axis=1)
    if not np.all(np.abs(norm - 1.0) <= 1.0e-6):
        raise OmniDronesRateControllerError("quaternion_wxyz must be unit-length WXYZ")
    angular_velocity = _batch_array(
        world_angular_velocity_radps,
        width=3,
        label="world_angular_velocity_radps",
    )
    target_rate = _batch_array(
        target_body_rate_radps, width=3, label="target_body_rate_radps"
    )
    if angular_velocity.shape[0] != count or target_rate.shape[0] != count:
        raise OmniDronesRateControllerError(
            "quaternion, world angular velocity, and target rate must have the same batch size"
        )
    try:
        collective = np.asarray(target_collective_thrust_n, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise OmniDronesRateControllerError(
            "target_collective_thrust_n must be finite"
        ) from exc
    if collective.shape == (count,):
        collective = collective[:, None]
    if collective.shape != (count, 1) or not np.all(np.isfinite(collective)):
        raise OmniDronesRateControllerError(
            f"target_collective_thrust_n must have shape [{count}, 1] or [{count}]"
        )
    if np.any(collective < 0.0):
        raise OmniDronesRateControllerError(
            "target_collective_thrust_n must be non-negative"
        )

    body_rate = _body_rate_from_world_wxyz(quaternion, angular_velocity)
    gain = np.asarray((0.52, 0.52, 0.025), dtype=np.float64) / profile.inertia_kg_m2
    # The audited upstream implementation adds cross(angular_velocity,
    # angular_velocity), which is algebraically zero.  Preserve that exact
    # behavior rather than silently changing it to a rigid-body gyroscopic term.
    angular_acceleration = -(body_rate - target_rate) * gain
    requested = (
        profile.mixer @ np.concatenate((angular_acceleration, collective), axis=1).T
    ).T
    max_thrust = profile.max_thrust_per_rotor_n
    normalized = requested / max_thrust[None, :] * 2.0 - 1.0
    clipped = np.clip(requested, 0.0, max_thrust[None, :])
    return OmniDronesRateControllerOutput(normalized, requested, clipped, body_rate)


def decode_bounded_omnidrones_rate_action(
    profile: OmniDronesRateControllerProfile, action: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Map a bounded four-value policy action to body-rate and collective force.

    OmniDrones' transform maps the first three values to ``[-pi, pi]`` rad/s
    and the last value to ``[0, sum(max thrust)]`` N.  The upstream transform's
    action spec is unbounded, but a Rivermark adapter refuses values outside
    ``[-1, 1]`` rather than relying on a downstream rotor clamp.
    """

    values = _batch_array(action, width=4, label="action")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise OmniDronesRateControllerError(
            "action must stay in the closed interval [-1, 1]"
        )
    target_rate = values[:, :3] * np.pi
    collective = ((values[:, 3:4] + 1.0) / 2.0) * np.sum(profile.max_thrust_per_rotor_n)
    return _readonly(target_rate), _readonly(collective)


__all__ = [
    "OMNIDRONES_RATE_CONTROLLER_SCHEMA",
    "OMNIDRONES_REFERENCE_ACTION_ABI",
    "OmniDronesRateControllerError",
    "OmniDronesRateControllerOutput",
    "OmniDronesRateControllerProfile",
    "compute_omnidrones_rate_controller",
    "decode_bounded_omnidrones_rate_action",
    "omnidrones_crazyflie_snapshot_profile",
]
