"""Development-only SB3 state-only control transfer for Isaac City-Lite.

This module deliberately has no Isaac, Torch, Gymnasium, or Stable-Baselines3
imports at module import time.  An Isaac caller copies the four public rigid
body arrays to CPU NumPy arrays, then this module derives the exact 8-D state
used by the local state-only pilot and maps one authenticated SB3 action back
to a bounded world-frame velocity/yaw command.

It is a control-wiring pilot, not Isaac training, a formal benchmark method,
or a dataset admission path.  It never accepts image, semantic, lidar, radar,
language, evaluator, reward, target, or seed inputs.
"""

from __future__ import annotations

import importlib.metadata
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import numpy as np

from .citylite_scene import AGENT_COUNT, PUBLIC_ROUTES_W_M


if TYPE_CHECKING:
    from .methods import StableBaselines3CheckpointPolicy


STATE_ONLY_PROFILE = "state_only"
SB3_ADAPTER_V2_SCHEMA = "org.rivermark.sb3-adapter.v2"
STATE_ONLY_PROPRIOCEPTION_ABI = "org.rivermark.state-only-proprioception.v1"
STATE_ONLY_VELOCITY_ACTION_ABI = "org.rivermark.state-only-velocity-yaw-action.v1"
CITYLITE_ROUTE_ANCHOR_HEADING_TO_PILOT_V1 = (
    "citylite_route_anchor_heading_to_pilot_v1"
)
TRANSFER_SCHEMA = "org.rivermark.isaac-sb3-state-transfer.v1"
TRANSFER_SOURCE = "sb3_state_only_isaac_transfer_development"
PILOT_BASE_ORIGIN_M = (1.75, 2.0, 3.05)
STATE_FIELDS = (
    "position_x_m",
    "position_y_m",
    "position_z_m",
    "velocity_x_mps",
    "velocity_y_mps",
    "velocity_z_mps",
    "yaw_rad",
    "yaw_rate_radps",
)
ACTION_FIELDS = (
    "velocity_x_mps",
    "velocity_y_mps",
    "velocity_z_mps",
    "yaw_rate_radps",
)
EXCLUDED_POLICY_INPUTS = (
    "rgb",
    "depth",
    "semantic_segmentation",
    "lidar",
    "radar",
    "imu",
    "language",
    "public_task_state",
    "public_team_messages",
    "high_level_action_history",
    "evaluator_private",
    "private_target_truth",
    "reward",
    "seed",
)
REQUIRED_RUNTIME_VERSION_KEYS = ("python", "numpy", "gymnasium", "stable_baselines3")
DEVELOPMENT_CLAIM_BOUNDARY = "development_state_only_control_wiring_smoke_only"


class StateOnlyTransferError(ValueError):
    """Raised when a state, action, policy, or cadence contract is invalid."""


def _finite_array(value: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Copy one exact CPU numeric array without importing a tensor runtime."""

    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise StateOnlyTransferError(f"{label} must be a finite numeric array") from exc
    if result.shape != shape:
        raise StateOnlyTransferError(f"{label} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise StateOnlyTransferError(f"{label} must contain only finite values")
    return result.copy()


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _wrap_angle(angle_rad: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _canonical_agent_ids(agent_ids: Iterable[int] | None, count: int) -> tuple[int, ...]:
    if count < 1:
        raise StateOnlyTransferError("at least one agent state is required")
    if agent_ids is None:
        return tuple(range(count))
    values = tuple(agent_ids)
    if len(values) != count:
        raise StateOnlyTransferError("agent_ids length must match state rows")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise StateOnlyTransferError("agent_ids must be integer row identifiers")
        normalized.append(int(value))
    if tuple(normalized) != tuple(range(count)):
        raise StateOnlyTransferError(
            "agent_ids must be canonical row order [0, ..., N-1]; reorder arrays before transfer"
        )
    return tuple(normalized)


def quaternion_wxyz_to_yaw(quaternion_wxyz: Any) -> np.ndarray:
    """Return world yaw in ``[-pi, pi)`` from finite, nonzero WXYZ quaternions.

    IsaacLab exposes root orientation as WXYZ.  The implementation normalizes
    each input first so an otherwise valid scaled quaternion has the same
    orientation, but rejects zero and non-finite quaternions.
    """

    try:
        quaternions = np.asarray(quaternion_wxyz, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise StateOnlyTransferError("quaternion_wxyz must be a finite [N,4] array") from exc
    if quaternions.ndim != 2 or quaternions.shape[1:] != (4,):
        raise StateOnlyTransferError(
            f"quaternion_wxyz must have shape [N, 4], got {quaternions.shape}"
        )
    if quaternions.shape[0] < 1 or not np.all(np.isfinite(quaternions)):
        raise StateOnlyTransferError("quaternion_wxyz must contain finite values")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms <= 1.0e-12):
        raise StateOnlyTransferError("quaternion_wxyz contains a zero-norm quaternion")
    q = quaternions / norms[:, None]
    w, x, y, z = (q[:, index] for index in range(4))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray(_wrap_angle(yaw), dtype=np.float64)


@dataclass(frozen=True)
class PhysicalState8D:
    """Exact public rigid-body state in stable Isaac fleet row order."""

    agent_ids: tuple[int, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = _finite_array(
            self.values,
            shape=(len(self.agent_ids), 8),
            label="physical state",
        )
        object.__setattr__(self, "agent_ids", _canonical_agent_ids(self.agent_ids, len(self.agent_ids)))
        object.__setattr__(self, "values", _readonly(values))

    def copy_values(self) -> np.ndarray:
        return self.values.copy()


def derive_physical_state_8d(
    position_w_m: Any,
    linear_velocity_w_mps: Any,
    quaternion_wxyz: Any,
    angular_velocity_b_radps: Any,
    *,
    agent_ids: Iterable[int] | None = None,
) -> PhysicalState8D:
    """Derive ``[world xyz, world velocity xyz, yaw, body yaw rate]``.

    Inputs are strictly batched CPU arrays with row ``i`` bound to Isaac agent
    ``i``.  No sensor or evaluator object can enter this function's ABI.
    """

    try:
        count = int(np.asarray(position_w_m).shape[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise StateOnlyTransferError("position_w_m must be a [N,3] array") from exc
    if count < 1:
        raise StateOnlyTransferError("position_w_m must contain at least one row")
    positions = _finite_array(position_w_m, shape=(count, 3), label="position_w_m")
    velocity = _finite_array(
        linear_velocity_w_mps,
        shape=(count, 3),
        label="linear_velocity_w_mps",
    )
    quaternion = _finite_array(
        quaternion_wxyz,
        shape=(count, 4),
        label="quaternion_wxyz",
    )
    angular_velocity = _finite_array(
        angular_velocity_b_radps,
        shape=(count, 3),
        label="angular_velocity_b_radps",
    )
    yaw = quaternion_wxyz_to_yaw(quaternion)
    values = np.concatenate(
        (positions, velocity, yaw[:, None], angular_velocity[:, 2:3]), axis=1
    )
    return PhysicalState8D(_canonical_agent_ids(agent_ids, count), values)


def _route_anchor_headings() -> tuple[tuple[float, float, float], tuple[float, ...]]:
    anchors: list[tuple[float, float, float]] = []
    headings: list[float] = []
    for route in PUBLIC_ROUTES_W_M:
        anchors.append(tuple(float(component) for component in route[0]))
        for start, end in zip(route, route[1:]):
            dx = float(end[0]) - float(start[0])
            dy = float(end[1]) - float(start[1])
            if math.hypot(dx, dy) > 1.0e-9:
                headings.append(math.atan2(dy, dx))
                break
        else:
            raise RuntimeError("every City-Lite public route needs a horizontal heading")
    return tuple(anchors), tuple(headings)


@dataclass(frozen=True)
class CityLiteRouteAnchorTransform:
    """Public route-anchor transform between City-Lite and pilot coordinates."""

    anchors_w_m: np.ndarray
    initial_route_heading_rad: np.ndarray
    pilot_base_origin_m: np.ndarray

    def __post_init__(self) -> None:
        anchors = _finite_array(
            self.anchors_w_m,
            shape=(AGENT_COUNT, 3),
            label="anchors_w_m",
        )
        headings = _finite_array(
            self.initial_route_heading_rad,
            shape=(AGENT_COUNT,),
            label="initial_route_heading_rad",
        )
        base = _finite_array(
            self.pilot_base_origin_m,
            shape=(3,),
            label="pilot_base_origin_m",
        )
        object.__setattr__(self, "anchors_w_m", _readonly(anchors))
        object.__setattr__(self, "initial_route_heading_rad", _readonly(headings))
        object.__setattr__(self, "pilot_base_origin_m", _readonly(base))

    @classmethod
    def from_public_routes(cls) -> "CityLiteRouteAnchorTransform":
        anchors, headings = _route_anchor_headings()
        return cls(
            anchors_w_m=np.asarray(anchors, dtype=np.float64),
            initial_route_heading_rad=np.asarray(headings, dtype=np.float64),
            pilot_base_origin_m=np.asarray(PILOT_BASE_ORIGIN_M, dtype=np.float64),
        )

    def physical_to_pilot(self, physical_state: PhysicalState8D) -> PhysicalState8D:
        """Rotate each physical state about its public anchor into pilot space."""

        if physical_state.agent_ids != tuple(range(AGENT_COUNT)):
            raise StateOnlyTransferError("City-Lite transfer requires all eight canonical agent rows")
        raw = physical_state.values
        heading = self.initial_route_heading_rad
        cosine = np.cos(heading)
        sine = np.sin(heading)
        position_offset = raw[:, :3] - self.anchors_w_m
        local_position = position_offset.copy()
        # R(-heading) applied to world-frame position/velocity XY.
        local_position[:, 0] = cosine * position_offset[:, 0] + sine * position_offset[:, 1]
        local_position[:, 1] = -sine * position_offset[:, 0] + cosine * position_offset[:, 1]
        local_position += self.pilot_base_origin_m
        local_velocity = raw[:, 3:6].copy()
        local_velocity[:, 0] = cosine * raw[:, 3] + sine * raw[:, 4]
        local_velocity[:, 1] = -sine * raw[:, 3] + cosine * raw[:, 4]
        pilot = np.concatenate(
            (
                local_position,
                local_velocity,
                np.asarray(_wrap_angle(raw[:, 6] - heading))[:, None],
                raw[:, 7:8],
            ),
            axis=1,
        )
        return PhysicalState8D(physical_state.agent_ids, pilot)

    def pilot_velocity_to_world(
        self, local_velocity_mps: Any, *, agent_ids: Iterable[int] | None = None
    ) -> np.ndarray:
        """Apply ``R(+heading)`` to pilot local velocity commands."""

        velocity = _finite_array(
            local_velocity_mps,
            shape=(AGENT_COUNT, 3),
            label="local_velocity_mps",
        )
        ids = _canonical_agent_ids(agent_ids, AGENT_COUNT)
        if ids != tuple(range(AGENT_COUNT)):
            raise StateOnlyTransferError("City-Lite transfer requires canonical agent rows")
        cosine = np.cos(self.initial_route_heading_rad)
        sine = np.sin(self.initial_route_heading_rad)
        world = velocity.copy()
        world[:, 0] = cosine * velocity[:, 0] - sine * velocity[:, 1]
        world[:, 1] = sine * velocity[:, 0] + cosine * velocity[:, 1]
        return _readonly(world)

    def provenance(self) -> dict[str, Any]:
        return {
            "coordinate_contract": CITYLITE_ROUTE_ANCHOR_HEADING_TO_PILOT_V1,
            "agent_count": AGENT_COUNT,
            "anchors_w_m": self.anchors_w_m.tolist(),
            "initial_route_heading_rad": self.initial_route_heading_rad.tolist(),
            "pilot_base_origin_m": self.pilot_base_origin_m.tolist(),
            "forward": "pilot_position=base+R(-heading)*(world_position-anchor)",
            "velocity_inverse": "world_velocity=R(+heading)*pilot_velocity",
        }


@dataclass(frozen=True)
class WorldCommandBounds:
    """Physical-command limits applied after local-to-world rotation."""

    max_horizontal_speed_mps: float = 2.3
    max_vertical_speed_mps: float = 1.25
    max_yaw_rate_rad_s: float = 1.4

    def __post_init__(self) -> None:
        for field_name in (
            "max_horizontal_speed_mps",
            "max_vertical_speed_mps",
            "max_yaw_rate_rad_s",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise StateOnlyTransferError(f"{field_name} must be finite and positive")
            object.__setattr__(self, field_name, value)

    def apply(self, world_velocity_mps: Any, yaw_rate_rad_s: Any) -> tuple[np.ndarray, np.ndarray]:
        velocity = _finite_array(
            world_velocity_mps,
            shape=(AGENT_COUNT, 3),
            label="world_velocity_mps",
        )
        yaw_rate = _finite_array(
            yaw_rate_rad_s,
            shape=(AGENT_COUNT,),
            label="yaw_rate_rad_s",
        )
        horizontal_norm = np.linalg.norm(velocity[:, :2], axis=1)
        scale = np.minimum(1.0, self.max_horizontal_speed_mps / np.maximum(horizontal_norm, 1.0e-12))
        velocity[:, :2] *= scale[:, None]
        velocity[:, 2] = np.clip(
            velocity[:, 2], -self.max_vertical_speed_mps, self.max_vertical_speed_mps
        )
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate_rad_s, self.max_yaw_rate_rad_s)
        return _readonly(velocity), _readonly(yaw_rate)


@dataclass(frozen=True)
class FixedDecisionCadence:
    """Integer physics-step cadence with no accumulated floating-point drift."""

    every_physics_steps: int

    def __post_init__(self) -> None:
        if isinstance(self.every_physics_steps, bool) or not isinstance(
            self.every_physics_steps, (int, np.integer)
        ) or int(self.every_physics_steps) < 1:
            raise StateOnlyTransferError("every_physics_steps must be a positive integer")
        object.__setattr__(self, "every_physics_steps", int(self.every_physics_steps))

    def is_due(self, physics_step: int) -> bool:
        if isinstance(physics_step, bool) or not isinstance(physics_step, (int, np.integer)):
            raise StateOnlyTransferError("physics_step must be an integer")
        if int(physics_step) < 0:
            raise StateOnlyTransferError("physics_step must be nonnegative")
        return int(physics_step) % self.every_physics_steps == 0

    def decision_index(self, physics_step: int) -> int:
        if not self.is_due(physics_step):
            raise StateOnlyTransferError(
                f"physics_step {physics_step} is not on the fixed decision cadence"
            )
        return int(physics_step) // self.every_physics_steps


@dataclass(frozen=True)
class StateOnlyTransferDecision:
    """Complete raw-to-emitted action evidence for one decision tick."""

    physics_step: int
    decision_index: int
    physical_state_8d: np.ndarray
    pilot_state_8d: np.ndarray
    normalized_observation_8d: np.ndarray
    raw_action: np.ndarray
    normalized_action: np.ndarray
    local_velocity_yaw_command: np.ndarray
    prebound_world_velocity_yaw_command: np.ndarray
    emitted_world_velocity_yaw_command: np.ndarray

    def __post_init__(self) -> None:
        for field_name, shape in (
            ("physical_state_8d", (AGENT_COUNT, 8)),
            ("pilot_state_8d", (AGENT_COUNT, 8)),
            ("normalized_observation_8d", (AGENT_COUNT, 8)),
            ("raw_action", (AGENT_COUNT, 4)),
            ("normalized_action", (AGENT_COUNT, 4)),
            ("local_velocity_yaw_command", (AGENT_COUNT, 4)),
            ("prebound_world_velocity_yaw_command", (AGENT_COUNT, 4)),
            ("emitted_world_velocity_yaw_command", (AGENT_COUNT, 4)),
        ):
            object.__setattr__(
                self,
                field_name,
                _readonly(_finite_array(getattr(self, field_name), shape=shape, label=field_name)),
            )

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": TRANSFER_SCHEMA,
            "claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
            "information_profile": STATE_ONLY_PROFILE,
            "policy_input_fields": list(STATE_FIELDS),
            "excluded_policy_inputs": list(EXCLUDED_POLICY_INPUTS),
            "action_source": TRANSFER_SOURCE,
            "physics_step": self.physics_step,
            "decision_index": self.decision_index,
            "raw_action": self.raw_action.tolist(),
            "normalized_action": self.normalized_action.tolist(),
            "emitted_world_velocity_yaw_command": self.emitted_world_velocity_yaw_command.tolist(),
        }


def _metadata_vector(metadata: Mapping[str, Any], key: str, *, shape: tuple[int, ...]) -> np.ndarray:
    try:
        vector = _finite_array(metadata.get(key), shape=shape, label=f"SB3 metadata {key}")
    except StateOnlyTransferError as exc:
        raise StateOnlyTransferError(str(exc)) from exc
    return vector


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _require_matching_runtime_versions(runtime_versions: Mapping[str, Any]) -> None:
    """Fail closed when the v2 checkpoint's declared runtime is not current."""

    if not isinstance(runtime_versions, Mapping) or any(
        not isinstance(runtime_versions.get(key), str) or not runtime_versions[key].strip()
        for key in REQUIRED_RUNTIME_VERSION_KEYS
    ):
        raise StateOnlyTransferError("SB3 checkpoint requires complete runtime version metadata")
    actual_versions = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "gymnasium": importlib.metadata.version("gymnasium"),
        "stable_baselines3": importlib.metadata.version("stable-baselines3"),
    }
    mismatches = [
        key
        for key, actual in actual_versions.items()
        if str(runtime_versions[key]) != actual
    ]
    if mismatches:
        raise StateOnlyTransferError(
            "SB3 checkpoint runtime version metadata does not match this process: "
            + ", ".join(mismatches)
        )


def _require_policy_hash_provenance(
    policy: "StableBaselines3CheckpointPolicy", metadata: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Require fresh checkpoint and sidecar hashes from the loaded adapter."""

    try:
        provenance = policy.provenance()
    except Exception as exc:
        raise StateOnlyTransferError("unable to read fresh SB3 checkpoint provenance") from exc
    if not isinstance(provenance, Mapping):
        raise StateOnlyTransferError("SB3 checkpoint provenance must be an object")
    checkpoint_hash = provenance.get("checkpoint_sha256")
    metadata_hash = provenance.get("adapter_metadata_sha256")
    if not _is_sha256(checkpoint_hash) or not _is_sha256(metadata_hash):
        raise StateOnlyTransferError(
            "SB3 policy provenance requires valid checkpoint and metadata SHA-256 hashes"
        )
    if metadata.get("checkpoint_sha256") != checkpoint_hash:
        raise StateOnlyTransferError(
            "SB3 checkpoint metadata commitment does not match loaded checkpoint provenance"
        )
    return provenance


def validate_state_only_sb3_policy(
    policy: "StableBaselines3CheckpointPolicy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a real v2 SB3 policy before it can command Isaac.

    The nominal adapter validates the checkpoint hash and loads the actual SB3
    PPO/SAC object.  This stricter transfer gate additionally checks that its
    exact local state/action ABI and coordinate contract are appropriate for
    the development-only City-Lite bridge.
    """

    from .methods import StableBaselines3CheckpointPolicy

    if not isinstance(policy, StableBaselines3CheckpointPolicy):
        raise StateOnlyTransferError(
            "Isaac transfer accepts only a real StableBaselines3CheckpointPolicy"
        )
    metadata = getattr(policy, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise StateOnlyTransferError("SB3 checkpoint policy has no validated metadata")
    if metadata.get("schema") != SB3_ADAPTER_V2_SCHEMA:
        raise StateOnlyTransferError("Isaac transfer requires SB3 adapter metadata v2")
    if metadata.get("information_profile") != STATE_ONLY_PROFILE:
        raise StateOnlyTransferError("Isaac transfer accepts only state_only SB3 policies")
    if metadata.get("implementation_kind") != "trained_sb3_pilot_checkpoint":
        raise StateOnlyTransferError("Isaac transfer requires a trained local SB3 pilot checkpoint")
    if metadata.get("training_backend") != "rivermark-kinematic-pilot-v1":
        raise StateOnlyTransferError("SB3 checkpoint training backend is not the declared pilot backend")
    if metadata.get("formal_benchmark_admission") is not False:
        raise StateOnlyTransferError("SB3 transfer metadata must explicitly deny formal benchmark admission")
    checkpoint_sha256 = metadata.get("checkpoint_sha256")
    if not _is_sha256(checkpoint_sha256):
        raise StateOnlyTransferError("SB3 checkpoint requires a lowercase SHA-256 commitment")
    runtime_versions = metadata.get("runtime_versions")
    _require_matching_runtime_versions(runtime_versions)
    _require_policy_hash_provenance(policy, metadata)
    transfer = metadata.get("isaac_control_transfer")
    if not isinstance(transfer, Mapping) or transfer.get("eligible") is not True:
        raise StateOnlyTransferError("SB3 checkpoint is not marked eligible for development transfer")
    if transfer.get("coordinate_contract") != CITYLITE_ROUTE_ANCHOR_HEADING_TO_PILOT_V1:
        raise StateOnlyTransferError("SB3 checkpoint coordinate transfer contract is not City-Lite v1")
    if transfer.get("physical_training") is not False or transfer.get("isaac_training") is not False:
        raise StateOnlyTransferError("SB3 transfer metadata must not claim physical or Isaac training")
    if transfer.get("claim_boundary") != DEVELOPMENT_CLAIM_BOUNDARY:
        raise StateOnlyTransferError("SB3 transfer metadata claim boundary is not development-only")
    observation_abi = metadata.get("observation_abi")
    if not isinstance(observation_abi, Mapping) or observation_abi.get("schema") != STATE_ONLY_PROPRIOCEPTION_ABI:
        raise StateOnlyTransferError("SB3 checkpoint lacks the required 8D state-only observation ABI")
    if (
        observation_abi.get("shape") != [8]
        or observation_abi.get("fields") != list(STATE_FIELDS)
        or observation_abi.get("coordinate_frame") != "pilot_world_right_handed_z_up"
    ):
        raise StateOnlyTransferError("SB3 checkpoint observation ABI fields/order do not match Isaac transfer")
    action_abi = metadata.get("action_abi")
    if not isinstance(action_abi, Mapping) or action_abi.get("schema") != STATE_ONLY_VELOCITY_ACTION_ABI:
        raise StateOnlyTransferError("SB3 checkpoint lacks the required velocity/yaw action ABI")
    if (
        action_abi.get("shape") != [4]
        or action_abi.get("fields") != list(ACTION_FIELDS)
        or action_abi.get("normalized_range") != [-1.0, 1.0]
        or action_abi.get("frame") != "pilot_world"
    ):
        raise StateOnlyTransferError("SB3 checkpoint action ABI fields/order do not match Isaac transfer")
    mean = _metadata_vector(metadata, "observation_mean", shape=(8,))
    standard_deviation = _metadata_vector(metadata, "observation_std", shape=(8,))
    action_scale = _metadata_vector(metadata, "action_scale", shape=(4,))
    if np.any(standard_deviation <= 0.0) or np.any(action_scale <= 0.0):
        raise StateOnlyTransferError("SB3 normalization std and action scale must be positive")
    model = getattr(policy, "model", None)
    if not callable(getattr(model, "predict", None)):
        raise StateOnlyTransferError("SB3 checkpoint policy does not expose a loaded model.predict")
    observation_space = getattr(model, "observation_space", None)
    action_space = getattr(model, "action_space", None)
    if tuple(getattr(observation_space, "shape", ())) != (8,):
        raise StateOnlyTransferError("loaded SB3 model observation space must be exactly [8]")
    if tuple(getattr(action_space, "shape", ())) != (4,):
        raise StateOnlyTransferError("loaded SB3 model action space must be exactly [4]")
    return _readonly(mean), _readonly(standard_deviation), _readonly(action_scale)


class StateOnlySB3IsaacTransfer:
    """Run an authenticated state-only SB3 policy on all eight CF2X rows."""

    def __init__(
        self,
        policy: "StableBaselines3CheckpointPolicy",
        *,
        cadence: FixedDecisionCadence,
        transform: CityLiteRouteAnchorTransform | None = None,
        bounds: WorldCommandBounds | None = None,
    ) -> None:
        if not isinstance(cadence, FixedDecisionCadence):
            raise StateOnlyTransferError("cadence must be a FixedDecisionCadence")
        self._mean, self._std, self._action_scale = validate_state_only_sb3_policy(policy)
        self._policy = policy
        self.cadence = cadence
        self.transform = transform or CityLiteRouteAnchorTransform.from_public_routes()
        if not isinstance(self.transform, CityLiteRouteAnchorTransform):
            raise StateOnlyTransferError("transform must be CityLiteRouteAnchorTransform")
        self.bounds = bounds or WorldCommandBounds()
        if not isinstance(self.bounds, WorldCommandBounds):
            raise StateOnlyTransferError("bounds must be WorldCommandBounds")

    @property
    def policy(self) -> "StableBaselines3CheckpointPolicy":
        return self._policy

    def decide(
        self,
        physics_step: int,
        position_w_m: Any,
        linear_velocity_w_mps: Any,
        quaternion_wxyz: Any,
        angular_velocity_b_radps: Any,
    ) -> StateOnlyTransferDecision:
        """Make exactly one due decision from public rigid-body state arrays."""

        decision_index = self.cadence.decision_index(physics_step)
        physical = derive_physical_state_8d(
            position_w_m,
            linear_velocity_w_mps,
            quaternion_wxyz,
            angular_velocity_b_radps,
            agent_ids=range(AGENT_COUNT),
        )
        if physical.values.shape != (AGENT_COUNT, 8):
            raise StateOnlyTransferError("City-Lite SB3 transfer requires exactly eight CF2X rows")
        pilot = self.transform.physical_to_pilot(physical)
        normalized_observation = (pilot.values - self._mean) / self._std
        try:
            raw_action, _ = self._policy.model.predict(
                normalized_observation.astype(np.float32, copy=False), deterministic=True
            )
        except Exception as exc:
            raise StateOnlyTransferError("SB3 model.predict failed during state-only transfer") from exc
        raw = _finite_array(raw_action, shape=(AGENT_COUNT, 4), label="SB3 raw action")
        normalized_action = np.clip(raw, -1.0, 1.0)
        local_command = normalized_action * self._action_scale
        prebound_world_velocity = self.transform.pilot_velocity_to_world(local_command[:, :3])
        bounded_velocity, bounded_yaw_rate = self.bounds.apply(
            prebound_world_velocity, local_command[:, 3]
        )
        prebound = np.concatenate((prebound_world_velocity, local_command[:, 3:4]), axis=1)
        emitted = np.concatenate((bounded_velocity, bounded_yaw_rate[:, None]), axis=1)
        return StateOnlyTransferDecision(
            physics_step=int(physics_step),
            decision_index=decision_index,
            physical_state_8d=physical.values,
            pilot_state_8d=pilot.values,
            normalized_observation_8d=normalized_observation,
            raw_action=raw,
            normalized_action=normalized_action,
            local_velocity_yaw_command=local_command,
            prebound_world_velocity_yaw_command=prebound,
            emitted_world_velocity_yaw_command=emitted,
        )

    def provenance(self) -> dict[str, Any]:
        policy_provenance = self._policy.provenance()
        return {
            "schema": TRANSFER_SCHEMA,
            "claim_boundary": DEVELOPMENT_CLAIM_BOUNDARY,
            "formal_benchmark_admission": False,
            "physical_training": False,
            "isaac_training": False,
            "information_profile": STATE_ONLY_PROFILE,
            "policy_input_fields": list(STATE_FIELDS),
            "excluded_policy_inputs": list(EXCLUDED_POLICY_INPUTS),
            # These values come from the authenticated v2 adapter metadata.
            # Persisting them lets a pure-Python verifier replay the public
            # state-to-command ABI without loading SB3, Isaac, or sensors.
            "observation_mean": self._mean.tolist(),
            "observation_std": self._std.tolist(),
            "action_scale": self._action_scale.tolist(),
            "action_source": TRANSFER_SOURCE,
            "decision_cadence_physics_steps": self.cadence.every_physics_steps,
            "coordinate_transform": self.transform.provenance(),
            "world_command_bounds": {
                "max_horizontal_speed_mps": self.bounds.max_horizontal_speed_mps,
                "max_vertical_speed_mps": self.bounds.max_vertical_speed_mps,
                "max_yaw_rate_rad_s": self.bounds.max_yaw_rate_rad_s,
            },
            "policy": policy_provenance,
        }


def create_state_only_sb3_isaac_transfer(
    checkpoint: Path,
    metadata_path: Path | None = None,
    *,
    cadence: FixedDecisionCadence,
    transform: CityLiteRouteAnchorTransform | None = None,
    bounds: WorldCommandBounds | None = None,
) -> StateOnlySB3IsaacTransfer:
    """Load the real provenance-checked SB3 checkpoint and build this bridge."""

    from .methods import create_sb3_checkpoint_policy

    policy = create_sb3_checkpoint_policy(checkpoint, metadata_path)
    return StateOnlySB3IsaacTransfer(
        policy,
        cadence=cadence,
        transform=transform,
        bounds=bounds,
    )
