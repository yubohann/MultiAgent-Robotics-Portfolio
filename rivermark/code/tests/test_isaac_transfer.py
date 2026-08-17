from __future__ import annotations

import inspect
import math
import platform
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.citylite_scene import AGENT_COUNT
from rivermark_benchmark.isaac_transfer import (
    ACTION_FIELDS,
    CITYLITE_ROUTE_ANCHOR_HEADING_TO_PILOT_V1,
    EXCLUDED_POLICY_INPUTS,
    STATE_FIELDS,
    CityLiteRouteAnchorTransform,
    FixedDecisionCadence,
    SB3_ADAPTER_V2_SCHEMA,
    STATE_ONLY_PROPRIOCEPTION_ABI,
    STATE_ONLY_VELOCITY_ACTION_ABI,
    StateOnlySB3IsaacTransfer,
    StateOnlyTransferError,
    WorldCommandBounds,
    derive_physical_state_8d,
    quaternion_wxyz_to_yaw,
    validate_state_only_sb3_policy,
)
from rivermark_benchmark.methods import StableBaselines3CheckpointPolicy


class _Space:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _Model:
    def __init__(self, action: np.ndarray, *, observation_shape: tuple[int, ...] = (8,), action_shape: tuple[int, ...] = (4,)) -> None:
        self.action = np.asarray(action, dtype=np.float64)
        self.observation_space = _Space(observation_shape)
        self.action_space = _Space(action_shape)
        self.calls: list[tuple[np.ndarray, bool]] = []

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        self.calls.append((np.asarray(observation).copy(), deterministic))
        return self.action.copy(), None


def _metadata() -> dict[str, object]:
    return {
        "schema": SB3_ADAPTER_V2_SCHEMA,
        "implementation_kind": "trained_sb3_pilot_checkpoint",
        "algorithm": "ppo",
        "information_profile": "state_only",
        "training_backend": "rivermark-kinematic-pilot-v1",
        "observation_mean": [0.0] * 8,
        "observation_std": [1.0] * 8,
        "action_scale": [2.3, 2.3, 1.25, 1.4],
        "observation_abi": {
            "schema": STATE_ONLY_PROPRIOCEPTION_ABI,
            "shape": [8],
            "fields": list(STATE_FIELDS),
            "coordinate_frame": "pilot_world_right_handed_z_up",
        },
        "action_abi": {
            "schema": STATE_ONLY_VELOCITY_ACTION_ABI,
            "shape": [4],
            "fields": list(ACTION_FIELDS),
            "normalized_range": [-1.0, 1.0],
            "frame": "pilot_world",
        },
        "isaac_control_transfer": {
            "eligible": True,
            "coordinate_contract": CITYLITE_ROUTE_ANCHOR_HEADING_TO_PILOT_V1,
            "physical_training": False,
            "isaac_training": False,
            "claim_boundary": "development_state_only_control_wiring_smoke_only",
        },
        "formal_benchmark_admission": False,
        "checkpoint_sha256": "a" * 64,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": "test",
            "gymnasium": "test",
            "stable_baselines3": "test",
        },
    }


def _policy(action: np.ndarray, *, metadata: dict[str, object] | None = None, model: _Model | None = None) -> StableBaselines3CheckpointPolicy:
    """Construct a type-valid stand-in after the real loader's hash gate.

    The production factory owns SB3 loading and SHA validation.  This focused
    NumPy test only needs a loaded policy object to exercise the pure bridge.
    """

    policy = object.__new__(StableBaselines3CheckpointPolicy)
    policy.metadata = _metadata() if metadata is None else metadata
    policy.model = _Model(action) if model is None else model
    policy.provenance = lambda: {
        "checkpoint_sha256": "a" * 64,
        "adapter_metadata_sha256": "b" * 64,
        "external_dependency": "stable_baselines3",
    }
    return policy


def _initial_physical_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transform = CityLiteRouteAnchorTransform.from_public_routes()
    positions = transform.anchors_w_m.copy()
    velocity = np.zeros((AGENT_COUNT, 3), dtype=np.float64)
    yaw = transform.initial_route_heading_rad
    quaternion = np.column_stack(
        (np.cos(yaw / 2.0), np.zeros(AGENT_COUNT), np.zeros(AGENT_COUNT), np.sin(yaw / 2.0))
    )
    angular_velocity = np.zeros((AGENT_COUNT, 3), dtype=np.float64)
    return positions, velocity, quaternion, angular_velocity


class IsaacStateOnlyTransferTests(unittest.TestCase):
    def _matching_runtime_versions(self):
        return patch(
            "rivermark_benchmark.isaac_transfer.importlib.metadata.version",
            side_effect=lambda distribution: {
                "numpy": "test",
                "gymnasium": "test",
                "stable-baselines3": "test",
            }[distribution],
        )

    def test_state_order_uses_wxyz_yaw_and_body_yaw_rate(self) -> None:
        positions = np.array(((1.0, 2.0, 3.0), (-4.0, 5.0, 6.0)))
        velocity = np.array(((0.1, 0.2, 0.3), (-0.4, 0.5, -0.6)))
        half = math.pi / 4.0
        quaternion = np.array(
            ((math.cos(half), 0.0, 0.0, math.sin(half)), (1.0, 0.0, 0.0, 0.0))
        )
        angular_velocity = np.array(((9.0, 8.0, 0.7), (6.0, 5.0, -0.8)))

        state = derive_physical_state_8d(positions, velocity, quaternion, angular_velocity)

        np.testing.assert_allclose(
            state.values,
            np.array(
                ((1.0, 2.0, 3.0, 0.1, 0.2, 0.3, math.pi / 2.0, 0.7), (-4.0, 5.0, 6.0, -0.4, 0.5, -0.6, 0.0, -0.8))
            ),
            atol=1.0e-12,
        )
        self.assertEqual(state.agent_ids, (0, 1))
        self.assertFalse(state.values.flags.writeable)

    def test_state_rejects_nonfinite_wrong_shape_zero_quaternion_and_reordered_ids(self) -> None:
        positions, velocity, quaternion, angular_velocity = _initial_physical_state()
        with self.assertRaisesRegex(StateOnlyTransferError, "shape"):
            derive_physical_state_8d(positions[:, :2], velocity, quaternion, angular_velocity)
        invalid = velocity.copy()
        invalid[0, 1] = math.nan
        with self.assertRaisesRegex(StateOnlyTransferError, "finite"):
            derive_physical_state_8d(positions, invalid, quaternion, angular_velocity)
        zero = quaternion.copy()
        zero[0] = 0.0
        with self.assertRaisesRegex(StateOnlyTransferError, "zero-norm"):
            derive_physical_state_8d(positions, velocity, zero, angular_velocity)
        with self.assertRaisesRegex(StateOnlyTransferError, "canonical row order"):
            derive_physical_state_8d(
                positions, velocity, quaternion, angular_velocity, agent_ids=reversed(range(AGENT_COUNT))
            )

    def test_route_anchor_heading_transform_and_inverse_velocity(self) -> None:
        transform = CityLiteRouteAnchorTransform.from_public_routes()
        positions, _, quaternion, angular_velocity = _initial_physical_state()
        headings = transform.initial_route_heading_rad
        velocity = np.column_stack((np.cos(headings), np.sin(headings), np.full(AGENT_COUNT, 0.2)))
        physical = derive_physical_state_8d(positions, velocity, quaternion, angular_velocity)

        pilot = transform.physical_to_pilot(physical)
        expected_position = np.broadcast_to(transform.pilot_base_origin_m, (AGENT_COUNT, 3))
        np.testing.assert_allclose(pilot.values[:, :3], expected_position)
        np.testing.assert_allclose(pilot.values[:, 3:6], np.column_stack((np.ones(AGENT_COUNT), np.zeros(AGENT_COUNT), np.full(AGENT_COUNT, 0.2))))
        np.testing.assert_allclose(pilot.values[:, 6], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(transform.pilot_velocity_to_world(pilot.values[:, 3:6]), velocity)

    def test_fixed_integer_cadence_is_deterministic(self) -> None:
        cadence = FixedDecisionCadence(3)
        self.assertEqual([step for step in range(10) if cadence.is_due(step)], [0, 3, 6, 9])
        self.assertEqual(cadence.decision_index(6), 2)
        with self.assertRaisesRegex(StateOnlyTransferError, "not on the fixed decision cadence"):
            cadence.decision_index(7)
        with self.assertRaises(StateOnlyTransferError):
            FixedDecisionCadence(0)
        with self.assertRaises(StateOnlyTransferError):
            cadence.is_due(-1)

    def test_bridge_preserves_raw_normalized_and_bounded_world_action_provenance(self) -> None:
        raw = np.tile(np.array((2.0, -2.0, 0.9, 4.0)), (AGENT_COUNT, 1))
        with self._matching_runtime_versions():
            bridge = StateOnlySB3IsaacTransfer(
                _policy(raw),
                cadence=FixedDecisionCadence(2),
                bounds=WorldCommandBounds(
                    max_horizontal_speed_mps=1.0,
                    max_vertical_speed_mps=0.5,
                    max_yaw_rate_rad_s=0.3,
                ),
            )
        positions, velocity, quaternion, angular_velocity = _initial_physical_state()
        decision = bridge.decide(2, positions, velocity, quaternion, angular_velocity)

        np.testing.assert_allclose(decision.raw_action, raw)
        np.testing.assert_allclose(decision.normalized_action, np.tile((1.0, -1.0, 0.9, 1.0), (AGENT_COUNT, 1)))
        self.assertTrue(np.all(np.linalg.norm(decision.emitted_world_velocity_yaw_command[:, :2], axis=1) <= 1.0 + 1.0e-12))
        self.assertTrue(np.all(np.abs(decision.emitted_world_velocity_yaw_command[:, 2]) <= 0.5 + 1.0e-12))
        self.assertTrue(np.all(np.abs(decision.emitted_world_velocity_yaw_command[:, 3]) <= 0.3 + 1.0e-12))
        self.assertFalse(np.allclose(decision.prebound_world_velocity_yaw_command, decision.emitted_world_velocity_yaw_command))
        evidence = decision.provenance()
        self.assertEqual(evidence["information_profile"], "state_only")
        self.assertEqual(evidence["policy_input_fields"], list(STATE_FIELDS))
        self.assertEqual(evidence["excluded_policy_inputs"], list(EXCLUDED_POLICY_INPUTS))
        self.assertIn("raw_action", evidence)
        self.assertFalse(decision.raw_action.flags.writeable)

    def test_bridge_has_no_sensor_or_source_injection_surface(self) -> None:
        signature = inspect.signature(StateOnlySB3IsaacTransfer.decide)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "self",
                "physics_step",
                "position_w_m",
                "linear_velocity_w_mps",
                "quaternion_wxyz",
                "angular_velocity_b_radps",
            ),
        )
        with self._matching_runtime_versions():
            bridge = StateOnlySB3IsaacTransfer(_policy(np.zeros((AGENT_COUNT, 4))), cadence=FixedDecisionCadence(1))
        positions, velocity, quaternion, angular_velocity = _initial_physical_state()
        with self.assertRaises(TypeError):
            bridge.decide(
                0,
                positions,
                velocity,
                quaternion,
                angular_velocity,
                semantic_segmentation=np.zeros((1, 1)),
            )
        receipt = bridge.provenance()
        self.assertEqual(receipt["formal_benchmark_admission"], False)
        self.assertEqual(receipt["physical_training"], False)
        self.assertEqual(receipt["isaac_training"], False)

    def test_policy_gate_rejects_stale_or_wrong_profile_abi_and_model_spaces(self) -> None:
        action = np.zeros((AGENT_COUNT, 4))
        metadata = _metadata()
        metadata["schema"] = "org.rivermark.sb3-adapter.v1"
        with self._matching_runtime_versions(), self.assertRaisesRegex(StateOnlyTransferError, "v2"):
            validate_state_only_sb3_policy(_policy(action, metadata=metadata))

        metadata = _metadata()
        metadata["information_profile"] = "egocentric_rgb_state"
        with self._matching_runtime_versions(), self.assertRaisesRegex(StateOnlyTransferError, "state_only"):
            validate_state_only_sb3_policy(_policy(action, metadata=metadata))

        metadata = _metadata()
        metadata["isaac_control_transfer"] = {"eligible": False}
        with self._matching_runtime_versions(), self.assertRaisesRegex(StateOnlyTransferError, "eligible"):
            validate_state_only_sb3_policy(_policy(action, metadata=metadata))

        metadata = _metadata()
        metadata["action_abi"] = {"shape": [4]}
        with self._matching_runtime_versions(), self.assertRaisesRegex(StateOnlyTransferError, "action ABI"):
            validate_state_only_sb3_policy(_policy(action, metadata=metadata))

        model = _Model(action, observation_shape=(7,))
        with self._matching_runtime_versions(), self.assertRaisesRegex(StateOnlyTransferError, "observation space"):
            validate_state_only_sb3_policy(_policy(action, model=model))

        metadata = _metadata()
        metadata["runtime_versions"] = {"python": "test"}
        with self._matching_runtime_versions(), self.assertRaisesRegex(StateOnlyTransferError, "runtime version"):
            validate_state_only_sb3_policy(_policy(action, metadata=metadata))

        metadata = _metadata()
        metadata["runtime_versions"]["numpy"] = "stale"
        with self._matching_runtime_versions(), self.assertRaisesRegex(StateOnlyTransferError, "does not match"):
            validate_state_only_sb3_policy(_policy(action, metadata=metadata))

        metadata = _metadata()
        metadata["formal_benchmark_admission"] = True
        with self._matching_runtime_versions(), self.assertRaisesRegex(StateOnlyTransferError, "formal benchmark"):
            validate_state_only_sb3_policy(_policy(action, metadata=metadata))

        policy = _policy(action)
        policy.provenance = lambda: {"checkpoint_sha256": "a" * 64}
        with self._matching_runtime_versions(), self.assertRaisesRegex(StateOnlyTransferError, "metadata SHA-256"):
            validate_state_only_sb3_policy(policy)

        policy = _policy(action)
        policy.provenance = lambda: {
            "checkpoint_sha256": "c" * 64,
            "adapter_metadata_sha256": "b" * 64,
        }
        with self._matching_runtime_versions(), self.assertRaisesRegex(StateOnlyTransferError, "does not match"):
            validate_state_only_sb3_policy(policy)

    def test_bridge_rejects_nondue_tick_and_nonfinite_model_action(self) -> None:
        positions, velocity, quaternion, angular_velocity = _initial_physical_state()
        with self._matching_runtime_versions():
            bridge = StateOnlySB3IsaacTransfer(_policy(np.zeros((AGENT_COUNT, 4))), cadence=FixedDecisionCadence(2))
        with self.assertRaisesRegex(StateOnlyTransferError, "not on the fixed decision cadence"):
            bridge.decide(1, positions, velocity, quaternion, angular_velocity)
        raw = np.zeros((AGENT_COUNT, 4))
        raw[3, 2] = math.inf
        with self._matching_runtime_versions():
            bridge = StateOnlySB3IsaacTransfer(_policy(raw), cadence=FixedDecisionCadence(1))
        with self.assertRaisesRegex(StateOnlyTransferError, "finite"):
            bridge.decide(0, positions, velocity, quaternion, angular_velocity)

    def test_quaternion_yaw_rejects_xyzw_style_shape_and_wraps_pi(self) -> None:
        yaw = quaternion_wxyz_to_yaw(np.array(((0.0, 0.0, 0.0, 1.0),)))
        np.testing.assert_allclose(yaw, np.array((-math.pi,)))
        with self.assertRaises(StateOnlyTransferError):
            quaternion_wxyz_to_yaw(np.zeros((4,)))


if __name__ == "__main__":
    unittest.main()
