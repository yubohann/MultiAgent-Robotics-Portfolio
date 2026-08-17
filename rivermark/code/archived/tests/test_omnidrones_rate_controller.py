from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.omnidrones_rate_controller import (
    OmniDronesRateControllerError,
    OmniDronesRateControllerProfile,
    compute_omnidrones_rate_controller,
    decode_bounded_omnidrones_rate_action,
    omnidrones_crazyflie_snapshot_profile,
)

FIXTURE = ROOT / "tests" / "fixtures" / "omnidrones_rate_controller_reference_v1.json"


def test_reference_math_matches_outputs_generated_by_upstream_rate_controller() -> None:
    reference = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert (
        reference["schema"] == "org.rivermark.omnidrones-rate-controller-reference.v1"
    )
    profile = OmniDronesRateControllerProfile.from_mapping(reference["profile"])
    assert profile.as_dict() == omnidrones_crazyflie_snapshot_profile().as_dict()

    for case in reference["cases"]:
        result = compute_omnidrones_rate_controller(
            profile,
            quaternion_wxyz=case["quaternion_wxyz"],
            world_angular_velocity_radps=case["world_angular_velocity_radps"],
            target_body_rate_radps=case["target_body_rate_radps"],
            target_collective_thrust_n=case["target_collective_thrust_n"],
        )
        np.testing.assert_allclose(
            result.normalized_rotor_command,
            np.asarray(case["normalized_rotor_command"], dtype=np.float64),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        assert np.all(result.clipped_rotor_thrust_n >= 0.0)
        assert np.all(result.clipped_rotor_thrust_n <= profile.max_thrust_per_rotor_n)


def test_rate_action_has_explicit_bounded_semantics() -> None:
    profile = omnidrones_crazyflie_snapshot_profile()
    target_rate, collective = decode_bounded_omnidrones_rate_action(
        profile, np.asarray([[-1.0, 0.0, 1.0, 0.0]])
    )
    np.testing.assert_allclose(target_rate, np.asarray([[-np.pi, 0.0, np.pi]]))
    np.testing.assert_allclose(
        collective,
        np.asarray([[np.sum(profile.max_thrust_per_rotor_n) / 2.0]]),
    )
    with pytest.raises(OmniDronesRateControllerError, match="closed interval"):
        decode_bounded_omnidrones_rate_action(
            profile, np.asarray([[0.0, 0.0, 0.0, 1.01]])
        )


def test_rate_controller_rejects_nonunit_or_misaligned_inputs() -> None:
    profile = omnidrones_crazyflie_snapshot_profile()
    kwargs = {
        "quaternion_wxyz": np.asarray([[2.0, 0.0, 0.0, 0.0]]),
        "world_angular_velocity_radps": np.zeros((1, 3)),
        "target_body_rate_radps": np.zeros((1, 3)),
        "target_collective_thrust_n": np.asarray((0.2,)),
    }
    with pytest.raises(OmniDronesRateControllerError, match="unit-length"):
        compute_omnidrones_rate_controller(profile, **kwargs)
    kwargs["quaternion_wxyz"] = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    kwargs["world_angular_velocity_radps"] = np.zeros((2, 3))
    with pytest.raises(OmniDronesRateControllerError, match="same batch size"):
        compute_omnidrones_rate_controller(profile, **kwargs)
