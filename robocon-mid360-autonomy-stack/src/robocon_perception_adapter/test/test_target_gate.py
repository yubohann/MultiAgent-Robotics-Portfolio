import time

import pytest

from robocon_perception_adapter.target_gate import (
    parse_observation,
    validate_observation,
)
from robocon_perception_adapter.synthetic_target_source import normalize_synthetic_truth


def test_camera_example_distance_mm_is_normalized():
    observation = parse_observation({
        "confidence": 0.9,
        "distance_mm": 3000,
        "stable": True,
        "observed_at_ns": 100,
        "target_type": "hoop",
    })
    assert observation.distance_m == 3.0


def test_target_gate_rejects_low_confidence_and_stale_observations():
    observation = parse_observation({
        "confidence": 0.5,
        "distance_m": 3.0,
        "stable": True,
        "observed_at_ns": 1,
    })
    valid, reason = validate_observation(
        observation,
        now_ns=time.time_ns(),
        min_confidence=0.7,
        min_distance_m=1.0,
        max_distance_m=10.0,
        max_age_sec=0.5,
        require_stable=True,
    )
    assert not valid
    assert reason == "target_observation_stale"


def test_target_gate_accepts_fresh_stable_observation():
    now = time.time_ns()
    observation = parse_observation({
        "confidence": 0.9,
        "distance_m": 3.0,
        "stable": True,
        "observed_at_ns": now,
    })
    valid, reason = validate_observation(
        observation,
        now_ns=now + 10_000_000,
        min_confidence=0.7,
        min_distance_m=1.0,
        max_distance_m=10.0,
        max_age_sec=0.5,
        require_stable=True,
    )
    assert valid
    assert reason == "target_valid"


def test_synthetic_target_truth_requires_explicit_provenance():
    observation, source = normalize_synthetic_truth({
        "confidence": 0.9,
        "distance_m": 3.0,
        "stable": True,
        "observed_at_ns": 123,
        "target_type": "hoop",
        "evidence_level": "synthetic",
        "source": "gazebo_model_state",
    })
    assert observation.target_type == "hoop"
    assert observation.evidence_level == "synthetic"
    assert observation.evidence_source == "gazebo_model_state"
    assert source == "gazebo_model_state"

    with pytest.raises(ValueError, match="evidence_level"):
        normalize_synthetic_truth({
            "confidence": 0.9,
            "distance_m": 3.0,
            "stable": True,
            "observed_at_ns": 123,
            "source": "gazebo_model_state",
        })
