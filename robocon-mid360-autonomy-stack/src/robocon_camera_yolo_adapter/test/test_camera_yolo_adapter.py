import time

import pytest

from robocon_camera_yolo_adapter.camera_yolo_adapter import (
    normalize_legacy_candidate,
    observation_to_json,
    select_best_legacy_candidate,
)


def test_normalize_legacy_detector_mm_output():
    now_ns = time.time_ns()
    observation = normalize_legacy_candidate(
        {
            "class": "basket",
            "confidence": 0.86,
            "distance": 3200.0,
            "position": (320, 240),
            "bbox": (100, 120, 540, 420),
            "is_stable": True,
            "timestamp": now_ns / 1_000_000_000.0,
        },
        now_ns=now_ns,
    )
    assert observation is not None
    assert observation.distance_m == pytest.approx(3.2)
    assert observation.center_x == 320
    assert observation.stable


def test_select_best_candidate_ignores_person_and_unknown_class():
    observation = select_best_legacy_candidate(
        [
            {"class": "person", "confidence": 0.99, "distance": 2000, "stable": True},
            {"class": "unknown", "confidence": 0.98, "distance": 2000, "stable": True},
            {"class": "basket", "confidence": 0.70, "distance": 3000, "stable": False},
            {"class": "basket", "confidence": 0.80, "distance": 3100, "stable": True},
        ],
        now_ns=time.time_ns(),
    )
    assert observation is not None
    assert observation.target_type == "basket"
    assert observation.confidence == pytest.approx(0.80)


def test_observation_serialization_preserves_contract_and_evidence():
    observation = normalize_legacy_candidate(
        {
            "class": "hoop",
            "confidence": 0.9,
            "distance_mm": 2500,
            "stable": True,
            "observed_at_ns": 123,
            "position": (4, 5),
            "bbox": (1, 2, 3, 4),
        },
        now_ns=456,
    )
    assert observation is not None
    payload = observation_to_json(observation, evidence_level="camera_yolo_runtime")
    assert '"distance_m":2.5' in payload
    assert '"evidence_level":"camera_yolo_runtime"' in payload
    assert '"bbox":[1,2,3,4]' in payload
