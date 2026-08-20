import pytest

from robocon_camera_yolo_adapter.metrics import evaluate_records, iou


def test_iou_and_class_metrics_are_deterministic():
    assert iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175
    result = evaluate_records([
        {
            "evidence_level": "bag_replay",
            "ground_truth": [{"class": "hoop", "bbox": [0, 0, 10, 10], "depth_m": 3.0}],
            "predictions": [{
                "class": "hoop", "bbox": [0, 0, 10, 10], "confidence": 0.9,
                "depth_m": 3.1, "valid_depth_ratio": 1.0, "inference_ms": 12.0,
            }],
        },
        {
            "evidence_level": "bag_replay",
            "ground_truth": [{"class": "hoop", "bbox": [0, 0, 10, 10]}],
            "predictions": [],
        },
    ])
    assert result["per_class"]["hoop"]["tp"] == 1
    assert result["per_class"]["hoop"]["fn"] == 1
    assert result["per_class"]["hoop"]["precision"] == 1.0
    assert result["per_class"]["hoop"]["recall"] == 0.5
    assert result["depth_abs_error_m_p50"] == pytest.approx(0.1)
    assert result["evidence_levels"] == ["bag_replay"]


def test_confidence_threshold_excludes_low_confidence_prediction():
    result = evaluate_records([
        {
            "evidence_level": "synthetic",
            "ground_truth": [],
            "predictions": [{"class": "hoop", "bbox": [0, 0, 2, 2], "confidence": 0.2}],
        },
    ], confidence_threshold=0.5)
    assert result["per_class"] == {}
