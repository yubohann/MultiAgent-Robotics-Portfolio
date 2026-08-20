"""Deterministic offline metrics for recorded target detections.

The evaluator deliberately consumes an explicit ground-truth stream.  It does
not infer labels from images and therefore cannot turn synthetic or missing
truth into a detector-quality claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must contain four coordinates")
    coordinates = tuple(float(item) for item in value)
    x1, y1, x2, y2 = coordinates
    if not all(math.isfinite(item) for item in coordinates) or x2 <= x1 or y2 <= y1:
        raise ValueError("bbox coordinates must be finite and ordered")
    return coordinates


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def evaluate_records(
    records: Iterable[Mapping[str, Any]], *, iou_threshold: float = 0.5,
    confidence_threshold: float = 0.0,
) -> dict[str, Any]:
    """Evaluate frame records using one-to-one greedy class-aware matching."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be within [0, 1]")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1]")

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    matched_ious: list[float] = []
    depth_errors: list[float] = []
    depth_valid_ratios: list[float] = []
    latencies: list[float] = []
    evidence_levels: set[str] = set()
    frame_count = 0

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each record must be an object")
        frame_count += 1
        evidence = str(record.get("evidence_level", "unknown")).strip() or "unknown"
        evidence_levels.add(evidence)
        truth = list(record.get("ground_truth", []))
        predictions = []
        for candidate in record.get("predictions", []):
            if not isinstance(candidate, Mapping):
                raise ValueError("predictions must contain objects")
            confidence = _finite_number(candidate.get("confidence", 0.0))
            if confidence is None or confidence < confidence_threshold:
                continue
            if confidence < 0.0 or confidence > 1.0:
                raise ValueError("prediction confidence must be within [0, 1]")
            predictions.append({**candidate, "confidence": confidence, "_bbox": _bbox(candidate.get("bbox"))})
            latency = _finite_number(candidate.get("inference_ms"))
            if latency is not None and latency >= 0.0:
                latencies.append(latency)
            depth_ratio = _finite_number(candidate.get("valid_depth_ratio"))
            if depth_ratio is not None and 0.0 <= depth_ratio <= 1.0:
                depth_valid_ratios.append(depth_ratio)

        truth_items = []
        for item in truth:
            if not isinstance(item, Mapping):
                raise ValueError("ground_truth must contain objects")
            truth_items.append({**item, "_bbox": _bbox(item.get("bbox"))})

        used_truth: set[int] = set()
        for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
            label = str(prediction.get("class", prediction.get("target_type", "unknown")))
            best_index = None
            best_score = iou_threshold
            for index, target in enumerate(truth_items):
                if index in used_truth:
                    continue
                target_label = str(target.get("class", target.get("target_type", "unknown")))
                if target_label != label:
                    continue
                score = iou(prediction["_bbox"], target["_bbox"])
                if score >= best_score:
                    best_index, best_score = index, score
            if best_index is None:
                counts[label]["fp"] += 1
                continue
            used_truth.add(best_index)
            counts[label]["tp"] += 1
            matched_ious.append(best_score)
            predicted_depth = _finite_number(prediction.get("depth_m"))
            truth_depth = _finite_number(truth_items[best_index].get("depth_m"))
            if predicted_depth is not None and truth_depth is not None and predicted_depth >= 0.0 and truth_depth >= 0.0:
                depth_errors.append(abs(predicted_depth - truth_depth))

        for index, target in enumerate(truth_items):
            if index not in used_truth:
                label = str(target.get("class", target.get("target_type", "unknown")))
                counts[label]["fn"] += 1

    per_class: dict[str, dict[str, Any]] = {}
    for label in sorted(counts):
        values = counts[label]
        precision = values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else 0.0
        recall = values["tp"] / (values["tp"] + values["fn"]) if values["tp"] + values["fn"] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {**values, "precision": precision, "recall": recall, "f1": f1}

    macro = {
        metric: (sum(item[metric] for item in per_class.values()) / len(per_class) if per_class else 0.0)
        for metric in ("precision", "recall", "f1")
    }
    return {
        "schema_version": 1,
        "frame_count": frame_count,
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "evidence_levels": sorted(evidence_levels),
        "per_class": per_class,
        "macro": macro,
        "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
        "depth_abs_error_m_p50": _percentile(depth_errors, 0.50),
        "depth_abs_error_m_p95": _percentile(depth_errors, 0.95),
        "valid_depth_ratio_mean": sum(depth_valid_ratios) / len(depth_valid_ratios) if depth_valid_ratios else None,
        "inference_ms_p50": _percentile(latencies, 0.50),
        "inference_ms_p95": _percentile(latencies, 0.95),
        "matched_count": len(matched_ious),
    }


def evaluate_jsonl(path: Path, *, iou_threshold: float = 0.5, confidence_threshold: float = 0.0) -> dict[str, Any]:
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    result = evaluate_records(records, iou_threshold=iou_threshold, confidence_threshold=confidence_threshold)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["input_file"] = str(path)
    return result
