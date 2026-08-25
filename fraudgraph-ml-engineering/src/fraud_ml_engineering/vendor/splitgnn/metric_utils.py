from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_FIXED_PRECISION_TARGET = 0.50


def gmean_from_confusion(conf: np.ndarray) -> float:
    conf = np.asarray(conf)
    if conf.shape != (2, 2):
        padded = np.zeros((2, 2), dtype=np.int64)
        padded[: conf.shape[0], : conf.shape[1]] = conf
        conf = padded
    tn, fp, fn, tp = conf.ravel()
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return float(np.sqrt(tpr * tnr))


def threshold_candidates(probs: np.ndarray) -> np.ndarray:
    probs = np.unique(np.asarray(probs, dtype=np.float64))
    if probs.size == 0:
        return np.array([0.5], dtype=np.float64)
    if probs.size == 1:
        value = float(probs[0])
        candidates = np.array([value - 1e-9, value, value + 1e-9], dtype=np.float64)
        return np.clip(candidates, 1e-6, 1.0 - 1e-6)
    midpoints = (probs[:-1] + probs[1:]) / 2.0
    candidates = np.concatenate(([probs[0] - 1e-9], midpoints, [probs[-1] + 1e-9]))
    return np.clip(candidates, 1e-6, 1.0 - 1e-6)


def safe_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, probs))
    except Exception:
        return 0.5


def safe_average_precision(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.size == 0:
        return 0.0
    if np.unique(labels).size < 2:
        return float(labels.mean()) if labels.size else 0.0
    try:
        return float(average_precision_score(labels, probs))
    except Exception:
        return float(labels.mean()) if labels.size else 0.0


def recall_at_precision_target(
    labels: np.ndarray,
    probs: np.ndarray,
    precision_target: float = DEFAULT_FIXED_PRECISION_TARGET,
) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.size == 0:
        return 0.0, 1.0

    try:
        precision, recall, thresholds = precision_recall_curve(labels, probs)
    except Exception:
        return 0.0, 1.0

    valid_indices = np.flatnonzero(precision >= float(precision_target))
    if valid_indices.size == 0:
        return 0.0, 1.0

    best_index = int(valid_indices[np.argmax(recall[valid_indices])])
    best_recall = float(recall[best_index])
    if thresholds.size == 0:
        return best_recall, 1.0
    if best_index >= thresholds.size:
        best_threshold = float(np.nextafter(np.max(probs), np.inf))
    else:
        best_threshold = float(thresholds[best_index])
    best_threshold = float(np.clip(best_threshold, 1e-6, 1.0))
    return best_recall, best_threshold


def compute_binary_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    precision_target: float = DEFAULT_FIXED_PRECISION_TARGET,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= float(threshold)).astype(np.int32)
    conf = confusion_matrix(labels, preds, labels=[0, 1])
    recall_at_precision, recall_at_precision_threshold = recall_at_precision_target(
        labels,
        probs,
        precision_target=precision_target,
    )
    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(labels, preds)) if preds.size else 0.0,
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "auc": safe_roc_auc(labels, probs),
        "pr_auc": safe_average_precision(labels, probs),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "f1_score": float(f1_score(labels, preds, zero_division=0)),
        "f1_binary": float(f1_score(labels, preds, zero_division=0)),
        "f1_pos": float(f1_score(labels, preds, zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "gmean": float(gmean_from_confusion(conf)),
        "positive_rate": float(preds.mean()) if preds.size else 0.0,
        "prob_mean": float(probs.mean()) if probs.size else 0.0,
        "prob_std": float(probs.std()) if probs.size else 0.0,
        "fixed_precision_target": float(precision_target),
        "recall_at_precision": float(recall_at_precision),
        "recall_at_precision_threshold": float(recall_at_precision_threshold),
    }


def find_best_threshold_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    precision_target: float = DEFAULT_FIXED_PRECISION_TARGET,
) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.std() < 1e-6:
        fallback = float(np.clip(float(probs.mean()), 1e-6, 1.0 - 1e-6))
        return compute_binary_metrics(
            labels,
            probs,
            fallback,
            precision_target=precision_target,
        )

    best_metrics: dict[str, float] | None = None
    best_rank: tuple[float, ...] | None = None
    label_positive_rate = float(np.mean(labels)) if len(labels) else 0.0
    for threshold in threshold_candidates(probs):
        current = compute_binary_metrics(
            labels,
            probs,
            float(threshold),
            precision_target=precision_target,
        )
        precision_shortfall = max(0.0, float(precision_target) - float(current["precision"]))
        # Favor balanced classification quality because the mainline protocol
        # reports macro-F1/gmean, and positive-class F1 alone tends to pick
        # overly aggressive thresholds on heavily imbalanced splits.
        rank = (
            float(current["f1_macro"]),
            float(current["gmean"]),
            -precision_shortfall,
            float(current["f1_score"]),
            float(current["recall"]),
            -abs(float(current["positive_rate"]) - label_positive_rate),
            -abs(float(current["threshold"]) - 0.5),
        )
        if best_rank is None or rank > best_rank:
            best_metrics = current
            best_rank = rank
    if best_metrics is None:
        raise RuntimeError("Threshold search failed to produce any metrics.")
    return best_metrics


def coerce_metric_dict(metrics: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in metrics.items()}
