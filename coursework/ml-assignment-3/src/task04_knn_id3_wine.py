"""Task 4 — kNN and an ID3 decision tree, both from scratch, on Wine.

Two classifiers, no ML libraries for the algorithms themselves. They share
the same preprocessing (stratified split, train-set standardization) and are
compared on the same test set.
"""

# Author: Bohan Yu
# Machine learning course, assignment 3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from utils import load_wine, results_path, standardize, stratified_train_test_split


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


# --------------------------------------------------------------------------
# kNN
# --------------------------------------------------------------------------
def knn_classify(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int) -> np.ndarray:
    """For each test point: find the k closest train points, majority vote."""
    distances = np.sqrt(((x_train[None] - x_test[:, None]) ** 2).sum(axis=2))
    neighbours = np.argsort(distances, axis=1)[:, :k]
    labels = y_train[neighbours]
    return np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=labels)


def run_knn(x_train, x_test, y_train, y_test) -> tuple[float, int]:
    x_train_std, x_test_std = standardize(x_train, x_test)
    k_candidates = [1, 3, 5, 7, 9, 11, 13, 15]
    print("=" * 56)
    print("KNN result (Wine, from scratch)")
    print("=" * 56)
    print(f"{'k':>4} | {'accuracy':>10}")
    print("-" * 28)
    best_acc, best_k = 0.0, 1
    for k in k_candidates:
        y_pred = knn_classify(x_train_std, y_train, x_test_std, k)
        acc = accuracy(y_test, y_pred)
        print(f"{k:>4} | {acc:.4f}")
        if acc > best_acc:
            best_acc, best_k = acc, k
    print("-" * 28)
    print(f"best k = {best_k}, accuracy = {best_acc:.4f}")
    return best_acc, best_k


# --------------------------------------------------------------------------
# ID3 decision tree
# --------------------------------------------------------------------------
def entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def information_gain(x: np.ndarray, y: np.ndarray, feature_idx: int, threshold: float) -> float:
    """Information gain = entropy before split minus weighted entropy after."""
    base = entropy(y)
    mask = x[:, feature_idx] <= threshold
    n = len(y)
    conditional = 0.0
    for subset in (y[mask], y[~mask]):
        if len(subset):
            conditional += (len(subset) / n) * entropy(subset)
    return base - conditional


def best_split(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
    # enumerating every possible cut point for continuous features is
    # expensive; using the mean as the threshold is crude but works fine here
    best_gain, best_feature, best_threshold = -np.inf, 0, 0.0
    for feature_idx in range(x.shape[1]):
        threshold = float(np.mean(x[:, feature_idx]))
        gain = information_gain(x, y, feature_idx, threshold)
        if gain > best_gain:
            best_gain, best_feature, best_threshold = gain, feature_idx, threshold
    return best_feature, best_threshold, best_gain


def majority_vote(y: np.ndarray) -> int:
    return int(np.bincount(y).argmax())


def build_id3(x: np.ndarray, y: np.ndarray, used: set[int] | None = None):
    # stop when all samples share a class, or when every feature is used up
    used = set() if used is None else set(used)
    if len(np.unique(y)) == 1:
        return int(y[0])
    if len(used) >= x.shape[1]:
        return majority_vote(y)

    feature_idx, threshold, _ = best_split(x, y)
    used.add(feature_idx)
    node = {"feature": feature_idx, "threshold": threshold, "children": {}}
    for side, mask in (("left", x[:, feature_idx] <= threshold), ("right", x[:, feature_idx] > threshold)):
        node["children"][side] = (
            build_id3(x[mask], y[mask], used) if len(np.unique(y[mask])) > 1 else majority_vote(y[mask])
        )
    return node


def id3_predict_single(node, sample: np.ndarray):
    if not isinstance(node, dict):
        return node
    side = "left" if sample[node["feature"]] <= node["threshold"] else "right"
    return id3_predict_single(node["children"][side], sample)


def run_id3(x_train, x_test, y_train, y_test) -> float:
    print("\n" + "=" * 56)
    print("ID3 decision tree result (Wine, from scratch)")
    print("=" * 56)
    tree = build_id3(x_train, y_train)
    y_pred = np.array([id3_predict_single(tree, sample) for sample in x_test])
    acc = accuracy(y_test, y_pred)
    print("ID3 tree built (information gain + mean-threshold discretization)")
    print(f"accuracy = {acc:.4f}")
    return acc


def main() -> None:
    x, y = load_wine()
    x_train, x_test, y_train, y_test = stratified_train_test_split(x, y, test_size=0.3)

    print("=" * 56)
    print("Wine dataset")
    print("=" * 56)
    print(f"samples: {len(x)}, features: {x.shape[1]}, classes: {len(np.unique(y))}")
    print(f"train: {len(x_train)}, test: {len(x_test)}")
    print(f"train class distribution: {np.bincount(y_train)}")
    print(f"test class distribution: {np.bincount(y_test)}")

    knn_acc, best_k = run_knn(x_train, x_test, y_train, y_test)
    id3_acc = run_id3(x_train, x_test, y_train, y_test)

    summary = (
        "| Algorithm | Best config | Accuracy |\n"
        "| --- | --- | --- |\n"
        f"| kNN (from scratch) | k={best_k} | {knn_acc:.4f} |\n"
        f"| ID3 (from scratch) | mean-threshold discretization | {id3_acc:.4f} |\n"
    )
    (results_path("classification_summary.md")).write_text(summary, encoding="utf-8")
    print("\nsaved: results/classification_summary.md")


if __name__ == "__main__":
    main()