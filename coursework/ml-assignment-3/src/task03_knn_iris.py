"""Task 3 — k-Nearest Neighbors from scratch on Iris.

The whole pipeline is hand-written: Euclidean distance, neighbour voting,
stratified split, standardization. I sweep odd k values and plot how the
accuracy changes.
"""

# Author: Bohan Yu
# Machine learning course, assignment 3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils import IRIS_LABEL_NAMES, load_iris, results_path, standardize, stratified_train_test_split

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def euclidean_distances(x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Distance from every test point to every train point in one shot.

    Broadcasting is much faster than a double loop, and easier to read once
    you get used to it.
    """
    diff = x_train[None, :, :] - x_test[:, None, :]
    return np.sqrt((diff**2).sum(axis=2))


def knn_predict(x_train, y_train, x_test, k: int) -> np.ndarray:
    """Take the k nearest neighbours and let them vote."""
    distances = euclidean_distances(x_train, x_test)
    neighbours = np.argsort(distances, axis=1)[:, :k]
    neighbour_labels = y_train[neighbours]
    votes = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=neighbour_labels)
    return votes


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def main() -> None:
    x, y = load_iris()
    x_train, x_test, y_train, y_test = stratified_train_test_split(x, y, test_size=0.3)
    x_train_std, x_test_std = standardize(x_train, x_test)

    print("=" * 56)
    print("Iris dataset")
    print("=" * 56)
    print(f"samples: {len(x)}, features: {x.shape[1]}, classes: {len(np.unique(y))}")
    print(f"train: {len(x_train)}, test: {len(x_test)}")

    # odd k values only, so ties can't happen in the voting
    k_candidates = [1, 3, 5, 7, 9, 11, 13, 15]
    scores = []
    print("\n" + "=" * 56)
    print("KNN result (Iris, from scratch)")
    print("=" * 56)
    print(f"{'k':>4} | {'accuracy':>10}")
    print("-" * 28)
    for k in k_candidates:
        y_pred = knn_predict(x_train_std, y_train, x_test_std, k)
        acc = accuracy(y_test, y_pred)
        scores.append(acc)
        print(f"{k:>4} | {acc:.4f}")

    best_idx = int(np.argmax(scores))
    best_k = k_candidates[best_idx]
    best_acc = scores[best_idx]
    print("-" * 28)
    print(f"best k = {best_k}, accuracy = {best_acc:.4f}")

    y_pred_best = knn_predict(x_train_std, y_train, x_test_std, best_k)
    print("\n20 random test samples, predicted vs. true")
    print(f"{'true label':<18} {'predicted':<18} {'match':>6}")
    print("-" * 46)
    rng = np.random.default_rng(42)
    for i in rng.choice(len(y_test), 20, replace=False):
        ok = "OK" if y_test[i] == y_pred_best[i] else "NO"
        print(f"{IRIS_LABEL_NAMES[y_test[i]]:<18} {IRIS_LABEL_NAMES[y_pred_best[i]]:<18} {ok:>6}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_candidates, scores, marker="o", linewidth=2, color="#2E86AB")
    ax.scatter([best_k], [best_acc], s=120, zorder=5, color="#E4572E", label=f"best k={best_k}, acc={best_acc:.3f}")
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("test accuracy", fontsize=11)
    ax.set_title("KNN accuracy vs k (Iris)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_path("knn_accuracy_vs_k.png"), dpi=300, bbox_inches="tight")
    print("\nsaved: results/knn_accuracy_vs_k.png")


if __name__ == "__main__":
    main()