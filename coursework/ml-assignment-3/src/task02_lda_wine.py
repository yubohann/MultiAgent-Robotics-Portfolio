"""Task 2 — LDA on the Wine dataset (classes 1 vs 2).

Unlike PCA, LDA uses the class labels: it looks for the axis that separates
the two classes best. I project onto that single axis and check how far apart
the class means land.
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
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

from utils import load_wine, results_path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    x, y = load_wine()
    mask = np.isin(y, [0, 1])
    x_bin, y_bin = x[mask], y[mask]
    y_bin = y_bin + 1

    print("=" * 56)
    print("LDA data preprocessing")
    print("=" * 56)
    print(f"binary dataset: {x_bin.shape[0]} samples x {x_bin.shape[1]} features")
    print(f"class distribution: class1({(y_bin == 1).sum()}), class2({(y_bin == 2).sum()})")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_bin)

    # with two classes there is at most one discriminant axis
    lda = LDA(n_components=1)
    x_lda = lda.fit_transform(x_scaled, y_bin)

    frame = pd.DataFrame({"sample_id": np.arange(len(x_lda)), "LD1": x_lda[:, 0], "class": y_bin})
    frame.to_csv(results_path("lda_projection.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 56)
    print("LDA result")
    print("=" * 56)
    print(f"discriminant axis dims: {x_lda.shape[1]}")
    class_means = {cls: x_lda[y_bin == cls].mean() for cls in [1, 2]}
    print(f"class1 mean on the axis: {class_means[1]:.4f}")
    print(f"class2 mean on the axis: {class_means[2]:.4f}")
    print(f"gap between means: {abs(class_means[1] - class_means[2]):.4f}")
    print("saved: results/lda_projection.csv")

    # draw each class on its own horizontal line so the separation is obvious
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for cls, color, label in zip([1, 2], ["#E4572E", "#2E86AB"], ["class 1", "class 2"]):
        values = x_lda[y_bin == cls, 0]
        ax.scatter(values, np.full_like(values, cls), s=48, alpha=0.7, c=color, label=f"{label} (n={len(values)})")
    ax.set_xlabel("LD1 (discriminant projection)", fontsize=11)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["class 1", "class 2"])
    ax.set_title("LDA projection of Wine (classes 1 vs 2)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_path("lda_visualization.png"), dpi=300, bbox_inches="tight")
    print("saved: results/lda_visualization.png")


if __name__ == "__main__":
    main()