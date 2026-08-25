"""Task 1 — PCA on the Wine dataset (classes 1 vs 2).

Standardize the 13 wine-chemical features, project the binary subset onto
two principal components, save the projection as CSV, and plot it.
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import load_wine, results_path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    x, y = load_wine()
    mask = np.isin(y, [0, 1])  # keep only classes 1 and 2
    x_bin, y_bin = x[mask], y[mask]
    y_bin = y_bin + 1  # shift back to 1/2 for nicer labels on the plot

    print("=" * 56)
    print("PCA data preprocessing")
    print("=" * 56)
    print(f"binary dataset: {x_bin.shape[0]} samples x {x_bin.shape[1]} features")
    print(f"class distribution: class1({(y_bin == 1).sum()}), class2({(y_bin == 2).sum()})")

    # features are on very different scales, PCA needs standardization first
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_bin)
    print(f"post-standardization means (first 3 dims): {np.round(x_scaled.mean(axis=0)[:3], 6)}")
    print(f"post-standardization variances (first 3 dims): {np.round(x_scaled.var(axis=0)[:3], 6)}")

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)
    ratio = pca.explained_variance_ratio_

    frame = pd.DataFrame(
        {
            "sample_id": np.arange(len(x_pca)),
            "PC1": x_pca[:, 0],
            "PC2": x_pca[:, 1],
            "class": y_bin,
        }
    )
    frame.to_csv(results_path("pca_projection.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 56)
    print("PCA result")
    print("=" * 56)
    print(f"PC1 explained variance ratio: {ratio[0]:.4f}")
    print(f"PC2 explained variance ratio: {ratio[1]:.4f}")
    print(f"cumulative: {ratio.sum():.4f}")
    print("saved: results/pca_projection.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    for cls, color, marker, label in zip([1, 2], ["#E4572E", "#2E86AB"], ["o", "s"], ["class 1", "class 2"]):
        sub = frame[frame["class"] == cls]
        ax.scatter(sub["PC1"], sub["PC2"], c=color, marker=marker, s=55, alpha=0.75, label=label)
    ax.set_xlabel(f"PC1 (explained variance {ratio[0]:.1%})", fontsize=11)
    ax.set_ylabel(f"PC2 (explained variance {ratio[1]:.1%})", fontsize=11)
    ax.set_title("PCA projection of Wine (classes 1 vs 2)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_path("pca_visualization.png"), dpi=300, bbox_inches="tight")
    print("saved: results/pca_visualization.png")


if __name__ == "__main__":
    main()