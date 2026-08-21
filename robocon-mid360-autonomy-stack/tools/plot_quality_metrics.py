#!/usr/bin/env python3
"""Create publication-style figures from exported quality metric tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]


def _read(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--rgbd-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth = _read(args.truth_csv)
    rgbd = _read(args.rgbd_csv)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18,
    })

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.8), constrained_layout=True)
    labels = ["P50", "P95", "Max"]
    x = np.arange(3)
    axes[0].bar(x, [truth["translation_error_p50"], truth["translation_error_p95"], truth["translation_error_max"]], color=COLORS[0], width=0.65)
    axes[0].set_title("Fixed-map translation error")
    axes[0].set_ylabel("Error (m)")
    axes[0].set_xticks(x, labels)
    axes[1].bar(x, [truth["yaw_error_p50"], truth["yaw_error_p95"], truth["yaw_error_max"]], color=COLORS[2], width=0.65)
    axes[1].set_title("Fixed-map yaw error")
    axes[1].set_ylabel("Error (deg)")
    axes[1].set_xticks(x, labels)
    _save(fig, args.output_dir / "fig_09_fixed_map_truth_error")

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.8), constrained_layout=True)
    axes[0].bar(["RGB", "Depth"], [rgbd["color_rate"], rgbd["depth_rate"]], color=[COLORS[0], COLORS[1]], width=0.6)
    axes[0].set_title("Gazebo RGB-D publication rate")
    axes[0].set_ylabel("Wall-clock rate (Hz)")
    axes[1].bar(["Valid depth", "Stamp skew P95"], [rgbd["depth_valid_ratio"], rgbd["stamp_skew_p95"]], color=[COLORS[1], COLORS[3]], width=0.6)
    axes[1].set_title("RGB-D content/timing contract")
    axes[1].set_ylabel("Ratio / seconds")
    _save(fig, args.output_dir / "fig_10_rgbd_quality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
