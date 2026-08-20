#!/usr/bin/env python3
"""Create publication-style figures from run-summary CSV data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"blue": "#0072B2", "green": "#009E73", "orange": "#E69F00", "red": "#D55E00"}


def _number(row: dict[str, str], key: str) -> float:
    value = row[key]
    return float("nan") if value in {"", "unknown", "TBD"} else float(value)


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("input CSV has no data rows")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18,
    })
    labels = [row["run_id"].replace("gazebo_", "") for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.3), constrained_layout=True)
    axes[0].plot(x, [_number(row, "odom_rate_hz") for row in rows], color=COLORS["blue"], marker="o", label="Local odometry")
    axes[0].axhline(10.0, color=COLORS["red"], linestyle="--", linewidth=1.2, label="10 Hz gate")
    axes[0].set_ylabel("Wall-clock rate (Hz)")
    axes[0].set_title("Controlled Gazebo runs: output rate and pose age")
    axes[0].legend(loc="best")
    axes[1].plot(x, [_number(row, "pose_age_p95_ms") for row in rows], color=COLORS["orange"], marker="s", label="Pose age P95")
    axes[1].axhline(150.0, color=COLORS["red"], linestyle="--", linewidth=1.2, label="150 ms target")
    axes[1].set_ylabel("Pose age P95 (ms, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Run profile")
    axes[1].legend(loc="best")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    _save(fig, args.output_dir / "fig_07_rate_pose_age")

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, [_number(row, "lidar_raw_points_median") for row in rows], width, color=COLORS["blue"], label="Raw points median")
    axes[0].bar(x + width / 2, [_number(row, "lidar_finite_points_min") for row in rows], width, color=COLORS["green"], label="Finite points minimum")
    axes[0].set_title("Controlled Gazebo input summary")
    axes[0].set_ylabel("Point count")
    axes[0].legend(ncol=2, loc="best")
    axes[1].plot(x, [_number(row, "offset_span_ms_median") for row in rows], color=COLORS["orange"], marker="D", label="Offset span")
    axes[1].plot(x, [_number(row, "imu_rate_hz") for row in rows], color=COLORS["green"], marker="^", label="IMU rate")
    axes[1].set_ylabel("Milliseconds / Hz")
    axes[1].set_xlabel("Run profile")
    axes[1].legend(ncol=2, loc="best")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    _save(fig, args.output_dir / "fig_05_input_quality")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
