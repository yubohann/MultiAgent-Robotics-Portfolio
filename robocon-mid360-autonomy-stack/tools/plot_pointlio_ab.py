#!/usr/bin/env python3
"""Export and plot the paired FAST-LIO2 / Point-LIO replay metrics.

The script is intentionally log-only: it never launches ROS or changes a run.
It creates a CSV, JSON summary, and publication-style PNG/PDF/SVG figure from
one completed run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RATE_RE = re.compile(
    r"average rate:\s*([0-9.]+).*?\n\s*min:\s*([0-9.]+)s\s*"
    r"max:\s*([0-9.]+)s",
    re.MULTILINE,
)
POINT_RE = re.compile(r"^POINTS\s+(\d+)\s*$", re.MULTILINE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate_metrics(path: Path) -> dict[str, float | int | None]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    matches = list(RATE_RE.finditer(text))
    if not matches:
        return {"final_rate_hz": None, "min_period_s": None, "max_period_s": None,
                "samples": 0}
    match = matches[-1]
    return {
        "final_rate_hz": float(match.group(1)),
        "min_period_s": float(match.group(2)),
        "max_period_s": float(match.group(3)),
        "samples": len(matches),
    }


def _count(path: Path, pattern: str) -> int:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _input_counts(bag_info: Path) -> tuple[int | None, int | None, int | None]:
    text = bag_info.read_text(encoding="utf-8", errors="replace") if bag_info.is_file() else ""
    counts: dict[str, int] = {}
    for topic in ("/livox/lidar", "/livox/imu", "/clock"):
        match = re.search(rf"Topic:\s*{re.escape(topic)}.*?Count:\s*(\d+)\b", text)
        if match:
            counts[topic] = int(match.group(1))
    return counts.get("/livox/lidar"), counts.get("/livox/imu"), counts.get("/clock")


def _truth_metrics(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect(run_dir: Path) -> list[dict[str, object]]:
    db3_files = sorted((run_dir / "input_bag").glob("*.db3"))
    bag_hash = _sha256(db3_files[0]) if db3_files else None
    lidar_frames, imu_frames, clock_frames = _input_counts(run_dir / "bag_info.txt")
    rows: list[dict[str, object]] = []
    for method, topic in (("FAST-LIO2", "/Odometry"), ("Point-LIO", "/Odometry")):
        node_dir = run_dir / ("fast_lio" if method == "FAST-LIO2" else "point_lio")
        metrics = _rate_metrics(node_dir / "topic_hz.log")
        node_log = node_dir / "node.log"
        truth = _truth_metrics(node_dir / "truth_metrics.json")
        rows.append({
            "run_id": run_dir.name,
            "method": method,
            "output_topic": topic,
            "final_rate_hz": metrics["final_rate_hz"],
            "min_period_s": metrics["min_period_s"],
            "max_period_s": metrics["max_period_s"],
            "rate_samples": metrics["samples"],
            "no_point_events": _count(node_log, r"No point"),
            "too_few_input_events": _count(node_log, r"Too few input point cloud"),
            "input_lidar_frames": lidar_frames,
            "input_imu_frames": imu_frames,
            "input_clock_frames": clock_frames,
            "bag_metadata_sha256": bag_hash,
            "evidence_level": "gazebo_simulation/bag_replay",
            "formal_rate_gate_hz": 10.0,
            "truth_matched_samples": truth.get("matched_samples"),
            "translation_error_p50_m": truth.get("translation_error_p50_m"),
            "translation_error_p95_m": truth.get("translation_error_p95_m"),
            "translation_error_max_m": truth.get("translation_error_max_m"),
            "yaw_error_p50_deg": truth.get("yaw_error_p50_deg"),
            "yaw_error_p95_deg": truth.get("yaw_error_p95_deg"),
            "yaw_error_max_deg": truth.get("yaw_error_max_deg"),
            "truth_status": truth.get("status", "missing"),
        })
    return rows


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "pointlio_fastlio_ab.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "pointlio_fastlio_ab.json").write_text(
        json.dumps({"rows": rows, "diagnostic_only": True}, indent=2) + "\n",
        encoding="utf-8",
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18,
    })
    labels = [str(row["method"]) for row in rows]
    values = [float(row["final_rate_hz"]) if row["final_rate_hz"] is not None else np.nan for row in rows]
    colors = ["#0072B2", "#009E73"]
    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=0.58, edgecolor="white", linewidth=0.7)
    ax.axhline(10.0, color="#D55E00", linestyle="--", linewidth=1.2, label="10 Hz formal gate")
    ax.set_ylabel("Wall-clock odometry rate (Hz)")
    ax.set_title("Same-bag diagnostic replay: FAST-LIO2 vs Point-LIO")
    ax.set_ylim(0, max(11.0, max((v for v in values if not np.isnan(v)), default=10.0) * 1.25))
    for bar, row in zip(bars, rows):
        rate = row["final_rate_hz"]
        errors = int(row["no_point_events"] or 0) + int(row["too_few_input_events"] or 0)
        label = "unknown" if rate is None else f"{float(rate):.2f} Hz"
        if errors:
            label += f"\n{errors} input warnings"
        ax.text(bar.get_x() + bar.get_width() / 2, 0.25, label,
                ha="center", va="bottom", fontsize=8, color="#222")
    ax.legend(loc="upper right")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"fig_08_pointlio_fastlio_ab.{suffix}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rows = collect(args.run_dir)
    if not rows:
        raise SystemExit("no A/B rows found")
    write_outputs(rows, args.output_dir)
    print(f"wrote A/B metrics and figure to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
