#!/usr/bin/env python3
"""Export controlled-run summaries to a reproducible CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run_id", "evidence_level", "profile", "scene", "duration_sec",
    "lidar_raw_points_min", "lidar_raw_points_median", "lidar_finite_points_min",
    "offset_span_ms_median", "imu_rate_hz", "odom_rate_hz", "pose_age_p95_ms",
    "pose_age_status", "error_count",
]


def _ns_to_ms(value: object) -> object:
    return "unknown" if value is None else float(value) / 1_000_000.0


def _sec_to_ms(value: object) -> object:
    return "unknown" if value is None else float(value) * 1_000.0


def _pose_age_metrics(value: object) -> tuple[object, str]:
    if value is None:
        return "unknown", "missing"
    seconds = float(value)
    # A multi-day value can only come from the pre-readiness simulation clock;
    # retain the run row but keep that startup artifact out of performance plots.
    if seconds < 0.0 or seconds > 60.0:
        return "unknown", "startup_clock_artifact"
    return seconds * 1_000.0, "valid_motion_window"


def row_for(run_dir: Path) -> dict[str, object]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    errors = summary.get("error_counts") or {}
    pose_age_ms, pose_age_status = _pose_age_metrics(summary.get("pose_age_p95_sec"))
    return {
        "run_id": run_dir.name,
        "evidence_level": manifest.get("evidence_level", "gazebo_simulation"),
        "profile": manifest.get("profile", "unknown"),
        "scene": manifest.get("scene", "unknown"),
        "duration_sec": summary.get("duration_sec", "unknown"),
        "lidar_raw_points_min": summary.get("lidar_raw_points_min", "unknown"),
        "lidar_raw_points_median": summary.get("lidar_raw_points_median", "unknown"),
        "lidar_finite_points_min": summary.get("lidar_finite_points_min", "unknown"),
        "offset_span_ms_median": _ns_to_ms(summary.get("offset_span_ns_median")),
        "imu_rate_hz": summary.get("imu_rate_hz", "unknown"),
        "odom_rate_hz": summary.get("odom_rate_hz", "unknown"),
        "pose_age_p95_ms": pose_age_ms,
        "pose_age_status": pose_age_status,
        "error_count": sum(int(value) for value in errors.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [row_for(path) for path in sorted(args.runs.iterdir())
            if path.is_dir() and (path / "summary.json").is_file()]
    if not rows:
        raise SystemExit(f"no summary.json found below {args.runs}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} run rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
