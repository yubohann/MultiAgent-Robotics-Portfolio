#!/usr/bin/env python3
"""Export fixed-map truth, RGB-D, and service lifecycle JSON to CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "unit", "source_run", "evidence_level"])
        writer.writeheader()
        writer.writerows(rows)


def _row(metric: str, value: object, unit: str, source_run: Path, evidence: str) -> dict[str, object]:
    return {"metric": metric, "value": value, "unit": unit, "source_run": source_run.name, "evidence_level": evidence}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-json", type=Path, required=True)
    parser.add_argument("--rgbd-json", type=Path, required=True)
    parser.add_argument("--systemd-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    truth = json.loads(args.truth_json.read_text(encoding="utf-8"))
    rgbd = json.loads(args.rgbd_json.read_text(encoding="utf-8"))
    systemd = json.loads(args.systemd_json.read_text(encoding="utf-8"))
    truth_rows = [
        _row("samples", truth["ground_truth_samples"], "count", args.truth_json.parent, truth["evidence_level"]),
        _row("translation_error_p50", truth["translation_error_p50_m"], "m", args.truth_json.parent, truth["evidence_level"]),
        _row("translation_error_p95", truth["translation_error_p95_m"], "m", args.truth_json.parent, truth["evidence_level"]),
        _row("translation_error_max", truth["translation_error_max_m"], "m", args.truth_json.parent, truth["evidence_level"]),
        _row("yaw_error_p50", truth["yaw_error_p50_deg"], "deg", args.truth_json.parent, truth["evidence_level"]),
        _row("yaw_error_p95", truth["yaw_error_p95_deg"], "deg", args.truth_json.parent, truth["evidence_level"]),
        _row("yaw_error_max", truth["yaw_error_max_deg"], "deg", args.truth_json.parent, truth["evidence_level"]),
    ]
    rgbd_rows = [
        _row("color_rate", rgbd["color_rate_hz_wall"], "Hz", args.rgbd_json.parent, rgbd["evidence_level"]),
        _row("depth_rate", rgbd["depth_rate_hz_wall"], "Hz", args.rgbd_json.parent, rgbd["evidence_level"]),
        _row("depth_valid_ratio", rgbd["depth_valid_ratio"], "ratio", args.rgbd_json.parent, rgbd["evidence_level"]),
        _row("stamp_skew_p95", rgbd["rgb_depth_stamp_skew_p95_sec"], "s", args.rgbd_json.parent, rgbd["evidence_level"]),
        _row("width", rgbd["color"]["width"], "px", args.rgbd_json.parent, rgbd["evidence_level"]),
        _row("height", rgbd["color"]["height"], "px", args.rgbd_json.parent, rgbd["evidence_level"]),
    ]
    service_rows = [
        _row("start_pass", int(systemd["start_pass"]), "bool", args.systemd_json.parent, systemd["evidence_level"]),
        _row("restart_pass", int(systemd["restart_pass"]), "bool", args.systemd_json.parent, systemd["evidence_level"]),
        _row("restart_count", systemd["restart_count"], "count", args.systemd_json.parent, systemd["evidence_level"]),
        _row("stop_timeout_pass", int(systemd["stop_timeout_pass"]), "bool", args.systemd_json.parent, systemd["evidence_level"]),
        _row("stop_timeout_elapsed", systemd["stop_timeout_elapsed_sec"], "s", args.systemd_json.parent, systemd["evidence_level"]),
    ]
    _write(args.output_dir / "fixed_map_truth_errors.csv", truth_rows)
    _write(args.output_dir / "rgbd_quality_contract.csv", rgbd_rows)
    _write(args.output_dir / "systemd_lifecycle.csv", service_rows)
    print(f"wrote quality tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
