"""Aggregate gate-density evaluation runs into tables and curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CURVE_FIELDS = (
    "success_rate",
    "collision_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "path_length_m_mean",
    "flight_time_s_mean",
    "min_clearance_m_mean",
    "planner_call_count_mean",
    "global_planner_trigger_count_mean",
    "planner_latency_ms_p95_mean",
    "guidance_query_count_mean",
    "guidance_failure_count_mean",
    "guidance_fallback_count_mean",
    "guidance_latency_ms_p95_mean",
    "guidance_non_fallback_rate_mean",
    "route_guidance_tracking_error_m_mean",
    "guidance_replan_urgency_mean",
    "guidance_waypoint_bias_y_mean",
    "guidance_dynamic_clearance_margin_m_mean",
    "actual_gate_motion_range_m_mean",
    "actual_gate_motion_range_x_m_mean",
    "actual_gate_motion_range_y_m_mean",
    "actual_gate_motion_range_mean_m_mean",
    "actual_gate_max_displacement_m_mean",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_stage_summaries(input_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(input_dir.rglob("stage_summary.json")):
        payload = _read_json(path)
        payload["stage_summary_path"] = str(path)
        summaries.append(payload)
    return summaries


def _aggregate_by_gate_count(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_count: dict[int, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_count.setdefault(int(summary["gate_count"]), []).append(summary)

    rows = []
    for gate_count in sorted(by_count):
        group = by_count[gate_count]
        row: dict[str, Any] = {
            "gate_count": gate_count,
            "seed_count": len({int(item["seed"]) for item in group}),
            "run_count": len(group),
            "episodes": int(sum(int(item.get("episodes", 0)) for item in group)),
        }
        for field in CURVE_FIELDS:
            values = [float(item[field]) for item in group if item.get(field) is not None]
            row[f"{field}_mean"] = float(np.mean(values) if values else 0.0)
            row[f"{field}_std"] = float(np.std(values) if values else 0.0)
        shield_values = [item.get("shield_activation_ratio") for item in group if item.get("shield_activation_ratio") is not None]
        row["shield_activation_ratio_mean"] = float(np.mean(shield_values)) if shield_values else None
        row["shield_note"] = "N/A" if not shield_values else ""
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(rows: list[dict[str, Any]], figures_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    x = [int(row["gate_count"]) for row in rows]
    specs = [
        ("success_rate_mean", "success_rate_vs_gate_count.png", "Success rate"),
        ("collision_rate_mean", "collision_rate_vs_gate_count.png", "Collision rate"),
        ("timeout_rate_mean", "timeout_rate_vs_gate_count.png", "Timeout rate"),
        ("path_length_m_mean_mean", "path_length_vs_gate_count.png", "Path length (m)"),
        ("flight_time_s_mean_mean", "flight_time_vs_gate_count.png", "Flight time (s)"),
        ("min_clearance_m_mean_mean", "min_clearance_vs_gate_count.png", "Min clearance (m)"),
        ("planner_call_count_mean_mean", "replan_count_vs_gate_count.png", "Planner calls"),
        ("global_planner_trigger_count_mean_mean", "global_planner_triggers_vs_gate_count.png", "Global planner triggers"),
        ("planner_latency_ms_p95_mean_mean", "planner_latency_p95_vs_gate_count.png", "Planner latency p95 (ms)"),
        ("guidance_latency_ms_p95_mean_mean", "guidance_latency_p95_vs_gate_count.png", "Guidance latency p95 (ms)"),
        ("guidance_non_fallback_rate_mean_mean", "guidance_non_fallback_rate_vs_gate_count.png", "Guidance non-fallback rate"),
        ("route_guidance_tracking_error_m_mean_mean", "route_guidance_error_vs_gate_count.png", "Route guidance error (m)"),
    ]
    output_paths = []
    for field, filename, ylabel in specs:
        y = [float(row.get(field, 0.0) or 0.0) for row in rows]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        ax.plot(x, y, marker="o", linewidth=2.0)
        ax.set_xlabel("gate count")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28)
        ax.set_xticks(x)
        fig.tight_layout()
        output_path = figures_dir / filename
        fig.savefig(output_path)
        plt.close(fig)
        output_paths.append(str(output_path))
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = _collect_stage_summaries(args.input_dir)
    rows = _aggregate_by_gate_count(summaries)
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    _write_csv(tables_dir / "gate_density_runs.csv", summaries)
    _write_csv(tables_dir / "gate_density_by_count.csv", rows)
    figure_paths = _plot_curves(rows, figures_dir)
    aggregate = {
        "input_dir": str(args.input_dir),
        "run_count": len(summaries),
        "gate_count_rows": rows,
        "curve_fields": list(CURVE_FIELDS),
        "figures": figure_paths,
        "shield_activation_ratio": "N/A if no underlying run reports a shield ratio",
    }
    _write_json(args.output_dir / "gate_density_aggregate_summary.json", aggregate)
    print("gate-density aggregation complete")
    print(f"run_count={len(summaries)}")
    print(f"by_count_csv={tables_dir / 'gate_density_by_count.csv'}")
    print(f"aggregate_summary={args.output_dir / 'gate_density_aggregate_summary.json'}")


if __name__ == "__main__":
    main()

