"""Export an eight-metric availability/supplement table for planner baseline comparison."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = (
    ROOT
    / "results"
    / "planner_baselines"
    / "planner_only_single_E1_E2_fixed_mainline_fixed_dyn_v1_20260510_142652"
)
TO60_ROOT = BASELINE_ROOT / "latest_model_eval_to60_svg_20260515_113944" / "tables"
BASE_SUMMARY_CSV = TO60_ROOT / "planner_vs_mainline_dynamic_to60_real.csv"
MAINLINE_BY_SEED_CSV = TO60_ROOT / "mainline_by_seed_to60_real.csv"
PLANNER_ROWS_JSONL = BASELINE_ROOT / "planner_baseline_rows.jsonl"
OUT_DIR = ROOT / "docs" / "2026-05-20_single_static_dynamic_comparison" / "09_planner_baseline_eight_metrics"

METHOD_ORDER = [
    "ours_mainline",
    "astar",
    "theta_star",
    "rrt_star",
    "informed_rrt_star",
    "heuristic",
    "ego_planner",
    "fast_planner",
]

METRICS = [
    "success_rate",
    "collision_rate",
    "progress_distance_m_mean",
    "path_length_m_mean",
    "flight_time_s_mean",
    "mean_speed_mps_mean",
    "min_clearance_m_mean",
    "dynamic_swept_collision_count_mean",
]


def finite(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean_or_blank(values: list[float | None]) -> float | str:
    filtered = [v for v in values if v is not None and math.isfinite(v)]
    return mean(filtered) if filtered else ""


def count_available(values: list[float | None]) -> int:
    return sum(1 for v in values if v is not None and math.isfinite(v))


def aggregate_planner_rows() -> dict[tuple[str, int], dict[str, object]]:
    grouped: dict[tuple[str, int], dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
    with PLANNER_ROWS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("scenario_kind") != "single_dynamic":
                continue
            method = row.get("method")
            if method not in METHOD_ORDER or method == "ours_mainline":
                continue
            gate_count = int(row["gate_count"])
            key = (method, gate_count)
            path_length = finite(row.get("flown_path_length_m"))
            if path_length is None:
                path_length = finite(row.get("path_length_m"))
            grouped[key]["path_length_m_mean"].append(path_length)
            grouped[key]["mean_speed_mps_mean"].append(finite(row.get("mean_speed_mps")))
            grouped[key]["min_clearance_m_mean"].append(finite(row.get("min_clearance_m")))
            grouped[key]["dynamic_swept_collision_count_mean"].append(
                finite(row.get("dynamic_swept_collision_count"))
            )

    aggregated: dict[tuple[str, int], dict[str, object]] = {}
    for key, metrics in grouped.items():
        aggregated[key] = {}
        for metric, values in metrics.items():
            aggregated[key][metric] = mean_or_blank(values)
            aggregated[key][f"{metric}_n"] = count_available(values)
    return aggregated


def aggregate_mainline_seed_rows() -> dict[tuple[str, int], dict[str, object]]:
    grouped: dict[int, dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
    for row in read_csv_rows(MAINLINE_BY_SEED_CSV):
        gate_count = int(row["gate_count"])
        source_path = Path(row["source_path"])
        if not source_path.exists():
            continue
        with source_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        grouped[gate_count]["path_length_m_mean"].append(finite(summary.get("path_length_m_mean")))
        grouped[gate_count]["mean_speed_mps_mean"].append(finite(summary.get("mean_speed_mps_mean")))
        grouped[gate_count]["min_clearance_m_mean"].append(finite(summary.get("min_clearance_m_mean")))
        grouped[gate_count]["dynamic_swept_collision_count_mean"].append(
            finite(summary.get("dynamic_swept_collision_count_mean"))
        )

    aggregated: dict[tuple[str, int], dict[str, object]] = {}
    for gate_count, metrics in grouped.items():
        key = ("ours_mainline", gate_count)
        aggregated[key] = {}
        for metric, values in metrics.items():
            aggregated[key][metric] = mean_or_blank(values)
            aggregated[key][f"{metric}_n"] = count_available(values)
    return aggregated


def write_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_rows = read_csv_rows(BASE_SUMMARY_CSV)
    planner_extra = aggregate_planner_rows()
    mainline_extra = aggregate_mainline_seed_rows()
    extra = {**planner_extra, **mainline_extra}

    output_rows: list[dict[str, object]] = []
    for row in base_rows:
        method = row["method"]
        if method not in METHOD_ORDER:
            continue
        gate_count = int(row["gate_count"])
        key = (method, gate_count)
        out: dict[str, object] = {
            "method": method,
            "gate_count": gate_count,
            "seed_count": row.get("seed_count", ""),
            "status": row.get("status", ""),
            "success_rate": row.get("success_rate", ""),
            "collision_rate": row.get("collision_rate", ""),
            "progress_distance_m_mean": row.get("progress_distance_m_mean", ""),
            "flight_time_s_mean": row.get("flight_time_s_mean", ""),
            "moving_gate_swept_clearance_m_min_mean": row.get("moving_gate_swept_clearance_m_min_mean", ""),
        }
        for metric in (
            "path_length_m_mean",
            "mean_speed_mps_mean",
            "min_clearance_m_mean",
            "dynamic_swept_collision_count_mean",
        ):
            out[metric] = extra.get(key, {}).get(metric, "")
            out[f"{metric}_n"] = extra.get(key, {}).get(f"{metric}_n", 0)
        output_rows.append(out)

    output_rows.sort(key=lambda r: (int(r["gate_count"]), METHOD_ORDER.index(str(r["method"]))))

    table_path = OUT_DIR / "planner_baseline_dynamic_to60_eight_metrics_supplemented.csv"
    fieldnames = [
        "method",
        "gate_count",
        "seed_count",
        "status",
        "success_rate",
        "collision_rate",
        "progress_distance_m_mean",
        "path_length_m_mean",
        "path_length_m_mean_n",
        "flight_time_s_mean",
        "mean_speed_mps_mean",
        "mean_speed_mps_mean_n",
        "min_clearance_m_mean",
        "min_clearance_m_mean_n",
        "dynamic_swept_collision_count_mean",
        "dynamic_swept_collision_count_mean_n",
        "moving_gate_swept_clearance_m_min_mean",
    ]
    with table_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    availability = []
    for metric in METRICS:
        total = len(output_rows)
        available = 0
        if metric in {"success_rate", "collision_rate", "progress_distance_m_mean", "flight_time_s_mean"}:
            available = sum(1 for r in output_rows if str(r.get(metric, "")) != "")
        else:
            available = sum(1 for r in output_rows if str(r.get(metric, "")) != "")
        availability.append(
            {
                "metric": metric,
                "available_rows": available,
                "total_rows": total,
                "complete": available == total,
            }
        )
    availability_path = OUT_DIR / "planner_baseline_dynamic_to60_eight_metrics_availability.csv"
    with availability_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "available_rows", "total_rows", "complete"])
        writer.writeheader()
        writer.writerows(availability)

    report_path = OUT_DIR / "planner_baseline_eight_metrics_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Planner baseline eight-metric availability report",
                "",
                "Data sources:",
                f"- Base comparison table: `{BASE_SUMMARY_CSV}`",
                f"- Planner per-episode JSONL: `{PLANNER_ROWS_JSONL}`",
                f"- Mainline per-seed table: `{MAINLINE_BY_SEED_CSV}`",
                "",
                "Metric status:",
                "- success_rate, collision_rate, progress_distance_m_mean, and flight_time_s_mean are from the existing to60 comparison table.",
                "- path_length_m_mean, mean_speed_mps_mean, and min_clearance_m_mean are supplemented from planner per-episode JSONL and mainline stage summaries.",
                "- dynamic_swept_collision_count_mean is available for mainline stage summaries, but not for all planner-baseline methods over the full to60 density grid.",
                "- moving_gate_swept_clearance_m_min_mean is retained as an aligned swept-clearance safety metric, but it is not the same as swept-collision count.",
                "",
                "Outputs:",
                f"- `{table_path}`",
                f"- `{availability_path}`",
            ]
        ),
        encoding="utf-8",
    )

    print(table_path)
    print(availability_path)
    print(report_path)


if __name__ == "__main__":
    write_outputs()

