"""Audit planner-only baselines against mainline results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "results" / "planner_baselines"
METHODS = ("astar", "theta_star", "rrt_star", "informed_rrt_star", "heuristic", "ego_planner", "fast_planner")
KIND_ORDER = ("single_static", "single_dynamic", "multi_static", "multi_dynamic")
HIGHER_BETTER = (
    "success_rate",
    "full_route_success_rate",
    "progress_ratio",
    "corridor_through_success_rate",
    "height_contract_passed_rate",
    "min_clearance_m",
    "min_pair_distance_mean_m",
)
LOWER_BETTER = (
    "hard_failure_rate",
    "collision_rate",
    "obstacle_collision_rate",
    "agent_agent_collision_rate",
    "timeout_rate",
    "planning_failure_rate",
    "no_path_rate",
    "out_of_bounds_rate",
    "safety_violation_rate",
    "side_bypass_failure_rate",
    "height_out_of_bounds_rate",
    "dispersed_termination_rate",
    "formation_slot_error_mean_m",
    "formation_slot_error_max_m",
    "path_length_ratio_success_only",
)
SECONDARY_QUALITY_METRICS = {
    "path_length_ratio_success_only",
    "formation_slot_error_mean_m",
    "formation_slot_error_max_m",
    "min_pair_distance_mean_m",
}
METRIC_MAP = {
    "success_rate": ("success_rate", "mainline_success_rate"),
    "full_route_success_rate": ("full_route_success_rate", "mainline_full_route_success_rate"),
    "progress_ratio": ("progress_ratio", "mainline_progress_ratio"),
    "corridor_through_success_rate": ("corridor_through_success_rate", "mainline_corridor_through_success_rate"),
    "height_contract_passed_rate": ("height_contract_passed_rate", "mainline_height_contract_passed_rate"),
    "min_clearance_m": ("min_clearance_m", None),
    "min_pair_distance_mean_m": ("min_pair_distance_mean_m", "mainline_min_pair_distance_mean_m"),
    "hard_failure_rate": ("hard_failure_rate", "mainline_hard_failure_rate"),
    "collision_rate": ("collision_rate", "mainline_collision_rate"),
    "obstacle_collision_rate": ("obstacle_collision_rate", "mainline_obstacle_collision_rate"),
    "agent_agent_collision_rate": ("agent_agent_collision_rate", "mainline_agent_agent_collision_rate"),
    "timeout_rate": ("timeout_rate", "mainline_timeout_rate"),
    "planning_failure_rate": ("planning_failure_rate", "mainline_planning_failure_rate"),
    "no_path_rate": ("no_path_rate", "mainline_no_path_rate"),
    "out_of_bounds_rate": ("out_of_bounds_rate", "mainline_out_of_bounds_rate"),
    "safety_violation_rate": ("safety_violation_rate", "mainline_safety_violation_rate"),
    "side_bypass_failure_rate": ("side_bypass_failure_rate", "mainline_side_bypass_failure_rate"),
    "height_out_of_bounds_rate": ("height_out_of_bounds_rate", "mainline_height_out_of_bounds_rate"),
    "dispersed_termination_rate": ("dispersed_termination_rate", "mainline_dispersed_termination_rate"),
    "formation_slot_error_mean_m": ("formation_slot_error_mean_m", "mainline_formation_slot_error_mean_m"),
    "formation_slot_error_max_m": ("formation_slot_error_max_m", "mainline_formation_slot_error_max_m"),
    "path_length_ratio_success_only": ("path_length_ratio_success_only", "mainline_path_length_ratio_success_only"),
}


@dataclass(frozen=True)
class Stats:
    n: int
    mean: float
    std: float
    sem: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--variability-sem-threshold", type=float, default=0.12)
    parser.add_argument("--dominance-eps", type=float, default=1.0e-9)
    args = parser.parse_args()

    baseline_dir = args.baseline_dir or latest_baseline_dir()
    rows = read_jsonl(baseline_dir / "planner_baseline_rows.jsonl")
    if not rows:
        raise FileNotFoundError(baseline_dir / "planner_baseline_rows.jsonl")

    out_dir = baseline_dir / "fairness_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = audit_seed_coverage(rows, expected_seeds=int(args.expected_seeds))
    dominance_rows = audit_mainline_dominance(rows, eps=float(args.dominance_eps))
    variability_rows = audit_variability(rows, threshold=float(args.variability_sem_threshold))
    monotonic_rows = audit_collision_monotonicity(rows)
    semantic_rows = audit_semantics(rows)

    write_csv(out_dir / "seed_coverage.csv", seed_rows)
    write_csv(out_dir / "mainline_dominance_violations.csv", dominance_rows)
    write_csv(out_dir / "high_variability_flags.csv", variability_rows)
    write_csv(out_dir / "collision_monotonicity_flags.csv", monotonic_rows)
    write_csv(out_dir / "semantic_flags.csv", semantic_rows)

    report = make_report(
        baseline_dir=baseline_dir,
        rows=rows,
        expected_seeds=int(args.expected_seeds),
        seed_rows=seed_rows,
        dominance_rows=dominance_rows,
        variability_rows=variability_rows,
        monotonic_rows=monotonic_rows,
        semantic_rows=semantic_rows,
    )
    (out_dir / "fairness_audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "baseline_dir": str(baseline_dir),
        "out_dir": str(out_dir),
        "rows": len(rows),
        "seed_coverage_flags": sum(1 for row in seed_rows if row["status"] != "ok"),
        "dominance_violations": len(dominance_rows),
        "high_variability_flags": len(variability_rows),
        "collision_monotonicity_flags": len(monotonic_rows),
        "semantic_flags": len(semantic_rows),
    }, indent=2, ensure_ascii=False))


def latest_baseline_dir() -> Path:
    candidates = sorted(
        [path for path in BASELINE_ROOT.glob("planner_only_baselines*") if (path / "planner_baseline_rows.jsonl").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(BASELINE_ROOT)
    return candidates[0]


def audit_seed_coverage(rows: list[dict[str, Any]], *, expected_seeds: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], set[int]] = defaultdict(set)
    for row in rows:
        grouped[(str(row["experiment"]), str(row["scenario_kind"]), int(row["gate_count"]), str(row["method"]))].add(int(row["seed"]))
    out = []
    for (experiment, kind, gate, method), seeds in sorted(grouped.items()):
        status = "ok" if len(seeds) >= expected_seeds else "insufficient_seed_count"
        out.append({
            "experiment": experiment,
            "scenario_kind": kind,
            "gate_count": gate,
            "method": method,
            "seed_count": len(seeds),
            "expected_seeds": expected_seeds,
            "seeds": " ".join(str(seed) for seed in sorted(seeds)),
            "status": status,
        })
    return out


def audit_mainline_dominance(rows: list[dict[str, Any]], *, eps: float) -> list[dict[str, Any]]:
    planner_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    mainline_groups: dict[tuple[str, int], dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        kind = str(row["scenario_kind"])
        gate = int(row["gate_count"])
        method = str(row["method"])
        seed = int(row["seed"])
        planner_groups[(kind, gate, method)].append(row)
        mainline_groups[(kind, gate)][(str(row["scenario_id"]), seed)] = row

    violations = []
    for kind in KIND_ORDER:
        gates = sorted({gate for k, gate, _m in planner_groups if k == kind})
        for gate in gates:
            mainline_source = list(mainline_groups[(kind, gate)].values())
            for metric in (*HIGHER_BETTER, *LOWER_BETTER):
                planner_key, mainline_key = METRIC_MAP[metric]
                if mainline_key is None:
                    continue
                main_stats = stat(row.get(mainline_key) for row in mainline_source)
                if main_stats.n == 0:
                    continue
                for method in METHODS:
                    method_rows = planner_groups.get((kind, gate, method), [])
                    planner_stats = stat(row.get(planner_key) for row in method_rows)
                    if planner_stats.n == 0:
                        continue
                    if metric in SECONDARY_QUALITY_METRICS:
                        planner_success = stat(row.get("success_rate") for row in method_rows)
                        main_success = stat(row.get("mainline_success_rate") for row in mainline_source)
                        planner_hard = stat(row.get("hard_failure_rate") for row in method_rows)
                        main_hard = stat(row.get("mainline_hard_failure_rate") for row in mainline_source)
                        if (
                            planner_success.n
                            and main_success.n
                            and planner_hard.n
                            and main_hard.n
                            and (
                                planner_success.mean + eps < main_success.mean
                                or planner_hard.mean > main_hard.mean + eps
                            )
                        ):
                            continue
                    higher = metric in HIGHER_BETTER
                    planner_better = planner_stats.mean > main_stats.mean + eps if higher else planner_stats.mean < main_stats.mean - eps
                    if planner_better:
                        violations.append({
                            "scenario_kind": kind,
                            "gate_count": gate,
                            "metric": metric,
                            "direction": "higher_better" if higher else "lower_better",
                            "planner_method": method,
                            "planner_mean": planner_stats.mean,
                            "planner_n": planner_stats.n,
                            "mainline_mean": main_stats.mean,
                            "mainline_n": main_stats.n,
                            "delta_planner_minus_mainline": planner_stats.mean - main_stats.mean,
                            "status": "mainline_not_best",
                        })
    return violations


def audit_variability(rows: list[dict[str, Any]], *, threshold: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in ("success_rate", "hard_failure_rate", "collision_rate", "timeout_rate", "progress_ratio"):
            value = fnum(row.get(metric))
            if math.isfinite(value):
                grouped[(str(row["scenario_kind"]), int(row["gate_count"]), str(row["method"]), metric)].append(value)
    out = []
    for (kind, gate, method, metric), values in sorted(grouped.items()):
        s = stat(values)
        if s.n >= 2 and s.sem > threshold:
            out.append({
                "scenario_kind": kind,
                "gate_count": gate,
                "method": method,
                "metric": metric,
                "n": s.n,
                "mean": s.mean,
                "std": s.std,
                "sem": s.sem,
                "threshold": threshold,
                "status": "high_seed_variability",
            })
    return out


def audit_collision_monotonicity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(str(row["scenario_kind"]), str(row["method"]))][int(row["gate_count"])].append(row)
    out = []
    for (kind, method), by_gate in sorted(grouped.items()):
        prev_gate = None
        prev_collision = None
        prev_hard = None
        prev_planning = None
        for gate in sorted(by_gate):
            collision = stat(row.get("collision_rate") for row in by_gate[gate]).mean
            hard = stat(row.get("hard_failure_rate") for row in by_gate[gate]).mean
            planning = stat(row.get("planning_failure_rate") for row in by_gate[gate]).mean
            if prev_gate is not None and collision + 1.0e-9 < prev_collision:
                explained = hard >= prev_hard - 1.0e-9 and planning >= prev_planning - 1.0e-9
                out.append({
                    "scenario_kind": kind,
                    "method": method,
                    "prev_gate_count": prev_gate,
                    "gate_count": gate,
                    "prev_collision_rate": prev_collision,
                    "collision_rate": collision,
                    "prev_hard_failure_rate": prev_hard,
                    "hard_failure_rate": hard,
                    "prev_planning_failure_rate": prev_planning,
                    "planning_failure_rate": planning,
                    "status": "collision_drop_explained_by_planning_failure" if explained else "collision_drop_unexplained",
                })
            prev_gate = gate
            prev_collision = collision
            prev_hard = hard
            prev_planning = planning
    return out


def audit_semantics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        success = fnum(row.get("success_rate"))
        path_len = fnum(row.get("path_length_m"))
        path_ratio = fnum(row.get("path_length_ratio_success_only"))
        full_route = fnum(row.get("full_route_distance_m"))
        if math.isfinite(success) and success < 1.0 and (math.isfinite(path_len) or math.isfinite(path_ratio)):
            out.append(flag(row, "failed_row_has_success_only_path_metric"))
        if math.isfinite(full_route) and abs(full_route - 54.0) > 1.0e-6:
            out.append(flag(row, "full_route_distance_not_54m"))
        if fnum(row.get("no_path_rate")) > 0.0:
            if not (fnum(row.get("timeout_rate")) >= 1.0 and fnum(row.get("hard_failure_rate")) >= 1.0 and str(row.get("done_reason")) == "planning_failure_timeout"):
                out.append(flag(row, "no_path_not_counted_as_hard_timeout"))
    return out


def flag(row: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "experiment": row.get("experiment"),
        "scenario_kind": row.get("scenario_kind"),
        "scenario_id": row.get("scenario_id"),
        "gate_count": row.get("gate_count"),
        "seed": row.get("seed"),
        "method": row.get("method"),
        "status": status,
    }


def stat(values: Any) -> Stats:
    data = [fnum(value) for value in values]
    data = [value for value in data if math.isfinite(value)]
    if not data:
        return Stats(0, float("nan"), float("nan"), float("nan"))
    mean = float(statistics.fmean(data))
    std = float(statistics.stdev(data)) if len(data) >= 2 else 0.0
    sem = std / math.sqrt(len(data)) if len(data) >= 2 else 0.0
    return Stats(len(data), mean, std, sem)


def make_report(
    *,
    baseline_dir: Path,
    rows: list[dict[str, Any]],
    expected_seeds: int,
    seed_rows: list[dict[str, Any]],
    dominance_rows: list[dict[str, Any]],
    variability_rows: list[dict[str, Any]],
    monotonic_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
) -> str:
    seed_bad = [row for row in seed_rows if row["status"] != "ok"]
    collision_unexplained = [row for row in monotonic_rows if row["status"] == "collision_drop_unexplained"]
    lines = [
        "# Fairness Audit Report",
        "",
        f"- baseline_dir: `{baseline_dir}`",
        f"- rows: {len(rows)}",
        f"- expected_seeds_per_gate_method: {expected_seeds}",
        f"- seed_coverage_flags: {len(seed_bad)}",
        f"- mainline_dominance_violations: {len(dominance_rows)}",
        f"- high_variability_flags: {len(variability_rows)}",
        f"- collision_monotonicity_flags: {len(monotonic_rows)}",
        f"- unexplained_collision_drop_flags: {len(collision_unexplained)}",
        f"- semantic_flags: {len(semantic_rows)}",
        "",
        "## Required Action",
        "",
    ]
    if seed_bad:
        lines.append(f"- Not enough seeds: rerun mainline and baselines to at least {expected_seeds} seeds.")
    if dominance_rows:
        lines.append("- Mainline is not best on at least one metric/gate/method cell; fix by valid rerun/retraining, not by editing plots.")
    if variability_rows:
        lines.append("- Seed variance is too high in flagged cells; use more seeds and report mean/std or confidence intervals.")
    if collision_unexplained:
        lines.append("- Some collision-rate drops are not explained by planning failure or hard-failure increase; inspect evaluator semantics.")
    if semantic_rows:
        lines.append("- Metric semantics are invalid in flagged rows; fix runner before plotting.")
    if not (seed_bad or dominance_rows or variability_rows or collision_unexplained or semantic_rows):
        lines.append("- No blocking fairness issue detected.")
    lines.extend([
        "",
        "## Top Mainline Dominance Violations",
        "",
        "| scenario | gate | metric | planner | planner | mainline | delta |",
        "|---|---:|---|---|---:|---:|---:|",
    ])
    for row in dominance_rows[:30]:
        lines.append(
            f"| {row['scenario_kind']} | {row['gate_count']} | {row['metric']} | {row['planner_method']} | "
            f"{float(row['planner_mean']):.4f} | {float(row['mainline_mean']):.4f} | {float(row['delta_planner_minus_mainline']):+.4f} |"
        )
    return "\n".join(lines) + "\n"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fnum(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    main()



