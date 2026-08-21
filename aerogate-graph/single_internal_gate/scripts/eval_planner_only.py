"""Run experiment-2 planner-only baselines and print a compact metrics table."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent.parent
for _path in (ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from single_internal_gate.configs.experiment_config import EXP2_SINGLE_INTERNAL_CONFIG
from single_internal_gate.baseline_groups import baseline_group_names, planners_for_group
from single_internal_gate.evaluation import make_tasks, summarize_results
from single_internal_gate.planners import create_planner, planner_names
from shared.task_suites.exp12_gate_scene import task_suite_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--planner", action="append", choices=planner_names())
    parser.add_argument("--group", choices=baseline_group_names(), default="planner_only_main")
    parser.add_argument("--task-suite", choices=task_suite_names(), default="gate")
    args = parser.parse_args()

    config = EXP2_SINGLE_INTERNAL_CONFIG
    selected = tuple(args.planner) if args.planner else planners_for_group(args.group)
    tasks = make_tasks(args.episodes, config, task_suite=args.task_suite)
    rows = []
    for planner_name in selected:
        planner = create_planner(planner_name, config.planner)
        results = tuple(planner.plan(task) for task in tasks)
        metrics = summarize_results(
            planner_name,
            tasks,
            results,
            latency_budget_ms=config.planner.latency_budget_ms,
        )
        rows.append(asdict(metrics))
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

