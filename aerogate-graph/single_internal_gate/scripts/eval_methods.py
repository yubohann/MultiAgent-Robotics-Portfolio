"""Run experiment-2 closed-loop method variants and ablations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from single_internal_gate.ablation import get_variant, method_variants, variant_names
from single_internal_gate.method_evaluation import evaluate_method_variant
from shared.task_suites.exp12_gate_scene import task_suite_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--variant", action="append", choices=variant_names())
    parser.add_argument("--task-suite", choices=task_suite_names(), default="gate")
    args = parser.parse_args()

    selected = tuple(get_variant(name) for name in args.variant) if args.variant else method_variants()
    rows = [asdict(evaluate_method_variant(variant, episodes=args.episodes, task_suite=args.task_suite)) for variant in selected]
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

