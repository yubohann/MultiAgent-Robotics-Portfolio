"""Build a fresh-process P07 train-outcome collection plan.

The plan uses the persistent-collection plan schema and writes one atomic
worker run per episode.  It deliberately only accepts a single frozen scene
and fixed P03/P04/P05/P06/controller artifacts, so extending the plan means
extending the audited scene assets rather than silently changing contracts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import write_json_atomic  # noqa: E402

PLAN_SCHEMA_VERSION = "hm3d-p07-persistent-collection-plan-v1"
DEFAULT_STRATEGIES = ("random", "frontier_3d", "auction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--collision-usd", required=True, type=Path)
    parser.add_argument("--start-reset-json", required=True, type=Path)
    parser.add_argument("--flight-space-audit", required=True, type=Path)
    parser.add_argument("--p03-artifact", required=True, type=Path)
    parser.add_argument("--p04-artifact", required=True, type=Path)
    parser.add_argument("--p05-artifact", required=True, type=Path)
    parser.add_argument("--p06-artifact", required=True, type=Path)
    parser.add_argument("--transit-time-model-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plan-output", required=True, type=Path)
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="comma-separated strategy list",
    )
    parser.add_argument("--episodes-per-strategy", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=20260811)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--max-decision-count", type=int, default=128)
    parser.add_argument("--action-budget-s", type=float, default=40.0)
    parser.add_argument(
        "--controller-id",
        default="isaac-so3-feedback-v6",
    )
    return parser.parse_args()


def _as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def main() -> int:
    args = parse_args()
    if args.episodes_per_strategy < 1:
        raise ValueError("episodes per strategy must be positive")
    strategies = tuple(
        item.strip() for item in args.strategies.split(",") if item.strip()
    )
    if not strategies:
        raise ValueError("at least one strategy is required")
    scene_prefix = args.scene_id.split("-", 1)[0]
    runs: list[dict[str, object]] = []
    seed = args.seed_base
    for strategy in strategies:
        for _ in range(args.episodes_per_strategy):
            output_path = (
                args.output_dir
                / f"{scene_prefix}_{strategy}_train_seed{seed}.json"
            )
            run_id = f"{scene_prefix}-{strategy}-train-outcome-seed{seed}"
            runs.append(
                {
                    "run_id": run_id,
                    "runner_arguments": [
                        "--scene-id", args.scene_id,
                        "--split", args.split,
                        "--record-purpose", "train_outcome",
                        "--collision-usd", _as_posix(args.collision_usd),
                        "--start-reset-json", _as_posix(args.start_reset_json),
                        "--flight-space-audit", _as_posix(args.flight_space_audit),
                        "--p03-artifact", _as_posix(args.p03_artifact),
                        "--p04-artifact", _as_posix(args.p04_artifact),
                        "--p05-artifact", _as_posix(args.p05_artifact),
                        "--p06-artifact", _as_posix(args.p06_artifact),
                        "--transit-time-model-json", _as_posix(args.transit_time_model_json),
                        "--output", _as_posix(output_path),
                        "--strategy", strategy,
                        "--candidate-limit", str(args.candidate_limit),
                        "--max-decision-count", str(args.max_decision_count),
                        "--action-budget-s", str(args.action_budget_s),
                        "--controller-id", args.controller_id,
                        "--random-key", str(seed),
                    ],
                }
            )
            seed += 1
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "claim_limit": (
            "Development train-only P07 outcome collection on the frozen scene "
            f"{args.scene_id}. Every episode runs in a fresh Isaac process. "
            "This plan is not a formal holdout result."
        ),
        "runs": runs,
    }
    write_json_atomic(args.plan_output, payload)
    print(
        json.dumps(
            {
                "plan": str(args.plan_output),
                "episodes": len(runs),
                "strategies": strategies,
                "seed_interval": [args.seed_base, seed - 1],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
