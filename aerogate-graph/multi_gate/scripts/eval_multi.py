"""Evaluation entry for the multi-agent Graph-FlashSAC experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> None:
    _bootstrap_imports()
    from multi_gate.configs import list_multi_experiment_config_names, resolve_multi_experiment_config
    from multi_gate.training import evaluate_checkpoint, evaluate_size_buckets
    from shared.runtime.artifacts import allocate_replay_artifacts, default_run_name, write_json

    parser = argparse.ArgumentParser(description="Evaluate a Graph-FlashSAC checkpoint in the multi-agent task.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config-name", type=str, default="auto", choices=["auto", *list_multi_experiment_config_names()])
    parser.add_argument("--bucket-eval", action="store_true")
    parser.add_argument("--team-sizes", type=str, default=None, help="Comma-separated team-size buckets, e.g. 2,3,5,8.")
    parser.add_argument("--episodes-per-bucket", type=int, default=None)
    args = parser.parse_args()
    resolved_config_name, experiment_config = resolve_multi_experiment_config(
        args.config_name,
        checkpoint_path=args.checkpoint,
    )

    if args.bucket_eval:
        team_sizes = None
        if args.team_sizes:
            team_sizes = [int(value.strip()) for value in args.team_sizes.split(",") if value.strip()]
        summary = evaluate_size_buckets(
            checkpoint_path=args.checkpoint,
            episodes_per_bucket=args.episodes_per_bucket or args.episodes,
            seed=args.seed,
            device=args.device,
            team_sizes=team_sizes,
            experiment_config=experiment_config,
        )
    else:
        summary = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            num_agents=args.num_agents,
            experiment_config=experiment_config,
        )
    if args.output_dir is None:
        eval_team_size = "buckets" if args.bucket_eval else summary["num_agents"]
        artifacts = allocate_replay_artifacts(
            "multi",
            run_name=default_run_name(f"{experiment_config.experiment_id}_eval_{eval_team_size}"),
        )
        output_dir = artifacts.output_dir
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "eval_summary.json"
    summary["summary_path"] = str(summary_path)
    write_json(summary_path, summary)
    print("multi-agent Graph-FlashSAC evaluation complete")
    print(f"config_name={resolved_config_name}")
    print(f"summary_path={summary_path}")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

