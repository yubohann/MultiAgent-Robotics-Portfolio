"""Batched multi-agent evaluation entry for aerogate_graph."""

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
    from multi_gate.env.vector_eval import evaluate_checkpoint_batched
    from shared.runtime.artifacts import allocate_replay_artifacts, default_run_name, write_json

    parser = argparse.ArgumentParser(description="Run Python-level batched evaluation for multi-agent Graph-FlashSAC.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config-name", type=str, default="auto", choices=["auto", *list_multi_experiment_config_names()])
    parser.add_argument("--team-sizes", type=str, default="2,3,5,7,8,9")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    team_sizes = [int(value.strip()) for value in args.team_sizes.split(",") if value.strip()]
    oversized_team_sizes = [team_size for team_size in team_sizes if team_size > 9]
    if oversized_team_sizes:
        raise SystemExit(
            "Current aerogate_graph formal Graph-FlashSAC evaluation is capped at 9 drones; "
            f"got oversized team sizes: {oversized_team_sizes}. "
            "Use a historical diagnostic-only path for 12-drone pressure tests."
        )
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    resolved_config_name, experiment_config = resolve_multi_experiment_config(
        args.config_name,
        checkpoint_path=args.checkpoint,
    )
    summary = evaluate_checkpoint_batched(
        checkpoint_path=args.checkpoint,
        experiment_config=experiment_config,
        team_sizes=team_sizes,
        seeds=seeds,
        episodes=args.episodes,
        device=args.device,
    )
    if args.output_dir is None:
        artifacts = allocate_replay_artifacts(
            "multi",
            run_name=default_run_name(f"{experiment_config.experiment_id}_batched_eval"),
        )
        output_dir = artifacts.output_dir
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "batched_eval_summary.json"
    summary["summary_path"] = str(summary_path)
    write_json(summary_path, summary)
    print("multi-agent batched evaluation complete")
    print(f"config_name={resolved_config_name}")
    print(f"summary_path={summary_path}")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

