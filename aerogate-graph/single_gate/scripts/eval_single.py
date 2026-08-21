"""Evaluation entry for the single-agent Graph-FlashSAC experiment."""

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
    from single_gate.training import evaluate_checkpoint
    from shared.runtime.artifacts import allocate_replay_artifacts, default_run_name, write_json

    parser = argparse.ArgumentParser(description="Evaluate a Graph-FlashSAC checkpoint in the single-agent task.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )
    if args.output_dir is None:
        artifacts = allocate_replay_artifacts("single", run_name=default_run_name("single_eval"))
        output_dir = artifacts.output_dir
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "eval_summary.json"
    summary["summary_path"] = str(summary_path)
    write_json(summary_path, summary)
    print("single-agent Graph-FlashSAC evaluation complete")
    print(f"summary_path={summary_path}")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

