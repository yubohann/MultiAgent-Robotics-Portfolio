"""Replay entry for the single-agent Graph-FlashSAC and heuristic controllers."""

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

    from single_gate.replay import run_single_replay

    parser = argparse.ArgumentParser(description="Replay the single-agent task with a checkpoint or heuristic controller.")
    parser.add_argument("--mode", type=str, default="heuristic", choices=["heuristic", "checkpoint"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    summary = run_single_replay(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        device=args.device,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    print("single-agent replay complete")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

