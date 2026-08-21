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
    from single_gate.replay_video import export_single_replay_mp4

    parser = argparse.ArgumentParser(description="Replay the single-agent task with a checkpoint or heuristic controller.")
    parser.add_argument("--mode", type=str, default="heuristic", choices=["heuristic", "checkpoint"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--export-mp4", action="store_true")
    parser.add_argument("--mp4-path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--trail-length", type=int, default=40)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    summary = run_single_replay(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        device=args.device,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    if args.export_mp4 or args.mp4_path is not None:
        mp4_summary = export_single_replay_mp4(
            trajectory_path=summary["trajectory_path"],
            output_path=args.mp4_path,
            report_path=summary.get("report_path"),
            fps=args.fps,
            frame_skip=args.frame_skip,
            trail_length=args.trail_length,
            dpi=args.dpi,
        )
        summary["mp4_path"] = mp4_summary["output_path"]
        summary["mp4_export_summary_path"] = mp4_summary["summary_path"]
    print("single-agent replay complete")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

