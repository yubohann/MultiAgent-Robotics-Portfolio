"""Minimal training entry for the single-agent Graph-FlashSAC experiment."""

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
    from single_gate.training import run_training
    from shared.runtime.artifacts import RUNTIME_POLICY, allocate_training_artifacts

    parser = argparse.ArgumentParser(description="Train the single-agent Graph-FlashSAC gate experiment.")
    parser.add_argument("--train-steps", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--updates-per-step", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=64)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--resume-mode", type=str, default=None, choices=["reset_train_state", "keep_optimizer_state"])
    parser.add_argument("--checkpoint-interval-steps", type=int, default=None)
    parser.add_argument("--selection-eval-episodes", type=int, default=None)
    parser.add_argument("--periodic-eval-episodes", type=int, default=0)
    parser.add_argument("--periodic-eval-interval-steps", type=int, default=None)
    parser.add_argument("--periodic-replay-mode", type=str, default="skip", choices=["skip", "heuristic", "checkpoint"])
    parser.add_argument("--periodic-replay-interval-steps", type=int, default=None)
    parser.add_argument("--periodic-replay-max-steps", type=int, default=None)
    parser.add_argument("--periodic-replay-render-isaaclab", action="store_true")
    parser.add_argument("--periodic-replay-export-video", action="store_true")
    parser.add_argument("--periodic-replay-fps", type=int, default=10)
    args = parser.parse_args()

    if args.save_dir is None:
        artifacts = allocate_training_artifacts("single", run_name=args.run_name)
        log_dir = artifacts.log_dir
        checkpoint_dir = artifacts.checkpoint_dir
    else:
        base_dir = Path(args.save_dir)
        log_dir = base_dir / "logs"
        checkpoint_dir = base_dir / "checkpoints"

    summary = run_training(
        train_steps=args.train_steps,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        log_every=args.log_every,
        resume_checkpoint=args.resume_checkpoint,
        resume_mode=args.resume_mode,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        selection_eval_episodes=args.selection_eval_episodes,
        periodic_eval_episodes=args.periodic_eval_episodes,
        periodic_eval_interval_steps=args.periodic_eval_interval_steps,
        periodic_replay_mode=args.periodic_replay_mode,
        periodic_replay_interval_steps=args.periodic_replay_interval_steps,
        periodic_replay_max_steps=args.periodic_replay_max_steps,
        periodic_replay_render_isaaclab=args.periodic_replay_render_isaaclab,
        periodic_replay_export_video=args.periodic_replay_export_video,
        periodic_replay_fps=args.periodic_replay_fps,
    )
    print(f"runtime_policy={RUNTIME_POLICY}")
    print("single-agent Graph-FlashSAC training complete")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

