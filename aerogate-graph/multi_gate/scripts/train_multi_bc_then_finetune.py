"""Collect experts, run BC warm start, then fine-tune with Graph-FlashSAC."""

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
    from multi_gate.configs import get_multi_experiment_config, list_multi_experiment_config_names
    from multi_gate.imitation import run_bc_warmstart_then_finetune
    from shared.runtime.artifacts import RUNTIME_POLICY, allocate_training_artifacts

    parser = argparse.ArgumentParser(description="Run expert collection -> BC warm start -> Graph-FlashSAC fine-tuning.")
    parser.add_argument("--config-name", type=str, default="variable", choices=list_multi_experiment_config_names())
    parser.add_argument("--train-steps", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--initial-actor-checkpoint", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--max-sampled-agents", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--updates-per-step", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=64)
    parser.add_argument("--expert-episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--retain-failed-episodes", action="store_true")
    parser.add_argument("--bc-epochs", type=int, default=None)
    parser.add_argument("--bc-batch-size", type=int, default=None)
    parser.add_argument("--bc-learning-rate", type=float, default=None)
    parser.add_argument("--bc-weight-decay", type=float, default=None)
    parser.add_argument("--bc-validation-split", type=float, default=None)
    parser.add_argument("--bc-target-log-std", type=float, default=None)
    parser.add_argument("--bc-log-std-penalty-scale", type=float, default=None)
    parser.add_argument("--bc-output-dir", type=str, default=None)
    parser.add_argument("--bc-run-name", type=str, default=None)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=None)
    parser.add_argument("--selection-eval-episodes", type=int, default=None)
    args = parser.parse_args()

    experiment_config = get_multi_experiment_config(args.config_name)
    if args.save_dir is None:
        artifacts = allocate_training_artifacts("multi", run_name=args.run_name)
        log_dir = artifacts.log_dir
        checkpoint_dir = artifacts.checkpoint_dir
    else:
        base_dir = Path(args.save_dir)
        log_dir = base_dir / "logs"
        checkpoint_dir = base_dir / "checkpoints"

    summary = run_bc_warmstart_then_finetune(
        experiment_config=experiment_config,
        train_steps=args.train_steps,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        initial_actor_checkpoint=args.initial_actor_checkpoint,
        num_agents=args.num_agents,
        max_sampled_agents=args.max_sampled_agents,
        expert_episodes=args.expert_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        retain_failed_episodes=args.retain_failed_episodes,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_learning_rate=args.bc_learning_rate,
        bc_weight_decay=args.bc_weight_decay,
        bc_validation_split=args.bc_validation_split,
        bc_target_log_std=args.bc_target_log_std,
        bc_log_std_penalty_scale=args.bc_log_std_penalty_scale,
        bc_output_dir=args.bc_output_dir,
        bc_run_name=args.bc_run_name,
        save_dir=args.save_dir,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        log_every=args.log_every,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        selection_eval_episodes=args.selection_eval_episodes,
    )
    print(f"runtime_policy={RUNTIME_POLICY}")
    print(f"config_name={args.config_name}")
    print("multi-agent BC warm start + fine-tuning complete")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

