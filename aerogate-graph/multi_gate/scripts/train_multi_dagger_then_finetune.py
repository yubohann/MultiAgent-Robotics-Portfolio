"""Run DAgger-style BC correction followed by Graph-FlashSAC fine-tuning."""

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
    from multi_gate.dagger import run_dagger_warmstart_then_finetune
    from shared.runtime.artifacts import RUNTIME_POLICY, allocate_training_artifacts

    parser = argparse.ArgumentParser(description="Run multi-agent DAgger warm start -> Graph-FlashSAC fine-tuning.")
    parser.add_argument("--config-name", type=str, default="variable", choices=list_multi_experiment_config_names())
    parser.add_argument("--train-steps", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--max-sampled-agents", type=int, default=None)
    parser.add_argument("--initial-actor-checkpoint", type=str, default=None)
    parser.add_argument("--expert-episodes", type=int, default=None)
    parser.add_argument("--initial-bc-epochs", type=int, default=None)
    parser.add_argument("--refresh-initial-bc", action="store_true")
    parser.add_argument("--dagger-iterations", type=int, default=None)
    parser.add_argument("--dagger-rollout-episodes", type=int, default=None)
    parser.add_argument("--dagger-bc-epochs", type=int, default=None)
    parser.add_argument("--dagger-output-dir", type=str, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--updates-per-step", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=64)
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

    summary = run_dagger_warmstart_then_finetune(
        experiment_config=experiment_config,
        train_steps=args.train_steps,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        num_agents=args.num_agents,
        max_sampled_agents=args.max_sampled_agents,
        initial_actor_checkpoint=args.initial_actor_checkpoint,
        expert_episodes=args.expert_episodes,
        initial_bc_epochs=args.initial_bc_epochs,
        refresh_initial_bc=bool(args.refresh_initial_bc),
        dagger_iterations=args.dagger_iterations,
        dagger_rollout_episodes=args.dagger_rollout_episodes,
        dagger_bc_epochs=args.dagger_bc_epochs,
        output_dir=args.dagger_output_dir,
        run_name=args.run_name,
        save_dir=args.save_dir,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        log_every=args.log_every,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        selection_eval_episodes=args.selection_eval_episodes,
    )
    print(f"runtime_policy={RUNTIME_POLICY}")
    print(f"config_name={args.config_name}")
    print("multi-agent DAgger warm start + fine-tuning complete")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

