"""Run behavior cloning for the multi-agent actor from a saved expert dataset."""

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
    from multi_gate.imitation import run_actor_behavior_cloning

    parser = argparse.ArgumentParser(description="Train a BC warm-start actor for the multi-agent experiment.")
    parser.add_argument("--config-name", type=str, default="variable", choices=list_multi_experiment_config_names())
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--validation-split", type=float, default=None)
    parser.add_argument("--target-log-std", type=float, default=None)
    parser.add_argument("--log-std-penalty-scale", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--actor-checkpoint-name", type=str, default=None)
    args = parser.parse_args()

    summary = run_actor_behavior_cloning(
        dataset_path=args.dataset,
        experiment_config=get_multi_experiment_config(args.config_name),
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_split=args.validation_split,
        target_log_std=args.target_log_std,
        log_std_penalty_scale=args.log_std_penalty_scale,
        output_dir=args.output_dir,
        run_name=args.run_name,
        actor_checkpoint_name=args.actor_checkpoint_name,
    )
    print("multi-agent BC actor training complete")
    print(f"config_name={args.config_name}")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

