"""Collect heuristic expert trajectories for the multi-agent experiment."""

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
    from multi_gate.imitation import collect_expert_demonstrations

    parser = argparse.ArgumentParser(description="Collect heuristic expert trajectories for multi-agent BC warm start.")
    parser.add_argument("--config-name", type=str, default="variable", choices=list_multi_experiment_config_names())
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--max-sampled-agents", type=int, default=None)
    parser.add_argument("--expert-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--retain-failed-episodes", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    summary = collect_expert_demonstrations(
        experiment_config=get_multi_experiment_config(args.config_name),
        num_agents=args.num_agents,
        max_sampled_agents=args.max_sampled_agents,
        expert_episodes=args.expert_episodes,
        seed=args.seed,
        max_steps_per_episode=args.max_steps_per_episode,
        retain_failed_episodes=args.retain_failed_episodes,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    print("multi-agent expert collection complete")
    print(f"config_name={args.config_name}")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

