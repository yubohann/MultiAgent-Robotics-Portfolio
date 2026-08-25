"""Python-level batched evaluation helpers for multi-agent aerogate_graph."""

from __future__ import annotations

from pathlib import Path

from multi_gate.configs.experiment_config import MULTI_EXPERIMENT_CONFIG, MultiExperimentConfig
from multi_gate.training import evaluate_checkpoint


def evaluate_checkpoint_batched(
    *,
    checkpoint_path: str | Path,
    experiment_config: MultiExperimentConfig | None = None,
    team_sizes: tuple[int, ...] | list[int],
    seeds: tuple[int, ...] | list[int],
    episodes: int = 1,
    device: str | None = None,
) -> dict[str, object]:
    """Evaluate many team-size/seed jobs through one explicit batch API."""

    selected_config = experiment_config or MULTI_EXPERIMENT_CONFIG
    jobs: list[dict[str, object]] = []
    total_episodes = 0
    weighted_successes = 0.0
    weighted_rewards = 0.0
    for team_size in team_sizes:
        for seed in seeds:
            summary = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                episodes=int(episodes),
                seed=int(seed),
                device=device,
                num_agents=int(team_size),
                experiment_config=selected_config,
            )
            episode_count = int(summary.get("episodes") or 0)
            total_episodes += episode_count
            weighted_successes += float(summary.get("success_rate") or 0.0) * episode_count
            weighted_rewards += float(summary.get("mean_episode_reward") or 0.0) * episode_count
            jobs.append(
                {
                    "team_size": int(team_size),
                    "seed": int(seed),
                    "summary": summary,
                }
            )
    return {
        "checkpoint_path": str(checkpoint_path),
        "experiment_id": selected_config.experiment_id,
        "jobs": jobs,
        "job_count": len(jobs),
        "episodes": int(total_episodes),
        "success_rate": weighted_successes / max(total_episodes, 1),
        "mean_episode_reward": weighted_rewards / max(total_episodes, 1),
        "notes": "Python-level batched evaluation; keeps the single-env API unchanged for compatibility.",
    }

