"""Checkpoint aliasing, selection scoring, and formal promotion gates."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Callable


class PromotionGateError(RuntimeError):
    """Raised when a trained checkpoint fails the formal promotion gate."""


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_min(values: list[float]) -> float:
    return min(values) if values else 0.0


def _episode_dicts(eval_summary: dict[str, object]) -> list[dict[str, object]]:
    return [episode for episode in list(eval_summary.get("episode_summaries") or []) if isinstance(episode, dict)]


def _reason_rate(done_reason_counts: dict[str, int], reason: str, episodes: int) -> float:
    return float(done_reason_counts.get(reason, 0)) / max(int(episodes), 1)


def _float_values(
    episodes: list[dict[str, object]],
    key: str,
    *,
    filter_reason: str | None = None,
) -> list[float]:
    values: list[float] = []
    for episode in episodes:
        if filter_reason is not None and str(episode.get("done_reason") or "") != filter_reason:
            continue
        value = episode.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _summary_float(eval_summary: dict[str, object], key: str) -> float | None:
    value = eval_summary.get(key)
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved):
        return None
    return resolved


def _single_selection_details(eval_summary: dict[str, object]) -> dict[str, object]:
    episodes = max(int(eval_summary.get("episodes") or 0), 1)
    done_reason_counts = {
        str(key): int(value)
        for key, value in dict(eval_summary.get("done_reason_counts") or {}).items()
    }
    episode_summaries = _episode_dicts(eval_summary)
    success_rate = float(eval_summary.get("success_rate") or 0.0)
    collision_rate = _reason_rate(done_reason_counts, "collision", episodes)
    out_of_bounds_rate = _reason_rate(done_reason_counts, "out_of_bounds", episodes)
    timeout_rate = _reason_rate(done_reason_counts, "timeout", episodes)
    timeout_counts_as_success = bool(eval_summary.get("timeout_counts_as_success") or False)
    collision_out_of_bounds_rate = collision_rate + out_of_bounds_rate
    mean_episode_reward = float(eval_summary.get("mean_episode_reward") or 0.0)
    mean_steps = _safe_mean(_float_values(episode_summaries, "steps"))
    mean_success_steps = _safe_mean(_float_values(episode_summaries, "steps", filter_reason="goal_reached"))
    mean_goal_distance_m = _safe_mean(_float_values(episode_summaries, "goal_distance_m"))
    mean_signed_clearance_m = _safe_mean(_float_values(episode_summaries, "signed_clearance_m"))
    min_signed_clearance_m = _safe_min(_float_values(episode_summaries, "signed_clearance_m"))
    success_mean_clearance_m = _safe_mean(
        _float_values(episode_summaries, "signed_clearance_m", filter_reason="goal_reached")
    )
    timeout_term = 0.0 if timeout_counts_as_success else (-timeout_rate * 120_000.0)
    score = (
        success_rate * 1_000_000.0
        - collision_out_of_bounds_rate * 400_000.0
        + timeout_term
        + mean_episode_reward * 1_000.0
        + success_mean_clearance_m * 350.0
        + mean_signed_clearance_m * 120.0
        - mean_success_steps * 18.0
        - mean_steps * 4.0
        - mean_goal_distance_m * 120.0
        + min_signed_clearance_m * 40.0
    )
    return {
        "task_type": "single",
        "score": float(score),
        "metrics": {
            "episodes": episodes,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "out_of_bounds_rate": out_of_bounds_rate,
            "timeout_rate": timeout_rate,
            "timeout_counts_as_success": timeout_counts_as_success,
            "collision_out_of_bounds_rate": collision_out_of_bounds_rate,
            "mean_episode_reward": mean_episode_reward,
            "mean_steps": mean_steps,
            "mean_success_steps": mean_success_steps,
            "mean_goal_distance_m": mean_goal_distance_m,
            "mean_signed_clearance_m": mean_signed_clearance_m,
            "min_signed_clearance_m": min_signed_clearance_m,
            "success_mean_clearance_m": success_mean_clearance_m,
        },
    }


def _count_multi_safety_violations(episode_summaries: list[dict[str, object]]) -> int:
    count = 0
    for episode in episode_summaries:
        reason = str(episode.get("done_reason") or "")
        min_clearance_m = float(episode.get("min_clearance_m") or 0.0)
        min_pair_distance_m = float(episode.get("min_pair_distance_m") or float("inf"))
        if reason in {"gate_post_collision", "agent_collision", "out_of_bounds"}:
            count += 1
        elif min_clearance_m < 0.5 or min_pair_distance_m < 1.2:
            count += 1
    return count


def _multi_selection_details(eval_summary: dict[str, object]) -> dict[str, object]:
    episodes = max(int(eval_summary.get("episodes") or 0), 1)
    done_reason_counts = {
        str(key): int(value)
        for key, value in dict(eval_summary.get("done_reason_counts") or {}).items()
    }
    episode_summaries = _episode_dicts(eval_summary)
    success_rate = float(eval_summary.get("success_rate") or 0.0)
    mean_episode_reward = float(eval_summary.get("mean_episode_reward") or 0.0)
    gate_post_collision_rate = _reason_rate(done_reason_counts, "gate_post_collision", episodes)
    agent_collision_rate = _reason_rate(done_reason_counts, "agent_collision", episodes)
    out_of_bounds_rate = _reason_rate(done_reason_counts, "out_of_bounds", episodes)
    timeout_rate = _reason_rate(done_reason_counts, "timeout", episodes)
    timeout_counts_as_success = bool(eval_summary.get("timeout_counts_as_success") or False)
    hard_failure_rate = gate_post_collision_rate + agent_collision_rate + out_of_bounds_rate
    safety_violation_rate_value = eval_summary.get("safety_violation_rate")
    if safety_violation_rate_value is None:
        safety_violation_rate = _count_multi_safety_violations(episode_summaries) / episodes
    else:
        safety_violation_rate = float(safety_violation_rate_value)
    min_bucket_success_rate_value = eval_summary.get("min_bucket_success_rate")
    min_bucket_success_rate = (
        None if min_bucket_success_rate_value is None else float(min_bucket_success_rate_value)
    )
    mean_steps = _summary_float(eval_summary, "mean_steps")
    if mean_steps is None:
        mean_steps = _safe_mean(_float_values(episode_summaries, "steps"))
    mean_goal_distance_m = _summary_float(eval_summary, "mean_goal_distance_m")
    if mean_goal_distance_m is None:
        mean_goal_distance_m = _safe_mean(_float_values(episode_summaries, "goal_distance_m"))
    mean_slot_error_m = _summary_float(eval_summary, "mean_slot_error_m")
    if mean_slot_error_m is None:
        mean_slot_error_m = _safe_mean(_float_values(episode_summaries, "mean_slot_error_m"))
    mean_min_clearance_m = _summary_float(eval_summary, "mean_min_clearance_m")
    if mean_min_clearance_m is None:
        mean_min_clearance_m = _safe_mean(_float_values(episode_summaries, "min_clearance_m"))
    min_min_clearance_m = _summary_float(eval_summary, "min_min_clearance_m")
    if min_min_clearance_m is None:
        min_min_clearance_m = _safe_min(_float_values(episode_summaries, "min_clearance_m"))
    mean_min_pair_distance_m = _summary_float(eval_summary, "mean_min_pair_distance_m")
    if mean_min_pair_distance_m is None:
        mean_min_pair_distance_m = _safe_mean(_float_values(episode_summaries, "min_pair_distance_m"))
    min_min_pair_distance_m = _summary_float(eval_summary, "min_min_pair_distance_m")
    if min_min_pair_distance_m is None:
        min_min_pair_distance_m = _safe_min(_float_values(episode_summaries, "min_pair_distance_m"))
    bucket_term = 0.0 if min_bucket_success_rate is None else min_bucket_success_rate * 500_000.0
    timeout_term = 0.0 if timeout_counts_as_success else (-timeout_rate * 80_000.0)
    score = (
        success_rate * 1_000_000.0
        + bucket_term
        - hard_failure_rate * 420_000.0
        - agent_collision_rate * 180_000.0
        - safety_violation_rate * 180_000.0
        + timeout_term
        # Keep raw return informative, but do not let curriculum-specific reward scale dominate
        # bucket success and hard-safety signals during checkpoint selection.
        + mean_episode_reward * 200.0
        - mean_slot_error_m * 600.0
        + mean_min_clearance_m * 240.0
        + min_min_clearance_m * 70.0
        + mean_min_pair_distance_m * 60.0
        + min_min_pair_distance_m * 20.0
        - mean_steps * 10.0
        - mean_goal_distance_m * 80.0
    )
    return {
        "task_type": "multi",
        "score": float(score),
        "metrics": {
            "episodes": episodes,
            "success_rate": success_rate,
            "gate_post_collision_rate": gate_post_collision_rate,
            "agent_collision_rate": agent_collision_rate,
            "out_of_bounds_rate": out_of_bounds_rate,
            "timeout_rate": timeout_rate,
            "timeout_counts_as_success": timeout_counts_as_success,
            "hard_failure_rate": hard_failure_rate,
            "safety_violation_rate": safety_violation_rate,
            "min_bucket_success_rate": min_bucket_success_rate,
            "mean_episode_reward": mean_episode_reward,
            "mean_steps": mean_steps,
            "mean_goal_distance_m": mean_goal_distance_m,
            "mean_slot_error_m": mean_slot_error_m,
            "mean_min_clearance_m": mean_min_clearance_m,
            "min_min_clearance_m": min_min_clearance_m,
            "mean_min_pair_distance_m": mean_min_pair_distance_m,
            "min_min_pair_distance_m": min_min_pair_distance_m,
        },
    }


def build_checkpoint_selection_details(eval_summary: dict[str, object]) -> dict[str, object]:
    """Build a richer, task-aware checkpoint selection summary."""

    if (
        "bucket_evaluation" in eval_summary
        or "num_agents" in eval_summary
        or "min_bucket_success_rate" in eval_summary
        or "safety_violation_rate" in eval_summary
        or "mean_slot_error_m" in eval_summary
    ):
        return _multi_selection_details(eval_summary)
    return _single_selection_details(eval_summary)


def compute_checkpoint_selection_score(eval_summary: dict[str, object]) -> float:
    """Score one checkpoint using robust deterministic evaluation results."""

    return float(build_checkpoint_selection_details(eval_summary)["score"])


def refresh_best_checkpoint_alias(
    checkpoint_path: str | Path,
    *,
    checkpoint_dir: str | Path,
    alias_name: str = "best_agent.pt",
) -> Path:
    """Refresh one checkpoint alias by copying one selected artifact."""

    source_path = Path(checkpoint_path)
    target_path = Path(checkpoint_dir) / alias_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != target_path.resolve() and target_path.exists():
        target_path.unlink()
    # Preserve the selected artifact contents, but let the alias timestamp
    # reflect when the alias itself was refreshed.
    shutil.copy(source_path, target_path)
    return target_path


def shortlist_checkpoint_records(
    checkpoint_records: list[dict[str, object]],
    *,
    shortlist_size: int,
) -> list[dict[str, object]]:
    """Return the top unique checkpoint records ranked by their selection score."""

    unique_records: dict[str, dict[str, object]] = {}
    for record in checkpoint_records:
        checkpoint_path = str(record.get("checkpoint_path") or "").strip()
        if not checkpoint_path:
            continue
        selection_score = float(record.get("selection_score") or float("-inf"))
        existing = unique_records.get(checkpoint_path)
        if existing is None or selection_score > float(existing.get("selection_score") or float("-inf")):
            unique_records[checkpoint_path] = record

    sorted_records = sorted(
        unique_records.values(),
        key=lambda item: float(item.get("selection_score") or float("-inf")),
        reverse=True,
    )
    return sorted_records[: max(int(shortlist_size), 1)]


def reselect_best_checkpoint_alias(
    *,
    checkpoint_records: list[dict[str, object]],
    checkpoint_dir: str | Path,
    alias_name: str,
    shortlist_size: int,
    final_eval_episodes: int,
    evaluate_summary_fn: Callable[[str, int], dict[str, object]],
    report_path: str | Path | None = None,
) -> dict[str, object] | None:
    """Run a stricter post-training best-checkpoint reselection pass."""

    resolved_final_eval_episodes = max(int(final_eval_episodes), 0)
    if resolved_final_eval_episodes <= 0:
        return None

    shortlist = shortlist_checkpoint_records(
        checkpoint_records,
        shortlist_size=shortlist_size,
    )
    if not shortlist:
        return None

    shortlist_results: list[dict[str, object]] = []
    final_results: list[dict[str, object]] = []
    for shortlist_rank, record in enumerate(shortlist, start=1):
        checkpoint_path = str(record["checkpoint_path"])
        shortlist_details = dict(record.get("selection_details") or {})
        shortlist_metrics = dict(shortlist_details.get("metrics") or {})
        shortlist_results.append(
            {
                "rank": int(shortlist_rank),
                "checkpoint_path": checkpoint_path,
                "shortlist_score": float(record.get("selection_score") or float("-inf")),
                "shortlist_eval_episodes": int(record.get("selection_eval_episodes") or 0),
                "shortlist_metrics": shortlist_metrics,
            }
        )

        final_eval_summary = evaluate_summary_fn(checkpoint_path, resolved_final_eval_episodes)
        final_selection_details = build_checkpoint_selection_details(final_eval_summary)
        final_results.append(
            {
                "rank_from_shortlist": int(shortlist_rank),
                "checkpoint_path": checkpoint_path,
                "final_eval_episodes": int(resolved_final_eval_episodes),
                "final_score": float(final_selection_details["score"]),
                "task_type": str(final_selection_details.get("task_type") or ""),
                "final_metrics": dict(final_selection_details.get("metrics") or {}),
            }
        )

    final_results.sort(key=lambda item: float(item["final_score"]), reverse=True)
    selected_result = final_results[0]
    refreshed_alias = refresh_best_checkpoint_alias(
        selected_result["checkpoint_path"],
        checkpoint_dir=checkpoint_dir,
        alias_name=alias_name,
    )

    report = {
        "checkpoint_dir": str(Path(checkpoint_dir)),
        "alias_name": str(alias_name),
        "shortlist_count": len(shortlist_results),
        "shortlist": shortlist_results,
        "final_results": final_results,
        "selected_checkpoint_path": str(selected_result["checkpoint_path"]),
        "selected_score": float(selected_result["final_score"]),
        "final_eval_episodes": int(resolved_final_eval_episodes),
        "best_alias_path": str(refreshed_alias),
    }
    if report_path is not None:
        resolved_report_path = Path(report_path)
        resolved_report_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        report["report_path"] = str(resolved_report_path)
    return report


def assess_single_promotion_gate(
    eval_summary: dict[str, object],
    gate_config: object,
    *,
    override_min_success_rate: float = 0.0,
) -> dict[str, object]:
    """Check the single-agent promotion gate."""

    episodes = max(int(eval_summary.get("episodes") or 0), 1)
    success_rate = float(eval_summary.get("success_rate") or 0.0)
    mean_episode_reward = float(eval_summary.get("mean_episode_reward") or 0.0)
    done_reason_counts = {
        str(key): int(value)
        for key, value in dict(eval_summary.get("done_reason_counts") or {}).items()
    }
    collision_out_of_bounds_rate = (
        done_reason_counts.get("collision", 0) + done_reason_counts.get("out_of_bounds", 0)
    ) / episodes
    effective_min_success_rate = max(
        float(getattr(gate_config, "min_success_rate", 0.0) or 0.0),
        float(override_min_success_rate or 0.0),
    )
    min_mean_episode_reward = getattr(gate_config, "min_mean_episode_reward", None)
    checks = {
        "success_rate": success_rate >= effective_min_success_rate,
        "collision_out_of_bounds_rate": collision_out_of_bounds_rate <= float(
            getattr(gate_config, "max_collision_out_of_bounds_rate", 1.0) or 1.0
        ),
        "mean_episode_reward": (
            True
            if min_mean_episode_reward is None
            else mean_episode_reward >= float(min_mean_episode_reward)
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "enabled": bool(getattr(gate_config, "enabled", True)),
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "episodes": episodes,
        "success_rate": success_rate,
        "effective_min_success_rate": effective_min_success_rate,
        "collision_out_of_bounds_rate": collision_out_of_bounds_rate,
        "max_collision_out_of_bounds_rate": float(
            getattr(gate_config, "max_collision_out_of_bounds_rate", 1.0) or 1.0
        ),
        "mean_episode_reward": mean_episode_reward,
        "min_mean_episode_reward": (
            None if min_mean_episode_reward is None else float(min_mean_episode_reward)
        ),
        "done_reason_counts": done_reason_counts,
    }


def assess_multi_promotion_gate(
    eval_summary: dict[str, object],
    gate_config: object,
    *,
    override_min_success_rate: float = 0.0,
) -> dict[str, object]:
    """Check the multi-agent promotion gate."""

    episodes = max(int(eval_summary.get("episodes") or 0), 1)
    success_rate = float(eval_summary.get("success_rate") or 0.0)
    mean_episode_reward = float(eval_summary.get("mean_episode_reward") or 0.0)
    done_reason_counts = {
        str(key): int(value)
        for key, value in dict(eval_summary.get("done_reason_counts") or {}).items()
    }
    agent_collision_rate = done_reason_counts.get("agent_collision", 0) / episodes
    hard_failure_rate = (
        done_reason_counts.get("gate_post_collision", 0)
        + done_reason_counts.get("agent_collision", 0)
        + done_reason_counts.get("out_of_bounds", 0)
    ) / episodes
    safety_violation_rate = eval_summary.get("safety_violation_rate")
    safety_violation_rate = None if safety_violation_rate is None else float(safety_violation_rate)
    min_bucket_success_rate = eval_summary.get("min_bucket_success_rate")
    min_bucket_success_rate = None if min_bucket_success_rate is None else float(min_bucket_success_rate)
    effective_min_success_rate = max(
        float(getattr(gate_config, "min_success_rate", 0.0) or 0.0),
        float(override_min_success_rate or 0.0),
    )
    min_mean_episode_reward = getattr(gate_config, "min_mean_episode_reward", None)
    max_safety_violation_rate = getattr(gate_config, "max_safety_violation_rate", None)
    gate_min_bucket_success_rate = getattr(gate_config, "min_bucket_success_rate", None)
    checks = {
        "success_rate": success_rate >= effective_min_success_rate,
        "agent_collision_rate": agent_collision_rate <= float(
            getattr(gate_config, "max_agent_collision_rate", 1.0) or 1.0
        ),
        "hard_failure_rate": hard_failure_rate <= float(
            getattr(gate_config, "max_hard_failure_rate", 1.0) or 1.0
        ),
        "mean_episode_reward": (
            True
            if min_mean_episode_reward is None
            else mean_episode_reward >= float(min_mean_episode_reward)
        ),
        "safety_violation_rate": (
            True
            if max_safety_violation_rate is None or safety_violation_rate is None
            else safety_violation_rate <= float(max_safety_violation_rate)
        ),
        "bucket_success_rate": (
            True
            if gate_min_bucket_success_rate is None or min_bucket_success_rate is None
            else min_bucket_success_rate >= float(gate_min_bucket_success_rate)
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "enabled": bool(getattr(gate_config, "enabled", True)),
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "episodes": episodes,
        "success_rate": success_rate,
        "effective_min_success_rate": effective_min_success_rate,
        "agent_collision_rate": agent_collision_rate,
        "max_agent_collision_rate": float(getattr(gate_config, "max_agent_collision_rate", 1.0) or 1.0),
        "hard_failure_rate": hard_failure_rate,
        "max_hard_failure_rate": float(getattr(gate_config, "max_hard_failure_rate", 1.0) or 1.0),
        "safety_violation_rate": safety_violation_rate,
        "max_safety_violation_rate": (
            None if max_safety_violation_rate is None else float(max_safety_violation_rate)
        ),
        "min_bucket_success_rate": min_bucket_success_rate,
        "required_min_bucket_success_rate": (
            None if gate_min_bucket_success_rate is None else float(gate_min_bucket_success_rate)
        ),
        "mean_episode_reward": mean_episode_reward,
        "min_mean_episode_reward": (
            None if min_mean_episode_reward is None else float(min_mean_episode_reward)
        ),
        "done_reason_counts": done_reason_counts,
    }

