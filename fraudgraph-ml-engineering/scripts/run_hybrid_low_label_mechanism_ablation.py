from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_ml_engineering.experiment_protocol import (
    CHECKPOINT_MODES,
    aggregate_metric,
    dedupe_float_values,
    hybrid_checkpoint_path,
    hybrid_summary_path,
    label_fraction_slug,
    load_hybrid_summary,
    load_summary_payload,
    resolve_checkpoint_mode,
    resolve_seeds,
)
from fraud_ml_engineering.paths import ARTIFACTS_ROOT

RESULT_ROOT = ARTIFACTS_ROOT / "experiments" / "low_label_mechanism_ablation"
SUPPORTED_DATASETS = ("yelp", "amazon", "comp")
SUPPORTED_LABEL_FRACTIONS = (0.10, 0.05, 0.01)
MECHANISM_LADDER = {
    "graph_only_low_label": {
        "description": "Graph-only low-label baseline",
        "disable_gnn": False,
        "disable_transformer": True,
        "pure_label_fraction": True,
        "fusion_variant": "single_branch",
        "teacher_ema_decay_override": 0.0,
        "pseudo_warmup_rounds_override": 0,
        "pseudo_ramp_rounds_override": 0,
        "open_set_novelty_threshold_override": 0.0,
        "open_set_loss_weight_override": 0.0,
        "prototype_loss_weight_override": 0.0,
        "shared_private_loss_weight_override": 0.0,
        "context_alignment_loss_weight_override": 0.0,
        "uncertainty_loss_weight_override": 0.0,
        "active_learning_budget_per_round": 0,
        "active_learning_delay_rounds": 0,
    },
    "hybrid_pseudo": {
        "description": "Hybrid + pseudo-label",
        "disable_gnn": False,
        "disable_transformer": False,
        "pure_label_fraction": False,
        "fusion_variant": "graph_dominant_residual",
        "open_set_novelty_threshold_override": 0.0,
        "open_set_loss_weight_override": 0.0,
        "prototype_loss_weight_override": 0.0,
        "shared_private_loss_weight_override": 0.0,
        "context_alignment_loss_weight_override": 0.0,
        "uncertainty_loss_weight_override": 0.0,
        "active_learning_budget_per_round": 0,
        "active_learning_delay_rounds": 0,
    },
    "hybrid_pseudo_open_set": {
        "description": "Hybrid + pseudo-label + open-set",
        "disable_gnn": False,
        "disable_transformer": False,
        "pure_label_fraction": False,
        "fusion_variant": "graph_dominant_residual",
        "prototype_loss_weight_override": 0.0,
        "shared_private_loss_weight_override": 0.0,
        "context_alignment_loss_weight_override": 0.0,
        "uncertainty_loss_weight_override": 0.0,
        "active_learning_budget_per_round": 0,
        "active_learning_delay_rounds": 0,
    },
    "hybrid_pseudo_open_set_prototype": {
        "description": "Hybrid + pseudo-label + open-set + prototype",
        "disable_gnn": False,
        "disable_transformer": False,
        "pure_label_fraction": False,
        "fusion_variant": "shared_private_prototype",
        "prototype_loss_weight_override": 0.08,
        "shared_private_loss_weight_override": 0.05,
        "context_alignment_loss_weight_override": 0.0,
        "uncertainty_loss_weight_override": 0.04,
        "active_learning_budget_per_round": 0,
        "active_learning_delay_rounds": 0,
    },
}
@dataclass
class MechanismResult:
    dataset: str
    mechanism: str
    label_fraction: float
    seed: int
    result_root: Path
    summary_path: Path
    summary: dict[str, Any]
    reused: bool


def _progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the low-label mechanism ladder for the hybrid mainline.")
    parser.add_argument("--datasets", nargs="+", default=list(SUPPORTED_DATASETS), choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--mechanisms", nargs="+", default=list(MECHANISM_LADDER), choices=list(MECHANISM_LADDER))
    parser.add_argument("--label_fractions", nargs="+", type=float, default=list(SUPPORTED_LABEL_FRACTIONS))
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=-1, help="Deprecated single-seed fallback.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[30, 31, 40])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--checkpoint_mode", type=str, default="reuse", choices=list(CHECKPOINT_MODES))
    parser.add_argument("--disable_tb", action="store_true")
    parser.add_argument("--graph_warmup_rounds", type=int, default=-1)
    parser.add_argument("--fusion_bootstrap_rounds", type=int, default=-1)
    parser.add_argument("--modality_dropout", type=float, default=0.10)
    parser.add_argument("--graph_anchor_loss_weight", type=float, default=-1.0)
    parser.add_argument("--graph_anchor_temperature", type=float, default=-1.0)
    parser.add_argument("--graph_teacher_checkpoint_path", type=str, default="")
    parser.add_argument("--graph_teacher_distill_weight", type=float, default=0.0)
    parser.add_argument("--graph_teacher_temperature", type=float, default=1.5)
    return parser.parse_args()


def _summary_is_usable(
    summary: dict[str, Any],
    *,
    mechanism: str,
    seed: int,
    rounds: int | None,
    label_fraction: float,
    require_completed: bool = True,
) -> bool:
    config = MECHANISM_LADDER[mechanism]
    if require_completed and not bool(summary.get("completed")):
        return False
    if int(summary.get("seed", -1)) != int(seed):
        return False
    if rounds is not None and int(summary.get("rounds_ran", 0)) != int(rounds):
        return False
    if abs(float(summary.get("label_fraction", -1.0)) - float(label_fraction)) > 1e-12:
        return False
    if bool(summary.get("gnn_enabled", True)) != (not bool(config["disable_gnn"])):
        return False
    if bool(summary.get("transformer_enabled", True)) != (not bool(config["disable_transformer"])):
        return False
    if bool(summary.get("federated_enabled", True)):
        return False
    if bool(summary.get("drl_enabled", False)):
        return False
    if str(summary.get("planner_mode", "")).lower() != "deterministic":
        return False
    if bool(summary.get("pure_label_fraction", False)) != bool(config["pure_label_fraction"]):
        return False
    if str(summary.get("fusion_variant", "")).lower() != str(config["fusion_variant"]).lower():
        return False
    if mechanism == "graph_only_low_label":
        return float(summary.get("open_set_loss_weight", -1.0)) == 0.0
    if mechanism == "hybrid_pseudo":
        return (
            float(summary.get("open_set_loss_weight", -1.0)) == 0.0
            and float(summary.get("prototype_loss_weight", -1.0)) == 0.0
        )
    if mechanism == "hybrid_pseudo_open_set":
        return (
            float(summary.get("open_set_loss_weight", 0.0)) > 0.0
            and float(summary.get("prototype_loss_weight", -1.0)) == 0.0
        )
    if mechanism == "hybrid_pseudo_open_set_prototype":
        return (
            float(summary.get("open_set_loss_weight", 0.0)) > 0.0
            and float(summary.get("prototype_loss_weight", 0.0)) > 0.0
            and float(summary.get("shared_private_loss_weight", 0.0)) > 0.0
        )
    return True


def _summary_continue_compatible(
    summary: dict[str, Any],
    *,
    mechanism: str,
    seed: int,
    target_rounds: int,
    label_fraction: float,
) -> bool:
    existing_rounds = int(summary.get("rounds_ran", 0))
    if existing_rounds <= 0 or existing_rounds >= int(target_rounds):
        return False
    return _summary_is_usable(
        summary,
        mechanism=mechanism,
        seed=seed,
        rounds=None,
        label_fraction=label_fraction,
        require_completed=False,
    )


def _metric_bundle(summary: dict[str, Any]) -> dict[str, float]:
    test_metrics = dict(summary.get("test", {}) or {})
    return {
        "best_valid_auc": float(summary.get("best_valid_auc", 0.0)),
        "best_valid_f1_macro": float(summary.get("best_valid_f1_macro", 0.0)),
        "test_auc": float(test_metrics.get("auc", 0.0)),
        "test_f1_macro": float(test_metrics.get("f1_macro", 0.0)),
        "test_recall": float(test_metrics.get("recall", 0.0)),
    }


def _run_one(
    *,
    dataset_name: str,
    mechanism: str,
    label_fraction: float,
    seed: int,
    rounds: int,
    run_tag: str,
    force_rerun: bool,
    checkpoint_mode: str,
    enable_tensorboard: bool,
    graph_warmup_rounds: int,
    fusion_bootstrap_rounds: int,
    modality_dropout: float,
    graph_anchor_loss_weight: float,
    graph_anchor_temperature: float,
    graph_teacher_checkpoint_path: str,
    graph_teacher_distill_weight: float,
    graph_teacher_temperature: float,
    device: str,
) -> MechanismResult:
    config = MECHANISM_LADDER[mechanism]
    fraction_slug = label_fraction_slug(label_fraction)
    seed_dir = f"seed{seed}" if not str(run_tag).strip() else f"seed{seed}_{str(run_tag).strip()}"
    result_root = RESULT_ROOT / mechanism / f"label_fraction_{fraction_slug}" / seed_dir
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = hybrid_summary_path(result_root, dataset_name)
    effective_checkpoint_mode = resolve_checkpoint_mode(force_rerun, checkpoint_mode)
    summary_payload = None if effective_checkpoint_mode == "fresh" else load_summary_payload(summary_path)
    cached_summary = None
    if summary_payload is not None:
        cached_summary = summary_payload.get("summary", summary_payload)
        if not isinstance(cached_summary, dict):
            cached_summary = None
    if cached_summary is not None and _summary_is_usable(
        cached_summary,
        mechanism=mechanism,
        seed=seed,
        rounds=rounds,
        label_fraction=label_fraction,
    ):
        return MechanismResult(
            dataset=dataset_name,
            mechanism=mechanism,
            label_fraction=label_fraction,
            seed=seed,
            result_root=result_root,
            summary_path=summary_path,
            summary=cached_summary,
            reused=True,
        )

    resume_path = ""
    resume_round_offset = 0
    total_target_rounds = None
    preload_history = None
    if (
        effective_checkpoint_mode == "continue"
        and cached_summary is not None
        and _summary_continue_compatible(
            cached_summary,
            mechanism=mechanism,
            seed=seed,
            target_rounds=rounds,
            label_fraction=label_fraction,
        )
    ):
        checkpoint_path = hybrid_checkpoint_path(result_root, dataset_name)
        history_payload = summary_payload.get("history", []) if summary_payload is not None else []
        existing_rounds = int(cached_summary.get("rounds_ran", 0))
        if checkpoint_path.exists() and isinstance(history_payload, list) and len(history_payload) == existing_rounds:
            resume_path = str(checkpoint_path)
            resume_round_offset = existing_rounds
            total_target_rounds = int(rounds)
            preload_history = [dict(item) for item in history_payload if isinstance(item, dict)]

    training_config = {
        "federated_rounds": int(rounds) - int(resume_round_offset),
        "local_epochs": 2,
        "extra_local_epochs": 1,
        "dataset": dataset_name,
        "num_clients": 1,
        "label_fraction": float(label_fraction),
        "rl_timesteps": 0,
        "device": str(device),
        "enable_tensorboard": bool(enable_tensorboard),
        "planner_mode": "deterministic",
        "disable_federated": True,
        "disable_gnn": bool(config["disable_gnn"]),
        "disable_transformer": bool(config["disable_transformer"]),
        "pure_label_fraction": bool(config["pure_label_fraction"]),
        "result_root": str(result_root),
        "seed": int(seed),
        "fusion_variant_override": str(config["fusion_variant"]),
        "modality_dropout_prob_override": float(modality_dropout),
        "active_learning_budget_per_round": int(config["active_learning_budget_per_round"]),
        "active_learning_delay_rounds": int(config["active_learning_delay_rounds"]),
        "test_every": 0,
    }
    if resume_path:
        training_config["resume_path"] = resume_path
        training_config["resume_round_offset"] = int(resume_round_offset)
        training_config["total_target_rounds"] = int(total_target_rounds)
        training_config["preload_history"] = preload_history or []
    for override_key in (
        "teacher_ema_decay_override",
        "pseudo_warmup_rounds_override",
        "pseudo_ramp_rounds_override",
        "open_set_novelty_threshold_override",
        "open_set_loss_weight_override",
        "prototype_loss_weight_override",
        "shared_private_loss_weight_override",
        "context_alignment_loss_weight_override",
        "uncertainty_loss_weight_override",
    ):
        if override_key in config:
            training_config[override_key] = config[override_key]
    if int(graph_warmup_rounds) >= 0:
        training_config["graph_warmup_rounds_override"] = int(graph_warmup_rounds)
    if int(fusion_bootstrap_rounds) >= 0:
        training_config["fusion_bootstrap_rounds_override"] = int(fusion_bootstrap_rounds)
    if float(graph_anchor_loss_weight) >= 0.0:
        training_config["graph_anchor_loss_weight_override"] = float(graph_anchor_loss_weight)
    if float(graph_anchor_temperature) > 0.0:
        training_config["graph_anchor_temperature_override"] = float(graph_anchor_temperature)
    if str(graph_teacher_checkpoint_path).strip():
        training_config["graph_teacher_checkpoint_path"] = str(graph_teacher_checkpoint_path).strip()
    if float(graph_teacher_distill_weight) > 0.0:
        training_config["graph_teacher_distill_weight"] = float(graph_teacher_distill_weight)
    if float(graph_teacher_temperature) > 0.0:
        training_config["graph_teacher_temperature"] = float(graph_teacher_temperature)

    from fraud_ml_engineering.algorithms import run_hybrid_fraud_training

    summaries = run_hybrid_fraud_training(**training_config)
    summary = summaries[dataset_name]
    loaded_summary = load_hybrid_summary(summary_path) or summary
    if not _summary_is_usable(
        loaded_summary,
        mechanism=mechanism,
        seed=seed,
        rounds=rounds,
        label_fraction=label_fraction,
    ):
        raise RuntimeError(
            f"Completed mechanism run mismatch for dataset={dataset_name}, mechanism={mechanism}, "
            f"seed={seed}, label_fraction={label_fraction}."
        )
    return MechanismResult(
        dataset=dataset_name,
        mechanism=mechanism,
        label_fraction=label_fraction,
        seed=seed,
        result_root=result_root,
        summary_path=summary_path,
        summary=loaded_summary,
        reused=False,
    )


def _aggregate_results(results: list[MechanismResult]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for dataset_name in SUPPORTED_DATASETS:
        dataset_items = [item for item in results if item.dataset == dataset_name]
        if not dataset_items:
            continue
        aggregate[dataset_name] = {}
        fractions = sorted({float(item.label_fraction) for item in dataset_items}, reverse=True)
        for label_fraction in fractions:
            fraction_items = [item for item in dataset_items if abs(float(item.label_fraction) - label_fraction) <= 1e-12]
            fraction_key = format(label_fraction, ".12g")
            aggregate[dataset_name][fraction_key] = {}
            for mechanism in MECHANISM_LADDER:
                mechanism_items = [item for item in fraction_items if item.mechanism == mechanism]
                if not mechanism_items:
                    continue
                metric_payloads = [_metric_bundle(item.summary) for item in mechanism_items]
                aggregate[dataset_name][fraction_key][mechanism] = {
                    "description": str(MECHANISM_LADDER[mechanism]["description"]),
                    "num_runs": len(mechanism_items),
                    "seeds": [int(item.seed) for item in mechanism_items],
                    "best_valid_auc": aggregate_metric([item["best_valid_auc"] for item in metric_payloads]),
                    "best_valid_f1_macro": aggregate_metric([item["best_valid_f1_macro"] for item in metric_payloads]),
                    "test_auc": aggregate_metric([item["test_auc"] for item in metric_payloads]),
                    "test_f1_macro": aggregate_metric([item["test_f1_macro"] for item in metric_payloads]),
                    "test_recall": aggregate_metric([item["test_recall"] for item in metric_payloads]),
                    "summary_paths": [str(item.summary_path) for item in mechanism_items],
                }
    return aggregate


def _render_markdown(
    *,
    started_at: str,
    finished_at: str,
    rounds: int,
    seeds: list[int],
    label_fractions: list[float],
    aggregate: dict[str, Any],
) -> str:
    lines = [
        "# Low-Label Mechanism Ablation",
        "",
        f"- started_at: `{started_at}`",
        f"- finished_at: `{finished_at}`",
        f"- rounds: `{rounds}`",
        f"- seeds: `{seeds}`",
        f"- label_fractions: `{label_fractions}`",
        f"- datasets: `{list(SUPPORTED_DATASETS)}`",
        f"- mechanisms: `{list(MECHANISM_LADDER)}`",
        "",
    ]
    for dataset_name, dataset_payload in aggregate.items():
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append("| label_fraction | mechanism | description | best_valid_auc | test_auc | test_f1_macro | test_recall |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for label_fraction in sorted(dataset_payload.keys(), key=lambda item: float(item), reverse=True):
            fraction_payload = dataset_payload[label_fraction]
            for mechanism in MECHANISM_LADDER:
                payload = fraction_payload.get(mechanism)
                if payload is None:
                    lines.append(f"| {label_fraction} | {mechanism} | - | - | - | - | - |")
                    continue
                lines.append(
                    f"| {label_fraction} | {mechanism} | {payload['description']} | "
                    f"{payload['best_valid_auc']['mean']:.6f} +/- {payload['best_valid_auc']['std']:.6f} | "
                    f"{payload['test_auc']['mean']:.6f} +/- {payload['test_auc']['std']:.6f} | "
                    f"{payload['test_f1_macro']['mean']:.6f} +/- {payload['test_f1_macro']['std']:.6f} | "
                    f"{payload['test_recall']['mean']:.6f} +/- {payload['test_recall']['std']:.6f} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    seeds = resolve_seeds(deprecated_seed=int(args.seed), seeds=args.seeds)
    label_fractions = dedupe_float_values(args.label_fractions)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    results: list[MechanismResult] = []
    jobs = [
        (dataset_name, mechanism, label_fraction, seed)
        for dataset_name in args.datasets
        for mechanism in args.mechanisms
        for label_fraction in label_fractions
        for seed in seeds
    ]
    _progress(
        f"low-label mechanism ablation started: jobs={len(jobs)} datasets={list(args.datasets)} "
        f"mechanisms={list(args.mechanisms)} rounds={int(args.rounds)}"
    )

    for job_index, (dataset_name, mechanism, label_fraction, seed) in enumerate(jobs, start=1):
        _progress(
            f"job {job_index}/{len(jobs)}: dataset={dataset_name} mechanism={mechanism} "
            f"label_fraction={label_fraction:.4f} seed={seed}"
        )
        started = time.perf_counter()
        result = _run_one(
            dataset_name=str(dataset_name),
            mechanism=str(mechanism),
            label_fraction=float(label_fraction),
            seed=int(seed),
            rounds=int(args.rounds),
            run_tag=str(args.run_tag),
            force_rerun=bool(args.force_rerun),
            checkpoint_mode=str(args.checkpoint_mode),
            enable_tensorboard=not bool(args.disable_tb),
            graph_warmup_rounds=int(args.graph_warmup_rounds),
            fusion_bootstrap_rounds=int(args.fusion_bootstrap_rounds),
            modality_dropout=float(args.modality_dropout),
            graph_anchor_loss_weight=float(args.graph_anchor_loss_weight),
            graph_anchor_temperature=float(args.graph_anchor_temperature),
            graph_teacher_checkpoint_path=str(args.graph_teacher_checkpoint_path),
            graph_teacher_distill_weight=float(args.graph_teacher_distill_weight),
            graph_teacher_temperature=float(args.graph_teacher_temperature),
            device=str(args.device),
        )
        results.append(result)
        metrics = _metric_bundle(result.summary)
        _progress(
            f"job {job_index}/{len(jobs)} finished in {time.perf_counter() - started:.1f}s: "
            f"reused={result.reused} test_auc={metrics['test_auc']:.6f} test_f1_macro={metrics['test_f1_macro']:.6f}"
        )

    aggregate = _aggregate_results(results)
    finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "protocol": {
            "datasets": list(args.datasets),
            "mechanisms": list(args.mechanisms),
            "label_fractions": label_fractions,
            "rounds": int(args.rounds),
            "seeds": seeds,
            "device": str(args.device),
            "checkpoint_mode": resolve_checkpoint_mode(bool(args.force_rerun), str(args.checkpoint_mode)),
            "graph_warmup_rounds": int(args.graph_warmup_rounds),
            "fusion_bootstrap_rounds": int(args.fusion_bootstrap_rounds),
            "modality_dropout": float(args.modality_dropout),
        },
        "records": [
            {
                "dataset": str(item.dataset),
                "mechanism": str(item.mechanism),
                "label_fraction": float(item.label_fraction),
                "seed": int(item.seed),
                "reused": bool(item.reused),
                "summary_path": str(item.summary_path),
                "result_root": str(item.result_root),
                "best_valid_auc": float(item.summary.get("best_valid_auc", 0.0)),
                "test_auc": float(item.summary.get("test", {}).get("auc", 0.0)),
                "test_f1_macro": float(item.summary.get("test", {}).get("f1_macro", 0.0)),
            }
            for item in results
        ],
        "aggregate": aggregate,
    }

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_json_path = RESULT_ROOT / "low_label_mechanism_ablation_summary.json"
    summary_md_path = RESULT_ROOT / "low_label_mechanism_ablation_summary.md"
    summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md_path.write_text(
        _render_markdown(
            started_at=started_at,
            finished_at=finished_at,
            rounds=int(args.rounds),
            seeds=seeds,
            label_fractions=label_fractions,
            aggregate=aggregate,
        ),
        encoding="utf-8",
    )
    _progress(f"low-label mechanism ablation finished: json={summary_json_path}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON: {summary_json_path}")
    print(f"Saved Markdown: {summary_md_path}")


if __name__ == "__main__":
    main()
