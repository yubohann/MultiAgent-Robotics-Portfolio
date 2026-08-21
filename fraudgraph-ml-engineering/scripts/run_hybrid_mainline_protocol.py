from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_ml_engineering.experiment_protocol import (
    CHECKPOINT_MODES,
    hybrid_checkpoint_path,
    hybrid_summary_path,
    label_fraction_slug,
    load_summary_payload,
    mean_std_metric,
    resolve_checkpoint_mode,
)
from fraud_ml_engineering.paths import ARTIFACTS_ROOT

RESULT_ROOT = ARTIFACTS_ROOT / "experiments" / "mainline_protocol"
EXPERIMENTS = {
    "full": {
        "disable_gnn": False,
        "disable_transformer": False,
        "description": "SplitGNN + Transformer",
    },
    "no_gnn": {
        "disable_gnn": True,
        "disable_transformer": False,
        "description": "Transformer only",
    },
    "no_transformer": {
        "disable_gnn": False,
        "disable_transformer": True,
        "description": "SplitGNN only",
    },
}
SUPPORTED_FUSION_VARIANTS = ("graph_dominant_residual", "late_fusion", "shared_private_prototype")
def _progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen mainline protocol for centralized SplitGNN + Transformer experiments."
    )
    parser.add_argument("--datasets", nargs="+", default=["yelp", "amazon", "comp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[30, 31, 40])
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--extra_local_epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--label_fractions", nargs="+", type=float, default=[0.10, 0.05, 0.01])
    parser.add_argument(
        "--full_supervision_experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=["full", "no_gnn", "no_transformer"],
    )
    parser.add_argument(
        "--low_label_experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=["full", "no_transformer"],
    )
    parser.add_argument("--skip_full_supervision", action="store_true")
    parser.add_argument("--skip_low_label", action="store_true")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--checkpoint_mode", type=str, default="reuse", choices=list(CHECKPOINT_MODES))
    parser.add_argument("--disable_tb", action="store_true")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument(
        "--fusion_variant",
        type=str,
        default="graph_dominant_residual",
        choices=list(SUPPORTED_FUSION_VARIANTS),
    )
    parser.add_argument("--graph_warmup_rounds", type=int, default=-1)
    parser.add_argument("--fusion_bootstrap_rounds", type=int, default=-1)
    parser.add_argument("--modality_dropout", type=float, default=0.10)
    parser.add_argument("--graph_anchor_loss_weight", type=float, default=-1.0)
    parser.add_argument("--graph_anchor_temperature", type=float, default=-1.0)
    parser.add_argument("--graph_teacher_checkpoint_path", type=str, default="")
    parser.add_argument("--graph_teacher_distill_weight", type=float, default=0.0)
    parser.add_argument("--graph_teacher_temperature", type=float, default=1.5)
    return parser.parse_args()


def _expected_fusion_variant(experiment_name: str, fusion_variant: str) -> str:
    experiment = EXPERIMENTS[experiment_name]
    if bool(experiment["disable_gnn"]) or bool(experiment["disable_transformer"]):
        return "single_branch"
    return str(fusion_variant).strip().lower()


def _summary_matches(
    summary: dict[str, Any],
    *,
    experiment_name: str,
    seed: int,
    rounds: int | None,
    label_fraction: float,
    fusion_variant: str,
    require_completed: bool = True,
) -> bool:
    experiment = EXPERIMENTS[experiment_name]
    if require_completed and not bool(summary.get("completed")):
        return False
    if int(summary.get("seed", -1)) != int(seed):
        return False
    if rounds is not None and int(summary.get("rounds_ran", 0)) != int(rounds):
        return False
    if abs(float(summary.get("label_fraction", -1.0)) - float(label_fraction)) > 1e-12:
        return False
    if bool(summary.get("gnn_enabled", True)) != (not bool(experiment["disable_gnn"])):
        return False
    if bool(summary.get("transformer_enabled", True)) != (not bool(experiment["disable_transformer"])):
        return False
    if bool(summary.get("federated_enabled", True)):
        return False
    if bool(summary.get("drl_enabled", False)):
        return False
    if str(summary.get("planner_mode", "")).lower() != "deterministic":
        return False
    if str(summary.get("fusion_variant", "")).lower() != _expected_fusion_variant(experiment_name, fusion_variant):
        return False
    return True


def _summary_continue_compatible(
    summary: dict[str, Any],
    *,
    experiment_name: str,
    seed: int,
    target_rounds: int,
    label_fraction: float,
    fusion_variant: str,
) -> bool:
    existing_rounds = int(summary.get("rounds_ran", 0))
    if existing_rounds <= 0 or existing_rounds >= int(target_rounds):
        return False
    return _summary_matches(
        summary,
        experiment_name=experiment_name,
        seed=seed,
        rounds=None,
        label_fraction=label_fraction,
        fusion_variant=fusion_variant,
        require_completed=False,
    )


def _run_trial(
    *,
    output_root: Path,
    phase_name: str,
    dataset_name: str,
    experiment_name: str,
    seed: int,
    label_fraction: float,
    rounds: int,
    local_epochs: int,
    extra_local_epochs: int,
    device: str,
    force_rerun: bool,
    checkpoint_mode: str,
    enable_tensorboard: bool,
    fusion_variant: str,
    graph_warmup_rounds: int,
    fusion_bootstrap_rounds: int,
    modality_dropout: float,
    graph_anchor_loss_weight: float,
    graph_anchor_temperature: float,
    graph_teacher_checkpoint_path: str,
    graph_teacher_distill_weight: float,
    graph_teacher_temperature: float,
) -> dict[str, Any]:
    experiment = EXPERIMENTS[experiment_name]
    trial_root = (
        output_root
        / phase_name
        / experiment_name
        / f"seed{int(seed)}"
        / f"label_fraction_{label_fraction_slug(label_fraction)}"
    )
    trial_root.mkdir(parents=True, exist_ok=True)
    summary_path = hybrid_summary_path(trial_root, dataset_name)
    effective_checkpoint_mode = resolve_checkpoint_mode(force_rerun, checkpoint_mode)
    summary_payload = None if effective_checkpoint_mode == "fresh" else load_summary_payload(summary_path)
    cached_summary = None
    if summary_payload is not None:
        cached_summary = summary_payload.get("summary", summary_payload)
        if not isinstance(cached_summary, dict):
            cached_summary = None
    if cached_summary is not None and _summary_matches(
        cached_summary,
        experiment_name=experiment_name,
        seed=seed,
        rounds=rounds,
        label_fraction=label_fraction,
        fusion_variant=fusion_variant,
    ):
        return {
            "dataset": dataset_name,
            "experiment": experiment_name,
            "phase": phase_name,
            "seed": int(seed),
            "label_fraction": float(label_fraction),
            "reused": True,
            "summary": cached_summary,
            "summary_path": str(summary_path),
            "diagnostics_path": str(cached_summary.get("diagnostics_path", "")),
            "trial_root": str(trial_root),
        }

    resume_path = ""
    resume_round_offset = 0
    total_target_rounds = None
    preload_history = None
    if (
        effective_checkpoint_mode == "continue"
        and cached_summary is not None
        and _summary_continue_compatible(
            cached_summary,
            experiment_name=experiment_name,
            seed=seed,
            target_rounds=rounds,
            label_fraction=label_fraction,
            fusion_variant=fusion_variant,
        )
    ):
        checkpoint_path = hybrid_checkpoint_path(trial_root, dataset_name)
        history_payload = summary_payload.get("history", []) if summary_payload is not None else []
        existing_rounds = int(cached_summary.get("rounds_ran", 0))
        if checkpoint_path.exists() and isinstance(history_payload, list) and len(history_payload) == existing_rounds:
            resume_path = str(checkpoint_path)
            resume_round_offset = existing_rounds
            total_target_rounds = int(rounds)
            preload_history = [dict(item) for item in history_payload if isinstance(item, dict)]

    config = {
        "federated_rounds": int(rounds) - int(resume_round_offset),
        "local_epochs": int(local_epochs),
        "extra_local_epochs": int(extra_local_epochs),
        "dataset": str(dataset_name),
        "num_clients": 1,
        "label_fraction": float(label_fraction),
        "rl_timesteps": 0,
        "device": str(device),
        "enable_tensorboard": bool(enable_tensorboard),
        "planner_mode": "deterministic",
        "disable_federated": True,
        "test_every": 0,
        "seed": int(seed),
        "result_root": str(trial_root),
        "disable_gnn": bool(experiment["disable_gnn"]),
        "disable_transformer": bool(experiment["disable_transformer"]),
        "fusion_variant_override": str(fusion_variant),
        "modality_dropout_prob_override": float(modality_dropout),
    }
    if resume_path:
        config["resume_path"] = resume_path
        config["resume_round_offset"] = int(resume_round_offset)
        config["total_target_rounds"] = int(total_target_rounds)
        config["preload_history"] = preload_history or []
    if int(graph_warmup_rounds) >= 0:
        config["graph_warmup_rounds_override"] = int(graph_warmup_rounds)
    if int(fusion_bootstrap_rounds) >= 0:
        config["fusion_bootstrap_rounds_override"] = int(fusion_bootstrap_rounds)
    if float(graph_anchor_loss_weight) >= 0.0:
        config["graph_anchor_loss_weight_override"] = float(graph_anchor_loss_weight)
    if float(graph_anchor_temperature) > 0.0:
        config["graph_anchor_temperature_override"] = float(graph_anchor_temperature)
    if str(graph_teacher_checkpoint_path).strip():
        config["graph_teacher_checkpoint_path"] = str(graph_teacher_checkpoint_path).strip()
    if float(graph_teacher_distill_weight) > 0.0:
        config["graph_teacher_distill_weight"] = float(graph_teacher_distill_weight)
    if float(graph_teacher_temperature) > 0.0:
        config["graph_teacher_temperature"] = float(graph_teacher_temperature)

    from fraud_ml_engineering.algorithms import run_hybrid_fraud_training

    summaries = run_hybrid_fraud_training(**config)
    summary = summaries[str(dataset_name)]
    return {
        "dataset": dataset_name,
        "experiment": experiment_name,
        "phase": phase_name,
        "seed": int(seed),
        "label_fraction": float(label_fraction),
        "reused": False,
        "summary": summary,
        "summary_path": str(summary_path),
        "diagnostics_path": str(summary.get("diagnostics_path", "")),
        "trial_root": str(trial_root),
    }


def _aggregate_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["phase"]),
            str(record["dataset"]),
            float(record["label_fraction"]),
            str(record["experiment"]),
        )
        grouped[key].append(record)

    aggregates: list[dict[str, Any]] = []
    for (phase, dataset, label_fraction, experiment_name), grouped_records in sorted(grouped.items()):
        best_valid_auc_values = [float(item["summary"].get("best_valid_auc", 0.0)) for item in grouped_records]
        test_auc_values = [float(item["summary"].get("test", {}).get("auc", 0.0)) for item in grouped_records]
        test_f1_values = [float(item["summary"].get("test", {}).get("f1_macro", 0.0)) for item in grouped_records]
        aggregates.append(
            {
                "phase": phase,
                "dataset": dataset,
                "label_fraction": float(label_fraction),
                "experiment": experiment_name,
                "description": str(EXPERIMENTS[experiment_name]["description"]),
                "seed_count": int(len(grouped_records)),
                "seeds": [int(item["seed"]) for item in grouped_records],
                "best_valid_auc": mean_std_metric(best_valid_auc_values),
                "test_auc": mean_std_metric(test_auc_values),
                "test_f1_macro": mean_std_metric(test_f1_values),
                "summary_paths": [str(item["summary_path"]) for item in grouped_records],
                "diagnostics_paths": [str(item["diagnostics_path"]) for item in grouped_records if item["diagnostics_path"]],
            }
        )
    return aggregates


def _markdown_table(rows: list[dict[str, Any]], *, include_label_fraction: bool) -> list[str]:
    header = [
        "| dataset |",
        "label_fraction |" if include_label_fraction else "",
        "experiment | mean_best_valid_auc | std_best_valid_auc | mean_test_auc | std_test_auc | mean_test_f1_macro | std_test_f1_macro | seeds |",
    ]
    lines = [
        "".join(header),
        "| --- |"
        + (" ---: |" if include_label_fraction else "")
        + " --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        line_parts = [
            f"| {row['dataset']} |",
        ]
        if include_label_fraction:
            line_parts.append(f" {row['label_fraction']:.2f} |")
        line_parts.extend(
            [
                f" {row['experiment']} |",
                f" {row['best_valid_auc']['mean']:.6f} |",
                f" {row['best_valid_auc']['std']:.6f} |",
                f" {row['test_auc']['mean']:.6f} |",
                f" {row['test_auc']['std']:.6f} |",
                f" {row['test_f1_macro']['mean']:.6f} |",
                f" {row['test_f1_macro']['std']:.6f} |",
                f" {','.join(str(seed) for seed in row['seeds'])} |",
            ]
        )
        lines.append("".join(line_parts))
    return lines


def _build_markdown(aggregates: list[dict[str, Any]], *, args: argparse.Namespace) -> str:
    full_rows = [row for row in aggregates if row["phase"] == "full_supervision"]
    low_rows = [row for row in aggregates if row["phase"] == "low_label"]
    lines = [
        "# Hybrid Mainline Frozen Protocol",
        "",
        f"- generated_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- datasets: `{', '.join(args.datasets)}`",
        f"- seeds: `{', '.join(str(seed) for seed in args.seeds)}`",
        f"- rounds: `{int(args.rounds)}`",
        f"- local_epochs: `{int(args.local_epochs)}`",
        f"- extra_local_epochs: `{int(args.extra_local_epochs)}`",
        f"- device: `{args.device}`",
        f"- planner_mode: `deterministic`",
        f"- disable_federated: `True`",
        f"- fusion_variant: `{args.fusion_variant}`",
        f"- graph_warmup_rounds: `{int(args.graph_warmup_rounds)}`",
        f"- fusion_bootstrap_rounds: `{int(args.fusion_bootstrap_rounds)}`",
        f"- modality_dropout: `{float(args.modality_dropout):.4f}`",
        "",
    ]
    if full_rows:
        lines.extend(
            [
                "## Full Supervision",
                "",
                "Full supervision comparison: `full / no_gnn / no_transformer`.",
                "",
            ]
        )
        lines.extend(_markdown_table(full_rows, include_label_fraction=False))
        lines.append("")
    if low_rows:
        lines.extend(
            [
                "## Low Label",
                "",
                "Low-label comparison: `full / no_transformer`.",
                "",
            ]
        )
        lines.extend(_markdown_table(low_rows, include_label_fraction=True))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if bool(args.skip_full_supervision) and bool(args.skip_low_label):
        raise SystemExit("At least one of full supervision or low label must remain enabled.")

    output_root = Path(args.output_root).resolve() if args.output_root else RESULT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    if not bool(args.skip_full_supervision):
        for dataset_name in args.datasets:
            for seed in args.seeds:
                for experiment_name in args.full_supervision_experiments:
                    jobs.append(
                        {
                            "phase": "full_supervision",
                            "dataset": dataset_name,
                            "seed": int(seed),
                            "experiment": experiment_name,
                            "label_fraction": 1.0,
                        }
                    )
    if not bool(args.skip_low_label):
        for dataset_name in args.datasets:
            for seed in args.seeds:
                for label_fraction in args.label_fractions:
                    for experiment_name in args.low_label_experiments:
                        jobs.append(
                            {
                                "phase": "low_label",
                                "dataset": dataset_name,
                                "seed": int(seed),
                                "experiment": experiment_name,
                                "label_fraction": float(label_fraction),
                            }
                        )

    _progress(
        f"mainline protocol started: jobs={len(jobs)} datasets={list(args.datasets)} "
        f"seeds={list(args.seeds)} rounds={int(args.rounds)} fusion_variant={args.fusion_variant}"
    )

    records: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        _progress(
            f"job {index}/{len(jobs)}: phase={job['phase']} dataset={job['dataset']} "
            f"experiment={job['experiment']} seed={job['seed']} label_fraction={job['label_fraction']:.4f}"
        )
        started = time.perf_counter()
        record = _run_trial(
            output_root=output_root,
            phase_name=str(job["phase"]),
            dataset_name=str(job["dataset"]),
            experiment_name=str(job["experiment"]),
            seed=int(job["seed"]),
            label_fraction=float(job["label_fraction"]),
            rounds=int(args.rounds),
            local_epochs=int(args.local_epochs),
            extra_local_epochs=int(args.extra_local_epochs),
            device=str(args.device),
            force_rerun=bool(args.force_rerun),
            checkpoint_mode=str(args.checkpoint_mode),
            enable_tensorboard=not bool(args.disable_tb),
            fusion_variant=str(args.fusion_variant),
            graph_warmup_rounds=int(args.graph_warmup_rounds),
            fusion_bootstrap_rounds=int(args.fusion_bootstrap_rounds),
            modality_dropout=float(args.modality_dropout),
            graph_anchor_loss_weight=float(args.graph_anchor_loss_weight),
            graph_anchor_temperature=float(args.graph_anchor_temperature),
            graph_teacher_checkpoint_path=str(args.graph_teacher_checkpoint_path),
            graph_teacher_distill_weight=float(args.graph_teacher_distill_weight),
            graph_teacher_temperature=float(args.graph_teacher_temperature),
        )
        records.append(record)
        summary = record["summary"]
        _progress(
            f"job {index}/{len(jobs)} finished in {time.perf_counter() - started:.1f}s: "
            f"reused={record['reused']} test_auc={float(summary.get('test', {}).get('auc', 0.0)):.6f} "
            f"test_f1_macro={float(summary.get('test', {}).get('f1_macro', 0.0)):.6f}"
        )

    aggregates = _aggregate_runs(records)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": str(output_root),
        "protocol": {
            "planner_mode": "deterministic",
            "disable_federated": True,
            "rounds": int(args.rounds),
            "local_epochs": int(args.local_epochs),
            "extra_local_epochs": int(args.extra_local_epochs),
            "device": str(args.device),
            "checkpoint_mode": resolve_checkpoint_mode(bool(args.force_rerun), str(args.checkpoint_mode)),
            "datasets": list(args.datasets),
            "seeds": [int(seed) for seed in args.seeds],
            "low_label_fractions": [float(value) for value in args.label_fractions],
            "full_supervision_experiments": list(args.full_supervision_experiments),
            "low_label_experiments": list(args.low_label_experiments),
            "fusion_variant": str(args.fusion_variant),
            "graph_warmup_rounds": int(args.graph_warmup_rounds),
            "fusion_bootstrap_rounds": int(args.fusion_bootstrap_rounds),
            "modality_dropout": float(args.modality_dropout),
        },
        "records": [
            {
                "phase": str(item["phase"]),
                "dataset": str(item["dataset"]),
                "experiment": str(item["experiment"]),
                "seed": int(item["seed"]),
                "label_fraction": float(item["label_fraction"]),
                "reused": bool(item["reused"]),
                "summary_path": str(item["summary_path"]),
                "diagnostics_path": str(item["diagnostics_path"]),
                "trial_root": str(item["trial_root"]),
                "best_valid_auc": float(item["summary"].get("best_valid_auc", 0.0)),
                "test_auc": float(item["summary"].get("test", {}).get("auc", 0.0)),
                "test_f1_macro": float(item["summary"].get("test", {}).get("f1_macro", 0.0)),
            }
            for item in records
        ],
        "aggregates": aggregates,
    }

    summary_json_path = output_root / "hybrid_mainline_frozen_protocol_summary.json"
    summary_md_path = output_root / "hybrid_mainline_frozen_protocol_summary.md"
    summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md_path.write_text(_build_markdown(aggregates, args=args), encoding="utf-8")

    _progress(f"mainline protocol finished: json={summary_json_path}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON: {summary_json_path}")
    print(f"Saved Markdown: {summary_md_path}")


if __name__ == "__main__":
    main()
