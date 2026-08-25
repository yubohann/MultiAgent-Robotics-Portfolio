from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_ml_engineering.paths import ARTIFACTS_ROOT

DEFAULT_OUTPUT_ROOT = ARTIFACTS_ROOT / "smoke_suite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project smoke suite with five named cases.")
    parser.add_argument("--dataset", type=str, default="comp")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def _dataset_loading_case(dataset_name: str) -> dict[str, Any]:
    from fraud_ml_engineering.dataset_registry import bundle_protocol_summary, load_registered_dataset_bundle

    args = SimpleNamespace(
        seed=30,
        active_learning_feedback_path="",
        archive_data_root="",
        archive_max_users=800,
        archive_max_transactions=5000,
        archive_risk_positive_ratio=0.15,
        archive_force_preview=True,
    )
    bundle = load_registered_dataset_bundle(
        dataset_name=dataset_name,
        args=args,
        effective_num_clients=1,
        client_hops=1,
        label_fraction=1.0,
    )
    return {
        "case": "dataset_loading",
        "dataset": dataset_name,
        "bundle_protocol": bundle_protocol_summary(bundle, dataset_name),
    }


def _metrics_case() -> dict[str, Any]:
    import numpy as np
    import torch

    from fraud_ml_engineering.vendor.splitgnn.utils import evaluate

    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    logits = torch.tensor(
        [
            [3.0, -3.0],
            [1.5, -1.0],
            [-1.0, 1.5],
            [-2.5, 3.5],
        ],
        dtype=torch.float32,
    )
    metrics = evaluate(
        labels,
        logits,
        return_details=True,
        precision_target=0.5,
    )
    return {
        "case": "metric_calculation",
        "metrics": {key: float(value) for key, value in metrics.items()},
    }


def _mini_train_case(dataset_name: str, device: str, run_root: Path) -> dict[str, Any]:
    from fraud_ml_engineering.algorithms import run_hybrid_fraud_training

    summaries = run_hybrid_fraud_training(
        federated_rounds=1,
        local_epochs=1,
        extra_local_epochs=1,
        edge_loss_weight=1.0,
        dataset=dataset_name,
        num_clients=1,
        client_hops=1,
        label_fraction=1.0,
        rl_timesteps=0,
        device=device,
        enable_tensorboard=False,
        classification_loss="cb_focal",
        focal_gamma=2.0,
        class_balance_beta=0.999,
        pseudo_label_threshold=0.9,
        pseudo_label_weight=0.0,
        pseudo_label_novelty_threshold=2.5,
        consistency_weight=0.0,
        active_learning_budget_per_round=0,
        active_learning_delay_rounds=0,
        active_learning_novelty_weight=0.0,
        active_learning_diversity_weight=0.0,
        active_learning_candidate_pool_scale=1,
        fedprox_mu=0.0,
        dp_noise_std=0.0,
        transformer_hidden_dim=32,
        transformer_num_layers=1,
        fusion_hidden_dim=32,
        planner_mode="deterministic",
        early_stop=0,
        test_every=0,
        fixed_precision_target=0.5,
        result_root=str(run_root),
        disable_federated=True,
        graph_aux_loss_weight_override=0.0,
        sequence_aux_loss_weight_override=0.0,
        graph_gate_logit_bias_override=0.0,
        eval_graph_gate_logit_bias_override=0.0,
        graph_residual_min_gate_override=0.0,
        sequence_residual_scale_override=1.0,
        seed=30,
    )
    summary = dict(summaries[dataset_name])
    return {
        "case": "mini_training",
        "dataset": dataset_name,
        "summary": summary,
        "summary_path": str(run_root / dataset_name / f"{dataset_name}_hybrid_summary.json"),
        "checkpoint_path": str(run_root / dataset_name / f"{dataset_name}_hybrid_fraudgraph.pt"),
    }


def _checkpoint_restore_case(dataset_name: str, device: str, mini_train_payload: dict[str, Any]) -> dict[str, Any]:
    from fraud_ml_engineering.evaluator import evaluate_saved_hybrid_checkpoint

    restored = evaluate_saved_hybrid_checkpoint(
        dataset_name=dataset_name,
        checkpoint_path=mini_train_payload["checkpoint_path"],
        summary_path=mini_train_payload["summary_path"],
        device=device,
    )
    return {
        "case": "checkpoint_restore",
        "dataset": dataset_name,
        "valid_metrics": restored["valid_metrics"],
        "test_metrics": restored["test_metrics"],
        "summary_path": restored["summary_path"],
    }


def _report_generation_case(artifacts_root: Path, case_results: dict[str, Any]) -> dict[str, Any]:
    from fraud_ml_engineering.run_artifacts import write_json

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_root": str(artifacts_root),
        "cases": list(case_results.keys()),
    }
    write_json(artifacts_root / "manifest.json", manifest)
    write_json(artifacts_root / "summary.json", case_results)

    lines = [
        "# SplitGNN Smoke Suite",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- run_root: `{manifest['run_root']}`",
        "",
    ]
    for case_name, payload in case_results.items():
        lines.append(f"## {case_name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    report_path = artifacts_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "case": "report_generation",
        "manifest_path": str(artifacts_root / "manifest.json"),
        "summary_path": str(artifacts_root / "summary.json"),
        "report_path": str(report_path),
    }


def main() -> None:
    from fraud_ml_engineering.run_artifacts import create_run_artifacts, write_json

    args = parse_args()
    artifacts = create_run_artifacts(args.output_root, prefix="smoke")
    mini_train_root = artifacts.run_root / "mini_train"
    mini_train_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    results["dataset_loading"] = _dataset_loading_case(args.dataset)
    results["metric_calculation"] = _metrics_case()
    results["mini_training"] = _mini_train_case(args.dataset, args.device, mini_train_root)
    results["checkpoint_restore"] = _checkpoint_restore_case(args.dataset, args.device, results["mini_training"])
    results["report_generation"] = _report_generation_case(artifacts.run_root, dict(results))

    write_json(artifacts.summary_path, results)
    print(json.dumps({"run_root": str(artifacts.run_root), "cases": list(results.keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
