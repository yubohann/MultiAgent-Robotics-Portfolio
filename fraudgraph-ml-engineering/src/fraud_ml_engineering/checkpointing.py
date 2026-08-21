from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .hybrid_task_model import sanitize_legacy_hybrid_state_dict


def generate_run_id() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    milliseconds = int((time.time() % 1) * 1000)
    return f"{timestamp}-{milliseconds:03d}"


def normalize_resume_identity_path(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return os.path.normcase(str(Path(raw).expanduser().resolve(strict=False)))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def run_metadata_payload(
    dataset_name: str,
    run_id: str,
    status: str,
    federated_rounds: int,
    local_epochs: int,
    planner_mode: str,
    test_every: int,
    resume_path: str,
    tb_logdir: str,
    seed: int | None = None,
    summary_path: str = "",
    model_path: str = "",
    rounds_ran: int = 0,
    best_round: int = -1,
    best_valid_auc: float | None = None,
    test_auc: float | None = None,
    finished_at: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset": dataset_name,
        "run_id": run_id,
        "status": status,
        "requested_rounds": int(federated_rounds),
        "base_local_epochs": int(local_epochs),
        "planner_mode": str(planner_mode),
        "test_every": int(test_every),
        "resume_path": str(resume_path),
        "tb_logdir": str(tb_logdir),
        "summary_path": str(summary_path),
        "model_path": str(model_path),
        "rounds_ran": int(rounds_ran),
        "best_round": int(best_round),
        "finished_at": str(finished_at),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if best_valid_auc is not None:
        payload["best_valid_auc"] = float(best_valid_auc)
    if test_auc is not None:
        payload["test_auc"] = float(test_auc)
    return payload


def resume_identity_payload(config: dict[str, Any] | SimpleNamespace | None) -> dict[str, Any]:
    source = vars(config) if isinstance(config, SimpleNamespace) else dict(config or {})
    transformer_hidden_dim = int(source.get("transformer_hidden_dim", source.get("seq_hidden_dim", 64)))
    requested_num_clients = int(source.get("requested_num_clients", source.get("num_clients", 1)))
    return {
        "dataset": str(source.get("dataset", "")).lower(),
        "planner_mode": str(source.get("planner_mode", "deterministic")).lower(),
        "disable_gnn": bool(source.get("disable_gnn", False)),
        "disable_transformer": bool(source.get("disable_transformer", False)),
        "disable_federated": bool(source.get("disable_federated", False)),
        "requested_num_clients": requested_num_clients,
        "client_hops": int(source.get("client_hops", 1)),
        "label_fraction": float(source.get("label_fraction", 1.0)),
        "active_learning_feedback_path": normalize_resume_identity_path(
            source.get("active_learning_feedback_path", "")
        ),
        "ethereum_phishing_data_root": normalize_resume_identity_path(
            source.get("ethereum_phishing_data_root", "")
        ),
        "ethereum_phishing_max_users": source.get("ethereum_phishing_max_users", None),
        "ethereum_phishing_max_transactions": source.get("ethereum_phishing_max_transactions", None),
        "ethereum_phishing_force_preview": bool(source.get("ethereum_phishing_force_preview", False)),
        "ethereum_ponzi_data_root": normalize_resume_identity_path(
            source.get("ethereum_ponzi_data_root", "")
        ),
        "ethereum_ponzi_negative_users_path": normalize_resume_identity_path(
            source.get("ethereum_ponzi_negative_users_path", "")
        ),
        "ethereum_ponzi_force_preview": bool(source.get("ethereum_ponzi_force_preview", False)),
        "defi_rug_pull_data_root": normalize_resume_identity_path(
            source.get("defi_rug_pull_data_root", "")
        ),
        "defi_rug_pull_negative_users_path": normalize_resume_identity_path(
            source.get("defi_rug_pull_negative_users_path", "")
        ),
        "defi_rug_pull_force_preview": bool(source.get("defi_rug_pull_force_preview", False)),
        "seed": None if source.get("seed", None) is None else int(source.get("seed")),
        "transformer_hidden_dim": transformer_hidden_dim,
        "transformer_num_layers": int(source.get("transformer_num_layers", 1)),
        "fusion_hidden_dim": int(source.get("fusion_hidden_dim", 64)),
        "edge_loss_weight": float(source.get("edge_loss_weight", source.get("gamma", 1.0))),
        "classification_loss": str(
            source.get(
                "classification_loss",
                source.get("requested_classification_loss", "cb_focal"),
            )
        ).lower(),
        "focal_gamma": float(source.get("focal_gamma", 2.0)),
        "class_balance_beta": float(source.get("class_balance_beta", 0.999)),
        "pseudo_label_threshold": float(source.get("pseudo_label_threshold", 0.9)),
        "pseudo_label_min_threshold": float(source.get("pseudo_label_min_threshold", 0.0)),
        "pseudo_label_top_fraction": float(source.get("pseudo_label_top_fraction", 0.0)),
        "pseudo_label_weight": float(source.get("pseudo_label_weight", 0.0)),
        "pseudo_label_novelty_threshold": float(source.get("pseudo_label_novelty_threshold", 0.0)),
        "consistency_weight": float(source.get("consistency_weight", 0.0)),
        "teacher_ema_decay": float(source.get("teacher_ema_decay", 0.0)),
        "teacher_temperature": float(source.get("teacher_temperature", 1.0)),
        "pseudo_warmup_rounds": int(source.get("pseudo_warmup_rounds", 0)),
        "pseudo_ramp_rounds": int(source.get("pseudo_ramp_rounds", 0)),
        "open_set_novelty_threshold": float(source.get("open_set_novelty_threshold", 0.0)),
        "open_set_loss_weight": float(source.get("open_set_loss_weight", 0.0)),
        "active_learning_budget_per_round": int(source.get("active_learning_budget_per_round", 0)),
        "active_learning_delay_rounds": int(source.get("active_learning_delay_rounds", 0)),
        "active_learning_novelty_weight": float(source.get("active_learning_novelty_weight", 0.0)),
        "active_learning_diversity_weight": float(source.get("active_learning_diversity_weight", 0.0)),
        "active_learning_candidate_pool_scale": int(source.get("active_learning_candidate_pool_scale", 1)),
        "fedprox_mu": float(source.get("fedprox_mu", 0.0)),
        "dp_noise_std": float(source.get("dp_noise_std", 0.0)),
        "target_prob_std": float(source.get("target_prob_std", 0.0)),
        "prob_std_regularization_weight": float(source.get("prob_std_regularization_weight", 0.0)),
        "graph_aux_loss_weight": float(source.get("graph_aux_loss_weight", 0.0)),
        "sequence_aux_loss_weight": float(source.get("sequence_aux_loss_weight", 0.0)),
    }


def validated_resume_state_dict(
    current_args: SimpleNamespace,
    warm_start_payload: dict[str, Any],
    resume_file: Path,
    current_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if "model_state" not in warm_start_payload:
        raise KeyError(f"Resume checkpoint is missing 'model_state': {resume_file}")

    checkpoint_args = dict(warm_start_payload.get("args", {}) or {})
    checkpoint_args.setdefault("planner_mode", warm_start_payload.get("planner_mode", "deterministic"))
    current_identity = resume_identity_payload(current_args)
    checkpoint_identity = resume_identity_payload(checkpoint_args)
    ignored_mismatch_keys: set[str] = set()
    if bool(getattr(current_args, "splitgnn_runtime_policy", False)):
        ignored_mismatch_keys.update({"planner_mode", "disable_federated"})
    mismatches = []
    for key, current_value in current_identity.items():
        if key in ignored_mismatch_keys:
            continue
        checkpoint_value = checkpoint_identity.get(key)
        if checkpoint_value != current_value:
            mismatches.append(f"{key}: checkpoint={checkpoint_value!r}, current={current_value!r}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Resume checkpoint configuration mismatch. Refusing to continue with a different experiment identity: "
            f"{resume_file} ({mismatch_text})"
        )

    resume_state = sanitize_legacy_hybrid_state_dict(
        warm_start_payload["model_state"],
        current_state_dict=current_state_dict,
    )
    if current_state_dict is not None:
        for key, value in current_state_dict.items():
            if key not in resume_state:
                resume_state[key] = value
    return resume_state


def resume_reference_best_metrics(warm_start_payload: dict[str, Any]) -> dict[str, float]:
    stored_best = dict(warm_start_payload.get("best_valid_metrics", {}) or {})
    return {
        "auc": float(warm_start_payload.get("best_valid_auc", stored_best.get("auc", -1.0))),
        "gmean": float(warm_start_payload.get("best_valid_gmean", stored_best.get("gmean", -1.0))),
        "pr_auc": float(warm_start_payload.get("best_valid_pr_auc", stored_best.get("pr_auc", -1.0))),
        "recall_at_precision": float(
            warm_start_payload.get(
                "best_valid_recall_at_precision",
                stored_best.get("recall_at_precision", -1.0),
            )
        ),
        "f1_macro": float(warm_start_payload.get("best_valid_f1_macro", stored_best.get("f1_macro", -1.0))),
        "threshold": float(warm_start_payload.get("best_valid_threshold", stored_best.get("threshold", 0.5))),
    }


def should_inherit_resume_best_metrics(
    current_valid_metrics: dict[str, float],
    stored_best_metrics: dict[str, float],
) -> tuple[bool, str]:
    stored_auc = float(stored_best_metrics.get("auc", -1.0))
    stored_gmean = float(stored_best_metrics.get("gmean", -1.0))
    stored_pr_auc = float(stored_best_metrics.get("pr_auc", -1.0))
    stored_f1_macro = float(stored_best_metrics.get("f1_macro", -1.0))
    if stored_auc < 0.0 or stored_gmean < 0.0 or stored_pr_auc < 0.0 or stored_f1_macro < 0.0:
        return False, "resume_checkpoint_missing_best_metrics"

    current_auc = float(current_valid_metrics.get("auc", 0.0))
    current_gmean = float(current_valid_metrics.get("gmean", 0.0))
    current_pr_auc = float(current_valid_metrics.get("pr_auc", 0.0))
    current_f1_macro = float(current_valid_metrics.get("f1_macro", 0.0))
    auc_gap = abs(current_auc - stored_auc)
    gmean_gap = abs(current_gmean - stored_gmean)
    pr_auc_gap = abs(current_pr_auc - stored_pr_auc)
    f1_macro_gap = abs(current_f1_macro - stored_f1_macro)

    if auc_gap <= 0.01 and gmean_gap <= 0.03 and pr_auc_gap <= 0.03 and f1_macro_gap <= 0.03:
        return (
            True,
            "resume_eval_matches_checkpoint("
            f"auc_gap={auc_gap:.4f}, gmean_gap={gmean_gap:.4f}, "
            f"pr_auc_gap={pr_auc_gap:.4f}, f1_macro_gap={f1_macro_gap:.4f})",
        )
    return (
        False,
        "resume_eval_diverged_from_checkpoint("
        f"auc_gap={auc_gap:.4f}, gmean_gap={gmean_gap:.4f}, "
        f"pr_auc_gap={pr_auc_gap:.4f}, f1_macro_gap={f1_macro_gap:.4f})",
    )
