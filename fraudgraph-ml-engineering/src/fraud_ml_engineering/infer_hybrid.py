"""Hybrid fraud model inference script."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score

from .device_utils import DEFAULT_DEVICE_REQUEST, resolve_dgl_training_device
from .paths import ARTIFACTS_ROOT, DATA_ROOT
from .vendor.splitgnn.utils import evaluate

IMPORT_ERROR: Exception | None = None
try:
    from .fraud_dataset import load_graph_for_inference, load_splitgnn_dataset
    from .hybrid_task_model import (
        HybridFraudModel,
        _novelty_scores,
        checkpoint_legacy_fusion_only,
        seed_legacy_hybrid_compatibility,
        sanitize_legacy_hybrid_state_dict,
    )
except Exception as error:  # pragma: no cover - runtime env dependent
    IMPORT_ERROR = error

DEFAULT_DATA_DIR = DATA_ROOT / "splitgnn"
DEFAULT_RESULT_ROOT = ARTIFACTS_ROOT / "inference"
SUPPORTED_DATASETS = ["amazon", "comp", "yelp"]


def _ensure_runtime_dependencies() -> None:
    """Validate inference dependencies at the boundary and provide an actionable remedy."""
    if IMPORT_ERROR is None:
        return
    message = str(IMPORT_ERROR).lower()
    if "dgl" in message:
        raise RuntimeError(
            "Missing dgl dependency. Install the pinned graph-learning profile:\n"
            "python -m pip install -r requirements/requirements-cpu.txt\n"
            "Then retry the documented inference command."
        ) from IMPORT_ERROR
    raise RuntimeError(
        "Inference dependencies could not be imported. Install the pinned profile and retry:\n"
        "python -m pip install -r requirements/requirements-cpu.txt"
    ) from IMPORT_ERROR


def _resolve_device(device_arg: str) -> torch.device:
    return resolve_dgl_training_device(device_arg)


def _mask_from_split(graph, split: str) -> torch.Tensor:
    if split == "all":
        return torch.ones(graph.num_nodes(graph.ntypes[0]), dtype=torch.bool, device=graph.device)
    key = f"{split}_mask"
    if key not in graph.ndata:
        raise KeyError(f"No mask named '{key}' exists for split '{split}'.")
    return graph.ndata[key].bool()


def _export_label_visibility_mask(graph, mask: torch.Tensor) -> np.ndarray:
    visible = torch.ones(graph.num_nodes(graph.ntypes[0]), dtype=torch.bool, device=graph.device)
    if "train_mask" in graph.ndata and "train_supervised_mask" in graph.ndata:
        train_mask = graph.ndata["train_mask"].bool()
        train_supervised_mask = graph.ndata["train_supervised_mask"].bool()
        visible[train_mask] = train_supervised_mask[train_mask]
    return visible[mask].detach().cpu().numpy().astype(bool)


def _build_model_args(checkpoint_args: dict, device: torch.device, dataset: str) -> SimpleNamespace:
    args = dict(checkpoint_args)
    args["dataset"] = dataset
    args["device"] = device
    args.setdefault("dropout", 0.1)
    args.setdefault("n_class", 2)
    args.setdefault("intra_dim", 8)
    args.setdefault("gamma", 1.0)
    args.setdefault("C", 1)
    args.setdefault("K", 0)
    transformer_hidden_dim = args.get("transformer_hidden_dim", args.get("seq_hidden_dim", 64))
    args["transformer_hidden_dim"] = int(transformer_hidden_dim)
    args.setdefault("seq_hidden_dim", int(transformer_hidden_dim))
    args.setdefault("transformer_num_layers", 1)
    args.setdefault("fusion_hidden_dim", 64)
    args.setdefault("classification_loss", "cb_focal")
    args.setdefault("focal_gamma", 2.0)
    args.setdefault("class_balance_beta", 0.999)
    return SimpleNamespace(**args)


def _checkpoint_feedback_path_for_reload(checkpoint_args: dict) -> str:
    raw = str(checkpoint_args.get("active_learning_feedback_path", "")).strip()
    if not raw:
        return ""
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return ""


def _bucketize_probs(probs: np.ndarray, bins: int = 10) -> List[dict]:
    if probs.size == 0:
        return []
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(probs, bins=edges)
    buckets = []
    for index in range(bins):
        buckets.append(
            {
                "bin": index,
                "range": f"{edges[index]:.2f}-{edges[index+1]:.2f}",
                "count": int(counts[index]),
            }
        )
    return buckets


def _binary_gmean(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    tp = float(np.sum((labels == 1) & (preds == 1)))
    tn = float(np.sum((labels == 0) & (preds == 0)))
    fp = float(np.sum((labels == 0) & (preds == 1)))
    fn = float(np.sum((labels == 1) & (preds == 0)))
    pos_recall = tp / max(tp + fn, 1.0)
    neg_recall = tn / max(tn + fp, 1.0)
    return float(np.sqrt(pos_recall * neg_recall))


def _log_softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - log_sum_exp


def _temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(max(temperature, 1e-6))
    return logits / temperature


def _sweep_temperature(
    labels: np.ndarray,
    logits: np.ndarray,
    temp_min: float = 0.5,
    temp_max: float = 5.0,
    steps: int = 91,
) -> tuple[float, dict]:
    temperatures = np.linspace(temp_min, temp_max, max(int(steps), 2))
    best_temperature = 1.0
    best_nll = float("inf")
    best_summary: dict = {}
    labels = np.asarray(labels).astype(np.int64)
    logits = np.asarray(logits).astype(np.float64)

    for temperature in temperatures:
        scaled_logits = _temperature_scale_logits(logits, float(temperature))
        log_probs = _log_softmax_numpy(scaled_logits)
        nll = float(-np.mean(log_probs[np.arange(len(labels)), labels]))
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
            best_summary = {"nll": float(nll)}
    return best_temperature, best_summary


def _sweep_calibration_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: str = "f1_macro",
    grid_step: float = 0.01,
) -> tuple[float, dict]:
    thresholds = np.arange(0.05, 0.951, max(grid_step, 1e-3))
    best_threshold = 0.5
    best_score = -1.0
    best_detail = {}
    for threshold in thresholds:
        preds = (probs >= threshold).astype(np.int32)
        if metric == "gmean":
            score = _binary_gmean(labels, preds)
            detail = {"gmean": float(score)}
        else:
            score = float(f1_score(labels, preds, average="macro"))
            detail = {"f1_macro": float(score)}
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
            best_detail = detail
    return best_threshold, best_detail


def _write_topk_csv(path: Path, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["rank", "node_id", "prob", "pred", "label", "novelty", "open_set_candidate"],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def _write_uncertain_csv(path: Path, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["rank", "node_id", "prob", "margin", "pred", "label", "novelty", "open_set_candidate"],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def _write_novelty_csv(path: Path, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["rank", "node_id", "novelty", "prob", "margin", "pred", "label", "open_set_candidate"],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def _write_feedback_template(path: Path, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["rank", "node_id", "review_label", "prob", "margin", "pred"])
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "node_id": row["node_id"],
                    "review_label": "",
                    "prob": row["prob"],
                    "margin": row["margin"],
                    "pred": row["pred"],
                }
            )


def _build_closed_loop_feedback_records(
    uncertain_records: List[dict],
    novelty_records: List[dict],
    max_records: int,
) -> List[dict]:
    merged: Dict[int, dict] = {}

    def _merge_ranked_records(records: List[dict], source_name: str) -> None:
        for rank, record in enumerate(records, start=1):
            node_id = int(record["node_id"])
            item = merged.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "label": int(record.get("label", -1)),
                    "prob": float(record.get("prob", 0.0)),
                    "margin": float(record.get("margin", 0.0)),
                    "pred": int(record.get("pred", 0)),
                    "novelty": float(record.get("novelty", 0.0)),
                    "open_set_candidate": bool(record.get("open_set_candidate", False)),
                    "priority_score": 0.0,
                    "sources": [],
                },
            )
            item["label"] = int(record.get("label", item["label"]))
            item["prob"] = float(record.get("prob", item["prob"]))
            item["margin"] = float(record.get("margin", item["margin"]))
            item["pred"] = int(record.get("pred", item["pred"]))
            item["novelty"] = float(record.get("novelty", item["novelty"]))
            item["open_set_candidate"] = bool(record.get("open_set_candidate", item["open_set_candidate"]))
            item["priority_score"] += 1.0 / float(rank)
            item["sources"].append(source_name)

    _merge_ranked_records(uncertain_records, "uncertainty")
    _merge_ranked_records(novelty_records, "novelty")
    ranked = sorted(merged.values(), key=lambda item: item["priority_score"], reverse=True)
    return ranked[: max(int(max_records), 1)]


def _write_feedback_json(path: Path, dataset: str, split: str, records: List[dict]) -> None:
    payload = {
        "dataset": dataset,
        "split": split,
        "records": [
            {
                "node_id": int(item["node_id"]),
                "label": int(item["label"]),
                "prob": float(item.get("prob", 0.0)),
                "margin": float(item.get("margin", 0.0)),
                "pred": int(item.get("pred", 0)),
                "novelty": float(item.get("novelty", 0.0)),
                "open_set_candidate": bool(item.get("open_set_candidate", False)),
                "priority_score": float(item.get("priority_score", 0.0)),
                "sources": list(item.get("sources", [])),
            }
            for item in records
        ],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _run_closed_loop_retrain(
    dataset: str,
    model_path: Path,
    checkpoint: dict,
    feedback_path: Path,
    device: torch.device,
    retrain_rounds: int,
) -> dict:
    from .algorithms import run_hybrid_fraud_training

    checkpoint_args = checkpoint.get("args", {})
    planner_mode = str(checkpoint_args.get("planner_mode", checkpoint.get("planner_mode", "deterministic")))
    requested_num_clients = max(int(checkpoint_args.get("requested_num_clients", checkpoint_args.get("num_clients", 3))), 1)
    requested_base_local_epochs = max(
        int(checkpoint_args.get("requested_base_local_epochs", checkpoint_args.get("local_epochs", 2))),
        1,
    )
    requested_extra_local_epochs = max(int(checkpoint_args.get("requested_extra_local_epochs", 1)), 1)
    retrain_seed = checkpoint_args.get("seed", None)
    closed_loop_result_root = feedback_path.parent / "closed_loop_retrain"
    retrain_summary = run_hybrid_fraud_training(
        federated_rounds=max(int(retrain_rounds), 1),
        local_epochs=requested_base_local_epochs,
        extra_local_epochs=requested_extra_local_epochs,
        edge_loss_weight=float(checkpoint_args.get("edge_loss_weight", checkpoint_args.get("gamma", 1.0))),
        dataset=dataset,
        num_clients=requested_num_clients,
        client_hops=max(int(checkpoint_args.get("client_hops", 1)), 1),
        label_fraction=float(checkpoint_args.get("label_fraction", 1.0)),
        rl_timesteps=max(
            int(checkpoint_args.get("rl_timesteps", checkpoint_args.get("controller_timesteps", 512))),
            1,
        ),
        device=str(device),
        enable_tensorboard=False,
        classification_loss=str(checkpoint_args.get("classification_loss", "cb_focal")),
        focal_gamma=float(checkpoint_args.get("focal_gamma", 2.0)),
        class_balance_beta=float(checkpoint_args.get("class_balance_beta", 0.999)),
        pseudo_label_threshold=float(checkpoint_args.get("pseudo_label_threshold", 0.9)),
        pseudo_label_weight=float(checkpoint_args.get("pseudo_label_weight", 0.15)),
        pseudo_label_novelty_threshold=float(checkpoint_args.get("pseudo_label_novelty_threshold", 2.5)),
        consistency_weight=float(checkpoint_args.get("consistency_weight", 0.1)),
        active_learning_budget_per_round=max(int(checkpoint_args.get("active_learning_budget_per_round", 0)), 0),
        active_learning_delay_rounds=max(int(checkpoint_args.get("active_learning_delay_rounds", 0)), 0),
        active_learning_novelty_weight=float(checkpoint_args.get("active_learning_novelty_weight", 0.35)),
        active_learning_diversity_weight=float(checkpoint_args.get("active_learning_diversity_weight", 0.25)),
        active_learning_candidate_pool_scale=max(int(checkpoint_args.get("active_learning_candidate_pool_scale", 4)), 1),
        fedprox_mu=float(checkpoint_args.get("fedprox_mu", 0.01)),
        dp_noise_std=float(checkpoint_args.get("dp_noise_std", 0.0)),
        transformer_hidden_dim=int(checkpoint_args.get("transformer_hidden_dim", checkpoint_args.get("seq_hidden_dim", 64))),
        transformer_num_layers=max(int(checkpoint_args.get("transformer_num_layers", 1)), 1),
        fusion_hidden_dim=int(checkpoint_args.get("fusion_hidden_dim", 64)),
        planner_mode=planner_mode,
        early_stop=max(int(checkpoint_args.get("early_stop", 0)), 0),
        test_every=0,
        resume_path=str(model_path),
        active_learning_feedback_path=str(feedback_path),
        export_embedding_viz=False,
        seed=int(retrain_seed) if retrain_seed is not None else None,
        result_root=str(closed_loop_result_root),
        disable_gnn=bool(checkpoint_args.get("disable_gnn", False)),
        disable_transformer=bool(checkpoint_args.get("disable_transformer", False)),
        disable_federated=bool(checkpoint_args.get("disable_federated", False)),
    )
    return retrain_summary.get(dataset, retrain_summary)


def run_inference_for_dataset(
    dataset: str,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    split: str,
    threshold: float | None,
    top_k: int,
    device: torch.device,
    no_labels: bool,
    graph_path: Path | None,
    export_embeddings: bool,
    calibration_strategy: str = "none",
    calibration_split: str = "valid",
    calibration_metric: str = "f1_macro",
    calibration_grid_step: float = 0.01,
    calibration_temperature_min: float = 0.5,
    calibration_temperature_max: float = 5.0,
    calibration_temperature_steps: int = 91,
    active_learning_top_k: int = 50,
    auto_feedback_oracle: bool = False,
    closed_loop_retrain_rounds: int = 0,
) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_dataset = str(checkpoint_args.get("dataset", "")).lower()
    if checkpoint_dataset and checkpoint_dataset != dataset:
        raise ValueError(
            f"Checkpoint dataset ({checkpoint_dataset}) does not match requested dataset ({dataset})."
        )
    if graph_path is None:
        bundle = load_splitgnn_dataset(
            dataset_name=dataset,
            data_dir=str(data_dir),
            num_clients=int(checkpoint_args.get("requested_num_clients", checkpoint_args.get("num_clients", 3))),
            seed=int(checkpoint_args.get("seed", 42)),
            client_hops=int(checkpoint_args.get("client_hops", 1)),
            label_fraction=float(checkpoint_args.get("label_fraction", 1.0)),
            active_learning_feedback_path=_checkpoint_feedback_path_for_reload(checkpoint_args),
        )
        graph = bundle.graph
    else:
        graph, _, _ = load_graph_for_inference(
            dataset_name=dataset,
            data_dir=str(data_dir),
            graph_path=str(graph_path),
        )

    model_args = _build_model_args(checkpoint_args, device=device, dataset=dataset)
    seed_legacy_hybrid_compatibility(checkpoint_args.get("seed", None))
    model = HybridFraudModel(model_args, graph).to(device)
    model.load_state_dict(
        sanitize_legacy_hybrid_state_dict(
            checkpoint["model_state"],
            current_state_dict=model.state_dict(),
        ),
        strict=True,
    )
    model.legacy_fusion_only = checkpoint_legacy_fusion_only(checkpoint)
    model.eval()

    graph = graph.to(device)
    with torch.no_grad():
        logits, _, graph_embeddings, sequence_embeddings, fused_embeddings = model.forward_with_details(graph)
        mask = _mask_from_split(graph, split)
        masked_logits = logits[mask].cpu()
        labels_available = "label" in graph.ndata and not no_labels
        labels = (
            graph.ndata["label"][mask].cpu().numpy()
            if labels_available
            else np.zeros(int(mask.sum().item()), dtype=np.int64)
        )
        reference_mask = None
        if "label" in graph.ndata:
            if "train_supervised_mask" in graph.ndata:
                reference_mask = graph.ndata["train_supervised_mask"].bool()
            elif "train_mask" in graph.ndata:
                reference_mask = graph.ndata["train_mask"].bool()
        novelty_scores = None
        if reference_mask is not None and bool(reference_mask.any()):
            novelty_scores = _novelty_scores(
                reference_embeddings=fused_embeddings[reference_mask].detach(),
                reference_labels=graph.ndata["label"][reference_mask].detach(),
                query_embeddings=fused_embeddings.detach(),
            )

    label_visibility = (
        _export_label_visibility_mask(graph, mask)
        if labels_available
        else np.zeros(int(mask.sum().item()), dtype=bool)
    )
    calibration_summary = {}
    calibrated_threshold = None
    calibrated_temperature = None
    effective_logits = masked_logits.clone()
    if calibration_strategy != "none" and labels_available:
        cal_mask = _mask_from_split(graph, calibration_split)
        cal_visibility = _export_label_visibility_mask(graph, cal_mask)
        if bool(np.any(cal_visibility)):
            cal_logits_all = logits[cal_mask].cpu()
            cal_visible_tensor = torch.from_numpy(cal_visibility).bool()
            cal_logits = cal_logits_all[cal_visible_tensor]
            cal_labels_all = graph.ndata["label"][cal_mask].cpu().numpy()
            cal_labels = cal_labels_all[cal_visibility]
            if calibration_strategy == "temperature":
                calibrated_temperature, calibration_summary = _sweep_temperature(
                    labels=cal_labels,
                    logits=cal_logits.numpy(),
                    temp_min=calibration_temperature_min,
                    temp_max=calibration_temperature_max,
                    steps=calibration_temperature_steps,
                )
                effective_logits = effective_logits / max(calibrated_temperature, 1e-6)
            elif calibration_strategy == "threshold_sweep":
                cal_probs = torch.softmax(cal_logits, dim=1)[:, 1].numpy()
                calibrated_threshold, calibration_summary = _sweep_calibration_threshold(
                    labels=cal_labels,
                    probs=cal_probs,
                    metric=calibration_metric,
                    grid_step=calibration_grid_step,
                )
            calibration_summary["visible_label_count"] = int(np.sum(cal_visibility))
        else:
            calibration_summary = {
                "skipped": True,
                "reason": f"no visible labels available for calibration split '{calibration_split}'",
                "visible_label_count": 0,
            }

    use_threshold = float(threshold) if threshold is not None else float(
        calibrated_threshold if calibrated_threshold is not None else checkpoint.get("best_valid_threshold", 0.5)
    )
    result_prefix = output_dir / f"{dataset}_{split}_infer"
    metrics = {}
    visible_label_count = int(np.sum(label_visibility)) if labels_available else 0
    if labels_available and visible_label_count > 0:
        metric_visible_tensor = torch.from_numpy(label_visibility).bool()
        metrics = evaluate(
            labels=labels[label_visibility],
            logits=effective_logits[metric_visible_tensor],
            result_path=str(result_prefix),
            threshold=use_threshold,
            return_details=True,
        )

    probs = torch.softmax(effective_logits, dim=1)[:, 1].numpy()
    threshold_used = float(metrics["threshold"]) if metrics else float(use_threshold)
    preds = (probs >= threshold_used).astype(np.int32)
    node_ids = mask.nonzero(as_tuple=False).flatten().cpu().numpy().astype(np.int64)
    preds_file = Path(str(result_prefix) + "_result_preds.npy")
    probs_file = Path(str(result_prefix) + "_result_probs.npy")
    np.save(preds_file, preds)
    np.save(probs_file, probs)
    masked_novelty_scores = (
        novelty_scores[mask].detach().cpu().numpy()
        if novelty_scores is not None
        else np.zeros(int(mask.sum().item()), dtype=np.float32)
    )
    open_set_threshold = float(checkpoint_args.get("open_set_novelty_threshold", 0.0))
    open_set_flags = (
        masked_novelty_scores >= open_set_threshold
        if open_set_threshold > 0.0
        else np.zeros_like(masked_novelty_scores, dtype=bool)
    )
    np.save(output_dir / f"{dataset}_{split}_infer_node_ids.npy", node_ids)
    if export_embeddings:
        np.save(output_dir / f"{dataset}_{split}_graph_embeddings.npy", graph_embeddings[mask].cpu().numpy())
        np.save(output_dir / f"{dataset}_{split}_sequence_embeddings.npy", sequence_embeddings[mask].cpu().numpy())
        np.save(output_dir / f"{dataset}_{split}_fused_embeddings.npy", fused_embeddings[mask].cpu().numpy())
        np.save(output_dir / f"{dataset}_{split}_logits.npy", effective_logits.numpy())
        np.save(output_dir / f"{dataset}_{split}_raw_logits.npy", masked_logits.numpy())

    uncertainty_scores = np.abs(probs - threshold_used)
    uncertain_order = np.argsort(uncertainty_scores)
    uncertain_top_k = int(max(active_learning_top_k, 1))
    uncertain_indices = uncertain_order[: min(uncertain_top_k, len(uncertain_order))]
    uncertain_records = []
    uncertain_export_records = []
    for rank, idx in enumerate(uncertain_indices, start=1):
        oracle_label = int(labels[idx]) if labels_available and idx < len(labels) else -1
        base_record = {
            "rank": rank,
            "node_id": int(node_ids[idx]),
            "prob": float(probs[idx]),
            "margin": float(uncertainty_scores[idx]),
            "pred": int(preds[idx]),
            "label": oracle_label,
            "novelty": float(masked_novelty_scores[idx]),
            "open_set_candidate": bool(open_set_flags[idx]),
        }
        uncertain_records.append(base_record)
        uncertain_export_records.append(
            {**base_record, "label": oracle_label if labels_available and bool(label_visibility[idx]) else -1}
        )
    uncertain_top_k_file = output_dir / f"{dataset}_{split}_infer_uncertain_topk.csv"
    _write_uncertain_csv(uncertain_top_k_file, uncertain_export_records)
    feedback_template_file = output_dir / f"{dataset}_{split}_active_learning_feedback.csv"
    _write_feedback_template(feedback_template_file, uncertain_export_records)

    novelty_order = np.argsort(-masked_novelty_scores)
    novelty_indices = novelty_order[: min(uncertain_top_k, len(novelty_order))]
    novelty_records = []
    novelty_export_records = []
    for rank, idx in enumerate(novelty_indices, start=1):
        oracle_label = int(labels[idx]) if labels_available and idx < len(labels) else -1
        base_record = {
            "rank": rank,
            "node_id": int(node_ids[idx]),
            "novelty": float(masked_novelty_scores[idx]),
            "prob": float(probs[idx]),
            "margin": float(uncertainty_scores[idx]),
            "pred": int(preds[idx]),
            "label": oracle_label,
            "open_set_candidate": bool(open_set_flags[idx]),
        }
        novelty_records.append(base_record)
        novelty_export_records.append(
            {**base_record, "label": oracle_label if labels_available and bool(label_visibility[idx]) else -1}
        )
    novelty_top_k_file = output_dir / f"{dataset}_{split}_infer_novelty_topk.csv"
    _write_novelty_csv(novelty_top_k_file, novelty_export_records)

    top_k = int(max(top_k, 1))
    top_indices = np.argsort(-probs)[: min(top_k, len(probs))]
    top_records = []
    top_export_records = []
    for rank, idx in enumerate(top_indices, start=1):
        oracle_label = int(labels[idx]) if labels_available and idx < len(labels) else -1
        base_record = {
            "rank": rank,
            "node_id": int(node_ids[idx]),
            "prob": float(probs[idx]),
            "pred": int(preds[idx]),
            "label": oracle_label,
            "novelty": float(masked_novelty_scores[idx]),
            "open_set_candidate": bool(open_set_flags[idx]),
        }
        top_records.append(base_record)
        top_export_records.append(
            {**base_record, "label": oracle_label if labels_available and bool(label_visibility[idx]) else -1}
        )
    _write_topk_csv(output_dir / f"{dataset}_{split}_infer_topk.csv", top_export_records)

    summary = {
        "dataset": dataset,
        "model_path": str(model_path),
        "split": split,
        "threshold_used": threshold_used,
        "num_samples": int(len(labels)),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "checkpoint_best_valid_auc": float(checkpoint.get("best_valid_auc", -1.0)),
        "checkpoint_best_round": int(checkpoint.get("best_round", -1)),
        "graph_path": str(graph_path) if graph_path is not None else "",
        "labels_available": bool(labels_available),
        "visible_label_count": int(visible_label_count),
        "hidden_train_labels_excluded_from_metrics": bool(labels_available and visible_label_count < len(labels)),
        "ranking_labels_mask_hidden_train_nodes": bool(labels_available and not bool(np.all(label_visibility))),
        "embeddings_exported": bool(export_embeddings),
        "calibration_strategy": calibration_strategy,
        "calibration_split": calibration_split,
        "calibration_metric": calibration_metric,
        "calibrated_threshold": float(calibrated_threshold) if calibrated_threshold is not None else None,
        "calibration_summary": calibration_summary,
        "risk_buckets": _bucketize_probs(probs),
        "top_k_file": str(output_dir / f"{dataset}_{split}_infer_topk.csv"),
        "node_ids_file": str(output_dir / f"{dataset}_{split}_infer_node_ids.npy"),
        "preds_file": str(preds_file),
        "probs_file": str(probs_file),
        "graph_embeddings_file": str(output_dir / f"{dataset}_{split}_graph_embeddings.npy") if export_embeddings else "",
        "sequence_embeddings_file": str(output_dir / f"{dataset}_{split}_sequence_embeddings.npy") if export_embeddings else "",
        "fused_embeddings_file": str(output_dir / f"{dataset}_{split}_fused_embeddings.npy") if export_embeddings else "",
        "logits_file": str(output_dir / f"{dataset}_{split}_logits.npy") if export_embeddings else "",
        "raw_logits_file": str(output_dir / f"{dataset}_{split}_raw_logits.npy") if export_embeddings else "",
    }
    summary["calibrated_temperature"] = float(calibrated_temperature) if calibrated_temperature is not None else None
    summary["calibration_temperature_min"] = float(calibration_temperature_min)
    summary["calibration_temperature_max"] = float(calibration_temperature_max)
    summary["calibration_temperature_steps"] = int(calibration_temperature_steps)
    summary["active_learning_top_k"] = int(active_learning_top_k)
    summary["uncertain_top_k_file"] = str(uncertain_top_k_file)
    summary["active_learning_feedback_template_file"] = str(feedback_template_file)
    summary["novelty_top_k_file"] = str(novelty_top_k_file)
    summary["open_set_novelty_threshold"] = float(open_set_threshold)
    summary["open_set_candidate_count"] = int(np.sum(open_set_flags))
    summary["novelty_mean"] = float(masked_novelty_scores.mean()) if masked_novelty_scores.size > 0 else 0.0
    summary["novelty_std"] = float(masked_novelty_scores.std()) if masked_novelty_scores.size > 0 else 0.0
    summary["auto_feedback_oracle_enabled"] = bool(auto_feedback_oracle)
    summary["closed_loop_retrain_rounds"] = int(max(closed_loop_retrain_rounds, 0))
    if auto_feedback_oracle or closed_loop_retrain_rounds > 0:
        if not labels_available:
            raise ValueError("Auto feedback / closed-loop retraining requires labels to be available in the current graph.")
        closed_loop_feedback_records = _build_closed_loop_feedback_records(
            uncertain_records=uncertain_records,
            novelty_records=novelty_records,
            max_records=active_learning_top_k,
        )
        closed_loop_feedback_file = output_dir / f"{dataset}_{split}_active_learning_feedback.auto.json"
        _write_feedback_json(
            path=closed_loop_feedback_file,
            dataset=dataset,
            split=split,
            records=closed_loop_feedback_records,
        )
        summary["active_learning_feedback_oracle_file"] = str(closed_loop_feedback_file)
        summary["active_learning_feedback_oracle_count"] = int(len(closed_loop_feedback_records))
        summary["closed_loop_offline_oracle_only"] = bool(split != "train")
        if closed_loop_retrain_rounds > 0:
            if split != "train":
                raise ValueError(
                    "Closed-loop retraining is only allowed for --split train. "
                    "Refusing to retrain on validation/test oracle feedback."
                )
            summary["closed_loop_retrain_summary"] = _run_closed_loop_retrain(
                dataset=dataset,
                model_path=model_path,
                checkpoint=checkpoint,
                feedback_path=closed_loop_feedback_file,
                device=device,
                retrain_rounds=closed_loop_retrain_rounds,
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for hybrid fraud checkpoints.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all"] + SUPPORTED_DATASETS,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Override the default checkpoint path when inferring on one dataset.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="",
        help="Optionally run inference against an external .dgl graph.",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=str(DEFAULT_RESULT_ROOT),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_RESULT_ROOT / "inference_analysis"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test", "all"],
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Skip metric calculation and export probabilities and predictions only.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--calibration_strategy",
        type=str,
        default="none",
        choices=["none", "threshold_sweep", "temperature"],
    )
    parser.add_argument(
        "--calibration_split",
        type=str,
        default="valid",
        choices=["train", "valid", "test", "all"],
    )
    parser.add_argument(
        "--calibration_metric",
        type=str,
        default="f1_macro",
        choices=["f1_macro", "gmean"],
    )
    parser.add_argument(
        "--calibration_grid_step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--calibration_temperature_min",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--calibration_temperature_max",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--calibration_temperature_steps",
        type=int,
        default=91,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--active_learning_top_k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--auto_feedback_oracle",
        action="store_true",
        help="When labels are available, generate an oracle-labeled JSON feedback file for active-learning records.",
    )
    parser.add_argument(
        "--closed_loop_retrain_rounds",
        type=int,
        default=0,
        help="If > 0, resume training from the checkpoint after generating oracle feedback.",
    )
    parser.add_argument(
        "--export_embeddings",
        action="store_true",
        help="Export graph, sequence, and logits embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE_REQUEST,
    )
    return parser.parse_args()


def main() -> None:
    _ensure_runtime_dependencies()
    args = parse_args()
    device = _resolve_device(args.device)
    datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
    if args.model_path and len(datasets) > 1:
        raise ValueError("When --model_path is provided, please run a single dataset at a time.")

    data_dir = Path(args.data_dir).resolve()
    result_root = Path(args.result_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    graph_path = Path(args.graph_path).resolve() if args.graph_path else None

    run_report: Dict[str, dict] = {}
    for dataset in datasets:
        if graph_path is not None and len(datasets) > 1:
            raise ValueError("When --graph_path is provided, please run a single dataset at a time.")
        model_path = Path(args.model_path).resolve() if args.model_path and len(datasets) == 1 else (
            result_root / dataset / f"{dataset}_hybrid_fraudgraph.pt"
        )
        dataset_out_dir = output_root / dataset
        dataset_out_dir.mkdir(parents=True, exist_ok=True)
        summary = run_inference_for_dataset(
            dataset=dataset,
            model_path=model_path,
            data_dir=data_dir,
            output_dir=dataset_out_dir,
            split=args.split,
            threshold=args.threshold,
            top_k=args.top_k,
            device=device,
            no_labels=bool(args.no_labels),
            graph_path=graph_path,
            export_embeddings=bool(args.export_embeddings),
            calibration_strategy=str(args.calibration_strategy),
            calibration_split=str(args.calibration_split),
            calibration_metric=str(args.calibration_metric),
            calibration_grid_step=float(args.calibration_grid_step),
            calibration_temperature_min=float(args.calibration_temperature_min),
            calibration_temperature_max=float(args.calibration_temperature_max),
            calibration_temperature_steps=int(args.calibration_temperature_steps),
            active_learning_top_k=int(args.active_learning_top_k),
            auto_feedback_oracle=bool(args.auto_feedback_oracle),
            closed_loop_retrain_rounds=int(args.closed_loop_retrain_rounds),
        )
        run_report[dataset] = summary
        with open(dataset_out_dir / f"{dataset}_{args.split}_infer_summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

    combined_path = output_root / f"combined_{args.split}_infer_summary.json"
    with open(combined_path, "w", encoding="utf-8") as file:
        json.dump(run_report, file, indent=2, ensure_ascii=False)

    print(json.dumps(run_report, indent=2, ensure_ascii=False))
    print(f"\nCombined summary: {combined_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(str(error))
        raise SystemExit(1)
