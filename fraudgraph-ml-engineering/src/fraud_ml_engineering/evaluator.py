from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dgl
import torch

from .checkpointing import atomic_write_json, normalize_resume_identity_path
from .vendor.splitgnn.utils import evaluate
from .dataset_registry import load_registered_dataset_bundle
from .device_utils import DEFAULT_DEVICE_REQUEST, resolve_dgl_training_device
from .fraud_dataset import DatasetBundle
from .hybrid_task_model import HybridFraudModel, checkpoint_legacy_fusion_only, sanitize_legacy_hybrid_state_dict


def _graph_on_device(graph: dgl.DGLHeteroGraph, device: torch.device) -> dgl.DGLHeteroGraph:
    graph_device = getattr(graph, "device", None)
    if isinstance(graph_device, torch.device) and graph_device == device:
        return graph
    return graph.to(device)


def _normalize_branch_priority(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("main",)
    if isinstance(value, str):
        items = [part.strip().lower() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(part).strip().lower() for part in value]
    else:
        items = [str(value).strip().lower()]
    normalized = tuple(item for item in items if item)
    return normalized if normalized else ("main",)


def _preferred_branch_priority(model: HybridFraudModel) -> tuple[str, ...]:
    args = getattr(model, "args", None)
    priority = _normalize_branch_priority(getattr(args, "eval_branch_priority", None))
    preferred = str(getattr(args, "preferred_eval_branch", "") or "").strip().lower()
    if preferred and preferred not in priority:
        priority = (preferred,) + tuple(item for item in priority if item != preferred)
    if "main" not in priority:
        priority = priority + ("main",)
    return priority


def _branch_logits_from_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    branch_outputs = dict(payload.get("branch_outputs", {}) or {})
    branch_logits = {
        "main": payload["logits"],
    }
    if branch_outputs.get("fusion_logits") is not None:
        branch_logits["fusion"] = branch_outputs["fusion_logits"]
    if branch_outputs.get("graph_residual_logits") is not None:
        branch_logits["graph_residual"] = branch_outputs["graph_residual_logits"]
    if branch_outputs.get("sequence_residual_logits") is not None:
        branch_logits["sequence_residual"] = branch_outputs["sequence_residual_logits"]
    if branch_outputs.get("raw_branch_logits") is not None:
        branch_logits["raw_branch"] = branch_outputs["raw_branch_logits"]
    return branch_logits


def _select_branch_name(
    branch_logits: dict[str, torch.Tensor],
    branch_priority: tuple[str, ...],
) -> str:
    for branch_name in branch_priority:
        if branch_name in branch_logits:
            return branch_name
    if "main" in branch_logits:
        return "main"
    return next(iter(branch_logits.keys()))


def evaluate_model(
    model: HybridFraudModel,
    graph: dgl.DGLHeteroGraph,
    split: str,
    device: torch.device,
    result_path: str = "",
    threshold: float | None = None,
) -> dict[str, float]:
    model.eval()
    eval_graph = _graph_on_device(graph, device)
    with torch.no_grad():
        payload = model.forward_with_branch_details(eval_graph)
        branch_priority = _preferred_branch_priority(model)
        branch_logits = _branch_logits_from_payload(payload)
        selected_branch = _select_branch_name(branch_logits, branch_priority)
        mask = eval_graph.ndata[split].bool()
        labels = eval_graph.ndata["label"][mask].cpu().numpy()
        split_logits = branch_logits[selected_branch][mask].cpu()
        metrics = evaluate(
            labels,
            split_logits,
            result_path=result_path,
            threshold=threshold,
            return_details=True,
            precision_target=float(getattr(model.args, "fixed_precision_target", 0.5)),
        )
    setattr(model, "_last_eval_branch", selected_branch)
    setattr(model, "_last_eval_branch_priority", branch_priority)
    return {key: float(value) for key, value in metrics.items()}


def _masked_tensor_stats(
    tensor: torch.Tensor | None,
    mask: torch.Tensor,
) -> dict[str, float]:
    if tensor is None:
        return {"mean": 0.0, "std": 0.0, "var": 0.0, "count": 0.0}
    values = tensor if tensor.ndim == 0 else tensor[mask]
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "var": 0.0, "count": 0.0}
    flattened = values.float().reshape(-1)
    return {
        "mean": float(flattened.mean().item()),
        "std": float(flattened.std(unbiased=False).item()),
        "var": float(flattened.var(unbiased=False).item()),
        "count": float(flattened.numel()),
    }


def collect_model_diagnostics(
    model: HybridFraudModel,
    graph: dgl.DGLHeteroGraph,
    device: torch.device,
    *,
    splits: tuple[str, ...] = ("valid_mask", "test_mask"),
) -> dict[str, Any]:
    model.eval()
    eval_graph = _graph_on_device(graph, device)
    precision_target = float(getattr(model.args, "fixed_precision_target", 0.5))
    branch_priority = _preferred_branch_priority(model)
    with torch.no_grad():
        payload = model.forward_with_branch_details(eval_graph)

    branch_logits = _branch_logits_from_payload(payload)
    selected_branch = _select_branch_name(branch_logits, branch_priority)

    diagnostic_tensors = dict(payload.get("diagnostics", {}) or {})
    split_payload: dict[str, Any] = {}
    branch_thresholds: dict[str, float] = {}
    statistic_names = (
        "shared_gap",
        "private_interaction",
        "context_gate",
        "graph_branch_gate",
        "graph_correction_support",
        "sequence_branch_gate",
        "raw_branch_gate",
        "fusion_delta_gate",
        "delta_correction_support",
        "time_reliability",
        "graph_temporal_gate",
        "shared_gate",
        "private_gate",
        "conflict_score",
        "graph_embedding_norm",
        "sequence_embedding_norm",
        "raw_embedding_norm",
        "fused_embedding_norm",
        "sequence_token_valid_ratio",
        "sequence_valid_length",
        "graph_sequence_prob_gap",
    )

    for split in splits:
        if split not in eval_graph.ndata:
            continue
        mask = eval_graph.ndata[split].bool()
        labels = eval_graph.ndata["label"][mask].cpu().numpy()
        split_name = str(split).replace("_mask", "")
        branch_metrics: dict[str, dict[str, float]] = {}
        for branch_name, branch_logits_tensor in branch_logits.items():
            branch_threshold = branch_thresholds.get(branch_name)
            metrics = evaluate(
                labels,
                branch_logits_tensor[mask].cpu(),
                threshold=branch_threshold,
                return_details=True,
                precision_target=precision_target,
            )
            branch_metrics[branch_name] = {key: float(value) for key, value in metrics.items()}
            if split_name == "valid":
                branch_thresholds[branch_name] = float(metrics.get("threshold", 0.5))

        stats: dict[str, float] = {}
        for statistic_name in statistic_names:
            for stat_name, stat_value in _masked_tensor_stats(diagnostic_tensors.get(statistic_name), mask).items():
                stats[f"{statistic_name}_{stat_name}"] = float(stat_value)

        split_payload[split_name] = {
            "num_nodes": int(mask.sum().item()),
            "branches": branch_metrics,
            "selected_branch": selected_branch,
            "selected_metrics": dict(branch_metrics.get(selected_branch, {})),
            "stats": stats,
        }

    return {
        "available_branches": list(branch_logits.keys()),
        "preferred_branch_priority": list(branch_priority),
        "selected_branch": selected_branch,
        "branch_thresholds": {key: float(value) for key, value in branch_thresholds.items()},
        "splits": split_payload,
        "gnn_enabled": bool(getattr(model.args, "gnn_enabled", True)),
        "transformer_enabled": bool(getattr(model.args, "transformer_enabled", True)),
        "use_multimodal_fusion": bool(getattr(model, "use_multimodal_fusion", False)),
    }


def _checkpoint_args_namespace(
    *,
    dataset_name: str,
    checkpoint: dict[str, Any],
    summary: dict[str, Any],
    device: torch.device,
) -> SimpleNamespace:
    checkpoint_args = dict(checkpoint.get("args", {}) or {})
    checkpoint_args["dataset"] = dataset_name
    checkpoint_args["device"] = str(device)
    checkpoint_args.setdefault("fixed_precision_target", float(summary.get("fixed_precision_target", 0.5)))
    args = SimpleNamespace(**checkpoint_args)
    args.device = device
    args.seed = int(checkpoint_args.get("seed", summary.get("seed", 42) or 42))
    args.active_learning_feedback_path = normalize_resume_identity_path(
        checkpoint_args.get("active_learning_feedback_path", "")
    )
    return args


def _load_bundle_for_checkpoint(
    dataset_name: str,
    *,
    args: SimpleNamespace,
) -> DatasetBundle:
    effective_num_clients = int(
        getattr(
            args,
            "effective_num_clients",
            getattr(args, "requested_num_clients", getattr(args, "num_clients", 1)),
        )
    )
    return load_registered_dataset_bundle(
        dataset_name=dataset_name,
        args=args,
        effective_num_clients=max(effective_num_clients, 1),
        client_hops=int(getattr(args, "client_hops", 1)),
        label_fraction=float(getattr(args, "label_fraction", 1.0)),
    )


def evaluate_saved_hybrid_checkpoint(
    *,
    dataset_name: str,
    checkpoint_path: str | Path,
    summary_path: str | Path,
    device: str = DEFAULT_DEVICE_REQUEST,
    result_prefix: str | Path | None = None,
) -> dict[str, Any]:
    resolved_device = resolve_dgl_training_device(device)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    summary_file = Path(summary_path).expanduser().resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary not found: {summary_file}")

    payload = json.loads(summary_file.read_text(encoding="utf-8-sig"))
    summary = dict(payload.get("summary", {}) or {})
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    args = _checkpoint_args_namespace(
        dataset_name=dataset_name,
        checkpoint=checkpoint,
        summary=summary,
        device=resolved_device,
    )
    bundle = _load_bundle_for_checkpoint(dataset_name, args=args)
    model = HybridFraudModel(args, bundle.graph).to(args.device)
    model.load_state_dict(
        sanitize_legacy_hybrid_state_dict(
            checkpoint["model_state"],
            current_state_dict=model.state_dict(),
        ),
        strict=True,
    )
    model.legacy_fusion_only = checkpoint_legacy_fusion_only(checkpoint)
    args.legacy_fusion_only = bool(model.legacy_fusion_only)
    eval_graph = _graph_on_device(bundle.graph, args.device)
    stored_best_valid_threshold = float(summary.get("best_valid_threshold", checkpoint.get("best_valid_threshold", 0.5)))
    resolved_result_prefix = (
        Path(result_prefix).expanduser().resolve()
        if result_prefix is not None
        else summary_file.parent / f"{dataset_name}_hybrid_fraudgraph"
    )
    valid_metrics = evaluate_model(model, eval_graph, "valid_mask", args.device)
    best_valid_threshold = float(valid_metrics.get("threshold", stored_best_valid_threshold))
    test_metrics = evaluate_model(
        model,
        eval_graph,
        "test_mask",
        args.device,
        result_path=str(resolved_result_prefix),
        threshold=best_valid_threshold,
    )
    diagnostics = collect_model_diagnostics(model, eval_graph, args.device, splits=("valid_mask", "test_mask"))
    summary["best_valid_metrics"] = valid_metrics
    summary["best_valid_auc"] = float(valid_metrics.get("auc", summary.get("best_valid_auc", 0.0)))
    summary["best_valid_gmean"] = float(valid_metrics.get("gmean", summary.get("best_valid_gmean", 0.0)))
    summary["best_valid_pr_auc"] = float(valid_metrics.get("pr_auc", summary.get("best_valid_pr_auc", 0.0)))
    summary["best_valid_recall_at_precision"] = float(
        valid_metrics.get("recall_at_precision", summary.get("best_valid_recall_at_precision", 0.0))
    )
    summary["best_valid_f1_macro"] = float(valid_metrics.get("f1_macro", summary.get("best_valid_f1_macro", 0.0)))
    summary["best_valid_threshold"] = float(best_valid_threshold)
    summary["diagnostics_available_branches"] = list(diagnostics.get("available_branches", []))
    summary["preferred_eval_branch_priority"] = list(diagnostics.get("preferred_branch_priority", []))
    summary["diagnostics_branch_thresholds"] = {
        key: float(value) for key, value in dict(diagnostics.get("branch_thresholds", {})).items()
    }
    summary["best_valid_selected_branch"] = str(
        diagnostics.get("splits", {}).get("valid", {}).get("selected_branch", getattr(model, "_last_eval_branch", "main"))
    )
    summary["test_selected_branch"] = str(
        diagnostics.get("splits", {}).get("test", {}).get("selected_branch", getattr(model, "_last_eval_branch", "main"))
    )
    summary["test"] = test_metrics
    summary["test_evaluated"] = True
    summary["test_evaluation_policy"] = "single_post_selection_final_report"
    payload["summary"] = summary
    atomic_write_json(summary_file, payload)
    return {
        "summary_path": str(summary_file),
        "checkpoint_path": str(checkpoint_file),
        "result_prefix": str(resolved_result_prefix),
        "best_valid_threshold": float(best_valid_threshold),
        "stored_best_valid_threshold": float(stored_best_valid_threshold),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "diagnostics": diagnostics,
    }
