from __future__ import annotations

import copy
import gc
import time
from collections import Counter
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch

from .hybrid_task_model import HybridFraudModel
from .paper_runner_runtime import select_amp_dtype


def _release_cuda_memory(device: torch.device | str | None = None) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.empty_cache()
        return
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    if device_obj.type != "cuda":
        return
    if device_obj.index is None:
        torch.cuda.empty_cache()
        return
    with torch.cuda.device(device_obj):
        torch.cuda.empty_cache()


def apply_dp_noise_to_state_dict(
    state_dict: Dict[str, torch.Tensor],
    noise_std: float,
) -> Dict[str, torch.Tensor]:
    if noise_std <= 0:
        return state_dict
    noisy_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if torch.is_floating_point(value):
            scale = float(max(value.detach().float().std().item(), 1e-6))
            noisy_state[key] = value + torch.randn_like(value) * (noise_std * scale)
        else:
            noisy_state[key] = value.clone()
    return noisy_state


def ema_update_model(
    teacher_model: HybridFraudModel,
    student_model: HybridFraudModel,
    decay: float,
) -> None:
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
        for teacher_buffer, student_buffer in zip(teacher_model.buffers(), student_model.buffers()):
            teacher_buffer.data.copy_(student_buffer.data)


def _stable_majority_value(values: list[str], default: str = "none") -> str:
    if not values:
        return str(default)
    counts = Counter(str(value) for value in values)
    first_seen: dict[str, int] = {}
    for index, value in enumerate(values):
        normalized = str(value)
        if normalized not in first_seen:
            first_seen[normalized] = index
    best_value = None
    best_score = None
    for value, count in counts.items():
        score = (int(count), -int(first_seen[value]))
        if best_score is None or score > best_score:
            best_value = value
            best_score = score
    return str(best_value if best_value is not None else default)


def _normalize_epoch_metric_recompute_mode(mode: str | None) -> str:
    normalized = str(mode or "all_local_epochs").strip().lower()
    if normalized not in {"all_local_epochs", "last_local_epoch_only", "disabled"}:
        return "all_local_epochs"
    return normalized


def _detach_loss_items(loss_items: dict) -> dict:
    detached: dict = {}
    for key, value in loss_items.items():
        if torch.is_tensor(value):
            detached[key] = value.detach()
        else:
            detached[key] = value
    return detached


def _fedprox_penalty(
    model: HybridFraudModel,
    global_state: Dict[str, torch.Tensor],
    mu: float,
    device: torch.device,
) -> torch.Tensor:
    if mu <= 0.0:
        return torch.tensor(0.0, device=device)
    prox_term = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in global_state:
            continue
        reference = global_state[name]
        if reference.device != param.device or reference.dtype != param.dtype:
            reference = reference.to(device=param.device, dtype=param.dtype, non_blocking=True)
        prox_term = prox_term + torch.sum((param - reference) ** 2)
    return 0.5 * float(mu) * prox_term


def _set_module_trainable(module: torch.nn.Module | None, trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(trainable)


def _resolve_training_stage(local_model: HybridFraudModel, args: SimpleNamespace, current_round: int) -> str:
    if not bool(getattr(local_model, "use_multimodal_fusion", False)):
        return "single_branch"
    graph_warmup_rounds = max(int(getattr(args, "graph_warmup_rounds", 0)), 0)
    fusion_bootstrap_rounds = max(int(getattr(args, "fusion_bootstrap_rounds", 0)), 0)
    if current_round < graph_warmup_rounds:
        return "graph_warmup"
    if current_round < graph_warmup_rounds + fusion_bootstrap_rounds:
        return "fusion_bootstrap"
    return "joint_finetune"


def _resolve_document_stage_name(args: SimpleNamespace, training_stage: str) -> str:
    stage = str(training_stage)
    if stage == "graph_warmup":
        return "stage_a_graph_sequence_raw"
    if stage == "fusion_bootstrap":
        return "stage_b_time_enhancement"
    if float(getattr(args, "label_fraction", 1.0)) < 0.999 and not bool(getattr(args, "pure_label_fraction", False)):
        return "stage_d_lowlabel_ssl"
    if str(getattr(args, "fusion_variant", "")).lower() == "shared_private_prototype":
        return "stage_c_decouple_and_proto"
    return "stage_b_time_enhancement"


def _configure_stage_trainability(local_model: HybridFraudModel, args: SimpleNamespace, stage_name: str) -> None:
    graph_modules = (
        local_model.graph_encoder,
        local_model.graph_diffusion_residual,
        local_model.coassociation_encoder,
        local_model.graph_residual_head,
    )
    sequence_modules = (
        local_model.sequence_encoder,
        local_model.event_encoder,
        local_model.temporal_context_encoder,
        local_model.wavelet_lite_head,
        local_model.utg_lite_fusion,
        local_model.graph_temporal_proj,
        local_model.graph_temporal_gate,
        local_model.context_norm,
        local_model.sequence_residual_head,
    )
    raw_modules = (
        local_model.raw_anchor_encoder,
        local_model.raw_residual_head,
    )
    fusion_modules = (
        local_model.raw_fusion,
        local_model.prototype_memory,
        local_model.shared_private_fusion,
        local_model.graph_dominant_fusion,
        local_model.tri_stream_fusion,
        local_model.uncertainty_head,
        local_model.context_event_gate,
        local_model.feature_encoder,
        local_model.fusion,
    )
    if stage_name == "graph_warmup":
        for module in graph_modules:
            _set_module_trainable(module, True)
        for module in sequence_modules:
            _set_module_trainable(module, False)
        for module in raw_modules:
            _set_module_trainable(module, False)
        for module in fusion_modules:
            _set_module_trainable(module, False)
        return
    if stage_name == "fusion_bootstrap":
        train_graph_during_bootstrap = bool(getattr(args, "fusion_bootstrap_train_graph", False))
        for module in graph_modules:
            _set_module_trainable(module, train_graph_during_bootstrap)
        for module in sequence_modules:
            _set_module_trainable(module, True)
        for module in raw_modules:
            _set_module_trainable(module, True)
        for module in fusion_modules:
            _set_module_trainable(module, True)
        return
    for module in graph_modules:
        _set_module_trainable(module, True)
    for module in sequence_modules:
        _set_module_trainable(module, True)
    for module in raw_modules:
        _set_module_trainable(module, True)
    for module in fusion_modules:
        _set_module_trainable(module, True)


def _parameter_group_name(parameter_name: str) -> str:
    if parameter_name.startswith("graph_encoder.") or parameter_name.startswith("graph_residual_head."):
        return "graph"
    if (
        parameter_name.startswith("sequence_encoder.")
        or parameter_name.startswith("event_encoder.")
        or parameter_name.startswith("temporal_context_encoder.")
        or parameter_name.startswith("graph_temporal_proj.")
        or parameter_name.startswith("graph_temporal_gate.")
        or parameter_name.startswith("sequence_residual_head.")
    ):
        return "sequence"
    return "fusion"


def _stage_learning_rate_scales(args: SimpleNamespace, stage_name: str) -> dict[str, float]:
    graph_scale = float(getattr(args, "graph_learning_rate_scale", 1.0))
    sequence_scale = float(getattr(args, "sequence_learning_rate_scale", 1.0))
    fusion_scale = float(getattr(args, "fusion_learning_rate_scale", 1.0))
    if stage_name == "graph_warmup":
        return {"graph": graph_scale, "sequence": 0.0, "fusion": 0.0}
    if stage_name == "fusion_bootstrap":
        train_graph_during_bootstrap = bool(getattr(args, "fusion_bootstrap_train_graph", False))
        return {
            "graph": float(getattr(args, "graph_follow_learning_rate_scale", 0.15)) if train_graph_during_bootstrap else 0.0,
            "sequence": sequence_scale,
            "fusion": fusion_scale,
        }
    return {"graph": graph_scale, "sequence": sequence_scale, "fusion": fusion_scale}


def _build_optimizer(
    local_model: HybridFraudModel,
    args: SimpleNamespace,
    *,
    stage_name: str,
    learning_rate: float,
) -> tuple[torch.optim.Optimizer, dict[str, float]]:
    lr_scales = _stage_learning_rate_scales(args, stage_name)
    grouped_params: dict[str, list[torch.nn.Parameter]] = {"graph": [], "sequence": [], "fusion": []}
    for name, param in local_model.named_parameters():
        if not param.requires_grad:
            continue
        grouped_params[_parameter_group_name(name)].append(param)

    param_groups = []
    effective_lrs = {"graph": 0.0, "sequence": 0.0, "fusion": 0.0}
    for group_name in ("graph", "sequence", "fusion"):
        params = grouped_params[group_name]
        if not params:
            continue
        group_lr = float(max(learning_rate * lr_scales[group_name], 0.0))
        effective_lrs[group_name] = group_lr
        param_groups.append(
            {
                "params": params,
                "lr": group_lr,
                "weight_decay": float(getattr(args, "weight_decay", 0.0)),
            }
        )

    if not param_groups:
        param_groups = [
            {
                "params": [param for param in local_model.parameters() if param.requires_grad],
                "lr": float(learning_rate),
                "weight_decay": float(getattr(args, "weight_decay", 0.0)),
            }
        ]
        effective_lrs = {"graph": 0.0, "sequence": 0.0, "fusion": float(learning_rate)}

    return torch.optim.Adam(param_groups), effective_lrs


def local_train_round(
    global_model: HybridFraudModel,
    graph_teacher_model: HybridFraudModel | None,
    subgraph,
    global_state: Dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    class_counts: torch.Tensor,
    args: SimpleNamespace,
    current_round: int,
    local_epochs: int,
    grad_clip: float,
    edge_loss_weight: float,
    learning_rate: float,
    fedprox_mu: float,
    dp_noise_std: float,
) -> tuple[Dict[str, torch.Tensor], dict]:
    """Run one round of local client training."""
    _release_cuda_memory(args.device)
    local_model = copy.deepcopy(global_model).to(args.device)
    local_model.edge_loss_weight = edge_loss_weight
    training_stage = _resolve_training_stage(local_model, args, current_round)
    local_model.active_training_stage = str(training_stage)
    args.active_training_stage = str(training_stage)
    _configure_stage_trainability(local_model, args, training_stage)
    fedprox_reference_state = (
        {
            name: global_state[name].detach().cpu().clone()
            for name, param in local_model.named_parameters()
            if param.requires_grad and name in global_state
        }
        if fedprox_mu > 0.0
        else {}
    )
    local_graph = subgraph.to(args.device)
    device_obj = args.device if isinstance(args.device, torch.device) else torch.device(args.device)
    optimizer, effective_lrs = _build_optimizer(
        local_model,
        args,
        stage_name=training_stage,
        learning_rate=learning_rate,
    )
    amp_enabled, amp_dtype, amp_note = select_amp_dtype(device_obj, getattr(args, "amp_dtype", "auto"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    epoch_metric_recompute_mode = _normalize_epoch_metric_recompute_mode(
        getattr(args, "epoch_metric_recompute_mode", "all_local_epochs")
    )
    has_unlabeled_training_nodes = bool(
        "train_unlabeled_mask" in local_graph.ndata and local_graph.ndata["train_unlabeled_mask"].bool().any().item()
    )
    ssl_runtime_enabled = bool(
        float(getattr(args, "pseudo_label_weight", 0.0)) > 0.0
        or float(getattr(args, "consistency_weight", 0.0)) > 0.0
        or float(getattr(args, "open_set_loss_weight", 0.0)) > 0.0
    )
    use_teacher_model = has_unlabeled_training_nodes and ssl_runtime_enabled and training_stage != "graph_warmup"
    use_graph_teacher = bool(
        graph_teacher_model is not None and float(getattr(args, "graph_teacher_distill_weight", 0.0)) > 0.0
    )
    epoch_losses = []
    cls_losses = []
    ranking_cls_losses = []
    pseudo_losses = []
    consistency_losses = []
    open_set_losses = []
    spread_losses = []
    graph_aux_losses = []
    sequence_aux_losses = []
    graph_anchor_losses = []
    graph_teacher_losses = []
    tabular_teacher_losses = []
    coassociation_losses = []
    wavelet_alignment_losses = []
    utg_alignment_losses = []
    edge_losses = []
    supervised_nodes = []
    balanced_supervised_nodes = []
    regularizer_nodes = []
    effective_ssl_nodes = []
    pseudo_nodes = []
    novel_nodes = []
    open_set_nodes = []
    pseudo_thresholds = []
    shared_gap_means = []
    shared_gap_stds = []
    private_interaction_means = []
    private_interaction_stds = []
    context_gate_means = []
    context_gate_stds = []
    graph_branch_gate_means = []
    graph_branch_gate_stds = []
    graph_correction_support_means = []
    graph_correction_support_stds = []
    sequence_branch_gate_means = []
    sequence_branch_gate_stds = []
    raw_branch_gate_means = []
    raw_branch_gate_stds = []
    fusion_delta_gate_means = []
    fusion_delta_gate_stds = []
    delta_correction_support_means = []
    delta_correction_support_stds = []
    time_reliability_means = []
    time_reliability_stds = []
    graph_temporal_gate_means = []
    graph_temporal_gate_stds = []
    shared_gate_means = []
    shared_gate_stds = []
    private_gate_means = []
    private_gate_stds = []
    conflict_score_means = []
    conflict_score_stds = []
    graph_embedding_norm_means = []
    sequence_embedding_norm_means = []
    raw_embedding_norm_means = []
    fused_embedding_norm_means = []
    sequence_token_valid_ratio_means = []
    sequence_valid_length_means = []
    graph_sequence_prob_gap_means = []
    pseudo_reliability_means = []
    pseudo_reliability_pass_rates = []
    pseudo_modality_agreement_means = []
    pseudo_uncertainty_confidence_means = []
    pseudo_uncertainty_pass_rates = []
    pseudo_prototype_margin_means = []
    pseudo_nearest_distance_means = []
    pseudo_cycle_agreement_means = []
    pseudo_cycle_support_rates = []
    ssl_reference_spaces = []
    wavelet_gate_means = []
    coassociation_gate_means = []
    coassociation_density_means = []
    diffusion_gate_means = []
    diffusion_neighbor_strength_means = []
    utg_temporal_gate_means = []
    teacher_model = None
    teacher_ema_decay = float(getattr(args, "teacher_ema_decay", 0.0))
    if teacher_ema_decay > 0.0 and use_teacher_model:
        teacher_model = copy.deepcopy(global_model).to(args.device)
        teacher_model.edge_loss_weight = edge_loss_weight
        teacher_model.eval()
    metric_seed_base = int(getattr(args, "seed", 42))
    metric_cuda_devices = []
    if torch.cuda.is_available():
        metric_device = torch.device(args.device)
        if metric_device.type == "cuda":
            metric_cuda_devices = [metric_device.index if metric_device.index is not None else torch.cuda.current_device()]
    class_weights_device = class_weights.to(args.device)
    class_counts_device = class_counts.to(args.device)
    fixed_graph_teacher_logits = None
    if use_graph_teacher:
        graph_teacher_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device_obj.type, dtype=amp_dtype, enabled=amp_enabled):
                fixed_graph_teacher_logits = graph_teacher_model(local_graph)
    teacher_logits = None
    for local_epoch_index in range(local_epochs):
        local_model.train()
        teacher_logits = None
        if teacher_model is not None:
            teacher_model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=device_obj.type, dtype=amp_dtype, enabled=amp_enabled):
                    teacher_logits = teacher_model(local_graph)
        stage_timer = getattr(args, "stage_timer", None)
        forward_start = time.perf_counter()
        with torch.autocast(device_type=device_obj.type, dtype=amp_dtype, enabled=amp_enabled):
            loss, loss_items = local_model.loss(
                local_graph,
                class_weights=class_weights_device,
                class_counts=class_counts_device,
                teacher_logits=teacher_logits,
                graph_teacher_logits=fixed_graph_teacher_logits,
                current_round=current_round,
            )
        if stage_timer is not None:
            stage_timer.add("forward", time.perf_counter() - forward_start)
        if fedprox_mu > 0.0:
            loss = loss + _fedprox_penalty(local_model, fedprox_reference_state, fedprox_mu, args.device)
        metric_loss = loss.detach()
        metric_loss_items = _detach_loss_items(loss_items)
        optimizer.zero_grad(set_to_none=True)
        backward_start = time.perf_counter()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if stage_timer is not None:
            stage_timer.add("backward", time.perf_counter() - backward_start)
        should_recompute_metrics = (
            epoch_metric_recompute_mode == "all_local_epochs"
            or (
                epoch_metric_recompute_mode == "last_local_epoch_only"
                and local_epoch_index == max(local_epochs - 1, 0)
            )
        )
        if should_recompute_metrics:
            recompute_seed = metric_seed_base + current_round * 1009 + local_epoch_index
            with torch.random.fork_rng(devices=metric_cuda_devices):
                torch.manual_seed(recompute_seed)
                if metric_cuda_devices:
                    torch.cuda.manual_seed_all(recompute_seed)
                local_model.train()
                with torch.no_grad():
                    eval_forward_start = time.perf_counter()
                    with torch.autocast(device_type=device_obj.type, dtype=amp_dtype, enabled=amp_enabled):
                        metric_loss, metric_loss_items = local_model.loss(
                            local_graph,
                            class_weights=class_weights_device,
                            class_counts=class_counts_device,
                            teacher_logits=teacher_logits,
                            graph_teacher_logits=fixed_graph_teacher_logits,
                            current_round=current_round,
                        )
                    if stage_timer is not None:
                        stage_timer.add("eval", time.perf_counter() - eval_forward_start)
                    if fedprox_mu > 0.0:
                        metric_loss = metric_loss + _fedprox_penalty(
                            local_model,
                            fedprox_reference_state,
                            fedprox_mu,
                            args.device,
                        )
                    metric_loss_items = _detach_loss_items(metric_loss_items)
        local_model.train()
        epoch_losses.append(float(metric_loss.item()))
        cls_losses.append(float(metric_loss_items["cls_loss"].item()))
        ranking_cls_losses.append(
            float(metric_loss_items.get("classification_ranking_loss", torch.tensor(0.0, device=args.device)).item())
        )
        pseudo_losses.append(float(metric_loss_items.get("pseudo_loss", torch.tensor(0.0, device=args.device)).item()))
        consistency_losses.append(float(metric_loss_items.get("consistency_loss", torch.tensor(0.0, device=args.device)).item()))
        open_set_losses.append(float(metric_loss_items.get("open_set_loss", torch.tensor(0.0, device=args.device)).item()))
        spread_losses.append(float(metric_loss_items.get("spread_loss", torch.tensor(0.0, device=args.device)).item()))
        graph_aux_losses.append(float(metric_loss_items.get("graph_aux_loss", torch.tensor(0.0, device=args.device)).item()))
        sequence_aux_losses.append(
            float(metric_loss_items.get("sequence_aux_loss", torch.tensor(0.0, device=args.device)).item())
        )
        graph_anchor_losses.append(
            float(metric_loss_items.get("graph_anchor_loss", torch.tensor(0.0, device=args.device)).item())
        )
        graph_teacher_losses.append(
            float(metric_loss_items.get("graph_teacher_loss", torch.tensor(0.0, device=args.device)).item())
        )
        tabular_teacher_losses.append(
            float(metric_loss_items.get("tabular_teacher_loss", torch.tensor(0.0, device=args.device)).item())
        )
        coassociation_losses.append(
            float(metric_loss_items.get("coassociation_loss", torch.tensor(0.0, device=args.device)).item())
        )
        wavelet_alignment_losses.append(
            float(metric_loss_items.get("wavelet_alignment_loss", torch.tensor(0.0, device=args.device)).item())
        )
        utg_alignment_losses.append(
            float(metric_loss_items.get("utg_alignment_loss", torch.tensor(0.0, device=args.device)).item())
        )
        edge_losses.append(float(metric_loss_items["edge_loss"].item()))
        supervised_nodes.append(float(metric_loss_items.get("supervised_nodes", 0)))
        balanced_supervised_nodes.append(float(metric_loss_items.get("balanced_supervised_nodes", 0)))
        regularizer_nodes.append(float(metric_loss_items.get("regularizer_nodes", 0)))
        effective_ssl_nodes.append(float(metric_loss_items.get("effective_ssl_nodes", 0.0)))
        pseudo_nodes.append(float(metric_loss_items.get("pseudo_nodes", 0)))
        novel_nodes.append(float(metric_loss_items.get("novel_nodes", 0)))
        open_set_nodes.append(float(metric_loss_items.get("open_set_nodes", 0)))
        pseudo_thresholds.append(float(metric_loss_items.get("pseudo_threshold_used", 0.0)))
        shared_gap_means.append(float(metric_loss_items.get("shared_gap_mean", torch.tensor(0.0, device=args.device)).item()))
        shared_gap_stds.append(float(metric_loss_items.get("shared_gap_std", torch.tensor(0.0, device=args.device)).item()))
        private_interaction_means.append(
            float(metric_loss_items.get("private_interaction_mean", torch.tensor(0.0, device=args.device)).item())
        )
        private_interaction_stds.append(
            float(metric_loss_items.get("private_interaction_std", torch.tensor(0.0, device=args.device)).item())
        )
        context_gate_means.append(float(metric_loss_items.get("context_gate_mean", torch.tensor(0.0, device=args.device)).item()))
        context_gate_stds.append(float(metric_loss_items.get("context_gate_std", torch.tensor(0.0, device=args.device)).item()))
        graph_branch_gate_means.append(
            float(metric_loss_items.get("graph_branch_gate_mean", torch.tensor(0.0, device=args.device)).item())
        )
        graph_branch_gate_stds.append(
            float(metric_loss_items.get("graph_branch_gate_std", torch.tensor(0.0, device=args.device)).item())
        )
        graph_correction_support_means.append(
            float(metric_loss_items.get("graph_correction_support_mean", torch.tensor(0.0, device=args.device)).item())
        )
        graph_correction_support_stds.append(
            float(metric_loss_items.get("graph_correction_support_std", torch.tensor(0.0, device=args.device)).item())
        )
        sequence_branch_gate_means.append(
            float(metric_loss_items.get("sequence_branch_gate_mean", torch.tensor(0.0, device=args.device)).item())
        )
        sequence_branch_gate_stds.append(
            float(metric_loss_items.get("sequence_branch_gate_std", torch.tensor(0.0, device=args.device)).item())
        )
        raw_branch_gate_means.append(
            float(metric_loss_items.get("raw_branch_gate_mean", torch.tensor(0.0, device=args.device)).item())
        )
        raw_branch_gate_stds.append(
            float(metric_loss_items.get("raw_branch_gate_std", torch.tensor(0.0, device=args.device)).item())
        )
        fusion_delta_gate_means.append(
            float(metric_loss_items.get("fusion_delta_gate_mean", torch.tensor(0.0, device=args.device)).item())
        )
        fusion_delta_gate_stds.append(
            float(metric_loss_items.get("fusion_delta_gate_std", torch.tensor(0.0, device=args.device)).item())
        )
        delta_correction_support_means.append(
            float(metric_loss_items.get("delta_correction_support_mean", torch.tensor(0.0, device=args.device)).item())
        )
        delta_correction_support_stds.append(
            float(metric_loss_items.get("delta_correction_support_std", torch.tensor(0.0, device=args.device)).item())
        )
        time_reliability_means.append(
            float(metric_loss_items.get("time_reliability_mean", torch.tensor(0.0, device=args.device)).item())
        )
        time_reliability_stds.append(
            float(metric_loss_items.get("time_reliability_std", torch.tensor(0.0, device=args.device)).item())
        )
        graph_temporal_gate_means.append(
            float(metric_loss_items.get("graph_temporal_gate_mean", torch.tensor(0.0, device=args.device)).item())
        )
        graph_temporal_gate_stds.append(
            float(metric_loss_items.get("graph_temporal_gate_std", torch.tensor(0.0, device=args.device)).item())
        )
        shared_gate_means.append(float(metric_loss_items.get("shared_gate_mean", torch.tensor(0.0, device=args.device)).item()))
        shared_gate_stds.append(float(metric_loss_items.get("shared_gate_std", torch.tensor(0.0, device=args.device)).item()))
        private_gate_means.append(
            float(metric_loss_items.get("private_gate_mean", torch.tensor(0.0, device=args.device)).item())
        )
        private_gate_stds.append(float(metric_loss_items.get("private_gate_std", torch.tensor(0.0, device=args.device)).item()))
        conflict_score_means.append(
            float(metric_loss_items.get("conflict_score_mean", torch.tensor(0.0, device=args.device)).item())
        )
        conflict_score_stds.append(
            float(metric_loss_items.get("conflict_score_std", torch.tensor(0.0, device=args.device)).item())
        )
        graph_embedding_norm_means.append(
            float(metric_loss_items.get("graph_embedding_norm_mean", torch.tensor(0.0, device=args.device)).item())
        )
        sequence_embedding_norm_means.append(
            float(metric_loss_items.get("sequence_embedding_norm_mean", torch.tensor(0.0, device=args.device)).item())
        )
        raw_embedding_norm_means.append(
            float(metric_loss_items.get("raw_embedding_norm_mean", torch.tensor(0.0, device=args.device)).item())
        )
        fused_embedding_norm_means.append(
            float(metric_loss_items.get("fused_embedding_norm_mean", torch.tensor(0.0, device=args.device)).item())
        )
        sequence_token_valid_ratio_means.append(
            float(metric_loss_items.get("sequence_token_valid_ratio_mean", torch.tensor(0.0, device=args.device)).item())
        )
        sequence_valid_length_means.append(
            float(metric_loss_items.get("sequence_valid_length_mean", torch.tensor(0.0, device=args.device)).item())
        )
        graph_sequence_prob_gap_means.append(
            float(metric_loss_items.get("graph_sequence_prob_gap_mean", torch.tensor(0.0, device=args.device)).item())
        )
        pseudo_reliability_means.append(
            float(metric_loss_items.get("pseudo_reliability_mean", torch.tensor(0.0, device=args.device)).item())
        )
        pseudo_reliability_pass_rates.append(
            float(metric_loss_items.get("pseudo_reliability_pass_rate", torch.tensor(0.0, device=args.device)).item())
        )
        pseudo_modality_agreement_means.append(
            float(metric_loss_items.get("pseudo_modality_agreement_mean", torch.tensor(0.0, device=args.device)).item())
        )
        pseudo_uncertainty_confidence_means.append(
            float(metric_loss_items.get("pseudo_uncertainty_confidence_mean", torch.tensor(0.0, device=args.device)).item())
        )
        pseudo_uncertainty_pass_rates.append(
            float(metric_loss_items.get("pseudo_uncertainty_pass_rate", torch.tensor(0.0, device=args.device)).item())
        )
        pseudo_prototype_margin_means.append(
            float(metric_loss_items.get("pseudo_prototype_margin_mean", torch.tensor(0.0, device=args.device)).item())
        )
        pseudo_nearest_distance_means.append(
            float(metric_loss_items.get("pseudo_nearest_distance_mean", torch.tensor(0.0, device=args.device)).item())
        )
        pseudo_cycle_agreement_means.append(
            float(metric_loss_items.get("pseudo_cycle_agreement_mean", torch.tensor(0.0, device=args.device)).item())
        )
        pseudo_cycle_support_rates.append(
            float(metric_loss_items.get("pseudo_cycle_support_rate", torch.tensor(0.0, device=args.device)).item())
        )
        ssl_reference_spaces.append(str(metric_loss_items.get("ssl_reference_space", "none")))
        wavelet_gate_means.append(
            float(metric_loss_items.get("wavelet_gate_mean", torch.tensor(0.0, device=args.device)).item())
        )
        coassociation_gate_means.append(
            float(metric_loss_items.get("coassociation_gate_mean", torch.tensor(0.0, device=args.device)).item())
        )
        coassociation_density_means.append(
            float(metric_loss_items.get("coassociation_density_mean", torch.tensor(0.0, device=args.device)).item())
        )
        diffusion_gate_means.append(
            float(metric_loss_items.get("diffusion_gate_mean", torch.tensor(0.0, device=args.device)).item())
        )
        diffusion_neighbor_strength_means.append(
            float(metric_loss_items.get("diffusion_neighbor_strength_mean", torch.tensor(0.0, device=args.device)).item())
        )
        utg_temporal_gate_means.append(
            float(metric_loss_items.get("utg_temporal_gate_mean", torch.tensor(0.0, device=args.device)).item())
        )
        if teacher_model is not None:
            ema_update_model(teacher_model, local_model, teacher_ema_decay)
    document_stage = _resolve_document_stage_name(args, training_stage)
    metrics = {
        "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
        "cls_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
        "classification_ranking_loss": float(np.mean(ranking_cls_losses)) if ranking_cls_losses else 0.0,
        "pseudo_loss": float(np.mean(pseudo_losses)) if pseudo_losses else 0.0,
        "consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else 0.0,
        "open_set_loss": float(np.mean(open_set_losses)) if open_set_losses else 0.0,
        "spread_loss": float(np.mean(spread_losses)) if spread_losses else 0.0,
        "graph_aux_loss": float(np.mean(graph_aux_losses)) if graph_aux_losses else 0.0,
        "sequence_aux_loss": float(np.mean(sequence_aux_losses)) if sequence_aux_losses else 0.0,
        "graph_anchor_loss": float(np.mean(graph_anchor_losses)) if graph_anchor_losses else 0.0,
        "graph_teacher_loss": float(np.mean(graph_teacher_losses)) if graph_teacher_losses else 0.0,
        "tabular_teacher_loss": float(np.mean(tabular_teacher_losses)) if tabular_teacher_losses else 0.0,
        "coassociation_loss": float(np.mean(coassociation_losses)) if coassociation_losses else 0.0,
        "wavelet_alignment_loss": float(np.mean(wavelet_alignment_losses)) if wavelet_alignment_losses else 0.0,
        "utg_alignment_loss": float(np.mean(utg_alignment_losses)) if utg_alignment_losses else 0.0,
        "edge_loss": float(np.mean(edge_losses)) if edge_losses else 0.0,
        "supervised_nodes": float(np.mean(supervised_nodes)) if supervised_nodes else 0.0,
        "balanced_supervised_nodes": float(np.mean(balanced_supervised_nodes)) if balanced_supervised_nodes else 0.0,
        "regularizer_nodes": float(np.mean(regularizer_nodes)) if regularizer_nodes else 0.0,
        "effective_ssl_nodes": float(np.mean(effective_ssl_nodes)) if effective_ssl_nodes else 0.0,
        "pseudo_nodes": float(np.mean(pseudo_nodes)) if pseudo_nodes else 0.0,
        "novel_nodes": float(np.mean(novel_nodes)) if novel_nodes else 0.0,
        "open_set_nodes": float(np.mean(open_set_nodes)) if open_set_nodes else 0.0,
        "total_effective_ssl_nodes": float(effective_ssl_nodes[-1]) if effective_ssl_nodes else 0.0,
        "total_pseudo_nodes": float(pseudo_nodes[-1]) if pseudo_nodes else 0.0,
        "total_novel_nodes": float(novel_nodes[-1]) if novel_nodes else 0.0,
        "total_open_set_nodes": float(open_set_nodes[-1]) if open_set_nodes else 0.0,
        "pseudo_threshold_used": float(np.mean(pseudo_thresholds)) if pseudo_thresholds else 0.0,
        "shared_gap_mean": float(np.mean(shared_gap_means)) if shared_gap_means else 0.0,
        "shared_gap_std": float(np.mean(shared_gap_stds)) if shared_gap_stds else 0.0,
        "private_interaction_mean": float(np.mean(private_interaction_means)) if private_interaction_means else 0.0,
        "private_interaction_std": float(np.mean(private_interaction_stds)) if private_interaction_stds else 0.0,
        "context_gate_mean": float(np.mean(context_gate_means)) if context_gate_means else 0.0,
        "context_gate_std": float(np.mean(context_gate_stds)) if context_gate_stds else 0.0,
        "graph_branch_gate_mean": float(np.mean(graph_branch_gate_means)) if graph_branch_gate_means else 0.0,
        "graph_branch_gate_std": float(np.mean(graph_branch_gate_stds)) if graph_branch_gate_stds else 0.0,
        "graph_correction_support_mean": float(np.mean(graph_correction_support_means))
        if graph_correction_support_means
        else 0.0,
        "graph_correction_support_std": float(np.mean(graph_correction_support_stds))
        if graph_correction_support_stds
        else 0.0,
        "sequence_branch_gate_mean": float(np.mean(sequence_branch_gate_means)) if sequence_branch_gate_means else 0.0,
        "sequence_branch_gate_std": float(np.mean(sequence_branch_gate_stds)) if sequence_branch_gate_stds else 0.0,
        "raw_branch_gate_mean": float(np.mean(raw_branch_gate_means)) if raw_branch_gate_means else 0.0,
        "raw_branch_gate_std": float(np.mean(raw_branch_gate_stds)) if raw_branch_gate_stds else 0.0,
        "fusion_delta_gate_mean": float(np.mean(fusion_delta_gate_means)) if fusion_delta_gate_means else 0.0,
        "fusion_delta_gate_std": float(np.mean(fusion_delta_gate_stds)) if fusion_delta_gate_stds else 0.0,
        "delta_correction_support_mean": float(np.mean(delta_correction_support_means))
        if delta_correction_support_means
        else 0.0,
        "delta_correction_support_std": float(np.mean(delta_correction_support_stds))
        if delta_correction_support_stds
        else 0.0,
        "time_reliability_mean": float(np.mean(time_reliability_means)) if time_reliability_means else 0.0,
        "time_reliability_std": float(np.mean(time_reliability_stds)) if time_reliability_stds else 0.0,
        "graph_temporal_gate_mean": float(np.mean(graph_temporal_gate_means)) if graph_temporal_gate_means else 0.0,
        "graph_temporal_gate_std": float(np.mean(graph_temporal_gate_stds)) if graph_temporal_gate_stds else 0.0,
        "shared_gate_mean": float(np.mean(shared_gate_means)) if shared_gate_means else 0.0,
        "shared_gate_std": float(np.mean(shared_gate_stds)) if shared_gate_stds else 0.0,
        "private_gate_mean": float(np.mean(private_gate_means)) if private_gate_means else 0.0,
        "private_gate_std": float(np.mean(private_gate_stds)) if private_gate_stds else 0.0,
        "conflict_score_mean": float(np.mean(conflict_score_means)) if conflict_score_means else 0.0,
        "conflict_score_std": float(np.mean(conflict_score_stds)) if conflict_score_stds else 0.0,
        "graph_embedding_norm_mean": float(np.mean(graph_embedding_norm_means))
        if graph_embedding_norm_means
        else 0.0,
        "sequence_embedding_norm_mean": float(np.mean(sequence_embedding_norm_means))
        if sequence_embedding_norm_means
        else 0.0,
        "raw_embedding_norm_mean": float(np.mean(raw_embedding_norm_means))
        if raw_embedding_norm_means
        else 0.0,
        "fused_embedding_norm_mean": float(np.mean(fused_embedding_norm_means)) if fused_embedding_norm_means else 0.0,
        "sequence_token_valid_ratio_mean": float(np.mean(sequence_token_valid_ratio_means))
        if sequence_token_valid_ratio_means
        else 0.0,
        "sequence_valid_length_mean": float(np.mean(sequence_valid_length_means))
        if sequence_valid_length_means
        else 0.0,
        "graph_sequence_prob_gap_mean": float(np.mean(graph_sequence_prob_gap_means))
        if graph_sequence_prob_gap_means
        else 0.0,
        "pseudo_reliability_mean": float(np.mean(pseudo_reliability_means)) if pseudo_reliability_means else 0.0,
        "pseudo_reliability_pass_rate": float(np.mean(pseudo_reliability_pass_rates))
        if pseudo_reliability_pass_rates
        else 0.0,
        "pseudo_modality_agreement_mean": float(np.mean(pseudo_modality_agreement_means))
        if pseudo_modality_agreement_means
        else 0.0,
        "pseudo_uncertainty_confidence_mean": float(np.mean(pseudo_uncertainty_confidence_means))
        if pseudo_uncertainty_confidence_means
        else 0.0,
        "pseudo_uncertainty_pass_rate": float(np.mean(pseudo_uncertainty_pass_rates))
        if pseudo_uncertainty_pass_rates
        else 0.0,
        "pseudo_prototype_margin_mean": float(np.mean(pseudo_prototype_margin_means))
        if pseudo_prototype_margin_means
        else 0.0,
        "pseudo_nearest_distance_mean": float(np.mean(pseudo_nearest_distance_means))
        if pseudo_nearest_distance_means
        else 0.0,
        "pseudo_cycle_agreement_mean": float(np.mean(pseudo_cycle_agreement_means))
        if pseudo_cycle_agreement_means
        else 0.0,
        "pseudo_cycle_support_rate": float(np.mean(pseudo_cycle_support_rates))
        if pseudo_cycle_support_rates
        else 0.0,
        "ssl_reference_space": _stable_majority_value(ssl_reference_spaces, "none"),
        "wavelet_gate_mean": float(np.mean(wavelet_gate_means)) if wavelet_gate_means else 0.0,
        "coassociation_gate_mean": float(np.mean(coassociation_gate_means)) if coassociation_gate_means else 0.0,
        "coassociation_density_mean": float(np.mean(coassociation_density_means)) if coassociation_density_means else 0.0,
        "diffusion_gate_mean": float(np.mean(diffusion_gate_means)) if diffusion_gate_means else 0.0,
        "diffusion_neighbor_strength_mean": float(np.mean(diffusion_neighbor_strength_means))
        if diffusion_neighbor_strength_means
        else 0.0,
        "utg_temporal_gate_mean": float(np.mean(utg_temporal_gate_means)) if utg_temporal_gate_means else 0.0,
        "training_stage": str(training_stage),
        "document_stage": str(document_stage),
        "epoch_metric_recompute_mode": str(epoch_metric_recompute_mode),
        "amp_enabled": bool(amp_enabled),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "off",
        "amp_note": str(amp_note),
        "graph_learning_rate": float(effective_lrs.get("graph", 0.0)),
        "sequence_learning_rate": float(effective_lrs.get("sequence", 0.0)),
        "fusion_learning_rate": float(effective_lrs.get("fusion", 0.0)),
        "train_nodes": int(local_graph.ndata["train_mask"].sum().item()),
        "num_nodes": int(local_graph.num_nodes(local_graph.ntypes[0])),
    }
    state_dict = {key: value.detach().cpu() for key, value in local_model.state_dict().items()}
    state_dict = apply_dp_noise_to_state_dict(state_dict, dp_noise_std)
    del fedprox_reference_state
    del fixed_graph_teacher_logits
    del class_weights_device
    del class_counts_device
    del scaler
    if teacher_model is not None:
        del teacher_model
    if teacher_logits is not None:
        del teacher_logits
    del local_graph
    del local_model
    del optimizer
    _release_cuda_memory(args.device)
    return state_dict, metrics
