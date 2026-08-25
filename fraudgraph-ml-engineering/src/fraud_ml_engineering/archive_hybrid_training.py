from __future__ import annotations

"""Archive-dataset training entry for the active GNN + Transformer mainline."""

import copy
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .checkpointing import (
    atomic_torch_save as _atomic_torch_save,
    atomic_write_json as _atomic_write_json,
    generate_run_id as _generate_run_id,
    normalize_resume_identity_path as _normalize_resume_identity_path,
    resume_reference_best_metrics as _resume_reference_best_metrics,
    run_metadata_payload as _run_metadata_payload,
    should_inherit_resume_best_metrics as _should_inherit_resume_best_metrics,
    validated_resume_state_dict as _validated_resume_state_dict,
)
from .evaluator import evaluate_model as _evaluate_model
from .algorithms import (
    _apply_active_learning_reveal,
    _apply_label_scarcity_runtime_defaults,
    _checkpoint_selection_guard,
    _close_rl_models,
    _controller_reward_from_round,
    _dataset_multimodal_aux_loss_profile,
    _dataset_multimodal_fusion_profile,
    _dataset_output_regularization,
    _force_mainline_gnn_transformer_mode,
    _plan_round,
    _refresh_client_training_masks,
    _resolve_label_scarcity_profile,
    _resolve_profile_backed_weight,
    _resolve_structure_only_ablation_mode,
    _resolve_training_device,
    _select_active_learning_queries,
    _should_update_best_checkpoint,
    train_controller_models,
)
from .archive_dataset import ARCHIVE_DEFAULT_ROOT, load_archive_dataset
from .device_utils import DEFAULT_DEVICE_REQUEST
from .vendor.splitgnn.utils import setup_seed
from .hybrid_task_model import HybridFraudModel, checkpoint_legacy_fusion_only
from .model_state_utils import snapshot_model_state_to_cpu
from .paths import ARTIFACTS_ROOT
from .trainer_local import local_train_round as _local_train

ARCHIVE_RESULT_ROOT = ARTIFACTS_ROOT / "archive_training"


def _resolve_archive_result_root(result_root: str | Path | None = None) -> Path:
    root = Path(result_root).expanduser() if result_root else ARCHIVE_RESULT_ROOT
    return root.resolve()


def _state_dict_average(
    state_dicts: List[dict[str, torch.Tensor]],
    weights: List[float],
) -> dict[str, torch.Tensor]:
    """Average model state dicts locally for the archived-FL/RL-free mainline."""
    normalized_weights = [weight / max(sum(weights), 1.0) for weight in weights]
    averaged_state = copy.deepcopy(state_dicts[0])
    for key in averaged_state.keys():
        reference_tensor = averaged_state[key]
        if torch.is_floating_point(reference_tensor):
            combined = None
            for state_dict, weight in zip(state_dicts, normalized_weights):
                value = state_dict[key].detach().cpu() * weight
                combined = value if combined is None else combined + value
            averaged_state[key] = combined.to(reference_tensor.dtype)
        else:
            averaged_state[key] = reference_tensor.clone()
    return averaged_state


def _build_archive_training_args(
    *,
    dataset_name: str,
    device: str,
    federated_rounds: int,
    local_epochs: int,
    num_clients: int,
    client_hops: int,
    label_fraction: float,
    edge_loss_weight: float,
    classification_loss: str,
    focal_gamma: float,
    class_balance_beta: float,
    pseudo_label_threshold: float,
    pseudo_label_weight: float,
    pseudo_label_novelty_threshold: float,
    consistency_weight: float,
    active_learning_budget_per_round: int,
    active_learning_delay_rounds: int,
    active_learning_novelty_weight: float,
    active_learning_diversity_weight: float,
    active_learning_candidate_pool_scale: int,
    fedprox_mu: float,
    dp_noise_std: float,
    transformer_hidden_dim: int,
    transformer_num_layers: int,
    fusion_hidden_dim: int,
    planner_mode: str,
    early_stop: int,
    test_every: int,
    fixed_precision_target: float,
    active_learning_feedback_path: str,
    seed: int | None,
    result_root: str,
    disable_gnn: bool,
    disable_transformer: bool,
    disable_federated: bool,
    learning_rate_override: float | None,
    weight_decay_override: float | None,
    dropout_override: float | None,
    graph_aux_loss_weight_override: float | None,
    sequence_aux_loss_weight_override: float | None,
    graph_gate_logit_bias_override: float | None,
    eval_graph_gate_logit_bias_override: float | None,
    graph_residual_min_gate_override: float | None,
    sequence_residual_scale_override: float | None,
    skip_test_evaluation: bool,
    data_root: str | Path,
    max_users: int | None,
    max_transactions: int | None,
    risk_positive_ratio: float,
    force_preview: bool,
) -> SimpleNamespace:
    args = SimpleNamespace()
    args.seed = None if seed is None else int(seed)
    args.requested_device = str(device)
    args.device = _resolve_training_device(device)
    args.dataset = str(dataset_name)
    base_lr = float(learning_rate_override) if learning_rate_override is not None else 5e-3
    args.lr = base_lr
    args.base_lr = float(args.lr)
    args.weight_decay = float(weight_decay_override) if weight_decay_override is not None else 5e-5
    args.dropout = float(dropout_override) if dropout_override is not None else 0.1
    args.gamma = 1.0
    args.C = 1
    args.K = 0
    args.intra_dim = 8
    args.n_class = 2
    args.transformer_hidden_dim = int(transformer_hidden_dim)
    args.seq_hidden_dim = int(transformer_hidden_dim)
    args.transformer_num_layers = max(int(transformer_num_layers), 1)
    args.fusion_hidden_dim = int(fusion_hidden_dim)
    args.feature_hidden_dim = max(int(fusion_hidden_dim), int(transformer_hidden_dim))
    args.edge_loss_weight = float(edge_loss_weight)
    output_regularization = _dataset_output_regularization(dataset_name)
    args.target_prob_std = float(output_regularization["target_prob_std"])
    args.prob_std_regularization_weight = float(output_regularization["prob_std_regularization_weight"])
    multimodal_aux_enabled = not bool(disable_gnn) and not bool(disable_transformer)
    multimodal_aux_profile = _dataset_multimodal_aux_loss_profile(dataset_name)
    default_graph_aux_loss_weight = float(multimodal_aux_profile["graph_aux_loss_weight"]) if multimodal_aux_enabled else 0.0
    default_sequence_aux_loss_weight = (
        float(multimodal_aux_profile["sequence_aux_loss_weight"]) if multimodal_aux_enabled else 0.0
    )
    args.graph_aux_loss_weight = (
        float(graph_aux_loss_weight_override)
        if multimodal_aux_enabled and graph_aux_loss_weight_override is not None
        else default_graph_aux_loss_weight
    )
    args.sequence_aux_loss_weight = (
        float(sequence_aux_loss_weight_override)
        if multimodal_aux_enabled and sequence_aux_loss_weight_override is not None
        else default_sequence_aux_loss_weight
    )
    fusion_profile = _dataset_multimodal_fusion_profile(dataset_name)
    default_graph_gate_logit_bias = float(fusion_profile["graph_gate_logit_bias"]) if multimodal_aux_enabled else 0.0
    default_eval_graph_gate_logit_bias = (
        float(fusion_profile.get("eval_graph_gate_logit_bias", fusion_profile["graph_gate_logit_bias"]))
        if multimodal_aux_enabled
        else 0.0
    )
    default_graph_residual_min_gate = float(fusion_profile["graph_residual_min_gate"]) if multimodal_aux_enabled else 0.0
    default_sequence_residual_scale = float(fusion_profile["sequence_residual_scale"]) if multimodal_aux_enabled else 1.0
    args.graph_gate_logit_bias = (
        float(graph_gate_logit_bias_override)
        if multimodal_aux_enabled and graph_gate_logit_bias_override is not None
        else default_graph_gate_logit_bias
    )
    if multimodal_aux_enabled and eval_graph_gate_logit_bias_override is not None:
        args.eval_graph_gate_logit_bias = float(eval_graph_gate_logit_bias_override)
    elif multimodal_aux_enabled and graph_gate_logit_bias_override is not None:
        args.eval_graph_gate_logit_bias = float(args.graph_gate_logit_bias)
    else:
        args.eval_graph_gate_logit_bias = default_eval_graph_gate_logit_bias
    args.graph_residual_min_gate = (
        float(graph_residual_min_gate_override)
        if multimodal_aux_enabled and graph_residual_min_gate_override is not None
        else default_graph_residual_min_gate
    )
    args.sequence_residual_scale = (
        float(sequence_residual_scale_override)
        if multimodal_aux_enabled and sequence_residual_scale_override is not None
        else default_sequence_residual_scale
    )
    args.requested_classification_loss = str(classification_loss).lower()
    args.classification_loss = str(classification_loss).lower()
    args.focal_gamma = float(focal_gamma)
    args.class_balance_beta = float(class_balance_beta)
    args.label_fraction = float(label_fraction)
    scarcity_profile = _resolve_label_scarcity_profile(dataset_name, args.label_fraction)
    args.label_scarcity_profile = str(scarcity_profile["name"])
    args.label_scarcity_profile_settings = scarcity_profile
    args.requested_pseudo_label_threshold = float(pseudo_label_threshold)
    args.requested_pseudo_label_weight = float(pseudo_label_weight)
    args.requested_pseudo_label_novelty_threshold = float(pseudo_label_novelty_threshold)
    args.requested_consistency_weight = float(consistency_weight)
    args.pseudo_label_threshold = float(max(pseudo_label_threshold, scarcity_profile["pseudo_label_threshold"]))
    args.pseudo_label_min_threshold = float(scarcity_profile["pseudo_label_min_threshold"])
    args.pseudo_label_top_fraction = float(scarcity_profile["pseudo_label_top_fraction"])
    args.pseudo_label_weight = float(min(pseudo_label_weight, scarcity_profile["pseudo_label_weight"]))
    args.pseudo_label_novelty_threshold = float(
        min(pseudo_label_novelty_threshold, scarcity_profile["pseudo_label_novelty_threshold"])
    )
    args.teacher_ema_decay = float(scarcity_profile["teacher_ema_decay"])
    args.teacher_temperature = float(scarcity_profile["teacher_temperature"])
    args.pseudo_warmup_rounds = int(scarcity_profile["pseudo_warmup_rounds"])
    args.pseudo_ramp_rounds = int(scarcity_profile["pseudo_ramp_rounds"])
    args.open_set_novelty_threshold = float(scarcity_profile["open_set_novelty_threshold"])
    args.open_set_loss_weight = float(scarcity_profile["open_set_loss_weight"])
    args.requested_active_learning_budget_per_round = int(active_learning_budget_per_round)
    args.requested_active_learning_delay_rounds = int(active_learning_delay_rounds)
    args.requested_active_learning_novelty_weight = float(active_learning_novelty_weight)
    args.requested_active_learning_diversity_weight = float(active_learning_diversity_weight)
    args.active_learning_budget_per_round = max(int(active_learning_budget_per_round), 0)
    args.active_learning_delay_rounds = max(int(active_learning_delay_rounds), 0)
    args.consistency_weight = _resolve_profile_backed_weight(
        consistency_weight,
        scarcity_profile["consistency_weight"],
        zero_disables=True,
    )
    args.active_learning_novelty_weight = _resolve_profile_backed_weight(
        active_learning_novelty_weight,
        scarcity_profile["active_learning_novelty_weight"],
        zero_disables=True,
    )
    args.active_learning_diversity_weight = _resolve_profile_backed_weight(
        active_learning_diversity_weight,
        scarcity_profile["active_learning_diversity_weight"],
        zero_disables=True,
    )
    args.active_learning_candidate_pool_scale = max(int(active_learning_candidate_pool_scale), 1)
    args.fedprox_mu = float(fedprox_mu)
    args.dp_noise_std = float(dp_noise_std)
    args.requested_planner_mode = str(planner_mode).lower()
    args.requested_disable_federated = bool(disable_federated)
    args.fixed_precision_target = float(fixed_precision_target)
    args.skip_test_evaluation = bool(skip_test_evaluation)
    args.planner_mode = str(planner_mode).lower()
    args.disable_gnn = bool(disable_gnn)
    args.disable_transformer = bool(disable_transformer)
    args.disable_federated = bool(disable_federated)
    args.gnn_enabled = not args.disable_gnn
    args.transformer_enabled = not args.disable_transformer
    args.federated_enabled = not args.disable_federated
    args.drl_enabled = args.planner_mode == "rl"
    args.ablation_mode = _resolve_structure_only_ablation_mode(
        disable_gnn=args.disable_gnn,
        disable_transformer=args.disable_transformer,
    )
    args.active_learning_feedback_path = _normalize_resume_identity_path(active_learning_feedback_path)
    args.federated_rounds = int(federated_rounds)
    args.local_epochs = int(local_epochs)
    args.num_clients = int(num_clients)
    args.requested_num_clients = int(num_clients)
    args.client_hops = int(client_hops)
    args.result_root = str(_resolve_archive_result_root(result_root))
    args.result_path = str(Path(args.result_root) / dataset_name)
    args.early_stop = max(int(early_stop), 0)
    args.test_every = max(int(test_every), 0)
    args.archive_data_root = str(Path(data_root).expanduser().resolve())
    args.archive_max_users = None if max_users is None else int(max_users)
    args.archive_max_transactions = None if max_transactions is None else int(max_transactions)
    args.archive_risk_positive_ratio = float(risk_positive_ratio)
    args.archive_force_preview = bool(force_preview)
    return args


def _train_archive_dataset(
    *,
    dataset_name: str,
    federated_rounds: int,
    local_epochs: int,
    requested_base_local_epochs: int | None,
    requested_extra_local_epochs: int | None,
    num_clients: int,
    client_hops: int,
    label_fraction: float,
    rl_timesteps: int,
    device: str,
    edge_loss_weight: float,
    classification_loss: str,
    focal_gamma: float,
    class_balance_beta: float,
    pseudo_label_threshold: float,
    pseudo_label_weight: float,
    pseudo_label_novelty_threshold: float,
    consistency_weight: float,
    active_learning_budget_per_round: int,
    active_learning_delay_rounds: int,
    active_learning_novelty_weight: float,
    active_learning_diversity_weight: float,
    active_learning_candidate_pool_scale: int,
    fedprox_mu: float,
    dp_noise_std: float,
    planner_mode: str,
    early_stop: int,
    test_every: int,
    fixed_precision_target: float,
    resume_path: str,
    enable_tensorboard: bool,
    transformer_hidden_dim: int,
    transformer_num_layers: int,
    fusion_hidden_dim: int,
    active_learning_feedback_path: str,
    rl_models: List | None,
    seed: int | None,
    result_root: str,
    disable_gnn: bool,
    disable_transformer: bool,
    disable_federated: bool,
    learning_rate_override: float | None,
    weight_decay_override: float | None,
    dropout_override: float | None,
    graph_aux_loss_weight_override: float | None,
    sequence_aux_loss_weight_override: float | None,
    graph_gate_logit_bias_override: float | None,
    eval_graph_gate_logit_bias_override: float | None,
    graph_residual_min_gate_override: float | None,
    sequence_residual_scale_override: float | None,
    skip_test_evaluation: bool,
    data_root: str | Path,
    max_users: int | None,
    max_transactions: int | None,
    risk_positive_ratio: float,
    force_preview: bool,
) -> dict:
    resolved_result_root = _resolve_archive_result_root(result_root)
    effective_num_clients = 1 if disable_federated else int(num_clients)
    args = _build_archive_training_args(
        dataset_name=dataset_name,
        device=device,
        federated_rounds=federated_rounds,
        local_epochs=local_epochs,
        num_clients=effective_num_clients,
        client_hops=client_hops,
        label_fraction=label_fraction,
        edge_loss_weight=edge_loss_weight,
        classification_loss=classification_loss,
        focal_gamma=focal_gamma,
        class_balance_beta=class_balance_beta,
        pseudo_label_threshold=pseudo_label_threshold,
        pseudo_label_weight=pseudo_label_weight,
        pseudo_label_novelty_threshold=pseudo_label_novelty_threshold,
        consistency_weight=consistency_weight,
        active_learning_budget_per_round=active_learning_budget_per_round,
        active_learning_delay_rounds=active_learning_delay_rounds,
        active_learning_novelty_weight=active_learning_novelty_weight,
        active_learning_diversity_weight=active_learning_diversity_weight,
        active_learning_candidate_pool_scale=active_learning_candidate_pool_scale,
        fedprox_mu=fedprox_mu,
        dp_noise_std=dp_noise_std,
        transformer_hidden_dim=transformer_hidden_dim,
        transformer_num_layers=transformer_num_layers,
        fusion_hidden_dim=fusion_hidden_dim,
        planner_mode=planner_mode,
        early_stop=early_stop,
        test_every=test_every,
        fixed_precision_target=fixed_precision_target,
        active_learning_feedback_path=active_learning_feedback_path,
        seed=seed,
        result_root=str(resolved_result_root),
        disable_gnn=disable_gnn,
        disable_transformer=disable_transformer,
        disable_federated=disable_federated,
        learning_rate_override=learning_rate_override,
        weight_decay_override=weight_decay_override,
        dropout_override=dropout_override,
        graph_aux_loss_weight_override=graph_aux_loss_weight_override,
        sequence_aux_loss_weight_override=sequence_aux_loss_weight_override,
        graph_gate_logit_bias_override=graph_gate_logit_bias_override,
        eval_graph_gate_logit_bias_override=eval_graph_gate_logit_bias_override,
        graph_residual_min_gate_override=graph_residual_min_gate_override,
        sequence_residual_scale_override=sequence_residual_scale_override,
        skip_test_evaluation=skip_test_evaluation,
        data_root=data_root,
        max_users=max_users,
        max_transactions=max_transactions,
        risk_positive_ratio=risk_positive_ratio,
        force_preview=force_preview,
    )
    args.requested_num_clients = int(num_clients)
    args.requested_base_local_epochs = int(max(requested_base_local_epochs or local_epochs, 1))
    args.requested_extra_local_epochs = int(max(requested_extra_local_epochs or 1, 1))
    os.makedirs(args.result_path, exist_ok=True)

    bundle = load_archive_dataset(
        data_root=data_root,
        dataset_name=dataset_name,
        num_clients=effective_num_clients,
        seed=42 if args.seed is None else int(args.seed),
        client_hops=client_hops,
        label_fraction=label_fraction,
        active_learning_feedback_path=str(getattr(args, "active_learning_feedback_path", "")),
        max_users=max_users,
        max_transactions=max_transactions,
        risk_positive_ratio=risk_positive_ratio,
        force_preview=force_preview,
    )
    bundle.base_lr = float(args.lr)
    _apply_label_scarcity_runtime_defaults(bundle, args)
    _refresh_client_training_masks(bundle)
    initial_supervised_train_nodes = int(bundle.graph.ndata["train_supervised_mask"].sum().item())
    initial_unlabeled_train_nodes = int(bundle.graph.ndata["train_unlabeled_mask"].sum().item())

    normalized_planner_mode = str(planner_mode).lower()
    if normalized_planner_mode not in {"rl", "deterministic"}:
        raise ValueError(f"Unsupported planner mode: {planner_mode}")
    owns_rl_models = normalized_planner_mode == "rl" and rl_models is None
    dataset_seed = 42 if getattr(args, "seed", None) is None else int(args.seed)
    if normalized_planner_mode == "rl" and rl_models is None:
        rl_models = train_controller_models(rl_timesteps=rl_timesteps, seed=dataset_seed)
    if normalized_planner_mode == "deterministic":
        rl_models = None

    setup_seed(dataset_seed)
    global_model = HybridFraudModel(args, bundle.graph).to(args.device)
    args.legacy_fusion_only = bool(getattr(args, "legacy_fusion_only", False))
    warm_start_payload = None
    if resume_path:
        resume_file = Path(resume_path)
        if not resume_file.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_file}")
        warm_start_payload = torch.load(resume_file, map_location=args.device)
        global_model.load_state_dict(
            _validated_resume_state_dict(
                current_args=args,
                warm_start_payload=warm_start_payload,
                resume_file=resume_file,
                current_state_dict=global_model.state_dict(),
            ),
            strict=True,
        )
        legacy_fusion_only = checkpoint_legacy_fusion_only(warm_start_payload)
        global_model.legacy_fusion_only = legacy_fusion_only
        args.legacy_fusion_only = legacy_fusion_only

    run_id = _generate_run_id()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    run_status_path = Path(args.result_path) / f"{dataset_name}_hybrid_run_status.json"
    _atomic_write_json(
        run_status_path,
        {
            "dataset": dataset_name,
            "run_id": run_id,
            "status": "running",
            "started_at": started_at,
            "planner_mode": normalized_planner_mode,
            "resolved_device": str(args.device),
            "seed": None if getattr(args, "seed", None) is None else int(args.seed),
            "data_summary": getattr(bundle, "data_summary", {}),
        },
    )

    writer = None
    tb_logdir = ""
    run_metadata_path: Path | None = None
    model_path = Path(args.result_path) / f"{dataset_name}_hybrid_fraudgraph.pt"
    result_prefix = Path(args.result_path) / f"{dataset_name}_hybrid_fraudgraph"
    summary_path = Path(args.result_path) / f"{dataset_name}_hybrid_summary.json"
    if enable_tensorboard:
        tb_dir = resolved_result_root / dataset_name / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir / run_id))
        tb_logdir = writer.log_dir
        run_metadata_path = Path(tb_logdir) / "run_metadata.json"
        _atomic_write_json(
            run_metadata_path,
            _run_metadata_payload(
                dataset_name=dataset_name,
                run_id=run_id,
                status="running",
                federated_rounds=federated_rounds,
                local_epochs=local_epochs,
                planner_mode=args.planner_mode,
                test_every=args.test_every,
                resume_path=resume_path,
                tb_logdir=tb_logdir,
                seed=None if getattr(args, "seed", None) is None else int(args.seed),
            ),
        )

    best_state = snapshot_model_state_to_cpu(global_model)
    peak_state = snapshot_model_state_to_cpu(global_model)
    best_valid_auc = -1.0
    best_valid_gmean = -1.0
    best_valid_pr_auc = -1.0
    best_valid_recall_at_precision = -1.0
    best_valid_f1_macro = -1.0
    peak_valid_auc_any_round = -1.0
    best_round = -1
    peak_valid_round_any_round = -1
    best_valid_threshold = 0.5
    peak_valid_threshold_any_round = 0.5
    resume_metrics_rebased = False
    resume_best_metric_inheritance = "not_applicable"
    resume_best_metric_reason = ""
    resume_reference_best_metrics: dict[str, float] = {}
    resume_recheck_valid_metrics: dict[str, float] = {}
    if warm_start_payload is not None:
        resume_reference_best_metrics = _resume_reference_best_metrics(warm_start_payload)
        resume_recheck_valid_metrics = _evaluate_model(global_model, bundle.graph, "valid_mask", args.device)
        stored_best_round = int(warm_start_payload.get("best_round", -1))
        if bool(warm_start_payload.get("protocol_isolated", False)):
            resume_metrics_rebased = True
            resume_best_metric_inheritance = "rebased"
            resume_best_metric_reason = "protocol_isolated_checkpoint"
        else:
            inherit_resume_best_metrics, resume_best_metric_reason = _should_inherit_resume_best_metrics(
                current_valid_metrics=resume_recheck_valid_metrics,
                stored_best_metrics=resume_reference_best_metrics,
            )
            resume_metrics_rebased = not inherit_resume_best_metrics
            resume_best_metric_inheritance = "inherited" if inherit_resume_best_metrics else "rebased"
        if not resume_metrics_rebased:
            best_valid_auc = float(warm_start_payload.get("best_valid_auc", best_valid_auc))
            best_valid_gmean = float(
                warm_start_payload.get(
                    "best_valid_gmean",
                    warm_start_payload.get("best_valid_metrics", {}).get("gmean", best_valid_gmean),
                )
            )
            best_valid_pr_auc = float(
                warm_start_payload.get(
                    "best_valid_pr_auc",
                    warm_start_payload.get("best_valid_metrics", {}).get("pr_auc", best_valid_pr_auc),
                )
            )
            best_valid_recall_at_precision = float(
                warm_start_payload.get(
                    "best_valid_recall_at_precision",
                    warm_start_payload.get("best_valid_metrics", {}).get(
                        "recall_at_precision",
                        best_valid_recall_at_precision,
                    ),
                )
            )
            best_valid_f1_macro = float(
                warm_start_payload.get(
                    "best_valid_f1_macro",
                    warm_start_payload.get("best_valid_metrics", {}).get("f1_macro", best_valid_f1_macro),
                )
            )
            best_round = int(warm_start_payload.get("best_round", best_round))
            best_valid_threshold = float(warm_start_payload.get("best_valid_threshold", best_valid_threshold))
            peak_valid_auc_any_round = float(warm_start_payload.get("peak_valid_auc_any_round", best_valid_auc))
            peak_valid_round_any_round = int(warm_start_payload.get("peak_valid_round_any_round", best_round))
            peak_valid_threshold_any_round = float(
                warm_start_payload.get("peak_valid_threshold_any_round", best_valid_threshold)
            )
        else:
            best_valid_auc = float(resume_recheck_valid_metrics["auc"])
            best_valid_gmean = float(resume_recheck_valid_metrics["gmean"])
            best_valid_pr_auc = float(resume_recheck_valid_metrics["pr_auc"])
            best_valid_recall_at_precision = float(resume_recheck_valid_metrics["recall_at_precision"])
            best_valid_f1_macro = float(resume_recheck_valid_metrics["f1_macro"])
            best_round = stored_best_round
            best_valid_threshold = float(resume_recheck_valid_metrics["threshold"])
            peak_valid_auc_any_round = float(resume_recheck_valid_metrics["auc"])
            peak_valid_round_any_round = stored_best_round
            peak_valid_threshold_any_round = float(resume_recheck_valid_metrics["threshold"])
        print(
            f"[{dataset_name}] resume_metric_check "
            f"decision={resume_best_metric_inheritance} "
            f"stored_auc={float(resume_reference_best_metrics.get('auc', -1.0)):.6f} "
            f"current_auc={float(resume_recheck_valid_metrics.get('auc', -1.0)):.6f} "
            f"reason={resume_best_metric_reason}",
            flush=True,
        )

    patience = 0
    history = []
    has_test_timeseries = False
    pending_active_learning_feedback: List[dict] = []
    data_summary = copy.deepcopy(getattr(bundle, "data_summary", {}))
    try:
        for round_index in tqdm(range(federated_rounds), desc=f"ArchiveHybrid-{dataset_name}"):
            pending_active_learning_feedback, revealed_feedback = _apply_active_learning_reveal(
                bundle=bundle,
                pending_feedback=pending_active_learning_feedback,
                current_round=round_index,
            )
            round_plan = _plan_round(
                rl_models=rl_models,
                planner_mode=normalized_planner_mode,
                bundle=bundle,
                round_index=round_index,
                total_rounds=federated_rounds,
                base_local_epochs=local_epochs,
                base_edge_loss_weight=edge_loss_weight,
                metrics_history=history,
            )
            selected_clients = [client for client in bundle.clients if client.client_id in set(round_plan["selected_clients"])]
            local_states = []
            local_weights = []
            local_losses = []
            local_cls_losses = []
            local_pseudo_losses = []
            local_consistency_losses = []
            local_open_set_losses = []
            local_spread_losses = []
            local_graph_aux_losses = []
            local_sequence_aux_losses = []
            local_edge_losses = []
            local_pseudo_nodes = []
            local_novel_nodes = []
            local_open_set_nodes = []
            local_pseudo_thresholds = []
            global_state = global_model.state_dict()

            for client in selected_clients:
                state_dict, local_metrics = _local_train(
                    global_model=global_model,
                    subgraph=client.subgraph,
                    global_state=global_state,
                    class_weights=bundle.class_weights,
                    class_counts=bundle.class_counts,
                    args=args,
                    current_round=round_index,
                    local_epochs=round_plan["local_epochs"],
                    grad_clip=round_plan["grad_clip"],
                    edge_loss_weight=round_plan["edge_loss_weight"],
                    learning_rate=round_plan["learning_rate"],
                    fedprox_mu=float(getattr(args, "fedprox_mu", 0.0)),
                    dp_noise_std=float(getattr(args, "dp_noise_std", 0.0)),
                )
                local_states.append(state_dict)
                local_weights.append(max(client.train_nodes, 1))
                local_losses.append(local_metrics["loss"])
                local_cls_losses.append(local_metrics["cls_loss"])
                local_pseudo_losses.append(local_metrics["pseudo_loss"])
                local_consistency_losses.append(local_metrics["consistency_loss"])
                local_open_set_losses.append(local_metrics["open_set_loss"])
                local_spread_losses.append(local_metrics["spread_loss"])
                local_graph_aux_losses.append(local_metrics.get("graph_aux_loss", 0.0))
                local_sequence_aux_losses.append(local_metrics.get("sequence_aux_loss", 0.0))
                local_edge_losses.append(local_metrics["edge_loss"])
                local_pseudo_nodes.append(local_metrics["pseudo_nodes"])
                local_novel_nodes.append(local_metrics["novel_nodes"])
                local_open_set_nodes.append(local_metrics["open_set_nodes"])
                local_pseudo_thresholds.append(local_metrics["pseudo_threshold_used"])

            if not local_states:
                continue

            aggregated_state = _state_dict_average(local_states, local_weights)
            global_model.load_state_dict(aggregated_state)
            valid_metrics = _evaluate_model(global_model, bundle.graph, "valid_mask", args.device)
            round_metrics = {
                "round": round_index,
                "mean_local_loss": float(np.mean(local_losses)) if local_losses else 0.0,
                "mean_local_cls_loss": float(np.mean(local_cls_losses)) if local_cls_losses else 0.0,
                "mean_local_pseudo_loss": float(np.mean(local_pseudo_losses)) if local_pseudo_losses else 0.0,
                "mean_local_consistency_loss": float(np.mean(local_consistency_losses)) if local_consistency_losses else 0.0,
                "mean_local_open_set_loss": float(np.mean(local_open_set_losses)) if local_open_set_losses else 0.0,
                "mean_local_spread_loss": float(np.mean(local_spread_losses)) if local_spread_losses else 0.0,
                "mean_local_graph_aux_loss": float(np.mean(local_graph_aux_losses)) if local_graph_aux_losses else 0.0,
                "mean_local_sequence_aux_loss": float(np.mean(local_sequence_aux_losses))
                if local_sequence_aux_losses
                else 0.0,
                "mean_local_edge_loss": float(np.mean(local_edge_losses)) if local_edge_losses else 0.0,
                "mean_local_pseudo_nodes": float(np.mean(local_pseudo_nodes)) if local_pseudo_nodes else 0.0,
                "mean_local_novel_nodes": float(np.mean(local_novel_nodes)) if local_novel_nodes else 0.0,
                "mean_local_open_set_nodes": float(np.mean(local_open_set_nodes)) if local_open_set_nodes else 0.0,
                "mean_local_pseudo_threshold": float(np.mean(local_pseudo_thresholds)) if local_pseudo_thresholds else 0.0,
                "valid_acc": valid_metrics["acc"],
                "valid_f1_binary": valid_metrics["f1_binary"],
                "valid_f1_pos": valid_metrics["f1_pos"],
                "valid_auc": valid_metrics["auc"],
                "valid_pr_auc": valid_metrics["pr_auc"],
                "valid_f1_macro": valid_metrics["f1_macro"],
                "valid_gmean": valid_metrics["gmean"],
                "valid_recall": valid_metrics["recall"],
                "valid_recall_at_precision": valid_metrics["recall_at_precision"],
                "valid_recall_at_precision_threshold": valid_metrics["recall_at_precision_threshold"],
                "fixed_precision_target": valid_metrics["fixed_precision_target"],
                "valid_threshold": valid_metrics["threshold"],
                "valid_positive_rate": valid_metrics["positive_rate"],
                "valid_prob_mean": valid_metrics["prob_mean"],
                "valid_prob_std": valid_metrics["prob_std"],
                "selected_clients": round_plan["selected_clients"],
                "local_epochs": round_plan["local_epochs"],
                "grad_clip": round_plan["grad_clip"],
                "selected_ratio": round_plan["selected_ratio"],
                "actual_selected_ratio": round_plan["actual_selected_ratio"],
                "selection_phase": round_plan["selection_phase"],
                "raw_action": round_plan["raw_action"],
                "planner_mode": round_plan["planner_mode"],
                "edge_loss_weight": round_plan["edge_loss_weight"],
                "learning_rate": round_plan["learning_rate"],
                "probability_collapse": round_plan["probability_collapse"],
                "severe_probability_collapse": round_plan["severe_probability_collapse"],
                "auc_plateau": round_plan["auc_plateau"],
                "stagnation_rounds": round_plan["stagnation_rounds"],
                "sharp_auc_drop": round_plan["sharp_auc_drop"],
                "recent_auc_delta": round_plan["recent_auc_delta"],
                "observation": round_plan["observation"],
                "round_progress": float((round_index + 1) / max(federated_rounds, 1)),
                "active_learning_revealed": int(len(revealed_feedback)),
                "active_learning_pending": int(len(pending_active_learning_feedback)),
                "supervised_train_nodes": int(bundle.graph.ndata["train_supervised_mask"].sum().item()),
                "unlabeled_train_nodes": int(bundle.graph.ndata["train_unlabeled_mask"].sum().item()),
            }
            controller_reward, reward_details = _controller_reward_from_round(
                dataset_name=dataset_name,
                round_index=round_index,
                total_rounds=federated_rounds,
                base_local_epochs=local_epochs,
                previous_metrics=history[-1] if history else None,
                current_metrics=round_metrics,
                round_plan=round_plan,
            )
            round_metrics["controller_reward"] = float(controller_reward)
            round_metrics["controller_reward_details"] = reward_details
            queried_feedback = _select_active_learning_queries(
                model=global_model,
                bundle=bundle,
                args=args,
                threshold=float(valid_metrics["threshold"]),
                current_round=round_index,
                pending_node_ids={int(item["node_id"]) for item in pending_active_learning_feedback},
            )
            if queried_feedback:
                pending_active_learning_feedback.extend(queried_feedback)
            round_metrics["active_learning_queried"] = int(len(queried_feedback))
            round_metrics["active_learning_pending"] = int(len(pending_active_learning_feedback))
            checkpoint_guard = _checkpoint_selection_guard(dataset_name, valid_metrics)
            round_metrics["checkpoint_selection_eligible"] = bool(checkpoint_guard["eligible"])
            round_metrics["checkpoint_selection_block_reason"] = str(checkpoint_guard["reason"])
            history.append(round_metrics)

            current_best_auc = max(float(best_valid_auc), float(valid_metrics["auc"])) if best_valid_auc >= 0 else float(valid_metrics["auc"])
            current_best_round = best_round if best_round >= 0 else round_index
            print(
                f"[{dataset_name}] round {round_index + 1}/{federated_rounds} "
                f"valid_auc={float(valid_metrics['auc']):.6f} "
                f"valid_pr_auc={float(valid_metrics['pr_auc']):.6f} "
                f"valid_f1_macro={float(valid_metrics['f1_macro']):.6f} "
                f"valid_gmean={float(valid_metrics['gmean']):.6f} "
                f"best_valid_auc={current_best_auc:.6f} "
                f"best_round={current_best_round + 1}",
                flush=True,
            )

            if writer is not None:
                writer.add_scalar("train/mean_local_loss", round_metrics["mean_local_loss"], round_index)
                writer.add_scalar("train/mean_local_cls_loss", round_metrics["mean_local_cls_loss"], round_index)
                writer.add_scalar("train/mean_local_pseudo_loss", round_metrics["mean_local_pseudo_loss"], round_index)
                writer.add_scalar("train/mean_local_consistency_loss", round_metrics["mean_local_consistency_loss"], round_index)
                writer.add_scalar("train/mean_local_open_set_loss", round_metrics["mean_local_open_set_loss"], round_index)
                writer.add_scalar("train/mean_local_spread_loss", round_metrics["mean_local_spread_loss"], round_index)
                writer.add_scalar("train/mean_local_graph_aux_loss", round_metrics["mean_local_graph_aux_loss"], round_index)
                writer.add_scalar(
                    "train/mean_local_sequence_aux_loss",
                    round_metrics["mean_local_sequence_aux_loss"],
                    round_index,
                )
                writer.add_scalar("train/mean_local_edge_loss", round_metrics["mean_local_edge_loss"], round_index)
                writer.add_scalar("train/mean_local_pseudo_nodes", round_metrics["mean_local_pseudo_nodes"], round_index)
                writer.add_scalar("train/mean_local_novel_nodes", round_metrics["mean_local_novel_nodes"], round_index)
                writer.add_scalar("train/mean_local_open_set_nodes", round_metrics["mean_local_open_set_nodes"], round_index)
                writer.add_scalar("train/mean_local_pseudo_threshold", round_metrics["mean_local_pseudo_threshold"], round_index)
                writer.add_scalar("valid/acc", round_metrics["valid_acc"], round_index)
                writer.add_scalar("valid/f1_binary", round_metrics["valid_f1_binary"], round_index)
                writer.add_scalar("valid/auc", round_metrics["valid_auc"], round_index)
                writer.add_scalar("valid/pr_auc", round_metrics["valid_pr_auc"], round_index)
                writer.add_scalar("valid/f1_macro", round_metrics["valid_f1_macro"], round_index)
                writer.add_scalar("valid/gmean", round_metrics["valid_gmean"], round_index)
                writer.add_scalar("valid/recall", round_metrics["valid_recall"], round_index)
                writer.add_scalar("valid/recall_at_precision", round_metrics["valid_recall_at_precision"], round_index)
                writer.add_scalar("valid/threshold", round_metrics["valid_threshold"], round_index)
                writer.add_scalar("valid/positive_rate", round_metrics["valid_positive_rate"], round_index)
                writer.add_scalar("valid/prob_std", round_metrics["valid_prob_std"], round_index)
                writer.add_scalar("plan/local_epochs", round_metrics["local_epochs"], round_index)
                writer.add_scalar("plan/grad_clip", round_metrics["grad_clip"], round_index)
                writer.add_scalar("plan/selected_clients", len(round_metrics["selected_clients"]), round_index)
                writer.add_scalar("plan/selected_ratio", round_metrics["selected_ratio"], round_index)
                writer.add_scalar("plan/actual_selected_ratio", round_metrics["actual_selected_ratio"], round_index)
                writer.add_scalar("plan/selection_phase", round_metrics["selection_phase"], round_index)
                writer.add_scalar("plan/edge_loss_weight", round_metrics["edge_loss_weight"], round_index)
                writer.add_scalar("plan/learning_rate", round_metrics["learning_rate"], round_index)
                writer.add_scalar("reward/controller", round_metrics["controller_reward"], round_index)
                writer.add_scalar("active_learning/queried", round_metrics["active_learning_queried"], round_index)
                writer.add_scalar("active_learning/revealed", round_metrics["active_learning_revealed"], round_index)
                writer.add_scalar("active_learning/pending", round_metrics["active_learning_pending"], round_index)
                writer.add_scalar("active_learning/supervised_train_nodes", round_metrics["supervised_train_nodes"], round_index)
                writer.add_scalar("active_learning/unlabeled_train_nodes", round_metrics["unlabeled_train_nodes"], round_index)
                for client in bundle.clients:
                    writer.add_scalar(
                        f"plan/client_{client.client_id}_selected",
                        1.0 if client.client_id in round_metrics["selected_clients"] else 0.0,
                        round_index,
                    )
                if not bool(getattr(args, "skip_test_evaluation", False)) and args.test_every > 0 and ((round_index + 1) % args.test_every == 0):
                    interim_test = _evaluate_model(
                        global_model,
                        bundle.graph,
                        "test_mask",
                        args.device,
                        threshold=float(round_metrics["valid_threshold"]),
                    )
                    writer.add_scalar("test/acc", interim_test["acc"], round_index)
                    writer.add_scalar("test/f1_binary", interim_test["f1_binary"], round_index)
                    writer.add_scalar("test/auc", interim_test["auc"], round_index)
                    writer.add_scalar("test/pr_auc", interim_test["pr_auc"], round_index)
                    writer.add_scalar("test/f1_macro", interim_test["f1_macro"], round_index)
                    writer.add_scalar("test/gmean", interim_test["gmean"], round_index)
                    writer.add_scalar("test/recall", interim_test["recall"], round_index)
                    writer.add_scalar("test/recall_at_precision", interim_test["recall_at_precision"], round_index)
                    writer.add_scalar("test/threshold", interim_test["threshold"], round_index)
                    writer.add_scalar("test/positive_rate", interim_test["positive_rate"], round_index)
                    has_test_timeseries = True
                writer.flush()

            if valid_metrics["auc"] > peak_valid_auc_any_round:
                peak_valid_auc_any_round = float(valid_metrics["auc"])
                peak_state = snapshot_model_state_to_cpu(global_model)
                peak_valid_round_any_round = round_index
                peak_valid_threshold_any_round = float(valid_metrics["threshold"])
                patience = 0
            else:
                patience += 1

            if checkpoint_guard["eligible"] and _should_update_best_checkpoint(
                valid_metrics,
                best_round=best_round,
                best_valid_auc=best_valid_auc,
                best_valid_gmean=best_valid_gmean,
                best_valid_pr_auc=best_valid_pr_auc,
                best_valid_recall_at_precision=best_valid_recall_at_precision,
                best_valid_f1_macro=best_valid_f1_macro,
                splitgnn_policy_active=False,
            ):
                best_valid_auc = float(valid_metrics["auc"])
                best_valid_gmean = float(valid_metrics["gmean"])
                best_valid_pr_auc = float(valid_metrics.get("pr_auc", best_valid_pr_auc))
                best_valid_recall_at_precision = float(
                    valid_metrics.get("recall_at_precision", best_valid_recall_at_precision)
                )
                best_valid_f1_macro = float(valid_metrics.get("f1_macro", best_valid_f1_macro))
                best_state = snapshot_model_state_to_cpu(global_model)
                best_round = round_index
                best_valid_threshold = float(valid_metrics["threshold"])
                print(
                    f"[{dataset_name}] new_best_checkpoint round={round_index + 1} "
                    f"valid_auc={best_valid_auc:.6f} "
                    f"valid_pr_auc={best_valid_pr_auc:.6f} "
                    f"valid_f1_macro={best_valid_f1_macro:.6f}",
                    flush=True,
                )

            _atomic_write_json(
                run_status_path,
                {
                    "dataset": dataset_name,
                    "run_id": run_id,
                    "status": "running",
                    "started_at": started_at,
                    "latest_round": round_index + 1,
                    "rounds_target": int(federated_rounds),
                    "summary_path": str(summary_path),
                    "model_path": str(model_path),
                    "tb_logdir": tb_logdir,
                    "planner_mode": normalized_planner_mode,
                    "resolved_device": str(args.device),
                    "seed": None if getattr(args, "seed", None) is None else int(args.seed),
                    "latest_valid_auc": float(valid_metrics["auc"]),
                    "latest_valid_pr_auc": float(valid_metrics["pr_auc"]),
                    "latest_valid_f1_macro": float(valid_metrics["f1_macro"]),
                    "latest_valid_gmean": float(valid_metrics["gmean"]),
                    "latest_valid_threshold": float(valid_metrics["threshold"]),
                    "best_valid_auc": float(best_valid_auc if best_round >= 0 else valid_metrics["auc"]),
                    "best_valid_pr_auc": float(best_valid_pr_auc if best_round >= 0 else valid_metrics["pr_auc"]),
                    "best_valid_f1_macro": float(best_valid_f1_macro if best_round >= 0 else valid_metrics["f1_macro"]),
                    "best_round": int(best_round if best_round >= 0 else round_index),
                    "best_valid_threshold": float(best_valid_threshold if best_round >= 0 else valid_metrics["threshold"]),
                    "peak_valid_auc_any_round": float(peak_valid_auc_any_round),
                    "peak_valid_round_any_round": int(peak_valid_round_any_round),
                    "peak_valid_threshold_any_round": float(peak_valid_threshold_any_round),
                    "test_evaluated": False,
                    "data_summary": data_summary,
                },
            )

            if args.early_stop > 0 and patience >= args.early_stop:
                break
        checkpoint_selection_fallback_used = bool(best_round < 0 and peak_valid_round_any_round >= 0)
        if checkpoint_selection_fallback_used:
            best_state = peak_state
            best_valid_auc = peak_valid_auc_any_round
            best_round = peak_valid_round_any_round
            best_valid_threshold = peak_valid_threshold_any_round

        global_model.load_state_dict(best_state)
        best_valid_metrics = _evaluate_model(global_model, bundle.graph, "valid_mask", args.device)
        best_valid_auc = float(best_valid_metrics["auc"])
        best_valid_gmean = float(best_valid_metrics["gmean"])
        best_valid_pr_auc = float(best_valid_metrics["pr_auc"])
        best_valid_recall_at_precision = float(best_valid_metrics["recall_at_precision"])
        best_valid_f1_macro = float(best_valid_metrics["f1_macro"])
        best_valid_threshold = float(best_valid_metrics["threshold"])
        if bool(getattr(args, "skip_test_evaluation", False)):
            test_metrics = None
        else:
            test_metrics = _evaluate_model(
                global_model,
                bundle.graph,
                "test_mask",
                args.device,
                result_path=str(result_prefix),
                threshold=best_valid_threshold,
            )
        has_test_metrics = isinstance(test_metrics, dict) and "auc" in test_metrics
        final_supervised_train_nodes = int(bundle.graph.ndata["train_supervised_mask"].sum().item())
        final_unlabeled_train_nodes = int(bundle.graph.ndata["train_unlabeled_mask"].sum().item())

        if writer is not None and not has_test_timeseries and has_test_metrics:
            test_step = len(history)
            writer.add_scalar("test/acc", test_metrics["acc"], test_step)
            writer.add_scalar("test/auc", test_metrics["auc"], test_step)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], test_step)
            writer.add_scalar("test/recall", test_metrics["recall"], test_step)
            writer.flush()

        checkpoint_payload = {
            "model_state": global_model.state_dict(),
            "legacy_fusion_only": bool(getattr(global_model, "legacy_fusion_only", False)),
            "best_valid_auc": best_valid_auc,
            "best_valid_gmean": best_valid_gmean,
            "best_valid_pr_auc": best_valid_pr_auc,
            "best_valid_recall_at_precision": best_valid_recall_at_precision,
            "best_valid_f1_macro": best_valid_f1_macro,
            "best_round": best_round,
            "best_valid_threshold": best_valid_threshold,
            "peak_valid_auc_any_round": peak_valid_auc_any_round,
            "peak_valid_round_any_round": peak_valid_round_any_round,
            "peak_valid_threshold_any_round": peak_valid_threshold_any_round,
            "checkpoint_selection_fallback_used": checkpoint_selection_fallback_used,
            "resume_metrics_rebased": bool(resume_metrics_rebased),
            "args": vars(args),
            "planner_mode": normalized_planner_mode,
            "ablation_mode": str(getattr(args, "ablation_mode", "full")),
            "relation_order": bundle.relation_order,
            "run_id": run_id,
            "tb_logdir": tb_logdir,
            "completed": True,
            "test_evaluated": bool(has_test_metrics),
            "data_summary": data_summary,
        }
        if warm_start_payload is not None:
            checkpoint_payload["resume_best_metric_inheritance"] = str(resume_best_metric_inheritance)
            checkpoint_payload["resume_best_metric_reason"] = str(resume_best_metric_reason)
            checkpoint_payload["resume_reference_best_valid_metrics"] = resume_reference_best_metrics
            checkpoint_payload["resume_recheck_valid_metrics"] = resume_recheck_valid_metrics
        summary = {
            "dataset": dataset_name,
            "run_id": run_id,
            "completed": True,
            "tb_logdir": tb_logdir,
            "planner_mode": normalized_planner_mode,
            "requested_planner_mode": str(getattr(args, "requested_planner_mode", normalized_planner_mode)),
            "ablation_mode": str(getattr(args, "ablation_mode", "full")),
            "gnn_enabled": bool(getattr(args, "gnn_enabled", True)),
            "transformer_enabled": bool(getattr(args, "transformer_enabled", True)),
            "federated_enabled": bool(getattr(args, "federated_enabled", True)),
            "requested_disable_federated": bool(getattr(args, "requested_disable_federated", False)),
            "drl_enabled": bool(getattr(args, "drl_enabled", False)),
            "splitgnn_runtime_policy": bool(getattr(args, "splitgnn_runtime_policy", False)),
            "runtime_policy_notes": list(getattr(args, "runtime_policy_notes", [])),
            "requested_num_clients": int(num_clients),
            "effective_num_clients": int(len(bundle.clients)),
            "resolved_device": str(args.device),
            "seed": None if getattr(args, "seed", None) is None else int(args.seed),
            "requested_classification_loss": str(
                getattr(args, "requested_classification_loss", getattr(args, "classification_loss", "cb_focal"))
            ),
            "classification_loss": str(getattr(args, "classification_loss", "cb_focal")),
            "best_round": best_round,
            "best_valid_auc": best_valid_auc,
            "best_valid_gmean": best_valid_gmean,
            "best_valid_pr_auc": best_valid_pr_auc,
            "best_valid_recall_at_precision": best_valid_recall_at_precision,
            "best_valid_f1_macro": best_valid_f1_macro,
            "best_valid_threshold": best_valid_threshold,
            "best_valid_metrics": best_valid_metrics,
            "peak_valid_auc_any_round": peak_valid_auc_any_round,
            "peak_valid_round_any_round": peak_valid_round_any_round,
            "peak_valid_threshold_any_round": peak_valid_threshold_any_round,
            "checkpoint_selection_fallback_used": checkpoint_selection_fallback_used,
            "resume_metrics_rebased": bool(resume_metrics_rebased),
            "test": test_metrics if has_test_metrics else {},
            "test_evaluated": bool(has_test_metrics),
            "rounds_ran": len(history),
            "label_fraction": float(args.label_fraction),
            "label_scarcity_profile": str(getattr(args, "label_scarcity_profile", "fully_supervised")),
            "teacher_ema_decay": float(getattr(args, "teacher_ema_decay", 0.0)),
            "teacher_temperature": float(getattr(args, "teacher_temperature", 1.0)),
            "pseudo_warmup_rounds": int(getattr(args, "pseudo_warmup_rounds", 0)),
            "pseudo_ramp_rounds": int(getattr(args, "pseudo_ramp_rounds", 0)),
            "pseudo_label_threshold": float(getattr(args, "pseudo_label_threshold", 0.0)),
            "pseudo_label_min_threshold": float(getattr(args, "pseudo_label_min_threshold", 0.0)),
            "pseudo_label_top_fraction": float(getattr(args, "pseudo_label_top_fraction", 0.0)),
            "pseudo_label_weight": float(getattr(args, "pseudo_label_weight", 0.0)),
            "pseudo_label_novelty_threshold": float(getattr(args, "pseudo_label_novelty_threshold", 0.0)),
            "requested_pseudo_label_threshold": float(getattr(args, "requested_pseudo_label_threshold", 0.0)),
            "requested_pseudo_label_weight": float(getattr(args, "requested_pseudo_label_weight", 0.0)),
            "requested_pseudo_label_novelty_threshold": float(
                getattr(args, "requested_pseudo_label_novelty_threshold", 0.0)
            ),
            "requested_consistency_weight": float(getattr(args, "requested_consistency_weight", 0.0)),
            "consistency_weight": float(getattr(args, "consistency_weight", 0.0)),
            "fixed_precision_target": float(getattr(args, "fixed_precision_target", 0.5)),
            "graph_aux_loss_weight": float(getattr(args, "graph_aux_loss_weight", 0.0)),
            "sequence_aux_loss_weight": float(getattr(args, "sequence_aux_loss_weight", 0.0)),
            "graph_gate_logit_bias": float(getattr(args, "graph_gate_logit_bias", 0.0)),
            "eval_graph_gate_logit_bias": float(getattr(args, "eval_graph_gate_logit_bias", 0.0)),
            "graph_residual_min_gate": float(getattr(args, "graph_residual_min_gate", 0.0)),
            "sequence_residual_scale": float(getattr(args, "sequence_residual_scale", 1.0)),
            "open_set_novelty_threshold": float(getattr(args, "open_set_novelty_threshold", 0.0)),
            "open_set_loss_weight": float(getattr(args, "open_set_loss_weight", 0.0)),
            "requested_active_learning_budget_per_round": int(
                getattr(args, "requested_active_learning_budget_per_round", getattr(args, "active_learning_budget_per_round", 0))
            ),
            "requested_active_learning_delay_rounds": int(
                getattr(args, "requested_active_learning_delay_rounds", getattr(args, "active_learning_delay_rounds", 0))
            ),
            "active_learning_budget_per_round": int(getattr(args, "active_learning_budget_per_round", 0)),
            "active_learning_delay_rounds": int(getattr(args, "active_learning_delay_rounds", 0)),
            "requested_active_learning_novelty_weight": float(
                getattr(args, "requested_active_learning_novelty_weight", getattr(args, "active_learning_novelty_weight", 0.0))
            ),
            "requested_active_learning_diversity_weight": float(
                getattr(args, "requested_active_learning_diversity_weight", getattr(args, "active_learning_diversity_weight", 0.0))
            ),
            "active_learning_novelty_weight": float(getattr(args, "active_learning_novelty_weight", 0.0)),
            "active_learning_diversity_weight": float(getattr(args, "active_learning_diversity_weight", 0.0)),
            "mean_controller_reward": float(np.mean([item.get("controller_reward", 0.0) for item in history])) if history else 0.0,
            "best_controller_reward": float(max([item.get("controller_reward", -1e9) for item in history])) if history else 0.0,
            "initial_supervised_train_nodes": int(initial_supervised_train_nodes),
            "initial_unlabeled_train_nodes": int(initial_unlabeled_train_nodes),
            "final_supervised_train_nodes": int(final_supervised_train_nodes),
            "final_unlabeled_train_nodes": int(final_unlabeled_train_nodes),
            "mean_local_pseudo_nodes": float(np.mean([item.get("mean_local_pseudo_nodes", 0.0) for item in history])) if history else 0.0,
            "mean_local_pseudo_threshold": float(np.mean([item.get("mean_local_pseudo_threshold", 0.0) for item in history])) if history else 0.0,
            "mean_local_open_set_nodes": float(np.mean([item.get("mean_local_open_set_nodes", 0.0) for item in history])) if history else 0.0,
            "mean_active_learning_queried": float(np.mean([item.get("active_learning_queried", 0.0) for item in history])) if history else 0.0,
            "mean_active_learning_revealed": float(np.mean([item.get("active_learning_revealed", 0.0) for item in history])) if history else 0.0,
            "result_root": str(resolved_result_root),
            "result_path": str(result_prefix),
            "model_path": str(model_path),
            "data_summary": data_summary,
        }
        if warm_start_payload is not None:
            summary["resume_best_metric_inheritance"] = str(resume_best_metric_inheritance)
            summary["resume_best_metric_reason"] = str(resume_best_metric_reason)
            summary["resume_reference_best_valid_metrics"] = resume_reference_best_metrics
            summary["resume_recheck_valid_metrics"] = resume_recheck_valid_metrics
        _atomic_torch_save(model_path, checkpoint_payload)
        _atomic_write_json(summary_path, {"history": history, "summary": summary})
        if run_metadata_path is not None:
            _atomic_write_json(
                run_metadata_path,
                _run_metadata_payload(
                    dataset_name=dataset_name,
                    run_id=run_id,
                    status="completed",
                    federated_rounds=federated_rounds,
                    local_epochs=local_epochs,
                    planner_mode=args.planner_mode,
                    test_every=args.test_every,
                    resume_path=resume_path,
                    tb_logdir=tb_logdir,
                    seed=None if getattr(args, "seed", None) is None else int(args.seed),
                    summary_path=str(summary_path),
                    model_path=str(model_path),
                    rounds_ran=len(history),
                    best_round=best_round,
                    best_valid_auc=best_valid_auc,
                    test_auc=float(test_metrics["auc"]) if has_test_metrics else None,
                    finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        _atomic_write_json(
            run_status_path,
            {
                "dataset": dataset_name,
                "run_id": run_id,
                "status": "completed",
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rounds_ran": len(history),
                "summary_path": str(summary_path),
                "model_path": str(model_path),
                "tb_logdir": tb_logdir,
                "planner_mode": normalized_planner_mode,
                "resolved_device": str(args.device),
                "seed": None if getattr(args, "seed", None) is None else int(args.seed),
                "test_evaluated": bool(has_test_metrics),
                "test_auc": float(test_metrics["auc"]) if has_test_metrics else None,
                "data_summary": data_summary,
            },
        )
        return summary
    except BaseException as error:
        interrupted_fallback_used = bool(best_round < 0 and peak_valid_round_any_round >= 0)
        interrupted_best_state = peak_state if interrupted_fallback_used else best_state
        interrupted_best_valid_auc = peak_valid_auc_any_round if interrupted_fallback_used else best_valid_auc
        interrupted_best_valid_gmean = best_valid_gmean
        interrupted_best_valid_pr_auc = best_valid_pr_auc
        interrupted_best_valid_recall_at_precision = best_valid_recall_at_precision
        interrupted_best_valid_f1_macro = best_valid_f1_macro
        interrupted_best_round = peak_valid_round_any_round if interrupted_fallback_used else best_round
        interrupted_best_valid_threshold = peak_valid_threshold_any_round if interrupted_fallback_used else best_valid_threshold
        if history:
            _atomic_torch_save(
                model_path,
                {
                    "model_state": interrupted_best_state,
                    "legacy_fusion_only": bool(getattr(global_model, "legacy_fusion_only", False)),
                    "best_valid_auc": interrupted_best_valid_auc,
                    "best_valid_gmean": interrupted_best_valid_gmean,
                    "best_valid_pr_auc": interrupted_best_valid_pr_auc,
                    "best_valid_recall_at_precision": interrupted_best_valid_recall_at_precision,
                    "best_valid_f1_macro": interrupted_best_valid_f1_macro,
                    "best_round": interrupted_best_round,
                    "best_valid_threshold": interrupted_best_valid_threshold,
                    "peak_valid_auc_any_round": peak_valid_auc_any_round,
                    "peak_valid_round_any_round": peak_valid_round_any_round,
                    "peak_valid_threshold_any_round": peak_valid_threshold_any_round,
                    "checkpoint_selection_fallback_used": interrupted_fallback_used,
                    "resume_metrics_rebased": bool(resume_metrics_rebased),
                    "args": vars(args),
                    "planner_mode": normalized_planner_mode,
                    "ablation_mode": str(getattr(args, "ablation_mode", "full")),
                    "relation_order": bundle.relation_order,
                    "run_id": run_id,
                    "tb_logdir": tb_logdir,
                    "completed": False,
                    "interrupted": True,
                    "test_evaluated": False,
                    "data_summary": data_summary,
                },
            )
        if warm_start_payload is not None and history:
            interrupted_checkpoint_payload = torch.load(model_path, map_location="cpu")
            interrupted_checkpoint_payload["resume_best_metric_inheritance"] = str(resume_best_metric_inheritance)
            interrupted_checkpoint_payload["resume_best_metric_reason"] = str(resume_best_metric_reason)
            interrupted_checkpoint_payload["resume_reference_best_valid_metrics"] = resume_reference_best_metrics
            interrupted_checkpoint_payload["resume_recheck_valid_metrics"] = resume_recheck_valid_metrics
            _atomic_torch_save(model_path, interrupted_checkpoint_payload)
        interrupted_summary = {
            "dataset": dataset_name,
            "run_id": run_id,
            "completed": False,
            "tb_logdir": tb_logdir,
            "planner_mode": normalized_planner_mode,
            "requested_planner_mode": str(getattr(args, "requested_planner_mode", normalized_planner_mode)),
            "ablation_mode": str(getattr(args, "ablation_mode", "full")),
            "gnn_enabled": bool(getattr(args, "gnn_enabled", True)),
            "transformer_enabled": bool(getattr(args, "transformer_enabled", True)),
            "federated_enabled": bool(getattr(args, "federated_enabled", True)),
            "requested_disable_federated": bool(getattr(args, "requested_disable_federated", False)),
            "drl_enabled": bool(getattr(args, "drl_enabled", False)),
            "resolved_device": str(args.device),
            "seed": None if getattr(args, "seed", None) is None else int(args.seed),
            "requested_classification_loss": str(
                getattr(args, "requested_classification_loss", getattr(args, "classification_loss", "cb_focal"))
            ),
            "classification_loss": str(getattr(args, "classification_loss", "cb_focal")),
            "graph_aux_loss_weight": float(getattr(args, "graph_aux_loss_weight", 0.0)),
            "sequence_aux_loss_weight": float(getattr(args, "sequence_aux_loss_weight", 0.0)),
            "graph_gate_logit_bias": float(getattr(args, "graph_gate_logit_bias", 0.0)),
            "eval_graph_gate_logit_bias": float(getattr(args, "eval_graph_gate_logit_bias", 0.0)),
            "graph_residual_min_gate": float(getattr(args, "graph_residual_min_gate", 0.0)),
            "sequence_residual_scale": float(getattr(args, "sequence_residual_scale", 1.0)),
            "fixed_precision_target": float(getattr(args, "fixed_precision_target", 0.5)),
            "test_evaluated": False,
            "best_round": interrupted_best_round,
            "best_valid_auc": interrupted_best_valid_auc,
            "best_valid_gmean": interrupted_best_valid_gmean,
            "best_valid_pr_auc": interrupted_best_valid_pr_auc,
            "best_valid_recall_at_precision": interrupted_best_valid_recall_at_precision,
            "best_valid_f1_macro": interrupted_best_valid_f1_macro,
            "best_valid_threshold": interrupted_best_valid_threshold,
            "peak_valid_auc_any_round": peak_valid_auc_any_round,
            "peak_valid_round_any_round": peak_valid_round_any_round,
            "peak_valid_threshold_any_round": peak_valid_threshold_any_round,
            "checkpoint_selection_fallback_used": interrupted_fallback_used,
            "resume_metrics_rebased": bool(resume_metrics_rebased),
            "rounds_ran": len(history),
            "result_root": str(resolved_result_root),
            "result_path": str(result_prefix),
            "model_path": str(model_path),
            "error": f"{type(error).__name__}: {error}",
            "data_summary": data_summary,
        }
        if warm_start_payload is not None:
            interrupted_summary["resume_best_metric_inheritance"] = str(resume_best_metric_inheritance)
            interrupted_summary["resume_best_metric_reason"] = str(resume_best_metric_reason)
            interrupted_summary["resume_reference_best_valid_metrics"] = resume_reference_best_metrics
            interrupted_summary["resume_recheck_valid_metrics"] = resume_recheck_valid_metrics
        _atomic_write_json(summary_path, {"history": history, "summary": interrupted_summary})
        if run_metadata_path is not None:
            _atomic_write_json(
                run_metadata_path,
                _run_metadata_payload(
                    dataset_name=dataset_name,
                    run_id=run_id,
                    status="interrupted",
                    federated_rounds=federated_rounds,
                    local_epochs=local_epochs,
                    planner_mode=args.planner_mode,
                    test_every=args.test_every,
                    resume_path=resume_path,
                    tb_logdir=tb_logdir,
                    seed=None if getattr(args, "seed", None) is None else int(args.seed),
                    summary_path=str(summary_path),
                    model_path=str(model_path),
                    rounds_ran=len(history),
                    best_round=interrupted_best_round,
                    best_valid_auc=interrupted_best_valid_auc if interrupted_best_valid_auc >= 0 else None,
                    finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        _atomic_write_json(
            run_status_path,
            {
                "dataset": dataset_name,
                "run_id": run_id,
                "status": "interrupted",
                "interrupted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rounds_ran": len(history),
                "summary_path": str(summary_path),
                "model_path": str(model_path),
                "tb_logdir": tb_logdir,
                "planner_mode": normalized_planner_mode,
                "resolved_device": str(args.device),
                "seed": None if getattr(args, "seed", None) is None else int(args.seed),
                "error": f"{type(error).__name__}: {error}",
                "data_summary": data_summary,
            },
        )
        raise
    finally:
        if writer is not None:
            try:
                writer.flush()
                writer.close()
            except Exception:
                pass
        if owns_rl_models:
            _close_rl_models(rl_models)


def train_archive_hybrid_pipeline(
    *,
    dataset: str = "archive",
    federated_rounds: int = 12,
    base_local_epochs: int = 2,
    extra_local_epochs: int = 1,
    edge_loss_weight: float = 1.0,
    num_clients: int = 3,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    rl_timesteps: int = 0,
    device: str = DEFAULT_DEVICE_REQUEST,
    enable_tensorboard: bool = True,
    classification_loss: str = "cb_focal",
    focal_gamma: float = 2.0,
    class_balance_beta: float = 0.999,
    pseudo_label_threshold: float = 0.9,
    pseudo_label_weight: float = 0.15,
    pseudo_label_novelty_threshold: float = 2.5,
    consistency_weight: float = 0.1,
    active_learning_budget_per_round: int = -1,
    active_learning_delay_rounds: int = -1,
    active_learning_novelty_weight: float = 0.35,
    active_learning_diversity_weight: float = 0.25,
    active_learning_candidate_pool_scale: int = 4,
    fedprox_mu: float = 0.01,
    dp_noise_std: float = 0.0,
    planner_mode: str = "deterministic",
    early_stop: int = 0,
    test_every: int = 0,
    fixed_precision_target: float = 0.5,
    resume_path: str = "",
    transformer_hidden_dim: int = 64,
    transformer_num_layers: int = 1,
    fusion_hidden_dim: int = 64,
    active_learning_feedback_path: str = "",
    seed: int | None = None,
    result_root: str = "",
    disable_gnn: bool = False,
    disable_transformer: bool = False,
    disable_federated: bool = False,
    learning_rate_override: float | None = None,
    weight_decay_override: float | None = None,
    dropout_override: float | None = None,
    graph_aux_loss_weight_override: float | None = None,
    sequence_aux_loss_weight_override: float | None = None,
    graph_gate_logit_bias_override: float | None = None,
    eval_graph_gate_logit_bias_override: float | None = None,
    graph_residual_min_gate_override: float | None = None,
    sequence_residual_scale_override: float | None = None,
    skip_test_evaluation: bool = False,
    data_root: str | Path = ARCHIVE_DEFAULT_ROOT,
    max_users: int | None = 4000,
    max_transactions: int | None = 50000,
    risk_positive_ratio: float = 0.15,
    force_preview: bool = False,
) -> dict:
    # FL/RL has been archived to `legacy_federated_rl_backup/`; keep this
    # entry point stable but force the active archive pipeline onto the
    # GNN + Transformer mainline.
    planner_mode, disable_federated = _force_mainline_gnn_transformer_mode(
        planner_mode=planner_mode,
        disable_federated=disable_federated,
    )
    normalized_dataset = str(dataset).strip().lower()
    if normalized_dataset != "archive":
        raise ValueError(
            f"archive pipeline only supports dataset='archive', but got {dataset!r}."
        )
    base_seed = 42 if seed is None else int(seed)
    setup_seed(base_seed)
    shared_rl_models = train_controller_models(rl_timesteps=rl_timesteps, seed=base_seed) if str(planner_mode).lower() == "rl" else None
    try:
        return _train_archive_dataset(
            dataset_name=normalized_dataset,
            federated_rounds=federated_rounds,
            local_epochs=max(int(base_local_epochs), 1) + max(int(extra_local_epochs) - 1, 0),
            requested_base_local_epochs=base_local_epochs,
            requested_extra_local_epochs=extra_local_epochs,
            num_clients=num_clients,
            client_hops=client_hops,
            label_fraction=label_fraction,
            rl_timesteps=rl_timesteps,
            device=device,
            edge_loss_weight=edge_loss_weight,
            classification_loss=classification_loss,
            focal_gamma=focal_gamma,
            class_balance_beta=class_balance_beta,
            pseudo_label_threshold=pseudo_label_threshold,
            pseudo_label_weight=pseudo_label_weight,
            pseudo_label_novelty_threshold=pseudo_label_novelty_threshold,
            consistency_weight=consistency_weight,
            active_learning_budget_per_round=active_learning_budget_per_round,
            active_learning_delay_rounds=active_learning_delay_rounds,
            active_learning_novelty_weight=active_learning_novelty_weight,
            active_learning_diversity_weight=active_learning_diversity_weight,
            active_learning_candidate_pool_scale=active_learning_candidate_pool_scale,
            fedprox_mu=fedprox_mu,
            dp_noise_std=dp_noise_std,
            planner_mode=planner_mode,
            early_stop=early_stop,
            test_every=test_every,
            fixed_precision_target=fixed_precision_target,
            resume_path=resume_path,
            enable_tensorboard=enable_tensorboard,
            transformer_hidden_dim=transformer_hidden_dim,
            transformer_num_layers=transformer_num_layers,
            fusion_hidden_dim=fusion_hidden_dim,
            active_learning_feedback_path=active_learning_feedback_path,
            rl_models=shared_rl_models,
            seed=seed,
            result_root=result_root,
            disable_gnn=disable_gnn,
            disable_transformer=disable_transformer,
            disable_federated=disable_federated,
            learning_rate_override=learning_rate_override,
            weight_decay_override=weight_decay_override,
            dropout_override=dropout_override,
            graph_aux_loss_weight_override=graph_aux_loss_weight_override,
            sequence_aux_loss_weight_override=sequence_aux_loss_weight_override,
            graph_gate_logit_bias_override=graph_gate_logit_bias_override,
            eval_graph_gate_logit_bias_override=eval_graph_gate_logit_bias_override,
            graph_residual_min_gate_override=graph_residual_min_gate_override,
            sequence_residual_scale_override=sequence_residual_scale_override,
            skip_test_evaluation=skip_test_evaluation,
            data_root=data_root,
            max_users=max_users,
            max_transactions=max_transactions,
            risk_positive_ratio=risk_positive_ratio,
            force_preview=force_preview,
        )
    finally:
        if shared_rl_models is not None:
            _close_rl_models(shared_rl_models)


def run_archive_hybrid_training(
    *,
    dataset: str = "archive",
    federated_rounds: int = 12,
    local_epochs: int = 2,
    extra_local_epochs: int = 1,
    edge_loss_weight: float = 1.0,
    num_clients: int = 3,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    rl_timesteps: int = 0,
    device: str = DEFAULT_DEVICE_REQUEST,
    enable_tensorboard: bool = True,
    classification_loss: str = "cb_focal",
    focal_gamma: float = 2.0,
    class_balance_beta: float = 0.999,
    pseudo_label_threshold: float = 0.9,
    pseudo_label_weight: float = 0.15,
    pseudo_label_novelty_threshold: float = 2.5,
    consistency_weight: float = 0.1,
    active_learning_budget_per_round: int = -1,
    active_learning_delay_rounds: int = -1,
    active_learning_novelty_weight: float = 0.35,
    active_learning_diversity_weight: float = 0.25,
    active_learning_candidate_pool_scale: int = 4,
    fedprox_mu: float = 0.01,
    dp_noise_std: float = 0.0,
    planner_mode: str = "deterministic",
    early_stop: int = 0,
    test_every: int = 0,
    fixed_precision_target: float = 0.5,
    resume_path: str = "",
    transformer_hidden_dim: int = 64,
    transformer_num_layers: int = 1,
    fusion_hidden_dim: int = 64,
    active_learning_feedback_path: str = "",
    seed: int | None = None,
    result_root: str = "",
    disable_gnn: bool = False,
    disable_transformer: bool = False,
    disable_federated: bool = False,
    learning_rate_override: float | None = None,
    weight_decay_override: float | None = None,
    dropout_override: float | None = None,
    graph_aux_loss_weight_override: float | None = None,
    sequence_aux_loss_weight_override: float | None = None,
    graph_gate_logit_bias_override: float | None = None,
    eval_graph_gate_logit_bias_override: float | None = None,
    graph_residual_min_gate_override: float | None = None,
    sequence_residual_scale_override: float | None = None,
    skip_test_evaluation: bool = False,
    data_root: str | Path = ARCHIVE_DEFAULT_ROOT,
    max_users: int | None = 4000,
    max_transactions: int | None = 50000,
    risk_positive_ratio: float = 0.15,
    force_preview: bool = False,
) -> dict:
    return train_archive_hybrid_pipeline(
        dataset=dataset,
        federated_rounds=federated_rounds,
        base_local_epochs=local_epochs,
        extra_local_epochs=extra_local_epochs,
        edge_loss_weight=edge_loss_weight,
        num_clients=num_clients,
        client_hops=client_hops,
        label_fraction=label_fraction,
        rl_timesteps=rl_timesteps,
        device=device,
        enable_tensorboard=enable_tensorboard,
        classification_loss=classification_loss,
        focal_gamma=focal_gamma,
        class_balance_beta=class_balance_beta,
        pseudo_label_threshold=pseudo_label_threshold,
        pseudo_label_weight=pseudo_label_weight,
        pseudo_label_novelty_threshold=pseudo_label_novelty_threshold,
        consistency_weight=consistency_weight,
        active_learning_budget_per_round=active_learning_budget_per_round,
        active_learning_delay_rounds=active_learning_delay_rounds,
        active_learning_novelty_weight=active_learning_novelty_weight,
        active_learning_diversity_weight=active_learning_diversity_weight,
        active_learning_candidate_pool_scale=active_learning_candidate_pool_scale,
        fedprox_mu=fedprox_mu,
        dp_noise_std=dp_noise_std,
        planner_mode=planner_mode,
        early_stop=early_stop,
        test_every=test_every,
        fixed_precision_target=fixed_precision_target,
        resume_path=resume_path,
        transformer_hidden_dim=transformer_hidden_dim,
        transformer_num_layers=transformer_num_layers,
        fusion_hidden_dim=fusion_hidden_dim,
        active_learning_feedback_path=active_learning_feedback_path,
        seed=seed,
        result_root=result_root,
        disable_gnn=disable_gnn,
        disable_transformer=disable_transformer,
        disable_federated=disable_federated,
        learning_rate_override=learning_rate_override,
        weight_decay_override=weight_decay_override,
        dropout_override=dropout_override,
        graph_aux_loss_weight_override=graph_aux_loss_weight_override,
        sequence_aux_loss_weight_override=sequence_aux_loss_weight_override,
        graph_gate_logit_bias_override=graph_gate_logit_bias_override,
        eval_graph_gate_logit_bias_override=eval_graph_gate_logit_bias_override,
        graph_residual_min_gate_override=graph_residual_min_gate_override,
        sequence_residual_scale_override=sequence_residual_scale_override,
        skip_test_evaluation=skip_test_evaluation,
        data_root=data_root,
        max_users=max_users,
        max_transactions=max_transactions,
        risk_positive_ratio=risk_positive_ratio,
        force_preview=force_preview,
    )
