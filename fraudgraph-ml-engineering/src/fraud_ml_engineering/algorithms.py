"""Hybrid fraud training pipeline with federated learning and controller control."""

from __future__ import annotations

import copy
from contextlib import nullcontext
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

try:
    import dgl
except Exception as error:  # pragma: no cover - runtime env dependent
    active_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    env_hint = "python -m pip install -r requirements/requirements-cpu.txt"
    raise RuntimeError(
        "Failed to import the DGL runtime stack"
        + (f" (CONDA_DEFAULT_ENV={active_env})" if active_env else "")
        + f". Original error: {type(error).__name__}: {error}\n"
        + "This usually means the current environment is missing `dgl` itself or a transitive dependency such as `torchdata`.\n"
        + "Run the project inside the configured training environment, for example:\n"
        + env_hint
    ) from error
import numpy as np
import torch
import yaml
from tqdm import tqdm

_RL_IMPORT_ERROR: Exception | None = None
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as error:  # pragma: no cover - runtime env dependent
    gym = None
    spaces = None
    _RL_IMPORT_ERROR = error

try:
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.noise import NormalActionNoise
except Exception as error:  # pragma: no cover - runtime env dependent
    ReplayBuffer = None
    NormalActionNoise = None
    if _RL_IMPORT_ERROR is None:
        _RL_IMPORT_ERROR = error

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter as _TensorboardSummaryWriter
else:
    _TensorboardSummaryWriter = Any

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - runtime env dependent
    SummaryWriter = None

warnings.filterwarnings(
    "ignore",
    message=".*Torch was not compiled with flash attention.*",
    category=UserWarning,
)

from .checkpointing import (
    atomic_torch_save as checkpoint_atomic_torch_save,
    atomic_write_json as checkpoint_atomic_write_json,
    generate_run_id as checkpoint_generate_run_id,
    normalize_resume_identity_path as checkpoint_normalize_resume_identity_path,
    resume_identity_payload as checkpoint_resume_identity_payload,
    resume_reference_best_metrics as checkpoint_resume_reference_best_metrics,
    run_metadata_payload as checkpoint_run_metadata_payload,
    should_inherit_resume_best_metrics as checkpoint_should_inherit_resume_best_metrics,
    validated_resume_state_dict as checkpoint_validated_resume_state_dict,
)
from .amlsim_dataset import AMLSIM_DEFAULT_ROOT
from .dataset_registry import (
    attach_bundle_protocol,
    load_registered_dataset_bundle,
    registered_dataset_names,
)
from .cli_contract import (
    DATASET_SELECTION_ALIASES,
    DATASET_SELECTION_CHOICES,
    DEFAULT_HYBRID_MAINLINE_ROUNDS,
    LEGACY_BATCH_DATASETS,
    SUPPORTED_HYBRID_DATASETS,
)
from .device_utils import DEFAULT_DEVICE_REQUEST, resolve_dgl_training_device
from .vendor.splitgnn.utils import evaluate, setup_seed
from .defi_rug_pull_dataset import DEFI_RUG_PULL_DEFAULT_ROOT, load_defi_rug_pull_dataset
from .elliptic_dataset import ELLIPTIC_DEFAULT_ROOT
from .ethereum_phishing_dataset import ETHEREUM_PHISHING_DEFAULT_ROOT, load_ethereum_phishing_dataset
from .ethereum_ponzi_dataset import ETHEREUM_PONZI_DEFAULT_ROOT, load_ethereum_ponzi_dataset
from .evaluator import (
    collect_model_diagnostics as platform_collect_model_diagnostics,
    evaluate_model as platform_evaluate_model,
    evaluate_saved_hybrid_checkpoint as platform_evaluate_saved_hybrid_checkpoint,
)
from .fraud_dataset import DatasetBundle, load_splitgnn_dataset
from .hybrid_task_model import (
    HybridFraudModel,
    _novelty_scores,
    checkpoint_legacy_fusion_only,
    sanitize_legacy_hybrid_state_dict,
)
from .ieee_cis_dataset import load_ieee_cis_dataset
try:
    from .model import FRModel
except Exception as error:  # pragma: no cover - runtime env dependent
    FRModel = None
    if _RL_IMPORT_ERROR is None:
        _RL_IMPORT_ERROR = error
from .paper_baseline_optimization import ieee_full_gpu_cuda_ready, stabilize_ieee_full_runtime
from .paper_runner_runtime import StageTimer
from .model_state_utils import snapshot_model_state_to_cpu
from .resource_guard import estimate_runtime_resource_plan, recommend_smaller_phishing_limits
from .trainer_local import (
    apply_dp_noise_to_state_dict as platform_apply_dp_noise_to_state_dict,
    ema_update_model as platform_ema_update_model,
    local_train_round as platform_local_train_round,
)
from .paths import ARTIFACTS_ROOT, CONFIG_ROOT, GRAPH_ROOT, REPO_ROOT

PROJECT_ROOT = REPO_ROOT
SPLITGNN_ROOT = REPO_ROOT / "src" / "fraud_ml_engineering" / "vendor" / "splitgnn"
SPLITGNN_DATA_DIR = GRAPH_ROOT
SPLITGNN_CONFIG_DIR = CONFIG_ROOT / "splitgnn"
HYBRID_RESULT_ROOT = ARTIFACTS_ROOT / "training"
SPLITGNN_DATASET_NAMES = frozenset(SUPPORTED_HYBRID_DATASETS)
AGGREGATED_REPLAY_BUFFER_SIZE = 10_000
TD3_REPLAY_BUFFER_SIZE = 10_000
PROBABILITY_COLLAPSE_STD = 4e-3
PLATEAU_STAGNATION_ROUNDS = 3
_GymEnvBase = gym.Env if gym is not None else object


def _require_rl_dependencies() -> None:
    if _RL_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "RL controller dependencies are unavailable. Install gymnasium and stable-baselines3 "
        "to use the archived RL path. The active deterministic mainline does not require them."
    ) from _RL_IMPORT_ERROR


def _require_tensorboard_dependency():
    if SummaryWriter is not None:
        return SummaryWriter
    raise RuntimeError(
        "TensorBoard support is unavailable because the 'tensorboard' package is not installed. "
        "Re-run with --disable_tb or install tensorboard to enable logging."
    )


def _resolve_result_root(result_root: str | Path | None = None) -> Path:
    root = Path(result_root).expanduser() if result_root else HYBRID_RESULT_ROOT
    return root.resolve()


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


def _log_progress(dataset_name: str, message: str) -> None:
    print(f"[{dataset_name}] {message}", flush=True)


def _ieee_runtime_uses_raw_assets(args: SimpleNamespace) -> bool:
    data_profile = str(getattr(args, "ieee_data_profile", "raw") or "raw").strip().lower()
    return data_profile == "raw"


def _apply_main_ieee_full_gpu_profile(args: SimpleNamespace, dataset_name: str) -> None:
    if str(dataset_name).lower() != "ieee" or not bool(getattr(args, "profile_ieee_full_gpu", False)):
        return
    stabilize_ieee_full_runtime(enabled=True)
    if ieee_full_gpu_cuda_ready(args) and _ieee_runtime_uses_raw_assets(args):
        args.ieee_max_transactions = 0
    args.transformer_hidden_dim = max(int(getattr(args, "transformer_hidden_dim", 64)), 192)
    args.seq_hidden_dim = max(int(getattr(args, "seq_hidden_dim", 64)), 192)
    args.fusion_hidden_dim = max(int(getattr(args, "fusion_hidden_dim", 64)), 192)
    args.feature_hidden_dim = max(int(getattr(args, "feature_hidden_dim", 64)), 192)
    args.raw_anchor_dim = max(int(getattr(args, "raw_anchor_dim", 64)), 192)
    args.transformer_num_layers = max(int(getattr(args, "transformer_num_layers", 1)), 3)
    args.graph_warmup_rounds = max(int(getattr(args, "graph_warmup_rounds", 0)), 2)
    args.fusion_bootstrap_rounds = max(int(getattr(args, "fusion_bootstrap_rounds", 0)), 2)
    args.patience = max(int(getattr(args, "patience", 0)), 8)
    args.lr = min(float(getattr(args, "lr", 3e-3)), 2e-3)
    args.export_embedding_viz = False
    args.test_every = 0
    if str(getattr(args, "amp_dtype", "auto")).strip().lower() == "auto":
        args.amp_dtype = "bf16"


def _main_important_parameters(
    args: SimpleNamespace,
    *,
    dataset_name: str,
    federated_rounds: int,
    local_epochs: int,
) -> dict[str, object]:
    return {
        "profile_ieee_full_gpu": bool(getattr(args, "profile_ieee_full_gpu", False)),
        "lightweight_valid_eval": bool(getattr(args, "lightweight_valid_eval", False)),
        "amp_dtype": str(getattr(args, "amp_dtype", "auto")),
        "lr": float(getattr(args, "lr", 0.0)),
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "dropout": float(getattr(args, "dropout", 0.0)),
        "federated_rounds": int(federated_rounds),
        "local_epochs": int(local_epochs),
        "transformer_hidden_dim": int(getattr(args, "transformer_hidden_dim", getattr(args, "seq_hidden_dim", 0))),
        "transformer_num_layers": int(getattr(args, "transformer_num_layers", 0)),
        "sequence_batch_chunk_size": (
            int(getattr(args, "sequence_batch_chunk_size", 0))
            if getattr(args, "sequence_batch_chunk_size", None) is not None
            else None
        ),
        "event_batch_chunk_size": (
            int(getattr(args, "event_batch_chunk_size", 0))
            if getattr(args, "event_batch_chunk_size", None) is not None
            else None
        ),
        "transformer_activation_checkpointing": bool(getattr(args, "transformer_activation_checkpointing", False)),
        "fusion_hidden_dim": int(getattr(args, "fusion_hidden_dim", 0)),
        "graph_aux_loss_weight": float(getattr(args, "graph_aux_loss_weight", 0.0)),
        "sequence_aux_loss_weight": float(getattr(args, "sequence_aux_loss_weight", 0.0)),
        "graph_gate_logit_bias": float(getattr(args, "graph_gate_logit_bias", 0.0)),
        "eval_graph_gate_logit_bias": float(getattr(args, "eval_graph_gate_logit_bias", 0.0)),
        "graph_residual_min_gate": float(getattr(args, "graph_residual_min_gate", 0.0)),
        "sequence_residual_scale": float(getattr(args, "sequence_residual_scale", 1.0)),
        "diffusion_residual_scale": float(getattr(args, "diffusion_residual_scale", 0.0)),
        "disable_relation_sequence_encoder": bool(getattr(args, "disable_relation_sequence_encoder", False)),
        "disable_event_transformer_encoder": bool(getattr(args, "disable_event_transformer_encoder", False)),
        "disable_temporal_context_encoder": bool(getattr(args, "disable_temporal_context_encoder", False)),
        "disable_graph_temporal_fusion": bool(getattr(args, "disable_graph_temporal_fusion", False)),
        "force_disable_wavelet_lite": bool(getattr(args, "force_disable_wavelet_lite", False)),
        "force_disable_utg_lite": bool(getattr(args, "force_disable_utg_lite", False)),
        "force_disable_coassociation": bool(getattr(args, "force_disable_coassociation", False)),
        "force_disable_diffusion_residual": bool(getattr(args, "force_disable_diffusion_residual", False)),
        "label_fraction": float(getattr(args, "label_fraction", 1.0)),
        "fixed_precision_target": float(getattr(args, "fixed_precision_target", 0.5)),
        "dataset": str(dataset_name),
        "requested_ieee_max_transactions": (
            int(getattr(args, "requested_ieee_max_transactions", 0))
            if getattr(args, "requested_ieee_max_transactions", None) is not None
            else None
        ),
        "ieee_data_profile": str(getattr(args, "ieee_data_profile", "raw")),
        "ieee_loader_view": str(getattr(args, "ieee_loader_view", "hybrid")),
        "ieee_relation_profile": str(getattr(args, "ieee_relation_profile", "core")),
        "ieee_feature_profile": str(getattr(args, "ieee_feature_profile", "typed_256")),
        "ieee_history_len": int(getattr(args, "ieee_history_len", 6)),
        "ieee_sampling_profile": str(getattr(args, "ieee_sampling_profile", "fraud_hardneg")),
        "ieee_max_transactions": (
            int(getattr(args, "ieee_max_transactions", 0))
            if getattr(args, "ieee_max_transactions", None) is not None
            else None
        ),
        "ieee_time_bins": int(getattr(args, "ieee_time_bins", 24)),
        "ieee_relation_window_neighbors": int(getattr(args, "ieee_relation_window_neighbors", 2)),
        "ieee_train_ratio": float(getattr(args, "ieee_train_ratio", 0.70)),
        "ieee_valid_ratio": float(getattr(args, "ieee_valid_ratio", 0.15)),
        "ieee_full_compact_sequences": bool(getattr(args, "ieee_full_compact_sequences", True)),
        "ieee_sequence_feature_dim": int(getattr(args, "ieee_sequence_feature_dim", 64)),
        "ieee_event_feature_dim": int(getattr(args, "ieee_event_feature_dim", 64)),
        "ieee_build_cache_only": bool(getattr(args, "ieee_build_cache_only", False)),
        "ieee_build_light_cache_only": bool(getattr(args, "ieee_build_light_cache_only", False)),
        "ieee_rebuild_cache": bool(getattr(args, "ieee_rebuild_cache", False)),
        "ieee_rebuild_light_cache": bool(getattr(args, "ieee_rebuild_light_cache", False)),
        "ieee_skip_training": bool(getattr(args, "ieee_skip_training", False)),
        "amlsim_data_root": str(getattr(args, "amlsim_data_root", "")),
        "amlsim_train_ratio": float(getattr(args, "amlsim_train_ratio", 0.70)),
        "amlsim_valid_ratio": float(getattr(args, "amlsim_valid_ratio", 0.15)),
        "amlsim_relation_window_neighbors": int(getattr(args, "amlsim_relation_window_neighbors", 4)),
        "amlsim_activity_bins": int(getattr(args, "amlsim_activity_bins", 8)),
        "amlsim_event_history_len": int(getattr(args, "amlsim_event_history_len", 12)),
        "amlsim_rebuild_cache": bool(getattr(args, "amlsim_rebuild_cache", False)),
        "amlsim_allow_sample_fallback": bool(getattr(args, "amlsim_allow_sample_fallback", False)),
        "amlsim_diffusion_residual_scale": float(getattr(args, "amlsim_diffusion_residual_scale", 0.18)),
        "amlsim_pseudo_refresh_interval": int(getattr(args, "amlsim_pseudo_refresh_interval", 0)),
        "amlsim_pseudo_refresh_start_round": int(getattr(args, "amlsim_pseudo_refresh_start_round", 0)),
        "amlsim_pseudo_refresh_momentum": float(getattr(args, "amlsim_pseudo_refresh_momentum", 0.65)),
        "amlsim_pseudo_refresh_max_fraction": float(getattr(args, "amlsim_pseudo_refresh_max_fraction", 0.0)),
        "amlsim_coassociation_loss_weight": float(getattr(args, "amlsim_coassociation_loss_weight", 0.0)),
        "amlsim_wavelet_loss_weight": float(getattr(args, "amlsim_wavelet_loss_weight", 0.0)),
        "amlsim_utg_align_loss_weight": float(getattr(args, "amlsim_utg_align_loss_weight", 0.0)),
        "elliptic_data_root": str(getattr(args, "elliptic_data_root", "")),
        "elliptic_train_time_end": int(getattr(args, "elliptic_train_time_end", 34)),
        "elliptic_valid_time_end": int(getattr(args, "elliptic_valid_time_end", 39)),
        "elliptic_history_len": int(getattr(args, "elliptic_history_len", 8)),
        "elliptic_sequence_topk": int(getattr(args, "elliptic_sequence_topk", 8)),
        "elliptic_coassociation_topk": int(getattr(args, "elliptic_coassociation_topk", 3)),
        "elliptic_coassociation_time_window": int(getattr(args, "elliptic_coassociation_time_window", 2)),
        "elliptic_use_unknown_ssl": bool(getattr(args, "elliptic_use_unknown_ssl", True)),
        "elliptic_rebuild_cache": bool(getattr(args, "elliptic_rebuild_cache", False)),
        "elliptic_pseudo_refresh_interval": int(getattr(args, "elliptic_pseudo_refresh_interval", 4)),
        "elliptic_pseudo_refresh_start_round": int(getattr(args, "elliptic_pseudo_refresh_start_round", 4)),
        "elliptic_pseudo_refresh_momentum": float(getattr(args, "elliptic_pseudo_refresh_momentum", 0.65)),
        "elliptic_pseudo_refresh_max_fraction": float(getattr(args, "elliptic_pseudo_refresh_max_fraction", 0.10)),
        "elliptic_diffusion_residual_scale": float(getattr(args, "elliptic_diffusion_residual_scale", 0.18)),
        "elliptic_coassociation_loss_weight": float(getattr(args, "elliptic_coassociation_loss_weight", 0.05)),
        "elliptic_wavelet_loss_weight": float(getattr(args, "elliptic_wavelet_loss_weight", 0.03)),
        "elliptic_utg_align_loss_weight": float(getattr(args, "elliptic_utg_align_loss_weight", 0.04)),
    }


def _force_mainline_gnn_transformer_mode(*, planner_mode: str, disable_federated: bool) -> tuple[str, bool]:
    """Force the active runtime onto the archived-FL/RL-free mainline."""
    _ = str(planner_mode).lower()
    _ = bool(disable_federated)
    return "deterministic", True


def _generate_run_id() -> str:
    return checkpoint_generate_run_id()


def _normalize_resume_identity_path(value: object) -> str:
    return checkpoint_normalize_resume_identity_path(value)


def is_splitgnn_dataset_name(dataset_name: str) -> bool:
    return str(dataset_name).lower() in SPLITGNN_DATASET_NAMES


def resolve_requested_datasets(dataset: str) -> list[str]:
    normalized_dataset = str(dataset).lower()
    if normalized_dataset in DATASET_SELECTION_ALIASES:
        return list(DATASET_SELECTION_ALIASES[normalized_dataset])
    if normalized_dataset in SPLITGNN_DATASET_NAMES:
        return [normalized_dataset]
    raise ValueError(
        f"Unsupported dataset selector: {dataset!r}. "
        f"Choose from {', '.join(DATASET_SELECTION_CHOICES)}."
    )


def resolve_splitgnn_runtime_policy(
    *,
    dataset_name: str,
    planner_mode: str,
    disable_federated: bool,
) -> dict:
    normalized_dataset = str(dataset_name).lower()
    requested_planner_mode = str(planner_mode).lower()
    requested_disable_federated = bool(disable_federated)
    splitgnn_policy_active = is_splitgnn_dataset_name(normalized_dataset)
    effective_planner_mode = requested_planner_mode
    effective_disable_federated = requested_disable_federated
    notes: list[str] = []

    if splitgnn_policy_active:
        effective_planner_mode = "deterministic"
        effective_disable_federated = True
        notes.append("SplitGNN datasets are restricted to structural ablations only.")
        if requested_planner_mode != effective_planner_mode:
            notes.append("Requested planner_mode was overridden to deterministic.")
        if requested_disable_federated != effective_disable_federated:
            notes.append("Federated learning was disabled by policy.")

    return {
        "dataset": normalized_dataset,
        "splitgnn_policy_active": splitgnn_policy_active,
        "requested_planner_mode": requested_planner_mode,
        "effective_planner_mode": effective_planner_mode,
        "requested_disable_federated": requested_disable_federated,
        "effective_disable_federated": effective_disable_federated,
        "notes": notes,
    }


def _resume_identity_payload(config: dict | SimpleNamespace | None) -> dict:
    return checkpoint_resume_identity_payload(config)


def _validated_resume_state_dict(
    current_args: SimpleNamespace,
    warm_start_payload: dict,
    resume_file: Path,
    current_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    return checkpoint_validated_resume_state_dict(
        current_args=current_args,
        warm_start_payload=warm_start_payload,
        resume_file=resume_file,
        current_state_dict=current_state_dict,
    )


def _resume_reference_best_metrics(warm_start_payload: dict) -> dict[str, float]:
    return checkpoint_resume_reference_best_metrics(warm_start_payload)


def _should_inherit_resume_best_metrics(
    current_valid_metrics: Dict[str, float],
    stored_best_metrics: Dict[str, float],
) -> tuple[bool, str]:
    return checkpoint_should_inherit_resume_best_metrics(current_valid_metrics, stored_best_metrics)


def _resolve_ablation_mode(
    disable_gnn: bool,
    disable_transformer: bool,
    disable_federated: bool,
    planner_mode: str,
) -> str:
    disabled_modules = []
    if disable_gnn:
        disabled_modules.append("gnn")
    if disable_transformer:
        disabled_modules.append("transformer")
    if disable_federated:
        disabled_modules.append("fl")
    if str(planner_mode).lower() != "rl":
        disabled_modules.append("drl")
    if not disabled_modules:
        return "full"
    return "no_" + "_".join(disabled_modules)


def _load_fixed_graph_teacher_model(
    *,
    checkpoint_path: str,
    args: SimpleNamespace,
    bundle: DatasetBundle,
) -> HybridFraudModel:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Graph teacher checkpoint not found: {checkpoint_file}")
    checkpoint_payload = torch.load(checkpoint_file, map_location="cpu")
    teacher_args = copy.deepcopy(args)
    checkpoint_args = dict(checkpoint_payload.get("args", {}) or {})
    for key, value in checkpoint_args.items():
        setattr(teacher_args, key, value)
    teacher_args.dataset = args.dataset
    teacher_args.device = args.device
    teacher_args.requested_device = str(getattr(args, "requested_device", args.device))
    teacher_args.disable_transformer = bool(checkpoint_args.get("disable_transformer", True))
    teacher_args.disable_gnn = bool(checkpoint_args.get("disable_gnn", False))
    teacher_args.transformer_enabled = not bool(teacher_args.disable_transformer)
    teacher_args.gnn_enabled = not bool(teacher_args.disable_gnn)
    teacher_args.fusion_variant = str(checkpoint_args.get("fusion_variant", "single_branch"))
    teacher_model = HybridFraudModel(teacher_args, bundle.graph).to(args.device)
    teacher_model.load_state_dict(
        sanitize_legacy_hybrid_state_dict(
            checkpoint_payload["model_state"],
            current_state_dict=teacher_model.state_dict(),
        ),
        strict=True,
    )
    teacher_model.legacy_fusion_only = checkpoint_legacy_fusion_only(checkpoint_payload)
    teacher_model.edge_loss_weight = float(getattr(args, "edge_loss_weight", teacher_model.edge_loss_weight))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model


def _resolve_structure_only_ablation_mode(
    disable_gnn: bool,
    disable_transformer: bool,
) -> str:
    disabled_modules = []
    if disable_gnn:
        disabled_modules.append("gnn")
    if disable_transformer:
        disabled_modules.append("transformer")
    if not disabled_modules:
        return "full"
    return "no_" + "_".join(disabled_modules)


DATASET_PROBABILITY_RULES = {
    "default": {
        "collapse_std": PROBABILITY_COLLAPSE_STD,
        "severe_collapse_std": PROBABILITY_COLLAPSE_STD * 0.3,
        "reward_low_prob_std_floor": PROBABILITY_COLLAPSE_STD,
        "weak_auc_floor": 0.60,
        "weak_peak_auc_floor": 0.62,
        "severe_auc_floor": 0.56,
        "severe_peak_auc_floor": 0.58,
        "require_prob_std_decay": False,
        "prob_std_decay_ratio": 1.0,
        "sharp_drop_delta": -0.08,
    },
    "yelp": {
        # Yelp should tolerate sharper probabilities than the default rule, but
        # the previous 1e-5-level threshold was letting obviously collapsed
        # runs pass without controller feedback.
        "collapse_std": 5e-3,
        "severe_collapse_std": 2e-3,
        "reward_low_prob_std_floor": 5e-3,
        "weak_auc_floor": 0.60,
        "weak_peak_auc_floor": 0.62,
        "severe_auc_floor": 0.56,
        "severe_peak_auc_floor": 0.58,
        "require_prob_std_decay": True,
        "prob_std_decay_ratio": 0.50,
        "sharp_drop_delta": -0.10,
    },
    "archive": {
        "collapse_std": 6e-3,
        "severe_collapse_std": 2.5e-3,
        "reward_low_prob_std_floor": 6e-3,
        "weak_auc_floor": 0.76,
        "weak_peak_auc_floor": 0.80,
        "severe_auc_floor": 0.70,
        "severe_peak_auc_floor": 0.74,
        "require_prob_std_decay": True,
        "prob_std_decay_ratio": 0.55,
        "sharp_drop_delta": -0.06,
    },
    "ccfd": {
        "collapse_std": 5e-3,
        "severe_collapse_std": 2e-3,
        "reward_low_prob_std_floor": 5e-3,
        "weak_auc_floor": 0.88,
        "weak_peak_auc_floor": 0.91,
        "severe_auc_floor": 0.84,
        "severe_peak_auc_floor": 0.88,
        "require_prob_std_decay": True,
        "prob_std_decay_ratio": 0.60,
        "sharp_drop_delta": -0.05,
    },
    "ieee": {
        "collapse_std": 5e-3,
        "severe_collapse_std": 2e-3,
        "reward_low_prob_std_floor": 5e-3,
        "weak_auc_floor": 0.84,
        "weak_peak_auc_floor": 0.88,
        "severe_auc_floor": 0.80,
        "severe_peak_auc_floor": 0.84,
        "require_prob_std_decay": True,
        "prob_std_decay_ratio": 0.60,
        "sharp_drop_delta": -0.05,
    },
    "ethereum_phishing": {
        "collapse_std": 5e-3,
        "severe_collapse_std": 2e-3,
        "reward_low_prob_std_floor": 5e-3,
        "weak_auc_floor": 0.70,
        "weak_peak_auc_floor": 0.74,
        "severe_auc_floor": 0.64,
        "severe_peak_auc_floor": 0.68,
        "require_prob_std_decay": True,
        "prob_std_decay_ratio": 0.60,
        "sharp_drop_delta": -0.07,
    },
    "ethereum_ponzi": {
        "collapse_std": 4e-3,
        "severe_collapse_std": 1.5e-3,
        "reward_low_prob_std_floor": 4e-3,
        "weak_auc_floor": 0.60,
        "weak_peak_auc_floor": 0.64,
        "severe_auc_floor": 0.56,
        "severe_peak_auc_floor": 0.60,
        "require_prob_std_decay": False,
        "prob_std_decay_ratio": 1.0,
        "sharp_drop_delta": -0.08,
    },
    "defi_rug_pull": {
        "collapse_std": 4e-3,
        "severe_collapse_std": 1.5e-3,
        "reward_low_prob_std_floor": 4e-3,
        "weak_auc_floor": 0.62,
        "weak_peak_auc_floor": 0.66,
        "severe_auc_floor": 0.58,
        "severe_peak_auc_floor": 0.62,
        "require_prob_std_decay": False,
        "prob_std_decay_ratio": 1.0,
        "sharp_drop_delta": -0.08,
    },
}
CONTROLLER_REWARD_WEIGHTS = {
    "auc_gain": 1.2,  # Avoid over-optimizing AUC at the expense of other fraud metrics.
    "f1_gain": 0.8,  # Balance precision and recall through macro F1.
    "recall_gain": 1.0,  # Preserve recall sensitivity for rare fraud cases.
    "prob_std_gain": 0.5,  # Stabilize the predicted-probability distribution.
    "alignment": 1.0,  # Prefer controller actions that remain policy-consistent.
    "quality_bonus": 0.6,  # Reward stable, high-quality rounds.
    "communication_cost": 0.8,  # Penalize unnecessary client communication.
    "compute_cost": 0.5,  # Trade model quality against compute cost.
    "clip_cost": 0.3,  # Account for gradient-clipping pressure.
    "collapse_penalty": 0.8,  # Penalize degenerate prediction distributions.
    "plateau_penalty": 0.2,  # Retain limited room to adapt during plateaus.
}
DATASET_CONTROLLER_REWARD_PROFILES = {
    "default": {
        "quality_auc_floor": 0.62,
        "quality_f1_floor": 0.52,
        "plateau_auc_floor": 0.64,
        "plateau_f1_floor": 0.54,
    },
    "yelp": {
        "quality_auc_floor": 0.74,
        "quality_f1_floor": 0.60,
        "plateau_auc_floor": 0.76,
        "plateau_f1_floor": 0.62,
    },
    "amazon": {
        "quality_auc_floor": 0.78,
        "quality_f1_floor": 0.66,
        "plateau_auc_floor": 0.80,
        "plateau_f1_floor": 0.68,
    },
    "comp": {
        "quality_auc_floor": 0.53,
        "quality_f1_floor": 0.50,
        "plateau_auc_floor": 0.56,
        "plateau_f1_floor": 0.52,
    },
    "archive": {
        "quality_auc_floor": 0.88,
        "quality_f1_floor": 0.74,
        "plateau_auc_floor": 0.90,
        "plateau_f1_floor": 0.76,
    },
    "ccfd": {
        "quality_auc_floor": 0.94,
        "quality_f1_floor": 0.78,
        "plateau_auc_floor": 0.95,
        "plateau_f1_floor": 0.80,
    },
    "ieee": {
        "quality_auc_floor": 0.88,
        "quality_f1_floor": 0.22,
        "plateau_auc_floor": 0.90,
        "plateau_f1_floor": 0.24,
    },
    "ethereum_phishing": {
        "quality_auc_floor": 0.80,
        "quality_f1_floor": 0.72,
        "plateau_auc_floor": 0.82,
        "plateau_f1_floor": 0.74,
    },
    "ethereum_ponzi": {
        "quality_auc_floor": 0.70,
        "quality_f1_floor": 0.62,
        "plateau_auc_floor": 0.74,
        "plateau_f1_floor": 0.64,
    },
    "defi_rug_pull": {
        "quality_auc_floor": 0.72,
        "quality_f1_floor": 0.64,
        "plateau_auc_floor": 0.75,
        "plateau_f1_floor": 0.66,
    },
}
DATASET_SCHEDULER_PROFILES = {
    "amazon": {
        "min_edge_weight": 0.08,
        "startup_edge_cap": 0.26,
        "stable_edge_cap": 0.40,
        "collapse_edge_ratio": 0.95,
        "stable_edge_ratio": 1.45,
        "startup_lr_scale": 0.90,
        "collapse_lr_scale": 0.80,
        "startup_epoch_scale": 0.40,
        "mid_epoch_scale": 0.55,
        "late_epoch_scale": 0.48,
        "collapse_epoch_cap": 0.48,
        "max_local_epoch_multiplier": 4,
        "plateau_epoch_boost": 0.06,
        "deterministic_edge_loss_weight": 1.0,
        "deterministic_min_local_epochs": 4,
        "plateau_min_local_epochs": 5,
        "deterministic_lr_scale": 0.95,
        "sharp_drop_epoch_scale": 0.28,
        "weak_metric_min_selected_clients": 3,
        "recall_guard_floor": 0.66,
        "positive_rate_guard_floor": 0.18,
    },
    "yelp": {
        "min_edge_weight": 0.08,
        "startup_edge_cap": 0.22,
        "stable_edge_cap": 0.34,
        "collapse_edge_ratio": 0.85,
        "stable_edge_ratio": 1.25,
        "startup_lr_scale": 0.90,
        "collapse_lr_scale": 0.50,
        "startup_epoch_scale": 0.36,
        "mid_epoch_scale": 0.50,
        "late_epoch_scale": 0.42,
        "collapse_epoch_cap": 0.12,
        "max_local_epoch_multiplier": 4,
        "plateau_epoch_boost": 0.06,
        "deterministic_edge_loss_weight": 1.0,
        "deterministic_min_local_epochs": 3,
        "plateau_min_local_epochs": 4,
        "deterministic_lr_scale": 0.65,
        "sharp_drop_epoch_scale": 0.18,
        "weak_metric_min_selected_clients": 3,
        "recall_guard_floor": 0.70,
        "positive_rate_guard_floor": 0.30,
    },
    "comp": {
        "min_edge_weight": 0.03,
        "startup_edge_cap": 0.08,
        "stable_edge_cap": 0.12,
        "collapse_edge_ratio": 0.30,
        "stable_edge_ratio": 0.50,
        "startup_lr_scale": 0.70,
        "collapse_lr_scale": 0.35,
        "startup_epoch_scale": 0.18,
        "mid_epoch_scale": 0.03,
        "late_epoch_scale": 0.03,
        "collapse_epoch_cap": 0.08,
        "max_local_epoch_multiplier": 4,
        "plateau_epoch_boost": 0.03,
        "deterministic_edge_loss_weight": 0.08,
        "deterministic_min_local_epochs": 3,
        "plateau_min_local_epochs": 4,
        "deterministic_lr_scale": 0.80,
        "sharp_drop_epoch_scale": 0.12,
        "weak_metric_min_selected_clients": 3,
        "recall_guard_floor": 0.55,
        "positive_rate_guard_floor": 0.40,
        "weak_auc_guard_floor": 0.58,
    },
    "archive": {
        "min_edge_weight": 0.10,
        "startup_edge_cap": 0.24,
        "stable_edge_cap": 0.34,
        "collapse_edge_ratio": 0.75,
        "stable_edge_ratio": 1.18,
        "startup_lr_scale": 0.85,
        "collapse_lr_scale": 0.55,
        "startup_epoch_scale": 0.32,
        "mid_epoch_scale": 0.58,
        "late_epoch_scale": 0.52,
        "collapse_epoch_cap": 0.18,
        "max_local_epoch_multiplier": 5,
        "plateau_epoch_boost": 0.10,
        "deterministic_edge_loss_weight": 0.22,
        "deterministic_min_local_epochs": 4,
        "plateau_min_local_epochs": 5,
        "deterministic_lr_scale": 0.78,
        "sharp_drop_epoch_scale": 0.20,
        "weak_metric_min_selected_clients": 2,
        "recall_guard_floor": 0.78,
        "positive_rate_guard_floor": 0.14,
    },
    "ccfd": {
        "min_edge_weight": 0.08,
        "startup_edge_cap": 0.18,
        "stable_edge_cap": 0.28,
        "collapse_edge_ratio": 0.72,
        "stable_edge_ratio": 1.12,
        "startup_lr_scale": 0.82,
        "collapse_lr_scale": 0.58,
        "startup_epoch_scale": 0.26,
        "mid_epoch_scale": 0.48,
        "late_epoch_scale": 0.44,
        "collapse_epoch_cap": 0.18,
        "max_local_epoch_multiplier": 5,
        "plateau_epoch_boost": 0.08,
        "deterministic_edge_loss_weight": 0.18,
        "deterministic_min_local_epochs": 4,
        "plateau_min_local_epochs": 5,
        "deterministic_lr_scale": 0.84,
        "sharp_drop_epoch_scale": 0.18,
        "weak_metric_min_selected_clients": 1,
        "recall_guard_floor": 0.70,
        "positive_rate_guard_floor": 0.01,
        "weak_auc_guard_floor": 0.90,
    },
    "ieee": {
        "min_edge_weight": 0.08,
        "startup_edge_cap": 0.20,
        "stable_edge_cap": 0.28,
        "collapse_edge_ratio": 0.70,
        "stable_edge_ratio": 1.10,
        "startup_lr_scale": 0.82,
        "collapse_lr_scale": 0.60,
        "startup_epoch_scale": 0.28,
        "mid_epoch_scale": 0.52,
        "late_epoch_scale": 0.46,
        "collapse_epoch_cap": 0.18,
        "max_local_epoch_multiplier": 5,
        "plateau_epoch_boost": 0.08,
        "deterministic_edge_loss_weight": 0.18,
        "deterministic_min_local_epochs": 4,
        "plateau_min_local_epochs": 5,
        "deterministic_lr_scale": 0.85,
        "sharp_drop_epoch_scale": 0.20,
        "weak_metric_min_selected_clients": 1,
        "recall_guard_floor": 0.18,
        "positive_rate_guard_floor": 0.02,
        "weak_auc_guard_floor": 0.86,
    },
    "ethereum_phishing": {
        "min_edge_weight": 0.08,
        "startup_edge_cap": 0.20,
        "stable_edge_cap": 0.30,
        "collapse_edge_ratio": 0.72,
        "stable_edge_ratio": 1.12,
        "startup_lr_scale": 0.84,
        "collapse_lr_scale": 0.58,
        "startup_epoch_scale": 0.28,
        "mid_epoch_scale": 0.50,
        "late_epoch_scale": 0.44,
        "collapse_epoch_cap": 0.18,
        "max_local_epoch_multiplier": 5,
        "plateau_epoch_boost": 0.08,
        "deterministic_edge_loss_weight": 0.16,
        "deterministic_min_local_epochs": 4,
        "plateau_min_local_epochs": 5,
        "deterministic_lr_scale": 0.85,
        "sharp_drop_epoch_scale": 0.18,
        "weak_metric_min_selected_clients": 1,
        "recall_guard_floor": 0.70,
        "positive_rate_guard_floor": 0.12,
        "weak_auc_guard_floor": 0.72,
    },
    "ethereum_ponzi": {
        "min_edge_weight": 0.05,
        "startup_edge_cap": 0.14,
        "stable_edge_cap": 0.22,
        "collapse_edge_ratio": 0.55,
        "stable_edge_ratio": 0.90,
        "startup_lr_scale": 0.86,
        "collapse_lr_scale": 0.55,
        "startup_epoch_scale": 0.22,
        "mid_epoch_scale": 0.34,
        "late_epoch_scale": 0.28,
        "collapse_epoch_cap": 0.16,
        "max_local_epoch_multiplier": 4,
        "plateau_epoch_boost": 0.05,
        "deterministic_edge_loss_weight": 0.10,
        "deterministic_min_local_epochs": 3,
        "plateau_min_local_epochs": 4,
        "deterministic_lr_scale": 0.88,
        "sharp_drop_epoch_scale": 0.14,
        "weak_metric_min_selected_clients": 1,
        "recall_guard_floor": 0.58,
        "positive_rate_guard_floor": 0.08,
        "weak_auc_guard_floor": 0.62,
    },
    "defi_rug_pull": {
        "min_edge_weight": 0.06,
        "startup_edge_cap": 0.16,
        "stable_edge_cap": 0.24,
        "collapse_edge_ratio": 0.58,
        "stable_edge_ratio": 0.94,
        "startup_lr_scale": 0.86,
        "collapse_lr_scale": 0.56,
        "startup_epoch_scale": 0.24,
        "mid_epoch_scale": 0.36,
        "late_epoch_scale": 0.30,
        "collapse_epoch_cap": 0.16,
        "max_local_epoch_multiplier": 4,
        "plateau_epoch_boost": 0.05,
        "deterministic_edge_loss_weight": 0.12,
        "deterministic_min_local_epochs": 3,
        "plateau_min_local_epochs": 4,
        "deterministic_lr_scale": 0.90,
        "sharp_drop_epoch_scale": 0.14,
        "weak_metric_min_selected_clients": 1,
        "recall_guard_floor": 0.60,
        "positive_rate_guard_floor": 0.08,
        "weak_auc_guard_floor": 0.64,
    },
}
DATASET_OUTPUT_REGULARIZATION = {
    "default": {
        "target_prob_std": 0.0,
        "prob_std_regularization_weight": 0.0,
    },
    "amazon": {
        "target_prob_std": 0.05,
        "prob_std_regularization_weight": 0.02,
    },
    "yelp": {
        "target_prob_std": 0.02,
        "prob_std_regularization_weight": 0.15,
    },
    "comp": {
        "target_prob_std": 0.03,
        "prob_std_regularization_weight": 0.20,
    },
    "ieee": {
        "target_prob_std": 0.04,
        "prob_std_regularization_weight": 0.02,
    },
    "ccfd": {
        "target_prob_std": 0.04,
        "prob_std_regularization_weight": 0.02,
    },
    "elliptic": {
        "target_prob_std": 0.03,
        "prob_std_regularization_weight": 0.04,
    },
    "ethereum_phishing": {
        "target_prob_std": 0.03,
        "prob_std_regularization_weight": 0.06,
    },
    "ethereum_ponzi": {
        "target_prob_std": 0.02,
        "prob_std_regularization_weight": 0.03,
    },
    "defi_rug_pull": {
        "target_prob_std": 0.02,
        "prob_std_regularization_weight": 0.03,
    },
}
DATASET_MULTIMODAL_AUX_LOSS = {
    "default": {
        "graph_aux_loss_weight": 0.08,
        "sequence_aux_loss_weight": 0.03,
    },
    "amazon": {
        # Amazon benefits more from protecting the graph branch than from
        # forcing the Transformer branch to predict on its own.
        "graph_aux_loss_weight": 0.10,
        "sequence_aux_loss_weight": 0.0,
    },
    "yelp": {
        "graph_aux_loss_weight": 0.10,
        "sequence_aux_loss_weight": 0.05,
    },
    "comp": {
        # Keep the graph trunk explicitly supervised so the hybrid mainline
        # does not drift below the graph-only baseline on comp.
        "graph_aux_loss_weight": 0.12,
        "sequence_aux_loss_weight": 0.03,
    },
    "ieee": {
        "graph_aux_loss_weight": 0.08,
        "sequence_aux_loss_weight": 0.03,
    },
    "ccfd": {
        "graph_aux_loss_weight": 0.08,
        "sequence_aux_loss_weight": 0.03,
    },
    "elliptic": {
        "graph_aux_loss_weight": 0.10,
        "sequence_aux_loss_weight": 0.05,
    },
    "ethereum_phishing": {
        "graph_aux_loss_weight": 0.08,
        "sequence_aux_loss_weight": 0.04,
    },
    "ethereum_ponzi": {
        "graph_aux_loss_weight": 0.10,
        "sequence_aux_loss_weight": 0.0,
    },
    "defi_rug_pull": {
        "graph_aux_loss_weight": 0.10,
        "sequence_aux_loss_weight": 0.02,
    },
}
DATASET_GRAPH_ANCHOR_PROFILES = {
    "default": {
        "graph_anchor_loss_weight": 0.12,
        "graph_anchor_temperature": 1.50,
    },
    "amazon": {
        "graph_anchor_loss_weight": 0.10,
        "graph_anchor_temperature": 1.40,
    },
    "yelp": {
        "graph_anchor_loss_weight": 0.14,
        "graph_anchor_temperature": 1.45,
    },
    "comp": {
        # Comp remains graph-dominant; keep the fusion logits close enough to
        # the graph trunk during multimodal training so sequence gains stay
        # corrective instead of destabilizing the baseline.
        "graph_anchor_loss_weight": 0.20,
        "graph_anchor_temperature": 1.35,
    },
    "ieee": {
        "graph_anchor_loss_weight": 0.10,
        "graph_anchor_temperature": 1.45,
    },
    "ccfd": {
        "graph_anchor_loss_weight": 0.10,
        "graph_anchor_temperature": 1.45,
    },
    "elliptic": {
        "graph_anchor_loss_weight": 0.16,
        "graph_anchor_temperature": 1.35,
    },
    "ethereum_phishing": {
        "graph_anchor_loss_weight": 0.12,
        "graph_anchor_temperature": 1.40,
    },
    "ethereum_ponzi": {
        "graph_anchor_loss_weight": 0.16,
        "graph_anchor_temperature": 1.35,
    },
    "defi_rug_pull": {
        "graph_anchor_loss_weight": 0.14,
        "graph_anchor_temperature": 1.40,
    },
}
DATASET_MULTIMODAL_FUSION_PROFILES = {
    "default": {
        "graph_gate_logit_bias": 0.0,
        "eval_graph_gate_logit_bias": 0.0,
        "graph_residual_min_gate": 0.0,
        "sequence_residual_scale": 1.0,
    },
    "amazon": {
        "graph_gate_logit_bias": 0.18,
        "eval_graph_gate_logit_bias": 0.40,
        "graph_residual_min_gate": 0.62,
        "sequence_residual_scale": 0.85,
    },
    "comp": {
        # Restore the stronger historical comp fusion profile that previously
        # produced the best non-federated baseline.
        "graph_gate_logit_bias": 0.24,
        "eval_graph_gate_logit_bias": 0.24,
        "graph_residual_min_gate": 0.70,
        "sequence_residual_scale": 0.75,
    },
    "ieee": {
        "graph_gate_logit_bias": 0.12,
        "eval_graph_gate_logit_bias": 0.18,
        "graph_residual_min_gate": 0.40,
        "sequence_residual_scale": 0.90,
    },
    "ccfd": {
        "graph_gate_logit_bias": 0.12,
        "eval_graph_gate_logit_bias": 0.18,
        "graph_residual_min_gate": 0.40,
        "sequence_residual_scale": 0.90,
    },
    "elliptic": {
        "graph_gate_logit_bias": 0.08,
        "eval_graph_gate_logit_bias": 0.05,
        "graph_residual_min_gate": 0.42,
        "sequence_residual_scale": 1.08,
    },
    "ethereum_phishing": {
        "graph_gate_logit_bias": 0.10,
        "eval_graph_gate_logit_bias": 0.16,
        "graph_residual_min_gate": 0.45,
        "sequence_residual_scale": 0.95,
    },
    "ethereum_ponzi": {
        "graph_gate_logit_bias": 0.28,
        "eval_graph_gate_logit_bias": 0.32,
        "graph_residual_min_gate": 0.70,
        "sequence_residual_scale": 0.55,
    },
    "defi_rug_pull": {
        "graph_gate_logit_bias": 0.22,
        "eval_graph_gate_logit_bias": 0.26,
        "graph_residual_min_gate": 0.62,
        "sequence_residual_scale": 0.70,
    },
}

ELLIPTIC_EVAL_BRANCH_PRIORITY = (
    "sequence_residual",
    "raw_branch",
    "main",
    "fusion",
    "graph_residual",
)
LABEL_SCARCITY_PROFILES = [
    {
        "max_fraction": 0.01,
        "name": "scarcity_1pct",
        "pseudo_label_threshold": 0.97,
        "pseudo_label_min_threshold": 0.52,
        "pseudo_label_top_fraction": 0.005,
        "pseudo_label_weight": 0.04,
        "pseudo_label_novelty_threshold": 1.20,
        "consistency_weight": 0.28,
        "teacher_ema_decay": 0.997,
        "teacher_temperature": 0.85,
        "pseudo_warmup_rounds": 2,
        "pseudo_ramp_rounds": 4,
        "open_set_novelty_threshold": 1.90,
        "open_set_loss_weight": 0.14,
        "active_learning_budget_scale": 0.20,
        "active_learning_budget_min": 4,
        "active_learning_budget_max": 32,
        "active_learning_delay_rounds": 1,
        "active_learning_novelty_weight": 0.55,
        "active_learning_diversity_weight": 0.35,
    },
    {
        "max_fraction": 0.05,
        "name": "scarcity_5pct_stable",
        "pseudo_label_threshold": 0.95,
        "pseudo_label_min_threshold": 0.52,
        "pseudo_label_top_fraction": 0.015,
        "pseudo_label_weight": 0.08,
        "pseudo_label_novelty_threshold": 1.45,
        "consistency_weight": 0.22,
        "teacher_ema_decay": 0.995,
        "teacher_temperature": 0.90,
        "pseudo_warmup_rounds": 1,
        "pseudo_ramp_rounds": 3,
        "open_set_novelty_threshold": 2.10,
        "open_set_loss_weight": 0.10,
        "active_learning_budget_scale": 0.08,
        "active_learning_budget_min": 4,
        "active_learning_budget_max": 24,
        "active_learning_delay_rounds": 1,
        "active_learning_novelty_weight": 0.48,
        "active_learning_diversity_weight": 0.30,
    },
    {
        "max_fraction": 0.10,
        "name": "scarcity_10pct",
        "pseudo_label_threshold": 0.93,
        "pseudo_label_min_threshold": 0.53,
        "pseudo_label_top_fraction": 0.02,
        "pseudo_label_weight": 0.10,
        "pseudo_label_novelty_threshold": 1.80,
        "consistency_weight": 0.16,
        "teacher_ema_decay": 0.992,
        "teacher_temperature": 0.95,
        "pseudo_warmup_rounds": 1,
        "pseudo_ramp_rounds": 2,
        "open_set_novelty_threshold": 2.35,
        "open_set_loss_weight": 0.07,
        "active_learning_budget_scale": 0.04,
        "active_learning_budget_min": 4,
        "active_learning_budget_max": 16,
        "active_learning_delay_rounds": 1,
        "active_learning_novelty_weight": 0.42,
        "active_learning_diversity_weight": 0.27,
    },
]


def _resolve_label_scarcity_profile(dataset_name: str, label_fraction: float) -> dict:
    profile = {
        "name": "fully_supervised",
        "pseudo_label_threshold": 0.0,
        "pseudo_label_min_threshold": 0.0,
        "pseudo_label_top_fraction": 0.0,
        "pseudo_label_weight": 1.0,
        "pseudo_label_novelty_threshold": float("inf"),
        "consistency_weight": 0.0,
        "teacher_ema_decay": 0.0,
        "teacher_temperature": 1.0,
        "pseudo_warmup_rounds": 0,
        "pseudo_ramp_rounds": 0,
        "open_set_novelty_threshold": 0.0,
        "open_set_loss_weight": 0.0,
        "active_learning_budget_scale": 0.0,
        "active_learning_budget_min": 0,
        "active_learning_budget_max": 0,
        "active_learning_delay_rounds": 0,
        "active_learning_novelty_weight": 0.0,
        "active_learning_diversity_weight": 0.0,
    }
    if float(label_fraction) >= 0.999:
        return profile
    for candidate in LABEL_SCARCITY_PROFILES:
        if float(label_fraction) <= float(candidate["max_fraction"]) + 1e-9:
            profile.update(candidate)
            break
    profile["dataset"] = str(dataset_name)
    profile["label_fraction"] = float(label_fraction)
    return profile


def _resolve_profile_backed_weight(requested_value: float, profile_value: float, *, zero_disables: bool = False) -> float:
    requested_value = float(requested_value)
    profile_value = float(profile_value)
    if zero_disables and abs(requested_value) <= 1e-12:
        return 0.0
    return float(np.clip(max(requested_value, profile_value), 0.0, 1.0))


def _dataset_multimodal_aux_loss_profile(dataset_name: str) -> dict:
    return DATASET_MULTIMODAL_AUX_LOSS.get(dataset_name, DATASET_MULTIMODAL_AUX_LOSS["default"])


def _dataset_multimodal_fusion_profile(dataset_name: str) -> dict:
    return DATASET_MULTIMODAL_FUSION_PROFILES.get(
        dataset_name,
        DATASET_MULTIMODAL_FUSION_PROFILES["default"],
    )


def _dataset_graph_anchor_profile(dataset_name: str) -> dict:
    return DATASET_GRAPH_ANCHOR_PROFILES.get(
        dataset_name,
        DATASET_GRAPH_ANCHOR_PROFILES["default"],
    )




def _atomic_write_json(path: Path, payload: dict) -> None:
    checkpoint_atomic_write_json(path, payload)


def _atomic_torch_save(path: Path, payload: dict) -> None:
    checkpoint_atomic_torch_save(path, payload)


def _run_metadata_payload(
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
) -> dict:
    return checkpoint_run_metadata_payload(
        dataset_name=dataset_name,
        run_id=run_id,
        status=status,
        federated_rounds=federated_rounds,
        local_epochs=local_epochs,
        planner_mode=planner_mode,
        test_every=test_every,
        resume_path=resume_path,
        tb_logdir=tb_logdir,
        seed=seed,
        summary_path=summary_path,
        model_path=model_path,
        rounds_ran=rounds_ran,
        best_round=best_round,
        best_valid_auc=best_valid_auc,
        test_auc=test_auc,
        finished_at=finished_at,
    )


def _dataset_probability_rule(dataset_name: str) -> dict:
    rule = dict(DATASET_PROBABILITY_RULES["default"])
    rule.update(DATASET_PROBABILITY_RULES.get(dataset_name, {}))
    return rule


def _dataset_output_regularization(dataset_name: str) -> dict:
    regularization = dict(DATASET_OUTPUT_REGULARIZATION["default"])
    regularization.update(DATASET_OUTPUT_REGULARIZATION.get(dataset_name, {}))
    return regularization


def _dataset_controller_reward_profile(dataset_name: str) -> dict:
    profile = dict(DATASET_CONTROLLER_REWARD_PROFILES["default"])
    profile.update(DATASET_CONTROLLER_REWARD_PROFILES.get(dataset_name, {}))
    return profile


def _round_stability_flags(dataset_name: str, metrics_history: List[dict]) -> dict:
    """Summarize collapse and plateau signals from recent validation metrics."""
    if not metrics_history:
        return {
            "probability_collapse": False,
            "severe_probability_collapse": False,
            "auc_plateau": False,
            "stagnation_rounds": 0,
            "sharp_auc_drop": False,
            "recent_auc_delta": 0.0,
        }

    probability_rule = _dataset_probability_rule(dataset_name)
    collapse_std = float(probability_rule["collapse_std"])
    severe_collapse_std = float(probability_rule["severe_collapse_std"])
    weak_auc_floor = float(probability_rule["weak_auc_floor"])
    weak_peak_auc_floor = float(probability_rule["weak_peak_auc_floor"])
    severe_auc_floor = float(probability_rule["severe_auc_floor"])
    severe_peak_auc_floor = float(probability_rule["severe_peak_auc_floor"])
    sharp_drop_delta = float(probability_rule.get("sharp_drop_delta", -0.08))

    last_prob_std = float(metrics_history[-1].get("valid_prob_std", 0.0))
    last_auc = float(metrics_history[-1].get("valid_auc", 0.0))
    previous_auc = float(metrics_history[-2].get("valid_auc", last_auc)) if len(metrics_history) >= 2 else last_auc
    recent_auc_delta = float(last_auc - previous_auc)
    recent_window = metrics_history[-3:]
    recent_prob_stds = [float(metrics.get("valid_prob_std", 0.0)) for metrics in recent_window]
    recent_aucs = [float(metrics.get("valid_auc", 0.0)) for metrics in recent_window]
    persistent_low_prob_std = len(recent_prob_stds) >= 2 and all(
        prob_std < collapse_std for prob_std in recent_prob_stds[-2:]
    )
    recent_auc_peak = max(recent_aucs) if recent_aucs else last_auc
    sustained_weak_auc = len(recent_aucs) >= 2 and max(recent_aucs[-2:]) < weak_auc_floor
    sustained_severe_auc = len(recent_aucs) >= 2 and max(recent_aucs[-2:]) < severe_auc_floor
    prob_std_decay = True
    if bool(probability_rule.get("require_prob_std_decay", False)) and len(recent_prob_stds) >= 3:
        recent_prob_peak = max(recent_prob_stds)
        prob_std_decay = recent_prob_peak > 0.0 and recent_prob_stds[-1] <= recent_prob_peak * float(
            probability_rule["prob_std_decay_ratio"]
        )
    probability_collapse = (
        len(metrics_history) >= 3
        and last_prob_std < collapse_std
        and persistent_low_prob_std
        and prob_std_decay
        and last_auc < weak_auc_floor
        and recent_auc_peak < weak_peak_auc_floor
        and sustained_weak_auc
    )
    severe_probability_collapse = (
        len(metrics_history) >= 3
        and last_prob_std < severe_collapse_std
        and persistent_low_prob_std
        and prob_std_decay
        and last_auc < severe_auc_floor
        and recent_auc_peak < severe_peak_auc_floor
        and sustained_severe_auc
    )

    auc_values = [float(metrics.get("valid_auc", 0.0)) for metrics in metrics_history]
    best_index = int(np.argmax(auc_values))
    stagnation_rounds = len(auc_values) - 1 - best_index
    auc_plateau = stagnation_rounds >= PLATEAU_STAGNATION_ROUNDS and len(metrics_history) >= 4
    sharp_auc_drop = len(metrics_history) >= 2 and recent_auc_delta <= sharp_drop_delta
    return {
        "probability_collapse": probability_collapse,
        "severe_probability_collapse": severe_probability_collapse,
        "auc_plateau": auc_plateau,
        "stagnation_rounds": stagnation_rounds,
        "sharp_auc_drop": sharp_auc_drop,
        "recent_auc_delta": recent_auc_delta,
    }


def _checkpoint_selection_guard(dataset_name: str, valid_metrics: dict) -> dict:
    """Reject weak checkpoints that only win because of an early collapsed AUC bump."""
    probability_rule = _dataset_probability_rule(dataset_name)
    valid_auc = float(valid_metrics.get("auc", valid_metrics.get("valid_auc", 0.0)))
    valid_prob_std = float(valid_metrics.get("prob_std", valid_metrics.get("valid_prob_std", 0.0)))
    severe_collapse_std = float(probability_rule["severe_collapse_std"])
    severe_auc_floor = float(probability_rule["severe_auc_floor"])
    collapse_std = float(probability_rule["collapse_std"])
    weak_auc_floor = float(probability_rule["weak_auc_floor"])
    if valid_prob_std < severe_collapse_std and valid_auc < severe_auc_floor:
        return {"eligible": False, "reason": "severe_low_prob_std_and_auc"}
    if valid_prob_std < collapse_std and valid_auc < weak_auc_floor:
        return {"eligible": False, "reason": "low_prob_std_and_auc"}
    return {"eligible": True, "reason": "ok"}


def _dataset_scheduler_profile(dataset_name: str) -> dict:
    return DATASET_SCHEDULER_PROFILES.get(dataset_name, DATASET_SCHEDULER_PROFILES["yelp"])


def _adaptive_edge_loss_weight(
    dataset_name: str,
    base_edge_loss_weight: float,
    metrics_history: List[dict],
    stability_flags: dict,
    progress: float,
) -> float:
    profile = _dataset_scheduler_profile(dataset_name)
    minimum_edge_weight = float(profile["min_edge_weight"])
    startup_cap = float(profile["startup_edge_cap"])
    stable_cap = float(profile["stable_edge_cap"])

    if not metrics_history:
        return float(np.clip(min(base_edge_loss_weight, startup_cap), minimum_edge_weight, stable_cap))

    last_metrics = metrics_history[-1]
    cls_loss = max(float(last_metrics.get("mean_local_cls_loss", 0.0)), 1e-6)
    edge_loss = max(float(last_metrics.get("mean_local_edge_loss", 0.0)), 1e-6)

    if stability_flags["probability_collapse"]:
        cap = startup_cap
        target_ratio = float(profile["collapse_edge_ratio"])
    else:
        cap = stable_cap if progress >= 0.33 else (startup_cap + stable_cap) / 2.0
        target_ratio = float(profile["stable_edge_ratio"])

    balanced_weight = target_ratio * cls_loss / edge_loss
    resolved = min(float(base_edge_loss_weight), cap, balanced_weight)
    resolved = float(np.clip(resolved, minimum_edge_weight, cap))
    if metrics_history:
        previous_weight = float(metrics_history[-1].get("edge_loss_weight", resolved))
        lower_bound = max(minimum_edge_weight, previous_weight * 0.85)
        upper_bound = min(cap, previous_weight * 1.15)
        resolved = float(np.clip(resolved, lower_bound, upper_bound))
        resolved = float(0.70 * previous_weight + 0.30 * resolved)
    return float(np.clip(resolved, minimum_edge_weight, cap))


def _adaptive_learning_rate(
    dataset_name: str,
    base_learning_rate: float,
    metrics_history: List[dict],
    stability_flags: dict,
    progress: float,
) -> float:
    profile = _dataset_scheduler_profile(dataset_name)
    lr_scale = 1.0
    if progress <= 0.20:
        lr_scale = min(lr_scale, float(profile["startup_lr_scale"]))
    if stability_flags["probability_collapse"]:
        lr_scale = min(lr_scale, float(profile["collapse_lr_scale"]))
    resolved = float(max(base_learning_rate * lr_scale, base_learning_rate * 0.4))
    if metrics_history:
        previous_lr = float(metrics_history[-1].get("learning_rate", resolved))
        lower_bound = max(base_learning_rate * 0.4, previous_lr * 0.85)
        upper_bound = min(base_learning_rate, previous_lr * 1.10)
        resolved = float(np.clip(resolved, lower_bound, upper_bound))
        resolved = float(0.70 * previous_lr + 0.30 * resolved)
    return resolved


def _apply_dp_noise_to_state_dict(state_dict: Dict[str, torch.Tensor], noise_std: float) -> Dict[str, torch.Tensor]:
    return platform_apply_dp_noise_to_state_dict(state_dict, noise_std)


def _controller_reward_from_round(
    dataset_name: str,
    round_index: int,
    total_rounds: int,
    base_local_epochs: int,
    previous_metrics: dict | None,
    current_metrics: dict,
    round_plan: dict,
) -> tuple[float, dict]:
    """Compute the controller reward for a training round."""

    profile = _dataset_scheduler_profile(dataset_name)
    reward_profile = _dataset_controller_reward_profile(dataset_name)
    observation = np.asarray(round_plan.get("observation", []), dtype=np.float32)
    if observation.size < 5:
        observation = np.array([0.0, 0.0, 5.0, 5.0, 5.0], dtype=np.float32)

    imbalance_signal = float(np.clip(observation[0], -1.0, 1.0))
    round_progress = float(np.clip(observation[1] / 10.0, 0.0, 1.0))
    relation_complexity = float(np.clip(observation[2] / 10.0, 0.0, 1.0))
    validation_pressure = float(np.clip(observation[3] / 10.0, 0.0, 1.0))
    loss_pressure = float(np.clip(observation[4] / 10.0, 0.0, 1.0))

    previous_metrics = previous_metrics or {}
    prev_auc = float(previous_metrics.get("valid_auc", 0.5))
    prev_f1 = float(previous_metrics.get("valid_f1_macro", 0.5))
    prev_recall = float(previous_metrics.get("valid_recall", 0.5))
    prev_prob_std = float(previous_metrics.get("valid_prob_std", 0.02))

    current_auc = float(current_metrics.get("valid_auc", prev_auc))
    current_f1 = float(current_metrics.get("valid_f1_macro", prev_f1))
    current_recall = float(current_metrics.get("valid_recall", prev_recall))
    current_prob_std = float(current_metrics.get("valid_prob_std", prev_prob_std))
    probability_rule = _dataset_probability_rule(dataset_name)

    auc_gain = current_auc - prev_auc
    f1_gain = current_f1 - prev_f1
    recall_gain = current_recall - prev_recall
    prob_std_gain = current_prob_std - prev_prob_std

    actual_selected_ratio = float(round_plan.get("actual_selected_ratio", round_plan.get("selected_ratio", 0.0)))
    local_epochs = float(round_plan.get("local_epochs", 1.0))
    grad_clip = float(round_plan.get("grad_clip", 1.0))
    selection_phase = float(round_plan.get("selection_phase", 0.5))

    max_local_epochs = max(
        int(round(base_local_epochs * float(profile["max_local_epoch_multiplier"]))),
        int(base_local_epochs) + 2,
        4,
    )
    max_local_epochs = max(max_local_epochs, int(round_plan.get("local_epochs", 1)))
    epoch_scale = float(np.clip(local_epochs / max(max_local_epochs, 1), 0.0, 1.0))
    norm_clip = float(np.clip((grad_clip - 0.5) / 1.5, 0.0, 1.0))

    target_selected_ratio = float(np.clip(
        0.35 + 0.25 * abs(imbalance_signal) + 0.20 * validation_pressure + 0.10 * loss_pressure + 0.05 * relation_complexity,
        0.20,
        1.0,
    ))
    target_epoch_scale = float(np.clip(
        0.22 + 0.35 * round_progress + 0.18 * validation_pressure + 0.15 * loss_pressure,
        0.10,
        1.0,
    ))
    target_grad_clip = float(np.clip(0.85 + 0.10 * loss_pressure + 0.05 * validation_pressure, 0.6, 1.4))
    target_phase = float(np.clip((imbalance_signal + 1.0) / 2.0, 0.0, 1.0))

    alignment = 1.0 - (
        0.42 * abs(actual_selected_ratio - target_selected_ratio)
        + 0.30 * abs(epoch_scale - target_epoch_scale)
        + 0.18 * abs(norm_clip - target_grad_clip)
        + 0.10 * abs(selection_phase - target_phase)
    )
    alignment = float(np.clip(alignment, -1.0, 1.0))

    communication_cost = actual_selected_ratio
    compute_cost = epoch_scale
    clip_cost = abs(norm_clip - 2.0 / 3.0)
    quality_bonus = max(current_auc - float(reward_profile["quality_auc_floor"]), 0.0)
    quality_bonus += 0.5 * max(current_f1 - float(reward_profile["quality_f1_floor"]), 0.0)
    quality_bonus = float(np.clip(quality_bonus, 0.0, 1.0))

    collapse_penalty = 0.0
    weak_auc_floor = float(probability_rule["weak_auc_floor"])
    reward_low_prob_std_floor = probability_rule.get("reward_low_prob_std_floor")
    collapse_triggered = bool(current_metrics.get("probability_collapse", False))
    low_prob_std_triggered = (
        reward_low_prob_std_floor is not None
        and current_prob_std < float(reward_low_prob_std_floor)
        and current_auc < weak_auc_floor
    )
    if collapse_triggered or low_prob_std_triggered:
        collapse_gap = max(weak_auc_floor - current_auc, 0.0)
        collapse_penalty = float(np.clip(0.35 + 3.0 * collapse_gap, 0.0, 1.0))

    plateau_penalty = 0.0
    plateau_triggered = bool(current_metrics.get("auc_plateau", False))
    plateau_auc_gap = max(float(reward_profile["plateau_auc_floor"]) - current_auc, 0.0)
    plateau_f1_gap = max(float(reward_profile["plateau_f1_floor"]) - current_f1, 0.0)
    if plateau_triggered and (plateau_auc_gap > 0.0 or plateau_f1_gap > 0.0):
        plateau_penalty = float(np.clip(plateau_auc_gap + 0.5 * plateau_f1_gap, 0.0, 1.0))
    stagnation_rounds = float(current_metrics.get("stagnation_rounds", 0.0))
    stagnation_penalty = 0.0
    if plateau_penalty > 0.0:
        stagnation_penalty = 0.03 * stagnation_rounds
    progress_bonus = float(np.clip(round_index / max(total_rounds - 1, 1), 0.0, 1.0))

    reward = (
        CONTROLLER_REWARD_WEIGHTS["auc_gain"] * auc_gain
        + CONTROLLER_REWARD_WEIGHTS["f1_gain"] * f1_gain
        + CONTROLLER_REWARD_WEIGHTS["recall_gain"] * recall_gain
        + CONTROLLER_REWARD_WEIGHTS["prob_std_gain"] * prob_std_gain
        + CONTROLLER_REWARD_WEIGHTS["alignment"] * alignment
        + CONTROLLER_REWARD_WEIGHTS["quality_bonus"] * quality_bonus
        + 0.10 * progress_bonus
        - CONTROLLER_REWARD_WEIGHTS["communication_cost"] * communication_cost
        - CONTROLLER_REWARD_WEIGHTS["compute_cost"] * compute_cost
        - CONTROLLER_REWARD_WEIGHTS["clip_cost"] * clip_cost
        - CONTROLLER_REWARD_WEIGHTS["collapse_penalty"] * collapse_penalty
        - CONTROLLER_REWARD_WEIGHTS["plateau_penalty"] * plateau_penalty
        - stagnation_penalty
    )
    reward = float(np.clip(reward, -5.0, 5.0))
    details = {
        "auc_gain": float(auc_gain),
        "f1_gain": float(f1_gain),
        "recall_gain": float(recall_gain),
        "prob_std_gain": float(prob_std_gain),
        "alignment": float(alignment),
        "quality_bonus": float(quality_bonus),
        "communication_cost": float(communication_cost),
        "compute_cost": float(compute_cost),
        "clip_cost": float(clip_cost),
        "collapse_penalty": float(collapse_penalty),
        "plateau_penalty": float(plateau_penalty),
        "stagnation_rounds": float(stagnation_rounds),
        "stagnation_penalty": float(stagnation_penalty),
        "reward": float(reward),
    }
    return reward, details


def _proxy_controller_reward(state: np.ndarray, action: np.ndarray, step_index: int, episode_length: int) -> tuple[float, dict]:
    """Proxy reward used to pretrain the controller before real training data exists."""

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if state.size < 5:
        state = np.array([0.0, 0.0, 5.0, 5.0, 5.0], dtype=np.float32)
    if action.size < 4:
        action = np.pad(action, (0, max(0, 4 - action.size)), mode="constant")

    imbalance_signal = float(np.clip(state[0], -1.0, 1.0))
    round_progress = float(np.clip(state[1] / 10.0, 0.0, 1.0))
    relation_complexity = float(np.clip(state[2] / 10.0, 0.0, 1.0))
    validation_pressure = float(np.clip(state[3] / 10.0, 0.0, 1.0))
    loss_pressure = float(np.clip(state[4] / 10.0, 0.0, 1.0))

    selection_phase = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
    selected_ratio = float(np.clip(action[1], 1e-3, 1.0))
    local_epoch_scale = float(np.clip(action[2], 1e-3, 1.0))
    grad_clip = float(np.clip(0.5 + action[3] / 10.0, 0.5, 2.0))
    norm_clip = float(np.clip((grad_clip - 0.5) / 1.5, 0.0, 1.0))

    target_selected_ratio = float(np.clip(
        0.30 + 0.30 * abs(imbalance_signal) + 0.20 * validation_pressure + 0.10 * loss_pressure + 0.05 * relation_complexity,
        0.20,
        1.0,
    ))
    target_epoch_scale = float(np.clip(
        0.20 + 0.40 * round_progress + 0.20 * validation_pressure + 0.10 * loss_pressure,
        0.10,
        1.0,
    ))
    target_grad_clip = float(np.clip(0.80 + 0.12 * loss_pressure + 0.05 * validation_pressure, 0.55, 1.35))
    target_phase = float(np.clip((imbalance_signal + 1.0) / 2.0, 0.0, 1.0))

    alignment = 1.0 - (
        0.42 * abs(selected_ratio - target_selected_ratio)
        + 0.30 * abs(local_epoch_scale - target_epoch_scale)
        + 0.18 * abs(norm_clip - target_grad_clip)
        + 0.10 * abs(selection_phase - target_phase)
    )
    alignment = float(np.clip(alignment, -1.0, 1.0))

    progress_bonus = float(np.clip(step_index / max(episode_length - 1, 1), 0.0, 1.0))
    cost = 0.45 * selected_ratio + 0.25 * local_epoch_scale + 0.10 * abs(norm_clip - 2.0 / 3.0)
    reward = float(np.clip(1.5 * alignment + 0.15 * progress_bonus - cost, -5.0, 5.0))
    details = {
        "alignment": float(alignment),
        "progress_bonus": float(progress_bonus),
        "cost": float(cost),
        "reward": float(reward),
    }
    return reward, details


class HybridControlEnv(_GymEnvBase):
    """Lightweight RL environment for controller pretraining."""

    metadata = {"render_modes": []}

    def __init__(self, episode_length: int = 12):
        _require_rl_dependencies()
        super().__init__()
        self.episode_length = max(int(episode_length), 1)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 1e-3, 1e-3, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._step_index = 0
        self.state = np.zeros(5, dtype=np.float32)

    def _initial_state(self) -> np.ndarray:
        return np.array([0.15, 0.0, 5.0, 6.0, 5.0], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._step_index = 0
        self.state = self._initial_state().copy()
        return self.state.copy(), {}

    def step(self, action):  # type: ignore[override]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        reward, reward_details = _proxy_controller_reward(self.state, action, self._step_index, self.episode_length)
        reward_details = dict(reward_details)

        imbalance_signal = float(np.clip(self.state[0], -1.0, 1.0))
        round_progress = float(np.clip(self.state[1] / 10.0, 0.0, 1.0))
        relation_complexity = float(np.clip(self.state[2] / 10.0, 0.0, 1.0))
        validation_pressure = float(np.clip(self.state[3] / 10.0, 0.0, 1.0))
        loss_pressure = float(np.clip(self.state[4] / 10.0, 0.0, 1.0))

        selection_phase = float(np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0))
        selected_ratio = float(np.clip(action[1], 1e-3, 1.0))
        local_epoch_scale = float(np.clip(action[2], 1e-3, 1.0))
        grad_clip = float(np.clip(0.5 + action[3] / 10.0, 0.5, 2.0))
        norm_clip = float(np.clip((grad_clip - 0.5) / 1.5, 0.0, 1.0))

        alignment = float(reward_details["alignment"])
        adaptation = float(np.clip(0.35 + 0.40 * alignment + 0.20 * validation_pressure + 0.10 * loss_pressure, 0.0, 1.0))
        next_validation_pressure = float(np.clip(validation_pressure - 0.22 * adaptation + 0.08 * (1.0 - alignment), 0.0, 1.0))
        next_loss_pressure = float(np.clip(loss_pressure - 0.25 * adaptation + 0.10 * (1.0 - alignment), 0.0, 1.0))
        next_imbalance = float(np.clip(imbalance_signal * (1.0 - 0.04 * adaptation), -1.0, 1.0))
        next_progress = float(np.clip(round_progress + 1.0 / max(self.episode_length, 1), 0.0, 1.0))
        next_complexity = float(np.clip(relation_complexity * (1.0 - 0.03 * alignment), 0.0, 1.0))

        self.state = np.array(
            [
                next_imbalance,
                next_progress * 10.0,
                next_complexity * 10.0,
                next_validation_pressure * 10.0,
                next_loss_pressure * 10.0,
            ],
            dtype=np.float32,
        )
        self._step_index += 1
        terminated = self._step_index >= self.episode_length
        truncated = False
        info = {
            "reward_details": reward_details,
            "selected_ratio": selected_ratio,
            "local_epoch_scale": local_epoch_scale,
            "grad_clip": grad_clip,
            "normalized_grad_clip": norm_clip,
            "selection_phase": selection_phase,
        }
        return self.state.copy(), float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None


def create_replay_buffer() -> ReplayBuffer:
    """Create a replay buffer compatible with the controller environment."""
    _require_rl_dependencies()
    return ReplayBuffer(
        buffer_size=AGGREGATED_REPLAY_BUFFER_SIZE,
        observation_space=spaces.Box(
            low=np.array([-1, 0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1, 10.0, 10.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        ),
        action_space=spaces.Box(
            low=np.array([-1, 1e-3, 1e-3, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 10], dtype=np.float32),
            dtype=np.float32,
        ),
    )


def init_model(rl_timesteps: int = 512, seed: int = 42) -> List[FRModel]:
    """Initialize a small ensemble of TD3 controllers."""
    _require_rl_dependencies()
    probe_env = HybridControlEnv()
    n_actions = probe_env.action_space.shape[-1]
    probe_env.close()
    steps_per_model = max(rl_timesteps // 3, 64)
    learning_starts = min(100, max(10, steps_per_model // 4))
    return [
        FRModel(
            "MlpPolicy",
            HybridControlEnv(),
            action_noise=NormalActionNoise(
                mean=np.zeros(n_actions, dtype=np.float32),
                sigma=0.1 * np.ones(n_actions, dtype=np.float32),
            ),
            buffer_size=TD3_REPLAY_BUFFER_SIZE,
            learning_starts=learning_starts,
            verbose=0,
            seed=int(seed) + index,
        )
        for index in range(3)
    ]


def _close_rl_models(models: List[FRModel]) -> None:
    """Close controller environments and release resources."""
    for model in models:
        try:
            env = model.get_env()
            if env is not None:
                env.close()
        except Exception:
            continue


def merge_replay_buffer(buffer_1: ReplayBuffer, buffer_2: ReplayBuffer) -> ReplayBuffer:
    """Merge two replay buffers in insertion order."""
    temp = copy.deepcopy(buffer_1)
    for index in range(buffer_2.pos):
        temp.add(
            obs=buffer_2.observations[index],
            next_obs=buffer_2.next_observations[index],
            action=buffer_2.actions[index],
            reward=buffer_2.rewards[index],
            done=buffer_2.dones[index],
            infos=[{}],
        )
    return temp


def sample_replay_buffer(replay_buffer: ReplayBuffer, batch_size: int) -> ReplayBuffer:
    """Sample transitions with replacement into a fresh replay buffer."""
    temp = create_replay_buffer()
    if replay_buffer.pos <= 0:
        return temp
    max_index = max(replay_buffer.pos, 1)
    sample_indexes = np.random.randint(low=0, high=max_index, size=batch_size)
    for index in sample_indexes:
        temp.add(
            obs=replay_buffer.observations[index],
            next_obs=replay_buffer.next_observations[index],
            action=replay_buffer.actions[index],
            reward=replay_buffer.rewards[index],
            done=replay_buffer.dones[index],
            infos=[{}],
        )
    return temp


def fed_avg(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Compute the standard FedAvg weighted average."""
    weight_avg = copy.deepcopy(weights[0])
    for key in weight_avg.keys():
        for index in range(1, len(weights)):
            weight_avg[key] += weights[index][key]
        weight_avg[key] = torch.div(weight_avg[key], len(weights))
    return weight_avg


def train_controller_models(rl_timesteps: int = 512, seed: int | None = None) -> List[FRModel]:
    """Pretrain the controller ensemble on the proxy environment."""
    _require_rl_dependencies()
    base_seed = 42 if seed is None else int(seed)
    setup_seed(base_seed)
    all_replay_buffers = create_replay_buffer()
    models = init_model(rl_timesteps=rl_timesteps, seed=base_seed)
    steps_per_model = max(rl_timesteps // max(len(models), 1), 64)

    for model in models:
        model.learn(total_timesteps=steps_per_model, log_interval=max(steps_per_model, 1))
        all_replay_buffers = merge_replay_buffer(all_replay_buffers, model.replay_buffer)

    for model in models:
        model.replay_buffer = sample_replay_buffer(all_replay_buffers, 100)
    return models


def _default_splitgnn_training_config(dataset_name: str) -> dict[str, float | int]:
    """Return minimal SplitGNN-compatible defaults when legacy YAML files are absent."""
    normalized_dataset = str(dataset_name).lower()
    defaults: dict[str, float | int] = {
        "lr": 3e-3,
        "weight_decay": 5e-5,
        "dropout": 0.1,
        "gamma": 1.0,
        "C": 1,
        "K": 0,
        "intra_dim": 8,
        "n_class": 2,
        "early_stop": 0,
    }
    if normalized_dataset == "comp":
        defaults["lr"] = 2e-3
    elif normalized_dataset == "amazon":
        defaults["lr"] = 5e-3
    elif normalized_dataset == "defi_rug_pull":
        defaults["lr"] = 4e-3
    return defaults


def _load_splitgnn_yaml(dataset_name: str) -> dict:
    """Load the dataset-specific SplitGNN YAML config."""
    config_path = SPLITGNN_CONFIG_DIR / f"{dataset_name}.yaml"
    defaults = _default_splitgnn_training_config(dataset_name)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        return {**defaults, **dict(loaded)}

    warnings.warn(
        f"SplitGNN config not found at {config_path}. Using built-in minimal defaults for {dataset_name}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return defaults


def _resolve_hybrid_learning_rate(dataset_name: str, base_lr: float) -> float:
    """Resolve a dataset-aware learning-rate cap for the hybrid model."""
    if dataset_name == "amazon":
        return min(base_lr, 5e-3)
    if dataset_name == "comp":
        return min(base_lr, 3e-3)
    if dataset_name == "elliptic":
        return min(base_lr, 3e-3)
    if dataset_name == "ieee":
        return min(base_lr, 3e-3)
    if dataset_name == "ethereum_phishing":
        return min(base_lr, 3e-3)
    if dataset_name == "ethereum_ponzi":
        return min(base_lr, 5e-3)
    if dataset_name == "defi_rug_pull":
        return min(base_lr, 4e-3)
    return min(base_lr, 1e-2)


def _resolve_training_device(device_name: str) -> torch.device:
    """Resolve training device and safely fallback when CUDA/DGL-CUDA is unavailable."""
    return resolve_dgl_training_device(device_name)


def _env_bool_override(name: str, default: bool) -> bool:
    raw_value = str(os.getenv(name, "")).strip().lower()
    if not raw_value:
        return bool(default)
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _build_training_args(
    dataset_name: str,
    device: str,
    amp_dtype: str,
    federated_rounds: int,
    schedule_total_rounds: int | None,
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
    seq_hidden_dim: int,
    fusion_hidden_dim: int,
    early_stop: int | None,
    planner_mode: str,
    test_every: int,
    fixed_precision_target: float = 0.5,
    transformer_hidden_dim: int | None = None,
    transformer_num_layers: int = 1,
    sequence_batch_chunk_size: int | None = None,
    event_batch_chunk_size: int | None = None,
    transformer_activation_checkpointing: bool = True,
    active_learning_feedback_path: str = "",
    seed: int | None = None,
    result_root: str = "",
    disable_gnn: bool = False,
    disable_transformer: bool = False,
    disable_federated: bool = False,
    disable_relation_sequence_encoder: bool = False,
    disable_event_transformer_encoder: bool = False,
    disable_temporal_context_encoder: bool = False,
    disable_graph_temporal_fusion: bool = False,
    force_disable_wavelet_lite: bool = False,
    force_disable_utg_lite: bool = False,
    force_disable_coassociation: bool = False,
    force_disable_diffusion_residual: bool = False,
    learning_rate_override: float | None = None,
    weight_decay_override: float | None = None,
    dropout_override: float | None = None,
    graph_aux_loss_weight_override: float | None = None,
    sequence_aux_loss_weight_override: float | None = None,
    graph_gate_logit_bias_override: float | None = None,
    eval_graph_gate_logit_bias_override: float | None = None,
    graph_residual_min_gate_override: float | None = None,
    sequence_residual_scale_override: float | None = None,
    preferred_eval_branch_override: str | None = None,
    eval_branch_priority_override: str | list[str] | tuple[str, ...] | None = None,
    fusion_variant_override: str | None = None,
    modality_dropout_prob_override: float | None = None,
    graph_learning_rate_scale_override: float | None = None,
    sequence_learning_rate_scale_override: float | None = None,
    fusion_learning_rate_scale_override: float | None = None,
    graph_follow_learning_rate_scale_override: float | None = None,
    graph_warmup_rounds_override: int | None = None,
    fusion_bootstrap_rounds_override: int | None = None,
    teacher_ema_decay_override: float | None = None,
    pseudo_warmup_rounds_override: int | None = None,
    pseudo_ramp_rounds_override: int | None = None,
    open_set_novelty_threshold_override: float | None = None,
    open_set_loss_weight_override: float | None = None,
    prototype_loss_weight_override: float | None = None,
    shared_private_loss_weight_override: float | None = None,
    context_alignment_loss_weight_override: float | None = None,
    uncertainty_loss_weight_override: float | None = None,
    graph_anchor_loss_weight_override: float | None = None,
    graph_anchor_temperature_override: float | None = None,
    graph_teacher_checkpoint_path: str = "",
    graph_teacher_distill_weight: float = 0.0,
    graph_teacher_temperature: float = 1.5,
    legacy_fusion_only_override: bool | None = None,
    epoch_metric_recompute_mode: str | None = None,
    pure_label_fraction: bool = False,
) -> SimpleNamespace:
    """Build the hybrid training namespace from the dataset config."""
    runtime_policy = resolve_splitgnn_runtime_policy(
        dataset_name=dataset_name,
        planner_mode=planner_mode,
        disable_federated=disable_federated,
    )
    config = _load_splitgnn_yaml(dataset_name)
    args = SimpleNamespace(**config)
    if seed is not None:
        args.seed = int(seed)
    args.requested_device = str(device)
    args.device = _resolve_training_device(device)
    args.amp_dtype = str(amp_dtype)
    args.dataset = dataset_name
    base_lr = float(learning_rate_override) if learning_rate_override is not None else float(args.lr)
    args.base_lr = base_lr
    args.lr = _resolve_hybrid_learning_rate(dataset_name, base_lr)
    if weight_decay_override is not None:
        args.weight_decay = float(weight_decay_override)
    if dropout_override is not None:
        args.dropout = float(dropout_override)
    resolved_transformer_hidden_dim = int(transformer_hidden_dim if transformer_hidden_dim is not None else seq_hidden_dim)
    args.transformer_hidden_dim = resolved_transformer_hidden_dim
    args.seq_hidden_dim = resolved_transformer_hidden_dim
    args.transformer_num_layers = max(int(transformer_num_layers), 1)
    default_transformer_chunk_size = 256 if str(dataset_name).lower() == "ieee" else None
    args.sequence_batch_chunk_size = (
        max(int(sequence_batch_chunk_size), 1)
        if sequence_batch_chunk_size is not None
        else default_transformer_chunk_size
    )
    args.event_batch_chunk_size = (
        max(int(event_batch_chunk_size), 1)
        if event_batch_chunk_size is not None
        else default_transformer_chunk_size
    )
    args.requested_transformer_activation_checkpointing = bool(transformer_activation_checkpointing)
    args.transformer_activation_checkpointing = (
        True if str(dataset_name).lower() == "ieee" else bool(transformer_activation_checkpointing)
    )
    args.fusion_hidden_dim = fusion_hidden_dim
    args.feature_hidden_dim = max(int(fusion_hidden_dim), int(resolved_transformer_hidden_dim))
    args.raw_anchor_dim = int(resolved_transformer_hidden_dim)
    dataset_name_normalized = str(dataset_name).lower()
    args.edge_loss_weight = (
        min(float(edge_loss_weight), 0.12)
        if dataset_name_normalized == "ieee"
        else float(edge_loss_weight)
    )
    output_regularization = _dataset_output_regularization(dataset_name)
    args.target_prob_std = float(output_regularization["target_prob_std"])
    args.prob_std_regularization_weight = float(output_regularization["prob_std_regularization_weight"])
    multimodal_aux_profile = _dataset_multimodal_aux_loss_profile(dataset_name)
    multimodal_aux_enabled = not bool(disable_gnn) and not bool(disable_transformer)
    default_graph_aux_loss_weight = float(multimodal_aux_profile["graph_aux_loss_weight"]) if multimodal_aux_enabled else 0.0
    default_sequence_aux_loss_weight = (
        float(multimodal_aux_profile["sequence_aux_loss_weight"]) if multimodal_aux_enabled else 0.0
    )
    args.graph_aux_loss_weight = (
        float(graph_aux_loss_weight_override)
        if multimodal_aux_enabled and graph_aux_loss_weight_override is not None
        else (
            0.04
            if dataset_name_normalized == "ieee" and multimodal_aux_enabled
            else default_graph_aux_loss_weight
        )
    )
    args.sequence_aux_loss_weight = (
        float(sequence_aux_loss_weight_override)

        if multimodal_aux_enabled and sequence_aux_loss_weight_override is not None
        else (
            0.03
            if dataset_name_normalized == "ieee" and multimodal_aux_enabled
            else default_sequence_aux_loss_weight
        )
    )
    args.raw_aux_loss_weight = (
        0.08
        if dataset_name_normalized == "ieee" and multimodal_aux_enabled
        else 0.03
        if dataset_name_normalized == "elliptic" and multimodal_aux_enabled
        else 0.06
        if multimodal_aux_enabled
        else 0.0
    )
    fusion_profile = _dataset_multimodal_fusion_profile(dataset_name)
    graph_anchor_profile = _dataset_graph_anchor_profile(dataset_name)
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
    elif multimodal_aux_enabled:
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
    args.fusion_variant = (
        str(fusion_variant_override).strip().lower()
        if multimodal_aux_enabled and fusion_variant_override is not None
        else (
            "graph_dominant_residual"
            if multimodal_aux_enabled and dataset_name_normalized == "elliptic"
            else "tri_stream_gate"
            if multimodal_aux_enabled
            else "single_branch"
        )
    )
    args.modality_dropout_prob = (
        float(modality_dropout_prob_override)
        if multimodal_aux_enabled and modality_dropout_prob_override is not None
        else (0.10 if multimodal_aux_enabled else 0.0)
    )
    args.graph_learning_rate_scale = (
        float(graph_learning_rate_scale_override)
        if graph_learning_rate_scale_override is not None
        else 1.0
    )
    args.sequence_learning_rate_scale = (
        float(sequence_learning_rate_scale_override)
        if sequence_learning_rate_scale_override is not None
        else (0.85 if multimodal_aux_enabled else 1.0)
    )
    args.fusion_learning_rate_scale = (
        float(fusion_learning_rate_scale_override)
        if fusion_learning_rate_scale_override is not None
        else (1.10 if multimodal_aux_enabled else 1.0)
    )
    args.graph_follow_learning_rate_scale = (
        float(graph_follow_learning_rate_scale_override)
        if graph_follow_learning_rate_scale_override is not None
        else (0.18 if multimodal_aux_enabled else 1.0)
    )
    schedule_rounds = int(schedule_total_rounds) if schedule_total_rounds is not None else int(federated_rounds)
    if multimodal_aux_enabled and dataset_name_normalized in {"ieee", "elliptic"}:
        default_graph_warmup_rounds = 2
        default_fusion_bootstrap_rounds = 4
    else:
        default_graph_warmup_rounds = int(max(round(schedule_rounds * 0.24), 2)) if multimodal_aux_enabled else 0
        default_fusion_bootstrap_rounds = int(max(round(schedule_rounds * 0.32), 4)) if multimodal_aux_enabled else 0
    if multimodal_aux_enabled and schedule_rounds > 0:
        max_prefinetune_rounds = max(int(schedule_rounds) - 1, 0)
        total_prefinetune_rounds = default_graph_warmup_rounds + default_fusion_bootstrap_rounds
        if total_prefinetune_rounds > max_prefinetune_rounds:
            graph_ratio = float(default_graph_warmup_rounds / max(total_prefinetune_rounds, 1))
            default_graph_warmup_rounds = int(round(max_prefinetune_rounds * graph_ratio))
            default_graph_warmup_rounds = int(np.clip(default_graph_warmup_rounds, 0, max_prefinetune_rounds))
            default_fusion_bootstrap_rounds = max_prefinetune_rounds - default_graph_warmup_rounds
    args.graph_warmup_rounds = (
        max(int(graph_warmup_rounds_override), 0)
        if graph_warmup_rounds_override is not None
        else default_graph_warmup_rounds
    )
    args.fusion_bootstrap_rounds = (
        max(int(fusion_bootstrap_rounds_override), 0)
        if fusion_bootstrap_rounds_override is not None
        else default_fusion_bootstrap_rounds
    )
    args.fusion_bootstrap_train_graph = (
        True
        if dataset_name_normalized in {"ieee", "elliptic"} and multimodal_aux_enabled
        else bool(multimodal_aux_enabled)
    )
    requested_preferred_eval_branch = str(preferred_eval_branch_override or "").strip().lower()
    if isinstance(eval_branch_priority_override, str):
        requested_eval_branch_priority = [
            item.strip().lower()
            for item in eval_branch_priority_override.split(",")
            if str(item).strip()
        ]
    elif isinstance(eval_branch_priority_override, (list, tuple)):
        requested_eval_branch_priority = [
            str(item).strip().lower()
            for item in eval_branch_priority_override
            if str(item).strip()
        ]
    else:
        requested_eval_branch_priority = []
    args.requested_preferred_eval_branch = requested_preferred_eval_branch
    args.requested_eval_branch_priority = list(requested_eval_branch_priority)
    args.schedule_total_rounds = int(schedule_rounds)
    args.active_training_stage = "graph_warmup" if args.graph_warmup_rounds > 0 else "joint_finetune"
    normalized_epoch_metric_recompute_mode = str(
        epoch_metric_recompute_mode
        if epoch_metric_recompute_mode is not None
        else ("last_local_epoch_only" if str(dataset_name).lower() == "ieee" else "all_local_epochs")
    ).strip().lower()
    if normalized_epoch_metric_recompute_mode not in {"all_local_epochs", "last_local_epoch_only", "disabled"}:
        normalized_epoch_metric_recompute_mode = "last_local_epoch_only" if str(dataset_name).lower() == "ieee" else "all_local_epochs"
    args.epoch_metric_recompute_mode = normalized_epoch_metric_recompute_mode
    args.requested_classification_loss = str(classification_loss).lower()
    args.classification_loss = (
        "weighted_bce_auc"
        if dataset_name_normalized in {"ieee", "elliptic"} and args.requested_classification_loss == "cb_focal"
        else args.requested_classification_loss
    )
    args.focal_gamma = focal_gamma
    args.class_balance_beta = class_balance_beta
    args.ranking_loss_weight = (
        0.35
        if dataset_name_normalized == "ieee"
        else 0.20
        if dataset_name_normalized == "elliptic"
        else 0.0
    )
    args.ranking_max_pairs = 4096
    args.tabular_teacher_distill_weight = 0.12 if dataset_name_normalized == "ieee" and multimodal_aux_enabled else 0.0
    args.tabular_teacher_temperature = 1.0
    args.label_fraction = float(label_fraction)
    scarcity_profile = _resolve_label_scarcity_profile(dataset_name, args.label_fraction)
    if bool(pure_label_fraction):
        scarcity_profile = _resolve_label_scarcity_profile(dataset_name, 1.0)
        scarcity_profile = dict(scarcity_profile)
        scarcity_profile["name"] = "pure_label_fraction"
        scarcity_profile["dataset"] = str(dataset_name)
        scarcity_profile["label_fraction"] = float(args.label_fraction)
    args.label_scarcity_profile = str(scarcity_profile["name"])
    args.label_scarcity_profile_settings = scarcity_profile
    args.pure_label_fraction = bool(pure_label_fraction)
    args.requested_pseudo_label_threshold = float(pseudo_label_threshold)
    args.requested_pseudo_label_weight = float(pseudo_label_weight)
    args.requested_pseudo_label_novelty_threshold = float(pseudo_label_novelty_threshold)
    args.requested_consistency_weight = float(consistency_weight)
    if bool(pure_label_fraction):
        args.pseudo_label_threshold = 0.0
        args.pseudo_label_min_threshold = 0.0
        args.pseudo_label_top_fraction = 0.0
        args.pseudo_label_weight = 0.0
        args.pseudo_label_novelty_threshold = float("inf")
        args.teacher_ema_decay = 0.0
        args.teacher_temperature = 1.0
        args.pseudo_warmup_rounds = 0
        args.pseudo_ramp_rounds = 0
        args.open_set_novelty_threshold = 0.0
        args.open_set_loss_weight = 0.0
    else:
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
    if bool(pure_label_fraction):
        args.active_learning_budget_per_round = 0
        args.active_learning_delay_rounds = 0
        args.consistency_weight = 0.0
        args.active_learning_novelty_weight = 0.0
        args.active_learning_diversity_weight = 0.0
    else:
        args.active_learning_budget_per_round = max(args.requested_active_learning_budget_per_round, 0)
        args.active_learning_delay_rounds = max(args.requested_active_learning_delay_rounds, 0)
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
    if not bool(pure_label_fraction):
        if teacher_ema_decay_override is not None:
            args.teacher_ema_decay = float(max(teacher_ema_decay_override, 0.0))
        if pseudo_warmup_rounds_override is not None:
            args.pseudo_warmup_rounds = max(int(pseudo_warmup_rounds_override), 0)
        if pseudo_ramp_rounds_override is not None:
            args.pseudo_ramp_rounds = max(int(pseudo_ramp_rounds_override), 0)
        if open_set_novelty_threshold_override is not None:
            args.open_set_novelty_threshold = float(max(open_set_novelty_threshold_override, 0.0))
        if open_set_loss_weight_override is not None:
            args.open_set_loss_weight = float(max(open_set_loss_weight_override, 0.0))
    args.prototype_loss_weight = (
        float(max(prototype_loss_weight_override, 0.0))
        if multimodal_aux_enabled and prototype_loss_weight_override is not None
        else (0.08 if multimodal_aux_enabled else 0.0)
    )
    args.shared_private_loss_weight = (
        float(max(shared_private_loss_weight_override, 0.0))
        if multimodal_aux_enabled and shared_private_loss_weight_override is not None
        else (
            0.05
            if multimodal_aux_enabled and str(args.fusion_variant).lower() == "shared_private_prototype"
            else 0.0
        )
    )
    args.context_alignment_loss_weight = (
        float(max(context_alignment_loss_weight_override, 0.0))
        if multimodal_aux_enabled and context_alignment_loss_weight_override is not None
        else 0.0
    )
    args.uncertainty_loss_weight = (
        float(max(uncertainty_loss_weight_override, 0.0))
        if multimodal_aux_enabled and uncertainty_loss_weight_override is not None
        else (0.04 if multimodal_aux_enabled else 0.0)
    )
    args.graph_anchor_loss_weight = (
        float(max(graph_anchor_loss_weight_override, 0.0))
        if multimodal_aux_enabled and graph_anchor_loss_weight_override is not None
        else (float(graph_anchor_profile["graph_anchor_loss_weight"]) if multimodal_aux_enabled else 0.0)
    )
    args.graph_anchor_temperature = (
        float(max(graph_anchor_temperature_override, 1e-6))
        if multimodal_aux_enabled and graph_anchor_temperature_override is not None
        else (float(graph_anchor_profile["graph_anchor_temperature"]) if multimodal_aux_enabled else 1.0)
    )
    args.legacy_fusion_only = bool(legacy_fusion_only_override) if legacy_fusion_only_override is not None else False
    args.graph_teacher_checkpoint_path = _normalize_resume_identity_path(graph_teacher_checkpoint_path)
    args.graph_teacher_distill_weight = (
        float(max(graph_teacher_distill_weight, 0.0)) if multimodal_aux_enabled else 0.0
    )
    args.graph_teacher_temperature = float(max(graph_teacher_temperature, 1e-6)) if multimodal_aux_enabled else 1.0
    if dataset_name_normalized == "elliptic":
        # Keep the Elliptic mainline focused on the light, direct path:
        # weak-graph / strong-temporal fusion + direct branch selection.
        # Do not silently re-enable heavier teacher/prototype/shared-private
        # families through low-label profiles or caller overrides.
        args.teacher_ema_decay = 0.0
        args.graph_teacher_distill_weight = 0.0
        args.tabular_teacher_distill_weight = 0.0
        args.prototype_loss_weight = 0.0
        args.shared_private_loss_weight = 0.0
    args.active_learning_candidate_pool_scale = max(int(active_learning_candidate_pool_scale), 1)
    args.fedprox_mu = float(fedprox_mu)
    args.dp_noise_std = float(dp_noise_std)
    args.requested_planner_mode = str(planner_mode).lower()
    args.requested_disable_federated = bool(disable_federated)
    args.fixed_precision_target = float(fixed_precision_target)
    args.planner_mode = str(runtime_policy["effective_planner_mode"]).lower()
    args.disable_gnn = bool(disable_gnn)
    args.disable_transformer = bool(disable_transformer)
    args.disable_federated = bool(runtime_policy["effective_disable_federated"])
    args.disable_relation_sequence_encoder = bool(disable_transformer or disable_relation_sequence_encoder)
    args.disable_event_transformer_encoder = bool(disable_transformer or disable_event_transformer_encoder)
    args.disable_temporal_context_encoder = bool(disable_transformer or disable_temporal_context_encoder)
    args.disable_graph_temporal_fusion = bool(
        disable_gnn or disable_transformer or disable_graph_temporal_fusion
    )
    args.force_disable_wavelet_lite = bool(disable_transformer or force_disable_wavelet_lite)
    args.force_disable_utg_lite = bool(disable_transformer or force_disable_utg_lite)
    args.force_disable_coassociation = bool(disable_gnn or force_disable_coassociation)
    args.force_disable_diffusion_residual = bool(disable_gnn or force_disable_diffusion_residual)
    args.gnn_enabled = not args.disable_gnn
    args.transformer_enabled = not args.disable_transformer
    args.federated_enabled = not args.disable_federated
    args.drl_enabled = args.planner_mode == "rl"
    args.splitgnn_runtime_policy = bool(runtime_policy["splitgnn_policy_active"])
    args.runtime_policy_notes = list(runtime_policy["notes"])
    if args.splitgnn_runtime_policy:
        args.ablation_mode = _resolve_structure_only_ablation_mode(
            disable_gnn=args.disable_gnn,
            disable_transformer=args.disable_transformer,
        )
    else:
        args.ablation_mode = _resolve_ablation_mode(
            disable_gnn=args.disable_gnn,
            disable_transformer=args.disable_transformer,
            disable_federated=args.disable_federated,
            planner_mode=args.planner_mode,
        )
    args.active_learning_feedback_path = _normalize_resume_identity_path(active_learning_feedback_path)
    args.federated_rounds = federated_rounds
    args.local_epochs = local_epochs
    args.num_clients = num_clients
    args.requested_num_clients = num_clients
    args.client_hops = client_hops
    args.result_root = str(_resolve_result_root(result_root))
    args.result_path = str(Path(args.result_root) / dataset_name)
    if early_stop is None:
        args.early_stop = int(getattr(args, "early_stop", 0))
    else:
        args.early_stop = max(int(early_stop), 0)
    args.test_every = max(int(test_every), 0)
    return args


def _load_hybrid_dataset_bundle(
    *,
    dataset_name: str,
    args: SimpleNamespace,
    effective_num_clients: int,
    client_hops: int,
    label_fraction: float,
) -> DatasetBundle:
    return load_registered_dataset_bundle(
        dataset_name=dataset_name,
        args=args,
        effective_num_clients=effective_num_clients,
        client_hops=client_hops,
        label_fraction=label_fraction,
    )


def _build_round_observation(
    bundle: DatasetBundle,
    round_index: int,
    total_rounds: int,
    metrics_history: List[dict],
) -> np.ndarray:
    """Build the controller observation vector for the current round."""
    train_mask = bundle.graph.ndata["train_mask"].bool()
    supervised_mask = bundle.graph.ndata["train_supervised_mask"].bool() if "train_supervised_mask" in bundle.graph.ndata else train_mask
    visible_train_labels = bundle.graph.ndata["label"][supervised_mask & train_mask]
    if visible_train_labels.numel() == 0:
        imbalance_signal = 0.0
    else:
        fraud_ratio = float(visible_train_labels.float().mean().item())
        imbalance_signal = np.clip(2 * fraud_ratio - 1, -1, 1)
    round_progress = 10.0 * (round_index + 1) / max(total_rounds, 1)
    relation_complexity = min(10.0, float(len(bundle.relation_order) * 2))
    if metrics_history:
        validation_pressure = min(10.0, 10.0 * (1.0 - metrics_history[-1]["valid_auc"]))
        loss_pressure = min(10.0, metrics_history[-1]["mean_local_loss"] * 10.0)
    else:
        validation_pressure = 5.0
        loss_pressure = 5.0
    return np.array(
        [imbalance_signal, round_progress, relation_complexity, validation_pressure, loss_pressure],
        dtype=np.float32,
    )


def _client_participation_counts(bundle: DatasetBundle, metrics_history: List[dict]) -> Dict[int, int]:
    """Count how often each client has been selected recently."""
    counts = {client.client_id: 0 for client in bundle.clients}
    for round_metrics in metrics_history:
        for client_id in round_metrics.get("selected_clients", []):
            if client_id in counts:
                counts[client_id] += 1
    return counts


def _stochastic_round(value: float, minimum: int, maximum: int) -> int:
    """Round a value to a bounded integer deterministically."""
    if maximum <= minimum:
        return int(minimum)
    clipped = float(np.clip(value, minimum, maximum))
    return int(np.clip(int(np.rint(clipped)), minimum, maximum))


def _deterministic_support_requirements(
    profile: dict,
    metrics_history: List[dict],
    stability_flags: dict,
    num_clients_total: int,
) -> tuple[int, int]:
    """Compute clean deterministic guardrails for client count and local epochs."""
    minimum_selected_clients = min(2, num_clients_total)
    minimum_local_epochs = int(profile.get("deterministic_min_local_epochs", 1))
    if not metrics_history:
        return minimum_selected_clients, minimum_local_epochs

    last_metrics = metrics_history[-1]
    recall = float(last_metrics.get("valid_recall", 1.0))
    positive_rate = float(last_metrics.get("valid_positive_rate", 1.0))
    valid_auc = float(last_metrics.get("valid_auc", 1.0))

    low_recall = recall < float(profile.get("recall_guard_floor", -1.0))
    low_positive_rate = positive_rate < float(profile.get("positive_rate_guard_floor", -1.0))
    weak_auc = valid_auc < float(profile.get("weak_auc_guard_floor", -1.0))
    need_extra_support = bool(stability_flags["auc_plateau"]) or low_recall or low_positive_rate or weak_auc
    sharp_drop = bool(stability_flags.get("sharp_auc_drop", False))

    required_selected_clients = minimum_selected_clients
    if need_extra_support:
        required_selected_clients = max(
            required_selected_clients,
            min(int(profile.get("weak_metric_min_selected_clients", minimum_selected_clients)), num_clients_total),
        )

    required_local_epochs = minimum_local_epochs
    if need_extra_support and not sharp_drop:
        required_local_epochs = max(
            required_local_epochs,
            int(profile.get("plateau_min_local_epochs", required_local_epochs)),
        )
    return required_selected_clients, required_local_epochs


def _deterministic_planner_action(
    bundle: DatasetBundle,
    round_index: int,
    total_rounds: int,
    metrics_history: List[dict],
) -> np.ndarray:
    """Rule-based scheduler aligned with fraud metrics."""
    num_clients_total = max(len(bundle.clients), 1)
    stability_flags = _round_stability_flags(bundle.name, metrics_history)
    progress = (round_index + 1) / max(total_rounds, 1)
    profile = _dataset_scheduler_profile(bundle.name)

    base_ratio = 2.0 / num_clients_total if num_clients_total <= 3 else 0.6
    selected_ratio = float(np.clip(base_ratio, 0.5, 1.0))
    if stability_flags["auc_plateau"]:
        selected_ratio = min(1.0, selected_ratio + 0.15)
    if stability_flags["probability_collapse"]:
        selected_ratio = 1.0
    if stability_flags.get("sharp_auc_drop", False):
        selected_ratio = min(1.0, selected_ratio + 0.20)

    if progress < 0.33:
        local_epoch_scale = float(profile["startup_epoch_scale"])
    elif progress < 0.66:
        local_epoch_scale = float(profile["mid_epoch_scale"])
    else:
        local_epoch_scale = float(profile["late_epoch_scale"])
    if stability_flags["auc_plateau"]:
        local_epoch_scale = min(1.0, local_epoch_scale + float(profile["plateau_epoch_boost"]))
    if stability_flags["probability_collapse"]:
        local_epoch_scale = min(local_epoch_scale, float(profile["collapse_epoch_cap"]))
    if stability_flags.get("sharp_auc_drop", False):
        local_epoch_scale = min(local_epoch_scale, float(profile.get("sharp_drop_epoch_scale", local_epoch_scale)))

    grad_clip_target = 1.0 + 0.05 * min(stability_flags["stagnation_rounds"], 6)
    if stability_flags["probability_collapse"]:
        grad_clip_target = max(grad_clip_target, 1.25)
    if stability_flags.get("sharp_auc_drop", False):
        grad_clip_target = max(grad_clip_target, 1.15)
    grad_clip_action = float(np.clip((grad_clip_target - 0.5) * 10.0, 0.0, 10.0))

    if num_clients_total > 1:
        selection_phase = ((round_index % num_clients_total) + 0.5) / num_clients_total
    else:
        selection_phase = 0.5
    phase_action = float(np.clip(selection_phase * 2.0 - 1.0, -1.0, 1.0))
    return np.array([phase_action, selected_ratio, local_epoch_scale, grad_clip_action], dtype=np.float32)


def _plan_round(
    rl_models: List[FRModel] | None,
    planner_mode: str,
    bundle: DatasetBundle,
    round_index: int,
    total_rounds: int,
    base_local_epochs: int,
    base_edge_loss_weight: float,
    metrics_history: List[dict],
) -> dict:
    """Plan the next federated round from RL or deterministic control."""
    observation = _build_round_observation(bundle, round_index, total_rounds, metrics_history)
    normalized_planner_mode = str(planner_mode).lower()
    stability_flags = _round_stability_flags(bundle.name, metrics_history)
    progress = (round_index + 1) / max(total_rounds, 1)
    profile = _dataset_scheduler_profile(bundle.name)
    deterministic_action = _deterministic_planner_action(
        bundle=bundle,
        round_index=round_index,
        total_rounds=total_rounds,
        metrics_history=metrics_history,
    )

    if normalized_planner_mode == "rl":
        if not rl_models:
            raise ValueError("planner_mode='rl' requires initialized RL models.")
        rl_actions = []
        for model in rl_models:
            action, _ = model.predict(observation, deterministic=True)
            rl_actions.append(np.asarray(action, dtype=np.float32))
        mean_action = np.mean(np.stack(rl_actions, axis=0), axis=0)
        if abs(float(mean_action[0])) >= 0.98:
            mean_action[0] = deterministic_action[0]
        mean_action[1:] = 0.55 * mean_action[1:] + 0.45 * deterministic_action[1:]
        if stability_flags["probability_collapse"]:
            mean_action[1:] = 0.25 * mean_action[1:] + 0.75 * deterministic_action[1:]
    elif normalized_planner_mode == "deterministic":
        mean_action = deterministic_action
    else:
        raise ValueError(f"Unsupported planner mode: {planner_mode}")

    selection_phase = float(np.clip((mean_action[0] + 1.0) / 2.0, 0.0, 0.999))
    selected_ratio = float(np.clip(mean_action[1], 1e-3, 1.0))
    local_epoch_scale = float(np.clip(mean_action[2], 1e-3, 1.0))
    grad_clip = float(np.clip(0.5 + mean_action[3] / 10.0, 0.5, 2.0))
    num_clients_total = max(len(bundle.clients), 1)

    if metrics_history:
        previous_metrics = metrics_history[-1]
        previous_selected_ratio = float(
            previous_metrics.get("actual_selected_ratio", previous_metrics.get("selected_ratio", 0.0))
        )
        previous_local_epochs = float(previous_metrics.get("local_epochs", base_local_epochs))
        previous_grad_clip = float(previous_metrics.get("grad_clip", 1.0))
        max_local_epochs_anchor = max(
            int(round(base_local_epochs * float(profile["max_local_epoch_multiplier"]))),
            base_local_epochs + 2,
            4,
        )
        previous_epoch_scale = float(np.clip(previous_local_epochs / max(max_local_epochs_anchor, 1), 0.0, 1.0))
        selected_ratio = float(0.80 * selected_ratio + 0.20 * previous_selected_ratio)
        local_epoch_scale = float(0.80 * local_epoch_scale + 0.20 * previous_epoch_scale)
        grad_clip = float(0.85 * grad_clip + 0.15 * previous_grad_clip)

    recent_history = metrics_history[-3:]
    recent_selected_counts = [len(item.get("selected_clients", [])) for item in recent_history]
    recent_selected_ratios = [
        float(item.get("actual_selected_ratio", item.get("selected_ratio", 0.0)))
        for item in recent_history
    ]
    recent_selection_phases = [float(item.get("selection_phase", 0.0)) for item in recent_history]
    planner_saturated = (
        len(recent_history) >= 2
        and len(set(recent_selected_counts)) == 1
        and (max(recent_selected_ratios) - min(recent_selected_ratios) < 1e-6)
        and (max(recent_selection_phases) - min(recent_selection_phases) < 1e-6)
        and (
            recent_selected_ratios[-1] >= 0.99
            or recent_selection_phases[-1] <= 0.01
            or recent_selection_phases[-1] >= 0.99
        )
    )

    if not metrics_history:
        startup_ratio_floor = min(0.75, max(2.0 / max(num_clients_total, 1), 0.40))
        selected_ratio = float(np.clip(selected_ratio, startup_ratio_floor, 0.85))
        selection_phase = float(np.clip(selection_phase, 0.15, 0.85))
        startup_epoch_scale = float(profile["startup_epoch_scale"])
        local_epoch_scale = float(
            np.clip(
                local_epoch_scale,
                max(0.14, startup_epoch_scale - 0.06),
                min(0.75, startup_epoch_scale + 0.06),
            )
        )
        grad_clip = float(np.clip(grad_clip, 0.95, 1.10))

    if num_clients_total == 3 and (selected_ratio >= 0.95 or planner_saturated):
        selected_ratio = min(selected_ratio, 0.80)
    if num_clients_total > 1 and (
        selection_phase <= 0.01 or selection_phase >= 0.99 or planner_saturated
    ):
        selection_phase = float(((round_index % num_clients_total) + 0.5) / num_clients_total)

    if stability_flags["auc_plateau"]:
        local_epoch_scale = min(1.0, local_epoch_scale + 0.05)
    if stability_flags["probability_collapse"]:
        selected_ratio = float(np.clip(selected_ratio, 0.65, 0.85))
        collapse_epoch_cap = float(profile["collapse_epoch_cap"])
        if stability_flags["severe_probability_collapse"]:
            collapse_epoch_cap = min(collapse_epoch_cap, 0.35)
        local_epoch_scale = min(local_epoch_scale, collapse_epoch_cap)
        grad_clip = min(1.5, max(grad_clip, 1.0) + 0.02)

    minimum_selected = min(2, num_clients_total)
    desired_selected = selected_ratio * num_clients_total
    selected_count = _stochastic_round(desired_selected, minimum_selected, num_clients_total)
    if num_clients_total == 3 and (planner_saturated or not metrics_history) and not stability_flags["probability_collapse"]:
        selected_count = min(selected_count, 2)

    max_local_epochs = max(
        int(round(base_local_epochs * float(profile["max_local_epoch_multiplier"]))),
        base_local_epochs + 2,
        4,
    )
    if stability_flags["auc_plateau"] and not stability_flags["probability_collapse"]:
        max_local_epochs = max(max_local_epochs, base_local_epochs + 3, 5)
    local_epochs = int(
        np.clip(np.round(local_epoch_scale * (max_local_epochs - 1)) + 1, 1, max_local_epochs)
    )

    if normalized_planner_mode == "deterministic":
        required_selected_clients, required_local_epochs = _deterministic_support_requirements(
            profile=profile,
            metrics_history=metrics_history,
            stability_flags=stability_flags,
            num_clients_total=num_clients_total,
        )
        selected_count = int(np.clip(max(selected_count, required_selected_clients), minimum_selected, num_clients_total))
        local_epochs = int(np.clip(max(local_epochs, required_local_epochs), 1, max_local_epochs))
        # Deterministic mode is our stable baseline: keep optimizer strength and
        # edge-loss coefficient fixed instead of layering in adaptive shrinkage.
        edge_loss_weight = float(profile.get("deterministic_edge_loss_weight", base_edge_loss_weight))
        learning_rate = float(getattr(bundle, "base_lr", 1e-3)) * float(profile.get("deterministic_lr_scale", 1.0))
    else:
        edge_loss_weight = _adaptive_edge_loss_weight(
            dataset_name=bundle.name,
            base_edge_loss_weight=base_edge_loss_weight,
            metrics_history=metrics_history,
            stability_flags=stability_flags,
            progress=progress,
        )
        learning_rate = _adaptive_learning_rate(
            dataset_name=bundle.name,
            base_learning_rate=float(getattr(bundle, "base_lr", 1e-3)),
            metrics_history=metrics_history,
            stability_flags=stability_flags,
            progress=progress,
        )

    anchor_index = (round_index + int(np.floor(selection_phase * num_clients_total))) % num_clients_total
    participation_counts = _client_participation_counts(bundle, metrics_history)
    ordered_clients = sorted(
        bundle.clients,
        key=lambda client: (
            participation_counts[client.client_id],
            (client.client_id - anchor_index) % num_clients_total,
        ),
    )
    selected_clients = ordered_clients[:selected_count]
    actual_selected_ratio = float(len(selected_clients) / max(num_clients_total, 1))
    return {
        "observation": observation.tolist(),
        "selected_clients": [client.client_id for client in selected_clients],
        "local_epochs": local_epochs,
        "grad_clip": grad_clip,
        "selection_phase": selection_phase,
        "selected_ratio": selected_ratio,
        "actual_selected_ratio": actual_selected_ratio,
        "raw_action": mean_action.tolist(),
        "planner_mode": normalized_planner_mode,
        "edge_loss_weight": edge_loss_weight,
        "learning_rate": learning_rate,
        "probability_collapse": bool(stability_flags["probability_collapse"]),
        "severe_probability_collapse": bool(stability_flags["severe_probability_collapse"]),
        "auc_plateau": bool(stability_flags["auc_plateau"]),
        "stagnation_rounds": int(stability_flags["stagnation_rounds"]),
        "sharp_auc_drop": bool(stability_flags.get("sharp_auc_drop", False)),
        "recent_auc_delta": float(stability_flags.get("recent_auc_delta", 0.0)),
    }

def _state_dict_average(state_dicts: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    """Average model state dicts locally for the single-mainline training path."""
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


def _evaluate_model(
    model: HybridFraudModel,
    graph: dgl.DGLHeteroGraph,
    split: str,
    device: torch.device,
    result_path: str = "",
    threshold: float | None = None,
) -> dict:
    return platform_evaluate_model(
        model,
        graph,
        split,
        device,
        result_path=result_path,
        threshold=threshold,
    )


def _eval_branch_priority(args: SimpleNamespace | None) -> tuple[str, ...]:
    raw_priority = getattr(args, "eval_branch_priority", None) if args is not None else None
    if isinstance(raw_priority, str):
        items = [part.strip().lower() for part in raw_priority.split(",")]
    elif isinstance(raw_priority, (list, tuple)):
        items = [str(part).strip().lower() for part in raw_priority]
    else:
        items = []
    normalized = tuple(item for item in items if item)
    return normalized if normalized else ("main",)


def _select_branch_name_from_metrics(
    branch_metrics: dict[str, dict],
    args: SimpleNamespace | None,
) -> str:
    if branch_metrics:
        for branch_name in _eval_branch_priority(args):
            if branch_name in branch_metrics:
                return branch_name
        if "main" in branch_metrics:
            return "main"
        return next(iter(branch_metrics.keys()))
    preferred = str(getattr(args, "preferred_eval_branch", "main") if args is not None else "main").strip().lower()
    return preferred or "main"


def _resolve_selected_branch_payload(
    branch_metrics: dict[str, dict],
    payload: dict[str, object],
    args: SimpleNamespace | None,
) -> tuple[str, dict]:
    selected_branch = str(payload.get("selected_branch", "") or "")
    if not selected_branch:
        selected_branch = _select_branch_name_from_metrics(branch_metrics, args)
    selected_metrics = dict(payload.get("selected_metrics", {}) or {})
    if not selected_metrics:
        selected_metrics = dict(branch_metrics.get(selected_branch, {}) or {})
    if not selected_metrics and "main" in branch_metrics:
        selected_branch = "main"
        selected_metrics = dict(branch_metrics.get("main", {}) or {})
    return selected_branch, selected_metrics


def _collect_model_diagnostics(
    model: HybridFraudModel,
    graph: dgl.DGLHeteroGraph,
    device: torch.device,
    *,
    splits: tuple[str, ...] = ("valid_mask", "test_mask"),
) -> dict:
    return platform_collect_model_diagnostics(
        model,
        graph,
        device,
        splits=splits,
    )


def evaluate_saved_hybrid_checkpoint(
    *,
    dataset_name: str,
    checkpoint_path: str | Path,
    summary_path: str | Path,
    device: str = DEFAULT_DEVICE_REQUEST,
    result_prefix: str | Path | None = None,
) -> dict:
    return platform_evaluate_saved_hybrid_checkpoint(
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        device=device,
        result_prefix=result_prefix,
    )


def _should_update_best_checkpoint(
    current_metrics: Dict[str, float],
    *,
    best_round: int,
    best_valid_auc: float,
    best_valid_gmean: float,
    best_valid_pr_auc: float,
    best_valid_recall_at_precision: float,
    best_valid_f1_macro: float,
    splitgnn_policy_active: bool = False,
) -> bool:
    """Select checkpoints by AUC first, then F1-macro/GMean, then auxiliaries."""
    current_auc = float(current_metrics.get("auc", 0.0))
    current_gmean = float(current_metrics.get("gmean", 0.0))
    current_pr_auc = float(current_metrics.get("pr_auc", 0.0))
    current_recall_at_precision = float(current_metrics.get("recall_at_precision", 0.0))
    current_f1_macro = float(current_metrics.get("f1_macro", 0.0))
    current_recall = float(current_metrics.get("recall", 0.0))

    # For SplitGNN structural ablations we avoid very conservative checkpoints
    # unless the ranking improvement is clearly meaningful.
    if splitgnn_policy_active and current_recall < 0.22 and current_recall_at_precision < 0.18:
        if best_round >= 0 and current_auc <= best_valid_auc + 0.02 and current_pr_auc <= best_valid_pr_auc + 0.02:
            return False
        if best_round < 0:
            return False

    if best_round < 0:
        return True

    auc_margin = 0.0005
    auc_tie_tolerance = 0.0005
    gmean_gain_margin = 0.001
    pr_auc_margin = 0.004
    recall_gain_margin = 0.02
    f1_gain_margin = 0.001

    if current_auc > best_valid_auc + auc_margin:
        return True
    if current_auc < best_valid_auc - auc_tie_tolerance:
        return False
    if current_f1_macro > best_valid_f1_macro + f1_gain_margin:
        return True
    if current_f1_macro < best_valid_f1_macro - f1_gain_margin:
        return False
    if current_gmean > best_valid_gmean + gmean_gain_margin:
        return True
    if current_gmean < best_valid_gmean - gmean_gain_margin:
        return False
    if current_pr_auc > best_valid_pr_auc + pr_auc_margin:
        return True
    if current_recall_at_precision > best_valid_recall_at_precision + recall_gain_margin:
        return True
    return False


def _recompute_bundle_class_stats(bundle: DatasetBundle) -> None:
    supervised_mask = bundle.graph.ndata["train_supervised_mask"].bool()
    labels = bundle.graph.ndata["label"][supervised_mask]
    if labels.numel() == 0:
        bundle.class_counts = torch.ones(2, dtype=torch.float32)
        bundle.class_weights = torch.ones(2, dtype=torch.float32)
        return
    counts = torch.bincount(labels, minlength=2).float().clamp(min=1.0)
    weights = counts.sum() / (counts * len(counts))
    bundle.class_counts = counts.cpu()
    bundle.class_weights = weights.cpu()


def _apply_label_scarcity_runtime_defaults(bundle: DatasetBundle, args: SimpleNamespace) -> None:
    profile = getattr(args, "label_scarcity_profile_settings", {}) or {}
    if float(getattr(args, "label_fraction", 1.0)) >= 0.999 or not profile:
        return

    supervised_count = int(bundle.graph.ndata["train_supervised_mask"].sum().item())
    requested_budget = int(
        getattr(
            args,
            "requested_active_learning_budget_per_round",
            getattr(args, "active_learning_budget_per_round", 0),
        )
    )
    if requested_budget < 0:
        budget = int(
            np.clip(
                round(supervised_count * float(profile.get("active_learning_budget_scale", 0.0))),
                int(profile.get("active_learning_budget_min", 0)),
                int(profile.get("active_learning_budget_max", 0)),
            )
        )
        args.active_learning_budget_per_round = max(budget, 0)

    requested_delay = int(
        getattr(
            args,
            "requested_active_learning_delay_rounds",
            getattr(args, "active_learning_delay_rounds", 0),
        )
    )
    if requested_delay < 0:
        args.active_learning_delay_rounds = max(int(profile.get("active_learning_delay_rounds", 0)), 0)


def _refresh_client_training_masks(bundle: DatasetBundle) -> None:
    node_type = bundle.node_type
    global_graph = bundle.graph
    global_supervised_mask = global_graph.nodes[node_type].data["train_supervised_mask"].bool()
    global_unlabeled_mask = global_graph.nodes[node_type].data["train_unlabeled_mask"].bool()
    if "homo" in global_graph.etypes:
        global_src, global_dst = global_graph.edges(etype="homo")
        global_graph.edges["homo"].data["train_mask"] = (
            global_supervised_mask[global_src] & global_supervised_mask[global_dst]
        ).bool()

    for client in bundle.clients:
        subgraph = client.subgraph
        node_data = subgraph.nodes[node_type].data
        if dgl.NID in node_data:
            global_node_ids = node_data[dgl.NID].long()
        else:
            global_node_ids = torch.arange(
                subgraph.num_nodes(node_type),
                dtype=torch.long,
            )
        owned_mask = subgraph.nodes[node_type].data["train_mask"].bool()
        subgraph.nodes[node_type].data["train_supervised_mask"] = global_supervised_mask[global_node_ids] & owned_mask
        subgraph.nodes[node_type].data["train_unlabeled_mask"] = global_unlabeled_mask[global_node_ids] & owned_mask
        if "homo" in subgraph.etypes:
            local_src, local_dst = subgraph.edges(etype="homo")
            local_supervised_mask = subgraph.nodes[node_type].data["train_supervised_mask"].bool()
            subgraph.edges["homo"].data["train_mask"] = (
                local_supervised_mask[local_src] & local_supervised_mask[local_dst]
            ).bool()
        client.train_nodes = int(owned_mask.sum().item())

    _sync_global_node_fields_to_clients(
        bundle,
        [
            "label",
            "label_scarcity_ratio",
            "pseudo_cycle_mask",
            "pseudo_cycle_label",
            "pseudo_cycle_weight",
            "pseudo_cycle_confidence",
            "pseudo_cycle_round",
        ],
    )
    _recompute_bundle_class_stats(bundle)


def _sync_global_node_fields_to_clients(bundle: DatasetBundle, field_names: Iterable[str]) -> None:
    node_type = bundle.node_type
    global_graph = bundle.graph
    global_node_data = global_graph.nodes[node_type].data
    available_fields = [str(name) for name in field_names if str(name) in global_node_data]
    if not available_fields:
        return

    for client in bundle.clients:
        subgraph = client.subgraph
        node_data = subgraph.nodes[node_type].data
        if dgl.NID in node_data:
            global_node_ids = node_data[dgl.NID].long()
        else:
            global_node_ids = torch.arange(subgraph.num_nodes(node_type), dtype=torch.long)
        for field_name in available_fields:
            node_data[field_name] = global_node_data[field_name][global_node_ids]


def _initialize_pseudo_cycle_cache_fields(bundle: DatasetBundle, *, reset: bool = False) -> None:
    node_type = bundle.node_type
    graph = bundle.graph
    node_data = graph.nodes[node_type].data
    num_nodes = int(graph.num_nodes(node_type))
    device = node_data["train_mask"].device
    label_dtype = node_data["label"].dtype if "label" in node_data else torch.long
    float_dtype = node_data["feature"].dtype if "feature" in node_data else torch.float32
    field_specs = {
        "pseudo_cycle_mask": torch.zeros(num_nodes, dtype=torch.bool, device=device),
        "pseudo_cycle_label": torch.full((num_nodes,), -1, dtype=label_dtype, device=device),
        "pseudo_cycle_weight": torch.zeros(num_nodes, dtype=float_dtype, device=device),
        "pseudo_cycle_confidence": torch.zeros(num_nodes, dtype=float_dtype, device=device),
        "pseudo_cycle_round": torch.full((num_nodes,), -1, dtype=torch.long, device=device),
    }
    for field_name, default_value in field_specs.items():
        needs_init = (
            reset
            or field_name not in node_data
            or int(node_data[field_name].shape[0]) != num_nodes
        )
        if needs_init:
            node_data[field_name] = default_value.clone()
    _sync_global_node_fields_to_clients(bundle, field_specs.keys())


def _cap_pseudo_cycle_cache_by_fraction(
    cache_payload: dict[str, torch.Tensor | dict[str, float]],
    *,
    unlabeled_mask: torch.Tensor,
    max_fraction: float,
) -> dict[str, torch.Tensor | dict[str, float]]:
    normalized_fraction = float(np.clip(max_fraction, 0.0, 1.0))
    allowed_nodes = int(unlabeled_mask.sum().item())
    if allowed_nodes <= 0:
        normalized_fraction = 0.0
    max_nodes = int(round(allowed_nodes * normalized_fraction)) if normalized_fraction > 0.0 else 0
    mask = cache_payload["mask"].bool() & unlabeled_mask.bool()
    selected_nodes = int(mask.sum().item())
    if selected_nodes <= max_nodes or max_nodes >= selected_nodes:
        cache_payload["mask"] = mask
        return cache_payload

    keep_mask = torch.zeros_like(mask)
    if max_nodes > 0:
        selected_index = torch.nonzero(mask, as_tuple=False).reshape(-1)
        selected_weights = cache_payload["weight"][selected_index].float()
        top_index = torch.topk(selected_weights, k=max_nodes, largest=True).indices
        keep_mask[selected_index[top_index]] = True
    cache_payload["mask"] = keep_mask
    cache_payload["label"] = torch.where(
        keep_mask,
        cache_payload["label"].long(),
        torch.full_like(cache_payload["label"].long(), -1),
    )
    cache_payload["weight"] = torch.where(
        keep_mask,
        cache_payload["weight"].float(),
        torch.zeros_like(cache_payload["weight"].float()),
    )
    cache_payload["confidence"] = torch.where(
        keep_mask,
        cache_payload["confidence"].float(),
        torch.zeros_like(cache_payload["confidence"].float()),
    )
    cache_payload["round"] = torch.where(
        keep_mask,
        cache_payload["round"].long(),
        torch.full_like(cache_payload["round"].long(), -1),
    )
    return cache_payload


def _maybe_refresh_pseudo_cycle_cache(
    *,
    bundle: DatasetBundle,
    global_model: HybridFraudModel,
    args: SimpleNamespace,
    current_round: int,
) -> dict[str, float | int | bool]:
    enabled = bool(getattr(args, "pseudo_cycle_refresh_enabled", False))
    node_type = bundle.node_type
    graph = bundle.graph
    if not enabled:
        return {
            "refreshed": False,
            "nodes": 0,
            "threshold": 0.0,
            "reliability_mean": 0.0,
            "support_rate": 0.0,
        }

    _initialize_pseudo_cycle_cache_fields(bundle)
    start_round = max(int(getattr(args, "pseudo_cycle_refresh_start_round", 0)), 0)
    refresh_interval = max(int(getattr(args, "pseudo_cycle_refresh_interval", 0)), 0)
    max_fraction = float(np.clip(getattr(args, "pseudo_cycle_refresh_max_fraction", 0.0), 0.0, 1.0))
    if current_round < start_round or refresh_interval <= 0 or max_fraction <= 0.0:
        _initialize_pseudo_cycle_cache_fields(bundle, reset=True)
        return {
            "refreshed": False,
            "nodes": 0,
            "threshold": 0.0,
            "reliability_mean": 0.0,
            "support_rate": 0.0,
        }

    if (current_round - start_round) % refresh_interval != 0:
        current_mask = graph.nodes[node_type].data["pseudo_cycle_mask"].bool()
        return {
            "refreshed": False,
            "nodes": int(current_mask.sum().item()),
            "threshold": 0.0,
            "reliability_mean": 0.0,
            "support_rate": float(current_mask.float().mean().item()) if current_mask.numel() > 0 else 0.0,
        }

    unlabeled_mask = graph.nodes[node_type].data["train_unlabeled_mask"].bool()
    if not unlabeled_mask.any():
        _initialize_pseudo_cycle_cache_fields(bundle, reset=True)
        return {
            "refreshed": True,
            "nodes": 0,
            "threshold": 0.0,
            "reliability_mean": 0.0,
            "support_rate": 0.0,
        }

    was_training = bool(global_model.training)
    device_graph = None
    try:
        global_model.eval()
        stage_timer = getattr(args, "stage_timer", None)
        track_context = (
            stage_timer.track("pseudo_cycle_refresh")
            if stage_timer is not None
            else nullcontext()
        )
        with track_context:
            device_graph = graph.to(args.device)
            with torch.no_grad():
                cache_payload = global_model.build_pseudo_cycle_cache(device_graph, current_round=current_round)
        cache_payload = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in cache_payload.items()
        }
        cache_payload = _cap_pseudo_cycle_cache_by_fraction(
            cache_payload,
            unlabeled_mask=unlabeled_mask.cpu(),
            max_fraction=max_fraction,
        )

        global_node_data = graph.nodes[node_type].data
        previous_mask = global_node_data["pseudo_cycle_mask"].bool().cpu()
        previous_label = global_node_data["pseudo_cycle_label"].long().cpu()
        previous_weight = global_node_data["pseudo_cycle_weight"].float().cpu()
        previous_confidence = global_node_data["pseudo_cycle_confidence"].float().cpu()
        previous_round = global_node_data["pseudo_cycle_round"].long().cpu()

        new_mask = cache_payload["mask"].bool() & unlabeled_mask.cpu()
        new_label = cache_payload["label"].long()
        new_weight = cache_payload["weight"].float().clamp(min=0.0, max=1.0)
        new_confidence = cache_payload["confidence"].float().clamp(min=0.0, max=1.0)
        new_round = cache_payload["round"].long()

        momentum = float(np.clip(getattr(args, "pseudo_cycle_refresh_momentum", 0.0), 0.0, 1.0))
        overlap_mask = previous_mask & new_mask
        agree_mask = overlap_mask & previous_label.eq(new_label)
        if momentum > 0.0 and agree_mask.any():
            new_weight = torch.where(
                agree_mask,
                momentum * previous_weight + (1.0 - momentum) * new_weight,
                new_weight,
            )
        conflict_mask = overlap_mask & ~previous_label.eq(new_label)
        prefer_previous = conflict_mask & previous_weight.gt(new_weight)
        if prefer_previous.any():
            new_label = torch.where(prefer_previous, previous_label, new_label)
            new_weight = torch.where(prefer_previous, previous_weight, new_weight)
            new_confidence = torch.where(prefer_previous, previous_confidence, new_confidence)
            new_round = torch.where(prefer_previous, previous_round, new_round)

        global_node_data["pseudo_cycle_mask"] = new_mask.to(device=graph.device)
        global_node_data["pseudo_cycle_label"] = torch.where(
            new_mask,
            new_label.clamp(min=0),
            torch.full_like(new_label, -1),
        ).to(device=graph.device)
        global_node_data["pseudo_cycle_weight"] = torch.where(
            new_mask,
            new_weight,
            torch.zeros_like(new_weight),
        ).to(device=graph.device)
        global_node_data["pseudo_cycle_confidence"] = torch.where(
            new_mask,
            new_confidence,
            torch.zeros_like(new_confidence),
        ).to(device=graph.device)
        global_node_data["pseudo_cycle_round"] = torch.where(
            new_mask,
            new_round,
            torch.full_like(new_round, -1),
        ).to(device=graph.device)
        _sync_global_node_fields_to_clients(
            bundle,
            [
                "pseudo_cycle_mask",
                "pseudo_cycle_label",
                "pseudo_cycle_weight",
                "pseudo_cycle_confidence",
                "pseudo_cycle_round",
            ],
        )
        refresh_stats = dict(cache_payload.get("stats", {}) or {})
        return {
            "refreshed": True,
            "nodes": int(new_mask.sum().item()),
            "threshold": float(refresh_stats.get("threshold", 0.0)),
            "reliability_mean": float(refresh_stats.get("reliability_mean", 0.0)),
            "support_rate": float(new_mask.float().mean().item()) if new_mask.numel() > 0 else 0.0,
        }
    finally:
        if was_training:
            global_model.train()
        if device_graph is not None:
            del device_graph
        _release_cuda_memory(args.device)


def _apply_active_learning_reveal(
    bundle: DatasetBundle,
    pending_feedback: List[dict],
    current_round: int,
) -> tuple[List[dict], List[dict]]:
    if not pending_feedback:
        return pending_feedback, []

    graph = bundle.graph
    train_mask = graph.ndata["train_mask"].bool()
    supervised_mask = graph.ndata["train_supervised_mask"].bool().clone()
    unlabeled_mask = graph.ndata["train_unlabeled_mask"].bool().clone()
    labels = graph.ndata["label"].clone()

    remaining_feedback = []
    revealed_feedback = []
    for item in pending_feedback:
        reveal_round = int(item.get("reveal_round", current_round + 1))
        node_id = int(item["node_id"])
        if reveal_round > current_round:
            remaining_feedback.append(item)
            continue
        if node_id < 0 or node_id >= graph.num_nodes(graph.ntypes[0]):
            continue
        if not bool(train_mask[node_id]) or not bool(unlabeled_mask[node_id]):
            continue
        labels[node_id] = int(item["label"])
        supervised_mask[node_id] = True
        unlabeled_mask[node_id] = False
        revealed_feedback.append(item)

    if revealed_feedback:
        graph.ndata["label"] = labels
        graph.ndata["train_supervised_mask"] = supervised_mask & train_mask
        graph.ndata["train_unlabeled_mask"] = unlabeled_mask & train_mask
        scarcity_ratio = float(graph.ndata["train_supervised_mask"].sum().item()) / max(
            float(train_mask.sum().item()), 1.0
        )
        graph.ndata["label_scarcity_ratio"] = torch.full(
            (graph.num_nodes(graph.ntypes[0]),),
            scarcity_ratio,
            dtype=torch.float32,
        )
        _refresh_client_training_masks(bundle)
    return remaining_feedback, revealed_feedback


def _select_active_learning_queries(
    model: HybridFraudModel,
    bundle: DatasetBundle,
    args: SimpleNamespace,
    threshold: float,
    current_round: int,
    pending_node_ids: set[int],
) -> List[dict]:
    budget = max(int(getattr(args, "active_learning_budget_per_round", 0)), 0)
    if budget <= 0:
        return []

    graph = bundle.graph.to(args.device)
    node_type = graph.ntypes[0]
    train_unlabeled_mask = graph.nodes[node_type].data["train_unlabeled_mask"].bool()
    if not train_unlabeled_mask.any():
        return []

    candidate_mask = train_unlabeled_mask.clone()
    if pending_node_ids:
        pending_tensor = torch.tensor(sorted(pending_node_ids), dtype=torch.long, device=graph.device)
        candidate_mask[pending_tensor] = False
    if not candidate_mask.any():
        return []

    model.eval()
    with torch.no_grad():
        logits, _, _, _, fused_embeddings = model.forward_with_details(graph)
        probs = torch.softmax(logits, dim=1)[:, 1]

    max_margin = max(float(threshold), 1.0 - float(threshold), 1e-6)
    uncertainty = 1.0 - torch.clamp(torch.abs(probs - float(threshold)) / max_margin, 0.0, 1.0)
    novelty_component = torch.zeros_like(uncertainty)
    supervised_mask = graph.nodes[node_type].data["train_supervised_mask"].bool()
    if supervised_mask.any():
        novelty_scores = _novelty_scores(
            reference_embeddings=fused_embeddings[supervised_mask].detach(),
            reference_labels=graph.ndata["label"][supervised_mask].detach(),
            query_embeddings=fused_embeddings.detach(),
        )
        if novelty_scores is not None:
            novelty_component = torch.sigmoid(novelty_scores)

    novelty_weight = float(np.clip(getattr(args, "active_learning_novelty_weight", 0.35), 0.0, 1.0))
    diversity_weight = float(np.clip(getattr(args, "active_learning_diversity_weight", 0.25), 0.0, 1.0))
    base_acquisition = (1.0 - novelty_weight) * uncertainty + novelty_weight * novelty_component
    candidate_ids = candidate_mask.nonzero(as_tuple=False).flatten()
    candidate_scores = base_acquisition[candidate_ids]
    if candidate_scores.numel() == 0:
        return []

    candidate_pool_scale = max(int(getattr(args, "active_learning_candidate_pool_scale", 4)), 1)
    candidate_pool_size = min(candidate_scores.numel(), max(budget * candidate_pool_scale, budget))
    candidate_order = torch.argsort(candidate_scores, descending=True)[:candidate_pool_size]
    candidate_pool_ids = candidate_ids[candidate_order]
    candidate_pool_embeddings = fused_embeddings[candidate_pool_ids].detach()
    candidate_pool_base_scores = base_acquisition[candidate_pool_ids]

    selected_positions: list[int] = []
    selected_nodes: list[int] = []
    diversity_scores: dict[int, float] = {}
    for _ in range(min(budget, candidate_pool_ids.numel())):
        candidate_diversities: list[tuple[int, float]] = []
        max_diversity_value = 1.0
        for position, _node_id_tensor in enumerate(candidate_pool_ids):
            if position in selected_positions:
                continue
            node_embedding = candidate_pool_embeddings[position].unsqueeze(0)
            if not selected_positions:
                diversity_value = 1.0
            else:
                chosen_embeddings = candidate_pool_embeddings[selected_positions]
                min_distance = torch.cdist(node_embedding.float(), chosen_embeddings.float(), p=2).min().item()
                diversity_value = float(min_distance)
            candidate_diversities.append((position, diversity_value))
            max_diversity_value = max(max_diversity_value, float(diversity_value))

        best_position = None
        best_score = None
        best_diversity = 0.0
        for position, diversity_value in candidate_diversities:
            normalized_diversity = float(np.clip(diversity_value / max(max_diversity_value, 1e-6), 0.0, 1.0))
            blended_score = (
                (1.0 - diversity_weight) * float(candidate_pool_base_scores[position].item())
                + diversity_weight * normalized_diversity
            )
            if best_score is None or blended_score > best_score:
                best_score = blended_score
                best_position = position
                best_diversity = normalized_diversity
        if best_position is None:
            break
        selected_positions.append(best_position)
        selected_node_id = int(candidate_pool_ids[best_position].item())
        selected_nodes.append(selected_node_id)
        diversity_scores[selected_node_id] = float(best_diversity)

    delay_rounds = max(int(getattr(args, "active_learning_delay_rounds", 0)), 0)
    labels = graph.ndata["label"]
    queries = []
    for node_id in selected_nodes:
        queries.append(
            {
                "node_id": int(node_id),
                "label": int(labels[node_id].item()),
                "uncertainty": float(uncertainty[node_id].item()),
                "novelty": float(novelty_component[node_id].item()),
                "diversity": float(diversity_scores.get(int(node_id), 0.0)),
                "score": float(
                    (1.0 - diversity_weight) * float(base_acquisition[node_id].item())
                    + diversity_weight * float(diversity_scores.get(int(node_id), 0.0))
                ),
                "requested_round": int(current_round),
                "reveal_round": int(current_round + delay_rounds + 1),
            }
        )
    return queries


def _ema_update_model(teacher_model: HybridFraudModel, student_model: HybridFraudModel, decay: float) -> None:
    platform_ema_update_model(teacher_model, student_model, decay)


def _local_train(
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
    return platform_local_train_round(
        global_model=global_model,
        graph_teacher_model=graph_teacher_model,
        subgraph=subgraph,
        global_state=global_state,
        class_weights=class_weights,
        class_counts=class_counts,
        args=args,
        current_round=current_round,
        local_epochs=local_epochs,
        grad_clip=grad_clip,
        edge_loss_weight=edge_loss_weight,
        learning_rate=learning_rate,
        fedprox_mu=fedprox_mu,
        dp_noise_std=dp_noise_std,
    )


def _train_one_dataset(
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
    amp_dtype: str,
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
    seq_hidden_dim: int,
    fusion_hidden_dim: int,
    planner_mode: str,
    early_stop: int | None,
    test_every: int,
    fixed_precision_target: float,
    resume_path: str,
    resume_round_offset: int,
    total_target_rounds: int | None,
    preload_history: list[dict[str, object]] | None,
    enable_tensorboard: bool,
    export_embedding_viz: bool,
    transformer_hidden_dim: int | None = None,
    transformer_num_layers: int = 1,
    sequence_batch_chunk_size: int | None = None,
    event_batch_chunk_size: int | None = None,
    transformer_activation_checkpointing: bool = True,
    active_learning_feedback_path: str = "",
    profile_ieee_full_gpu: bool = False,
    ieee_data_root: str = "",
    ieee_data_profile: str = "light_v1",
    ieee_loader_view: str = "hybrid",
    ieee_relation_profile: str = "core",
    ieee_feature_profile: str = "typed_256",
    ieee_history_len: int = 6,
    ieee_sampling_profile: str = "fraud_hardneg",
    ieee_max_transactions: int | None = None,
    ieee_time_bins: int = 24,
    ieee_relation_window_neighbors: int = 2,
    ieee_train_ratio: float = 0.70,
    ieee_valid_ratio: float = 0.15,
    ieee_full_compact_sequences: bool = True,
    ieee_sequence_feature_dim: int = 64,
    ieee_event_feature_dim: int = 64,
    ieee_build_light_cache_only: bool = False,
    ieee_rebuild_light_cache: bool = False,
    ieee_build_cache_only: bool = False,
    ieee_rebuild_cache: bool = False,
    ieee_skip_training: bool = False,
    amlsim_data_root: str = "",
    amlsim_train_ratio: float = 0.70,
    amlsim_valid_ratio: float = 0.15,
    amlsim_relation_window_neighbors: int = 4,
    amlsim_activity_bins: int = 8,
    amlsim_event_history_len: int = 12,
    amlsim_rebuild_cache: bool = False,
    amlsim_allow_sample_fallback: bool = False,
    amlsim_diffusion_residual_scale: float = 0.18,
    amlsim_pseudo_refresh_interval: int = 0,
    amlsim_pseudo_refresh_start_round: int = 0,
    amlsim_pseudo_refresh_momentum: float = 0.65,
    amlsim_pseudo_refresh_max_fraction: float = 0.0,
    amlsim_coassociation_loss_weight: float = 0.0,
    amlsim_wavelet_loss_weight: float = 0.0,
    amlsim_utg_align_loss_weight: float = 0.0,
    elliptic_data_root: str = "",
    elliptic_train_time_end: int = 34,
    elliptic_valid_time_end: int = 39,
    elliptic_history_len: int = 8,
    elliptic_sequence_topk: int = 8,
    elliptic_coassociation_topk: int = 3,
    elliptic_coassociation_time_window: int = 2,
    elliptic_use_unknown_ssl: bool = True,
    elliptic_rebuild_cache: bool = False,
    elliptic_pseudo_refresh_interval: int = 4,
    elliptic_pseudo_refresh_start_round: int = 4,
    elliptic_pseudo_refresh_momentum: float = 0.65,
    elliptic_pseudo_refresh_max_fraction: float = 0.10,
    elliptic_diffusion_residual_scale: float = 0.18,
    elliptic_coassociation_loss_weight: float = 0.05,
    elliptic_wavelet_loss_weight: float = 0.03,
    elliptic_utg_align_loss_weight: float = 0.04,
    ethereum_phishing_data_root: str = "",
    ethereum_phishing_max_users: int | None = None,
    ethereum_phishing_max_transactions: int | None = None,
    ethereum_phishing_force_preview: bool = False,
    ethereum_ponzi_data_root: str = "",
    ethereum_ponzi_negative_users_path: str = "",
    ethereum_ponzi_force_preview: bool = False,
    defi_rug_pull_data_root: str = "",
    defi_rug_pull_negative_users_path: str = "",
    defi_rug_pull_force_preview: bool = False,
    rl_models: List[FRModel] | None = None,
    seed: int | None = None,
    result_root: str = "",
    disable_gnn: bool = False,
    disable_transformer: bool = False,
    disable_federated: bool = False,
    disable_relation_sequence_encoder: bool = False,
    disable_event_transformer_encoder: bool = False,
    disable_temporal_context_encoder: bool = False,
    disable_graph_temporal_fusion: bool = False,
    force_disable_wavelet_lite: bool = False,
    force_disable_utg_lite: bool = False,
    force_disable_coassociation: bool = False,
    force_disable_diffusion_residual: bool = False,
    learning_rate_override: float | None = None,
    weight_decay_override: float | None = None,
    dropout_override: float | None = None,
    graph_aux_loss_weight_override: float | None = None,
    sequence_aux_loss_weight_override: float | None = None,
    graph_gate_logit_bias_override: float | None = None,
    eval_graph_gate_logit_bias_override: float | None = None,
    graph_residual_min_gate_override: float | None = None,
    sequence_residual_scale_override: float | None = None,
    preferred_eval_branch_override: str | None = None,
    eval_branch_priority_override: str | list[str] | tuple[str, ...] | None = None,
    fusion_variant_override: str | None = None,
    modality_dropout_prob_override: float | None = None,
    graph_learning_rate_scale_override: float | None = None,
    sequence_learning_rate_scale_override: float | None = None,
    fusion_learning_rate_scale_override: float | None = None,
    graph_follow_learning_rate_scale_override: float | None = None,
    graph_warmup_rounds_override: int | None = None,
    fusion_bootstrap_rounds_override: int | None = None,
    teacher_ema_decay_override: float | None = None,
    pseudo_warmup_rounds_override: int | None = None,
    pseudo_ramp_rounds_override: int | None = None,
    open_set_novelty_threshold_override: float | None = None,
    open_set_loss_weight_override: float | None = None,
    prototype_loss_weight_override: float | None = None,
    shared_private_loss_weight_override: float | None = None,
    context_alignment_loss_weight_override: float | None = None,
    uncertainty_loss_weight_override: float | None = None,
    graph_anchor_loss_weight_override: float | None = None,
    graph_anchor_temperature_override: float | None = None,
    graph_teacher_checkpoint_path: str = "",
    graph_teacher_distill_weight: float = 0.0,
    graph_teacher_temperature: float = 1.5,
    legacy_fusion_only_override: bool | None = None,
    skip_test_evaluation: bool = False,
    lightweight_valid_eval: bool = False,
    epoch_metric_recompute_mode: str | None = None,
    pure_label_fraction: bool = False,
) -> dict:
    """Train the active GNN + Transformer mainline."""
    # FL/RL has been archived to `legacy_federated_rl_backup/`; keep the public
    # API stable but force the active project runtime onto the structure-only
    # GNN + Transformer path.
    planner_mode, disable_federated = _force_mainline_gnn_transformer_mode(
        planner_mode=planner_mode,
        disable_federated=disable_federated,
    )
    resolved_result_root = _resolve_result_root(result_root)
    runtime_policy = resolve_splitgnn_runtime_policy(
        dataset_name=dataset_name,
        planner_mode=planner_mode,
        disable_federated=disable_federated,
    )
    effective_num_clients = 1 if bool(runtime_policy["effective_disable_federated"]) else num_clients
    args = _build_training_args(
        dataset_name=dataset_name,
        device=device,
        amp_dtype=amp_dtype,
        federated_rounds=federated_rounds,
        schedule_total_rounds=total_target_rounds,
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
        seq_hidden_dim=seq_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        early_stop=early_stop,
        planner_mode=planner_mode,
        test_every=test_every,
        fixed_precision_target=fixed_precision_target,
        transformer_hidden_dim=transformer_hidden_dim,
        transformer_num_layers=transformer_num_layers,
        sequence_batch_chunk_size=sequence_batch_chunk_size,
        event_batch_chunk_size=event_batch_chunk_size,
        transformer_activation_checkpointing=transformer_activation_checkpointing,
        active_learning_feedback_path=active_learning_feedback_path,
        seed=seed,
        result_root=str(resolved_result_root),
        disable_gnn=disable_gnn,
        disable_transformer=disable_transformer,
        disable_federated=disable_federated,
        disable_relation_sequence_encoder=disable_relation_sequence_encoder,
        disable_event_transformer_encoder=disable_event_transformer_encoder,
        disable_temporal_context_encoder=disable_temporal_context_encoder,
        disable_graph_temporal_fusion=disable_graph_temporal_fusion,
        force_disable_wavelet_lite=force_disable_wavelet_lite,
        force_disable_utg_lite=force_disable_utg_lite,
        force_disable_coassociation=force_disable_coassociation,
        force_disable_diffusion_residual=force_disable_diffusion_residual,
        learning_rate_override=learning_rate_override,
        weight_decay_override=weight_decay_override,
        dropout_override=dropout_override,
        graph_aux_loss_weight_override=graph_aux_loss_weight_override,
        sequence_aux_loss_weight_override=sequence_aux_loss_weight_override,
        graph_gate_logit_bias_override=graph_gate_logit_bias_override,
        eval_graph_gate_logit_bias_override=eval_graph_gate_logit_bias_override,
        graph_residual_min_gate_override=graph_residual_min_gate_override,
        sequence_residual_scale_override=sequence_residual_scale_override,
        preferred_eval_branch_override=preferred_eval_branch_override,
        eval_branch_priority_override=eval_branch_priority_override,
        fusion_variant_override=fusion_variant_override,
        modality_dropout_prob_override=modality_dropout_prob_override,
        graph_learning_rate_scale_override=graph_learning_rate_scale_override,
        sequence_learning_rate_scale_override=sequence_learning_rate_scale_override,
        fusion_learning_rate_scale_override=fusion_learning_rate_scale_override,
        graph_follow_learning_rate_scale_override=graph_follow_learning_rate_scale_override,
        graph_warmup_rounds_override=graph_warmup_rounds_override,
        fusion_bootstrap_rounds_override=fusion_bootstrap_rounds_override,
        teacher_ema_decay_override=teacher_ema_decay_override,
        pseudo_warmup_rounds_override=pseudo_warmup_rounds_override,
        pseudo_ramp_rounds_override=pseudo_ramp_rounds_override,
        open_set_novelty_threshold_override=open_set_novelty_threshold_override,
        open_set_loss_weight_override=open_set_loss_weight_override,
        prototype_loss_weight_override=prototype_loss_weight_override,
        shared_private_loss_weight_override=shared_private_loss_weight_override,
        context_alignment_loss_weight_override=context_alignment_loss_weight_override,
        uncertainty_loss_weight_override=uncertainty_loss_weight_override,
        graph_anchor_loss_weight_override=graph_anchor_loss_weight_override,
        graph_anchor_temperature_override=graph_anchor_temperature_override,
        graph_teacher_checkpoint_path=graph_teacher_checkpoint_path,
        graph_teacher_distill_weight=graph_teacher_distill_weight,
        graph_teacher_temperature=graph_teacher_temperature,
        legacy_fusion_only_override=legacy_fusion_only_override,
        epoch_metric_recompute_mode=epoch_metric_recompute_mode,
        pure_label_fraction=pure_label_fraction,
    )
    args.requested_num_clients = int(num_clients)
    args.requested_base_local_epochs = int(max(requested_base_local_epochs if requested_base_local_epochs is not None else local_epochs, 1))
    args.requested_extra_local_epochs = int(max(requested_extra_local_epochs if requested_extra_local_epochs is not None else 1, 1))
    args.skip_test_evaluation = bool(skip_test_evaluation)
    args.lightweight_valid_eval = bool(lightweight_valid_eval)
    args.profile_ieee_full_gpu = bool(profile_ieee_full_gpu)
    args.ieee_data_root = str(Path(ieee_data_root).expanduser().resolve()) if str(ieee_data_root).strip() else ""
    args.ieee_data_profile = str(ieee_data_profile or "light_v1")
    args.ieee_loader_view = str(ieee_loader_view or "hybrid")
    args.ieee_relation_profile = str(ieee_relation_profile or "core")
    args.ieee_feature_profile = str(ieee_feature_profile or "typed_256")
    args.ieee_history_len = int(ieee_history_len or 6)
    args.ieee_sampling_profile = str(ieee_sampling_profile or "fraud_hardneg")
    args.requested_ieee_max_transactions = None if ieee_max_transactions is None else int(ieee_max_transactions)
    args.ieee_max_transactions = None if ieee_max_transactions is None else int(ieee_max_transactions)
    args.ieee_time_bins = int(ieee_time_bins)
    args.ieee_relation_window_neighbors = int(ieee_relation_window_neighbors)
    args.ieee_train_ratio = float(ieee_train_ratio)
    args.ieee_valid_ratio = float(ieee_valid_ratio)
    args.ieee_full_compact_sequences = bool(ieee_full_compact_sequences)
    args.ieee_sequence_feature_dim = max(int(ieee_sequence_feature_dim), 1)
    args.ieee_event_feature_dim = max(int(ieee_event_feature_dim), 1)
    args.ieee_build_light_cache_only = bool(ieee_build_light_cache_only)
    args.ieee_rebuild_light_cache = bool(ieee_rebuild_light_cache)
    args.ieee_build_cache_only = bool(ieee_build_cache_only)
    args.ieee_rebuild_cache = bool(ieee_rebuild_cache)
    args.ieee_skip_training = bool(ieee_skip_training or ieee_build_cache_only or ieee_build_light_cache_only)
    _apply_main_ieee_full_gpu_profile(args, dataset_name)
    args.amlsim_data_root = str(
        Path(amlsim_data_root).expanduser().resolve()
    ) if str(amlsim_data_root).strip() else str(AMLSIM_DEFAULT_ROOT.resolve())
    args.amlsim_train_ratio = float(amlsim_train_ratio)
    args.amlsim_valid_ratio = float(amlsim_valid_ratio)
    args.amlsim_relation_window_neighbors = max(int(amlsim_relation_window_neighbors), 1)
    args.amlsim_activity_bins = max(int(amlsim_activity_bins), 1)
    args.amlsim_event_history_len = max(int(amlsim_event_history_len), 1)
    args.amlsim_rebuild_cache = bool(amlsim_rebuild_cache)
    args.amlsim_allow_sample_fallback = bool(amlsim_allow_sample_fallback)
    args.amlsim_diffusion_residual_scale = float(max(amlsim_diffusion_residual_scale, 0.0))
    args.amlsim_pseudo_refresh_interval = max(int(amlsim_pseudo_refresh_interval), 0)
    args.amlsim_pseudo_refresh_start_round = max(int(amlsim_pseudo_refresh_start_round), 0)
    args.amlsim_pseudo_refresh_momentum = float(np.clip(amlsim_pseudo_refresh_momentum, 0.0, 1.0))
    args.amlsim_pseudo_refresh_max_fraction = float(np.clip(amlsim_pseudo_refresh_max_fraction, 0.0, 1.0))
    args.amlsim_coassociation_loss_weight = float(max(amlsim_coassociation_loss_weight, 0.0))
    args.amlsim_wavelet_loss_weight = float(max(amlsim_wavelet_loss_weight, 0.0))
    args.amlsim_utg_align_loss_weight = float(max(amlsim_utg_align_loss_weight, 0.0))
    args.elliptic_data_root = str(
        Path(elliptic_data_root).expanduser().resolve()
    ) if str(elliptic_data_root).strip() else str(ELLIPTIC_DEFAULT_ROOT.resolve())
    args.elliptic_train_time_end = int(elliptic_train_time_end)
    args.elliptic_valid_time_end = int(elliptic_valid_time_end)
    args.elliptic_history_len = max(int(elliptic_history_len), 1)
    args.elliptic_sequence_topk = max(int(elliptic_sequence_topk), 1)
    args.elliptic_coassociation_topk = max(int(elliptic_coassociation_topk), 0)
    args.elliptic_coassociation_time_window = max(int(elliptic_coassociation_time_window), 0)
    args.elliptic_use_unknown_ssl = bool(elliptic_use_unknown_ssl)
    args.elliptic_rebuild_cache = bool(elliptic_rebuild_cache)
    args.elliptic_pseudo_refresh_interval = max(int(elliptic_pseudo_refresh_interval), 0)
    args.elliptic_pseudo_refresh_start_round = max(int(elliptic_pseudo_refresh_start_round), 0)
    args.elliptic_pseudo_refresh_momentum = float(np.clip(elliptic_pseudo_refresh_momentum, 0.0, 1.0))
    args.elliptic_pseudo_refresh_max_fraction = float(np.clip(elliptic_pseudo_refresh_max_fraction, 0.0, 1.0))
    args.elliptic_diffusion_residual_scale = float(max(elliptic_diffusion_residual_scale, 0.0))
    args.elliptic_coassociation_loss_weight = float(max(elliptic_coassociation_loss_weight, 0.0))
    args.elliptic_wavelet_loss_weight = float(max(elliptic_wavelet_loss_weight, 0.0))
    args.elliptic_utg_align_loss_weight = float(max(elliptic_utg_align_loss_weight, 0.0))
    dataset_name_normalized = str(dataset_name).lower()
    if dataset_name_normalized == "elliptic":
        args.preferred_eval_branch = "sequence_residual"
        args.eval_branch_priority = list(ELLIPTIC_EVAL_BRANCH_PRIORITY)
    else:
        args.preferred_eval_branch = "main"
        args.eval_branch_priority = ["main", "fusion", "graph_residual", "sequence_residual", "raw_branch"]
    requested_preferred_eval_branch = str(getattr(args, "requested_preferred_eval_branch", "") or "").strip().lower()
    requested_eval_branch_priority = [
        str(item).strip().lower()
        for item in getattr(args, "requested_eval_branch_priority", [])
        if str(item).strip()
    ]
    if requested_preferred_eval_branch:
        args.preferred_eval_branch = requested_preferred_eval_branch
        args.eval_branch_priority = [
            requested_preferred_eval_branch,
            *[item for item in args.eval_branch_priority if item != requested_preferred_eval_branch],
        ]
    if requested_eval_branch_priority:
        args.eval_branch_priority = list(requested_eval_branch_priority)
        if args.preferred_eval_branch not in args.eval_branch_priority:
            args.eval_branch_priority = [
                str(args.preferred_eval_branch),
                *[item for item in args.eval_branch_priority if item != str(args.preferred_eval_branch)],
            ]
    elliptic_mainline_enabled = dataset_name_normalized == "elliptic" and bool(args.gnn_enabled) and bool(args.transformer_enabled)
    amlsim_mainline_enabled = dataset_name_normalized == "amlsim" and bool(args.gnn_enabled) and bool(args.transformer_enabled)
    args.wavelet_lite_enabled = (
        False
        if bool(args.force_disable_wavelet_lite)
        else _env_bool_override(
            "SPLITGNN_WAVELET_LITE_ENABLED",
            bool(elliptic_mainline_enabled),
        )
    )
    args.utg_lite_enabled = (
        False
        if bool(args.force_disable_utg_lite)
        else _env_bool_override(
            "SPLITGNN_UTG_LITE_ENABLED",
            bool(elliptic_mainline_enabled),
        )
    )
    args.coassociation_enabled = (
        False
        if bool(args.force_disable_coassociation)
        else _env_bool_override(
            "SPLITGNN_COASSOCIATION_ENABLED",
            bool(dataset_name_normalized == "elliptic" and bool(args.gnn_enabled)),
        )
    )
    args.diffusion_residual_enabled = (
        False
        if bool(args.force_disable_diffusion_residual)
        else _env_bool_override(
            "SPLITGNN_DIFFUSION_RESIDUAL_ENABLED",
            bool(dataset_name_normalized in {"elliptic", "amlsim"} and bool(args.gnn_enabled)),
        )
    )
    if dataset_name_normalized == "elliptic":
        args.diffusion_residual_scale = float(args.elliptic_diffusion_residual_scale)
    elif dataset_name_normalized == "amlsim":
        args.diffusion_residual_scale = float(args.amlsim_diffusion_residual_scale)
    else:
        args.diffusion_residual_scale = 0.0
    args.coassociation_loss_weight = (
        float(args.elliptic_coassociation_loss_weight)
        if elliptic_mainline_enabled
        else float(args.amlsim_coassociation_loss_weight)
        if amlsim_mainline_enabled
        else 0.0
    )
    args.wavelet_alignment_loss_weight = (
        float(args.elliptic_wavelet_loss_weight)
        if elliptic_mainline_enabled
        else float(args.amlsim_wavelet_loss_weight)
        if amlsim_mainline_enabled
        else 0.0
    )
    args.utg_alignment_loss_weight = (
        float(args.elliptic_utg_align_loss_weight)
        if elliptic_mainline_enabled
        else float(args.amlsim_utg_align_loss_weight)
        if amlsim_mainline_enabled
        else 0.0
    )
    if dataset_name_normalized == "elliptic":
        args.pseudo_cycle_refresh_enabled = bool(
            not bool(args.pure_label_fraction)
            and float(args.label_fraction) < 0.999
            and int(args.elliptic_pseudo_refresh_interval) > 0
            and float(args.elliptic_pseudo_refresh_max_fraction) > 0.0
        )
        args.pseudo_cycle_refresh_interval = int(args.elliptic_pseudo_refresh_interval)
        args.pseudo_cycle_refresh_start_round = int(args.elliptic_pseudo_refresh_start_round)
        args.pseudo_cycle_refresh_momentum = float(args.elliptic_pseudo_refresh_momentum)
        args.pseudo_cycle_refresh_max_fraction = float(args.elliptic_pseudo_refresh_max_fraction)
    elif dataset_name_normalized == "amlsim":
        args.pseudo_cycle_refresh_enabled = bool(
            not bool(args.pure_label_fraction)
            and float(args.label_fraction) < 0.999
            and int(args.amlsim_pseudo_refresh_interval) > 0
            and float(args.amlsim_pseudo_refresh_max_fraction) > 0.0
        )
        args.pseudo_cycle_refresh_interval = int(args.amlsim_pseudo_refresh_interval)
        args.pseudo_cycle_refresh_start_round = int(args.amlsim_pseudo_refresh_start_round)
        args.pseudo_cycle_refresh_momentum = float(args.amlsim_pseudo_refresh_momentum)
        args.pseudo_cycle_refresh_max_fraction = float(args.amlsim_pseudo_refresh_max_fraction)
    else:
        args.pseudo_cycle_refresh_enabled = False
        args.pseudo_cycle_refresh_interval = 0
        args.pseudo_cycle_refresh_start_round = 0
        args.pseudo_cycle_refresh_momentum = 0.65
        args.pseudo_cycle_refresh_max_fraction = 0.0
    args.ethereum_phishing_data_root = str(
        Path(ethereum_phishing_data_root).expanduser().resolve()
    ) if str(ethereum_phishing_data_root).strip() else str(ETHEREUM_PHISHING_DEFAULT_ROOT.resolve())
    args.ethereum_phishing_max_users = (
        None if ethereum_phishing_max_users is None else int(ethereum_phishing_max_users)
    )
    args.ethereum_phishing_max_transactions = (
        None if ethereum_phishing_max_transactions is None else int(ethereum_phishing_max_transactions)
    )
    args.ethereum_phishing_force_preview = bool(ethereum_phishing_force_preview)
    args.ethereum_ponzi_data_root = str(
        Path(ethereum_ponzi_data_root).expanduser().resolve()
    ) if str(ethereum_ponzi_data_root).strip() else str(ETHEREUM_PONZI_DEFAULT_ROOT.resolve())
    args.ethereum_ponzi_negative_users_path = _normalize_resume_identity_path(ethereum_ponzi_negative_users_path)
    args.ethereum_ponzi_force_preview = bool(ethereum_ponzi_force_preview)
    args.defi_rug_pull_data_root = str(
        Path(defi_rug_pull_data_root).expanduser().resolve()
    ) if str(defi_rug_pull_data_root).strip() else str(DEFI_RUG_PULL_DEFAULT_ROOT.resolve())
    args.defi_rug_pull_negative_users_path = _normalize_resume_identity_path(defi_rug_pull_negative_users_path)
    args.defi_rug_pull_force_preview = bool(defi_rug_pull_force_preview)
    args.stage_timer = StageTimer()
    os.makedirs(args.result_path, exist_ok=True)
    with args.stage_timer.track("dataset_load"):
        bundle = _load_hybrid_dataset_bundle(
            dataset_name=dataset_name,
            args=args,
            effective_num_clients=effective_num_clients,
            client_hops=client_hops,
            label_fraction=label_fraction,
        )
    bundle.base_lr = float(args.lr)
    _apply_label_scarcity_runtime_defaults(bundle, args)
    _refresh_client_training_masks(bundle)
    initial_supervised_train_nodes = int(bundle.graph.ndata["train_supervised_mask"].sum().item())
    initial_unlabeled_train_nodes = int(bundle.graph.ndata["train_unlabeled_mask"].sum().item())
    normalized_planner_mode = str(args.planner_mode).lower()
    if normalized_planner_mode not in {"rl", "deterministic"}:
        raise ValueError(f"Unsupported planner mode: {planner_mode}")
    owns_rl_models = normalized_planner_mode == "rl" and rl_models is None
    dataset_seed = 42 if getattr(args, "seed", None) is None else int(args.seed)
    if normalized_planner_mode == "rl" and rl_models is None:
        rl_models = train_controller_models(rl_timesteps=rl_timesteps, seed=dataset_seed)
    if normalized_planner_mode == "deterministic":
        rl_models = None
    setup_seed(dataset_seed)
    data_summary = copy.deepcopy(getattr(bundle, "data_summary", {}))
    if str(dataset_name).lower() == "ieee" and bool(getattr(args, "ieee_skip_training", False)):
        cache_only_summary_path = Path(args.result_path) / f"{dataset_name}_cache_build_summary.json"
        cache_only_summary = {
            "dataset": dataset_name,
            "completed": True,
            "training_skipped": True,
            "status": "cache_build_only" if bool(getattr(args, "ieee_build_cache_only", False)) else "dataset_loaded_only",
            "skip_reason": "ieee_build_cache_only"
            if bool(getattr(args, "ieee_build_cache_only", False))
            else "ieee_skip_training",
            "cache_rebuild_requested": bool(getattr(args, "ieee_rebuild_cache", False)),
            "resolved_device": str(args.device),
            "seed": None if getattr(args, "seed", None) is None else int(args.seed),
            "rounds_ran": 0,
            "result_root": str(resolved_result_root),
            "summary_path": str(cache_only_summary_path),
            "data_summary": data_summary,
            "important_parameters": _main_important_parameters(
                args,
                dataset_name=dataset_name,
                federated_rounds=federated_rounds,
                local_epochs=local_epochs,
            ),
        }
        _log_progress(
            dataset_name,
            "dataset_load_only: skipping training "
            f"build_cache_only={bool(getattr(args, 'ieee_build_cache_only', False))} "
            f"rebuild_cache={bool(getattr(args, 'ieee_rebuild_cache', False))}",
        )
        _atomic_write_json(cache_only_summary_path, {"history": [], "summary": cache_only_summary})
        return cache_only_summary
    global_model = HybridFraudModel(args, bundle.graph)
    graph_teacher_model = None
    args.legacy_fusion_only = bool(getattr(args, "legacy_fusion_only", False))
    warm_start_payload = None
    if resume_path:
        resume_file = Path(resume_path)
        if not resume_file.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_file}")
        warm_start_payload = torch.load(resume_file, map_location="cpu")
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
    if str(getattr(args, "graph_teacher_checkpoint_path", "")).strip():
        graph_teacher_model = _load_fixed_graph_teacher_model(
            checkpoint_path=str(args.graph_teacher_checkpoint_path),
            args=args,
            bundle=bundle,
        )
    resource_guard = estimate_runtime_resource_plan(
        bundle=bundle,
        model=global_model,
        device=args.device,
        use_teacher_model=bool(
            graph_teacher_model is not None
            or (
                float(getattr(args, "teacher_ema_decay", 0.0)) > 0.0
                and (
                    float(getattr(args, "pseudo_label_weight", 0.0)) > 0.0
                    or float(getattr(args, "consistency_weight", 0.0)) > 0.0
                    or float(getattr(args, "open_set_loss_weight", 0.0)) > 0.0
                )
            )
        ),
    )
    resource_guard["requested_device"] = str(getattr(args, "requested_device", args.device))
    resource_guard["dataset"] = dataset_name
    if not bool(resource_guard["fits"]) and str(args.device).lower().startswith("cuda"):
        recommendation_text = ""
        if dataset_name == "ethereum_phishing":
            current_transactions = int(
                dict(data_summary.get("relation_edge_counts", {}) or {}).get(
                    "transfer_out",
                    data_summary.get("max_transactions", 0) or 0,
                )
            )
            recommended_limits = recommend_smaller_phishing_limits(
                current_num_nodes=int(data_summary.get("num_nodes", 0) or 0),
                current_num_transactions=max(current_transactions, 0),
                estimated_vram_gib=float(resource_guard["estimated_vram_gib"]),
                budget_gib=float(resource_guard["budget_gib"] or 0.0),
            )
            resource_guard["recommended_limits"] = recommended_limits
            recommendation_text = (
                " Recommended ethereum_phishing caps: "
                f"max_users={recommended_limits['max_users']}, "
                f"max_transactions={recommended_limits['max_transactions']}."
            )
        raise RuntimeError(
            f"Preflight GPU budget check failed for {dataset_name}: "
            f"estimated_vram_gib={float(resource_guard['estimated_vram_gib']):.2f}, "
            f"budget_gib={float(resource_guard['budget_gib']):.2f}."
            f"{recommendation_text}"
        )
    global_model = global_model.to(args.device)
    run_id = _generate_run_id()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    run_status_path = Path(args.result_path) / f"{dataset_name}_hybrid_run_status.json"
    live_progress_path = Path(args.result_path) / f"{dataset_name}_hybrid_progress.json"
    live_summary_path = Path(args.result_path) / f"{dataset_name}_hybrid_live_summary.json"
    live_resume_checkpoint_path = Path(args.result_path) / f"{dataset_name}_hybrid_live_resume.pt"
    _atomic_write_json(
        run_status_path,
        {
            "dataset": dataset_name,
            "run_id": run_id,
            "status": "running",
            "started_at": started_at,
            "rounds_target": int(total_target_rounds) if total_target_rounds is not None else int(
                max(federated_rounds + max(resume_round_offset, 0), federated_rounds)
            ),
            "latest_round": int(max(resume_round_offset, 0)),
            "live_progress_path": str(live_progress_path),
            "live_summary_path": str(live_summary_path),
            "live_resume_checkpoint_path": str(live_resume_checkpoint_path),
            "planner_mode": normalized_planner_mode,
            "requested_planner_mode": str(getattr(args, "requested_planner_mode", normalized_planner_mode)),
            "requested_disable_federated": bool(getattr(args, "requested_disable_federated", False)),
            "splitgnn_runtime_policy": bool(getattr(args, "splitgnn_runtime_policy", False)),
            "resolved_device": str(args.device),
            "seed": None if getattr(args, "seed", None) is None else int(args.seed),
            "data_summary": data_summary,
            "resource_guard": resource_guard,
        },
    )

    writer = None
    tb_logdir = ""
    run_metadata_path: Path | None = None
    model_path = Path(args.result_path) / f"{dataset_name}_hybrid_fraudgraph.pt"
    result_prefix = Path(args.result_path) / f"{dataset_name}_hybrid_fraudgraph"
    summary_path = Path(args.result_path) / f"{dataset_name}_hybrid_summary.json"
    diagnostics_path = Path(args.result_path) / f"{dataset_name}_diagnostics.json"
    if enable_tensorboard:
        summary_writer_cls = _require_tensorboard_dependency()
        tb_dir = resolved_result_root / dataset_name / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = summary_writer_cls(log_dir=str(tb_dir / run_id))
        tb_logdir = writer.log_dir
        run_metadata_path = Path(tb_logdir) / "run_metadata.json"
        _atomic_write_json(
            run_status_path,
            {
                "dataset": dataset_name,
                "run_id": run_id,
                "status": "running",
                "started_at": started_at,
                "rounds_target": int(total_target_rounds) if total_target_rounds is not None else int(
                    max(federated_rounds + max(resume_round_offset, 0), federated_rounds)
                ),
                "latest_round": int(max(resume_round_offset, 0)),
                "tb_logdir": tb_logdir,
                "live_progress_path": str(live_progress_path),
                "live_summary_path": str(live_summary_path),
                "live_resume_checkpoint_path": str(live_resume_checkpoint_path),
                "planner_mode": normalized_planner_mode,
                "requested_planner_mode": str(getattr(args, "requested_planner_mode", normalized_planner_mode)),
                "requested_disable_federated": bool(getattr(args, "requested_disable_federated", False)),
                "splitgnn_runtime_policy": bool(getattr(args, "splitgnn_runtime_policy", False)),
                "resolved_device": str(args.device),
                "seed": None if getattr(args, "seed", None) is None else int(args.seed),
                "data_summary": data_summary,
                "resource_guard": resource_guard,
            },
        )
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
                rounds_ran=int(max(resume_round_offset, 0)),
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
    history = [dict(item) for item in (preload_history or [])]
    live_progress_enabled = str(os.getenv("SPLITGNN_FORCE_LIVE_PROGRESS", "0")).strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }
    try:
        live_resume_checkpoint_interval = max(
            int(os.getenv("SPLITGNN_LIVE_RESUME_CHECKPOINT_INTERVAL", "0") or 0),
            0,
        )
    except ValueError:
        live_resume_checkpoint_interval = 0
    resume_metrics_rebased = False
    resume_best_metric_inheritance = "not_applicable"
    resume_best_metric_reason = ""
    resume_reference_best_metrics: dict[str, float] = {}
    resume_recheck_valid_metrics: dict[str, float] = {}

    def _runtime_total_round_budget() -> int:
        return int(total_target_rounds) if total_target_rounds is not None else int(
            max(federated_rounds + max(resume_round_offset, 0), federated_rounds)
        )

    def _live_runtime_payload(
        *,
        status: str,
        latest_round: int | None = None,
        round_metrics_payload: dict | None = None,
        error_message: str = "",
    ) -> dict:
        latest_round_value = int(latest_round) if latest_round is not None else int(max(len(history), max(resume_round_offset, 0)))
        latest_history = dict(round_metrics_payload) if round_metrics_payload is not None else (dict(history[-1]) if history else {})
        latest_valid_auc_value = float(
            latest_history.get(
                "valid_auc",
                resume_recheck_valid_metrics.get("auc", best_valid_auc if best_valid_auc >= 0 else 0.0),
            )
        )
        latest_valid_pr_auc_value = float(
            latest_history.get(
                "valid_pr_auc",
                resume_recheck_valid_metrics.get("pr_auc", best_valid_pr_auc if best_valid_pr_auc >= 0 else 0.0),
            )
        )
        latest_valid_f1_macro_value = float(
            latest_history.get(
                "valid_f1_macro",
                resume_recheck_valid_metrics.get("f1_macro", best_valid_f1_macro if best_valid_f1_macro >= 0 else 0.0),
            )
        )
        latest_valid_threshold_value = float(
            latest_history.get(
                "valid_threshold",
                resume_recheck_valid_metrics.get("threshold", best_valid_threshold),
            )
        )
        summary_payload = {
            "dataset": dataset_name,
            "run_id": run_id,
            "status": str(status),
            "started_at": started_at,
            "rounds_target": int(_runtime_total_round_budget()),
            "rounds_requested_this_run": int(federated_rounds),
            "latest_round": int(latest_round_value),
            "rounds_ran": int(len(history)),
            "summary_path": str(summary_path),
            "model_path": str(model_path),
            "run_status_path": str(run_status_path),
            "live_progress_path": str(live_progress_path),
            "live_summary_path": str(live_summary_path),
            "live_resume_checkpoint_path": str(live_resume_checkpoint_path),
            "tb_logdir": tb_logdir,
            "planner_mode": normalized_planner_mode,
            "requested_planner_mode": str(getattr(args, "requested_planner_mode", normalized_planner_mode)),
            "requested_disable_federated": bool(getattr(args, "requested_disable_federated", False)),
            "splitgnn_runtime_policy": bool(getattr(args, "splitgnn_runtime_policy", False)),
            "resolved_device": str(args.device),
            "resume_path": str(resume_path),
            "resume_round_offset": int(max(resume_round_offset, 0)),
            "seed": None if getattr(args, "seed", None) is None else int(args.seed),
            "latest_valid_auc": latest_valid_auc_value,
            "latest_valid_pr_auc": latest_valid_pr_auc_value,
            "latest_valid_f1_macro": latest_valid_f1_macro_value,
            "latest_valid_threshold": latest_valid_threshold_value,
            "best_valid_auc": float(best_valid_auc if best_valid_auc >= 0 else latest_valid_auc_value),
            "best_valid_pr_auc": float(best_valid_pr_auc if best_valid_pr_auc >= 0 else latest_valid_pr_auc_value),
            "best_valid_f1_macro": float(best_valid_f1_macro if best_valid_f1_macro >= 0 else latest_valid_f1_macro_value),
            "best_valid_recall_at_precision": float(
                best_valid_recall_at_precision
                if best_valid_recall_at_precision >= 0
                else resume_recheck_valid_metrics.get("recall_at_precision", 0.0)
            ),
            "best_valid_threshold": float(best_valid_threshold),
            "best_round": int(best_round),
            "peak_valid_auc_any_round": float(peak_valid_auc_any_round),
            "peak_valid_round_any_round": int(peak_valid_round_any_round),
            "peak_valid_threshold_any_round": float(peak_valid_threshold_any_round),
            "history_length": int(len(history)),
            "latest_history": latest_history,
        }
        if error_message:
            summary_payload["error"] = str(error_message)
        return {"history": history, "summary": summary_payload}

    def _write_live_progress(
        *,
        status: str,
        latest_round: int | None = None,
        round_metrics_payload: dict | None = None,
        error_message: str = "",
    ) -> None:
        if not live_progress_enabled:
            return
        payload = _live_runtime_payload(
            status=status,
            latest_round=latest_round,
            round_metrics_payload=round_metrics_payload,
            error_message=error_message,
        )
        _atomic_write_json(live_progress_path, dict(payload["summary"]))

    def _write_live_summary(
        *,
        status: str,
        latest_round: int | None = None,
        round_metrics_payload: dict | None = None,
        error_message: str = "",
    ) -> None:
        if not live_progress_enabled:
            return
        _atomic_write_json(
            live_summary_path,
            _live_runtime_payload(
                status=status,
                latest_round=latest_round,
                round_metrics_payload=round_metrics_payload,
                error_message=error_message,
            ),
        )

    if warm_start_payload is not None:
        resume_reference_best_metrics = _resume_reference_best_metrics(warm_start_payload)
        with args.stage_timer.track("eval"):
            resume_recheck_valid_metrics = _evaluate_model(global_model, bundle.graph, "valid_mask", args.device)
        _release_cuda_memory(args.device)
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
            peak_valid_auc_any_round = float(
                warm_start_payload.get(
                    "peak_valid_auc_any_round",
                    warm_start_payload.get("best_valid_auc", peak_valid_auc_any_round),
                )
            )
            peak_valid_round_any_round = int(
                warm_start_payload.get(
                    "peak_valid_round_any_round",
                    warm_start_payload.get("best_round", peak_valid_round_any_round),
                )
            )
            peak_valid_threshold_any_round = float(
                warm_start_payload.get(
                    "peak_valid_threshold_any_round",
                    warm_start_payload.get("best_valid_threshold", peak_valid_threshold_any_round),
                )
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
        _release_cuda_memory(args.device)
    patience = 0
    has_test_timeseries = False
    pending_active_learning_feedback: List[dict] = []
    total_round_budget = int(total_target_rounds) if total_target_rounds is not None else int(
        max(federated_rounds + max(resume_round_offset, 0), federated_rounds)
    )
    absolute_round_start = int(max(resume_round_offset, 0))
    _write_live_progress(status="running", latest_round=absolute_round_start)
    _write_live_summary(status="running", latest_round=absolute_round_start)
    try:
        for round_index in tqdm(
            range(absolute_round_start, absolute_round_start + federated_rounds),
            desc=f"HybridFraud-{dataset_name}",
            file=sys.stdout,
            dynamic_ncols=True,
        ):
            pending_active_learning_feedback, revealed_feedback = _apply_active_learning_reveal(
                bundle=bundle,
                pending_feedback=pending_active_learning_feedback,
                current_round=round_index,
            )
            pseudo_cycle_refresh = _maybe_refresh_pseudo_cycle_cache(
                bundle=bundle,
                global_model=global_model,
                args=args,
                current_round=round_index,
            )
            round_plan = _plan_round(
                rl_models=rl_models,
                planner_mode=normalized_planner_mode,
                bundle=bundle,
                round_index=round_index,
                total_rounds=total_round_budget,
                base_local_epochs=local_epochs,
                base_edge_loss_weight=edge_loss_weight,
                metrics_history=history,
            )
            selected_clients = [
                client for client in bundle.clients if client.client_id in set(round_plan["selected_clients"])
            ]
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
            local_graph_teacher_losses = []
            local_coassociation_losses = []
            local_wavelet_alignment_losses = []
            local_utg_alignment_losses = []
            local_edge_losses = []
            local_pseudo_nodes = []
            local_novel_nodes = []
            local_open_set_nodes = []
            local_pseudo_thresholds = []
            local_shared_gap_means = []
            local_shared_gap_stds = []
            local_private_interaction_means = []
            local_private_interaction_stds = []
            local_context_gate_means = []
            local_context_gate_stds = []
            local_graph_branch_gate_means = []
            local_graph_branch_gate_stds = []
            local_sequence_branch_gate_means = []
            local_sequence_branch_gate_stds = []
            local_fusion_delta_gate_means = []
            local_fusion_delta_gate_stds = []
            local_graph_embedding_norm_means = []
            local_sequence_embedding_norm_means = []
            local_fused_embedding_norm_means = []
            local_sequence_token_valid_ratio_means = []
            local_sequence_valid_length_means = []
            local_graph_sequence_prob_gap_means = []
            local_pseudo_cycle_agreement_means = []
            local_pseudo_cycle_support_rates = []
            local_wavelet_gate_means = []
            local_coassociation_gate_means = []
            local_diffusion_gate_means = []
            local_utg_temporal_gate_means = []
            local_training_stages = []
            local_graph_learning_rates = []
            local_sequence_learning_rates = []
            local_fusion_learning_rates = []
            global_state = global_model.state_dict()

            for client in selected_clients:
                state_dict, local_metrics = _local_train(
                    global_model=global_model,
                    graph_teacher_model=graph_teacher_model,
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
                local_graph_teacher_losses.append(local_metrics.get("graph_teacher_loss", 0.0))
                local_coassociation_losses.append(local_metrics.get("coassociation_loss", 0.0))
                local_wavelet_alignment_losses.append(local_metrics.get("wavelet_alignment_loss", 0.0))
                local_utg_alignment_losses.append(local_metrics.get("utg_alignment_loss", 0.0))
                local_edge_losses.append(local_metrics["edge_loss"])
                local_pseudo_nodes.append(local_metrics["pseudo_nodes"])
                local_novel_nodes.append(local_metrics["novel_nodes"])
                local_open_set_nodes.append(local_metrics["open_set_nodes"])
                local_pseudo_thresholds.append(local_metrics["pseudo_threshold_used"])
                local_shared_gap_means.append(local_metrics.get("shared_gap_mean", 0.0))
                local_shared_gap_stds.append(local_metrics.get("shared_gap_std", 0.0))
                local_private_interaction_means.append(local_metrics.get("private_interaction_mean", 0.0))
                local_private_interaction_stds.append(local_metrics.get("private_interaction_std", 0.0))
                local_context_gate_means.append(local_metrics.get("context_gate_mean", 0.0))
                local_context_gate_stds.append(local_metrics.get("context_gate_std", 0.0))
                local_graph_branch_gate_means.append(local_metrics.get("graph_branch_gate_mean", 0.0))
                local_graph_branch_gate_stds.append(local_metrics.get("graph_branch_gate_std", 0.0))
                local_sequence_branch_gate_means.append(local_metrics.get("sequence_branch_gate_mean", 0.0))
                local_sequence_branch_gate_stds.append(local_metrics.get("sequence_branch_gate_std", 0.0))
                local_fusion_delta_gate_means.append(local_metrics.get("fusion_delta_gate_mean", 0.0))
                local_fusion_delta_gate_stds.append(local_metrics.get("fusion_delta_gate_std", 0.0))
                local_graph_embedding_norm_means.append(local_metrics.get("graph_embedding_norm_mean", 0.0))
                local_sequence_embedding_norm_means.append(local_metrics.get("sequence_embedding_norm_mean", 0.0))
                local_fused_embedding_norm_means.append(local_metrics.get("fused_embedding_norm_mean", 0.0))
                local_sequence_token_valid_ratio_means.append(
                    local_metrics.get("sequence_token_valid_ratio_mean", 0.0)
                )
                local_sequence_valid_length_means.append(local_metrics.get("sequence_valid_length_mean", 0.0))
                local_graph_sequence_prob_gap_means.append(local_metrics.get("graph_sequence_prob_gap_mean", 0.0))
                local_pseudo_cycle_agreement_means.append(local_metrics.get("pseudo_cycle_agreement_mean", 0.0))
                local_pseudo_cycle_support_rates.append(local_metrics.get("pseudo_cycle_support_rate", 0.0))
                local_wavelet_gate_means.append(local_metrics.get("wavelet_gate_mean", 0.0))
                local_coassociation_gate_means.append(local_metrics.get("coassociation_gate_mean", 0.0))
                local_diffusion_gate_means.append(local_metrics.get("diffusion_gate_mean", 0.0))
                local_utg_temporal_gate_means.append(local_metrics.get("utg_temporal_gate_mean", 0.0))
                local_training_stages.append(str(local_metrics.get("training_stage", "joint_finetune")))
                local_graph_learning_rates.append(float(local_metrics.get("graph_learning_rate", 0.0)))
                local_sequence_learning_rates.append(float(local_metrics.get("sequence_learning_rate", 0.0)))
                local_fusion_learning_rates.append(float(local_metrics.get("fusion_learning_rate", 0.0)))
                _release_cuda_memory(args.device)

            if not local_states:
                del global_state
                _release_cuda_memory(args.device)
                continue

            aggregated_state = _state_dict_average(local_states, local_weights)
            global_model.load_state_dict(aggregated_state)
            del aggregated_state
            del global_state
            _release_cuda_memory(args.device)
            if bool(getattr(args, "lightweight_valid_eval", False)):
                with args.stage_timer.track("eval"):
                    valid_metrics = _evaluate_model(global_model, bundle.graph, "valid_mask", args.device)
                selected_valid_branch = str(
                    getattr(global_model, "_last_eval_branch", getattr(args, "preferred_eval_branch", "main"))
                )
                valid_branch_metrics = {selected_valid_branch: dict(valid_metrics)}
                valid_stats = {}
            else:
                with args.stage_timer.track("eval"):
                    valid_diagnostics = _collect_model_diagnostics(
                        global_model,
                        bundle.graph,
                        args.device,
                        splits=("valid_mask",),
                    )
                valid_payload = valid_diagnostics.get("splits", {}).get("valid", {})
                valid_branch_metrics = dict(valid_payload.get("branches", {}) or {})
                valid_stats = dict(valid_payload.get("stats", {}) or {})
                selected_valid_branch, valid_metrics = _resolve_selected_branch_payload(
                    valid_branch_metrics,
                    valid_payload,
                    args,
                )
            if not valid_metrics:
                selected_valid_branch = _select_branch_name_from_metrics(valid_branch_metrics, args)
                valid_metrics = dict(valid_branch_metrics.get(selected_valid_branch, {}) or {})
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
                "mean_local_graph_teacher_loss": float(np.mean(local_graph_teacher_losses))
                if local_graph_teacher_losses
                else 0.0,
                "mean_local_coassociation_loss": float(np.mean(local_coassociation_losses))
                if local_coassociation_losses
                else 0.0,
                "mean_local_wavelet_alignment_loss": float(np.mean(local_wavelet_alignment_losses))
                if local_wavelet_alignment_losses
                else 0.0,
                "mean_local_utg_alignment_loss": float(np.mean(local_utg_alignment_losses))
                if local_utg_alignment_losses
                else 0.0,
                "mean_local_edge_loss": float(np.mean(local_edge_losses)) if local_edge_losses else 0.0,
                "mean_local_pseudo_nodes": float(np.mean(local_pseudo_nodes)) if local_pseudo_nodes else 0.0,
                "mean_local_novel_nodes": float(np.mean(local_novel_nodes)) if local_novel_nodes else 0.0,
                "mean_local_open_set_nodes": float(np.mean(local_open_set_nodes)) if local_open_set_nodes else 0.0,
                "mean_local_pseudo_threshold": float(np.mean(local_pseudo_thresholds)) if local_pseudo_thresholds else 0.0,
                "mean_local_shared_gap_mean": float(np.mean(local_shared_gap_means)) if local_shared_gap_means else 0.0,
                "mean_local_shared_gap_std": float(np.mean(local_shared_gap_stds)) if local_shared_gap_stds else 0.0,
                "mean_local_private_interaction_mean": float(np.mean(local_private_interaction_means))
                if local_private_interaction_means
                else 0.0,
                "mean_local_private_interaction_std": float(np.mean(local_private_interaction_stds))
                if local_private_interaction_stds
                else 0.0,
                "mean_local_context_gate_mean": float(np.mean(local_context_gate_means))
                if local_context_gate_means
                else 0.0,
                "mean_local_context_gate_std": float(np.mean(local_context_gate_stds))
                if local_context_gate_stds
                else 0.0,
                "mean_local_graph_branch_gate_mean": float(np.mean(local_graph_branch_gate_means))
                if local_graph_branch_gate_means
                else 0.0,
                "mean_local_graph_branch_gate_std": float(np.mean(local_graph_branch_gate_stds))
                if local_graph_branch_gate_stds
                else 0.0,
                "mean_local_sequence_branch_gate_mean": float(np.mean(local_sequence_branch_gate_means))
                if local_sequence_branch_gate_means
                else 0.0,
                "mean_local_sequence_branch_gate_std": float(np.mean(local_sequence_branch_gate_stds))
                if local_sequence_branch_gate_stds
                else 0.0,
                "mean_local_fusion_delta_gate_mean": float(np.mean(local_fusion_delta_gate_means))
                if local_fusion_delta_gate_means
                else 0.0,
                "mean_local_fusion_delta_gate_std": float(np.mean(local_fusion_delta_gate_stds))
                if local_fusion_delta_gate_stds
                else 0.0,
                "mean_local_graph_embedding_norm_mean": float(np.mean(local_graph_embedding_norm_means))
                if local_graph_embedding_norm_means
                else 0.0,
                "mean_local_sequence_embedding_norm_mean": float(np.mean(local_sequence_embedding_norm_means))
                if local_sequence_embedding_norm_means
                else 0.0,
                "mean_local_fused_embedding_norm_mean": float(np.mean(local_fused_embedding_norm_means))
                if local_fused_embedding_norm_means
                else 0.0,
                "mean_local_sequence_token_valid_ratio_mean": float(np.mean(local_sequence_token_valid_ratio_means))
                if local_sequence_token_valid_ratio_means
                else 0.0,
                "mean_local_sequence_valid_length_mean": float(np.mean(local_sequence_valid_length_means))
                if local_sequence_valid_length_means
                else 0.0,
                "mean_local_graph_sequence_prob_gap_mean": float(np.mean(local_graph_sequence_prob_gap_means))
                if local_graph_sequence_prob_gap_means
                else 0.0,
                "mean_local_pseudo_cycle_agreement_mean": float(np.mean(local_pseudo_cycle_agreement_means))
                if local_pseudo_cycle_agreement_means
                else 0.0,
                "mean_local_pseudo_cycle_support_rate": float(np.mean(local_pseudo_cycle_support_rates))
                if local_pseudo_cycle_support_rates
                else 0.0,
                "mean_local_wavelet_gate_mean": float(np.mean(local_wavelet_gate_means))
                if local_wavelet_gate_means
                else 0.0,
                "mean_local_coassociation_gate_mean": float(np.mean(local_coassociation_gate_means))
                if local_coassociation_gate_means
                else 0.0,
                "mean_local_diffusion_gate_mean": float(np.mean(local_diffusion_gate_means))
                if local_diffusion_gate_means
                else 0.0,
                "mean_local_utg_temporal_gate_mean": float(np.mean(local_utg_temporal_gate_means))
                if local_utg_temporal_gate_means
                else 0.0,
                "training_stage": str(local_training_stages[0]) if local_training_stages else "joint_finetune",
                "graph_learning_rate": float(np.mean(local_graph_learning_rates)) if local_graph_learning_rates else 0.0,
                "sequence_learning_rate": float(np.mean(local_sequence_learning_rates))
                if local_sequence_learning_rates
                else 0.0,
                "fusion_learning_rate": float(np.mean(local_fusion_learning_rates)) if local_fusion_learning_rates else 0.0,
                "valid_selected_branch": str(selected_valid_branch),
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
                "valid_selected_auc": float(valid_metrics["auc"]),
                "valid_selected_pr_auc": float(valid_metrics["pr_auc"]),
                "valid_selected_f1_macro": float(valid_metrics["f1_macro"]),
                "valid_main_auc": float(valid_branch_metrics.get("main", {}).get("auc", 0.0)),
                "valid_main_f1_macro": float(valid_branch_metrics.get("main", {}).get("f1_macro", 0.0)),
                "valid_fusion_auc": float(valid_branch_metrics.get("fusion", {}).get("auc", 0.0)),
                "valid_fusion_f1_macro": float(valid_branch_metrics.get("fusion", {}).get("f1_macro", 0.0)),
                "valid_graph_residual_auc": float(valid_branch_metrics.get("graph_residual", {}).get("auc", 0.0)),
                "valid_graph_residual_f1_macro": float(
                    valid_branch_metrics.get("graph_residual", {}).get("f1_macro", 0.0)
                ),
                "valid_sequence_residual_auc": float(
                    valid_branch_metrics.get("sequence_residual", {}).get("auc", 0.0)
                ),
                "valid_sequence_residual_f1_macro": float(
                    valid_branch_metrics.get("sequence_residual", {}).get("f1_macro", 0.0)
                ),
                "valid_raw_branch_auc": float(valid_branch_metrics.get("raw_branch", {}).get("auc", 0.0)),
                "valid_raw_branch_f1_macro": float(valid_branch_metrics.get("raw_branch", {}).get("f1_macro", 0.0)),
                "valid_shared_gap_mean": float(valid_stats.get("shared_gap_mean", 0.0)),
                "valid_shared_gap_std": float(valid_stats.get("shared_gap_std", 0.0)),
                "valid_private_interaction_mean": float(valid_stats.get("private_interaction_mean", 0.0)),
                "valid_private_interaction_std": float(valid_stats.get("private_interaction_std", 0.0)),
                "valid_context_gate_mean": float(valid_stats.get("context_gate_mean", 0.0)),
                "valid_context_gate_std": float(valid_stats.get("context_gate_std", 0.0)),
                "valid_graph_branch_gate_mean": float(valid_stats.get("graph_branch_gate_mean", 0.0)),
                "valid_graph_branch_gate_std": float(valid_stats.get("graph_branch_gate_std", 0.0)),
                "valid_sequence_branch_gate_mean": float(valid_stats.get("sequence_branch_gate_mean", 0.0)),
                "valid_sequence_branch_gate_std": float(valid_stats.get("sequence_branch_gate_std", 0.0)),
                "valid_fusion_delta_gate_mean": float(valid_stats.get("fusion_delta_gate_mean", 0.0)),
                "valid_fusion_delta_gate_std": float(valid_stats.get("fusion_delta_gate_std", 0.0)),
                "valid_graph_embedding_norm_mean": float(valid_stats.get("graph_embedding_norm_mean", 0.0)),
                "valid_sequence_embedding_norm_mean": float(valid_stats.get("sequence_embedding_norm_mean", 0.0)),
                "valid_fused_embedding_norm_mean": float(valid_stats.get("fused_embedding_norm_mean", 0.0)),
                "valid_sequence_token_valid_ratio_mean": float(
                    valid_stats.get("sequence_token_valid_ratio_mean", 0.0)
                ),
                "valid_sequence_valid_length_mean": float(valid_stats.get("sequence_valid_length_mean", 0.0)),
                "valid_graph_sequence_prob_gap_mean": float(
                    valid_stats.get("graph_sequence_prob_gap_mean", 0.0)
                ),
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
                "round_progress": float((round_index + 1) / max(total_round_budget, 1)),
                "active_learning_revealed": int(len(revealed_feedback)),
                "active_learning_pending": int(len(pending_active_learning_feedback)),
                "supervised_train_nodes": int(bundle.graph.ndata["train_supervised_mask"].sum().item()),
                "unlabeled_train_nodes": int(bundle.graph.ndata["train_unlabeled_mask"].sum().item()),
                "pseudo_cycle_refreshed": bool(pseudo_cycle_refresh["refreshed"]),
                "pseudo_cycle_nodes": int(pseudo_cycle_refresh["nodes"]),
                "pseudo_cycle_threshold": float(pseudo_cycle_refresh["threshold"]),
                "pseudo_cycle_reliability_mean": float(pseudo_cycle_refresh["reliability_mean"]),
                "pseudo_cycle_support_rate": float(pseudo_cycle_refresh["support_rate"]),
            }
            controller_reward, reward_details = _controller_reward_from_round(
                dataset_name=dataset_name,
                round_index=round_index,
                total_rounds=total_round_budget,
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
                f"[{dataset_name}] round {round_index + 1}/{total_round_budget} "
                f"eval_branch={str(round_metrics['valid_selected_branch'])} "
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
                writer.add_scalar(
                    "train/mean_local_graph_teacher_loss",
                    round_metrics["mean_local_graph_teacher_loss"],
                    round_index,
                )
                writer.add_scalar("train/mean_local_edge_loss", round_metrics["mean_local_edge_loss"], round_index)
                writer.add_scalar("train/mean_local_pseudo_nodes", round_metrics["mean_local_pseudo_nodes"], round_index)
                writer.add_scalar("train/mean_local_novel_nodes", round_metrics["mean_local_novel_nodes"], round_index)
                writer.add_scalar("train/mean_local_open_set_nodes", round_metrics["mean_local_open_set_nodes"], round_index)
                writer.add_scalar("train/mean_local_pseudo_threshold", round_metrics["mean_local_pseudo_threshold"], round_index)
                writer.add_scalar("train/mean_local_shared_gap_mean", round_metrics["mean_local_shared_gap_mean"], round_index)
                writer.add_scalar("train/mean_local_private_interaction_mean", round_metrics["mean_local_private_interaction_mean"], round_index)
                writer.add_scalar("train/mean_local_context_gate_mean", round_metrics["mean_local_context_gate_mean"], round_index)
                writer.add_scalar(
                    "train/mean_local_graph_branch_gate_mean",
                    round_metrics["mean_local_graph_branch_gate_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "train/mean_local_sequence_branch_gate_mean",
                    round_metrics["mean_local_sequence_branch_gate_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "train/mean_local_fusion_delta_gate_mean",
                    round_metrics["mean_local_fusion_delta_gate_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "train/mean_local_graph_embedding_norm_mean",
                    round_metrics["mean_local_graph_embedding_norm_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "train/mean_local_sequence_embedding_norm_mean",
                    round_metrics["mean_local_sequence_embedding_norm_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "train/mean_local_sequence_token_valid_ratio_mean",
                    round_metrics["mean_local_sequence_token_valid_ratio_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "train/mean_local_graph_sequence_prob_gap_mean",
                    round_metrics["mean_local_graph_sequence_prob_gap_mean"],
                    round_index,
                )
                writer.add_scalar("valid/acc", round_metrics["valid_acc"], round_index)
                writer.add_scalar("valid/f1_binary", round_metrics["valid_f1_binary"], round_index)
                writer.add_scalar("valid/auc", round_metrics["valid_auc"], round_index)
                writer.add_scalar("valid/pr_auc", round_metrics["valid_pr_auc"], round_index)
                writer.add_scalar("valid/f1_macro", round_metrics["valid_f1_macro"], round_index)
                writer.add_scalar("valid/gmean", round_metrics["valid_gmean"], round_index)
                writer.add_scalar("valid/recall", round_metrics["valid_recall"], round_index)
                writer.add_scalar("valid/selected_auc", round_metrics["valid_selected_auc"], round_index)
                writer.add_scalar("valid/selected_pr_auc", round_metrics["valid_selected_pr_auc"], round_index)
                writer.add_scalar("valid/selected_f1_macro", round_metrics["valid_selected_f1_macro"], round_index)
                writer.add_scalar(
                    "valid/recall_at_precision",
                    round_metrics["valid_recall_at_precision"],
                    round_index,
                )
                writer.add_scalar("valid/threshold", round_metrics["valid_threshold"], round_index)
                writer.add_scalar("valid/positive_rate", round_metrics["valid_positive_rate"], round_index)
                writer.add_scalar("valid/prob_std", round_metrics["valid_prob_std"], round_index)
                writer.add_scalar("branch_valid/main_auc", round_metrics["valid_main_auc"], round_index)
                writer.add_scalar("branch_valid/main_f1_macro", round_metrics["valid_main_f1_macro"], round_index)
                writer.add_scalar("branch_valid/fusion_auc", round_metrics["valid_fusion_auc"], round_index)
                writer.add_scalar("branch_valid/fusion_f1_macro", round_metrics["valid_fusion_f1_macro"], round_index)
                writer.add_scalar(
                    "branch_valid/graph_residual_auc",
                    round_metrics["valid_graph_residual_auc"],
                    round_index,
                )
                writer.add_scalar(
                    "branch_valid/graph_residual_f1_macro",
                    round_metrics["valid_graph_residual_f1_macro"],
                    round_index,
                )
                writer.add_scalar(
                    "branch_valid/sequence_residual_auc",
                    round_metrics["valid_sequence_residual_auc"],
                    round_index,
                )
                writer.add_scalar(
                    "branch_valid/sequence_residual_f1_macro",
                    round_metrics["valid_sequence_residual_f1_macro"],
                    round_index,
                )
                writer.add_scalar("branch_valid/raw_branch_auc", round_metrics["valid_raw_branch_auc"], round_index)
                writer.add_scalar(
                    "branch_valid/raw_branch_f1_macro",
                    round_metrics["valid_raw_branch_f1_macro"],
                    round_index,
                )
                writer.add_scalar("diagnostic/valid_shared_gap_mean", round_metrics["valid_shared_gap_mean"], round_index)
                writer.add_scalar(
                    "diagnostic/valid_private_interaction_mean",
                    round_metrics["valid_private_interaction_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "diagnostic/valid_context_gate_mean",
                    round_metrics["valid_context_gate_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "diagnostic/valid_graph_branch_gate_mean",
                    round_metrics["valid_graph_branch_gate_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "diagnostic/valid_sequence_branch_gate_mean",
                    round_metrics["valid_sequence_branch_gate_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "diagnostic/valid_fusion_delta_gate_mean",
                    round_metrics["valid_fusion_delta_gate_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "diagnostic/valid_graph_embedding_norm_mean",
                    round_metrics["valid_graph_embedding_norm_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "diagnostic/valid_sequence_embedding_norm_mean",
                    round_metrics["valid_sequence_embedding_norm_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "diagnostic/valid_sequence_token_valid_ratio_mean",
                    round_metrics["valid_sequence_token_valid_ratio_mean"],
                    round_index,
                )
                writer.add_scalar(
                    "diagnostic/valid_graph_sequence_prob_gap_mean",
                    round_metrics["valid_graph_sequence_prob_gap_mean"],
                    round_index,
                )
                writer.add_scalar("plan/local_epochs", round_metrics["local_epochs"], round_index)
                writer.add_scalar("plan/grad_clip", round_metrics["grad_clip"], round_index)
                writer.add_scalar("plan/selected_clients", len(round_metrics["selected_clients"]), round_index)
                writer.add_scalar("plan/selected_ratio", round_metrics["selected_ratio"], round_index)
                writer.add_scalar("plan/actual_selected_ratio", round_metrics["actual_selected_ratio"], round_index)
                writer.add_scalar("plan/selection_phase", round_metrics["selection_phase"], round_index)
                writer.add_scalar("plan/edge_loss_weight", round_metrics["edge_loss_weight"], round_index)
                writer.add_scalar("plan/learning_rate", round_metrics["learning_rate"], round_index)
                writer.add_scalar("plan/graph_learning_rate", round_metrics["graph_learning_rate"], round_index)
                writer.add_scalar("plan/sequence_learning_rate", round_metrics["sequence_learning_rate"], round_index)
                writer.add_scalar("plan/fusion_learning_rate", round_metrics["fusion_learning_rate"], round_index)
                writer.add_scalar(
                    "plan/training_stage",
                    {"graph_warmup": 0.0, "fusion_bootstrap": 1.0, "joint_finetune": 2.0}.get(
                        str(round_metrics["training_stage"]),
                        2.0,
                    ),
                    round_index,
                )
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
                writer.add_scalar("reward/controller", round_metrics["controller_reward"], round_index)
                writer.add_scalar("reward/auc_gain", reward_details["auc_gain"], round_index)
                writer.add_scalar("reward/f1_gain", reward_details["f1_gain"], round_index)
                writer.add_scalar("reward/recall_gain", reward_details["recall_gain"], round_index)
                writer.add_scalar("reward/alignment", reward_details["alignment"], round_index)
                writer.add_scalar("reward/quality_bonus", reward_details["quality_bonus"], round_index)
                writer.add_scalar("reward/communication_cost", reward_details["communication_cost"], round_index)
                writer.add_scalar("reward/compute_cost", reward_details["compute_cost"], round_index)
                writer.add_scalar("reward/collapse_penalty", reward_details["collapse_penalty"], round_index)
                writer.add_scalar("reward/plateau_penalty", reward_details["plateau_penalty"], round_index)
                writer.add_scalar("reward/stagnation_penalty", reward_details["stagnation_penalty"], round_index)
                writer.add_scalar("reward/prob_std_gain", reward_details["prob_std_gain"], round_index)
                writer.add_scalar(
                    "diagnostic/reward_reference_only",
                    1.0 if normalized_planner_mode != "rl" else 0.0,
                    round_index,
                )
                writer.add_scalar("diagnostic/controller_score", round_metrics["controller_reward"], round_index)
                if args.test_every and (round_index + 1) % args.test_every == 0:
                    with args.stage_timer.track("eval"):
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
                if args.early_stop > 0 and patience >= args.early_stop:
                    break

            if checkpoint_guard["eligible"] and _should_update_best_checkpoint(
                valid_metrics,
                best_round=best_round,
                best_valid_auc=best_valid_auc,
                best_valid_gmean=best_valid_gmean,
                best_valid_pr_auc=best_valid_pr_auc,
                best_valid_recall_at_precision=best_valid_recall_at_precision,
                best_valid_f1_macro=best_valid_f1_macro,
                splitgnn_policy_active=bool(getattr(args, "splitgnn_runtime_policy", False)),
            ):
                best_valid_auc = float(valid_metrics["auc"])
                best_valid_gmean = float(valid_metrics["gmean"])
                best_valid_pr_auc = float(valid_metrics["pr_auc"])
                best_valid_recall_at_precision = float(valid_metrics["recall_at_precision"])
                best_valid_f1_macro = float(valid_metrics["f1_macro"])
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
                    "rounds_target": int(total_round_budget),
                    "summary_path": str(summary_path),
                    "model_path": str(model_path),
                    "live_progress_path": str(live_progress_path),
                    "live_summary_path": str(live_summary_path),
                    "live_resume_checkpoint_path": str(live_resume_checkpoint_path),
                    "tb_logdir": tb_logdir,
                    "planner_mode": normalized_planner_mode,
                    "requested_planner_mode": str(getattr(args, "requested_planner_mode", normalized_planner_mode)),
                    "requested_disable_federated": bool(getattr(args, "requested_disable_federated", False)),
                    "splitgnn_runtime_policy": bool(getattr(args, "splitgnn_runtime_policy", False)),
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
            _write_live_progress(
                status="running",
                latest_round=round_index + 1,
                round_metrics_payload=round_metrics,
            )
            _write_live_summary(
                status="running",
                latest_round=round_index + 1,
                round_metrics_payload=round_metrics,
            )
            if live_resume_checkpoint_interval > 0 and (round_index + 1) % live_resume_checkpoint_interval == 0:
                _atomic_torch_save(
                    live_resume_checkpoint_path,
                    {
                        "model_state": snapshot_model_state_to_cpu(global_model),
                        "legacy_fusion_only": bool(getattr(global_model, "legacy_fusion_only", False)),
                        "best_valid_auc": float(best_valid_auc if best_valid_auc >= 0 else valid_metrics["auc"]),
                        "best_valid_gmean": float(best_valid_gmean if best_valid_gmean >= 0 else valid_metrics["gmean"]),
                        "best_valid_pr_auc": float(best_valid_pr_auc if best_valid_pr_auc >= 0 else valid_metrics["pr_auc"]),
                        "best_valid_recall_at_precision": float(
                            best_valid_recall_at_precision
                            if best_valid_recall_at_precision >= 0
                            else valid_metrics["recall_at_precision"]
                        ),
                        "best_valid_f1_macro": float(
                            best_valid_f1_macro if best_valid_f1_macro >= 0 else valid_metrics["f1_macro"]
                        ),
                        "best_round": int(best_round if best_round >= 0 else round_index),
                        "best_valid_threshold": float(
                            best_valid_threshold if best_valid_threshold >= 0 else valid_metrics["threshold"]
                        ),
                        "peak_valid_auc_any_round": float(peak_valid_auc_any_round),
                        "peak_valid_round_any_round": int(peak_valid_round_any_round),
                        "peak_valid_threshold_any_round": float(peak_valid_threshold_any_round),
                        "args": vars(args),
                        "planner_mode": normalized_planner_mode,
                        "ablation_mode": str(getattr(args, "ablation_mode", "full")),
                        "relation_order": bundle.relation_order,
                        "run_id": run_id,
                        "tb_logdir": tb_logdir,
                        "completed": False,
                        "interrupted": False,
                        "protocol_isolated": True,
                        "rounds_ran": len(history),
                        "latest_round": int(round_index + 1),
                    },
                )

        checkpoint_selection_fallback_used = bool(best_round < 0 and peak_valid_round_any_round >= 0)
        if checkpoint_selection_fallback_used:
            best_state = peak_state
            best_valid_auc = peak_valid_auc_any_round
            best_valid_pr_auc = (
                float(best_valid_metrics.get("pr_auc", best_valid_pr_auc))
                if "best_valid_metrics" in locals()
                else best_valid_pr_auc
            )
            best_valid_recall_at_precision = (
                float(best_valid_metrics.get("recall_at_precision", best_valid_recall_at_precision))
                if "best_valid_metrics" in locals()
                else best_valid_recall_at_precision
            )
            best_valid_f1_macro = float(best_valid_metrics.get("f1_macro", best_valid_f1_macro)) if "best_valid_metrics" in locals() else best_valid_f1_macro
            best_round = peak_valid_round_any_round
            best_valid_threshold = peak_valid_threshold_any_round
        _log_progress(dataset_name, "finalize: loading best checkpoint state into global model")
        global_model.load_state_dict(best_state)
        _log_progress(dataset_name, "finalize: starting final valid evaluation")
        final_valid_eval_start = time.perf_counter()
        with args.stage_timer.track("eval"):
            best_valid_metrics = _evaluate_model(global_model, bundle.graph, "valid_mask", args.device)
        best_valid_selected_branch = str(
            getattr(global_model, "_last_eval_branch", getattr(args, "preferred_eval_branch", "main"))
        )
        _log_progress(
            dataset_name,
            "finalize: completed final valid evaluation "
            f"in {time.perf_counter() - final_valid_eval_start:.2f}s "
            f"(branch={best_valid_selected_branch}, auc={float(best_valid_metrics.get('auc', 0.0)):.6f})",
        )
        best_valid_auc = float(best_valid_metrics["auc"])
        best_valid_gmean = float(best_valid_metrics["gmean"])
        best_valid_pr_auc = float(best_valid_metrics["pr_auc"])
        best_valid_recall_at_precision = float(best_valid_metrics["recall_at_precision"])
        best_valid_f1_macro = float(best_valid_metrics["f1_macro"])
        best_valid_threshold = float(best_valid_metrics["threshold"])
        test_selected_branch = ""
        if bool(getattr(args, "skip_test_evaluation", False)):
            _log_progress(dataset_name, "finalize: skipping final test evaluation by configuration")
            test_metrics = None
        else:
            _log_progress(
                dataset_name,
                "finalize: starting final test evaluation "
                f"(threshold={best_valid_threshold:.6f})",
            )
            final_test_eval_start = time.perf_counter()
            with args.stage_timer.track("eval"):
                test_metrics = _evaluate_model(
                    global_model,
                    bundle.graph,
                    "test_mask",
                    args.device,
                    result_path=str(result_prefix),
                    threshold=best_valid_threshold,
                )
            test_selected_branch = str(
                getattr(global_model, "_last_eval_branch", getattr(args, "preferred_eval_branch", "main"))
            )
            _log_progress(
                dataset_name,
                "finalize: completed final test evaluation "
                f"in {time.perf_counter() - final_test_eval_start:.2f}s "
                f"(branch={test_selected_branch}, auc={float(test_metrics.get('auc', 0.0)):.6f})",
            )
        has_test_metrics = isinstance(test_metrics, dict) and "auc" in test_metrics
        final_supervised_train_nodes = int(bundle.graph.ndata["train_supervised_mask"].sum().item())
        final_unlabeled_train_nodes = int(bundle.graph.ndata["train_unlabeled_mask"].sum().item())
        skip_final_diagnostics = (
            bool(getattr(args, "skip_test_evaluation", False)) and len(history) == 0
        ) or _env_bool_override("SPLITGNN_SKIP_FINAL_DIAGNOSTICS", False)
        if skip_final_diagnostics:
            _log_progress(
                dataset_name,
                "finalize: skipping final diagnostics collection by runtime policy/flag",
            )
            final_diagnostics = {
                "available_branches": [],
                "branch_thresholds": {},
                "splits": {},
            }
        else:
            _log_progress(dataset_name, "finalize: starting final diagnostics collection")
            final_diagnostics_start = time.perf_counter()
            with args.stage_timer.track("eval"):
                final_diagnostics = _collect_model_diagnostics(
                    global_model,
                    bundle.graph,
                    args.device,
                    splits=("valid_mask", "test_mask"),
                )
            _log_progress(
                dataset_name,
                "finalize: completed final diagnostics collection "
                f"in {time.perf_counter() - final_diagnostics_start:.2f}s",
            )
        final_valid_payload = dict(final_diagnostics.get("splits", {}).get("valid", {}) or {})
        final_test_payload = dict(final_diagnostics.get("splits", {}).get("test", {}) or {})
        final_valid_branches = dict(final_valid_payload.get("branches", {}) or {})
        final_test_branches = dict(final_test_payload.get("branches", {}) or {})
        graph_gate_curve = [float(item.get("valid_graph_branch_gate_mean", 0.0)) for item in history]
        graph_gate_trend = {
            "start": float(graph_gate_curve[0]) if graph_gate_curve else 0.0,
            "end": float(graph_gate_curve[-1]) if graph_gate_curve else 0.0,
            "delta": float(graph_gate_curve[-1] - graph_gate_curve[0]) if len(graph_gate_curve) >= 2 else 0.0,
            "max": float(max(graph_gate_curve)) if graph_gate_curve else 0.0,
            "min": float(min(graph_gate_curve)) if graph_gate_curve else 0.0,
        }
        diagnostic_curve_keys = (
            "round",
            "training_stage",
            "graph_learning_rate",
            "sequence_learning_rate",
            "fusion_learning_rate",
            "valid_selected_branch",
            "valid_selected_auc",
            "valid_selected_pr_auc",
            "valid_selected_f1_macro",
            "valid_main_auc",
            "valid_main_f1_macro",
            "valid_fusion_auc",
            "valid_fusion_f1_macro",
            "valid_graph_residual_auc",
            "valid_graph_residual_f1_macro",
            "valid_sequence_residual_auc",
            "valid_sequence_residual_f1_macro",
            "valid_raw_branch_auc",
            "valid_raw_branch_f1_macro",
            "valid_shared_gap_mean",
            "valid_private_interaction_mean",
            "valid_context_gate_mean",
            "valid_graph_branch_gate_mean",
            "valid_sequence_branch_gate_mean",
            "valid_fusion_delta_gate_mean",
            "valid_graph_embedding_norm_mean",
            "valid_sequence_embedding_norm_mean",
            "valid_sequence_token_valid_ratio_mean",
            "valid_graph_sequence_prob_gap_mean",
        )
        diagnostics_payload = {
            "dataset": dataset_name,
            "run_id": run_id,
            "summary_path": str(summary_path),
            "model_path": str(model_path),
            "available_branches": list(final_diagnostics.get("available_branches", [])),
            "preferred_branch_priority": list(final_diagnostics.get("preferred_branch_priority", [])),
            "selected_branch": str(final_diagnostics.get("selected_branch", getattr(args, "preferred_eval_branch", "main"))),
            "branch_thresholds": dict(final_diagnostics.get("branch_thresholds", {})),
            "splits": dict(final_diagnostics.get("splits", {})),
            "sequence_quality": dict(data_summary.get("sequence_quality", {}) or {}),
            "graph_gate_trend": graph_gate_trend,
            "round_curves": [
                {key: item.get(key) for key in diagnostic_curve_keys}
                for item in history
            ],
        }
        _log_progress(dataset_name, f"finalize: writing diagnostics json -> {diagnostics_path}")
        diagnostics_write_start = time.perf_counter()
        with args.stage_timer.track("json_write"):
            _atomic_write_json(diagnostics_path, diagnostics_payload)
        _log_progress(
            dataset_name,
            "finalize: completed diagnostics json write "
            f"in {time.perf_counter() - diagnostics_write_start:.2f}s",
        )

        if writer is not None and not has_test_timeseries and has_test_metrics:
            # Keep `test/*` semantically clean: either periodic evaluation during
            # training, or one final point when periodic test logging is disabled.
            test_step = len(history)
            writer.add_scalar("test/acc", test_metrics["acc"], test_step)
            writer.add_scalar("test/f1_binary", test_metrics["f1_binary"], test_step)
            writer.add_scalar("test/auc", test_metrics["auc"], test_step)
            writer.add_scalar("test/pr_auc", test_metrics["pr_auc"], test_step)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], test_step)
            writer.add_scalar("test/gmean", test_metrics["gmean"], test_step)
            writer.add_scalar("test/recall", test_metrics["recall"], test_step)
            writer.add_scalar("test/recall_at_precision", test_metrics["recall_at_precision"], test_step)
            writer.add_scalar("test/threshold", test_metrics["threshold"], test_step)
            writer.add_scalar("test/positive_rate", test_metrics["positive_rate"], test_step)
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
            "preferred_eval_branch": str(getattr(args, "preferred_eval_branch", "main")),
            "eval_branch_priority": list(getattr(args, "eval_branch_priority", ["main"])),
            "best_valid_selected_branch": str(best_valid_selected_branch),
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
            "requested_classification_loss": str(getattr(args, "requested_classification_loss", getattr(args, "classification_loss", "cb_focal"))),
            "classification_loss": str(getattr(args, "classification_loss", "cb_focal")),
            "best_round": best_round,
            "best_valid_acc": float(best_valid_metrics.get("acc", 0.0)),
            "best_valid_auc": best_valid_auc,
            "best_valid_gmean": best_valid_gmean,
            "best_valid_pr_auc": best_valid_pr_auc,
            "best_valid_recall_at_precision": best_valid_recall_at_precision,
            "best_valid_f1_macro": best_valid_f1_macro,
            "best_valid_threshold": best_valid_threshold,
            "peak_valid_auc_any_round": peak_valid_auc_any_round,
            "peak_valid_round_any_round": peak_valid_round_any_round,
            "peak_valid_threshold_any_round": peak_valid_threshold_any_round,
            "checkpoint_selection_fallback_used": checkpoint_selection_fallback_used,
            "resume_metrics_rebased": bool(resume_metrics_rebased),
            "best_valid_metrics": best_valid_metrics,
            "test": test_metrics if has_test_metrics else {},
            "test_acc": float(test_metrics["acc"]) if has_test_metrics else None,
            "test_recall": float(test_metrics["recall"]) if has_test_metrics else None,
            "test_precision": float(test_metrics["precision"]) if has_test_metrics else None,
            "test_selected_branch": str(test_selected_branch) if has_test_metrics else None,
            "test_evaluated": bool(has_test_metrics),
            "mean_controller_reward": float(np.mean([item.get("controller_reward", 0.0) for item in history])) if history else 0.0,
            "best_controller_reward": float(max([item.get("controller_reward", -1e9) for item in history])) if history else 0.0,
            "rounds_ran": len(history),
            "num_clients": len(bundle.clients),
            "label_fraction": float(args.label_fraction),
            "pure_label_fraction": bool(getattr(args, "pure_label_fraction", False)),
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
            "raw_aux_loss_weight": float(getattr(args, "raw_aux_loss_weight", 0.0)),
            "raw_anchor_dim": int(getattr(args, "raw_anchor_dim", 0)),
            "graph_anchor_loss_weight": float(getattr(args, "graph_anchor_loss_weight", 0.0)),
            "graph_anchor_temperature": float(getattr(args, "graph_anchor_temperature", 1.5)),
            "graph_teacher_checkpoint_path": str(getattr(args, "graph_teacher_checkpoint_path", "")),
            "graph_teacher_distill_weight": float(getattr(args, "graph_teacher_distill_weight", 0.0)),
            "graph_teacher_temperature": float(getattr(args, "graph_teacher_temperature", 1.5)),
            "graph_gate_logit_bias": float(getattr(args, "graph_gate_logit_bias", 0.0)),
            "eval_graph_gate_logit_bias": float(getattr(args, "eval_graph_gate_logit_bias", 0.0)),
            "graph_residual_min_gate": float(getattr(args, "graph_residual_min_gate", 0.0)),
            "sequence_residual_scale": float(getattr(args, "sequence_residual_scale", 1.0)),
            "fusion_variant": str(getattr(args, "fusion_variant", "single_branch")),
            "modality_dropout_prob": float(getattr(args, "modality_dropout_prob", 0.0)),
            "graph_learning_rate_scale": float(getattr(args, "graph_learning_rate_scale", 1.0)),
            "sequence_learning_rate_scale": float(getattr(args, "sequence_learning_rate_scale", 1.0)),
            "fusion_learning_rate_scale": float(getattr(args, "fusion_learning_rate_scale", 1.0)),
            "graph_follow_learning_rate_scale": float(getattr(args, "graph_follow_learning_rate_scale", 1.0)),
            "graph_warmup_rounds": int(getattr(args, "graph_warmup_rounds", 0)),
            "fusion_bootstrap_rounds": int(getattr(args, "fusion_bootstrap_rounds", 0)),
            "open_set_novelty_threshold": float(getattr(args, "open_set_novelty_threshold", 0.0)),
            "open_set_loss_weight": float(getattr(args, "open_set_loss_weight", 0.0)),
            "prototype_loss_weight": float(getattr(args, "prototype_loss_weight", 0.0)),
            "shared_private_loss_weight": float(getattr(args, "shared_private_loss_weight", 0.0)),
            "context_alignment_loss_weight": float(getattr(args, "context_alignment_loss_weight", 0.0)),
            "uncertainty_loss_weight": float(getattr(args, "uncertainty_loss_weight", 0.0)),
            "legacy_fusion_only": bool(getattr(args, "legacy_fusion_only", False)),
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
            "initial_supervised_train_nodes": int(initial_supervised_train_nodes),
            "initial_unlabeled_train_nodes": int(initial_unlabeled_train_nodes),
            "final_supervised_train_nodes": int(final_supervised_train_nodes),
            "final_unlabeled_train_nodes": int(final_unlabeled_train_nodes),
            "mean_local_pseudo_nodes": float(np.mean([item.get("mean_local_pseudo_nodes", 0.0) for item in history])) if history else 0.0,
            "mean_local_pseudo_threshold": float(np.mean([item.get("mean_local_pseudo_threshold", 0.0) for item in history])) if history else 0.0,
            "mean_local_open_set_nodes": float(np.mean([item.get("mean_local_open_set_nodes", 0.0) for item in history])) if history else 0.0,
            "mean_local_graph_teacher_loss": float(np.mean([item.get("mean_local_graph_teacher_loss", 0.0) for item in history])) if history else 0.0,
            "stage_schedule": [str(item.get("training_stage", "")) for item in history],
            "graph_gate_trend": graph_gate_trend,
            "final_valid_branch_metrics": final_valid_branches,
            "final_test_branch_metrics": final_test_branches,
            "mean_active_learning_queried": float(np.mean([item.get("active_learning_queried", 0.0) for item in history])) if history else 0.0,
            "mean_active_learning_revealed": float(np.mean([item.get("active_learning_revealed", 0.0) for item in history])) if history else 0.0,
            "result_root": str(resolved_result_root),
            "result_path": str(result_prefix),
            "model_path": str(model_path),
            "diagnostics_path": str(diagnostics_path),
            "final_diagnostics": final_diagnostics,
            "data_summary": data_summary,
            "resource_guard": resource_guard,
            "important_parameters": _main_important_parameters(
                args,
                dataset_name=dataset_name,
                federated_rounds=federated_rounds,
                local_epochs=local_epochs,
            ),
        }
        if warm_start_payload is not None:
            summary["resume_best_metric_inheritance"] = str(resume_best_metric_inheritance)
            summary["resume_best_metric_reason"] = str(resume_best_metric_reason)
            summary["resume_reference_best_valid_metrics"] = resume_reference_best_metrics
            summary["resume_recheck_valid_metrics"] = resume_recheck_valid_metrics
        if export_embedding_viz:
            _log_progress(dataset_name, "finalize: starting embedding analysis export")
            embedding_export_start = time.perf_counter()
            from .embedding_analysis import export_embedding_analysis_for_graph

            embedding_summary = export_embedding_analysis_for_graph(
                dataset=dataset_name,
                run_id=run_id,
                model=global_model,
                graph=bundle.graph,
                output_root=resolved_result_root,
                device=args.device,
            )
            summary["embedding_analysis_summary"] = embedding_summary.get("summary_file", "")
            _log_progress(
                dataset_name,
                "finalize: completed embedding analysis export "
                f"in {time.perf_counter() - embedding_export_start:.2f}s",
            )
        _log_progress(dataset_name, f"finalize: writing checkpoint -> {model_path}")
        checkpoint_write_start = time.perf_counter()
        with args.stage_timer.track("checkpoint_write"):
            _atomic_torch_save(model_path, checkpoint_payload)
        _log_progress(
            dataset_name,
            "finalize: completed checkpoint write "
            f"in {time.perf_counter() - checkpoint_write_start:.2f}s",
        )
        summary["stage_timings"] = args.stage_timer.as_dict()
        _log_progress(dataset_name, f"finalize: writing summary json -> {summary_path}")
        summary_write_start = time.perf_counter()
        with args.stage_timer.track("json_write"):
            _atomic_write_json(summary_path, {"history": history, "summary": summary})
        _log_progress(
            dataset_name,
            "finalize: completed summary json write "
            f"in {time.perf_counter() - summary_write_start:.2f}s",
        )
        if live_progress_enabled:
            _atomic_write_json(live_summary_path, {"history": history, "summary": summary})
        if run_metadata_path is not None:
            _log_progress(dataset_name, f"finalize: writing run metadata -> {run_metadata_path}")
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
            _log_progress(dataset_name, "finalize: completed run metadata write")
        _log_progress(dataset_name, f"finalize: writing run status -> {run_status_path}")
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
                "live_progress_path": str(live_progress_path),
                "live_summary_path": str(live_summary_path),
                "live_resume_checkpoint_path": str(live_resume_checkpoint_path),
                "tb_logdir": tb_logdir,
                "planner_mode": normalized_planner_mode,
                "resolved_device": str(args.device),
                "seed": None if getattr(args, "seed", None) is None else int(args.seed),
                "test_auc": float(test_metrics["auc"]) if has_test_metrics else None,
                "test_evaluated": bool(has_test_metrics),
                "data_summary": data_summary,
                "resource_guard": resource_guard,
            },
        )
        _write_live_progress(status="completed", latest_round=len(history))
        _log_progress(dataset_name, "finalize: completed run status write")
        return summary
    except BaseException as error:
        interrupted_fallback_used = bool(best_round < 0 and peak_valid_round_any_round >= 0)
        interrupted_best_state = best_state
        interrupted_best_valid_auc = best_valid_auc
        interrupted_best_valid_gmean = best_valid_gmean
        interrupted_best_valid_pr_auc = best_valid_pr_auc
        interrupted_best_valid_recall_at_precision = best_valid_recall_at_precision
        interrupted_best_valid_f1_macro = best_valid_f1_macro
        interrupted_best_round = best_round
        interrupted_best_valid_threshold = best_valid_threshold
        if interrupted_fallback_used:
            interrupted_best_state = peak_state
            interrupted_best_valid_auc = peak_valid_auc_any_round
            interrupted_best_round = peak_valid_round_any_round
            interrupted_best_valid_threshold = peak_valid_threshold_any_round
        if history:
            interrupted_checkpoint = {
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
                "args": vars(args),
                "planner_mode": normalized_planner_mode,
                "ablation_mode": str(getattr(args, "ablation_mode", "full")),
                "relation_order": bundle.relation_order,
                "run_id": run_id,
                "tb_logdir": tb_logdir,
                "completed": False,
                "interrupted": True,
            }
            if warm_start_payload is not None:
                interrupted_checkpoint["resume_metrics_rebased"] = bool(resume_metrics_rebased)
                interrupted_checkpoint["resume_best_metric_inheritance"] = str(resume_best_metric_inheritance)
                interrupted_checkpoint["resume_best_metric_reason"] = str(resume_best_metric_reason)
                interrupted_checkpoint["resume_reference_best_valid_metrics"] = resume_reference_best_metrics
                interrupted_checkpoint["resume_recheck_valid_metrics"] = resume_recheck_valid_metrics
            _atomic_torch_save(model_path, interrupted_checkpoint)
        final_supervised_train_nodes = int(bundle.graph.ndata["train_supervised_mask"].sum().item())
        final_unlabeled_train_nodes = int(bundle.graph.ndata["train_unlabeled_mask"].sum().item())
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
            "splitgnn_runtime_policy": bool(getattr(args, "splitgnn_runtime_policy", False)),
            "runtime_policy_notes": list(getattr(args, "runtime_policy_notes", [])),
            "resolved_device": str(args.device),
            "seed": None if getattr(args, "seed", None) is None else int(args.seed),
            "requested_classification_loss": str(getattr(args, "requested_classification_loss", getattr(args, "classification_loss", "cb_focal"))),
            "classification_loss": str(getattr(args, "classification_loss", "cb_focal")),
            "graph_aux_loss_weight": float(getattr(args, "graph_aux_loss_weight", 0.0)),
            "sequence_aux_loss_weight": float(getattr(args, "sequence_aux_loss_weight", 0.0)),
            "graph_anchor_loss_weight": float(getattr(args, "graph_anchor_loss_weight", 0.0)),
            "graph_anchor_temperature": float(getattr(args, "graph_anchor_temperature", 1.5)),
            "graph_teacher_checkpoint_path": str(getattr(args, "graph_teacher_checkpoint_path", "")),
            "graph_teacher_distill_weight": float(getattr(args, "graph_teacher_distill_weight", 0.0)),
            "graph_teacher_temperature": float(getattr(args, "graph_teacher_temperature", 1.5)),
            "graph_gate_logit_bias": float(getattr(args, "graph_gate_logit_bias", 0.0)),
            "eval_graph_gate_logit_bias": float(getattr(args, "eval_graph_gate_logit_bias", 0.0)),
            "graph_residual_min_gate": float(getattr(args, "graph_residual_min_gate", 0.0)),
            "sequence_residual_scale": float(getattr(args, "sequence_residual_scale", 1.0)),
            "fixed_precision_target": float(getattr(args, "fixed_precision_target", 0.5)),
            "test_evaluated": False,
            "best_round": interrupted_best_round,
            "best_valid_acc": float(resume_recheck_valid_metrics.get("acc", 0.0)) if resume_recheck_valid_metrics else None,
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
            "rounds_ran": len(history),
            "num_clients": len(bundle.clients),
            "initial_supervised_train_nodes": int(initial_supervised_train_nodes),
            "initial_unlabeled_train_nodes": int(initial_unlabeled_train_nodes),
            "final_supervised_train_nodes": int(final_supervised_train_nodes),
            "final_unlabeled_train_nodes": int(final_unlabeled_train_nodes),
            "result_root": str(resolved_result_root),
            "result_path": str(result_prefix),
            "model_path": str(model_path),
            "mean_controller_reward": float(np.mean([item.get("controller_reward", 0.0) for item in history])) if history else 0.0,
            "error": f"{type(error).__name__}: {error}",
            "data_summary": data_summary,
            "resource_guard": resource_guard,
            "important_parameters": _main_important_parameters(
                args,
                dataset_name=dataset_name,
                federated_rounds=federated_rounds,
                local_epochs=local_epochs,
            ),
        }
        if warm_start_payload is not None:
            interrupted_summary["resume_metrics_rebased"] = bool(resume_metrics_rebased)
            interrupted_summary["resume_best_metric_inheritance"] = str(resume_best_metric_inheritance)
            interrupted_summary["resume_best_metric_reason"] = str(resume_best_metric_reason)
            interrupted_summary["resume_reference_best_valid_metrics"] = resume_reference_best_metrics
            interrupted_summary["resume_recheck_valid_metrics"] = resume_recheck_valid_metrics
        interrupted_summary["stage_timings"] = args.stage_timer.as_dict()
        with args.stage_timer.track("json_write"):
            _atomic_write_json(summary_path, {"history": history, "summary": interrupted_summary})
        if live_progress_enabled:
            _atomic_write_json(live_summary_path, {"history": history, "summary": interrupted_summary})
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
                "live_progress_path": str(live_progress_path),
                "live_summary_path": str(live_summary_path),
                "live_resume_checkpoint_path": str(live_resume_checkpoint_path),
                "tb_logdir": tb_logdir,
                "planner_mode": normalized_planner_mode,
                "resolved_device": str(args.device),
                "seed": None if getattr(args, "seed", None) is None else int(args.seed),
                "error": f"{type(error).__name__}: {error}",
                "data_summary": data_summary,
                "resource_guard": resource_guard,
            },
        )
        _write_live_progress(
            status="interrupted",
            latest_round=len(history),
            error_message=f"{type(error).__name__}: {error}",
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


def train_hybrid_fraud_pipeline(
    federated_rounds: int = DEFAULT_HYBRID_MAINLINE_ROUNDS,
    base_local_epochs: int = 2,
    extra_local_epochs: int = 1,
    edge_loss_weight: float = 1.0,
    dataset: str = "all",
    num_clients: int = 3,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    rl_timesteps: int = 0,
    device: str = DEFAULT_DEVICE_REQUEST,
    amp_dtype: str = "auto",
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
    seq_hidden_dim: int = 64,
    fusion_hidden_dim: int = 64,
    planner_mode: str = "deterministic",
    early_stop: int = 0,
    test_every: int = 0,
    fixed_precision_target: float = 0.5,
    resume_path: str = "",
    resume_round_offset: int = 0,
    total_target_rounds: int | None = None,
    preload_history: list[dict[str, object]] | None = None,
    export_embedding_viz: bool = False,
    transformer_hidden_dim: int | None = None,
    transformer_num_layers: int = 1,
    sequence_batch_chunk_size: int | None = None,
    event_batch_chunk_size: int | None = None,
    transformer_activation_checkpointing: bool = True,
    active_learning_feedback_path: str = "",
    profile_ieee_full_gpu: bool = False,
    ieee_data_root: str = "",
    ieee_data_profile: str = "light_v1",
    ieee_loader_view: str = "hybrid",
    ieee_relation_profile: str = "core",
    ieee_feature_profile: str = "typed_256",
    ieee_history_len: int = 6,
    ieee_sampling_profile: str = "fraud_hardneg",
    ieee_max_transactions: int | None = None,
    ieee_time_bins: int = 24,
    ieee_relation_window_neighbors: int = 2,
    ieee_train_ratio: float = 0.70,
    ieee_valid_ratio: float = 0.15,
    ieee_full_compact_sequences: bool = True,
    ieee_sequence_feature_dim: int = 64,
    ieee_event_feature_dim: int = 64,
    ieee_build_light_cache_only: bool = False,
    ieee_rebuild_light_cache: bool = False,
    ieee_build_cache_only: bool = False,
    ieee_rebuild_cache: bool = False,
    ieee_skip_training: bool = False,
    amlsim_data_root: str = "",
    amlsim_train_ratio: float = 0.70,
    amlsim_valid_ratio: float = 0.15,
    amlsim_relation_window_neighbors: int = 4,
    amlsim_activity_bins: int = 8,
    amlsim_event_history_len: int = 12,
    amlsim_rebuild_cache: bool = False,
    amlsim_allow_sample_fallback: bool = False,
    amlsim_diffusion_residual_scale: float = 0.18,
    amlsim_pseudo_refresh_interval: int = 0,
    amlsim_pseudo_refresh_start_round: int = 0,
    amlsim_pseudo_refresh_momentum: float = 0.65,
    amlsim_pseudo_refresh_max_fraction: float = 0.0,
    amlsim_coassociation_loss_weight: float = 0.0,
    amlsim_wavelet_loss_weight: float = 0.0,
    amlsim_utg_align_loss_weight: float = 0.0,
    elliptic_data_root: str = "",
    elliptic_train_time_end: int = 34,
    elliptic_valid_time_end: int = 39,
    elliptic_history_len: int = 8,
    elliptic_sequence_topk: int = 8,
    elliptic_coassociation_topk: int = 3,
    elliptic_coassociation_time_window: int = 2,
    elliptic_use_unknown_ssl: bool = True,
    elliptic_rebuild_cache: bool = False,
    elliptic_pseudo_refresh_interval: int = 4,
    elliptic_pseudo_refresh_start_round: int = 4,
    elliptic_pseudo_refresh_momentum: float = 0.65,
    elliptic_pseudo_refresh_max_fraction: float = 0.10,
    elliptic_diffusion_residual_scale: float = 0.18,
    elliptic_coassociation_loss_weight: float = 0.05,
    elliptic_wavelet_loss_weight: float = 0.03,
    elliptic_utg_align_loss_weight: float = 0.04,
    ethereum_phishing_data_root: str = "",
    ethereum_phishing_max_users: int | None = None,
    ethereum_phishing_max_transactions: int | None = None,
    ethereum_phishing_force_preview: bool = False,
    ethereum_ponzi_data_root: str = "",
    ethereum_ponzi_negative_users_path: str = "",
    ethereum_ponzi_force_preview: bool = False,
    defi_rug_pull_data_root: str = "",
    defi_rug_pull_negative_users_path: str = "",
    defi_rug_pull_force_preview: bool = False,
    seed: int | None = None,
    result_root: str = "",
    disable_gnn: bool = False,
    disable_transformer: bool = False,
    disable_federated: bool = False,
    disable_relation_sequence_encoder: bool = False,
    disable_event_transformer_encoder: bool = False,
    disable_temporal_context_encoder: bool = False,
    disable_graph_temporal_fusion: bool = False,
    force_disable_wavelet_lite: bool = False,
    force_disable_utg_lite: bool = False,
    force_disable_coassociation: bool = False,
    force_disable_diffusion_residual: bool = False,
    learning_rate_override: float | None = None,
    weight_decay_override: float | None = None,
    dropout_override: float | None = None,
    graph_aux_loss_weight_override: float | None = None,
    sequence_aux_loss_weight_override: float | None = None,
    graph_gate_logit_bias_override: float | None = None,
    eval_graph_gate_logit_bias_override: float | None = None,
    graph_residual_min_gate_override: float | None = None,
    sequence_residual_scale_override: float | None = None,
    preferred_eval_branch_override: str | None = None,
    eval_branch_priority_override: str | list[str] | tuple[str, ...] | None = None,
    fusion_variant_override: str | None = None,
    modality_dropout_prob_override: float | None = None,
    graph_learning_rate_scale_override: float | None = None,
    sequence_learning_rate_scale_override: float | None = None,
    fusion_learning_rate_scale_override: float | None = None,
    graph_follow_learning_rate_scale_override: float | None = None,
    graph_warmup_rounds_override: int | None = None,
    fusion_bootstrap_rounds_override: int | None = None,
    teacher_ema_decay_override: float | None = None,
    pseudo_warmup_rounds_override: int | None = None,
    pseudo_ramp_rounds_override: int | None = None,
    open_set_novelty_threshold_override: float | None = None,
    open_set_loss_weight_override: float | None = None,
    prototype_loss_weight_override: float | None = None,
    shared_private_loss_weight_override: float | None = None,
    context_alignment_loss_weight_override: float | None = None,
    uncertainty_loss_weight_override: float | None = None,
    graph_anchor_loss_weight_override: float | None = None,
    graph_anchor_temperature_override: float | None = None,
    graph_teacher_checkpoint_path: str = "",
    graph_teacher_distill_weight: float = 0.0,
    graph_teacher_temperature: float = 1.5,
    legacy_fusion_only_override: bool | None = None,
    skip_test_evaluation: bool = False,
    lightweight_valid_eval: bool = False,
    epoch_metric_recompute_mode: str | None = None,
    pure_label_fraction: bool = False,
) -> dict:
    """Compatibility wrapper for older scripts."""
    base_seed = 42 if seed is None else int(seed)
    setup_seed(base_seed)
    datasets = resolve_requested_datasets(dataset)
    if (
        bool(ieee_build_light_cache_only)
        or bool(ieee_rebuild_light_cache)
        or bool(ieee_build_cache_only)
        or bool(ieee_rebuild_cache)
        or bool(ieee_skip_training)
    ) and datasets != ["ieee"]:
        raise ValueError(
            "IEEE cache-control flags are only supported when the active dataset selection is exactly ['ieee']."
        )
    normalized_negative_users_path = str(ethereum_ponzi_negative_users_path).strip()
    normalized_defi_negative_users_path = str(defi_rug_pull_negative_users_path).strip()
    if "ethereum_ponzi" in datasets and not normalized_negative_users_path:
        raise ValueError(
            "Selections including 'ethereum_ponzi' require "
            "`ethereum_ponzi_negative_users_path=...` to provide an external negative set."
        )
    if "defi_rug_pull" in datasets and not normalized_defi_negative_users_path:
        raise ValueError(
            "Selections including 'defi_rug_pull' require "
            "`defi_rug_pull_negative_users_path=...` to provide an external negative set."
        )
    resolved_result_root = _resolve_result_root(result_root)
    os.makedirs(resolved_result_root, exist_ok=True)
    normalized_planner_mode = str(planner_mode).lower()
    if normalized_planner_mode not in {"rl", "deterministic"}:
        raise ValueError(f"Unsupported planner mode: {planner_mode}")
    if len(datasets) > 1 and str(resume_path).strip():
        raise ValueError(
            "When a multi-dataset selector is used, --resume_path must be empty to avoid checkpoint-dataset mismatch."
        )
    summaries = {}
    dataset_runtime_policies = {
        dataset_name: resolve_splitgnn_runtime_policy(
            dataset_name=dataset_name,
            planner_mode=normalized_planner_mode,
            disable_federated=disable_federated,
        )
        for dataset_name in datasets
    }
    needs_shared_rl_models = any(
        policy["effective_planner_mode"] == "rl" for policy in dataset_runtime_policies.values()
    )
    shared_rl_models = train_controller_models(rl_timesteps=rl_timesteps, seed=base_seed) if needs_shared_rl_models else None
    try:
        for dataset_name in datasets:
            setup_seed(base_seed)
            summaries[dataset_name] = _train_one_dataset(
                dataset_name=dataset_name,
                federated_rounds=federated_rounds,
                local_epochs=max(base_local_epochs, 1) + max(extra_local_epochs - 1, 0),
                requested_base_local_epochs=base_local_epochs,
                requested_extra_local_epochs=extra_local_epochs,
                num_clients=num_clients,
                client_hops=client_hops,
                label_fraction=label_fraction,
                rl_timesteps=rl_timesteps,
                device=device,
                amp_dtype=amp_dtype,
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
                seq_hidden_dim=seq_hidden_dim,
                fusion_hidden_dim=fusion_hidden_dim,
                planner_mode=normalized_planner_mode,
                early_stop=early_stop,
                test_every=test_every,
                fixed_precision_target=fixed_precision_target,
                resume_path=resume_path,
                resume_round_offset=resume_round_offset,
                total_target_rounds=total_target_rounds,
                preload_history=preload_history,
                enable_tensorboard=enable_tensorboard,
                export_embedding_viz=export_embedding_viz,
                transformer_hidden_dim=transformer_hidden_dim,
                transformer_num_layers=transformer_num_layers,
                sequence_batch_chunk_size=sequence_batch_chunk_size,
                event_batch_chunk_size=event_batch_chunk_size,
                transformer_activation_checkpointing=transformer_activation_checkpointing,
                active_learning_feedback_path=active_learning_feedback_path,
                profile_ieee_full_gpu=profile_ieee_full_gpu,
                ieee_data_root=ieee_data_root,
                ieee_data_profile=ieee_data_profile,
                ieee_loader_view=ieee_loader_view,
                ieee_relation_profile=ieee_relation_profile,
                ieee_feature_profile=ieee_feature_profile,
                ieee_history_len=ieee_history_len,
                ieee_sampling_profile=ieee_sampling_profile,
                ieee_max_transactions=ieee_max_transactions,
                ieee_time_bins=ieee_time_bins,
                ieee_relation_window_neighbors=ieee_relation_window_neighbors,
                ieee_train_ratio=ieee_train_ratio,
                ieee_valid_ratio=ieee_valid_ratio,
                ieee_full_compact_sequences=ieee_full_compact_sequences,
                ieee_sequence_feature_dim=ieee_sequence_feature_dim,
                ieee_event_feature_dim=ieee_event_feature_dim,
                ieee_build_light_cache_only=ieee_build_light_cache_only,
                ieee_rebuild_light_cache=ieee_rebuild_light_cache,
                ieee_build_cache_only=ieee_build_cache_only,
                ieee_rebuild_cache=ieee_rebuild_cache,
                ieee_skip_training=ieee_skip_training,
                amlsim_data_root=amlsim_data_root,
                amlsim_train_ratio=amlsim_train_ratio,
                amlsim_valid_ratio=amlsim_valid_ratio,
                amlsim_relation_window_neighbors=amlsim_relation_window_neighbors,
                amlsim_activity_bins=amlsim_activity_bins,
                amlsim_event_history_len=amlsim_event_history_len,
                amlsim_rebuild_cache=amlsim_rebuild_cache,
                amlsim_allow_sample_fallback=amlsim_allow_sample_fallback,
                amlsim_diffusion_residual_scale=amlsim_diffusion_residual_scale,
                amlsim_pseudo_refresh_interval=amlsim_pseudo_refresh_interval,
                amlsim_pseudo_refresh_start_round=amlsim_pseudo_refresh_start_round,
                amlsim_pseudo_refresh_momentum=amlsim_pseudo_refresh_momentum,
                amlsim_pseudo_refresh_max_fraction=amlsim_pseudo_refresh_max_fraction,
                amlsim_coassociation_loss_weight=amlsim_coassociation_loss_weight,
                amlsim_wavelet_loss_weight=amlsim_wavelet_loss_weight,
                amlsim_utg_align_loss_weight=amlsim_utg_align_loss_weight,
                elliptic_data_root=elliptic_data_root,
                elliptic_train_time_end=elliptic_train_time_end,
                elliptic_valid_time_end=elliptic_valid_time_end,
                elliptic_history_len=elliptic_history_len,
                elliptic_sequence_topk=elliptic_sequence_topk,
                elliptic_coassociation_topk=elliptic_coassociation_topk,
                elliptic_coassociation_time_window=elliptic_coassociation_time_window,
                elliptic_use_unknown_ssl=elliptic_use_unknown_ssl,
                elliptic_rebuild_cache=elliptic_rebuild_cache,
                elliptic_pseudo_refresh_interval=elliptic_pseudo_refresh_interval,
                elliptic_pseudo_refresh_start_round=elliptic_pseudo_refresh_start_round,
                elliptic_pseudo_refresh_momentum=elliptic_pseudo_refresh_momentum,
                elliptic_pseudo_refresh_max_fraction=elliptic_pseudo_refresh_max_fraction,
                elliptic_diffusion_residual_scale=elliptic_diffusion_residual_scale,
                elliptic_coassociation_loss_weight=elliptic_coassociation_loss_weight,
                elliptic_wavelet_loss_weight=elliptic_wavelet_loss_weight,
                elliptic_utg_align_loss_weight=elliptic_utg_align_loss_weight,
                ethereum_phishing_data_root=ethereum_phishing_data_root,
                ethereum_phishing_max_users=ethereum_phishing_max_users,
                ethereum_phishing_max_transactions=ethereum_phishing_max_transactions,
                ethereum_phishing_force_preview=ethereum_phishing_force_preview,
                ethereum_ponzi_data_root=ethereum_ponzi_data_root,
                ethereum_ponzi_negative_users_path=normalized_negative_users_path,
                ethereum_ponzi_force_preview=ethereum_ponzi_force_preview,
                defi_rug_pull_data_root=defi_rug_pull_data_root,
                defi_rug_pull_negative_users_path=normalized_defi_negative_users_path,
                defi_rug_pull_force_preview=defi_rug_pull_force_preview,
                rl_models=shared_rl_models,
                seed=seed,
                result_root=str(resolved_result_root),
                disable_gnn=disable_gnn,
                disable_transformer=disable_transformer,
                disable_federated=disable_federated,
                disable_relation_sequence_encoder=disable_relation_sequence_encoder,
                disable_event_transformer_encoder=disable_event_transformer_encoder,
                disable_temporal_context_encoder=disable_temporal_context_encoder,
                disable_graph_temporal_fusion=disable_graph_temporal_fusion,
                force_disable_wavelet_lite=force_disable_wavelet_lite,
                force_disable_utg_lite=force_disable_utg_lite,
                force_disable_coassociation=force_disable_coassociation,
                force_disable_diffusion_residual=force_disable_diffusion_residual,
                learning_rate_override=learning_rate_override,
                weight_decay_override=weight_decay_override,
                dropout_override=dropout_override,
                graph_aux_loss_weight_override=graph_aux_loss_weight_override,
                sequence_aux_loss_weight_override=sequence_aux_loss_weight_override,
                graph_gate_logit_bias_override=graph_gate_logit_bias_override,
                eval_graph_gate_logit_bias_override=eval_graph_gate_logit_bias_override,
                graph_residual_min_gate_override=graph_residual_min_gate_override,
                sequence_residual_scale_override=sequence_residual_scale_override,
                preferred_eval_branch_override=preferred_eval_branch_override,
                eval_branch_priority_override=eval_branch_priority_override,
                fusion_variant_override=fusion_variant_override,
                modality_dropout_prob_override=modality_dropout_prob_override,
                graph_learning_rate_scale_override=graph_learning_rate_scale_override,
                sequence_learning_rate_scale_override=sequence_learning_rate_scale_override,
                fusion_learning_rate_scale_override=fusion_learning_rate_scale_override,
                graph_follow_learning_rate_scale_override=graph_follow_learning_rate_scale_override,
                graph_warmup_rounds_override=graph_warmup_rounds_override,
                fusion_bootstrap_rounds_override=fusion_bootstrap_rounds_override,
                teacher_ema_decay_override=teacher_ema_decay_override,
                pseudo_warmup_rounds_override=pseudo_warmup_rounds_override,
                pseudo_ramp_rounds_override=pseudo_ramp_rounds_override,
                open_set_novelty_threshold_override=open_set_novelty_threshold_override,
                open_set_loss_weight_override=open_set_loss_weight_override,
                prototype_loss_weight_override=prototype_loss_weight_override,
                shared_private_loss_weight_override=shared_private_loss_weight_override,
                context_alignment_loss_weight_override=context_alignment_loss_weight_override,
                uncertainty_loss_weight_override=uncertainty_loss_weight_override,
                graph_anchor_loss_weight_override=graph_anchor_loss_weight_override,
                graph_anchor_temperature_override=graph_anchor_temperature_override,
                graph_teacher_checkpoint_path=graph_teacher_checkpoint_path,
                graph_teacher_distill_weight=graph_teacher_distill_weight,
                graph_teacher_temperature=graph_teacher_temperature,
                legacy_fusion_only_override=legacy_fusion_only_override,
                skip_test_evaluation=skip_test_evaluation,
                lightweight_valid_eval=lightweight_valid_eval,
                epoch_metric_recompute_mode=epoch_metric_recompute_mode,
                pure_label_fraction=pure_label_fraction,
            )
    finally:
        if shared_rl_models is not None:
            _close_rl_models(shared_rl_models)

    report_path = resolved_result_root / "hybrid_fraudgraph_run_summary.json"
    _atomic_write_json(report_path, summaries)
    return summaries


def MAFRL(
    T_SDN: int = DEFAULT_HYBRID_MAINLINE_ROUNDS,
    T_UCH: int = 2,
    T_MID: int = 1,
    eta_1: float = 1.0,
    eta_2: float = 0.5,
    **kwargs,
) -> dict:
    """Compatibility wrapper for older scripts.

    `eta_2` is ignored because the active hybrid pipeline no longer uses the
    legacy second edge-loss coefficient.
    """
    _ = eta_2
    return train_hybrid_fraud_pipeline(
        federated_rounds=T_SDN,
        base_local_epochs=T_UCH,
        extra_local_epochs=T_MID,
        edge_loss_weight=eta_1,
        **kwargs,
    )


def run_hybrid_fraud_training(
    federated_rounds: int = DEFAULT_HYBRID_MAINLINE_ROUNDS,
    local_epochs: int = 2,
    extra_local_epochs: int = 1,
    edge_loss_weight: float = 1.0,
    dataset: str = "all",
    num_clients: int = 3,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    rl_timesteps: int = 0,
    device: str = DEFAULT_DEVICE_REQUEST,
    amp_dtype: str = "auto",
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
    seq_hidden_dim: int = 64,
    fusion_hidden_dim: int = 64,
    planner_mode: str = "deterministic",
    early_stop: int = 0,
    test_every: int = 0,
    fixed_precision_target: float = 0.5,
    resume_path: str = "",
    resume_round_offset: int = 0,
    total_target_rounds: int | None = None,
    preload_history: list[dict[str, object]] | None = None,
    export_embedding_viz: bool = False,
    transformer_hidden_dim: int | None = None,
    transformer_num_layers: int = 1,
    sequence_batch_chunk_size: int | None = None,
    event_batch_chunk_size: int | None = None,
    transformer_activation_checkpointing: bool = True,
    active_learning_feedback_path: str = "",
    profile_ieee_full_gpu: bool = False,
    ieee_data_root: str = "",
    ieee_data_profile: str = "light_v1",
    ieee_loader_view: str = "hybrid",
    ieee_relation_profile: str = "core",
    ieee_feature_profile: str = "typed_256",
    ieee_history_len: int = 6,
    ieee_sampling_profile: str = "fraud_hardneg",
    ieee_max_transactions: int | None = None,
    ieee_time_bins: int = 24,
    ieee_relation_window_neighbors: int = 2,
    ieee_train_ratio: float = 0.70,
    ieee_valid_ratio: float = 0.15,
    ieee_full_compact_sequences: bool = True,
    ieee_sequence_feature_dim: int = 64,
    ieee_event_feature_dim: int = 64,
    ieee_build_light_cache_only: bool = False,
    ieee_rebuild_light_cache: bool = False,
    ieee_build_cache_only: bool = False,
    ieee_rebuild_cache: bool = False,
    ieee_skip_training: bool = False,
    amlsim_data_root: str = "",
    amlsim_train_ratio: float = 0.70,
    amlsim_valid_ratio: float = 0.15,
    amlsim_relation_window_neighbors: int = 4,
    amlsim_activity_bins: int = 8,
    amlsim_event_history_len: int = 12,
    amlsim_rebuild_cache: bool = False,
    amlsim_allow_sample_fallback: bool = False,
    amlsim_diffusion_residual_scale: float = 0.18,
    amlsim_pseudo_refresh_interval: int = 0,
    amlsim_pseudo_refresh_start_round: int = 0,
    amlsim_pseudo_refresh_momentum: float = 0.65,
    amlsim_pseudo_refresh_max_fraction: float = 0.0,
    amlsim_coassociation_loss_weight: float = 0.0,
    amlsim_wavelet_loss_weight: float = 0.0,
    amlsim_utg_align_loss_weight: float = 0.0,
    elliptic_data_root: str = "",
    elliptic_train_time_end: int = 34,
    elliptic_valid_time_end: int = 39,
    elliptic_history_len: int = 8,
    elliptic_sequence_topk: int = 8,
    elliptic_coassociation_topk: int = 3,
    elliptic_coassociation_time_window: int = 2,
    elliptic_use_unknown_ssl: bool = True,
    elliptic_rebuild_cache: bool = False,
    elliptic_pseudo_refresh_interval: int = 4,
    elliptic_pseudo_refresh_start_round: int = 4,
    elliptic_pseudo_refresh_momentum: float = 0.65,
    elliptic_pseudo_refresh_max_fraction: float = 0.10,
    elliptic_diffusion_residual_scale: float = 0.18,
    elliptic_coassociation_loss_weight: float = 0.05,
    elliptic_wavelet_loss_weight: float = 0.03,
    elliptic_utg_align_loss_weight: float = 0.04,
    ethereum_phishing_data_root: str = "",
    ethereum_phishing_max_users: int | None = None,
    ethereum_phishing_max_transactions: int | None = None,
    ethereum_phishing_force_preview: bool = False,
    ethereum_ponzi_data_root: str = "",
    ethereum_ponzi_negative_users_path: str = "",
    ethereum_ponzi_force_preview: bool = False,
    defi_rug_pull_data_root: str = "",
    defi_rug_pull_negative_users_path: str = "",
    defi_rug_pull_force_preview: bool = False,
    seed: int | None = None,
    result_root: str = "",
    disable_gnn: bool = False,
    disable_transformer: bool = False,
    disable_federated: bool = False,
    disable_relation_sequence_encoder: bool = False,
    disable_event_transformer_encoder: bool = False,
    disable_temporal_context_encoder: bool = False,
    disable_graph_temporal_fusion: bool = False,
    force_disable_wavelet_lite: bool = False,
    force_disable_utg_lite: bool = False,
    force_disable_coassociation: bool = False,
    force_disable_diffusion_residual: bool = False,
    learning_rate_override: float | None = None,
    weight_decay_override: float | None = None,
    dropout_override: float | None = None,
    graph_aux_loss_weight_override: float | None = None,
    sequence_aux_loss_weight_override: float | None = None,
    graph_gate_logit_bias_override: float | None = None,
    eval_graph_gate_logit_bias_override: float | None = None,
    graph_residual_min_gate_override: float | None = None,
    sequence_residual_scale_override: float | None = None,
    preferred_eval_branch_override: str | None = None,
    eval_branch_priority_override: str | list[str] | tuple[str, ...] | None = None,
    fusion_variant_override: str | None = None,
    modality_dropout_prob_override: float | None = None,
    graph_learning_rate_scale_override: float | None = None,
    sequence_learning_rate_scale_override: float | None = None,
    fusion_learning_rate_scale_override: float | None = None,
    graph_follow_learning_rate_scale_override: float | None = None,
    graph_warmup_rounds_override: int | None = None,
    fusion_bootstrap_rounds_override: int | None = None,
    teacher_ema_decay_override: float | None = None,
    pseudo_warmup_rounds_override: int | None = None,
    pseudo_ramp_rounds_override: int | None = None,
    open_set_novelty_threshold_override: float | None = None,
    open_set_loss_weight_override: float | None = None,
    prototype_loss_weight_override: float | None = None,
    shared_private_loss_weight_override: float | None = None,
    context_alignment_loss_weight_override: float | None = None,
    uncertainty_loss_weight_override: float | None = None,
    graph_anchor_loss_weight_override: float | None = None,
    graph_anchor_temperature_override: float | None = None,
    graph_teacher_checkpoint_path: str = "",
    graph_teacher_distill_weight: float = 0.0,
    graph_teacher_temperature: float = 1.5,
    legacy_fusion_only_override: bool | None = None,
    skip_test_evaluation: bool = False,
    lightweight_valid_eval: bool = False,
    epoch_metric_recompute_mode: str | None = None,
    pure_label_fraction: bool = False,
) -> dict:
    """Main entry point for the active GNN + Transformer training pipeline."""
    return train_hybrid_fraud_pipeline(
        federated_rounds=federated_rounds,
        base_local_epochs=local_epochs,
        extra_local_epochs=extra_local_epochs,
        edge_loss_weight=edge_loss_weight,
        dataset=dataset,
        num_clients=num_clients,
        client_hops=client_hops,
        label_fraction=label_fraction,
        rl_timesteps=rl_timesteps,
        device=device,
        amp_dtype=amp_dtype,
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
        seq_hidden_dim=seq_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        planner_mode=planner_mode,
        early_stop=early_stop,
        test_every=test_every,
        fixed_precision_target=fixed_precision_target,
        resume_path=resume_path,
        resume_round_offset=resume_round_offset,
        total_target_rounds=total_target_rounds,
        preload_history=preload_history,
        export_embedding_viz=export_embedding_viz,
        transformer_hidden_dim=transformer_hidden_dim,
        transformer_num_layers=transformer_num_layers,
        sequence_batch_chunk_size=sequence_batch_chunk_size,
        event_batch_chunk_size=event_batch_chunk_size,
        transformer_activation_checkpointing=transformer_activation_checkpointing,
        active_learning_feedback_path=active_learning_feedback_path,
        profile_ieee_full_gpu=profile_ieee_full_gpu,
        ieee_data_root=ieee_data_root,
        ieee_data_profile=ieee_data_profile,
        ieee_loader_view=ieee_loader_view,
        ieee_relation_profile=ieee_relation_profile,
        ieee_feature_profile=ieee_feature_profile,
        ieee_history_len=ieee_history_len,
        ieee_sampling_profile=ieee_sampling_profile,
        ieee_max_transactions=ieee_max_transactions,
        ieee_time_bins=ieee_time_bins,
        ieee_relation_window_neighbors=ieee_relation_window_neighbors,
        ieee_train_ratio=ieee_train_ratio,
        ieee_valid_ratio=ieee_valid_ratio,
        ieee_full_compact_sequences=ieee_full_compact_sequences,
        ieee_sequence_feature_dim=ieee_sequence_feature_dim,
        ieee_event_feature_dim=ieee_event_feature_dim,
        ieee_build_light_cache_only=ieee_build_light_cache_only,
        ieee_rebuild_light_cache=ieee_rebuild_light_cache,
        ieee_build_cache_only=ieee_build_cache_only,
        ieee_rebuild_cache=ieee_rebuild_cache,
        ieee_skip_training=ieee_skip_training,
        amlsim_data_root=amlsim_data_root,
        amlsim_train_ratio=amlsim_train_ratio,
        amlsim_valid_ratio=amlsim_valid_ratio,
        amlsim_relation_window_neighbors=amlsim_relation_window_neighbors,
        amlsim_activity_bins=amlsim_activity_bins,
        amlsim_event_history_len=amlsim_event_history_len,
        amlsim_rebuild_cache=amlsim_rebuild_cache,
        amlsim_allow_sample_fallback=amlsim_allow_sample_fallback,
        amlsim_diffusion_residual_scale=amlsim_diffusion_residual_scale,
        amlsim_pseudo_refresh_interval=amlsim_pseudo_refresh_interval,
        amlsim_pseudo_refresh_start_round=amlsim_pseudo_refresh_start_round,
        amlsim_pseudo_refresh_momentum=amlsim_pseudo_refresh_momentum,
        amlsim_pseudo_refresh_max_fraction=amlsim_pseudo_refresh_max_fraction,
        amlsim_coassociation_loss_weight=amlsim_coassociation_loss_weight,
        amlsim_wavelet_loss_weight=amlsim_wavelet_loss_weight,
        amlsim_utg_align_loss_weight=amlsim_utg_align_loss_weight,
        elliptic_data_root=elliptic_data_root,
        elliptic_train_time_end=elliptic_train_time_end,
        elliptic_valid_time_end=elliptic_valid_time_end,
        elliptic_history_len=elliptic_history_len,
        elliptic_sequence_topk=elliptic_sequence_topk,
        elliptic_coassociation_topk=elliptic_coassociation_topk,
        elliptic_coassociation_time_window=elliptic_coassociation_time_window,
        elliptic_use_unknown_ssl=elliptic_use_unknown_ssl,
        elliptic_rebuild_cache=elliptic_rebuild_cache,
        elliptic_pseudo_refresh_interval=elliptic_pseudo_refresh_interval,
        elliptic_pseudo_refresh_start_round=elliptic_pseudo_refresh_start_round,
        elliptic_pseudo_refresh_momentum=elliptic_pseudo_refresh_momentum,
        elliptic_pseudo_refresh_max_fraction=elliptic_pseudo_refresh_max_fraction,
        elliptic_diffusion_residual_scale=elliptic_diffusion_residual_scale,
        elliptic_coassociation_loss_weight=elliptic_coassociation_loss_weight,
        elliptic_wavelet_loss_weight=elliptic_wavelet_loss_weight,
        elliptic_utg_align_loss_weight=elliptic_utg_align_loss_weight,
        ethereum_phishing_data_root=ethereum_phishing_data_root,
        ethereum_phishing_max_users=ethereum_phishing_max_users,
        ethereum_phishing_max_transactions=ethereum_phishing_max_transactions,
        ethereum_phishing_force_preview=ethereum_phishing_force_preview,
        ethereum_ponzi_data_root=ethereum_ponzi_data_root,
        ethereum_ponzi_negative_users_path=ethereum_ponzi_negative_users_path,
        ethereum_ponzi_force_preview=ethereum_ponzi_force_preview,
        defi_rug_pull_data_root=defi_rug_pull_data_root,
        defi_rug_pull_negative_users_path=defi_rug_pull_negative_users_path,
        defi_rug_pull_force_preview=defi_rug_pull_force_preview,
        seed=seed,
        result_root=result_root,
        disable_gnn=disable_gnn,
        disable_transformer=disable_transformer,
        disable_federated=disable_federated,
        disable_relation_sequence_encoder=disable_relation_sequence_encoder,
        disable_event_transformer_encoder=disable_event_transformer_encoder,
        disable_temporal_context_encoder=disable_temporal_context_encoder,
        disable_graph_temporal_fusion=disable_graph_temporal_fusion,
        force_disable_wavelet_lite=force_disable_wavelet_lite,
        force_disable_utg_lite=force_disable_utg_lite,
        force_disable_coassociation=force_disable_coassociation,
        force_disable_diffusion_residual=force_disable_diffusion_residual,
        learning_rate_override=learning_rate_override,
        weight_decay_override=weight_decay_override,
        dropout_override=dropout_override,
        graph_aux_loss_weight_override=graph_aux_loss_weight_override,
        sequence_aux_loss_weight_override=sequence_aux_loss_weight_override,
        graph_gate_logit_bias_override=graph_gate_logit_bias_override,
        eval_graph_gate_logit_bias_override=eval_graph_gate_logit_bias_override,
        graph_residual_min_gate_override=graph_residual_min_gate_override,
        sequence_residual_scale_override=sequence_residual_scale_override,
        preferred_eval_branch_override=preferred_eval_branch_override,
        eval_branch_priority_override=eval_branch_priority_override,
        fusion_variant_override=fusion_variant_override,
        modality_dropout_prob_override=modality_dropout_prob_override,
        graph_learning_rate_scale_override=graph_learning_rate_scale_override,
        sequence_learning_rate_scale_override=sequence_learning_rate_scale_override,
        fusion_learning_rate_scale_override=fusion_learning_rate_scale_override,
        graph_follow_learning_rate_scale_override=graph_follow_learning_rate_scale_override,
        graph_warmup_rounds_override=graph_warmup_rounds_override,
        fusion_bootstrap_rounds_override=fusion_bootstrap_rounds_override,
        teacher_ema_decay_override=teacher_ema_decay_override,
        pseudo_warmup_rounds_override=pseudo_warmup_rounds_override,
        pseudo_ramp_rounds_override=pseudo_ramp_rounds_override,
        open_set_novelty_threshold_override=open_set_novelty_threshold_override,
        open_set_loss_weight_override=open_set_loss_weight_override,
        prototype_loss_weight_override=prototype_loss_weight_override,
        shared_private_loss_weight_override=shared_private_loss_weight_override,
        context_alignment_loss_weight_override=context_alignment_loss_weight_override,
        uncertainty_loss_weight_override=uncertainty_loss_weight_override,
        graph_anchor_loss_weight_override=graph_anchor_loss_weight_override,
        graph_anchor_temperature_override=graph_anchor_temperature_override,
        graph_teacher_checkpoint_path=graph_teacher_checkpoint_path,
        graph_teacher_distill_weight=graph_teacher_distill_weight,
        graph_teacher_temperature=graph_teacher_temperature,
        legacy_fusion_only_override=legacy_fusion_only_override,
        skip_test_evaluation=skip_test_evaluation,
        lightweight_valid_eval=lightweight_valid_eval,
        epoch_metric_recompute_mode=epoch_metric_recompute_mode,
        pure_label_fraction=pure_label_fraction,
    )
