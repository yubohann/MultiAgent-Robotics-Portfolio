"""CLI entry point for the hybrid fraud training pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .cli_contract import (
    DATASET_SELECTION_CHOICES,
    DEFAULT_HYBRID_MAINLINE_ROUNDS,
    LEGACY_BATCH_DATASETS,
    SUPPORTED_HYBRID_DATASETS,
)
from .cli_contract import DEFAULT_DEVICE_REQUEST


def _add_core_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Register model-agnostic training, selection, and evaluation options."""

    parser.add_argument(
        "--dataset",
        type=str,
        default="elliptic",
        choices=list(DATASET_SELECTION_CHOICES),
        help=(
            f"Dataset to run. 'all' keeps the active runtime batch: {', '.join(LEGACY_BATCH_DATASETS)}. "
            f"'all_supported' runs every wired dataset: {', '.join(SUPPORTED_HYBRID_DATASETS)}."
        ),
    )
    parser.add_argument(
        "--federated_rounds",
        "--rounds",
        dest="federated_rounds",
        type=int,
        default=DEFAULT_HYBRID_MAINLINE_ROUNDS,
        help="Number of training rounds to run on the active mainline.",
    )
    parser.add_argument(
        "--base_local_epochs",
        "--local_epochs",
        dest="base_local_epochs",
        type=int,
        default=2,
        help="Base epochs per round on the active mainline.",
    )
    parser.add_argument(
        "--extra_local_epochs",
        "--mid_epochs",
        dest="extra_local_epochs",
        type=int,
        default=1,
        help="Additional epochs used by the active training schedule.",
    )
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--client_hops", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument(
        "--controller_timesteps",
        "--rl_timesteps",
        dest="controller_timesteps",
        type=int,
        default=0,
        help="Legacy compatibility flag. Parsed for old scripts but ignored by the active non-RL mainline.",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE_REQUEST)
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "off"])
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--classification_loss",
        type=str,
        default="cb_focal",
        choices=["ce", "weighted_ce", "focal", "cb_ce", "cb_focal", "weighted_bce_auc"],
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--class_balance_beta", type=float, default=0.999)
    parser.add_argument("--pseudo_label_threshold", type=float, default=0.9)
    parser.add_argument("--pseudo_label_weight", type=float, default=0.15)
    parser.add_argument("--pseudo_label_novelty_threshold", type=float, default=2.5)
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.1,
        help="Consistency regularization weight. Use 0 to disable explicitly.",
    )
    parser.add_argument(
        "--active_learning_budget_per_round",
        type=int,
        default=-1,
        help="Per-round active-learning budget. Use 0 to disable, or a negative value to use the low-label profile default.",
    )
    parser.add_argument(
        "--active_learning_delay_rounds",
        type=int,
        default=-1,
        help="Reveal delay for active-learning labels. Use 0 to disable delay, or a negative value to use the low-label profile default.",
    )
    parser.add_argument(
        "--active_learning_novelty_weight",
        type=float,
        default=0.35,
        help="Novelty weight in active-learning acquisition. Use 0 to disable explicitly.",
    )
    parser.add_argument(
        "--active_learning_diversity_weight",
        type=float,
        default=0.25,
        help="Diversity weight in active-learning acquisition. Use 0 to disable explicitly.",
    )
    parser.add_argument("--active_learning_candidate_pool_scale", type=int, default=4)
    parser.add_argument("--fedprox_mu", type=float, default=0.01)
    parser.add_argument("--dp_noise_std", type=float, default=0.0)
    parser.add_argument(
        "--transformer_hidden_dim",
        "--seq_hidden_dim",
        dest="transformer_hidden_dim",
        type=int,
        default=64,
        help="Hidden width for the Transformer sequence encoder.",
    )
    parser.add_argument(
        "--transformer_num_layers",
        type=int,
        default=1,
        help="Number of Transformer encoder layers used in the sequence branch.",
    )
    parser.add_argument(
        "--sequence_batch_chunk_size",
        type=int,
        default=None,
        help="Optional batch chunk size for the relation-sequence encoder. Smaller values trade speed for memory.",
    )
    parser.add_argument(
        "--event_batch_chunk_size",
        type=int,
        default=None,
        help="Optional batch chunk size for the event-sequence encoder. Smaller values trade speed for memory.",
    )
    parser.add_argument(
        "--transformer_activation_checkpointing",
        dest="transformer_activation_checkpointing",
        action="store_true",
        default=True,
        help="Enable activation checkpointing for transformer encoders. The active IEEE-safe runtime keeps this on.",
    )
    parser.add_argument(
        "--no_transformer_activation_checkpointing",
        dest="transformer_activation_checkpointing",
        action="store_false",
        help="Disable transformer activation checkpointing. The active IEEE-safe runtime may override this to stay memory-safe.",
    )
    parser.add_argument("--fusion_hidden_dim", type=int, default=64)
    parser.add_argument("--graph_aux_loss_weight", type=float, default=None)
    parser.add_argument("--sequence_aux_loss_weight", type=float, default=None)
    parser.add_argument("--graph_gate_logit_bias", type=float, default=None)
    parser.add_argument("--eval_graph_gate_logit_bias", type=float, default=None)
    parser.add_argument("--graph_residual_min_gate", type=float, default=None)
    parser.add_argument("--sequence_residual_scale", type=float, default=None)
    parser.add_argument(
        "--preferred_eval_branch",
        type=str,
        default="",
        help="Optional evaluation branch override, for example `main` or `sequence_residual`.",
    )
    parser.add_argument(
        "--eval_branch_priority",
        type=str,
        default="",
        help="Optional comma-separated evaluation branch priority override.",
    )
    parser.add_argument(
        "--planner_mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "rl"],
        help="Legacy compatibility flag. Parsed for old scripts but forced to deterministic by the active mainline.",
    )
    parser.add_argument("--early_stop", type=int, default=0)
    parser.add_argument("--test_every", type=int, default=0)
    parser.add_argument(
        "--fixed_precision_target",
        type=float,
        default=0.5,
        help="Precision target used for Recall@Precision evaluation and checkpoint selection.",
    )
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--active_learning_feedback_path", type=str, default="")
    parser.add_argument("--result_root", type=str, default="")


def _add_ieee_arguments(parser: argparse.ArgumentParser) -> None:
    """Register IEEE-CIS data profiles, cache controls, and sequence settings."""

    parser.add_argument("--profile_ieee_full_gpu", action="store_true")
    parser.add_argument("--ieee_data_root", type=str, default="")
    parser.add_argument(
        "--ieee_data_profile",
        type=str,
        default="light_v1",
        choices=["raw", "light_v1", "light_v2", "tabular_full", "custom"],
        help="Controls IEEE data scale / asset family independently from runtime GPU profile.",
    )
    parser.add_argument(
        "--ieee_loader_view",
        type=str,
        default="hybrid",
        choices=["graph", "tabular", "sequence", "hybrid"],
        help="Primary IEEE asset view requested by the active consumer.",
    )
    parser.add_argument(
        "--ieee_relation_profile",
        type=str,
        default="core",
        choices=["core", "extended"],
        help="Controls which IEEE relation families are materialized into light graph assets.",
    )
    parser.add_argument(
        "--ieee_feature_profile",
        type=str,
        default="typed_256",
        choices=["typed_full", "typed_256", "typed_160", "paper_pruned", "paper_v30"],
        help="Controls IEEE feature pruning / compression profile.",
    )
    parser.add_argument(
        "--ieee_history_len",
        type=int,
        default=6,
        help="History budget used by IEEE sequence / hybrid assets.",
    )
    parser.add_argument(
        "--ieee_sampling_profile",
        type=str,
        default="fraud_hardneg",
        choices=["chrono_full", "chrono_stratified", "fraud_hardneg", "normal_only_train"],
        help="Controls IEEE manifest sampling policy before pass-2 subset loading.",
    )
    parser.add_argument(
        "--ieee_max_transactions",
        type=int,
        default=0,
        help="Optional explicit IEEE transaction cap. Use 0 to defer to --ieee_data_profile.",
    )
    parser.add_argument("--ieee_time_bins", type=int, default=24)
    parser.add_argument("--ieee_relation_window_neighbors", type=int, default=2)
    parser.add_argument("--ieee_train_ratio", type=float, default=0.70)
    parser.add_argument("--ieee_valid_ratio", type=float, default=0.15)
    parser.add_argument(
        "--ieee_full_compact_sequences",
        dest="ieee_full_compact_sequences",
        action="store_true",
        default=True,
        help="Enable IEEE full-profile compact base features for relation/event sequence branches.",
    )
    parser.add_argument(
        "--no_ieee_full_compact_sequences",
        dest="ieee_full_compact_sequences",
        action="store_false",
        help="Disable IEEE full-profile compact base features and keep sequence/event branches on raw full features.",
    )
    parser.add_argument(
        "--ieee_sequence_feature_dim",
        type=int,
        default=64,
        help="Compact feature width used by the IEEE relation-sequence branch when full compact mode is enabled.",
    )
    parser.add_argument(
        "--ieee_event_feature_dim",
        type=int,
        default=64,
        help="Compact feature width used by the IEEE event branch when full compact mode is enabled.",
    )
    parser.add_argument(
        "--ieee_build_light_cache_only",
        action="store_true",
        help="Build the new IEEE light-cache asset family and exit before training.",
    )
    parser.add_argument(
        "--ieee_rebuild_light_cache",
        action="store_true",
        help="Force rebuild of the new IEEE light-cache asset family before any model run.",
    )
    parser.add_argument(
        "--ieee_build_cache_only",
        action="store_true",
        help="For --dataset ieee only: build or validate the IEEE cache, write artifacts, then exit before model initialization.",
    )
    parser.add_argument(
        "--ieee_rebuild_cache",
        action="store_true",
        help="For --dataset ieee only: ignore any existing IEEE cache and rebuild the core graph cache plus artifact shards.",
    )
    parser.add_argument(
        "--ieee_skip_training",
        action="store_true",
        help="For --dataset ieee only: load the IEEE dataset/cache, then exit before training.",
    )


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """Register dataset-specific roots, sampling controls, and negative-set inputs."""

    parser.add_argument("--amlsim_data_root", type=str, default="")
    parser.add_argument("--amlsim_train_ratio", type=float, default=0.70)
    parser.add_argument("--amlsim_valid_ratio", type=float, default=0.15)
    parser.add_argument("--amlsim_relation_window_neighbors", type=int, default=4)
    parser.add_argument("--amlsim_activity_bins", type=int, default=8)
    parser.add_argument("--amlsim_event_history_len", type=int, default=12)
    parser.add_argument("--amlsim_rebuild_cache", action="store_true")
    parser.add_argument(
        "--amlsim_allow_sample_fallback",
        action="store_true",
        help="Allow AMLSim sample outputs when the requested root does not contain generated outputs yet.",
    )
    parser.add_argument("--elliptic_data_root", type=str, default="")
    parser.add_argument("--elliptic_train_time_end", type=int, default=34)
    parser.add_argument("--elliptic_valid_time_end", type=int, default=39)
    parser.add_argument("--elliptic_history_len", type=int, default=8)
    parser.add_argument("--elliptic_sequence_topk", type=int, default=8)
    parser.add_argument("--elliptic_coassociation_topk", type=int, default=3)
    parser.add_argument("--elliptic_coassociation_time_window", type=int, default=2)
    parser.add_argument("--elliptic_pseudo_refresh_interval", type=int, default=4)
    parser.add_argument("--elliptic_pseudo_refresh_start_round", type=int, default=4)
    parser.add_argument("--elliptic_pseudo_refresh_momentum", type=float, default=0.65)
    parser.add_argument("--elliptic_pseudo_refresh_max_fraction", type=float, default=0.10)
    parser.add_argument("--elliptic_diffusion_residual_scale", type=float, default=0.18)
    parser.add_argument("--elliptic_coassociation_loss_weight", type=float, default=0.05)
    parser.add_argument("--elliptic_wavelet_loss_weight", type=float, default=0.03)
    parser.add_argument("--elliptic_utg_align_loss_weight", type=float, default=0.04)
    parser.add_argument(
        "--elliptic_use_unknown_ssl",
        dest="elliptic_use_unknown_ssl",
        action="store_true",
        default=True,
        help="Keep unknown Elliptic train-window nodes as unlabeled SSL candidates.",
    )
    parser.add_argument(
        "--no_elliptic_use_unknown_ssl",
        dest="elliptic_use_unknown_ssl",
        action="store_false",
        help="Exclude unknown Elliptic train-window nodes from the SSL pool.",
    )
    parser.add_argument("--elliptic_rebuild_cache", action="store_true")
    parser.add_argument("--ethereum_phishing_data_root", type=str, default="")
    parser.add_argument("--ethereum_phishing_max_users", type=int, default=None)
    parser.add_argument("--ethereum_phishing_max_transactions", type=int, default=None)
    parser.add_argument("--ethereum_phishing_force_preview", action="store_true")
    parser.add_argument("--ethereum_ponzi_data_root", type=str, default="")
    parser.add_argument(
        "--ethereum_ponzi_negative_users_path",
        type=str,
        default="",
        help="Required whenever the selected dataset set includes ethereum_ponzi. Points to an external negative set.",
    )
    parser.add_argument("--ethereum_ponzi_force_preview", action="store_true")
    parser.add_argument("--defi_rug_pull_data_root", type=str, default="")
    parser.add_argument(
        "--defi_rug_pull_negative_users_path",
        type=str,
        default="",
        help="Required whenever the selected dataset set includes defi_rug_pull. Points to an external negative set.",
    )
    parser.add_argument("--defi_rug_pull_force_preview", action="store_true")


def _add_runtime_control_arguments(parser: argparse.ArgumentParser) -> None:
    """Register model-branch and output controls shared by all datasets."""

    parser.add_argument("--disable_gnn", action="store_true")
    parser.add_argument("--disable_transformer", action="store_true")
    parser.add_argument(
        "--disable_federated",
        action="store_true",
        help="Legacy compatibility flag. Parsed for old scripts but the active mainline already disables federated training.",
    )
    parser.add_argument("--export_embedding_viz", action="store_true")
    parser.add_argument("--skip_test_evaluation", action="store_true")
    parser.add_argument("--disable_tb", action="store_true")
    parser.add_argument(
        "--lightweight_valid_eval",
        action="store_true",
        help="Use main-branch-only validation during training rounds to reduce CPU overhead.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the CLI parser and parse an optional argument sequence."""

    parser = argparse.ArgumentParser(description="Run the GNN + Transformer fraud training pipeline.")
    _add_core_training_arguments(parser)
    _add_ieee_arguments(parser)
    _add_dataset_arguments(parser)
    _add_runtime_control_arguments(parser)
    return parser.parse_args(argv)


def _normalize_active_mainline_args(args: argparse.Namespace, selected_datasets: Sequence[str]) -> None:
    """Apply current deterministic-run policy while accepting legacy CLI flags."""

    if bool(getattr(args, "ieee_build_light_cache_only", False)):
        args.ieee_build_cache_only = True
    if bool(getattr(args, "ieee_rebuild_light_cache", False)):
        args.ieee_rebuild_cache = True
    if bool(args.ieee_build_cache_only):
        args.ieee_skip_training = True
    if (
        bool(args.ieee_build_cache_only)
        or bool(args.ieee_rebuild_cache)
        or bool(args.ieee_skip_training)
    ) and list(selected_datasets) != ["ieee"]:
        raise SystemExit(
            "IEEE cache-control flags (--ieee_build_cache_only, --ieee_rebuild_cache, --ieee_skip_training) "
            "require --dataset ieee."
        )
    if "ethereum_ponzi" in selected_datasets and not str(args.ethereum_ponzi_negative_users_path).strip():
        raise SystemExit(
            "Selections including ethereum_ponzi require --ethereum_ponzi_negative_users_path to point to an external negative set."
        )
    if "defi_rug_pull" in selected_datasets and not str(args.defi_rug_pull_negative_users_path).strip():
        raise SystemExit(
            "Selections including defi_rug_pull require --defi_rug_pull_negative_users_path to point to an external negative set."
        )

    args.controller_timesteps = 0
    args.planner_mode = "deterministic"
    args.disable_federated = True


def _core_training_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Map normalized CLI values to model-agnostic training API arguments."""

    return dict(
        federated_rounds=args.federated_rounds,
        local_epochs=args.base_local_epochs,
        extra_local_epochs=args.extra_local_epochs,
        edge_loss_weight=args.edge_loss_weight,
        dataset=args.dataset,
        num_clients=args.num_clients,
        client_hops=args.client_hops,
        seed=args.seed,
        label_fraction=args.label_fraction,
        rl_timesteps=0,
        device=args.device,
        amp_dtype=args.amp_dtype,
        enable_tensorboard=not args.disable_tb,
        classification_loss=args.classification_loss,
        focal_gamma=args.focal_gamma,
        class_balance_beta=args.class_balance_beta,
        pseudo_label_threshold=args.pseudo_label_threshold,
        pseudo_label_weight=args.pseudo_label_weight,
        pseudo_label_novelty_threshold=args.pseudo_label_novelty_threshold,
        consistency_weight=args.consistency_weight,
        active_learning_budget_per_round=args.active_learning_budget_per_round,
        active_learning_delay_rounds=args.active_learning_delay_rounds,
        active_learning_novelty_weight=args.active_learning_novelty_weight,
        active_learning_diversity_weight=args.active_learning_diversity_weight,
        active_learning_candidate_pool_scale=args.active_learning_candidate_pool_scale,
        fedprox_mu=args.fedprox_mu,
        dp_noise_std=args.dp_noise_std,
        transformer_hidden_dim=args.transformer_hidden_dim,
        transformer_num_layers=args.transformer_num_layers,
        sequence_batch_chunk_size=args.sequence_batch_chunk_size,
        event_batch_chunk_size=args.event_batch_chunk_size,
        transformer_activation_checkpointing=bool(args.transformer_activation_checkpointing),
        fusion_hidden_dim=args.fusion_hidden_dim,
        graph_aux_loss_weight_override=args.graph_aux_loss_weight,
        sequence_aux_loss_weight_override=args.sequence_aux_loss_weight,
        graph_gate_logit_bias_override=args.graph_gate_logit_bias,
        eval_graph_gate_logit_bias_override=args.eval_graph_gate_logit_bias,
        graph_residual_min_gate_override=args.graph_residual_min_gate,
        sequence_residual_scale_override=args.sequence_residual_scale,
        preferred_eval_branch_override=str(args.preferred_eval_branch).strip() or None,
        eval_branch_priority_override=str(args.eval_branch_priority).strip() or None,
        planner_mode="deterministic",
        early_stop=args.early_stop,
        test_every=args.test_every,
        fixed_precision_target=args.fixed_precision_target,
        resume_path=str(args.resume_path).strip(),
        active_learning_feedback_path=str(args.active_learning_feedback_path).strip(),
        disable_gnn=bool(args.disable_gnn),
        disable_transformer=bool(args.disable_transformer),
        disable_federated=True,
        export_embedding_viz=bool(args.export_embedding_viz),
        skip_test_evaluation=bool(args.skip_test_evaluation),
        lightweight_valid_eval=bool(args.lightweight_valid_eval),
    )


def _dataset_training_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Map normalized CLI values to dataset adapter and cache arguments."""

    return dict(
        profile_ieee_full_gpu=bool(args.profile_ieee_full_gpu),
        ieee_data_root=str(args.ieee_data_root).strip(),
        ieee_data_profile=str(args.ieee_data_profile).strip(),
        ieee_loader_view=str(args.ieee_loader_view).strip(),
        ieee_relation_profile=str(args.ieee_relation_profile).strip(),
        ieee_feature_profile=str(args.ieee_feature_profile).strip(),
        ieee_history_len=args.ieee_history_len,
        ieee_sampling_profile=str(args.ieee_sampling_profile).strip(),
        ieee_max_transactions=args.ieee_max_transactions,
        ieee_time_bins=args.ieee_time_bins,
        ieee_relation_window_neighbors=args.ieee_relation_window_neighbors,
        ieee_train_ratio=args.ieee_train_ratio,
        ieee_valid_ratio=args.ieee_valid_ratio,
        ieee_full_compact_sequences=bool(args.ieee_full_compact_sequences),
        ieee_sequence_feature_dim=args.ieee_sequence_feature_dim,
        ieee_event_feature_dim=args.ieee_event_feature_dim,
        ieee_build_light_cache_only=bool(args.ieee_build_light_cache_only),
        ieee_rebuild_light_cache=bool(args.ieee_rebuild_light_cache),
        ieee_build_cache_only=bool(args.ieee_build_cache_only),
        ieee_rebuild_cache=bool(args.ieee_rebuild_cache),
        ieee_skip_training=bool(args.ieee_skip_training),
        amlsim_data_root=str(args.amlsim_data_root).strip(),
        amlsim_train_ratio=args.amlsim_train_ratio,
        amlsim_valid_ratio=args.amlsim_valid_ratio,
        amlsim_relation_window_neighbors=args.amlsim_relation_window_neighbors,
        amlsim_activity_bins=args.amlsim_activity_bins,
        amlsim_event_history_len=args.amlsim_event_history_len,
        amlsim_rebuild_cache=bool(args.amlsim_rebuild_cache),
        amlsim_allow_sample_fallback=bool(args.amlsim_allow_sample_fallback),
        elliptic_data_root=str(args.elliptic_data_root).strip(),
        elliptic_train_time_end=args.elliptic_train_time_end,
        elliptic_valid_time_end=args.elliptic_valid_time_end,
        elliptic_history_len=args.elliptic_history_len,
        elliptic_sequence_topk=args.elliptic_sequence_topk,
        elliptic_coassociation_topk=args.elliptic_coassociation_topk,
        elliptic_coassociation_time_window=args.elliptic_coassociation_time_window,
        elliptic_use_unknown_ssl=bool(args.elliptic_use_unknown_ssl),
        elliptic_rebuild_cache=bool(args.elliptic_rebuild_cache),
        elliptic_pseudo_refresh_interval=args.elliptic_pseudo_refresh_interval,
        elliptic_pseudo_refresh_start_round=args.elliptic_pseudo_refresh_start_round,
        elliptic_pseudo_refresh_momentum=args.elliptic_pseudo_refresh_momentum,
        elliptic_pseudo_refresh_max_fraction=args.elliptic_pseudo_refresh_max_fraction,
        elliptic_diffusion_residual_scale=args.elliptic_diffusion_residual_scale,
        elliptic_coassociation_loss_weight=args.elliptic_coassociation_loss_weight,
        elliptic_wavelet_loss_weight=args.elliptic_wavelet_loss_weight,
        elliptic_utg_align_loss_weight=args.elliptic_utg_align_loss_weight,
        ethereum_phishing_data_root=str(args.ethereum_phishing_data_root).strip(),
        ethereum_phishing_max_users=args.ethereum_phishing_max_users,
        ethereum_phishing_max_transactions=args.ethereum_phishing_max_transactions,
        ethereum_phishing_force_preview=bool(args.ethereum_phishing_force_preview),
        ethereum_ponzi_data_root=str(args.ethereum_ponzi_data_root).strip(),
        ethereum_ponzi_negative_users_path=str(args.ethereum_ponzi_negative_users_path).strip(),
        ethereum_ponzi_force_preview=bool(args.ethereum_ponzi_force_preview),
        defi_rug_pull_data_root=str(args.defi_rug_pull_data_root).strip(),
        defi_rug_pull_negative_users_path=str(args.defi_rug_pull_negative_users_path).strip(),
        defi_rug_pull_force_preview=bool(args.defi_rug_pull_force_preview),
        result_root=str(args.result_root).strip(),
    )


def _build_training_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Combine the explicit CLI-to-training API mappings into one call payload."""

    return _core_training_kwargs(args) | _dataset_training_kwargs(args)


def main() -> None:
    args = parse_args()
    from .algorithms import resolve_requested_datasets, run_hybrid_fraud_training

    selected_datasets = resolve_requested_datasets(args.dataset)
    _normalize_active_mainline_args(args, selected_datasets)
    summaries = run_hybrid_fraud_training(**_build_training_kwargs(args))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
