from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .archive_dataset import ARCHIVE_DEFAULT_ROOT, load_archive_dataset
from .amlsim_dataset import AMLSIM_DEFAULT_ROOT, load_amlsim_dataset
from .ccfd_dataset import CCFD_DEFAULT_ROOT, load_ccfd_dataset
from .defi_rug_pull_dataset import DEFI_RUG_PULL_DEFAULT_ROOT, load_defi_rug_pull_dataset
from .elliptic_dataset import ELLIPTIC_DEFAULT_ROOT, load_elliptic_dataset
from .ethereum_phishing_dataset import ETHEREUM_PHISHING_DEFAULT_ROOT, load_ethereum_phishing_dataset
from .ethereum_ponzi_dataset import ETHEREUM_PONZI_DEFAULT_ROOT, load_ethereum_ponzi_dataset
from .fraud_dataset import DatasetBundle, load_splitgnn_dataset
from .ieee_cis_dataset import IEEE_DEFAULT_ROOT, load_ieee_cis_dataset
from .runtime_dataset_policy import active_runtime_datasets, ensure_dataset_enabled
from .paths import GRAPH_ROOT

SPLITGNN_DATA_DIR = GRAPH_ROOT


@dataclass(frozen=True)
class DatasetResourceHints:
    family: str
    default_data_root: str
    time_features_available: bool = False
    requires_negative_users_path: bool = False
    default_max_users: int | None = None
    default_max_transactions: int | None = None
    safe_preview_available: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetDescriptor:
    name: str
    loader: Callable[..., DatasetBundle]
    resource_hints: DatasetResourceHints


def _splitgnn_descriptor(name: str) -> DatasetDescriptor:
    return DatasetDescriptor(
        name=name,
        loader=load_splitgnn_dataset,
        resource_hints=DatasetResourceHints(
            family="splitgnn",
            default_data_root=str(SPLITGNN_DATA_DIR.resolve()),
            time_features_available=False,
            safe_preview_available=False,
            notes=("Uses the shared SplitGNN graph directory.",),
        ),
    )


DATASET_REGISTRY: dict[str, DatasetDescriptor] = {
    "amazon": _splitgnn_descriptor("amazon"),
    "yelp": _splitgnn_descriptor("yelp"),
    "comp": _splitgnn_descriptor("comp"),
    "ieee": DatasetDescriptor(
        name="ieee",
        loader=load_ieee_cis_dataset,
        resource_hints=DatasetResourceHints(
            family="ieee",
            default_data_root=str(IEEE_DEFAULT_ROOT.resolve()),
            time_features_available=True,
            default_max_transactions=None,
            safe_preview_available=True,
            notes=("Supports temporal relation features and cached graph builds.",),
        ),
    ),
    "archive": DatasetDescriptor(
        name="archive",
        loader=load_archive_dataset,
        resource_hints=DatasetResourceHints(
            family="defi_protocol",
            default_data_root=str(ARCHIVE_DEFAULT_ROOT.resolve()),
            time_features_available=True,
            default_max_users=4000,
            default_max_transactions=50000,
            safe_preview_available=True,
            notes=("DeFi Protocol Data on Ethereum with preview-mode caps.",),
        ),
    ),
    "ccfd": DatasetDescriptor(
        name="ccfd",
        loader=load_ccfd_dataset,
        resource_hints=DatasetResourceHints(
            family="ccfd",
            default_data_root=str(CCFD_DEFAULT_ROOT.resolve()),
            time_features_available=True,
            default_max_transactions=None,
            safe_preview_available=True,
            notes=("Chronological credit-card fraud dataset built from creditcard.csv.",),
        ),
    ),
    "amlsim": DatasetDescriptor(
        name="amlsim",
        loader=load_amlsim_dataset,
        resource_hints=DatasetResourceHints(
            family="synthetic_aml",
            default_data_root=str(AMLSIM_DEFAULT_ROOT.resolve()),
            time_features_available=True,
            default_max_transactions=None,
            safe_preview_available=True,
            notes=(
                "Loads AMLSim account-level CSV outputs into the hybrid graph pipeline.",
                "Supports fallback to AMLSim sample outputs when explicitly enabled.",
            ),
        ),
    ),
    "elliptic": DatasetDescriptor(
        name="elliptic",
        loader=load_elliptic_dataset,
        resource_hints=DatasetResourceHints(
            family="crypto_graph",
            default_data_root=str(ELLIPTIC_DEFAULT_ROOT.resolve()),
            time_features_available=True,
            default_max_transactions=None,
            safe_preview_available=True,
            notes=("Elliptic Bitcoin transaction graph with chronological SSL-aware splits.",),
        ),
    ),
    "ethereum_phishing": DatasetDescriptor(
        name="ethereum_phishing",
        loader=load_ethereum_phishing_dataset,
        resource_hints=DatasetResourceHints(
            family="onchain",
            default_data_root=str(ETHEREUM_PHISHING_DEFAULT_ROOT.resolve()),
            time_features_available=True,
            default_max_users=50000,
            default_max_transactions=None,
            safe_preview_available=True,
            notes=("On-chain phishing dataset with optional max_users/max_transactions caps.",),
        ),
    ),
    "ethereum_ponzi": DatasetDescriptor(
        name="ethereum_ponzi",
        loader=load_ethereum_ponzi_dataset,
        resource_hints=DatasetResourceHints(
            family="onchain",
            default_data_root=str(ETHEREUM_PONZI_DEFAULT_ROOT.resolve()),
            time_features_available=False,
            requires_negative_users_path=True,
            safe_preview_available=True,
            notes=("Requires an external negative address set to form a binary task.",),
        ),
    ),
    "defi_rug_pull": DatasetDescriptor(
        name="defi_rug_pull",
        loader=load_defi_rug_pull_dataset,
        resource_hints=DatasetResourceHints(
            family="onchain",
            default_data_root=str(DEFI_RUG_PULL_DEFAULT_ROOT.resolve()),
            time_features_available=False,
            requires_negative_users_path=True,
            safe_preview_available=True,
            notes=("Requires an external negative address set to form a binary task.",),
        ),
    ),
}


def registered_dataset_names() -> tuple[str, ...]:
    return tuple(name for name in active_runtime_datasets() if name in DATASET_REGISTRY)


def get_dataset_descriptor(dataset_name: str) -> DatasetDescriptor:
    normalized_name = str(dataset_name).lower()
    if normalized_name not in DATASET_REGISTRY:
        raise KeyError(f"Unsupported dataset: {dataset_name}")
    ensure_dataset_enabled(normalized_name, context="dataset_registry.get_dataset_descriptor")
    return DATASET_REGISTRY[normalized_name]


def bundle_protocol_summary(bundle: DatasetBundle, dataset_name: str | None = None) -> dict[str, Any]:
    descriptor = get_dataset_descriptor(dataset_name or bundle.name)
    graph = bundle.graph
    node_type = bundle.node_type
    node_data = graph.nodes[node_type].data
    feature_dim = 0
    if "feature" in node_data and getattr(node_data["feature"], "ndim", 0) >= 2:
        feature_dim = int(node_data["feature"].shape[-1])
    mask_names = sorted(
        key
        for key, value in node_data.items()
        if key.endswith("_mask") and getattr(value, "shape", None) is not None
    )
    return {
        "dataset": descriptor.name,
        "family": descriptor.resource_hints.family,
        "node_type": node_type,
        "feature_dim": int(feature_dim),
        "num_nodes": int(graph.num_nodes(node_type)),
        "relation_count": int(len(bundle.relation_order)),
        "relation_order": list(bundle.relation_order),
        "mask_names": mask_names,
        "num_clients": int(len(bundle.clients)),
        "client_train_nodes": [int(client.train_nodes) for client in bundle.clients],
        "time_features_available": bool(descriptor.resource_hints.time_features_available),
        "resource_hints": {
            "default_data_root": descriptor.resource_hints.default_data_root,
            "requires_negative_users_path": bool(descriptor.resource_hints.requires_negative_users_path),
            "default_max_users": descriptor.resource_hints.default_max_users,
            "default_max_transactions": descriptor.resource_hints.default_max_transactions,
            "safe_preview_available": bool(descriptor.resource_hints.safe_preview_available),
            "notes": list(descriptor.resource_hints.notes),
        },
    }


def attach_bundle_protocol(bundle: DatasetBundle, dataset_name: str | None = None) -> DatasetBundle:
    descriptor = get_dataset_descriptor(dataset_name or bundle.name)
    data_summary = dict(getattr(bundle, "data_summary", {}) or {})
    data_summary.setdefault("dataset", descriptor.name)
    data_summary.setdefault("family", descriptor.resource_hints.family)
    data_summary.setdefault("default_data_root", descriptor.resource_hints.default_data_root)
    data_summary["bundle_protocol"] = bundle_protocol_summary(bundle, descriptor.name)
    bundle.data_summary = data_summary
    return bundle


def _path_arg_or_default(value: object, default_path: Path) -> str:
    raw = str(value or "").strip()
    if not raw:
        return str(default_path)
    return raw


def _seed_arg_or_default(args: SimpleNamespace, default: int = 42) -> int:
    value = getattr(args, "seed", None)
    if value is None:
        return int(default)
    return int(value)


def load_registered_dataset_bundle(
    *,
    dataset_name: str,
    args: SimpleNamespace,
    effective_num_clients: int,
    client_hops: int,
    label_fraction: float,
) -> DatasetBundle:
    descriptor = get_dataset_descriptor(dataset_name)
    common_kwargs = {
        "dataset_name": descriptor.name,
        "num_clients": int(effective_num_clients),
        "seed": _seed_arg_or_default(args),
        "client_hops": int(client_hops),
        "label_fraction": float(label_fraction),
        "active_learning_feedback_path": str(getattr(args, "active_learning_feedback_path", "")),
    }
    if descriptor.resource_hints.family == "splitgnn":
        bundle = descriptor.loader(
            data_dir=str(SPLITGNN_DATA_DIR),
            **common_kwargs,
        )
        return attach_bundle_protocol(bundle, descriptor.name)
    if descriptor.name == "ieee":
        bundle = descriptor.loader(
            data_root=_path_arg_or_default(getattr(args, "ieee_data_root", IEEE_DEFAULT_ROOT), IEEE_DEFAULT_ROOT),
            max_transactions=(
                None
                if getattr(args, "ieee_max_transactions", None) is None
                or int(getattr(args, "ieee_max_transactions", 0)) <= 0
                else int(getattr(args, "ieee_max_transactions"))
            ),
            time_bins=int(getattr(args, "ieee_time_bins", 24)),
            relation_window_neighbors=int(getattr(args, "ieee_relation_window_neighbors", 2)),
            train_ratio=float(getattr(args, "ieee_train_ratio", 0.70)),
            valid_ratio=float(getattr(args, "ieee_valid_ratio", 0.15)),
            rebuild_cache=bool(getattr(args, "ieee_rebuild_cache", False)),
            ieee_full_compact_sequences=bool(getattr(args, "ieee_full_compact_sequences", True)),
            ieee_sequence_feature_dim=int(getattr(args, "ieee_sequence_feature_dim", 64)),
            ieee_event_feature_dim=int(getattr(args, "ieee_event_feature_dim", 64)),
            data_profile=str(getattr(args, "ieee_data_profile", "light_v1") or "light_v1"),
            loader_view=str(getattr(args, "ieee_loader_view", "hybrid") or "hybrid"),
            relation_profile=str(getattr(args, "ieee_relation_profile", "core") or "core"),
            feature_profile=str(getattr(args, "ieee_feature_profile", "typed_256") or "typed_256"),
            history_len=int(getattr(args, "ieee_history_len", 6) or 6),
            sampling_profile=str(getattr(args, "ieee_sampling_profile", "fraud_hardneg") or "fraud_hardneg"),
            build_light_cache_only=bool(getattr(args, "ieee_build_light_cache_only", False)),
            rebuild_light_cache=bool(getattr(args, "ieee_rebuild_light_cache", False)),
            **common_kwargs,
        )
        return attach_bundle_protocol(bundle, descriptor.name)
    if descriptor.name == "archive":
        bundle = descriptor.loader(
            data_root=_path_arg_or_default(getattr(args, "archive_data_root", ARCHIVE_DEFAULT_ROOT), ARCHIVE_DEFAULT_ROOT),
            max_users=getattr(args, "archive_max_users", None),
            max_transactions=getattr(args, "archive_max_transactions", None),
            risk_positive_ratio=float(getattr(args, "archive_risk_positive_ratio", 0.15)),
            force_preview=bool(getattr(args, "archive_force_preview", False)),
            **common_kwargs,
        )
        return attach_bundle_protocol(bundle, descriptor.name)
    if descriptor.name == "ccfd":
        bundle = descriptor.loader(
            data_root=_path_arg_or_default(getattr(args, "ccfd_data_root", CCFD_DEFAULT_ROOT), CCFD_DEFAULT_ROOT),
            max_transactions=getattr(args, "ccfd_max_transactions", None),
            time_bins=int(getattr(args, "ccfd_time_bins", 24)),
            amount_bins=int(getattr(args, "ccfd_amount_bins", 24)),
            relation_window_neighbors=int(getattr(args, "ccfd_relation_window_neighbors", 2)),
            train_ratio=float(getattr(args, "ccfd_train_ratio", 0.70)),
            valid_ratio=float(getattr(args, "ccfd_valid_ratio", 0.15)),
            **common_kwargs,
        )
        return attach_bundle_protocol(bundle, descriptor.name)
    if descriptor.name == "amlsim":
        bundle = descriptor.loader(
            data_root=_path_arg_or_default(getattr(args, "amlsim_data_root", AMLSIM_DEFAULT_ROOT), AMLSIM_DEFAULT_ROOT),
            train_ratio=float(getattr(args, "amlsim_train_ratio", 0.70)),
            valid_ratio=float(getattr(args, "amlsim_valid_ratio", 0.15)),
            relation_window_neighbors=int(getattr(args, "amlsim_relation_window_neighbors", 4)),
            activity_bins=int(getattr(args, "amlsim_activity_bins", 8)),
            event_history_len=int(getattr(args, "amlsim_event_history_len", 12)),
            rebuild_cache=bool(getattr(args, "amlsim_rebuild_cache", False)),
            allow_sample_fallback=bool(getattr(args, "amlsim_allow_sample_fallback", False)),
            **common_kwargs,
        )
        return attach_bundle_protocol(bundle, descriptor.name)
    if descriptor.name == "elliptic":
        bundle = descriptor.loader(
            data_root=_path_arg_or_default(getattr(args, "elliptic_data_root", ELLIPTIC_DEFAULT_ROOT), ELLIPTIC_DEFAULT_ROOT),
            train_time_end=int(getattr(args, "elliptic_train_time_end", 34)),
            valid_time_end=int(getattr(args, "elliptic_valid_time_end", 39)),
            history_len=int(getattr(args, "elliptic_history_len", 8)),
            sequence_topk=int(getattr(args, "elliptic_sequence_topk", 8)),
            coassociation_topk=int(getattr(args, "elliptic_coassociation_topk", 3)),
            coassociation_time_window=int(getattr(args, "elliptic_coassociation_time_window", 2)),
            use_unknown_ssl=bool(getattr(args, "elliptic_use_unknown_ssl", True)),
            rebuild_cache=bool(getattr(args, "elliptic_rebuild_cache", False)),
            **common_kwargs,
        )
        return attach_bundle_protocol(bundle, descriptor.name)
    if descriptor.name == "ethereum_phishing":
        bundle = descriptor.loader(
            data_root=_path_arg_or_default(
                getattr(args, "ethereum_phishing_data_root", ETHEREUM_PHISHING_DEFAULT_ROOT),
                ETHEREUM_PHISHING_DEFAULT_ROOT,
            ),
            max_users=getattr(args, "ethereum_phishing_max_users", None),
            max_transactions=getattr(args, "ethereum_phishing_max_transactions", None),
            force_preview=bool(getattr(args, "ethereum_phishing_force_preview", False)),
            **common_kwargs,
        )
        return attach_bundle_protocol(bundle, descriptor.name)
    if descriptor.name == "ethereum_ponzi":
        bundle = descriptor.loader(
            data_root=_path_arg_or_default(
                getattr(args, "ethereum_ponzi_data_root", ETHEREUM_PONZI_DEFAULT_ROOT),
                ETHEREUM_PONZI_DEFAULT_ROOT,
            ),
            negative_users_path=str(getattr(args, "ethereum_ponzi_negative_users_path", "")).strip() or None,
            force_preview=bool(getattr(args, "ethereum_ponzi_force_preview", False)),
            **common_kwargs,
        )
        return attach_bundle_protocol(bundle, descriptor.name)
    if descriptor.name == "defi_rug_pull":
        bundle = descriptor.loader(
            data_root=_path_arg_or_default(
                getattr(args, "defi_rug_pull_data_root", DEFI_RUG_PULL_DEFAULT_ROOT),
                DEFI_RUG_PULL_DEFAULT_ROOT,
            ),
            negative_users_path=str(getattr(args, "defi_rug_pull_negative_users_path", "")).strip() or None,
            force_preview=bool(getattr(args, "defi_rug_pull_force_preview", False)),
            **common_kwargs,
        )
        return attach_bundle_protocol(bundle, descriptor.name)
    raise KeyError(f"Unsupported dataset registry dispatch for {dataset_name}")
