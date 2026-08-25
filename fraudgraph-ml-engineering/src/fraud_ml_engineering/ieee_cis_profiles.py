from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import DATA_ROOT

IEEE_DEFAULT_DATA_ROOT = DATA_ROOT / "ieee_cis"

IEEE_DATA_PROFILE_RAW = "raw"
IEEE_DATA_PROFILE_LIGHT_V1 = "light_v1"
IEEE_DATA_PROFILE_LIGHT_V2 = "light_v2"
IEEE_DATA_PROFILE_TABULAR_FULL = "tabular_full"
IEEE_DATA_PROFILE_CUSTOM = "custom"

IEEE_LOADER_VIEW_GRAPH = "graph"
IEEE_LOADER_VIEW_TABULAR = "tabular"
IEEE_LOADER_VIEW_SEQUENCE = "sequence"
IEEE_LOADER_VIEW_HYBRID = "hybrid"

IEEE_RELATION_PROFILE_CORE = "core"
IEEE_RELATION_PROFILE_EXTENDED = "extended"

IEEE_FEATURE_PROFILE_TYPED_FULL = "typed_full"
IEEE_FEATURE_PROFILE_TYPED_256 = "typed_256"
IEEE_FEATURE_PROFILE_TYPED_160 = "typed_160"
IEEE_FEATURE_PROFILE_PAPER_PRUNED = "paper_pruned"
IEEE_FEATURE_PROFILE_PAPER_V30 = "paper_v30"

IEEE_SAMPLING_PROFILE_CHRONO_FULL = "chrono_full"
IEEE_SAMPLING_PROFILE_CHRONO_STRATIFIED = "chrono_stratified"
IEEE_SAMPLING_PROFILE_FRAUD_HARDNEG = "fraud_hardneg"
IEEE_SAMPLING_PROFILE_NORMAL_ONLY_TRAIN = "normal_only_train"

IEEE_DEFAULT_DATA_PROFILE = IEEE_DATA_PROFILE_LIGHT_V1
IEEE_DEFAULT_LOADER_VIEW = IEEE_LOADER_VIEW_HYBRID
IEEE_DEFAULT_RELATION_PROFILE = IEEE_RELATION_PROFILE_CORE
IEEE_DEFAULT_FEATURE_PROFILE = IEEE_FEATURE_PROFILE_TYPED_256
IEEE_DEFAULT_HISTORY_LEN = 6
IEEE_DEFAULT_SAMPLING_PROFILE = IEEE_SAMPLING_PROFILE_FRAUD_HARDNEG


@dataclass(frozen=True)
class IEEEDataProfileSpec:
    name: str
    max_transactions: int | None
    default_sampling_profile: str
    default_history_len: int
    description: str


@dataclass(frozen=True)
class IEEEFeatureProfileSpec:
    name: str
    dense_target_dim: int | None
    missing_threshold: float | None
    compress_v_block_to: int | None
    description: str


IEEE_DATA_PROFILES: dict[str, IEEEDataProfileSpec] = {
    IEEE_DATA_PROFILE_RAW: IEEEDataProfileSpec(
        name=IEEE_DATA_PROFILE_RAW,
        max_transactions=None,
        default_sampling_profile=IEEE_SAMPLING_PROFILE_CHRONO_FULL,
        default_history_len=8,
        description="Full raw IEEE-CIS training table and graph assets.",
    ),
    IEEE_DATA_PROFILE_LIGHT_V1: IEEEDataProfileSpec(
        name=IEEE_DATA_PROFILE_LIGHT_V1,
        max_transactions=260_000,
        default_sampling_profile=IEEE_SAMPLING_PROFILE_FRAUD_HARDNEG,
        default_history_len=6,
        description="Development-sized light asset profile for graph and hybrid experiments.",
    ),
    IEEE_DATA_PROFILE_LIGHT_V2: IEEEDataProfileSpec(
        name=IEEE_DATA_PROFILE_LIGHT_V2,
        max_transactions=360_000,
        default_sampling_profile=IEEE_SAMPLING_PROFILE_FRAUD_HARDNEG,
        default_history_len=8,
        description="Larger light asset profile for final model comparison.",
    ),
    IEEE_DATA_PROFILE_TABULAR_FULL: IEEEDataProfileSpec(
        name=IEEE_DATA_PROFILE_TABULAR_FULL,
        max_transactions=None,
        default_sampling_profile=IEEE_SAMPLING_PROFILE_CHRONO_FULL,
        default_history_len=6,
        description="Full-row tabular asset family with strong column pruning but no graph down-sampling.",
    ),
    IEEE_DATA_PROFILE_CUSTOM: IEEEDataProfileSpec(
        name=IEEE_DATA_PROFILE_CUSTOM,
        max_transactions=None,
        default_sampling_profile=IEEE_SAMPLING_PROFILE_FRAUD_HARDNEG,
        default_history_len=6,
        description="User-specified IEEE asset profile controlled by explicit overrides.",
    ),
}


IEEE_FEATURE_PROFILES: dict[str, IEEEFeatureProfileSpec] = {
    IEEE_FEATURE_PROFILE_TYPED_FULL: IEEEFeatureProfileSpec(
        name=IEEE_FEATURE_PROFILE_TYPED_FULL,
        dense_target_dim=None,
        missing_threshold=None,
        compress_v_block_to=None,
        description="Keep the typed feature schema intact with no dense compression.",
    ),
    IEEE_FEATURE_PROFILE_TYPED_256: IEEEFeatureProfileSpec(
        name=IEEE_FEATURE_PROFILE_TYPED_256,
        dense_target_dim=256,
        missing_threshold=0.97,
        compress_v_block_to=None,
        description="Keep typed schema and compress the dense feature block to 256 dimensions.",
    ),
    IEEE_FEATURE_PROFILE_TYPED_160: IEEEFeatureProfileSpec(
        name=IEEE_FEATURE_PROFILE_TYPED_160,
        dense_target_dim=160,
        missing_threshold=0.97,
        compress_v_block_to=None,
        description="Keep typed schema and compress the dense feature block to 160 dimensions.",
    ),
    IEEE_FEATURE_PROFILE_PAPER_PRUNED: IEEEFeatureProfileSpec(
        name=IEEE_FEATURE_PROFILE_PAPER_PRUNED,
        dense_target_dim=256,
        missing_threshold=0.85,
        compress_v_block_to=None,
        description="Aggressive paper-style pruning with high-missing column removal.",
    ),
    IEEE_FEATURE_PROFILE_PAPER_V30: IEEEFeatureProfileSpec(
        name=IEEE_FEATURE_PROFILE_PAPER_V30,
        dense_target_dim=160,
        missing_threshold=0.85,
        compress_v_block_to=30,
        description="Paper-style pruning with the V block reduced to 30 representative columns.",
    ),
}


IEEE_RELATION_PROFILE_COLUMNS: dict[str, tuple[str, ...]] = {
    IEEE_RELATION_PROFILE_CORE: (
        "uid",
        "uid_addr",
        "uid_email",
        "device_browser",
    ),
    IEEE_RELATION_PROFILE_EXTENDED: (
        "uid",
        "uid_addr",
        "uid_email",
        "device_browser",
        "temporal_past",
    ),
}


def normalize_ieee_data_profile(value: str | None) -> str:
    text = str(value or IEEE_DEFAULT_DATA_PROFILE).strip().lower()
    if text not in IEEE_DATA_PROFILES:
        return IEEE_DEFAULT_DATA_PROFILE
    return text


def normalize_ieee_loader_view(value: str | None) -> str:
    text = str(value or IEEE_DEFAULT_LOADER_VIEW).strip().lower()
    if text not in {
        IEEE_LOADER_VIEW_GRAPH,
        IEEE_LOADER_VIEW_TABULAR,
        IEEE_LOADER_VIEW_SEQUENCE,
        IEEE_LOADER_VIEW_HYBRID,
    }:
        return IEEE_DEFAULT_LOADER_VIEW
    return text


def normalize_ieee_relation_profile(value: str | None) -> str:
    text = str(value or IEEE_DEFAULT_RELATION_PROFILE).strip().lower()
    if text not in IEEE_RELATION_PROFILE_COLUMNS:
        return IEEE_DEFAULT_RELATION_PROFILE
    return text


def normalize_ieee_feature_profile(value: str | None) -> str:
    text = str(value or IEEE_DEFAULT_FEATURE_PROFILE).strip().lower()
    if text not in IEEE_FEATURE_PROFILES:
        return IEEE_DEFAULT_FEATURE_PROFILE
    return text


def normalize_ieee_sampling_profile(value: str | None, *, data_profile: str | None = None) -> str:
    text = str(value or "").strip().lower()
    if text:
        return text
    resolved_profile = resolve_ieee_data_profile(data_profile)
    return resolved_profile.default_sampling_profile


def resolve_ieee_data_profile(value: str | None) -> IEEEDataProfileSpec:
    return IEEE_DATA_PROFILES[normalize_ieee_data_profile(value)]


def resolve_ieee_feature_profile(value: str | None) -> IEEEFeatureProfileSpec:
    return IEEE_FEATURE_PROFILES[normalize_ieee_feature_profile(value)]


def resolve_ieee_relation_columns(relation_profile: str | None) -> tuple[str, ...]:
    normalized = normalize_ieee_relation_profile(relation_profile)
    return IEEE_RELATION_PROFILE_COLUMNS[normalized]


def resolve_ieee_history_len(
    history_len: int | None,
    *,
    data_profile: str | None = None,
) -> int:
    if history_len is not None and int(history_len) > 0:
        return int(history_len)
    return int(resolve_ieee_data_profile(data_profile).default_history_len)


def resolve_ieee_max_transactions(
    max_transactions: int | None,
    *,
    data_profile: str | None = None,
) -> int | None:
    if max_transactions is not None and int(max_transactions) > 0:
        return int(max_transactions)
    profile = resolve_ieee_data_profile(data_profile)
    return None if profile.max_transactions is None else int(profile.max_transactions)


def asset_family_for_view(
    *,
    data_profile: str | None,
    loader_view: str | None,
) -> str:
    resolved_profile = normalize_ieee_data_profile(data_profile)
    resolved_view = normalize_ieee_loader_view(loader_view)
    if resolved_view == IEEE_LOADER_VIEW_TABULAR:
        if resolved_profile in {IEEE_DATA_PROFILE_TABULAR_FULL, IEEE_DATA_PROFILE_RAW}:
            return "tabular_full_pruned"
        return f"tabular_{resolved_profile}"
    if resolved_view == IEEE_LOADER_VIEW_GRAPH:
        return "graph_raw" if resolved_profile == IEEE_DATA_PROFILE_RAW else f"graph_{resolved_profile}"
    if resolved_view == IEEE_LOADER_VIEW_SEQUENCE:
        return "hybrid_raw" if resolved_profile == IEEE_DATA_PROFILE_RAW else f"hybrid_{resolved_profile}"
    if resolved_view == IEEE_LOADER_VIEW_HYBRID:
        return "hybrid_raw" if resolved_profile == IEEE_DATA_PROFILE_RAW else f"hybrid_{resolved_profile}"
    return f"{resolved_view}_{resolved_profile}"


def ieee_cache_asset_root(data_root: str | Path, *, asset_family: str) -> Path:
    return Path(data_root).expanduser().resolve() / "cache_assets" / str(asset_family)


def ieee_profile_runtime_summary(
    *,
    data_profile: str | None,
    loader_view: str | None,
    relation_profile: str | None,
    feature_profile: str | None,
    history_len: int | None,
    sampling_profile: str | None,
    max_transactions: int | None,
) -> dict[str, Any]:
    resolved_data_profile = resolve_ieee_data_profile(data_profile)
    resolved_feature_profile = resolve_ieee_feature_profile(feature_profile)
    resolved_loader_view = normalize_ieee_loader_view(loader_view)
    resolved_relation_profile = normalize_ieee_relation_profile(relation_profile)
    resolved_history_len = resolve_ieee_history_len(history_len, data_profile=resolved_data_profile.name)
    resolved_sampling_profile = normalize_ieee_sampling_profile(
        sampling_profile,
        data_profile=resolved_data_profile.name,
    )
    return {
        "data_profile": resolved_data_profile.name,
        "loader_view": resolved_loader_view,
        "relation_profile": resolved_relation_profile,
        "feature_profile": resolved_feature_profile.name,
        "history_len": int(resolved_history_len),
        "sampling_profile": resolved_sampling_profile,
        "max_transactions": resolve_ieee_max_transactions(
            max_transactions,
            data_profile=resolved_data_profile.name,
        ),
        "asset_family": asset_family_for_view(
            data_profile=resolved_data_profile.name,
            loader_view=resolved_loader_view,
        ),
        "data_profile_summary": asdict(resolved_data_profile),
        "feature_profile_summary": asdict(resolved_feature_profile),
    }
