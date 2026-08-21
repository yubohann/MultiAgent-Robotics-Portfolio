from __future__ import annotations

"""IEEE-CIS fraud dataset loader with strict chronological sampling and graph construction."""

import hashlib
import json
import math
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import dgl
except Exception as error:  # pragma: no cover - runtime env dependent
    raise RuntimeError(
        "Missing dgl dependency. Install the pinned graph-learning profile and retry:"
        "\npython -m pip install -r requirements/requirements-cpu.txt"
    ) from error
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .fraud_dataset import (
    ClientShard,
    DatasetBundle,
    IEEE_FULL_SEQUENCE_COMPACT_DIM,
    IEEE_FULL_SEQUENCE_NODE_THRESHOLD,
    SEQUENCE_BUILDER_VERSION,
    _apply_active_learning_feedback,
    _apply_label_scarcity,
    _attach_dataset_context_defaults,
    _attach_relation_sequence,
    _build_client_subgraph,
    _memory_log,
    _merge_partitions,
    _random_partition,
    _start_memory_log_session,
    _stratified_partition,
    _sequence_quality_summary,
    _stop_memory_log_session,
)
from .runtime_dataset_policy import ensure_dataset_enabled
from .paths import DATA_ROOT, GRAPH_ROOT

IEEE_DEFAULT_ROOT = DATA_ROOT / "ieee_cis"
IEEE_RAW_ROOT = IEEE_DEFAULT_ROOT / "raw"
IEEE_CACHE_GRAPH_PATH = GRAPH_ROOT / "ieee.dgl"
IEEE_CACHE_METADATA_PATH = GRAPH_ROOT / "ieee_metadata.json"
NODE_TYPE = "transaction"

IEEE_FEATURE_LEAKAGE_GUARD_COLUMNS = frozenset({"TransactionID", "TransactionDT", "isFraud"})
IEEE_DEFAULT_RELATION_COLUMNS = (
    "uid",
    "uid_addr",
    "uid_email",
    "device_browser",
)
IEEE_DEFAULT_MAX_TRANSACTIONS = None
IEEE_DEFAULT_TIME_BINS = 24
IEEE_DEFAULT_RELATION_WINDOW_NEIGHBORS = 2
IEEE_DEFAULT_TEMPORAL_WINDOW_NEIGHBORS = 3
IEEE_EVENT_SEQUENCE_CHANNELS: tuple[str, ...] = (
    "uid_addr",
    "uid_email",
    "device_browser",
    "global_recent",
)
IEEE_EVENT_HISTORY_PER_CHANNEL = 2
IEEE_EVENT_SEQUENCE_LENGTH = 1 + len(IEEE_EVENT_SEQUENCE_CHANNELS) * IEEE_EVENT_HISTORY_PER_CHANNEL
IEEE_TEMPORAL_CONTEXT_WINDOWS: tuple[int, ...] = (3600, 21600, 86400, 259200, 604800)
IEEE_DEFAULT_TRAIN_RATIO = 0.70
IEEE_DEFAULT_VALID_RATIO = 0.15
IEEE_CACHE_LAYOUT_VERSION = "core_graph_plus_artifact_shards_v3_typed_schema_topk_relation"
IEEE_CACHE_ARTIFACT_VERSION = "ieee_sequence_event_artifacts_v3_typed_schema_topk_relation"
IEEE_RELATION_TOPK = 2
IEEE_HISTORY_GROUP_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("uid_addr", ("card1", "card2", "card3", "card5", "addr1", "addr2")),
    ("uid_email", ("card1", "card2", "card3", "card5", "P_emaildomain")),
    ("device_browser", ("DeviceType", "id_31")),
)
IEEE_MIN_RELATION_COVERAGE = 0.001
IEEE_DEFAULT_FULL_COMPACT_SEQUENCES = True
IEEE_DEFAULT_SEQUENCE_FEATURE_DIM = IEEE_FULL_SEQUENCE_COMPACT_DIM
IEEE_FULL_EVENT_COMPACT_DIM = IEEE_FULL_SEQUENCE_COMPACT_DIM
IEEE_DEFAULT_EVENT_FEATURE_DIM = IEEE_FULL_EVENT_COMPACT_DIM
IEEE_GRAPH_BUILDER_VERSION = (
    f"temporal_entity_history_v7_typed_schema::{SEQUENCE_BUILDER_VERSION}::{IEEE_CACHE_LAYOUT_VERSION}"
)
IEEE_SEQUENCE_ARTIFACT_FIELDS: tuple[str, ...] = (
    "sequence_relation_degree",
    "sequence_relation_topk_indices",
)
IEEE_EVENT_ARTIFACT_FIELDS: tuple[str, ...] = (
    "event_history_indices",
    "event_mask",
    "event_time_deltas",
    "event_token_weights",
    "event_token_types",
    "event_source_ids",
)
IEEE_ARTIFACT_SHARD_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sequence", IEEE_SEQUENCE_ARTIFACT_FIELDS),
    ("event", IEEE_EVENT_ARTIFACT_FIELDS),
)


def _ieee_id_columns(start: int, end: int) -> tuple[str, ...]:
    return tuple(f"id_{index:02d}" for index in range(int(start), int(end) + 1))


IEEE_SCHEMA_CATEGORICAL_COLUMNS = frozenset(
    {
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "id_12",
        "id_15",
        "id_16",
        "id_23",
        *[f"M{index}" for index in range(1, 10)],
        *_ieee_id_columns(27, 38),
    }
)
IEEE_SCHEMA_CONTINUOUS_COLUMNS = frozenset(
    {
        "TransactionAmt",
        "dist1",
        "dist2",
        *[f"id_{index:02d}" for index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 24, 25, 26]],
    }
)
IEEE_SCHEMA_CONTINUOUS_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^C\d+$"),
    re.compile(r"^D\d+$"),
    re.compile(r"^V\d+$"),
)
IEEE_RELATION_SPEC_COLUMNS: dict[str, tuple[str, ...]] = {
    "uid": ("card1", "card2", "card3", "card5"),
    "uid_addr": ("card1", "card2", "card3", "card5", "addr1", "addr2"),
    "uid_email": ("card1", "card2", "card3", "card5", "P_emaildomain"),
    "device_browser": ("DeviceType", "id_31"),
}
IEEE_RELATION_PRIORITY_ORDER: tuple[str, ...] = (
    "uid",
    "uid_addr",
    "uid_email",
    "device_browser",
    "temporal_past",
)


@dataclass(frozen=True)
class IEEECachedPaths:
    graph_path: Path
    metadata_path: Path
    artifact_dir: Path


def _normalize_missing_token(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "__missing__"}:
        return None
    return text


def _ieee_feature_type(column_name: str) -> str:
    column = str(column_name)
    if column in IEEE_SCHEMA_CATEGORICAL_COLUMNS:
        return "categorical"
    if column in IEEE_SCHEMA_CONTINUOUS_COLUMNS:
        return "continuous"
    for pattern in IEEE_SCHEMA_CONTINUOUS_PREFIX_PATTERNS:
        if pattern.match(column):
            return "continuous"
    return "continuous"


def _normalize_composite_relation_value(parts: list[str | None]) -> str | None:
    normalized_parts = [part for part in parts if part is not None]
    if len(normalized_parts) != len(parts) or not normalized_parts:
        return None
    return "||".join(normalized_parts)


def _build_composite_series(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.Series:
    normalized_columns = {
        column: [_normalize_missing_token(value) for value in frame[column].tolist()]
        for column in columns
        if column in frame.columns
    }
    values: list[str | None] = []
    for row_index in range(len(frame)):
        parts: list[str | None] = []
        valid = True
        for column in columns:
            column_values = normalized_columns.get(column)
            if column_values is None:
                valid = False
                break
            parts.append(column_values[row_index])
        normalized = _normalize_composite_relation_value(parts) if valid else None
        values.append(normalized)
    return pd.Series(values, index=frame.index, dtype="object")


def _chronological_bin_ids(size: int, bins: int) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=np.int32)
    effective_bins = int(max(1, min(int(bins), size)))
    edges = np.linspace(0, size, effective_bins + 1, dtype=np.int64)
    codes = np.empty(size, dtype=np.int32)
    for bin_id in range(effective_bins):
        start = int(edges[bin_id])
        end = int(edges[bin_id + 1])
        codes[start:end] = bin_id
    return codes


def _allocate_targets(
    group_sizes: list[int],
    total_target: int,
    *,
    preserve_present_groups: bool = False,
) -> list[int]:
    total_available = int(sum(max(int(size), 0) for size in group_sizes))
    if total_available <= 0:
        return [0 for _ in group_sizes]
    if total_target >= total_available:
        return [int(size) for size in group_sizes]
    target = max(int(total_target), 0)
    allocation = [0 for _ in group_sizes]
    positive_groups = [index for index, size in enumerate(group_sizes) if int(size) > 0]

    if preserve_present_groups and target >= len(positive_groups):
        for index in positive_groups:
            allocation[index] = 1
        target -= len(positive_groups)
        reduced_sizes = [max(int(size) - allocation[index], 0) for index, size in enumerate(group_sizes)]
    else:
        reduced_sizes = [max(int(size), 0) for size in group_sizes]

    reduced_total = int(sum(reduced_sizes))
    if reduced_total <= 0 or target <= 0:
        return allocation

    raw = [size / reduced_total * target for size in reduced_sizes]
    extra = [int(math.floor(value)) for value in raw]
    allocation = [allocation[index] + extra[index] for index in range(len(group_sizes))]
    remainder = target - int(sum(extra))
    fractions = sorted(
        ((raw[index] - extra[index], index) for index in range(len(group_sizes))),
        reverse=True,
    )
    for _, index in fractions:
        if remainder <= 0:
            break
        if allocation[index] >= int(group_sizes[index]):
            continue
        allocation[index] += 1
        remainder -= 1
    return allocation


def _time_stratified_sample(
    frame: pd.DataFrame,
    *,
    max_transactions: int | None,
    time_bins: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if max_transactions is None or len(frame) <= int(max_transactions):
        return frame.reset_index(drop=True).copy(), {
            "sampling_applied": False,
            "original_rows": int(len(frame)),
            "sampled_rows": int(len(frame)),
            "time_bins": int(max(1, min(int(time_bins), len(frame)))) if len(frame) > 0 else 0,
            "sampling_strategy": "full_chronological_train_set",
        }

    target_size = int(max(max_transactions, 1000))
    bin_ids = _chronological_bin_ids(len(frame), time_bins)
    bin_sizes = [int(np.sum(bin_ids == bin_id)) for bin_id in range(int(bin_ids.max()) + 1)]
    bin_targets = _allocate_targets(bin_sizes, target_size, preserve_present_groups=False)
    rng = np.random.default_rng(seed)
    sampled_indices: list[np.ndarray] = []

    labels = frame["isFraud"].to_numpy(dtype=np.int32)
    for bin_id, bin_target in enumerate(bin_targets):
        if bin_target <= 0:
            continue
        bin_index = np.flatnonzero(bin_ids == bin_id)
        if bin_index.size <= bin_target:
            sampled_indices.append(bin_index.astype(np.int64))
            continue
        neg_index = bin_index[labels[bin_index] == 0]
        pos_index = bin_index[labels[bin_index] == 1]
        label_targets = _allocate_targets(
            [int(len(neg_index)), int(len(pos_index))],
            int(bin_target),
            preserve_present_groups=True,
        )
        chosen_parts: list[np.ndarray] = []
        if label_targets[0] > 0 and len(neg_index) > 0:
            chosen_parts.append(np.sort(rng.choice(neg_index, size=label_targets[0], replace=False)).astype(np.int64))
        if label_targets[1] > 0 and len(pos_index) > 0:
            chosen_parts.append(np.sort(rng.choice(pos_index, size=label_targets[1], replace=False)).astype(np.int64))
        chosen = np.sort(np.concatenate(chosen_parts, axis=0)) if chosen_parts else np.empty(0, dtype=np.int64)
        if len(chosen) < bin_target:
            remaining_candidates = np.setdiff1d(bin_index.astype(np.int64), chosen, assume_unique=False)
            extra_needed = min(int(bin_target) - int(len(chosen)), int(len(remaining_candidates)))
            if extra_needed > 0:
                extra = np.sort(rng.choice(remaining_candidates, size=extra_needed, replace=False)).astype(np.int64)
                chosen = np.sort(np.concatenate([chosen, extra], axis=0))
        sampled_indices.append(chosen.astype(np.int64))

    merged_indices = np.sort(np.concatenate(sampled_indices, axis=0)).astype(np.int64)
    if len(merged_indices) > target_size:
        merged_indices = np.sort(rng.choice(merged_indices, size=target_size, replace=False)).astype(np.int64)
    elif len(merged_indices) < target_size:
        full_index = np.arange(len(frame), dtype=np.int64)
        remaining = np.setdiff1d(full_index, merged_indices, assume_unique=False)
        extra_needed = min(target_size - len(merged_indices), len(remaining))
        if extra_needed > 0:
            extra = np.sort(rng.choice(remaining, size=extra_needed, replace=False)).astype(np.int64)
            merged_indices = np.sort(np.concatenate([merged_indices, extra], axis=0))

    sampled = frame.iloc[merged_indices].copy().reset_index(drop=True)
    return sampled, {
        "sampling_applied": True,
        "original_rows": int(len(frame)),
        "sampled_rows": int(len(sampled)),
        "time_bins": int(bin_ids.max() + 1),
        "sampling_strategy": "chronological_time_bin_stratified_sample",
    }


def _chronological_split_masks(
    frame: pd.DataFrame,
    *,
    train_ratio: float,
    valid_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_rows = int(len(frame))
    if num_rows < 10:
        raise ValueError("IEEE-CIS sample is too small to build chronological train/valid/test splits.")
    train_end = int(round(num_rows * float(train_ratio)))
    valid_end = int(round(num_rows * float(train_ratio + valid_ratio)))
    train_end = min(max(train_end, 1), num_rows - 2)
    valid_end = min(max(valid_end, train_end + 1), num_rows - 1)

    train_mask = torch.zeros(num_rows, dtype=torch.bool)
    valid_mask = torch.zeros(num_rows, dtype=torch.bool)
    test_mask = torch.zeros(num_rows, dtype=torch.bool)
    train_mask[:train_end] = True
    valid_mask[train_end:valid_end] = True
    test_mask[valid_end:] = True
    return train_mask, valid_mask, test_mask


def _fit_feature_preprocessor(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, Any], dict[str, np.ndarray]]:
    feature_frame = frame.drop(columns=list(IEEE_FEATURE_LEAKAGE_GUARD_COLUMNS), errors="ignore").copy()
    raw_feature_columns = list(feature_frame.columns)
    categorical_columns = [column for column in raw_feature_columns if _ieee_feature_type(column) == "categorical"]
    numeric_columns = [column for column in raw_feature_columns if column not in categorical_columns]
    train_frame = feature_frame.loc[train_mask].copy()

    dense_columns: list[np.ndarray] = []
    dense_feature_names: list[str] = []
    category_sizes: dict[str, int] = {}
    numeric_fill_values: dict[str, float] = {}
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    categorical_frequency_means: dict[str, float] = {}
    categorical_frequency_stds: dict[str, float] = {}
    numeric_value_columns: list[np.ndarray] = []
    numeric_missing_columns: list[np.ndarray] = []
    categorical_id_columns: list[np.ndarray] = []
    categorical_missing_columns: list[np.ndarray] = []
    categorical_frequency_columns: list[np.ndarray] = []

    for column in numeric_columns:
        train_numeric = pd.to_numeric(train_frame[column], errors="coerce")
        fill_value = float(train_numeric.median()) if train_numeric.notna().any() else 0.0
        full_numeric_series = pd.to_numeric(feature_frame[column], errors="coerce")
        missing_mask = full_numeric_series.isna().to_numpy(dtype=np.float32)
        full_numeric = full_numeric_series.fillna(fill_value).to_numpy(dtype=np.float32)
        train_values = full_numeric[train_mask]
        mean = float(train_values.mean()) if train_values.size > 0 else 0.0
        std = float(train_values.std()) if train_values.size > 0 else 1.0
        if std < 1e-6:
            std = 1.0
        normalized_numeric = ((full_numeric - mean) / std).reshape(-1, 1).astype(np.float32)
        numeric_missing = missing_mask.reshape(-1, 1).astype(np.float32)
        numeric_value_columns.append(normalized_numeric)
        numeric_missing_columns.append(numeric_missing)
        dense_columns.append(normalized_numeric)
        dense_columns.append(numeric_missing)
        dense_feature_names.append(f"num::{column}")
        dense_feature_names.append(f"num_missing::{column}")
        numeric_fill_values[column] = fill_value
        numeric_means[column] = mean
        numeric_stds[column] = std

    for column in categorical_columns:
        train_tokens = [
            _normalize_missing_token(value) or "__missing__"
            for value in train_frame[column].tolist()
        ]
        known_values = sorted(set(train_tokens))
        mapping = {value: index + 1 for index, value in enumerate(known_values)}
        full_tokens = [_normalize_missing_token(value) or "__missing__" for value in feature_frame[column].tolist()]
        encoded = np.asarray([mapping.get(value, 0) for value in full_tokens], dtype=np.int64)
        missing_mask = np.asarray([1.0 if value == "__missing__" else 0.0 for value in full_tokens], dtype=np.float32)
        train_counts = pd.Series(train_tokens, dtype="object").value_counts(dropna=False)
        raw_frequency = np.asarray([float(train_counts.get(value, 0.0)) for value in full_tokens], dtype=np.float32)
        log_frequency = np.log1p(raw_frequency).astype(np.float32).reshape(-1, 1)
        train_log_frequency = log_frequency[train_mask]
        freq_mean = float(train_log_frequency.mean()) if train_log_frequency.size > 0 else 0.0
        freq_std = float(train_log_frequency.std()) if train_log_frequency.size > 0 else 1.0
        if freq_std < 1e-6:
            freq_std = 1.0
        normalized_frequency = ((log_frequency - freq_mean) / freq_std).astype(np.float32)
        categorical_id_columns.append(encoded.reshape(-1, 1))
        categorical_missing_columns.append(missing_mask.reshape(-1, 1))
        categorical_frequency_columns.append(normalized_frequency)
        dense_columns.append(normalized_frequency)
        dense_columns.append(missing_mask.reshape(-1, 1))
        dense_feature_names.append(f"cat_freq::{column}")
        dense_feature_names.append(f"cat_missing::{column}")
        category_sizes[column] = int(len(mapping))
        categorical_frequency_means[column] = freq_mean
        categorical_frequency_stds[column] = freq_std

    feature_matrix = (
        np.concatenate(dense_columns, axis=1).astype(np.float32)
        if dense_columns
        else np.zeros((len(feature_frame), 0), dtype=np.float32)
    )
    typed_artifacts = {
        "typed_numeric": (
            np.concatenate(numeric_value_columns, axis=1).astype(np.float32)
            if numeric_value_columns
            else np.zeros((len(feature_frame), 0), dtype=np.float32)
        ),
        "typed_numeric_missing": (
            np.concatenate(numeric_missing_columns, axis=1).astype(np.float32)
            if numeric_missing_columns
            else np.zeros((len(feature_frame), 0), dtype=np.float32)
        ),
        "typed_categorical": (
            np.concatenate(categorical_id_columns, axis=1).astype(np.int64)
            if categorical_id_columns
            else np.zeros((len(feature_frame), 0), dtype=np.int64)
        ),
        "typed_categorical_missing": (
            np.concatenate(categorical_missing_columns, axis=1).astype(np.float32)
            if categorical_missing_columns
            else np.zeros((len(feature_frame), 0), dtype=np.float32)
        ),
        "typed_categorical_frequency": (
            np.concatenate(categorical_frequency_columns, axis=1).astype(np.float32)
            if categorical_frequency_columns
            else np.zeros((len(feature_frame), 0), dtype=np.float32)
        ),
    }
    metadata = {
        "raw_feature_columns": raw_feature_columns,
        "feature_columns": dense_feature_names,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "category_sizes": category_sizes,
        "numeric_fill_values": numeric_fill_values,
        "numeric_means": numeric_means,
        "numeric_stds": numeric_stds,
        "categorical_frequency_means": categorical_frequency_means,
        "categorical_frequency_stds": categorical_frequency_stds,
        "typed_numeric_dim": int(typed_artifacts["typed_numeric"].shape[1]),
        "typed_categorical_dim": int(typed_artifacts["typed_categorical"].shape[1]),
    }
    return feature_matrix, dense_feature_names, metadata, typed_artifacts


def _label_purity_from_groups(
    value_to_nodes: dict[str, list[int]],
    labels: np.ndarray,
) -> float:
    weighted_purity = 0.0
    total_weight = 0
    for nodes in value_to_nodes.values():
        if len(nodes) < 2:
            continue
        node_index = np.asarray(nodes, dtype=np.int64)
        group_labels = labels[node_index]
        positive_rate = float(group_labels.mean()) if len(group_labels) > 0 else 0.0
        purity = max(positive_rate, 1.0 - positive_rate)
        weighted_purity += purity * len(nodes)
        total_weight += len(nodes)
    if total_weight <= 0:
        return 0.0
    return float(weighted_purity / total_weight)


def _resolve_relation_series(
    frame: pd.DataFrame,
    relation_name: str,
) -> tuple[pd.Series | None, tuple[str, ...]]:
    if relation_name in IEEE_RELATION_SPEC_COLUMNS:
        source_columns = IEEE_RELATION_SPEC_COLUMNS[relation_name]
        if not all(column in frame.columns for column in source_columns):
            return None, source_columns
        return _build_composite_series(frame, source_columns), source_columns
    if relation_name not in frame.columns:
        return None, (relation_name,)
    return frame[relation_name], (relation_name,)


def _relation_topk_from_edges(
    *,
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    neighbor_indices = np.full((int(num_nodes), int(topk)), -1, dtype=np.int64)
    relation_degree = np.zeros(int(num_nodes), dtype=np.float32)
    if len(src) == 0 or len(dst) == 0:
        return neighbor_indices, relation_degree

    order = np.lexsort((src, dst))
    src_sorted = src[order]
    dst_sorted = dst[order]
    current_dst = -1
    current_sources: list[int] = []

    def _flush_bucket(target_node: int, sources: list[int]) -> None:
        if target_node < 0 or not sources:
            return
        recent_sources = sources[-int(topk) :]
        insert_start = int(topk) - len(recent_sources)
        neighbor_indices[target_node, insert_start:] = np.asarray(recent_sources, dtype=np.int64)

    for source_node, target_node in zip(src_sorted.tolist(), dst_sorted.tolist()):
        relation_degree[int(target_node)] += 1.0
        if int(target_node) != current_dst:
            _flush_bucket(current_dst, current_sources)
            current_dst = int(target_node)
            current_sources = []
        current_sources.append(int(source_node))
    _flush_bucket(current_dst, current_sources)
    return neighbor_indices, relation_degree


def _relation_edges_from_series(
    series: pd.Series,
    *,
    max_neighbors: int,
    relation_id: int,
    transaction_dt: np.ndarray,
    transaction_amt: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    value_to_nodes: dict[str, list[int]] = {}
    for node_id, raw_value in enumerate(series.tolist()):
        normalized = _normalize_missing_token(raw_value)
        if normalized is None:
            continue
        value_to_nodes.setdefault(normalized, []).append(int(node_id))

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    rarity_parts: list[np.ndarray] = []
    neighbor_limit = max(int(max_neighbors), 1)
    total_observed = sum(len(nodes) for nodes in value_to_nodes.values())
    for nodes in value_to_nodes.values():
        if len(nodes) < 2:
            continue
        ordered = np.asarray(nodes, dtype=np.int64)
        rarity_value = np.float32(1.0 / float(len(ordered)))
        for offset in range(1, min(neighbor_limit, len(ordered) - 1) + 1):
            src = ordered[:-offset]
            dst = ordered[offset:]
            src_parts.append(src)
            dst_parts.append(dst)
            rarity_parts.append(np.full(len(src), rarity_value, dtype=np.float32))

    coverage = float(total_observed / len(series)) if len(series) > 0 else 0.0
    stats = {
        "coverage_ratio": coverage,
        "non_missing_nodes": int(total_observed),
        "unique_values": int(len(value_to_nodes)),
        "multi_occurrence_values": int(sum(1 for nodes in value_to_nodes.values() if len(nodes) >= 2)),
        "label_purity_reference": _label_purity_from_groups(value_to_nodes, labels),
        "selected": False,
        "selection_reason": "no_edges",
    }
    if not src_parts:
        empty = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=np.float32)
        return empty, empty, {
            "delta_t": empty_float,
            "log_delta_t": empty_float,
            "delta_amt": empty_float,
            "relation_type_id": np.empty(0, dtype=np.int64),
            "relation_rarity": empty_float,
            "missing_relation_flag": np.empty(0, dtype=np.float32),
        }, stats

    src = np.concatenate(src_parts, axis=0)
    dst = np.concatenate(dst_parts, axis=0)
    relation_rarity = np.concatenate(rarity_parts, axis=0) if rarity_parts else np.empty(0, dtype=np.float32)
    delta_t = np.maximum(transaction_dt[dst] - transaction_dt[src], 0.0).astype(np.float32)
    delta_amt = np.abs(transaction_amt[dst] - transaction_amt[src]).astype(np.float32)
    edge_features = {
        "delta_t": delta_t,
        "log_delta_t": np.log1p(delta_t).astype(np.float32),
        "delta_amt": delta_amt,
        "relation_type_id": np.full(len(src), int(relation_id), dtype=np.int64),
        "relation_rarity": relation_rarity.astype(np.float32),
        "missing_relation_flag": np.zeros(len(src), dtype=np.float32),
    }
    stats["edge_count"] = int(len(src))
    stats["avg_in_degree"] = float(len(src) / max(1, total_observed))
    stats["selected"] = bool(
        coverage >= IEEE_MIN_RELATION_COVERAGE and stats["multi_occurrence_values"] > 0 and len(src) > 0
    )
    stats["selection_reason"] = "selected" if stats["selected"] else "coverage_or_support_too_low"
    stats["mean_delta_t"] = float(delta_t.mean()) if len(delta_t) > 0 else 0.0
    stats["mean_delta_amt"] = float(delta_amt.mean()) if len(delta_amt) > 0 else 0.0
    return src, dst, edge_features, stats


def _temporal_past_edges(
    num_nodes: int,
    max_neighbors: int,
    *,
    transaction_dt: np.ndarray,
    transaction_amt: np.ndarray,
    relation_id: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if num_nodes <= 1:
        empty = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=np.float32)
        return empty, empty, {
            "delta_t": empty_float,
            "log_delta_t": empty_float,
            "delta_amt": empty_float,
            "relation_type_id": np.empty(0, dtype=np.int64),
            "relation_rarity": empty_float,
            "missing_relation_flag": empty_float,
        }

    ordered = np.arange(num_nodes, dtype=np.int64)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    neighbor_limit = max(int(max_neighbors), 1)
    for offset in range(1, min(neighbor_limit, num_nodes - 1) + 1):
        src_parts.append(ordered[:-offset])
        dst_parts.append(ordered[offset:])
    if not src_parts:
        empty = np.empty(0, dtype=np.int64)
        empty_float = np.empty(0, dtype=np.float32)
        return empty, empty, {
            "delta_t": empty_float,
            "log_delta_t": empty_float,
            "delta_amt": empty_float,
            "relation_type_id": np.empty(0, dtype=np.int64),
            "relation_rarity": empty_float,
            "missing_relation_flag": empty_float,
        }
    src = np.concatenate(src_parts, axis=0)
    dst = np.concatenate(dst_parts, axis=0)
    delta_t = np.maximum(transaction_dt[dst] - transaction_dt[src], 0.0).astype(np.float32)
    delta_amt = np.abs(transaction_amt[dst] - transaction_amt[src]).astype(np.float32)
    return src, dst, {
        "delta_t": delta_t,
        "log_delta_t": np.log1p(delta_t).astype(np.float32),
        "delta_amt": delta_amt,
        "relation_type_id": np.full(len(src), int(relation_id), dtype=np.int64),
        "relation_rarity": np.ones(len(src), dtype=np.float32),
        "missing_relation_flag": np.zeros(len(src), dtype=np.float32),
    }


def _event_storage_dtype(events: torch.Tensor) -> torch.dtype:
    return torch.float16 if events.numel() >= 25_000_000 else torch.float32


def _use_ieee_full_event_compact_mode(graph: dgl.DGLHeteroGraph) -> bool:
    return int(graph.num_nodes(NODE_TYPE)) >= int(IEEE_FULL_SEQUENCE_NODE_THRESHOLD)


def _attach_event_base_feature_bank(
    graph: dgl.DGLHeteroGraph,
    *,
    ieee_full_compact_sequences: bool = IEEE_DEFAULT_FULL_COMPACT_SEQUENCES,
    ieee_event_feature_dim: int = IEEE_DEFAULT_EVENT_FEATURE_DIM,
) -> dict[str, Any]:
    node_data = graph.nodes[NODE_TYPE].data
    features = node_data["feature"].float()
    raw_feature_dim = int(features.shape[1])
    if "event_base_feature" in node_data:
        del node_data["event_base_feature"]
    return {
        "event_base_feature_dim": int(max(int(ieee_event_feature_dim), 1)),
        "event_raw_feature_dim": int(raw_feature_dim),
        "event_compact_mode_enabled": False,
        "event_requested_feature_dim": int(max(int(ieee_event_feature_dim), 1)),
        "event_base_feature_storage_dtype": "runtime_shared_typed_projector",
    }


def _has_ieee_event_artifacts(graph: dgl.DGLHeteroGraph) -> bool:
    node_data = graph.nodes[NODE_TYPE].data
    required_fields = (
        "event_history_indices",
        "event_mask",
        "event_time_deltas",
        "event_token_weights",
        "event_token_types",
        "event_source_ids",
        "temporal_context",
    )
    return all(field_name in node_data for field_name in required_fields)


def _has_ieee_sequence_artifacts(graph: dgl.DGLHeteroGraph) -> bool:
    node_data = graph.nodes[NODE_TYPE].data
    runtime_dynamic_fields = (
        "sequence_relation_degree",
        "sequence_relation_topk_indices",
    )
    dense_fields = (
        "sequence",
        "sequence_mask",
        "sequence_token_weights",
        "sequence_token_types",
        "sequence_relation_ids",
    )
    lazy_fields = (
        "sequence_mask",
        "sequence_token_weights",
        "sequence_token_types",
        "sequence_relation_ids",
        "sequence_relation_mean_feature",
        "sequence_relation_max_feature",
        "sequence_relation_degree",
    )
    has_dense_payload = all(field_name in node_data for field_name in dense_fields)
    has_lazy_payload = all(field_name in node_data for field_name in lazy_fields)
    has_runtime_dynamic_payload = all(field_name in node_data for field_name in runtime_dynamic_fields)
    return bool(has_dense_payload or has_lazy_payload or has_runtime_dynamic_payload)


def _build_history_group_keys(frame: pd.DataFrame) -> dict[str, list[str | None]]:
    all_columns = {column for _, columns in IEEE_HISTORY_GROUP_SPECS for column in columns}
    normalized_columns: dict[str, list[str | None]] = {}
    for column in all_columns:
        if column not in frame.columns:
            continue
        normalized_columns[column] = [_normalize_missing_token(value) for value in frame[column].tolist()]

    group_keys: dict[str, list[str | None]] = {}
    for source_name, columns in IEEE_HISTORY_GROUP_SPECS:
        source_keys: list[str | None] = [None] * len(frame)
        for node_index in range(len(frame)):
            parts: list[str | None] = []
            valid = True
            for column in columns:
                column_values = normalized_columns.get(column)
                if column_values is None:
                    valid = False
                    break
                parts.append(column_values[node_index])
            source_keys[node_index] = _normalize_composite_relation_value(parts) if valid else None
        group_keys[source_name] = source_keys
    return group_keys


def _build_ieee_event_tensors(
    frame: pd.DataFrame,
    transaction_dt: torch.Tensor,
    history_len: int = IEEE_EVENT_SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    times = transaction_dt.float().cpu()
    num_nodes = int(times.shape[0])
    sequence_length = max(int(history_len), 1)
    history_slot_budget = max(sequence_length - 1, 0)
    channel_names = list(IEEE_EVENT_SEQUENCE_CHANNELS)
    channel_capacities = {source_name: 0 for source_name in channel_names}
    if history_slot_budget > 0:
        # Distribute the reduced history budget across channels without exceeding
        # the per-channel cap, so light profiles like history_len=6 remain valid.
        remaining_budget = int(history_slot_budget)
        for source_name in channel_names:
            if remaining_budget <= 0:
                break
            channel_capacities[source_name] = 1
            remaining_budget -= 1
        while remaining_budget > 0:
            assigned_any = False
            for source_name in channel_names:
                current_capacity = int(channel_capacities[source_name])
                if current_capacity >= int(IEEE_EVENT_HISTORY_PER_CHANNEL):
                    continue
                channel_capacities[source_name] = current_capacity + 1
                remaining_budget -= 1
                assigned_any = True
                if remaining_budget <= 0:
                    break
            if not assigned_any:
                break
    channel_ranges: dict[str, tuple[int, int]] = {}
    history_cursor = 0
    for source_name in channel_names:
        channel_capacity = int(channel_capacities[source_name])
        channel_ranges[source_name] = (history_cursor, history_cursor + channel_capacity)
        history_cursor += channel_capacity
    event_history_indices = torch.full((num_nodes, sequence_length), -1, dtype=torch.long)
    event_mask = torch.zeros((num_nodes, sequence_length), dtype=torch.bool)
    event_time_deltas = torch.zeros((num_nodes, sequence_length), dtype=torch.float32)
    event_token_weights = torch.zeros((num_nodes, sequence_length), dtype=torch.float32)
    event_token_types = torch.zeros((num_nodes, sequence_length), dtype=torch.long)
    event_source_ids = torch.zeros((num_nodes, sequence_length), dtype=torch.long)
    history_buckets: dict[str, dict[str, list[int]]] = {
        source_name: {}
        for source_name, _ in IEEE_HISTORY_GROUP_SPECS
    }
    source_name_to_id = {source_name: index + 1 for index, source_name in enumerate(IEEE_EVENT_SEQUENCE_CHANNELS)}
    source_name_to_id["current"] = len(source_name_to_id) + 1
    source_usage_counts: dict[str, int] = {source_name: 0 for source_name in IEEE_EVENT_SEQUENCE_CHANNELS}
    channel_nonempty_counts: dict[str, int] = {source_name: 0 for source_name in IEEE_EVENT_SEQUENCE_CHANNELS}
    history_lengths: list[int] = []
    fallback_global_history: list[int] = []
    history_group_keys = _build_history_group_keys(frame)

    for node_index in range(num_nodes):
        total_history_count = 0
        for source_name in channel_names:
            start, end = channel_ranges[source_name]
            channel_capacity = max(end - start, 0)
            if channel_capacity <= 0:
                continue
            if source_name == "global_recent":
                candidate_history = fallback_global_history
            else:
                source_key = history_group_keys.get(source_name, [None] * num_nodes)[node_index]
                candidate_history = history_buckets.get(source_name, {}).get(source_key, []) if source_key is not None else []
            source_usage_counts[source_name] = int(source_usage_counts.get(source_name, 0)) + 1
            recent_history = candidate_history[-channel_capacity:]
            if recent_history:
                channel_nonempty_counts[source_name] = int(channel_nonempty_counts.get(source_name, 0)) + 1
            total_history_count += int(len(recent_history))
            insert_start = end - len(recent_history)
            for position, history_index in enumerate(recent_history, start=insert_start):
                event_history_indices[node_index, position] = int(history_index)
                event_mask[node_index, position] = True
                delta = torch.log1p((times[node_index] - times[int(history_index)]).clamp(min=0.0))
                event_time_deltas[node_index, position] = delta
                event_token_weights[node_index, position] = 1.0 / (1.0 + delta)
                event_token_types[node_index, position] = int(source_name_to_id[source_name])
                event_source_ids[node_index, position] = int(source_name_to_id[source_name])
        history_lengths.append(total_history_count)
        current_position = sequence_length - 1
        event_history_indices[node_index, current_position] = int(node_index)
        event_mask[node_index, current_position] = True
        event_token_weights[node_index, current_position] = 1.0
        event_token_types[node_index, current_position] = int(source_name_to_id["current"])
        event_source_ids[node_index, current_position] = int(source_name_to_id["current"])
        for source_name, _ in IEEE_HISTORY_GROUP_SPECS:
            source_key = history_group_keys.get(source_name, [None] * num_nodes)[node_index]
            if source_key is not None:
                history_buckets.setdefault(source_name, {}).setdefault(source_key, []).append(int(node_index))
        fallback_global_history.append(int(node_index))

    sequence_stats = {
        "event_sequence_strategy": "indexed_multichannel_history_with_source_attention",
        "event_sequence_storage_mode": "history_indices",
        "history_group_specs": [name for name, _ in IEEE_HISTORY_GROUP_SPECS],
        "history_channels": channel_names,
        "history_per_channel": int(IEEE_EVENT_HISTORY_PER_CHANNEL),
        "requested_history_len": int(history_len),
        "effective_history_len": int(sequence_length),
        "history_slot_budget": int(history_slot_budget),
        "history_channel_capacities": {key: int(value) for key, value in channel_capacities.items()},
        "history_source_to_id": {key: int(value) for key, value in source_name_to_id.items()},
        "history_source_counts": {key: int(value) for key, value in source_usage_counts.items()},
        "history_channel_nonempty_counts": {key: int(value) for key, value in channel_nonempty_counts.items()},
        "history_channel_coverage_ratio": {
            key: float(value / max(num_nodes, 1))
            for key, value in channel_nonempty_counts.items()
        },
        "entity_history_coverage_ratio": float(sum(length > 0 for length in history_lengths) / max(num_nodes, 1)),
        "mean_history_length": float(np.mean(history_lengths)) if history_lengths else 0.0,
        "max_history_length": int(max(history_lengths)) if history_lengths else 0,
        "nodes_with_nonempty_history": int(sum(length > 0 for length in history_lengths)),
    }
    return (
        event_history_indices,
        event_mask,
        event_time_deltas,
        event_token_weights,
        event_token_types,
        {
            "event_source_ids": event_source_ids,
            "sequence_stats": sequence_stats,
        },
    )


def _attach_ieee_event_sequence(
    graph: dgl.DGLHeteroGraph,
    frame: pd.DataFrame,
    history_len: int = IEEE_EVENT_SEQUENCE_LENGTH,
    *,
    ieee_full_compact_sequences: bool = IEEE_DEFAULT_FULL_COMPACT_SEQUENCES,
    ieee_event_feature_dim: int = IEEE_DEFAULT_EVENT_FEATURE_DIM,
) -> dict[str, Any]:
    event_base_feature_stats = _attach_event_base_feature_bank(
        graph,
        ieee_full_compact_sequences=ieee_full_compact_sequences,
        ieee_event_feature_dim=ieee_event_feature_dim,
    )
    _memory_log(
        "event_sequence: begin "
        f"nodes={int(graph.num_nodes(NODE_TYPE))} feature_dim={int(graph.nodes[NODE_TYPE].data['feature'].shape[1])} "
        f"history_len={int(history_len)}"
    )
    transaction_dt = graph.nodes[NODE_TYPE].data["transaction_dt"].float()
    (
        event_history_indices,
        event_mask,
        event_time_deltas,
        event_token_weights,
        event_token_types,
        event_metadata,
    ) = _build_ieee_event_tensors(
        frame=frame,
        transaction_dt=transaction_dt,
        history_len=history_len,
    )
    event_metadata["sequence_stats"].update(event_base_feature_stats)
    _memory_log(
        "event_sequence: built "
        f"indices_shape={tuple(int(item) for item in event_history_indices.shape)} "
        f"base_feature_dim={int(event_base_feature_stats['event_base_feature_dim'])} "
        f"compact_mode={bool(event_base_feature_stats['event_compact_mode_enabled'])}"
    )
    if "event_sequence" in graph.nodes[NODE_TYPE].data:
        del graph.nodes[NODE_TYPE].data["event_sequence"]
    graph.nodes[NODE_TYPE].data["event_history_indices"] = event_history_indices.long()
    graph.nodes[NODE_TYPE].data["event_mask"] = event_mask.bool()
    graph.nodes[NODE_TYPE].data["event_time_deltas"] = event_time_deltas.float()
    graph.nodes[NODE_TYPE].data["event_token_weights"] = event_token_weights.float()
    graph.nodes[NODE_TYPE].data["event_token_types"] = event_token_types.long()
    graph.nodes[NODE_TYPE].data["event_source_ids"] = event_metadata["event_source_ids"].long()
    _memory_log(
        "event_sequence: attached "
        f"indices_dtype={str(graph.nodes[NODE_TYPE].data['event_history_indices'].dtype).replace('torch.', '')} "
        f"base_feature_storage_dtype={str(event_base_feature_stats['event_base_feature_storage_dtype'])}"
    )
    return event_metadata["sequence_stats"]


def _build_temporal_context_features(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = IEEE_TEMPORAL_CONTEXT_WINDOWS,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    transaction_dt = frame["TransactionDT"].to_numpy(dtype=np.float64)
    transaction_amt = (
        pd.to_numeric(frame["TransactionAmt"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "TransactionAmt" in frame.columns
        else np.zeros(len(frame), dtype=np.float64)
    )
    num_rows = int(len(frame))
    if num_rows == 0:
        return torch.zeros((0, 0), dtype=torch.float32), [], {
            "temporal_windows": list(windows),
            "temporal_context_dim": 0,
        }

    prefix_amt = np.concatenate([[0.0], np.cumsum(transaction_amt, dtype=np.float64)])
    feature_columns: list[np.ndarray] = []
    feature_names: list[str] = []

    for window in windows:
        left_index = np.searchsorted(transaction_dt, transaction_dt - float(window), side="left")
        counts = np.arange(num_rows, dtype=np.float64) - left_index.astype(np.float64)
        amt_sum = prefix_amt[1:] - prefix_amt[left_index]
        amt_mean = np.divide(amt_sum, np.maximum(counts, 1.0), dtype=np.float64)
        feature_columns.extend(
            [
                np.log1p(np.maximum(counts, 0.0)),
                np.log1p(np.maximum(amt_sum, 0.0)),
                np.log1p(np.maximum(amt_mean, 0.0)),
            ]
        )
        feature_names.extend(
            [
                f"log_count_{int(window)}s",
                f"log_amount_sum_{int(window)}s",
                f"log_amount_mean_{int(window)}s",
            ]
        )

    prev_gap = np.zeros(num_rows, dtype=np.float64)
    if num_rows > 1:
        prev_gap[1:] = np.maximum(np.diff(transaction_dt), 0.0)
    log_prev_gap = np.log1p(prev_gap)
    short_span = min(5, num_rows)
    long_span = min(20, num_rows)
    rolling_gap_short = pd.Series(prev_gap).rolling(window=short_span, min_periods=1).mean().to_numpy(dtype=np.float64)
    rolling_gap_long = pd.Series(prev_gap).rolling(window=long_span, min_periods=1).mean().to_numpy(dtype=np.float64)
    gap_ratio = np.divide(
        rolling_gap_short + 1.0,
        rolling_gap_long + 1.0,
        out=np.ones_like(rolling_gap_short),
        where=(rolling_gap_long + 1.0) != 0.0,
    )
    amount_delta = np.zeros(num_rows, dtype=np.float64)
    if num_rows > 1:
        amount_delta[1:] = np.abs(np.diff(transaction_amt))
    log_amount_delta = np.log1p(np.maximum(amount_delta, 0.0))

    feature_columns.extend(
        [
            log_prev_gap,
            np.log1p(np.maximum(rolling_gap_short, 0.0)),
            np.log1p(np.maximum(rolling_gap_long, 0.0)),
            np.log(gap_ratio),
            log_amount_delta,
        ]
    )
    feature_names.extend(
        [
            "log_prev_gap",
            "log_gap_mean_short",
            "log_gap_mean_long",
            "log_gap_ratio_short_long",
            "log_amount_delta_prev",
        ]
    )

    feature_matrix = np.stack(feature_columns, axis=1).astype(np.float32)
    tensor = torch.from_numpy(feature_matrix)
    summary = {
        "temporal_windows": [int(window) for window in windows],
        "temporal_context_dim": int(feature_matrix.shape[1]),
        "temporal_context_feature_names": list(feature_names),
        "temporal_context_mean": [float(value) for value in feature_matrix.mean(axis=0)],
        "temporal_context_std": [float(value) for value in feature_matrix.std(axis=0)],
    }
    return tensor, feature_names, summary


def _build_ieee_teacher_feature_matrix(typed_artifacts: dict[str, np.ndarray]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in (
        "typed_numeric",
        "typed_numeric_missing",
        "typed_categorical_frequency",
        "typed_categorical_missing",
    ):
        value = typed_artifacts.get(key)
        if value is not None and value.size > 0:
            parts.append(value.astype(np.float32))
    categorical_ids = typed_artifacts.get("typed_categorical")
    if categorical_ids is not None and categorical_ids.size > 0:
        parts.append(categorical_ids.astype(np.float32))
    if not parts:
        node_count = int(next(iter(typed_artifacts.values())).shape[0]) if typed_artifacts else 0
        return np.zeros((node_count, 0), dtype=np.float32)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _fit_ieee_tabular_teacher(
    *,
    typed_artifacts: dict[str, np.ndarray],
    labels: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    teacher_features = _build_ieee_teacher_feature_matrix(typed_artifacts)
    if teacher_features.size == 0 or not bool(np.any(train_mask)):
        return None, {
            "enabled": False,
            "teacher_name": "disabled",
            "reason": "missing_features_or_train_nodes",
        }

    train_x = teacher_features[train_mask]
    train_y = labels[train_mask].astype(np.int64)
    positive_count = float(max(float(train_y.sum()), 1.0))
    negative_count = float(max(float(len(train_y) - train_y.sum()), 1.0))
    sample_weight = np.where(train_y > 0, negative_count / positive_count, 1.0).astype(np.float32)

    model = None
    teacher_name = "unknown"
    last_error = ""
    try:
        import xgboost as xgb  # type: ignore

        model = xgb.XGBClassifier(
            n_estimators=320,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=4.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=8,
            random_state=42,
        )
        teacher_name = "xgboost_hist"
        model.fit(train_x, train_y, sample_weight=sample_weight)
    except Exception as error:
        last_error = str(error)
        try:
            import lightgbm as lgb  # type: ignore

            model = lgb.LGBMClassifier(
                n_estimators=320,
                learning_rate=0.05,
                num_leaves=128,
                max_depth=-1,
                min_child_samples=40,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary",
                random_state=42,
                n_jobs=8,
                verbose=-1,
            )
            teacher_name = "lightgbm_gbdt"
            model.fit(train_x, train_y, sample_weight=sample_weight)
        except Exception as fallback_error:
            last_error = f"{last_error} | {fallback_error}".strip(" |")
            try:
                from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore

                model = HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=8,
                    max_iter=280,
                    min_samples_leaf=40,
                    max_bins=255,
                    l2_regularization=0.0,
                    random_state=42,
                )
                teacher_name = "sklearn_hist_gradient_boosting"
                model.fit(train_x, train_y, sample_weight=sample_weight)
            except Exception as final_error:
                return None, {
                    "enabled": False,
                    "teacher_name": "disabled",
                    "reason": "teacher_fit_failed",
                    "error": f"{last_error} | {final_error}".strip(" |"),
                }

    if model is None:
        return None, {
            "enabled": False,
            "teacher_name": "disabled",
            "reason": "teacher_fit_failed",
            "error": last_error,
        }

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(teacher_features), dtype=np.float32)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            positive_prob = probabilities[:, 1]
        else:
            positive_prob = probabilities.reshape(-1)
    else:
        decision = np.asarray(model.decision_function(teacher_features), dtype=np.float32).reshape(-1)
        positive_prob = 1.0 / (1.0 + np.exp(-decision))
    positive_prob = np.clip(positive_prob.astype(np.float32), 1e-5, 1.0 - 1e-5)
    logits = np.log(positive_prob / (1.0 - positive_prob)).astype(np.float32)
    teacher_logits = np.stack([-logits, logits], axis=1).astype(np.float32)

    summary: dict[str, Any] = {
        "enabled": True,
        "teacher_name": teacher_name,
        "feature_dim": int(teacher_features.shape[1]),
        "train_nodes": int(train_mask.sum()),
        "valid_nodes": int(valid_mask.sum()),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

        summary["train_auc"] = float(roc_auc_score(labels[train_mask], positive_prob[train_mask]))
        summary["train_pr_auc"] = float(average_precision_score(labels[train_mask], positive_prob[train_mask]))
        if bool(np.any(valid_mask)) and len(np.unique(labels[valid_mask])) > 1:
            summary["valid_auc"] = float(roc_auc_score(labels[valid_mask], positive_prob[valid_mask]))
            summary["valid_pr_auc"] = float(average_precision_score(labels[valid_mask], positive_prob[valid_mask]))
    except Exception:
        pass
    return teacher_logits, summary


def _build_edge_dict(
    frame: pd.DataFrame,
    *,
    relation_columns: tuple[str, ...],
    relation_window_neighbors: int,
) -> tuple[
    dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]],
    dict[str, int],
    dict[str, dict[str, Any]],
    dict[str, dict[str, torch.Tensor]],
    dict[str, Any],
]:
    edge_dict: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    relation_edge_counts: dict[str, int] = {}
    relation_stats: dict[str, dict[str, Any]] = {}
    edge_feature_dict: dict[str, dict[str, torch.Tensor]] = {}
    relation_topk_columns: list[np.ndarray] = []
    relation_degree_columns: list[np.ndarray] = []
    relation_sequence_order: list[str] = []
    homo_src_parts: list[np.ndarray] = []
    homo_dst_parts: list[np.ndarray] = []
    homo_feature_parts: dict[str, list[np.ndarray]] = {
        "delta_t": [],
        "log_delta_t": [],
        "delta_amt": [],
        "relation_type_id": [],
        "relation_rarity": [],
        "missing_relation_flag": [],
    }
    transaction_dt = frame["TransactionDT"].to_numpy(dtype=np.float32)
    if "TransactionAmt" in frame.columns:
        transaction_amt = pd.to_numeric(frame["TransactionAmt"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        transaction_amt = np.zeros(len(frame), dtype=np.float32)
    labels = frame["isFraud"].to_numpy(dtype=np.int64)

    relation_id = 1
    for relation in relation_columns:
        relation_series, source_columns = _resolve_relation_series(frame, relation)
        if relation_series is None:
            continue
        src, dst, edge_features, stats = _relation_edges_from_series(
            relation_series,
            max_neighbors=relation_window_neighbors,
            relation_id=relation_id,
            transaction_dt=transaction_dt,
            transaction_amt=transaction_amt,
            labels=labels,
        )
        stats["source_columns"] = [str(column) for column in source_columns]
        relation_stats[relation] = stats
        relation_id += 1
        if not stats["selected"]:
            continue
        if len(src) == 0:
            continue
        edge_dict[(NODE_TYPE, relation, NODE_TYPE)] = (
            torch.from_numpy(src.astype(np.int64)),
            torch.from_numpy(dst.astype(np.int64)),
        )
        relation_edge_counts[relation] = int(len(src))
        edge_feature_dict[relation] = {
            key: torch.from_numpy(value.copy())
            for key, value in edge_features.items()
        }
        topk_indices, relation_degree = _relation_topk_from_edges(
            num_nodes=len(frame),
            src=src,
            dst=dst,
            topk=IEEE_RELATION_TOPK,
        )
        relation_topk_columns.append(topk_indices)
        relation_degree_columns.append(relation_degree.reshape(-1, 1))
        relation_sequence_order.append(str(relation))
        homo_src_parts.append(src)
        homo_dst_parts.append(dst)
        for key, value in edge_features.items():
            homo_feature_parts[key].append(value)

    temporal_relation_id = relation_id
    temporal_src, temporal_dst, temporal_features = _temporal_past_edges(
        len(frame),
        max_neighbors=max(IEEE_DEFAULT_TEMPORAL_WINDOW_NEIGHBORS, relation_window_neighbors),
        transaction_dt=transaction_dt,
        transaction_amt=transaction_amt,
        relation_id=temporal_relation_id,
    )
    if len(temporal_src) > 0:
        edge_dict[(NODE_TYPE, "temporal_past", NODE_TYPE)] = (
            torch.from_numpy(temporal_src.astype(np.int64)),
            torch.from_numpy(temporal_dst.astype(np.int64)),
        )
        relation_edge_counts["temporal_past"] = int(len(temporal_src))
        edge_feature_dict["temporal_past"] = {
            key: torch.from_numpy(value.copy())
            for key, value in temporal_features.items()
        }
        temporal_topk_indices, temporal_degree = _relation_topk_from_edges(
            num_nodes=len(frame),
            src=temporal_src,
            dst=temporal_dst,
            topk=IEEE_RELATION_TOPK,
        )
        relation_topk_columns.append(temporal_topk_indices)
        relation_degree_columns.append(temporal_degree.reshape(-1, 1))
        relation_sequence_order.append("temporal_past")
        relation_stats["temporal_past"] = {
            "coverage_ratio": 1.0,
            "non_missing_nodes": int(len(frame)),
            "unique_values": int(len(frame)),
            "multi_occurrence_values": int(len(frame)),
            "label_purity_reference": 0.0,
            "selected": True,
            "selection_reason": "always_on_temporal_backbone",
            "edge_count": int(len(temporal_src)),
            "avg_in_degree": float(len(temporal_src) / max(1, len(frame))),
            "mean_delta_t": float(temporal_features["delta_t"].mean()) if len(temporal_src) > 0 else 0.0,
            "mean_delta_amt": float(temporal_features["delta_amt"].mean()) if len(temporal_src) > 0 else 0.0,
        }

    if homo_src_parts:
        homo_src = np.concatenate(homo_src_parts, axis=0).astype(np.int64)
        homo_dst = np.concatenate(homo_dst_parts, axis=0).astype(np.int64)
        homo_features = {
            key: np.concatenate(parts, axis=0)
            for key, parts in homo_feature_parts.items()
            if parts
        }
    else:
        ordered_nodes = np.arange(len(frame), dtype=np.int64)
        homo_src = np.concatenate([ordered_nodes[:-1], ordered_nodes[1:]], axis=0)
        homo_dst = np.concatenate([ordered_nodes[1:], ordered_nodes[:-1]], axis=0)
        delta_t = np.maximum(transaction_dt[homo_dst] - transaction_dt[homo_src], 0.0).astype(np.float32)
        homo_features = {
            "delta_t": delta_t,
            "log_delta_t": np.log1p(delta_t).astype(np.float32),
            "delta_amt": np.abs(transaction_amt[homo_dst] - transaction_amt[homo_src]).astype(np.float32),
            "relation_type_id": np.zeros(len(homo_src), dtype=np.int64),
            "relation_rarity": np.ones(len(homo_src), dtype=np.float32),
            "missing_relation_flag": np.zeros(len(homo_src), dtype=np.float32),
        }

    edge_dict[(NODE_TYPE, "homo", NODE_TYPE)] = (
        torch.from_numpy(homo_src.astype(np.int64)),
        torch.from_numpy(homo_dst.astype(np.int64)),
    )
    relation_edge_counts["homo"] = int(len(homo_src))
    edge_feature_dict["homo"] = {
        key: torch.from_numpy(value.copy())
        for key, value in homo_features.items()
    }
    relation_stats["homo"] = {
        "coverage_ratio": 1.0,
        "non_missing_nodes": int(len(frame)),
        "unique_values": int(len(frame)),
        "multi_occurrence_values": int(len(frame)),
        "label_purity_reference": 0.0,
        "selected": True,
        "selection_reason": "aggregated_relation_graph",
        "edge_count": int(len(homo_src)),
        "avg_in_degree": float(len(homo_src) / max(1, len(frame))),
        "mean_delta_t": float(homo_features["delta_t"].mean()) if len(homo_src) > 0 else 0.0,
        "mean_delta_amt": float(homo_features["delta_amt"].mean()) if len(homo_src) > 0 else 0.0,
    }
    relation_sequence_payload = {
        "relation_order": list(relation_sequence_order),
        "topk_indices": (
            np.stack(relation_topk_columns, axis=1).astype(np.int64)
            if relation_topk_columns
            else np.zeros((len(frame), 0, IEEE_RELATION_TOPK), dtype=np.int64)
        ),
        "relation_degree": (
            np.concatenate(relation_degree_columns, axis=1).astype(np.float32)
            if relation_degree_columns
            else np.zeros((len(frame), 0), dtype=np.float32)
        ),
        "relation_topk": int(IEEE_RELATION_TOPK),
    }
    return edge_dict, relation_edge_counts, relation_stats, edge_feature_dict, relation_sequence_payload


def _cache_signature(
    *,
    data_root: Path,
    max_transactions: int | None,
    time_bins: int,
    relation_columns: tuple[str, ...],
    relation_window_neighbors: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    ieee_full_compact_sequences: bool,
    ieee_sequence_feature_dim: int,
    ieee_event_feature_dim: int,
) -> dict[str, Any]:
    return {
        "data_root": str(data_root),
        "graph_builder_version": IEEE_GRAPH_BUILDER_VERSION,
        "cache_layout_version": IEEE_CACHE_LAYOUT_VERSION,
        "artifact_cache_version": IEEE_CACHE_ARTIFACT_VERSION,
        "event_sequence_length": int(IEEE_EVENT_SEQUENCE_LENGTH),
        "event_sequence_channels": list(IEEE_EVENT_SEQUENCE_CHANNELS),
        "event_history_per_channel": int(IEEE_EVENT_HISTORY_PER_CHANNEL),
        "temporal_context_windows": [int(window) for window in IEEE_TEMPORAL_CONTEXT_WINDOWS],
        "history_group_specs": [[str(name), [str(column) for column in columns]] for name, columns in IEEE_HISTORY_GROUP_SPECS],
        "min_relation_coverage": float(IEEE_MIN_RELATION_COVERAGE),
        "relation_topk": int(IEEE_RELATION_TOPK),
        "default_temporal_window_neighbors": int(IEEE_DEFAULT_TEMPORAL_WINDOW_NEIGHBORS),
        "max_transactions": None if max_transactions is None else int(max_transactions),
        "time_bins": int(time_bins),
        "relation_columns": list(relation_columns),
        "relation_window_neighbors": int(relation_window_neighbors),
        "train_ratio": float(train_ratio),
        "valid_ratio": float(valid_ratio),
        "seed": int(seed) if max_transactions is not None else None,
        "ieee_full_compact_sequences": bool(ieee_full_compact_sequences),
        "ieee_sequence_feature_dim": int(ieee_sequence_feature_dim),
        "ieee_event_feature_dim": int(ieee_event_feature_dim),
        "feature_leakage_guard_columns": sorted(IEEE_FEATURE_LEAKAGE_GUARD_COLUMNS),
        "categorical_schema_columns": sorted(IEEE_SCHEMA_CATEGORICAL_COLUMNS),
        "continuous_schema_columns": sorted(IEEE_SCHEMA_CONTINUOUS_COLUMNS),
    }


def _resolve_cache_paths(signature: dict[str, Any]) -> IEEECachedPaths:
    default_signature = _cache_signature(
        data_root=IEEE_DEFAULT_ROOT.expanduser().resolve(),
        max_transactions=IEEE_DEFAULT_MAX_TRANSACTIONS,
        time_bins=IEEE_DEFAULT_TIME_BINS,
        relation_columns=IEEE_DEFAULT_RELATION_COLUMNS,
        relation_window_neighbors=IEEE_DEFAULT_RELATION_WINDOW_NEIGHBORS,
        train_ratio=IEEE_DEFAULT_TRAIN_RATIO,
        valid_ratio=IEEE_DEFAULT_VALID_RATIO,
        seed=42,
        ieee_full_compact_sequences=IEEE_DEFAULT_FULL_COMPACT_SEQUENCES,
        ieee_sequence_feature_dim=IEEE_DEFAULT_SEQUENCE_FEATURE_DIM,
        ieee_event_feature_dim=IEEE_DEFAULT_EVENT_FEATURE_DIM,
    )
    if signature == default_signature:
        return IEEECachedPaths(
            graph_path=IEEE_CACHE_GRAPH_PATH,
            metadata_path=IEEE_CACHE_METADATA_PATH,
            artifact_dir=IEEE_CACHE_GRAPH_PATH.parent / "ieee_artifacts",
        )

    digest = hashlib.sha1(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    cache_dir = IEEE_CACHE_GRAPH_PATH.parent / "cache"
    return IEEECachedPaths(
        graph_path=cache_dir / f"ieee_{digest}.dgl",
        metadata_path=cache_dir / f"ieee_{digest}.json",
        artifact_dir=cache_dir / f"ieee_{digest}_artifacts",
    )


def _artifact_tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "numel": int(tensor.numel()),
        "bytes": int(tensor.numel()) * int(tensor.element_size()),
    }


def _detach_ieee_artifact_payloads(graph: dgl.DGLHeteroGraph) -> dict[str, dict[str, torch.Tensor]]:
    node_data = graph.nodes[NODE_TYPE].data
    detached_payloads: dict[str, dict[str, torch.Tensor]] = {}
    for shard_name, field_names in IEEE_ARTIFACT_SHARD_FIELDS:
        shard_payload: dict[str, torch.Tensor] = {}
        for field_name in field_names:
            if field_name not in node_data:
                continue
            shard_payload[field_name] = node_data[field_name]
            del node_data[field_name]
        if shard_payload:
            detached_payloads[shard_name] = shard_payload
    return detached_payloads


def _restore_ieee_artifact_payloads(
    graph: dgl.DGLHeteroGraph,
    artifact_payloads: dict[str, dict[str, torch.Tensor]],
) -> None:
    node_data = graph.nodes[NODE_TYPE].data
    for shard_payload in artifact_payloads.values():
        for field_name, tensor in shard_payload.items():
            node_data[field_name] = tensor


def _save_ieee_artifact_shards(
    *,
    artifact_payloads: dict[str, dict[str, torch.Tensor]],
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shard_summaries: list[dict[str, Any]] = []
    for shard_name, shard_payload in artifact_payloads.items():
        shard_path = artifact_dir / f"{shard_name}.pt"
        _memory_log(
            "cache_write: saving_artifact_shard "
            f"name={shard_name} file={shard_path.name} tensors={int(len(shard_payload))}"
        )
        torch.save(shard_payload, shard_path)
        tensor_summaries = {
            field_name: _artifact_tensor_summary(tensor)
            for field_name, tensor in shard_payload.items()
        }
        shard_summary = {
            "name": str(shard_name),
            "file_name": str(shard_path.name),
            "tensor_count": int(len(shard_payload)),
            "fields": [str(field_name) for field_name in shard_payload.keys()],
            "total_numel": int(sum(int(item["numel"]) for item in tensor_summaries.values())),
            "total_bytes": int(sum(int(item["bytes"]) for item in tensor_summaries.values())),
            "tensors": tensor_summaries,
        }
        shard_summaries.append(shard_summary)
        _memory_log(
            "cache_write: artifact_shard_saved "
            f"name={shard_name} total_bytes={int(shard_summary['total_bytes'])}"
        )
    return {
        "version": IEEE_CACHE_ARTIFACT_VERSION,
        "layout": IEEE_CACHE_LAYOUT_VERSION,
        "storage_mode": "torch_shards",
        "artifact_dir": str(artifact_dir),
        "shard_count": int(len(shard_summaries)),
        "shards": shard_summaries,
    }


def _load_ieee_artifact_shards(
    *,
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    artifact_dir: Path,
) -> bool:
    artifact_cache = dict(metadata.get("artifact_cache", {}) or {})
    if str(artifact_cache.get("version", "")) != IEEE_CACHE_ARTIFACT_VERSION:
        _memory_log("cache_read: artifact_cache_missing_or_version_mismatch")
        return False
    shard_entries = list(artifact_cache.get("shards", []) or [])
    if not shard_entries:
        _memory_log("cache_read: artifact_cache_missing_shards")
        return False

    loaded_payloads: dict[str, dict[str, torch.Tensor]] = {}
    for shard_entry in shard_entries:
        shard_name = str(shard_entry.get("name", "")).strip() or "unknown"
        file_name = str(shard_entry.get("file_name", "")).strip()
        if not file_name:
            _memory_log(f"cache_read: artifact_shard_missing_filename name={shard_name}")
            return False
        shard_path = artifact_dir / file_name
        if not shard_path.exists():
            _memory_log(f"cache_read: artifact_shard_missing_file name={shard_name} file={shard_path.name}")
            return False
        _memory_log(f"cache_read: loading_artifact_shard name={shard_name} file={shard_path.name}")
        shard_payload = torch.load(shard_path, map_location="cpu")
        if not isinstance(shard_payload, dict):
            _memory_log(f"cache_read: artifact_shard_invalid_payload name={shard_name}")
            return False
        loaded_payloads[shard_name] = {
            str(field_name): tensor
            for field_name, tensor in shard_payload.items()
            if isinstance(tensor, torch.Tensor)
        }
        _memory_log(
            "cache_read: artifact_shard_loaded "
            f"name={shard_name} tensors={int(len(loaded_payloads[shard_name]))}"
        )

    if not loaded_payloads:
        _memory_log("cache_read: artifact_cache_empty")
        return False

    _restore_ieee_artifact_payloads(graph, loaded_payloads)
    _memory_log(
        "cache_read: artifacts_attached "
        f"shards={int(len(loaded_payloads))} "
        f"fields={int(sum(len(payload) for payload in loaded_payloads.values()))}"
    )
    return True


def _inject_cache_paths_into_metadata(
    *,
    metadata: dict[str, Any],
    cache_paths: IEEECachedPaths,
    artifact_cache: dict[str, Any],
) -> None:
    metadata["cache_layout_version"] = str(IEEE_CACHE_LAYOUT_VERSION)
    metadata["artifact_cache"] = copy.deepcopy(artifact_cache)
    data_summary = dict(metadata.get("data_summary", {}) or {})
    data_summary["cache_graph_path"] = str(cache_paths.graph_path)
    data_summary["cache_metadata_path"] = str(cache_paths.metadata_path)
    data_summary["cache_artifact_dir"] = str(cache_paths.artifact_dir)
    data_summary["cache_artifact_layout"] = str(IEEE_CACHE_LAYOUT_VERSION)
    data_summary["cache_artifact_version"] = str(artifact_cache.get("version", ""))
    data_summary["cache_artifact_shard_count"] = int(artifact_cache.get("shard_count", 0))
    data_summary["cache_artifact_shards"] = copy.deepcopy(list(artifact_cache.get("shards", []) or []))
    metadata["data_summary"] = data_summary


def _load_cached_graph(
    *,
    signature: dict[str, Any],
    cache_paths: IEEECachedPaths,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]] | None:
    _memory_log(
        "cache_read: begin "
        f"graph={cache_paths.graph_path.name} metadata={cache_paths.metadata_path.name} "
        f"artifact_dir={cache_paths.artifact_dir.name}"
    )
    if not cache_paths.graph_path.exists() or not cache_paths.metadata_path.exists():
        _memory_log("cache_read: miss_missing_files")
        return None
    metadata = json.loads(cache_paths.metadata_path.read_text(encoding="utf-8-sig"))
    _memory_log("cache_read: metadata_loaded")
    if dict(metadata.get("cache_signature", {})) != signature:
        _memory_log("cache_read: miss_signature_mismatch")
        return None
    graph = dgl.load_graphs(str(cache_paths.graph_path))[0][0]
    _memory_log(
        "cache_read: graph_loaded "
        f"nodes={int(graph.num_nodes(NODE_TYPE))} etypes={len(graph.etypes)}"
    )
    artifact_loaded = _load_ieee_artifact_shards(
        graph=graph,
        metadata=metadata,
        artifact_dir=cache_paths.artifact_dir,
    )
    _memory_log(
        "cache_read: artifact_status "
        f"loaded={bool(artifact_loaded)} shard_count={int(dict(metadata.get('artifact_cache', {}) or {}).get('shard_count', 0))}"
    )
    return graph, metadata


def _write_cache(
    *,
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    cache_paths: IEEECachedPaths,
) -> None:
    _memory_log(
        "cache_write: begin "
        f"graph={cache_paths.graph_path.name} metadata={cache_paths.metadata_path.name} "
        f"artifact_dir={cache_paths.artifact_dir.name}"
    )
    cache_paths.graph_path.parent.mkdir(parents=True, exist_ok=True)
    cache_paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    detached_artifacts = _detach_ieee_artifact_payloads(graph)
    try:
        _memory_log("cache_write: saving_core_graph")
        dgl.save_graphs(str(cache_paths.graph_path), [graph])
        _memory_log("cache_write: core_graph_saved")
        artifact_cache = _save_ieee_artifact_shards(
            artifact_payloads=detached_artifacts,
            artifact_dir=cache_paths.artifact_dir,
        )
        _inject_cache_paths_into_metadata(
            metadata=metadata,
            cache_paths=cache_paths,
            artifact_cache=artifact_cache,
        )
        cache_paths.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig")
        _memory_log("cache_write: metadata_saved")
    finally:
        _restore_ieee_artifact_payloads(graph, detached_artifacts)


def _load_sampled_ieee_frame(
    *,
    data_root: Path,
    max_transactions: int | None,
    time_bins: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    transaction_path = data_root / "hf_graph" / "raw" / "train_transaction.csv"
    identity_path = data_root / "hf_graph" / "raw" / "train_identity.csv"
    if not transaction_path.exists():
        raise FileNotFoundError(f"Missing IEEE-CIS transaction file: {transaction_path}")
    if not identity_path.exists():
        raise FileNotFoundError(f"Missing IEEE-CIS identity file: {identity_path}")

    _memory_log("build_graph_payload: reading_transaction_csv")
    transactions = pd.read_csv(transaction_path, low_memory=False)
    _memory_log(f"build_graph_payload: transaction_csv_loaded rows={int(len(transactions))}")
    _memory_log("build_graph_payload: reading_identity_csv")
    identities = pd.read_csv(identity_path, low_memory=False)
    _memory_log(f"build_graph_payload: identity_csv_loaded rows={int(len(identities))}")
    frame = transactions.merge(identities, on="TransactionID", how="left", sort=False)
    _memory_log(f"build_graph_payload: merged_raw_frame rows={int(len(frame))} cols={int(len(frame.columns))}")
    frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    _memory_log("build_graph_payload: sorted_merged_frame")
    sampled_frame, sampling_info = _time_stratified_sample(
        frame,
        max_transactions=max_transactions,
        time_bins=time_bins,
        seed=seed,
    )
    sampled_frame = sampled_frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    _memory_log(
        "build_graph_payload: sampled_frame_ready "
        f"rows={int(len(sampled_frame))} sampling_applied={bool(sampling_info['sampling_applied'])}"
    )
    return sampled_frame, sampling_info, transaction_path, identity_path


def _refresh_ieee_artifact_metadata(
    *,
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    relation_order: list[str],
    event_stats: dict[str, Any],
    dataset_name: str,
    ieee_full_compact_sequences: bool,
    ieee_sequence_feature_dim: int,
    ieee_event_feature_dim: int,
) -> None:
    relation_sequence_quality = _sequence_quality_summary(graph, relation_order, dataset_name=dataset_name)
    data_summary = dict(metadata.get("data_summary", {}) or {})
    data_summary["relation_columns_used"] = [str(item) for item in relation_order]
    data_summary["event_sequence_length"] = int(IEEE_EVENT_SEQUENCE_LENGTH)
    data_summary["event_sequence_strategy"] = str(event_stats.get("event_sequence_strategy", ""))
    data_summary["event_sequence_storage_mode"] = str(event_stats.get("event_sequence_storage_mode", ""))
    data_summary["sequence_quality"] = relation_sequence_quality
    data_summary["sequence_storage_mode"] = str(relation_sequence_quality.get("storage_mode", ""))
    data_summary["ieee_full_compact_sequences"] = bool(ieee_full_compact_sequences)
    data_summary["ieee_sequence_feature_dim"] = int(ieee_sequence_feature_dim)
    data_summary["ieee_event_feature_dim"] = int(ieee_event_feature_dim)
    data_summary["event_quality"] = copy.deepcopy(event_stats)
    metadata["data_summary"] = data_summary
    _memory_log(
        "artifact_metadata: refreshed "
        f"selected_relations={int(len(relation_order))} "
        f"sequence_length={int(relation_sequence_quality.get('sequence_length', 0))} "
        f"sequence_dim={int(relation_sequence_quality.get('sequence_feature_dim', 0))} "
        f"sequence_storage_mode={str(relation_sequence_quality.get('storage_mode', ''))} "
        f"sequence_compact_dim={int(ieee_sequence_feature_dim)} "
        f"event_compact_dim={int(ieee_event_feature_dim)} "
        f"event_length={int(IEEE_EVENT_SEQUENCE_LENGTH)}"
    )


def _rebuild_ieee_artifacts_only(
    *,
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    data_root: Path,
    dataset_name: str,
    max_transactions: int | None,
    time_bins: int,
    seed: int,
    ieee_full_compact_sequences: bool,
    ieee_sequence_feature_dim: int,
    ieee_event_feature_dim: int,
) -> list[str]:
    _memory_log("artifact_rebuild: begin")
    sampled_frame, _, _, _ = _load_sampled_ieee_frame(
        data_root=data_root,
        max_transactions=max_transactions,
        time_bins=time_bins,
        seed=seed,
    )
    if int(len(sampled_frame)) != int(graph.num_nodes(NODE_TYPE)):
        raise RuntimeError(
            "IEEE artifact rebuild sampled frame size mismatch: "
            f"frame_rows={int(len(sampled_frame))} graph_nodes={int(graph.num_nodes(NODE_TYPE))}"
        )
    event_stats = _attach_ieee_event_sequence(
        graph,
        sampled_frame,
        ieee_full_compact_sequences=ieee_full_compact_sequences,
        ieee_event_feature_dim=ieee_event_feature_dim,
    )
    relation_order = _attach_relation_sequence(
        graph,
        dataset_name=dataset_name,
        ieee_full_compact_sequences=ieee_full_compact_sequences,
        ieee_sequence_feature_dim=ieee_sequence_feature_dim,
    )
    _refresh_ieee_artifact_metadata(
        graph=graph,
        metadata=metadata,
        relation_order=relation_order,
        event_stats=event_stats,
        dataset_name=dataset_name,
        ieee_full_compact_sequences=ieee_full_compact_sequences,
        ieee_sequence_feature_dim=ieee_sequence_feature_dim,
        ieee_event_feature_dim=ieee_event_feature_dim,
    )
    _memory_log(
        "artifact_rebuild: complete "
        f"selected_relations={int(len(relation_order))} "
        f"event_history_shape={tuple(int(item) for item in graph.nodes[NODE_TYPE].data['event_history_indices'].shape)}"
    )
    return relation_order


def _build_graph_payload(
    *,
    data_root: Path,
    max_transactions: int | None,
    time_bins: int,
    relation_columns: tuple[str, ...],
    relation_window_neighbors: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    ieee_full_compact_sequences: bool,
    ieee_sequence_feature_dim: int,
    ieee_event_feature_dim: int,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]]:
    _memory_log(
        "build_graph_payload: begin "
        f"max_transactions={max_transactions} time_bins={int(time_bins)} relation_window_neighbors={int(relation_window_neighbors)} "
        f"full_compact={bool(ieee_full_compact_sequences)} sequence_dim={int(ieee_sequence_feature_dim)} "
        f"event_dim={int(ieee_event_feature_dim)}"
    )
    sampled_frame, sampling_info, transaction_path, identity_path = _load_sampled_ieee_frame(
        data_root=data_root,
        max_transactions=max_transactions,
        time_bins=time_bins,
        seed=seed,
    )

    train_mask, valid_mask, test_mask = _chronological_split_masks(
        sampled_frame,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    labels = sampled_frame["isFraud"].to_numpy(dtype=np.int64)
    train_mask_np = train_mask.cpu().numpy().astype(bool)
    valid_mask_np = valid_mask.cpu().numpy().astype(bool)
    _memory_log("build_graph_payload: fitting_feature_preprocessor")
    feature_matrix, feature_columns, feature_metadata, typed_artifacts = _fit_feature_preprocessor(
        sampled_frame,
        train_mask=train_mask_np,
    )
    _memory_log(
        "build_graph_payload: feature_matrix_ready "
        f"shape={tuple(int(item) for item in feature_matrix.shape)}"
    )
    _memory_log("build_graph_payload: building_edge_dict")
    edge_dict, relation_edge_counts, relation_stats, edge_feature_dict, relation_sequence_payload = _build_edge_dict(
        sampled_frame,
        relation_columns=relation_columns,
        relation_window_neighbors=relation_window_neighbors,
    )
    _memory_log(
        "build_graph_payload: edge_dict_ready "
        f"relations={len(relation_edge_counts)} total_edges={int(sum(int(count) for count in relation_edge_counts.values()))}"
    )

    graph = dgl.heterograph(edge_dict, num_nodes_dict={NODE_TYPE: len(sampled_frame)})
    _memory_log("build_graph_payload: heterograph_created")
    graph.nodes[NODE_TYPE].data["feature"] = torch.from_numpy(feature_matrix.astype(np.float32))
    graph.nodes[NODE_TYPE].data["label"] = torch.from_numpy(labels.astype(np.int64))
    graph.nodes[NODE_TYPE].data["train_mask"] = train_mask.bool()
    graph.nodes[NODE_TYPE].data["valid_mask"] = valid_mask.bool()
    graph.nodes[NODE_TYPE].data["test_mask"] = test_mask.bool()
    graph.nodes[NODE_TYPE].data["transaction_dt"] = torch.from_numpy(
        sampled_frame["TransactionDT"].to_numpy(dtype=np.float32)
    )
    graph.nodes[NODE_TYPE].data["typed_numeric"] = torch.from_numpy(typed_artifacts["typed_numeric"]).float()
    graph.nodes[NODE_TYPE].data["typed_numeric_missing"] = torch.from_numpy(typed_artifacts["typed_numeric_missing"]).float()
    graph.nodes[NODE_TYPE].data["typed_categorical"] = torch.from_numpy(typed_artifacts["typed_categorical"]).long()
    graph.nodes[NODE_TYPE].data["typed_categorical_missing"] = torch.from_numpy(
        typed_artifacts["typed_categorical_missing"]
    ).float()
    graph.nodes[NODE_TYPE].data["typed_categorical_frequency"] = torch.from_numpy(
        typed_artifacts["typed_categorical_frequency"]
    ).float()
    graph.nodes[NODE_TYPE].data["sequence_relation_topk_indices"] = torch.from_numpy(
        relation_sequence_payload["topk_indices"]
    ).long()
    graph.nodes[NODE_TYPE].data["sequence_relation_degree"] = torch.from_numpy(
        relation_sequence_payload["relation_degree"]
    ).float()
    _memory_log("build_graph_payload: node_tensors_attached")
    temporal_context, temporal_feature_names, temporal_summary = _build_temporal_context_features(sampled_frame)
    graph.nodes[NODE_TYPE].data["temporal_context"] = temporal_context.float()
    _memory_log(
        "build_graph_payload: temporal_context_attached "
        f"shape={tuple(int(item) for item in temporal_context.shape)}"
    )

    graph.nodes[NODE_TYPE].data["train_supervised_mask"] = train_mask.bool().clone()
    graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros_like(train_mask.bool())
    graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = torch.full(
        (graph.num_nodes(NODE_TYPE),),
        1.0,
        dtype=torch.float32,
    )
    for relation_name, feature_payload in edge_feature_dict.items():
        for feature_name, feature_tensor in feature_payload.items():
            graph.edges[relation_name].data[feature_name] = feature_tensor
    _attach_dataset_context_defaults(graph, dataset_name="ieee")
    _memory_log("build_graph_payload: edge_features_and_context_attached")
    teacher_logits, teacher_summary = _fit_ieee_tabular_teacher(
        typed_artifacts=typed_artifacts,
        labels=labels,
        train_mask=train_mask_np,
        valid_mask=valid_mask_np,
    )
    if teacher_logits is not None:
        graph.nodes[NODE_TYPE].data["tabular_teacher_logits"] = torch.from_numpy(teacher_logits).float()
    _memory_log(
        "build_graph_payload: tabular_teacher_ready "
        f"enabled={bool(teacher_summary.get('enabled', False))} "
        f"name={str(teacher_summary.get('teacher_name', 'disabled'))}"
    )

    homo_src, homo_dst = graph.edges(etype="homo")
    homo_edge_labels = torch.where(
        graph.nodes[NODE_TYPE].data["label"][homo_src] == graph.nodes[NODE_TYPE].data["label"][homo_dst],
        torch.ones_like(homo_src, dtype=torch.float32),
        -torch.ones_like(homo_src, dtype=torch.float32),
    )
    edge_train_mask = train_mask[homo_src] & train_mask[homo_dst]
    graph.edges["homo"].data["label"] = homo_edge_labels
    graph.edges["homo"].data["train_mask"] = edge_train_mask.bool()
    _memory_log("build_graph_payload: homo_edge_labels_attached")
    sequence_stats = _attach_ieee_event_sequence(
        graph,
        sampled_frame,
        ieee_full_compact_sequences=ieee_full_compact_sequences,
        ieee_event_feature_dim=ieee_event_feature_dim,
    )
    relation_order = _attach_relation_sequence(
        graph,
        dataset_name="ieee",
        ieee_full_compact_sequences=ieee_full_compact_sequences,
        ieee_sequence_feature_dim=ieee_sequence_feature_dim,
    )
    _memory_log(
        "build_graph_payload: sequence_artifacts_ready "
        f"selected_relations={len(relation_order)}"
    )
    causal_edge_summary: dict[str, dict[str, Any]] = {}
    for relation_name, feature_payload in edge_feature_dict.items():
        delta_t_tensor = feature_payload.get("delta_t")
        delta_t_array = delta_t_tensor.cpu().numpy() if delta_t_tensor is not None else np.empty(0, dtype=np.float32)
        causal_edge_summary[relation_name] = {
            "edge_count": int(len(delta_t_array)),
            "non_negative_delta_t": bool(np.all(delta_t_array >= 0.0)) if len(delta_t_array) > 0 else True,
            "mean_delta_t": float(delta_t_array.mean()) if len(delta_t_array) > 0 else 0.0,
            "max_delta_t": float(delta_t_array.max()) if len(delta_t_array) > 0 else 0.0,
        }

    split_stats = {
        "train_positive": int(labels[train_mask_np].sum()),
        "valid_positive": int(labels[valid_mask.cpu().numpy().astype(bool)].sum()),
        "test_positive": int(labels[test_mask.cpu().numpy().astype(bool)].sum()),
    }
    metadata = {
        "cache_signature": _cache_signature(
            data_root=data_root,
            max_transactions=max_transactions,
            time_bins=time_bins,
            relation_columns=relation_columns,
            relation_window_neighbors=relation_window_neighbors,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
            ieee_full_compact_sequences=ieee_full_compact_sequences,
            ieee_sequence_feature_dim=ieee_sequence_feature_dim,
            ieee_event_feature_dim=ieee_event_feature_dim,
        ),
        "data_summary": {
            "data_root": str(data_root),
            "transaction_path": str(transaction_path),
            "identity_path": str(identity_path),
            "feature_columns": feature_columns,
            "raw_feature_columns": feature_metadata["raw_feature_columns"],
            "feature_dim": int(feature_matrix.shape[1]),
            "numeric_feature_count": int(len(feature_metadata["numeric_columns"])),
            "categorical_feature_count": int(len(feature_metadata["categorical_columns"])),
            "category_sizes": {key: int(value) for key, value in feature_metadata["category_sizes"].items()},
            "typed_numeric_dim": int(feature_metadata["typed_numeric_dim"]),
            "typed_categorical_dim": int(feature_metadata["typed_categorical_dim"]),
            "num_nodes": int(graph.num_nodes(NODE_TYPE)),
            "relation_columns_used": list(relation_order),
            "relation_candidate_columns": list(relation_columns),
            "relation_edge_counts": relation_edge_counts,
            "relation_field_stats": relation_stats,
            "relation_topk": int(relation_sequence_payload["relation_topk"]),
            "relation_runtime_order": list(relation_sequence_payload["relation_order"]),
            "causal_edge_summary": causal_edge_summary,
            "causal_graph_enforced": bool(all(item["non_negative_delta_t"] for item in causal_edge_summary.values())),
            "train_nodes": int(train_mask.sum().item()),
            "valid_nodes": int(valid_mask.sum().item()),
            "test_nodes": int(test_mask.sum().item()),
            "positive_labels": int(labels.sum()),
            "positive_ratio": float(labels.mean()),
            "train_positive": split_stats["train_positive"],
            "valid_positive": split_stats["valid_positive"],
            "test_positive": split_stats["test_positive"],
            "sampling_strategy": str(sampling_info["sampling_strategy"]),
            "sampling_applied": bool(sampling_info["sampling_applied"]),
            "original_train_rows": int(sampling_info["original_rows"]),
            "sampled_rows": int(sampling_info["sampled_rows"]),
            "time_bins": int(sampling_info["time_bins"]),
            "split_strategy": "chronological_transactiondt_holdout",
            "train_ratio": float(train_ratio),
            "valid_ratio": float(valid_ratio),
            "test_ratio": float(max(1.0 - train_ratio - valid_ratio, 0.0)),
            "graph_builder_version": IEEE_GRAPH_BUILDER_VERSION,
            "temporal_windows": temporal_summary["temporal_windows"],
            "temporal_context_dim": int(temporal_summary["temporal_context_dim"]),
            "temporal_context_feature_names": temporal_feature_names,
            "temporal_context_mean": temporal_summary["temporal_context_mean"],
            "temporal_context_std": temporal_summary["temporal_context_std"],
            "feature_leakage_guard_columns": sorted(IEEE_FEATURE_LEAKAGE_GUARD_COLUMNS),
            "typed_feature_schema": {
                "categorical_columns": list(feature_metadata["categorical_columns"]),
                "numeric_columns": list(feature_metadata["numeric_columns"]),
            },
            "relation_window_neighbors": int(relation_window_neighbors),
            "max_transactions": None if max_transactions is None else int(max_transactions),
            "transactiondt_min": float(sampled_frame["TransactionDT"].min()),
            "transactiondt_max": float(sampled_frame["TransactionDT"].max()),
            "ieee_full_compact_sequences": bool(ieee_full_compact_sequences),
            "ieee_sequence_feature_dim": int(ieee_sequence_feature_dim),
            "ieee_event_feature_dim": int(ieee_event_feature_dim),
            "tabular_teacher": teacher_summary,
            "cache_artifact_dir": "",
            "cache_artifact_layout": str(IEEE_CACHE_LAYOUT_VERSION),
            "cache_artifact_version": str(IEEE_CACHE_ARTIFACT_VERSION),
            "cache_artifact_shard_count": 0,
            "cache_artifact_shards": [],
            "cache_graph_path": "",
            "cache_metadata_path": "",
        },
    }
    _refresh_ieee_artifact_metadata(
        graph=graph,
        metadata=metadata,
        relation_order=relation_order,
        event_stats=sequence_stats,
        dataset_name="ieee",
        ieee_full_compact_sequences=ieee_full_compact_sequences,
        ieee_sequence_feature_dim=ieee_sequence_feature_dim,
        ieee_event_feature_dim=ieee_event_feature_dim,
    )
    _memory_log("build_graph_payload: metadata_ready")
    return graph, metadata


def _clone_graph_for_runtime(graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
    # The graph is already uniquely owned after cache load/build on this path.
    # Re-cloning here only duplicates a large IEEE graph in memory.
    return graph


def _reset_runtime_masks(graph: dgl.DGLHeteroGraph) -> None:
    train_mask = graph.nodes[NODE_TYPE].data["train_mask"].bool()
    graph.nodes[NODE_TYPE].data["train_supervised_mask"] = train_mask.clone()
    graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros_like(train_mask)
    graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = torch.full(
        (graph.num_nodes(NODE_TYPE),),
        1.0,
        dtype=torch.float32,
    )
    if "homo" in graph.etypes:
        src, dst = graph.edges(etype="homo")
        graph.edges["homo"].data["train_mask"] = (train_mask[src] & train_mask[dst]).bool()


def load_ieee_cis_dataset(
    *,
    data_root: str | Path = IEEE_DEFAULT_ROOT,
    dataset_name: str = "ieee",
    num_clients: int = 3,
    seed: int = 42,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    active_learning_feedback_path: str = "",
    max_transactions: int | None = IEEE_DEFAULT_MAX_TRANSACTIONS,
    time_bins: int = IEEE_DEFAULT_TIME_BINS,
    relation_columns: tuple[str, ...] = IEEE_DEFAULT_RELATION_COLUMNS,
    relation_window_neighbors: int = IEEE_DEFAULT_RELATION_WINDOW_NEIGHBORS,
    train_ratio: float = IEEE_DEFAULT_TRAIN_RATIO,
    valid_ratio: float = IEEE_DEFAULT_VALID_RATIO,
    rebuild_cache: bool = False,
    ieee_full_compact_sequences: bool = IEEE_DEFAULT_FULL_COMPACT_SEQUENCES,
    ieee_sequence_feature_dim: int = IEEE_DEFAULT_SEQUENCE_FEATURE_DIM,
    ieee_event_feature_dim: int = IEEE_DEFAULT_EVENT_FEATURE_DIM,
    data_profile: str = "raw",
    loader_view: str = "hybrid",
    relation_profile: str = "core",
    feature_profile: str = "typed_256",
    history_len: int | None = None,
    sampling_profile: str | None = None,
    build_light_cache_only: bool = False,
    rebuild_light_cache: bool = False,
) -> DatasetBundle:
    ensure_dataset_enabled("ieee", context="ieee_cis_dataset.load_ieee_cis_dataset")
    memory_session = _start_memory_log_session("ieee_dataset_load", dataset_name=dataset_name)
    try:
        resolved_loader_view = str(loader_view or "hybrid").strip().lower()
        if resolved_loader_view not in {"graph", "hybrid"}:
            raise ValueError(
                f"load_ieee_cis_dataset only returns graph/hybrid DatasetBundle objects, got loader_view={loader_view!r}. "
                "Use ieee_cis_views.load_ieee_tabular_view or load_ieee_sequence_view for non-graph consumers."
            )
        from .ieee_cis_views import load_ieee_graph_view, load_ieee_hybrid_view

        _memory_log(
            "load_ieee_cis_dataset: delegated_light_asset_loader "
            f"data_profile={str(data_profile)} loader_view={resolved_loader_view} "
            f"relation_profile={str(relation_profile)} feature_profile={str(feature_profile)} "
            f"history_len={str(history_len)} sampling_profile={str(sampling_profile)} "
            f"build_light_cache_only={bool(build_light_cache_only)} rebuild_light_cache={bool(rebuild_light_cache or rebuild_cache)}"
        )
        view_kwargs = {
            "data_root": data_root,
            "dataset_name": dataset_name,
            "num_clients": num_clients,
            "seed": seed,
            "client_hops": client_hops,
            "label_fraction": label_fraction,
            "active_learning_feedback_path": active_learning_feedback_path,
            "max_transactions": max_transactions,
            "time_bins": time_bins,
            "relation_window_neighbors": relation_window_neighbors,
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "data_profile": data_profile,
            "loader_view": resolved_loader_view,
            "relation_profile": relation_profile,
            "feature_profile": feature_profile,
            "history_len": (
                int(history_len)
                if history_len is not None and int(history_len) > 0
                else int(IEEE_EVENT_SEQUENCE_LENGTH)
            ),
            "sampling_profile": sampling_profile,
            "rebuild_light_cache": bool(rebuild_light_cache or rebuild_cache),
        }
        bundle = load_ieee_graph_view(**view_kwargs) if resolved_loader_view == "graph" else load_ieee_hybrid_view(**view_kwargs)
        _memory_log(
            "load_ieee_cis_dataset: delegated_bundle_ready "
            f"nodes={int(bundle.graph.num_nodes(bundle.node_type))} relations={len(bundle.relation_order)}"
        )
        return bundle

        resolved_root = Path(data_root).expanduser().resolve()
        _memory_log(
            "load_ieee_cis_dataset: begin "
            f"dataset={dataset_name} max_transactions={max_transactions} time_bins={int(time_bins)} "
            f"relation_window_neighbors={int(relation_window_neighbors)} rebuild_cache={bool(rebuild_cache)} "
            f"full_compact={bool(ieee_full_compact_sequences)} sequence_dim={int(ieee_sequence_feature_dim)} "
            f"event_dim={int(ieee_event_feature_dim)}"
        )
        signature = _cache_signature(
            data_root=resolved_root,
            max_transactions=max_transactions,
            time_bins=time_bins,
            relation_columns=relation_columns,
            relation_window_neighbors=relation_window_neighbors,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
            ieee_full_compact_sequences=ieee_full_compact_sequences,
            ieee_sequence_feature_dim=ieee_sequence_feature_dim,
            ieee_event_feature_dim=ieee_event_feature_dim,
        )
        cache_paths = _resolve_cache_paths(signature)
        _memory_log(
            "load_ieee_cis_dataset: cache_paths_resolved "
            f"graph={cache_paths.graph_path.name} metadata={cache_paths.metadata_path.name} "
            f"artifact_dir={cache_paths.artifact_dir.name}"
        )
        cached = None
        if bool(rebuild_cache):
            _memory_log("load_ieee_cis_dataset: force_rebuild_requested")
        else:
            cached = _load_cached_graph(
                signature=signature,
                cache_paths=cache_paths,
            )
        if cached is None:
            _memory_log("load_ieee_cis_dataset: cache_miss_building_payload")
            graph, metadata = _build_graph_payload(
                data_root=resolved_root,
                max_transactions=max_transactions,
                time_bins=time_bins,
                relation_columns=relation_columns,
                relation_window_neighbors=relation_window_neighbors,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                seed=seed,
                ieee_full_compact_sequences=ieee_full_compact_sequences,
                ieee_sequence_feature_dim=ieee_sequence_feature_dim,
                ieee_event_feature_dim=ieee_event_feature_dim,
            )
            _write_cache(
                graph=graph,
                metadata=metadata,
                cache_paths=cache_paths,
            )
        else:
            graph, metadata = cached
            _memory_log(
                "load_ieee_cis_dataset: cache_hit "
                f"nodes={int(graph.num_nodes(NODE_TYPE))} etypes={len(graph.etypes)}"
            )

        cache_requires_core_rebuild = False
        cache_requires_artifact_rebuild = False
        _memory_log("load_ieee_cis_dataset: auditing_cached_artifacts")
        if not _has_ieee_sequence_artifacts(graph):
            cache_requires_artifact_rebuild = True
        if not _has_ieee_event_artifacts(graph):
            cache_requires_artifact_rebuild = True
        for relation_name in graph.etypes:
            if relation_name == "self_loop":
                continue
            if "delta_t" not in graph.edges[relation_name].data:
                cache_requires_core_rebuild = True
                break
        _memory_log(
            "load_ieee_cis_dataset: cache_audit_complete "
            f"core_rebuild={bool(cache_requires_core_rebuild)} artifact_rebuild={bool(cache_requires_artifact_rebuild)}"
        )
        if cache_requires_core_rebuild:
            _memory_log("load_ieee_cis_dataset: rebuilding_full_payload_due_to_cache_audit")
            graph, metadata = _build_graph_payload(
                data_root=resolved_root,
                max_transactions=max_transactions,
                time_bins=time_bins,
                relation_columns=relation_columns,
                relation_window_neighbors=relation_window_neighbors,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                seed=seed,
                ieee_full_compact_sequences=ieee_full_compact_sequences,
                ieee_sequence_feature_dim=ieee_sequence_feature_dim,
                ieee_event_feature_dim=ieee_event_feature_dim,
            )
            _write_cache(
                graph=graph,
                metadata=metadata,
                cache_paths=cache_paths,
            )
        elif cache_requires_artifact_rebuild:
            _memory_log("load_ieee_cis_dataset: rebuilding_artifacts_due_to_cache_audit")
            _rebuild_ieee_artifacts_only(
                graph=graph,
                metadata=metadata,
                data_root=resolved_root,
                dataset_name=dataset_name,
                max_transactions=max_transactions,
                time_bins=time_bins,
                seed=seed,
                ieee_full_compact_sequences=ieee_full_compact_sequences,
                ieee_sequence_feature_dim=ieee_sequence_feature_dim,
                ieee_event_feature_dim=ieee_event_feature_dim,
            )
            _write_cache(
                graph=graph,
                metadata=metadata,
                cache_paths=cache_paths,
            )

        _attach_dataset_context_defaults(graph, dataset_name=dataset_name)
        _reset_runtime_masks(graph)
        _memory_log("load_ieee_cis_dataset: runtime_masks_reset")
        if float(label_fraction) < 0.999:
            _apply_label_scarcity(graph, label_fraction=float(label_fraction), seed=seed)
            _memory_log(f"load_ieee_cis_dataset: label_scarcity_applied fraction={float(label_fraction):.4f}")
        if active_learning_feedback_path:
            _apply_active_learning_feedback(graph, active_learning_feedback_path, dataset_name=dataset_name)
            _memory_log("load_ieee_cis_dataset: active_learning_feedback_applied")
        relation_order = [
            str(item)
            for item in list(dict(metadata.get("data_summary", {}) or {}).get("relation_columns_used", []))
            if str(item) not in {"self_loop", "homo"}
        ]
        if not relation_order:
            relation_order = [str(item) for item in graph.etypes if str(item) not in {"self_loop", "homo"}]
        if not _has_ieee_sequence_artifacts(graph):
            raise RuntimeError(
                "IEEE-CIS cached graph is missing relation-sequence tensors after cache audit; please rebuild the dataset cache."
            )
        if not _has_ieee_event_artifacts(graph):
            raise RuntimeError(
                "IEEE-CIS cached graph is missing indexed event or temporal tensors; please rebuild the dataset cache."
            )

        train_supervised_mask = graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool() & graph.nodes[NODE_TYPE].data[
            "train_mask"
        ].bool()
        train_unlabeled_mask = graph.nodes[NODE_TYPE].data["train_unlabeled_mask"].bool() & graph.nodes[NODE_TYPE].data[
            "train_mask"
        ].bool()
        supervised_nodes = train_supervised_mask.nonzero(as_tuple=False).flatten()
        supervised_labels = graph.nodes[NODE_TYPE].data["label"][train_supervised_mask]
        unlabeled_nodes = train_unlabeled_mask.nonzero(as_tuple=False).flatten()
        resolved_num_clients = max(int(num_clients), 1)
        clients: list[ClientShard] = []
        client_subgraph_mode = "materialized_subgraph"
        if resolved_num_clients == 1:
            owned_nodes = graph.nodes[NODE_TYPE].data["train_mask"].bool().nonzero(as_tuple=False).flatten().long()
            clients.append(
                ClientShard(
                    client_id=0,
                    owned_global_nodes=owned_nodes,
                    subgraph=graph,
                    train_nodes=int(owned_nodes.numel()),
                )
            )
            client_subgraph_mode = "shared_global_graph"
        else:
            supervised_partitions = _stratified_partition(
                supervised_nodes,
                supervised_labels,
                num_clients=resolved_num_clients,
                seed=seed,
            )
            unlabeled_partitions = _random_partition(unlabeled_nodes, num_clients=resolved_num_clients, seed=seed + 1)
            owned_partitions = _merge_partitions(supervised_partitions, unlabeled_partitions)

            for client_id, owned_nodes in enumerate(owned_partitions):
                if len(owned_nodes) == 0:
                    continue
                subgraph = _build_client_subgraph(graph, NODE_TYPE, owned_nodes, hops=client_hops)
                local_train_nodes = int(subgraph.nodes[NODE_TYPE].data["train_mask"].sum().item())
                clients.append(
                    ClientShard(
                        client_id=client_id,
                        owned_global_nodes=owned_nodes,
                        subgraph=subgraph,
                        train_nodes=local_train_nodes,
                    )
                )
        _memory_log(
            "load_ieee_cis_dataset: client_shards_ready "
            f"num_clients={int(len(clients))} mode={client_subgraph_mode}"
        )

        class_labels = graph.nodes[NODE_TYPE].data["label"][graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool()]
        if class_labels.numel() == 0:
            class_counts = torch.ones(2, dtype=torch.float32)
        else:
            class_counts = torch.bincount(class_labels.long(), minlength=2).float().clamp(min=1.0)
        class_weights = class_counts.sum() / (class_counts * len(class_counts))

        bundle = DatasetBundle(
            name=dataset_name,
            graph=graph,
            node_type=NODE_TYPE,
            relation_order=relation_order,
            class_weights=class_weights,
            class_counts=class_counts,
            clients=clients,
            base_lr=1e-3,
        )
        data_summary = dict(metadata.get("data_summary", {}) or {})
        data_summary["cache_graph_path"] = str(cache_paths.graph_path)
        data_summary["cache_metadata_path"] = str(cache_paths.metadata_path)
        data_summary["cache_artifact_dir"] = str(cache_paths.artifact_dir)
        data_summary["num_clients"] = int(len(clients))
        data_summary["client_subgraph_mode"] = str(client_subgraph_mode)
        data_summary["label_fraction"] = float(label_fraction)
        data_summary["active_learning_feedback_path"] = str(active_learning_feedback_path or "")
        bundle.data_summary = data_summary
        _memory_log(
            "load_ieee_cis_dataset: bundle_ready "
            f"nodes={int(bundle.graph.num_nodes(NODE_TYPE))} relations={len(bundle.relation_order)}"
        )
        return bundle
    finally:
        _stop_memory_log_session(memory_session, stage_name="load_ieee_cis_dataset: finished")
