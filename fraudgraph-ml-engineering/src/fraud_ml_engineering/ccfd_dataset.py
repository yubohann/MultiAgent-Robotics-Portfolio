from __future__ import annotations

"""Credit-card fraud CSV loader used by the legacy ``ccfd`` registry entry."""

import hashlib
import json
import math
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

from .fraud_dataset import (
    ClientShard,
    DatasetBundle,
    SEQUENCE_BUILDER_VERSION,
    _apply_active_learning_feedback,
    _apply_label_scarcity,
    _attach_dataset_context_defaults,
    _attach_relation_sequence,
    _build_client_subgraph,
    _merge_partitions,
    _random_partition,
    _stratified_partition,
)
from .paths import DATA_ROOT, GRAPH_ROOT

CCFD_EXPLICIT_ROOT = DATA_ROOT / "ccfd"
ULB_FALLBACK_ROOT = DATA_ROOT / "ulb"
CCFD_DEFAULT_ROOT = CCFD_EXPLICIT_ROOT
CCFD_RAW_PATH = CCFD_DEFAULT_ROOT / "creditcard.csv"
CCFD_CACHE_GRAPH_PATH = GRAPH_ROOT / "ccfd.dgl"
CCFD_CACHE_METADATA_PATH = GRAPH_ROOT / "ccfd_metadata.json"
CCFD_PUBLIC_NAME = "Credit Card Fraud CSV"
CCFD_SOURCE_PUBLIC_NAMES = {
    "ccfd": "CCFD",
    "ulb": "ULB",
    "creditcard_csv": CCFD_PUBLIC_NAME,
}
NODE_TYPE = "transaction"
CCFD_FEATURE_LEAKAGE_GUARD_COLUMNS = frozenset({"Class"})
CCFD_REQUIRED_COLUMNS = frozenset({"Time", "Amount", "Class"})
CCFD_DEFAULT_MAX_TRANSACTIONS = 60000
CCFD_DEFAULT_TIME_BINS = 24
CCFD_DEFAULT_AMOUNT_BINS = 24
CCFD_DEFAULT_RELATION_WINDOW_NEIGHBORS = 2
CCFD_DEFAULT_TEMPORAL_WINDOW_NEIGHBORS = 3
CCFD_EVENT_SEQUENCE_LENGTH = 8
CCFD_DEFAULT_TRAIN_RATIO = 0.70
CCFD_DEFAULT_VALID_RATIO = 0.15
CCFD_GRAPH_BUILDER_VERSION = f"creditcard_sourceaware_v2::{SEQUENCE_BUILDER_VERSION}"


def _source_display_name(source_dataset_key: str) -> str:
    normalized_key = str(source_dataset_key).lower().strip()
    return str(CCFD_SOURCE_PUBLIC_NAMES.get(normalized_key, CCFD_PUBLIC_NAME))


def _infer_source_dataset_key(*, data_root: Path, csv_path: Path) -> str:
    root_name = str(data_root.name).strip().lower()
    parent_name = str(csv_path.parent.name).strip().lower()
    if root_name == "ulb" or parent_name == "ulb":
        return "ulb"
    if root_name == "ccfd" or parent_name == "ccfd":
        return "ccfd"
    return "creditcard_csv"


def _resolve_ccfd_source(data_root: Path) -> dict[str, Any]:
    requested_root = Path(data_root).expanduser()
    candidate_specs: list[tuple[Path, Path, str]] = []
    seen_candidates: set[tuple[str, str]] = set()

    def _append_candidate(candidate_root: Path, candidate_csv: Path, resolution: str) -> None:
        candidate_root_resolved = candidate_root.resolve()
        candidate_csv_resolved = candidate_csv.resolve()
        candidate_key = (str(candidate_root_resolved), str(candidate_csv_resolved))
        if candidate_key in seen_candidates:
            return
        seen_candidates.add(candidate_key)
        candidate_specs.append((candidate_root_resolved, candidate_csv_resolved, resolution))

    if requested_root.suffix.lower() == ".csv":
        _append_candidate(requested_root.parent, requested_root, "direct_csv")
    else:
        _append_candidate(requested_root, requested_root / "creditcard.csv", "requested_root")

    for fallback_root, resolution in (
        (CCFD_EXPLICIT_ROOT, "fallback_ccfd_root"),
        (ULB_FALLBACK_ROOT, "fallback_ulb_root"),
    ):
        _append_candidate(fallback_root, fallback_root / "creditcard.csv", resolution)

    tried_paths = [str(candidate_csv) for _, candidate_csv, _ in candidate_specs]
    for candidate_root, candidate_csv, resolution in candidate_specs:
        if not candidate_csv.exists():
            continue
        source_dataset_key = _infer_source_dataset_key(data_root=candidate_root, csv_path=candidate_csv)
        return {
            "requested_data_root": requested_root.resolve(),
            "data_root": candidate_root,
            "csv_path": candidate_csv,
            "source_dataset_key": source_dataset_key,
            "source_display_name": _source_display_name(source_dataset_key),
            "source_resolution": resolution,
        }

    tried_summary = "\n".join(f"  - {path}" for path in tried_paths)
    raise FileNotFoundError(
        "Unable to locate a usable credit-card fraud CSV for the `ccfd` loader.\n"
        "Tried these paths:\n"
        f"{tried_summary}\n"
        "Expected schema: creditcard.csv with columns Time, Amount, Class."
    )


def _validate_creditcard_schema(frame: pd.DataFrame, *, csv_path: Path) -> None:
    missing_columns = sorted(CCFD_REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{csv_path} does not match the supported creditcard.csv schema. Missing columns: {missing_columns}"
        )


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
            "sampling_strategy": "full_chronological_credit_card_dataset",
        }

    target_size = int(max(max_transactions, 1000))
    bin_ids = _chronological_bin_ids(len(frame), time_bins)
    bin_sizes = [int(np.sum(bin_ids == bin_id)) for bin_id in range(int(bin_ids.max()) + 1)]
    bin_targets = _allocate_targets(bin_sizes, target_size, preserve_present_groups=False)
    rng = np.random.default_rng(seed)
    sampled_indices: list[np.ndarray] = []

    labels = frame["Class"].to_numpy(dtype=np.int32)
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
        raise ValueError("CCFD sample is too small to build chronological train/valid/test splits.")
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
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    feature_frame = frame.drop(columns=list(CCFD_FEATURE_LEAKAGE_GUARD_COLUMNS), errors="ignore").copy()
    feature_columns = list(feature_frame.columns)
    train_frame = feature_frame.loc[train_mask].copy()

    processed_columns: list[np.ndarray] = []
    numeric_fill_values: dict[str, float] = {}
    for column in feature_columns:
        train_numeric = pd.to_numeric(train_frame[column], errors="coerce")
        fill_value = float(train_numeric.median()) if train_numeric.notna().any() else 0.0
        full_numeric = pd.to_numeric(feature_frame[column], errors="coerce").fillna(fill_value).to_numpy(dtype=np.float32)
        train_values = full_numeric[train_mask]
        mean = float(train_values.mean()) if train_values.size > 0 else 0.0
        std = float(train_values.std()) if train_values.size > 0 else 1.0
        if std < 1e-6:
            std = 1.0
        processed_columns.append(((full_numeric - mean) / std).reshape(-1, 1).astype(np.float32))
        numeric_fill_values[column] = fill_value

    feature_matrix = np.concatenate(processed_columns, axis=1).astype(np.float32)
    metadata = {
        "feature_columns": feature_columns,
        "numeric_columns": feature_columns,
        "numeric_fill_values": numeric_fill_values,
    }
    return feature_matrix, feature_columns, metadata


def _series_to_quantile_codes(series: pd.Series, bins: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        fill_value = float(numeric.median())
        numeric = numeric.fillna(fill_value)
    else:
        numeric = pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    effective_bins = int(max(1, min(int(bins), len(numeric))))
    if effective_bins <= 1:
        codes = pd.Series(np.zeros(len(numeric), dtype=np.int32), index=numeric.index)
    else:
        ranked = numeric.rank(method="first")
        codes = pd.qcut(ranked, q=effective_bins, labels=False, duplicates="drop")
        if codes is None:
            codes = pd.Series(np.zeros(len(numeric), dtype=np.int32), index=numeric.index)
    codes = pd.Series(codes, index=numeric.index).fillna(0).astype(np.int32)
    return codes.map(lambda value: f"{prefix}_{int(value)}")


def _relation_edges_from_series(series: pd.Series, max_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    value_to_nodes: dict[str, list[int]] = {}
    for node_id, raw_value in enumerate(series.tolist()):
        text = str(raw_value).strip()
        if not text:
            continue
        value_to_nodes.setdefault(text, []).append(int(node_id))

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    neighbor_limit = max(int(max_neighbors), 1)
    for nodes in value_to_nodes.values():
        if len(nodes) < 2:
            continue
        ordered = np.asarray(nodes, dtype=np.int64)
        for offset in range(1, min(neighbor_limit, len(ordered) - 1) + 1):
            src = ordered[:-offset]
            dst = ordered[offset:]
            src_parts.append(src)
            dst_parts.append(dst)
            src_parts.append(dst)
            dst_parts.append(src)
    if not src_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(src_parts, axis=0), np.concatenate(dst_parts, axis=0)


def _temporal_past_edges(num_nodes: int, max_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    if num_nodes <= 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    ordered = np.arange(num_nodes, dtype=np.int64)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    neighbor_limit = max(int(max_neighbors), 1)
    for offset in range(1, min(neighbor_limit, num_nodes - 1) + 1):
        src_parts.append(ordered[:-offset])
        dst_parts.append(ordered[offset:])
    if not src_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(src_parts, axis=0), np.concatenate(dst_parts, axis=0)


def _event_storage_dtype(events: torch.Tensor) -> torch.dtype:
    return torch.float16 if events.numel() >= 25_000_000 else torch.float32


def _build_ccfd_event_tensors(
    feature_tensor: torch.Tensor,
    transaction_time: torch.Tensor,
    history_len: int = CCFD_EVENT_SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = feature_tensor.float().cpu()
    times = transaction_time.float().cpu()
    num_nodes = int(features.shape[0])
    sequence_length = max(int(history_len), 1)
    event_sequence = torch.zeros((num_nodes, sequence_length, features.shape[1]), dtype=torch.float32)
    event_mask = torch.zeros((num_nodes, sequence_length), dtype=torch.bool)
    event_time_deltas = torch.zeros((num_nodes, sequence_length), dtype=torch.float32)
    event_token_weights = torch.zeros((num_nodes, sequence_length), dtype=torch.float32)
    event_token_types = torch.zeros((num_nodes, sequence_length), dtype=torch.long)

    for node_index in range(num_nodes):
        start = max(0, node_index - sequence_length + 1)
        history_indices = torch.arange(start, node_index + 1, dtype=torch.long)
        valid_length = int(history_indices.numel())
        insert_start = sequence_length - valid_length
        event_sequence[node_index, insert_start:] = features[history_indices]
        event_mask[node_index, insert_start:] = True
        deltas = (times[node_index] - times[history_indices]).clamp(min=0.0)
        log_deltas = torch.log1p(deltas)
        event_time_deltas[node_index, insert_start:] = log_deltas
        event_token_weights[node_index, insert_start:] = 1.0 / (1.0 + log_deltas)
        if valid_length > 1:
            event_token_types[node_index, insert_start : sequence_length - 1] = 1
        event_token_types[node_index, sequence_length - 1] = 2
    return event_sequence, event_mask, event_time_deltas, event_token_weights, event_token_types


def _attach_ccfd_event_sequence(
    graph: dgl.DGLHeteroGraph,
    history_len: int = CCFD_EVENT_SEQUENCE_LENGTH,
) -> None:
    feature_tensor = graph.nodes[NODE_TYPE].data["feature"].float()
    transaction_time = graph.nodes[NODE_TYPE].data["transaction_time"].float()
    event_sequence, event_mask, event_time_deltas, event_token_weights, event_token_types = _build_ccfd_event_tensors(
        feature_tensor=feature_tensor,
        transaction_time=transaction_time,
        history_len=history_len,
    )
    graph.nodes[NODE_TYPE].data["event_sequence"] = event_sequence.to(dtype=_event_storage_dtype(event_sequence))
    graph.nodes[NODE_TYPE].data["event_mask"] = event_mask.bool()
    graph.nodes[NODE_TYPE].data["event_time_deltas"] = event_time_deltas.float()
    graph.nodes[NODE_TYPE].data["event_token_weights"] = event_token_weights.float()
    graph.nodes[NODE_TYPE].data["event_token_types"] = event_token_types.long()


def _build_edge_dict(
    frame: pd.DataFrame,
    *,
    time_bins: int,
    amount_bins: int,
    relation_window_neighbors: int,
) -> tuple[dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]], dict[str, int]]:
    edge_dict: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    relation_edge_counts: dict[str, int] = {}
    homo_src_parts: list[np.ndarray] = []
    homo_dst_parts: list[np.ndarray] = []

    relation_series = {
        "time_bin": pd.Series(_chronological_bin_ids(len(frame), time_bins), index=frame.index).map(
            lambda value: f"time_bin_{int(value)}"
        ),
        "amount_bin": _series_to_quantile_codes(np.log1p(frame["Amount"].clip(lower=0.0)), amount_bins, "amount_bin"),
    }

    for relation, series in relation_series.items():
        src, dst = _relation_edges_from_series(series, max_neighbors=relation_window_neighbors)
        if len(src) == 0:
            continue
        edge_dict[(NODE_TYPE, relation, NODE_TYPE)] = (
            torch.from_numpy(src.astype(np.int64)),
            torch.from_numpy(dst.astype(np.int64)),
        )
        relation_edge_counts[relation] = int(len(src))
        homo_src_parts.append(src)
        homo_dst_parts.append(dst)

    temporal_src, temporal_dst = _temporal_past_edges(
        len(frame),
        max_neighbors=max(CCFD_DEFAULT_TEMPORAL_WINDOW_NEIGHBORS, relation_window_neighbors),
    )
    if len(temporal_src) > 0:
        edge_dict[(NODE_TYPE, "temporal_past", NODE_TYPE)] = (
            torch.from_numpy(temporal_src.astype(np.int64)),
            torch.from_numpy(temporal_dst.astype(np.int64)),
        )
        relation_edge_counts["temporal_past"] = int(len(temporal_src))
        homo_src_parts.append(temporal_src)
        homo_dst_parts.append(temporal_dst)

    if homo_src_parts:
        homo_src = np.concatenate(homo_src_parts, axis=0).astype(np.int64)
        homo_dst = np.concatenate(homo_dst_parts, axis=0).astype(np.int64)
    else:
        ordered_nodes = np.arange(len(frame), dtype=np.int64)
        homo_src = np.concatenate([ordered_nodes[:-1], ordered_nodes[1:]], axis=0)
        homo_dst = np.concatenate([ordered_nodes[1:], ordered_nodes[:-1]], axis=0)

    edge_dict[(NODE_TYPE, "homo", NODE_TYPE)] = (
        torch.from_numpy(homo_src.astype(np.int64)),
        torch.from_numpy(homo_dst.astype(np.int64)),
    )
    relation_edge_counts["homo"] = int(len(homo_src))
    return edge_dict, relation_edge_counts


def _cache_signature(
    *,
    data_root: Path,
    csv_path: Path,
    source_dataset_key: str,
    max_transactions: int | None,
    time_bins: int,
    amount_bins: int,
    relation_window_neighbors: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "data_root": str(data_root),
        "csv_path": str(csv_path),
        "source_dataset_key": str(source_dataset_key),
        "graph_builder_version": CCFD_GRAPH_BUILDER_VERSION,
        "max_transactions": None if max_transactions is None else int(max_transactions),
        "time_bins": int(time_bins),
        "amount_bins": int(amount_bins),
        "relation_window_neighbors": int(relation_window_neighbors),
        "train_ratio": float(train_ratio),
        "valid_ratio": float(valid_ratio),
        "seed": int(seed),
        "feature_leakage_guard_columns": sorted(CCFD_FEATURE_LEAKAGE_GUARD_COLUMNS),
    }


def _default_cache_signature() -> dict[str, Any] | None:
    try:
        default_source_info = _resolve_ccfd_source(CCFD_DEFAULT_ROOT)
    except FileNotFoundError:
        return None
    return _cache_signature(
        data_root=Path(default_source_info["data_root"]),
        csv_path=Path(default_source_info["csv_path"]),
        source_dataset_key=str(default_source_info["source_dataset_key"]),
        max_transactions=CCFD_DEFAULT_MAX_TRANSACTIONS,
        time_bins=CCFD_DEFAULT_TIME_BINS,
        amount_bins=CCFD_DEFAULT_AMOUNT_BINS,
        relation_window_neighbors=CCFD_DEFAULT_RELATION_WINDOW_NEIGHBORS,
        train_ratio=CCFD_DEFAULT_TRAIN_RATIO,
        valid_ratio=CCFD_DEFAULT_VALID_RATIO,
        seed=42,
    )


def _resolve_cache_paths(signature: dict[str, Any]) -> tuple[Path, Path]:
    default_signature = _default_cache_signature()
    if default_signature is not None and signature == default_signature:
        return CCFD_CACHE_GRAPH_PATH, CCFD_CACHE_METADATA_PATH

    digest = hashlib.sha1(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    cache_dir = CCFD_CACHE_GRAPH_PATH.parent / "cache"
    return cache_dir / f"ccfd_{digest}.dgl", cache_dir / f"ccfd_{digest}.json"


def _load_cached_graph(
    *,
    signature: dict[str, Any],
    graph_path: Path,
    metadata_path: Path,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]] | None:
    if not graph_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if dict(metadata.get("cache_signature", {})) != signature:
        return None
    graph = dgl.load_graphs(str(graph_path))[0][0]
    return graph, metadata


def _write_cache(
    *,
    graph: dgl.DGLHeteroGraph,
    metadata: dict[str, Any],
    graph_path: Path,
    metadata_path: Path,
) -> None:
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(graph_path), [graph])
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def _build_graph_payload(
    *,
    source_info: dict[str, Any],
    cache_signature: dict[str, Any],
    max_transactions: int | None,
    time_bins: int,
    amount_bins: int,
    relation_window_neighbors: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[dgl.DGLHeteroGraph, dict[str, Any]]:
    requested_data_root = Path(source_info["requested_data_root"])
    data_root = Path(source_info["data_root"])
    csv_path = Path(source_info["csv_path"])
    source_dataset_key = str(source_info["source_dataset_key"])
    source_display_name = str(source_info["source_display_name"])
    source_resolution = str(source_info["source_resolution"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing credit-card fraud CSV: {csv_path}")

    frame = pd.read_csv(csv_path, low_memory=False)
    _validate_creditcard_schema(frame, csv_path=csv_path)
    frame = frame.sort_values(["Time"], kind="mergesort").reset_index(drop=True)
    sampled_frame, sampling_info = _time_stratified_sample(
        frame,
        max_transactions=max_transactions,
        time_bins=time_bins,
        seed=seed,
    )
    sampled_frame = sampled_frame.sort_values(["Time"], kind="mergesort").reset_index(drop=True)

    train_mask, valid_mask, test_mask = _chronological_split_masks(
        sampled_frame,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    labels = sampled_frame["Class"].to_numpy(dtype=np.int64)
    train_mask_np = train_mask.cpu().numpy().astype(bool)
    feature_matrix, feature_columns, feature_metadata = _fit_feature_preprocessor(sampled_frame, train_mask=train_mask_np)
    edge_dict, relation_edge_counts = _build_edge_dict(
        sampled_frame,
        time_bins=time_bins,
        amount_bins=amount_bins,
        relation_window_neighbors=relation_window_neighbors,
    )

    graph = dgl.heterograph(edge_dict, num_nodes_dict={NODE_TYPE: len(sampled_frame)})
    graph.nodes[NODE_TYPE].data["feature"] = torch.from_numpy(feature_matrix.astype(np.float32))
    graph.nodes[NODE_TYPE].data["label"] = torch.from_numpy(labels.astype(np.int64))
    graph.nodes[NODE_TYPE].data["train_mask"] = train_mask.bool()
    graph.nodes[NODE_TYPE].data["valid_mask"] = valid_mask.bool()
    graph.nodes[NODE_TYPE].data["test_mask"] = test_mask.bool()
    graph.nodes[NODE_TYPE].data["transaction_time"] = torch.from_numpy(
        sampled_frame["Time"].to_numpy(dtype=np.float32)
    )

    graph.nodes[NODE_TYPE].data["train_supervised_mask"] = train_mask.bool().clone()
    graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros_like(train_mask.bool())
    graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = torch.full(
        (graph.num_nodes(NODE_TYPE),),
        1.0,
        dtype=torch.float32,
    )
    _attach_dataset_context_defaults(graph, dataset_name="ccfd")

    homo_src, homo_dst = graph.edges(etype="homo")
    homo_edge_labels = torch.where(
        graph.nodes[NODE_TYPE].data["label"][homo_src] == graph.nodes[NODE_TYPE].data["label"][homo_dst],
        torch.ones_like(homo_src, dtype=torch.float32),
        -torch.ones_like(homo_src, dtype=torch.float32),
    )
    edge_train_mask = train_mask[homo_src] & train_mask[homo_dst]
    graph.edges["homo"].data["label"] = homo_edge_labels
    graph.edges["homo"].data["train_mask"] = edge_train_mask.bool()
    _attach_ccfd_event_sequence(graph)
    relation_order = _attach_relation_sequence(graph, dataset_name="ccfd")

    split_stats = {
        "train_positive": int(labels[train_mask_np].sum()),
        "valid_positive": int(labels[valid_mask.cpu().numpy().astype(bool)].sum()),
        "test_positive": int(labels[test_mask.cpu().numpy().astype(bool)].sum()),
    }
    metadata = {
        "cache_signature": cache_signature,
        "data_summary": {
            "dataset_display_name": source_display_name,
            "source_display_name": source_display_name,
            "source_dataset_key": source_dataset_key,
            "source_resolution": source_resolution,
            "requested_data_root": str(requested_data_root),
            "data_root": str(data_root),
            "csv_path": str(csv_path),
            "feature_columns": feature_columns,
            "feature_dim": int(feature_matrix.shape[1]),
            "numeric_feature_count": int(len(feature_metadata["numeric_columns"])),
            "num_nodes": int(graph.num_nodes(NODE_TYPE)),
            "relation_columns_used": list(relation_order),
            "relation_edge_counts": relation_edge_counts,
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
            "original_rows": int(sampling_info["original_rows"]),
            "sampled_rows": int(sampling_info["sampled_rows"]),
            "time_bins": int(sampling_info["time_bins"]),
            "amount_bins": int(amount_bins),
            "split_strategy": "chronological_time_holdout",
            "train_ratio": float(train_ratio),
            "valid_ratio": float(valid_ratio),
            "test_ratio": float(max(1.0 - train_ratio - valid_ratio, 0.0)),
            "graph_builder_version": CCFD_GRAPH_BUILDER_VERSION,
            "event_sequence_length": int(CCFD_EVENT_SEQUENCE_LENGTH),
            "feature_leakage_guard_columns": sorted(CCFD_FEATURE_LEAKAGE_GUARD_COLUMNS),
            "relation_window_neighbors": int(relation_window_neighbors),
            "max_transactions": None if max_transactions is None else int(max_transactions),
            "transaction_time_min": float(sampled_frame["Time"].min()),
            "transaction_time_max": float(sampled_frame["Time"].max()),
            "cache_graph_path": str(CCFD_CACHE_GRAPH_PATH),
            "cache_metadata_path": str(CCFD_CACHE_METADATA_PATH),
        },
    }
    return graph, metadata


def _clone_graph_for_runtime(graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
    # The loader already owns a fresh graph instance from disk/build cache,
    # so cloning here only duplicates memory with no runtime benefit.
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


def load_ccfd_dataset(
    *,
    data_root: str | Path = CCFD_DEFAULT_ROOT,
    dataset_name: str = "ccfd",
    num_clients: int = 3,
    seed: int = 42,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    active_learning_feedback_path: str = "",
    max_transactions: int | None = CCFD_DEFAULT_MAX_TRANSACTIONS,
    time_bins: int = CCFD_DEFAULT_TIME_BINS,
    amount_bins: int = CCFD_DEFAULT_AMOUNT_BINS,
    relation_window_neighbors: int = CCFD_DEFAULT_RELATION_WINDOW_NEIGHBORS,
    train_ratio: float = CCFD_DEFAULT_TRAIN_RATIO,
    valid_ratio: float = CCFD_DEFAULT_VALID_RATIO,
) -> DatasetBundle:
    requested_root = Path(data_root).expanduser()
    source_info = _resolve_ccfd_source(requested_root)
    resolved_root = Path(source_info["data_root"])
    csv_path = Path(source_info["csv_path"])
    signature = _cache_signature(
        data_root=resolved_root,
        csv_path=csv_path,
        source_dataset_key=str(source_info["source_dataset_key"]),
        max_transactions=max_transactions,
        time_bins=time_bins,
        amount_bins=amount_bins,
        relation_window_neighbors=relation_window_neighbors,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    cache_graph_path, cache_metadata_path = _resolve_cache_paths(signature)
    cached = _load_cached_graph(
        signature=signature,
        graph_path=cache_graph_path,
        metadata_path=cache_metadata_path,
    )
    if cached is None:
        graph, metadata = _build_graph_payload(
            source_info=source_info,
            cache_signature=signature,
            max_transactions=max_transactions,
            time_bins=time_bins,
            amount_bins=amount_bins,
            relation_window_neighbors=relation_window_neighbors,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            seed=seed,
        )
        _write_cache(
            graph=graph,
            metadata=metadata,
            graph_path=cache_graph_path,
            metadata_path=cache_metadata_path,
        )
    else:
        graph, metadata = cached

    graph = _clone_graph_for_runtime(graph)
    _attach_dataset_context_defaults(graph, dataset_name=dataset_name)
    _reset_runtime_masks(graph)
    if float(label_fraction) < 0.999:
        _apply_label_scarcity(graph, label_fraction=float(label_fraction), seed=seed)
    if active_learning_feedback_path:
        _apply_active_learning_feedback(graph, active_learning_feedback_path, dataset_name=dataset_name)
    relation_order = list(graph.etypes)
    if (
        "sequence" not in graph.nodes[NODE_TYPE].data
        or "sequence_token_weights" not in graph.nodes[NODE_TYPE].data
        or "sequence_token_types" not in graph.nodes[NODE_TYPE].data
        or "sequence_relation_ids" not in graph.nodes[NODE_TYPE].data
    ):
        relation_order = _attach_relation_sequence(graph, dataset_name=dataset_name)
    if (
        "event_sequence" not in graph.nodes[NODE_TYPE].data
        or "event_mask" not in graph.nodes[NODE_TYPE].data
        or "event_time_deltas" not in graph.nodes[NODE_TYPE].data
        or "event_token_weights" not in graph.nodes[NODE_TYPE].data
        or "event_token_types" not in graph.nodes[NODE_TYPE].data
    ):
        _attach_ccfd_event_sequence(graph)

    train_supervised_mask = graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool() & graph.nodes[NODE_TYPE].data[
        "train_mask"
    ].bool()
    train_unlabeled_mask = graph.nodes[NODE_TYPE].data["train_unlabeled_mask"].bool() & graph.nodes[NODE_TYPE].data[
        "train_mask"
    ].bool()
    supervised_nodes = train_supervised_mask.nonzero(as_tuple=False).flatten()
    supervised_labels = graph.nodes[NODE_TYPE].data["label"][train_supervised_mask]
    unlabeled_nodes = train_unlabeled_mask.nonzero(as_tuple=False).flatten()
    supervised_partitions = _stratified_partition(
        supervised_nodes,
        supervised_labels,
        num_clients=max(int(num_clients), 1),
        seed=seed,
    )
    unlabeled_partitions = _random_partition(unlabeled_nodes, num_clients=max(int(num_clients), 1), seed=seed + 1)
    owned_partitions = _merge_partitions(supervised_partitions, unlabeled_partitions)

    clients: list[ClientShard] = []
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

    class_labels = graph.nodes[NODE_TYPE].data["label"][graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool()]
    if class_labels.numel() == 0:
        class_counts = torch.ones(2, dtype=torch.float32)
    else:
        class_counts = torch.bincount(class_labels.long(), minlength=2).float()
        class_counts = torch.clamp(class_counts, min=1.0)
    class_weights = class_counts.sum() / (class_counts * len(class_counts))

    bundle = DatasetBundle(
        name=dataset_name,
        graph=graph,
        node_type=NODE_TYPE,
        relation_order=relation_order,
        class_weights=class_weights.float(),
        class_counts=class_counts.float(),
        clients=clients,
        base_lr=1e-3,
    )
    data_summary = dict(metadata.get("data_summary", {}) or {})
    data_summary["dataset"] = str(dataset_name)
    data_summary["dataset_registry_name"] = str(dataset_name)
    data_summary["dataset_display_name"] = str(
        data_summary.get("source_display_name")
        or data_summary.get("dataset_display_name")
        or _source_display_name(str(source_info["source_dataset_key"]))
    )
    data_summary["num_clients"] = int(len(clients))
    bundle.data_summary = data_summary
    return bundle
