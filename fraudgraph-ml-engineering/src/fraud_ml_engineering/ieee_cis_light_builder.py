from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ieee_cis_feature_profiles import pass2_required_columns, scan_columns_for_manifest
from .ieee_cis_profiles import (
    IEEE_DEFAULT_DATA_ROOT,
    ieee_cache_asset_root,
    ieee_profile_runtime_summary,
    normalize_ieee_sampling_profile,
    resolve_ieee_history_len,
    resolve_ieee_max_transactions,
    resolve_ieee_relation_columns,
)

MANIFEST_CHUNK_SIZE = 100_000


@dataclass(frozen=True)
class IEEEAssetLayout:
    root_dir: Path
    metadata_path: Path
    manifest_path: Path
    transactions_subset_path: Path
    identity_subset_path: Path
    merged_subset_path: Path
    views_dir: Path
    typed_static_path: Path
    sequence_view_path: Path
    graph_view_path: Path
    hybrid_graph_path: Path
    graph_metadata_path: Path
    hybrid_metadata_path: Path
    edge_tables_dir: Path


def _preferred_layout(root_dir: Path) -> IEEEAssetLayout:
    views_dir = root_dir / "views"
    graph_cache_dir = views_dir / "graph_cache"
    return IEEEAssetLayout(
        root_dir=root_dir,
        metadata_path=root_dir / "metadata.json",
        manifest_path=root_dir / "manifest.parquet",
        transactions_subset_path=root_dir / "transactions_subset.parquet",
        identity_subset_path=root_dir / "identity_subset.parquet",
        merged_subset_path=root_dir / "merged_subset.parquet",
        views_dir=views_dir,
        typed_static_path=views_dir / "typed_static.npz",
        sequence_view_path=views_dir / "sequence_view.npz",
        graph_view_path=graph_cache_dir / "graph_view.dgl",
        hybrid_graph_path=graph_cache_dir / "hybrid_view.dgl",
        graph_metadata_path=graph_cache_dir / "graph_view.json",
        hybrid_metadata_path=graph_cache_dir / "hybrid_view.json",
        edge_tables_dir=views_dir / "edge_tables",
    )


def _stable_digest(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:12]


def _source_paths(data_root: str | Path) -> tuple[Path, Path]:
    root = Path(data_root).expanduser().resolve()
    transaction_path = root / "hf_graph" / "raw" / "train_transaction.csv"
    identity_path = root / "hf_graph" / "raw" / "train_identity.csv"
    if not transaction_path.exists():
        raise FileNotFoundError(f"Missing IEEE-CIS transaction file: {transaction_path}")
    if not identity_path.exists():
        raise FileNotFoundError(f"Missing IEEE-CIS identity file: {identity_path}")
    return transaction_path, identity_path


def _source_file_stamp(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    }


def resolve_ieee_asset_layout(
    *,
    data_root: str | Path = IEEE_DEFAULT_DATA_ROOT,
    data_profile: str,
    loader_view: str,
    relation_profile: str,
    feature_profile: str,
    history_len: int,
    sampling_profile: str,
    max_transactions: int | None,
    time_bins: int,
    relation_window_neighbors: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> IEEEAssetLayout:
    transaction_path, identity_path = _source_paths(data_root)
    runtime_summary = ieee_profile_runtime_summary(
        data_profile=data_profile,
        loader_view=loader_view,
        relation_profile=relation_profile,
        feature_profile=feature_profile,
        history_len=history_len,
        sampling_profile=sampling_profile,
        max_transactions=max_transactions,
    )
    signature_payload = {
        **runtime_summary,
        "time_bins": int(time_bins),
        "relation_window_neighbors": int(relation_window_neighbors),
        "train_ratio": float(train_ratio),
        "valid_ratio": float(valid_ratio),
        "seed": int(seed),
        "transaction_source": _source_file_stamp(transaction_path),
        "identity_source": _source_file_stamp(identity_path),
    }
    asset_root = ieee_cache_asset_root(
        data_root,
        asset_family=str(runtime_summary["asset_family"]),
    ) / _stable_digest(signature_payload)
    return _preferred_layout(asset_root)


def _write_dataframe_cache(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return {"path": str(path), "format": "parquet"}
    except Exception:
        csv_path = path.with_suffix(".csv.gz")
        frame.to_csv(csv_path, index=False, compression="gzip")
        return {"path": str(csv_path), "format": "csv.gz"}


def read_dataframe_cache(info: dict[str, Any]) -> pd.DataFrame:
    path = Path(str(info.get("path", ""))).expanduser()
    format_name = str(info.get("format", "")).strip().lower()
    if format_name == "parquet":
        return pd.read_parquet(path)
    if format_name == "csv.gz":
        return pd.read_csv(path, low_memory=False, compression="gzip")
    raise FileNotFoundError(f"Unsupported cached frame format: {info}")


def load_ieee_asset_metadata(layout: IEEEAssetLayout) -> dict[str, Any] | None:
    if not layout.metadata_path.exists():
        return None
    return json.loads(layout.metadata_path.read_text(encoding="utf-8-sig"))


def _chronological_bin_ids(size: int, bins: int) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=np.int32)
    effective_bins = int(max(1, min(int(bins), size)))
    edges = np.linspace(0, size, effective_bins + 1, dtype=np.int64)
    codes = np.empty(size, dtype=np.int32)
    for bin_id in range(effective_bins):
        start = int(edges[bin_id])
        end = int(edges[bin_id + 1])
        codes[start:end] = int(bin_id)
    return codes


def _allocate_targets(group_sizes: list[int], total_target: int) -> list[int]:
    total_available = int(sum(max(int(size), 0) for size in group_sizes))
    if total_available <= 0:
        return [0 for _ in group_sizes]
    if int(total_target) >= total_available:
        return [int(size) for size in group_sizes]
    target = max(int(total_target), 0)
    scaled = [int(np.floor((int(size) / total_available) * target)) for size in group_sizes]
    scaled = [min(int(group_sizes[index]), int(value)) for index, value in enumerate(scaled)]
    remainder = int(target - sum(scaled))
    if remainder > 0:
        fractions = [
            ((int(group_sizes[index]) / total_available) * target) - scaled[index]
            for index in range(len(group_sizes))
        ]
        for index in np.argsort(np.asarray(fractions))[::-1].tolist():
            if remainder <= 0:
                break
            if scaled[index] < int(group_sizes[index]):
                scaled[index] += 1
                remainder -= 1
    return [int(value) for value in scaled]


def _normalize_relation_token(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "__missing__"}:
        return None
    return text


def _compute_hard_negative_scores(frame: pd.DataFrame, relation_columns: tuple[str, ...]) -> np.ndarray:
    positive_mask = (frame["split_name"].astype(str) == "train") & (frame["isFraud"].fillna(0).astype(int) == 1)
    if not bool(np.any(positive_mask)):
        return np.zeros(len(frame), dtype=np.float32)
    from .ieee_cis_feature_profiles import relation_source_columns

    value_sets: dict[str, set[str]] = {}
    for column in relation_source_columns(relation_columns):
        if column not in frame.columns:
            continue
        tokens = {
            token
            for token in (_normalize_relation_token(value) for value in frame.loc[positive_mask, column].tolist())
            if token is not None
        }
        if tokens:
            value_sets[str(column)] = tokens
    if not value_sets:
        return np.zeros(len(frame), dtype=np.float32)
    scores = np.zeros(len(frame), dtype=np.float32)
    for column, token_set in value_sets.items():
        scores += np.asarray(
            [
                1.0 if _normalize_relation_token(value) in token_set else 0.0
                for value in frame[column].tolist()
            ],
            dtype=np.float32,
        )
    return scores


def _assign_split_names(frame: pd.DataFrame, *, train_ratio: float, valid_ratio: float) -> np.ndarray:
    num_rows = int(len(frame))
    train_end = int(round(num_rows * float(train_ratio)))
    valid_end = int(round(num_rows * float(train_ratio + valid_ratio)))
    train_end = min(max(train_end, 1), max(num_rows - 2, 1))
    valid_end = min(max(valid_end, train_end + 1), max(num_rows - 1, 1))
    split_names = np.full(num_rows, "test", dtype=object)
    split_names[:train_end] = "train"
    split_names[train_end:valid_end] = "valid"
    return split_names


def _sample_one_split(
    split_frame: pd.DataFrame,
    *,
    split_name: str,
    target_size: int | None,
    time_bins: int,
    seed: int,
    sampling_profile: str,
) -> pd.DataFrame:
    if target_size is None or target_size >= len(split_frame):
        result = split_frame.copy()
        result["keep"] = True
        result["sampling_reason"] = "full_split_kept"
        return result
    rng = np.random.default_rng(int(seed))
    split_frame = split_frame.copy()
    labels = split_frame["isFraud"].fillna(0).astype(int).to_numpy(dtype=np.int64)
    positive_index = np.flatnonzero(labels == 1)
    negative_index = np.flatnonzero(labels == 0)
    if str(sampling_profile) == "normal_only_train" and split_name == "train":
        positive_index = np.empty(0, dtype=np.int64)
    chosen_positive = positive_index
    if len(chosen_positive) > int(target_size):
        pos_bin_ids = _chronological_bin_ids(len(chosen_positive), time_bins)
        allocations = _allocate_targets(
            [int(np.sum(pos_bin_ids == bin_id)) for bin_id in np.unique(pos_bin_ids).tolist()],
            int(target_size),
        )
        chosen_parts: list[np.ndarray] = []
        for position, bin_id in enumerate(np.unique(pos_bin_ids).tolist()):
            bin_members = chosen_positive[pos_bin_ids == int(bin_id)]
            take = min(len(bin_members), int(allocations[position]))
            if take > 0:
                chosen_parts.append(np.sort(rng.choice(bin_members, size=take, replace=False)).astype(np.int64))
        chosen_positive = np.sort(np.concatenate(chosen_parts, axis=0)) if chosen_parts else np.empty(0, dtype=np.int64)

    remaining_budget = max(int(target_size) - int(len(chosen_positive)), 0)
    if remaining_budget >= len(negative_index):
        chosen_negative = negative_index
    else:
        negative_bin_ids = _chronological_bin_ids(len(negative_index), time_bins)
        allocations = _allocate_targets(
            [int(np.sum(negative_bin_ids == bin_id)) for bin_id in np.unique(negative_bin_ids).tolist()],
            remaining_budget,
        )
        chosen_parts = []
        for position, bin_id in enumerate(np.unique(negative_bin_ids).tolist()):
            bin_members = negative_index[negative_bin_ids == int(bin_id)]
            take = min(len(bin_members), int(allocations[position]))
            if take <= 0:
                continue
            local_scores = split_frame.iloc[bin_members]["hard_negative_score"].to_numpy(dtype=np.float32)
            noise = rng.random(len(bin_members))
            order = np.lexsort((noise, -local_scores))
            chosen_parts.append(np.sort(bin_members[order[:take]]).astype(np.int64))
        chosen_negative = np.sort(np.concatenate(chosen_parts, axis=0)) if chosen_parts else np.empty(0, dtype=np.int64)

    chosen_index = np.sort(np.concatenate([chosen_positive, chosen_negative], axis=0)).astype(np.int64)
    sampled = split_frame.iloc[chosen_index].copy()
    sampled["keep"] = True
    sampled["sampling_reason"] = str(sampling_profile)
    return sampled


def _sample_manifest(
    frame: pd.DataFrame,
    *,
    max_transactions: int | None,
    time_bins: int,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    relation_columns: tuple[str, ...],
    sampling_profile: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = frame.copy()
    manifest["split_name"] = _assign_split_names(manifest, train_ratio=train_ratio, valid_ratio=valid_ratio)
    manifest["hard_negative_score"] = _compute_hard_negative_scores(manifest, relation_columns)
    if max_transactions is None or int(max_transactions) <= 0 or int(max_transactions) >= len(manifest):
        manifest["keep"] = True
        manifest["sampling_reason"] = "full_dataset_kept"
        return manifest, {
            "sampling_applied": False,
            "sampling_strategy": "chrono_full",
            "original_rows": int(len(frame)),
            "sampled_rows": int(len(frame)),
            "requested_max_transactions": None if max_transactions is None else int(max_transactions),
        }

    split_names = ["train", "valid", "test"]
    split_sizes = [int(np.sum(manifest["split_name"].to_numpy(dtype=object) == split_name)) for split_name in split_names]
    allocations = _allocate_targets(split_sizes, int(max_transactions))
    sampled_parts: list[pd.DataFrame] = []
    for offset, split_name in enumerate(split_names):
        split_frame = manifest.loc[manifest["split_name"].astype(str) == split_name].reset_index(drop=True)
        if split_frame.empty:
            continue
        sampled_parts.append(
            _sample_one_split(
                split_frame,
                split_name=split_name,
                target_size=int(allocations[offset]),
                time_bins=time_bins,
                seed=seed + offset,
                sampling_profile=sampling_profile,
            )
        )
    sampled = pd.concat(sampled_parts, axis=0, ignore_index=True) if sampled_parts else manifest.iloc[0:0].copy()
    sampled = sampled.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    return sampled, {
        "sampling_applied": True,
        "sampling_strategy": str(sampling_profile),
        "original_rows": int(len(frame)),
        "sampled_rows": int(len(sampled)),
        "requested_max_transactions": int(max_transactions),
        "split_sizes_before": {split_names[index]: int(split_sizes[index]) for index in range(len(split_names))},
        "split_sizes_after": {
            split_name: int(np.sum(sampled["split_name"].astype(str) == split_name))
            for split_name in split_names
        },
    }


def _read_header(path: Path) -> list[str]:
    return [str(item) for item in pd.read_csv(path, nrows=0).columns.tolist()]


def _load_identity_minimal(identity_path: Path, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=["TransactionID"])
    return pd.read_csv(identity_path, usecols=["TransactionID", *columns], low_memory=False)


def _build_manifest_frame(
    *,
    transaction_path: Path,
    identity_path: Path,
    relation_columns: tuple[str, ...],
    time_bins: int,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    max_transactions: int | None,
    sampling_profile: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    transaction_columns = _read_header(transaction_path)
    identity_columns = _read_header(identity_path)
    scan_columns = scan_columns_for_manifest(relation_columns)
    transaction_scan_columns = [column for column in scan_columns if column in transaction_columns]
    identity_scan_columns = [column for column in scan_columns if column in identity_columns and column != "TransactionID"]
    identity_minimal = _load_identity_minimal(identity_path, identity_scan_columns)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        transaction_path,
        usecols=transaction_scan_columns,
        chunksize=MANIFEST_CHUNK_SIZE,
        low_memory=False,
    ):
        chunks.append(chunk.merge(identity_minimal, on="TransactionID", how="left", sort=False))
    manifest_frame = pd.concat(chunks, axis=0, ignore_index=True) if chunks else pd.DataFrame(columns=transaction_scan_columns)
    manifest_frame = manifest_frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    sampled_manifest, sampling_info = _sample_manifest(
        manifest_frame,
        max_transactions=max_transactions,
        time_bins=time_bins,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        relation_columns=relation_columns,
        sampling_profile=sampling_profile,
    )
    return sampled_manifest, {
        **sampling_info,
        "manifest_columns": [str(item) for item in sampled_manifest.columns],
        "manifest_minimal_scan_columns": [str(item) for item in scan_columns],
    }


def _load_subset_from_csv(
    path: Path,
    *,
    usecols: list[str],
    keep_ids: set[int],
) -> pd.DataFrame:
    if not usecols:
        return pd.DataFrame()
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=MANIFEST_CHUNK_SIZE, low_memory=False):
        ids = pd.to_numeric(chunk["TransactionID"], errors="coerce").fillna(-1).astype(np.int64)
        mask = ids.isin(list(keep_ids))
        if bool(mask.any()):
            chunks.append(chunk.loc[mask].copy())
    return pd.concat(chunks, axis=0, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def _manifest_hash(manifest: pd.DataFrame) -> str:
    payload = {
        "transaction_ids": manifest["TransactionID"].astype(str).tolist(),
        "split_name": manifest["split_name"].astype(str).tolist(),
        "isFraud": manifest["isFraud"].fillna(0).astype(int).tolist(),
    }
    return _stable_digest(payload)


def ensure_ieee_light_assets(
    *,
    data_root: str | Path = IEEE_DEFAULT_DATA_ROOT,
    data_profile: str,
    loader_view: str,
    relation_profile: str,
    feature_profile: str,
    history_len: int | None,
    sampling_profile: str | None,
    max_transactions: int | None,
    time_bins: int,
    relation_window_neighbors: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    rebuild: bool = False,
) -> tuple[IEEEAssetLayout, dict[str, Any]]:
    resolved_relation_columns = resolve_ieee_relation_columns(relation_profile)
    resolved_history_len = resolve_ieee_history_len(history_len, data_profile=data_profile)
    resolved_max_transactions = resolve_ieee_max_transactions(max_transactions, data_profile=data_profile)
    resolved_sampling_profile = normalize_ieee_sampling_profile(sampling_profile, data_profile=data_profile)
    layout = resolve_ieee_asset_layout(
        data_root=data_root,
        data_profile=data_profile,
        loader_view=loader_view,
        relation_profile=relation_profile,
        feature_profile=feature_profile,
        history_len=resolved_history_len,
        sampling_profile=resolved_sampling_profile,
        max_transactions=resolved_max_transactions,
        time_bins=time_bins,
        relation_window_neighbors=relation_window_neighbors,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    if not rebuild:
        metadata = load_ieee_asset_metadata(layout)
        if metadata is not None:
            return layout, metadata

    transaction_path, identity_path = _source_paths(data_root)
    manifest, manifest_info = _build_manifest_frame(
        transaction_path=transaction_path,
        identity_path=identity_path,
        relation_columns=resolved_relation_columns,
        time_bins=time_bins,
        seed=seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        max_transactions=resolved_max_transactions,
        sampling_profile=resolved_sampling_profile,
    )
    keep_ids = {
        int(value)
        for value in pd.to_numeric(manifest["TransactionID"], errors="coerce").fillna(-1).astype(np.int64).tolist()
        if int(value) >= 0
    }
    transaction_columns = _read_header(transaction_path)
    identity_columns = _read_header(identity_path)
    pass2_transaction_columns = [
        column
        for column in pass2_required_columns(
            transaction_columns,
            relation_columns=resolved_relation_columns,
            feature_profile=feature_profile,
        )
        if column in transaction_columns
    ]
    pass2_identity_columns = [
        column
        for column in pass2_required_columns(
            identity_columns,
            relation_columns=resolved_relation_columns,
            feature_profile=feature_profile,
        )
        if column in identity_columns
    ]
    if "TransactionID" not in pass2_transaction_columns:
        pass2_transaction_columns = ["TransactionID", *pass2_transaction_columns]
    if "TransactionID" not in pass2_identity_columns:
        pass2_identity_columns = ["TransactionID", *pass2_identity_columns]

    transactions_subset = _load_subset_from_csv(
        transaction_path,
        usecols=pass2_transaction_columns,
        keep_ids=keep_ids,
    )
    identity_subset = _load_subset_from_csv(
        identity_path,
        usecols=pass2_identity_columns,
        keep_ids=keep_ids,
    )
    merged_subset = transactions_subset.merge(identity_subset, on="TransactionID", how="left", sort=False)
    merged_subset = merged_subset.merge(
        manifest[["TransactionID", "split_name", "hard_negative_score", "sampling_reason"]],
        on="TransactionID",
        how="left",
        sort=False,
    )
    merged_subset = merged_subset.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    manifest = manifest.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)

    layout.root_dir.mkdir(parents=True, exist_ok=True)
    transactions_info = _write_dataframe_cache(layout.transactions_subset_path, transactions_subset)
    identity_info = _write_dataframe_cache(layout.identity_subset_path, identity_subset)
    merged_info = _write_dataframe_cache(layout.merged_subset_path, merged_subset)
    manifest_info_payload = _write_dataframe_cache(layout.manifest_path, manifest)

    runtime_summary = ieee_profile_runtime_summary(
        data_profile=data_profile,
        loader_view=loader_view,
        relation_profile=relation_profile,
        feature_profile=feature_profile,
        history_len=resolved_history_len,
        sampling_profile=resolved_sampling_profile,
        max_transactions=resolved_max_transactions,
    )
    metadata = {
        "asset_family": str(runtime_summary["asset_family"]),
        "data_profile": str(runtime_summary["data_profile"]),
        "loader_view": str(runtime_summary["loader_view"]),
        "relation_profile": str(runtime_summary["relation_profile"]),
        "feature_profile": str(runtime_summary["feature_profile"]),
        "history_len": int(runtime_summary["history_len"]),
        "sampling_profile": str(runtime_summary["sampling_profile"]),
        "requested_max_transactions": None if max_transactions is None else int(max_transactions),
        "effective_max_transactions": None if resolved_max_transactions is None else int(resolved_max_transactions),
        "time_bins": int(time_bins),
        "relation_window_neighbors": int(relation_window_neighbors),
        "train_ratio": float(train_ratio),
        "valid_ratio": float(valid_ratio),
        "test_ratio": float(max(1.0 - float(train_ratio) - float(valid_ratio), 0.0)),
        "seed": int(seed),
        "transaction_source": _source_file_stamp(transaction_path),
        "identity_source": _source_file_stamp(identity_path),
        "tables": {
            "manifest": manifest_info_payload,
            "transactions_subset": transactions_info,
            "identity_subset": identity_info,
            "merged_subset": merged_info,
        },
        "original_rows": int(manifest_info["original_rows"]),
        "sampled_rows": int(manifest_info["sampled_rows"]),
        "train_size": int(np.sum(manifest["split_name"].astype(str) == "train")),
        "valid_size": int(np.sum(manifest["split_name"].astype(str) == "valid")),
        "test_size": int(np.sum(manifest["split_name"].astype(str) == "test")),
        "positive_labels": int(manifest["isFraud"].fillna(0).astype(int).sum()),
        "selected_relation_columns": [str(item) for item in resolved_relation_columns],
        "sampling_strategy": str(manifest_info["sampling_strategy"]),
        "sampling_applied": bool(manifest_info["sampling_applied"]),
        "manifest_hash": _manifest_hash(manifest),
        "split_policy": "chronological_transactiondt_holdout",
        "feature_block_summary": {
            "transaction_columns": [str(item) for item in pass2_transaction_columns],
            "identity_columns": [str(item) for item in pass2_identity_columns],
        },
        "paper_compatible_mode": bool(str(feature_profile).startswith("paper_")),
        "paths": {
            "root_dir": str(layout.root_dir),
            "typed_static_path": str(layout.typed_static_path),
            "sequence_view_path": str(layout.sequence_view_path),
            "graph_view_path": str(layout.graph_view_path),
            "hybrid_graph_path": str(layout.hybrid_graph_path),
            "graph_metadata_path": str(layout.graph_metadata_path),
            "hybrid_metadata_path": str(layout.hybrid_metadata_path),
            "edge_tables_dir": str(layout.edge_tables_dir),
        },
    }
    layout.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return layout, metadata


def load_merged_subset_frame(layout: IEEEAssetLayout, metadata: dict[str, Any]) -> pd.DataFrame:
    tables = dict(metadata.get("tables", {}) or {})
    merged_info = dict(tables.get("merged_subset", {}) or {})
    if not merged_info:
        raise FileNotFoundError(f"IEEE merged subset metadata missing under {layout.metadata_path}")
    return read_dataframe_cache(merged_info)


def load_manifest_frame(layout: IEEEAssetLayout, metadata: dict[str, Any]) -> pd.DataFrame:
    tables = dict(metadata.get("tables", {}) or {})
    manifest_info = dict(tables.get("manifest", {}) or {})
    if not manifest_info:
        raise FileNotFoundError(f"IEEE manifest metadata missing under {layout.metadata_path}")
    return read_dataframe_cache(manifest_info)
