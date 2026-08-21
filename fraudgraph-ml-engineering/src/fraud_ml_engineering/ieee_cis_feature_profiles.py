from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .ieee_cis_profiles import (
    IEEE_FEATURE_PROFILE_PAPER_PRUNED,
    IEEE_FEATURE_PROFILE_PAPER_V30,
    IEEE_FEATURE_PROFILE_TYPED_160,
    IEEE_FEATURE_PROFILE_TYPED_256,
    IEEE_FEATURE_PROFILE_TYPED_FULL,
    resolve_ieee_feature_profile,
)

IEEE_BASE_COLUMNS: tuple[str, ...] = (
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",
)

IEEE_TRANSACTION_RELATION_SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "uid": ("card1", "card2", "card3", "card5"),
    "uid_addr": ("card1", "card2", "card3", "card5", "addr1", "addr2"),
    "uid_email": ("card1", "card2", "card3", "card5", "P_emaildomain"),
    "device_browser": ("DeviceType", "id_31"),
}

IEEE_STATIC_PRIORITY_COLUMNS: tuple[str, ...] = (
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
)

IEEE_MINIMAL_SCAN_COLUMNS: tuple[str, ...] = (
    *IEEE_BASE_COLUMNS,
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "DeviceType",
    "id_31",
)


@dataclass(frozen=True)
class PreparedIEEEFeatures:
    frame: pd.DataFrame
    feature_columns: list[str]
    metadata: dict[str, Any]


def relation_source_columns(relation_columns: tuple[str, ...]) -> tuple[str, ...]:
    required: list[str] = []
    for relation in relation_columns:
        for column in IEEE_TRANSACTION_RELATION_SOURCE_COLUMNS.get(str(relation), ()):
            if column not in required:
                required.append(column)
    return tuple(required)


def scan_columns_for_manifest(relation_columns: tuple[str, ...]) -> tuple[str, ...]:
    required = list(IEEE_MINIMAL_SCAN_COLUMNS)
    for column in relation_source_columns(relation_columns):
        if column not in required:
            required.append(column)
    return tuple(required)


def pass2_required_columns(
    all_columns: list[str],
    *,
    relation_columns: tuple[str, ...],
    feature_profile: str,
) -> tuple[str, ...]:
    resolved_profile = resolve_ieee_feature_profile(feature_profile)
    required: list[str] = [column for column in IEEE_BASE_COLUMNS if column in all_columns]
    for column in relation_source_columns(relation_columns):
        if column in all_columns and column not in required:
            required.append(column)
    priority_prefixes = ("C", "D", "V", "id_", "M")
    for column in IEEE_STATIC_PRIORITY_COLUMNS:
        if column in all_columns and column not in required:
            required.append(column)
    for column in all_columns:
        if column in required:
            continue
        if str(column).startswith(priority_prefixes):
            required.append(str(column))
    if resolved_profile.name == IEEE_FEATURE_PROFILE_TYPED_FULL:
        return tuple(required)
    for column in all_columns:
        if column not in required:
            required.append(str(column))
    return tuple(required)


def _missing_rate(frame: pd.DataFrame, column_name: str) -> float:
    series = frame[column_name]
    if len(series) == 0:
        return 1.0
    return float(series.isna().mean())


def _select_v_columns(frame: pd.DataFrame, *, keep_count: int, train_mask: np.ndarray) -> list[str]:
    candidates = [column for column in frame.columns if str(column).startswith("V")]
    if keep_count <= 0 or len(candidates) <= keep_count:
        return candidates
    if train_mask.size != len(frame):
        train_frame = frame[candidates]
    else:
        train_frame = frame.loc[train_mask, candidates]
    numeric = train_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    variances = numeric.var(axis=0).sort_values(ascending=False)
    return [str(item) for item in variances.index[:keep_count].tolist()]


def _top_variance_feature_indices(
    feature_matrix: np.ndarray,
    *,
    target_dim: int,
    train_mask: np.ndarray | None = None,
) -> np.ndarray:
    if target_dim <= 0 or feature_matrix.ndim != 2 or feature_matrix.shape[1] <= target_dim:
        return np.arange(feature_matrix.shape[1], dtype=np.int64)
    if train_mask is None or train_mask.size != feature_matrix.shape[0] or not np.any(train_mask):
        train_matrix = feature_matrix
    else:
        train_matrix = feature_matrix[train_mask]
    variances = np.var(train_matrix, axis=0)
    order = np.argsort(variances)[::-1]
    return np.asarray(order[:target_dim], dtype=np.int64)


def compress_dense_feature_matrix(
    feature_matrix: np.ndarray,
    feature_columns: list[str],
    *,
    feature_profile: str,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    resolved_profile = resolve_ieee_feature_profile(feature_profile)
    target_dim = resolved_profile.dense_target_dim
    if target_dim is None or feature_matrix.shape[1] <= int(target_dim):
        return feature_matrix, list(feature_columns), {
            "dense_target_dim": None if target_dim is None else int(target_dim),
            "dense_compression_applied": False,
            "dense_selected_feature_indices": list(range(int(feature_matrix.shape[1]))),
        }
    keep_indices = _top_variance_feature_indices(
        feature_matrix,
        target_dim=int(target_dim),
        train_mask=train_mask,
    )
    compressed = np.asarray(feature_matrix[:, keep_indices], dtype=np.float32)
    selected_columns = [str(feature_columns[int(index)]) for index in keep_indices.tolist()]
    return compressed, selected_columns, {
        "dense_target_dim": int(target_dim),
        "dense_compression_applied": True,
        "dense_selected_feature_indices": [int(index) for index in keep_indices.tolist()],
    }


def prepare_ieee_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_profile: str,
    relation_columns: tuple[str, ...],
    train_mask: np.ndarray,
) -> PreparedIEEEFeatures:
    resolved_profile = resolve_ieee_feature_profile(feature_profile)
    relation_source = set(relation_source_columns(relation_columns))
    keep_columns: list[str] = []
    metadata: dict[str, Any] = {
        "feature_profile": resolved_profile.name,
        "missing_threshold": resolved_profile.missing_threshold,
        "compress_v_block_to": resolved_profile.compress_v_block_to,
        "dropped_columns": [],
        "kept_columns": [],
        "selected_relation_source_columns": sorted(relation_source),
    }

    missing_threshold = resolved_profile.missing_threshold
    keep_v_columns: set[str] | None = None
    if resolved_profile.compress_v_block_to is not None:
        keep_v_columns = set(
            _select_v_columns(
                frame,
                keep_count=int(resolved_profile.compress_v_block_to),
                train_mask=np.asarray(train_mask, dtype=bool),
            )
        )
        metadata["v_block_compression"] = {
            "mode": "variance_topk",
            "kept_v_columns": sorted(keep_v_columns),
        }
    else:
        metadata["v_block_compression"] = {
            "mode": "disabled",
            "kept_v_columns": [],
        }

    for column in frame.columns:
        column_name = str(column)
        if column_name in {"TransactionID", "TransactionDT", "isFraud", "split_name"}:
            keep_columns.append(column_name)
            continue
        if column_name in relation_source:
            keep_columns.append(column_name)
            continue
        if column_name.startswith("V") and keep_v_columns is not None and column_name not in keep_v_columns:
            metadata["dropped_columns"].append(column_name)
            continue
        if missing_threshold is not None and _missing_rate(frame, column_name) >= float(missing_threshold):
            metadata["dropped_columns"].append(column_name)
            continue
        keep_columns.append(column_name)

    filtered = frame.loc[:, keep_columns].copy()
    metadata["kept_columns"] = [str(item) for item in filtered.columns]
    return PreparedIEEEFeatures(
        frame=filtered,
        feature_columns=[
            str(column)
            for column in filtered.columns
            if str(column) not in {"TransactionID", "TransactionDT", "isFraud", "split_name"}
            and str(column) not in relation_source
        ],
        metadata=metadata,
    )
