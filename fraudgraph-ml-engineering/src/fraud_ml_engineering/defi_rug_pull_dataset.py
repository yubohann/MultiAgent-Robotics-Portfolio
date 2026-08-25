from __future__ import annotations

"""Loader for curated DeFi rug-pull incidents adapted to SplitGNN + Transformer.

The public local source is positive-only, so a usable binary benchmark still
needs an external negative set.
"""

from pathlib import Path
from typing import Any
import re
from urllib.parse import urlparse

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

from .archive_dataset import (
    _compute_class_counts,
    _compute_class_weights,
    _stratified_split_masks,
    _window_edge_pairs,
)
from .fraud_dataset import (
    ClientShard,
    DatasetBundle,
    _apply_active_learning_feedback,
    _apply_label_scarcity,
    _attach_dataset_context_defaults,
    _attach_relation_sequence,
    _build_client_subgraph,
    _merge_partitions,
    _random_partition,
    _stratified_partition,
)
from .paths import DATA_ROOT

DEFI_RUG_PULL_DEFAULT_ROOT = DATA_ROOT / "defi_rug_pull"
NODE_TYPE = "contract"
RUG_PULL_EVENT_SEQUENCE_LENGTH = 6
UNKNOWN_TOKEN = "unknown"
FEATURE_EXCLUDE_COLUMNS = {
    "incident_id",
    "address",
    "label",
    "label_name",
    "chain",
    "type",
    "root_causes",
    "sources",
    "url",
    "chain_clean",
    "type_clean",
    "root_cause_clean",
    "source_clean",
    "domain_clean",
    "sequence_available",
    "is_anchor",
}


def _normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    for old, new in (
        ("(", "_"),
        (")", ""),
        ("[", "_"),
        ("]", ""),
        (" ", "_"),
        ("-", "_"),
        ("/", "_"),
        (".", "_"),
    ):
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _rename_known_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        normalized = _normalize_column_name(column)
        if normalized in {"no", "incident_id", "id"}:
            rename_map[column] = "incident_id"
        elif normalized in {"chain", "network"}:
            rename_map[column] = "chain"
        elif normalized in {"address", "contract_address"}:
            rename_map[column] = "address"
        elif normalized in {"losses", "loss", "loss_usd", "losses_usd"}:
            rename_map[column] = "losses"
        elif normalized in {"type", "rugpull_type", "incident_type"}:
            rename_map[column] = "type"
        elif normalized in {"root_causes", "root_cause", "rootcause", "rootcauses"}:
            rename_map[column] = "root_causes"
        elif normalized in {"sources", "source"}:
            rename_map[column] = "sources"
        elif normalized in {"url", "urls", "reference_url", "report_url"}:
            rename_map[column] = "url"
        elif normalized in {"label", "target"}:
            rename_map[column] = "label"
        elif normalized in {"label_name", "labelname"}:
            rename_map[column] = "label_name"
    return frame.rename(columns=rename_map)


def _series_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _normalize_category(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "nan", "none", "null", "n/a", "na"}:
        return UNKNOWN_TOKEN
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or UNKNOWN_TOKEN


def _normalize_address_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _parse_loss_series(series: pd.Series) -> np.ndarray:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, ".": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)


def _extract_domain(url_value: object) -> str:
    text = str(url_value or "").strip()
    if text.lower() in {"", "nan", "none", "null", "n/a", "na"}:
        return UNKNOWN_TOKEN
    parsed = urlparse(text if "://" in text else f"https://{text}")
    domain = (parsed.netloc or "").strip().lower()
    if not domain:
        domain = str(parsed.path or "").split("/")[0].strip().lower()
    domain = domain.removeprefix("www.")
    return _normalize_category(domain)


def _resolve_data_root(path_like: str | Path) -> Path:
    requested_root = Path(path_like).expanduser().resolve()
    candidates = [requested_root, requested_root / "rug_pull_dataset-main"]
    for candidate in candidates:
        if (candidate / "rugpull_full_dataset_new.csv").exists() or (candidate / "rugpull_full_dataset_new.parquet").exists():
            return candidate
        if (candidate / "rugpull_dataset.csv").exists() or (candidate / "rugpull_dataset.parquet").exists():
            return candidate
    raise FileNotFoundError(
        "DeFi Rug Pull dataset root is missing expected files. "
        f"Checked: {requested_root} and {requested_root / 'rug_pull_dataset-main'}."
    )


def _read_incident_table(resolved_root: Path, *, force_preview: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    for stem in ("rugpull_full_dataset_new", "rugpull_dataset"):
        parquet_path = resolved_root / f"{stem}.parquet"
        csv_path = resolved_root / f"{stem}.csv"
        if not force_preview and parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path), {"path": str(parquet_path), "format": "parquet"}
            except Exception:
                pass
        if csv_path.exists():
            return pd.read_csv(csv_path, low_memory=False), {"path": str(csv_path), "format": "csv"}
    raise FileNotFoundError(
        f"Missing rug-pull incident table under {resolved_root}. "
        "Expected rugpull_full_dataset_new.[parquet/csv] or rugpull_dataset.[parquet/csv]."
    )


def _load_negative_users(path_like: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Negative user file not found: {path}")

    source_meta = {"path": str(path)}
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        source_meta["format"] = "parquet"
    elif path.suffix.lower() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=separator, low_memory=False)
        source_meta["format"] = path.suffix.lower().lstrip(".")
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            frame = pd.DataFrame({"address": [line.strip() for line in handle if line.strip()]})
        source_meta["format"] = "txt"

    frame = _rename_known_columns(frame.copy())
    if "address" not in frame.columns:
        raise ValueError(f"Negative user file must contain an 'address' column: {path}")

    address_only_input = {"chain", "losses", "type", "root_causes", "sources", "url"}.isdisjoint(frame.columns)
    negative = pd.DataFrame(
        {
            "incident_id": _series_or_default(frame, "incident_id", ""),
            "address": _normalize_address_series(frame["address"]),
            "chain": _series_or_default(frame, "chain", UNKNOWN_TOKEN),
            "losses": _series_or_default(frame, "losses", 0.0),
            "type": _series_or_default(frame, "type", UNKNOWN_TOKEN),
            "root_causes": _series_or_default(frame, "root_causes", UNKNOWN_TOKEN),
            "sources": _series_or_default(frame, "sources", UNKNOWN_TOKEN),
            "url": _series_or_default(frame, "url", ""),
        }
    )
    label_series = pd.to_numeric(_series_or_default(frame, "label", 0), errors="coerce").fillna(0).astype(np.int64)
    negative["label"] = label_series
    negative["label_name"] = _series_or_default(frame, "label_name", "non_rug_pull").astype(str)
    negative["is_anchor"] = 1
    negative = negative[negative["address"] != ""].drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)
    source_meta["address_only_input"] = bool(address_only_input)
    return negative, source_meta


def _top_categories(series: pd.Series, top_k: int) -> list[str]:
    valid = series[series != UNKNOWN_TOKEN]
    if valid.empty:
        return []
    return valid.value_counts().head(int(top_k)).index.tolist()


def _add_top_category_flags(frame: pd.DataFrame, column: str, prefix: str, top_k: int) -> list[str]:
    selected = _top_categories(frame[column], top_k=top_k)
    for value in selected:
        frame[f"{prefix}_flag_{value}"] = (frame[column] == value).astype(np.float32)
    return selected


def _add_frequency_features(frame: pd.DataFrame, column: str, prefix: str) -> None:
    valid = frame[column][frame[column] != UNKNOWN_TOKEN]
    counts = valid.value_counts()
    denominator = max(float(len(valid)), 1.0)
    frame[f"{prefix}_count"] = frame[column].map(counts).fillna(0.0).astype(np.float32)
    frame[f"{prefix}_frequency"] = (frame[f"{prefix}_count"] / denominator).astype(np.float32)


def _engineer_incident_table(raw_users: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    users = _rename_known_columns(raw_users.copy())
    users["address"] = _normalize_address_series(_series_or_default(users, "address", ""))
    users = users[users["address"] != ""].drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)

    users["incident_id"] = _series_or_default(users, "incident_id", "").astype(str)
    users["label"] = pd.to_numeric(_series_or_default(users, "label", 0), errors="coerce").fillna(0).astype(np.int64)
    users["label_name"] = _series_or_default(users, "label_name", "non_rug_pull").astype(str)
    users["is_anchor"] = pd.to_numeric(_series_or_default(users, "is_anchor", 1), errors="coerce").fillna(1).astype(np.int64)
    users["chain"] = _series_or_default(users, "chain", UNKNOWN_TOKEN).astype(str)
    users["type"] = _series_or_default(users, "type", UNKNOWN_TOKEN).astype(str)
    users["root_causes"] = _series_or_default(users, "root_causes", UNKNOWN_TOKEN).astype(str)
    users["sources"] = _series_or_default(users, "sources", UNKNOWN_TOKEN).astype(str)
    users["url"] = _series_or_default(users, "url", "").astype(str)

    users["chain_clean"] = users["chain"].map(_normalize_category)
    users["type_clean"] = users["type"].map(_normalize_category)
    users["root_cause_clean"] = users["root_causes"].map(_normalize_category)
    users["source_clean"] = users["sources"].map(_normalize_category)
    users["domain_clean"] = users["url"].map(_extract_domain)

    loss_usd = _parse_loss_series(users["losses"] if "losses" in users.columns else pd.Series([0.0] * len(users)))
    users["loss_usd"] = loss_usd.astype(np.float32)
    users["loss_log_usd"] = np.log1p(loss_usd).astype(np.float32)
    users["loss_rank"] = pd.Series(loss_usd).rank(pct=True, method="average").fillna(0.0).to_numpy(dtype=np.float32)
    users["loss_severity_bucket"] = np.select(
        [loss_usd >= 1_000_000.0, loss_usd >= 100_000.0, loss_usd >= 10_000.0],
        [3.0, 2.0, 1.0],
        default=0.0,
    ).astype(np.float32)

    users["has_chain"] = (users["chain_clean"] != UNKNOWN_TOKEN).astype(np.float32)
    users["has_type"] = (users["type_clean"] != UNKNOWN_TOKEN).astype(np.float32)
    users["has_root_cause"] = (users["root_cause_clean"] != UNKNOWN_TOKEN).astype(np.float32)
    users["has_source"] = (users["source_clean"] != UNKNOWN_TOKEN).astype(np.float32)
    users["has_domain"] = (users["domain_clean"] != UNKNOWN_TOKEN).astype(np.float32)
    users["metadata_density"] = (
        users[["has_chain", "has_type", "has_root_cause", "has_source", "has_domain"]].mean(axis=1).astype(np.float32)
    )
    users["sequence_available"] = ((users["metadata_density"] > 0.0) | (users["loss_usd"] > 0.0)).astype(np.int64)

    users["chain_eth"] = (users["chain_clean"] == "eth").astype(np.float32)
    users["chain_bsc"] = (users["chain_clean"] == "bsc").astype(np.float32)
    users["chain_other"] = (
        (users["chain_clean"] != UNKNOWN_TOKEN)
        & (users["chain_clean"] != "eth")
        & (users["chain_clean"] != "bsc")
    ).astype(np.float32)

    _add_frequency_features(users, "type_clean", "type")
    _add_frequency_features(users, "root_cause_clean", "root_cause")
    _add_frequency_features(users, "source_clean", "source")
    _add_frequency_features(users, "domain_clean", "domain")

    users["semantic_profile_strength"] = (
        0.45 * users["type_frequency"]
        + 0.35 * users["root_cause_frequency"]
        + 0.20 * users["metadata_density"]
    ).astype(np.float32)
    users["provenance_strength"] = (
        0.45 * users["source_frequency"]
        + 0.35 * users["domain_frequency"]
        + 0.20 * users["metadata_density"]
    ).astype(np.float32)
    users["incident_context_score"] = (
        users["loss_rank"] + 0.75 * users["semantic_profile_strength"] + 0.50 * users["provenance_strength"]
    ).astype(np.float32)

    users["type_transaction_limitation"] = (users["type_clean"] == "transaction_limitation").astype(np.float32)
    users["type_fee"] = (users["type_clean"] == "fee").astype(np.float32)
    users["type_token_generation"] = (users["type_clean"] == "token_generation").astype(np.float32)
    users["type_combination"] = (users["type_clean"] == "combination").astype(np.float32)
    users["type_lp_drain"] = (users["type_clean"] == "lp_drain").astype(np.float32)

    users["root_sale_restrict"] = (users["root_cause_clean"] == "sale_restrict").astype(np.float32)
    users["root_fee_modification"] = (users["root_cause_clean"] == "fee_modification").astype(np.float32)
    users["root_freeze_account"] = (users["root_cause_clean"] == "freezeaccount").astype(np.float32)
    users["root_mint"] = (users["root_cause_clean"] == "mint").astype(np.float32)
    users["root_transfer_block"] = (users["root_cause_clean"] == "transfer_block").astype(np.float32)
    users["root_combination"] = (users["root_cause_clean"] == "combination").astype(np.float32)

    users["source_uniswap_study"] = (
        users["source_clean"].str.contains("programming_bugs_to_multimillion_dollar_scams", regex=False)
    ).astype(np.float32)
    users["source_honeypot_study"] = (
        users["source_clean"].str.contains("art_of_the_scam", regex=False)
    ).astype(np.float32)
    users["source_pied_piper"] = (users["source_clean"] == "pied_piper").astype(np.float32)
    users["source_security_vendor"] = users["source_clean"].isin(
        {
            "certik",
            "aegisweb3",
            "beosin",
            "metatrust",
            "de_fi",
            "peckshield",
            "web3_security_expert",
            "slowmist",
        }
    ).astype(np.float32)

    users["domain_github"] = (users["domain_clean"] == "github_com").astype(np.float32)
    users["domain_twitter"] = (users["domain_clean"] == "twitter_com").astype(np.float32)
    users["domain_x"] = (users["domain_clean"] == "x_com").astype(np.float32)
    users["domain_security_site"] = users["domain_clean"].isin(
        {
            "certik_com",
            "research_checkpoint_com",
            "de_fi",
            "numencyber_com",
            "slowmist_medium_com",
        }
    ).astype(np.float32)
    users["url_present"] = (users["url"].astype(str).str.strip() != "").astype(np.float32)

    selected_top_flags = {
        "type_flags": _add_top_category_flags(users, "type_clean", "type", top_k=8),
        "root_cause_flags": _add_top_category_flags(users, "root_cause_clean", "root_cause", top_k=10),
        "source_flags": _add_top_category_flags(users, "source_clean", "source", top_k=6),
        "domain_flags": _add_top_category_flags(users, "domain_clean", "domain", top_k=6),
    }
    return users, selected_top_flags


def _build_feature_matrix(users: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_columns: list[str] = []
    for column in users.columns:
        if column in FEATURE_EXCLUDE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(users[column]):
            feature_columns.append(column)
    if not feature_columns:
        raise ValueError("DeFi Rug Pull combined table did not yield any numeric feature columns.")
    frame = users[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    values = frame.to_numpy(dtype=np.float32)
    values = np.sign(values) * np.log1p(np.abs(values))
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    stds[stds < 1e-6] = 1.0
    return ((values - means) / stds).astype(np.float32), feature_columns


def _numeric_column(users: pd.DataFrame, column: str, *, default: float = 0.0) -> np.ndarray:
    if column not in users.columns:
        return np.full(len(users), float(default), dtype=np.float32)
    return pd.to_numeric(users[column], errors="coerce").fillna(default).to_numpy(dtype=np.float32)


def _window_edges_from_signal(signal: np.ndarray, neighbors: int = 4) -> tuple[np.ndarray, np.ndarray]:
    if len(signal) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    safe_signal = np.nan_to_num(np.asarray(signal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(safe_signal, kind="mergesort")
    src, dst = _window_edge_pairs(order.astype(np.int64), neighbors=neighbors)
    if len(src) == 0:
        singleton = np.arange(len(safe_signal), dtype=np.int64)
        return singleton.copy(), singleton.copy()
    return src, dst


def _group_window_edges(group_values: pd.Series, order_signal: np.ndarray, neighbors: int = 4) -> tuple[np.ndarray, np.ndarray]:
    src_all: list[np.ndarray] = []
    dst_all: list[np.ndarray] = []
    normalized_values = group_values.fillna(UNKNOWN_TOKEN).astype(str).map(_normalize_category)
    group_array = normalized_values.to_numpy()
    for value in sorted(normalized_values.unique()):
        if not value or value == UNKNOWN_TOKEN:
            continue
        group_index = np.flatnonzero(group_array == value)
        if len(group_index) < 2:
            continue
        group_order = group_index[np.argsort(order_signal[group_index], kind="mergesort")]
        src, dst = _window_edge_pairs(group_order.astype(np.int64), neighbors=neighbors)
        if len(src) > 0:
            src_all.append(src)
            dst_all.append(dst)
    if not src_all:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(src_all), np.concatenate(dst_all)


def _append_relation_edges(
    edge_dict: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]],
    relation_edge_counts: dict[str, int],
    relation_name: str,
    src: np.ndarray,
    dst: np.ndarray,
) -> None:
    if len(src) == 0:
        return
    edge_dict[(NODE_TYPE, relation_name, NODE_TYPE)] = (
        torch.from_numpy(src.astype(np.int64)),
        torch.from_numpy(dst.astype(np.int64)),
    )
    relation_edge_counts[relation_name] = int(len(src))


def _build_synthetic_edges(users: pd.DataFrame) -> tuple[dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]], dict[str, int]]:
    severity_signal = (
        _numeric_column(users, "loss_rank")
        + 0.75 * _numeric_column(users, "semantic_profile_strength")
        + 0.50 * _numeric_column(users, "provenance_strength")
        + 0.25 * _numeric_column(users, "metadata_density")
    )
    src, dst = _window_edges_from_signal(severity_signal, neighbors=4)
    edge_dict = {
        (NODE_TYPE, "homo", NODE_TYPE): (torch.from_numpy(src), torch.from_numpy(dst)),
        (NODE_TYPE, "severity_peer", NODE_TYPE): (torch.from_numpy(src.copy()), torch.from_numpy(dst.copy())),
    }
    relation_edge_counts = {"homo": int(len(src)), "severity_peer": int(len(src))}

    chain_signal = _numeric_column(users, "loss_log_usd") + 0.5 * _numeric_column(users, "semantic_profile_strength")
    chain_src, chain_dst = _group_window_edges(users["chain_clean"], chain_signal, neighbors=5)
    _append_relation_edges(edge_dict, relation_edge_counts, "chain_peer", chain_src, chain_dst)

    type_signal = _numeric_column(users, "loss_log_usd") + 2.0 * _numeric_column(users, "type_frequency")
    type_src, type_dst = _group_window_edges(users["type_clean"], type_signal, neighbors=4)
    _append_relation_edges(edge_dict, relation_edge_counts, "type_peer", type_src, type_dst)

    root_signal = _numeric_column(users, "loss_log_usd") + 2.0 * _numeric_column(users, "root_cause_frequency")
    root_src, root_dst = _group_window_edges(users["root_cause_clean"], root_signal, neighbors=4)
    _append_relation_edges(edge_dict, relation_edge_counts, "root_cause_peer", root_src, root_dst)

    source_signal = _numeric_column(users, "provenance_strength") + 0.5 * _numeric_column(users, "loss_rank")
    source_src, source_dst = _group_window_edges(users["source_clean"], source_signal, neighbors=4)
    _append_relation_edges(edge_dict, relation_edge_counts, "source_peer", source_src, source_dst)

    domain_signal = _numeric_column(users, "provenance_strength") + 0.5 * _numeric_column(users, "loss_rank")
    domain_src, domain_dst = _group_window_edges(users["domain_clean"], domain_signal, neighbors=4)
    _append_relation_edges(edge_dict, relation_edge_counts, "domain_peer", domain_src, domain_dst)

    provenance_signal = _numeric_column(users, "provenance_strength") + 0.4 * _numeric_column(users, "domain_frequency")
    provenance_src, provenance_dst = _window_edges_from_signal(provenance_signal, neighbors=4)
    _append_relation_edges(edge_dict, relation_edge_counts, "provenance_peer", provenance_src, provenance_dst)
    return edge_dict, relation_edge_counts


def _build_incident_event_tensors(
    users: pd.DataFrame,
    history_len: int = RUG_PULL_EVENT_SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = int(len(users))
    sequence_length = max(int(history_len), 1)
    event_dim = 8
    event_sequence = np.zeros((num_nodes, sequence_length, event_dim), dtype=np.float32)
    event_mask = np.zeros((num_nodes, sequence_length), dtype=bool)
    event_time_deltas = np.zeros((num_nodes, sequence_length), dtype=np.float32)
    event_token_weights = np.zeros((num_nodes, sequence_length), dtype=np.float32)
    event_token_types = np.zeros((num_nodes, sequence_length), dtype=np.int64)
    if num_nodes == 0:
        return (
            torch.from_numpy(event_sequence),
            torch.from_numpy(event_mask),
            torch.from_numpy(event_time_deltas),
            torch.from_numpy(event_token_weights),
            torch.from_numpy(event_token_types),
        )

    chain_eth = _numeric_column(users, "chain_eth")
    chain_bsc = _numeric_column(users, "chain_bsc")
    chain_other = _numeric_column(users, "chain_other")
    loss_log = _numeric_column(users, "loss_log_usd")
    loss_rank = _numeric_column(users, "loss_rank")
    severity_bucket = _numeric_column(users, "loss_severity_bucket") / 3.0
    metadata_density = _numeric_column(users, "metadata_density")
    sequence_available = np.clip(_numeric_column(users, "sequence_available"), 0.0, 1.0)

    type_count = np.log1p(_numeric_column(users, "type_count"))
    type_frequency = _numeric_column(users, "type_frequency")
    type_transaction_limitation = _numeric_column(users, "type_transaction_limitation")
    type_fee = _numeric_column(users, "type_fee")
    type_token_generation = _numeric_column(users, "type_token_generation")
    type_combination = _numeric_column(users, "type_combination")
    type_lp_drain = _numeric_column(users, "type_lp_drain")

    root_count = np.log1p(_numeric_column(users, "root_cause_count"))
    root_frequency = _numeric_column(users, "root_cause_frequency")
    root_sale_restrict = _numeric_column(users, "root_sale_restrict")
    root_fee_modification = _numeric_column(users, "root_fee_modification")
    root_freeze_account = _numeric_column(users, "root_freeze_account")
    root_mint = _numeric_column(users, "root_mint")
    root_transfer_block = _numeric_column(users, "root_transfer_block")
    root_combination = _numeric_column(users, "root_combination")

    source_count = np.log1p(_numeric_column(users, "source_count"))
    source_frequency = _numeric_column(users, "source_frequency")
    source_uniswap_study = _numeric_column(users, "source_uniswap_study")
    source_honeypot_study = _numeric_column(users, "source_honeypot_study")
    source_pied_piper = _numeric_column(users, "source_pied_piper")
    source_security_vendor = _numeric_column(users, "source_security_vendor")

    domain_count = np.log1p(_numeric_column(users, "domain_count"))
    domain_frequency = _numeric_column(users, "domain_frequency")
    domain_github = _numeric_column(users, "domain_github")
    domain_twitter = _numeric_column(users, "domain_twitter")
    domain_x = _numeric_column(users, "domain_x")
    domain_security_site = _numeric_column(users, "domain_security_site")
    url_present = _numeric_column(users, "url_present")

    has_chain = _numeric_column(users, "has_chain")
    has_type = _numeric_column(users, "has_type")
    has_root_cause = _numeric_column(users, "has_root_cause")
    has_source = _numeric_column(users, "has_source")
    has_domain = _numeric_column(users, "has_domain")
    labels = _numeric_column(users, "label")
    is_anchor = np.clip(_numeric_column(users, "is_anchor", default=1.0), 0.0, 1.0)

    token_vectors = []
    token_masks = []
    token_types = [0, 1, 1, 2, 3, 4]

    token_0 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_0[:, 0] = chain_eth
    token_0[:, 1] = chain_bsc
    token_0[:, 2] = chain_other
    token_0[:, 3] = loss_log
    token_0[:, 4] = loss_rank
    token_0[:, 5] = severity_bucket
    token_0[:, 6] = metadata_density
    token_0[:, 7] = sequence_available
    token_vectors.append(token_0)
    token_masks.append((has_chain > 0.0) | (loss_log > 0.0))

    token_1 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_1[:, 0] = type_count
    token_1[:, 1] = type_frequency
    token_1[:, 2] = type_transaction_limitation
    token_1[:, 3] = type_fee
    token_1[:, 4] = type_token_generation
    token_1[:, 5] = type_combination
    token_1[:, 6] = type_lp_drain
    token_1[:, 7] = metadata_density
    token_vectors.append(token_1)
    token_masks.append(has_type > 0.0)

    token_2 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_2[:, 0] = root_count
    token_2[:, 1] = root_frequency
    token_2[:, 2] = root_sale_restrict
    token_2[:, 3] = root_fee_modification
    token_2[:, 4] = root_freeze_account
    token_2[:, 5] = root_mint
    token_2[:, 6] = root_transfer_block
    token_2[:, 7] = root_combination
    token_vectors.append(token_2)
    token_masks.append(has_root_cause > 0.0)

    token_3 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_3[:, 0] = source_count
    token_3[:, 1] = source_frequency
    token_3[:, 2] = source_uniswap_study
    token_3[:, 3] = source_honeypot_study
    token_3[:, 4] = source_pied_piper
    token_3[:, 5] = source_security_vendor
    token_3[:, 6] = metadata_density
    token_3[:, 7] = sequence_available
    token_vectors.append(token_3)
    token_masks.append(has_source > 0.0)

    token_4 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_4[:, 0] = domain_count
    token_4[:, 1] = domain_frequency
    token_4[:, 2] = domain_github
    token_4[:, 3] = domain_twitter
    token_4[:, 4] = domain_x
    token_4[:, 5] = domain_security_site
    token_4[:, 6] = url_present
    token_4[:, 7] = metadata_density
    token_vectors.append(token_4)
    token_masks.append((has_domain > 0.0) | (url_present > 0.0))

    token_5 = np.zeros((num_nodes, event_dim), dtype=np.float32)
    token_5[:, 0] = has_chain
    token_5[:, 1] = has_type
    token_5[:, 2] = has_root_cause
    token_5[:, 3] = has_source
    token_5[:, 4] = has_domain
    token_5[:, 5] = metadata_density
    token_5[:, 6] = labels
    token_5[:, 7] = is_anchor
    token_vectors.append(token_5)
    token_masks.append(np.ones(num_nodes, dtype=bool))

    max_tokens = min(sequence_length, len(token_vectors))
    for token_index in range(max_tokens):
        event_sequence[:, token_index, :] = token_vectors[token_index]
        event_mask[:, token_index] = token_masks[token_index]
        event_token_types[:, token_index] = int(token_types[token_index])
        event_time_deltas[:, token_index] = float(max_tokens - 1 - token_index)

    token_strength = np.log1p(np.abs(event_sequence).sum(axis=-1))
    event_token_weights = np.where(event_mask, 1.0 + np.clip(token_strength, 0.0, 3.0), 0.0).astype(np.float32)
    return (
        torch.from_numpy(event_sequence),
        torch.from_numpy(event_mask),
        torch.from_numpy(event_time_deltas),
        torch.from_numpy(event_token_weights),
        torch.from_numpy(event_token_types),
    )


def load_defi_rug_pull_dataset(
    *,
    data_root: str | Path = DEFI_RUG_PULL_DEFAULT_ROOT,
    dataset_name: str = "defi_rug_pull",
    num_clients: int = 3,
    seed: int = 42,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    active_learning_feedback_path: str = "",
    negative_users_path: str | Path | None = None,
    force_preview: bool = False,
) -> DatasetBundle:
    requested_root = Path(data_root).expanduser().resolve()
    resolved_root = _resolve_data_root(requested_root)
    positive_users, incidents_source = _read_incident_table(resolved_root, force_preview=force_preview)
    positive_users = _rename_known_columns(positive_users.copy())
    if "address" not in positive_users.columns:
        raise ValueError("DeFi Rug Pull incident table must contain an 'address' column.")
    positive_users["label"] = 1
    positive_users["label_name"] = "rug_pull"
    positive_users["is_anchor"] = 1

    if negative_users_path is None:
        raise ValueError(
            "DeFi Rug Pull loader is draft-only right now: the local public dataset is positive-only. "
            "Provide `negative_users_path=...` with an external non-rug-pull address/contract set to build a binary benchmark."
        )

    negative_users, negative_source = _load_negative_users(negative_users_path)
    raw_users = pd.concat([positive_users, negative_users], ignore_index=True, sort=False)
    raw_users = raw_users.drop_duplicates(subset=["address"], keep="first").reset_index(drop=True)
    users, selected_top_flags = _engineer_incident_table(raw_users)

    if users["label"].nunique() < 2:
        raise ValueError(
            "DeFi Rug Pull loader still has only one class after merging negatives. "
            "Please provide a valid negative set with label 0."
        )

    numeric_columns = [column for column in users.columns if pd.api.types.is_numeric_dtype(users[column])]
    users[numeric_columns] = users[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_matrix, feature_columns = _build_feature_matrix(users)

    train_mask, valid_mask, test_mask = _stratified_split_masks(users["label"].to_numpy(dtype=np.int64), seed=seed)
    edge_dict, relation_edge_counts = _build_synthetic_edges(users)
    graph = dgl.heterograph(edge_dict, num_nodes_dict={NODE_TYPE: len(users)})
    graph.nodes[NODE_TYPE].data["feature"] = torch.from_numpy(feature_matrix)
    graph.nodes[NODE_TYPE].data["label"] = torch.from_numpy(users["label"].to_numpy(dtype=np.int64, copy=True))
    graph.nodes[NODE_TYPE].data["train_mask"] = train_mask.bool()
    graph.nodes[NODE_TYPE].data["valid_mask"] = valid_mask.bool()
    graph.nodes[NODE_TYPE].data["test_mask"] = test_mask.bool()
    graph.nodes[NODE_TYPE].data["label_confidence_target"] = torch.ones(graph.num_nodes(NODE_TYPE), dtype=torch.float32)
    _attach_dataset_context_defaults(graph, dataset_name=dataset_name)

    if float(label_fraction) < 0.999:
        _apply_label_scarcity(graph, label_fraction=float(label_fraction), seed=seed)
    else:
        graph.nodes[NODE_TYPE].data["train_supervised_mask"] = train_mask.clone()
        graph.nodes[NODE_TYPE].data["train_unlabeled_mask"] = torch.zeros_like(train_mask)
        graph.nodes[NODE_TYPE].data["label_scarcity_ratio"] = torch.full(
            (graph.num_nodes(NODE_TYPE),),
            1.0,
            dtype=torch.float32,
        )
    if active_learning_feedback_path:
        _apply_active_learning_feedback(graph, active_learning_feedback_path, dataset_name=dataset_name)

    homo_src, homo_dst = graph.edges(etype="homo")
    labels = graph.nodes[NODE_TYPE].data["label"].long()
    supervised_mask = graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool()
    graph.edges["homo"].data["label"] = torch.where(
        labels[homo_src] == labels[homo_dst],
        torch.ones_like(homo_src, dtype=torch.float32),
        -torch.ones_like(homo_src, dtype=torch.float32),
    )
    graph.edges["homo"].data["train_mask"] = supervised_mask[homo_src] & supervised_mask[homo_dst]

    event_sequence, event_mask, event_time_deltas, event_token_weights, event_token_types = _build_incident_event_tensors(
        users=users,
        history_len=RUG_PULL_EVENT_SEQUENCE_LENGTH,
    )
    graph.nodes[NODE_TYPE].data["event_sequence"] = event_sequence.float()
    graph.nodes[NODE_TYPE].data["event_mask"] = event_mask.bool()
    graph.nodes[NODE_TYPE].data["event_time_deltas"] = event_time_deltas.float()
    graph.nodes[NODE_TYPE].data["event_token_weights"] = event_token_weights.float()
    graph.nodes[NODE_TYPE].data["event_token_types"] = event_token_types.long()
    relation_order = _attach_relation_sequence(graph, dataset_name="defi_rug_pull")

    train_supervised_mask = (
        graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool() & graph.nodes[NODE_TYPE].data["train_mask"].bool()
    )
    train_unlabeled_mask = (
        graph.nodes[NODE_TYPE].data["train_unlabeled_mask"].bool() & graph.nodes[NODE_TYPE].data["train_mask"].bool()
    )
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
        clients.append(
            ClientShard(
                client_id=client_id,
                owned_global_nodes=owned_nodes,
                subgraph=subgraph,
                train_nodes=int(subgraph.nodes[NODE_TYPE].data["train_mask"].sum().item()),
            )
        )

    class_labels = graph.nodes[NODE_TYPE].data["label"][graph.nodes[NODE_TYPE].data["train_supervised_mask"].bool()]
    bundle = DatasetBundle(
        name=dataset_name,
        graph=graph,
        node_type=NODE_TYPE,
        relation_order=relation_order,
        class_weights=_compute_class_weights(class_labels) if class_labels.numel() > 0 else torch.ones(2, dtype=torch.float32),
        class_counts=_compute_class_counts(class_labels) if class_labels.numel() > 0 else torch.ones(2, dtype=torch.float32),
        clients=clients,
        base_lr=1e-3,
    )
    bundle.addresses = users["address"].tolist()
    bundle.data_summary = {
        "data_root": str(resolved_root),
        "requested_data_root": str(requested_root),
        "incidents_source": incidents_source,
        "negative_users_path": str(Path(negative_users_path).expanduser().resolve()),
        "negative_users_source": negative_source,
        "feature_columns": feature_columns,
        "feature_dim": int(feature_matrix.shape[1]),
        "num_nodes": int(graph.num_nodes(NODE_TYPE)),
        "num_clients": int(len(clients)),
        "relation_edge_counts": relation_edge_counts,
        "graph_source": "synthetic_incident_semantic_graph",
        "train_nodes": int(train_mask.sum().item()),
        "valid_nodes": int(valid_mask.sum().item()),
        "test_nodes": int(test_mask.sum().item()),
        "positive_labels": int(users["label"].sum()),
        "positive_ratio": float(users["label"].mean()),
        "metadata_ready_ratio": float(users["sequence_available"].mean()),
        "event_sequence_length": int(event_sequence.shape[1]),
        "event_sequence_strategy": "incident_semantic_capsules",
        "selected_top_flags": selected_top_flags,
        "draft_loader": True,
        "force_preview": bool(force_preview),
    }
    return bundle
