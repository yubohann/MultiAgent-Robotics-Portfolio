from __future__ import annotations

"""SplitGNN dataset loading, client partitioning, and label-scarcity helpers."""

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

try:
    import dgl
except Exception as error:  # pragma: no cover - runtime env dependent
    raise RuntimeError(
        "Missing dgl dependency. Install the pinned graph-learning profile and retry:\n"
        "python -m pip install -r requirements/requirements-cpu.txt"
    ) from error
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F

from .paths import REPO_ROOT
from .vendor.splitgnn.utils import normalize, resolve_graph_path

PROJECT_ROOT = REPO_ROOT
_ACTIVE_MEMORY_LOG_STACK: list[dict[str, object]] = []
SEQUENCE_BUILDER_VERSION = "relation_capsule_v7_ieee_runtime_topk"
IEEE_FULL_SEQUENCE_NODE_THRESHOLD = 200_000
IEEE_FULL_SEQUENCE_COMPACT_DIM = 64
SEQUENCE_DATASET_PROFILES = {
    "default": {
        "name": "relation_capsule_v4_default_semantic",
        "max_relations": None,
        "token_order": ("local", "motif", "reliability"),
        "self_blend": 0.25,
        "relation_gain": 0.35,
        "local_delta_gain": 0.20,
        "motif_gain": 0.20,
        "reliability_gain": 0.15,
        "min_relation_score": 0.08,
        "min_relation_coverage": 0.01,
    },
    "amazon": {
        "name": "amazon_relation_context_v2_semantic_shortseq",
        "max_relations": 3,
        "token_order": ("local", "reliability"),
        "self_blend": 0.18,
        "relation_gain": 0.24,
        "local_delta_gain": 0.12,
        "motif_gain": 0.08,
        "reliability_gain": 0.22,
        "min_relation_score": 0.12,
        "min_relation_coverage": 0.02,
    },
    "yelp": {
        "name": "yelp_relation_context_v2_semantic_shortseq",
        "max_relations": 3,
        "token_order": ("local", "reliability"),
        "self_blend": 0.20,
        "relation_gain": 0.26,
        "local_delta_gain": 0.15,
        "motif_gain": 0.10,
        "reliability_gain": 0.24,
        "min_relation_score": 0.12,
        "min_relation_coverage": 0.02,
    },
    "comp": {
        "name": "comp_hetero_role_v2_semantic_shortseq",
        "max_relations": 4,
        "token_order": ("local", "motif", "reliability"),
        "self_blend": 0.20,
        "relation_gain": 0.30,
        "local_delta_gain": 0.24,
        "motif_gain": 0.35,
        "reliability_gain": 0.20,
        "min_relation_score": 0.10,
        "min_relation_coverage": 0.015,
    },
    "ieee": {
        "name": "ieee_relation_context_v1_semantic_shortseq",
        "max_relations": 6,
        "token_order": ("local", "motif", "reliability"),
        "self_blend": 0.18,
        "relation_gain": 0.22,
        "local_delta_gain": 0.10,
        "motif_gain": 0.08,
        "reliability_gain": 0.12,
        "min_relation_score": 0.08,
        "min_relation_coverage": 0.01,
    },
    "elliptic": {
        "name": "elliptic_causal_relation_v1",
        "max_relations": 3,
        "token_order": ("local", "motif", "reliability"),
        "self_blend": 0.18,
        "relation_gain": 0.28,
        "local_delta_gain": 0.16,
        "motif_gain": 0.20,
        "reliability_gain": 0.18,
        "min_relation_score": 0.0,
        "min_relation_coverage": 0.0,
        "excluded_relations": ("homo",),
        "priority_relations": ("causal_forward", "causal_reverse", "self_loop"),
        "lazy_materialize": False,
        "runtime_dynamic": False,
    },
    "amlsim": {
        "name": "amlsim_account_relation_v1",
        "max_relations": 6,
        "token_order": ("local", "motif", "reliability"),
        "self_blend": 0.18,
        "relation_gain": 0.26,
        "local_delta_gain": 0.14,
        "motif_gain": 0.16,
        "reliability_gain": 0.16,
        "min_relation_score": 0.0,
        "min_relation_coverage": 0.0,
        "priority_relations": ("transfer", "transfer_reverse", "temporal_near", "same_bank"),
        "excluded_relations": (),
        "lazy_materialize": False,
        "runtime_dynamic": False,
    },
}
DATASET_CONTEXT_IDS = {
    "yelp": 0,
    "amazon": 1,
    "comp": 2,
    "ieee": 3,
    "archive": 4,
    "ethereum_phishing": 5,
    "ethereum_ponzi": 6,
    "defi_rug_pull": 7,
    "ccfd": 8,
    "elliptic": 9,
    "amlsim": 10,
}
SEQUENCE_TOKEN_TYPE_SELF = 0
SEQUENCE_TOKEN_TYPE_LOCAL = 1
SEQUENCE_TOKEN_TYPE_MOTIF = 2
SEQUENCE_TOKEN_TYPE_RELIABILITY = 3
SEQUENCE_TOKEN_TYPE_GLOBAL = 4


def _memory_log_enabled(dataset_name: str = "") -> bool:
    raw = str(os.environ.get("SPLITGNN_MEMORY_LOG", "auto")).strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "on", "yes"}:
        return True
    return str(dataset_name).strip().lower() == "ieee"


def _process_rss_bytes() -> int | None:
    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        except Exception:
            pass
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _format_rss_gib(rss_bytes: int | None) -> str:
    if rss_bytes is None:
        return "n/a"
    return f"{float(rss_bytes) / float(1024 ** 3):.2f}GiB"


def _start_memory_log_session(label: str, *, dataset_name: str = "") -> dict[str, object] | None:
    if not _memory_log_enabled(dataset_name):
        return None
    session = {
        "label": str(label),
        "dataset_name": str(dataset_name),
        "start_time": float(time.perf_counter()),
        "peak_rss_bytes": int(_process_rss_bytes() or 0),
    }
    _ACTIVE_MEMORY_LOG_STACK.append(session)
    _memory_log("session_start", session=session)
    return session


def _stop_memory_log_session(session: dict[str, object] | None, stage_name: str = "session_end") -> None:
    if session is None:
        return
    _memory_log(stage_name, session=session)
    if _ACTIVE_MEMORY_LOG_STACK and _ACTIVE_MEMORY_LOG_STACK[-1] is session:
        _ACTIVE_MEMORY_LOG_STACK.pop()
        return
    if session in _ACTIVE_MEMORY_LOG_STACK:
        _ACTIVE_MEMORY_LOG_STACK.remove(session)


def _memory_log(stage_name: str, *, session: dict[str, object] | None = None) -> None:
    active_session = session if session is not None else (_ACTIVE_MEMORY_LOG_STACK[-1] if _ACTIVE_MEMORY_LOG_STACK else None)
    if active_session is None:
        return
    rss_bytes = _process_rss_bytes()
    current_peak = int(active_session.get("peak_rss_bytes", 0) or 0)
    if rss_bytes is not None:
        current_peak = max(current_peak, int(rss_bytes))
        active_session["peak_rss_bytes"] = current_peak
    elapsed_seconds = float(time.perf_counter()) - float(active_session.get("start_time", 0.0) or 0.0)
    label = str(active_session.get("label", "memory"))
    print(
        f"[memory][{label}] elapsed={elapsed_seconds:.1f}s rss={_format_rss_gib(rss_bytes)} "
        f"peak={_format_rss_gib(current_peak if current_peak > 0 else rss_bytes)} stage={stage_name}",
        flush=True,
    )


@dataclass
class ClientShard:
    """Data partition owned by one client in the training protocol."""

    client_id: int
    owned_global_nodes: torch.Tensor
    subgraph: dgl.DGLHeteroGraph
    train_nodes: int


@dataclass
class DatasetBundle:
    """Complete dataset bundle consumed by the training protocol."""

    name: str
    graph: dgl.DGLHeteroGraph
    node_type: str
    relation_order: List[str]
    class_weights: torch.Tensor
    class_counts: torch.Tensor
    clients: List[ClientShard]
    base_lr: float = 1e-3
    data_summary: dict[str, Any] | None = None
    data_profile: str = ""
    loader_view: str = ""
    feature_profile: str = ""
    relation_profile: str = ""
    history_len: int = 0


def _supervised_training_mask(graph: dgl.DGLHeteroGraph) -> torch.Tensor:
    node_type = graph.ntypes[0]
    if "train_supervised_mask" in graph.nodes[node_type].data:
        return graph.nodes[node_type].data["train_supervised_mask"].bool()
    return graph.nodes[node_type].data["train_mask"].bool()


def _unlabeled_training_mask(graph: dgl.DGLHeteroGraph) -> torch.Tensor:
    node_type = graph.ntypes[0]
    if "train_unlabeled_mask" in graph.nodes[node_type].data:
        return graph.nodes[node_type].data["train_unlabeled_mask"].bool()
    return torch.zeros_like(graph.nodes[node_type].data["train_mask"].bool())


def _refresh_homo_edge_train_mask(graph: dgl.DGLHeteroGraph) -> None:
    node_type = graph.ntypes[0]
    if "homo" not in graph.etypes:
        return
    supervised_mask = _supervised_training_mask(graph).bool()
    src_nodes, dst_nodes = graph.edges(etype="homo")
    graph.edges["homo"].data["train_mask"] = (supervised_mask[src_nodes] & supervised_mask[dst_nodes]).bool()


def _apply_label_scarcity(graph: dgl.DGLHeteroGraph, label_fraction: float, seed: int) -> None:
    """Create a low-label training view while preserving an unlabeled pool."""

    node_type = graph.ntypes[0]
    train_mask = graph.nodes[node_type].data["train_mask"].bool()
    label_fraction = float(max(min(label_fraction, 1.0), 0.0))

    if label_fraction >= 0.999 or not train_mask.any():
        graph.nodes[node_type].data["train_supervised_mask"] = train_mask.clone()
        graph.nodes[node_type].data["train_unlabeled_mask"] = torch.zeros_like(train_mask)
        graph.nodes[node_type].data["label_scarcity_ratio"] = torch.full(
            (graph.num_nodes(node_type),),
            1.0,
            dtype=torch.float32,
        )
        return

    if label_fraction <= 0.0:
        graph.nodes[node_type].data["train_supervised_mask"] = torch.zeros_like(train_mask)
        graph.nodes[node_type].data["train_unlabeled_mask"] = train_mask.clone()
        graph.nodes[node_type].data["label_scarcity_ratio"] = torch.zeros(
            graph.num_nodes(node_type),
            dtype=torch.float32,
        )
        return

    rng = np.random.default_rng(seed)
    train_node_ids = train_mask.nonzero(as_tuple=False).flatten().cpu().numpy()
    train_labels = graph.nodes[node_type].data["label"][train_mask].cpu().numpy()
    selected_global_nodes: list[int] = []

    for label in np.unique(train_labels):
        label_positions = np.flatnonzero(train_labels == label)
        if label_positions.size == 0:
            continue
        target_count = max(1, int(round(label_positions.size * label_fraction)))
        target_count = min(target_count, label_positions.size)
        chosen_positions = rng.choice(label_positions, size=target_count, replace=False)
        selected_global_nodes.extend(train_node_ids[chosen_positions].tolist())

    supervised_mask = torch.zeros_like(train_mask)
    if selected_global_nodes:
        supervised_mask[torch.tensor(sorted(set(selected_global_nodes)), dtype=torch.long)] = True
    supervised_mask &= train_mask
    unlabeled_mask = train_mask & ~supervised_mask

    graph.nodes[node_type].data["train_supervised_mask"] = supervised_mask
    graph.nodes[node_type].data["train_unlabeled_mask"] = unlabeled_mask
    scarcity_ratio = float(supervised_mask.sum().item()) / max(float(train_mask.sum().item()), 1.0)
    graph.nodes[node_type].data["label_scarcity_ratio"] = torch.full(
        (graph.num_nodes(node_type),),
        scarcity_ratio,
        dtype=torch.float32,
    )
    _refresh_homo_edge_train_mask(graph)


def _resolve_feedback_path(feedback_path: str) -> Path | None:
    raw = str(feedback_path).strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    search_paths = [candidate] if candidate.is_absolute() else [candidate, PROJECT_ROOT / candidate]
    for path in search_paths:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        f"Active learning feedback file not found: {raw}. "
        "Pass an existing path or leave --active_learning_feedback_path empty."
    )


def _load_active_learning_feedback(feedback_path: str, expected_dataset: str = "") -> list[tuple[int, int]]:
    path = _resolve_feedback_path(feedback_path)
    if path is None:
        return []

    deduped_records: dict[int, int] = {}
    ordered_node_ids: list[int] = []

    def _normalize_mapping(item: dict) -> dict:
        return {str(key).lstrip("\ufeff").strip(): value for key, value in item.items()}

    def _parse_feedback_record(node_value, label_value) -> tuple[int, int] | None:
        if node_value is None or label_value is None or str(label_value).strip() == "":
            return None
        node_id = int(node_value)
        label = int(label_value)
        if label not in (0, 1):
            raise ValueError(
                f"Active learning feedback labels must be binary 0/1, but got label={label} for node_id={node_id}."
            )
        return node_id, label

    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8-sig") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            normalized_payload = _normalize_mapping(payload)
            payload_dataset = str(normalized_payload.get("dataset", "")).strip().lower()
            payload_split = str(normalized_payload.get("split", "")).strip().lower()
            normalized_expected_dataset = str(expected_dataset).strip().lower()
            if normalized_expected_dataset and payload_dataset and payload_dataset != normalized_expected_dataset:
                raise ValueError(
                    f"Active learning feedback dataset mismatch: expected '{normalized_expected_dataset}', "
                    f"but feedback file declares '{payload_dataset}'."
                )
            if payload_split and payload_split != "train":
                raise ValueError(
                    f"Active learning feedback split must be 'train' for retraining state, but got '{payload_split}'."
                )
            if "records" in normalized_payload:
                payload = normalized_payload["records"]
            elif "items" in normalized_payload:
                payload = normalized_payload["items"]
            else:
                payload = [normalized_payload]
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError(f"Unsupported active learning feedback format: {path}")
        for item in payload:
            if not isinstance(item, dict):
                continue
            normalized_item = _normalize_mapping(item)
            node_id = normalized_item.get("node_id", normalized_item.get("id"))
            label = normalized_item.get("label", normalized_item.get("review_label"))
            parsed = _parse_feedback_record(node_id, label)
            if parsed is None:
                continue
            node_id, label = parsed
            previous_label = deduped_records.get(node_id)
            if previous_label is not None and previous_label != label:
                raise ValueError(
                    f"Active learning feedback contains conflicting labels for node_id={node_id}: "
                    f"{previous_label} vs {label}."
                )
            if previous_label is None:
                ordered_node_ids.append(node_id)
            deduped_records[node_id] = label
        return [(node_id, deduped_records[node_id]) for node_id in ordered_node_ids]

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            normalized_row = _normalize_mapping(row)
            node_id = normalized_row.get("node_id", normalized_row.get("id"))
            label = normalized_row.get("label", normalized_row.get("review_label"))
            parsed = _parse_feedback_record(node_id, label)
            if parsed is None:
                continue
            node_id, label = parsed
            previous_label = deduped_records.get(node_id)
            if previous_label is not None and previous_label != label:
                raise ValueError(
                    f"Active learning feedback contains conflicting labels for node_id={node_id}: "
                    f"{previous_label} vs {label}."
                )
            if previous_label is None:
                ordered_node_ids.append(node_id)
            deduped_records[node_id] = label
    return [(node_id, deduped_records[node_id]) for node_id in ordered_node_ids]


def _apply_active_learning_feedback(graph: dgl.DGLHeteroGraph, feedback_path: str, dataset_name: str = "") -> None:
    feedback = _load_active_learning_feedback(feedback_path, expected_dataset=dataset_name)
    if not feedback:
        return

    node_type = graph.ntypes[0]
    train_mask = graph.nodes[node_type].data["train_mask"].bool()
    if "train_supervised_mask" in graph.nodes[node_type].data:
        supervised_mask = graph.nodes[node_type].data["train_supervised_mask"].bool().clone()
    else:
        supervised_mask = train_mask.clone()
    if "train_unlabeled_mask" in graph.nodes[node_type].data:
        unlabeled_mask = graph.nodes[node_type].data["train_unlabeled_mask"].bool().clone()
    else:
        unlabeled_mask = train_mask & ~supervised_mask
    labels = graph.nodes[node_type].data["label"].clone()

    for node_id, label in feedback:
        if node_id < 0 or node_id >= graph.num_nodes(node_type):
            raise IndexError(
                f"Active learning feedback node_id={node_id} is outside the valid range [0, {graph.num_nodes(node_type) - 1}]."
            )
        if not bool(train_mask[node_id]):
            raise ValueError(
                f"Active learning feedback node_id={node_id} is not part of the training split for dataset '{dataset_name or 'unknown'}'."
            )
        if bool(supervised_mask[node_id]) and not bool(unlabeled_mask[node_id]):
            raise ValueError(
                f"Active learning feedback node_id={node_id} is already supervised; feedback files must only reveal unlabeled train nodes."
            )
        labels[node_id] = int(label)
        supervised_mask[node_id] = True
        unlabeled_mask[node_id] = False

    graph.nodes[node_type].data["label"] = labels
    graph.nodes[node_type].data["train_supervised_mask"] = supervised_mask & train_mask
    graph.nodes[node_type].data["train_unlabeled_mask"] = unlabeled_mask & train_mask
    scarcity_ratio = float(graph.nodes[node_type].data["train_supervised_mask"].sum().item()) / max(
        float(train_mask.sum().item()), 1.0
    )
    graph.nodes[node_type].data["label_scarcity_ratio"] = torch.full(
        (graph.num_nodes(node_type),),
        scarcity_ratio,
        dtype=torch.float32,
    )
    _refresh_homo_edge_train_mask(graph)


def _prepare_graph(
    dataset_name: str,
    graph: dgl.DGLHeteroGraph,
    require_labels_and_masks: bool = True,
) -> dgl.DGLHeteroGraph:
    """Normalize graph features, labels, and split masks into the package contract."""

    node_type = graph.ntypes[0]
    if "feature" not in graph.nodes[node_type].data:
        raise KeyError(f"Graph for dataset '{dataset_name}' is missing node feature data.")

    features = graph.nodes[node_type].data["feature"].cpu().numpy()
    if dataset_name == "amazon" and features.shape[1] > 19:
        # Drop the known leakage-prone 20th feature in the Amazon variant.
        features = np.delete(features, 19, axis=1)
    features = normalize(features)
    graph.nodes[node_type].data["feature"] = torch.from_numpy(features).float()

    if "label" in graph.nodes[node_type].data:
        graph.nodes[node_type].data["label"] = graph.nodes[node_type].data["label"].long()
    elif require_labels_and_masks:
        raise KeyError(f"Graph for dataset '{dataset_name}' is missing node labels.")

    for key in ["train_mask", "valid_mask", "test_mask"]:
        if key in graph.nodes[node_type].data:
            graph.nodes[node_type].data[key] = graph.nodes[node_type].data[key].bool()
        elif require_labels_and_masks:
            raise KeyError(f"Graph for dataset '{dataset_name}' is missing node mask '{key}'.")
    return graph


def _load_graph(dataset_name: str, data_dir: str) -> dgl.DGLHeteroGraph:
    graph_path = resolve_graph_path(data_dir, dataset_name)
    graph = dgl.load_graphs(graph_path)[0][0]
    return _prepare_graph(dataset_name=dataset_name, graph=graph, require_labels_and_masks=True)


def _attach_dataset_context_defaults(graph: dgl.DGLHeteroGraph, dataset_name: str) -> None:
    node_type = graph.ntypes[0]
    dataset_id = DATASET_CONTEXT_IDS.get(str(dataset_name).strip().lower(), len(DATASET_CONTEXT_IDS))
    graph.nodes[node_type].data["dataset_context_id"] = torch.full(
        (graph.num_nodes(node_type),),
        int(dataset_id),
        dtype=torch.long,
    )
    if "label_confidence_target" not in graph.nodes[node_type].data:
        graph.nodes[node_type].data["label_confidence_target"] = torch.ones(
            graph.num_nodes(node_type),
            dtype=torch.float32,
        )


def _sequence_relation_order(graph: dgl.DGLHeteroGraph) -> List[str]:
    relation_order = [relation for relation in graph.etypes if relation != "homo"]
    if "homo" in graph.etypes:
        relation_order.append("homo")
    return relation_order


def _resolve_sequence_dataset_profile(dataset_name: str) -> dict:
    normalized = str(dataset_name).strip().lower()
    profile = dict(SEQUENCE_DATASET_PROFILES["default"])
    profile.update(SEQUENCE_DATASET_PROFILES.get(normalized, {}))
    return profile


def _use_ieee_full_sequence_compact_mode(
    graph: dgl.DGLHeteroGraph,
    dataset_name: str,
) -> bool:
    return str(dataset_name).strip().lower() == "ieee" and int(graph.num_nodes(graph.ntypes[0])) >= IEEE_FULL_SEQUENCE_NODE_THRESHOLD


def _sequence_profile_for_graph(
    graph: dgl.DGLHeteroGraph,
    dataset_name: str,
    *,
    ieee_full_compact_sequences: bool | None = None,
    ieee_sequence_feature_dim: int | None = None,
) -> dict:
    profile = dict(_resolve_sequence_dataset_profile(dataset_name))
    if _use_ieee_full_sequence_compact_mode(graph, dataset_name):
        compact_enabled = True if ieee_full_compact_sequences is None else bool(ieee_full_compact_sequences)
        compact_feature_dim = (
            int(IEEE_FULL_SEQUENCE_COMPACT_DIM)
            if ieee_sequence_feature_dim is None
            else max(int(ieee_sequence_feature_dim), 1)
        )
        profile.update(
            {
                "name": "ieee_full_runtime_topk_v1",
                "max_relations": None,
                "token_order": ("local", "motif", "reliability"),
                "self_blend": 0.16,
                "relation_gain": 0.24,
                "local_delta_gain": 0.12,
                "motif_gain": 0.12,
                "reliability_gain": 0.14,
                "min_relation_score": 0.0,
                "min_relation_coverage": 0.0,
                "compact_feature_dim": compact_feature_dim if compact_enabled else None,
                "excluded_relations": ("homo",),
                "priority_relations": ("uid", "uid_addr", "uid_email", "device_browser", "temporal_past"),
                "streaming_mode": False,
                "lazy_materialize": True,
                "runtime_dynamic": True,
            }
        )
    else:
        profile.setdefault("compact_feature_dim", None)
        profile.setdefault("excluded_relations", tuple())
        profile.setdefault("priority_relations", tuple())
        profile.setdefault("streaming_mode", False)
        profile.setdefault("lazy_materialize", False)
        profile.setdefault("runtime_dynamic", False)
    return profile


def _build_sequence_base_features(
    graph: dgl.DGLHeteroGraph,
    dataset_name: str,
    profile: dict,
) -> torch.Tensor:
    node_type = graph.ntypes[0]
    features = graph.nodes[node_type].data["feature"].float()
    if bool(profile.get("runtime_dynamic", False)):
        if "sequence_base_feature" in graph.nodes[node_type].data:
            del graph.nodes[node_type].data["sequence_base_feature"]
        if "sequence_base_feature_dim" in graph.nodes[node_type].data:
            del graph.nodes[node_type].data["sequence_base_feature_dim"]
        return features
    compact_feature_dim = profile.get("compact_feature_dim")
    if compact_feature_dim is None or int(compact_feature_dim) <= 0 or int(compact_feature_dim) >= int(features.shape[1]):
        if "sequence_base_feature" in graph.nodes[node_type].data:
            del graph.nodes[node_type].data["sequence_base_feature"]
        if "sequence_base_feature_dim" in graph.nodes[node_type].data:
            del graph.nodes[node_type].data["sequence_base_feature_dim"]
        return features
    compact_dim = int(compact_feature_dim)
    compact_features = F.adaptive_avg_pool1d(features.unsqueeze(1), compact_dim).squeeze(1).contiguous()
    graph.nodes[node_type].data["sequence_base_feature"] = compact_features
    return compact_features


def _sequence_candidate_relations(
    graph: dgl.DGLHeteroGraph,
    profile: dict,
) -> List[str]:
    excluded_relations = {str(item) for item in tuple(profile.get("excluded_relations", tuple()) or tuple())}
    return [relation for relation in _sequence_relation_order(graph) if str(relation) not in excluded_relations]


def _scan_relation_payload(
    graph: dgl.DGLHeteroGraph,
    relation: str,
    features: torch.Tensor,
) -> dict[str, float | str]:
    mean_feature, max_feature, relation_degree = _aggregate_relation_features(graph, relation, features)
    relation_delta = mean_feature - features
    role_delta = max_feature - mean_feature
    degree_strength = torch.log1p(relation_degree).unsqueeze(-1)
    coverage = relation_degree.gt(0).float().mean().item()
    local_shift_norm = torch.norm(relation_delta.float(), dim=-1, keepdim=True).div(max(features.size(-1), 1) ** 0.5)
    role_shift_norm = torch.norm(role_delta.float(), dim=-1, keepdim=True).div(max(features.size(-1), 1) ** 0.5)
    relation_score = float(
        degree_strength.mean().item()
        + 0.18 * float(local_shift_norm.mean().item())
        + 0.10 * float(role_shift_norm.mean().item())
        + 0.12 * float(coverage)
    )
    return {
        "relation": relation,
        "coverage": float(coverage),
        "score": relation_score,
    }


def _aggregate_relation_features(
    graph: dgl.DGLHeteroGraph,
    relation: str,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    subgraph = graph.edge_type_subgraph([relation])
    degrees = subgraph.in_degrees().to(features.device).float()
    if subgraph.num_edges() == 0:
        zero = torch.zeros_like(features)
        return zero, zero, degrees

    with subgraph.local_scope():
        subgraph.ndata["h"] = features
        subgraph.update_all(fn.copy_u("h", "m"), fn.mean("m", "mean_agg"))
        mean_agg = subgraph.ndata["mean_agg"].float()

    with subgraph.local_scope():
        subgraph.ndata["h"] = features
        subgraph.update_all(fn.copy_u("h", "m"), fn.max("m", "max_agg"))
        max_agg = subgraph.ndata["max_agg"].float()

    valid_mask = degrees.gt(0).unsqueeze(-1)
    zero = torch.zeros_like(features)
    mean_agg = torch.where(valid_mask, mean_agg, zero)
    max_agg = torch.where(valid_mask, max_agg, zero)
    return mean_agg, max_agg, degrees


def _sequence_semantic_features(
    *,
    base_token: torch.Tensor,
    token_type_id: int,
    relation_rank: float,
    relation_strength: torch.Tensor,
    relation_reliability: torch.Tensor,
    relation_presence: torch.Tensor,
    local_shift_norm: torch.Tensor,
    role_shift_norm: torch.Tensor,
    order_position: float,
) -> torch.Tensor:
    semantic_channels = torch.cat(
        [
            torch.full_like(relation_strength, float(token_type_id) / 4.0),
            torch.full_like(relation_strength, float(relation_rank)),
            relation_strength,
            relation_reliability,
            relation_presence,
            local_shift_norm,
            role_shift_norm,
            torch.full_like(relation_strength, float(order_position)),
        ],
        dim=-1,
    )
    return torch.cat([base_token, semantic_channels], dim=-1)


def _resolve_sequence_base_feature_bank(graph: dgl.DGLHeteroGraph) -> torch.Tensor | None:
    node_type = graph.ntypes[0]
    node_data = graph.nodes[node_type].data
    if "sequence_base_feature" in node_data:
        return node_data["sequence_base_feature"]
    if "feature" in node_data:
        return node_data["feature"]
    return None


def _has_lazy_relation_sequence_payload(graph: dgl.DGLHeteroGraph) -> bool:
    node_type = graph.ntypes[0]
    node_data = graph.nodes[node_type].data
    required_fields = (
        "sequence_mask",
        "sequence_token_weights",
        "sequence_token_types",
        "sequence_relation_ids",
        "sequence_relation_mean_feature",
        "sequence_relation_max_feature",
        "sequence_relation_degree",
    )
    if not all(field_name in node_data for field_name in required_fields):
        return False
    return _resolve_sequence_base_feature_bank(graph) is not None


def _materialize_relation_sequence_chunk(
    graph: dgl.DGLHeteroGraph,
    dataset_name: str,
    start: int,
    end: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    node_type = graph.ntypes[0]
    node_data = graph.nodes[node_type].data
    if "sequence" in node_data:
        return (
            node_data["sequence"][start:end],
            node_data["sequence_mask"][start:end] if "sequence_mask" in node_data else None,
            node_data["sequence_token_weights"][start:end] if "sequence_token_weights" in node_data else None,
            node_data["sequence_token_types"][start:end] if "sequence_token_types" in node_data else None,
            node_data["sequence_relation_ids"][start:end] if "sequence_relation_ids" in node_data else None,
        )
    if not _has_lazy_relation_sequence_payload(graph):
        return None, None, None, None, None

    base_features = _resolve_sequence_base_feature_bank(graph)
    if base_features is None:
        return None, None, None, None, None

    base_chunk = base_features[start:end].float()
    sequence_mask = node_data["sequence_mask"][start:end].bool()
    sequence_token_weights = node_data["sequence_token_weights"][start:end].float()
    sequence_token_types = node_data["sequence_token_types"][start:end].long()
    sequence_relation_ids = node_data["sequence_relation_ids"][start:end].long()
    relation_mean_feature = node_data["sequence_relation_mean_feature"][start:end].float()
    relation_max_feature = node_data["sequence_relation_max_feature"][start:end].float()
    relation_degree = node_data["sequence_relation_degree"][start:end].float()
    global_context = (
        node_data["sequence_global_context"][start:end].float()
        if "sequence_global_context" in node_data
        else None
    )

    batch_size = int(base_chunk.shape[0])
    feature_dim = int(base_chunk.shape[1])
    sequence_length = int(sequence_mask.shape[1]) if sequence_mask.ndim >= 2 else 0
    token_feature_dim = feature_dim + 8
    sequences = torch.zeros(
        (batch_size, sequence_length, token_feature_dim),
        dtype=base_chunk.dtype,
        device=base_chunk.device,
    )
    if sequence_length <= 0:
        return sequences, sequence_mask, sequence_token_weights, sequence_token_types, sequence_relation_ids

    relation_count = int(relation_mean_feature.shape[1]) if relation_mean_feature.ndim >= 3 else 0
    profile = _sequence_profile_for_graph(graph, dataset_name)
    token_order = tuple(str(item) for item in tuple(profile.get("token_order", ("local", "motif", "reliability"))) or tuple())
    total_context_slots = max(len(token_order) * max(relation_count, 1), 1)
    if relation_count > 0:
        degree_strength_bank = torch.log1p(relation_degree).unsqueeze(-1)
        strength_denominator = degree_strength_bank.sum(dim=1).clamp(min=1e-6)
        max_strength = degree_strength_bank.amax(dim=1).clamp(min=1e-6)
    else:
        strength_denominator = torch.ones((batch_size, 1), dtype=base_chunk.dtype, device=base_chunk.device)
        max_strength = torch.ones((batch_size, 1), dtype=base_chunk.dtype, device=base_chunk.device)

    for relation_index in range(relation_count):
        mean_feature = relation_mean_feature[:, relation_index, :]
        max_feature = relation_max_feature[:, relation_index, :]
        relation_degree_column = relation_degree[:, relation_index]
        relation_mask = relation_degree_column.gt(0).unsqueeze(-1)
        degree_strength = torch.log1p(relation_degree_column).unsqueeze(-1)
        relation_strength = torch.where(
            relation_mask,
            degree_strength / strength_denominator,
            torch.zeros_like(degree_strength),
        )
        relation_reliability = torch.where(
            relation_mask,
            0.55 * relation_strength + 0.45 * (degree_strength / max_strength),
            torch.zeros_like(degree_strength),
        )
        relation_delta = mean_feature - base_chunk
        role_delta = max_feature - mean_feature
        local_shift_norm = torch.norm(relation_delta.float(), dim=-1, keepdim=True).div(max(feature_dim, 1) ** 0.5)
        role_shift_norm = torch.norm(role_delta.float(), dim=-1, keepdim=True).div(max(feature_dim, 1) ** 0.5)
        local_stack = mean_feature + float(profile["relation_gain"]) * torch.tanh(max_feature - mean_feature)
        local_stack = local_stack + float(profile["local_delta_gain"]) * relation_strength * relation_delta
        motif_stack = local_stack + float(profile["motif_gain"]) * torch.tanh(role_delta + 0.5 * relation_delta)
        reliability_stack = (
            relation_reliability * local_stack
            + (1.0 - relation_reliability)
            * (0.55 * base_chunk + 0.45 * (motif_stack + float(profile["reliability_gain"]) * relation_delta))
        )
        relation_rank = float(relation_index + 1) / float(max(relation_count, 1))
        relation_presence = relation_mask.float()
        for token_name_index, token_name in enumerate(token_order):
            slot_index = 1 + (relation_index * len(token_order)) + token_name_index
            if slot_index >= sequence_length - 1:
                break
            if token_name == "local":
                token_stack = local_stack
                token_type_id = SEQUENCE_TOKEN_TYPE_LOCAL
            elif token_name == "motif":
                token_stack = motif_stack
                token_type_id = SEQUENCE_TOKEN_TYPE_MOTIF
            elif token_name == "reliability":
                token_stack = reliability_stack
                token_type_id = SEQUENCE_TOKEN_TYPE_RELIABILITY
            else:
                continue
            sequences[:, slot_index, :] = _sequence_semantic_features(
                base_token=token_stack,
                token_type_id=int(token_type_id),
                relation_rank=relation_rank,
                relation_strength=relation_strength,
                relation_reliability=relation_reliability,
                relation_presence=relation_presence,
                local_shift_norm=local_shift_norm,
                role_shift_norm=role_shift_norm,
                order_position=float(token_name_index + 1 + relation_index * len(token_order))
                / float(total_context_slots + 1),
            )

    if global_context is None:
        if relation_count > 0:
            weighted_context = torch.zeros_like(base_chunk)
            for relation_index in range(relation_count):
                mean_feature = relation_mean_feature[:, relation_index, :]
                max_feature = relation_max_feature[:, relation_index, :]
                relation_degree_column = relation_degree[:, relation_index]
                relation_mask = relation_degree_column.gt(0).unsqueeze(-1)
                degree_strength = torch.log1p(relation_degree_column).unsqueeze(-1)
                relation_strength = torch.where(
                    relation_mask,
                    degree_strength / strength_denominator,
                    torch.zeros_like(degree_strength),
                )
                relation_reliability = torch.where(
                    relation_mask,
                    0.55 * relation_strength + 0.45 * (degree_strength / max_strength),
                    torch.zeros_like(degree_strength),
                )
                relation_delta = mean_feature - base_chunk
                role_delta = max_feature - mean_feature
                local_stack = mean_feature + float(profile["relation_gain"]) * torch.tanh(max_feature - mean_feature)
                local_stack = local_stack + float(profile["local_delta_gain"]) * relation_strength * relation_delta
                motif_stack = local_stack + float(profile["motif_gain"]) * torch.tanh(role_delta + 0.5 * relation_delta)
                reliability_stack = (
                    relation_reliability * local_stack
                    + (1.0 - relation_reliability)
                    * (0.55 * base_chunk + 0.45 * (motif_stack + float(profile["reliability_gain"]) * relation_delta))
                )
                weighted_context = weighted_context + (
                    local_stack * relation_reliability
                    + 0.45 * motif_stack * relation_strength
                    + 0.5 * reliability_stack * relation_reliability
                )
            has_relation_context = relation_degree.gt(0).any(dim=1, keepdim=True)
            global_context = torch.where(has_relation_context, weighted_context, base_chunk)
        else:
            global_context = base_chunk

    ones = torch.ones((batch_size, 1), dtype=base_chunk.dtype, device=base_chunk.device)
    zeros = torch.zeros((batch_size, 1), dtype=base_chunk.dtype, device=base_chunk.device)
    sequences[:, 0, :] = _sequence_semantic_features(
        base_token=base_chunk + float(profile["self_blend"]) * (global_context - base_chunk),
        token_type_id=SEQUENCE_TOKEN_TYPE_SELF,
        relation_rank=0.0,
        relation_strength=ones,
        relation_reliability=ones,
        relation_presence=ones,
        local_shift_norm=zeros,
        role_shift_norm=zeros,
        order_position=0.0,
    )
    sequences[:, sequence_length - 1, :] = _sequence_semantic_features(
        base_token=global_context,
        token_type_id=SEQUENCE_TOKEN_TYPE_GLOBAL,
        relation_rank=1.0,
        relation_strength=ones,
        relation_reliability=ones,
        relation_presence=ones,
        local_shift_norm=zeros,
        role_shift_norm=zeros,
        order_position=1.0,
    )
    return sequences, sequence_mask, sequence_token_weights, sequence_token_types, sequence_relation_ids


def _build_relation_sequence(
    graph: dgl.DGLHeteroGraph,
    dataset_name: str,
    *,
    ieee_full_compact_sequences: bool | None = None,
    ieee_sequence_feature_dim: int | None = None,
) -> dict[str, torch.Tensor | List[str] | str]:
    node_type = graph.ntypes[0]
    profile = _sequence_profile_for_graph(
        graph,
        dataset_name,
        ieee_full_compact_sequences=ieee_full_compact_sequences,
        ieee_sequence_feature_dim=ieee_sequence_feature_dim,
    )
    if bool(profile.get("runtime_dynamic", False)):
        candidate_relations = _sequence_candidate_relations(graph, profile)
        priority_relations = tuple(str(item) for item in tuple(profile.get("priority_relations", tuple()) or tuple()))
        relation_order = [relation for relation in priority_relations if relation in candidate_relations]
        relation_order.extend([relation for relation in candidate_relations if relation not in relation_order])
        return {
            "relation_order": relation_order,
            "storage_mode": "runtime_dynamic_topk",
        }
    features = _build_sequence_base_features(graph, dataset_name, profile)
    candidate_relations = _sequence_candidate_relations(graph, profile)
    _memory_log(
        "relation_sequence: streaming_scan_begin "
        f"dataset={dataset_name} feature_dim={int(features.shape[1])} candidates={len(candidate_relations)} "
        f"profile={str(profile.get('name', 'default'))}"
    )
    relation_payloads: list[dict[str, float | str]] = []
    for relation_index, relation in enumerate(candidate_relations, start=1):
        _memory_log(
            "relation_sequence: scan_relation_begin "
            f"index={int(relation_index)}/{int(len(candidate_relations))} relation={str(relation)}"
        )
        payload = _scan_relation_payload(graph, relation, features)
        relation_payloads.append(payload)
        _memory_log(
            "relation_sequence: scan_relation_complete "
            f"index={int(relation_index)}/{int(len(candidate_relations))} relation={str(relation)} "
            f"score={float(payload['score']):.4f} coverage={float(payload['coverage']):.4f}"
        )
    priority_relations = tuple(str(item) for item in tuple(profile.get("priority_relations", tuple()) or tuple()))
    priority_relation_set = set(priority_relations)
    relation_payloads = sorted(
        relation_payloads,
        key=lambda item: (
            str(item["relation"]) in priority_relation_set,
            float(item["score"]),
        ),
        reverse=True,
    )
    all_sorted_payloads = list(relation_payloads)
    filtered_payloads = [
        item
        for item in relation_payloads
        if float(item["score"]) >= float(profile.get("min_relation_score", 0.0))
        and float(item.get("coverage", 0.0)) >= float(profile.get("min_relation_coverage", 0.0))
    ]
    if filtered_payloads:
        relation_payloads = filtered_payloads
    elif relation_payloads:
        relation_payloads = relation_payloads[:1]
    if priority_relations:
        prioritized_payloads = [
            item
            for relation_name in priority_relations
            for item in all_sorted_payloads
            if str(item["relation"]) == relation_name
        ]
        relation_payloads = prioritized_payloads + [
            item for item in relation_payloads if str(item["relation"]) not in priority_relation_set
        ]
    max_relations = profile.get("max_relations")
    if max_relations is not None and len(relation_payloads) > int(max_relations):
        relation_payloads = relation_payloads[: int(max_relations)]
    relation_order = [str(item["relation"]) for item in relation_payloads]
    _memory_log(
        "relation_sequence: streaming_scan_complete "
        f"selected_relations={len(relation_order)} relation_order={relation_order}"
    )

    num_nodes = int(features.shape[0])
    feature_dim = int(features.shape[1])
    token_order = tuple(str(item) for item in tuple(profile.get("token_order", ("local", "motif", "reliability"))) or tuple())
    total_context_slots = max(len(token_order) * max(len(relation_order), 1), 1)
    sequence_length = 2 + (len(relation_order) * len(token_order))
    token_feature_dim = feature_dim + 8
    lazy_materialize = bool(profile.get("lazy_materialize", False))
    _memory_log(
        "relation_sequence: plan_ready "
        f"sequence_length={int(sequence_length)} token_feature_dim={int(token_feature_dim)} "
        f"compact_feature_dim={int(feature_dim)} token_order={list(token_order)} "
        f"storage_mode={'lazy' if lazy_materialize else 'dense'}"
    )

    if relation_order:
        strength_denominator = torch.zeros((num_nodes, 1), dtype=features.dtype, device=features.device)
        max_strength = torch.zeros((num_nodes, 1), dtype=features.dtype, device=features.device)
        has_relation_context = torch.zeros((num_nodes, 1), dtype=torch.bool, device=features.device)
        _memory_log("relation_sequence: streaming_strength_stats_begin")
        for relation in relation_order:
            _, _, relation_degree = _aggregate_relation_features(graph, relation, features)
            degree_strength = torch.log1p(relation_degree).unsqueeze(-1)
            strength_denominator = strength_denominator + degree_strength
            max_strength = torch.maximum(max_strength, degree_strength)
            has_relation_context |= relation_degree.gt(0).unsqueeze(-1)
        strength_denominator = strength_denominator.clamp(min=1e-6)
        max_strength = max_strength.clamp(min=1e-6)
        weighted_context = torch.zeros_like(features)
    else:
        strength_denominator = torch.ones((num_nodes, 1), dtype=features.dtype, device=features.device)
        max_strength = torch.ones((num_nodes, 1), dtype=features.dtype, device=features.device)
        has_relation_context = torch.zeros((num_nodes, 1), dtype=torch.bool, device=features.device)
        weighted_context = torch.zeros_like(features)

    sequence_mask = torch.zeros((num_nodes, sequence_length), dtype=torch.bool, device=features.device)
    sequence_token_weights = torch.zeros((num_nodes, sequence_length), dtype=features.dtype, device=features.device)
    sequence_token_types = torch.zeros((num_nodes, sequence_length), dtype=torch.long, device=features.device)
    sequence_relation_ids = torch.zeros((num_nodes, sequence_length), dtype=torch.long, device=features.device)
    sequences = None
    relation_mean_features = None
    relation_max_features = None
    relation_degree_matrix = None
    sequence_global_context = None
    if lazy_materialize:
        relation_mean_features = torch.zeros(
            (num_nodes, len(relation_order), feature_dim),
            dtype=features.dtype,
            device=features.device,
        )
        relation_max_features = torch.zeros_like(relation_mean_features)
        relation_degree_matrix = torch.zeros(
            (num_nodes, len(relation_order)),
            dtype=features.dtype,
            device=features.device,
        )
        sequence_global_context = torch.zeros_like(features)
    else:
        sequences = torch.zeros((num_nodes, sequence_length, token_feature_dim), dtype=features.dtype, device=features.device)
    _memory_log(
        "relation_sequence: output_buffers_ready "
        f"sequence_shape={(num_nodes, sequence_length, token_feature_dim)} "
        f"relation_bank_shape={(num_nodes, len(relation_order), feature_dim)}"
    )

    def _assign_token_metadata(
        slot_index: int,
        *,
        token_mask_value: torch.Tensor,
        token_weight_value: torch.Tensor,
        token_type_id: int,
        relation_id_value: int,
    ) -> None:
        sequence_mask[:, slot_index] = token_mask_value.bool().squeeze(-1)
        sequence_token_weights[:, slot_index] = token_weight_value.reshape(-1)
        sequence_token_types[:, slot_index] = int(token_type_id)
        sequence_relation_ids[:, slot_index] = int(relation_id_value)

    def _fill_dense_token_slot(
        slot_index: int,
        *,
        base_token: torch.Tensor,
        token_type_id: int,
        relation_rank: float,
        relation_strength: torch.Tensor,
        relation_reliability: torch.Tensor,
        relation_presence: torch.Tensor,
        local_shift_norm: torch.Tensor,
        role_shift_norm: torch.Tensor,
        order_position: float,
        token_mask_value: torch.Tensor,
        token_weight_value: torch.Tensor,
        relation_id_value: int,
    ) -> None:
        if sequences is None:
            return
        sequences[:, slot_index, :] = _sequence_semantic_features(
            base_token=base_token,
            token_type_id=token_type_id,
            relation_rank=relation_rank,
            relation_strength=relation_strength,
            relation_reliability=relation_reliability,
            relation_presence=relation_presence,
            local_shift_norm=local_shift_norm,
            role_shift_norm=role_shift_norm,
            order_position=order_position,
        )
        _assign_token_metadata(
            slot_index,
            token_mask_value=token_mask_value,
            token_weight_value=token_weight_value,
            token_type_id=token_type_id,
            relation_id_value=relation_id_value,
        )

    if relation_order:
        _memory_log("relation_sequence: materialize_begin")
        for relation_index, relation in enumerate(relation_order):
            _memory_log(
                "relation_sequence: materialize_relation_begin "
                f"index={int(relation_index + 1)}/{int(len(relation_order))} relation={str(relation)}"
            )
            mean_feature, max_feature, relation_degree = _aggregate_relation_features(graph, relation, features)
            relation_delta = mean_feature - features
            role_delta = max_feature - mean_feature
            degree_strength = torch.log1p(relation_degree).unsqueeze(-1)
            relation_mask = relation_degree.gt(0).unsqueeze(-1)
            relation_strength = torch.where(
                relation_mask,
                degree_strength / strength_denominator,
                torch.zeros_like(degree_strength),
            )
            relation_reliability = torch.where(
                relation_mask,
                0.55 * relation_strength + 0.45 * (degree_strength / max_strength),
                torch.zeros_like(degree_strength),
            )
            if relation_mean_features is not None:
                relation_mean_features[:, relation_index, :] = mean_feature
            if relation_max_features is not None:
                relation_max_features[:, relation_index, :] = max_feature
            if relation_degree_matrix is not None:
                relation_degree_matrix[:, relation_index] = relation_degree.float()
            local_shift_norm = torch.norm(relation_delta.float(), dim=-1, keepdim=True).div(max(feature_dim, 1) ** 0.5)
            role_shift_norm = torch.norm(role_delta.float(), dim=-1, keepdim=True).div(max(feature_dim, 1) ** 0.5)
            local_stack = mean_feature + float(profile["relation_gain"]) * torch.tanh(max_feature - mean_feature)
            local_stack = local_stack + float(profile["local_delta_gain"]) * relation_strength * relation_delta
            motif_stack = local_stack + float(profile["motif_gain"]) * torch.tanh(role_delta + 0.5 * relation_delta)
            reliability_stack = (
                relation_reliability * local_stack
                + (1.0 - relation_reliability)
                * (0.55 * features + 0.45 * (motif_stack + float(profile["reliability_gain"]) * relation_delta))
            )
            weighted_context = weighted_context + (
                local_stack * relation_reliability
                + 0.45 * motif_stack * relation_strength
                + 0.5 * reliability_stack * relation_reliability
            )
            relation_id_value = relation_index + 1
            relation_rank = float(relation_index + 1) / float(max(len(relation_order), 1))
            relation_presence = relation_mask.float()
            for token_name_index, token_name in enumerate(token_order):
                order_slot = (relation_index * len(token_order)) + token_name_index + 1
                slot_index = 1 + (relation_index * len(token_order)) + token_name_index
                if token_name == "local":
                    token_stack = local_stack
                    token_weight_value = torch.where(
                        relation_mask,
                        1.0 + relation_strength,
                        torch.zeros_like(relation_strength),
                    )
                    token_type_id = SEQUENCE_TOKEN_TYPE_LOCAL
                elif token_name == "motif":
                    token_stack = motif_stack
                    token_weight_value = torch.where(
                        relation_mask,
                        1.0 + 0.5 * (relation_strength + relation_reliability),
                        torch.zeros_like(relation_strength),
                    )
                    token_type_id = SEQUENCE_TOKEN_TYPE_MOTIF
                elif token_name == "reliability":
                    token_stack = reliability_stack
                    token_weight_value = torch.where(
                        relation_mask,
                        1.0 + relation_reliability,
                        torch.zeros_like(relation_strength),
                    )
                    token_type_id = SEQUENCE_TOKEN_TYPE_RELIABILITY
                else:
                    continue
                token_kwargs = {
                    "base_token": token_stack,
                    "token_type_id": int(token_type_id),
                    "relation_rank": relation_rank,
                    "relation_strength": relation_strength,
                    "relation_reliability": relation_reliability,
                    "relation_presence": relation_presence,
                    "local_shift_norm": local_shift_norm,
                    "role_shift_norm": role_shift_norm,
                    "order_position": float(order_slot) / float(total_context_slots + 1),
                    "token_mask_value": relation_mask,
                    "token_weight_value": token_weight_value,
                    "relation_id_value": relation_id_value,
                }
                if lazy_materialize:
                    _assign_token_metadata(
                        slot_index,
                        token_mask_value=relation_mask,
                        token_weight_value=token_weight_value,
                        token_type_id=int(token_type_id),
                        relation_id_value=relation_id_value,
                    )
                else:
                    _fill_dense_token_slot(slot_index, **token_kwargs)
            _memory_log(
                "relation_sequence: materialize_relation_complete "
                f"index={int(relation_index + 1)}/{int(len(relation_order))} relation={str(relation)}"
            )
        global_token = torch.where(has_relation_context, weighted_context, features)
    else:
        global_token = features

    self_token_kwargs = {
        "base_token": features + float(profile["self_blend"]) * (global_token - features),
        "token_type_id": SEQUENCE_TOKEN_TYPE_SELF,
        "relation_rank": 0.0,
        "relation_strength": torch.ones((num_nodes, 1), dtype=features.dtype, device=features.device),
        "relation_reliability": torch.ones((num_nodes, 1), dtype=features.dtype, device=features.device),
        "relation_presence": torch.ones((num_nodes, 1), dtype=features.dtype, device=features.device),
        "local_shift_norm": torch.zeros((num_nodes, 1), dtype=features.dtype, device=features.device),
        "role_shift_norm": torch.zeros((num_nodes, 1), dtype=features.dtype, device=features.device),
        "order_position": 0.0,
        "token_mask_value": torch.ones((num_nodes, 1), dtype=torch.bool, device=features.device),
        "token_weight_value": torch.ones((num_nodes, 1), dtype=features.dtype, device=features.device),
        "relation_id_value": 0,
    }
    global_token_kwargs = {
        "base_token": global_token,
        "token_type_id": SEQUENCE_TOKEN_TYPE_GLOBAL,
        "relation_rank": 1.0,
        "relation_strength": torch.ones((num_nodes, 1), dtype=features.dtype, device=features.device),
        "relation_reliability": torch.ones((num_nodes, 1), dtype=features.dtype, device=features.device),
        "relation_presence": torch.ones((num_nodes, 1), dtype=features.dtype, device=features.device),
        "local_shift_norm": torch.zeros((num_nodes, 1), dtype=features.dtype, device=features.device),
        "role_shift_norm": torch.zeros((num_nodes, 1), dtype=features.dtype, device=features.device),
        "order_position": 1.0,
        "token_mask_value": torch.ones((num_nodes, 1), dtype=torch.bool, device=features.device),
        "token_weight_value": torch.ones((num_nodes, 1), dtype=features.dtype, device=features.device),
        "relation_id_value": 0,
    }
    if lazy_materialize:
        _assign_token_metadata(
            0,
            token_mask_value=self_token_kwargs["token_mask_value"],
            token_weight_value=self_token_kwargs["token_weight_value"],
            token_type_id=int(self_token_kwargs["token_type_id"]),
            relation_id_value=int(self_token_kwargs["relation_id_value"]),
        )
        _assign_token_metadata(
            sequence_length - 1,
            token_mask_value=global_token_kwargs["token_mask_value"],
            token_weight_value=global_token_kwargs["token_weight_value"],
            token_type_id=int(global_token_kwargs["token_type_id"]),
            relation_id_value=int(global_token_kwargs["relation_id_value"]),
        )
        if sequence_global_context is not None:
            sequence_global_context[:, :] = global_token
    else:
        _fill_dense_token_slot(0, **self_token_kwargs)
        _fill_dense_token_slot(sequence_length - 1, **global_token_kwargs)
    _memory_log(
        "relation_sequence: materialize_complete "
        f"sequence_length={int(sequence_length)} token_feature_dim={int(token_feature_dim)}"
    )
    payload: dict[str, torch.Tensor | List[str] | str] = {
        "storage_mode": "lazy" if lazy_materialize else "dense",
        "relation_order": relation_order,
        "sequence_mask": sequence_mask,
        "sequence_token_weights": sequence_token_weights.float(),
        "sequence_token_types": sequence_token_types,
        "sequence_relation_ids": sequence_relation_ids,
    }
    if lazy_materialize:
        payload["sequence_relation_mean_feature"] = relation_mean_features.float() if relation_mean_features is not None else torch.zeros(0)
        payload["sequence_relation_max_feature"] = relation_max_features.float() if relation_max_features is not None else torch.zeros(0)
        payload["sequence_relation_degree"] = relation_degree_matrix.float() if relation_degree_matrix is not None else torch.zeros(0)
        if sequence_global_context is not None:
            payload["sequence_global_context"] = sequence_global_context.float()
    else:
        payload["sequence"] = sequences.float() if sequences is not None else torch.zeros(0)
    return payload


def _sequence_storage_dtype(sequences: torch.Tensor) -> torch.dtype:
    # IEEE-scale graphs store a very large precomputed token tensor. Keep the
    # runtime math in fp32, but store the cached sequence tensor more compactly.
    return torch.float16 if sequences.numel() >= 25_000_000 else torch.float32


def _attach_relation_sequence(
    graph: dgl.DGLHeteroGraph,
    dataset_name: str,
    *,
    ieee_full_compact_sequences: bool | None = None,
    ieee_sequence_feature_dim: int | None = None,
) -> List[str]:
    _memory_log(f"relation_sequence: begin dataset={dataset_name}")
    payload = _build_relation_sequence(
        graph,
        dataset_name=dataset_name,
        ieee_full_compact_sequences=ieee_full_compact_sequences,
        ieee_sequence_feature_dim=ieee_sequence_feature_dim,
    )
    relation_order = [str(item) for item in list(payload.get("relation_order", []))]
    storage_mode = str(payload.get("storage_mode", "dense"))
    if storage_mode == "runtime_dynamic_topk":
        node_type = graph.ntypes[0]
        for field_name in (
            "sequence",
            "sequence_mask",
            "sequence_token_weights",
            "sequence_token_types",
            "sequence_relation_ids",
            "sequence_relation_mean_feature",
            "sequence_relation_max_feature",
            "sequence_global_context",
            "sequence_base_feature",
        ):
            if field_name in graph.nodes[node_type].data:
                del graph.nodes[node_type].data[field_name]
        _memory_log(
            "relation_sequence: attached "
            f"storage_mode={storage_mode} relations={len(relation_order)}"
        )
        return relation_order
    sequence_mask = payload["sequence_mask"].bool()
    sequence_token_weights = payload["sequence_token_weights"].float()
    sequence_token_types = payload["sequence_token_types"].long()
    sequence_relation_ids = payload["sequence_relation_ids"].long()
    sequence_tensor = payload["sequence"].float() if "sequence" in payload else None
    base_feature_bank = _resolve_sequence_base_feature_bank(graph)
    sequence_feature_dim = (
        int(sequence_tensor.shape[2])
        if sequence_tensor is not None and sequence_tensor.ndim >= 3
        else int(base_feature_bank.shape[1] + 8)
        if base_feature_bank is not None and base_feature_bank.ndim >= 2
        else 0
    )
    _memory_log(
        "relation_sequence: built "
        f"nodes={int(sequence_mask.shape[0])} seq_len={int(sequence_mask.shape[1]) if sequence_mask.ndim >= 2 else 0} "
        f"seq_dim={int(sequence_feature_dim)} relations={len(relation_order)} storage_mode={storage_mode}"
    )
    node_type = graph.ntypes[0]
    if sequence_tensor is not None:
        graph.nodes[node_type].data["sequence"] = sequence_tensor.to(dtype=_sequence_storage_dtype(sequence_tensor))
    elif "sequence" in graph.nodes[node_type].data:
        del graph.nodes[node_type].data["sequence"]
    graph.nodes[node_type].data["sequence_mask"] = sequence_mask.bool()
    graph.nodes[node_type].data["sequence_token_weights"] = sequence_token_weights.float()
    graph.nodes[node_type].data["sequence_token_types"] = sequence_token_types.long()
    graph.nodes[node_type].data["sequence_relation_ids"] = sequence_relation_ids.long()
    if "sequence_relation_mean_feature" in payload:
        relation_mean_features = payload["sequence_relation_mean_feature"].float()
        graph.nodes[node_type].data["sequence_relation_mean_feature"] = relation_mean_features.to(
            dtype=_sequence_storage_dtype(relation_mean_features)
        )
    elif "sequence_relation_mean_feature" in graph.nodes[node_type].data:
        del graph.nodes[node_type].data["sequence_relation_mean_feature"]
    if "sequence_relation_max_feature" in payload:
        relation_max_features = payload["sequence_relation_max_feature"].float()
        graph.nodes[node_type].data["sequence_relation_max_feature"] = relation_max_features.to(
            dtype=_sequence_storage_dtype(relation_max_features)
        )
    elif "sequence_relation_max_feature" in graph.nodes[node_type].data:
        del graph.nodes[node_type].data["sequence_relation_max_feature"]
    if "sequence_relation_degree" in payload:
        graph.nodes[node_type].data["sequence_relation_degree"] = payload["sequence_relation_degree"].float()
    elif "sequence_relation_degree" in graph.nodes[node_type].data:
        del graph.nodes[node_type].data["sequence_relation_degree"]
    if "sequence_global_context" in payload:
        sequence_global_context = payload["sequence_global_context"].float()
        graph.nodes[node_type].data["sequence_global_context"] = sequence_global_context.to(
            dtype=_sequence_storage_dtype(sequence_global_context)
        )
    elif "sequence_global_context" in graph.nodes[node_type].data:
        del graph.nodes[node_type].data["sequence_global_context"]
    if "sequence_base_feature" in graph.nodes[node_type].data:
        sequence_base_feature = graph.nodes[node_type].data["sequence_base_feature"].float()
        graph.nodes[node_type].data["sequence_base_feature"] = sequence_base_feature.to(
            dtype=_sequence_storage_dtype(sequence_base_feature)
        )
    _memory_log(
        "relation_sequence: attached "
        f"storage_mode={storage_mode} "
        f"sequence_dtype={str(graph.nodes[node_type].data['sequence'].dtype).replace('torch.', '') if 'sequence' in graph.nodes[node_type].data else 'n/a'} "
        f"relation_bank_dtype={str(graph.nodes[node_type].data['sequence_relation_mean_feature'].dtype).replace('torch.', '') if 'sequence_relation_mean_feature' in graph.nodes[node_type].data else 'n/a'}"
    )
    return relation_order


def _sequence_quality_summary(
    graph: dgl.DGLHeteroGraph,
    relation_order: List[str],
    dataset_name: str,
) -> dict:
    node_type = graph.ntypes[0]
    node_data = graph.nodes[node_type].data
    if "sequence" in node_data:
        sequence = node_data["sequence"].float()
        storage_mode = "dense"
    elif _has_lazy_relation_sequence_payload(graph):
        sequence = None
        storage_mode = "lazy_chunk_materialized"
    else:
        profile = _sequence_profile_for_graph(graph, dataset_name)
        if not bool(profile.get("runtime_dynamic", False)) or "sequence_relation_degree" not in node_data:
            return {}
        relation_degree = node_data["sequence_relation_degree"].float()
        raw_feature_dim = int(node_data["feature"].shape[1]) if "feature" in node_data else 0
        token_order = tuple(profile.get("token_order", tuple()) or tuple())
        relation_coverage = {}
        for relation_index, relation_name in enumerate(relation_order):
            if relation_index >= relation_degree.size(1):
                break
            relation_coverage[str(relation_name)] = float(relation_degree[:, relation_index].gt(0).float().mean().item())
        return {
            "builder_version": str(SEQUENCE_BUILDER_VERSION),
            "builder_profile": str(profile.get("name", "default")),
            "streaming_mode": False,
            "storage_mode": "runtime_dynamic_topk",
            "lazy_materialization": True,
            "sequence_length": int(2 + len(relation_order) * len(token_order)),
            "sequence_feature_dim": int(raw_feature_dim + 8),
            "base_feature_dim": int(raw_feature_dim),
            "raw_feature_dim": int(raw_feature_dim),
            "compact_mode_enabled": False,
            "semantic_feature_dim": 8,
            "selected_relation_count": int(len(relation_order)),
            "relation_order": [str(item) for item in relation_order],
            "priority_relations": [str(item) for item in tuple(profile.get("priority_relations", tuple()) or tuple())],
            "token_order": [str(item) for item in token_order],
            "relation_coverage": relation_coverage,
            "runtime_projector": True,
        }
    profile = _sequence_profile_for_graph(graph, dataset_name)

    sequence_mask = (
        node_data["sequence_mask"].bool()
        if "sequence_mask" in node_data
        else torch.ones(sequence.shape[:2], dtype=torch.bool, device=sequence.device)
        if sequence is not None
        else torch.ones(
            (
                int(node_data["feature"].shape[0]) if "feature" in node_data else 0,
                1,
            ),
            dtype=torch.bool,
        )
    )
    sequence_types = node_data["sequence_token_types"].long() if "sequence_token_types" in node_data else None
    sequence_relation_ids = node_data["sequence_relation_ids"].long() if "sequence_relation_ids" in node_data else None

    valid_ratio = sequence_mask.float().mean(dim=1)
    valid_length = sequence_mask.float().sum(dim=1)
    nonzero_ratio = sequence.abs().sum(dim=-1).gt(0).float().mean(dim=1) if sequence is not None else valid_ratio
    raw_feature_dim = (
        int(node_data["feature"].shape[1])
        if "feature" in node_data
        else int(sequence.size(-1))
        if sequence is not None
        else 0
    )
    if "sequence_base_feature" in node_data:
        base_feature_dim = int(node_data["sequence_base_feature"].shape[1])
    else:
        base_feature_dim = raw_feature_dim
    sequence_feature_dim = int(sequence.size(-1)) if sequence is not None else int(base_feature_dim + 8)
    semantic_feature_dim = max(int(sequence_feature_dim) - base_feature_dim, 0)

    token_type_labels = {
        SEQUENCE_TOKEN_TYPE_SELF: "self",
        SEQUENCE_TOKEN_TYPE_LOCAL: "local",
        SEQUENCE_TOKEN_TYPE_MOTIF: "motif",
        SEQUENCE_TOKEN_TYPE_RELIABILITY: "reliability",
        SEQUENCE_TOKEN_TYPE_GLOBAL: "global",
    }
    token_type_coverage: dict[str, float] = {}
    if sequence_types is not None:
        for token_type_id, token_type_name in token_type_labels.items():
            type_mask = sequence_mask & sequence_types.eq(int(token_type_id))
            token_type_coverage[token_type_name] = float(type_mask.any(dim=1).float().mean().item())

    relation_coverage: dict[str, float] = {}
    if sequence_relation_ids is not None:
        for relation_index, relation_name in enumerate(relation_order, start=1):
            relation_mask = sequence_mask & sequence_relation_ids.eq(int(relation_index))
            relation_coverage[str(relation_name)] = float(relation_mask.any(dim=1).float().mean().item())

    informative_context_ratio = valid_ratio
    if sequence_types is not None:
        context_mask = sequence_mask & ~sequence_types.eq(SEQUENCE_TOKEN_TYPE_SELF) & ~sequence_types.eq(
            SEQUENCE_TOKEN_TYPE_GLOBAL
        )
        informative_context_ratio = context_mask.float().sum(dim=1) / sequence_mask.float().sum(dim=1).clamp(min=1.0)

    return {
        "builder_version": str(SEQUENCE_BUILDER_VERSION),
        "builder_profile": str(profile.get("name", "default")),
        "streaming_mode": bool(profile.get("streaming_mode", False)),
        "storage_mode": str(storage_mode),
        "lazy_materialization": bool(sequence is None),
        "sequence_length": int(sequence_mask.size(1)),
        "sequence_feature_dim": int(sequence_feature_dim),
        "base_feature_dim": int(base_feature_dim),
        "raw_feature_dim": int(raw_feature_dim),
        "compact_mode_enabled": bool(base_feature_dim != raw_feature_dim),
        "semantic_feature_dim": int(semantic_feature_dim),
        "selected_relation_count": int(len(relation_order)),
        "relation_order": [str(item) for item in relation_order],
        "priority_relations": [str(item) for item in tuple(profile.get("priority_relations", tuple()) or tuple())],
        "token_order": [str(item) for item in tuple(profile.get("token_order", tuple()) or tuple())],
        "valid_ratio_mean": float(valid_ratio.mean().item()),
        "valid_ratio_std": float(valid_ratio.std(unbiased=False).item()),
        "valid_length_mean": float(valid_length.mean().item()),
        "valid_length_std": float(valid_length.std(unbiased=False).item()),
        "nonzero_ratio_mean": float(nonzero_ratio.mean().item()),
        "nonzero_ratio_std": float(nonzero_ratio.std(unbiased=False).item()),
        "informative_context_ratio_mean": float(informative_context_ratio.mean().item()),
        "informative_context_ratio_std": float(informative_context_ratio.std(unbiased=False).item()),
        "token_type_coverage": token_type_coverage,
        "relation_coverage": relation_coverage,
    }


def _stratified_partition(
    node_ids: torch.Tensor,
    labels: torch.Tensor,
    num_clients: int,
    seed: int,
) -> List[torch.Tensor]:
    rng = np.random.default_rng(seed)
    node_ids_np = node_ids.cpu().numpy()
    labels_np = labels.cpu().numpy()
    partitions = [[] for _ in range(num_clients)]

    for label in np.unique(labels_np):
        label_nodes = node_ids_np[labels_np == label].copy()
        rng.shuffle(label_nodes)
        for client_id, shard in enumerate(np.array_split(label_nodes, num_clients)):
            if len(shard) > 0:
                partitions[client_id].append(torch.from_numpy(shard.copy()).long())

    merged_partitions = []
    for client_parts in partitions:
        if not client_parts:
            merged_partitions.append(torch.empty(0, dtype=torch.long))
            continue
        merged = torch.cat(client_parts)
        merged, _ = torch.sort(torch.unique(merged))
        merged_partitions.append(merged)
    return merged_partitions


def _random_partition(
    node_ids: torch.Tensor,
    num_clients: int,
    seed: int,
) -> List[torch.Tensor]:
    rng = np.random.default_rng(seed)
    node_ids_np = node_ids.cpu().numpy().copy()
    rng.shuffle(node_ids_np)
    partitions = []
    for shard in np.array_split(node_ids_np, num_clients):
        if len(shard) == 0:
            partitions.append(torch.empty(0, dtype=torch.long))
            continue
        partitions.append(torch.from_numpy(shard.copy()).long())
    return partitions


def _merge_partitions(*partition_sets: List[torch.Tensor]) -> List[torch.Tensor]:
    if not partition_sets:
        return []
    num_clients = len(partition_sets[0])
    merged_partitions = []
    for client_id in range(num_clients):
        client_parts = []
        for partition_set in partition_sets:
            if client_id >= len(partition_set):
                continue
            part = partition_set[client_id]
            if len(part) > 0:
                client_parts.append(part.long())
        if not client_parts:
            merged_partitions.append(torch.empty(0, dtype=torch.long))
            continue
        merged = torch.cat(client_parts)
        merged = torch.sort(torch.unique(merged)).values
        merged_partitions.append(merged)
    return merged_partitions


def _expand_client_nodes(
    graph: dgl.DGLHeteroGraph,
    node_type: str,
    owned_nodes: torch.Tensor,
    hops: int,
) -> torch.Tensor:
    frontier_nodes = owned_nodes
    all_nodes = owned_nodes
    for _ in range(hops):
        frontier_mask = torch.zeros(graph.num_nodes(node_type), dtype=torch.bool)
        frontier_mask[frontier_nodes.long()] = True
        candidate_nodes = [all_nodes]
        for relation in graph.etypes:
            src, dst = graph.edges(etype=relation)
            neighbor_mask = frontier_mask[src.long()] | frontier_mask[dst.long()]
            if neighbor_mask.any():
                candidate_nodes.append(src[neighbor_mask].long())
                candidate_nodes.append(dst[neighbor_mask].long())
        merged = torch.unique(torch.cat(candidate_nodes))
        frontier_nodes = merged
        all_nodes = merged
    return torch.sort(all_nodes).values


def _build_client_subgraph(
    graph: dgl.DGLHeteroGraph,
    node_type: str,
    owned_nodes: torch.Tensor,
    hops: int,
) -> dgl.DGLHeteroGraph:
    local_nodes = _expand_client_nodes(graph, node_type, owned_nodes, hops=hops)
    subgraph = dgl.node_subgraph(graph, {node_type: local_nodes.to(graph.idtype)})
    global_node_ids = subgraph.nodes[node_type].data[dgl.NID].long()
    owned_mask = torch.isin(global_node_ids, owned_nodes.long())

    subgraph.nodes[node_type].data["train_mask"] = owned_mask
    subgraph.nodes[node_type].data["valid_mask"] = torch.zeros_like(owned_mask)
    subgraph.nodes[node_type].data["test_mask"] = torch.zeros_like(owned_mask)

    if "train_supervised_mask" in subgraph.nodes[node_type].data:
        supervised_mask = subgraph.nodes[node_type].data["train_supervised_mask"].bool() & owned_mask
    else:
        supervised_mask = owned_mask.clone()
    if "train_unlabeled_mask" in subgraph.nodes[node_type].data:
        unlabeled_mask = subgraph.nodes[node_type].data["train_unlabeled_mask"].bool() & owned_mask
    else:
        unlabeled_mask = owned_mask & ~supervised_mask

    subgraph.nodes[node_type].data["train_supervised_mask"] = supervised_mask
    subgraph.nodes[node_type].data["train_unlabeled_mask"] = unlabeled_mask
    _refresh_homo_edge_train_mask(subgraph)

    global_node_ids = subgraph.nodes[node_type].data[dgl.NID].long()
    lookup_size = int(global_node_ids.max().item()) + 1 if global_node_ids.numel() > 0 else 0
    global_to_local = torch.full((lookup_size,), -1, dtype=torch.long, device=global_node_ids.device)
    if global_node_ids.numel() > 0:
        global_to_local[global_node_ids] = torch.arange(global_node_ids.numel(), device=global_node_ids.device)

    if "event_history_indices" in subgraph.nodes[node_type].data:
        history_indices = subgraph.nodes[node_type].data["event_history_indices"].long()
        remapped_history = torch.full_like(history_indices, -1)
        valid_history = history_indices.ge(0)
        if valid_history.any() and global_node_ids.numel() > 0:
            required_lookup_size = int(max(global_node_ids.max().item(), history_indices[valid_history].max().item())) + 1
            if required_lookup_size > lookup_size:
                expanded = torch.full((required_lookup_size,), -1, dtype=torch.long, device=global_node_ids.device)
                expanded[:lookup_size] = global_to_local
                global_to_local = expanded
                global_to_local[global_node_ids] = torch.arange(global_node_ids.numel(), device=global_node_ids.device)
                lookup_size = required_lookup_size
            remapped_history[valid_history] = global_to_local[history_indices[valid_history]]
        valid_history = valid_history & remapped_history.ge(0)
        subgraph.nodes[node_type].data["event_history_indices"] = remapped_history
        if "event_mask" in subgraph.nodes[node_type].data:
            subgraph.nodes[node_type].data["event_mask"] = (
                subgraph.nodes[node_type].data["event_mask"].bool() & valid_history
            )
        if "event_time_deltas" in subgraph.nodes[node_type].data:
            subgraph.nodes[node_type].data["event_time_deltas"] = subgraph.nodes[node_type].data[
                "event_time_deltas"
            ].float() * valid_history.to(dtype=torch.float32)
        if "event_token_weights" in subgraph.nodes[node_type].data:
            subgraph.nodes[node_type].data["event_token_weights"] = subgraph.nodes[node_type].data[
                "event_token_weights"
            ].float() * valid_history.to(dtype=torch.float32)
        if "event_token_types" in subgraph.nodes[node_type].data:
            subgraph.nodes[node_type].data["event_token_types"] = subgraph.nodes[node_type].data[
                "event_token_types"
            ].long() * valid_history.to(dtype=torch.long)
        if "event_source_ids" in subgraph.nodes[node_type].data:
            subgraph.nodes[node_type].data["event_source_ids"] = subgraph.nodes[node_type].data[
                "event_source_ids"
            ].long() * valid_history.to(dtype=torch.long)

    if "sequence_relation_topk_indices" in subgraph.nodes[node_type].data:
        relation_indices = subgraph.nodes[node_type].data["sequence_relation_topk_indices"].long()
        remapped_relation_indices = torch.full_like(relation_indices, -1)
        valid_relation_indices = relation_indices.ge(0)
        if valid_relation_indices.any() and global_node_ids.numel() > 0:
            required_lookup_size = int(max(global_node_ids.max().item(), relation_indices[valid_relation_indices].max().item())) + 1
            if required_lookup_size > lookup_size:
                expanded = torch.full((required_lookup_size,), -1, dtype=torch.long, device=global_node_ids.device)
                expanded[:lookup_size] = global_to_local
                global_to_local = expanded
                global_to_local[global_node_ids] = torch.arange(global_node_ids.numel(), device=global_node_ids.device)
                lookup_size = required_lookup_size
            remapped_relation_indices[valid_relation_indices] = global_to_local[relation_indices[valid_relation_indices]]
        subgraph.nodes[node_type].data["sequence_relation_topk_indices"] = remapped_relation_indices

    if "homo" in subgraph.etypes:
        src, dst = subgraph.edges(etype="homo")
        edge_mask = subgraph.nodes[node_type].data["train_supervised_mask"].bool()[src] & subgraph.nodes[node_type].data[
            "train_supervised_mask"
        ].bool()[dst]
        subgraph.edges["homo"].data["train_mask"] = edge_mask
    return subgraph


def _compute_class_weights(graph: dgl.DGLHeteroGraph) -> torch.Tensor:
    labels = graph.ndata["label"][_supervised_training_mask(graph)]
    counts = torch.bincount(labels, minlength=2).float().clamp(min=1.0)
    weights = counts.sum() / (counts * len(counts))
    return weights


def _compute_class_counts(graph: dgl.DGLHeteroGraph) -> torch.Tensor:
    labels = graph.ndata["label"][_supervised_training_mask(graph)]
    return torch.bincount(labels, minlength=2).float().clamp(min=1.0)


def load_splitgnn_dataset(
    dataset_name: str,
    data_dir: str,
    num_clients: int = 3,
    seed: int = 42,
    client_hops: int = 1,
    label_fraction: float = 1.0,
    active_learning_feedback_path: str = "",
) -> DatasetBundle:
    graph = _load_graph(dataset_name, data_dir)
    node_type = graph.ntypes[0]
    _attach_dataset_context_defaults(graph, dataset_name=dataset_name)
    relation_order = _attach_relation_sequence(graph, dataset_name=dataset_name)
    _apply_label_scarcity(graph, label_fraction=label_fraction, seed=seed)
    _apply_active_learning_feedback(graph, active_learning_feedback_path, dataset_name=dataset_name)

    # Keep the full train pool in client subgraphs, but only stratify the
    # supervised subset. Unlabeled nodes must not use hidden ground-truth
    # labels during client assignment.
    train_mask = graph.nodes[node_type].data["train_mask"].bool()
    train_supervised_mask = graph.nodes[node_type].data["train_supervised_mask"].bool() & train_mask
    train_unlabeled_mask = graph.nodes[node_type].data["train_unlabeled_mask"].bool() & train_mask
    supervised_nodes = train_supervised_mask.nonzero(as_tuple=False).flatten()
    supervised_labels = graph.nodes[node_type].data["label"][train_supervised_mask]
    unlabeled_nodes = train_unlabeled_mask.nonzero(as_tuple=False).flatten()
    supervised_partitions = _stratified_partition(
        supervised_nodes,
        supervised_labels,
        num_clients=num_clients,
        seed=seed,
    )
    unlabeled_partitions = _random_partition(unlabeled_nodes, num_clients=num_clients, seed=seed + 1)
    owned_partitions = _merge_partitions(supervised_partitions, unlabeled_partitions)

    clients = []
    for client_id, owned_nodes in enumerate(owned_partitions):
        if len(owned_nodes) == 0:
            continue
        subgraph = _build_client_subgraph(graph, node_type, owned_nodes, hops=client_hops)
        local_train_nodes = int(subgraph.nodes[node_type].data["train_mask"].sum().item())
        clients.append(
            ClientShard(
                client_id=client_id,
                owned_global_nodes=owned_nodes,
                subgraph=subgraph,
                train_nodes=local_train_nodes,
            )
        )

    bundle = DatasetBundle(
        name=dataset_name,
        graph=graph,
        node_type=node_type,
        relation_order=relation_order,
        class_weights=_compute_class_weights(graph),
        class_counts=_compute_class_counts(graph),
        clients=clients,
    )
    bundle.data_summary = {
        "dataset": str(dataset_name),
        "num_clients": int(len(clients)),
        "label_fraction": float(label_fraction),
        "active_learning_feedback_path": str(active_learning_feedback_path or ""),
        "sequence_quality": _sequence_quality_summary(graph, relation_order, dataset_name=dataset_name),
    }
    return bundle


def load_graph_for_inference(
    dataset_name: str,
    data_dir: str | None = None,
    graph_path: str | None = None,
) -> tuple[dgl.DGLHeteroGraph, str, List[str]]:
    """Load an inference graph from the repository layout or an external DGL file."""

    if graph_path is None:
        if data_dir is None:
            raise ValueError("Either data_dir or graph_path must be provided for inference.")
        resolved_graph_path = resolve_graph_path(data_dir, dataset_name)
    else:
        resolved_graph_path = str(Path(graph_path).resolve())
        if not os.path.exists(resolved_graph_path):
            raise FileNotFoundError(f"Cannot find inference graph at {resolved_graph_path}")

    graph = dgl.load_graphs(resolved_graph_path)[0][0]
    require_labels_and_masks = graph_path is None
    graph = _prepare_graph(
        dataset_name=dataset_name,
        graph=graph,
        require_labels_and_masks=require_labels_and_masks,
    )
    node_type = graph.ntypes[0]
    _attach_dataset_context_defaults(graph, dataset_name=dataset_name)
    relation_order = _attach_relation_sequence(graph, dataset_name=dataset_name)
    return graph, node_type, relation_order
