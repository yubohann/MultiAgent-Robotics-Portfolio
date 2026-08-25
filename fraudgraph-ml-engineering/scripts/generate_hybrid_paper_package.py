from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_ml_engineering.paths import ARTIFACTS_ROOT, GRAPH_ROOT

PAPER_PROTOCOL_ROOT = ARTIFACTS_ROOT / "experiments"
SUPERVISED_ROOT = PAPER_PROTOCOL_ROOT / "supervised_structure"
LOW_LABEL_ROOT = PAPER_PROTOCOL_ROOT / "label_scarcity"
FUSION_ROOT = PAPER_PROTOCOL_ROOT / "fusion_ablation"
OUTPUT_ROOT = PAPER_PROTOCOL_ROOT / "paper_package"
SPLITGNN_DATA_DIR = GRAPH_ROOT

SUPPORTED_DATASETS = ("yelp", "amazon", "comp")
SUPPORTED_SUPERVISED_EXPERIMENTS = ("full", "no_gnn", "no_transformer")
SUPPORTED_LOW_LABEL_EXPERIMENTS = ("full", "no_transformer")
SUPPORTED_FUSION_VARIANTS = (
    "graph_only",
    "late_fusion",
    "graph_dominant_residual",
    "shared_private_prototype",
)
EXCLUDE_TAGS = ("smoke", "probe", "test", "debug", "stagecheck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the SplitGNN hybrid paper package.")
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--datasets", nargs="+", default=list(SUPPORTED_DATASETS), choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--allow_smoke", action="store_true")
    parser.add_argument("--case_topk", type=int, default=10)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _aggregate_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _extract_seed(seed_dir: str) -> int:
    match = re.match(r"seed(\d+)", str(seed_dir).strip().lower())
    return int(match.group(1)) if match else -1


def _skip_seed_dir(seed_dir: str, allow_smoke: bool) -> bool:
    if allow_smoke:
        return False
    lowered = str(seed_dir).lower()
    return any(tag in lowered for tag in EXCLUDE_TAGS)


def _metric_bundle(summary: dict[str, Any]) -> dict[str, float]:
    test_metrics = dict(summary.get("test", {}) or {})
    return {
        "best_valid_auc": _safe_float(summary.get("best_valid_auc")),
        "best_valid_f1_macro": _safe_float(summary.get("best_valid_f1_macro")),
        "test_auc": _safe_float(test_metrics.get("auc")),
        "test_f1_macro": _safe_float(test_metrics.get("f1_macro")),
        "test_recall": _safe_float(test_metrics.get("recall")),
    }


def _scan_supervised(datasets: set[str], allow_smoke: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not SUPERVISED_ROOT.exists():
        return records
    for summary_path in SUPERVISED_ROOT.rglob("*_hybrid_summary.json"):
        parts = summary_path.relative_to(SUPERVISED_ROOT).parts
        if len(parts) < 4:
            continue
        experiment_name, seed_dir, dataset_name = parts[0], parts[1], parts[2]
        if experiment_name not in SUPPORTED_SUPERVISED_EXPERIMENTS or dataset_name not in datasets:
            continue
        if _skip_seed_dir(seed_dir, allow_smoke):
            continue
        payload = _load_json(summary_path)
        summary = dict(payload.get("summary", payload) or {})
        if not bool(summary.get("completed")):
            continue
        diagnostics_path = Path(str(summary.get("diagnostics_path", ""))).resolve() if summary.get("diagnostics_path") else summary_path.parent / f"{dataset_name}_diagnostics.json"
        records.append(
            {
                "dataset": dataset_name,
                "experiment": experiment_name,
                "seed_dir": seed_dir,
                "seed": _extract_seed(seed_dir),
                "summary_path": str(summary_path),
                "diagnostics_path": str(diagnostics_path),
                "summary": summary,
                "history": list(payload.get("history", []) or []),
                **_metric_bundle(summary),
            }
        )
    return records


def _parse_label_fraction_dir(label_dir: str) -> float:
    normalized = str(label_dir).replace("label_fraction_", "").replace("p", ".")
    return _safe_float(normalized, default=0.0)


def _scan_low_label(datasets: set[str], allow_smoke: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not LOW_LABEL_ROOT.exists():
        return records
    for summary_path in LOW_LABEL_ROOT.rglob("*_hybrid_summary.json"):
        parts = summary_path.relative_to(LOW_LABEL_ROOT).parts
        if len(parts) < 5:
            continue
        experiment_name, label_dir, seed_dir, dataset_name = parts[0], parts[1], parts[2], parts[3]
        if experiment_name not in SUPPORTED_LOW_LABEL_EXPERIMENTS or dataset_name not in datasets:
            continue
        if _skip_seed_dir(seed_dir, allow_smoke):
            continue
        payload = _load_json(summary_path)
        summary = dict(payload.get("summary", payload) or {})
        if not bool(summary.get("completed")):
            continue
        diagnostics_path = Path(str(summary.get("diagnostics_path", ""))).resolve() if summary.get("diagnostics_path") else summary_path.parent / f"{dataset_name}_diagnostics.json"
        records.append(
            {
                "dataset": dataset_name,
                "experiment": experiment_name,
                "label_fraction": _parse_label_fraction_dir(label_dir),
                "label_fraction_dir": label_dir,
                "seed_dir": seed_dir,
                "seed": _extract_seed(seed_dir),
                "summary_path": str(summary_path),
                "diagnostics_path": str(diagnostics_path),
                "summary": summary,
                "history": list(payload.get("history", []) or []),
                **_metric_bundle(summary),
            }
        )
    return records


def _scan_fusion(datasets: set[str], allow_smoke: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not FUSION_ROOT.exists():
        return records
    for summary_path in FUSION_ROOT.rglob("*_hybrid_summary.json"):
        parts = summary_path.relative_to(FUSION_ROOT).parts
        if len(parts) < 4:
            continue
        variant_name, seed_dir, dataset_name = parts[0], parts[1], parts[2]
        if variant_name not in SUPPORTED_FUSION_VARIANTS or dataset_name not in datasets:
            continue
        if _skip_seed_dir(seed_dir, allow_smoke):
            continue
        payload = _load_json(summary_path)
        summary = dict(payload.get("summary", payload) or {})
        if not bool(summary.get("completed")):
            continue
        diagnostics_path = Path(str(summary.get("diagnostics_path", ""))).resolve() if summary.get("diagnostics_path") else summary_path.parent / f"{dataset_name}_diagnostics.json"
        records.append(
            {
                "dataset": dataset_name,
                "variant": variant_name,
                "seed_dir": seed_dir,
                "seed": _extract_seed(seed_dir),
                "summary_path": str(summary_path),
                "diagnostics_path": str(diagnostics_path),
                "summary": summary,
                "history": list(payload.get("history", []) or []),
                **_metric_bundle(summary),
            }
        )
    return records


def _aggregate_supervised(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for dataset_name in sorted({item["dataset"] for item in records}):
        dataset_records = [item for item in records if item["dataset"] == dataset_name]
        aggregate[dataset_name] = {}
        for experiment_name in SUPPORTED_SUPERVISED_EXPERIMENTS:
            group = [item for item in dataset_records if item["experiment"] == experiment_name]
            if not group:
                continue
            payload = {
                "num_runs": len(group),
                "seeds": [int(item["seed"]) for item in group],
                "best_valid_auc": _aggregate_metric([item["best_valid_auc"] for item in group]),
                "best_valid_f1_macro": _aggregate_metric([item["best_valid_f1_macro"] for item in group]),
                "test_auc": _aggregate_metric([item["test_auc"] for item in group]),
                "test_f1_macro": _aggregate_metric([item["test_f1_macro"] for item in group]),
                "test_recall": _aggregate_metric([item["test_recall"] for item in group]),
                "summary_paths": [item["summary_path"] for item in group],
            }
            aggregate[dataset_name][experiment_name] = payload
            rows.append(
                {
                    "dataset": dataset_name,
                    "experiment": experiment_name,
                    "num_runs": len(group),
                    "seeds": ",".join(str(item["seed"]) for item in group),
                    "best_valid_auc_mean": payload["best_valid_auc"]["mean"],
                    "best_valid_auc_std": payload["best_valid_auc"]["std"],
                    "test_auc_mean": payload["test_auc"]["mean"],
                    "test_auc_std": payload["test_auc"]["std"],
                    "test_f1_macro_mean": payload["test_f1_macro"]["mean"],
                    "test_f1_macro_std": payload["test_f1_macro"]["std"],
                    "test_recall_mean": payload["test_recall"]["mean"],
                    "test_recall_std": payload["test_recall"]["std"],
                }
            )
    return aggregate, rows


def _aggregate_low_label(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate: dict[str, Any] = {}
    curve_rows: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []
    for dataset_name in sorted({item["dataset"] for item in records}):
        dataset_records = [item for item in records if item["dataset"] == dataset_name]
        aggregate[dataset_name] = {}
        for experiment_name in SUPPORTED_LOW_LABEL_EXPERIMENTS:
            experiment_records = [item for item in dataset_records if item["experiment"] == experiment_name]
            if not experiment_records:
                continue
            by_fraction: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for item in experiment_records:
                by_fraction[float(item["label_fraction"])].append(item)
            fraction_payload: dict[str, Any] = {}
            ordered_fractions = sorted(by_fraction)
            for label_fraction in ordered_fractions:
                group = by_fraction[label_fraction]
                payload = {
                    "num_runs": len(group),
                    "seeds": [int(item["seed"]) for item in group],
                    "best_valid_auc": _aggregate_metric([item["best_valid_auc"] for item in group]),
                    "test_auc": _aggregate_metric([item["test_auc"] for item in group]),
                    "test_f1_macro": _aggregate_metric([item["test_f1_macro"] for item in group]),
                    "test_recall": _aggregate_metric([item["test_recall"] for item in group]),
                    "summary_paths": [item["summary_path"] for item in group],
                }
                fraction_key = f"{label_fraction:.4f}"
                fraction_payload[fraction_key] = payload
                curve_rows.append(
                    {
                        "dataset": dataset_name,
                        "experiment": experiment_name,
                        "label_fraction": label_fraction,
                        "num_runs": len(group),
                        "seeds": ",".join(str(item["seed"]) for item in group),
                        "best_valid_auc_mean": payload["best_valid_auc"]["mean"],
                        "best_valid_auc_std": payload["best_valid_auc"]["std"],
                        "test_auc_mean": payload["test_auc"]["mean"],
                        "test_auc_std": payload["test_auc"]["std"],
                        "test_f1_macro_mean": payload["test_f1_macro"]["mean"],
                        "test_f1_macro_std": payload["test_f1_macro"]["std"],
                        "test_recall_mean": payload["test_recall"]["mean"],
                        "test_recall_std": payload["test_recall"]["std"],
                    }
                )
            aggregate[dataset_name][experiment_name] = fraction_payload
            if len(ordered_fractions) >= 2:
                high_fraction = ordered_fractions[-1]
                low_fraction = ordered_fractions[0]
                high_payload = fraction_payload[f"{high_fraction:.4f}"]
                low_payload = fraction_payload[f"{low_fraction:.4f}"]
                auc_drop = high_payload["test_auc"]["mean"] - low_payload["test_auc"]["mean"]
                f1_drop = high_payload["test_f1_macro"]["mean"] - low_payload["test_f1_macro"]["mean"]
                denom = max(high_fraction - low_fraction, 1e-12)
                drop_rows.append(
                    {
                        "dataset": dataset_name,
                        "experiment": experiment_name,
                        "high_fraction": high_fraction,
                        "low_fraction": low_fraction,
                        "test_auc_drop": auc_drop,
                        "test_f1_macro_drop": f1_drop,
                        "normalized_auc_drop_per_fraction": auc_drop / denom,
                        "normalized_f1_drop_per_fraction": f1_drop / denom,
                    }
                )
    return aggregate, curve_rows, drop_rows


def _aggregate_fusion(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for dataset_name in sorted({item["dataset"] for item in records}):
        dataset_records = [item for item in records if item["dataset"] == dataset_name]
        aggregate[dataset_name] = {}
        for variant_name in SUPPORTED_FUSION_VARIANTS:
            group = [item for item in dataset_records if item["variant"] == variant_name]
            if not group:
                continue
            payload = {
                "num_runs": len(group),
                "seeds": [int(item["seed"]) for item in group],
                "best_valid_auc": _aggregate_metric([item["best_valid_auc"] for item in group]),
                "test_auc": _aggregate_metric([item["test_auc"] for item in group]),
                "test_f1_macro": _aggregate_metric([item["test_f1_macro"] for item in group]),
                "test_recall": _aggregate_metric([item["test_recall"] for item in group]),
                "summary_paths": [item["summary_path"] for item in group],
            }
            aggregate[dataset_name][variant_name] = payload
            rows.append(
                {
                    "dataset": dataset_name,
                    "variant": variant_name,
                    "num_runs": len(group),
                    "seeds": ",".join(str(item["seed"]) for item in group),
                    "best_valid_auc_mean": payload["best_valid_auc"]["mean"],
                    "best_valid_auc_std": payload["best_valid_auc"]["std"],
                    "test_auc_mean": payload["test_auc"]["mean"],
                    "test_auc_std": payload["test_auc"]["std"],
                    "test_f1_macro_mean": payload["test_f1_macro"]["mean"],
                    "test_f1_macro_std": payload["test_f1_macro"]["std"],
                    "test_recall_mean": payload["test_recall"]["mean"],
                    "test_recall_std": payload["test_recall"]["std"],
                }
            )
    return aggregate, rows


def _pick_best_supervised_full(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best_records: dict[str, dict[str, Any]] = {}
    for dataset_name in sorted({item["dataset"] for item in records}):
        group = [item for item in records if item["dataset"] == dataset_name and item["experiment"] == "full"]
        if group:
            best_records[dataset_name] = max(group, key=lambda item: (item["best_valid_auc"], item["test_auc"]))
    return best_records


def _load_optional_json(path_str: str) -> dict[str, Any] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return _load_json(path)


def _build_diagnostics_rows(best_records: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for dataset_name, record in best_records.items():
        diagnostics_payload = _load_optional_json(record["diagnostics_path"])
        if diagnostics_payload is None:
            continue
        valid_payload = dict(diagnostics_payload.get("splits", {}).get("valid", {}) or {})
        test_payload = dict(diagnostics_payload.get("splits", {}).get("test", {}) or {})
        valid_branches = dict(valid_payload.get("branches", {}) or {})
        valid_stats = dict(valid_payload.get("stats", {}) or {})
        summary = dict(record.get("summary", {}) or {})
        output[dataset_name] = {
            "summary_path": record["summary_path"],
            "diagnostics_path": record["diagnostics_path"],
            "graph_gate_trend": dict(summary.get("graph_gate_trend", {}) or {}),
            "stage_schedule": list(summary.get("stage_schedule", []) or []),
            "valid": valid_payload,
            "test": test_payload,
        }
        rows.append(
            {
                "dataset": dataset_name,
                "seed": record["seed"],
                "best_valid_auc": record["best_valid_auc"],
                "test_auc": record["test_auc"],
                "valid_main_auc": _safe_float(valid_branches.get("main", {}).get("auc")),
                "valid_graph_residual_auc": _safe_float(valid_branches.get("graph_residual", {}).get("auc")),
                "valid_sequence_residual_auc": _safe_float(valid_branches.get("sequence_residual", {}).get("auc")),
                "valid_fusion_auc": _safe_float(valid_branches.get("fusion", {}).get("auc")),
                "valid_graph_gate_mean": _safe_float(valid_stats.get("graph_branch_gate_mean")),
                "valid_sequence_gate_mean": _safe_float(valid_stats.get("sequence_branch_gate_mean")),
                "valid_fusion_delta_gate_mean": _safe_float(valid_stats.get("fusion_delta_gate_mean")),
                "valid_shared_gap_mean": _safe_float(valid_stats.get("shared_gap_mean")),
                "valid_private_interaction_mean": _safe_float(valid_stats.get("private_interaction_mean")),
                "valid_graph_embedding_norm_mean": _safe_float(valid_stats.get("graph_embedding_norm_mean")),
                "valid_sequence_embedding_norm_mean": _safe_float(valid_stats.get("sequence_embedding_norm_mean")),
                "valid_sequence_token_valid_ratio_mean": _safe_float(valid_stats.get("sequence_token_valid_ratio_mean")),
                "valid_graph_sequence_prob_gap_mean": _safe_float(valid_stats.get("graph_sequence_prob_gap_mean")),
                "graph_gate_trend_delta": _safe_float(summary.get("graph_gate_trend", {}).get("delta")),
            }
        )
    return output, rows


def _export_curves(best_records: dict[str, dict[str, Any]], output_root: Path) -> dict[str, str]:
    curve_paths: dict[str, str] = {}
    for dataset_name, record in best_records.items():
        history = list(record.get("history", []) or [])
        if not history:
            continue
        rows = []
        for item in history:
            rows.append(
                {
                    "round": _safe_int(item.get("round")),
                    "training_stage": str(item.get("training_stage", "")),
                    "valid_auc": _safe_float(item.get("valid_auc")),
                    "valid_main_auc": _safe_float(item.get("valid_main_auc")),
                    "valid_fusion_auc": _safe_float(item.get("valid_fusion_auc")),
                    "valid_graph_residual_auc": _safe_float(item.get("valid_graph_residual_auc")),
                    "valid_sequence_residual_auc": _safe_float(item.get("valid_sequence_residual_auc")),
                    "valid_graph_branch_gate_mean": _safe_float(item.get("valid_graph_branch_gate_mean")),
                    "valid_sequence_branch_gate_mean": _safe_float(item.get("valid_sequence_branch_gate_mean")),
                    "valid_fusion_delta_gate_mean": _safe_float(item.get("valid_fusion_delta_gate_mean")),
                    "valid_shared_gap_mean": _safe_float(item.get("valid_shared_gap_mean")),
                    "valid_private_interaction_mean": _safe_float(item.get("valid_private_interaction_mean")),
                    "valid_graph_embedding_norm_mean": _safe_float(item.get("valid_graph_embedding_norm_mean")),
                    "valid_sequence_embedding_norm_mean": _safe_float(item.get("valid_sequence_embedding_norm_mean")),
                    "valid_sequence_token_valid_ratio_mean": _safe_float(item.get("valid_sequence_token_valid_ratio_mean")),
                    "valid_graph_sequence_prob_gap_mean": _safe_float(item.get("valid_graph_sequence_prob_gap_mean")),
                }
            )
        curve_path = output_root / "curves" / f"{dataset_name}_gate_branch_curve.csv"
        _write_csv(curve_path, rows, list(rows[0].keys()))
        curve_paths[dataset_name] = str(curve_path)
    return curve_paths


def _case_rows_from_masks(dataset_name: str, summary: dict[str, Any], probs: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    import numpy as np

    from fraud_ml_engineering.fraud_dataset import load_splitgnn_dataset

    bundle = load_splitgnn_dataset(
        dataset_name=dataset_name,
        data_dir=str(SPLITGNN_DATA_DIR),
        num_clients=1,
        seed=int(summary.get("seed", 30)),
        client_hops=1,
        label_fraction=float(summary.get("label_fraction", 1.0)),
        active_learning_feedback_path=str(summary.get("active_learning_feedback_path", "")),
    )
    graph = bundle.graph
    test_mask = graph.ndata["test_mask"].bool().cpu().numpy()
    labels = graph.ndata["label"].cpu().numpy().astype(np.int32)[test_mask]
    node_ids = np.nonzero(test_mask)[0]
    if len(node_ids) != len(probs) or len(labels) != len(preds):
        raise ValueError(
            f"Case-study size mismatch for {dataset_name}: nodes={len(node_ids)} probs={len(probs)} preds={len(preds)}"
        )
    return {"node_ids": node_ids, "labels": labels}


def _build_case_studies(best_records: dict[str, dict[str, Any]], output_root: Path, topk: int) -> dict[str, str]:
    import numpy as np

    case_paths: dict[str, str] = {}
    for dataset_name, record in best_records.items():
        summary_path = Path(record["summary_path"])
        probs_path = summary_path.parent / f"{dataset_name}_hybrid_fraudgraph_result_probs.npy"
        preds_path = summary_path.parent / f"{dataset_name}_hybrid_fraudgraph_result_preds.npy"
        if not probs_path.exists() or not preds_path.exists():
            continue
        probs = np.load(probs_path)
        preds = np.load(preds_path)
        mask_payload = _case_rows_from_masks(dataset_name, record["summary"], probs=probs, preds=preds)
        examples = []
        for node_id, label, pred, prob in zip(
            mask_payload["node_ids"].tolist(),
            mask_payload["labels"].tolist(),
            preds.tolist(),
            probs.tolist(),
        ):
            examples.append(
                {
                    "node_id": int(node_id),
                    "label": int(label),
                    "prediction": int(pred),
                    "positive_probability": float(prob),
                    "negative_probability": float(1.0 - prob),
                    "confidence": float(max(prob, 1.0 - prob)),
                }
            )

        def select_rows(predicate: Any, *, reverse: bool, sort_key: str) -> list[dict[str, Any]]:
            rows = [item for item in examples if predicate(item)]
            rows.sort(key=lambda item: item[sort_key], reverse=reverse)
            return rows[:topk]

        case_payload = {
            "dataset": dataset_name,
            "summary_path": record["summary_path"],
            "diagnostics_path": record["diagnostics_path"],
            "counts": {
                "num_test_examples": len(examples),
                "num_predicted_positive": int(sum(1 for item in examples if item["prediction"] == 1)),
                "num_true_positive_labels": int(sum(1 for item in examples if item["label"] == 1)),
            },
            "top_true_positives": select_rows(lambda item: item["label"] == 1 and item["prediction"] == 1, reverse=True, sort_key="positive_probability"),
            "top_false_positives": select_rows(lambda item: item["label"] == 0 and item["prediction"] == 1, reverse=True, sort_key="positive_probability"),
            "top_false_negatives": select_rows(lambda item: item["label"] == 1 and item["prediction"] == 0, reverse=False, sort_key="positive_probability"),
            "top_true_negatives": select_rows(lambda item: item["label"] == 0 and item["prediction"] == 0, reverse=False, sort_key="positive_probability"),
        }
        case_path = output_root / "case_studies" / f"{dataset_name}_case_study.json"
        _dump_json(case_path, case_payload)
        case_paths[dataset_name] = str(case_path)
    return case_paths


def _render_markdown(
    *,
    supervised_aggregate: dict[str, Any],
    low_label_aggregate: dict[str, Any],
    fusion_aggregate: dict[str, Any],
    diagnostics_rows: list[dict[str, Any]],
    curve_paths: dict[str, str],
    case_paths: dict[str, str],
) -> str:
    lines = ["# SplitGNN Hybrid Paper Package", "", "## Supervised Structure Matrix", ""]
    for dataset_name in SUPPORTED_DATASETS:
        dataset_payload = supervised_aggregate.get(dataset_name)
        if not dataset_payload:
            continue
        lines.append(f"### {dataset_name}")
        lines.append("")
        lines.append("| experiment | best_valid_auc | test_auc | test_f1_macro | test_recall |")
        lines.append("| --- | --- | --- | --- | --- |")
        for experiment_name in SUPPORTED_SUPERVISED_EXPERIMENTS:
            payload = dataset_payload.get(experiment_name)
            if payload is None:
                continue
            lines.append(
                f"| {experiment_name} | {payload['best_valid_auc']['mean']:.4f} +/- {payload['best_valid_auc']['std']:.4f} | "
                f"{payload['test_auc']['mean']:.4f} +/- {payload['test_auc']['std']:.4f} | "
                f"{payload['test_f1_macro']['mean']:.4f} +/- {payload['test_f1_macro']['std']:.4f} | "
                f"{payload['test_recall']['mean']:.4f} +/- {payload['test_recall']['std']:.4f} |"
            )
        lines.append("")
    lines.extend(["## Low-Label Matrix", ""])
    for dataset_name in SUPPORTED_DATASETS:
        dataset_payload = low_label_aggregate.get(dataset_name)
        if not dataset_payload:
            continue
        lines.append(f"### {dataset_name}")
        lines.append("")
        lines.append("| experiment | label_fraction | test_auc | test_f1_macro | test_recall |")
        lines.append("| --- | --- | --- | --- | --- |")
        for experiment_name in SUPPORTED_LOW_LABEL_EXPERIMENTS:
            fraction_payload = dataset_payload.get(experiment_name)
            if fraction_payload is None:
                continue
            for fraction_key in sorted(fraction_payload, key=float, reverse=True):
                payload = fraction_payload[fraction_key]
                lines.append(
                    f"| {experiment_name} | {float(fraction_key):.2%} | {payload['test_auc']['mean']:.4f} +/- {payload['test_auc']['std']:.4f} | "
                    f"{payload['test_f1_macro']['mean']:.4f} +/- {payload['test_f1_macro']['std']:.4f} | "
                    f"{payload['test_recall']['mean']:.4f} +/- {payload['test_recall']['std']:.4f} |"
                )
        lines.append("")
    lines.extend(["## Fusion Ladder", ""])
    for dataset_name in SUPPORTED_DATASETS:
        dataset_payload = fusion_aggregate.get(dataset_name)
        if not dataset_payload:
            continue
        lines.append(f"### {dataset_name}")
        lines.append("")
        lines.append("| variant | best_valid_auc | test_auc | test_f1_macro | test_recall |")
        lines.append("| --- | --- | --- | --- | --- |")
        for variant_name in SUPPORTED_FUSION_VARIANTS:
            payload = dataset_payload.get(variant_name)
            if payload is None:
                continue
            lines.append(
                f"| {variant_name} | {payload['best_valid_auc']['mean']:.4f} +/- {payload['best_valid_auc']['std']:.4f} | "
                f"{payload['test_auc']['mean']:.4f} +/- {payload['test_auc']['std']:.4f} | "
                f"{payload['test_f1_macro']['mean']:.4f} +/- {payload['test_f1_macro']['std']:.4f} | "
                f"{payload['test_recall']['mean']:.4f} +/- {payload['test_recall']['std']:.4f} |"
            )
        lines.append("")
    if diagnostics_rows:
        lines.extend(["## Diagnostics Snapshot", "", "| dataset | valid_main_auc | valid_graph_residual_auc | valid_sequence_residual_auc | valid_graph_gate_mean | valid_sequence_gate_mean | valid_prob_gap |", "| --- | --- | --- | --- | --- | --- | --- |"])
        for row in diagnostics_rows:
            lines.append(
                f"| {row['dataset']} | {row['valid_main_auc']:.4f} | {row['valid_graph_residual_auc']:.4f} | {row['valid_sequence_residual_auc']:.4f} | "
                f"{row['valid_graph_gate_mean']:.4f} | {row['valid_sequence_gate_mean']:.4f} | {row['valid_graph_sequence_prob_gap_mean']:.4f} |"
            )
        lines.append("")
    if curve_paths:
        lines.extend(["## Curve Artifacts", ""])
        for dataset_name, curve_path in sorted(curve_paths.items()):
            lines.append(f"- {dataset_name}: `{curve_path}`")
        lines.append("")
    if case_paths:
        lines.extend(["## Case Study Artifacts", ""])
        for dataset_name, case_path in sorted(case_paths.items()):
            lines.append(f"- {dataset_name}: `{case_path}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = set(str(item) for item in args.datasets)
    supervised_records = _scan_supervised(datasets=datasets, allow_smoke=bool(args.allow_smoke))
    low_label_records = _scan_low_label(datasets=datasets, allow_smoke=bool(args.allow_smoke))
    fusion_records = _scan_fusion(datasets=datasets, allow_smoke=bool(args.allow_smoke))
    supervised_aggregate, supervised_rows = _aggregate_supervised(supervised_records)
    low_label_aggregate, low_label_curve_rows, low_label_drop_rows = _aggregate_low_label(low_label_records)
    fusion_aggregate, fusion_rows = _aggregate_fusion(fusion_records)
    best_supervised_full = _pick_best_supervised_full(supervised_records)
    diagnostics_payload, diagnostics_rows = _build_diagnostics_rows(best_supervised_full)
    curve_paths = _export_curves(best_supervised_full, output_root=output_root)
    case_paths = _build_case_studies(best_supervised_full, output_root=output_root, topk=int(args.case_topk))

    if supervised_rows:
        _write_csv(output_root / "supervised_structure_matrix.csv", supervised_rows, list(supervised_rows[0].keys()))
    if low_label_curve_rows:
        _write_csv(output_root / "low_label_curve.csv", low_label_curve_rows, list(low_label_curve_rows[0].keys()))
    if low_label_drop_rows:
        _write_csv(output_root / "low_label_drop_summary.csv", low_label_drop_rows, list(low_label_drop_rows[0].keys()))
    if fusion_rows:
        _write_csv(output_root / "fusion_ladder_matrix.csv", fusion_rows, list(fusion_rows[0].keys()))
    if diagnostics_rows:
        _write_csv(output_root / "fusion_diagnostics_snapshot.csv", diagnostics_rows, list(diagnostics_rows[0].keys()))

    package_payload = {
        "protocol_root": str(PAPER_PROTOCOL_ROOT),
        "datasets": sorted(datasets),
        "allow_smoke": bool(args.allow_smoke),
        "counts": {
            "supervised_runs": len(supervised_records),
            "low_label_runs": len(low_label_records),
            "fusion_runs": len(fusion_records),
        },
        "artifacts": {
            "curve_paths": curve_paths,
            "case_study_paths": case_paths,
        },
        "supervised_structure": supervised_aggregate,
        "low_label": low_label_aggregate,
        "fusion_ladder": fusion_aggregate,
        "diagnostics_snapshot": diagnostics_payload,
    }
    json_path = output_root / "hybrid_paper_package.json"
    md_path = output_root / "hybrid_paper_package.md"
    _dump_json(json_path, package_payload)
    md_path.write_text(
        _render_markdown(
            supervised_aggregate=supervised_aggregate,
            low_label_aggregate=low_label_aggregate,
            fusion_aggregate=fusion_aggregate,
            diagnostics_rows=diagnostics_rows,
            curve_paths=curve_paths,
            case_paths=case_paths,
        ),
        encoding="utf-8",
    )
    print(json.dumps(package_payload, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
