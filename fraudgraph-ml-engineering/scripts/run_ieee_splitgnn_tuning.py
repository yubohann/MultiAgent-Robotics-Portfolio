from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_ml_engineering.paths import ARTIFACTS_ROOT
from fraud_ml_engineering.selection import VALIDATION_ONLY_POLICY_NAME, validation_only_rank

DEFAULT_RESULT_ROOT = ARTIFACTS_ROOT / "experiments" / "ieee_splitgnn_tuning"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "ieee_splitgnn_tuning.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune SplitGNN-Transformer on IEEE-CIS with typed candidate configs.")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--result_root", type=str, default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--force_rerun", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _summary_from_path(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    payload = load_json(summary_path)
    summary = dict(payload.get("summary", payload) or {})
    if str(summary.get("dataset", "")).lower() != "ieee":
        return None
    return summary


def _selection_rank(summary: dict[str, Any]) -> tuple[float, ...]:
    return validation_only_rank(
        summary,
        metric_order=(
            "best_valid_auc",
            "best_valid_pr_auc",
            "best_valid_recall_at_precision",
            "best_valid_f1_macro",
        ),
        completed_first=True,
        rounds_ran_key="rounds_ran",
        prefer_lower_rounds_ran=True,
    )


def _config_note(candidate: dict[str, Any]) -> str:
    note_parts = [
        f"rounds={int(candidate['federated_rounds'])}",
        f"local_epochs={int(candidate['local_epochs'])}",
        f"extra_local_epochs={int(candidate['extra_local_epochs'])}",
        f"edge_loss_weight={float(candidate['edge_loss_weight']):.2f}",
        f"hidden={int(candidate['transformer_hidden_dim'])}/{int(candidate['fusion_hidden_dim'])}",
        f"layers={int(candidate['transformer_num_layers'])}",
    ]
    if candidate.get("learning_rate_override") is not None:
        note_parts.append(f"lr={float(candidate['learning_rate_override']):.6f}")
    if candidate.get("weight_decay_override") is not None:
        note_parts.append(f"wd={float(candidate['weight_decay_override']):.6f}")
    if candidate.get("dropout_override") is not None:
        note_parts.append(f"dropout={float(candidate['dropout_override']):.2f}")
    if candidate.get("consistency_weight") is not None:
        note_parts.append(f"consistency={float(candidate['consistency_weight']):.2f}")
    return ", ".join(note_parts)


def _candidate_row(candidate: dict[str, Any], summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    test_metrics = dict(summary.get("test", {}) or {})
    return {
        "tag": str(candidate["tag"]),
        "note": str(candidate.get("note", "")),
        "config_note": _config_note(candidate),
        "summary_path": str(summary_path),
        "completed": bool(summary.get("completed", False)),
        "best_valid_auc": float(summary.get("best_valid_auc", 0.0)),
        "best_valid_pr_auc": float(summary.get("best_valid_pr_auc", 0.0)),
        "best_valid_recall_at_precision": float(summary.get("best_valid_recall_at_precision", 0.0)),
        "best_valid_f1_macro": float(summary.get("best_valid_f1_macro", 0.0)),
        "test_auc": float(test_metrics.get("auc", 0.0)),
        "test_f1_macro": float(test_metrics.get("f1_macro", 0.0)),
        "test_recall": float(test_metrics.get("recall", 0.0)),
        "rounds_ran": int(summary.get("rounds_ran", 0)),
        "best_round": int(summary.get("best_round", -1)),
        "learning_rate_override": candidate.get("learning_rate_override"),
        "weight_decay_override": candidate.get("weight_decay_override"),
        "dropout_override": candidate.get("dropout_override"),
        "consistency_weight": float(candidate.get("consistency_weight", 0.0)),
        "error": str(summary.get("error", "") or ""),
        "summary": summary,
    }


def candidate_markdown(candidate_rows: list[dict[str, Any]], best_tag: str, config_path: Path) -> str:
    lines = [
        "# IEEE SplitGNN-Transformer Tuning",
        "",
        f"- config_path: `{config_path}`",
        f"- best_tag: `{best_tag}`",
        f"- selection_rule: `{VALIDATION_ONLY_POLICY_NAME}`",
        "- selection_policy: validation metrics only; test metrics are reported but not used for ranking.",
        "",
        "| candidate | completed | best_valid_auc | best_valid_pr_auc | best_valid_recall@P>=0.50 | test_auc | test_f1_macro | test_recall | rounds_ran | config |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in candidate_rows:
        lines.append(
            f"| {item['tag']} | {'yes' if item['completed'] else 'no'} | "
            f"{item['best_valid_auc']:.6f} | {item['best_valid_pr_auc']:.6f} | "
            f"{item['best_valid_recall_at_precision']:.6f} | {item['test_auc']:.6f} | "
            f"{item['test_f1_macro']:.6f} | {item['test_recall']:.6f} | {item['rounds_ran']} | "
            f"{item['config_note']} |"
        )
        if item["error"]:
            lines.append(f"- `{item['tag']}` error: `{item['error']}`")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    from fraud_ml_engineering.experiment_config import load_candidate_list_config

    result_root = Path(args.result_root).expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).expanduser().resolve()
    config = load_candidate_list_config(config_path)
    candidates = [item.as_dict() for item in config.candidates]

    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        tag = str(candidate["tag"])
        candidate_root = result_root / tag
        summary_path = candidate_root / "ieee" / "ieee_hybrid_summary.json"
        if args.force_rerun and candidate_root.exists():
            shutil.rmtree(candidate_root, ignore_errors=True)

        summary = _summary_from_path(summary_path)
        should_reuse = bool(summary is not None and summary.get("completed", False) and not args.force_rerun)
        if should_reuse:
            print(f"[ieee-tune] reusing completed {tag}", flush=True)
        else:
            candidate_root.mkdir(parents=True, exist_ok=True)
            print(f"[ieee-tune] running {tag}", flush=True)
            try:
                from fraud_ml_engineering.algorithms import run_hybrid_fraud_training

                run_hybrid_fraud_training(
                    federated_rounds=int(candidate["federated_rounds"]),
                    local_epochs=int(candidate["local_epochs"]),
                    extra_local_epochs=int(candidate["extra_local_epochs"]),
                    edge_loss_weight=float(candidate["edge_loss_weight"]),
                    dataset="ieee",
                    num_clients=1,
                    client_hops=1,
                    label_fraction=1.0,
                    rl_timesteps=0,
                    device=str(args.device),
                    enable_tensorboard=False,
                    classification_loss="cb_focal",
                    focal_gamma=2.0,
                    class_balance_beta=0.999,
                    pseudo_label_threshold=0.9,
                    pseudo_label_weight=0.0,
                    pseudo_label_novelty_threshold=2.5,
                    consistency_weight=float(candidate.get("consistency_weight", 0.0)),
                    active_learning_budget_per_round=0,
                    active_learning_delay_rounds=0,
                    active_learning_novelty_weight=0.0,
                    active_learning_diversity_weight=0.0,
                    active_learning_candidate_pool_scale=1,
                    fedprox_mu=0.0,
                    dp_noise_std=0.0,
                    seq_hidden_dim=int(candidate["seq_hidden_dim"]),
                    fusion_hidden_dim=int(candidate["fusion_hidden_dim"]),
                    planner_mode="deterministic",
                    early_stop=10,
                    test_every=0,
                    fixed_precision_target=0.5,
                    transformer_hidden_dim=int(candidate["transformer_hidden_dim"]),
                    transformer_num_layers=int(candidate["transformer_num_layers"]),
                    active_learning_feedback_path="",
                    seed=int(args.seed),
                    result_root=str(candidate_root),
                    disable_gnn=False,
                    disable_transformer=False,
                    disable_federated=True,
                    learning_rate_override=candidate.get("learning_rate_override"),
                    weight_decay_override=candidate.get("weight_decay_override"),
                    dropout_override=candidate.get("dropout_override"),
                )
            except Exception as error:  # pragma: no cover - runtime safeguard
                error_text = "".join(traceback.format_exception_only(type(error), error)).strip()
                print(f"[ieee-tune] {tag} raised: {error_text}", flush=True)

            summary = _summary_from_path(summary_path)
            if summary is None:
                summary = {
                    "dataset": "ieee",
                    "completed": False,
                    "error": "Missing summary after run.",
                    "test": {},
                }

        row = _candidate_row(candidate=candidate, summary_path=summary_path, summary=summary)
        candidate_rows.append(row)
        print(
            f"[ieee-tune] {tag}: completed={row['completed']} "
            f"valid_auc={row['best_valid_auc']:.6f} test_auc={row['test_auc']:.6f}",
            flush=True,
        )

    ranked = sorted(candidate_rows, key=lambda item: _selection_rank(item["summary"]), reverse=True)
    completed_rows = [item for item in ranked if item["completed"]]
    if not completed_rows:
        raise RuntimeError("No completed IEEE tuning candidate was produced.")

    best = completed_rows[0]
    output_json = result_root / "ieee_splitgnn_tuning.json"
    output_md = result_root / "ieee_splitgnn_tuning.md"
    output_payload = {
        "best_tag": best["tag"],
        "best_summary_path": best["summary_path"],
        "selection_rule": config.selection_policy or VALIDATION_ONLY_POLICY_NAME,
        "test_policy": config.test_policy or "candidate_stage_reports_only",
        "config_path": str(config_path),
        "candidates": [{key: value for key, value in item.items() if key != "summary"} for item in ranked],
    }
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    output_md.write_text(candidate_markdown(ranked, best_tag=best["tag"], config_path=config_path), encoding="utf-8-sig")
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
