from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = REPO_ROOT / "artifacts" / "experiments" / "ieee_acceptance_matrix"
PEAK_RSS_PATTERN = re.compile(r"peak=(?P<peak>[0-9.]+)GiB")


@dataclass
class StageConfig:
    name: str
    rounds: int
    extra_args: list[str]
    description: str


STAGES: tuple[StageConfig, ...] = (
    StageConfig(
        name="build_cache_only",
        rounds=0,
        extra_args=["--ieee_build_cache_only", "--ieee_rebuild_cache"],
        description="Build or rebuild the full IEEE cache without training.",
    ),
    StageConfig(
        name="round_1",
        rounds=1,
        extra_args=["--federated_rounds", "1"],
        description="Smoke-test a single training round on the cached full IEEE graph.",
    ),
    StageConfig(
        name="round_4",
        rounds=4,
        extra_args=["--federated_rounds", "4"],
        description="Short multi-round validation pass on the cached full IEEE graph.",
    ),
    StageConfig(
        name="round_24",
        rounds=24,
        extra_args=["--federated_rounds", "24"],
        description="Full target training schedule on the cached full IEEE graph.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged IEEE acceptance matrix and summarize outcomes.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch main.py.")
    parser.add_argument("--result_root", type=str, default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ieee_data_root", type=str, default="")
    parser.add_argument(
        "--ieee_data_profile",
        type=str,
        default="light_v1",
        choices=["raw", "light_v1", "light_v2", "tabular_full", "custom"],
    )
    parser.add_argument(
        "--ieee_loader_view",
        type=str,
        default="hybrid",
        choices=["graph", "tabular", "sequence", "hybrid"],
    )
    parser.add_argument("--ieee_relation_profile", type=str, default="core", choices=["core", "extended"])
    parser.add_argument(
        "--ieee_feature_profile",
        type=str,
        default="typed_256",
        choices=["typed_full", "typed_256", "typed_160", "paper_pruned", "paper_v30"],
    )
    parser.add_argument("--ieee_history_len", type=int, default=6)
    parser.add_argument(
        "--ieee_sampling_profile",
        type=str,
        default="fraud_hardneg",
        choices=["chrono_full", "chrono_stratified", "fraud_hardneg", "normal_only_train"],
    )
    parser.add_argument("--ieee_max_transactions", type=int, default=None)
    parser.add_argument("--ieee_time_bins", type=int, default=24)
    parser.add_argument("--ieee_relation_window_neighbors", type=int, default=2)
    parser.add_argument("--ieee_train_ratio", type=float, default=0.70)
    parser.add_argument("--ieee_valid_ratio", type=float, default=0.15)
    parser.add_argument("--transformer_hidden_dim", type=int, default=64)
    parser.add_argument("--transformer_num_layers", type=int, default=1)
    parser.add_argument("--sequence_batch_chunk_size", type=int, default=1024)
    parser.add_argument("--event_batch_chunk_size", type=int, default=1024)
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "off"])
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--disable_tb", action="store_true")
    parser.add_argument("--lightweight_valid_eval", action="store_true")
    parser.add_argument("--skip_test_evaluation", action="store_true")
    parser.add_argument("--profile_ieee_full_gpu", action="store_true")
    parser.add_argument(
        "--ieee_full_compact_sequences",
        dest="ieee_full_compact_sequences",
        action="store_true",
        default=True,
        help="Enable IEEE full compact base features during staged runs.",
    )
    parser.add_argument(
        "--no_ieee_full_compact_sequences",
        dest="ieee_full_compact_sequences",
        action="store_false",
        help="Disable IEEE full compact base features during staged runs.",
    )
    parser.add_argument("--ieee_sequence_feature_dim", type=int, default=64)
    parser.add_argument("--ieee_event_feature_dim", type=int, default=64)
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to later stages even if an earlier stage fails.",
    )
    return parser.parse_args()


def _extract_json_payload(stdout_text: str) -> dict[str, Any] | None:
    candidate_starts = [match.start() for match in re.finditer(r"(?m)^\{", stdout_text)]
    for start_index in reversed(candidate_starts):
        candidate = stdout_text[start_index:].strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _peak_rss_from_stdout(stdout_text: str) -> float | None:
    peak_values = [float(match.group("peak")) for match in PEAK_RSS_PATTERN.finditer(stdout_text)]
    if not peak_values:
        return None
    return max(peak_values)


def _stage_command(args: argparse.Namespace, stage: StageConfig, stage_result_root: Path) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "fraud_ml_engineering",
        "--dataset",
        "ieee",
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--amp_dtype",
        str(args.amp_dtype),
        "--label_fraction",
        str(float(args.label_fraction)),
        "--transformer_hidden_dim",
        str(int(args.transformer_hidden_dim)),
        "--transformer_num_layers",
        str(int(args.transformer_num_layers)),
        "--sequence_batch_chunk_size",
        str(int(args.sequence_batch_chunk_size)),
        "--event_batch_chunk_size",
        str(int(args.event_batch_chunk_size)),
        "--ieee_data_profile",
        str(args.ieee_data_profile),
        "--ieee_loader_view",
        str(args.ieee_loader_view),
        "--ieee_relation_profile",
        str(args.ieee_relation_profile),
        "--ieee_feature_profile",
        str(args.ieee_feature_profile),
        "--ieee_history_len",
        str(int(args.ieee_history_len)),
        "--ieee_sampling_profile",
        str(args.ieee_sampling_profile),
        "--ieee_time_bins",
        str(int(args.ieee_time_bins)),
        "--ieee_relation_window_neighbors",
        str(int(args.ieee_relation_window_neighbors)),
        "--ieee_train_ratio",
        str(float(args.ieee_train_ratio)),
        "--ieee_valid_ratio",
        str(float(args.ieee_valid_ratio)),
        "--ieee_sequence_feature_dim",
        str(int(args.ieee_sequence_feature_dim)),
        "--ieee_event_feature_dim",
        str(int(args.ieee_event_feature_dim)),
        "--result_root",
        str(stage_result_root),
    ]
    if str(args.ieee_data_root).strip():
        command.extend(["--ieee_data_root", str(args.ieee_data_root).strip()])
    if args.ieee_max_transactions is not None:
        command.extend(["--ieee_max_transactions", str(int(args.ieee_max_transactions))])
    if bool(args.profile_ieee_full_gpu):
        command.append("--profile_ieee_full_gpu")
    if bool(args.lightweight_valid_eval):
        command.append("--lightweight_valid_eval")
    if bool(args.skip_test_evaluation):
        command.append("--skip_test_evaluation")
    if bool(args.disable_tb):
        command.append("--disable_tb")
    if bool(args.ieee_full_compact_sequences):
        command.append("--ieee_full_compact_sequences")
    else:
        command.append("--no_ieee_full_compact_sequences")
    command.extend(stage.extra_args)
    return command


def _load_stage_summary(stage_result_root: Path, stdout_text: str) -> dict[str, Any] | None:
    payload = _extract_json_payload(stdout_text)
    if isinstance(payload, dict) and "ieee" in payload and isinstance(payload["ieee"], dict):
        return dict(payload["ieee"])
    if isinstance(payload, dict) and "dataset" in payload:
        return payload

    ieee_dir = stage_result_root / "ieee"
    for candidate_name in ("ieee_cache_build_summary.json", "ieee_hybrid_summary.json"):
        candidate_path = ieee_dir / candidate_name
        if candidate_path.exists():
            try:
                candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                continue
            if isinstance(candidate_payload, dict) and "summary" in candidate_payload:
                return dict(candidate_payload["summary"])
            if isinstance(candidate_payload, dict):
                return candidate_payload
    return None


def _stage_record(
    stage: StageConfig,
    *,
    command: list[str],
    return_code: int,
    stdout_text: str,
    stderr_text: str,
    stage_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    data_summary = dict(stage_summary.get("data_summary", {}) or {}) if stage_summary else {}
    sequence_quality = dict(data_summary.get("sequence_quality", {}) or {})
    resource_guard = dict(stage_summary.get("resource_guard", {}) or {}) if stage_summary else {}
    return {
        "stage": stage.name,
        "description": stage.description,
        "rounds": int(stage.rounds),
        "command": command,
        "return_code": int(return_code),
        "completed": bool(stage_summary.get("completed", False)) if stage_summary else False,
        "status": str(stage_summary.get("status", "completed" if return_code == 0 else "failed")) if stage_summary else ("failed" if return_code != 0 else "unknown"),
        "training_skipped": bool(stage_summary.get("training_skipped", False)) if stage_summary else False,
        "rounds_ran": int(stage_summary.get("rounds_ran", 0)) if stage_summary else 0,
        "peak_rss_gib_from_stdout": _peak_rss_from_stdout(stdout_text),
        "estimated_vram_gib": (
            float(resource_guard.get("estimated_vram_gib"))
            if resource_guard.get("estimated_vram_gib") is not None
            else None
        ),
        "sequence_storage_mode": str(sequence_quality.get("storage_mode", "")),
        "sequence_length": int(sequence_quality.get("sequence_length", 0) or 0),
        "sequence_feature_dim": int(sequence_quality.get("sequence_feature_dim", 0) or 0),
        "base_feature_dim": int(sequence_quality.get("base_feature_dim", 0) or 0),
        "selected_relation_count": int(sequence_quality.get("selected_relation_count", 0) or 0),
        "relation_order": list(sequence_quality.get("relation_order", []) or []),
        "best_valid_auc": stage_summary.get("best_valid_auc") if stage_summary else None,
        "test_auc": stage_summary.get("test_auc") if stage_summary else None,
        "summary_path": str(stage_summary.get("summary_path", "")) if stage_summary else "",
        "stdout_log": "",
        "stderr_log": "",
    }


def _render_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# IEEE Acceptance Matrix",
        "",
        "| Stage | Completed | Status | Rounds | Peak RSS GiB | Est. VRAM GiB | Seq Storage | Seq Len | Seq Dim | Relations | Best Valid AUC | Test AUC |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| {stage} | {completed} | {status} | {rounds_ran} | {peak_rss} | {estimated_vram} | {storage} | {seq_len} | {seq_dim} | {relations} | {best_valid_auc} | {test_auc} |".format(
                stage=record["stage"],
                completed="yes" if record["completed"] else "no",
                status=record["status"],
                rounds_ran=int(record["rounds_ran"]),
                peak_rss="n/a" if record["peak_rss_gib_from_stdout"] is None else f"{float(record['peak_rss_gib_from_stdout']):.2f}",
                estimated_vram="n/a" if record["estimated_vram_gib"] is None else f"{float(record['estimated_vram_gib']):.2f}",
                storage=record["sequence_storage_mode"] or "n/a",
                seq_len=int(record["sequence_length"]),
                seq_dim=int(record["sequence_feature_dim"]),
                relations=int(record["selected_relation_count"]),
                best_valid_auc="n/a" if record["best_valid_auc"] is None else f"{float(record['best_valid_auc']):.4f}",
                test_auc="n/a" if record["test_auc"] is None else f"{float(record['test_auc']):.4f}",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for stage in STAGES:
        stage_result_root = result_root / stage.name
        stage_result_root.mkdir(parents=True, exist_ok=True)
        command = _stage_command(args, stage, stage_result_root)
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout_path = stage_result_root / "stdout.log"
        stderr_path = stage_result_root / "stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        stage_summary = _load_stage_summary(stage_result_root, completed.stdout)
        record = _stage_record(
            stage,
            command=command,
            return_code=completed.returncode,
            stdout_text=completed.stdout,
            stderr_text=completed.stderr,
            stage_summary=stage_summary,
        )
        record["stdout_log"] = str(stdout_path)
        record["stderr_log"] = str(stderr_path)
        records.append(record)
        if completed.returncode != 0 and not bool(args.continue_on_error):
            break

    matrix_payload = {
        "project_root": str(REPO_ROOT),
        "result_root": str(result_root),
        "python": str(args.python),
        "stages": [asdict(stage) for stage in STAGES],
        "records": records,
    }
    matrix_json_path = result_root / "ieee_acceptance_matrix.json"
    matrix_md_path = result_root / "ieee_acceptance_matrix.md"
    matrix_json_path.write_text(json.dumps(matrix_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    matrix_md_path.write_text(_render_markdown(records), encoding="utf-8")
    print(json.dumps({"acceptance_matrix_json": str(matrix_json_path), "acceptance_matrix_md": str(matrix_md_path), "records": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
