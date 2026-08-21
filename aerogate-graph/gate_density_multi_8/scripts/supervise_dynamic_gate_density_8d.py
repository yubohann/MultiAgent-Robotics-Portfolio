"""Supervisor for resumable 8-drone dynamic gate-density training jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PYTHON_EXE = Path(os.environ.get("PYTHON", sys.executable))
ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_dynamic_gate_density_8d_curriculum.py"
RESULTS_ROOT = ROOT / "results"
CAMPAIGN_ROOT = RESULTS_ROOT / "e2d_dynamic_gate_density_8d_curriculum_v26_scheduled_supervised"
JOBS_ROOT = CAMPAIGN_ROOT / "jobs"
STATUS_LOG = CAMPAIGN_ROOT / "supervisor_status.jsonl"
REPORT_MD = CAMPAIGN_ROOT / "training_job_summary.md"

INIT_LOW_DENSITY = (
    RESULTS_ROOT
    / "e2d_dynamic_gate_density_8d_curriculum_v14_continuation_full"
    / "stages"
    / "05_C2c_gate16_speed04.json"
)
INIT_MEDIUM_DENSITY = (
    RESULTS_ROOT
    / "e2d_dynamic_gate_density_8d_curriculum_v15_from_c3_full"
    / "stages"
    / "08_C4a_gate28_speed09.json"
)


@dataclass
class JobResult:
    label: str
    output_dir: Path
    returncode: int
    latest_stage_json: Path | None
    latest_summary: dict[str, object] | None


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time": _now(), **payload}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _stage_files(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "stages").glob("*.json")) if (output_dir / "stages").exists() else []


def _load_stage_summary(stage_json: Path | None) -> dict[str, object] | None:
    if stage_json is None or not stage_json.exists():
        return None
    payload = json.loads(stage_json.read_text(encoding="utf-8"))
    return payload.get("eval_summary")


def _latest_stage_json(output_dir: Path) -> Path | None:
    files = _stage_files(output_dir)
    return files[-1] if files else None


def _selected_params_json(job: JobResult, fallback: Path) -> Path:
    return job.latest_stage_json if job.latest_stage_json is not None else fallback


def _summary_value(summary: dict[str, object] | None, key: str, default: float = 0.0) -> float:
    if not summary:
        return default
    try:
        return float(summary.get(key, default))
    except (TypeError, ValueError):
        return default


def run_job(
    label: str,
    *,
    start_stage_index: int,
    initial_params_json: Path,
    candidates_per_stage: int,
    stage_limit: int | None = None,
    skip_formal: bool = True,
    seed: int = 20260511,
) -> JobResult:
    output_dir = JOBS_ROOT / label
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON_EXE),
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--candidates-per-stage",
        str(candidates_per_stage),
        "--start-stage-index",
        str(start_stage_index),
        "--initial-params-json",
        str(initial_params_json),
    ]
    if stage_limit is not None:
        cmd.extend(["--stage-limit", str(stage_limit)])
    if skip_formal:
        cmd.append("--skip-formal")

    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    _write_jsonl(
        STATUS_LOG,
        {
            "event": "job_start",
            "label": label,
            "cmd": cmd,
            "output_dir": str(output_dir),
        },
    )
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=stdout, stderr=stderr)
        last_stage_count = -1
        while proc.poll() is None:
            stage_count = len(_stage_files(output_dir))
            latest = _latest_stage_json(output_dir)
            summary = _load_stage_summary(latest)
            _write_jsonl(
                STATUS_LOG,
                {
                    "event": "job_progress" if stage_count != last_stage_count else "job_heartbeat",
                    "label": label,
                    "pid": proc.pid,
                    "elapsed_s": round(time.time() - start, 1),
                    "stage_count": stage_count,
                    "latest_stage": latest.name if latest else None,
                    "latest_summary": summary,
                },
            )
            last_stage_count = stage_count
            time.sleep(60)
        returncode = int(proc.returncode)

    latest = _latest_stage_json(output_dir)
    summary = _load_stage_summary(latest)
    _write_jsonl(
        STATUS_LOG,
        {
            "event": "job_done",
            "label": label,
            "returncode": returncode,
            "elapsed_s": round(time.time() - start, 1),
            "latest_stage": latest.name if latest else None,
            "latest_summary": summary,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    )
    return JobResult(label, output_dir, returncode, latest, summary)


def write_report(results: list[JobResult], formal_job: JobResult | None, selected_final_params: Path) -> None:
    lines = [
        "# Dynamic Gate-density 8-drone Training Summary",
        "",
        f"- 更新时间: {_now()}",
        f"- 运行目录: `{CAMPAIGN_ROOT}`",
        f"- 评估参数来源: `{selected_final_params}`",
        f"- 状态日志: `{STATUS_LOG}`",
        "",
        "## 阶段结果",
        "",
        "| job | latest stage | success | obstacle collision | agent collision | progress | note |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in results:
        summary = item.latest_summary or {}
        note = "ok" if item.returncode == 0 else f"returncode={item.returncode}"
        lines.append(
            "| "
            + " | ".join(
                [
                    item.label,
                    item.latest_stage_json.name if item.latest_stage_json else "-",
                    f"{_summary_value(summary, 'team_success_rate'):.3f}",
                    f"{_summary_value(summary, 'obstacle_collision_rate'):.3f}",
                    f"{_summary_value(summary, 'agent_agent_collision_rate'):.3f}",
                    f"{_summary_value(summary, 'progress_distance_mean_m'):.2f}",
                    note,
                ]
            )
            + " |"
        )
    if formal_job is not None:
        summary_path = formal_job.output_dir / "curriculum_summary.json"
        lines.extend(
            [
                "",
                "## Evaluation",
                "",
                f"- formal job: `{formal_job.output_dir}`",
                f"- summary: `{summary_path}`",
                f"- E2D-2 CSV: `{formal_job.output_dir / 'e2d2_dynamic_gate_density_risk.csv'}`",
                f"- E2D-3 CSV: `{formal_job.output_dir / 'e2d3_slow_fast_safe_ablation.csv'}`",
            ]
        )
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            e2d2 = payload.get("e2d2_dynamic_gate_density_risk", [])
            if e2d2:
                lines.extend(["", "### E2D-2 风险曲线", ""])
                lines.append("| gate_count | speed | success | obstacle collision | progress |")
                lines.append("|---:|---:|---:|---:|---:|")
                for row in e2d2:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                str(row.get("gate_count")),
                                str(row.get("moving_gate_speed_mps")),
                                f"{float(row.get('team_success_rate', 0.0)):.3f}",
                                f"{float(row.get('obstacle_collision_rate', 0.0)):.3f}",
                                f"{float(row.get('progress_distance_mean_m', 0.0)):.2f}",
                            ]
                        )
                        + " |"
                    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        STATUS_LOG,
        {
            "event": "supervisor_start",
            "python": str(PYTHON_EXE),
            "train_script": str(TRAIN_SCRIPT),
            "curriculum_gate_counts": [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60],
            "init_low_density_exists": INIT_LOW_DENSITY.exists(),
            "init_medium_density_exists": INIT_MEDIUM_DENSITY.exists(),
        },
    )

    results: list[JobResult] = []

    c5_probe = run_job(
        "01_c5_gate30_probe",
        start_stage_index=5,
        stage_limit=1,
        initial_params_json=INIT_MEDIUM_DENSITY,
        candidates_per_stage=4,
        skip_formal=True,
        seed=20260511,
    )
    results.append(c5_probe)
    best_c5 = c5_probe

    if _summary_value(c5_probe.latest_summary, "team_success_rate") < 0.20:
        c5_retune = run_job(
            "02_c5_gate30_retune",
            start_stage_index=5,
            stage_limit=1,
            initial_params_json=INIT_MEDIUM_DENSITY,
            candidates_per_stage=10,
            skip_formal=True,
            seed=20260512,
        )
        results.append(c5_retune)
        if _summary_value(c5_retune.latest_summary, "team_success_rate") > _summary_value(
            c5_probe.latest_summary, "team_success_rate"
        ):
            best_c5 = c5_retune

    final_params = INIT_MEDIUM_DENSITY
    if _summary_value(best_c5.latest_summary, "team_success_rate") >= 0.20:
        final_params = _selected_params_json(best_c5, INIT_MEDIUM_DENSITY)
        c6_c10 = run_job(
            "03_c6_to_c10_pressure_continuation",
            start_stage_index=6,
            initial_params_json=final_params,
            candidates_per_stage=6,
            skip_formal=True,
            seed=20260513,
        )
        results.append(c6_c10)
        final_params = _selected_params_json(c6_c10, final_params)
    else:
        _write_jsonl(
            STATUS_LOG,
            {
                "event": "pressure_too_hard",
                "reason": "C5 gate30 below 0.20 success after probe/retune; keep medium-density policy and use higher density as pressure evaluation.",
                "best_c5_success": _summary_value(best_c5.latest_summary, "team_success_rate"),
            },
        )

    formal = run_job(
        "04_formal_e2d2_e2d3",
        start_stage_index=11,
        initial_params_json=final_params,
        candidates_per_stage=1,
        skip_formal=False,
        seed=20260514,
    )
    results.append(formal)
    write_report(results, formal, final_params)
    _write_jsonl(STATUS_LOG, {"event": "supervisor_done", "report": str(REPORT_MD)})


if __name__ == "__main__":
    main()

