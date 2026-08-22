"""Run P07 train-outcome episodes with one fresh Isaac process per episode.

The persistent collection script can aggregate several episodes in one Isaac
process, but the installed IsaacLab build hangs on the second context.new_stage()
inside the old SimulationContext stop callback. This batch runner keeps the same
plan schema and manifest validation while moving the process boundary to one
P07 worker per fresh Isaac process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import canonical_sha256, write_json_atomic

PERSISTENT_SCRIPT = ROOT / "scripts" / "run_hm3d_p07_persistent_collection.py"
WORKER_SCRIPT = ROOT / "scripts" / "run_hm3d_p07_exploration_episode.py"
DEFAULT_PYTHON = r"C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe"


def _load_persistent_module() -> Any:
    spec = importlib.util.spec_from_file_location("hm3d_p07_outcome_batch_persistent", PERSISTENT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load persistent P07 collection module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_one(
    module: Any,
    run: Any,
    *,
    python: Path,
    log_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    if run.output_path.exists():
        if not resume:
            raise FileExistsError(
                f"worker output already exists; inspect it or pass --resume: {run.output_path}"
            )
        summary = module._validated_worker_summary(run)
        summary["exit_code"] = 0 if summary.get("completed") is True else 2
        summary["resumed_existing"] = True
        return summary

    command = [str(python), str(WORKER_SCRIPT), *run.runner_arguments, "--headless"]
    stdout_path = log_dir / f"{run.run_id}.stdout.log"
    stderr_path = log_dir / f"{run.run_id}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            timeout=1800,
            check=False,
        )

    if completed.returncode != 0:
        return {
            "run_id": run.run_id,
            "output_path": str(run.output_path),
            "status": f"WORKER_EXIT_{completed.returncode}",
            "completed": False,
            "completion_reasons": [f"worker_exit_code_{completed.returncode}"],
            "exit_code": completed.returncode,
            "resumed_existing": False,
        }

    try:
        summary = module._validated_worker_summary(run)
    except Exception as error:
        summary = {
            "run_id": run.run_id,
            "output_path": str(run.output_path),
            "status": "OUTPUT_VALIDATION_FAILED",
            "error": f"{type(error).__name__}: {error}",
            "completed": False,
            "completion_reasons": ["output_validation_failed"],
            "exit_code": 2,
            "resumed_existing": False,
        }
        return summary

    summary["exit_code"] = 0 if summary.get("completed") is True else 2
    summary["resumed_existing"] = False
    return summary


def _build_manifest(
    *,
    module: Any,
    plan_path: Path,
    runs: tuple[Any, ...],
    rows: list[dict[str, Any]],
    started: float,
    max_parallel: int,
) -> dict[str, Any]:
    failed = sum(int(row.get("completed") is not True) for row in rows)
    complete_workers = sum(row.get("completed") is True for row in rows)
    all_runs_present = len(rows) == len(runs)
    all_workers_clean = all_runs_present and all(
        row.get("completed") is True for row in rows
    )
    status = (
        module.COLLECTION_COMPLETE_STATUS
        if failed == 0 and all_runs_present and all_workers_clean
        else module.COLLECTION_HAS_FAILURES_STATUS
    )
    manifest = {
        "schema_version": "hm3d-p07-outcome-batch-v1",
        "status": status,
        "claim_limit": (
            "Development real P07 outcome batch. Every episode is a fresh Isaac process. "
            "Each indexed transition remains bound to its original CF2X outcome."
        ),
        "plan_sha256": canonical_sha256(json.loads(plan_path.read_text(encoding="utf-8"))),
        "isaac_process_count": len(rows),
        "process_boundary_policy": "one_fresh_isaac_process_per_episode",
        "lifecycle_note": (
            "Fresh Isaac process per episode avoids the in-process context.new_stage "
            "stop-callback hang recorded in debug/findings.md iteration 7."
        ),
        "episode_count": len(rows),
        "planned_episode_count": len(runs),
        "aborted_episode_count": len(runs) - len(rows),
        "failed_episode_count": failed,
        "clean_worker_count": complete_workers,
        "max_parallel": max_parallel,
        "completion_audit": {
            "all_planned_runs_present": all_runs_present,
            "all_workers_clean": all_workers_clean,
            "requires": (
                "final=true, failed_episode_count=0, every planned run present, "
                "and every worker summary completed=true"
            ),
        },
        "real_decision_count": sum(int(row.get("transition_count", 0)) for row in rows),
        "total_physics_s": sum(float(row.get("elapsed_physics_s", 0.0)) for row in rows),
        "total_worker_wall_s": sum(float(row.get("wall_s", 0.0)) for row in rows),
        "batch_wall_s": time.perf_counter() - started,
        "coverage_audit": module._coverage_audit(rows),
        "runs": rows,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(DEFAULT_PYTHON))
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be positive")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)

    module = _load_persistent_module()
    runs = module._load_plan(args.plan_json, allow_cross_scene_process=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(
                _run_one,
                module,
                run,
                python=args.python,
                log_dir=args.log_dir,
                resume=args.resume,
            ): run
            for run in runs
        }
        for future in as_completed(futures):
            run = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {
                    "run_id": run.run_id,
                    "output_path": str(run.output_path),
                    "status": "LAUNCH_FAILED",
                    "error": f"{type(error).__name__}: {error}",
                    "completed": False,
                    "completion_reasons": ["launch_failed"],
                    "exit_code": 2,
                    "resumed_existing": False,
                }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "run_id": run.run_id,
                        "status": row.get("status"),
                        "completed": row.get("completed"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    manifest = _build_manifest(
        module=module,
        plan_path=args.plan_json,
        runs=runs,
        rows=rows,
        started=started,
        max_parallel=args.max_parallel,
    )
    write_json_atomic(args.manifest_output, manifest)
    print(
        json.dumps(
            {"status": manifest["status"], "manifest": str(args.manifest_output)},
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == module.COLLECTION_COMPLETE_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
