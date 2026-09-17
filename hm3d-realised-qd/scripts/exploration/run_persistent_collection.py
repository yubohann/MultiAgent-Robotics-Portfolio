"""Run same-scene Exploration episodes under one Isaac process."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from realised_qd.contracts.hm3d_public_schema import (  # noqa: E402
    PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
    require_current_public_schema,
)
from realised_qd.contracts.io import (  # noqa: E402
    require_identifier,
    write_json_atomic,
)

PLAN_SCHEMA_VERSION = "hm3d-persistent-collection-plan-v1"
RESULT_SCHEMA_VERSION = "hm3d-persistent-collection-v1"
EXPLORATION_EXECUTION_SMOKE_COMPLETE_STATUS = "EXPLORATION_EXECUTION_SMOKE_COMPLETE"
EXPLORATION_EXECUTION_SMOKE_FAILED_STATUS = "EXPLORATION_EXECUTION_SMOKE_FAILED"
COLLECTION_IN_PROGRESS_STATUS = "COLLECTION_IN_PROGRESS"
COLLECTION_COMPLETE_STATUS = "COLLECTION_COMPLETE"
COLLECTION_HAS_FAILURES_STATUS = "COLLECTION_HAS_FAILURES"
_FORBIDDEN_WORKER_ARGUMENTS = {
    "--headless",
    "--device",
    "--enable_cameras",
    "--experience",
}
_RETIRED_FIXED_CADENCE_ARGUMENTS = {"--decision-count"}


def _argument_value(arguments: tuple[str, ...], name: str) -> str | None:
    """Return one option value without importing the worker's Isaac parser."""

    positions = [index for index, value in enumerate(arguments) if value == name]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"persistent collection run must declare exactly one {name}")
    value = arguments[positions[0] + 1]
    if not value or value.startswith("-"):
        raise ValueError(f"persistent collection {name} value is missing")
    return value


@dataclass(frozen=True, slots=True)
class CollectionRun:
    run_id: str
    runner_arguments: tuple[str, ...]
    output_path: Path


def _load_worker_module() -> Any:
    path = ROOT / "scripts" / "exploration" / "run_exploration_episode.py"
    module_name = "hm3d_exploration_persistent_worker"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Exploration worker module")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec_module; dataclasses._is_type resolves the
    # class module and can otherwise raise AttributeError.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_plan(
    path: Path,
    *,
    allow_cross_scene_process: bool = False,
) -> tuple[CollectionRun, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("persistent collection plan schema mismatch")
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("persistent collection plan requires at least one run")
    runs: list[CollectionRun] = []
    for index, raw in enumerate(raw_runs):
        if not isinstance(raw, dict) or set(raw) != {"run_id", "runner_arguments"}:
            raise ValueError(f"runs[{index}] fields are invalid")
        run_id = require_identifier(raw["run_id"], f"runs[{index}].run_id")
        values = raw["runner_arguments"]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"runs[{index}].runner_arguments must be a string list")
        arguments = tuple(values)
        if any(value in _FORBIDDEN_WORKER_ARGUMENTS for value in arguments):
            raise ValueError("worker runs cannot override the shared Isaac application")
        retired = sorted(set(arguments).intersection(_RETIRED_FIXED_CADENCE_ARGUMENTS))
        if retired:
            raise ValueError(
                "persistent collection accepts only event-driven Exploration plans; "
                f"retired fixed-cadence argument(s): {', '.join(retired)}"
            )
        if arguments.count("--output") != 1:
            raise ValueError("each worker run must declare exactly one --output")
        output_index = arguments.index("--output")
        if output_index + 1 >= len(arguments):
            raise ValueError("worker --output lacks a path")
        runs.append(CollectionRun(run_id, arguments, Path(arguments[output_index + 1]).resolve()))
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("persistent collection run_id values must be unique")
    if len({run.output_path for run in runs}) != len(runs):
        raise ValueError("persistent collection outputs must be unique")
    scene_ids = {
        scene_id
        for run in runs
        if (scene_id := _argument_value(run.runner_arguments, "--scene-id")) is not None
    }
    if len(scene_ids) > 1 and not allow_cross_scene_process:
        raise ValueError(
            "persistent collection refuses multiple scene_id values in one Isaac process; "
            "launch one collection process per scene, or pass --allow-cross-scene-process "
            "only for an explicitly diagnosed engineering run"
        )
    return tuple(runs)


def _worker_completion_audit(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether a worker is a clean, budget-exhausted execution."""

    reasons: list[str] = []
    if payload.get("status") != EXPLORATION_EXECUTION_SMOKE_COMPLETE_STATUS:
        reasons.append("worker_status_not_complete")
    if payload.get("terminal_outcome") != "budget_exhausted":
        reasons.append("terminal_outcome_not_budget_exhausted")
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        reasons.append("execution_summary_missing")
    else:
        failed_fragments = execution.get("failed_fragment_count")
        if not isinstance(failed_fragments, int) or isinstance(failed_fragments, bool):
            reasons.append("failed_fragment_count_missing")
        elif failed_fragments != 0:
            reasons.append("failed_fragments_present")
    return not reasons, reasons


def _validated_worker_summary(run: CollectionRun) -> dict[str, Any]:
    payload = json.loads(run.output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Exploration worker output must be an object")
    runtime_record_id = require_identifier(payload.get("runtime_record_id"), "runtime record id")
    require_current_public_schema(payload, context="persistent-collection worker output")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("persistent-collection worker output lacks decision records")
    for index, decision in enumerate(decisions):
        require_current_public_schema(
            decision, context=f"persistent-collection decision[{index}]"
        )
    task_reservation = payload.get("task_reservation")
    if not isinstance(task_reservation, dict):
        raise ValueError("persistent-collection worker output lacks task_reservation")
    if task_reservation.get("schema_version") != PUBLIC_TASK_RESERVATION_SCHEMA_VERSION:
        raise ValueError("persistent-collection task_reservation schema mismatch")
    # MARVEL is a fixed-altitude source reference, so its legacy transition family
    # never enters a production collection manifest.
    for retired_key in (
        "marvel_training_transitions",
        "marvel_supplementary_reference_training_transitions",
    ):
        if retired_key in payload:
            raise ValueError(
                f"persistent-collection rejects retired MARVEL transition family ({retired_key}); "
                "use marl_ipp_training_transitions for the external learning transfer"
            )
    # A manifest may summarize engineering-only workers but never index a transition
    # whose public-task schema differs from its executed decision.
    for transition_key in (
        "single_rl_training_transitions",
        "marl_ipp_training_transitions",
    ):
        transitions = payload.get(transition_key)
        if transitions is None:
            continue
        if not isinstance(transitions, list):
            raise ValueError(f"persistent-collection {transition_key} must be a list")
        for index, transition in enumerate(transitions):
            require_current_public_schema(
                transition,
                context=f"persistent-collection {transition_key}[{index}]",
            )
    transitions = payload.get("single_rl_training_transitions")
    decision_count = len(decisions) if isinstance(decisions, list) else 0
    transition_count = len(transitions) if isinstance(transitions, list) else 0
    completed, completion_reasons = _worker_completion_audit(payload)
    return {
        "run_id": run.run_id,
        "output_path": str(run.output_path),
        "runtime_record_id": runtime_record_id,
        "status": payload.get("status"),
        "completed": completed,
        "completion_reasons": completion_reasons,
        "terminal_outcome": payload.get("terminal_outcome"),
        "scene_id": payload.get("scene_id"),
        "strategy": payload.get("strategy"),
        "decision_count": decision_count,
        "transition_count": transition_count,
        "elapsed_physics_s": payload.get("elapsed_physics_s", 0.0),
        "wall_s": payload.get("runtime_performance", {}).get("total_wall_s", 0.0),
        "auc": payload.get("metric_report", {}).get(
            "explored_free_flight_volume_auc_time", 0.0
        ),
        "final_coverage": payload.get("metric_report", {}).get(
            "final_coverage_at_budget", 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse immutable completed worker outputs and continue remaining atomic runs",
    )
    parser.add_argument(
        "--fresh-process",
        action="store_true",
        help="run every worker in a fresh Isaac subprocess",
    )
    parser.add_argument(
        "--allow-cross-scene-process",
        action="store_true",
        help=(
            "engineering-only escape hatch for diagnosing shared Isaac stage reuse; "
            "formal collection must use one scene per Isaac process"
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _coverage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize actual scene/strategy support without treating episodes as scenes."""

    scene_episodes: dict[str, int] = {}
    scene_transitions: dict[str, int] = {}
    strategy_episodes: dict[str, int] = {}
    strategy_transitions: dict[str, int] = {}
    scene_strategy_episodes: dict[str, int] = {}
    for row in rows:
        scene_id = row.get("scene_id")
        strategy = row.get("strategy")
        if not isinstance(scene_id, str) or not scene_id:
            continue
        if not isinstance(strategy, str) or not strategy:
            continue
        transitions = int(row.get("transition_count", 0))
        scene_episodes[scene_id] = scene_episodes.get(scene_id, 0) + 1
        scene_transitions[scene_id] = scene_transitions.get(scene_id, 0) + transitions
        strategy_episodes[strategy] = strategy_episodes.get(strategy, 0) + 1
        strategy_transitions[strategy] = strategy_transitions.get(strategy, 0) + transitions
        joint_key = f"{scene_id}::{strategy}"
        scene_strategy_episodes[joint_key] = scene_strategy_episodes.get(joint_key, 0) + 1
    return {
        "independent_scene_count": len(scene_episodes),
        "scene_episode_counts": dict(sorted(scene_episodes.items())),
        "scene_transition_counts": dict(sorted(scene_transitions.items())),
        "strategy_episode_counts": dict(sorted(strategy_episodes.items())),
        "strategy_transition_counts": dict(sorted(strategy_transitions.items())),
        "scene_strategy_episode_counts": dict(sorted(scene_strategy_episodes.items())),
        "claim_limit": (
            "Episodes, start seeds and vectorized clusters from one scene are repeated measures, "
            "not additional independent scenes."
        ),
    }


def _build_manifest(
    *,
    plan_file_id: str,
    runs: tuple[CollectionRun, ...],
    rows: list[dict[str, Any]],
    failed: int,
    started: float,
    final: bool,
    allow_cross_scene_process: bool,
    isaac_process_count: int = 1,
) -> dict[str, Any]:
    if failed < 0:
        raise ValueError("collection failure count cannot be negative")
    # A missing worker output or malformed summary is itself a failed collection and
    # stays visible to resume and audit tooling.
    complete_workers = sum(row.get("completed") is True for row in rows)
    all_runs_present = len(rows) == len(runs)
    all_workers_clean = all_runs_present and all(
        row.get("completed") is True for row in rows
    )
    status = COLLECTION_IN_PROGRESS_STATUS
    if final:
        status = (
            COLLECTION_COMPLETE_STATUS
            if failed == 0 and all_runs_present and all_workers_clean
            else COLLECTION_HAS_FAILURES_STATUS
        )
    planned_scene_ids = sorted(
        {
            scene_id
            for run in runs
            if (scene_id := _argument_value(run.runner_arguments, "--scene-id")) is not None
        }
    )
    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "claim_limit": (
            "Development real Exploration collection manifest. Gradient updates are not physical "
            "interactions; every indexed transition remains bound to its original CF2X outcome."
        ),
        "plan_file_id": plan_file_id,
        "isaac_process_count": isaac_process_count,
        "planned_scene_ids": planned_scene_ids,
        "cross_scene_process_allowed": allow_cross_scene_process,
        "process_boundary_policy": (
            "engineering_cross_scene_override"
            if allow_cross_scene_process
            else "one_scene_per_isaac_process"
        ),
        "episode_count": len(rows),
        "planned_episode_count": len(runs),
        "aborted_episode_count": len(runs) - len(rows),
        "failed_episode_count": failed,
        "clean_worker_count": complete_workers,
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
        "coverage_audit": _coverage_audit(rows),
        "runs": rows,
    }
    manifest["manifest_id"] = (
        f"exploration-collection:{len(rows)}-episodes:{len(runs)}-planned"
    )
    return manifest


def _run_worker_subprocess(run: CollectionRun, device: str) -> int:
    import subprocess

    command = [
        sys.executable,
        str(ROOT / "scripts" / "exploration" / "run_exploration_episode.py"),
    ]
    command += ["--headless"]
    if "--device" not in run.runner_arguments:
        command += ["--device", device]
    command += list(run.runner_arguments)
    print(
        json.dumps(
            {"status": "FRESH_PROCESS_RUN_STARTED", "run_id": run.run_id}, sort_keys=True
        ),
        flush=True,
    )
    result = subprocess.run(command, cwd=ROOT, check=False)
    return result.returncode


def main(args: argparse.Namespace, simulation_app: Any) -> int:
    worker = _load_worker_module()
    plan_path = args.plan_json.expanduser().resolve()
    manifest_output = args.manifest_output.expanduser().resolve()
    runs = _load_plan(
        plan_path,
        allow_cross_scene_process=bool(args.allow_cross_scene_process),
    )
    plan_file_id = f"{plan_path.name}:{plan_path.stat().st_size}"
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failed = 0
    process_count = 0
    for run in runs:
        if run.output_path.exists():
            if not args.resume:
                raise FileExistsError(
                    f"worker output already exists; inspect it or pass --resume: {run.output_path}"
                )
            summary = _validated_worker_summary(run)
            completed = summary.get("completed") is True
            summary["exit_code"] = 0 if completed else 2
            summary["resumed_existing"] = True
            rows.append(summary)
            failed += int(not completed)
            process_count = len(rows)
            write_json_atomic(
                manifest_output,
                _build_manifest(
                    plan_file_id=plan_file_id,
                    runs=runs,
                    rows=rows,
                    failed=failed,
                    started=started,
                    final=False,
                    allow_cross_scene_process=bool(args.allow_cross_scene_process),
                    isaac_process_count=process_count if args.fresh_process else 1,
                ),
            )
            continue
        worker_exception = False
        if args.fresh_process:
            try:
                exit_code = _run_worker_subprocess(run, args.device)
            except BaseException:
                worker_exception = True
                failed += 1
                exit_code = 2
                traceback.print_exc()
        else:
            import torch
            from isaacsim.core.api import SimulationContext

            print(
                json.dumps(
                    {"status": "COLLECTION_RUN_STARTED", "run_id": run.run_id}, sort_keys=True
                ),
                flush=True,
            )
            worker_args = worker.parse_args(run.runner_arguments)
            worker_args.device = args.device
            try:
                exit_code = worker.main(worker_args, simulation_app)
            except BaseException as error:
                worker_exception = True
                failed += 1
                exit_code = 2
                worker._write_failure(worker_args, error)
                traceback.print_exc()
            finally:
                SimulationContext.clear_instance()
                simulation_app.update()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        validation_failed = False
        try:
            summary = _validated_worker_summary(run)
        except BaseException as error:
            validation_failed = True
            summary = {
                "run_id": run.run_id,
                "output_path": str(run.output_path),
                "status": "OUTPUT_VALIDATION_FAILED",
                "error": f"{type(error).__name__}: {error}",
            }
        if not worker_exception and (
            validation_failed or exit_code != 0 or summary.get("completed") is not True
        ):
            failed += 1
        summary["exit_code"] = exit_code
        summary["resumed_existing"] = False
        rows.append(summary)
        process_count = len(rows)
        print(
            json.dumps(
                {
                    "status": "COLLECTION_RUN_FINISHED",
                    "run_id": run.run_id,
                    "exit_code": exit_code,
                    "worker_status": summary.get("status"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        write_json_atomic(
            manifest_output,
            _build_manifest(
                plan_file_id=plan_file_id,
                runs=runs,
                rows=rows,
                failed=failed,
                started=started,
                final=False,
                allow_cross_scene_process=bool(args.allow_cross_scene_process),
                isaac_process_count=process_count if args.fresh_process else 1,
            ),
        )
        if exit_code != 0:
            # A partial USD stage or SimulationContext after an exception made the
            # next run hang without trustworthy evidence.
            break
    manifest = _build_manifest(
        plan_file_id=plan_file_id,
        runs=runs,
        rows=rows,
        failed=failed,
        started=started,
        final=True,
        allow_cross_scene_process=bool(args.allow_cross_scene_process),
        isaac_process_count=process_count if args.fresh_process else 1,
    )
    write_json_atomic(manifest_output, manifest)
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_output)}))
    return 0 if manifest["status"] == COLLECTION_COMPLETE_STATUS else 2


def _entrypoint() -> int:
    args = parse_args()
    if args.fresh_process:
        try:
            exit_code = main(args, None)
        except BaseException:
            traceback.print_exc()
            return 2
        sys.stdout.flush()
        sys.stderr.flush()
        return exit_code
    from isaaclab.app import AppLauncher

    app = AppLauncher(args)
    try:
        exit_code = main(args, app.app)
    except BaseException:
        # A failed collection may have an incomplete stage; keep the normal Isaac
        # shutdown path so the traceback stays actionable.
        app.app.close()
        raise

    # The manifest and worker outcomes are already written when main returns; on this
    # Isaac build SimulationApp.close() can spin after a successful collection, so use
    # the same process-boundary rule as the Exploration worker.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
