"""Validate the paper 2D experiment curricula before launching long runs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
REPORTS_ROOT = OUTPUT_ROOT / "reports"
EXPECTED_GATE_COUNTS = (0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60)
EXPECTED_GATE_SPEED_SCHEDULE_MPS = (0.0, 0.0, 0.5, 0.8, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0)
EXPECTED_TRAINING_DRONE_SPEED_AXIS = (1.15, 1.45, 1.75, 2.05, 2.40, 2.75, 3.10, 3.50)
EXPECTED_TRAINING_DRONE_ACCEL_AXIS = (0.75, 1.00, 1.20, 1.40, 1.65, 1.90, 2.15, 2.45)
EXPECTED_TRAINING_DRONE_STAGE_SPEED_SCHEDULE = (1.15, 1.15, 1.45, 1.75, 2.05, 2.40, 2.75, 3.10, 3.50, 3.50, 3.50)
EXPECTED_TRAINING_DRONE_STAGE_ACCEL_SCHEDULE = (0.75, 0.75, 1.00, 1.20, 1.40, 1.65, 1.90, 2.15, 2.45, 2.45, 2.45)
EXPECTED_EVAL_DRONE_SPEED_AXIS = (0.80, 1.25, 1.70, 2.15, 2.60, 3.00, 3.25, 3.50)
EXPECTED_MAX_DRONE_SPEED_MPS = 3.50
EXPECTED_MAX_DRONE_ACCEL_MPS2 = 2.45
DEMO8_CHECKPOINT = Path(os.environ["GATE2D_DEMO8_CHECKPOINT"]) if os.environ.get("GATE2D_DEMO8_CHECKPOINT") else None


def _bootstrap() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _assert(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _float_tuple(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _validate_doc(failures: list[str]) -> dict[str, object]:
    doc_matches = sorted((REPORTS_ROOT / "curricula").glob("2026-05-02_*.md"))
    if not doc_matches:
        return {
            "checked": False,
            "skipped": True,
            "reason": "minimal package does not ship paper markdown documents",
        }
    doc_path = doc_matches[0]
    text = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    forbidden = ("Line -> Column", "motion_frequency_hz", "0.12Hz", "0.15Hz", "0.18Hz")
    for item in forbidden:
        _assert(item not in text, f"doc_forbidden_text_present: {item}", failures)
    gate_axis = "0,6,12,18,24,30,36,42,48,54,60"
    required_tokens = (
        gate_axis,
        "moving_gate_speed_mps",
        "2.0 m/s",
        "BC / DAgger",
        "gate_top_height_m",
        "`8.0`",
        "gate_center_height_m",
        "`4.0`",
        "height_contract_passed",
        "training_drone_speed_axis_mps",
        "training_drone_accel_axis_mps2",
        "evaluation_drone_speed_axis_mps",
        "3.50",
        "2.45",
    )
    for token in required_tokens:
        _assert(token in text, f"doc_missing_required_token: {token}", failures)
    return {
        "path": str(doc_path),
        "gate_axis_present": gate_axis in text,
        "moving_gate_speed_mps_present": "moving_gate_speed_mps" in text,
        "height_contract_present": "height_contract_passed" in text,
        "training_speed_axis_present": "training_drone_speed_axis_mps" in text,
        "training_accel_axis_present": "training_drone_accel_axis_mps2" in text,
        "evaluation_speed_axis_present": "evaluation_drone_speed_axis_mps" in text,
        "forbidden_hits": [item for item in forbidden if item in text],
    }


def _validate_multi_gate_runner(failures: list[str]) -> dict[str, object]:
    _bootstrap()
    from multi_gate.configs import get_multi_experiment_config
    from multi_gate.sanity import validate_dynamic_gate_density_environment
    from shared.configs.global_config import GLOBAL_CONFIG
    from shared.core.dynamic_gate_density_2d import (
        TRAINING_DRONE_ACCEL_AXIS_MPS2,
        TRAINING_DRONE_STAGE_ACCEL_SCHEDULE_MPS2,
        TRAINING_DRONE_STAGE_SPEED_SCHEDULE_MPS,
        TRAINING_DRONE_SPEED_AXIS_MPS,
        drone_accel_limit_for_speed_mps2,
        eval_drone_speed_axis_mps,
        validate_dynamic_gate_density_geometry,
    )

    runner_path = ROOT / "multi_gate" / "scripts" / "run_dynamic_gate_density_8d_curriculum.py"
    runner = _load_module("paper_dynamic_gate_runner", runner_path)
    base_config = get_multi_experiment_config("dynamic_gate_density_8d_v1")
    stages = list(runner._stages())
    counts = tuple(int(stage.gate_count) for stage in stages)
    speeds = tuple(float(stage.speed_mps) for stage in stages)
    drone_speeds = tuple(float(stage.drone_speed_mps) for stage in stages)
    drone_accels = tuple(float(stage.drone_accel_mps2) for stage in stages)

    _assert(counts == EXPECTED_GATE_COUNTS, f"multi_gate_stage_counts={counts}", failures)
    _assert(speeds == EXPECTED_GATE_SPEED_SCHEDULE_MPS, f"multi_gate_gate_speed_schedule={speeds}", failures)
    _assert(drone_speeds == EXPECTED_TRAINING_DRONE_STAGE_SPEED_SCHEDULE, f"multi_gate_drone_stage_speed_schedule={drone_speeds}", failures)
    _assert(drone_accels == EXPECTED_TRAINING_DRONE_STAGE_ACCEL_SCHEDULE, f"multi_gate_drone_stage_accel_schedule={drone_accels}", failures)
    _assert(_float_tuple(TRAINING_DRONE_SPEED_AXIS_MPS) == EXPECTED_TRAINING_DRONE_SPEED_AXIS, "shared_training_drone_speed_axis_changed", failures)
    _assert(_float_tuple(TRAINING_DRONE_ACCEL_AXIS_MPS2) == EXPECTED_TRAINING_DRONE_ACCEL_AXIS, "shared_training_drone_accel_axis_changed", failures)
    _assert(_float_tuple(TRAINING_DRONE_STAGE_SPEED_SCHEDULE_MPS) == EXPECTED_TRAINING_DRONE_STAGE_SPEED_SCHEDULE, "shared_training_drone_stage_speed_schedule_changed", failures)
    _assert(_float_tuple(TRAINING_DRONE_STAGE_ACCEL_SCHEDULE_MPS2) == EXPECTED_TRAINING_DRONE_STAGE_ACCEL_SCHEDULE, "shared_training_drone_stage_accel_schedule_changed", failures)
    _assert(eval_drone_speed_axis_mps() == EXPECTED_EVAL_DRONE_SPEED_AXIS, "shared_eval_drone_speed_axis_changed", failures)
    _assert(max(eval_drone_speed_axis_mps()) == EXPECTED_MAX_DRONE_SPEED_MPS, "eval_drone_speed_max_not_3p5", failures)
    _assert(drone_accel_limit_for_speed_mps2(EXPECTED_MAX_DRONE_SPEED_MPS) == EXPECTED_MAX_DRONE_ACCEL_MPS2, "shared_accel_for_max_speed_not_2p45", failures)
    _assert(float(GLOBAL_CONFIG.planar_max_speed_mps) == EXPECTED_MAX_DRONE_SPEED_MPS, "global_speed_cap_not_3p5", failures)
    _assert(float(GLOBAL_CONFIG.planar_max_accel_mps2) == EXPECTED_MAX_DRONE_ACCEL_MPS2, "global_accel_cap_not_2p45", failures)

    gate_cfg = base_config.dynamic_gate_density
    _assert(int(gate_cfg.max_gate_count) == 60, "multi_gate_max_gate_count_not_60", failures)
    _assert(float(gate_cfg.max_moving_gate_speed_mps) == 2.0, "multi_gate_max_moving_gate_speed_not_2", failures)
    _assert(base_config.default_agents == 8, "multi_gate_default_agents_not_8", failures)
    _assert(float(gate_cfg.gate_opening_bottom_height_m) == 0.0, "multi_gate_gate_bottom_not_0", failures)
    _assert(float(gate_cfg.gate_opening_top_height_m) == 8.0, "multi_gate_gate_top_not_8", failures)
    _assert(float(gate_cfg.gate_center_height_m) == 4.0, "multi_gate_gate_center_not_4", failures)
    _assert(float(gate_cfg.fixed_height_m) == 4.0, "multi_gate_fixed_height_not_gate_center", failures)

    geometry_reports: list[dict[str, object]] = []
    env_reports: list[dict[str, object]] = []
    for idx, stage in enumerate(stages):
        config = runner._stage_config(base_config, stage)
        geometry = validate_dynamic_gate_density_geometry(
            gate_count=int(stage.gate_count),
            speed_mps=float(stage.speed_mps),
            amplitude_m=float(stage.amplitude_m),
            seed=20260503 + idx,
            config=config.dynamic_gate_density,
            sample_times_s=(0.0, 0.4, 0.8),
        )
        geometry_reports.append(
            {
                "stage_index": idx,
                "gate_count": int(stage.gate_count),
                "speed_mps": float(stage.speed_mps),
                "drone_speed_mps": float(stage.drone_speed_mps),
                "drone_accel_mps2": float(stage.drone_accel_mps2),
                "passed": bool(geometry["passed"]),
                "max_center_motion_m": geometry.get("max_center_motion_m"),
            }
        )
        _assert(bool(geometry["passed"]), f"geometry_failed_stage_{idx}: {geometry.get('failures')}", failures)
        env_report = validate_dynamic_gate_density_environment(
            experiment_config=config,
            seed=20260503 + idx,
            num_agents=8,
            forward_steps=12,
        )
        env_reports.append(
            {
                "stage_index": idx,
                "gate_count": int(stage.gate_count),
                "speed_mps": float(stage.speed_mps),
                "drone_speed_mps": float(stage.drone_speed_mps),
                "passed": bool(env_report["passed"]),
                "failures": env_report.get("failures", []),
                "forward_delta_x_m": env_report["forward_progress"]["delta_x_m"],
                "collision_terminated": env_report["collision"].get("terminated"),
                "dynamic_motion_m": env_report["dynamic_motion"].get("motion_after_zero_action_m"),
            }
        )
        _assert(bool(env_report["passed"]), f"env_failed_stage_{idx}: {env_report.get('failures')}", failures)

    return {
        "runner": str(runner_path),
        "stage_counts": counts,
        "stage_gate_speeds": speeds,
        "drone_stage_speed_schedule": drone_speeds,
        "drone_stage_accel_schedule": drone_accels,
        "drone_speed_axis": _float_tuple(TRAINING_DRONE_SPEED_AXIS_MPS),
        "drone_accel_axis": _float_tuple(TRAINING_DRONE_ACCEL_AXIS_MPS2),
        "eval_drone_speed_axis": eval_drone_speed_axis_mps(),
        "height_contract": {
            "gate_bottom_height_m": float(gate_cfg.gate_opening_bottom_height_m),
            "gate_top_height_m": float(gate_cfg.gate_opening_top_height_m),
            "gate_center_height_m": float(gate_cfg.gate_center_height_m),
            "fixed_height_m": float(gate_cfg.fixed_height_m),
        },
        "geometry_reports": geometry_reports,
        "env_reports": env_reports,
    }


def _validate_self_contained_runner(failures: list[str]) -> dict[str, object]:
    runner_path = ROOT / "gate_density_multi_8" / "scripts" / "train_dynamic_gate_density_8d_curriculum.py"
    runner = _load_module("paper_self_contained_gate_runner", runner_path)
    stages = list(runner.build_curriculum())
    counts = tuple(int(stage.gate_count) for stage in stages)
    speeds = tuple(float(stage.moving_gate_speed_mps) for stage in stages)
    drone_speeds = tuple(float(stage.drone_base_speed_mps) for stage in stages)
    drone_accels = tuple(float(stage.drone_accel_limit_mps2) for stage in stages)
    source = runner_path.read_text(encoding="utf-8")

    _assert(counts == EXPECTED_GATE_COUNTS, f"self_contained_stage_counts={counts}", failures)
    _assert(speeds == EXPECTED_GATE_SPEED_SCHEDULE_MPS, f"self_contained_gate_speed_schedule={speeds}", failures)
    _assert(drone_speeds == EXPECTED_TRAINING_DRONE_STAGE_SPEED_SCHEDULE, f"self_contained_drone_stage_speed_schedule={drone_speeds}", failures)
    _assert(drone_accels == EXPECTED_TRAINING_DRONE_STAGE_ACCEL_SCHEDULE, f"self_contained_drone_stage_accel_schedule={drone_accels}", failures)
    _assert("stage02d_empty_8_full" not in source, "self_contained_uses_old_stage02d_checkpoint", failures)
    _assert("demo8_35_full_route_mixed_isaaclab_render" in source, "self_contained_missing_demo8_checkpoint", failures)
    _assert("2.85" not in source, "self_contained_controller_still_clips_to_2p85", failures)
    _assert("stage_speed_cap + 0.95" not in source, "self_contained_speed_limit_can_exceed_stage_cap", failures)
    return {
        "runner": str(runner_path),
        "stage_counts": counts,
        "stage_gate_speeds": speeds,
        "drone_stage_speed_schedule": drone_speeds,
        "drone_stage_accel_schedule": drone_accels,
        "controller_base_speed_clip_allows_3p5": "3.50" in source,
    }


def _validate_single_gate_height_contract(failures: list[str]) -> dict[str, object]:
    runner_path = ROOT / "gate_density_single" / "scripts" / "run_gate_density_eval.py"
    wrapper_path = ROOT / "gate_density_single" / "scripts" / "run_paper_gate_density_single_eval.py"
    runner = _load_module("paper_single_gate_density_runner", runner_path)
    runner_source = runner_path.read_text(encoding="utf-8")
    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    _assert(tuple(float(v) for v in runner.START_XYZ) == (-9.0, 0.0, 4.0), "single_start_xyz_not_center_4m", failures)
    _assert(tuple(float(v) for v in runner.GOAL_XYZ) == (9.0, 0.0, 4.0), "single_goal_xyz_not_center_4m", failures)
    _assert(float(runner.GATE_TOP_HEIGHT_M) == 8.0, "single_gate_top_not_8m", failures)
    _assert(float(runner.GATE_CENTER_HEIGHT_M) == 4.0, "single_gate_center_not_4m", failures)
    _assert("--drone-speed-mps" in runner_source, "single_eval_missing_drone_speed_arg", failures)
    _assert("DEFAULT_SINGLE_DRONE_SPEED_MPS = 3.50" in wrapper_source, "single_wrapper_default_speed_not_3p5", failures)
    _assert("drone_accel_limit_for_speed_mps2" in wrapper_source, "single_wrapper_not_using_shared_accel_axis", failures)
    _assert("height_contract_passed_rate" in runner_source, "single_eval_missing_height_contract_rate", failures)
    _assert("corridor_through_success_rate" in runner_source, "single_eval_missing_corridor_through_rate", failures)
    _assert(
        "drone_shell_top_m = float(fixed_height_m + DRONE_RADIUS_M)" in runner_source,
        "single_height_audit_not_using_fixed_height_for_shell",
        failures,
    )
    _assert("side_bypass_failure_rate" in wrapper_source, "single_wrapper_missing_side_bypass_rate_column", failures)
    _assert("E9_single_drone_speed_gradient" in wrapper_source, "single_wrapper_missing_speed_gradient_eval", failures)
    _assert("gate_visual_scale_xyz" in runner_source, "single_eval_missing_gate_visual_scale", failures)
    multi_wrapper_path = ROOT / "multi_gate" / "scripts" / "run_paper_multi_gate_density_eval.py"
    multi_legacy_wrapper_path = ROOT / "multi_gate" / "scripts" / "run_dynamic_gate_density_8d_paper_eval.py"
    multi_training_path = ROOT / "multi_gate" / "training.py"
    multi_replay_path = ROOT / "multi_gate" / "replay.py"
    multi_wrapper_source = multi_wrapper_path.read_text(encoding="utf-8")
    multi_legacy_wrapper_source = multi_legacy_wrapper_path.read_text(encoding="utf-8")
    multi_training_source = multi_training_path.read_text(encoding="utf-8")
    multi_replay_source = multi_replay_path.read_text(encoding="utf-8")
    _assert("DEFAULT_DRONE_SPEED_MPS = 3.50" in multi_wrapper_source, "multi_wrapper_default_speed_not_3p5", failures)
    _assert("DEFAULT_DRONE_SPEED_MPS = 3.50" in multi_legacy_wrapper_source, "multi_legacy_wrapper_default_speed_not_3p5", failures)
    _assert("height_contract_passed_rate" in multi_training_source, "multi_training_missing_height_contract_rate", failures)
    _assert("corridor_through_success_rate" in multi_training_source, "multi_training_missing_corridor_through_rate", failures)
    _assert("_multi_episode_success_from_info" in multi_training_source, "multi_training_missing_audited_success_helper", failures)
    _assert("_multi_episode_success_from_info" in multi_replay_source, "multi_replay_missing_audited_success_helper", failures)
    _assert("side_bypass_failure_rate" in multi_wrapper_source, "multi_wrapper_missing_side_bypass_rate_column", failures)
    return {
        "runner": str(runner_path),
        "wrapper": str(wrapper_path),
        "visual_replay": "not packaged",
        "start_xyz": list(runner.START_XYZ),
        "goal_xyz": list(runner.GOAL_XYZ),
        "gate_top_height_m": float(runner.GATE_TOP_HEIGHT_M),
        "gate_center_height_m": float(runner.GATE_CENTER_HEIGHT_M),
        "drone_speed_arg_present": "--drone-speed-mps" in runner_source,
        "single_speed_gradient_eval_present": "E9_single_drone_speed_gradient" in wrapper_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_ROOT / "paper_2d_curriculum_preflight" / "latest_report.json",
    )
    args = parser.parse_args()

    failures: list[str] = []
    report: dict[str, object] = {
        "expected_gate_counts": EXPECTED_GATE_COUNTS,
        "expected_gate_speed_schedule_mps": EXPECTED_GATE_SPEED_SCHEDULE_MPS,
        "demo8_checkpoint": str(DEMO8_CHECKPOINT) if DEMO8_CHECKPOINT else None,
        "demo8_checkpoint_exists": bool(DEMO8_CHECKPOINT and DEMO8_CHECKPOINT.exists()),
    }
    if DEMO8_CHECKPOINT is not None:
        _assert(DEMO8_CHECKPOINT.exists(), f"missing_demo8_checkpoint={DEMO8_CHECKPOINT}", failures)
    report["doc"] = _validate_doc(failures)
    report["multi_gate_runner"] = _validate_multi_gate_runner(failures)
    report["self_contained_runner"] = _validate_self_contained_runner(failures)
    report["single_gate_height_contract"] = _validate_single_gate_height_contract(failures)
    report["passed"] = len(failures) == 0
    report["failures"] = failures

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "failures": failures, "report": str(args.output)}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

