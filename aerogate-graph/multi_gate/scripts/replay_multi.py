"""Replay entry for the multi-agent Graph-FlashSAC and heuristic controllers."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import sys
from pathlib import Path
from typing import Any


def _bootstrap_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_dynamic_gate_runner(root: Path) -> Any:
    runner_path = root / "multi_gate" / "scripts" / "run_dynamic_gate_density_8d_curriculum.py"
    spec = importlib.util.spec_from_file_location("dynamic_gate_curriculum_runner_for_replay", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dynamic gate curriculum runner from {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_gate_curriculum_runner_for_replay"] = runner
    spec.loader.exec_module(runner)
    return runner


def _apply_dynamic_gate_stage_overrides(root: Path, experiment_config: Any, args: argparse.Namespace) -> Any:
    has_stage_override = any(
        value is not None
        for value in (
            args.stage_index,
            args.stage_name,
            args.gate_count,
            args.speed_mps,
            args.amplitude_m,
            args.drone_speed_mps,
            args.drone_accel_mps2,
        )
    )
    if not has_stage_override:
        return experiment_config
    if str(experiment_config.experiment_id) != "multi_gate_dynamic_gate_density_8d_v1":
        raise ValueError("Dynamic gate stage replay overrides require config-name=dynamic_gate_density_8d_v1 or auto.")

    runner = _load_dynamic_gate_runner(root)
    stages = runner._stages()
    stage_index = int(args.stage_index) if args.stage_index is not None else 0
    if stage_index < 0 or stage_index >= len(stages):
        raise ValueError(f"--stage-index must be in [0, {len(stages) - 1}]")
    stage = stages[stage_index]
    if args.stage_name is not None:
        requested_name = str(args.stage_name)
        matched_stage = None
        for candidate in stages:
            if str(candidate.name) == requested_name:
                matched_stage = candidate
                break
        stage = matched_stage or replace(stage, name=requested_name)

    stage = replace(
        stage,
        gate_count=int(args.gate_count) if args.gate_count is not None else int(stage.gate_count),
        speed_mps=float(args.speed_mps) if args.speed_mps is not None else float(stage.speed_mps),
        amplitude_m=float(args.amplitude_m) if args.amplitude_m is not None else float(stage.amplitude_m),
        drone_speed_mps=(
            float(args.drone_speed_mps) if args.drone_speed_mps is not None else float(stage.drone_speed_mps)
        ),
        drone_accel_mps2=(
            float(args.drone_accel_mps2) if args.drone_accel_mps2 is not None else float(stage.drone_accel_mps2)
        ),
    )
    return runner._stage_config(experiment_config, stage)


def main() -> None:
    _bootstrap_imports()
    from multi_gate.configs import list_multi_experiment_config_names, resolve_multi_experiment_config
    from multi_gate.replay import render_multi_replay_isaaclab_from_summary, run_multi_replay

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Replay the multi-agent task with a checkpoint or heuristic controller.")
    parser.add_argument("--mode", type=str, default="heuristic", choices=["heuristic", "checkpoint"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config-name", type=str, default="auto", choices=["auto", *list_multi_experiment_config_names()])
    parser.add_argument("--render-isaaclab", action="store_true")
    parser.add_argument("--export-video", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stage-index", type=int, default=None)
    parser.add_argument("--stage-name", type=str, default=None)
    parser.add_argument("--gate-count", type=int, default=None)
    parser.add_argument("--speed-mps", type=float, default=None)
    parser.add_argument("--amplitude-m", type=float, default=None)
    parser.add_argument("--drone-speed-mps", type=float, default=None)
    parser.add_argument("--drone-accel-mps2", type=float, default=None)
    parser.add_argument(
        "--camera-mode",
        type=str,
        default="picture_in_picture",
        choices=["global", "follow", "picture_in_picture"],
    )
    args = parser.parse_args()
    resolved_config_name, experiment_config = resolve_multi_experiment_config(
        args.config_name,
        checkpoint_path=args.checkpoint if args.mode == "checkpoint" else None,
    )
    experiment_config = _apply_dynamic_gate_stage_overrides(root, experiment_config, args)

    summary = run_multi_replay(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        num_agents=args.num_agents,
        seed=args.seed,
        device=args.device,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        experiment_config=experiment_config,
    )
    if args.render_isaaclab:
        summary["isaaclab_render"] = render_multi_replay_isaaclab_from_summary(
            replay_summary=summary,
            experiment_config=experiment_config,
            output_dir=Path(str(summary["report_path"])).resolve().parent / "isaaclab",
            export_video=args.export_video,
            fps=args.fps,
            camera_mode=args.camera_mode,
            headless=True,
        )
    print("multi-agent replay complete")
    print(f"config_name={resolved_config_name}")
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()

