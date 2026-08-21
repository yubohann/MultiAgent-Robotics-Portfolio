"""Run relaxed variable-team continuation training and full evaluation.

This runner keeps the current 8-drone Graph-FlashSAC checkpoint lineage, adds
equal fixed-team continuation stages for team sizes 2..7 under static and
dynamic gate18, then evaluates team sizes 1..7 with relaxed but still explicit
geometry auditing and top-view MP4 replay export.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_gate.configs import get_multi_experiment_config
from multi_gate.configs.experiment_config import MultiSizeInvarianceConfig
from multi_gate.scripts.run_dynamic_gate_density_8d_curriculum import (
    DynamicGateCurriculumStage,
    _stage_config,
)
from multi_gate.training import _load_checkpoint_metadata, run_training


DEFAULT_INPUT_CHECKPOINT = (
    ROOT
    / "runtime"
    / "dynamic_gate_density_8d_curriculum_v8_dyn7"
    / "stages"
    / "07_C1p47_micro_dynamic_gate07_speed020_amp010_drone115"
    / "checkpoints"
    / "best_agent.pt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results" / f"variable_team_size_full_continuation_{datetime.now():%Y%m%d_%H%M%S}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def parse_int_range(text: str) -> list[int]:
    value = str(text).strip()
    if ":" in value:
        start, stop, *rest = value.split(":")
        step = int(rest[0]) if rest and rest[0] else 1
        return list(range(int(start), int(stop), step))
    return [int(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]


def relaxed_stage(
    *,
    name: str,
    gate_count: int,
    speed_mps: float,
    amplitude_m: float,
    train_steps: int,
) -> DynamicGateCurriculumStage:
    return DynamicGateCurriculumStage(
        name=name,
        gate_count=int(gate_count),
        speed_mps=float(speed_mps),
        amplitude_m=float(amplitude_m),
        train_steps=int(train_steps),
        goal_slot_tolerance_m=5.8,
        slot_error_penalty_scale=0.28,
        slot_improvement_scale=0.45,
        max_slot_error_penalty_scale=0.0,
        inter_agent_safe_distance_m=0.82,
        shield_margin_m=1.05,
        drone_speed_mps=1.90 if float(speed_mps) > 0.0 else 1.65,
        drone_accel_mps2=1.25,
        notes=(
            "Relaxed continuation stage for variable team-size transfer. "
            "Uses policy-only Graph-FlashSAC training with safety shielding, "
            "not compact formation playback."
        ),
    )


def make_train_config(
    *,
    base_config: Any,
    stage: DynamicGateCurriculumStage,
    team_sizes: list[int],
    gate_post_radius_m: float,
    drone_radius_m: float,
    disable_guidance_runtime: bool,
    notes_suffix: str,
) -> Any:
    config = _stage_config(base_config, stage)
    gate_cfg = replace(
        config.dynamic_gate_density,
        gate_post_radius_m=float(gate_post_radius_m),
        drone_radius_m=float(drone_radius_m),
    )
    env_cfg = replace(
        config.environment,
        drone_radius_m=float(drone_radius_m),
        timeout_counts_as_success=False,
        formation_line_collapse_terminal=False,
        formation_line_collapse_min_lateral_bands=min(3, max(team_sizes)),
        formation_line_collapse_penalty_scale=max(
            float(getattr(config.environment, "formation_line_collapse_penalty_scale", 0.0) or 0.0),
            12.0,
        ),
    )
    reasoning_cfg = config.reasoning
    if bool(disable_guidance_runtime):
        reasoning_cfg = replace(
            reasoning_cfg,
            route_guidance_enabled=False,
            guidance_shadow_mode=False,
            guidance_async_enabled=False,
            guidance_cache_enabled=False,
            guidance_provider="none",
        )
    return replace(
        config,
        default_agents=max(team_sizes),
        dynamic_gate_density=gate_cfg,
        environment=env_cfg,
        reasoning=reasoning_cfg,
        size_invariance=MultiSizeInvarianceConfig(
            enabled=True,
            team_size_sampling_mode="uniform_buckets",
            bucket_team_sizes=tuple(int(size) for size in team_sizes),
            bucket_eval_episodes=0,
            min_bucket_success_rate=0.0,
        ),
        notes=f"{config.notes}\n{notes_suffix}",
    )


def build_training_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    team_sizes = [int(size) for size in args.train_team_sizes]
    plan: list[dict[str, Any]] = []
    for team_size in team_sizes:
        plan.append(
            {
                "name": f"team{team_size:02d}_static_gate18_relaxed",
                "team_size": int(team_size),
                "condition": "static_gate18",
                "gate_count": 18,
                "speed_mps": 0.0,
                "amplitude_m": 0.0,
                "train_steps": int(args.train_steps_per_stage),
                "fixed_team": True,
            }
        )
        plan.append(
            {
                "name": f"team{team_size:02d}_dynamic_gate18_relaxed",
                "team_size": int(team_size),
                "condition": "dynamic_gate18",
                "gate_count": 18,
                "speed_mps": 0.8,
                "amplitude_m": 0.75,
                "train_steps": int(args.train_steps_per_stage),
                "fixed_team": True,
            }
        )
    if int(args.mixed_train_steps_per_condition) > 0:
        plan.append(
            {
                "name": "mixed_2to7_static_gate18_relaxed",
                "team_size": None,
                "condition": "static_gate18",
                "gate_count": 18,
                "speed_mps": 0.0,
                "amplitude_m": 0.0,
                "train_steps": int(args.mixed_train_steps_per_condition),
                "fixed_team": False,
            }
        )
        plan.append(
            {
                "name": "mixed_2to7_dynamic_gate18_relaxed",
                "team_size": None,
                "condition": "dynamic_gate18",
                "gate_count": 18,
                "speed_mps": 0.8,
                "amplitude_m": 0.75,
                "train_steps": int(args.mixed_train_steps_per_condition),
                "fixed_team": False,
            }
        )
    return plan


def stage_summary_path(stage_dir: Path) -> Path:
    return stage_dir / "logs" / "training_summary.json"


def run_training_plan(args: argparse.Namespace, output_root: Path, input_checkpoint: Path) -> Path:
    base_config = get_multi_experiment_config("dynamic_gate_density_8d_v1")
    team_sizes = [int(size) for size in args.train_team_sizes]
    plan = build_training_plan(args)
    write_json(output_root / "training_plan.json", plan)
    current_checkpoint = input_checkpoint
    records: list[dict[str, Any]] = []

    for index, item in enumerate(plan, start=1):
        stage_name = str(item["name"])
        stage_dir = output_root / "training" / f"{index:02d}_{stage_name}"
        summary_path = stage_summary_path(stage_dir)
        if bool(args.resume) and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            next_checkpoint = (
                summary.get("checkpoint_path")
                or summary.get("best_alias_path")
                or summary.get("best_checkpoint_path")
                or summary.get("latest_alias_path")
                or summary.get("final_checkpoint_path")
            )
            if next_checkpoint and Path(str(next_checkpoint)).exists():
                current_checkpoint = Path(str(next_checkpoint))
                records.append(
                    {
                        "stage_index": index,
                        "stage_name": stage_name,
                        "status": "skipped_existing",
                        "checkpoint_out": str(current_checkpoint),
                        "summary_path": str(summary_path),
                    }
                )
                write_json(output_root / "training_state.json", {"current_checkpoint": current_checkpoint, "records": records})
                continue

        stage = relaxed_stage(
            name=stage_name,
            gate_count=int(item["gate_count"]),
            speed_mps=float(item["speed_mps"]),
            amplitude_m=float(item["amplitude_m"]),
            train_steps=int(item["train_steps"]),
        )
        config = make_train_config(
            base_config=base_config,
            stage=stage,
            team_sizes=team_sizes,
            gate_post_radius_m=float(args.train_gate_post_radius_m),
            drone_radius_m=float(args.train_drone_radius_m),
            disable_guidance_runtime=bool(args.disable_guidance_runtime),
            notes_suffix=(
                f"Full continuation stage {index}: condition={item['condition']}, "
                f"fixed_team={item['team_size']}."
            ),
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "stage": index,
                    "name": stage_name,
                    "input_checkpoint": str(current_checkpoint),
                    "team_size": item["team_size"],
                    "train_steps": item["train_steps"],
                    "gate_count": item["gate_count"],
                    "speed_mps": item["speed_mps"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        summary = run_training(
            train_steps=int(item["train_steps"]),
            num_envs=int(args.num_envs),
            seed=int(args.seed) + index * 1000,
            device=str(args.train_device),
            save_dir=stage_dir,
            log_dir=stage_dir / "logs",
            checkpoint_dir=stage_dir / "checkpoints",
            checkpoint_name="graph_flashsac_multi_dynamic_gate_density_8d.pt",
            num_agents=(None if item["team_size"] is None else int(item["team_size"])),
            max_sampled_agents=max(team_sizes),
            learning_starts=int(args.learning_starts),
            batch_size=int(args.batch_size),
            updates_per_step=int(args.updates_per_step),
            log_every=int(args.log_every),
            experiment_config=config,
            resume_checkpoint=current_checkpoint,
            resume_mode=str(args.resume_mode),
            checkpoint_interval_steps=int(args.checkpoint_interval_transitions),
            selection_eval_episodes=int(args.selection_eval_episodes),
            periodic_eval_episodes=0,
        )
        next_checkpoint = (
            summary.get("checkpoint_path")
            or summary.get("best_alias_path")
            or summary.get("best_checkpoint_path")
            or summary.get("latest_alias_path")
            or summary.get("final_checkpoint_path")
        )
        if not next_checkpoint or not Path(str(next_checkpoint)).exists():
            raise RuntimeError(f"Training stage {stage_name} did not produce a checkpoint.")
        current_checkpoint = Path(str(next_checkpoint))
        records.append(
            {
                "stage_index": index,
                "stage_name": stage_name,
                "status": "completed",
                "checkpoint_in": str(summary.get("resume_context", {}).get("resume_checkpoint_path") if isinstance(summary.get("resume_context"), dict) else ""),
                "checkpoint_out": str(current_checkpoint),
                "summary_path": summary.get("summary_path"),
                "team_sizes_seen": summary.get("team_sizes_seen"),
                "done_reason_counts": summary.get("done_reason_counts"),
            }
        )
        write_json(output_root / "training_state.json", {"current_checkpoint": current_checkpoint, "records": records})
    return current_checkpoint


def run_eval(args: argparse.Namespace, output_root: Path, checkpoint: Path) -> int:
    eval_root = output_root / "evaluation"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_variable_team_size_eval.py"),
        "--checkpoint",
        str(checkpoint),
        "--output-root",
        str(eval_root),
        "--mode",
        "all",
        "--include-stress",
        "--team-sizes",
        args.eval_team_sizes_text,
        "--pilot-team-sizes",
        args.pilot_team_sizes_text,
        "--pilot-seeds",
        args.pilot_seeds_text,
        "--formal-seeds",
        args.formal_seeds_text,
        "--render-seeds",
        args.render_seeds_text,
        "--gate-post-radius-m",
        str(float(args.eval_gate_post_radius_m)),
        "--drone-radius-m",
        str(float(args.eval_drone_radius_m)),
        "--clean-swept-clearance-m",
        str(float(args.clean_swept_clearance_m)),
        "--video-frame-stride",
        str(int(args.video_frame_stride)),
        "--video-fps",
        str(float(args.video_fps)),
        "--step-sample-stride",
        str(int(args.step_sample_stride)),
        "--candidate-limit",
        "1",
        "--disable-terminal-formation-collapse",
        "--resume",
    ]
    if args.max_episode_steps is not None:
        cmd.extend(["--max-episode-steps", str(int(args.max_episode_steps))])
    write_json(output_root / "eval_command.json", {"cmd": cmd, "eval_root": eval_root})
    print(json.dumps({"starting_eval": str(eval_root), "checkpoint": str(checkpoint)}, ensure_ascii=False), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-checkpoint", type=Path, default=DEFAULT_INPUT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-team-sizes", type=parse_int_range, default=list(range(2, 8)))
    parser.add_argument("--train-steps-per-stage", type=int, default=4096)
    parser.add_argument("--mixed-train-steps-per-condition", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--train-device", type=str, default="cuda")
    parser.add_argument("--learning-starts", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=256)
    parser.add_argument("--checkpoint-interval-transitions", type=int, default=8192)
    parser.add_argument("--selection-eval-episodes", type=int, default=0)
    parser.add_argument("--resume-mode", type=str, default="reset_train_state", choices=["reset_train_state", "keep_optimizer_state"])
    parser.add_argument("--train-gate-post-radius-m", type=float, default=0.38)
    parser.add_argument("--train-drone-radius-m", type=float, default=0.35)
    parser.add_argument("--eval-gate-post-radius-m", type=float, default=0.40)
    parser.add_argument("--eval-drone-radius-m", type=float, default=0.35)
    parser.add_argument("--clean-swept-clearance-m", type=float, default=0.02)
    parser.add_argument("--eval-team-sizes-text", type=str, default="1:8")
    parser.add_argument("--pilot-team-sizes-text", type=str, default="1,4,7")
    parser.add_argument("--pilot-seeds-text", type=str, default="0:5")
    parser.add_argument("--formal-seeds-text", type=str, default="0:20")
    parser.add_argument("--render-seeds-text", type=str, default="0")
    parser.add_argument("--video-frame-stride", type=int, default=8)
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--step-sample-stride", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--disable-guidance-runtime", action="store_true", default=True)
    parser.add_argument("--enable-guidance-runtime", dest="disable_guidance_runtime", action="store_false")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    input_checkpoint = Path(args.input_checkpoint)
    if not input_checkpoint.exists():
        raise FileNotFoundError(input_checkpoint)
    metadata = _load_checkpoint_metadata(input_checkpoint)
    write_json(
        output_root / "input_checkpoint_audit.json",
        {
            "path": input_checkpoint,
            "sha256": sha256_file(input_checkpoint),
            "metadata": metadata,
        },
    )
    write_json(
        output_root / "run_config.json",
        {
            "script": Path(__file__).resolve(),
            "args": vars(args),
            "python": sys.executable,
        },
    )
    final_checkpoint = input_checkpoint
    if not bool(args.skip_training):
        final_checkpoint = run_training_plan(args, output_root, input_checkpoint)
    write_json(
        output_root / "final_checkpoint.json",
        {
            "path": final_checkpoint,
            "sha256": sha256_file(final_checkpoint),
            "training_skipped": bool(args.skip_training),
        },
    )
    eval_exit_code = 0
    if not bool(args.skip_eval):
        eval_exit_code = run_eval(args, output_root, final_checkpoint)
    summary = {
        "output_root": output_root,
        "final_checkpoint": final_checkpoint,
        "evaluation_root": output_root / "evaluation",
        "eval_exit_code": int(eval_exit_code),
    }
    write_json(output_root / "run_summary.json", summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)
    if eval_exit_code != 0:
        raise SystemExit(eval_exit_code)


if __name__ == "__main__":
    main()

