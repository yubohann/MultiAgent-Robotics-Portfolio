"""Evaluate a multi-drone checkpoint over relaxed dynamic-gate difficulty settings."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any


def _bootstrap_imports() -> Path:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


ROOT = _bootstrap_imports()

from multi_gate.configs import get_multi_experiment_config  # noqa: E402
from multi_gate.scripts.run_dynamic_gate_density_8d_curriculum import _stage_config  # noqa: E402
from scripts.run_multi_c6a110r3_static_dynamic_continuation import (  # noqa: E402
    _build_stages,
)
from multi_gate.training import evaluate_checkpoint  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _build_config(
    *,
    gate_count: int,
    speed_scale: float,
    amplitude_scale: float,
    gate_post_radius_m: float,
    drone_radius_m: float,
    gate_half_width_m: float | None,
    gate_half_width_scale: float | None,
    safety_clearance_m: float,
    obstacle_shield_margin_m: float,
    separation_shield_margin_m: float,
    inter_agent_safe_distance_m: float,
    post_gate_cruise_min_pair_distance_m: float,
    lateral_spacing_m: float | None,
    longitudinal_spacing_m: float | None,
) -> Any:
    base_config = get_multi_experiment_config("dynamic_gate_density_8d_v1")
    stage = _build_stages(
        "dynamic",
        [int(gate_count)],
        1.0,
        "difficulty_diagnostic",
        float(speed_scale),
        float(amplitude_scale),
    )[0]
    config = _stage_config(base_config, stage)
    gate_half_width = (
        float(gate_half_width_m)
        if gate_half_width_m is not None
        else float(config.dynamic_gate_density.gate_half_width_m)
    )
    if gate_half_width_scale is not None:
        gate_half_width *= float(gate_half_width_scale)
    config = replace(
        config,
        dynamic_gate_density=replace(
            config.dynamic_gate_density,
            gate_post_radius_m=float(gate_post_radius_m),
            drone_radius_m=float(drone_radius_m),
            gate_half_width_m=float(gate_half_width),
        ),
        environment=replace(
            config.environment,
            drone_radius_m=float(drone_radius_m),
            inter_agent_safe_distance_m=float(inter_agent_safe_distance_m),
            safety_clearance_m=float(safety_clearance_m),
            action_safety_shield_obstacle_margin_m=float(obstacle_shield_margin_m),
            action_safety_shield_separation_margin_m=float(separation_shield_margin_m),
            action_safety_shield_post_gate_cruise_min_pair_distance_m=float(
                post_gate_cruise_min_pair_distance_m
            ),
        ),
        reasoning=replace(
            config.reasoning,
            route_guidance_enabled=False,
            guidance_shadow_mode=False,
            guidance_async_enabled=False,
            guidance_cache_enabled=False,
            guidance_provider="none",
        ),
    )
    if lateral_spacing_m is not None or longitudinal_spacing_m is not None:
        config = replace(
            config,
            formation=replace(
                config.formation,
                lateral_spacing_m=(
                    float(lateral_spacing_m)
                    if lateral_spacing_m is not None
                    else float(config.formation.lateral_spacing_m)
                ),
                longitudinal_spacing_m=(
                    float(longitudinal_spacing_m)
                    if longitudinal_spacing_m is not None
                    else float(config.formation.longitudinal_spacing_m)
                ),
            ),
        )
    return config


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gates", type=str, default="6,12,18")
    parser.add_argument("--speed-scales", type=str, default="0.0,0.25,0.35,0.50,1.0")
    parser.add_argument("--amplitude-scales", type=str, default="0.0,0.35,0.45,0.60,1.0")
    parser.add_argument("--radius-pairs", type=str, default="0.32:0.35,0.28:0.25,0.24:0.22")
    parser.add_argument("--gate-half-width-m", type=float, default=2.40)
    parser.add_argument("--gate-half-width-scale", type=float, default=None)
    parser.add_argument("--safety-clearance-m", type=float, default=0.35)
    parser.add_argument("--obstacle-shield-margin-m", type=float, default=1.0)
    parser.add_argument("--separation-shield-margin-m", type=float, default=0.75)
    parser.add_argument("--inter-agent-safe-distance-m", type=float, default=0.55)
    parser.add_argument("--post-gate-cruise-min-pair-distance-m", type=float, default=0.95)
    parser.add_argument("--lateral-spacing-m", type=float, default=None)
    parser.add_argument("--longitudinal-spacing-m", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gates = _parse_int_list(args.gates)
    speed_scales = _parse_float_list(args.speed_scales)
    amplitude_scales = _parse_float_list(args.amplitude_scales)
    radius_pairs: list[tuple[float, float]] = []
    for raw_pair in str(args.radius_pairs).split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        left, right = raw_pair.split(":", maxsplit=1)
        radius_pairs.append((float(left), float(right)))

    rows: list[dict[str, Any]] = []
    csv_path = args.output_dir / "difficulty_grid.csv"
    jsonl_path = args.output_dir / "difficulty_grid.jsonl"
    fieldnames: list[str] | None = None
    for gate_count in gates:
        for speed_scale in speed_scales:
            for amplitude_scale in amplitude_scales:
                for gate_post_radius_m, drone_radius_m in radius_pairs:
                    config = _build_config(
                        gate_count=gate_count,
                        speed_scale=speed_scale,
                        amplitude_scale=amplitude_scale,
                        gate_post_radius_m=gate_post_radius_m,
                        drone_radius_m=drone_radius_m,
                        gate_half_width_m=args.gate_half_width_m,
                        gate_half_width_scale=args.gate_half_width_scale,
                        safety_clearance_m=args.safety_clearance_m,
                        obstacle_shield_margin_m=args.obstacle_shield_margin_m,
                        separation_shield_margin_m=args.separation_shield_margin_m,
                        inter_agent_safe_distance_m=args.inter_agent_safe_distance_m,
                        post_gate_cruise_min_pair_distance_m=args.post_gate_cruise_min_pair_distance_m,
                        lateral_spacing_m=args.lateral_spacing_m,
                        longitudinal_spacing_m=args.longitudinal_spacing_m,
                    )
                    summary = evaluate_checkpoint(
                        checkpoint_path=args.checkpoint,
                        episodes=int(args.episodes),
                        seed=int(args.seed) + int(gate_count) * 1000 + int(speed_scale * 100) * 10,
                        device=args.device,
                        num_agents=int(args.num_agents),
                        experiment_config=config,
                    )
                    row = {
                        "gate_count": int(gate_count),
                        "speed_scale": float(speed_scale),
                        "amplitude_scale": float(amplitude_scale),
                        "moving_gate_speed_mps": float(config.dynamic_gate_density.moving_gate_speed_mps),
                        "moving_gate_amplitude_m": float(config.dynamic_gate_density.moving_gate_amplitude_m),
                        "gate_post_radius_m": float(gate_post_radius_m),
                        "drone_radius_m": float(drone_radius_m),
                        "combined_radius_m": float(gate_post_radius_m + drone_radius_m),
                        "gate_half_width_m": float(config.dynamic_gate_density.gate_half_width_m),
                        "effective_opening_m": float(
                            2.0
                            * (
                                float(config.dynamic_gate_density.gate_half_width_m)
                                - float(gate_post_radius_m)
                                - float(drone_radius_m)
                            )
                        ),
                        "safety_clearance_m": float(config.environment.safety_clearance_m),
                        "inter_agent_safe_distance_m": float(config.environment.inter_agent_safe_distance_m),
                        "obstacle_shield_margin_m": float(
                            config.environment.action_safety_shield_obstacle_margin_m
                        ),
                        "separation_shield_margin_m": float(
                            config.environment.action_safety_shield_separation_margin_m
                        ),
                        "post_gate_cruise_min_pair_distance_m": float(
                            config.environment.action_safety_shield_post_gate_cruise_min_pair_distance_m
                        ),
                        "lateral_spacing_m": float(config.formation.lateral_spacing_m),
                        "longitudinal_spacing_m": float(config.formation.longitudinal_spacing_m),
                        "episodes": int(args.episodes),
                        "success_rate": float(summary.get("success_rate") or 0.0),
                        "team_success_rate": float(summary.get("team_success_rate") or 0.0),
                        "dynamic_gate_collision_rate": float(summary.get("dynamic_gate_collision_rate") or 0.0),
                        "agent_collision_rate": float(summary.get("agent_collision_rate") or 0.0),
                        "timeout_rate": float(summary.get("timeout_rate") or 0.0),
                        "corridor_through_success_rate": float(
                            summary.get("corridor_through_success_rate") or 0.0
                        ),
                        "mean_min_clearance_m": summary.get("mean_min_clearance_m"),
                        "min_min_clearance_m": summary.get("min_min_clearance_m"),
                        "mean_min_pair_distance_m": summary.get("mean_min_pair_distance_m"),
                        "mean_slot_error_m": summary.get("mean_slot_error_m"),
                        "formation_line_collapse_failure_rate": float(
                            summary.get("formation_line_collapse_failure_rate") or 0.0
                        ),
                        "mean_formation_lateral_band_count": summary.get("mean_formation_lateral_band_count"),
                        "min_formation_lateral_band_count": summary.get("min_formation_lateral_band_count"),
                        "mean_formation_line_collapse_score": summary.get(
                            "mean_formation_line_collapse_score"
                        ),
                        "path_length_m_mean": summary.get("path_length_m_mean"),
                        "flight_time_s_mean": summary.get("flight_time_s_mean"),
                        "done_reason_counts": json.dumps(
                            summary.get("done_reason_counts") or {}, ensure_ascii=False, sort_keys=True
                        ),
                    }
                    rows.append(row)
                    if fieldnames is None:
                        fieldnames = list(row.keys())
                        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
                            writer = csv.DictWriter(handle, fieldnames=fieldnames)
                            writer.writeheader()
                    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        writer.writerow(row)
                    with jsonl_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
                    print(
                        "gate={gate_count} speed_scale={speed_scale:.2f} "
                        "amp_scale={amplitude_scale:.2f} radii={gate_post_radius_m:.2f}+{drone_radius_m:.2f} "
                        "success={success_rate:.3f} dyn_collision={dynamic_gate_collision_rate:.3f} "
                        "clearance={mean_min_clearance_m}".format(**row),
                        flush=True,
                    )

    (args.output_dir / "difficulty_grid.json").write_text(
        json.dumps(_json_safe(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()

