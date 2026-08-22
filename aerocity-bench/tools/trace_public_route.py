"""Trace public baseline route-state transitions on a development episode."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from aerocity_bench.baselines import BASELINES, create_baseline
from aerocity_bench.builder_v3 import validate_ordinary_release
from aerocity_bench.canonical import read_json, write_json
from aerocity_bench.geometry import colliders_from_city
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config
from aerocity_bench.public_boundary import audit_public_layout
from aerocity_bench.runtime import L0FleetRuntime


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("authority_root", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "calibration"), required=True)
    parser.add_argument("--layout-id")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--method", choices=tuple(BASELINES), required=True)
    parser.add_argument("--drone", required=True)
    parser.add_argument("--route-index", type=int, required=True)
    parser.add_argument("--index-radius", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--disable-local-scan-block", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _selected_layout(index: dict[str, Any], split: str, layout_id: str | None) -> dict[str, Any]:
    if split in FORMAL_SPLITS:
        raise ValueError("public route tracing cannot disclose a formal split")
    candidates = [layout for layout in index["layouts"] if layout["split"] == split]
    if layout_id is not None:
        candidates = [layout for layout in candidates if layout["layout_id"] == layout_id]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one matching development layout, got {len(candidates)}")
    return candidates[0]


def _route_state(policy: Any, drone_id: str) -> dict[str, Any]:
    index = int(policy.indices[drone_id])
    route = policy.routes[drone_id]
    pose = route[index].to_dict() if index < len(route) else None
    return {
        "route_index": index,
        "route_pose": pose,
        "is_observe_index": index in policy.observe_indices[drone_id],
        "observe_remaining": int(policy.observe_remaining[drone_id]),
        "scan_index_refined": index in policy.refined_scan_indices[drone_id],
        "stalled_steps": int(policy.stalled_steps[drone_id]),
        "return_phase": str(policy.return_phases[drone_id]),
    }


def main() -> int:
    args = _arguments()
    if args.method == "centralized-oracle":
        raise ValueError("trace_public_route accepts public baselines only")
    authority = args.authority_root.resolve()
    validate_ordinary_release(authority)
    index = read_json(authority / "release_index.json")
    layout = _selected_layout(index, args.split, args.layout_id)
    root = authority / "splits" / args.split / layout["layout_id"]
    audit_public_layout(root)
    config = load_ordinary_config(authority / "authority_private" / "release_config.json")
    task_spec = read_json(root / "method_public" / "task_spec.json")
    public_episode = read_json(
        root / "method_public" / "episodes" / f"episode-{args.episode_index:04d}.json"
    )
    private_episode = read_json(
        root / "evaluator_private" / "episodes" / f"episode-{args.episode_index:04d}.json"
    )
    city = read_json(root / "scene_authority" / "cityspec.json")
    policy: Any = create_baseline(args.method, config, task_spec, public_episode)
    if args.drone not in policy.routes:
        raise ValueError(f"unknown policy drone: {args.drone}")
    initial_route_window = [
        {"route_index": index, "pose": policy.routes[args.drone][index].to_dict()}
        for index in range(
            max(0, args.route_index - args.index_radius),
            min(len(policy.routes[args.drone]), args.route_index + args.index_radius + 1),
        )
    ]
    if args.disable_local_scan_block:
        policy._local_occupancy_blocks_scan = lambda observation, target: False
    runtime = L0FleetRuntime(
        config,
        city,
        private_episode,
        receipt_secret=b"public-route-development-trace-only",
        public_task_spec=task_spec,
        public_episode=public_episode,
    )
    observations = runtime.reset()
    authority_colliders = colliders_from_city(city)
    period = float(config.raw["execution_contract"]["control_period_s"])
    duration = float(config.raw["execution_contract"]["episode"]["duration_s"])
    limit = args.max_steps or math.ceil(duration / period)
    low = max(0, args.route_index - args.index_radius)
    high = args.route_index + args.index_radius
    trace: list[dict[str, Any]] = []
    previous_index = int(policy.indices[args.drone])
    for step_index in range(limit):
        before = _route_state(policy, args.drone)
        observation = observations[args.drone]
        before_index = int(before["route_index"])
        before_target = (
            policy.routes[args.drone][before_index]
            if before_index < len(policy.routes[args.drone])
            else None
        )
        local_boxes = policy._local_occupancy_boxes(observation)
        local_point_clearance = (
            min(
                (box.point_distance(before_target.position) for box in local_boxes),
                default=math.inf,
            )
            if before_target is not None
            else math.inf
        )
        authority_point_clearance = (
            min(
                (box.point_distance(before_target.position) for box in authority_colliders),
                default=math.inf,
            )
            if before_target is not None
            else math.inf
        )
        local_precheck_blocks = bool(
            before_target is not None
            and policy._local_occupancy_blocks_scan(observation, before_target)
        )
        actions = policy(observations)
        after_policy = _route_state(policy, args.drone)
        action = actions[args.drone]
        step = runtime.step(actions)
        receipt = next(item for item in step.execution_receipts if item.drone_id == args.drone)
        after_step = _route_state(policy, args.drone)
        current_indices = {
            int(before["route_index"]),
            int(after_policy["route_index"]),
            int(after_step["route_index"]),
            previous_index,
        }
        if any(low <= route_index <= high for route_index in current_indices):
            trace.append(
                {
                    "step_index": step_index,
                    "timestamp_s": observation.timestamp_s,
                    "source_observation_id": observation.observation_id,
                    "observation_pose": observation.pose.to_dict(),
                    "observation_linear_velocity_world_mps": list(
                        observation.linear_velocity_world_mps
                    ),
                    "observation_angular_speed_deg_s": observation.angular_speed_deg_s,
                    "before_target_local_point_clearance_m": local_point_clearance,
                    "before_target_authority_point_clearance_m": authority_point_clearance,
                    "local_precheck_blocks_before_policy": local_precheck_blocks,
                    "before_policy": before,
                    "after_policy": after_policy,
                    "action": action.to_dict(),
                    "execution_receipt": receipt.to_dict(),
                    "after_step": after_step,
                }
            )
        previous_index = int(after_step["route_index"])
        observations = step.observations
        if step.done:
            break
    report = {
        "schema": "org.aerocity.bench.public-route-trace.v1",
        "evidence_scope": "development_public_route_state_no_private_target_output",
        "formal_score_eligible": False,
        "authority_root": str(authority),
        "split": args.split,
        "layout_id": layout["layout_id"],
        "episode_index": args.episode_index,
        "method_id": args.method,
        "drone_id": args.drone,
        "requested_route_index": args.route_index,
        "local_scan_block_disabled": args.disable_local_scan_block,
        "initial_route_window": initial_route_window,
        "trace": trace,
    }
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
