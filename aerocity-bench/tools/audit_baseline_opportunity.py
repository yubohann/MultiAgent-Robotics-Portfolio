"""Audit whether public reference routes can ever observe private calibration targets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from aerocity_bench.baselines import BASELINES, create_baseline
from aerocity_bench.builder_v3 import validate_ordinary_release
from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json
from aerocity_bench.compiler import compile_method_task_spec
from aerocity_bench.contracts import ObservationPacket, Pose3D
from aerocity_bench.evaluator import PrivateEvaluator
from aerocity_bench.geometry import (
    colliders_from_city,
    distance,
    minimum_segment_clearance,
)
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config
from aerocity_bench.public_boundary import audit_public_layout
from aerocity_bench.runtime import L0FleetRuntime


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("authority_root", type=Path)
    parser.add_argument(
        "--derived-layout-root",
        type=Path,
        help=(
            "Local development-only compatibility layout derived from the authority root. "
            "Its DERIVATION_RECEIPT.json, source/derived file hashes, and recompiled "
            "public task contract are verified before the audit can run."
        ),
    )
    parser.add_argument("--split", choices=("train", "validation", "calibration"), required=True)
    parser.add_argument("--layout-id")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--method",
        action="append",
        choices=tuple(sorted(BASELINES)),
        required=True,
        help="One or more public reference methods to audit; repeat for multiple methods.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help="Positive L0 control-step cap for this development-only diagnostic.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.derived_layout_root is not None and args.layout_id is None:
        parser.error("--derived-layout-root requires an explicit --layout-id")
    if args.derived_layout_root is not None and args.episode_index != 0:
        parser.error("derived compatibility inputs only bind --episode-index 0")
    if any(BASELINES[method_id].requires_private_truth for method_id in args.method):
        parser.error("--method cannot select evaluator-private diagnostic methods")
    return args


def _selected_layout(index: dict[str, Any], split: str, layout_id: str | None) -> dict[str, Any]:
    if split in FORMAL_SPLITS:
        raise ValueError("baseline opportunity audit cannot disclose a formal split")
    candidates = [layout for layout in index["layouts"] if layout["split"] == split]
    if layout_id is not None:
        candidates = [layout for layout in candidates if layout["layout_id"] == layout_id]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one matching development layout, got {len(candidates)}")
    return candidates[0]


def _sha256(value: object, name: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 value")
    return text


def _receipt_hash(receipt: dict[str, Any]) -> str:
    declared = _sha256(receipt.get("receipt_hash"), "derivation receipt_hash")
    payload = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if content_hash(payload) != declared:
        raise ValueError("derivation receipt content hash mismatch")
    return declared


def _relative_layout_path(value: object, name: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a relative path below the release root")
    if len(path.parts) != 3 or path.parts[0] != "splits":
        raise ValueError(f"{name} must have the form splits/<development-split>/<layout-id>")
    if path.parts[1] in FORMAL_SPLITS:
        raise ValueError("derived compatibility inputs cannot reference a formal split")
    return path


def _load_derived_development_inputs(
    authority: Path,
    *,
    layout: dict[str, Any],
    split: str,
    episode_index: int,
    derived_root: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Load a receipt-bound local task-spec compatibility derivation.

    This narrow escape hatch exists because a historical development authority
    build predates the public safe-sky transit ABI used by current reference
    policies.  It cannot be used for a formal split, a formal score, or an
    arbitrary replacement layout.  The CitySpec and evaluator-private episode
    must be byte-identical to their authority-source counterparts; only the
    public task spec may be recompiled.
    """

    if split in FORMAL_SPLITS:
        raise ValueError("derived compatibility inputs cannot audit a formal split")
    if episode_index != 0:
        raise ValueError("derived compatibility inputs only bind episode index 0")
    source = authority.resolve()
    derived = derived_root.resolve()
    if derived == source:
        raise ValueError("derived compatibility root must be distinct from its authority source")
    receipt_path = derived / "DERIVATION_RECEIPT.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("derived compatibility root lacks DERIVATION_RECEIPT.json")
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("DERIVATION_RECEIPT.json must contain an object")
    if receipt.get("schema") != "org.aerocity.bench.cf2x-v4-derived-input.v1":
        raise ValueError("unsupported derived compatibility receipt schema")
    receipt_hash = _receipt_hash(receipt)
    if receipt.get("formal_score_eligible") is not False:
        raise ValueError("derived compatibility receipt must explicitly be non-formal")
    declared_source = Path(str(receipt.get("source_release_root", ""))).resolve()
    if declared_source != source:
        raise ValueError("derivation receipt source_release_root does not match authority_root")

    expected_relative = Path("splits") / split / str(layout["layout_id"])
    source_relative = _relative_layout_path(
        receipt.get("source_layout_relative_path"), "source_layout_relative_path"
    )
    derived_relative = _relative_layout_path(
        receipt.get("derived_layout_relative_path"), "derived_layout_relative_path"
    )
    if source_relative != expected_relative or derived_relative != expected_relative:
        raise ValueError(
            "derivation receipt layout binding differs from the selected development layout"
        )

    config_path = source / "authority_private" / "release_config.json"
    if _sha256(receipt.get("release_config_sha256"), "release_config_sha256") != file_hash(
        config_path
    ):
        raise ValueError("derivation receipt release-config hash mismatch")
    config = load_ordinary_config(config_path)
    source_layout_root = source / source_relative
    derived_layout_root = derived / derived_relative
    source_city_path = source_layout_root / "scene_authority" / "cityspec.json"
    derived_city_path = derived_layout_root / "scene_authority" / "cityspec.json"
    source_private_path = (
        source_layout_root / "evaluator_private" / "episodes" / "episode-0000.json"
    )
    derived_private_path = (
        derived_layout_root / "evaluator_private" / "episodes" / "episode-0000.json"
    )
    source_public_path = (
        source_layout_root / "method_public" / "episodes" / "episode-0000.json"
    )
    derived_public_path = (
        derived_layout_root / "method_public" / "episodes" / "episode-0000.json"
    )
    source_task_path = source_layout_root / "method_public" / "task_spec.json"
    derived_task_path = derived_layout_root / "method_public" / "task_spec.json"
    required = (
        source_city_path,
        derived_city_path,
        source_private_path,
        derived_private_path,
        source_public_path,
        derived_public_path,
        source_task_path,
        derived_task_path,
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("derived compatibility layout is missing a bound input file")
    city_hash = _sha256(receipt.get("cityspec_sha256_before_and_after"), "cityspec hash")
    if file_hash(source_city_path) != city_hash or file_hash(derived_city_path) != city_hash:
        raise ValueError("derived CitySpec is not byte-identical to the authority source")
    private_hash = _sha256(
        receipt.get("private_episode_0000_sha256_before_and_after"),
        "private episode-0000 hash",
    )
    if (
        file_hash(source_private_path) != private_hash
        or file_hash(derived_private_path) != private_hash
    ):
        raise ValueError(
            "derived evaluator-private episode is not byte-identical to the authority source"
        )
    if file_hash(source_public_path) != file_hash(derived_public_path):
        raise ValueError(
            "derived public episode must remain byte-identical to the authority source"
        )
    if _sha256(receipt.get("source_task_spec_sha256"), "source task-spec hash") != file_hash(
        source_task_path
    ):
        raise ValueError("derivation receipt source task-spec hash mismatch")
    if _sha256(receipt.get("derived_task_spec_sha256"), "derived task-spec hash") != file_hash(
        derived_task_path
    ):
        raise ValueError("derivation receipt derived task-spec hash mismatch")

    # Check byte-level derivation bindings before parsing the public task.  This
    # keeps a corrupted derived file attributable to the derivation receipt,
    # while the boundary audit below still rejects any semantically invalid
    # public artifact before a policy can consume it.
    audit_public_layout(source_layout_root)
    audit_public_layout(derived_layout_root)

    city = read_json(derived_city_path)
    task_spec = read_json(derived_task_path)
    expected_task_spec = compile_method_task_spec(
        city, config.raw["execution_contract"], config.raw["fleet"]
    )
    if task_spec != expected_task_spec:
        raise ValueError("derived public task spec is not the current deterministic recompilation")
    if not isinstance(task_spec.get("public_transit_contract"), dict):
        raise ValueError("derived public task spec lacks public_transit_contract")
    public_episode = read_json(derived_public_path)
    private_episode = read_json(derived_private_path)
    if str(private_episode.get("layout_id")) != str(layout["layout_id"]):
        raise ValueError("derived evaluator-private episode layout binding mismatch")
    if str(public_episode.get("episode_id")) != str(private_episode.get("episode_id")):
        raise ValueError("derived public/private episode identifiers differ")
    return (
        config,
        task_spec,
        public_episode,
        private_episode,
        city,
        {
            "derivation_receipt_hash": receipt_hash,
            "source_task_spec_sha256": file_hash(source_task_path),
            "derived_task_spec_sha256": file_hash(derived_task_path),
        },
    )


def _route_poses(policy: object) -> tuple[dict[str, list[Pose3D]], dict[str, set[int]]]:
    routes = getattr(policy, "routes", None)
    if not isinstance(routes, dict):
        raise TypeError("reference policy does not expose an auditable route")
    normalized = {str(drone_id): list(poses) for drone_id, poses in routes.items()}
    raw_observe_indices = getattr(policy, "observe_indices", None)
    if not isinstance(raw_observe_indices, dict):
        raise TypeError("reference policy does not expose observable route indices")
    observe_indices = {
        str(drone_id): {int(index) for index in indices}
        for drone_id, indices in raw_observe_indices.items()
    }
    if set(normalized) != set(observe_indices):
        raise ValueError("route and observable-index agents differ")
    return normalized, observe_indices


def _observation(episode_id: str, pose: Pose3D, index: int) -> ObservationPacket:
    return ObservationPacket(
        episode_id=episode_id,
        observation_id=f"audit-observation-{index:08d}",
        drone_id="audit-uav",
        sequence=index,
        timestamp_s=float(index),
        pose=pose,
        linear_velocity_world_mps=(0.0, 0.0, 0.0),
        angular_speed_deg_s=0.0,
        energy_remaining_j=1.0,
    )


def _execute_policy_with_trace(
    policy: object,
    *,
    config: Any,
    city: dict[str, Any],
    private_episode: dict[str, Any],
    public_task_spec: dict[str, Any] | None = None,
    public_episode: dict[str, Any] | None = None,
    max_steps: int | None = None,
) -> tuple[list[ObservationPacket], dict[str, Any]]:
    """Run the public policy and retain only observations it actually inspects."""

    if not callable(policy):
        raise TypeError("reference policy must be callable")
    runtime = L0FleetRuntime(
        config,
        city,
        private_episode,
        receipt_secret=b"baseline-opportunity-audit-only",
        public_task_spec=public_task_spec,
        public_episode=public_episode,
    )
    observations = runtime.reset()
    executed_observations: list[ObservationPacket] = []
    duration = float(config.raw["execution_contract"]["episode"]["duration_s"])
    period = float(config.raw["execution_contract"]["control_period_s"])
    limit = max_steps if max_steps is not None else math.ceil(duration / period)
    for _ in range(limit):
        actions = policy(observations)  # type: ignore[operator]
        for drone_id, action in actions.items():
            if action.kind == "OBSERVE":
                executed_observations.append(observations[drone_id])
        step = runtime.step(actions)
        observations = step.observations
        if step.done:
            break
    return executed_observations, runtime.result()


def _visibility_summary(
    observations: list[ObservationPacket],
    *,
    evaluator: PrivateEvaluator,
    private_episode: dict[str, Any],
) -> dict[str, Any]:
    visible_targets: set[str] = set()
    visible_pose_target_pairs = 0
    visibility_diagnostics: dict[str, int] = {}
    nearest_target_distances: list[float] = []
    nearest_witness_distances: list[float] = []
    for target in private_episode["targets"]:
        target_id = str(target["target_id"])
        target_position = tuple(float(value) for value in target["position"])
        witnesses = [Pose3D.from_dict(item["pose"]) for item in target["legal_witnesses"]]
        nearest_target_distances.append(
            min(
                (
                    distance(observation.pose.position, target_position)
                    for observation in observations
                ),
                default=math.inf,
            )
        )
        nearest_witness_distances.append(
            min(
                (
                    distance(observation.pose.position, witness.position)
                    for observation in observations
                    for witness in witnesses
                ),
                default=math.inf,
            )
        )
        for observation in observations:
            visible, reason = evaluator._visibility(  # noqa: SLF001 - evaluator-owner audit
                observation, target
            )
            visibility_diagnostics[reason] = visibility_diagnostics.get(reason, 0) + 1
            if visible:
                visible_targets.add(target_id)
                visible_pose_target_pairs += 1

    def distribution(values: list[float]) -> dict[str, float | None]:
        finite = sorted(value for value in values if math.isfinite(value))
        return {
            "minimum": round(finite[0], 6) if finite else None,
            "median": round(finite[len(finite) // 2], 6) if finite else None,
            "maximum": round(finite[-1], 6) if finite else None,
        }

    target_count = int(private_episode["target_count"])
    return {
        "observation_count": len(observations),
        "visible_target_count": len(visible_targets),
        "visible_target_fraction": round(
            len(visible_targets) / target_count if target_count else 0.0, 6
        ),
        "visible_pose_target_pair_count": visible_pose_target_pairs,
        "visibility_diagnostics": dict(sorted(visibility_diagnostics.items())),
        "nearest_target_distance_m": distribution(nearest_target_distances),
        "nearest_legal_witness_distance_m": distribution(nearest_witness_distances),
    }


def audit_method(
    method_id: str,
    *,
    config: Any,
    task_spec: dict[str, Any],
    public_episode: dict[str, Any],
    private_episode: dict[str, Any],
    city: dict[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("baseline opportunity audit max_steps must be positive")
    policy = create_baseline(method_id, config, task_spec, public_episode)
    routes, observe_indices = _route_poses(policy)
    colliders = colliders_from_city(city)
    vehicle = config.raw["execution_contract"]["vehicle"]
    safe_center_clearance = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    evaluator = PrivateEvaluator(config, city, private_episode, receipt_secret=b"audit-only-secret")
    starts = {
        str(start["drone_id"]): tuple(float(value) for value in start["position"])
        for start in public_episode["starts"]
    }

    route_pose_count = sum(len(route) for route in routes.values())
    blocked_segments = 0
    route_distance_m = 0.0
    for drone_id, route in routes.items():
        previous = starts[drone_id]
        for pose in route:
            route_distance_m += distance(previous, pose.position)
            clearance, _ = minimum_segment_clearance(previous, pose.position, colliders)
            if clearance + 1.0e-9 < safe_center_clearance:
                blocked_segments += 1
            previous = pose.position

    all_poses = [
        pose
        for drone_id, route in routes.items()
        for index, pose in enumerate(route)
        if index in observe_indices[drone_id]
    ]
    planned_observations = [
        _observation(private_episode["episode_id"], pose, pose_index)
        for pose_index, pose in enumerate(all_poses)
    ]
    planned = _visibility_summary(
        planned_observations,
        evaluator=evaluator,
        private_episode=private_episode,
    )
    executed_observations, run_result = _execute_policy_with_trace(
        policy,
        config=config,
        city=city,
        private_episode=private_episode,
        public_task_spec=task_spec,
        public_episode=public_episode,
        max_steps=max_steps,
    )
    executed = _visibility_summary(
        executed_observations,
        evaluator=evaluator,
        private_episode=private_episode,
    )
    executed.update(
        {
            "confirmed_target_count": len(run_result["confirmations"]),
            "collision_count": int(run_result["budget_ledger"]["collisions"]),
            "clearance_intervention_count": int(
                run_result["budget_ledger"]["clearance_interventions"]
            ),
            "out_of_bounds_action_count": int(
                run_result["budget_ledger"]["out_of_bounds_actions"]
            ),
            "failure_count": len(run_result["failures"]),
            "all_survivors_returned_home": all(run_result["returned_home"].values()),
            "task_time_s": round(float(run_result["task_time_s"]), 6),
        }
    )
    return {
        "method_id": method_id,
        "substantive_method": BASELINES[method_id].substantive_method,
        "observation_profile": BASELINES[method_id].observation_profile,
        "route_pose_count": route_pose_count,
        "observable_pose_count": len(all_poses),
        "route_distance_m_without_return": round(route_distance_m, 6),
        "blocked_direct_segment_count": blocked_segments,
        "blocked_direct_segment_fraction": round(
            blocked_segments / route_pose_count if route_pose_count else 0.0, 6
        ),
        "theoretically_visible_target_count": planned["visible_target_count"],
        "theoretically_visible_target_fraction": planned["visible_target_fraction"],
        "visible_route_pose_target_pair_count": planned["visible_pose_target_pair_count"],
        "nearest_legal_witness_distance_m": planned["nearest_legal_witness_distance_m"],
        "planned_full_route": planned,
        "executed_within_budget": executed,
    }


def main() -> int:
    args = _arguments()
    authority = args.authority_root.resolve()
    validation = validate_ordinary_release(authority)
    index = read_json(authority / "release_index.json")
    layout = _selected_layout(index, args.split, args.layout_id)
    derivation: dict[str, str] | None = None
    if args.derived_layout_root is not None:
        (
            config,
            task_spec,
            public_episode,
            private_episode,
            city,
            derivation,
        ) = _load_derived_development_inputs(
            authority,
            layout=layout,
            split=args.split,
            episode_index=args.episode_index,
            derived_root=args.derived_layout_root,
        )
    else:
        root = authority / "splits" / args.split / layout["layout_id"]
        public_path = root / "method_public" / "episodes" / f"episode-{args.episode_index:04d}.json"
        private_path = (
            root / "evaluator_private" / "episodes" / f"episode-{args.episode_index:04d}.json"
        )
        if not public_path.is_file() or not private_path.is_file():
            raise FileNotFoundError("requested paired development episode is absent")
        audit_public_layout(root)
        config = load_ordinary_config(authority / "authority_private" / "release_config.json")
        task_spec = read_json(root / "method_public" / "task_spec.json")
        public_episode = read_json(public_path)
        private_episode = read_json(private_path)
        city = read_json(root / "scene_authority" / "cityspec.json")
    methods = [
        audit_method(
            method_id,
            config=config,
            task_spec=task_spec,
            public_episode=public_episode,
            private_episode=private_episode,
            city=city,
            max_steps=args.max_steps,
        )
        for method_id in args.method
    ]
    substantive = [method for method in methods if method["substantive_method"]]
    execution_opportunity_gate = bool(substantive) and any(
        int(method["executed_within_budget"]["visible_target_count"]) > 0
        for method in substantive
    )
    gate_status = (
        "PASS"
        if execution_opportunity_gate
        else ("FAIL" if substantive else "NOT_APPLICABLE")
    )
    report = {
        "schema": "org.aerocity.bench.baseline-opportunity-audit.v2",
        "status": gate_status,
        "evidence_scope": "evaluator_owner_development_diagnostic_not_method_input_or_score",
        "formal_score_eligible": False,
        "input_mode": (
            "receipt_bound_local_development_task_spec_derivation"
            if derivation is not None
            else "authority_release_development_input"
        ),
        "authority_validation_status": validation["status"],
        "release_index_hash": index["release_index_hash"],
        "split": args.split,
        "layout_id": layout["layout_id"],
        "episode_id": private_episode["episode_id"],
        "target_count_private": private_episode["target_count"],
        "execution_step_limit": args.max_steps,
        "method_ids": list(args.method),
        "gates": {
            "substantive_method_has_executed_opportunity": {
                "status": gate_status,
                "detail": (
                    "at least one substantive public method observed a target "
                    "within the task budget"
                    if execution_opportunity_gate
                    else (
                        "no substantive public method was selected"
                        if not substantive
                        else "no substantive public method observed any target within the step cap"
                    )
                ),
            }
        },
        "methods": methods,
    }
    if derivation is not None:
        report["derivation"] = derivation
    report["report_hash"] = content_hash(report)
    write_json(args.output, report)
    return 0 if report["status"] in {"PASS", "NOT_APPLICABLE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
