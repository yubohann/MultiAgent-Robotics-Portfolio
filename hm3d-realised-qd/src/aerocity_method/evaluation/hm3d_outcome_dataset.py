"""Build a shared train-only index over real decision-level P07 outcomes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aerocity_method.contracts import FORMAL_FLEET_SIZE
from aerocity_method.contracts.io import canonical_sha256, read_json_object, require_sha256
from aerocity_method.contracts.hm3d_public_schema import require_current_public_schema
from aerocity_method.evaluation.hm3d_evidence_classification import (
    require_trainable_p07_outcome,
)
from aerocity_method.evaluation.hm3d_single_rl_training import (
    sample_from_p07_training_record,
    training_scene_ids_from_split_manifest,
)

OUTCOME_DATASET_SCHEMA_VERSION = "hm3d-real-decision-outcome-dataset-v4"


def _collection_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    require_current_public_schema(payload, context="outcome dataset worker record")
    contract = {
        "runner_version": payload.get("runner_version"),
        "fleet_size": payload.get("fleet_size"),
        "candidate_limit": payload.get("candidate_limit"),
        "action_budget_s": payload.get("action_budget_s"),
        "physics_dt_s": payload.get("physics_dt_s"),
        "arrival_tolerance_m": payload.get("arrival_tolerance_m"),
        "outcome_time_tolerance_s": payload.get("outcome_time_tolerance_s"),
        "cf2x_usd_sha256": payload.get("cf2x_usd_sha256"),
        "communication_contract_sha256": payload.get("communication_contract_sha256"),
        "sensor_profile_sha256": payload.get("sensor_profile_sha256"),
        "split_manifest_sha256": payload.get("split_manifest_sha256"),
        "transit_time_model_sha256": payload.get("transit_time_model_sha256"),
        "controller_id": payload.get("controller_id"),
        "action_completion_mode": payload.get("action_completion_mode"),
        "execution_profile_sha256": payload.get("execution_profile_sha256"),
        "candidate_pool_schema_version": payload.get("candidate_pool_schema_version"),
        "task_reservation_schema_version": payload.get("task_reservation_schema_version"),
        "evaluation_denominator_schema": _denominator_schema(payload),
    }
    if contract["fleet_size"] != FORMAL_FLEET_SIZE:
        raise ValueError("outcome dataset accepts only the frozen four-CF2X fleet")
    if not isinstance(contract["candidate_limit"], int) or contract["candidate_limit"] < FORMAL_FLEET_SIZE:
        raise ValueError("outcome dataset candidate_limit must support the frozen four-CF2X fleet")
    if not isinstance(contract["action_budget_s"], (int, float)) or float(contract["action_budget_s"]) <= 0.0:
        raise ValueError("outcome dataset action_budget_s must be positive")
    if contract["action_completion_mode"] != "event_driven_all_routes_completed_plus_minimum_dwell":
        raise ValueError("outcome dataset requires event-driven action completion")
    if not isinstance(contract["controller_id"], str) or not contract["controller_id"]:
        raise ValueError("outcome dataset requires a named frozen controller")
    for field in (
        "cf2x_usd_sha256",
        "communication_contract_sha256",
        "sensor_profile_sha256",
        "split_manifest_sha256",
        "transit_time_model_sha256",
        "execution_profile_sha256",
    ):
        require_sha256(contract[field], field)
    return contract


def _denominator_schema(payload: Mapping[str, Any]) -> str:
    denominator = payload.get("evaluation_denominator")
    if not isinstance(denominator, Mapping):
        raise ValueError("outcome dataset requires an episode evaluation denominator")
    if denominator.get("schema_version") != "hm3d-reachable-evaluation-denominator-v1":
        raise ValueError("outcome dataset requires the reachable-component denominator schema")
    return str(denominator["schema_version"])


def _episode_denominator_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep per-start reachability identity out of the cross-scene collection contract."""

    denominator = payload.get("evaluation_denominator")
    if not isinstance(denominator, Mapping):  # guarded above; preserve local type narrowing
        raise ValueError("outcome dataset requires an episode evaluation denominator")
    episode_hash = require_sha256(
        payload.get("evaluation_denominator_sha256"), "evaluation_denominator_sha256"
    )
    if denominator.get("denominator_sha256") != episode_hash:
        raise ValueError("episode denominator payload does not match its top-level hash")
    geometry_hash = require_sha256(
        payload.get("evaluation_geometry_denominator_sha256"),
        "evaluation_geometry_denominator_sha256",
    )
    if denominator.get("geometry_evaluation_denominator_sha256") != geometry_hash:
        raise ValueError("episode denominator does not bind its geometry-level denominator")
    component_ids = denominator.get("component_ids")
    if not isinstance(component_ids, list) or not component_ids:
        raise ValueError("episode denominator lacks reachable component provenance")
    return {
        "evaluation_denominator_sha256": episode_hash,
        "evaluation_geometry_denominator_sha256": geometry_hash,
        "reachable_component_ids": list(component_ids),
        "start_reset_manifest_sha256": require_sha256(
            denominator.get("start_reset_manifest_sha256"),
            "evaluation_denominator.start_reset_manifest_sha256",
        ),
    }


def build_outcome_dataset_manifest(
    record_paths: Sequence[Path], *, split_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and index unique real transitions without duplicating rollout payloads."""

    if not record_paths:
        raise ValueError("outcome dataset requires at least one P07 record")
    root = split_manifest.get("payload", split_manifest)
    if not isinstance(root, Mapping):
        raise ValueError("split manifest payload must be an object")
    split_manifest_sha256 = require_sha256(
        root.get("split_manifest_sha256"), "split_manifest_sha256"
    )
    allowed_scenes = training_scene_ids_from_split_manifest(root)
    rows: list[dict[str, Any]] = []
    transition_hashes: set[str] = set()
    record_hashes: set[str] = set()
    scene_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    episode_policy_keys: set[tuple[str, str, str]] = set()
    collection_contract: dict[str, Any] | None = None
    total_physics_s = 0.0
    total_wall_s = 0.0
    for raw_path in record_paths:
        path = raw_path.expanduser().resolve()
        payload = read_json_object(path)
        evidence_classification = require_trainable_p07_outcome(payload)
        samples = sample_from_p07_training_record(
            payload,
            allowed_train_scene_ids=allowed_scenes,
        )
        current_contract = _collection_contract(payload)
        if current_contract["split_manifest_sha256"] != split_manifest_sha256:
            raise ValueError("P07 rollout record split manifest differs from the supplied train split")
        denominator_identity = _episode_denominator_identity(payload)
        if collection_contract is None:
            collection_contract = current_contract
        elif current_contract != collection_contract:
            raise ValueError("P07 rollout records use different frozen collection contracts")
        record_hash = require_sha256(
            payload.get("runtime_record_sha256"), "runtime_record_sha256"
        )
        if record_hash in record_hashes:
            raise ValueError("duplicate P07 rollout record supplied to outcome dataset")
        record_hashes.add(record_hash)
        sample_hashes = [sample.transition_sha256 for sample in samples]
        overlap = transition_hashes.intersection(sample_hashes)
        if overlap:
            raise ValueError("duplicate decision transition supplied to outcome dataset")
        transition_hashes.update(sample_hashes)
        scene_id = samples[0].scene_id
        strategy = str(payload.get("strategy"))
        episode_policy_key = (scene_id, samples[0].public_episode_id, strategy)
        if episode_policy_key in episode_policy_keys:
            raise ValueError("duplicate scene/episode/behavior-policy rollout supplied")
        episode_policy_keys.add(episode_policy_key)
        scene_counts[scene_id] += len(samples)
        strategy_counts[strategy] += len(samples)
        elapsed_physics_s = float(payload.get("elapsed_physics_s", 0.0))
        runtime = payload.get("runtime_performance")
        wall_s = float(runtime.get("total_wall_s", 0.0)) if isinstance(runtime, Mapping) else 0.0
        total_physics_s += elapsed_physics_s
        total_wall_s += wall_s
        rows.append(
            {
                "path": str(path),
                "runtime_record_sha256": record_hash,
                "scene_id": scene_id,
                "strategy": strategy,
                "public_episode_id": samples[0].public_episode_id,
                "decision_count": len(samples),
                **denominator_identity,
                "transition_sha256": sample_hashes,
                "elapsed_physics_s": elapsed_physics_s,
                "wall_s": wall_s,
                "evidence_classification_sha256": evidence_classification[
                    "classification_sha256"
                ],
            }
        )
    rows.sort(key=lambda row: (row["scene_id"], row["public_episode_id"], row["strategy"]))
    manifest = {
        "schema_version": OUTCOME_DATASET_SCHEMA_VERSION,
        "claim_limit": (
            "Train-only shared real-decision index. The same physical transitions may support "
            "multiple offline gradient updates, but they count once in the interaction budget."
        ),
        "split_manifest_sha256": split_manifest_sha256,
        "collection_contract": collection_contract,
        "collection_contract_sha256": canonical_sha256(collection_contract),
        "physical_episode_count": len(rows),
        "real_decision_count": len(transition_hashes),
        "total_physics_s": total_physics_s,
        "total_wall_s": total_wall_s,
        "scene_decision_counts": dict(sorted(scene_counts.items())),
        "behavior_policy_decision_counts": dict(sorted(strategy_counts.items())),
        "records": rows,
    }
    manifest["dataset_sha256"] = canonical_sha256(manifest)
    return manifest


__all__ = ["OUTCOME_DATASET_SCHEMA_VERSION", "build_outcome_dataset_manifest"]
