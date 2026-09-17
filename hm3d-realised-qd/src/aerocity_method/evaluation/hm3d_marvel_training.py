"""Outcome-only training ingestion for the MARVEL-style HM3D transfer."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aerocity_method.adapters.hm3d_marvel import (
    MARVEL_AUTHOR_MODEL_COMMIT,
    MARVEL_SUPPLEMENTARY_REFERENCE_FEATURE_SCHEMA_VERSION,
    MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_TRAINING_TRANSITION_KEY,
    MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_KEY,
    MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_SCHEMA_VERSION,
    MarvelSupplementaryReferenceConfig,
    MarvelSupplementaryReferencePolicy,
    MarvelSupplementaryReferenceTrainingRow,
    build_marvel_checkpoint_payload,
)
from aerocity_method.adapters.hm3d_marvel_author_sac import (
    marvel_graph_observation_from_dict,
)
from aerocity_method.contracts.hm3d_public_schema import (
    public_schema_fields,
    require_current_public_schema,
)
from aerocity_method.contracts.io import finite_number, require_identifier
from aerocity_method.evaluation.hm3d_p07_matrix import P07ProbeRecord

MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_PROVENANCE_SCHEMA_VERSION = (
    "hm3d-marvel-author-sac-training-v3"
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_id(value: Any, name: str) -> str:
    return require_identifier(value, name)


def _vector(value: Any, name: str, width: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{name} must contain {width} values")
    return tuple(finite_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _agents(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("agent_features must be a non-empty list")
    return tuple(_vector(row, f"agent_features[{index}]", 5) for index, row in enumerate(value))


def _adjacency(value: Any, count: int) -> tuple[tuple[bool, ...], ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("communication_adjacency row count differs from agent count")
    rows = tuple(tuple(row) for row in value)
    if any(len(row) != count or any(not isinstance(item, bool) for item in row) for row in rows):
        raise ValueError("communication_adjacency must be a square boolean matrix")
    if any(not rows[index][index] for index in range(count)):
        raise ValueError("communication_adjacency must contain self loops")
    return rows


def _candidates(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("candidate_features must be a non-empty list")
    return tuple(_vector(row, f"candidate_features[{index}]", 8) for index, row in enumerate(value))


@dataclass(frozen=True, slots=True)
class MarvelSupplementaryReferenceTrainingSample:
    raw_record_id: str
    scene_id: str
    public_episode_id: str
    decision_id: str
    transition_id: str
    done: bool
    duration_s: float
    row: MarvelSupplementaryReferenceTrainingRow

    def __post_init__(self) -> None:
        _require_id(self.raw_record_id, "raw_record_id")
        require_identifier(self.scene_id, "scene_id")
        require_identifier(self.public_episode_id, "public_episode_id")
        require_identifier(self.decision_id, "decision_id")
        _require_id(self.transition_id, "transition_id")
        if not isinstance(self.done, bool) or finite_number(self.duration_s, "duration_s") <= 0.0:
            raise ValueError(
                "MARVEL supplementary reference sample terminal flag or duration is invalid"
            )
        if not isinstance(self.row, MarvelSupplementaryReferenceTrainingRow):
            raise TypeError("row must be a MarvelSupplementaryReferenceTrainingRow")


def training_scene_ids_from_split_manifest(payload: Mapping[str, Any]) -> tuple[str, ...]:
    root = _mapping(payload, "split manifest")
    if "payload" in root:
        root = _mapping(root["payload"], "split manifest payload")
    assignments = root.get("scene_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("split manifest lacks scene_assignments")
    normalized: list[dict[str, str]] = []
    train: list[str] = []
    for item in assignments:
        row = _mapping(item, "scene assignment")
        scene = require_identifier(row.get("scene_id"), "scene_id")
        split = row.get("split")
        asset = _require_id(row.get("asset_id"), "asset_id")
        if split not in {"train", "validation", "test"}:
            raise ValueError("scene assignment has an invalid split")
        normalized.append({"scene_id": scene, "split": split, "asset_id": asset})
        if split == "train":
            train.append(scene)
    if not train:
        raise ValueError("split manifest has no train scenes")
    return tuple(sorted(train))


def sample_from_p07_training_record(
    payload: Mapping[str, Any], *, allowed_train_scene_ids: Sequence[str]
) -> tuple[MarvelSupplementaryReferenceTrainingSample, ...]:
    require_current_public_schema(payload, context="MARVEL P07 worker record")
    strategy = str(payload.get("strategy"))
    probe = P07ProbeRecord.from_raw(strategy, payload)
    if probe.partition != "train" or probe.scene_id not in set(allowed_train_scene_ids):
        raise ValueError("MARVEL supplementary reference training accepts only frozen train scenes")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("MARVEL supplementary reference training record has no executed decision")
    if (
        MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_KEY in payload
        and MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_TRAINING_TRANSITION_KEY in payload
    ):
        raise ValueError(
            "MARVEL supplementary reference record must not contain both current and legacy "
            "transition keys"
        )
    emitted_key = MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_KEY
    emitted_rows = payload.get(emitted_key)
    if emitted_rows is None:
        emitted_key = MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_TRAINING_TRANSITION_KEY
        emitted_rows = payload.get(emitted_key)
    if not isinstance(emitted_rows, list) or len(emitted_rows) != len(decisions):
        raise ValueError(
            f"MARVEL supplementary reference {emitted_key} count must equal executed decisions"
        )
    samples: list[MarvelSupplementaryReferenceTrainingSample] = []
    reward_sum = 0.0
    for index, (raw_emitted, raw_decision) in enumerate(
        zip(emitted_rows, decisions, strict=True)
    ):
        emitted = _mapping(raw_emitted, f"{emitted_key}[{index}]")
        decision = _mapping(raw_decision, f"decisions[{index}]")
        require_current_public_schema(decision, context=f"decisions[{index}]")
        require_current_public_schema(
            emitted, context=f"{emitted_key}[{index}]"
        )
        if (
            emitted.get("schema_version")
            != MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_SCHEMA_VERSION
        ):
            raise ValueError("MARVEL supplementary reference transition schema mismatch")
        if emitted.get("author_model_commit") != MARVEL_AUTHOR_MODEL_COMMIT:
            raise ValueError("MARVEL supplementary reference transition author source mismatch")
        supplied_id = _require_id(emitted.get("transition_id"), "transition_id")
        decision_id = require_identifier(decision.get("decision_id"), "decision_id")
        if emitted.get("decision_id") != decision_id or emitted.get("scene_id") != probe.scene_id:
            raise ValueError("MARVEL supplementary reference decision or scene binding differs")
        for field, expected in public_schema_fields().items():
            if emitted.get(field) != decision.get(field) or emitted.get(field) != expected:
                raise ValueError(
                    "MARVEL supplementary reference public-task schema differs from execution"
                )
        # The episode anchor precedes bootstrap sensing.  The learning state is the
        # source-bound decisions[0] state, which is checked above.
        selection = _mapping(decision.get("selection"), "decision selection")
        execution = _mapping(decision.get("execution"), "decision execution")
        if emitted.get("selected_candidate_id") != selection.get("selected_candidate_id"):
            raise ValueError(
                "MARVEL supplementary reference candidate differs from executed decision"
            )
        if emitted.get("selected_manifest_id") != selection.get("selected_manifest_id"):
            raise ValueError(
                "MARVEL supplementary reference manifest differs from executed decision"
            )
        outcome_ids = execution.get("outcome_ids")
        if not isinstance(outcome_ids, list) or not outcome_ids:
            raise ValueError(
                "MARVEL supplementary reference transition has no execution outcome ids"
            )
        candidates = _candidates(emitted.get("candidate_features"))
        legal = tuple(emitted.get("legal_mask", ()))
        if len(legal) != len(candidates) or not any(legal):
            raise ValueError("MARVEL supplementary reference transition legal mask is malformed")
        action = emitted.get("selected_action_index")
        if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < len(legal):
            raise ValueError(
                "MARVEL supplementary reference selected action is outside the candidate pool"
            )
        if not legal[action]:
            raise ValueError("MARVEL supplementary reference selected action is illegal")
        reward = finite_number(
            emitted.get("reward_explored_free_flight_volume_auc_time_contribution"),
            "MARVEL supplementary reference AUC contribution",
        )
        if reward != finite_number(
            decision.get("reward_explored_free_flight_volume_auc_time_contribution"),
            "decision AUC contribution",
        ):
            raise ValueError(
                "MARVEL supplementary reference reward differs from actual decision contribution"
            )
        reward_sum += reward
        duration = finite_number(
            emitted.get("duration_s"), "MARVEL supplementary reference duration"
        )
        if duration != finite_number(decision.get("duration_s"), "decision duration"):
            raise ValueError("MARVEL supplementary reference duration differs from actual decision")
        is_final = index == len(decisions) - 1
        expected_terminated = (
            is_final and probe.terminal_outcome == "executed_terminal_safety_failure"
        )
        expected_truncated = is_final and probe.terminal_outcome == "budget_exhausted"
        terminated = emitted.get("terminated")
        truncated = emitted.get("truncated")
        if terminated is not expected_terminated or truncated is not expected_truncated:
            raise ValueError(
                "MARVEL supplementary reference terminal cause differs from P07 outcome"
            )
        done = terminated or truncated
        if not done:
            next_emitted = _mapping(emitted_rows[index + 1], "next MARVEL transition")
            if emitted.get("next_public_context_id") != next_emitted.get(
                "public_context_id"
            ) or emitted.get("next_public_candidate_pool_id") != next_emitted.get(
                "public_candidate_pool_id"
            ):
                raise ValueError(
                    "MARVEL supplementary reference next state is not the next real decision"
                )
        agent_features = _agents(emitted.get("agent_features"))
        _adjacency(emitted.get("communication_adjacency"), len(agent_features))
        _candidates(emitted.get("candidate_features"))
        graph = marvel_graph_observation_from_dict(
            _mapping(emitted.get("marvel_graph_observation"), "marvel_graph_observation")
        )
        next_graph = marvel_graph_observation_from_dict(
            _mapping(
                emitted.get("next_marvel_graph_observation"),
                "next_marvel_graph_observation",
            )
        )
        if graph.legal_mask != legal:
            raise ValueError("MARVEL graph action mask differs from the executed candidate pool")
        samples.append(
            MarvelSupplementaryReferenceTrainingSample(
                raw_record_id=probe.raw_record_id,
                scene_id=probe.scene_id,
                public_episode_id=probe.public_episode_id,
                decision_id=decision_id,
                transition_id=supplied_id,
                done=done,
                duration_s=duration,
                row=MarvelSupplementaryReferenceTrainingRow(
                    observation=graph,
                    next_observation=next_graph,
                    action=action,
                    reward=reward,
                    duration_s=duration,
                    done=done,
                ),
            )
        )
    if abs(reward_sum - probe.explored_free_flight_volume_auc_time) > 1.0e-9:
        raise ValueError(
            "MARVEL supplementary reference decision rewards do not reproduce episode AUC"
        )
    return tuple(samples)


def train_marvel_supplementary_reference_baseline(
    samples: Sequence[MarvelSupplementaryReferenceTrainingSample],
    *,
    split_manifest_id: str,
    updates: int,
    hidden_dim: int,
    seed: int,
    minimum_transitions: int,
    minimum_scenes: int,
    batch_size: int = 256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if updates < 1 or minimum_transitions < 1 or minimum_scenes < 1 or batch_size < 1:
        raise ValueError("MARVEL supplementary reference training counts must be positive")
    ordered = tuple(sorted(samples, key=lambda row: row.transition_id))
    if len(ordered) < minimum_transitions:
        raise ValueError(
            "not enough real decision transitions for MARVEL supplementary reference "
            "training"
        )
    if len({row.transition_id for row in ordered}) != len(ordered):
        raise ValueError("duplicate MARVEL supplementary reference decision transition")
    scenes = tuple(sorted({row.scene_id for row in ordered}))
    if len(scenes) < minimum_scenes:
        raise ValueError(
            "MARVEL supplementary reference training needs the requested number of "
            "scenes"
        )
    model = MarvelSupplementaryReferencePolicy(
        MarvelSupplementaryReferenceConfig(hidden_dim=hidden_dim), seed=seed
    )
    diagnostics: dict[str, float] = {}
    rows = tuple(sample.row for sample in ordered)
    sampler = random.Random(seed)
    for _ in range(updates):
        batch = (
            rows
            if len(rows) <= batch_size
            else tuple(rows[index] for index in sampler.sample(range(len(rows)), batch_size))
        )
        diagnostics = model.update(batch)
    provenance: dict[str, Any] = {
        "schema_version": MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_PROVENANCE_SCHEMA_VERSION,
        "claim_limit": (
            "Train-only MARVEL author PolicyNet/QNet and discrete SAC migration. All model "
            "inputs are public sparse-range candidate graphs and all rewards are actual "
            "decision-level outcome scores."
        ),
        "training_partition": "train",
        "split_manifest_id": _require_id(split_manifest_id, "split_manifest_id"),
        "feature_schema_version": MARVEL_SUPPLEMENTARY_REFERENCE_FEATURE_SCHEMA_VERSION,
        **public_schema_fields(),
        "training_scene_ids": list(scenes),
        "episode_count": len({(row.scene_id, row.public_episode_id) for row in ordered}),
        "transition_count": len(ordered),
        "rollout_record_file_id": sorted({row.raw_record_id for row in ordered}),
        "transition_id": [row.transition_id for row in ordered],
        "updates": updates,
        "batch_size": min(batch_size, len(rows)),
        "seed": seed,
        "model": {
            "hidden_dim": hidden_dim,
            "author_model_commit": MARVEL_AUTHOR_MODEL_COMMIT,
            "policy_net": "author",
            "twin_q": True,
            "target_q": True,
            "learned_temperature": True,
            "duration_discount": True,
            "qd": False,
            "ogfr": False,
        },
        "aggregate_training_diagnostics": diagnostics,
    }
    checkpoint = build_marvel_checkpoint_payload(
        model,
        training_scene_ids=scenes,
        training_updates=updates,
        training_provenance=provenance,
        split_manifest_id=split_manifest_id,
    )
    return checkpoint, provenance


__all__ = [
    "MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_PROVENANCE_SCHEMA_VERSION",
    "MarvelSupplementaryReferenceTrainingSample",
    "sample_from_p07_training_record",
    "train_marvel_supplementary_reference_baseline",
    "training_scene_ids_from_split_manifest",
]
