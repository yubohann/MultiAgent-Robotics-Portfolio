"""Fail-closed training ingestion for the ordinary P07 single-RL baseline.

This module intentionally trains only the archive-free, fragment-free weak
baseline.  It consumes every decision-level transition emitted by an actual
CF2X rollout, rather than one episode aggregate attached to the first action.
The actor still receives only target-free public candidate features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aerocity_method.adapters.hm3d_single_rl import (
    SINGLE_RL_FEATURE_SCHEMA_VERSION,
    SINGLE_RL_TRAINING_TRANSITION_SCHEMA_VERSION,
    build_single_rl_checkpoint_payload,
)
from aerocity_method.contracts.io import canonical_sha256, finite_number, require_identifier
from aerocity_method.contracts.hm3d_public_schema import (
    public_schema_fields,
    require_current_public_schema,
)
from aerocity_method.evaluation.hm3d_evidence_classification import (
    require_trainable_p07_outcome,
)
from aerocity_method.evaluation.hm3d_p07_matrix import P07ProbeRecord
from aerocity_method.learning.rb_sf_sac import RBSFSAC, RBSFSACConfig
from aerocity_method.learning.replay import CandidateTransition

SINGLE_RL_TRAINING_PROVENANCE_SCHEMA_VERSION = "hm3d-p07-single-rl-training-v2"
_CONTEXT_DIM = 4
_CANDIDATE_DIM = 5


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    int(value, 16)
    return value


def _finite_vector(value: Any, name: str, *, width: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{name} must contain exactly {width} numeric features")
    return tuple(finite_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _candidate_matrix(value: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("candidate_features must be a non-empty list")
    return tuple(
        _finite_vector(row, f"candidate_features[{index}]", width=_CANDIDATE_DIM)
        for index, row in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class SingleRLTrainingSample:
    """One real decision transition with no evaluator geometry."""

    raw_record_sha256: str
    scene_id: str
    public_episode_id: str
    decision_id: str
    transition_sha256: str
    transition: CandidateTransition

    def __post_init__(self) -> None:
        _sha(self.raw_record_sha256, "raw_record_sha256")
        require_identifier(self.scene_id, "scene_id")
        require_identifier(self.public_episode_id, "public_episode_id")
        require_identifier(self.decision_id, "decision_id")
        _sha(self.transition_sha256, "transition_sha256")
        if not isinstance(self.transition, CandidateTransition):
            raise TypeError("transition must be a CandidateTransition")


def training_scene_ids_from_split_manifest(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the immutable P05 train-scene set and reject malformed manifests."""

    root = _mapping(payload, "split manifest")
    if "payload" in root:
        root = _mapping(root["payload"], "split manifest payload")
    assignments = root.get("scene_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("P05 split manifest lacks scene_assignments")
    expected_hash = _sha(root.get("split_manifest_sha256"), "split_manifest_sha256")
    normalized: list[dict[str, str]] = []
    train_ids: list[str] = []
    for index, item in enumerate(assignments):
        row = _mapping(item, f"scene_assignments[{index}]")
        scene_id = require_identifier(row.get("scene_id"), "scene_id")
        split = row.get("split")
        asset_sha256 = _sha(row.get("asset_sha256"), "asset_sha256")
        if split not in {"train", "validation", "test"}:
            raise ValueError("scene assignment has an unknown partition")
        normalized.append({"scene_id": scene_id, "split": split, "asset_sha256": asset_sha256})
        if split == "train":
            train_ids.append(scene_id)
    if canonical_sha256(sorted(normalized, key=lambda row: row["scene_id"])) != expected_hash:
        raise ValueError("P05 split_manifest_sha256 does not match scene_assignments")
    if not train_ids:
        raise ValueError("P05 split manifest contains no train scenes")
    return tuple(sorted(train_ids))


def sample_from_p07_training_record(
    payload: Mapping[str, Any], *, allowed_train_scene_ids: Sequence[str]
) -> tuple[SingleRLTrainingSample, ...]:
    """Validate one worker record and reconstruct every real SAC transition."""

    require_trainable_p07_outcome(payload)
    require_current_public_schema(payload, context="single-RL P07 worker record")
    probe = P07ProbeRecord.from_raw(str(payload.get("strategy")), payload)
    if probe.partition != "train":
        raise ValueError("single-RL training accepts only train-partition worker records")
    if probe.scene_id not in set(allowed_train_scene_ids):
        raise ValueError("worker scene is absent from the frozen P05 train split")
    raw = _mapping(payload, "worker record")
    decisions = raw.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("single-RL training record has no executed decision")
    emitted_rows = raw.get("single_rl_training_transitions")
    if not isinstance(emitted_rows, list) or len(emitted_rows) != len(decisions):
        raise ValueError("single-RL transition count must equal the executed decision count")
    samples: list[SingleRLTrainingSample] = []
    reward_sum = 0.0
    for index, (raw_emitted, raw_decision) in enumerate(
        zip(emitted_rows, decisions, strict=True)
    ):
        emitted = _mapping(raw_emitted, f"single_rl_training_transitions[{index}]")
        decision = _mapping(raw_decision, f"decisions[{index}]")
        require_current_public_schema(decision, context=f"decisions[{index}]")
        require_current_public_schema(
            emitted, context=f"single_rl_training_transitions[{index}]"
        )
        if emitted.get("schema_version") != SINGLE_RL_TRAINING_TRANSITION_SCHEMA_VERSION:
            raise ValueError("single-RL training transition schema mismatch")
        supplied_hash = _sha(emitted.get("transition_sha256"), "transition_sha256")
        unsigned = dict(emitted)
        unsigned.pop("transition_sha256", None)
        if canonical_sha256(unsigned) != supplied_hash:
            raise ValueError("single-RL training transition content hash mismatch")
        decision_id = require_identifier(decision.get("decision_id"), "decision_id")
        if emitted.get("decision_id") != decision_id or emitted.get("scene_id") != probe.scene_id:
            raise ValueError("training transition decision or scene binding differs")
        context_hash = _sha(emitted.get("public_context_hash"), "public_context_hash")
        pool_hash = _sha(emitted.get("public_candidate_pool_hash"), "candidate pool hash")
        if context_hash != decision.get("public_context_hash"):
            raise ValueError("training transition context differs from its executed decision")
        if pool_hash != decision.get("public_candidate_pool_hash"):
            raise ValueError("training transition pool differs from its executed decision")
        for field, expected in public_schema_fields().items():
            if emitted.get(field) != decision.get(field) or emitted.get(field) != expected:
                raise ValueError("training transition public-task schema differs from execution")
        # The top-level public context is the pre-bootstrap audit anchor.  Training is
        # decision-level, so its first state is bound to decisions[0], after bootstrap
        # observations and communication have updated the public belief.
        selection = _mapping(decision.get("selection"), f"decisions[{index}].selection")
        execution = _mapping(decision.get("execution"), f"decisions[{index}].execution")
        if emitted.get("selected_candidate_id") != selection.get("selected_candidate_id"):
            raise ValueError("training transition selected candidate differs from execution")
        if emitted.get("selected_manifest_hash") != selection.get("selected_manifest_hash"):
            raise ValueError("training transition selected manifest differs from execution")
        context = _finite_vector(emitted.get("context_features"), "context_features", width=4)
        candidates = _candidate_matrix(emitted.get("candidate_features"))
        legal = tuple(emitted.get("legal_mask", ()))
        if len(legal) != len(candidates) or any(not isinstance(item, bool) for item in legal):
            raise ValueError("legal_mask must match candidate_features")
        action = emitted.get("selected_action_index")
        if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < len(legal):
            raise ValueError("selected_action_index is outside the public candidate pool")
        if not legal[action]:
            raise ValueError("training transition selects an illegal candidate")
        reward = finite_number(
            emitted.get("reward_explored_free_flight_volume_auc_time_contribution"),
            "transition AUC contribution",
        )
        expected_reward = finite_number(
            decision.get("reward_explored_free_flight_volume_auc_time_contribution"),
            "decision AUC contribution",
        )
        if not 0.0 <= reward <= 1.0 or reward != expected_reward:
            raise ValueError("training reward differs from its actual decision AUC contribution")
        reward_sum += reward
        cost = finite_number(emitted.get("cost_energy_j"), "transition cost")
        if cost != finite_number(execution.get("total_energy_used_j"), "actual energy"):
            raise ValueError("training cost differs from its actual decision energy")
        outcome_hash = _sha(emitted.get("outcome_hash"), "transition outcome_hash")
        expected_outcome_hash = canonical_sha256(
            {
                "manifest_hash": selection.get("selected_manifest_hash"),
                "outcome_hashes": execution.get("outcome_hashes"),
            }
        )
        if outcome_hash != expected_outcome_hash:
            raise ValueError("training outcome identity differs from actual execution")
        duration = finite_number(emitted.get("duration_s"), "duration_s")
        if duration != finite_number(decision.get("duration_s"), "decision duration_s"):
            raise ValueError("training duration differs from its actual decision duration")
        is_final = index == len(decisions) - 1
        expected_terminated = (
            is_final and probe.terminal_outcome == "executed_terminal_safety_failure"
        )
        expected_truncated = is_final and probe.terminal_outcome == "budget_exhausted"
        terminated = emitted.get("terminated")
        truncated = emitted.get("truncated")
        if terminated is not expected_terminated or truncated is not expected_truncated:
            raise ValueError("training transition terminal cause differs from P07 outcome")
        done = terminated or truncated
        next_context = _finite_vector(
            emitted.get("next_context_features"), "next_context_features", width=4
        )
        next_candidates = _candidate_matrix(emitted.get("next_candidate_features"))
        next_legal = tuple(emitted.get("next_legal_mask", ()))
        if len(next_legal) != len(next_candidates) or any(
            not isinstance(item, bool) for item in next_legal
        ):
            raise ValueError("next_legal_mask must match next_candidate_features")
        if not done:
            next_emitted = _mapping(
                emitted_rows[index + 1], f"single_rl_training_transitions[{index + 1}]"
            )
            if emitted.get("next_public_context_hash") != next_emitted.get(
                "public_context_hash"
            ) or emitted.get("next_public_candidate_pool_hash") != next_emitted.get(
                "public_candidate_pool_hash"
            ):
                raise ValueError("training transition next state is not the next real decision")
        samples.append(
            SingleRLTrainingSample(
                raw_record_sha256=probe.raw_record_sha256,
                scene_id=probe.scene_id,
                public_episode_id=probe.public_episode_id,
                decision_id=decision_id,
                transition_sha256=supplied_hash,
                transition=CandidateTransition(
                    context=context,
                    candidates=candidates,
                    legal_mask=legal,
                    action=action,
                    reward=reward,
                    cost=cost,
                    preference=(),
                    behavior_features=(),
                    next_context=next_context,
                    next_candidates=next_candidates,
                    next_legal_mask=next_legal,
                    next_preference=(),
                    done=done,
                    duration=duration,
                    outcome_hash=outcome_hash,
                    terminated=terminated,
                    truncated=truncated,
                ),
            )
        )
    if abs(reward_sum - probe.explored_free_flight_volume_auc_time) > 1.0e-9:
        raise ValueError("decision AUC contributions do not reproduce the episode metric")
    return tuple(samples)


def train_single_rl_baseline(
    samples: Sequence[SingleRLTrainingSample],
    *,
    split_manifest_sha256: str,
    updates: int,
    hidden_dim: int,
    seed: int,
    minimum_transitions: int,
    minimum_scenes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Perform real RB-SF-SAC updates and return checkpoint plus public provenance."""

    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
        raise ValueError("updates must be a positive integer")
    if not isinstance(minimum_transitions, int) or minimum_transitions < 1:
        raise ValueError("minimum_transitions must be positive")
    if not isinstance(minimum_scenes, int) or minimum_scenes < 1:
        raise ValueError("minimum_scenes must be positive")
    ordered = tuple(sorted(samples, key=lambda sample: sample.transition_sha256))
    if len(ordered) < minimum_transitions:
        raise ValueError("not enough real P07 decision transitions for single-RL training")
    if len({sample.transition_sha256 for sample in ordered}) != len(ordered):
        raise ValueError("duplicate P07 decision transition supplied to single-RL training")
    action_keys = {(row.scene_id, row.public_episode_id, row.decision_id) for row in ordered}
    if len(action_keys) != len(ordered):
        raise ValueError("duplicate scene/episode/decision supplied to single-RL training")
    scenes = tuple(sorted({sample.scene_id for sample in ordered}))
    if len(scenes) < minimum_scenes:
        raise ValueError("single-RL training requires the requested number of train scenes")
    model = RBSFSAC(
        RBSFSACConfig(
            context_dim=_CONTEXT_DIM,
            candidate_dim=_CANDIDATE_DIM,
            preference_dim=0,
            sf_dim=0,
            hidden_dim=hidden_dim,
        ),
        seed=seed,
    )
    transitions = tuple(sample.transition for sample in ordered)
    diagnostics: dict[str, float] = {}
    for _ in range(updates):
        diagnostics = model.update(transitions)
    provenance = {
        "schema_version": SINGLE_RL_TRAINING_PROVENANCE_SCHEMA_VERSION,
        "claim_limit": (
            "Train-only ordinary candidate-level RB-SF-SAC baseline. No realised-QD archive, "
            "outcome fragment reuse, evaluator geometry, target coordinates, or oracle actions "
            "are used as policy inputs."
        ),
        "training_partition": "train",
        "split_manifest_sha256": _sha(split_manifest_sha256, "split_manifest_sha256"),
        "feature_schema_version": SINGLE_RL_FEATURE_SCHEMA_VERSION,
        **public_schema_fields(),
        "training_scene_ids": list(scenes),
        "episode_count": len({(row.scene_id, row.public_episode_id) for row in ordered}),
        "transition_count": len(ordered),
        "rollout_record_sha256": sorted({sample.raw_record_sha256 for sample in ordered}),
        "transition_sha256": [sample.transition_sha256 for sample in ordered],
        "updates": updates,
        "seed": seed,
        "model": {"hidden_dim": hidden_dim, "sf_dim": 0, "archive": False, "ogfr": False},
        "aggregate_training_diagnostics": diagnostics,
    }
    checkpoint = build_single_rl_checkpoint_payload(
        model,
        training_scene_ids=scenes,
        training_updates=updates,
        training_provenance=provenance,
        split_manifest_sha256=split_manifest_sha256,
    )
    return checkpoint, provenance


__all__ = [
    "SINGLE_RL_TRAINING_PROVENANCE_SCHEMA_VERSION",
    "SingleRLTrainingSample",
    "sample_from_p07_training_record",
    "train_single_rl_baseline",
    "training_scene_ids_from_split_manifest",
]
