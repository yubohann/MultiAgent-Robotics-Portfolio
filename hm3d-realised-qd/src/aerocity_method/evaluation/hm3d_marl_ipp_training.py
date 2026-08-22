"""Outcome-only training ingestion for the MARL-IPP controlled transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerocity_method.adapters.hm3d_marl_ipp import (
    MARL_IPP_FEATURE_SCHEMA_VERSION,
    MARL_IPP_TRAINING_TRANSITION_SCHEMA_VERSION,
    MarlIPPGraphInput,
    MarlIPPPortConfig,
    MarlIPPPortPolicy,
    MarlIPPTrainingRow,
    build_marl_ipp_checkpoint_payload,
)
from aerocity_method.contracts.io import canonical_sha256, finite_number, require_identifier
from aerocity_method.contracts.hm3d_public_schema import (
    public_schema_fields,
    require_current_public_schema,
)
from aerocity_method.evaluation.hm3d_evidence_classification import (
    require_trainable_p07_outcome,
)
from aerocity_method.evaluation.hm3d_single_rl_training import (
    training_scene_ids_from_split_manifest,
)
from aerocity_method.evaluation.hm3d_p07_matrix import P07ProbeRecord

MARL_IPP_TRAINING_PROVENANCE_SCHEMA_VERSION = "hm3d-marl-ipp-port-training-v2"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    int(value, 16)
    return value


def _matrix(
    value: Any,
    name: str,
    *,
    width: int,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty matrix")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{name}[{row_index}] must contain {width} values")
        rows.append(
            tuple(
                finite_number(item, f"{name}[{row_index}][{column}]")
                for column, item in enumerate(row)
            )
        )
    return tuple(rows)


def _bool_matrix(value: Any, name: str, count: int) -> tuple[tuple[bool, ...], ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name} row count mismatch")
    rows = tuple(tuple(row) for row in value)
    if any(len(row) != count or any(not isinstance(item, bool) for item in row) for row in rows):
        raise ValueError(f"{name} must be a square boolean matrix")
    return rows


@dataclass(frozen=True, slots=True)
class MarlIPPTrainingSample:
    raw_record_sha256: str
    scene_id: str
    public_episode_id: str
    decision_id: str
    transition_sha256: str
    done: bool
    duration_s: float
    row: MarlIPPTrainingRow

    def __post_init__(self) -> None:
        _sha(self.raw_record_sha256, "raw_record_sha256")
        require_identifier(self.scene_id, "scene_id")
        require_identifier(self.public_episode_id, "public_episode_id")
        require_identifier(self.decision_id, "decision_id")
        _sha(self.transition_sha256, "transition_sha256")
        if not isinstance(self.done, bool) or finite_number(self.duration_s, "duration_s") <= 0.0:
            raise ValueError("MARL-IPP sample terminal flag or duration is invalid")
        if not isinstance(self.row, MarlIPPTrainingRow):
            raise TypeError("row must be MarlIPPTrainingRow")


def sample_from_p07_training_record(
    payload: Mapping[str, Any],
    *,
    allowed_train_scene_ids: Sequence[str],
) -> tuple[MarlIPPTrainingSample, ...]:
    require_trainable_p07_outcome(payload)
    require_current_public_schema(payload, context="MARL-IPP P07 worker record")
    probe = P07ProbeRecord.from_raw(str(payload.get("strategy")), payload)
    if probe.partition != "train" or probe.scene_id not in set(allowed_train_scene_ids):
        raise ValueError("MARL-IPP training accepts only frozen train scenes")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("MARL-IPP training record has no executed decision")
    emitted_rows = payload.get("marl_ipp_training_transitions")
    if not isinstance(emitted_rows, list) or len(emitted_rows) != len(decisions):
        raise ValueError("MARL-IPP transition count must equal executed decisions")
    samples: list[MarlIPPTrainingSample] = []
    reward_sum = 0.0
    for index, (raw_emitted, raw_decision) in enumerate(
        zip(emitted_rows, decisions, strict=True)
    ):
        emitted = _mapping(raw_emitted, f"marl_ipp_training_transitions[{index}]")
        decision = _mapping(raw_decision, f"decisions[{index}]")
        require_current_public_schema(decision, context=f"decisions[{index}]")
        require_current_public_schema(
            emitted, context=f"marl_ipp_training_transitions[{index}]"
        )
        if emitted.get("schema_version") != MARL_IPP_TRAINING_TRANSITION_SCHEMA_VERSION:
            raise ValueError("MARL-IPP transition schema mismatch")
        supplied_hash = _sha(emitted.get("transition_sha256"), "transition_sha256")
        unsigned = dict(emitted)
        unsigned.pop("transition_sha256", None)
        if canonical_sha256(unsigned) != supplied_hash:
            raise ValueError("MARL-IPP transition hash mismatch")
        decision_id = require_identifier(decision.get("decision_id"), "decision_id")
        if emitted.get("decision_id") != decision_id or emitted.get("scene_id") != probe.scene_id:
            raise ValueError("MARL-IPP decision or scene binding differs")
        context_hash = _sha(emitted.get("public_context_hash"), "public_context_hash")
        pool_hash = _sha(emitted.get("public_candidate_pool_hash"), "candidate pool hash")
        if context_hash != decision.get("public_context_hash") or pool_hash != decision.get(
            "public_candidate_pool_hash"
        ):
            raise ValueError("MARL-IPP public state differs from executed decision")
        for field, expected in public_schema_fields().items():
            if emitted.get(field) != decision.get(field) or emitted.get(field) != expected:
                raise ValueError("MARL-IPP public-task schema differs from execution")
        # The episode anchor precedes bootstrap sensing.  The learning state is the
        # source-bound decisions[0] state, which is checked above.
        selection = _mapping(decision.get("selection"), "decision selection")
        execution = _mapping(decision.get("execution"), "decision execution")
        if emitted.get("selected_candidate_id") != selection.get("selected_candidate_id"):
            raise ValueError("MARL-IPP candidate differs from executed decision")
        if emitted.get("selected_manifest_hash") != selection.get("selected_manifest_hash"):
            raise ValueError("MARL-IPP manifest differs from executed decision")
        outcome_hashes = emitted.get("execution_outcome_hashes")
        if not isinstance(outcome_hashes, list) or outcome_hashes != execution.get(
            "outcome_hashes"
        ):
            raise ValueError("MARL-IPP outcome identities differ from execution")
        expected_outcome = canonical_sha256(
            {"manifest_hash": execution.get("manifest_hash"), "outcome_hashes": outcome_hashes}
        )
        if emitted.get("outcome_hash") != expected_outcome:
            raise ValueError("MARL-IPP transition outcome hash mismatch")
        node_features = _matrix(emitted.get("node_features"), "node_features", width=8)
        node_count = len(node_features)
        adjacency = _bool_matrix(emitted.get("adjacency"), "adjacency", node_count)
        budgets = _matrix(emitted.get("budget_features"), "budget_features", width=1)
        positions = _matrix(emitted.get("position_encoding"), "position_encoding", width=32)
        legal_raw = emitted.get("legal_mask")
        if not isinstance(legal_raw, list) or len(legal_raw) != node_count:
            raise ValueError("MARL-IPP legal mask shape mismatch")
        legal = tuple(legal_raw)
        action = emitted.get("selected_action_index")
        if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < node_count:
            raise ValueError("MARL-IPP selected action is outside graph")
        reward = finite_number(
            emitted.get("reward_explored_free_flight_volume_auc_time_contribution"),
            "MARL-IPP AUC contribution",
        )
        if reward != finite_number(
            decision.get("reward_explored_free_flight_volume_auc_time_contribution"),
            "decision AUC contribution",
        ):
            raise ValueError("MARL-IPP reward differs from actual decision contribution")
        reward_sum += reward
        duration = finite_number(emitted.get("duration_s"), "MARL-IPP duration")
        if duration != finite_number(decision.get("duration_s"), "decision duration"):
            raise ValueError("MARL-IPP duration differs from actual decision")
        is_final = index == len(decisions) - 1
        expected_terminated = (
            is_final and probe.terminal_outcome == "executed_terminal_safety_failure"
        )
        expected_truncated = is_final and probe.terminal_outcome == "budget_exhausted"
        terminated = emitted.get("terminated")
        truncated = emitted.get("truncated")
        if terminated is not expected_terminated or truncated is not expected_truncated:
            raise ValueError("MARL-IPP terminal cause differs from P07 outcome")
        done = terminated or truncated
        if not done:
            next_emitted = _mapping(emitted_rows[index + 1], "next MARL-IPP transition")
            if emitted.get("next_public_context_hash") != next_emitted.get(
                "public_context_hash"
            ) or emitted.get("next_public_candidate_pool_hash") != next_emitted.get(
                "public_candidate_pool_hash"
            ):
                raise ValueError("MARL-IPP next state is not the next real decision")
        graph = MarlIPPGraphInput(
            node_features=node_features,
            adjacency=adjacency,
            budget_features=budgets,
            position_encoding=positions,
            legal_mask=legal,
        )
        samples.append(
            MarlIPPTrainingSample(
                raw_record_sha256=probe.raw_record_sha256,
                scene_id=probe.scene_id,
                public_episode_id=probe.public_episode_id,
                decision_id=decision_id,
                transition_sha256=supplied_hash,
                done=done,
                duration_s=duration,
                row=MarlIPPTrainingRow(graph=graph, action=action, reward=reward),
            )
        )
    if abs(reward_sum - probe.explored_free_flight_volume_auc_time) > 1.0e-9:
        raise ValueError("MARL-IPP decision rewards do not reproduce episode AUC")
    return tuple(samples)


def train_marl_ipp_port_baseline(
    samples: Sequence[MarlIPPTrainingSample],
    *,
    split_manifest_sha256: str,
    source_root: str | Path,
    source_checkpoint: str | Path,
    updates: int,
    seed: int,
    minimum_transitions: int,
    minimum_scenes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if updates < 1 or minimum_transitions < 1 or minimum_scenes < 1:
        raise ValueError("MARL-IPP training counts must be positive")
    ordered = tuple(sorted(samples, key=lambda row: row.transition_sha256))
    if len(ordered) < minimum_transitions:
        raise ValueError("not enough real decision transitions for MARL-IPP training")
    if len({row.transition_sha256 for row in ordered}) != len(ordered):
        raise ValueError("duplicate MARL-IPP decision transition")
    scenes = tuple(sorted({row.scene_id for row in ordered}))
    if len(scenes) < minimum_scenes:
        raise ValueError("MARL-IPP training needs the requested number of scenes")
    model = MarlIPPPortPolicy(
        source_root,
        MarlIPPPortConfig(),
        source_checkpoint=source_checkpoint,
        seed=seed,
    )
    diagnostics: dict[str, float] = {}
    rows = tuple(sample.row for sample in ordered)
    for _ in range(updates):
        diagnostics = model.update(rows)
    provenance: dict[str, Any] = {
        "schema_version": MARL_IPP_TRAINING_PROVENANCE_SCHEMA_VERSION,
        "claim_limit": (
            "The authors' AttentionNet is initialized from the published checkpoint and "
            "trained only on public HM3D candidate graphs and real CF2X outcome returns."
        ),
        "training_partition": "train",
        "split_manifest_sha256": _sha(split_manifest_sha256, "split_manifest_sha256"),
        "feature_schema_version": MARL_IPP_FEATURE_SCHEMA_VERSION,
        **public_schema_fields(),
        "training_scene_ids": list(scenes),
        "episode_count": len({(row.scene_id, row.public_episode_id) for row in ordered}),
        "transition_count": len(ordered),
        "rollout_record_sha256": sorted({row.raw_record_sha256 for row in ordered}),
        "transition_sha256": [row.transition_sha256 for row in ordered],
        "updates": updates,
        "seed": seed,
        "source_attention_net_sha256": model.source_attention_net_sha256,
        "source_checkpoint_sha256": model.source_checkpoint_sha256,
        "aggregate_training_diagnostics": diagnostics,
    }
    checkpoint = build_marl_ipp_checkpoint_payload(
        model,
        training_scene_ids=scenes,
        training_updates=updates,
        training_provenance=provenance,
        split_manifest_sha256=split_manifest_sha256,
    )
    return checkpoint, provenance


__all__ = [
    "MARL_IPP_TRAINING_PROVENANCE_SCHEMA_VERSION",
    "MarlIPPTrainingSample",
    "sample_from_p07_training_record",
    "train_marl_ipp_port_baseline",
    "training_scene_ids_from_split_manifest",
]
