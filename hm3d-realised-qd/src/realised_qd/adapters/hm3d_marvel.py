"""Outcome-bound MARVEL author-SAC migration for public HM3D team candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from realised_qd.adapters.hm3d_baselines import PublicSearchState
from realised_qd.adapters.hm3d_marvel_author_sac import (
    AGENT_DIM,
    CANDIDATE_DIM,
    CONTEXT_DIM,
    MARVEL_AUTHOR_MODEL_COMMIT,
    MarvelGraphObservation,
    MarvelSupplementaryReferenceConfig,
    MarvelSupplementaryReferencePolicy,
    MarvelSupplementaryReferenceTrainingRow,
    public_marvel_adjacency,
    public_marvel_agent_features,
    public_marvel_candidate_features,
    public_marvel_graph_observation,
)
from realised_qd.adapters.hm3d_single_rl import public_context_features
from realised_qd.contracts.hm3d_public_schema import (
    public_schema_fields,
    require_current_public_schema,
)
from realised_qd.contracts.io import finite_number, require_identifier
from realised_qd.contracts.models import CandidateFragmentManifest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - explicit at runtime
    torch = None  # type: ignore[assignment]

MARVEL_SUPPLEMENTARY_REFERENCE_ID = "marvel_supplementary_reference"
MARVEL_SUPPLEMENTARY_REFERENCE_CHECKPOINT_SCHEMA_VERSION = "hm3d-marvel-author-sac-checkpoint-v3"
MARVEL_SUPPLEMENTARY_REFERENCE_FEATURE_SCHEMA_VERSION = "hm3d-marvel-public-3d-candidate-graph-v2"
MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_SCHEMA_VERSION = (
    "hm3d-marvel-author-sac-transition-v5"
)
MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_KEY = (
    "marvel_supplementary_reference_training_transitions"
)
MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_TRAINING_TRANSITION_KEY = "marvel_training_transitions"
MARVEL_SUPPLEMENTARY_REFERENCE_STATE_KEY = "marvel_supplementary_reference_state"
MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_STATE_KEY = "marvel_port_state"
_CONTEXT_DIM = CONTEXT_DIM
_AGENT_DIM = AGENT_DIM
_CANDIDATE_DIM = CANDIDATE_DIM


def _file_id(path: Path) -> str:
    # Checkpoint identity from file name and size.
    return f"{path.name}:{path.stat().st_size}"


@dataclass(frozen=True, slots=True)
class MarvelSupplementaryReferenceSelection:
    selected_manifest_id: str
    selected_candidate_id: str
    scores: tuple[tuple[str, float], ...]
    checkpoint_id: str
    training_provenance_id: str

    def __post_init__(self) -> None:
        require_identifier(self.selected_manifest_id, "selected_manifest_id")
        require_identifier(self.selected_candidate_id, "selected_candidate_id")
        require_identifier(self.checkpoint_id, "checkpoint_id")
        require_identifier(self.training_provenance_id, "training_provenance_id")
        if not self.scores:
            raise ValueError("MARVEL selection requires public probabilities")

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": MARVEL_SUPPLEMENTARY_REFERENCE_ID,
            "schema_version": MARVEL_SUPPLEMENTARY_REFERENCE_FEATURE_SCHEMA_VERSION,
            "adaptation_status": "author_network_and_sac_controlled_3d_transfer",
            "author_model_commit": MARVEL_AUTHOR_MODEL_COMMIT,
            "selected_manifest_id": self.selected_manifest_id,
            "selected_candidate_id": self.selected_candidate_id,
            "scores": [list(row) for row in self.scores],
            "checkpoint_id": self.checkpoint_id,
            "training_provenance_id": self.training_provenance_id,
            "claim_limit": (
                "MARVEL author PolicyNet/QNet with discrete SAC on the common public 3D "
                "candidate graph. It has no QD archive, fragment replay, target data or "
                "evaluator geometry input."
            ),
        }


def build_marvel_supplementary_reference_training_transition(
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
    selected: CandidateFragmentManifest,
    *,
    next_state: PublicSearchState,
    next_pool: Sequence[CandidateFragmentManifest],
    scene_id: str,
    execution: Mapping[str, Any],
    explored_free_flight_volume_auc_time_contribution: float,
    duration_s: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, object]:
    """Serialize one actual decision for MARVEL author-architecture SAC training."""

    rows = tuple(pool)
    next_rows = tuple(next_pool)
    if not rows:
        raise ValueError("MARVEL supplementary reference transition needs a candidate pool")
    if not next_rows or not any(row.feasible for row in next_rows):
        raise ValueError("MARVEL supplementary reference transition needs a legal next pool")
    try:
        action = tuple(row.manifest_id for row in rows).index(selected.manifest_id)
    except ValueError as error:
        raise ValueError(
            "selected MARVEL supplementary reference candidate is absent from the public pool"
        ) from error
    if not selected.feasible:
        raise ValueError(
            "MARVEL supplementary reference training cannot record an illegal selected candidate"
        )
    outcome_ids = execution.get("outcome_ids")
    if not isinstance(outcome_ids, list) or not outcome_ids:
        raise ValueError("MARVEL supplementary reference training needs actual outcome ids")
    for outcome_id in outcome_ids:
        require_identifier(outcome_id, "execution outcome id")
    if execution.get("manifest_id") != selected.manifest_id:
        raise ValueError(
            "MARVEL supplementary reference execution manifest does not match selected candidate"
        )
    reward = finite_number(
        explored_free_flight_volume_auc_time_contribution,
        "exploration AUC contribution",
    )
    duration = finite_number(duration_s, "duration_s")
    if (
        not 0.0 <= reward <= 1.0
        or duration <= 0.0
        or not isinstance(terminated, bool)
        or not isinstance(truncated, bool)
        or (terminated and truncated)
    ):
        raise ValueError(
            "MARVEL supplementary reference reward, duration or terminal flag is outside "
            "the contract"
        )
    require_identifier(scene_id, "scene_id")
    graph = public_marvel_graph_observation(state, rows)
    next_graph = public_marvel_graph_observation(next_state, next_rows)
    transition: dict[str, object] = {
        "schema_version": MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_SCHEMA_VERSION,
        "claim_limit": (
            "Train-only decision-level MARVEL author-SAC transition. All graph inputs are "
            "public and the reward is this actual execution segment's AUC contribution."
        ),
        "author_model_commit": MARVEL_AUTHOR_MODEL_COMMIT,
        "scene_id": scene_id,
        "decision_id": state.context.decision_id,
        "public_context_id": state.context.context_id,
        "public_candidate_pool_id": "|".join(row.manifest_id for row in rows),
        **public_schema_fields(),
        "context_features": list(public_context_features(state)),
        "agent_features": [list(row) for row in public_marvel_agent_features(state)],
        "communication_adjacency": [list(row) for row in public_marvel_adjacency(state)],
        "candidate_features": [list(row) for row in public_marvel_candidate_features(state, rows)],
        "marvel_graph_observation": graph.to_dict(),
        "selected_action_index": action,
        "selected_candidate_id": selected.candidate_id,
        "selected_manifest_id": selected.manifest_id,
        "reward_explored_free_flight_volume_auc_time_contribution": reward,
        "cost_energy_j": finite_number(execution.get("total_energy_used_j"), "energy cost"),
        "duration_s": duration,
        "terminated": terminated,
        "truncated": truncated,
        "next_public_context_id": next_state.context.context_id,
        "next_public_candidate_pool_id": "|".join(row.manifest_id for row in next_rows),
        "next_context_features": list(public_context_features(next_state)),
        "next_agent_features": [
            list(row) for row in public_marvel_agent_features(next_state)
        ],
        "next_communication_adjacency": [
            list(row) for row in public_marvel_adjacency(next_state)
        ],
        "next_candidate_features": [
            list(row) for row in public_marvel_candidate_features(next_state, next_rows)
        ],
        "next_marvel_graph_observation": next_graph.to_dict(),
        "outcome_id": "|".join(str(value) for value in outcome_ids),
    }
    transition["transition_id"] = f"{scene_id}:{state.context.decision_id}:{action}"
    return transition


def build_marvel_checkpoint_payload(
    model: MarvelSupplementaryReferencePolicy,
    *,
    training_scene_ids: Sequence[str],
    training_updates: int,
    training_provenance: Mapping[str, Any],
    split_manifest_id: str,
) -> dict[str, Any]:
    if not isinstance(model, MarvelSupplementaryReferencePolicy):
        raise TypeError("model must be a MarvelSupplementaryReferencePolicy")
    if training_updates < 1 or not training_provenance:
        raise ValueError("MARVEL checkpoint requires real training updates and provenance")
    scenes = tuple(training_scene_ids)
    if not scenes or any(not isinstance(scene, str) or not scene for scene in scenes):
        raise ValueError("MARVEL checkpoint has no train scene provenance")
    split_id = require_identifier(split_manifest_id, "split_manifest_id")
    provenance_split_id = require_identifier(
        training_provenance.get("split_manifest_id"),
        "training provenance split_manifest_id",
    )
    if provenance_split_id != split_id:
        raise ValueError("MARVEL checkpoint provenance is not bound to the frozen scene split")
    require_current_public_schema(training_provenance, context="MARVEL training provenance")
    return {
        "schema_version": MARVEL_SUPPLEMENTARY_REFERENCE_CHECKPOINT_SCHEMA_VERSION,
        "training_partition": "train",
        "split_manifest_id": split_id,
        "training_scene_ids": list(scenes),
        "training_updates": training_updates,
        "training_provenance_id": f"marvel-provenance:{len(training_provenance)}-fields:{split_id}",
        "feature_schema_version": MARVEL_SUPPLEMENTARY_REFERENCE_FEATURE_SCHEMA_VERSION,
        **public_schema_fields(),
        "author_model_commit": MARVEL_AUTHOR_MODEL_COMMIT,
        MARVEL_SUPPLEMENTARY_REFERENCE_STATE_KEY: model.state_dict(),
    }


def _load_checkpoint(
    path: Path,
    *,
    expected_split_manifest_id: str | None = None,
) -> tuple[MarvelSupplementaryReferencePolicy, str, str]:
    if torch is None:
        raise RuntimeError("MARVEL baseline requires PyTorch")
    if not path.is_file():
        raise FileNotFoundError(f"MARVEL checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("MARVEL checkpoint must be a mapping")
    if payload.get("schema_version") != MARVEL_SUPPLEMENTARY_REFERENCE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("MARVEL checkpoint schema mismatch")
    if payload.get("training_partition") != "train":
        raise ValueError("MARVEL checkpoint must be trained on train scenes only")
    if payload.get("author_model_commit") != MARVEL_AUTHOR_MODEL_COMMIT:
        raise ValueError("MARVEL checkpoint author source mismatch")
    checkpoint_split_id = require_identifier(
        payload.get("split_manifest_id"), "split_manifest_id"
    )
    if (
        expected_split_manifest_id is not None
        and checkpoint_split_id != expected_split_manifest_id
    ):
        raise ValueError("MARVEL checkpoint belongs to a different frozen scene split")
    scenes = payload.get("training_scene_ids")
    if (
        not isinstance(scenes, list)
        or not scenes
        or any(not isinstance(row, str) or not row for row in scenes)
    ):
        raise ValueError("MARVEL checkpoint lacks training-scene provenance")
    updates = payload.get("training_updates")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
        raise ValueError("MARVEL checkpoint needs at least one real update")
    if (
        payload.get("feature_schema_version")
        != MARVEL_SUPPLEMENTARY_REFERENCE_FEATURE_SCHEMA_VERSION
    ):
        raise ValueError("MARVEL feature schema mismatch")
    require_current_public_schema(payload, context="MARVEL checkpoint")
    provenance = require_identifier(payload.get("training_provenance_id"), "training provenance")
    state = payload.get(MARVEL_SUPPLEMENTARY_REFERENCE_STATE_KEY)
    if not isinstance(state, Mapping):
        state = payload.get(MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_STATE_KEY)
    if not isinstance(state, Mapping) or not isinstance(state.get("config"), Mapping):
        raise ValueError("MARVEL checkpoint lacks policy state")
    config = MarvelSupplementaryReferenceConfig(**dict(state["config"]))
    model = MarvelSupplementaryReferencePolicy(config, seed=0)
    model.load_state_dict(state)
    return model, _file_id(path), provenance


def select_marvel_supplementary_reference(
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
    *,
    checkpoint_path: str | Path,
    expected_split_manifest_id: str | None = None,
) -> tuple[CandidateFragmentManifest, MarvelSupplementaryReferenceSelection]:
    """Select one legal common-pool candidate with the trained author policy."""

    rows = tuple(pool)
    if not rows or not any(row.feasible for row in rows):
        raise ValueError("MARVEL requires a non-empty legal public candidate pool")
    model, checkpoint_id, provenance_id = _load_checkpoint(
        Path(checkpoint_path),
        expected_split_manifest_id=expected_split_manifest_id,
    )
    probabilities = model.action_probabilities(public_marvel_graph_observation(state, rows))
    index = max(range(len(rows)), key=lambda item: probabilities[item])
    selected = rows[index]
    if not selected.feasible:
        raise RuntimeError("masked MARVEL policy selected an illegal candidate")
    return selected, MarvelSupplementaryReferenceSelection(
        selected_manifest_id=selected.manifest_id,
        selected_candidate_id=selected.candidate_id,
        scores=tuple(
            (row.candidate_id, score)
            for row, score in zip(rows, probabilities, strict=True)
        ),
        checkpoint_id=checkpoint_id,
        training_provenance_id=provenance_id,
    )


__all__ = [
    "MARVEL_AUTHOR_MODEL_COMMIT",
    "MARVEL_SUPPLEMENTARY_REFERENCE_CHECKPOINT_SCHEMA_VERSION",
    "MARVEL_SUPPLEMENTARY_REFERENCE_FEATURE_SCHEMA_VERSION",
    "MARVEL_SUPPLEMENTARY_REFERENCE_ID",
    "MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_STATE_KEY",
    "MARVEL_SUPPLEMENTARY_REFERENCE_LEGACY_TRAINING_TRANSITION_KEY",
    "MARVEL_SUPPLEMENTARY_REFERENCE_STATE_KEY",
    "MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_SCHEMA_VERSION",
    "MARVEL_SUPPLEMENTARY_REFERENCE_TRAINING_TRANSITION_KEY",
    "MarvelGraphObservation",
    "MarvelSupplementaryReferenceConfig",
    "MarvelSupplementaryReferencePolicy",
    "MarvelSupplementaryReferenceSelection",
    "MarvelSupplementaryReferenceTrainingRow",
    "build_marvel_checkpoint_payload",
    "build_marvel_supplementary_reference_training_transition",
    "public_marvel_adjacency",
    "public_marvel_agent_features",
    "public_marvel_candidate_features",
    "public_marvel_graph_observation",
    "select_marvel_supplementary_reference",
]
