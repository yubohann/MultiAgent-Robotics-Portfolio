"""Checkpoint-gated single-RL selector used as a Exploration weak baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from realised_qd.adapters.hm3d_baselines import PublicSearchState
from realised_qd.contracts.hm3d_public_schema import (
    public_schema_fields,
    require_current_public_schema,
)
from realised_qd.contracts.io import finite_number, require_identifier
from realised_qd.contracts.models import CandidateFragmentManifest
from realised_qd.learning.rb_sf_sac import RBSFSAC, RBSFSACConfig

SINGLE_RL_CHECKPOINT_SCHEMA_VERSION = "hm3d-single-rl-checkpoint-v2"
SINGLE_RL_FEATURE_SCHEMA_VERSION = "hm3d-public-candidate-features-v1"
SINGLE_RL_TRAINING_TRANSITION_SCHEMA_VERSION = "hm3d-single-rl-train-transition-v4"
_CONTEXT_DIM = 4
_CANDIDATE_DIM = 5


def _file_id(path: Path) -> str:
    # Checkpoint identity from file name and size.
    return f"{path.name}:{path.stat().st_size}"


def public_context_features(state: PublicSearchState) -> tuple[float, ...]:
    """Return stable, target-free policy context features."""

    agent_count = len(state.agents)
    max_degree = max(1, agent_count - 1)
    return (
        min(1.0, state.decision_duration_s / 60.0),
        sum(agent.remaining_energy_fraction for agent in state.agents) / agent_count,
        sum(agent.communication_degree / max_degree for agent in state.agents) / agent_count,
        min(1.0, agent_count / 6.0),
    )


def public_candidate_features(manifest: CandidateFragmentManifest) -> tuple[float, ...]:
    """Expose only the public candidate fields consumed by the weak policy."""

    if len(manifest.planned_descriptor) != 3:
        raise ValueError("Exploration single-RL expects the frozen three-dimensional descriptor")
    return (
        *tuple(float(value) for value in manifest.planned_descriptor),
        float(manifest.quality_hint),
        float(manifest.cost_hint),
    )


def build_single_rl_training_transition(
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
    """Serialize one decision-level, outcome-backed train-only SAC transition."""

    rows = tuple(pool)
    next_rows = tuple(next_pool)
    if not rows:
        raise ValueError("single-RL training transition needs a candidate pool")
    if not next_rows or not any(row.feasible for row in next_rows):
        raise ValueError("single-RL training transition needs a legal next candidate pool")
    try:
        action = tuple(row.manifest_id for row in rows).index(selected.manifest_id)
    except ValueError as error:
        raise ValueError("selected manifest is absent from the public candidate pool") from error
    if not rows[action].feasible:
        raise ValueError("single-RL training transition cannot select an illegal candidate")
    if not isinstance(execution, Mapping):
        raise ValueError("execution must be a public outcome mapping")
    outcome_ids = execution.get("outcome_ids")
    if not isinstance(outcome_ids, list) or not outcome_ids:
        raise ValueError("single-RL training requires one or more real outcome ids")
    for outcome_id in outcome_ids:
        require_identifier(outcome_id, "execution outcome id")
    manifest_id = execution.get("manifest_id")
    if manifest_id != selected.manifest_id:
        raise ValueError("execution outcome manifest does not match the selected candidate")
    energy_j = finite_number(execution.get("total_energy_used_j"), "total_energy_used_j")
    if energy_j < 0.0:
        raise ValueError("total_energy_used_j must be non-negative")
    reward = finite_number(
        explored_free_flight_volume_auc_time_contribution,
        "explored_free_flight_volume_auc_time_contribution",
    )
    if not 0.0 <= reward <= 1.0:
        raise ValueError("AUC-time contribution must lie in [0, 1]")
    duration = finite_number(duration_s, "duration_s")
    if duration <= 0.0:
        raise ValueError("duration_s must be positive")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise ValueError("terminated and truncated must be boolean")
    if terminated and truncated:
        raise ValueError("a transition cannot be both terminated and truncated")
    require_identifier(scene_id, "scene_id")
    transition = {
        "schema_version": SINGLE_RL_TRAINING_TRANSITION_SCHEMA_VERSION,
        "claim_limit": (
            "Train-only decision-level candidate-selection transition. Current and next candidate "
            "features are public; reward and cost come from this selected candidate's actual "
            "execution segment. It contains no target geometry, target identifiers, unselected "
            "rewards, or archive data."
        ),
        "scene_id": scene_id,
        "decision_id": state.context.decision_id,
        "public_context_id": state.context.context_id,
        "public_candidate_pool_id": "|".join(row.manifest_id for row in rows),
        **public_schema_fields(),
        "context_features": list(public_context_features(state)),
        "candidate_features": [list(public_candidate_features(row)) for row in rows],
        "legal_mask": [row.feasible for row in rows],
        "selected_action_index": action,
        "selected_candidate_id": selected.candidate_id,
        "selected_manifest_id": selected.manifest_id,
        "reward_explored_free_flight_volume_auc_time_contribution": reward,
        "cost_energy_j": energy_j,
        "duration_s": duration,
        "terminated": terminated,
        "truncated": truncated,
        "next_public_context_id": next_state.context.context_id,
        "next_public_candidate_pool_id": "|".join(row.manifest_id for row in next_rows),
        "next_context_features": list(public_context_features(next_state)),
        "next_candidate_features": [
            list(public_candidate_features(row)) for row in next_rows
        ],
        "next_legal_mask": [row.feasible for row in next_rows],
        "outcome_id": "|".join(str(value) for value in outcome_ids),
    }
    transition["transition_id"] = f"{scene_id}:{state.context.decision_id}:{action}"
    return transition


@dataclass(frozen=True, slots=True)
class SingleRLSelection:
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
            raise ValueError("single-RL selection needs public candidate probabilities")
        for candidate_id, probability in self.scores:
            require_identifier(candidate_id, "candidate_id")
            if finite_number(probability, "candidate probability") < 0.0:
                raise ValueError("candidate probability must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": "single_rl",
            "selected_manifest_id": self.selected_manifest_id,
            "selected_candidate_id": self.selected_candidate_id,
            "scores": list(self.scores),
            "checkpoint_id": self.checkpoint_id,
            "training_provenance_id": self.training_provenance_id,
            "feature_schema_version": SINGLE_RL_FEATURE_SCHEMA_VERSION,
            "claim_limit": (
                "Target-free, archive-free single-RL baseline selection. It does not use "
                "QD elites, OGFR fragments, private targets or evaluator geometry."
            ),
        }


def _load_checkpoint(
    path: Path,
    *,
    expected_split_manifest_id: str | None = None,
) -> tuple[RBSFSAC, str, str]:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - only the Isaac worker loads it
        raise RuntimeError("single-RL baseline needs PyTorch") from error
    if not path.is_file():
        raise FileNotFoundError(f"single-RL checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("single-RL checkpoint must be a mapping")
    if payload.get("schema_version") != SINGLE_RL_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("single-RL checkpoint schema mismatch")
    if payload.get("training_partition") != "train":
        raise ValueError("single-RL checkpoint must be trained only on the HM3D train split")
    checkpoint_split_id = require_identifier(
        payload.get("split_manifest_id"), "split_manifest_id"
    )
    if (
        expected_split_manifest_id is not None
        and checkpoint_split_id != expected_split_manifest_id
    ):
        raise ValueError("single-RL checkpoint belongs to a different frozen scene split")
    scene_ids = payload.get("training_scene_ids")
    if (
        not isinstance(scene_ids, list)
        or not scene_ids
        or any(not isinstance(scene_id, str) or not scene_id for scene_id in scene_ids)
    ):
        raise ValueError("single-RL checkpoint lacks training scene provenance")
    if payload.get("feature_schema_version") != SINGLE_RL_FEATURE_SCHEMA_VERSION:
        raise ValueError("single-RL feature schema mismatch")
    require_current_public_schema(payload, context="single-RL checkpoint")
    training_updates = payload.get("training_updates")
    if (
        not isinstance(training_updates, int)
        or isinstance(training_updates, bool)
        or training_updates < 1
    ):
        raise ValueError("single-RL checkpoint must contain at least one real training update")
    provenance_id = require_identifier(
        payload.get("training_provenance_id"), "training_provenance_id"
    )
    state = payload.get("rbsfsac_state")
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict):
        raise ValueError("single-RL checkpoint lacks an RB-SF-SAC state")
    config = RBSFSACConfig(**state["config"])
    if (
        config.context_dim != _CONTEXT_DIM
        or config.candidate_dim != _CANDIDATE_DIM
        or config.preference_dim != 0
        or config.sf_dim != 0
    ):
        raise ValueError(
            "single-RL checkpoint does not match the frozen public Exploration features"
        )
    model = RBSFSAC(config, seed=0)
    model.load_state_dict(state)
    return model, _file_id(path), provenance_id


def select_single_rl(
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
    *,
    checkpoint_path: str | Path,
    expected_split_manifest_id: str | None = None,
) -> tuple[CandidateFragmentManifest, SingleRLSelection]:
    """Load a train-provenanced actor and select a legal public candidate."""

    rows = tuple(pool)
    if not rows or not any(row.feasible for row in rows):
        raise ValueError("single-RL selection requires a non-empty legal candidate pool")
    model, checkpoint_id, provenance_file_id = _load_checkpoint(
        Path(checkpoint_path),
        expected_split_manifest_id=expected_split_manifest_id,
    )
    probabilities = model.action_probabilities(
        public_context_features(state),
        tuple(public_candidate_features(row) for row in rows),
        tuple(row.feasible for row in rows),
    )
    selected_index = max(range(len(rows)), key=lambda index: probabilities[index])
    selected = rows[selected_index]
    if not selected.feasible:
        raise RuntimeError("masked single-RL actor selected an illegal public candidate")
    return selected, SingleRLSelection(
        selected_manifest_id=selected.manifest_id,
        selected_candidate_id=selected.candidate_id,
        scores=tuple(
            (row.candidate_id, probability)
            for row, probability in zip(rows, probabilities, strict=True)
        ),
        checkpoint_id=checkpoint_id,
        training_provenance_id=provenance_file_id,
    )


def build_single_rl_checkpoint_payload(
    model: RBSFSAC,
    *,
    training_scene_ids: Sequence[str],
    training_updates: int,
    training_provenance: Mapping[str, Any],
    split_manifest_id: str,
) -> dict[str, Any]:
    """Create a serializable checkpoint envelope for the train-only baseline."""

    if not isinstance(model, RBSFSAC):
        raise TypeError("model must be an RBSFSAC instance")
    if model.config.context_dim != _CONTEXT_DIM or model.config.candidate_dim != _CANDIDATE_DIM:
        raise ValueError("model dimensions do not match Exploration public features")
    scenes = tuple(training_scene_ids)
    if not scenes or any(not isinstance(scene, str) or not scene for scene in scenes):
        raise ValueError("training scene provenance cannot be empty")
    if (
        not isinstance(training_updates, int)
        or isinstance(training_updates, bool)
        or training_updates < 1
    ):
        raise ValueError("training_updates must be positive")
    provenance = dict(training_provenance)
    if not provenance:
        raise ValueError("single-RL checkpoint requires real training provenance")
    split_id = require_identifier(split_manifest_id, "split_manifest_id")
    if provenance.get("split_manifest_id") != split_id:
        raise ValueError("single-RL checkpoint provenance is not bound to the frozen scene split")
    require_current_public_schema(provenance, context="single-RL training provenance")
    return {
        "schema_version": SINGLE_RL_CHECKPOINT_SCHEMA_VERSION,
        "training_partition": "train",
        "split_manifest_id": split_id,
        "training_scene_ids": list(scenes),
        "training_updates": training_updates,
        "training_provenance_id": f"single-rl-provenance:{len(provenance)}-fields:{split_id}",
        "feature_schema_version": SINGLE_RL_FEATURE_SCHEMA_VERSION,
        **public_schema_fields(),
        "rbsfsac_state": model.state_dict(),
    }


__all__ = [
    "SINGLE_RL_CHECKPOINT_SCHEMA_VERSION",
    "SINGLE_RL_FEATURE_SCHEMA_VERSION",
    "SINGLE_RL_TRAINING_TRANSITION_SCHEMA_VERSION",
    "SingleRLSelection",
    "build_single_rl_training_transition",
    "build_single_rl_checkpoint_payload",
    "public_candidate_features",
    "public_context_features",
    "select_single_rl",
]
