"""MARL-IPP controlled transfer on the public HM3D team-candidate interface.

The adapter imports the authors' ``AttentionNet`` directly from a pinned
source checkout.  It replaces the target-mapping environment and reward with
the common HM3D candidate graph and outcome-backed exploration return.  It is
therefore a controlled transfer, not an original-task reproduction.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from aerocity_method.adapters.hm3d_baselines import PublicSearchState
from aerocity_method.contracts.hm3d_public_schema import (
    public_schema_fields,
    require_current_public_schema,
)
from aerocity_method.contracts.io import canonical_sha256, finite_number, require_identifier
from aerocity_method.contracts.models import CandidateFragmentManifest
from aerocity_method.contracts.privacy import walk_public_payload

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - explicit runtime failure
    torch = None  # type: ignore[assignment]

MARL_IPP_PORT_ID = "marl_ipp_port"
MARL_IPP_CHECKPOINT_SCHEMA_VERSION = "hm3d-marl-ipp-port-checkpoint-v2"
MARL_IPP_FEATURE_SCHEMA_VERSION = "hm3d-marl-ipp-public-candidate-graph-v1"
MARL_IPP_TRAINING_TRANSITION_SCHEMA_VERSION = "hm3d-marl-ipp-train-transition-v4"
MARL_IPP_INPUT_DIM = 8
MARL_IPP_EMBEDDING_DIM = 128
MARL_IPP_POSITION_DIM = 32


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    int(value, 16)
    return value


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import source module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_author_attention_net(source_root: str | Path) -> tuple[type[Any], str]:
    """Load the authors' class without copying or silently rewriting it."""

    root = Path(source_root).expanduser().resolve()
    attention_path = root / "attention_net.py"
    parameters_path = root / "parameters.py"
    if not attention_path.is_file() or not parameters_path.is_file():
        raise FileNotFoundError("MARL-IPP source checkout lacks attention_net.py or parameters.py")
    source_hash = _sha256_file(attention_path)
    previous_parameters = sys.modules.get("parameters")
    parameters_module = _load_module(
        parameters_path, f"_aerocity_marl_ipp_parameters_{source_hash[:12]}"
    )
    sys.modules["parameters"] = parameters_module
    try:
        attention_module = _load_module(
            attention_path, f"_aerocity_marl_ipp_attention_{source_hash[:12]}"
        )
    finally:
        if previous_parameters is None:
            sys.modules.pop("parameters", None)
        else:
            sys.modules["parameters"] = previous_parameters
    attention_net = getattr(attention_module, "AttentionNet", None)
    if not isinstance(attention_net, type):
        raise RuntimeError("MARL-IPP source does not expose AttentionNet")
    return attention_net, source_hash


def _transit_endpoint_centroid(
    manifest: CandidateFragmentManifest,
) -> tuple[float, float, float]:
    endpoints = [
        tuple(fragment.path[-1])
        for fragment in manifest.fragments
        if fragment.type_signature.fragment_type == "transit"
    ]
    if not endpoints:
        raise ValueError("MARL-IPP candidate needs at least one transit fragment")
    return tuple(sum(point[axis] for point in endpoints) / len(endpoints) for axis in range(3))


def _normalized_laplacian_encoding(
    adjacency: Sequence[Sequence[bool]],
    *,
    width: int = MARL_IPP_POSITION_DIM,
) -> tuple[tuple[float, ...], ...]:
    if torch is None:
        raise RuntimeError("MARL-IPP controlled transfer requires PyTorch")
    graph = torch.tensor(adjacency, dtype=torch.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1] or graph.shape[0] < 2:
        raise ValueError("MARL-IPP graph adjacency must be a square matrix with an anchor")
    graph.fill_diagonal_(0.0)
    degree = graph.sum(dim=1)
    inverse = torch.where(degree > 0.0, degree.rsqrt(), torch.zeros_like(degree))
    laplacian = torch.eye(graph.shape[0], dtype=torch.float64)
    laplacian -= inverse[:, None] * graph * inverse[None, :]
    _, vectors = torch.linalg.eigh(laplacian)
    nontrivial = vectors[:, 1 : min(vectors.shape[1], width + 1)]
    if nontrivial.shape[1] < width:
        nontrivial = torch.nn.functional.pad(nontrivial, (0, width - nontrivial.shape[1]))
    return tuple(tuple(float(value) for value in row) for row in nontrivial.tolist())


@dataclass(frozen=True, slots=True)
class MarlIPPGraphInput:
    """One anchor node followed by one node per public team candidate."""

    node_features: tuple[tuple[float, ...], ...]
    adjacency: tuple[tuple[bool, ...], ...]
    budget_features: tuple[tuple[float], ...]
    position_encoding: tuple[tuple[float, ...], ...]
    legal_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        node_count = len(self.node_features)
        if node_count < 2 or any(len(row) != MARL_IPP_INPUT_DIM for row in self.node_features):
            raise ValueError("MARL-IPP graph needs an anchor and eight-wide candidate nodes")
        if len(self.adjacency) != node_count or any(
            len(row) != node_count for row in self.adjacency
        ):
            raise ValueError("MARL-IPP adjacency shape mismatch")
        if len(self.budget_features) != node_count or any(
            len(row) != 1 for row in self.budget_features
        ):
            raise ValueError("MARL-IPP budget shape mismatch")
        if len(self.position_encoding) != node_count or any(
            len(row) != MARL_IPP_POSITION_DIM for row in self.position_encoding
        ):
            raise ValueError("MARL-IPP position encoding shape mismatch")
        if len(self.legal_mask) != node_count or self.legal_mask[0] or not any(
            self.legal_mask[1:]
        ):
            raise ValueError("MARL-IPP legal mask must reject the anchor and admit a candidate")


def public_marl_ipp_graph_input(
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
) -> MarlIPPGraphInput:
    """Convert the common public pool to the authors' eight-channel graph input."""

    rows = tuple(pool)
    if not rows or not any(row.feasible for row in rows):
        raise ValueError("MARL-IPP needs a non-empty legal public candidate pool")
    team_centroid = tuple(
        sum(agent.position_m[axis] for agent in state.agents) / len(state.agents)
        for axis in range(3)
    )
    reference = max(state.communication_range_m, 1.0e-9)
    quality_scale = max((abs(row.quality_hint) for row in rows), default=1.0)
    quality_scale = max(quality_scale, 1.0e-9)
    cost_scale = max(state.decision_duration_s, 1.0e-9)
    centroids = tuple(_transit_endpoint_centroid(row) for row in rows)
    candidates = tuple(
        (
            (centroid[0] - team_centroid[0]) / reference,
            (centroid[1] - team_centroid[1]) / reference,
            (centroid[2] - team_centroid[2]) / reference,
            *tuple(float(value) for value in row.planned_descriptor),
            float(row.quality_hint) / quality_scale,
            min(1.0, float(row.cost_hint) / cost_scale),
        )
        for row, centroid in zip(rows, centroids, strict=True)
    )
    node_features = ((0.0,) * MARL_IPP_INPUT_DIM, *candidates)
    node_count = len(node_features)
    adjacency_rows: list[tuple[bool, ...]] = []
    for left in range(node_count):
        row: list[bool] = []
        for right in range(node_count):
            if left == right or left == 0 or right == 0:
                row.append(True)
                continue
            spatial_distance = math.dist(centroids[left - 1], centroids[right - 1])
            descriptor_distance = math.dist(
                rows[left - 1].planned_descriptor,
                rows[right - 1].planned_descriptor,
            )
            row.append(spatial_distance <= reference or descriptor_distance <= 0.35)
        adjacency_rows.append(tuple(row))
    adjacency = tuple(adjacency_rows)
    remaining_fraction = min(
        1.0,
        max(
            0.0,
            dict(state.context.budget).get("time_remaining_s", state.decision_duration_s)
            / state.decision_duration_s,
        ),
    )
    budgets = (
        (remaining_fraction,),
        *tuple(
            (max(0.0, 1.0 - float(row.cost_hint) / cost_scale),)
            for row in rows
        ),
    )
    return MarlIPPGraphInput(
        node_features=node_features,
        adjacency=adjacency,
        budget_features=budgets,
        position_encoding=_normalized_laplacian_encoding(adjacency),
        legal_mask=(False, *tuple(row.feasible for row in rows)),
    )


@dataclass(frozen=True, slots=True)
class MarlIPPPortConfig:
    input_dim: int = MARL_IPP_INPUT_DIM
    embedding_dim: int = MARL_IPP_EMBEDDING_DIM
    learning_rate: float = 1.0e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.input_dim != MARL_IPP_INPUT_DIM or self.embedding_dim != MARL_IPP_EMBEDDING_DIM:
            raise ValueError(
                "MARL-IPP author checkpoint requires input_dim=8 and embedding_dim=128"
            )
        for name in ("learning_rate", "entropy_coef", "value_coef"):
            value = finite_number(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class MarlIPPTrainingRow:
    graph: MarlIPPGraphInput
    action: int
    reward: float

    def __post_init__(self) -> None:
        candidate_count = len(self.graph.legal_mask) - 1
        if not isinstance(self.action, int) or not 0 <= self.action < candidate_count:
            raise ValueError("MARL-IPP training action is outside the public pool")
        if not self.graph.legal_mask[self.action + 1]:
            raise ValueError("MARL-IPP training action is illegal")
        reward = finite_number(self.reward, "MARL-IPP reward")
        if not 0.0 <= reward <= 1.0:
            raise ValueError("MARL-IPP reward must lie in [0, 1]")
        object.__setattr__(self, "reward", reward)


class MarlIPPPortPolicy:
    """Trainable wrapper around the unmodified author AttentionNet."""

    def __init__(
        self,
        source_root: str | Path,
        config: MarlIPPPortConfig | None = None,
        *,
        source_checkpoint: str | Path | None = None,
        seed: int = 0,
    ) -> None:
        if torch is None:
            raise RuntimeError("MARL-IPP controlled transfer requires PyTorch")
        config = MarlIPPPortConfig() if config is None else config
        attention_net, source_hash = load_author_attention_net(source_root)
        self.source_root = Path(source_root).expanduser().resolve()
        self.source_attention_net_sha256 = source_hash
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(seed)
        self.model = attention_net(config.input_dim, config.embedding_dim).to(self.device)
        self.source_checkpoint_sha256: str | None = None
        if source_checkpoint is not None:
            checkpoint = Path(source_checkpoint).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"MARL-IPP author checkpoint is missing: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
                raise ValueError("MARL-IPP author checkpoint lacks the model state")
            self.model.load_state_dict(dict(payload["model"]), strict=True)
            self.source_checkpoint_sha256 = _sha256_file(checkpoint)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def _forward(self, graph: MarlIPPGraphInput) -> tuple[Any, Any]:
        node_count = len(graph.node_features)
        edge = tuple(tuple(range(node_count)) for _ in range(node_count))
        mask = tuple(
            tuple(not allowed for allowed in graph.adjacency[row])
            for row in range(node_count)
        )
        mask = tuple(
            tuple(
                value or (row == 0 and not graph.legal_mask[column])
                for column, value in enumerate(mask_row)
            )
            for row, mask_row in enumerate(mask)
        )
        hidden = torch.zeros((1, 1, self.config.embedding_dim), device=self.device)
        cell = torch.zeros_like(hidden)
        log_probabilities, value, _, _ = self.model(
            torch.tensor([graph.node_features], dtype=torch.float32, device=self.device),
            torch.tensor([edge], dtype=torch.long, device=self.device),
            torch.tensor([graph.budget_features], dtype=torch.float32, device=self.device),
            torch.tensor([[[0]]], dtype=torch.long, device=self.device),
            hidden,
            cell,
            torch.tensor([graph.position_encoding], dtype=torch.float32, device=self.device),
            torch.tensor([mask], dtype=torch.long, device=self.device),
        )
        return log_probabilities[0, 1:], value.reshape(-1)[0]

    def action_probabilities(self, graph: MarlIPPGraphInput) -> tuple[float, ...]:
        self.model.eval()
        with torch.no_grad():
            logs, _ = self._forward(graph)
            probabilities = logs.exp()
            probabilities = probabilities / probabilities.sum().clamp_min(1.0e-12)
        return tuple(float(value) for value in probabilities.cpu().tolist())

    def update(self, rows: Sequence[MarlIPPTrainingRow]) -> dict[str, float]:
        rows = tuple(rows)
        if not rows:
            raise ValueError("MARL-IPP update requires outcome-backed training rows")
        self.model.train()
        policy_losses = []
        value_losses = []
        entropies = []
        for row in rows:
            logs, value = self._forward(row.graph)
            probability = logs.exp()
            probability = probability / probability.sum().clamp_min(1.0e-12)
            normalized_logs = torch.log(probability.clamp_min(1.0e-12))
            reward = torch.tensor(row.reward, dtype=torch.float32, device=self.device)
            advantage = reward - value.detach()
            policy_losses.append(-normalized_logs[row.action] * advantage)
            value_losses.append(torch.nn.functional.mse_loss(value, reward))
            entropies.append(-(probability * normalized_logs).sum())
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("MARL-IPP update produced non-finite gradients")
        self.optimizer.step()
        diagnostics = {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError("MARL-IPP update produced non-finite diagnostics")
        return diagnostics

    def state_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "source_attention_net_sha256": self.source_attention_net_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if payload.get("config") != asdict(self.config):
            raise ValueError("MARL-IPP checkpoint config mismatch")
        if payload.get("source_attention_net_sha256") != self.source_attention_net_sha256:
            raise ValueError("MARL-IPP source attention_net.py hash mismatch")
        model = payload.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("MARL-IPP checkpoint lacks model state")
        self.model.load_state_dict(dict(model), strict=True)
        source_checkpoint_hash = payload.get("source_checkpoint_sha256")
        if source_checkpoint_hash is not None:
            _require_sha256(source_checkpoint_hash, "source_checkpoint_sha256")
        self.source_checkpoint_sha256 = source_checkpoint_hash


@dataclass(frozen=True, slots=True)
class MarlIPPSelection:
    selected_manifest_hash: str
    selected_candidate_id: str
    scores: tuple[tuple[str, float], ...]
    checkpoint_sha256: str
    training_provenance_sha256: str
    source_attention_net_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.selected_manifest_hash, "selected_manifest_hash")
        require_identifier(self.selected_candidate_id, "selected_candidate_id")
        for name in (
            "checkpoint_sha256",
            "training_provenance_sha256",
            "source_attention_net_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not self.scores:
            raise ValueError("MARL-IPP selection needs public probabilities")

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": MARL_IPP_PORT_ID,
            "schema_version": MARL_IPP_FEATURE_SCHEMA_VERSION,
            "adaptation_status": "controlled_transfer_not_original_target_mapping_reproduction",
            "selected_manifest_hash": self.selected_manifest_hash,
            "selected_candidate_id": self.selected_candidate_id,
            "scores": [list(row) for row in self.scores],
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_provenance_sha256": self.training_provenance_sha256,
            "source_attention_net_sha256": self.source_attention_net_sha256,
            "claim_limit": (
                "The authors' AttentionNet is trained and evaluated on the common public HM3D "
                "team-candidate graph. Original target mapping, simulator and rewards are replaced."
            ),
        }


def build_marl_ipp_training_transition(
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
    rows = tuple(pool)
    next_rows = tuple(next_pool)
    if not next_rows or not any(row.feasible for row in next_rows):
        raise ValueError("MARL-IPP training transition needs a legal next candidate graph")
    try:
        action = tuple(row.manifest_hash for row in rows).index(selected.manifest_hash)
    except ValueError as error:
        raise ValueError("selected MARL-IPP candidate is absent from the public pool") from error
    outcome_hashes = execution.get("outcome_hashes")
    if not isinstance(outcome_hashes, list) or not outcome_hashes:
        raise ValueError("MARL-IPP training needs actual execution outcome hashes")
    for outcome_hash in outcome_hashes:
        _require_sha256(outcome_hash, "execution outcome hash")
    if execution.get("manifest_hash") != selected.manifest_hash:
        raise ValueError("MARL-IPP execution manifest does not match the selected candidate")
    reward = finite_number(
        explored_free_flight_volume_auc_time_contribution,
        "explored_free_flight_volume_auc_time_contribution",
    )
    duration = finite_number(duration_s, "duration_s")
    if (
        not 0.0 <= reward <= 1.0
        or duration <= 0.0
        or not isinstance(terminated, bool)
        or not isinstance(truncated, bool)
        or (terminated and truncated)
    ):
        raise ValueError("MARL-IPP reward or duration is outside the public contract")
    require_identifier(scene_id, "scene_id")
    graph = public_marl_ipp_graph_input(state, rows)
    next_graph = public_marl_ipp_graph_input(next_state, next_rows)
    transition: dict[str, object] = {
        "schema_version": MARL_IPP_TRAINING_TRANSITION_SCHEMA_VERSION,
        "scene_id": scene_id,
        "decision_id": state.context.decision_id,
        "public_context_hash": state.context.digest,
        "public_candidate_pool_hash": canonical_sha256([row.to_dict() for row in rows]),
        **public_schema_fields(),
        "node_features": [list(row) for row in graph.node_features],
        "adjacency": [list(row) for row in graph.adjacency],
        "budget_features": [list(row) for row in graph.budget_features],
        "position_encoding": [list(row) for row in graph.position_encoding],
        "legal_mask": list(graph.legal_mask),
        "selected_action_index": action,
        "selected_candidate_id": selected.candidate_id,
        "selected_manifest_hash": selected.manifest_hash,
        "reward_explored_free_flight_volume_auc_time_contribution": reward,
        "cost_energy_j": finite_number(execution.get("total_energy_used_j"), "energy cost"),
        "duration_s": duration,
        "terminated": terminated,
        "truncated": truncated,
        "next_public_context_hash": next_state.context.digest,
        "next_public_candidate_pool_hash": canonical_sha256(
            [row.to_dict() for row in next_rows]
        ),
        "next_node_features": [list(row) for row in next_graph.node_features],
        "next_adjacency": [list(row) for row in next_graph.adjacency],
        "next_budget_features": [list(row) for row in next_graph.budget_features],
        "next_position_encoding": [list(row) for row in next_graph.position_encoding],
        "next_legal_mask": list(next_graph.legal_mask),
        "execution_outcome_hashes": list(outcome_hashes),
        "outcome_hash": canonical_sha256(
            {"manifest_hash": selected.manifest_hash, "outcome_hashes": outcome_hashes}
        ),
        "claim_limit": (
            "Train-only public candidate graph and outcome-backed exploration return. "
            "No target or evaluator-private geometry is present."
        ),
    }
    transition["transition_sha256"] = canonical_sha256(transition)
    walk_public_payload(transition)
    return transition


def build_marl_ipp_checkpoint_payload(
    model: MarlIPPPortPolicy,
    *,
    training_scene_ids: Sequence[str],
    training_updates: int,
    training_provenance: Mapping[str, Any],
    split_manifest_sha256: str,
) -> dict[str, Any]:
    if not isinstance(model, MarlIPPPortPolicy):
        raise TypeError("model must be MarlIPPPortPolicy")
    if (
        not isinstance(training_updates, int)
        or isinstance(training_updates, bool)
        or training_updates < 1
    ):
        raise ValueError("MARL-IPP checkpoint requires at least one real update")
    scenes = tuple(training_scene_ids)
    if not scenes or any(not isinstance(scene, str) or not scene for scene in scenes):
        raise ValueError("MARL-IPP checkpoint lacks training scenes")
    split_hash = _require_sha256(split_manifest_sha256, "split_manifest_sha256")
    if training_provenance.get("split_manifest_sha256") != split_hash:
        raise ValueError("MARL-IPP training provenance is not bound to the frozen split")
    provenance = dict(training_provenance)
    if not provenance:
        raise ValueError("MARL-IPP checkpoint lacks training provenance")
    require_current_public_schema(provenance, context="MARL-IPP training provenance")
    return {
        "schema_version": MARL_IPP_CHECKPOINT_SCHEMA_VERSION,
        "training_partition": "train",
        "split_manifest_sha256": split_hash,
        "training_scene_ids": list(scenes),
        "training_updates": training_updates,
        "training_provenance_sha256": canonical_sha256(provenance),
        "feature_schema_version": MARL_IPP_FEATURE_SCHEMA_VERSION,
        **public_schema_fields(),
        "marl_ipp_port_state": model.state_dict(),
    }


def _load_checkpoint(
    path: Path,
    *,
    source_root: str | Path,
    expected_split_manifest_sha256: str | None = None,
) -> tuple[MarlIPPPortPolicy, str, str]:
    if torch is None:
        raise RuntimeError("MARL-IPP controlled transfer requires PyTorch")
    if not path.is_file():
        raise FileNotFoundError(f"MARL-IPP checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("MARL-IPP checkpoint must be a mapping")
    if payload.get("schema_version") != MARL_IPP_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("MARL-IPP checkpoint schema mismatch")
    if payload.get("training_partition") != "train":
        raise ValueError("MARL-IPP checkpoint must be trained only on train scenes")
    split_hash = _require_sha256(payload.get("split_manifest_sha256"), "split_manifest_sha256")
    if expected_split_manifest_sha256 is not None and split_hash != expected_split_manifest_sha256:
        raise ValueError("MARL-IPP checkpoint belongs to a different frozen scene split")
    updates = payload.get("training_updates")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
        raise ValueError("MARL-IPP checkpoint has no real training updates")
    scenes = payload.get("training_scene_ids")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("MARL-IPP checkpoint lacks training-scene provenance")
    if payload.get("feature_schema_version") != MARL_IPP_FEATURE_SCHEMA_VERSION:
        raise ValueError("MARL-IPP feature schema mismatch")
    require_current_public_schema(payload, context="MARL-IPP checkpoint")
    provenance_hash = _require_sha256(
        payload.get("training_provenance_sha256"), "training_provenance_sha256"
    )
    state = payload.get("marl_ipp_port_state")
    if not isinstance(state, Mapping) or not isinstance(state.get("config"), Mapping):
        raise ValueError("MARL-IPP checkpoint lacks policy state")
    config = MarlIPPPortConfig(**dict(state["config"]))
    model = MarlIPPPortPolicy(source_root, config, seed=0)
    model.load_state_dict(state)
    return model, _sha256_file(path), provenance_hash


def select_marl_ipp_port(
    state: PublicSearchState,
    pool: Sequence[CandidateFragmentManifest],
    *,
    checkpoint_path: str | Path,
    source_root: str | Path,
    expected_split_manifest_sha256: str | None = None,
) -> tuple[CandidateFragmentManifest, MarlIPPSelection]:
    rows = tuple(pool)
    model, checkpoint_hash, provenance_hash = _load_checkpoint(
        Path(checkpoint_path),
        source_root=source_root,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
    )
    graph = public_marl_ipp_graph_input(state, rows)
    probabilities = model.action_probabilities(graph)
    selected_index = max(range(len(rows)), key=lambda index: probabilities[index])
    selected = rows[selected_index]
    if not selected.feasible:
        raise RuntimeError("masked MARL-IPP policy selected an illegal public candidate")
    return selected, MarlIPPSelection(
        selected_manifest_hash=selected.manifest_hash,
        selected_candidate_id=selected.candidate_id,
        scores=tuple(
            (row.candidate_id, score)
            for row, score in zip(rows, probabilities, strict=True)
        ),
        checkpoint_sha256=checkpoint_hash,
        training_provenance_sha256=provenance_hash,
        source_attention_net_sha256=model.source_attention_net_sha256,
    )


__all__ = [
    "MARL_IPP_CHECKPOINT_SCHEMA_VERSION",
    "MARL_IPP_FEATURE_SCHEMA_VERSION",
    "MARL_IPP_PORT_ID",
    "MARL_IPP_TRAINING_TRANSITION_SCHEMA_VERSION",
    "MarlIPPGraphInput",
    "MarlIPPPortConfig",
    "MarlIPPPortPolicy",
    "MarlIPPSelection",
    "MarlIPPTrainingRow",
    "build_marl_ipp_checkpoint_payload",
    "build_marl_ipp_training_transition",
    "load_author_attention_net",
    "public_marl_ipp_graph_input",
    "select_marl_ipp_port",
]
