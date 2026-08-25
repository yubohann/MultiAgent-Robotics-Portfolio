"""Stable method adapters, capability negotiation, and replay trajectories."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .canonical import content_hash, file_hash, read_json
from .contracts import ActionPacket, MessagePacket, ObservationPacket, Pose3D
from .geometry import segment_segment_distance
from .public_boundary import validate_public_episode
from .runtime import L0FleetRuntime, RuntimeStep

CAPABILITY_PROFILES = {
    "G1": {
        "name": "occupancy_voxel",
        "observations": [
            "self_state",
            "local_occupancy",
            "bounded_teammate_state",
            "budgeted_messages",
            "anonymous_confirmations",
        ],
        "forbidden": [
            "full_cityspec",
            "target_count",
            "target_coordinates",
            "support_sites",
            "target_process",
            "split_label",
            "evaluator_witnesses",
        ],
    },
    "G2-I": {
        "name": "public_inspection_prior",
        "observations": [
            "self_state",
            "local_occupancy",
            "bounded_teammate_state",
            "budgeted_messages",
            "anonymous_confirmations",
            "public_inspection_prior_on_reset",
        ],
        "forbidden": [
            "target_count",
            "target_coordinates",
            "support_sites",
            "target_process",
            "split_label",
            "evaluator_witnesses",
        ],
    },
    "G2": {
        "name": "range_fov",
        "observations": ["self_state", "bounded_range_points", "budgeted_messages"],
        "forbidden": ["target_truth", "full_cityspec", "split_label"],
    },
    "P1": {
        "name": "rgbd_perception",
        "observations": ["self_state", "rgb", "depth", "camera_calibration"],
        "forbidden": ["target_truth", "geometry_oracle_detection", "split_label"],
    },
    "C1": {
        "name": "low_level_control",
        "observations": ["proprioception", "camera_or_range"],
        "forbidden": ["target_truth", "shared_low_level_controller"],
    },
}

_PROCESS_BOUNDARIES = frozenset({"in_process", "process", "container", "ros"})
_COPYLEFT_LICENSE_PREFIXES = ("gpl-", "agpl-", "lgpl-")
_FULL_GIT_REVISION = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_OCI_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXTERNAL_WIRE_FORBIDDEN_KEYS = frozenset(
    {
        "counterfactual_pairs",
        "distractors",
        "evaluator_seed",
        "evaluator_witnesses",
        "legal_witnesses",
        "layout_seed",
        "master_seed",
        "private_evaluator",
        "split_label",
        "support_sites",
        "target_id",
        "target_ids",
        "target_coordinates",
        "target_count",
        "target_process",
        "target_truth",
        "targets",
    }
)


def arbitrate_public_fleet_actions(
    actions: dict[str, ActionPacket],
    observations: dict[str, ObservationPacket],
    *,
    vehicle_radius_m: float,
) -> dict[str, ActionPacket]:
    """Apply a deterministic one-step yield to public fleet trajectories.

    External planners commonly emit independent waypoint actions.  The shared
    L0/L1 executor evaluates the resulting segments simultaneously, so an
    adapter must reject a predicted vehicle-to-vehicle crossing before it is
    handed to the executor.  This arbiter intentionally consumes no city,
    target, evaluator, or private state.
    """

    if len(actions) < 2:
        return actions
    if vehicle_radius_m <= 0.0:
        raise ValueError("public fleet arbitration requires a positive vehicle radius")
    conflict_distance = 2.0 * vehicle_radius_m + 0.02
    positions = {
        drone_id: observation.pose.position for drone_id, observation in observations.items()
    }
    for observation in observations.values():
        for teammate in observation.teammate_states:
            teammate_id = str(teammate.get("drone_id", ""))
            teammate_position = teammate.get("position")
            if (
                teammate_id
                and teammate_id not in positions
                and isinstance(teammate_position, (list, tuple))
                and len(teammate_position) == 3
            ):
                positions[teammate_id] = tuple(float(value) for value in teammate_position)

    def endpoint(drone_id: str, action: ActionPacket) -> tuple[float, float, float]:
        if action.kind == "WAYPOINT" and action.waypoint is not None:
            return action.waypoint.position
        return positions[drone_id]

    def hold(action: ActionPacket) -> ActionPacket:
        return ActionPacket(
            episode_id=action.episode_id,
            drone_id=action.drone_id,
            sequence=action.sequence,
            issued_at_s=action.issued_at_s,
            kind="HOVER",
            messages=action.messages,
        )

    adjusted = dict(actions)
    for first_index, first_id in enumerate(sorted(adjusted)):
        if first_id not in positions:
            continue
        for second_id in sorted(adjusted)[first_index + 1 :]:
            if second_id not in positions:
                continue
            first_start = positions[first_id]
            second_start = positions[second_id]
            first_end = endpoint(first_id, adjusted[first_id])
            second_end = endpoint(second_id, adjusted[second_id])
            predicted = segment_segment_distance(first_start, first_end, second_start, second_end)
            if predicted + 1.0e-9 >= conflict_distance:
                continue
            hold_first_distance = segment_segment_distance(
                first_start, first_start, second_start, second_end
            )
            hold_second_distance = segment_segment_distance(
                first_start, first_end, second_start, second_start
            )
            if hold_second_distance + 1.0e-9 >= conflict_distance:
                adjusted[second_id] = hold(adjusted[second_id])
            elif hold_first_distance + 1.0e-9 >= conflict_distance:
                adjusted[first_id] = hold(adjusted[first_id])
            else:
                adjusted[first_id] = hold(adjusted[first_id])
                adjusted[second_id] = hold(adjusted[second_id])
    return adjusted
_EXTERNAL_WIRE_FORBIDDEN_PREFIXES = ("evaluator_", "private_", "target_")
_EXTERNAL_AUDIT_FALSE_SENTINELS = frozenset(
    {"formal_split_label_public", "target_count_public", "target_process_public"}
)
_EXTERNAL_REQUEST_SCHEMA = "org.aerocity.bench.external-planner-request.v1"
_EXTERNAL_RESPONSE_SCHEMA = "org.aerocity.bench.external-planner-response.v1"


@dataclass(frozen=True)
class AdapterDeclaration:
    adapter_id: str
    method_id: str
    capability_profile: str
    upstream_url: str | None
    upstream_commit: str | None
    upstream_license: str
    process_boundary: str
    training_allowed: bool
    decentralized_execution: bool
    runtime_image_digest: str | None = None

    def validate(self) -> None:
        if not self.adapter_id or not self.method_id:
            raise ValueError("adapter and method IDs cannot be empty")
        if self.capability_profile not in CAPABILITY_PROFILES:
            raise ValueError(f"unknown capability profile: {self.capability_profile}")
        if bool(self.upstream_url) != bool(self.upstream_commit):
            raise ValueError("upstream URL and frozen upstream commit must be declared together")
        if self.upstream_commit and not _FULL_GIT_REVISION.fullmatch(self.upstream_commit):
            raise ValueError(
                "upstream_commit must be a full 40- or 64-character lowercase git revision"
            )
        license_name = self.upstream_license.lower().strip()
        if license_name in {"unknown", "none", ""}:
            raise ValueError("formal adapters require a known upstream license")
        if self.process_boundary not in _PROCESS_BOUNDARIES:
            raise ValueError("adapter process_boundary is unsupported")
        if license_name.startswith(_COPYLEFT_LICENSE_PREFIXES):
            if self.process_boundary not in {"container", "process", "ros"}:
                raise ValueError("copyleft baselines must remain behind an independent boundary")
        if self.process_boundary == "container":
            if not self.runtime_image_digest or not _OCI_IMAGE_DIGEST.fullmatch(
                self.runtime_image_digest
            ):
                raise ValueError("container adapters require a pinned sha256 OCI image digest")
        elif self.runtime_image_digest is not None:
            raise ValueError("only container adapters may declare a runtime_image_digest")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "org.aerocity.bench.adapter-declaration.v1",
            **self.__dict__,
        }


EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA = "org.aerocity.bench.external-l1-adapter-manifest.v3"
_EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA_V1 = "org.aerocity.bench.external-l1-adapter-manifest.v1"
_EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA_V2 = "org.aerocity.bench.external-l1-adapter-manifest.v2"
_EXTERNAL_L1_TASK_DOMAINS = frozenset({"2d_exploration_transfer", "3d_geometry_search"})
_EXTERNAL_L1_COMPARABILITY = frozenset({"transfer_diagnostic", "substantive_3d"})
_EXTERNAL_L1_EXECUTION_ARTIFACT_KINDS = frozenset({"checkpoint", "solver_lock"})


@dataclass(frozen=True)
class ExternalL1AdapterManifest:
    """Pinned local launch material for an external L1 process diagnostic.

    Paths are necessary to execute a locally installed upstream method, but no
    path or command is copied into public evidence.  The report carries only
    hashes and the upstream declaration, which binds the implementation without
    leaking a workstation layout or turning a process bridge into a sandbox.
    """

    declaration: AdapterDeclaration
    command: tuple[str, ...]
    adapter_source_path: Path
    adapter_source_sha256: str
    upstream_source_path: Path
    execution_artifact_kind: str
    execution_artifact_path: Path
    execution_artifact_sha256: str
    runtime_environment_sha256: str
    runtime_python_path: Path | None
    runtime_python_sha256: str | None
    task_domain: str
    comparability_claim: str
    manifest_file_sha256: str

    def validate(self) -> None:
        self.declaration.validate()
        if self.declaration.process_boundary != "process":
            raise ValueError("external L1 manifests require a real process boundary")
        if self.declaration.capability_profile != "G2-I":
            raise ValueError("external L1 manifests require the G2-I capability profile")
        if not self.command or any(not item or not isinstance(item, str) for item in self.command):
            raise ValueError("external L1 manifest command must be a non-empty string vector")
        required_command_tokens = {
            "{python_executable}",
            "{adapter_source}",
            "{upstream_source}",
        }
        if self.execution_artifact_kind == "checkpoint":
            required_command_tokens.add("{checkpoint}")
        if not required_command_tokens <= set(self.command):
            raise ValueError(
                "external L1 manifest command must bind the runtime, adapter source, "
                "upstream source, and any required execution artifact"
            )
        for expected, actual, label in (
            (self.adapter_source_sha256, file_hash(self.adapter_source_path), "adapter source"),
            (
                self.execution_artifact_sha256,
                file_hash(self.execution_artifact_path),
                "execution artifact",
            ),
        ):
            if not _SHA256.fullmatch(expected):
                raise ValueError(f"external L1 {label} hash is not SHA-256")
            if expected != actual:
                raise ValueError(f"external L1 {label} hash differs from the manifest")
        if not _SHA256.fullmatch(self.runtime_environment_sha256):
            raise ValueError("external L1 runtime environment hash is not SHA-256")
        if (self.runtime_python_path is None) != (self.runtime_python_sha256 is None):
            raise ValueError(
                "external L1 runtime Python path and hash must be declared together"
            )
        if self.runtime_python_path is not None:
            if not self.runtime_python_path.is_file():
                raise ValueError("external L1 runtime Python is not a readable file")
            assert self.runtime_python_sha256 is not None
            if not _SHA256.fullmatch(self.runtime_python_sha256):
                raise ValueError("external L1 runtime Python hash is not SHA-256")
            if file_hash(self.runtime_python_path) != self.runtime_python_sha256:
                raise ValueError("external L1 runtime Python hash differs from the manifest")
        if self.execution_artifact_kind not in _EXTERNAL_L1_EXECUTION_ARTIFACT_KINDS:
            raise ValueError("external L1 execution artifact kind is unsupported")
        if self.execution_artifact_kind == "solver_lock" and self.declaration.training_allowed:
            raise ValueError("an external solver lock cannot declare training")
        if not self.upstream_source_path.is_dir():
            raise ValueError("external L1 upstream source path is not a readable directory")
        try:
            head = subprocess.run(
                ["git", "-C", str(self.upstream_source_path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().lower()
            worktree_state = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.upstream_source_path),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("external L1 upstream source is not a readable Git worktree") from exc
        if head != self.declaration.upstream_commit:
            raise ValueError("external L1 upstream Git HEAD differs from the declared revision")
        unexpected_worktree_entries = []
        for entry in worktree_state.splitlines():
            # Python writes bytecode next to imported upstream modules.  Those
            # cache files are not executable source and are the only untracked
            # entries tolerated; tracked modifications and every other
            # untracked file remain a hard failure.
            path = entry[3:].replace("\\", "/") if entry.startswith("?? ") else ""
            bytecode_cache = entry.startswith("?? ") and "/__pycache__/" in f"/{path}"
            if not (bytecode_cache and path.endswith(".pyc")):
                unexpected_worktree_entries.append(entry)
        if unexpected_worktree_entries:
            raise ValueError("external L1 upstream Git worktree must be clean")
        if self.task_domain not in _EXTERNAL_L1_TASK_DOMAINS:
            raise ValueError("external L1 task domain is unsupported")
        if self.comparability_claim not in _EXTERNAL_L1_COMPARABILITY:
            raise ValueError("external L1 comparability claim is unsupported")
        if (
            self.task_domain != "3d_geometry_search"
            and self.comparability_claim != "transfer_diagnostic"
        ):
            raise ValueError("a non-3-D external method can only be a transfer diagnostic")
        if not _SHA256.fullmatch(self.manifest_file_sha256):
            raise ValueError("external L1 manifest file hash is not SHA-256")

    def launch_command(self, python_executable: str | None = None) -> list[str]:
        """Materialize a verified local process command without publishing paths.

        Version 3 manifests bind the isolated interpreter used by the external
        method.  Version 1 and 2 manifests are legacy inputs and require an
        explicit interpreter from their caller.
        """

        self.validate()
        if self.runtime_python_path is not None:
            locked_python = str(self.runtime_python_path)
            if (
                python_executable is not None
                and Path(python_executable).resolve() != self.runtime_python_path
            ):
                raise ValueError("external L1 launch Python differs from the manifest lock")
        elif not python_executable:
            raise ValueError("external L1 process requires a Python executable")
        else:
            locked_python = python_executable
        substitutions = {
            "{python_executable}": locked_python,
            "{adapter_source}": str(self.adapter_source_path),
            "{upstream_source}": str(self.upstream_source_path),
            "{checkpoint}": str(self.execution_artifact_path),
            "{execution_artifact}": str(self.execution_artifact_path),
        }
        return [substitutions.get(item, item) for item in self.command]

    def public_provenance(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "org.aerocity.bench.external-l1-adapter-provenance.v2",
            "adapter_manifest_sha256": self.manifest_file_sha256,
            "declaration": self.declaration.to_dict(),
            "adapter_source_sha256": self.adapter_source_sha256,
            "execution_artifact_kind": self.execution_artifact_kind,
            "execution_artifact_sha256": self.execution_artifact_sha256,
            "runtime_environment_sha256": self.runtime_environment_sha256,
            "runtime_python_sha256": self.runtime_python_sha256,
            "command_template_sha256": content_hash({"command": list(self.command)}),
            "task_domain": self.task_domain,
            "comparability_claim": self.comparability_claim,
        }


def load_external_l1_adapter_manifest(path: Path) -> ExternalL1AdapterManifest:
    """Load a pinned process declaration without publishing local path strings."""

    manifest_path = path.resolve()
    raw = read_json(manifest_path)
    if not isinstance(raw, dict):
        raise ValueError("external L1 adapter manifest must be a JSON object")
    expected_v1 = {
        "schema",
        "declaration",
        "command",
        "adapter_source_path",
        "adapter_source_sha256",
        "upstream_source_path",
        "checkpoint_path",
        "checkpoint_sha256",
        "runtime_environment_sha256",
        "task_domain",
        "comparability_claim",
    }
    expected_v2 = {
        "schema",
        "declaration",
        "command",
        "adapter_source_path",
        "adapter_source_sha256",
        "upstream_source_path",
        "execution_artifact_kind",
        "execution_artifact_path",
        "execution_artifact_sha256",
        "runtime_environment_sha256",
        "task_domain",
        "comparability_claim",
    }
    expected_v3 = {
        *expected_v2,
        "runtime_python_path",
        "runtime_python_sha256",
    }
    schema = raw.get("schema")
    if (
        (schema == _EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA_V1 and set(raw) == expected_v1)
        or (schema == _EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA_V2 and set(raw) == expected_v2)
        or (schema == EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA and set(raw) == expected_v3)
    ) is False:
        raise ValueError("external L1 adapter manifest schema or fields differ")
    declaration_raw = raw["declaration"]
    declaration_fields = {
        "adapter_id",
        "method_id",
        "capability_profile",
        "upstream_url",
        "upstream_commit",
        "upstream_license",
        "process_boundary",
        "training_allowed",
        "decentralized_execution",
        "runtime_image_digest",
    }
    if not isinstance(declaration_raw, dict) or set(declaration_raw) != declaration_fields:
        raise ValueError("external L1 adapter declaration fields differ")
    command = raw["command"]
    if not isinstance(command, list):
        raise ValueError("external L1 manifest command must be a JSON list")

    def resolve_file(field: str) -> Path:
        value = raw[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"external L1 manifest {field} must be a non-empty path")
        candidate = (manifest_path.parent / value).resolve()
        if not candidate.is_file():
            raise ValueError(f"external L1 manifest {field} is not a readable file")
        return candidate

    def resolve_directory(field: str) -> Path:
        value = raw[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"external L1 manifest {field} must be a non-empty path")
        candidate = (manifest_path.parent / value).resolve()
        if not candidate.is_dir():
            raise ValueError(f"external L1 manifest {field} is not a readable directory")
        return candidate

    def resolve_optional_file(field: str) -> Path | None:
        if schema != EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA:
            return None
        return resolve_file(field)

    artifact_kind = "checkpoint" if schema == _EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA_V1 else str(
        raw["execution_artifact_kind"]
    )
    artifact_path = "checkpoint_path" if schema == _EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA_V1 else (
        "execution_artifact_path"
    )
    artifact_hash = "checkpoint_sha256" if schema == _EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA_V1 else (
        "execution_artifact_sha256"
    )
    manifest = ExternalL1AdapterManifest(
        declaration=AdapterDeclaration(**declaration_raw),
        command=tuple(command),
        adapter_source_path=resolve_file("adapter_source_path"),
        adapter_source_sha256=str(raw["adapter_source_sha256"]),
        upstream_source_path=resolve_directory("upstream_source_path"),
        execution_artifact_kind=artifact_kind,
        execution_artifact_path=resolve_file(artifact_path),
        execution_artifact_sha256=str(raw[artifact_hash]),
        runtime_environment_sha256=str(raw["runtime_environment_sha256"]),
        runtime_python_path=resolve_optional_file("runtime_python_path"),
        runtime_python_sha256=(
            str(raw["runtime_python_sha256"])
            if schema == EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA
            else None
        ),
        task_domain=str(raw["task_domain"]),
        comparability_claim=str(raw["comparability_claim"]),
        manifest_file_sha256=file_hash(manifest_path),
    )
    manifest.validate()
    return manifest


class Planner(Protocol):
    def reset(self, task_spec: dict[str, Any]) -> None: ...

    def act(self, observations: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]: ...


class AgentPlanner(Protocol):
    def reset(self, task_spec: dict[str, Any], drone_id: str) -> None: ...

    def act(self, observation: dict[str, Any]) -> dict[str, Any]: ...


def project_g1(observation: ObservationPacket) -> dict[str, Any]:
    return {
        "schema": "org.aerocity.bench.g1-observation.v2",
        "episode_id": observation.episode_id,
        "observation_id": observation.observation_id,
        "drone_id": observation.drone_id,
        "sequence": observation.sequence,
        "timestamp_s": observation.timestamp_s,
        "self_state": {
            "pose": observation.pose.to_dict(),
            "sensor_pitch_deg": observation.sensor_pitch_deg,
            "linear_velocity_world_mps": list(observation.linear_velocity_world_mps),
            "angular_speed_deg_s": observation.angular_speed_deg_s,
            "energy_remaining_j": observation.energy_remaining_j,
            "health": observation.health,
        },
        "local_occupancy": [list(cell) for cell in observation.local_occupancy],
        "local_occupancy_origin_world_m": list(
            observation.local_occupancy_origin_world_m
        ),
        "local_occupancy_resolution_m": observation.local_occupancy_resolution_m,
        "local_occupancy_radius_m": observation.local_occupancy_radius_m,
        "teammate_states": list(observation.teammate_states),
        "received_messages": [message.to_dict() for message in observation.received_messages],
    }


def _messages_from_dict(node: object, observation: ObservationPacket) -> tuple[MessagePacket, ...]:
    if node is None:
        return ()
    if not isinstance(node, list):
        raise ValueError("adapter messages must be a list")
    messages = []
    expected = {"message_id", "destination_drone_ids", "expires_at_s", "payload_hex"}
    for record in node:
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError("adapter message fields differ from the canonical contract")
        destinations = record["destination_drone_ids"]
        if not isinstance(destinations, list):
            raise ValueError("adapter message destinations must be a list")
        try:
            payload = bytes.fromhex(str(record["payload_hex"]))
        except ValueError as exc:
            raise ValueError("adapter message payload_hex is invalid") from exc
        messages.append(
            MessagePacket(
                message_id=str(record["message_id"]),
                source_drone_id=observation.drone_id,
                destination_drone_ids=tuple(str(value) for value in destinations),
                created_at_s=observation.timestamp_s,
                expires_at_s=float(record["expires_at_s"]),
                payload=payload,
            )
        )
    return tuple(messages)


def _action_from_dict(
    node: dict[str, Any], observation: ObservationPacket, sequence: int
) -> ActionPacket:
    allowed = {
        "kind",
        "waypoint",
        "velocity_body_mps",
        "yaw_rate_deg_s",
        "sensor_pitch_deg",
        "source_observation_id",
        "messages",
    }
    extra = sorted(set(node) - allowed)
    if extra:
        raise ValueError(f"adapter action has unknown fields: {extra}")
    kind = str(node["kind"])
    waypoint = Pose3D.from_dict(node["waypoint"]) if node.get("waypoint") else None
    source = node.get("source_observation_id")
    if kind == "OBSERVE":
        source = source or observation.observation_id
    return ActionPacket(
        episode_id=observation.episode_id,
        drone_id=observation.drone_id,
        sequence=sequence,
        issued_at_s=observation.timestamp_s,
        kind=kind,  # type: ignore[arg-type]
        waypoint=waypoint,
        velocity_body_mps=(
            tuple(float(value) for value in node["velocity_body_mps"])
            if node.get("velocity_body_mps") is not None
            else None
        ),  # type: ignore[arg-type]
        yaw_rate_deg_s=float(node.get("yaw_rate_deg_s", 0.0)),
        sensor_pitch_deg=(
            float(node["sensor_pitch_deg"])
            if node.get("sensor_pitch_deg") is not None
            else None
        ),
        source_observation_id=str(source) if source is not None else None,
        messages=_messages_from_dict(node.get("messages"), observation),
    )


class PlannerAdapter:
    """Convert a language-neutral planner dictionary API to canonical packets."""

    def __init__(self, declaration: AdapterDeclaration, planner: Planner) -> None:
        declaration.validate()
        if declaration.capability_profile not in {"G1", "G2-I"}:
            raise ValueError("PlannerAdapter implements canonical G1 or G2-I observations")
        if declaration.process_boundary != "in_process":
            raise ValueError(
                "PlannerAdapter cannot execute a process/container/ROS declaration in-process; "
                "use a real external bridge"
            )
        if declaration.decentralized_execution:
            raise ValueError(
                "fleet PlannerAdapter exposes a centralized observation dictionary; "
                "use DecentralizedPlannerAdapter for decentralized execution"
            )
        self.declaration = declaration
        self.planner = planner
        self.adapter_latencies_s: list[float] = []

    def reset(
        self,
        public_episode: dict[str, Any],
        *,
        public_task_spec: dict[str, Any] | None = None,
    ) -> None:
        if self.declaration.capability_profile == "G2-I":
            if (
                not isinstance(public_task_spec, dict)
                or public_task_spec.get("task_track") != "G2-I"
            ):
                raise ValueError("G2-I adapter reset requires a public G2-I task spec")
            validate_public_episode(public_episode, public_task_spec)
            _assert_external_wire_is_public(public_task_spec)
            self.planner.reset(
                {
                    "schema": "org.aerocity.bench.g2-i-planner-reset.v1",
                    "public_episode": _external_process_episode_projection(public_episode),
                    "public_task_spec": _external_process_episode_projection(public_task_spec),
                }
            )
            return
        if public_task_spec is not None:
            raise ValueError("G1 adapter reset must not receive a G2-I task spec")
        self.planner.reset(public_episode)

    def act(
        self, observations: dict[str, ObservationPacket]
    ) -> tuple[dict[str, ActionPacket], dict[str, float]]:
        start = time.perf_counter()
        projection = {drone_id: project_g1(packet) for drone_id, packet in observations.items()}
        method_actions = self.planner.act(projection)
        elapsed = time.perf_counter() - start
        if set(method_actions) != set(observations):
            raise ValueError("planner actions must exactly match active observation IDs")
        actions = {
            drone_id: _action_from_dict(method_actions[drone_id], packet, packet.sequence)
            for drone_id, packet in observations.items()
        }
        latencies = {drone_id: elapsed for drone_id in actions}
        self.adapter_latencies_s.append(elapsed)
        return actions, latencies

    def adapter_tax_report(self) -> dict[str, Any]:
        ordered = sorted(self.adapter_latencies_s)
        return {
            "schema": "org.aerocity.bench.adapter-tax.v1",
            "adapter_id": self.declaration.adapter_id,
            "call_count": len(ordered),
            "total_s": sum(ordered),
            "median_s": ordered[len(ordered) // 2] if ordered else None,
            "maximum_s": ordered[-1] if ordered else None,
        }


class DecentralizedPlannerAdapter:
    """Run one isolated planner object per agent with no fleet observation dictionary."""

    def __init__(
        self,
        declaration: AdapterDeclaration,
        planners: dict[str, AgentPlanner],
    ) -> None:
        declaration.validate()
        if declaration.capability_profile != "G1" or not declaration.decentralized_execution:
            raise ValueError("decentralized adapter requires a decentralized G1 declaration")
        if declaration.process_boundary != "in_process":
            raise ValueError(
                "DecentralizedPlannerAdapter cannot execute a process/container/ROS declaration "
                "in-process; use a real external bridge"
            )
        if not planners:
            raise ValueError("decentralized adapter requires at least one agent planner")
        self.declaration = declaration
        self.planners = dict(planners)
        self.adapter_latencies_s: list[float] = []

    def reset(self, public_episode: dict[str, Any]) -> None:
        expected = {str(item["drone_id"]) for item in public_episode["starts"]}
        if set(self.planners) != expected:
            raise ValueError("agent planner IDs differ from the public fleet")
        for drone_id, planner in self.planners.items():
            planner.reset(public_episode, drone_id)

    def act(
        self, observations: dict[str, ObservationPacket]
    ) -> tuple[dict[str, ActionPacket], dict[str, float]]:
        if set(observations) - set(self.planners):
            raise ValueError("an observation has no isolated agent planner")
        actions: dict[str, ActionPacket] = {}
        latencies: dict[str, float] = {}
        for drone_id, observation in sorted(observations.items()):
            start = time.perf_counter()
            method_action = self.planners[drone_id].act(project_g1(observation))
            elapsed = time.perf_counter() - start
            actions[drone_id] = _action_from_dict(method_action, observation, observation.sequence)
            latencies[drone_id] = elapsed
            self.adapter_latencies_s.append(elapsed)
        return actions, latencies


def _normalized_wire_key(key: object) -> str:
    """Canonicalize a JSON object key before checking its information class.

    JSON object keys must be strings.  Rejecting non-string keys before
    serialization avoids a Python-only integer key being silently converted
    into a different wire key, while normalization prevents spelling variants
    such as ``Target-ID`` from bypassing the private-field policy.
    """

    if not isinstance(key, str):
        raise ValueError("external planner wire payload contains a non-string object key")
    if not key.isascii():
        raise ValueError("external planner wire payload contains a non-ASCII object key")
    return re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")


def _assert_external_wire_is_public(value: object, *, path: str = "$") -> None:
    """Reject an accidental evaluator-private field before it reaches a process.

    A process boundary is useful for licence isolation, but it is not a sandbox.
    This check consequently protects the benchmark-owned wire payload itself;
    formal blind execution still needs its separate read-only mount and network
    policy.
    """

    if isinstance(value, dict):
        forbidden = []
        for key, nested in value.items():
            normalized = _normalized_wire_key(key)
            if normalized in _EXTERNAL_AUDIT_FALSE_SENTINELS and nested is False:
                continue
            if (
                normalized in _EXTERNAL_WIRE_FORBIDDEN_KEYS
                or normalized.startswith(_EXTERNAL_WIRE_FORBIDDEN_PREFIXES)
            ):
                forbidden.append(key)
        if forbidden:
            raise ValueError(
                "external planner wire payload contains private fields at "
                f"{path}: {sorted(str(key) for key in forbidden)}"
            )
        for key, nested in value.items():
            _assert_external_wire_is_public(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_external_wire_is_public(nested, path=f"{path}[{index}]")


def _external_process_episode_projection(public_episode: dict[str, Any]) -> dict[str, Any]:
    """Remove local audit sentinels before serializing a process request.

    The regular public projection keeps these ``false`` fields so local
    builders can prove that target count, target process, and formal split
    labels were not published. They are contract metadata rather than method
    input, so the process wire removes them entirely after validating their
    value. This leaves no target or split-boundary field for an external method
    to branch on.
    """

    _assert_external_wire_is_public(public_episode)
    def scrub(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: scrub(nested)
                for key, nested in value.items()
                if _normalized_wire_key(key) not in _EXTERNAL_AUDIT_FALSE_SENTINELS
            }
        if isinstance(value, list):
            return [scrub(nested) for nested in value]
        if isinstance(value, tuple):
            return [scrub(nested) for nested in value]
        return value

    cleaned = scrub(public_episode)
    assert isinstance(cleaned, dict)  # ``public_episode`` is a dictionary input.
    return cleaned


class ExternalProcessPlannerBridge:
    """Run a centralized G1/G2-I planner through a persistent JSONL process.

    The bridge is deliberately limited to ``process`` declarations.  A
    ``container`` declaration must be launched by a container runner that
    verifies its OCI digest, and a ``ros`` declaration must use a real ROS
    graph.  Treating either as an ordinary local executable would create a
    false licence or isolation claim.

    The external executable receives exactly one JSON object per line and must
    return one matching response line::

        {"schema": "org.aerocity.bench.external-planner-request.v1",
         "request_id": "...", "kind": "reset", "public_episode": {...}}
        {"schema": "org.aerocity.bench.external-planner-response.v1",
         "request_id": "...", "status": "ok"}

    For ``act``, the request carries a G1 projection keyed by drone ID and the
    response additionally carries an ``actions`` object in the canonical
    language-neutral action dictionary format.  This is a development and
    integration bridge, not a blind-evaluator sandbox.
    """

    def __init__(
        self,
        declaration: AdapterDeclaration,
        command: list[str],
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
        response_timeout_s: float = 5.0,
        initialization_timeout_s: float | None = None,
        maximum_line_bytes: int = 1_000_000,
    ) -> None:
        declaration.validate()
        if declaration.capability_profile not in {"G1", "G2-I"}:
            raise ValueError(
                "ExternalProcessPlannerBridge implements canonical G1 or G2-I observations"
            )
        if declaration.process_boundary != "process":
            raise ValueError(
                "ExternalProcessPlannerBridge only runs a real process declaration; "
                "container and ROS declarations require their own runners"
            )
        if declaration.decentralized_execution:
            raise ValueError(
                "ExternalProcessPlannerBridge exposes a centralized observation dictionary; "
                "it cannot claim decentralized execution"
            )
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("external process command must be a non-empty string vector")
        if response_timeout_s <= 0.0:
            raise ValueError("external process response timeout must be positive")
        if initialization_timeout_s is not None and initialization_timeout_s <= 0.0:
            raise ValueError("external process initialization timeout must be positive")
        if maximum_line_bytes < 256:
            raise ValueError("external process maximum response line must be at least 256 bytes")

        self.declaration = declaration
        self.command = tuple(command)
        self.response_timeout_s = float(response_timeout_s)
        self.initialization_timeout_s = float(
            response_timeout_s if initialization_timeout_s is None else initialization_timeout_s
        )
        self.maximum_line_bytes = int(maximum_line_bytes)
        self.adapter_latencies_s: list[float] = []
        self.initialization_latency_s: float | None = None
        self._last_request_timing: dict[str, float] | None = None
        self._last_act_timing: dict[str, float] | None = None
        self._request_sequence = 0
        self._closed = False
        self._reset = False
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._termination_lock = threading.Lock()
        child_environment = self._child_environment(environment)
        self._process = subprocess.Popen(
            self.command,
            cwd=str(cwd) if cwd is not None else None,
            env=child_environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
            shell=False,
        )
        if self._process.stdin is None or self._process.stdout is None:  # pragma: no cover
            self.close()
            raise RuntimeError("external process did not expose a JSONL standard stream")
        self._reader = threading.Thread(
            target=self._pump_responses,
            name=f"aerocity-external-planner-{self._process.pid}",
            daemon=True,
        )
        self._reader.start()

    @staticmethod
    def _child_environment(environment: dict[str, str] | None) -> dict[str, str]:
        """Pass only explicitly supplied variables plus runtime essentials."""

        allowed = ("PATH", "SYSTEMROOT", "WINDIR", "HOME", "TMP", "TEMP")
        child = {key: os.environ[key] for key in allowed if key in os.environ}
        child["PYTHONUNBUFFERED"] = "1"
        if environment is not None:
            if any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in environment.items()
            ):
                raise ValueError("external process environment must contain only string pairs")
            child.update(environment)
        return child

    def _pump_responses(self) -> None:
        assert self._process.stdout is not None
        try:
            while True:
                line = self._process.stdout.readline(self.maximum_line_bytes + 1)
                if not line:
                    self._responses.put(None)
                    return
                if len(line.encode("utf-8")) > self.maximum_line_bytes:
                    self._responses.put(None)
                    self._terminate()
                    return
                if not line.endswith("\n"):
                    self._responses.put(None)
                    self._terminate()
                    return
                self._responses.put(line)
        except (OSError, UnicodeError):
            self._responses.put(None)

    def _terminate(self) -> None:
        """Stop only this bridge's owned process tree after a failed exchange."""

        with self._termination_lock:
            if self._process.poll() is not None:
                return
            if os.name == "nt":
                # An external ROS/planner launcher can retain children after
                # its Python parent exits.  The PID was created by this bridge,
                # so a tree kill remains scoped to the owned adapter attempt.
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(self._process.pid), "/T", "/F"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=10.0,
                    )
                except (OSError, subprocess.SubprocessError):
                    self._process.kill()
            else:
                self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)

    def _request(
        self, kind: str, *, timeout_s: float | None = None, **payload: Any
    ) -> dict[str, Any]:
        request_started = time.perf_counter()
        request_timing = {
            "request_public_audit_wall_clock_s": 0.0,
            "request_json_serialize_wall_clock_s": 0.0,
            "request_size_check_wall_clock_s": 0.0,
            "request_write_flush_wall_clock_s": 0.0,
            "response_wait_wall_clock_s": 0.0,
            "response_json_decode_wall_clock_s": 0.0,
            "response_validate_wall_clock_s": 0.0,
        }
        try:
            if self._closed:
                raise RuntimeError("external planner bridge is closed")
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"external planner process exited with {self._process.returncode}"
                )
            request_id = f"{self.declaration.adapter_id}-{self._request_sequence:08d}"
            self._request_sequence += 1
            request = {
                "schema": _EXTERNAL_REQUEST_SCHEMA,
                "request_id": request_id,
                "kind": kind,
                **payload,
            }
            stage_started = time.perf_counter()
            try:
                _assert_external_wire_is_public(request)
            finally:
                request_timing["request_public_audit_wall_clock_s"] = (
                    time.perf_counter() - stage_started
                )
            stage_started = time.perf_counter()
            try:
                encoded = json.dumps(request, ensure_ascii=True, separators=(",", ":"))
            finally:
                request_timing["request_json_serialize_wall_clock_s"] = (
                    time.perf_counter() - stage_started
                )
            stage_started = time.perf_counter()
            try:
                if len(encoded.encode("utf-8")) > self.maximum_line_bytes:
                    raise ValueError("external planner request exceeds the configured line limit")
            finally:
                request_timing["request_size_check_wall_clock_s"] = (
                    time.perf_counter() - stage_started
                )
            assert self._process.stdin is not None
            stage_started = time.perf_counter()
            try:
                self._process.stdin.write(encoded + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError("external planner process rejected its request stream") from exc
            finally:
                request_timing["request_write_flush_wall_clock_s"] = (
                    time.perf_counter() - stage_started
                )
            effective_timeout_s = self.response_timeout_s if timeout_s is None else timeout_s
            if effective_timeout_s <= 0.0:
                raise ValueError("external planner request timeout must be positive")
            stage_started = time.perf_counter()
            try:
                line = self._responses.get(timeout=effective_timeout_s)
            except queue.Empty as exc:
                request_timing["response_wait_wall_clock_s"] = time.perf_counter() - stage_started
                self._terminate()
                raise TimeoutError(
                    f"external planner did not return a {kind} JSONL response before its deadline"
                ) from exc
            else:
                request_timing["response_wait_wall_clock_s"] = time.perf_counter() - stage_started
            if line is None:
                raise RuntimeError("external planner closed or violated the JSONL response stream")
            stage_started = time.perf_counter()
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("external planner returned malformed JSON") from exc
            finally:
                request_timing["response_json_decode_wall_clock_s"] = (
                    time.perf_counter() - stage_started
                )
            stage_started = time.perf_counter()
            try:
                allowed_response_fields = {"schema", "request_id", "status", "actions"}
                if not isinstance(response, dict) or set(response) - allowed_response_fields:
                    raise ValueError("external planner response has unsupported fields")
                if response.get("schema") != _EXTERNAL_RESPONSE_SCHEMA:
                    raise ValueError(
                        "external planner response schema differs from the canonical bridge"
                    )
                if response.get("request_id") != request_id:
                    raise ValueError("external planner response does not bind the active request")
                if response.get("status") != "ok":
                    raise ValueError("external planner did not acknowledge a successful request")
            finally:
                request_timing["response_validate_wall_clock_s"] = (
                    time.perf_counter() - stage_started
                )
            return response
        finally:
            covered = sum(request_timing.values())
            request_timing["request_wall_clock_s"] = time.perf_counter() - request_started
            request_timing["request_internal_unattributed_wall_clock_s"] = max(
                0.0, request_timing["request_wall_clock_s"] - covered
            )
            self._last_request_timing = request_timing

    def reset(
        self,
        public_episode: dict[str, Any],
        *,
        public_task_spec: dict[str, Any] | None = None,
    ) -> None:
        external_episode = _external_process_episode_projection(public_episode)
        payload: dict[str, Any] = {"public_episode": external_episode}
        if self.declaration.capability_profile == "G2-I":
            if (
                not isinstance(public_task_spec, dict)
                or public_task_spec.get("task_track") != "G2-I"
            ):
                raise ValueError("G2-I process reset requires a public G2-I task spec")
            validate_public_episode(public_episode, public_task_spec)
            payload["public_task_spec"] = _external_process_episode_projection(
                public_task_spec
            )
        elif public_task_spec is not None:
            raise ValueError("G1 process reset must not receive a G2-I task spec")
        initialization_started = time.perf_counter()
        response = self._request("reset", timeout_s=self.initialization_timeout_s, **payload)
        self.initialization_latency_s = time.perf_counter() - initialization_started
        if "actions" in response:
            raise ValueError("external planner reset response must not contain actions")
        self._reset = True

    def act(
        self, observations: dict[str, ObservationPacket]
    ) -> tuple[dict[str, ActionPacket], dict[str, float]]:
        if not self._reset:
            raise RuntimeError("external planner bridge must be reset before act")
        act_started = time.perf_counter()
        act_timing = {
            "projection_wall_clock_s": 0.0,
            "request_public_audit_wall_clock_s": 0.0,
            "request_json_serialize_wall_clock_s": 0.0,
            "request_size_check_wall_clock_s": 0.0,
            "request_write_flush_wall_clock_s": 0.0,
            "response_wait_wall_clock_s": 0.0,
            "response_json_decode_wall_clock_s": 0.0,
            "response_validate_wall_clock_s": 0.0,
            "action_validation_conversion_wall_clock_s": 0.0,
        }
        try:
            stage_started = time.perf_counter()
            try:
                projection = {
                    drone_id: project_g1(packet) for drone_id, packet in observations.items()
                }
            finally:
                act_timing["projection_wall_clock_s"] = time.perf_counter() - stage_started
            response = self._request("act", observations=projection)
            if self._last_request_timing is not None:
                for field in tuple(act_timing):
                    if field in self._last_request_timing:
                        act_timing[field] = self._last_request_timing[field]
            stage_started = time.perf_counter()
            try:
                actions_node = response.get("actions")
                if not isinstance(actions_node, dict) or set(actions_node) != set(observations):
                    raise ValueError(
                        "external planner actions must exactly match active observation IDs"
                    )
                actions = {
                    drone_id: _action_from_dict(actions_node[drone_id], packet, packet.sequence)
                    for drone_id, packet in observations.items()
                }
            finally:
                act_timing["action_validation_conversion_wall_clock_s"] = (
                    time.perf_counter() - stage_started
                )
        finally:
            act_timing["bridge_act_wall_clock_s"] = time.perf_counter() - act_started
            covered = sum(act_timing.values())
            act_timing["bridge_internal_unattributed_wall_clock_s"] = max(
                0.0, act_timing["bridge_act_wall_clock_s"] - covered
            )
            self._last_act_timing = act_timing
        elapsed = act_timing["bridge_act_wall_clock_s"]
        self.adapter_latencies_s.append(elapsed)
        return actions, {drone_id: elapsed for drone_id in actions}

    def last_act_timing(self) -> dict[str, float] | None:
        """Return scalar timing for the most recent action exchange.

        This deliberately excludes every request and response payload.  It is
        diagnostic evidence, not an additional input channel for a method.
        """

        return None if self._last_act_timing is None else dict(self._last_act_timing)

    def adapter_tax_report(self) -> dict[str, Any]:
        ordered = sorted(self.adapter_latencies_s)
        return {
            "schema": "org.aerocity.bench.adapter-tax.v1",
            "adapter_id": self.declaration.adapter_id,
            "call_count": len(ordered),
            "total_s": sum(ordered),
            "median_s": ordered[len(ordered) // 2] if ordered else None,
            "maximum_s": ordered[-1] if ordered else None,
        }

    def initialization_report(self) -> dict[str, Any]:
        """Return the one-time public setup measurement separately from actions."""

        return {
            "schema": "org.aerocity.bench.external-process-initialization.v1",
            "deadline_s": self.initialization_timeout_s,
            "completed": self._reset,
            "elapsed_s": self.initialization_latency_s,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._terminate()
        if self._process.stdin is not None:
            self._process.stdin.close()
        if self._process.stdout is not None:
            self._process.stdout.close()

    def __enter__(self) -> ExternalProcessPlannerBridge:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


class ReplayWriter:
    def __init__(self, path: Path, provenance: dict[str, Any]) -> None:
        self.path = path
        self.provenance = dict(provenance)
        self._records: list[dict[str, Any]] = []

    def append(
        self,
        observations: dict[str, ObservationPacket],
        actions: dict[str, ActionPacket],
        result: RuntimeStep,
    ) -> None:
        record = {
            "step": len(self._records),
            "observations": {key: value.to_dict() for key, value in sorted(observations.items())},
            "actions": {key: value.to_dict() for key, value in sorted(actions.items())},
            "confirmations": list(result.confirmations),
            "execution_receipts": [item.to_dict() for item in result.execution_receipts],
            "failures": [item.to_dict() for item in result.failures],
            "task_time_s": result.task_time_s,
            "done": result.done,
        }
        record["record_hash"] = content_hash(record)
        self._records.append(record)

    def close(self) -> dict[str, Any]:
        payload = {
            "schema": "org.aerocity.bench.replay.v1",
            "provenance": self.provenance,
            "records": self._records,
        }
        payload["replay_hash"] = content_hash(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return payload


def validate_replay(path: Path) -> dict[str, Any]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = str(replay.pop("replay_hash", ""))
    if content_hash(replay) != expected_hash:
        raise ValueError("replay content hash mismatch")
    previous_time = -1.0
    for index, record in enumerate(replay["records"]):
        expected_record_hash = str(record.pop("record_hash", ""))
        if content_hash(record) != expected_record_hash:
            raise ValueError(f"replay record hash mismatch at step {index}")
        if float(record["task_time_s"]) < previous_time:
            raise ValueError("replay task time runs backwards")
        previous_time = float(record["task_time_s"])
    return {
        "status": "PASS",
        "record_count": len(replay["records"]),
        "replay_hash": expected_hash,
    }


def _reject_unsupported_g2_i_rl_wrapper(runtime: L0FleetRuntime) -> None:
    task_spec = runtime.public_task_spec
    if isinstance(task_spec, dict) and task_spec.get("task_track") == "G2-I":
        raise ValueError(
            "the legacy RL wrapper exposes only the G1 observation projection and "
            "confirmation-only reward; use a versioned G2-I training wrapper before training"
        )


class GymnasiumFleetWrapper:
    """Dependency-light Gymnasium-style wrapper around a fleet runtime.

    The wrapper follows reset/step signatures without making Gymnasium a core
    dependency.  Projects needing registered spaces install ``aerocity-bench[rl]``.
    """

    metadata = {"render_modes": []}

    def __init__(self, runtime: L0FleetRuntime) -> None:
        _reject_unsupported_g2_i_rl_wrapper(runtime)
        self.runtime = runtime

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        observations = {key: project_g1(value) for key, value in self.runtime.reset().items()}
        return observations, {"execution_level": self.runtime.execution_level}

    def step(self, actions: dict[str, ActionPacket]):
        result = self.runtime.step(actions)
        observations = {key: project_g1(value) for key, value in result.observations.items()}
        rewards = {
            drone_id: float(
                sum(receipt["drone_id"] == drone_id for receipt in result.confirmations)
            )
            for drone_id in observations
        }
        terminated = {drone_id: result.done for drone_id in observations}
        truncated = {drone_id: False for drone_id in observations}
        return observations, rewards, terminated, truncated, {"task_time_s": result.task_time_s}


class PettingZooParallelWrapper:
    """PettingZoo ParallelEnv-compatible surface without a hard dependency."""

    metadata = {"name": "aerocity_ordinary_v1", "is_parallelizable": True}

    def __init__(self, runtime: L0FleetRuntime) -> None:
        _reject_unsupported_g2_i_rl_wrapper(runtime)
        self.runtime = runtime
        self.possible_agents = sorted(runtime.reset())
        self.agents = list(self.possible_agents)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        observations = {key: project_g1(value) for key, value in self.runtime.reset().items()}
        self.agents = sorted(observations)
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions: dict[str, ActionPacket]):
        acting_agents = list(self.agents)
        result = self.runtime.step(actions)
        observations = {key: project_g1(value) for key, value in result.observations.items()}
        rewards = {agent: 0.0 for agent in acting_agents}
        for confirmation in result.confirmations:
            rewards[str(confirmation["drone_id"])] += 1.0
        active_agents = set(observations)
        terminations = {agent: result.done or agent not in active_agents for agent in acting_agents}
        truncations = {agent: False for agent in acting_agents}
        infos = {agent: {"task_time_s": result.task_time_s} for agent in acting_agents}
        self.agents = [] if result.done else sorted(active_agents)
        return observations, rewards, terminations, truncations, infos
