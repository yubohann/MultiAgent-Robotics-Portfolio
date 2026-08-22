"""Blind-evaluator submission policy, mount plan, and side-channel audit."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import content_hash, file_hash, read_json, write_json
from .errors import ValidationError

IMAGE_DIGEST = re.compile(r"^[a-z0-9./_-]+@sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class SubmissionPolicy:
    maximum_submissions_per_team: int = 10
    cpus: int = 8
    memory_gib: int = 24
    gpu_count: int = 1
    wall_time_s: int = 3600
    writable_tmp_gib: int = 8
    network: str = "none"
    read_only_root: bool = True
    drop_linux_capabilities: bool = True
    no_new_privileges: bool = True

    def __post_init__(self) -> None:
        if (
            min(
                self.maximum_submissions_per_team,
                self.cpus,
                self.memory_gib,
                self.wall_time_s,
                self.writable_tmp_gib,
            )
            < 1
        ):
            raise ValueError("blind evaluator resource limits must be positive")
        if self.network != "none":
            raise ValueError("formal blind submissions cannot access a network")
        if (
            not self.read_only_root
            or not self.drop_linux_capabilities
            or not self.no_new_privileges
        ):
            raise ValueError("blind evaluator process hardening cannot be disabled")


def submission_spec(
    *,
    team_id: str,
    submission_id: str,
    image: str,
    adapter_declaration_path: Path,
    policy: SubmissionPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or SubmissionPolicy()
    if not team_id or not submission_id:
        raise ValueError("team and submission IDs cannot be empty")
    if not IMAGE_DIGEST.fullmatch(image):
        raise ValueError("submission images must be immutable sha256 digests")
    adapter = read_json(adapter_declaration_path)
    spec = {
        "schema": "org.aerocity.bench.blind-submission.v1",
        "team_id": team_id,
        "submission_id": submission_id,
        "image": image,
        "adapter_declaration": adapter,
        "adapter_declaration_sha256": file_hash(adapter_declaration_path),
        "resources": {
            "cpus": policy.cpus,
            "memory_gib": policy.memory_gib,
            "gpu_count": policy.gpu_count,
            "wall_time_s": policy.wall_time_s,
            "writable_tmp_gib": policy.writable_tmp_gib,
        },
        "sandbox": {
            "network": policy.network,
            "read_only_root": policy.read_only_root,
            "drop_linux_capabilities": policy.drop_linux_capabilities,
            "no_new_privileges": policy.no_new_privileges,
            "mounts": [
                {"source_role": "method_public", "target": "/benchmark/input", "mode": "ro"},
                {"source_role": "ipc", "target": "/benchmark/ipc", "mode": "rw"},
                {"source_role": "scratch", "target": "/tmp", "mode": "rw"},
            ],
            "forbidden_mount_roles": [
                "evaluator_private",
                "authority_private",
                "scene_authority",
                "docker_socket",
                "host_home",
            ],
        },
        "side_channel_controls": {
            "hide_split_names": True,
            "uniform_file_metadata": True,
            "fixed_response_shapes": True,
            "deterministic_error_classes": True,
            "gpu_process_isolation": True,
            "cache_reset_between_submissions": True,
            "timing_padding_profile": "versioned-bounded-v1",
        },
    }
    spec["submission_spec_hash"] = content_hash(spec)
    return spec


def write_submission_spec(path: Path, **kwargs: Any) -> dict[str, Any]:
    spec = submission_spec(**kwargs)
    write_json(path, spec)
    return spec


def validate_submission_spec(path: Path) -> dict[str, Any]:
    spec = read_json(path)
    expected_hash = str(spec.pop("submission_spec_hash", ""))
    if content_hash(spec) != expected_hash:
        raise ValidationError("blind submission spec hash mismatch")
    sandbox = spec["sandbox"]
    mounted_roles = {item["source_role"] for item in sandbox["mounts"]}
    forbidden = set(sandbox["forbidden_mount_roles"])
    if mounted_roles & forbidden:
        raise ValidationError("blind submission mounts an evaluator-private role")
    if sandbox.get("network") != "none" or sandbox.get("read_only_root") is not True:
        raise ValidationError("blind submission sandbox is not fail closed")
    controls = spec["side_channel_controls"]
    if not all(value is True for key, value in controls.items() if key != "timing_padding_profile"):
        raise ValidationError("blind submission disables a side-channel control")
    return {
        "status": "PASS",
        "team_id": spec["team_id"],
        "submission_id": spec["submission_id"],
        "submission_spec_hash": expected_hash,
    }
