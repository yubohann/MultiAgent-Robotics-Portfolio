"""Fail-closed validation for method-visible benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import content_hash, read_json
from .inspection_atlas import validate_public_inspection_atlas, validate_public_mission_sector
from .ordinary_config import validate_public_execution_contract

_FORBIDDEN_KEY_FRAGMENTS = (
    "private",
    "target",
    "support",
    "witness",
    "evaluator",
    "split",
    "seed",
)
_FALSE_AUDIT_SENTINELS = frozenset(
    {"target_count_public", "target_process_public", "formal_split_label_public"}
)


def _normalized_key(key: object) -> str:
    if not isinstance(key, str) or not key.isascii():
        raise ValueError("public artifact contains a non-ASCII or non-string object key")
    return key.casefold().replace("-", "_")


def public_field_errors(value: object, *, path: str = "$") -> list[str]:
    """Return prohibited semantic key paths without serializing private values."""

    errors: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if normalized in _FALSE_AUDIT_SENTINELS:
                if nested is not False:
                    errors.append(f"{child_path} must be false")
                continue
            if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                errors.append(child_path)
            errors.extend(public_field_errors(nested, path=child_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            errors.extend(public_field_errors(nested, path=f"{path}[{index}]"))
    return errors


def assert_public_fields(value: object, *, path: str = "$") -> None:
    errors = public_field_errors(value, path=path)
    if errors:
        raise ValueError("public artifact contains forbidden fields: " + ", ".join(errors))


def validate_public_task_spec(task_spec: object) -> None:
    """Validate task integrity and information boundary before method exposure."""

    if not isinstance(task_spec, dict):
        raise ValueError("public task spec must be an object")
    expected_hash = str(task_spec.get("task_spec_hash", ""))
    payload = {key: value for key, value in task_spec.items() if key != "task_spec_hash"}
    if len(expected_hash) != 64 or expected_hash != content_hash(payload):
        raise ValueError("public task spec hash is invalid")
    assert_public_fields(payload, path="task_spec")
    contract = payload.get("execution_contract")
    if not isinstance(contract, dict):
        raise ValueError("public task spec lacks an execution contract")
    validate_public_execution_contract(contract)
    if payload.get("public_execution_contract_hash") != content_hash(contract):
        raise ValueError("public task spec execution-contract hash is invalid")
    atlas = payload.get("inspection_atlas")
    if atlas is not None:
        if not isinstance(atlas, dict):
            raise ValueError("public inspection atlas must be an object")
        validate_public_inspection_atlas(atlas)


def validate_public_episode(
    episode: object,
    task_spec: object,
) -> None:
    """Validate one method-visible episode before a public policy receives it.

    A public episode deliberately has no content hash: it is a projection of
    authority-held data, and its binding is verified against that projection by
    the evaluator/runtime.  This function validates only fields that are safe
    for a method to read.
    """

    validate_public_task_spec(task_spec)
    if not isinstance(episode, dict):
        raise ValueError("public episode must be an object")
    assert_public_fields(episode, path="public_episode")
    required = {
        "schema",
        "episode_id",
        "layout_id",
        "fleet_profile",
        "starts",
        "target_count_public",
        "target_process_public",
    }
    allowed = required | {"mission_sector", "mission_sector_hash", "coarse_region_ids"}
    if not required.issubset(episode) or set(episode) - allowed:
        raise ValueError("public episode fields differ")
    if episode["schema"] != "org.aerocity.bench.episode-public.ordinary.v1":
        raise ValueError("public episode schema differs")
    if not isinstance(episode["episode_id"], str) or not episode["episode_id"]:
        raise ValueError("public episode ID is invalid")
    if episode["layout_id"] != task_spec["layout_id"]:
        raise ValueError("public episode layout differs from task spec")
    if not isinstance(episode["fleet_profile"], dict) or not isinstance(episode["starts"], list):
        raise ValueError("public episode fleet roster is invalid")
    if not episode["starts"]:
        raise ValueError("public episode has no starts")
    if episode["target_count_public"] is not False or episode["target_process_public"] is not False:
        raise ValueError("public episode target visibility sentinels must be false")

    task_node = task_spec
    if task_node.get("task_track") != "G2-I":
        return
    atlas = task_node.get("inspection_atlas")
    sector = episode.get("mission_sector")
    sector_hash = episode.get("mission_sector_hash")
    if (sector is None) != (sector_hash is None):
        raise ValueError("public episode mission-sector binding is incomplete")
    if sector is not None:
        if not isinstance(atlas, dict):
            raise ValueError("public mission sector requires the full G2-I atlas")
        if not isinstance(sector, dict):
            raise ValueError("G2-I public episode mission sector is invalid")
        if sector_hash != sector.get("sector_hash"):
            raise ValueError("public episode mission-sector hash differs")
        contract = task_node["execution_contract"]
        validate_public_mission_sector(sector, atlas, episode["starts"], contract)


def audit_public_layout(layout_root: Path) -> dict[str, Any]:
    """Audit method-visible files of one materialized layout without truth access."""

    layout_root = layout_root.resolve()
    public_root = layout_root / "method_public"
    task_path = public_root / "task_spec.json"
    episode_root = public_root / "episodes"
    if not task_path.is_file() or not episode_root.is_dir():
        raise ValueError("layout lacks a public task spec or public episode directory")
    task_spec = read_json(task_path)
    validate_public_task_spec(task_spec)
    episodes = sorted(episode_root.glob("*.json"))
    if not episodes:
        raise ValueError("layout has no public episodes")
    task_track = str(task_spec.get("task_track", ""))
    for episode_path in episodes:
        episode = read_json(episode_path)
        try:
            validate_public_episode(episode, task_spec)
        except ValueError as exc:
            raise ValueError(f"invalid public episode: {episode_path}: {exc}") from exc
        if (
            task_track == "G2-I"
            and task_spec.get("inspection_prior_level") == "full-cells"
            and "mission_sector" not in episode
        ):
            raise ValueError(
                f"full-cell G2-I public episode lacks its mission sector: {episode_path}"
            )
    return {
        "schema": "org.aerocity.bench.public-boundary-audit.v1",
        "status": "PASS",
        "layout_id": str(task_spec["layout_id"]),
        "task_spec_sha256": content_hash(task_spec),
        "public_execution_contract_hash": str(task_spec["public_execution_contract_hash"]),
        "task_track": task_track,
        "public_episode_count": len(episodes),
    }
