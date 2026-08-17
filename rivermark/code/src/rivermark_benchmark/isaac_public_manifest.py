"""Public, deterministic projections of native Isaac control-plane metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .schema import forbidden_policy_key, forbidden_policy_value_token, iter_tree


class PublicManifestError(ValueError):
    """Raised when raw metadata cannot support a non-private public projection."""


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicManifestError(f"{label} must be an object")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicManifestError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PublicManifestError(f"{label} must be SHA-256")
    return text


def validate_public_payload(value: Any) -> None:
    """Reject evaluator truth, target truth, private paths, and related tokens."""

    for tree_path, key, child in iter_tree(value):
        if key is not None and forbidden_policy_key(key):
            raise PublicManifestError(
                f"public key {key!r} is forbidden at {tree_path}"
            )
        if isinstance(child, str):
            token = forbidden_policy_value_token(child)
            if token is not None:
                raise PublicManifestError(
                    f"public string references forbidden token {token!r} at {tree_path}"
                )


def build_public_scene_manifest(scene: Mapping[str, Any]) -> dict[str, Any]:
    """Project raw scene diagnostics to a path-free public layout manifest."""

    if scene.get("fresh_stage") is not True:
        raise PublicManifestError("scene is not a fresh stage")
    if scene.get("static_scene_authority_verified") is not True:
        raise PublicManifestError("static scene authority was not verified")
    if scene.get("legacy_route_or_target_imported") is not False:
        raise PublicManifestError("legacy route or target data was imported")
    unresolved = scene.get("unresolved_reference_count")
    if not isinstance(unresolved, int) or isinstance(unresolved, bool) or unresolved != 0:
        raise PublicManifestError("scene has unresolved references")
    agent_count = scene.get("agent_count")
    if (
        not isinstance(agent_count, int)
        or isinstance(agent_count, bool)
        or not 1 <= agent_count <= 32
    ):
        raise PublicManifestError("scene agent_count is invalid")

    contract = _mapping(scene.get("scene_contract"), label="scene_contract")
    gate_status = _text(contract.get("gate_status"), label="scene_contract.gate_status")
    if not gate_status.startswith("pass_"):
        raise PublicManifestError("scene contract construction gate did not pass")
    inventory = _mapping(
        scene.get("rivermark_layer_inventory"), label="rivermark_layer_inventory"
    )
    composition = _mapping(
        inventory.get("composition_scope"),
        label="rivermark_layer_inventory.composition_scope",
    )
    if composition.get("mode") != "selective_references_only":
        raise PublicManifestError("scene composition is not selective-references-only")
    references = composition.get("selective_references")
    if not isinstance(references, list) or not references:
        raise PublicManifestError("scene composition has no selective references")
    for index, reference in enumerate(references):
        item = _mapping(
            reference,
            label=f"rivermark_layer_inventory.composition_scope.selective_references[{index}]",
        )
        _text(item.get("source_prim"), label=f"selective_references[{index}].source_prim")
        _text(
            item.get("destination_prim"),
            label=f"selective_references[{index}].destination_prim",
        )
    if composition.get("whole_final_stage_inventory") is not False:
        raise PublicManifestError(
            "scene inventory must describe selective inputs, not the final stage"
        )

    local_count = inventory.get("local_authority_layer_count")
    external_count = inventory.get("rivermarksrc51_external_layer_count")
    input_count = inventory.get("input_resolved_layer_count")
    counts = (local_count, external_count, input_count)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in counts
    ):
        raise PublicManifestError("scene layer counts are invalid")

    payload = {
        "schema": "org.rivermark.benchmark.public-citylite-scene.v1",
        "source_scene_schema": _text(scene.get("schema"), label="scene.schema"),
        "environment_id": _text(scene.get("environment_id"), label="environment_id"),
        "agent_count": agent_count,
        "construction": {
            "fresh_stage": True,
            "static_scene_authority_verified": True,
            "legacy_imported": False,
            "unresolved_reference_count": 0,
        },
        "authority": {
            "contract_schema": _text(
                contract.get("schema"), label="scene_contract.schema"
            ),
            "contract_gate_status": gate_status,
            "contract_payload_sha256": _sha256(
                contract.get("payload_sha256"),
                label="scene_contract.payload_sha256",
            ),
            "contract_file_sha256": _sha256(
                contract.get("sha256"), label="scene_contract.sha256"
            ),
            "resolved_inventory_schema": _text(
                inventory.get("schema"), label="rivermark_layer_inventory.schema"
            ),
            "resolved_inventory_sha256": _sha256(
                inventory.get("inventory_sha256"),
                label="rivermark_layer_inventory.inventory_sha256",
            ),
            "local_authority_inventory_sha256": _sha256(
                inventory.get("local_authority_inventory_sha256"),
                label="rivermark_layer_inventory.local_authority_inventory_sha256",
            ),
            "external_asset_inventory_sha256": _sha256(
                inventory.get("rivermarksrc51_external_inventory_sha256"),
                label="rivermark_layer_inventory.rivermarksrc51_external_inventory_sha256",
            ),
            "local_authority_layer_count": local_count,
            "external_asset_layer_count": external_count,
            "input_resolved_layer_count": input_count,
        },
        "composition": {
            "mode": "selective_references_only",
            "reference_count": len(references),
            "whole_final_stage_inventory": False,
        },
    }
    validate_public_payload(payload)
    return payload


def canonical_public_bytes(value: Any) -> bytes:
    validate_public_payload(value)
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def public_manifest_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_public_bytes(value)).hexdigest()


__all__ = [
    "PublicManifestError",
    "build_public_scene_manifest",
    "canonical_public_bytes",
    "public_manifest_sha256",
    "validate_public_payload",
]
