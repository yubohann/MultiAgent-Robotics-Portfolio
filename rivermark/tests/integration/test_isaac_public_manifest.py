from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import copy
import json

import pytest

from rivermark_benchmark._identity import IdentityAccumulator
from rivermark_benchmark.pack.isaac_public_manifest import (
    PublicManifestError,
    build_public_scene_manifest,
    canonical_public_bytes,
    public_manifest_identity,
    validate_public_payload,
)


def _raw_scene() -> dict[str, object]:
    return {
        "schema": "org.rivermark.public-isaac-scene.v1",
        "environment_id": "RIVERMARK_CITY_LITE_v1",
        "agent_count": 8,
        "fresh_stage": True,
        "static_scene_authority_verified": True,
        "legacy_route_or_target_imported": False,
        "unresolved_reference_count": 0,
        "private_evaluator_manifest_identity": "e" * 16,
        "source_scene": r"C:\Users\operator\private\rivermark.usd",
        "target_diagnostics": [{"target_id": 3, "target_xyz": [1.0, 2.0, 3.0]}],
        "scene_contract": {
            "schema": "citylite-contract-v1",
            "gate_status": "pass_city_lite_static_construction",
            "payload_identity": "1" * 16,
            "identity": "2" * 16,
            "path": r"C:\private\scene_contract.json",
        },
        "rivermark_layer_inventory": {
            "schema": "resolved-layer-inventory-v1",
            "inventory_identity": "3" * 16,
            "local_authority_inventory_identity": "4" * 16,
            "rivermarksrc51_external_inventory_identity": "5" * 16,
            "local_authority_layer_count": 2,
            "rivermarksrc51_external_layer_count": 3,
            "input_resolved_layer_count": 5,
            "asset_root": r"C:\private\assets",
            "composition_scope": {
                "mode": "selective_references_only",
                "selective_references": [
                    {
                        "source_prim": "/World/City/Rivermark",
                        "destination_prim": "/World/StaticScene/City/Rivermark",
                    },
                    {
                        "source_prim": "/World/CityTaskObstacles",
                        "destination_prim": "/World/StaticScene/CityTaskObstacles",
                    },
                ],
                "whole_final_stage_inventory": False,
            },
        },
    }


def test_projection_excludes_private_truth_and_host_paths() -> None:
    public = build_public_scene_manifest(_raw_scene())
    serialized = canonical_public_bytes(public).decode("utf-8")

    assert public["schema"] == "org.rivermark.benchmark.public-citylite-scene.v1"
    assert public["agent_count"] == 8
    assert public["composition"]["reference_count"] == 2
    assert "private" not in serialized.lower()
    assert "target" not in serialized.lower()
    assert "C:\\" not in serialized
    validate_public_payload(public)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fresh_stage", False, "fresh stage"),
        ("static_scene_authority_verified", False, "authority"),
        ("legacy_route_or_target_imported", True, "legacy"),
        ("unresolved_reference_count", 1, "unresolved"),
    ],
)
def test_projection_fails_closed_on_untrusted_scene_state(
    field: str,
    value: object,
    message: str,
) -> None:
    scene = _raw_scene()
    scene[field] = value
    with pytest.raises(PublicManifestError, match=message):
        build_public_scene_manifest(scene)


def test_projection_requires_selective_input_inventory() -> None:
    scene = _raw_scene()
    inventory = scene["rivermark_layer_inventory"]
    assert isinstance(inventory, dict)
    composition = inventory["composition_scope"]
    assert isinstance(composition, dict)
    composition["whole_final_stage_inventory"] = True

    with pytest.raises(PublicManifestError, match="selective inputs"):
        build_public_scene_manifest(scene)


def test_projection_identity_is_stable_across_irrelevant_raw_diagnostics() -> None:
    first = _raw_scene()
    second = copy.deepcopy(first)
    second["private_evaluator_manifest_identity"] = "f" * 16
    second["source_scene"] = r"D:\another-host\rivermark.usd"
    second["target_diagnostics"] = [{"target_id": 99, "target_xyz": [9, 9, 9]}]

    first_public = build_public_scene_manifest(first)
    second_public = build_public_scene_manifest(second)
    assert first_public == second_public
    assert public_manifest_identity(first_public) == public_manifest_identity(second_public)
    assert public_manifest_identity(first_public) == IdentityAccumulator(
        canonical_public_bytes(first_public)
    ).hexdigest()


def test_public_payload_scanner_rejects_truth_keys_and_provenance_tokens() -> None:
    with pytest.raises(PublicManifestError, match="hidden_target_id"):
        validate_public_payload({"hidden_target_id": 1})
    with pytest.raises(PublicManifestError, match="ground_truth"):
        validate_public_payload({"description": "derived from ground_truth"})


def test_projection_is_json_round_trip_stable() -> None:
    public = build_public_scene_manifest(_raw_scene())
    decoded = json.loads(canonical_public_bytes(public))
    assert decoded == public
    assert public_manifest_identity(decoded) == public_manifest_identity(public)
