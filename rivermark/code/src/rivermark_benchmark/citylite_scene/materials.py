"""CityTaskObstacles local-material repair contract and receipts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .constants import (
    CITY_TASK_OBSTACLE_MATERIAL_CLOSURE_SCHEMA,
    CITY_TASK_OBSTACLE_MATERIAL_ROOT,
    CITY_TASK_OBSTACLE_MATERIAL_SPECS,
)
from .scene import CityLiteAuthorityError, canonical_payload_sha256


def city_task_obstacle_material_contract_payload() -> dict[str, Any]:
    """Return the exact local repair contract for scoped obstacle materials."""

    bindings: list[dict[str, Any]] = []
    for spec in CITY_TASK_OBSTACLE_MATERIAL_SPECS:
        obstacle_name = str(spec["obstacle_name"])
        material_name = str(spec["material_name"])
        local_material_prim = f"{CITY_TASK_OBSTACLE_MATERIAL_ROOT}/{material_name}"
        bindings.append(
            {
                "obstacle_prim": f"/World/StaticScene/CityTaskObstacles/{obstacle_name}",
                "source_material_prim": f"/World/Materials/{material_name}",
                "local_material_prim": local_material_prim,
                "shader_prim": f"{local_material_prim}/Shader",
                "surface_output_connection": f"{local_material_prim}/Shader.outputs:surface",
                "shader_id": "UsdPreviewSurface",
                "diffuse_color": [float(value) for value in spec["diffuse_color"]],
                "opacity": float(spec["opacity"]),
                "roughness": float(spec["roughness"]),
            }
        )
    return {
        "schema": CITY_TASK_OBSTACLE_MATERIAL_CLOSURE_SCHEMA,
        "repair_strategy": "local_usd_preview_surface_rebinding",
        "material_root": CITY_TASK_OBSTACLE_MATERIAL_ROOT,
        "binding_count": len(bindings),
        "bindings": bindings,
    }

CITY_TASK_OBSTACLE_MATERIAL_CONTRACT_SHA256 = canonical_payload_sha256(
    city_task_obstacle_material_contract_payload()
)

def city_task_obstacle_material_closure_receipt_template() -> dict[str, Any]:
    """Create an empty-diagnostic receipt for the exact local material repair."""

    contract = city_task_obstacle_material_contract_payload()
    return {
        **contract,
        "contract_sha256": CITY_TASK_OBSTACLE_MATERIAL_CONTRACT_SHA256,
        "post_repair_binding_closure": True,
        "observed_bindings": [
            {
                "obstacle_prim": binding["obstacle_prim"],
                "local_material_prim": binding["local_material_prim"],
                "surface_output_connection": binding["surface_output_connection"],
                "resolved": True,
            }
            for binding in contract["bindings"]
        ],
        "source_scope_diagnostics": {
            "known_external_material_binding_count": contract["binding_count"],
            "repair_applied_before_stage_load": True,
            "reported_warning_count": 0,
            "reported_warnings": [],
        },
    }

def validate_city_task_obstacle_material_closure_receipt(
    receipt: Mapping[str, Any],
) -> None:
    """Fail closed unless every scoped obstacle has its verified local material."""

    if not isinstance(receipt, Mapping):
        raise CityLiteAuthorityError(
            "CityTaskObstacles material closure receipt must be an object"
        )
    expected_contract = city_task_obstacle_material_contract_payload()
    for key, expected in expected_contract.items():
        if receipt.get(key) != expected:
            raise CityLiteAuthorityError(
                f"CityTaskObstacles material closure field is invalid: {key}"
            )
    if receipt.get("contract_sha256") != CITY_TASK_OBSTACLE_MATERIAL_CONTRACT_SHA256:
        raise CityLiteAuthorityError(
            "CityTaskObstacles material closure contract digest is invalid"
        )
    if receipt.get("post_repair_binding_closure") is not True:
        raise CityLiteAuthorityError(
            "CityTaskObstacles local material binding closure was not verified"
        )

    expected_observed = [
        {
            "obstacle_prim": binding["obstacle_prim"],
            "local_material_prim": binding["local_material_prim"],
            "surface_output_connection": binding["surface_output_connection"],
            "resolved": True,
        }
        for binding in expected_contract["bindings"]
    ]
    if receipt.get("observed_bindings") != expected_observed:
        raise CityLiteAuthorityError(
            "CityTaskObstacles observed material bindings are incomplete or stale"
        )

    diagnostics = receipt.get("source_scope_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise CityLiteAuthorityError(
            "CityTaskObstacles source scope diagnostics are missing"
        )
    if diagnostics.get("known_external_material_binding_count") != expected_contract[
        "binding_count"
    ]:
        raise CityLiteAuthorityError(
            "CityTaskObstacles source scope diagnostic count is stale"
        )
    if diagnostics.get("repair_applied_before_stage_load") is not True:
        raise CityLiteAuthorityError(
            "CityTaskObstacles material repair was not authored before stage load"
        )
    warnings = diagnostics.get("reported_warnings")
    count = diagnostics.get("reported_warning_count")
    if (
        not isinstance(warnings, list)
        or not all(isinstance(message, str) for message in warnings)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(warnings)
    ):
        raise CityLiteAuthorityError(
            "CityTaskObstacles source scope diagnostics are malformed"
        )
    for message in warnings:
        lowered = message.lower()
        if "outside the scope of the reference" not in lowered or "/world/materials/" not in lowered:
            raise CityLiteAuthorityError(
                "CityTaskObstacles source scope diagnostic is not an expected repaired binding warning"
            )
