from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from rivermark_benchmark.citylite_scene import (
    AGENT_COUNT,
    AUTHORITY_SHA256,
    CITY_LITE_COMMAND_VOLUME_W_M,
    CITY_LITE_FLIGHT_VOLUME_W_M,
    CITY_TASK_OBSTACLE_MATERIAL_CONTRACT_SHA256,
    CITY_TASK_OBSTACLE_MATERIAL_CLOSURE_SCHEMA,
    EXPECTED_NATIVE_COLLISION_COUNTS,
    EXPECTED_UPSTREAM_PERMISSIONS,
    FORMAL_SCORING_VOLUME_W_M,
    ROUTE_CLEARANCE_M,
    SCENE_CONTRACT_GATE_STATUS,
    SCENE_CONTRACT_PAYLOAD_SHA256,
    SCENE_CONTRACT_SCHEMA,
    SCENE_CONTRACT_SHA256,
    SELECTIVE_REFERENCES,
    TARGET_FREE_SAFE_STARTS_W_M,
    AABB,
    CityLiteAuthority,
    CityLiteAuthorityError,
    CityLiteRouteError,
    aabb_geometry_sha256,
    canonical_payload_sha256,
    city_task_obstacle_material_closure_receipt_template,
    city_task_obstacle_material_contract_payload,
    flight_contract_payload,
    forbidden_scene_paths,
    make_rivermark_layer_inventory,
    make_public_route_contract,
    resolve_city_lite_authority,
    segment_has_clearance,
    segment_intersects_aabb,
    sha256_file,
    validate_public_route_contract,
    validate_public_routes,
    validate_city_task_obstacle_material_closure_receipt,
    validate_rivermark_layer_inventory_receipt,
    validate_static_scene_receipt,
    validate_upstream_scene_contract,
)
from rivermark_benchmark.isaac_runtime_safety import CF2X_RUNTIME_GUARD_RADIUS_M


def _upstream_contract() -> dict:
    filenames = {
        "city_lite_base_usd": "rivermark_city_lite_base_v1.usda",
        "filtered_structural_props_usd": "rivermark_city_lite_structural_props_v1.usda",
        "final_combined_usd": "hi_fi_search_rescue_rivermark_city_lite_v1.usda",
    }
    payload = {
        "schema": SCENE_CONTRACT_SCHEMA,
        "scene_id": "RIVERMARK_CITY_LITE_v1",
        "gate_status": SCENE_CONTRACT_GATE_STATUS,
        "permissions": dict(EXPECTED_UPSTREAM_PERMISSIONS),
        "isaac_started": False,
        "simulation_app_started": False,
        "checks": {"static_construction": True, "zero_unresolved": True},
        "outputs": {
            key: {
                "path": f"C:\\authority\\{filename}",
                "sha256": AUTHORITY_SHA256[filename],
                "size_bytes": index + 1,
            }
            for index, (key, filename) in enumerate(filenames.items())
        },
        "created_utc": "2026-07-14T18:48:15+00:00",
    }
    payload["payload_sha256"] = canonical_payload_sha256(payload)
    return payload


def _receipt() -> dict:
    return {
        "environment_id": "RIVERMARK_CITY_LITE_v1",
        "static_scene_authority_verified": True,
        "unresolved_reference_count": 0,
        "legacy_prim_count": 0,
        "forbidden_decoration_prim_count": 0,
        "city_task_obstacle_material_closure": city_task_obstacle_material_closure_receipt_template(),
        "scene_contract": {
            "path": "/authority/rivermark_city_lite_scene_contract_v1.json",
            "sha256": SCENE_CONTRACT_SHA256,
            "payload_sha256": SCENE_CONTRACT_PAYLOAD_SHA256,
            "schema": SCENE_CONTRACT_SCHEMA,
            "gate_status": SCENE_CONTRACT_GATE_STATUS,
            "permissions": dict(EXPECTED_UPSTREAM_PERMISSIONS),
        },
        "authority_assets": {
            name: {"path": f"/authority/{name}", "sha256": digest}
            for name, digest in AUTHORITY_SHA256.items()
        },
        "selective_references": [
            {"source_prim": source, "destination_prim": destination}
            for source, destination in SELECTIVE_REFERENCES
        ],
        "stage_units": {
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "time_codes_per_second": 60.0,
            "frames_per_second": 60.0,
        },
        "flight_contract": flight_contract_payload(),
        "native_collision_counts": dict(EXPECTED_NATIVE_COLLISION_COUNTS),
        "scene_runtime_admission": False,
        "formal_collection": False,
        "formal_benchmark_admission": False,
    }


def _safe_routes() -> list[list[list[float]]]:
    routes: list[list[list[float]]] = []
    for x, y, z in TARGET_FREE_SAFE_STARTS_W_M:
        dx = -1.0 if x >= 40.0 else 1.0
        dy = -1.0 if y >= 40.0 else 1.0
        routes.append(
            [
                [x, y, z],
                [x + dx, y, z],
                [x + dx, y + dy, z + 0.25],
            ]
        )
    return routes


def _remote_box() -> AABB:
    return AABB(
        (20.0, 20.0, 9.0),
        (21.0, 21.0, 10.0),
        source_prim="/World/CollisionProxies/remote",
        category="structural_proxy",
    )


def _temporary_authority(tmp_path: Path) -> tuple[CityLiteAuthority, list[Path]]:
    root = tmp_path / "authority"
    root.mkdir()
    assets: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for index, filename in enumerate(sorted(AUTHORITY_SHA256), start=1):
        path = root / filename
        path.write_text(f"#usda 1.0\n# local authority {index}\n", encoding="ascii")
        assets[filename] = path
        hashes[filename] = sha256_file(path)
    authority = CityLiteAuthority(
        root=root,
        contract_path=root / "contract.json",
        final_scene_path=assets["hi_fi_search_rescue_rivermark_city_lite_v1.usda"],
        asset_paths=assets,
        sha256=hashes,
        contract_sha256="0" * 64,
        contract_payload_sha256="1" * 64,
    )
    return authority, [assets[name] for name in sorted(assets)]


def _temporary_rivermark_layers(tmp_path: Path) -> tuple[Path, list[Path]]:
    root = tmp_path / "RivermarkSrc51"
    paths = [
        root / "dsready_content" / "scene" / "sub_terrain.usd",
        root / "dsready_content" / "props" / "building.usda",
    ]
    for index, path in enumerate(paths, start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#usda 1.0\n# external layer {index}\n", encoding="ascii")
    return root, paths


def test_forbidden_scene_paths_rejects_only_exact_legacy_and_removed_roots() -> None:
    paths = [
        "/World/StaticScene/City/Rivermark/roads",
        "/World/StaticScene/Mission/Targets/t0",
        "/World/StaticScene/Drones/cf2x_0",
        "/World/StaticScene/City/Rivermark/foliage/tree_0",
        "/World/StaticScene/City/Rivermark/grass",
        "/World/StaticScene/City/Rivermark/sub_traffic_signs/sign_0",
        "/World/StaticScene/City/Rivermark/vegetation/tree_0",
        "/World/StaticScene/City/Rivermark/props/business_sign",
        "/World/StaticScene/City/Rivermark/materials/grass_green",
    ]
    assert forbidden_scene_paths(paths) == [
        "/World/StaticScene/City/Rivermark/foliage/tree_0",
        "/World/StaticScene/City/Rivermark/grass",
        "/World/StaticScene/City/Rivermark/sub_traffic_signs/sign_0",
        "/World/StaticScene/Drones/cf2x_0",
        "/World/StaticScene/Mission/Targets/t0",
    ]


def test_validate_upstream_scene_contract_accepts_self_bound_static_contract() -> None:
    contract = _upstream_contract()
    assert validate_upstream_scene_contract(contract) == contract["payload_sha256"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong", "schema"),
        ("scene_id", "wrong", "scene_id"),
        ("gate_status", "pending", "gate"),
        ("isaac_started", True, "isaac_started"),
        ("simulation_app_started", True, "simulation_app_started"),
    ],
)
def test_validate_upstream_scene_contract_rejects_invalid_authority_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    contract = _upstream_contract()
    contract[field] = value
    contract["payload_sha256"] = canonical_payload_sha256(
        {key: item for key, item in contract.items() if key != "payload_sha256"}
    )
    with pytest.raises(CityLiteAuthorityError, match=message):
        validate_upstream_scene_contract(contract)


def test_validate_upstream_scene_contract_rejects_positive_or_missing_permission() -> None:
    for mutation in ("positive", "missing"):
        contract = _upstream_contract()
        if mutation == "positive":
            contract["permissions"]["formal_collection"] = True
        else:
            del contract["permissions"]["formal_collection"]
        contract["payload_sha256"] = canonical_payload_sha256(
            {key: item for key, item in contract.items() if key != "payload_sha256"}
        )
        with pytest.raises(CityLiteAuthorityError, match="permissions"):
            validate_upstream_scene_contract(contract)


def test_validate_upstream_scene_contract_rejects_payload_tampering() -> None:
    contract = _upstream_contract()
    contract["created_utc"] = "tampered"
    with pytest.raises(CityLiteAuthorityError, match="payload hash mismatch"):
        validate_upstream_scene_contract(contract)
    with pytest.raises(CityLiteAuthorityError, match="unexpected.*payload"):
        validate_upstream_scene_contract(
            _upstream_contract(),
            expected_payload_sha256="0" * 64,
        )


def test_validate_upstream_scene_contract_rejects_failed_check_and_output_binding() -> None:
    contract = _upstream_contract()
    contract["checks"]["zero_unresolved"] = False
    contract["payload_sha256"] = canonical_payload_sha256(
        {key: item for key, item in contract.items() if key != "payload_sha256"}
    )
    with pytest.raises(CityLiteAuthorityError, match="checks"):
        validate_upstream_scene_contract(contract)

    contract = _upstream_contract()
    contract["outputs"]["final_combined_usd"]["sha256"] = "0" * 64
    contract["payload_sha256"] = canonical_payload_sha256(
        {key: item for key, item in contract.items() if key != "payload_sha256"}
    )
    with pytest.raises(CityLiteAuthorityError, match="output digest"):
        validate_upstream_scene_contract(contract)


def test_city_lite_volumes_and_target_free_starts_are_frozen() -> None:
    assert FORMAL_SCORING_VOLUME_W_M == AABB(
        (-46.0, -48.0, 0.0),
        (46.0, 44.0, 19.0),
    )
    assert CITY_LITE_FLIGHT_VOLUME_W_M == AABB(
        (-46.0, -48.0, 8.9),
        (46.0, 44.0, 15.0),
    )
    assert CITY_LITE_COMMAND_VOLUME_W_M == AABB(
        (-46.0, -48.0, 9.0),
        (46.0, 44.0, 14.25),
    )
    assert len(TARGET_FREE_SAFE_STARTS_W_M) == AGENT_COUNT
    assert len(set(TARGET_FREE_SAFE_STARTS_W_M)) == AGENT_COUNT
    assert all(
        CITY_LITE_COMMAND_VOLUME_W_M.contains(point)
        for point in TARGET_FREE_SAFE_STARTS_W_M
    )
    assert all(
        CITY_LITE_FLIGHT_VOLUME_W_M.contains(
            point, margin_m=CF2X_RUNTIME_GUARD_RADIUS_M
        )
        for point in TARGET_FREE_SAFE_STARTS_W_M
    )
    assert TARGET_FREE_SAFE_STARTS_W_M[1] == (-4.0, -32.0, 9.847)
    assert (-10.0, -42.0, 9.25) not in TARGET_FREE_SAFE_STARTS_W_M
    assert (-42.0, -46.0, 9.048005771636962) not in TARGET_FREE_SAFE_STARTS_W_M


def test_segment_aabb_clearance_is_conservative_and_treats_touch_as_collision() -> None:
    box = AABB((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert segment_intersects_aabb((-1.0, 0.5, 0.5), (2.0, 0.5, 0.5), box)
    assert segment_intersects_aabb((-1.0, 1.0, 0.5), (2.0, 1.0, 0.5), box)
    assert not segment_intersects_aabb(
        (-1.0, 1.5001, 0.5),
        (2.0, 1.5001, 0.5),
        box,
        clearance_m=0.5,
    )
    assert segment_intersects_aabb(
        (-1.0, 1.5, 0.5),
        (2.0, 1.5, 0.5),
        box,
        clearance_m=0.5,
    )
    assert segment_has_clearance(
        (-1.0, 1.5001, 0.5),
        (2.0, 1.5001, 0.5),
        [box],
        clearance_m=0.5,
    )


def test_aabb_hash_is_order_independent_and_binds_source_semantics() -> None:
    first = AABB((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), source_prim="/a")
    second = AABB((2.0, 2.0, 2.0), (3.0, 3.0, 3.0), source_prim="/b")
    assert aabb_geometry_sha256([first, second]) == aabb_geometry_sha256(
        [second, first]
    )
    assert aabb_geometry_sha256([first]) != aabb_geometry_sha256(
        [AABB(first.minimum, first.maximum, source_prim="/other")]
    )


def test_public_route_contract_accepts_target_free_clear_routes() -> None:
    routes = _safe_routes()
    boxes = [_remote_box()]
    contract = make_public_route_contract(boxes)
    report = validate_public_route_contract(contract, routes, boxes)
    assert report.agent_count == AGENT_COUNT
    assert report.waypoint_count_per_agent == 3
    assert report.segment_count == 16
    assert report.clearance_m == ROUTE_CLEARANCE_M
    assert report.aabb_geometry_sha256 == contract["aabb_geometry_sha256"]
    assert validate_public_routes(routes, boxes) == report


def test_public_route_contract_rejects_legacy_or_private_conditioning() -> None:
    routes = _safe_routes()
    boxes = [_remote_box()]
    contract = make_public_route_contract(boxes)
    contract["targets"] = []
    with pytest.raises(CityLiteRouteError, match="target/evaluator/legacy"):
        validate_public_route_contract(contract, routes, boxes)

    contract = make_public_route_contract(boxes)
    contract["target_or_evaluator_consumed"] = True
    with pytest.raises(CityLiteRouteError, match="target_or_evaluator_consumed"):
        validate_public_route_contract(contract, routes, boxes)


def test_public_route_contract_rejects_wrong_start_volume_and_aabb_hash() -> None:
    boxes = [_remote_box()]
    contract = make_public_route_contract(boxes)

    routes = _safe_routes()
    routes[0][0][0] += 0.1
    with pytest.raises(CityLiteRouteError, match="safe anchor"):
        validate_public_route_contract(contract, routes, boxes)

    routes = _safe_routes()
    routes[0][1][2] = 20.0
    with pytest.raises(CityLiteRouteError, match="command volume"):
        validate_public_route_contract(contract, routes, boxes)

    stale = deepcopy(contract)
    stale["aabb_geometry_sha256"] = "0" * 64
    with pytest.raises(CityLiteRouteError, match="geometry hash"):
        validate_public_route_contract(stale, _safe_routes(), boxes)


def test_public_route_contract_rejects_segment_clearance_violation() -> None:
    routes = _safe_routes()
    blocking = AABB(
        (-39.6, -12.1, 9.1),
        (-39.4, -11.9, 9.4),
        source_prim="/World/CollisionProxies/blocking",
        category="structural_proxy",
    )
    contract = make_public_route_contract([blocking])
    with pytest.raises(CityLiteRouteError, match="clearance"):
        validate_public_route_contract(contract, routes, [blocking])


def test_static_scene_receipt_accepts_exact_authority() -> None:
    validate_static_scene_receipt(_receipt())


def test_city_task_obstacle_material_contract_is_exact_and_closed() -> None:
    contract = city_task_obstacle_material_contract_payload()
    receipt = city_task_obstacle_material_closure_receipt_template()

    assert contract["schema"] == CITY_TASK_OBSTACLE_MATERIAL_CLOSURE_SCHEMA
    assert contract["binding_count"] == 8
    assert receipt["contract_sha256"] == CITY_TASK_OBSTACLE_MATERIAL_CONTRACT_SHA256
    assert receipt["post_repair_binding_closure"] is True
    assert [row["obstacle_prim"] for row in contract["bindings"]] == [
        "/World/StaticScene/CityTaskObstacles/south_collapsed_facade",
        "/World/StaticScene/CityTaskObstacles/midblock_service_wall",
        "/World/StaticScene/CityTaskObstacles/west_tower_rubble_screen",
        "/World/StaticScene/CityTaskObstacles/north_skybridge_debris",
        "/World/StaticScene/CityTaskObstacles/warning_light_00",
        "/World/StaticScene/CityTaskObstacles/warning_light_01",
        "/World/StaticScene/CityTaskObstacles/warning_light_02",
        "/World/StaticScene/CityTaskObstacles/warning_light_03",
    ]
    assert all(
        row["local_material_prim"].startswith(
            "/World/StaticScene/CityTaskObstacleMaterials/"
        )
        for row in contract["bindings"]
    )
    validate_city_task_obstacle_material_closure_receipt(receipt)


def test_city_task_obstacle_material_receipt_rejects_open_or_stale_binding() -> None:
    receipt = city_task_obstacle_material_closure_receipt_template()
    receipt["observed_bindings"][0]["resolved"] = False
    with pytest.raises(CityLiteAuthorityError, match="observed material bindings"):
        validate_city_task_obstacle_material_closure_receipt(receipt)

    static_receipt = _receipt()
    static_receipt["city_task_obstacle_material_closure"]["post_repair_binding_closure"] = False
    with pytest.raises(CityLiteAuthorityError, match="material closure receipt"):
        validate_static_scene_receipt(static_receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("environment_id", "rivermark-placeholder"),
        ("unresolved_reference_count", 1),
        ("legacy_prim_count", 1),
        ("forbidden_decoration_prim_count", 1),
        ("scene_runtime_admission", True),
        ("formal_collection", True),
        ("formal_benchmark_admission", True),
    ],
)
def test_static_scene_receipt_fails_closed(field: str, value: object) -> None:
    receipt = _receipt()
    receipt[field] = value
    with pytest.raises(CityLiteAuthorityError):
        validate_static_scene_receipt(receipt)


def test_static_scene_receipt_rejects_stale_contract_or_flight_contract() -> None:
    receipt = _receipt()
    receipt["scene_contract"]["payload_sha256"] = "0" * 64
    with pytest.raises(CityLiteAuthorityError, match="scene_contract"):
        validate_static_scene_receipt(receipt)

    receipt = _receipt()
    receipt["flight_contract"]["route_clearance_m"] = 0.0
    with pytest.raises(CityLiteAuthorityError, match="flight contract"):
        validate_static_scene_receipt(receipt)


def test_resolve_authority_rejects_missing_contract(tmp_path: Path) -> None:
    with pytest.raises(CityLiteAuthorityError, match="contract not found"):
        resolve_city_lite_authority(tmp_path)


def test_rivermark_layer_inventory_binds_selective_external_layers(
    tmp_path: Path,
) -> None:
    authority, local_layers = _temporary_authority(tmp_path)
    asset_root, external_layers = _temporary_rivermark_layers(tmp_path)
    receipt = make_rivermark_layer_inventory(
        authority,
        [
            "anon:root-layer",
            local_layers[2],
            external_layers[1],
            local_layers[0],
            external_layers[0],
            "anon:session-layer",
            local_layers[1],
        ],
    )

    assert receipt["composition_scope"] == {
        "mode": "selective_references_only",
        "selective_references": [
            {"source_prim": source, "destination_prim": destination}
            for source, destination in SELECTIVE_REFERENCES
        ],
        "whole_final_stage_inventory": False,
    }
    assert receipt["asset_root"]["path"] == str(asset_root.resolve())
    assert receipt["ignored_anonymous_layer_count"] == 2
    assert receipt["local_authority_layer_count"] == 3
    assert receipt["rivermarksrc51_external_layer_count"] == 2
    assert [
        row["root_relative_path"]
        for row in receipt["rivermarksrc51_external_layers"]
    ] == [
        "dsready_content/props/building.usda",
        "dsready_content/scene/sub_terrain.usd",
    ]
    assert all(
        row["classification"] == "city_lite_local_authority"
        for row in receipt["local_authority_layers"]
    )
    assert all(
        row["classification"] == "rivermarksrc51_external_authority"
        for row in receipt["rivermarksrc51_external_layers"]
    )
    assert len(receipt["inventory_sha256"]) == 64


def test_rivermark_layer_inventory_is_order_independent_and_deduplicated(
    tmp_path: Path,
) -> None:
    authority, local_layers = _temporary_authority(tmp_path)
    _, external_layers = _temporary_rivermark_layers(tmp_path)
    first = make_rivermark_layer_inventory(
        authority,
        [*local_layers, *external_layers],
    )
    second = make_rivermark_layer_inventory(
        authority,
        [
            external_layers[1],
            local_layers[2],
            external_layers[0],
            external_layers[1],
            local_layers[0],
            local_layers[1],
        ],
    )

    assert second["rivermarksrc51_external_layer_count"] == 2
    assert second["inventory_sha256"] == first["inventory_sha256"]
    assert (
        second["rivermarksrc51_external_inventory_sha256"]
        == first["rivermarksrc51_external_inventory_sha256"]
    )


def test_rivermark_layer_inventory_hash_detects_external_tampering(
    tmp_path: Path,
) -> None:
    authority, local_layers = _temporary_authority(tmp_path)
    _, external_layers = _temporary_rivermark_layers(tmp_path)
    before = make_rivermark_layer_inventory(
        authority,
        [*local_layers, *external_layers],
    )
    external_layers[0].write_text(
        "#usda 1.0\n# externally tampered\n",
        encoding="ascii",
    )
    after = make_rivermark_layer_inventory(
        authority,
        [*local_layers, *external_layers],
    )

    assert after["inventory_sha256"] != before["inventory_sha256"]
    assert (
        after["rivermarksrc51_external_inventory_sha256"]
        != before["rivermarksrc51_external_inventory_sha256"]
    )


def test_rivermark_layer_inventory_fails_on_missing_or_incomplete_layers(
    tmp_path: Path,
) -> None:
    authority, local_layers = _temporary_authority(tmp_path)
    asset_root, external_layers = _temporary_rivermark_layers(tmp_path)

    with pytest.raises(CityLiteAuthorityError, match="missing or unresolved"):
        make_rivermark_layer_inventory(
            authority,
            [*local_layers, *external_layers, asset_root / "missing.usd"],
        )
    with pytest.raises(CityLiteAuthorityError, match="missing local"):
        make_rivermark_layer_inventory(
            authority,
            [*local_layers[:-1], *external_layers],
        )
    with pytest.raises(CityLiteAuthorityError, match="no RivermarkSrc51"):
        make_rivermark_layer_inventory(authority, local_layers)


def test_rivermark_layer_inventory_rejects_escape_and_multiple_roots(
    tmp_path: Path,
) -> None:
    authority, local_layers = _temporary_authority(tmp_path)
    _, external_layers = _temporary_rivermark_layers(tmp_path)
    legacy_layer = tmp_path / "legacy" / "Mission_Drones_CF2X.usd"
    legacy_layer.parent.mkdir()
    legacy_layer.write_text("#usda 1.0\n", encoding="ascii")

    with pytest.raises(CityLiteAuthorityError, match="escapes"):
        make_rivermark_layer_inventory(
            authority,
            [*local_layers, *external_layers, legacy_layer],
        )

    other_layer = (
        tmp_path
        / "second_install"
        / "RivermarkSrc51"
        / "dsready_content"
        / "other.usd"
    )
    other_layer.parent.mkdir(parents=True)
    other_layer.write_text("#usda 1.0\n", encoding="ascii")
    with pytest.raises(CityLiteAuthorityError, match="multiple"):
        make_rivermark_layer_inventory(
            authority,
            [*local_layers, *external_layers, other_layer],
        )


def test_rivermark_layer_inventory_never_classifies_local_layers_as_external(
    tmp_path: Path,
) -> None:
    authority, local_layers = _temporary_authority(tmp_path)
    asset_root, external_layers = _temporary_rivermark_layers(tmp_path)
    receipt = make_rivermark_layer_inventory(
        authority,
        [*external_layers, *local_layers],
        asset_root=asset_root,
    )

    external_paths = {
        row["path"] for row in receipt["rivermarksrc51_external_layers"]
    }
    assert external_paths.isdisjoint({str(path.resolve()) for path in local_layers})
    assert {row["filename"] for row in receipt["local_authority_layers"]} == set(
        AUTHORITY_SHA256
    )


def test_validate_rivermark_layer_inventory_receipt_accepts_serialized_receipt(
    tmp_path: Path,
) -> None:
    authority, local_layers = _temporary_authority(tmp_path)
    _, external_layers = _temporary_rivermark_layers(tmp_path)
    receipt = make_rivermark_layer_inventory(
        authority,
        ["anon:root", *local_layers, *external_layers],
    )

    for path in [*local_layers, *external_layers]:
        path.unlink()
    validate_rivermark_layer_inventory_receipt(receipt)


def test_validate_rivermark_layer_inventory_receipt_rejects_tampering(
    tmp_path: Path,
) -> None:
    authority, local_layers = _temporary_authority(tmp_path)
    _, external_layers = _temporary_rivermark_layers(tmp_path)
    original = make_rivermark_layer_inventory(
        authority,
        ["anon:root", *local_layers, *external_layers],
    )

    tampered_receipts: list[dict] = []

    tampered = deepcopy(original)
    tampered["schema"] = "org.rivermark.invalid.v1"
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["composition_scope"]["whole_final_stage_inventory"] = True
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["local_authority_layers"].pop()
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["rivermarksrc51_external_layer_count"] = 0
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["rivermarksrc51_external_layers"][0]["root_relative_path"] = (
        "../Mission_Drones_CF2X.usd"
    )
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["rivermarksrc51_external_layers"][0]["classification"] = (
        "city_lite_local_authority"
    )
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["rivermarksrc51_external_layers"][0]["size_bytes"] = 0
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["rivermarksrc51_external_layers"][0]["sha256"] = "0" * 64
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["local_authority_inventory_sha256"] = "0" * 64
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["rivermarksrc51_external_inventory_sha256"] = "0" * 64
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["inventory_sha256"] = "0" * 64
    tampered_receipts.append(tampered)

    tampered = deepcopy(original)
    tampered["unexpected_field"] = False
    tampered_receipts.append(tampered)

    for receipt in tampered_receipts:
        with pytest.raises(CityLiteAuthorityError):
            validate_rivermark_layer_inventory_receipt(receipt)
