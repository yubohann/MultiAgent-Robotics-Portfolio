"""Frozen constants for the Rivermark City-Lite scene contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ENVIRONMENT_ID = "RIVERMARK_CITY_LITE_v1"
SCENE_CONTRACT_FILENAME = "rivermark_city_lite_scene_contract_v1.json"
SCENE_CONTRACT_SCHEMA = "md_qd_swarm_t32_rivermark_city_lite_scene_contract_v1"
SCENE_CONTRACT_GATE_STATUS = "pass_city_lite_static_construction"

SCENE_CONTRACT_SHA256 = "f7837d248b4797592c66d4b8b8bd48380de444eca305b8ef03d297dcf32051ea"
SCENE_CONTRACT_PAYLOAD_SHA256 = (
    "1d5838d90c9920a849fd68d4051c81ad519ce0b79fb3cb3ffa2bcb643ac8544d"
)

# These are the exact outputs admitted by the md_qd_swarm v1_r2 static
# composition contract. That contract is not runtime or dataset admission.
AUTHORITY_SHA256: Mapping[str, str] = {
    "rivermark_city_lite_base_v1.usda": (
        "162dfcf12e1a3f48257fab8c06e1de6b063e63559d3953fca22330750c63c6ad"
    ),
    "rivermark_city_lite_structural_props_v1.usda": (
        "a30874dc3ca4e3919ad0d6281879092cd4146b2a21b71bce269a8c20a5d76f5e"
    ),
    "hi_fi_search_rescue_rivermark_city_lite_v1.usda": (
        "c8ec943618322c07f1ce8799e366b5fb984d3b302566b4404892fb50bfa86567"
    ),
}

FINAL_SCENE_FILENAME = "hi_fi_search_rescue_rivermark_city_lite_v1.usda"
RIVERMARK_ASSET_ROOT_NAME = "RivermarkSrc51"
RIVERMARK_LAYER_INVENTORY_SCHEMA = (
    "org.rivermark.city-lite.resolved-layer-inventory.v1"
)
_OUTPUT_BINDINGS: Mapping[str, str] = {
    "city_lite_base_usd": "rivermark_city_lite_base_v1.usda",
    "filtered_structural_props_usd": "rivermark_city_lite_structural_props_v1.usda",
    "final_combined_usd": FINAL_SCENE_FILENAME,
}

EXPECTED_UPSTREAM_PERMISSIONS: Mapping[str, bool] = {
    "blind128": False,
    "c3_rollout": False,
    "c3_training": False,
    "c4": False,
    "c5": False,
    "formal_collection": False,
    "scene_runtime_admission": False,
    "validation64": False,
}

# Referencing the final layer's default /World would also import the legacy
# Mission, targets, route, presentation drones, materials, and Render prims.
SELECTIVE_REFERENCES: tuple[tuple[str, str], ...] = (
    ("/World/City/Rivermark", "/World/StaticScene/City/Rivermark"),
    ("/World/CityTaskObstacles", "/World/StaticScene/CityTaskObstacles"),
)

# The upstream task-obstacle meshes bind to absolute /World/Materials paths.
# Those targets sit outside the admitted /World/CityTaskObstacles reference
# scope, so City-Lite composes exact local USD Preview Surface replacements
# before the reference is loaded.  This is deliberately an eight-material
# allow-list, not a reference to the legacy material library.
CITY_TASK_OBSTACLE_MATERIAL_CLOSURE_SCHEMA = (
    "org.rivermark.city-lite.task-obstacle-material-closure.v1"
)
CITY_TASK_OBSTACLE_MATERIAL_ROOT = "/World/StaticScene/CityTaskObstacleMaterials"
CITY_TASK_OBSTACLE_MATERIAL_SPECS: tuple[Mapping[str, Any], ...] = (
    {
        "obstacle_name": "south_collapsed_facade",
        "material_name": "World_CityTaskObstacles_south_collapsed_facade_107_86_76_188",
        "diffuse_color": (0.42, 0.34, 0.30),
        "opacity": 0.74,
        "roughness": 0.82,
    },
    {
        "obstacle_name": "midblock_service_wall",
        "material_name": "World_CityTaskObstacles_midblock_service_wall_89_96_102_188",
        "diffuse_color": (0.35, 0.38, 0.40),
        "opacity": 0.74,
        "roughness": 0.82,
    },
    {
        "obstacle_name": "west_tower_rubble_screen",
        "material_name": "World_CityTaskObstacles_west_tower_rubble_screen_96_86_76_188",
        "diffuse_color": (0.38, 0.34, 0.30),
        "opacity": 0.74,
        "roughness": 0.82,
    },
    {
        "obstacle_name": "north_skybridge_debris",
        "material_name": "World_CityTaskObstacles_north_skybridge_debris_81_94_102_188",
        "diffuse_color": (0.32, 0.37, 0.40),
        "opacity": 0.74,
        "roughness": 0.82,
    },
    {
        "obstacle_name": "warning_light_00",
        "material_name": "World_CityTaskObstacles_warning_light_00_255_114_5_188",
        "diffuse_color": (1.00, 0.45, 0.02),
        "opacity": 0.74,
        "roughness": 0.82,
    },
    {
        "obstacle_name": "warning_light_01",
        "material_name": "World_CityTaskObstacles_warning_light_01_255_114_5_188",
        "diffuse_color": (1.00, 0.45, 0.02),
        "opacity": 0.74,
        "roughness": 0.82,
    },
    {
        "obstacle_name": "warning_light_02",
        "material_name": "World_CityTaskObstacles_warning_light_02_255_114_5_188",
        "diffuse_color": (1.00, 0.45, 0.02),
        "opacity": 0.74,
        "roughness": 0.82,
    },
    {
        "obstacle_name": "warning_light_03",
        "material_name": "World_CityTaskObstacles_warning_light_03_255_114_5_188",
        "diffuse_color": (1.00, 0.45, 0.02),
        "opacity": 0.74,
        "roughness": 0.82,
    },
)

FORBIDDEN_PRIM_PREFIXES: tuple[str, ...] = (
    "/World/Mission",
    "/World/Drones",
    "/World/StaticScene/Mission",
    "/World/StaticScene/Drones",
)

# These are exact upstream prim roots. Do not broaden this to generic words
# such as "sign" or "vegetation": those produce false positives.
FORBIDDEN_DECORATION_COMPONENTS = frozenset(
    {"foliage", "grass", "sub_traffic_signs"}
)
FORBIDDEN_DECORATION_PRIM_PREFIXES: tuple[str, ...] = tuple(
    f"{root}/{component}"
    for root in (
        "/World/City/Rivermark",
        "/World/StaticScene/City/Rivermark",
    )
    for component in sorted(FORBIDDEN_DECORATION_COMPONENTS)
)

EXPECTED_NATIVE_COLLISION_COUNTS: Mapping[str, int] = {
    "total": 4811,
    "drivable_surfaces": 4807,
    "city_task_obstacles": 4,
    "structural_props": 0,
}

ROUTE_CONTRACT_SCHEMA = "org.rivermark.city-lite-public-routes.v1"
ROUTE_GENERATION = "target-free-public-static-geometry-v1"
ROUTE_CONDITIONING = "public_static_geometry_only"
ROUTE_CLEARANCE_M = 0.85
AGENT_COUNT = 8
MINIMUM_ROUTE_WAYPOINTS = 3
_EPSILON = 1.0e-12
