"""Compile CitySpec into separate visual and physics USD layers."""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from .assets import AssetLock
from .canonical import content_hash, write_json
from .inspection_atlas import (
    ATLAS_PRIOR_COARSE,
    ATLAS_PRIOR_FULL,
    TASK_TRACK_G1_U,
    TASK_TRACK_G2_I,
    compile_inspection_atlas,
    project_inspection_atlas,
    validate_public_inspection_atlas,
)
from .ordinary_config import public_execution_contract
from .public_boundary import validate_public_task_spec

ROAD_SURFACE_COLORS = {
    "asphalt": [0.12, 0.13, 0.14],
    "weathered_asphalt": [0.17, 0.16, 0.15],
    "paved_local": [0.21, 0.20, 0.18],
}
SIDEWALK_COLOR = [0.34, 0.35, 0.35]
ROAD_MARKING_COLOR = [0.70, 0.68, 0.57]


def _name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _cube(
    name: str,
    center: list[float],
    size: list[float],
    color: list[float],
    indent: str = "        ",
    rotation_z_deg: float | None = None,
    collision_enabled: bool | None = None,
) -> str:
    order = ["xformOp:translate"]
    rotation_line: list[str] = []
    if rotation_z_deg is not None:
        rotation_line = [f"{indent}    double xformOp:rotateZ = {rotation_z_deg}"]
        order.append("xformOp:rotateZ")
    order.append("xformOp:scale")
    order_tokens = ", ".join(f'"{token}"' for token in order)
    return "\n".join(
        [
            f'{indent}def Cube "{_name(name)}"',
            f"{indent}{{",
            f"{indent}    double size = 2",
            f"{indent}    color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]",
            f"{indent}    double3 xformOp:scale = ({size[0] / 2}, {size[1] / 2}, {size[2] / 2})",
            f"{indent}    double3 xformOp:translate = ({center[0]}, {center[1]}, {center[2]})",
            *rotation_line,
            *(
                [
                    f"{indent}    bool physics:collisionEnabled = "
                    f"{'true' if collision_enabled else 'false'}"
                ]
                if collision_enabled is not None
                else []
            ),
            f"{indent}    uniform token[] xformOpOrder = [{order_tokens}]",
            f"{indent}}}",
        ]
    )


def _relative_asset_reference(record_root: str) -> str:
    parts = PurePosixPath(record_root).parts
    return PurePosixPath("../../../../_assets").joinpath(*parts).as_posix()


def _road_geometry(
    road: dict[str, Any], size_m: float
) -> tuple[list[float], float, float, float, tuple[float, float]]:
    """Return center, length, width, yaw and forward unit vector for a road."""

    width = float(road["width_m"])
    if "start" in road and "end" in road:
        start_x, start_y = [float(value) for value in road["start"]]
        end_x, end_y = [float(value) for value in road["end"]]
        delta_x, delta_y = end_x - start_x, end_y - start_y
        length = math.hypot(delta_x, delta_y)
        if length <= 1.0e-9:
            raise ValueError(f"road has a degenerate segment: {road['id']}")
        return (
            [(start_x + end_x) / 2.0, (start_y + end_y) / 2.0, 0.015],
            length,
            width,
            math.degrees(math.atan2(delta_y, delta_x)),
            (delta_x / length, delta_y / length),
        )
    if road["axis"] == "x":
        return [float(road["x"]), 0.0, 0.015], size_m, width, 90.0, (0.0, 1.0)
    return [0.0, float(road["y"]), 0.015], size_m, width, 0.0, (1.0, 0.0)


def compile_scene(city: dict[str, Any], public_dir: Path, lock: AssetLock) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    size = float(city["size_m"])
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Xform "Ground"',
        "    {",
        _cube("ground", [0.0, 0.0, -0.1], [size, size, 0.2], [0.18, 0.20, 0.18]),
        "    }",
        '    def Xform "Roads"',
        "    {",
    ]
    for road in city["roads"]:
        center, length, width, rotation_z_deg, _ = _road_geometry(road, size)
        lines.append(
            _cube(
                str(road["id"]),
                center,
                [length, width, 0.03],
                ROAD_SURFACE_COLORS.get(
                    str(road.get("surface_style")), ROAD_SURFACE_COLORS["asphalt"]
                ),
                rotation_z_deg=rotation_z_deg,
            )
        )
    lines.extend(["    }", '    def Xform "Buildings"', "    {"])
    for building in city["buildings"]:
        color = [float(value) for value in building["display_color"]]
        for component in building["components"]:
            lines.append(
                _cube(
                    f"{building['id']}-{component['id']}",
                    [float(value) for value in component["center"]],
                    [float(value) for value in component["size"]],
                    [
                        float(value)
                        for value in component.get("display_color", color)
                    ],
                )
            )
    lines.extend(["    }", '    def Xform "Obstacles"', "    {"])
    for obstacle in city["obstacles"]:
        lines.append(
            _cube(
                str(obstacle["id"]),
                [float(value) for value in obstacle["center"]],
                [float(value) for value in obstacle["size"]],
                [0.31, 0.28, 0.25],
            )
        )
    lines.extend(["    }", '    def Xform "UrbanGroundDetail"', "    {"])
    for road in city["roads"]:
        center, length, width, rotation_z_deg, forward = _road_geometry(road, size)
        normal = (-forward[1], forward[0])
        sidewalk_width = 0.78
        sidewalk_offset = width / 2.0 + sidewalk_width / 2.0 + 0.08
        for side in (-1.0, 1.0):
            sidewalk_center = [
                center[0] + normal[0] * sidewalk_offset * side,
                center[1] + normal[1] * sidewalk_offset * side,
                0.038,
            ]
            lines.append(
                _cube(
                    f"{road['id']}-sidewalk-{'left' if side < 0 else 'right'}",
                    sidewalk_center,
                    [length, sidewalk_width, 0.045],
                    SIDEWALK_COLOR,
                    rotation_z_deg=rotation_z_deg,
                )
            )
        marking_count = min(6, max(2, int(length / 15.0)))
        for marking_index in range(marking_count):
            longitudinal = (marking_index + 1) * length / (marking_count + 1) - length / 2.0
            marking_center = [
                center[0] + forward[0] * longitudinal,
                center[1] + forward[1] * longitudinal,
                0.052,
            ]
            lines.append(
                _cube(
                    f"{road['id']}-mark-{marking_index}",
                    marking_center,
                    [min(2.4, length / (marking_count + 2)), 0.14, 0.02],
                    ROAD_MARKING_COLOR,
                    rotation_z_deg=rotation_z_deg,
                )
            )
    lines.extend(["    }", '    def Xform "VisualDecorations"', "    {"])
    for decoration in city["decorations"]:
        record = lock.records[str(decoration["asset_id"])]
        reference = _relative_asset_reference(f"{lock.bundle}/{record.root_file}")
        position = decoration["position"]
        scale = float(decoration["scale"])
        translation = (
            f"            double3 xformOp:translate = ({position[0]}, {position[1]}, {position[2]})"
        )
        transform_order = (
            "            uniform token[] xformOpOrder = "
            '["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]'
        )
        lines.extend(
            [
                f'        def Xform "{_name(str(decoration["id"]))}" (',
                f"            prepend references = @{reference}@",
                "        )",
                "        {",
                "            bool physics:collisionEnabled = false",
                f"            double xformOp:rotateZ = {float(decoration['rotation_z_deg'])}",
                f"            double3 xformOp:scale = ({scale}, {scale}, {scale})",
                translation,
                transform_order,
                "        }",
            ]
        )
    lines.extend(["    }", '    def Xform "ProceduralVisualDetail"', "    {"])
    for accent in city.get("visual_facade_accents", []):
        lines.append(
            _cube(
                str(accent["id"]),
                [float(value) for value in accent["center"]],
                [float(value) for value in accent["size"]],
                [float(value) for value in accent["display_color"]],
                collision_enabled=False,
            )
        )
    lines.extend(["    }", "}", ""])
    (public_dir / "scene.usda").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    collision_groups: dict[str, list[tuple[str, str]]] = {
        "Ground": [("ground", "Cube")],
        "Buildings": [
            (_name(f"{building['id']}-{component['id']}"), "Cube")
            for building in city["buildings"]
            for component in building["components"]
        ],
        "Obstacles": [(_name(str(item["id"])), "Cube") for item in city["obstacles"]],
    }
    collision_lines = [
        "#usda 1.0",
        "(",
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'over Xform "World"',
        "{",
    ]
    for group, names in collision_groups.items():
        collision_lines.extend([f'    over Xform "{group}"', "    {"])
        for prim_name, prim_type in names:
            collision_lines.extend(
                [
                    f'        over {prim_type} "{prim_name}" (',
                    '            prepend apiSchemas = ["PhysicsCollisionAPI"]',
                    "        )",
                    "        {",
                    "            bool physics:collisionEnabled = true",
                    "        }",
                ]
            )
        collision_lines.extend(["    }", ""])
    collision_lines.extend(["}", ""])
    (public_dir / "collision.usda").write_text(
        "\n".join(collision_lines), encoding="utf-8", newline="\n"
    )
    stage = "\n".join(
        [
            "#usda 1.0",
            "(",
            '    defaultPrim = "World"',
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            "    subLayers = [",
            "        @scene.usda@,",
            "        @collision.usda@",
            "    ]",
            ")",
            "",
        ]
    )
    (public_dir / "stage.usda").write_text(stage, encoding="utf-8", newline="\n")


def compile_public_catalogue(city: dict[str, Any]) -> dict[str, Any]:
    poses: list[dict[str, Any]] = []
    for building in city["buildings"]:
        x, y, w, d = [float(value) for value in building["footprint"]]
        height = float(building["height_m"])
        offset = max(w, d) * 0.65 + 4.0
        for side, position, yaw in (
            ("south", [x, y - offset, min(height * 0.55, 32.0)], 90.0),
            ("east", [x + offset, y, min(height * 0.55, 32.0)], 180.0),
            ("north", [x, y + offset, min(height * 0.55, 32.0)], -90.0),
            ("west", [x - offset, y, min(height * 0.55, 32.0)], 0.0),
            ("roof", [x, y, height + 6.0], 0.0),
        ):
            poses.append(
                {
                    "pose_id": f"view-{building['id']}-{side}",
                    "position": [round(value, 4) for value in position],
                    "yaw_deg": yaw,
                    "pitch_deg": -35.0 if side == "roof" else 0.0,
                    "support_hint": side,
                }
            )
    return {
        "schema": "org.aerocity.bench.public-catalogue.v1",
        "layout_id": city["layout_id"],
        "layout_hash": city["layout_hash"],
        "view_poses": poses,
    }


def public_cityspec(city: dict[str, Any]) -> dict[str, Any]:
    """Project the internal generator record onto the method-visible contract."""

    public_fields = (
        "generator_version",
        "layout_id",
        "layout_hash",
        "size_m",
        "roads",
        "blocks",
        "buildings",
        "obstacles",
        "decorations",
        "flight_bounds",
        "topology_signature",
        "asset_set_hash",
        "metrics",
    )
    return {
        "schema": "org.aerocity.bench.cityspec.v2",
        **{field: city[field] for field in public_fields},
    }


def public_cityspec_v3(city: dict[str, Any]) -> dict[str, Any]:
    """Project ordinary-v3 geometry without split, grammar, seed, or target truth."""

    public_fields = (
        "generator_version",
        "layout_id",
        "layout_hash",
        "task_geometry_hash",
        "size_m",
        "road_graph",
        "roads",
        "buildings",
        "obstacles",
        "decorations",
        "flight_bounds",
        "topology_signature",
        "asset_set_hash",
        "metrics",
    )
    return {
        "schema": "org.aerocity.bench.cityspec.ordinary.v3",
        **{field: city[field] for field in public_fields},
    }


def compile_coarse_prior(city: dict[str, Any]) -> dict[str, Any]:
    task_geometry_hash = str(city.get("task_geometry_hash", city["layout_hash"]))
    seed = int(task_geometry_hash[:16], 16) ^ 0xA3C0_17
    # A local deterministic generator avoids coupling public corruption or
    # visual-only scene variants to coarse task geometry.
    import random

    rng = random.Random(seed)
    omission = 0.09
    coordinate_error = 1.5
    height_error = 0.12
    buildings = []
    for building in city["buildings"]:
        if rng.random() < omission:
            continue
        x, y, w, d = [float(value) for value in building["footprint"]]
        buildings.append(
            {
                "prior_id": f"prior-{building['id']}",
                "center_xy": [
                    round(x + rng.uniform(-coordinate_error, coordinate_error), 4),
                    round(y + rng.uniform(-coordinate_error, coordinate_error), 4),
                ],
                "size_xy": [round(w, 4), round(d, 4)],
                "height_m": round(
                    float(building["height_m"]) * (1.0 + rng.uniform(-height_error, height_error)),
                    4,
                ),
            }
        )
    false_obstacles = []
    for index in range(1):
        span = float(city["size_m"]) * 0.4
        false_obstacles.append(
            {
                "prior_id": f"false-obstacle-{index:02d}",
                "center_xy": [
                    round(rng.uniform(-span, span), 4),
                    round(rng.uniform(-span, span), 4),
                ],
                "radius_m": round(rng.uniform(2.0, 7.0), 4),
            }
        )
    prior = {
        "schema": "org.aerocity.bench.coarse-prior.v1",
        "layout_id": city["layout_id"],
        "profile": "standard-v1",
        "buildings": buildings,
        "false_obstacles": false_obstacles,
    }
    prior["prior_hash"] = content_hash(prior)
    return prior


def compile_method_task_spec(
    city: dict[str, Any],
    execution_contract: dict[str, Any],
    fleet_profile: dict[str, Any],
    *,
    task_track: str = TASK_TRACK_G1_U,
    inspection_prior_level: str = ATLAS_PRIOR_FULL,
    inspection_sampling_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned public task contract without evaluator-private truth.

    The current ordinary-v3 builder continues to emit ``G1-U`` for target-free
    exploration compatibility.  ``G2-I`` is an explicit opt-in compiler path:
    it adds a target-agnostic inspection atlas but does not promote any run to
    a formal score or make the existing G1 runtime atlas-aware.
    """

    if task_track not in {TASK_TRACK_G1_U, TASK_TRACK_G2_I}:
        raise ValueError(f"unknown method task track: {task_track}")
    if task_track == TASK_TRACK_G1_U and inspection_prior_level != ATLAS_PRIOR_FULL:
        raise ValueError("G1-U does not accept a G2-I inspection prior level")

    public_contract = public_execution_contract(execution_contract)
    vehicle = public_contract["vehicle"]
    collider_ceilings = [
        float(component["center"][2]) + float(component["size"][2]) / 2.0
        for building in city["buildings"]
        for component in building["components"]
    ] + [
        float(obstacle["center"][2]) + float(obstacle["size"][2]) / 2.0
        for obstacle in city["obstacles"]
    ]
    if not collider_ceilings:
        raise ValueError("city must contain physical colliders before compiling a task spec")
    body_margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    # This is an aggregate public geometry contract, not target truth.  It gives
    # G1 methods a provably safe sky corridor without exposing exact building
    # placement.  The extra half metre covers serialized geometry and controller
    # tracking error; any generator height-envelope change must update this hash.
    safe_sky_altitude = max(collider_ceilings) + body_margin + 0.5
    maximum_safe_altitude = float(city["flight_bounds"]["maximum"][2]) - body_margin
    if safe_sky_altitude > maximum_safe_altitude + 1.0e-9:
        raise ValueError(
            "city collider ceiling leaves no public safe-sky transit lane inside flight bounds"
        )
    task_spec = {
        "schema": "org.aerocity.bench.task-spec-public.ordinary.v1",
        "layout_id": city["layout_id"],
        "coordinate_frame": {
            "world": "right-handed-Z-up",
            "linear_unit": "meter",
            "angular_unit": "degree",
        },
        "flight_bounds": city["flight_bounds"],
        "fleet_profile": fleet_profile,
        "execution_contract": public_contract,
        "public_execution_contract_hash": content_hash(public_contract),
        "coarse_prior": compile_coarse_prior(city),
        "task_track": TASK_TRACK_G1_U,
        "public_transit_contract": {
            "safe_sky_altitude_m": round(safe_sky_altitude, 4),
            "height_envelope_basis": "aggregate_city_collider_ceiling_plus_vehicle_margin",
            "transit_clearance_margin_m": 0.5,
        },
        "exact_cityspec_public": False,
        "target_count_public": False,
        "target_process_public": False,
        "formal_split_label_public": False,
    }
    if task_track == TASK_TRACK_G2_I:
        atlas = compile_inspection_atlas(
            city,
            execution_contract,
            sampling_policy=inspection_sampling_policy,
        )
        validate_public_inspection_atlas(atlas)
        task_spec["schema"] = "org.aerocity.bench.task-spec-public.g2-i.v1"
        task_spec["task_track"] = TASK_TRACK_G2_I
        task_spec["inspection_prior_level"] = inspection_prior_level
        projection = project_inspection_atlas(atlas, inspection_prior_level)
        if inspection_prior_level == ATLAS_PRIOR_FULL:
            task_spec["inspection_atlas"] = atlas
        elif inspection_prior_level == ATLAS_PRIOR_COARSE:
            task_spec["inspection_atlas_projection"] = projection
        else:
            raise ValueError(f"unsupported G2-I inspection prior: {inspection_prior_level}")
    task_spec["task_spec_hash"] = content_hash(task_spec)
    validate_public_task_spec(task_spec)
    return task_spec


def compile_g2_i_task_spec(
    city: dict[str, Any],
    execution_contract: dict[str, Any],
    fleet_profile: dict[str, Any],
    *,
    inspection_prior_level: str = ATLAS_PRIOR_FULL,
    inspection_sampling_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile the planned G2-I public projection without changing G1-U output."""

    return compile_method_task_spec(
        city,
        execution_contract,
        fleet_profile,
        task_track=TASK_TRACK_G2_I,
        inspection_prior_level=inspection_prior_level,
        inspection_sampling_policy=inspection_sampling_policy,
    )


def write_compiled_public(city: dict[str, Any], public_dir: Path, lock: AssetLock) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    write_json(public_dir / "cityspec.json", public_cityspec(city))
    write_json(public_dir / "public_catalogue.json", compile_public_catalogue(city))
    write_json(public_dir / "coarse_prior.json", compile_coarse_prior(city))
    compile_scene(city, public_dir, lock)


def write_compiled_public_v3(city: dict[str, Any], public_dir: Path, lock: AssetLock) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    write_json(public_dir / "cityspec.json", public_cityspec_v3(city))
    write_json(public_dir / "coarse_prior.json", compile_coarse_prior(city))
    # This catalogue is diagnostic/training metadata. A formal G1 container never mounts it.
    write_json(public_dir / "developer_view_catalogue.json", compile_public_catalogue(city))
    compile_scene(city, public_dir, lock)


def usda_metadata(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "path": path.name,
        "nonblank": bool(text.strip()),
        "line_count": len(text.splitlines()),
        "default_prim_declared": 'defaultPrim = "World"' in text,
        "world_declared": 'def Xform "World"' in text or path.name != "scene.usda",
        "json_debug": json.dumps(
            {"bytes": path.stat().st_size, "finite": math.isfinite(path.stat().st_size)}
        ),
    }
