"""Ordinary-v3 procedural city generator without index-periodic shortcuts."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from .canonical import content_hash, derived_seed
from .errors import GenerationRejected
from .geometry import AABB
from .ordinary_config import OrdinaryReleaseConfig

DEVELOPMENT_TEMPLATES = ("rectangle", "l_shape", "slab", "tower_podium")
OOD_TEMPLATES = ("courtyard", "u_shape", "stepped_tower", "narrow_slab")
COLORS = (
    (0.55, 0.58, 0.61),
    (0.69, 0.55, 0.44),
    (0.42, 0.52, 0.57),
    (0.63, 0.65, 0.56),
    (0.52, 0.46, 0.43),
    (0.46, 0.49, 0.58),
)
ROAD_SURFACE_STYLES = ("asphalt", "weathered_asphalt", "paved_local")
PARAPET_COLOR = (0.30, 0.32, 0.34)
CANOPY_COLOR = (0.18, 0.20, 0.22)
ROOF_EQUIPMENT_COLOR = (0.28, 0.29, 0.30)
COLLIDER_JOIN_GAP_M = 0.02
VISUAL_DETAIL_PROFILE = "procedural-urban-detail-v1"
MAX_VISUAL_FACADE_ACCENTS_PER_BUILDING = 4
FACADE_ACCENT_COLORS = (
    (0.11, 0.19, 0.24),
    (0.18, 0.25, 0.29),
    (0.25, 0.30, 0.31),
)


def _node(node_id: str, x: float, y: float) -> dict[str, Any]:
    return {"id": node_id, "position": [round(x, 4), round(y, 4), 0.0]}


def _edge(
    edge_id: str,
    start: str,
    end: str,
    nodes: dict[str, dict[str, Any]],
    width: float,
    road_class: str = "local",
    blocked: bool = False,
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "start_node": start,
        "end_node": end,
        "start": nodes[start]["position"][:2],
        "end": nodes[end]["position"][:2],
        "width_m": round(width, 4),
        "class": road_class,
        "blocked": blocked,
    }


def _road_graph(
    family: str, size_m: int, width: float, rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    half = size_m / 2.0
    jitter = min(3.0, size_m * 0.035)
    nodes: dict[str, dict[str, Any]] = {}
    edge_specs: list[tuple[str, str, str, str, bool]] = []

    if family == "offset_grid":
        xs = [-size_m * 0.19 + rng.uniform(-jitter, jitter), size_m * 0.21]
        ys = [-size_m * 0.23, size_m * 0.17 + rng.uniform(-jitter, jitter)]
        for x_index, x in enumerate(xs):
            for y_index, y in enumerate([-half, *ys, half]):
                nodes[f"v{x_index}-{y_index}"] = _node(f"v{x_index}-{y_index}", x, y)
            for segment in range(3):
                edge_specs.append(
                    (
                        f"vertical-{x_index}-{segment}",
                        f"v{x_index}-{segment}",
                        f"v{x_index}-{segment + 1}",
                        "arterial" if x_index == 0 else "local",
                        False,
                    )
                )
        for y_index, y in enumerate(ys):
            left = f"h{y_index}-0"
            right = f"h{y_index}-3"
            nodes[left] = _node(left, -half, y)
            nodes[right] = _node(right, half, y)
            # Intersection nodes share coordinates but remain explicit graph vertices.
            for x_index, x in enumerate(xs, start=1):
                key = f"h{y_index}-{x_index}"
                nodes[key] = _node(key, x, y)
            for segment in range(3):
                edge_specs.append(
                    (
                        f"horizontal-{y_index}-{segment}",
                        f"h{y_index}-{segment}",
                        f"h{y_index}-{segment + 1}",
                        "local",
                        False,
                    )
                )
    elif family == "t_junction":
        points = {
            "west": (-half, rng.uniform(-jitter, jitter)),
            "junction": (rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)),
            "east": (half, rng.uniform(-jitter, jitter)),
            "south": (rng.uniform(-jitter, jitter), -half),
            "branch": (size_m * 0.28, size_m * 0.30),
        }
        nodes = {key: _node(key, *value) for key, value in points.items()}
        edge_specs = [
            ("main-west", "west", "junction", "arterial", False),
            ("main-east", "junction", "east", "arterial", False),
            ("stem", "south", "junction", "local", False),
            ("dead-end", "junction", "branch", "local", False),
        ]
    elif family == "dead_end_courts":
        points = {
            "south": (rng.uniform(-jitter, jitter), -half),
            "center": (0.0, 0.0),
            "north": (rng.uniform(-jitter, jitter), half),
            "west_end": (-size_m * 0.34, size_m * 0.12),
            "east_end": (size_m * 0.31, -size_m * 0.15),
        }
        nodes = {key: _node(key, *value) for key, value in points.items()}
        edge_specs = [
            ("spine-south", "south", "center", "arterial", False),
            ("spine-north", "center", "north", "arterial", False),
            ("west-court", "center", "west_end", "local", False),
            ("east-court", "center", "east_end", "local", False),
        ]
    elif family == "ring_spokes":
        radius = size_m * 0.31
        for index in range(8):
            angle = 2.0 * math.pi * index / 8.0 + rng.uniform(-0.035, 0.035)
            nodes[f"ring-{index}"] = _node(
                f"ring-{index}", radius * math.cos(angle), radius * math.sin(angle)
            )
        nodes["center"] = _node("center", rng.uniform(-jitter, jitter), 0.0)
        for index in range(8):
            edge_specs.append(
                (f"ring-edge-{index}", f"ring-{index}", f"ring-{(index + 1) % 8}", "local", False)
            )
        for index in (0, 2, 4, 6):
            edge_specs.append((f"spoke-{index}", "center", f"ring-{index}", "arterial", False))
    elif family == "staggered_loop":
        points = {
            "sw": (-size_m * 0.33, -size_m * 0.27),
            "se": (size_m * 0.28, -size_m * 0.34),
            "east": (size_m * 0.36, size_m * 0.08),
            "ne": (size_m * 0.18, size_m * 0.34),
            "nw": (-size_m * 0.35, size_m * 0.25),
            "inner": (rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)),
        }
        nodes = {key: _node(key, *value) for key, value in points.items()}
        loop = ("sw", "se", "east", "ne", "nw", "sw")
        edge_specs = [
            (f"loop-{index}", loop[index], loop[index + 1], "local", index == 2)
            for index in range(len(loop) - 1)
        ]
        edge_specs.extend(
            [
                ("inner-west", "nw", "inner", "arterial", False),
                ("inner-east", "inner", "east", "arterial", False),
            ]
        )
    else:
        raise GenerationRejected(f"unknown road grammar family: {family}")

    roads = [
        _edge(edge_id, start, end, nodes, width, road_class, blocked)
        for edge_id, start, end, road_class, blocked in edge_specs
    ]
    return [nodes[key] for key in sorted(nodes)], roads


def _point_segment_distance(point: tuple[float, float], road: dict[str, Any]) -> float:
    px, py = point
    ax, ay = (float(value) for value in road["start"])
    bx, by = (float(value) for value in road["end"])
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-9:
        return math.hypot(px - ax, py - ay)
    factor = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return math.hypot(px - (ax + factor * dx), py - (ay + factor * dy))


def _components(
    template: str, x: float, y: float, width: float, depth: float, height: float
) -> list[dict[str, Any]]:
    joint_gap = COLLIDER_JOIN_GAP_M

    def box(
        name: str, cx: float, cy: float, cz: float, sx: float, sy: float, sz: float
    ) -> dict[str, Any]:
        return {
            "id": name,
            "center": [round(cx, 4), round(cy, 4), round(cz, 4)],
            "size": [round(sx, 4), round(sy, 4), round(sz, 4)],
        }

    if template in {"rectangle", "slab", "narrow_slab"}:
        return [box("body", x, y, height / 2.0, width, depth, height)]
    thickness = min(width, depth) * 0.28
    if template == "l_shape":
        south_width = width - thickness - joint_gap
        return [
            box("west", x - (width - thickness) / 2.0, y, height / 2.0, thickness, depth, height),
            box(
                "south",
                x + (thickness + joint_gap) / 2.0,
                y - (depth - thickness) / 2.0,
                height / 2.0,
                south_width,
                thickness,
                height,
            ),
        ]
    if template == "u_shape":
        leg_depth = depth - thickness - joint_gap
        return [
            box("north", x, y + (depth - thickness) / 2.0, height / 2.0, width, thickness, height),
            box(
                "west",
                x - (width - thickness) / 2.0,
                y - (thickness + joint_gap) / 2.0,
                height / 2.0,
                thickness,
                leg_depth,
                height,
            ),
            box(
                "east",
                x + (width - thickness) / 2.0,
                y - (thickness + joint_gap) / 2.0,
                height / 2.0,
                thickness,
                leg_depth,
                height,
            ),
        ]
    if template == "courtyard":
        vertical_depth = depth - 2.0 * thickness - 2.0 * joint_gap
        return [
            box("north", x, y + (depth - thickness) / 2.0, height / 2.0, width, thickness, height),
            box("south", x, y - (depth - thickness) / 2.0, height / 2.0, width, thickness, height),
            box(
                "west",
                x - (width - thickness) / 2.0,
                y,
                height / 2.0,
                thickness,
                vertical_depth,
                height,
            ),
            box(
                "east",
                x + (width - thickness) / 2.0,
                y,
                height / 2.0,
                thickness,
                vertical_depth,
                height,
            ),
        ]
    if template == "tower_podium":
        podium_height = max(4.5, height * 0.22)
        tower_height = height - podium_height - joint_gap
        return [
            box("podium", x, y, podium_height / 2.0, width, depth, podium_height),
            box(
                "tower",
                x,
                y,
                podium_height + joint_gap + tower_height / 2.0,
                width * 0.58,
                depth * 0.58,
                tower_height,
            ),
        ]
    if template == "stepped_tower":
        low, middle = height * 0.30, height * 0.32
        top_height = height - low - middle - 2.0 * joint_gap
        return [
            box("base", x, y, low / 2.0, width, depth, low),
            box(
                "middle",
                x,
                y,
                low + joint_gap + middle / 2.0,
                width * 0.72,
                depth * 0.72,
                middle,
            ),
            box(
                "top",
                x,
                y,
                low + joint_gap + middle + joint_gap + top_height / 2.0,
                width * 0.45,
                depth * 0.45,
                top_height,
            ),
        ]
    raise GenerationRejected(f"unknown building template: {template}")


def _component_horizontal_bounds(component: dict[str, Any]) -> tuple[float, float, float, float]:
    center_x, center_y, _ = (float(value) for value in component["center"])
    size_x, size_y, _ = (float(value) for value in component["size"])
    return (
        center_x - size_x / 2.0,
        center_x + size_x / 2.0,
        center_y - size_y / 2.0,
        center_y + size_y / 2.0,
    )


def _component_roof_z(component: dict[str, Any]) -> float:
    return float(component["center"][2]) + float(component["size"][2]) / 2.0


def _face_anchor(
    components: list[dict[str, Any]], side: str
) -> tuple[dict[str, Any], tuple[float, float], float]:
    """Pick a real exterior component face for an entrance or canopy.

    The building footprint is only an outer envelope for compound grammars.
    Selecting an actual component face keeps details out of U-shaped openings,
    courtyards, and stepped-tower voids.
    """

    if side not in {"south", "east", "north", "west"}:
        raise ValueError(f"unknown facade side: {side}")
    candidates: list[tuple[float, float, str, dict[str, Any]]] = []
    for component in components:
        minimum_x, maximum_x, minimum_y, maximum_y = _component_horizontal_bounds(component)
        if side == "south":
            candidates.append((minimum_y, maximum_x - minimum_x, str(component["id"]), component))
        elif side == "north":
            candidates.append((-maximum_y, maximum_x - minimum_x, str(component["id"]), component))
        elif side == "west":
            candidates.append((minimum_x, maximum_y - minimum_y, str(component["id"]), component))
        else:
            candidates.append((-maximum_x, maximum_y - minimum_y, str(component["id"]), component))
    _, _, _, component = min(candidates, key=lambda item: (item[0], -item[1], item[2]))
    minimum_x, maximum_x, minimum_y, maximum_y = _component_horizontal_bounds(component)
    if side == "south":
        return component, (0.0, -1.0), maximum_x - minimum_x
    if side == "north":
        return component, (0.0, 1.0), maximum_x - minimum_x
    if side == "west":
        return component, (-1.0, 0.0), maximum_y - minimum_y
    return component, (1.0, 0.0), maximum_y - minimum_y


def _face_is_externally_clear(
    components: list[dict[str, Any]], component_index: int, side: str
) -> bool:
    """Reject full-width parapets whose exterior probe enters another roof.

    Compound footprint components may overlap at a corner or along part of a
    facade. A single box parapet cannot represent that partial boundary without
    creating an internal collider seam, so this conservative check omits it.
    """

    component = components[component_index]
    minimum_x, maximum_x, minimum_y, maximum_y = _component_horizontal_bounds(component)
    roof_z = _component_roof_z(component)
    epsilon = 0.04
    if side == "south":
        probe = (float(component["center"][0]), minimum_y - epsilon)
        tangent = (minimum_x, maximum_x)
        axis = "x"
    elif side == "north":
        probe = (float(component["center"][0]), maximum_y + epsilon)
        tangent = (minimum_x, maximum_x)
        axis = "x"
    elif side == "west":
        probe = (minimum_x - epsilon, float(component["center"][1]))
        tangent = (minimum_y, maximum_y)
        axis = "y"
    elif side == "east":
        probe = (maximum_x + epsilon, float(component["center"][1]))
        tangent = (minimum_y, maximum_y)
        axis = "y"
    else:
        raise ValueError(f"unknown facade side: {side}")

    for other_index, other in enumerate(components):
        if other_index == component_index or _component_roof_z(other) < roof_z - epsilon:
            continue
        other_minimum_x, other_maximum_x, other_minimum_y, other_maximum_y = (
            _component_horizontal_bounds(other)
        )
        if not (
            other_minimum_x <= probe[0] <= other_maximum_x
            and other_minimum_y <= probe[1] <= other_maximum_y
        ):
            continue
        other_tangent = (
            (other_minimum_x, other_maximum_x)
            if axis == "x"
            else (other_minimum_y, other_maximum_y)
        )
        if min(tangent[1], other_tangent[1]) > max(tangent[0], other_tangent[0]):
            return False
    return True


def _architectural_components(
    components: list[dict[str, Any]],
    *,
    entrance_side: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Add bounded, collider-backed architectural detail to one building.

    The details deliberately use simple boxes rather than a large unbounded mesh
    library.  They make rooflines and entrances legible in native review while
    keeping the L1 collision budget predictable.  They are never target support
    surfaces, so a method cannot exploit one repeated detail class as a target
    prior.
    """

    details: list[dict[str, Any]] = []
    parapet_height = rng.uniform(0.32, 0.50)
    parapet_thickness = rng.uniform(0.18, 0.28)
    parapet_roof_gap = 0.015
    parapet_candidates: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        center_x, center_y, center_z = (float(value) for value in component["center"])
        size_x, size_y, size_z = (float(value) for value in component["size"])
        roof_z = center_z + size_z / 2.0
        base_id = str(component["id"])
        # Perpendicular parapets must have a real join gap, not merely a
        # mathematical edge contact.  CitySpec serializes each box to four
        # decimals independently; nominal contact can otherwise become a
        # positive-volume corner overlap after serialization.
        horizontal_span = size_x - 2.0 * parapet_thickness - COLLIDER_JOIN_GAP_M
        vertical_span = size_y - 2.0 * parapet_thickness - COLLIDER_JOIN_GAP_M
        for suffix, detail_center, detail_size in (
            (
                "parapet_south",
                (
                    center_x,
                    center_y - size_y / 2.0 + parapet_thickness / 2.0,
                    roof_z + parapet_roof_gap + parapet_height / 2.0,
                ),
                (horizontal_span, parapet_thickness, parapet_height),
            ),
            (
                "parapet_north",
                (
                    center_x,
                    center_y + size_y / 2.0 - parapet_thickness / 2.0,
                    roof_z + parapet_roof_gap + parapet_height / 2.0,
                ),
                (horizontal_span, parapet_thickness, parapet_height),
            ),
            (
                "parapet_west",
                (
                    center_x - size_x / 2.0 + parapet_thickness / 2.0,
                    center_y,
                    roof_z + parapet_roof_gap + parapet_height / 2.0,
                ),
                (parapet_thickness, vertical_span, parapet_height),
            ),
            (
                "parapet_east",
                (
                    center_x + size_x / 2.0 - parapet_thickness / 2.0,
                    center_y,
                    roof_z + parapet_roof_gap + parapet_height / 2.0,
                ),
                (parapet_thickness, vertical_span, parapet_height),
            ),
        ):
            side = suffix.removeprefix("parapet_")
            if (
                max(detail_size[0], detail_size[1]) <= 0.60
                or not _face_is_externally_clear(components, component_index, side)
            ):
                continue
            parapet_candidates.append(
                {
                    "id": f"{base_id}_{suffix}",
                    "center": [round(value, 4) for value in detail_center],
                    "size": [round(value, 4) for value in detail_size],
                    "target_support": False,
                    "structural_role": "roof_parapet",
                    "host_component_id": base_id,
                    "display_color": list(PARAPET_COLOR),
                }
            )

    # Complex courtyard and stepped grammars have more component roofs than a
    # single tower.  A fixed upper bound prevents them from becoming a hidden
    # performance variable or a large collider budget.  The seed is city-local
    # and never derives from a target label.
    if len(parapet_candidates) > 8:
        selected = set(rng.sample(range(len(parapet_candidates)), 8))
        details.extend(
            candidate
            for index, candidate in enumerate(parapet_candidates)
            if index in selected
        )
    else:
        details.extend(parapet_candidates)

    host_component, normal, tangent_span = _face_anchor(components, entrance_side)
    host_center_x, host_center_y, _ = (float(value) for value in host_component["center"])
    host_size_x, host_size_y, _ = (float(value) for value in host_component["size"])
    canopy_width = min(
        4.2,
        max(1.0, min(tangent_span - 0.35, tangent_span * 0.55)),
    )
    canopy_depth = rng.uniform(0.85, 1.20)
    canopy_height = rng.uniform(0.18, 0.28)
    canopy_z = rng.uniform(2.45, 2.85)
    canopy_wall_gap = 0.015
    tangent_limit = max(0.0, tangent_span / 2.0 - canopy_width / 2.0 - 0.25)
    tangent_offset = rng.uniform(-min(tangent_limit, 0.75), min(tangent_limit, 0.75))
    if entrance_side in {"south", "north"}:
        facade_center = (
            host_center_x + tangent_offset,
            host_center_y + normal[1] * host_size_y / 2.0,
        )
        canopy_center = (
            facade_center[0],
            facade_center[1] + normal[1] * (canopy_depth / 2.0 + canopy_wall_gap),
            canopy_z,
        )
        canopy_size = (canopy_width, canopy_depth, canopy_height)
    else:
        facade_center = (
            host_center_x + normal[0] * host_size_x / 2.0,
            host_center_y + tangent_offset,
        )
        canopy_center = (
            facade_center[0] + normal[0] * (canopy_depth / 2.0 + canopy_wall_gap),
            facade_center[1],
            canopy_z,
        )
        canopy_size = (canopy_depth, canopy_width, canopy_height)
    details.append(
        {
            "id": "entrance_canopy",
            "center": [round(value, 4) for value in canopy_center],
            "size": [round(value, 4) for value in canopy_size],
            "target_support": False,
            "structural_role": "entrance_canopy",
            "host_component_id": str(host_component["id"]),
            "attachment_side": entrance_side,
            "display_color": list(CANOPY_COLOR),
        }
    )
    entrance = [
        round(facade_center[0] + normal[0] * 0.2, 4),
        round(facade_center[1] + normal[1] * 0.2, 4),
        1.4,
    ]
    return details, entrance


def _roof_equipment_component(
    components: list[dict[str, Any]], rng: random.Random
) -> dict[str, Any] | None:
    """Place equipment on an actual highest roof instead of a footprint envelope."""

    if rng.random() >= 0.72:
        return None
    highest_roof = max(_component_roof_z(component) for component in components)
    candidates = [
        component
        for component in components
        if abs(_component_roof_z(component) - highest_roof) <= 1.0e-6
    ]
    host = rng.choice(sorted(candidates, key=lambda component: str(component["id"])))
    center_x, center_y, _ = (float(value) for value in host["center"])
    size_x, size_y, _ = (float(value) for value in host["size"])
    equipment_height = rng.uniform(1.3, 3.2)
    equipment_width = min(3.2, max(0.7, size_x * rng.uniform(0.14, 0.26)), size_x - 0.4)
    equipment_depth = min(3.2, max(0.7, size_y * rng.uniform(0.14, 0.26)), size_y - 0.4)
    equipment_roof_gap = 0.015
    # Keep the equipment clear of the bounded roof parapets. The detail
    # geometry uses a 0.28 m maximum wall thickness, so 0.50 m guarantees
    # non-overlapping colliders after CitySpec's four-decimal serialization.
    x_room = max(0.0, (size_x - equipment_width) / 2.0 - 0.50)
    y_room = max(0.0, (size_y - equipment_depth) / 2.0 - 0.50)
    return {
        "id": "roof_equipment",
        "center": [
            round(center_x + rng.uniform(-x_room, x_room), 4),
            round(center_y + rng.uniform(-y_room, y_room), 4),
            round(_component_roof_z(host) + equipment_roof_gap + equipment_height / 2.0, 4),
        ],
        "size": [
            round(equipment_width, 4),
            round(equipment_depth, 4),
            round(equipment_height, 4),
        ],
        "target_support": False,
        "structural_role": "roof_equipment",
        "host_component_id": str(host["id"]),
        "display_color": list(ROOF_EQUIPMENT_COLOR),
    }


def _overlaps(candidate: AABB, existing: list[AABB], margin: float) -> bool:
    expanded = candidate.expanded(margin)
    return any(
        all(
            low_a <= high_b and high_a >= low_b
            for low_a, high_a, low_b, high_b in zip(
                expanded.minimum,
                expanded.maximum,
                other.minimum,
                other.maximum,
                strict=True,
            )
        )
        for other in existing
    )


def _sample_height(rng: random.Random, size_m: int) -> float:
    band = rng.choices(("low", "mid", "high"), weights=(0.36, 0.39, 0.25), k=1)[0]
    if band == "low":
        return rng.uniform(8.0, 15.0)
    if band == "mid":
        return rng.uniform(18.0, 31.0)
    return rng.uniform(34.0, 52.0 if size_m >= 96 else 46.0)


def _generate_buildings(
    size_m: int,
    roads: list[dict[str, Any]],
    templates: tuple[str, ...],
    admission: dict[str, Any],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], float]:
    target_low, target_high = (float(value) for value in admission["built_ratio"])
    target_ratio = rng.uniform(target_low + 0.025, min(target_high - 0.025, target_low + 0.13))
    occupied: list[AABB] = []
    buildings: list[dict[str, Any]] = []
    built_area = 0.0
    half = size_m / 2.0
    attempts = 0
    while built_area / (size_m * size_m) < target_ratio and attempts < 3000:
        attempts += 1
        width = rng.uniform(8.0, min(17.0, size_m * 0.19))
        depth = rng.uniform(8.0, min(17.0, size_m * 0.19))
        x = rng.uniform(-half + width / 2.0 + 2.0, half - width / 2.0 - 2.0)
        y = rng.uniform(-half + depth / 2.0 + 2.0, half - depth / 2.0 - 2.0)
        diagonal = math.hypot(width, depth) / 2.0
        if any(
            _point_segment_distance((x, y), road) < float(road["width_m"]) / 2.0 + diagonal + 1.2
            for road in roads
        ):
            continue
        footprint_box = AABB.from_center_size(
            "candidate", (x, y, 8.0), (width, depth, 16.0), "building"
        )
        if _overlaps(footprint_box, occupied, margin=2.5):
            continue
        template = rng.choice(templates)
        if template in {"slab", "narrow_slab"}:
            if rng.random() < 0.5:
                depth *= 0.72
            else:
                width *= 0.72
        height = _sample_height(rng, size_m)
        building_id = f"building-{content_hash([x, y, width, depth, height])[:10]}"
        components = _components(template, x, y, width, depth, height)
        side = rng.choice(("south", "east", "north", "west"))
        # Keep visual/architectural sampling separate from the core city stream:
        # visual richness must not alter street topology, building placement,
        # obstacle sampling, target processes, or difficulty admission.
        detail_rng = random.Random(derived_seed(building_id, "architectural-detail-v1"))
        structural_components = list(components)
        architectural_details, entrance = _architectural_components(
            structural_components,
            entrance_side=side,
            rng=detail_rng,
        )
        components.extend(architectural_details)
        equipment = _roof_equipment_component(structural_components, detail_rng)
        if equipment is not None:
            components.append(equipment)
        buildings.append(
            {
                "id": building_id,
                "template": template,
                "footprint": [round(x, 4), round(y, 4), round(width, 4), round(depth, 4)],
                "height_m": round(height, 4),
                "components": components,
                "entrances": [[round(value, 4) for value in entrance]],
                "display_color": list(rng.choice(COLORS)),
            }
        )
        occupied.append(
            AABB.from_center_size(
                building_id, (x, y, height / 2.0), (width, depth, height), "building"
            )
        )
        built_area += width * depth
    built_ratio = built_area / (size_m * size_m)
    if built_ratio < target_low:
        raise GenerationRejected(f"could not reach minimum built ratio: {built_ratio:.4f}")
    heights = sorted(float(item["height_m"]) for item in buildings)
    minimum_span = float(admission["minimum_vertical_span_m"])
    if not heights or heights[-1] - heights[0] < minimum_span:
        raise GenerationRejected("building heights do not create enough vertical search span")
    return buildings, built_ratio


def _component_boxes(buildings: list[dict[str, Any]]) -> list[AABB]:
    return [
        AABB.from_center_size(
            f"{building['id']}/{component['id']}",
            component["center"],
            component["size"],
            "building",
        )
        for building in buildings
        for component in building["components"]
    ]


def _generate_obstacles(
    buildings: list[dict[str, Any]], size_m: int, rng: random.Random
) -> list[dict[str, Any]]:
    colliders = _component_boxes(buildings)
    obstacles: list[dict[str, Any]] = []
    desired = max(6, round(len(buildings) * 0.75))
    shuffled = list(buildings)
    rng.shuffle(shuffled)
    for building in shuffled:
        if len(obstacles) >= desired:
            break
        x, y, width, depth = (float(value) for value in building["footprint"])
        sides = list(("south", "east", "north", "west"))
        rng.shuffle(sides)
        for side in sides:
            sx, sy, sz = rng.uniform(1.2, 3.2), rng.uniform(1.2, 3.2), rng.uniform(0.7, 2.1)
            gap = rng.uniform(0.7, 1.8)
            tangent = rng.uniform(-0.25, 0.25)
            if side == "south":
                center = (x + tangent * width, y - depth / 2.0 - gap - sy / 2.0, sz / 2.0)
            elif side == "north":
                center = (x + tangent * width, y + depth / 2.0 + gap + sy / 2.0, sz / 2.0)
            elif side == "west":
                center = (x - width / 2.0 - gap - sx / 2.0, y + tangent * depth, sz / 2.0)
            else:
                center = (x + width / 2.0 + gap + sx / 2.0, y + tangent * depth, sz / 2.0)
            if max(abs(center[0]), abs(center[1])) >= size_m / 2.0 - 1.0:
                continue
            candidate = AABB.from_center_size("candidate", center, (sx, sy, sz), "rubble")
            if _overlaps(candidate, colliders, margin=0.15):
                continue
            obstacle_id = f"obstacle-{content_hash([building['id'], center])[:10]}"
            obstacles.append(
                {
                    "id": obstacle_id,
                    "kind": rng.choice(("rubble", "barrier", "utility_debris")),
                    "center": [round(value, 4) for value in center],
                    "size": [round(sx, 4), round(sy, 4), round(sz, 4)],
                    "semantic_anchor": building["id"],
                    "support_domain": True,
                }
            )
            colliders.append(AABB.from_center_size(obstacle_id, center, (sx, sy, sz), "obstacle"))
            break
    if len(obstacles) < max(4, desired // 2):
        raise GenerationRejected("semantic obstacle placement produced too few legal obstacles")
    return obstacles


def _road_sample(road: dict[str, Any], factor: float, lateral: float) -> tuple[float, float]:
    ax, ay = (float(value) for value in road["start"])
    bx, by = (float(value) for value in road["end"])
    dx, dy = bx - ax, by - ay
    length = max(math.hypot(dx, dy), 1.0e-9)
    return (
        ax + factor * dx - lateral * dy / length,
        ay + factor * dy + lateral * dx / length,
    )


def _generate_decorations(
    asset_ids: list[str],
    roads: list[dict[str, Any]],
    buildings: list[dict[str, Any]],
    size_m: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    building_boxes = _component_boxes(buildings)
    decorations: list[dict[str, Any]] = []
    for asset_id in asset_ids:
        copies = rng.randint(1, 3)
        for copy_index in range(copies):
            for _ in range(40):
                road = rng.choice(roads)
                lateral = float(road["width_m"]) / 2.0 + rng.uniform(0.8, 2.2)
                lateral *= rng.choice((-1.0, 1.0))
                x, y = _road_sample(road, rng.uniform(0.08, 0.92), lateral)
                if max(abs(x), abs(y)) >= size_m / 2.0 - 0.6:
                    continue
                if any(box.point_distance((x, y, 0.8)) < 0.7 for box in building_boxes):
                    continue
                decorations.append(
                    {
                        "id": f"decor-{content_hash([asset_id, copy_index, x, y])[:10]}",
                        "asset_id": asset_id,
                        "position": [round(x, 4), round(y, 4), 0.0],
                        "rotation_z_deg": round(rng.uniform(-180.0, 180.0), 3),
                        "scale": round(rng.uniform(0.78, 1.16), 4),
                        "semantic_zone": "street_setback",
                        "physics_role": "visual_only",
                    }
                )
                break
    return decorations


def _generate_visual_facade_accents(
    buildings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add bounded render-only facade variation without changing task geometry.

    These thin panels make vertical scale and exterior faces legible in L2
    review.  They deliberately have no collider, target-support, or atlas
    role: a geometry-search method cannot receive a task advantage from a
    color or a panel placement.
    """

    accents: list[dict[str, Any]] = []
    for building in buildings:
        building_id = str(building["id"])
        rng = random.Random(derived_seed(building_id, VISUAL_DETAIL_PROFILE))
        structural = [
            component
            for component in building["components"]
            if component.get("target_support", True)
        ]
        candidates: list[tuple[str, dict[str, Any]]] = []
        for component_index, component in enumerate(structural):
            for side in ("south", "east", "north", "west"):
                if _face_is_externally_clear(structural, component_index, side):
                    candidates.append((side, component))
        rng.shuffle(candidates)
        for accent_index, (side, component) in enumerate(
            candidates[:MAX_VISUAL_FACADE_ACCENTS_PER_BUILDING]
        ):
            center_x, center_y, center_z = (
                float(value) for value in component["center"]
            )
            size_x, size_y, size_z = (float(value) for value in component["size"])
            tangent = size_x if side in {"south", "north"} else size_y
            panel_width = min(4.8, max(1.0, tangent * rng.uniform(0.28, 0.48)))
            panel_height = min(8.0, max(1.4, size_z * rng.uniform(0.16, 0.28)))
            panel_z = center_z + size_z * rng.uniform(-0.16, 0.16)
            outward = 0.026
            if side == "south":
                center = [center_x, center_y - size_y / 2.0 - outward, panel_z]
                size = [panel_width, 0.035, panel_height]
            elif side == "north":
                center = [center_x, center_y + size_y / 2.0 + outward, panel_z]
                size = [panel_width, 0.035, panel_height]
            elif side == "west":
                center = [center_x - size_x / 2.0 - outward, center_y, panel_z]
                size = [0.035, panel_width, panel_height]
            else:
                center = [center_x + size_x / 2.0 + outward, center_y, panel_z]
                size = [0.035, panel_width, panel_height]
            accents.append(
                {
                    "id": f"facade-{building_id.removeprefix('building-')}-{accent_index}",
                    "building_id": building_id,
                    "component_id": str(component["id"]),
                    "side": side,
                    "center": [round(value, 4) for value in center],
                    "size": [round(value, 4) for value in size],
                    "display_color": list(rng.choice(FACADE_ACCENT_COLORS)),
                    "physics_role": "visual_only",
                }
            )
    return accents


def _road_ratio(roads: list[dict[str, Any]], size_m: int) -> float:
    # Raster integration handles overlap and non-axis-aligned roads deterministically.
    resolution = 1.0
    cells = int(size_m / resolution)
    covered = 0
    for ix in range(cells):
        x = -size_m / 2.0 + (ix + 0.5) * resolution
        for iy in range(cells):
            y = -size_m / 2.0 + (iy + 0.5) * resolution
            if any(
                _point_segment_distance((x, y), road) <= float(road["width_m"]) / 2.0
                for road in roads
            ):
                covered += 1
    return covered / max(1, cells * cells)


def _task_geometry_payload(
    *,
    generator_version: str,
    size_m: int,
    graph_payload: dict[str, Any],
    roads: list[dict[str, Any]],
    buildings: list[dict[str, Any]],
    obstacles: list[dict[str, Any]],
    flight_bounds: dict[str, list[float]],
) -> dict[str, Any]:
    """Return visual-independent geometry that determines task semantics.

    The full layout hash remains the identity for scene bytes and replay. This
    payload is used for target, spawn, and coarse-prior randomness, so changing
    colors, road styles, or visual-only assets cannot alter task difficulty.
    """

    task_roads = [
        {key: value for key, value in road.items() if key != "surface_style"}
        for road in roads
    ]
    task_buildings = [
        {
            "id": building["id"],
            "template": building["template"],
            "footprint": building["footprint"],
            "height_m": building["height_m"],
            "entrances": building["entrances"],
            "components": [
                {
                    "id": component["id"],
                    "center": component["center"],
                    "size": component["size"],
                    "target_support": component.get("target_support", True),
                }
                for component in building["components"]
            ],
        }
        for building in buildings
    ]
    return {
        "generator_version": generator_version,
        "size_m": size_m,
        "road_graph": graph_payload,
        "roads": task_roads,
        "buildings": task_buildings,
        "obstacles": obstacles,
        "flight_bounds": flight_bounds,
    }


def generate_city_v3(
    config: OrdinaryReleaseConfig,
    split: str,
    index: int,
    attempt: int,
    asset_ids: list[str],
) -> dict[str, Any]:
    seed = derived_seed(config.master_seed, config.generator_version, split, index, attempt)
    rng = random.Random(seed)
    size_m = (
        int(config.raw["topology_ood_size_m"])
        if split == "test_topology"
        else int(rng.choice(config.raw["core_sizes_m"]))
    )
    families = config.raw["city_grammar"][
        "topology_ood_families" if split == "test_topology" else "development_families"
    ]
    family = str(rng.choice(families))
    width_low, width_high = (float(value) for value in config.raw["admission"]["street_width_m"])
    road_width = rng.uniform(width_low, width_high)
    nodes, roads = _road_graph(family, size_m, road_width, rng)
    style_rng = random.Random(derived_seed(seed, "road-surface-style-v1"))
    # The style is public, layout-hashed geometry metadata, but it must not
    # perturb the core RNG stream used to sample city structure.
    for road in roads:
        road["surface_style"] = style_rng.choice(ROAD_SURFACE_STYLES)
    templates = OOD_TEMPLATES if split == "test_topology" else DEVELOPMENT_TEMPLATES
    buildings, built_ratio = _generate_buildings(
        size_m, roads, templates, config.raw["admission"], rng
    )
    obstacles = _generate_obstacles(buildings, size_m, rng)
    decoration_rng = random.Random(derived_seed(seed, "visual-decoration-v1", sorted(asset_ids)))
    decorations = _generate_decorations(asset_ids, roads, buildings, size_m, decoration_rng)
    visual_facade_accents = _generate_visual_facade_accents(buildings)
    road_ratio = _road_ratio(roads, size_m)
    road_low, road_high = (float(value) for value in config.raw["admission"]["road_ratio"])
    if not road_low <= road_ratio <= road_high:
        raise GenerationRejected(f"road_ratio={road_ratio:.4f} outside admission range")

    graph_payload = {
        "nodes": [[round(value, 2) for value in node["position"][:2]] for node in nodes],
        "edges": sorted((road["start_node"], road["end_node"], road["blocked"]) for road in roads),
    }
    asset_set_hash = content_hash(sorted(asset_ids))
    height_values = [float(item["height_m"]) for item in buildings]
    flight_bounds = {
        "minimum": [-size_m / 2.0, -size_m / 2.0, 1.0],
        "maximum": [
            size_m / 2.0,
            size_m / 2.0,
            max(70.0, max(height_values) + 12.0),
        ],
    }
    task_geometry_hash = content_hash(
        _task_geometry_payload(
            generator_version=config.generator_version,
            size_m=size_m,
            graph_payload=graph_payload,
            roads=roads,
            buildings=buildings,
            obstacles=obstacles,
            flight_bounds=flight_bounds,
        )
    )
    geometry = {
        "generator_version": config.generator_version,
        "size_m": size_m,
        "road_graph": graph_payload,
        "roads": roads,
        "buildings": buildings,
        "obstacles": obstacles,
        "decorations": decorations,
        "visual_detail_profile": VISUAL_DETAIL_PROFILE,
        "visual_facade_accents": visual_facade_accents,
        "asset_set_hash": asset_set_hash,
    }
    layout_hash = content_hash(geometry)
    template_counts = Counter(str(item["template"]) for item in buildings)
    spawn_rng = random.Random(derived_seed(seed, "spawn-grammar-v1"))
    return {
        "schema": "org.aerocity.bench.cityspec-internal.ordinary.v3",
        "generator_version": config.generator_version,
        "layout_id": f"city-{layout_hash[:16]}",
        "layout_hash": layout_hash,
        "task_geometry_hash": task_geometry_hash,
        "generation_seed": seed,
        "split": split,
        "family_private": family,
        "size_m": size_m,
        "road_graph": {"nodes": nodes, "edges": roads},
        "roads": roads,
        "blocks": [],
        "buildings": buildings,
        "obstacles": obstacles,
        "decorations": decorations,
        "visual_detail_profile": VISUAL_DETAIL_PROFILE,
        "visual_facade_accents": visual_facade_accents,
        "flight_bounds": flight_bounds,
        "topology_signature": content_hash(graph_payload),
        "asset_set_hash": asset_set_hash,
        "spawn_grammar": str(spawn_rng.choice(config.raw["city_grammar"]["spawn_grammars"])),
        "metrics": {
            "built_ratio": round(built_ratio, 6),
            "road_ratio": round(road_ratio, 6),
            "road_node_count": len(nodes),
            "road_edge_count": len(roads),
            "building_count": len(buildings),
            "obstacle_count": len(obstacles),
            "architectural_detail_count": sum(
                1
                for building in buildings
                for component in building["components"]
                if component.get("structural_role") in {
                    "roof_parapet",
                    "entrance_canopy",
                    "roof_equipment",
                }
            ),
            "visual_facade_accent_count": len(visual_facade_accents),
            "road_surface_histogram": dict(
                sorted(Counter(str(road["surface_style"]) for road in roads).items())
            ),
            "height_min_m": round(min(height_values), 4),
            "height_max_m": round(max(height_values), 4),
            "template_histogram": dict(sorted(template_counts.items())),
        },
    }
