"""Deterministic constrained city grammar."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from .canonical import content_hash, derived_seed
from .config import ReleaseConfig
from .errors import GenerationRejected

TRAINING_FAMILIES = ("grid_corridor", "mixed_blocks", "industrial_spine")
CORE_TEMPLATES = ("rectangle", "l_shape", "slab", "tower_podium")
HOLDOUT_TEMPLATES = ("courtyard", "u_shape", "stepped_tower", "narrow_slab")
COLORS = (
    [0.55, 0.58, 0.61],
    [0.69, 0.55, 0.44],
    [0.42, 0.52, 0.57],
    [0.63, 0.65, 0.56],
    [0.52, 0.46, 0.43],
)


def _axis_count(size_m: int, family: str, axis: str) -> int:
    if size_m <= 64:
        return 1
    base = max(1, round(size_m / 100))
    if family == "industrial_spine":
        return 1 if axis == "x" else max(2, base)
    if family == "topology_holdout" and axis == "x":
        return max(2, base)
    return base


def _centres(count: int, size_m: int, rng: random.Random, jitter: float) -> list[float]:
    if count == 1:
        return [rng.uniform(-jitter, jitter)]
    spacing = size_m / (count + 1)
    return [
        -size_m / 2 + spacing * (index + 1) + rng.uniform(-jitter, jitter) for index in range(count)
    ]


def _intervals(size_m: int, roads: list[dict[str, Any]], axis: str) -> list[tuple[float, float]]:
    lower = -size_m / 2
    upper = size_m / 2
    cuts = sorted(
        (
            float(road[axis]) - float(road["width_m"]) / 2,
            float(road[axis]) + float(road["width_m"]) / 2,
        )
        for road in roads
        if road["axis"] == axis
    )
    result: list[tuple[float, float]] = []
    cursor = lower
    for start, end in cuts:
        if start - cursor >= 12:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if upper - cursor >= 12:
        result.append((cursor, upper))
    return result


def _components(
    template: str, x: float, y: float, w: float, d: float, h: float
) -> list[dict[str, Any]]:
    def box(
        name: str, cx: float, cy: float, cz: float, sx: float, sy: float, sz: float
    ) -> dict[str, Any]:
        return {
            "id": name,
            "center": [round(cx, 4), round(cy, 4), round(cz, 4)],
            "size": [round(sx, 4), round(sy, 4), round(sz, 4)],
        }

    if template in {"rectangle", "slab", "narrow_slab"}:
        return [box("base", x, y, h / 2, w, d, h)]
    if template == "l_shape":
        t = min(w, d) * 0.32
        return [
            box("west", x - (w - t) / 2, y, h / 2, t, d, h),
            box("south", x + t / 2, y - (d - t) / 2, h / 2, w - t, t, h),
        ]
    if template == "courtyard":
        t = min(w, d) * 0.22
        return [
            box("north", x, y + (d - t) / 2, h / 2, w, t, h),
            box("south", x, y - (d - t) / 2, h / 2, w, t, h),
            box("west", x - (w - t) / 2, y, h / 2, t, d - 2 * t, h),
            box("east", x + (w - t) / 2, y, h / 2, t, d - 2 * t, h),
        ]
    if template == "u_shape":
        t = min(w, d) * 0.24
        return [
            box("north", x, y + (d - t) / 2, h / 2, w, t, h),
            box("west", x - (w - t) / 2, y - t / 2, h / 2, t, d - t, h),
            box("east", x + (w - t) / 2, y - t / 2, h / 2, t, d - t, h),
        ]
    if template == "tower_podium":
        podium_h = max(5.0, h * 0.22)
        return [
            box("podium", x, y, podium_h / 2, w, d, podium_h),
            box("tower", x, y, podium_h + (h - podium_h) / 2, w * 0.58, d * 0.58, h - podium_h),
        ]
    if template == "stepped_tower":
        low = h * 0.30
        mid = h * 0.32
        return [
            box("base", x, y, low / 2, w, d, low),
            box("middle", x, y, low + mid / 2, w * 0.72, d * 0.72, mid),
            box("top", x, y, low + mid + (h - low - mid) / 2, w * 0.45, d * 0.45, h - low - mid),
        ]
    raise GenerationRejected(f"unknown building template: {template}")


def _height(index: int, size_m: int, rng: random.Random) -> float:
    band = index % 3
    if band == 0:
        return rng.uniform(8.0, 14.0)
    if band == 1:
        return rng.uniform(18.0, 30.0)
    high_floor = 40.0 if size_m >= 160 else 32.0
    high_ceiling = 58.0 if size_m >= 160 else 48.0
    return rng.uniform(high_floor, high_ceiling)


def _footprint_ratio(template: str) -> float:
    return {
        "rectangle": 1.0,
        "slab": 1.0,
        "narrow_slab": 1.0,
        "l_shape": 0.54,
        "courtyard": 0.61,
        "u_shape": 0.56,
        "tower_podium": 1.0,
        "stepped_tower": 1.0,
    }[template]


def generate_city(
    config: ReleaseConfig,
    split: str,
    index: int,
    attempt: int,
    asset_ids: list[str],
) -> dict[str, Any]:
    seed = derived_seed(config.master_seed, config.generator_version, split, index, attempt)
    rng = random.Random(seed)
    sizes = [int(value) for value in config.raw["core_sizes_m"]]
    size_m = (
        int(config.raw["scale_ood_size_m"]) if split == "test_scale" else sizes[index % len(sizes)]
    )
    family = (
        str(config.raw["topology_holdout_family"])
        if split == "test_topology"
        else str(
            config.raw["training_families"][(index + seed) % len(config.raw["training_families"])]
        )
    )
    width_min, width_max = [float(value) for value in config.raw["admission"]["street_width_m"]]
    if size_m <= 64:
        width_max = min(width_max, size_m * 0.18)
        width_min = min(width_min, width_max)
    road_width = rng.uniform(width_min, width_max)
    roads: list[dict[str, Any]] = []
    for axis in ("x", "y"):
        count = _axis_count(size_m, family, axis)
        centres = _centres(count, size_m, rng, min(road_width * 0.22, size_m * 0.025))
        for road_index, centre in enumerate(centres):
            roads.append(
                {
                    "id": f"road-{axis}-{road_index:02d}",
                    "axis": axis,
                    axis: round(centre, 4),
                    "width_m": round(road_width, 4),
                    "class": "arterial" if road_index == 0 else "local",
                }
            )
    x_intervals = _intervals(size_m, roads, "x")
    y_intervals = _intervals(size_m, roads, "y")
    blocks: list[dict[str, Any]] = []
    for ix, (x0, x1) in enumerate(x_intervals):
        for iy, (y0, y1) in enumerate(y_intervals):
            blocks.append(
                {
                    "id": f"block-{ix:02d}-{iy:02d}",
                    "bounds": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
                }
            )
    if len(blocks) < 2:
        raise GenerationRejected("road grammar produced fewer than two legal blocks")

    templates = HOLDOUT_TEMPLATES if family == "topology_holdout" else CORE_TEMPLATES
    buildings: list[dict[str, Any]] = []
    built_area = 0.0
    building_index = 0
    for block_index, block in enumerate(blocks):
        x0, y0, x1, y1 = [float(value) for value in block["bounds"]]
        block_w, block_d = x1 - x0, y1 - y0
        parcels = [(x0, y0, x1, y1)]
        split_allowed = (
            size_m >= 100
            and max(block_w, block_d) >= 34.0
            and min(block_w, block_d) >= 18.0
            and not (family == "industrial_spine" and block_index % 2 == 0)
        )
        if split_allowed:
            gap = rng.uniform(4.0, 6.0)
            if block_w >= block_d and (block_w - gap) / 2 >= 14.0:
                middle = (x0 + x1) / 2
                parcels = [(x0, y0, middle - gap / 2, y1), (middle + gap / 2, y0, x1, y1)]
            elif (block_d - gap) / 2 >= 14.0:
                middle = (y0 + y1) / 2
                parcels = [(x0, y0, x1, middle - gap / 2), (x0, middle + gap / 2, x1, y1)]
        for parcel_x0, parcel_y0, parcel_x1, parcel_y1 in parcels:
            parcel_w, parcel_d = parcel_x1 - parcel_x0, parcel_y1 - parcel_y0
            template = templates[(building_index + rng.randrange(len(templates))) % len(templates)]
            scale = rng.uniform(0.76, 0.86)
            w = parcel_w * scale
            d = parcel_d * scale
            if template in {"slab", "narrow_slab"}:
                if parcel_w >= parcel_d:
                    d *= 0.82 if template == "slab" else 0.68
                else:
                    w *= 0.82 if template == "slab" else 0.68
            x = (parcel_x0 + parcel_x1) / 2 + rng.uniform(-parcel_w * 0.02, parcel_w * 0.02)
            y = (parcel_y0 + parcel_y1) / 2 + rng.uniform(-parcel_d * 0.02, parcel_d * 0.02)
            h = _height(building_index, size_m, rng)
            building_id = f"building-{building_index:03d}"
            components = _components(template, x, y, w, d, h)
            if building_index % 2 == 1:
                roof = components[0]
                roof_x, roof_y, roof_z = [float(value) for value in roof["center"]]
                roof_w, roof_d, roof_h = [float(value) for value in roof["size"]]
                equipment_h = rng.uniform(1.8, 3.4)
                components.append(
                    {
                        "id": "roof_equipment",
                        "center": [
                            round(roof_x + roof_w * 0.18, 4),
                            round(roof_y - roof_d * 0.16, 4),
                            round(roof_z + roof_h / 2 + equipment_h / 2, 4),
                        ],
                        "size": [
                            round(max(1.5, roof_w * 0.18), 4),
                            round(max(1.5, roof_d * 0.18), 4),
                            round(equipment_h, 4),
                        ],
                    }
                )
            entrance_side = (building_index + rng.randrange(4)) % 4
            entrances = [
                [x, y - d / 2 - 0.25, 1.5],
                [x + w / 2 + 0.25, y, 1.5],
                [x, y + d / 2 + 0.25, 1.5],
                [x - w / 2 - 0.25, y, 1.5],
            ]
            buildings.append(
                {
                    "id": building_id,
                    "block_id": block["id"],
                    "template": template,
                    "footprint": [round(x, 4), round(y, 4), round(w, 4), round(d, 4)],
                    "height_m": round(h, 4),
                    "components": components,
                    "entrances": [[round(value, 4) for value in entrances[entrance_side]]],
                    "display_color": COLORS[(building_index + int(seed)) % len(COLORS)],
                }
            )
            built_area += w * d * _footprint_ratio(template)
            building_index += 1

    obstacle_count = max(4, len(blocks) // 2)
    obstacles: list[dict[str, Any]] = []
    for obstacle_index in range(obstacle_count):
        block = blocks[obstacle_index % len(blocks)]
        x0, y0, x1, y1 = [float(value) for value in block["bounds"]]
        corner = obstacle_index % 4
        x = x0 + 1.8 if corner in {0, 2} else x1 - 1.8
        y = y0 + 1.8 if corner in {0, 1} else y1 - 1.8
        x += rng.uniform(-0.6, 0.6)
        y += rng.uniform(-0.6, 0.6)
        sx, sy, sz = rng.uniform(1.2, 3.0), rng.uniform(1.2, 3.0), rng.uniform(0.8, 2.2)
        obstacles.append(
            {
                "id": f"rubble-{obstacle_index:03d}",
                "kind": "rubble",
                "center": [round(x, 4), round(y, 4), round(sz / 2, 4)],
                "size": [round(sx, 4), round(sy, 4), round(sz, 4)],
                "support_domain": True,
            }
        )

    decorations: list[dict[str, Any]] = []
    for asset_index, asset_id in enumerate(asset_ids):
        for copy_index in range(2):
            theta = 2 * math.pi * (asset_index * 2 + copy_index) / max(1, len(asset_ids) * 2)
            radius = size_m * (0.27 + 0.04 * copy_index)
            decorations.append(
                {
                    "id": f"decor-{asset_index:02d}-{copy_index:02d}",
                    "asset_id": asset_id,
                    "position": [
                        round(radius * math.cos(theta), 4),
                        round(radius * math.sin(theta), 4),
                        0.0,
                    ],
                    "rotation_z_deg": round(math.degrees(theta) + 90.0, 3),
                    "scale": round(rng.uniform(0.75, 1.2), 4),
                    "physics_role": "visual_only_pending_native_audit",
                }
            )

    vertical_width = sum(float(road["width_m"]) for road in roads if road["axis"] == "x")
    horizontal_width = sum(float(road["width_m"]) for road in roads if road["axis"] == "y")
    road_area = (
        vertical_width * size_m + horizontal_width * size_m - vertical_width * horizontal_width
    )
    road_ratio = road_area / (size_m * size_m)
    built_ratio = built_area / (size_m * size_m)
    admission = config.raw["admission"]
    if not float(admission["road_ratio"][0]) <= road_ratio <= float(admission["road_ratio"][1]):
        raise GenerationRejected(f"road_ratio={road_ratio:.4f} outside admission range")
    if not float(admission["built_ratio"][0]) <= built_ratio <= float(admission["built_ratio"][1]):
        raise GenerationRejected(f"built_ratio={built_ratio:.4f} outside admission range")

    template_counts = Counter(item["template"] for item in buildings)
    topology_payload = {
        "family": family,
        "axis_counts": [
            sum(road["axis"] == "x" for road in roads),
            sum(road["axis"] == "y" for road in roads),
        ],
        "templates": [item["template"] for item in buildings],
        "block_aspects": [
            round(
                (item["bounds"][2] - item["bounds"][0]) / (item["bounds"][3] - item["bounds"][1]), 1
            )
            for item in blocks
        ],
        "road_offsets": [round(float(road[road["axis"]]) / size_m, 2) for road in roads],
    }
    asset_set_hash = content_hash(sorted(asset_ids))
    geometry = {
        "generator_version": config.generator_version,
        "family": family,
        "size_m": size_m,
        "seed": seed,
        "roads": roads,
        "blocks": blocks,
        "buildings": buildings,
        "obstacles": obstacles,
        "decorations": decorations,
        "flight_bounds": {
            "minimum": [-size_m / 2, -size_m / 2, 1.0],
            "maximum": [
                size_m / 2,
                size_m / 2,
                max(90.0, max(item["height_m"] for item in buildings) + 15.0),
            ],
        },
        "asset_set_hash": asset_set_hash,
    }
    layout_hash = content_hash(geometry)
    return {
        "schema": "org.aerocity.bench.cityspec-internal.v2",
        "generator_version": config.generator_version,
        "layout_id": f"city-{layout_hash[:16]}",
        "layout_hash": layout_hash,
        "generation_seed": seed,
        "split": split,
        "family": family,
        "size_m": size_m,
        "roads": roads,
        "blocks": blocks,
        "buildings": buildings,
        "obstacles": obstacles,
        "decorations": decorations,
        "flight_bounds": geometry["flight_bounds"],
        "topology_signature": content_hash(topology_payload),
        "asset_set_hash": asset_set_hash,
        "support_site_rules": {
            "roof_spacing_m": 4.0,
            "opening_spacing_m": 4.0,
            "opening_altitudes_m": [4.0, 8.0, 16.0, 24.0, 36.0, 48.0],
        },
        "metrics": {
            "built_ratio": round(built_ratio, 6),
            "road_ratio": round(road_ratio, 6),
            "block_count": len(blocks),
            "building_count": len(buildings),
            "height_min_m": round(min(item["height_m"] for item in buildings), 4),
            "height_max_m": round(max(item["height_m"] for item in buildings), 4),
            "template_histogram": dict(sorted(template_counts.items())),
        },
    }
