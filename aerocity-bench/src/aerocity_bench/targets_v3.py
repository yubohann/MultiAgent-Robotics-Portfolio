"""Three-dimensional surface processes and immutable target-validity manifests."""

from __future__ import annotations

import copy
import math
import random
import threading
from collections import Counter, OrderedDict
from typing import Any

from .canonical import content_hash, derived_seed
from .errors import GenerationRejected
from .geometry import (
    AABB,
    Vec3,
    colliders_from_city,
    distance,
    line_of_sight,
    minimum_clearance,
    pose_looking_at,
    segment_intersection_fraction,
)
from .inspection_atlas import (
    TASK_TRACK_G2_I,
    compile_public_mission_sector,
    validate_public_mission_sector,
)
from .ordinary_config import OrdinaryReleaseConfig

SUPPORT_CLASSES = ("roof", "facade_marker_site", "entrance", "rubble")
ALTITUDE_BANDS = ("near_ground", "lower", "mid", "elevated", "highrise")
_REACHABILITY_CACHE_LIMIT = 32
_REACHABILITY_CACHE: OrderedDict[str, tuple[dict[str, Any], ...]] = OrderedDict()
_REACHABILITY_CACHE_LOCK = threading.RLock()

# This private lattice is deliberately independent from the public atlas-cell
# density.  The atlas says what public structure must be inspected; target
# sampling later chooses a hidden state on this fixed surface support.  Without
# that separation, a cell-density ablation would silently alter the target
# distribution as well as the public representation.
_G2_I_PRIVATE_SUPPORT_POLICY = {
    "policy_id": "g2-i-private-surface-support-v1",
    "surface_spacing_m": 2.5,
    "facade_inset_m": 0.55,
    "facade_height_inset_m": 0.75,
    "surface_offset_m": 0.12,
}


def _task_geometry_hash(city: dict[str, Any]) -> str:
    """Use visual-independent task geometry when available.

    Older pilot artifacts lack this field, so they keep their historical
    layout-hash behavior rather than becoming unreadable.
    """

    return str(city.get("task_geometry_hash", city["layout_hash"]))


def altitude_band(z_value: float) -> str:
    if z_value < 4.0:
        return "near_ground"
    if z_value < 12.0:
        return "lower"
    if z_value < 25.0:
        return "mid"
    if z_value < 40.0:
        return "elevated"
    return "highrise"


def _bounds(component: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    x, y, z = (float(value) for value in component["center"])
    sx, sy, sz = (float(value) for value in component["size"])
    return x - sx / 2.0, x + sx / 2.0, y - sy / 2.0, y + sy / 2.0, z - sz / 2.0, z + sz / 2.0


def _site(
    support_class: str,
    owner_id: str,
    position: Vec3,
    normal: Vec3,
    represented_area_m2: float,
    lineage: dict[str, str],
) -> dict[str, Any]:
    payload = {
        "support_class": support_class,
        "owner_collider_id": owner_id,
        "position": [round(value, 4) for value in position],
        "normal": [round(value, 6) for value in normal],
        "represented_area_m2": round(max(represented_area_m2, 0.01), 5),
        "surface_lineage": lineage,
    }
    payload["altitude_band"] = altitude_band(position[2])
    payload["site_id"] = f"site-{content_hash(payload)[:18]}"
    return payload


def _stratified_values(
    low: float, high: float, count: int, rng: random.Random, margin: float = 0.3
) -> list[float]:
    if count <= 0 or high <= low:
        return []
    span = high - low
    values = []
    for index in range(count):
        cell_low = low + span * index / count
        cell_high = low + span * (index + 1) / count
        inset = min(margin, (cell_high - cell_low) * 0.2)
        values.append(rng.uniform(cell_low + inset, cell_high - inset))
    rng.shuffle(values)
    return values


def _raw_surface_sites(city: dict[str, Any]) -> list[dict[str, Any]]:
    rng = random.Random(derived_seed(_task_geometry_hash(city), "surface-sites-v3"))
    sites: list[dict[str, Any]] = []
    for building in city["buildings"]:
        building_id = str(building["id"])
        for component in building["components"]:
            if component.get("target_support", True) is not True:
                continue
            owner_id = f"{building_id}/{component['id']}"
            x0, x1, y0, y1, z0, z1 = _bounds(component)
            width, depth, height = x1 - x0, y1 - y0, z1 - z0
            roof_count = max(2, round(width * depth / 30.0))
            roof_x = _stratified_values(x0 + 0.55, x1 - 0.55, roof_count, rng)
            roof_y = _stratified_values(y0 + 0.55, y1 - 0.55, roof_count, rng)
            rng.shuffle(roof_y)
            for px, py in zip(roof_x, roof_y, strict=True):
                sites.append(
                    _site(
                        "roof",
                        owner_id,
                        (px, py, z1 + 0.12),
                        (0.0, 0.0, 1.0),
                        width * depth / roof_count,
                        {
                            "building_id": building_id,
                            "component_id": str(component["id"]),
                            "face": "top",
                        },
                    )
                )

            # Each building gets a jittered floor grammar; no global fixed altitude list exists.
            floor_height = rng.uniform(2.75, 3.75)
            first_level = z0 + rng.uniform(2.0, min(3.2, max(2.05, height - 0.8)))
            levels: list[float] = []
            level = first_level
            while level <= z1 - 0.75:
                levels.append(level + rng.uniform(-0.16, 0.16))
                level += floor_height + rng.uniform(-0.18, 0.18)
            for face, fixed, span_low, span_high, normal, face_area in (
                ("south", y0 - 0.12, x0 + 0.55, x1 - 0.55, (0.0, -1.0, 0.0), width * height),
                ("north", y1 + 0.12, x0 + 0.55, x1 - 0.55, (0.0, 1.0, 0.0), width * height),
                ("west", x0 - 0.12, y0 + 0.55, y1 - 0.55, (-1.0, 0.0, 0.0), depth * height),
                ("east", x1 + 0.12, y0 + 0.55, y1 - 0.55, (1.0, 0.0, 0.0), depth * height),
            ):
                along_count = max(1, round((span_high - span_low) / rng.uniform(3.5, 5.5)))
                along_values = _stratified_values(span_low, span_high, along_count, rng)
                represented = face_area / max(1, len(levels) * along_count)
                for level_value in levels:
                    for along in along_values:
                        position = (
                            (along, fixed, level_value)
                            if face in {"south", "north"}
                            else (fixed, along, level_value)
                        )
                        sites.append(
                            _site(
                                "facade_marker_site",
                                owner_id,
                                position,
                                normal,
                                represented,
                                {
                                    "building_id": building_id,
                                    "component_id": str(component["id"]),
                                    "face": face,
                                },
                            )
                        )
        footprint_x, footprint_y, _, _ = (float(value) for value in building["footprint"])
        for entrance_index, entrance in enumerate(building["entrances"]):
            position = tuple(float(value) for value in entrance)
            delta_x, delta_y = position[0] - footprint_x, position[1] - footprint_y
            if abs(delta_x) > abs(delta_y):
                normal = (math.copysign(1.0, delta_x), 0.0, 0.0)
            else:
                normal = (0.0, math.copysign(1.0, delta_y), 0.0)
            owner_id = min(
                (
                    f"{building_id}/{component['id']}"
                    for component in building["components"]
                    if component.get("target_support", True) is True
                ),
                key=lambda collider_id: collider_id,
            )
            sites.append(
                _site(
                    "entrance",
                    owner_id,
                    position,  # type: ignore[arg-type]
                    normal,
                    3.0,
                    {
                        "building_id": building_id,
                        "component_id": "entrance",
                        "face": f"entrance-{entrance_index}",
                    },
                )
            )
    for obstacle in city["obstacles"]:
        if not obstacle.get("support_domain"):
            continue
        x, y, z = (float(value) for value in obstacle["center"])
        sx, sy, sz = (float(value) for value in obstacle["size"])
        sites.append(
            _site(
                "rubble",
                str(obstacle["id"]),
                (
                    x + rng.uniform(-sx * 0.18, sx * 0.18),
                    y + rng.uniform(-sy * 0.18, sy * 0.18),
                    z + sz / 2.0 + 0.1,
                ),
                (0.0, 0.0, 1.0),
                sx * sy,
                {
                    "building_id": str(obstacle["semantic_anchor"]),
                    "component_id": str(obstacle["id"]),
                    "face": "top",
                },
            )
        )
    unique = {site["site_id"]: site for site in sites}
    return [unique[key] for key in sorted(unique)]


def _fixed_surface_values(low: float, high: float, spacing: float) -> list[float]:
    if high < low:
        return []
    count = max(1, math.ceil((high - low) / spacing))
    return [low + (high - low) * (index + 0.5) / count for index in range(count)]


def _g2_i_private_surface_sites(city: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile a density-independent private support lattice from CitySpec."""

    spacing = float(_G2_I_PRIVATE_SUPPORT_POLICY["surface_spacing_m"])
    side_inset = float(_G2_I_PRIVATE_SUPPORT_POLICY["facade_inset_m"])
    height_inset = float(_G2_I_PRIVATE_SUPPORT_POLICY["facade_height_inset_m"])
    offset = float(_G2_I_PRIVATE_SUPPORT_POLICY["surface_offset_m"])
    sites: list[dict[str, Any]] = []
    for building in sorted(city["buildings"], key=lambda item: str(item["id"])):
        building_id = str(building["id"])
        components = sorted(building["components"], key=lambda item: str(item["id"]))
        for component in components:
            if component.get("target_support", True) is not True:
                continue
            component_id = str(component["id"])
            owner_id = f"{building_id}/{component_id}"
            x0, x1, y0, y1, z0, z1 = _bounds(component)
            roof_x = _fixed_surface_values(x0 + side_inset, x1 - side_inset, spacing)
            roof_y = _fixed_surface_values(y0 + side_inset, y1 - side_inset, spacing)
            roof_count = max(1, len(roof_x) * len(roof_y))
            for x_value in roof_x:
                for y_value in roof_y:
                    sites.append(
                        _site(
                            "roof",
                            owner_id,
                            (x_value, y_value, z1 + offset),
                            (0.0, 0.0, 1.0),
                            (x1 - x0) * (y1 - y0) / roof_count,
                            {
                                "building_id": building_id,
                                "component_id": component_id,
                                "face": "top",
                            },
                        )
                    )
            facade_specs = (
                ("south", (0.0, -1.0, 0.0), "y", y0 - offset, x0, x1, x1 - x0),
                ("north", (0.0, 1.0, 0.0), "y", y1 + offset, x0, x1, x1 - x0),
                ("west", (-1.0, 0.0, 0.0), "x", x0 - offset, y0, y1, y1 - y0),
                ("east", (1.0, 0.0, 0.0), "x", x1 + offset, y0, y1, y1 - y0),
            )
            z_values = _fixed_surface_values(z0 + height_inset, z1 - height_inset, spacing)
            for face, normal, fixed_axis, fixed, along_low, along_high, width in facade_specs:
                along_values = _fixed_surface_values(
                    along_low + side_inset, along_high - side_inset, spacing
                )
                count = max(1, len(z_values) * len(along_values))
                for z_value in z_values:
                    for along in along_values:
                        position = (
                            (fixed, along, z_value)
                            if fixed_axis == "x"
                            else (along, fixed, z_value)
                        )
                        sites.append(
                            _site(
                                "facade_marker_site",
                                owner_id,
                                position,
                                normal,
                                width * (z1 - z0) / count,
                                {
                                    "building_id": building_id,
                                    "component_id": component_id,
                                    "face": face,
                                },
                            )
                        )
        footprint_x, footprint_y, _, _ = (float(value) for value in building["footprint"])
        for entrance_index, entrance in enumerate(building["entrances"]):
            position = tuple(float(value) for value in entrance)
            delta_x, delta_y = position[0] - footprint_x, position[1] - footprint_y
            normal = (
                (math.copysign(1.0, delta_x), 0.0, 0.0)
                if abs(delta_x) > abs(delta_y)
                else (0.0, math.copysign(1.0, delta_y), 0.0)
            )
            owners = [
                f"{building_id}/{component['id']}"
                for component in components
                if component.get("target_support", True) is True
            ]
            if not owners:
                continue
            sites.append(
                _site(
                    "entrance",
                    min(owners),
                    position,  # type: ignore[arg-type]
                    normal,
                    3.0,
                    {
                        "building_id": building_id,
                        "component_id": "entrance",
                        "face": f"entrance-{entrance_index}",
                    },
                )
            )
    for obstacle in sorted(city["obstacles"], key=lambda item: str(item["id"])):
        if not obstacle.get("support_domain"):
            continue
        x, y, z = (float(value) for value in obstacle["center"])
        sx, sy, sz = (float(value) for value in obstacle["size"])
        x_values = _fixed_surface_values(
            x - sx / 2.0 + side_inset, x + sx / 2.0 - side_inset, spacing
        )
        y_values = _fixed_surface_values(
            y - sy / 2.0 + side_inset, y + sy / 2.0 - side_inset, spacing
        )
        count = max(1, len(x_values) * len(y_values))
        for x_value in x_values:
            for y_value in y_values:
                sites.append(
                    _site(
                        "rubble",
                        str(obstacle["id"]),
                        (x_value, y_value, z + sz / 2.0 + offset),
                        (0.0, 0.0, 1.0),
                        sx * sy / count,
                        {
                            "building_id": str(obstacle["semantic_anchor"]),
                            "component_id": str(obstacle["id"]),
                            "face": "top",
                        },
                    )
                )
    return [site for _, site in sorted({site["site_id"]: site for site in sites}.items())]


def _tangent(normal: Vec3) -> Vec3:
    if abs(normal[2]) > 0.8:
        return (1.0, 0.0, 0.0)
    return (-normal[1], normal[0], 0.0)


def _compile_witnesses(
    city: dict[str, Any],
    site: dict[str, Any],
    colliders: list[AABB],
    config: OrdinaryReleaseConfig,
) -> list[dict[str, Any]]:
    execution = config.raw["execution_contract"]
    observe = execution["observe"]
    vehicle = execution["vehicle"]
    bounds = city["flight_bounds"]
    minimum = tuple(float(value) for value in bounds["minimum"])
    maximum = tuple(float(value) for value in bounds["maximum"])
    position = tuple(float(value) for value in site["position"])
    normal = tuple(float(value) for value in site["normal"])
    tangent = _tangent(normal)  # type: ignore[arg-type]
    requested_distance = float(config.raw["admission"]["witness_distance_m"])
    distances = sorted(
        {
            max(1.1, requested_distance - 0.55),
            requested_distance,
            min(float(observe["max_range_m"]) - 0.12, requested_distance + 0.5),
        }
    )
    offsets = (0.0, -0.45, 0.45)
    witnesses: list[dict[str, Any]] = []
    for witness_distance in distances:
        for tangent_offset in offsets:
            witness_position = tuple(
                position[index] + normal[index] * witness_distance + tangent[index] * tangent_offset
                for index in range(3)
            )
            if any(
                value < low or value > high
                for value, low, high in zip(witness_position, minimum, maximum, strict=True)
            ):
                continue
            clearance, _ = minimum_clearance(witness_position, colliders)
            required_clearance = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
            if clearance < required_clearance:
                continue
            visible, _ = line_of_sight(
                witness_position,
                position,
                colliders,
                ignored_ids=frozenset({site["owner_collider_id"]}),
            )
            if not visible:
                continue
            pose = pose_looking_at(witness_position, position)
            witnesses.append(
                {
                    "witness_id": f"witness-{content_hash([site['site_id'], pose.to_dict()])[:16]}",
                    "pose": pose.to_dict(),
                    "target_distance_m": round(distance(witness_position, position), 5),
                    "clearance_m": round(clearance, 5),
                }
            )
    return witnesses


def _context_review_pose(
    city: dict[str, Any],
    site: dict[str, Any],
    config: OrdinaryReleaseConfig,
) -> dict[str, Any] | None:
    """Choose a non-scoring, context-first camera pose for one review target.

    A legal confirmation witness is intentionally close to a target and bounded
    by ``observe.max_range_m``.  Reusing it for L2 screenshots produces frames
    that prove a marker is present but often hide the supporting geometry.  This
    pose is private review metadata only: it is farther away, oblique when
    possible, and is never accepted by the evaluator as an observation witness.
    """

    colliders = colliders_from_city(city)
    target = tuple(float(value) for value in site["position"])
    normal = tuple(float(value) for value in site["normal"])
    tangent = _tangent(normal)  # type: ignore[arg-type]
    bounds = city["flight_bounds"]
    lower = tuple(float(value) for value in bounds["minimum"])
    upper = tuple(float(value) for value in bounds["maximum"])
    review_outside_margin = max(12.0, float(city["size_m"]) * 0.15)
    vehicle = config.raw["execution_contract"]["vehicle"]
    required_clearance = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    owner_id = str(site["owner_collider_id"])
    context_ids = frozenset(str(value) for value in site["surrounding_collider_ids"])

    # The candidate set is deterministic and deliberately well beyond the
    # three-metre confirmation radius.  Oblique views make facade/roof edges and
    # nearby context much more likely to be visible than a normal-only close-up.
    candidates: list[tuple[tuple[float, float, float, float, float], dict[str, Any]]] = []
    for distance_m in (5.5, 7.0, 8.5, 10.0):
        for lateral_factor in (-0.80, -0.50, 0.50, 0.80):
            vertical_factors = (0.0,) if abs(normal[2]) > 0.8 else (-0.25, 0.25)
            for vertical_factor in vertical_factors:
                candidate = tuple(
                    target[index]
                    + normal[index] * distance_m * 0.72
                    + tangent[index] * distance_m * lateral_factor
                    + (distance_m * vertical_factor if index == 2 else 0.0)
                    for index in range(3)
                )
                # This is an offline L2 camera, not an executable UAV pose.  A
                # bounded exterior ring is required to inspect outward-facing
                # facade targets placed near the public flight-envelope edge.
                # That relaxation is horizontal only: a camera below the flight
                # floor is occluded by the ground plane in Isaac and cannot be
                # accepted as visual evidence.
                if any(
                    value < low - review_outside_margin
                    or value > high + review_outside_margin
                    for value, low, high in zip(candidate[:2], lower[:2], upper[:2], strict=True)
                ) or not lower[2] <= candidate[2] <= upper[2]:
                    continue
                clearance, _ = minimum_clearance(candidate, colliders)
                if clearance < required_clearance:
                    continue
                visible, _ = line_of_sight(
                    candidate,
                    target,
                    colliders,
                    ignored_ids=frozenset({owner_id}),
                )
                if not visible:
                    continue
                pose = pose_looking_at(candidate, target)
                visible_context: list[tuple[str, Vec3]] = []
                for collider in colliders:
                    if collider.collider_id not in context_ids:
                        continue
                    probe = tuple(
                        min(max(candidate[axis], collider.minimum[axis]), collider.maximum[axis])
                        for axis in range(3)
                    )
                    context_visible, _ = line_of_sight(
                        candidate,
                        probe,
                        colliders,
                        ignored_ids=frozenset({owner_id, collider.collider_id}),
                    )
                    if context_visible:
                        visible_context.append((collider.collider_id, probe))
                focus_probe = max(
                    visible_context,
                    key=lambda item: (distance(target, item[1]), item[0]),
                    default=None,
                )
                context_look_at = (
                    target
                    if focus_probe is None
                    else tuple(
                        target[axis] + 0.30 * (focus_probe[1][axis] - target[axis])
                        for axis in range(3)
                    )
                )
                lateral_ratio = abs(lateral_factor) / math.sqrt(0.72**2 + lateral_factor**2)
                score = (
                    float(len(visible_context)),
                    min(float(clearance), 10.0),
                    lateral_ratio,
                    -abs(distance_m - 8.0),
                    -float(lateral_factor),
                )
                candidates.append(
                    (
                        score,
                        {
                            "pose": pose.to_dict(),
                            "target_distance_m": round(distance(candidate, target), 5),
                            "clearance_m": round(clearance, 5),
                            "oblique_lateral_ratio": round(lateral_ratio, 6),
                            "look_at": [round(value, 5) for value in context_look_at],
                            "visible_context_collider_ids": sorted(
                                collider_id for collider_id, _ in visible_context
                            ),
                            "camera_kind": "private_l2_context_review_only",
                        },
                    )
                )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def derive_support_sites_v3(
    city: dict[str, Any], config: OrdinaryReleaseConfig
) -> list[dict[str, Any]]:
    colliders = colliders_from_city(city)
    context_radius = float(config.raw["admission"]["witness_distance_m"]) * 1.8
    minimum_surrounding_context = int(config.raw["admission"]["minimum_surrounding_colliders"])
    result: list[dict[str, Any]] = []
    for site in _raw_surface_sites(city):
        position = tuple(float(value) for value in site["position"])
        context = sorted(
            collider.collider_id
            for collider in colliders
            if collider.point_distance(position) <= context_radius
        )
        surrounding_context = [
            collider_id for collider_id in context if collider_id != site["owner_collider_id"]
        ]
        if (
            len(surrounding_context) < minimum_surrounding_context
            or site["owner_collider_id"] not in context
        ):
            continue
        witnesses = _compile_witnesses(city, site, colliders, config)
        if not witnesses:
            continue
        density = len(context) / max(1.0, 4.0 / 3.0 * math.pi * context_radius**3)
        result.append(
            {
                **site,
                "context_collider_ids": context,
                "context_collider_count": len(context),
                "surrounding_collider_ids": surrounding_context,
                "surrounding_collider_count": len(surrounding_context),
                "local_structure_density": round(density, 8),
                "legal_witnesses": witnesses,
                "legal_witness_count": len(witnesses),
                "minimum_confirmation_cost_m": min(
                    witness["target_distance_m"] for witness in witnesses
                ),
            }
        )
    required = int(math.ceil(float(config.raw["admission"]["minimum_support_to_target_ratio"])))
    if len(result) < required:
        raise GenerationRejected("too few reachable structure-context support sites")
    return sorted(result, key=lambda item: item["site_id"])


def _distance(first: dict[str, Any], second: dict[str, Any], vertical_scale: float = 1.0) -> float:
    first_position = first["position"]
    second_position = second["position"]
    return math.sqrt(
        (first_position[0] - second_position[0]) ** 2
        + (first_position[1] - second_position[1]) ** 2
        + ((first_position[2] - second_position[2]) * vertical_scale) ** 2
    )


def _admissible(
    sites: list[dict[str, Any]], selected: list[dict[str, Any]], separation: float
) -> list[dict[str, Any]]:
    selected_ids = {item["site_id"] for item in selected}
    return [
        site
        for site in sites
        if site["site_id"] not in selected_ids
        and all(_distance(site, current) >= separation for current in selected)
    ]


def _weighted(
    candidates: list[dict[str, Any]], weights: list[float], rng: random.Random
) -> dict[str, Any]:
    if not candidates or len(candidates) != len(weights) or sum(weights) <= 0:
        raise GenerationRejected("target process exhausted positive-weight candidates")
    return rng.choices(candidates, weights=weights, k=1)[0]


def _sample_targets(
    sites: list[dict[str, Any]],
    count: int,
    separation: float,
    process_name: str,
    profile: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    if process_name in {"clustered_surface", "anisotropic_clustered_surface"}:
        low, high = (int(value) for value in profile["cluster_count"])
        anchors = rng.sample(sites, k=min(count, rng.randint(low, high)))
    required_bands: list[str] = []
    if process_name == "height_stratified":
        weights_by_band = {key: float(value) for key, value in profile["band_weights"].items()}
        available = [
            band
            for band in ALTITUDE_BANDS
            if weights_by_band[band] > 0 and any(site["altitude_band"] == band for site in sites)
        ]
        minimum_bands = min(count, int(profile["minimum_bands"]))
        if len(available) < minimum_bands:
            raise GenerationRejected("height process lacks the required 3D altitude support")
        required_bands = rng.sample(available, k=minimum_bands)
    while len(selected) < count:
        candidates = _admissible(sites, selected, separation)
        if not candidates:
            raise GenerationRejected(f"{process_name} exhausted separated support sites")
        if required_bands:
            band = required_bands.pop()
            candidates = [site for site in candidates if site["altitude_band"] == band]
            if not candidates:
                raise GenerationRejected(
                    "height process could not preserve separated band coverage"
                )
        area_weights = [float(site["represented_area_m2"]) for site in candidates]
        if process_name == "uniform_surface":
            weights = area_weights
        elif process_name in {"clustered_surface", "anisotropic_clustered_surface"}:
            bandwidth = float(profile["bandwidth_m"])
            vertical_scale = float(profile.get("vertical_scale", 1.0))
            weights = [
                area
                * math.exp(
                    -min(_distance(site, anchor, vertical_scale) for anchor in anchors) / bandwidth
                )
                for site, area in zip(candidates, area_weights, strict=True)
            ]
        elif process_name == "height_stratified":
            weights_by_band = {key: float(value) for key, value in profile["band_weights"].items()}
            weights = [
                area * weights_by_band[site["altitude_band"]]
                for site, area in zip(candidates, area_weights, strict=True)
            ]
        else:
            raise GenerationRejected(f"unknown ordinary target process: {process_name}")
        selected.append(_weighted(candidates, weights, rng))
    return selected


def _context_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    return (
        (0.0 if first["support_class"] == second["support_class"] else 8.0)
        + (0.0 if first["altitude_band"] == second["altitude_band"] else 3.0)
        + abs(first["context_collider_count"] - second["context_collider_count"])
        + 0.3 * abs(first["legal_witness_count"] - second["legal_witness_count"])
        + 0.15 * abs(first["minimum_confirmation_cost_m"] - second["minimum_confirmation_cost_m"])
    )


def _matched_distractors(
    selected: list[dict[str, Any]], sites: list[dict[str, Any]], rng: random.Random
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    unavailable = {item["site_id"] for item in selected}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    order = list(selected)
    rng.shuffle(order)
    for target in order:
        candidates = [site for site in sites if site["site_id"] not in unavailable]
        mission_region_id = target.get("_mission_region_id")
        if mission_region_id is not None:
            candidates = [
                site
                for site in candidates
                if site.get("_mission_region_id") == mission_region_id
            ]
        if not candidates:
            raise GenerationRejected(
                "not enough same-region candidates for matched target distractors"
            )
        best_distance = min(_context_distance(target, candidate) for candidate in candidates)
        tied = [
            candidate
            for candidate in candidates
            if _context_distance(target, candidate) == best_distance
        ]
        distractor = rng.choice(tied)
        unavailable.add(distractor["site_id"])
        pairs.append((target, distractor))
    return sorted(pairs, key=lambda pair: pair[0]["site_id"])


def start_contract_errors(
    config: OrdinaryReleaseConfig, city: dict[str, Any], starts: list[dict[str, Any]]
) -> list[str]:
    vehicle = config.raw["execution_contract"]["vehicle"]
    body_margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    collision_margin = float(vehicle["radius_m"])
    lower = tuple(float(value) + body_margin for value in city["flight_bounds"]["minimum"])
    upper = tuple(float(value) - body_margin for value in city["flight_bounds"]["maximum"])
    colliders = colliders_from_city(city)
    errors: list[str] = []
    positions: list[tuple[str, Vec3]] = []
    for start in starts:
        drone_id = str(start.get("drone_id", ""))
        raw_position = start.get("position", [])
        if not drone_id or not isinstance(raw_position, list) or len(raw_position) != 3:
            errors.append("start identity or 3-D position is invalid")
            continue
        position = tuple(float(value) for value in raw_position)
        if not all(
            low <= value <= high
            for value, low, high in zip(position, lower, upper, strict=True)
        ):
            errors.append(f"{drone_id} violates the body-clearance flight bounds")
        clearance, collider_id = minimum_clearance(position, colliders)
        if clearance + 1.0e-9 < body_margin:
            errors.append(f"{drone_id} lacks spawn clearance from {collider_id}")
        positions.append((drone_id, position))
    for first_index, (first_id, first) in enumerate(positions):
        for second_id, second in positions[first_index + 1 :]:
            if distance(first, second) + 1.0e-9 < 2.0 * collision_margin:
                errors.append(f"{first_id} and {second_id} overlap at spawn")
    return errors


def _inset_node_center(
    city: dict[str, Any], node: dict[str, Any], inset_m: float
) -> tuple[float, float]:
    x, y = (float(value) for value in node["position"][:2])
    size = float(city["size_m"])
    limit = size / 2.0 - inset_m
    if abs(x) <= limit and abs(y) <= limit:
        return x, y
    nodes = {str(item["id"]): item for item in city["road_graph"]["nodes"]}
    neighbor_ids = []
    for edge in city["road_graph"]["edges"]:
        if edge["start_node"] == node["id"]:
            neighbor_ids.append(str(edge["end_node"]))
        elif edge["end_node"] == node["id"]:
            neighbor_ids.append(str(edge["start_node"]))
    if not neighbor_ids:
        return max(-limit, min(limit, x)), max(-limit, min(limit, y))
    neighbor = min(
        (nodes[node_id] for node_id in neighbor_ids),
        key=lambda item: math.hypot(float(item["position"][0]), float(item["position"][1])),
    )
    dx = float(neighbor["position"][0]) - x
    dy = float(neighbor["position"][1]) - y
    length = math.hypot(dx, dy)
    if length <= 1.0e-9:
        return max(-limit, min(limit, x)), max(-limit, min(limit, y))
    shift = inset_m + 0.05
    return x + shift * dx / length, y + shift * dy / length


def _starts(
    city: dict[str, Any], config: OrdinaryReleaseConfig, fleet_count: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(derived_seed(seed, "spawn"))
    nodes = list(city["road_graph"]["nodes"])
    grammar = str(city["spawn_grammar"])
    degree = Counter()
    for edge in city["road_graph"]["edges"]:
        degree[str(edge["start_node"])] += 1
        degree[str(edge["end_node"])] += 1
    if grammar == "edge":
        candidates = sorted(nodes, key=lambda node: math.hypot(*node["position"][:2]), reverse=True)
    elif grammar == "intersection":
        candidates = sorted(
            nodes, key=lambda node: (-degree[str(node["id"])], math.hypot(*node["position"][:2]))
        )
    elif grammar == "courtyard":
        candidates = [node for node in nodes if degree[str(node["id"])] == 1]
        rng.shuffle(candidates)
        candidates.extend(node for node in nodes if node not in candidates)
    else:
        candidates = list(nodes)
        rng.shuffle(candidates)
    vehicle = config.raw["execution_contract"]["vehicle"]
    body_margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    formation_radius = max(1.25, 2.25 * float(vehicle["radius_m"]))
    shared = grammar != "distributed"
    for candidate_index in range(len(candidates)):
        selected = (
            [candidates[candidate_index]] * fleet_count
            if shared
            else [
                candidates[(candidate_index + index) % len(candidates)]
                for index in range(fleet_count)
            ]
        )
        starts = []
        for index, anchor in enumerate(selected):
            angle = 2.0 * math.pi * index / fleet_count
            radius = formation_radius if shared else 0.0
            center_x, center_y = _inset_node_center(city, anchor, body_margin + radius)
            starts.append(
                {
                    "drone_id": f"uav-{index:02d}",
                    "position": [
                        round(center_x + radius * math.cos(angle), 4),
                        round(center_y + radius * math.sin(angle), 4),
                        2.5,
                    ],
                    "yaw_deg": round(math.degrees(angle), 3),
                    "spawn_grammar": grammar,
                }
            )
        if not start_contract_errors(config, city, starts):
            return starts
    raise GenerationRejected(f"no safe {grammar} spawn formation exists")


def _episode_reachable_sites(
    city: dict[str, Any],
    support_sites: list[dict[str, Any]],
    starts: list[dict[str, Any]],
    config: OrdinaryReleaseConfig,
) -> list[dict[str, Any]]:
    colliders = colliders_from_city(city)
    vehicle = config.raw["execution_contract"]["vehicle"]
    episode = config.raw["execution_contract"]["episode"]
    body_margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    maximum_z = float(city["flight_bounds"]["maximum"][2]) - body_margin
    sky_z = max(collider.maximum[2] for collider in colliders) + body_margin + 1.0
    if sky_z > maximum_z:
        raise GenerationRejected("flight bounds do not contain a safe over-roof transit layer")
    maximum_time = float(episode["duration_s"]) - float(episode["return_reserve_s"])
    maximum_energy = float(vehicle["energy_budget_j"])
    horizontal_speed = float(vehicle["horizontal_speed_mps"])
    vertical_speed = float(vehicle["vertical_speed_mps"])
    energy_per_meter = float(vehicle["energy_per_meter_j"])

    def segment_clear(first: Vec3, second: Vec3) -> bool:
        return all(
            segment_intersection_fraction(first, second, collider.expanded(body_margin)) is None
            for collider in colliders
        )

    # The transit altitude is one metre above every expanded collider.  Thus
    # each horizontal sky segment is necessarily clear.  The vertical lift from
    # a start is invariant across all witnesses, and the descent to a witness
    # is invariant across all starts.  Hoisting both checks preserves the same
    # three-segment proof while avoiding repeated collider scans.
    start_sky_clear = {
        str(start["drone_id"]): segment_clear(
            tuple(float(value) for value in start["position"]),
            (
                float(start["position"][0]),
                float(start["position"][1]),
                sky_z,
            ),
        )
        for start in starts
    }
    reachable: list[dict[str, Any]] = []
    for site in support_sites:
        witnesses = []
        for witness in site["legal_witnesses"]:
            witness_position = tuple(float(value) for value in witness["pose"]["position"])
            witness_sky = (witness_position[0], witness_position[1], sky_z)
            if not segment_clear(witness_sky, witness_position):
                continue
            best_proof: dict[str, Any] | None = None
            for start in starts:
                start_position = tuple(float(value) for value in start["position"])
                if not start_sky_clear[str(start["drone_id"])]:
                    continue
                first_sky = (start_position[0], start_position[1], sky_z)
                horizontal_distance = distance(first_sky, witness_sky)
                vertical_distance = distance(start_position, first_sky) + distance(
                    witness_sky, witness_position
                )
                path_distance = horizontal_distance + vertical_distance
                travel_time = (
                    horizontal_distance / horizontal_speed + vertical_distance / vertical_speed
                )
                energy = path_distance * energy_per_meter
                if travel_time > maximum_time or energy > maximum_energy:
                    continue
                proof = {
                    "start_drone_id": start["drone_id"],
                    "path_model": "three_segment_safe_sky_corridor",
                    "transit_altitude_m": round(sky_z, 5),
                    "path_distance_upper_bound_m": round(path_distance, 5),
                    "travel_time_upper_bound_s": round(travel_time, 5),
                    "energy_upper_bound_j": round(energy, 5),
                }
                if (
                    best_proof is None
                    or proof["travel_time_upper_bound_s"] < best_proof["travel_time_upper_bound_s"]
                ):
                    best_proof = proof
            if best_proof is not None:
                witnesses.append({**witness, "reachability_proof": best_proof})
        if witnesses:
            reachable.append(
                {
                    **site,
                    "legal_witnesses": witnesses,
                    "legal_witness_count": len(witnesses),
                    "reachability_hash": content_hash(
                        [witness["reachability_proof"] for witness in witnesses]
                    ),
                }
            )
    return reachable


def _reachability_cache_key(
    city: dict[str, Any],
    support_sites: list[dict[str, Any]],
    starts: list[dict[str, Any]],
    config: OrdinaryReleaseConfig,
) -> str:
    """Bind cached proofs to every input that can change their safety meaning."""

    return content_hash(
        {
            "schema": "org.aerocity.bench.reachability-cache-key.v1",
            "task_geometry_hash": _task_geometry_hash(city),
            "support_sites_hash": content_hash(support_sites),
            "starts": starts,
            "execution_contract": config.raw["execution_contract"],
        }
    )


def _cached_episode_reachable_sites(
    city: dict[str, Any],
    support_sites: list[dict[str, Any]],
    starts: list[dict[str, Any]],
    config: OrdinaryReleaseConfig,
) -> list[dict[str, Any]]:
    """Return a detached, bounded cache of deterministic private reachability proofs.

    A process-specific cache only removes repeated compilation for paired
    target-process episodes. It is not a release artifact and it is never
    serialized or exposed to a planner.
    """

    key = _reachability_cache_key(city, support_sites, starts, config)
    with _REACHABILITY_CACHE_LOCK:
        cached = _REACHABILITY_CACHE.get(key)
        if cached is not None:
            _REACHABILITY_CACHE.move_to_end(key)
            return copy.deepcopy(list(cached))
    compiled = _episode_reachable_sites(city, support_sites, starts, config)
    with _REACHABILITY_CACHE_LOCK:
        _REACHABILITY_CACHE[key] = tuple(copy.deepcopy(compiled))
        _REACHABILITY_CACHE.move_to_end(key)
        while len(_REACHABILITY_CACHE) > _REACHABILITY_CACHE_LIMIT:
            _REACHABILITY_CACHE.popitem(last=False)
    return compiled


def _clear_reachability_cache_for_tests() -> None:
    """Reset process-local cached private proofs for deterministic regression tests."""

    with _REACHABILITY_CACHE_LOCK:
        _REACHABILITY_CACHE.clear()


def _episode_condition(
    config: OrdinaryReleaseConfig, city: dict[str, Any], episode_index: int
) -> tuple[str, int, str]:
    split = str(city["split"])
    processes = config.target_processes(split)
    group_index = episode_index // len(processes)
    process_name = processes[episode_index % len(processes)]
    group_seed = derived_seed(
        config.master_seed, _task_geometry_hash(city), "paired-process", group_index
    )
    return process_name, group_seed, f"process-pair-{group_index:04d}"


def _site_matches_mission_region(site: dict[str, Any], region: dict[str, Any]) -> bool:
    class_mapping = {
        "roof": "roof",
        "facade": "facade_marker_site",
        "entrance": "entrance",
        "rubble": "rubble",
    }
    if site["support_class"] != class_mapping.get(str(region["region_class"])):
        return False
    cells = region.get("cells")
    if not isinstance(cells, list) or not cells:
        return False
    region_normal = tuple(float(value) for value in cells[0]["surface_normal"])
    site_normal = tuple(float(value) for value in site["normal"])
    if sum(
        first * second for first, second in zip(region_normal, site_normal, strict=True)
    ) < 0.999:
        return False
    lower = tuple(float(value) for value in region["bounds"]["minimum"])
    upper = tuple(float(value) for value in region["bounds"]["maximum"])
    return all(
        low - 0.25 <= value <= high + 0.25
        for value, low, high in zip(site["position"], lower, upper, strict=True)
    )


def _mission_sector_support_sites(
    city: dict[str, Any],
    public_task_spec: dict[str, Any],
    mission_sector: dict[str, Any],
    starts: list[dict[str, Any]],
    config: OrdinaryReleaseConfig,
) -> list[dict[str, Any]]:
    """Derive fixed private candidates inside public mission regions.

    Public cells are inspection obligations, not hidden target supports.  This
    keeps target-process sampling invariant when a public atlas density changes.
    """

    atlas = public_task_spec.get("inspection_atlas")
    if not isinstance(atlas, dict):
        raise GenerationRejected("G2-I mission sampling requires the full public atlas")
    validate_public_mission_sector(
        mission_sector, atlas, starts, config.raw["execution_contract"]
    )
    selected_region_ids = {str(value) for value in mission_sector["selected_region_ids"]}
    selected_regions = [
        region for region in atlas["regions"] if str(region["region_id"]) in selected_region_ids
    ]
    if len(selected_regions) != len(selected_region_ids):
        raise GenerationRejected("mission sector does not resolve to unique public regions")
    colliders = colliders_from_city(city)
    context_radius = float(config.raw["admission"]["witness_distance_m"]) * 1.8
    minimum_context = int(config.raw["admission"]["minimum_surrounding_colliders"])
    result: list[dict[str, Any]] = []
    for raw_site in _g2_i_private_surface_sites(city):
        matching_regions = [
            region
            for region in selected_regions
            if _site_matches_mission_region(raw_site, region)
        ]
        if not matching_regions:
            continue
        mission_region = min(matching_regions, key=lambda region: str(region["region_id"]))
        position = tuple(float(value) for value in raw_site["position"])
        owner_id = str(raw_site["owner_collider_id"])
        owner = next((item for item in colliders if item.collider_id == owner_id), None)
        if owner is None or owner.point_distance(position) > 0.3:
            continue
        context = sorted(
            collider.collider_id
            for collider in colliders
            if collider.point_distance(position) <= context_radius
        )
        surrounding = [value for value in context if value != owner_id]
        if len(surrounding) < minimum_context or owner_id not in context:
            continue
        witnesses = _compile_witnesses(city, raw_site, colliders, config)
        if not witnesses:
            continue
        density = len(context) / max(1.0, 4.0 / 3.0 * math.pi * context_radius**3)
        result.append(
            {
                **raw_site,
                # This evaluator-private annotation is used only to create a
                # counterfactual distractor in the same public inspection
                # region. It is deliberately omitted from the episode schema.
                "_mission_region_id": str(mission_region["region_id"]),
                "context_collider_ids": context,
                "context_collider_count": len(context),
                "surrounding_collider_ids": surrounding,
                "surrounding_collider_count": len(surrounding),
                "local_structure_density": round(density, 8),
                "legal_witnesses": witnesses,
                "legal_witness_count": len(witnesses),
                "minimum_confirmation_cost_m": min(
                    witness["target_distance_m"] for witness in witnesses
                ),
            }
        )
    return sorted(
        {site["site_id"]: site for site in result}.values(),
        key=lambda item: item["site_id"],
    )


def sample_episode_v3(
    config: OrdinaryReleaseConfig,
    city: dict[str, Any],
    support_sites: list[dict[str, Any]],
    episode_index: int,
    *,
    public_task_spec: dict[str, Any] | None = None,
    precomputed_mission_sector: dict[str, Any] | None = None,
    precomputed_mission_sector_sites: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if (precomputed_mission_sector is None) != (
        precomputed_mission_sector_sites is None
    ):
        raise ValueError("precomputed G2-I sector and sites must be supplied together")
    if precomputed_mission_sector is not None and public_task_spec is None:
        raise ValueError("precomputed G2-I sector requires a public task spec")
    process_name, group_seed, group_id = _episode_condition(config, city, episode_index)
    count_rng = random.Random(derived_seed(group_seed, "fixed-target-count"))
    target_rng = random.Random(derived_seed(group_seed, "target-process", process_name))
    starts = _starts(city, config, config.fleet_count, group_seed)
    mission_sector: dict[str, Any] | None = None
    eligible_sites = support_sites
    if public_task_spec is not None:
        if public_task_spec.get("task_track") != TASK_TRACK_G2_I:
            raise ValueError("mission-sector sampling requires a public G2-I task spec")
        atlas = public_task_spec.get("inspection_atlas")
        if not isinstance(atlas, dict):
            raise ValueError("mission-sector sampling requires the full public atlas")
        if precomputed_mission_sector is None:
            mission_sector = compile_public_mission_sector(
                atlas, starts, config.raw["execution_contract"]
            )
            eligible_sites = _mission_sector_support_sites(
                city,
                public_task_spec,
                mission_sector,
                starts,
                config,
            )
        else:
            mission_sector = copy.deepcopy(precomputed_mission_sector)
            validate_public_mission_sector(
                mission_sector, atlas, starts, config.raw["execution_contract"]
            )
            eligible_sites = copy.deepcopy(precomputed_mission_sector_sites)
    reachable_sites = _cached_episode_reachable_sites(
        city, eligible_sites, starts, config
    )
    low, high = config.target_range(int(city["size_m"]))
    target_count = count_rng.randint(low, high)
    required_ratio = float(config.raw["admission"]["minimum_support_to_target_ratio"])
    if len(reachable_sites) < math.ceil(required_ratio * target_count):
        raise GenerationRejected(
            f"{len(reachable_sites)} reachable support sites are insufficient for "
            f"{target_count} targets"
        )
    profile = config.raw["target_processes"]["profiles"][process_name]
    selected = _sample_targets(
        reachable_sites,
        target_count,
        float(config.raw["admission"]["minimum_target_separation_m"]),
        process_name,
        profile,
        target_rng,
    )
    selected.sort(key=lambda item: item["site_id"])
    nearby_limit = float(config.raw["execution_contract"]["observe"]["max_range_m"]) * 2.0
    maximum_fraction = float(config.raw["admission"]["maximum_single_observation_target_fraction"])
    maximum_nearby = max(
        sum(_distance(target, other) <= nearby_limit for other in selected) for target in selected
    )
    if maximum_nearby / target_count > maximum_fraction:
        raise GenerationRejected(
            "target layout permits an excessive single-observation target fraction"
        )
    pairs = _matched_distractors(selected, reachable_sites, target_rng)
    pair_by_target = {target["site_id"]: distractor for target, distractor in pairs}
    targets = []
    distractors = []
    counterfactual_pairs = []
    for index, site in enumerate(selected):
        target_id = f"target-{index:03d}"
        distractor = pair_by_target[site["site_id"]]
        pair_id = f"counterfactual-{content_hash([site['site_id'], distractor['site_id']])[:14]}"
        target = {
            "target_id": target_id,
            "site_id": site["site_id"],
            "support_class": site["support_class"],
            "altitude_band": site["altitude_band"],
            "position": site["position"],
            "normal": site["normal"],
            "owner_collider_id": site["owner_collider_id"],
            "context_collider_ids": site["context_collider_ids"],
            "represented_area_m2": site["represented_area_m2"],
            "legal_witnesses": site["legal_witnesses"],
            "legal_witness_count": site["legal_witness_count"],
            "reachability_hash": site["reachability_hash"],
            "valid_before_run": True,
        }
        targets.append(target)
        distractors.append(
            {
                "distractor_id": f"distractor-{index:03d}",
                "site_id": distractor["site_id"],
                "support_class": distractor["support_class"],
                "altitude_band": distractor["altitude_band"],
                "position": distractor["position"],
                "normal": distractor["normal"],
                "owner_collider_id": distractor["owner_collider_id"],
                "context_collider_ids": distractor["context_collider_ids"],
                "legal_witnesses": distractor["legal_witnesses"],
                "legal_witness_count": distractor["legal_witness_count"],
                "reachability_hash": distractor["reachability_hash"],
            }
        )
        counterfactual_pairs.append(
            {
                "pair_id": pair_id,
                "target_site_id": site["site_id"],
                "distractor_site_id": distractor["site_id"],
            }
        )
    target_validity = {
        "schema": "org.aerocity.bench.target-validity-private.v1",
        "layout_hash": city["layout_hash"],
        "condition_group_id": group_id,
        "target_ids": [target["target_id"] for target in targets],
        "site_ids": [target["site_id"] for target in targets],
        "witness_hashes": [content_hash(target["legal_witnesses"]) for target in targets],
        "reachability_hashes": [target["reachability_hash"] for target in targets],
        "frozen_before_execution": True,
    }
    target_validity["validity_hash"] = content_hash(target_validity)
    episode_identifier = content_hash([city["layout_hash"], episode_index, process_name])[:18]
    episode = {
        "schema": "org.aerocity.bench.episode-private.ordinary.v3",
        "episode_id": f"episode-{episode_identifier}",
        "layout_id": city["layout_id"],
        "layout_hash": city["layout_hash"],
        "episode_index": episode_index,
        "episode_seed": derived_seed(group_seed, process_name),
        "condition_group_id": group_id,
        "target_process": process_name,
        "target_count": target_count,
        "targets": targets,
        "distractors": distractors,
        "counterfactual_pairs": counterfactual_pairs,
        "target_validity": target_validity,
        "fleet_profile": {
            "name": config.raw["fleet"]["profile"],
            "count": config.fleet_count,
        },
        "starts": starts,
        "execution_contract_hash": content_hash(config.raw["execution_contract"]),
        "formal_execution_level": config.raw["execution_contract"]["formal_execution_level"],
    }
    if mission_sector is not None:
        episode["mission_sector"] = mission_sector
        episode["mission_sector_hash"] = mission_sector["sector_hash"]
    episode["target_summary_private"] = {
        "support_histogram": dict(
            sorted(Counter(target["support_class"] for target in targets).items())
        ),
        "altitude_histogram": dict(
            sorted(Counter(target["altitude_band"] for target in targets).items())
        ),
        "vertical_span_m": round(
            max(target["position"][2] for target in targets)
            - min(target["position"][2] for target in targets),
            4,
        ),
    }
    episode["episode_hash"] = content_hash(episode)
    return episode


def sample_visual_review_episode_v3(
    config: OrdinaryReleaseConfig,
    city: dict[str, Any],
    support_sites: list[dict[str, Any]],
    starts: list[dict[str, Any]],
    *,
    target_count: int,
    process_name: str = "height_stratified",
) -> dict[str, Any]:
    """Generate a hashed, non-scoring target overlay for human scene review."""

    if target_count < 1:
        raise ValueError("visual-review target_count must be positive")
    profiles = config.raw["target_processes"]["profiles"]
    if process_name not in profiles:
        raise ValueError(f"unknown visual-review target process: {process_name}")
    start_errors = start_contract_errors(config, city, starts)
    if start_errors:
        raise GenerationRejected("visual-review starts are invalid: " + "; ".join(start_errors))
    reachable_sites = _episode_reachable_sites(city, support_sites, starts, config)
    formal_required_ratio = float(config.raw["admission"]["minimum_support_to_target_ratio"])
    review_required_ratio = min(formal_required_ratio, 4.0)
    if len(reachable_sites) < math.ceil(review_required_ratio * target_count):
        raise GenerationRejected(
            f"{len(reachable_sites)} reachable sites cannot support {target_count} review targets"
        )
    seed = derived_seed(_task_geometry_hash(city), "visual-review", process_name, target_count)
    nearby_limit = float(config.raw["execution_contract"]["observe"]["max_range_m"]) * 2.0
    maximum_fraction = float(config.raw["admission"]["maximum_single_observation_target_fraction"])
    selected: list[dict[str, Any]] | None = None
    selected_context_poses: dict[str, dict[str, Any]] = {}
    maximum_nearby = 0
    sampling_attempt = 0
    sampling_errors: Counter[str] = Counter()
    for attempt in range(128):
        rng = random.Random(derived_seed(seed, "admissible-review-sample", attempt))
        try:
            candidate = _sample_targets(
                reachable_sites,
                target_count,
                float(config.raw["admission"]["minimum_target_separation_m"]),
                process_name,
                profiles[process_name],
                rng,
            )
        except GenerationRejected as exc:
            sampling_errors[str(exc)] += 1
            continue
        candidate_context_poses = {
            str(site["site_id"]): _context_review_pose(city, site, config)
            for site in candidate
        }
        if any(pose is None for pose in candidate_context_poses.values()):
            sampling_errors["missing_private_l2_context_pose"] += 1
            continue
        candidate_nearby = max(
            sum(_distance(target, other) <= nearby_limit for other in candidate)
            for target in candidate
        )
        if candidate_nearby / target_count <= maximum_fraction:
            selected = candidate
            selected_context_poses = {
                site_id: pose
                for site_id, pose in candidate_context_poses.items()
                if pose is not None
            }
            maximum_nearby = candidate_nearby
            sampling_attempt = attempt
            break
        sampling_errors["excessive_co_visibility"] += 1
    if selected is None:
        raise GenerationRejected(
            "visual-review target layout has no admissible deterministic sample after "
            f"128 attempts: {dict(sorted(sampling_errors.items()))}"
        )
    selected.sort(key=lambda item: item["site_id"])
    targets = []
    for index, site in enumerate(selected):
        # The farthest high-clearance legal witness gives the private visual audit a
        # reproducible local camera without weakening the formal hidden-target split.
        review_witness = max(
            site["legal_witnesses"],
            key=lambda witness: (
                float(witness["target_distance_m"]),
                float(witness["clearance_m"]),
                str(witness["witness_id"]),
            ),
        )
        context_pose = selected_context_poses.get(str(site["site_id"]))
        if context_pose is None:
            raise GenerationRejected("selected visual-review site lacks a private L2 context pose")
        targets.append(
            {
                "target_id": f"review-target-{index:03d}",
                "site_id": site["site_id"],
                "support_class": site["support_class"],
                "altitude_band": site["altitude_band"],
                "position": site["position"],
                "normal": site["normal"],
                "owner_collider_id": site["owner_collider_id"],
                "context_collider_ids": site["context_collider_ids"],
                "surrounding_collider_count": site["surrounding_collider_count"],
                "legal_witness_count": site["legal_witness_count"],
                "legal_witness_hash": content_hash(site["legal_witnesses"]),
                "local_review_pose": review_witness["pose"],
                "local_review_witness_id": review_witness["witness_id"],
                "local_context_review_pose": context_pose["pose"],
                "local_context_review_look_at": context_pose["look_at"],
                "local_context_review_distance_m": context_pose["target_distance_m"],
                "local_context_review_clearance_m": context_pose["clearance_m"],
                "local_context_review_oblique_lateral_ratio": context_pose[
                    "oblique_lateral_ratio"
                ],
                "local_context_visible_collider_ids": context_pose[
                    "visible_context_collider_ids"
                ],
                "reachability_hash": site["reachability_hash"],
            }
        )
    positions = [target["position"] for target in targets]
    context_camera_heights = [
        float(target["local_context_review_pose"]["position"][2]) for target in targets
    ]
    flight_floor = float(city["flight_bounds"]["minimum"][2])
    flight_ceiling = float(city["flight_bounds"]["maximum"][2])
    pair_distances = [
        math.dist(first, second)
        for first_index, first in enumerate(positions)
        for second in positions[first_index + 1 :]
    ]
    episode = {
        "schema": "org.aerocity.bench.visual-review-private.v1",
        "purpose": "human_visual_distribution_review_only",
        "formal_score_eligible": False,
        "layout_id": city["layout_id"],
        "layout_hash": city["layout_hash"],
        "review_seed": seed,
        "target_process": process_name,
        "target_count": target_count,
        "targets": targets,
        "starts": starts,
        "audit": {
            "support_histogram": dict(
                sorted(Counter(target["support_class"] for target in targets).items())
            ),
            "altitude_histogram": dict(
                sorted(Counter(target["altitude_band"] for target in targets).items())
            ),
            "vertical_span_m": round(
                max(position[2] for position in positions)
                - min(position[2] for position in positions),
                6,
            ),
            "minimum_pair_distance_m": round(min(pair_distances), 6),
            "all_targets_have_surrounding_colliders": all(
                target["surrounding_collider_count"] >= 1 for target in targets
            ),
            "all_targets_have_legal_witnesses": all(
                target["legal_witness_count"] >= 1 for target in targets
            ),
            "all_targets_have_private_l2_context_pose": all(
                "local_context_review_pose" in target for target in targets
            ),
            "all_private_l2_context_poses_within_vertical_flight_bounds": all(
                flight_floor <= height <= flight_ceiling for height in context_camera_heights
            ),
            "minimum_private_l2_context_camera_height_m": round(
                min(context_camera_heights), 6
            ),
            "maximum_private_l2_context_camera_height_m": round(
                max(context_camera_heights), 6
            ),
            "minimum_private_l2_context_distance_m": round(
                min(target["local_context_review_distance_m"] for target in targets), 6
            ),
            "minimum_private_l2_context_clearance_m": round(
                min(target["local_context_review_clearance_m"] for target in targets), 6
            ),
            "targets_with_visible_structural_context": sum(
                bool(target["local_context_visible_collider_ids"]) for target in targets
            ),
            "maximum_nearby_target_fraction": round(maximum_nearby / target_count, 6),
            "reachable_support_site_count": len(reachable_sites),
            "reachable_support_to_target_ratio": round(len(reachable_sites) / target_count, 6),
            "review_minimum_support_to_target_ratio": review_required_ratio,
            "formal_minimum_support_to_target_ratio_not_applied": formal_required_ratio,
            "deterministic_sampling_attempt": sampling_attempt,
            "rejected_sampling_attempts": sum(sampling_errors.values()),
            "sampling_rejection_histogram": dict(sorted(sampling_errors.items())),
        },
    }
    episode["episode_hash"] = content_hash(episode)
    return episode


def public_episode_projection(episode: dict[str, Any]) -> dict[str, Any]:
    """Return the only episode fields a method may receive before reset."""

    projection = {
        "schema": "org.aerocity.bench.episode-public.ordinary.v1",
        "episode_id": episode["episode_id"],
        "layout_id": episode["layout_id"],
        # The projection crosses the authority boundary.  It must not share
        # mutable roster objects with the evaluator-held private episode.
        "fleet_profile": copy.deepcopy(episode["fleet_profile"]),
        "starts": copy.deepcopy(episode["starts"]),
        "target_count_public": False,
        "target_process_public": False,
    }
    if "mission_sector" in episode:
        projection["mission_sector"] = copy.deepcopy(episode["mission_sector"])
        projection["mission_sector_hash"] = str(episode["mission_sector_hash"])
    return projection


def validate_frozen_g2_i_episode(
    episode: dict[str, Any],
    city: dict[str, Any],
    task_spec: dict[str, Any],
    execution_contract: dict[str, Any],
) -> None:
    """Fail closed when a frozen private episode is replayed on another task."""

    if episode.get("schema") != "org.aerocity.bench.episode-private.ordinary.v3":
        raise ValueError("frozen G2-I episode schema differs")
    if task_spec.get("task_track") != TASK_TRACK_G2_I:
        raise ValueError("frozen G2-I episode requires a public G2-I task spec")
    atlas = task_spec.get("inspection_atlas")
    if not isinstance(atlas, dict):
        raise ValueError("frozen G2-I episode requires the authority full-cell atlas")
    if episode.get("layout_id") != city.get("layout_id"):
        raise ValueError("frozen G2-I episode is not bound to its city")
    if episode.get("layout_hash") != city.get("layout_hash"):
        raise ValueError("frozen G2-I episode layout hash differs from its city")
    if episode.get("execution_contract_hash") != content_hash(execution_contract):
        raise ValueError("frozen G2-I episode execution contract differs")
    stored_hash = episode.get("episode_hash")
    if not isinstance(stored_hash, str):
        raise ValueError("frozen G2-I episode lacks episode_hash")
    unhashed = dict(episode)
    unhashed.pop("episode_hash", None)
    if content_hash(unhashed) != stored_hash:
        raise ValueError("frozen G2-I episode hash does not match its contents")
    sector = episode.get("mission_sector")
    if not isinstance(sector, dict):
        raise ValueError("frozen G2-I episode lacks its public mission sector")
    if episode.get("mission_sector_hash") != sector.get("sector_hash"):
        raise ValueError("frozen G2-I episode mission-sector hash differs")
    validate_public_mission_sector(
        sector,
        atlas,
        episode.get("starts"),
        execution_contract,
    )
