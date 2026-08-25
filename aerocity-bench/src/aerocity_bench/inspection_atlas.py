"""Target-agnostic public inspection-atlas compilation for G2-I.

The atlas exposes *what classes of city structure should be inspected*, not
which structure contains a target.  It deliberately operates on CitySpec
geometry and the public observation contract only.  It never imports target
sampling, support-site, evaluator, or episode modules.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any

from .canonical import content_hash
from .contracts import Pose3D
from .geometry import (
    Vec3,
    colliders_from_city,
    distance,
    in_field_of_view,
    line_of_sight,
    minimum_clearance,
    minimum_segment_clearance,
    sensor_pose,
    surface_facing,
)

ATLAS_SCHEMA = "org.aerocity.bench.inspection-atlas-public.v1"
ATLAS_VERSION = "g2-i-atlas-v1"
ATLAS_PROJECTION_SCHEMA = "org.aerocity.bench.inspection-atlas-projection-public.v1"
ATLAS_PROJECTION_VERSION = "g2-i-prior-ablation-v1"
ATLAS_PRIOR_COARSE = "coarse-regions"
ATLAS_PRIOR_FULL = "full-cells"
ATLAS_PRIOR_LEVELS = frozenset({ATLAS_PRIOR_COARSE, ATLAS_PRIOR_FULL})
TASK_TRACK_G1_U = "G1-U"
TASK_TRACK_G2_I = "G2-I"
REGION_CLASSES = frozenset({"roof", "facade", "entrance", "rubble"})
MISSION_SECTOR_SCHEMA = "org.aerocity.bench.inspection-mission-sector-public.v2"
MISSION_SECTOR_POLICY = "g2-i-budgeted-public-sector-v1"

_SAMPLING_POLICY = {
    "schema": "org.aerocity.bench.inspection-atlas-sampling-policy-public.v1",
    "policy_id": "g2-i-geometric-sampling-calibration-candidate-v2",
    "nominal_standoff_range_fraction": 0.62,
    "footprint_spacing_fraction": 0.70,
    "clearance_buffer_m": 0.35,
    "minimum_cell_spacing_m": 1.0,
    "facade_surface_inset_m": 0.55,
    "rubble_surface_inset_m": 0.25,
    "entrance_half_width_m": 1.5,
    "entrance_height_offset_m": 0.45,
    "entrance_standoff_range_fraction": 0.82,
    "flight_bound_buffer_m": 0.5,
    "method_independent_calibration_required": True,
    "calibration_status": "frozen",
}

# Density conditions are explicit public calibration candidates.  They keep the
# same geometry, sensor, and safety semantics, changing only the footprint
# sampling fraction.  An arbitrary caller-supplied dictionary is never
# accepted: otherwise a nominal-atlas validator could silently bless a custom
# evaluation denominator.
_SAMPLING_POLICY_CANDIDATES = {
    str(_SAMPLING_POLICY["policy_id"]): _SAMPLING_POLICY,
    "g2-i-geometric-sampling-density-sparse-v1": {
        **_SAMPLING_POLICY,
        "policy_id": "g2-i-geometric-sampling-density-sparse-v1",
        "footprint_spacing_fraction": 0.75,
        "calibration_status": "ablation-only",
    },
    "g2-i-geometric-sampling-density-dense-v1": {
        **_SAMPLING_POLICY,
        "policy_id": "g2-i-geometric-sampling-density-dense-v1",
        "footprint_spacing_fraction": 0.60,
        "calibration_status": "ablation-only",
    },
}

# Normalized object keys which would turn the public atlas into an evaluator
# projection.  The check is intentionally recursive so a future serializer
# cannot hide a prohibited field inside a pose, graph node, or metadata block.
_PRIVATE_KEY_TOKENS = frozenset(
    {
        "target",
        "targetid",
        "targetcount",
        "targetprocess",
        "targetlabel",
        "supportsite",
        "supportsiteid",
        "siteid",
        "witness",
        "legalwitness",
        "evaluator",
        "private",
        "split",
        "seed",
        "distractor",
        "confirmation",
        "recall",
    }
)


def _normalised_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _assert_no_private_keys(node: object, path: str = "atlas") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            normalised = _normalised_key(key)
            if any(token in normalised for token in _PRIVATE_KEY_TOKENS):
                raise ValueError(f"public inspection atlas contains prohibited key at {path}.{key}")
            _assert_no_private_keys(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_no_private_keys(value, f"{path}[{index}]")


def _finite_number(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must be a {size}-vector")
    return tuple(_finite_number(item, name) for item in value)


def _vec3(value: object, name: str) -> Vec3:
    x_value, y_value, z_value = _vector(value, 3, name)
    return (x_value, y_value, z_value)


def _inspection_camera_pose(pose: Pose3D, execution_contract: dict[str, Any]) -> Pose3D:
    """Interpret atlas pitch as public camera pitch, never body pitch."""

    rig = execution_contract["sensor_rig"]
    body = pose if rig["gimbal_mode"] == "fixed" else Pose3D(pose.position, pose.yaw_deg)
    return sensor_pose(
        body,
        rig["translation_body_m"],
        sensor_pitch_deg=(pose.pitch_deg if rig["gimbal_mode"] == "bounded" else None),
    )


def _altitude_band(z_value: float) -> str:
    if z_value < 4.0:
        return "near_ground"
    if z_value < 12.0:
        return "lower"
    if z_value < 25.0:
        return "mid"
    if z_value < 40.0:
        return "elevated"
    return "highrise"


def _component_bounds(component: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    center_x, center_y, center_z = _vector(component["center"], 3, "component.center")
    size_x, size_y, size_z = _vector(component["size"], 3, "component.size")
    if min(size_x, size_y, size_z) <= 0.0:
        raise ValueError("inspection-atlas component sizes must be positive")
    return (
        center_x - size_x / 2.0,
        center_x + size_x / 2.0,
        center_y - size_y / 2.0,
        center_y + size_y / 2.0,
        center_z - size_z / 2.0,
        center_z + size_z / 2.0,
    )


def _centered_values(low: float, high: float, spacing: float) -> list[float]:
    if spacing <= 0.0:
        raise ValueError("inspection-atlas spacing must be positive")
    if high < low:
        return []
    count = max(1, math.ceil((high - low) / spacing))
    return [low + (high - low) * (index + 0.5) / count for index in range(count)]


def _rounded_vector(values: tuple[float, ...] | list[float]) -> list[float]:
    return [round(float(value), 4) for value in values]


def _region_id(
    region_class: str, bounds: dict[str, list[float]], ordinal: int, inspection_geometry_hash: str
) -> str:
    digest = content_hash([inspection_geometry_hash, region_class, bounds, ordinal])[:18]
    return f"atlas-region-{digest}"


def _cell_id(region_id: str, ordinal: int, pose: dict[str, Any]) -> str:
    return f"atlas-cell-{content_hash([region_id, ordinal, pose])[:18]}"


def _pose(position: tuple[float, float, float], yaw_deg: float, pitch_deg: float) -> dict[str, Any]:
    return {
        "position": _rounded_vector(position),
        "yaw_deg": round(yaw_deg, 4),
        "pitch_deg": round(pitch_deg, 4),
    }


def _sampling_policy_candidate(sampling_policy: dict[str, Any] | None) -> dict[str, Any]:
    """Return a recognized immutable-by-copy public sampling policy."""

    candidate = _SAMPLING_POLICY if sampling_policy is None else sampling_policy
    policy_id = candidate.get("policy_id") if isinstance(candidate, dict) else None
    expected = _SAMPLING_POLICY_CANDIDATES.get(str(policy_id))
    if expected is None or candidate != expected:
        raise ValueError("inspection atlas sampling policy is not a recognized candidate")
    return copy.deepcopy(expected)


def inspection_sampling_policy(policy_id: str) -> dict[str, Any]:
    """Return a copy of a public, versioned sampling calibration candidate."""

    candidate = _SAMPLING_POLICY_CANDIDATES.get(policy_id)
    if candidate is None:
        raise ValueError(f"unknown inspection-atlas sampling policy: {policy_id}")
    return copy.deepcopy(candidate)


def _inspection_parameters(
    execution_contract: dict[str, Any], sampling_policy: dict[str, Any]
) -> dict[str, Any]:
    observe = execution_contract["observe"]
    vehicle = execution_contract["vehicle"]
    maximum_range = _finite_number(observe["max_range_m"], "observe.max_range_m")
    horizontal_fov = _finite_number(observe["horizontal_fov_deg"], "observe.horizontal_fov_deg")
    vertical_fov = _finite_number(observe["vertical_fov_deg"], "observe.vertical_fov_deg")
    dwell = _finite_number(observe["continuous_dwell_s"], "observe.continuous_dwell_s")
    radius = _finite_number(vehicle["radius_m"], "vehicle.radius_m")
    clearance = _finite_number(vehicle["minimum_clearance_m"], "vehicle.minimum_clearance_m")
    if maximum_range <= 0.0 or dwell <= 0.0 or radius <= 0.0 or clearance <= 0.0:
        raise ValueError("inspection-atlas execution parameters must be positive")
    if not 0.0 < horizontal_fov <= 180.0 or not 0.0 < vertical_fov <= 180.0:
        raise ValueError("inspection-atlas field of view must lie in (0, 180]")

    # These dimensionless factors are a versioned calibration candidate in the
    # public atlas.  They are deliberately not described as physical truths;
    # the formal gate remains closed until method-independent calibration.
    clearance_buffer = float(sampling_policy["clearance_buffer_m"])
    range_fraction = float(sampling_policy["nominal_standoff_range_fraction"])
    spacing_fraction = float(sampling_policy["footprint_spacing_fraction"])
    minimum_spacing = float(sampling_policy["minimum_cell_spacing_m"])
    minimum_standoff = radius + clearance + clearance_buffer
    stand_off = max(
        minimum_standoff,
        min(maximum_range * range_fraction, maximum_range - clearance_buffer),
    )
    if stand_off >= maximum_range:
        raise ValueError("observation range leaves no public inspection stand-off")
    horizontal_spacing = max(
        minimum_spacing,
        2.0 * stand_off * math.tan(math.radians(horizontal_fov) / 2.0) * spacing_fraction,
    )
    vertical_spacing = max(
        minimum_spacing,
        2.0 * stand_off * math.tan(math.radians(vertical_fov) / 2.0) * spacing_fraction,
    )
    return {
        "maximum_range_m": maximum_range,
        "horizontal_fov_deg": horizontal_fov,
        "vertical_fov_deg": vertical_fov,
        "continuous_dwell_s": dwell,
        "minimum_clearance_m": clearance,
        "body_margin_m": radius + clearance,
        "nominal_standoff_m": stand_off,
        "horizontal_cell_spacing_m": horizontal_spacing,
        "vertical_cell_spacing_m": vertical_spacing,
        "sampling_policy": sampling_policy,
    }


def _region(
    region_class: str,
    bounds: tuple[float, float, float, float, float, float],
    area_m2: float,
    inspection_geometry_hash: str,
    ordinal: int,
) -> dict[str, Any]:
    x0, x1, y0, y1, z0, z1 = bounds
    public_bounds = {
        "minimum": _rounded_vector((x0, y0, z0)),
        "maximum": _rounded_vector((x1, y1, z1)),
    }
    center_z = (z0 + z1) / 2.0
    return {
        "region_id": _region_id(region_class, public_bounds, ordinal, inspection_geometry_hash),
        "region_class": region_class,
        "bounds": public_bounds,
        "represented_area_m2": round(max(0.01, area_m2), 4),
        "altitude_band": _altitude_band(center_z),
        "cells": [],
    }


def _append_cell(
    region: dict[str, Any],
    position: tuple[float, float, float],
    yaw_deg: float,
    pitch_deg: float,
    normal: tuple[float, float, float],
    parameters: dict[str, Any],
    *,
    surface_point: tuple[float, float, float] | None = None,
) -> None:
    pose = _pose(position, yaw_deg, pitch_deg)
    public_surface = surface_point or tuple(
        position[index] - normal[index] * parameters["nominal_standoff_m"]
        for index in range(3)
    )
    nominal_standoff = math.dist(position, public_surface)
    ordinal = len(region["cells"])
    region["cells"].append(
        {
            "cell_id": _cell_id(str(region["region_id"]), ordinal, pose),
            "pose": pose,
            "surface_point": _rounded_vector(public_surface),
            "surface_normal": _rounded_vector(normal),
            "represented_area_m2": round(
                float(region["represented_area_m2"]) / max(1, ordinal + 1), 4
            ),
            "pose_envelope": {
                "nominal_standoff_m": round(nominal_standoff, 4),
                "lateral_tolerance_m": round(parameters["horizontal_cell_spacing_m"] * 0.25, 4),
                "vertical_tolerance_m": round(parameters["vertical_cell_spacing_m"] * 0.25, 4),
            },
        }
    )


def _facade_regions(
    component: dict[str, Any],
    inspection_geometry_hash: str,
    ordinal: int,
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    x0, x1, y0, y1, z0, z1 = _component_bounds(component)
    regions: list[dict[str, Any]] = []
    facade_specs = (
        ("south", (0.0, -1.0, 0.0), y0, "y", x0, x1, x1 - x0),
        ("north", (0.0, 1.0, 0.0), y1, "y", x0, x1, x1 - x0),
        ("west", (-1.0, 0.0, 0.0), x0, "x", y0, y1, y1 - y0),
        ("east", (1.0, 0.0, 0.0), x1, "x", y0, y1, y1 - y0),
    )
    stand_off = parameters["nominal_standoff_m"]
    for face_index, (_, normal, fixed, fixed_axis, along_low, along_high, width) in enumerate(
        facade_specs
    ):
        region = _region(
            "facade",
            (x0, x1, y0, y1, z0, z1),
            width * (z1 - z0),
            inspection_geometry_hash,
            ordinal + face_index,
        )
        yaw_deg = math.degrees(math.atan2(-normal[1], -normal[0]))
        for z_value in _centered_values(
            z0 + float(parameters["sampling_policy"]["facade_surface_inset_m"]),
            z1 - float(parameters["sampling_policy"]["facade_surface_inset_m"]),
            parameters["vertical_cell_spacing_m"],
        ):
            for along in _centered_values(
                along_low + float(parameters["sampling_policy"]["facade_surface_inset_m"]),
                along_high - float(parameters["sampling_policy"]["facade_surface_inset_m"]),
                parameters["horizontal_cell_spacing_m"],
            ):
                surface = (fixed, along, z_value) if fixed_axis == "x" else (along, fixed, z_value)
                position = tuple(surface[index] + normal[index] * stand_off for index in range(3))
                _append_cell(region, position, yaw_deg, 0.0, normal, parameters)
        if region["cells"]:
            regions.append(region)
    return regions


def _roof_region(
    component: dict[str, Any],
    inspection_geometry_hash: str,
    ordinal: int,
    parameters: dict[str, Any],
) -> dict[str, Any] | None:
    x0, x1, y0, y1, _, z1 = _component_bounds(component)
    region = _region(
        "roof",
        (x0, x1, y0, y1, z1, z1),
        (x1 - x0) * (y1 - y0),
        inspection_geometry_hash,
        ordinal,
    )
    spacing = parameters["horizontal_cell_spacing_m"]
    facade_inset = float(parameters["sampling_policy"]["facade_surface_inset_m"])
    for x_value in _centered_values(x0 + facade_inset, x1 - facade_inset, spacing):
        for y_value in _centered_values(y0 + facade_inset, y1 - facade_inset, spacing):
            _append_cell(
                region,
                (x_value, y_value, z1 + parameters["nominal_standoff_m"]),
                0.0,
                -90.0,
                (0.0, 0.0, 1.0),
                parameters,
            )
    return region if region["cells"] else None


def _entrance_region(
    entrance: object,
    building: dict[str, Any],
    inspection_geometry_hash: str,
    ordinal: int,
    parameters: dict[str, Any],
) -> dict[str, Any] | None:
    x, y, z = _vector(entrance, 3, "building.entrance")
    footprint_x, footprint_y, _, _ = _vector(building["footprint"], 4, "building.footprint")
    delta_x, delta_y = x - footprint_x, y - footprint_y
    if abs(delta_x) >= abs(delta_y):
        normal = (1.0 if delta_x >= 0.0 else -1.0, 0.0, 0.0)
    else:
        normal = (0.0, 1.0 if delta_y >= 0.0 else -1.0, 0.0)
    half_width = float(parameters["sampling_policy"]["entrance_half_width_m"])
    region = _region(
        "entrance",
        (x - half_width, x + half_width, y - half_width, y + half_width, z - 1.0, z + 1.0),
        6.0,
        inspection_geometry_hash,
        ordinal,
    )
    position = (
        x,
        y,
        z,
    )
    surface = (
        x,
        y,
        z + float(parameters["sampling_policy"]["entrance_height_offset_m"]),
    )
    minimum_body_z = (
        parameters["flight_minimum_z_m"]
        + parameters["body_margin_m"]
        + float(parameters["sampling_policy"]["flight_bound_buffer_m"])
    )
    body_z = max(surface[2], minimum_body_z)
    vertical_offset = body_z - surface[2]
    maximum_distance = parameters["maximum_range_m"] - float(
        parameters["sampling_policy"]["flight_bound_buffer_m"]
    )
    maximum_horizontal = math.sqrt(max(0.0, maximum_distance**2 - vertical_offset**2))
    horizontal_standoff = min(
        maximum_horizontal,
        max(
            parameters["nominal_standoff_m"],
            parameters["maximum_range_m"]
            * float(parameters["sampling_policy"]["entrance_standoff_range_fraction"]),
        ),
    )
    if horizontal_standoff <= parameters["body_margin_m"]:
        return None
    position = (
        x + normal[0] * horizontal_standoff,
        y + normal[1] * horizontal_standoff,
        body_z,
    )
    yaw_deg = math.degrees(math.atan2(-normal[1], -normal[0]))
    pitch_deg = math.degrees(math.atan2(surface[2] - body_z, horizontal_standoff))
    _append_cell(
        region,
        position,
        yaw_deg,
        pitch_deg,
        normal,
        parameters,
        surface_point=surface,
    )
    return region


def _rubble_region(
    obstacle: dict[str, Any],
    inspection_geometry_hash: str,
    ordinal: int,
    parameters: dict[str, Any],
) -> dict[str, Any] | None:
    x0, x1, y0, y1, _, z1 = _component_bounds(obstacle)
    region = _region(
        "rubble",
        (x0, x1, y0, y1, z1, z1),
        (x1 - x0) * (y1 - y0),
        inspection_geometry_hash,
        ordinal,
    )
    spacing = parameters["horizontal_cell_spacing_m"]
    rubble_inset = float(parameters["sampling_policy"]["rubble_surface_inset_m"])
    for x_value in _centered_values(x0 + rubble_inset, x1 - rubble_inset, spacing):
        for y_value in _centered_values(y0 + rubble_inset, y1 - rubble_inset, spacing):
            _append_cell(
                region,
                (x_value, y_value, z1 + parameters["nominal_standoff_m"]),
                0.0,
                -90.0,
                (0.0, 0.0, 1.0),
                parameters,
            )
    return region if region["cells"] else None


def _safe_sky_altitude(city: dict[str, Any], execution_contract: dict[str, Any]) -> float:
    vehicle = execution_contract["vehicle"]
    body_margin = _finite_number(vehicle["radius_m"], "vehicle.radius_m") + _finite_number(
        vehicle["minimum_clearance_m"], "vehicle.minimum_clearance_m"
    )
    ceilings = [
        _component_bounds(component)[5]
        for building in city["buildings"]
        for component in building["components"]
    ] + [_component_bounds(obstacle)[5] for obstacle in city["obstacles"]]
    if not ceilings:
        raise ValueError("inspection atlas requires at least one structural collider")
    safe_sky = max(ceilings) + body_margin + 0.5
    maximum_z = (
        _vector(city["flight_bounds"]["maximum"], 3, "flight_bounds.maximum")[2]
        - body_margin
    )
    if safe_sky > maximum_z + 1.0e-9:
        raise ValueError("inspection atlas has no public safe-sky transit lane")
    return round(safe_sky, 4)


def _inspection_geometry_hash(city: dict[str, Any]) -> str:
    """Hash only the public, target-independent atlas authority geometry.

    ``task_geometry_hash`` remains the existing private task-layer identity and
    includes legacy target-support metadata.  Atlas IDs must not inherit it:
    otherwise a target-layer policy edit could change public route IDs without
    changing any declared inspection structure.
    """

    buildings = []
    for building in sorted(city["buildings"], key=lambda item: str(item["id"])):
        buildings.append(
            {
                "id": str(building["id"]),
                "footprint": list(building["footprint"]),
                "entrances": [list(entrance) for entrance in building.get("entrances", [])],
                "components": [
                    {
                        "id": str(component["id"]),
                        "center": list(component["center"]),
                        "size": list(component["size"]),
                        "structural_role": component.get("structural_role"),
                    }
                    for component in sorted(
                        building["components"], key=lambda item: str(item["id"])
                    )
                ],
            }
        )
    obstacles = [
        {
            "id": str(obstacle["id"]),
            "kind": str(obstacle.get("kind", "unknown")),
            "center": list(obstacle["center"]),
            "size": list(obstacle["size"]),
        }
        for obstacle in sorted(city["obstacles"], key=lambda item: str(item["id"]))
    ]
    return content_hash(
        {
            "atlas_version": ATLAS_VERSION,
            "buildings": buildings,
            "obstacles": obstacles,
            "flight_bounds": city["flight_bounds"],
        }
    )


def _inside_flight_bounds(
    point: Vec3, city: dict[str, Any], body_margin: float
) -> bool:
    minimum = _vector(city["flight_bounds"]["minimum"], 3, "flight_bounds.minimum")
    maximum = _vector(city["flight_bounds"]["maximum"], 3, "flight_bounds.maximum")
    return all(
        low + body_margin - 1.0e-9 <= value <= high - body_margin + 1.0e-9
        for value, low, high in zip(point, minimum, maximum, strict=True)
    )


def _geometrically_admit_cells(
    city: dict[str, Any],
    regions: list[dict[str, Any]],
    execution_contract: dict[str, Any],
    safe_sky_altitude_m: float,
    sampling_policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove public cells that fail the same geometry used by strict L0 credit.

    This compiler consumes only public city colliders and the public execution
    contract.  It does not prove CF2X dynamics, but it prevents impossible AABB
    obligations from entering either the coverage denominator or a method ABI.
    """

    colliders = colliders_from_city(city)
    observe = execution_contract["observe"]
    vehicle = execution_contract["vehicle"]
    body_margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    flight_bound_margin = body_margin + float(sampling_policy["flight_bound_buffer_m"])
    rejection_counts: Counter[str] = Counter()
    rejection_counts_by_region_class: dict[str, Counter[str]] = {}
    candidate_count = 0
    admitted_count = 0
    admitted_regions = []
    for region in regions:
        candidates = list(region["cells"])
        candidate_count += len(candidates)
        if not candidates:
            continue
        represented_area_per_candidate = float(region["represented_area_m2"]) / len(candidates)
        admitted = []
        for cell in candidates:
            pose = Pose3D.from_dict(cell["pose"])
            normal = _vec3(cell["surface_normal"], "inspection cell surface normal")
            surface_point = _vec3(cell["surface_point"], "inspection cell surface point")
            reasons = set()
            if not _inside_flight_bounds(pose.position, city, flight_bound_margin):
                reasons.add("body_outside_flight_bounds")
            clearance, _ = minimum_clearance(pose.position, colliders)
            if clearance + 1.0e-9 < body_margin:
                reasons.add("body_clearance")
            camera = _inspection_camera_pose(pose, execution_contract)
            if distance(camera.position, surface_point) > float(observe["max_range_m"]):
                reasons.add("sensor_range")
            in_view, _, _ = in_field_of_view(
                camera,
                surface_point,
                float(observe["horizontal_fov_deg"]),
                float(observe["vertical_fov_deg"]),
            )
            if not in_view:
                reasons.add("sensor_fov")
            facing, _ = surface_facing(
                camera.position,
                surface_point,
                normal,
                float(observe["surface_facing_min_cosine"]),
            )
            if not facing:
                reasons.add("surface_facing")
            visible, _ = line_of_sight(camera.position, surface_point, colliders)
            if not visible:
                reasons.add("surface_los")
            sky_point = (pose.position[0], pose.position[1], safe_sky_altitude_m)
            climb_clearance, _ = minimum_segment_clearance(
                pose.position, sky_point, colliders
            )
            if climb_clearance + 1.0e-9 < body_margin:
                reasons.add("safe_sky_climb_clearance")
            if reasons:
                rejection_counts.update(reasons)
                rejection_counts_by_region_class.setdefault(
                    str(region["region_class"]), Counter()
                ).update(reasons)
                continue
            cell["represented_area_m2"] = round(represented_area_per_candidate, 4)
            admitted.append(cell)
        if admitted:
            region["cells"] = admitted
            region["represented_area_m2"] = round(
                represented_area_per_candidate * len(admitted), 4
            )
            admitted_count += len(admitted)
            admitted_regions.append(region)
    if not admitted_regions or admitted_count == 0:
        raise ValueError("public geometry admission removed every inspection cell")
    return admitted_regions, {
        "schema": "org.aerocity.bench.inspection-atlas-geometric-admission-public.v1",
        "candidate_cell_count": candidate_count,
        "admitted_cell_count": admitted_count,
        "rejected_cell_count": candidate_count - admitted_count,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejection_counts_by_region_class": {
            region_class: dict(sorted(counts.items()))
            for region_class, counts in sorted(rejection_counts_by_region_class.items())
        },
        "flight_bound_margin_m": round(flight_bound_margin, 4),
        "admission_semantics": (
            "public_aabb_bounds_clearance_sensor_geometry_los_and_direct_safe_sky_path"
        ),
        "native_cf2x_validation_required": True,
    }


def _transit_graph(
    regions: list[dict[str, Any]],
    safe_sky_altitude_m: float,
    city: dict[str, Any],
    flight_bound_margin_m: float,
) -> dict[str, Any]:
    minimum = _vector(city["flight_bounds"]["minimum"], 3, "flight_bounds.minimum")
    maximum = _vector(city["flight_bounds"]["maximum"], 3, "flight_bounds.maximum")
    if flight_bound_margin_m <= 0.0:
        raise ValueError("transit graph flight-bound margin must be positive")
    x_min, x_max = minimum[0] + flight_bound_margin_m, maximum[0] - flight_bound_margin_m
    y_min, y_max = minimum[1] + flight_bound_margin_m, maximum[1] - flight_bound_margin_m
    if x_min > x_max or y_min > y_max:
        raise ValueError("flight bounds leave no safe-sky transit footprint")
    nodes = []
    for region in regions:
        lower = _vector(region["bounds"]["minimum"], 3, "region.bounds.minimum")
        upper = _vector(region["bounds"]["maximum"], 3, "region.bounds.maximum")
        # Region bounds can touch the city boundary even when every admitted
        # inspection cell has a valid standoff.  Keep the public transit
        # waypoint inside the same body-plus-buffer envelope used by cell
        # admission instead of emitting an unreachable route node.
        center_x = min(max((lower[0] + upper[0]) / 2.0, x_min), x_max)
        center_y = min(max((lower[1] + upper[1]) / 2.0, y_min), y_max)
        nodes.append(
            {
                "node_id": f"atlas-transit-{content_hash(region['region_id'])[:18]}",
                "region_id": region["region_id"],
                "position": _rounded_vector(
                    (center_x, center_y, safe_sky_altitude_m)
                ),
            }
        )
    nodes.sort(key=lambda item: str(item["node_id"]))
    positions = {str(node["node_id"]): tuple(node["position"][:2]) for node in nodes}
    edge_pairs: set[tuple[str, str]] = set()
    all_pairs = sorted(
        (
            (
                math.dist(positions[str(first["node_id"])], positions[str(second["node_id"])]),
                tuple(sorted((str(first["node_id"]), str(second["node_id"])))),
            )
            for first_index, first in enumerate(nodes)
            for second in nodes[first_index + 1 :]
            if math.dist(
                positions[str(first["node_id"])], positions[str(second["node_id"])]
            )
            > 1.0e-9
        ),
        key=lambda item: (item[0], item[1]),
    )

    # A nearest-neighbour graph is not guaranteed to be connected.  Build a
    # deterministic minimum spanning backbone first, then retain local edges
    # for useful route alternatives.  Nodes at the same projection connect to
    # a common non-zero-distance neighbour instead of introducing zero edges.
    parent = {str(node["node_id"]): str(node["node_id"]) for node in nodes}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(first_id: str, second_id: str) -> bool:
        first_root, second_root = find(first_id), find(second_id)
        if first_root == second_root:
            return False
        parent[max(first_root, second_root)] = min(first_root, second_root)
        return True

    for _, pair in all_pairs:
        if union(pair[0], pair[1]):
            edge_pairs.add(pair)
    if len({find(node_id) for node_id in parent}) != 1:
        raise ValueError("inspection-atlas transit nodes have no connected non-zero graph")

    for node in nodes:
        candidates = sorted(
            (
                (
                    math.dist(tuple(node["position"][:2]), tuple(other["position"][:2])),
                    str(other["node_id"]),
                )
                for other in nodes
                if other["node_id"] != node["node_id"]
            ),
            key=lambda item: (item[0], item[1]),
        )
        # Facades and roofs of one component can share a sky projection.  A
        # zero-length edge conveys no transfer cost and would violate the
        # public graph invariant, so retain only distinct public waypoints.
        for _distance_m, other_id in [item for item in candidates if item[0] > 1.0e-9][:3]:
            edge_pairs.add(tuple(sorted((str(node["node_id"]), other_id))))
    edges = [
        {
            "edge_id": f"atlas-edge-{content_hash(pair)[:18]}",
            "start_node_id": pair[0],
            "end_node_id": pair[1],
            "safe_sky_distance_m": round(
                math.dist(positions[pair[0]], positions[pair[1]]), 4
            ),
        }
        for pair in sorted(edge_pairs)
    ]
    return {
        "schema": "org.aerocity.bench.inspection-atlas-transit-public.v1",
        "safe_sky_altitude_m": safe_sky_altitude_m,
        "nodes": nodes,
        "edges": sorted(edges, key=lambda item: str(item["edge_id"])),
    }


def compile_inspection_atlas(
    city: dict[str, Any],
    execution_contract: dict[str, Any],
    *,
    sampling_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a deterministic, target-independent G2-I inspection atlas.

    ``city`` may contain private generation metadata.  This compiler never
    reads it and its output is invariant to target-process/episode additions.
    Runtime collision and evaluator checks remain authoritative: a public
    inspection cell is a declared inspection obligation, never a legal target
    witness or an automatic confirmation guarantee.
    """

    selected_policy = _sampling_policy_candidate(sampling_policy)
    inspection_geometry_hash = _inspection_geometry_hash(city)
    parameters = _inspection_parameters(execution_contract, selected_policy)
    parameters["flight_minimum_z_m"] = _vector(
        city["flight_bounds"]["minimum"], 3, "flight_bounds.minimum"
    )[2]
    regions: list[dict[str, Any]] = []
    ordinal = 0
    for building in sorted(city["buildings"], key=lambda item: str(item["id"])):
        components = sorted(building["components"], key=lambda item: str(item["id"]))
        for component in components:
            # Architectural detail is intentionally excluded because it is not
            # a public inspection surface in v1.  Do not consult the legacy
            # target-layer flag here: target eligibility must not decide what a
            # public method is asked to inspect.
            if component.get("structural_role") is not None:
                continue
            roof = _roof_region(component, inspection_geometry_hash, ordinal, parameters)
            ordinal += 1
            if roof is not None:
                regions.append(roof)
            facade_regions = _facade_regions(
                component, inspection_geometry_hash, ordinal, parameters
            )
            ordinal += 4
            regions.extend(facade_regions)
        for entrance in building.get("entrances", []):
            region = _entrance_region(
                entrance, building, inspection_geometry_hash, ordinal, parameters
            )
            ordinal += 1
            if region is not None:
                regions.append(region)
    for obstacle in sorted(city["obstacles"], key=lambda item: str(item["id"])):
        # Obstacles are publicly declared physical debris/barriers.  As above,
        # never use the legacy target-layer support_domain flag to decide atlas
        # membership.  The private target process may later select none, one,
        # or many of these public regions.
        region = _rubble_region(obstacle, inspection_geometry_hash, ordinal, parameters)
        ordinal += 1
        if region is not None:
            regions.append(region)
    regions.sort(key=lambda item: str(item["region_id"]))
    if not regions or not any(region["cells"] for region in regions):
        raise ValueError("inspection atlas contains no public inspection cells")
    safe_sky_altitude = _safe_sky_altitude(city, execution_contract)
    flight_bound_margin = (
        parameters["body_margin_m"] + float(selected_policy["flight_bound_buffer_m"])
    )
    regions, geometric_admission = _geometrically_admit_cells(
        city, regions, execution_contract, safe_sky_altitude, selected_policy
    )

    atlas = {
        "schema": ATLAS_SCHEMA,
        "atlas_version": ATLAS_VERSION,
        "layout_id": str(city["layout_id"]),
        "inspection_geometry_hash": inspection_geometry_hash,
        "sampling_policy": selected_policy,
        "geometric_admission": geometric_admission,
        "observation_contract": {
            "maximum_range_m": round(parameters["maximum_range_m"], 4),
            "horizontal_fov_deg": round(parameters["horizontal_fov_deg"], 4),
            "vertical_fov_deg": round(parameters["vertical_fov_deg"], 4),
            "continuous_dwell_s": round(parameters["continuous_dwell_s"], 4),
            "minimum_clearance_m": round(parameters["minimum_clearance_m"], 4),
            "nominal_standoff_m": round(parameters["nominal_standoff_m"], 4),
            "horizontal_cell_spacing_m": round(parameters["horizontal_cell_spacing_m"], 4),
            "vertical_cell_spacing_m": round(parameters["vertical_cell_spacing_m"], 4),
        },
        "regions": regions,
        "transit_graph": _transit_graph(
            regions,
            safe_sky_altitude,
            city,
            flight_bound_margin,
        ),
        "runtime_validation_required": True,
    }
    atlas["atlas_hash"] = content_hash(atlas)
    validate_public_inspection_atlas(atlas)
    return atlas


def _mission_motion_time_s(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    *,
    horizontal_speed_mps: float,
    vertical_speed_mps: float,
) -> float:
    horizontal = math.hypot(second[0] - first[0], second[1] - first[1])
    vertical = abs(second[2] - first[2])
    return max(horizontal / horizontal_speed_mps, vertical / vertical_speed_mps)


def _mission_route_lower_bound_s(
    start: tuple[float, float, float],
    cells: list[dict[str, Any]],
    *,
    safe_sky_altitude_m: float,
    horizontal_speed_mps: float,
    vertical_speed_mps: float,
    dwell_charge_s: float,
    return_reserve_s: float,
) -> float:
    """Optimistic public lower bound for scan, return, and frozen reserve."""

    current = start
    current_region_id: str | None = None
    total = 0.0
    for cell in cells:
        position = _vec3(cell["pose"]["position"], "mission cell pose")
        region_id = str(cell.get("_mission_region_id", ""))
        if current_region_id is None or region_id != current_region_id:
            current_sky = (current[0], current[1], safe_sky_altitude_m)
            target_sky = (position[0], position[1], safe_sky_altitude_m)
            total += _mission_motion_time_s(
                current,
                current_sky,
                horizontal_speed_mps=horizontal_speed_mps,
                vertical_speed_mps=vertical_speed_mps,
            )
            total += _mission_motion_time_s(
                current_sky,
                target_sky,
                horizontal_speed_mps=horizontal_speed_mps,
                vertical_speed_mps=vertical_speed_mps,
            )
            total += _mission_motion_time_s(
                target_sky,
                position,
                horizontal_speed_mps=horizontal_speed_mps,
                vertical_speed_mps=vertical_speed_mps,
            )
        else:
            total += _mission_motion_time_s(
                current,
                position,
                horizontal_speed_mps=horizontal_speed_mps,
                vertical_speed_mps=vertical_speed_mps,
            )
        total += dwell_charge_s
        current = position
        current_region_id = region_id
    if cells:
        current_sky = (current[0], current[1], safe_sky_altitude_m)
        home_sky = (start[0], start[1], safe_sky_altitude_m)
        total += _mission_motion_time_s(
            current,
            current_sky,
            horizontal_speed_mps=horizontal_speed_mps,
            vertical_speed_mps=vertical_speed_mps,
        )
        total += _mission_motion_time_s(
            current_sky,
            home_sky,
            horizontal_speed_mps=horizontal_speed_mps,
            vertical_speed_mps=vertical_speed_mps,
        )
        total += _mission_motion_time_s(
            home_sky,
            start,
            horizontal_speed_mps=horizontal_speed_mps,
            vertical_speed_mps=vertical_speed_mps,
        )
    return total + return_reserve_s


def _evenly_spaced_cells(region: dict[str, Any], maximum: int) -> list[dict[str, Any]]:
    cells = list(region.get("cells", []))
    if maximum < 1 or len(cells) <= maximum:
        return cells
    indices = sorted(
        {
            min(len(cells) - 1, math.floor(index * len(cells) / maximum))
            for index in range(maximum)
        }
    )
    return [cells[index] for index in indices]


def compile_public_mission_sector(
    atlas: dict[str, Any],
    starts: list[dict[str, Any]],
    execution_contract: dict[str, Any],
    *,
    region_allowlist: set[str] | None = None,
    require_all_allowed_regions: bool = False,
) -> dict[str, Any]:
    """Compile a target-independent, budget-bracketed public search sector.

    The sector is frozen from public geometry and public starts before private
    targets are sampled.  It selects inspection obligations, never target
    locations or legal witnesses.  The capacity certificate is deliberately
    an optimistic lower bound; native CF2X replay remains mandatory.
    """

    validate_public_inspection_atlas(atlas)
    if not starts:
        raise ValueError("mission sector requires public fleet starts")
    start_positions: dict[str, tuple[float, float, float]] = {}
    for start in starts:
        drone_id = str(start.get("drone_id", ""))
        if not drone_id or drone_id in start_positions:
            raise ValueError("mission sector start IDs must be unique")
        start_positions[drone_id] = _vec3(start.get("position"), "mission start.position")

    episode = execution_contract["episode"]
    observe = execution_contract["observe"]
    vehicle = execution_contract["vehicle"]
    duration_s = _finite_number(episode["duration_s"], "episode.duration_s")
    return_reserve_s = _finite_number(
        episode["return_reserve_s"], "episode.return_reserve_s"
    )
    period_s = _finite_number(execution_contract["control_period_s"], "control_period_s")
    dwell_s = _finite_number(observe["continuous_dwell_s"], "observe.continuous_dwell_s")
    dwell_charge_s = (math.ceil(dwell_s / period_s) + 1) * period_s
    # These conservative reference speeds match the transparent public
    # inspector and stay below the task's vehicle caps.
    horizontal_speed = min(1.5, _finite_number(vehicle["horizontal_speed_mps"], "horizontal speed"))
    vertical_speed = min(1.0, _finite_number(vehicle["vertical_speed_mps"], "vertical speed"))
    if min(duration_s, period_s, dwell_s, horizontal_speed, vertical_speed) <= 0.0:
        raise ValueError("mission-sector timing and speeds must be positive")
    if not 0.0 < return_reserve_s < duration_s:
        raise ValueError("mission-sector return reserve must lie inside the episode")
    safe_sky = _finite_number(
        atlas["transit_graph"]["safe_sky_altitude_m"], "mission safe-sky altitude"
    )

    regions = [region for region in atlas["regions"] if region.get("cells")]
    if region_allowlist is not None:
        allowed = {str(value) for value in region_allowlist}
        if not allowed:
            raise ValueError("mission sector region allowlist must not be empty")
        known = {str(region["region_id"]) for region in regions}
        unknown = allowed - known
        if unknown:
            raise ValueError(
                "mission sector region allowlist references unknown regions: "
                + ", ".join(sorted(unknown))
            )
        regions = [region for region in regions if str(region["region_id"]) in allowed]
    elif require_all_allowed_regions:
        raise ValueError("requiring all mission regions needs an explicit region allowlist")
    if not regions:
        raise ValueError("mission sector has no public inspection regions")
    region_by_id = {str(region["region_id"]): region for region in regions}
    if len(region_by_id) != len(regions):
        raise ValueError("mission sector requires unique public region IDs")

    def region_center(region: dict[str, Any]) -> tuple[float, float, float]:
        lower = _vec3(region["bounds"]["minimum"], "region minimum")
        upper = _vec3(region["bounds"]["maximum"], "region maximum")
        return tuple((low + high) / 2.0 for low, high in zip(lower, upper, strict=True))  # type: ignore[return-value]

    # Seed the public task with structural and altitude diversity.  This is a
    # task-domain prior, not a target-process realization: the same selection
    # is produced after arbitrary target metadata changes.
    selected_regions: list[dict[str, Any]] = []
    available = list(regions)
    altitude_bands = sorted({str(region["altitude_band"]) for region in available})
    desired_bands = altitude_bands[:]
    for band in desired_bands:
        candidates = [region for region in available if str(region["altitude_band"]) == band]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda region: (
                min(distance(region_center(region), start) for start in start_positions.values()),
                str(region["region_class"]),
                str(region["region_id"]),
            ),
        )
        selected_regions.append(selected)
        available.remove(selected)
    for region_class in sorted(REGION_CLASSES):
        if any(str(region["region_class"]) == region_class for region in selected_regions):
            continue
        candidates = [region for region in available if str(region["region_class"]) == region_class]
        if candidates:
            selected = min(
                candidates,
                key=lambda region: (
                    min(
                        distance(region_center(region), start)
                        for start in start_positions.values()
                    ),
                    str(region["region_id"]),
                ),
            )
            selected_regions.append(selected)
            available.remove(selected)
    available.sort(
        key=lambda region: (
            min(distance(region_center(region), start) for start in start_positions.values()),
            str(region["region_class"]),
            str(region["region_id"]),
        )
    )
    selected_regions.extend(available)

    assigned: dict[str, list[dict[str, Any]]] = {drone_id: [] for drone_id in start_positions}
    selected_cell_ids: set[str] = set()
    selected_region_ids: set[str] = set()
    # Hidden-target search is a partial-coverage task, not an instruction to
    # exhaust every public obligation.  The capacity certificate is nevertheless
    # a hard per-vehicle execution bound: route motion, discrete dwell, return
    # motion, and the declared reserve must fit inside the episode duration.
    # Earlier development used 1.35x here, which admitted 405-second lower
    # bounds into a 300-second episode; that candidate is retired.
    capacity_fraction = 1.0
    capacity_limit_s = duration_s * capacity_fraction
    maximum_cells_per_region = 96
    if require_all_allowed_regions:
        best_allocation: dict[str, list[dict[str, Any]]] | None = None
        low, high = 1, maximum_cells_per_region
        while low <= high:
            cap = (low + high) // 2
            candidate_assignment: dict[str, list[dict[str, Any]]] = {
                drone_id: [] for drone_id in start_positions
            }
            complete = True
            for region in selected_regions:
                cells = [
                    {**cell, "_mission_region_id": str(region["region_id"])}
                    for cell in _evenly_spaced_cells(region, cap)
                ]
                owner_order = sorted(
                    start_positions,
                    key=lambda drone_id: (
                        distance(region_center(region), start_positions[drone_id]),
                        len(candidate_assignment[drone_id]),
                        drone_id,
                    ),
                )
                accepted_owner = next(
                    (
                        drone_id
                        for drone_id in owner_order
                        if _mission_route_lower_bound_s(
                            start_positions[drone_id],
                            candidate_assignment[drone_id] + cells,
                            safe_sky_altitude_m=safe_sky,
                            horizontal_speed_mps=horizontal_speed,
                            vertical_speed_mps=vertical_speed,
                            dwell_charge_s=dwell_charge_s,
                            return_reserve_s=return_reserve_s,
                        )
                        <= capacity_limit_s + 1.0e-9
                    ),
                    None,
                )
                if accepted_owner is None:
                    complete = False
                    break
                candidate_assignment[accepted_owner].extend(cells)
            if complete and all(candidate_assignment.values()):
                best_allocation = candidate_assignment
                low = cap + 1
            else:
                high = cap - 1
        if best_allocation is None:
            raise ValueError(
                "mission-sector compiler cannot retain every required public region"
            )
        assigned = best_allocation
        selected_cell_ids = {
            str(cell["cell_id"])
            for cells in assigned.values()
            for cell in cells
        }
        selected_region_ids = {
            str(cell["_mission_region_id"])
            for cells in assigned.values()
            for cell in cells
        }
        if selected_region_ids != {str(value) for value in region_allowlist or set()}:
            raise ValueError("mission-sector required region cohort was not preserved")
    else:
        for region in selected_regions:
            cells = [
                {**cell, "_mission_region_id": str(region["region_id"])}
                for cell in _evenly_spaced_cells(region, maximum_cells_per_region)
            ]
            region_with_route_metadata = {**region, "cells": cells}
            owner_order = sorted(
                start_positions,
                key=lambda drone_id: (
                    distance(region_center(region), start_positions[drone_id]),
                    len(assigned[drone_id]),
                    drone_id,
                ),
            )
            accepted_owner: str | None = None
            accepted_cells: list[dict[str, Any]] = []
            for drone_id in owner_order:
                # Add the largest evenly-spaced prefix that preserves the complete
                # scan+return lower-bound certificate for this vehicle.
                low, high = 1, len(cells)
                best: list[dict[str, Any]] = []
                while low <= high:
                    count = (low + high) // 2
                    candidate_cells = assigned[drone_id] + _evenly_spaced_cells(
                        region_with_route_metadata, count
                    )
                    required = _mission_route_lower_bound_s(
                        start_positions[drone_id],
                        candidate_cells,
                        safe_sky_altitude_m=safe_sky,
                        horizontal_speed_mps=horizontal_speed,
                        vertical_speed_mps=vertical_speed,
                        dwell_charge_s=dwell_charge_s,
                        return_reserve_s=return_reserve_s,
                    )
                    if required <= capacity_limit_s + 1.0e-9:
                        best = candidate_cells
                        low = count + 1
                    else:
                        high = count - 1
                if best:
                    accepted_owner = drone_id
                    accepted_cells = best
                    break
            if accepted_owner is None:
                continue
            previous_ids = {str(cell["cell_id"]) for cell in assigned[accepted_owner]}
            assigned[accepted_owner] = accepted_cells
            new_ids = {str(cell["cell_id"]) for cell in accepted_cells} - previous_ids
            if new_ids:
                selected_cell_ids.update(new_ids)
                selected_region_ids.add(str(region["region_id"]))
            if all(
                _mission_route_lower_bound_s(
                    start_positions[drone_id],
                    assigned[drone_id],
                    safe_sky_altitude_m=safe_sky,
                    horizontal_speed_mps=horizontal_speed,
                    vertical_speed_mps=vertical_speed,
                    dwell_charge_s=dwell_charge_s,
                    return_reserve_s=return_reserve_s,
                )
                >= capacity_limit_s * 0.92
                for drone_id in assigned
            ):
                break
    if not selected_cell_ids or any(not cells for cells in assigned.values()):
        raise ValueError("mission-sector compiler could not allocate work to every vehicle")

    cell_lookup = {
        str(cell["cell_id"]): cell
        for region in regions
        for cell in region["cells"]
    }
    band_set = {
        str(region_by_id[region_id]["altitude_band"]) for region_id in selected_region_ids
    }
    if len(band_set) < min(3, len(altitude_bands)):
        raise ValueError("mission sector lacks three-dimensional altitude diversity")
    per_drone_required = {
        drone_id: round(
            _mission_route_lower_bound_s(
                start_positions[drone_id],
                cells,
                safe_sky_altitude_m=safe_sky,
                horizontal_speed_mps=horizontal_speed,
                vertical_speed_mps=vertical_speed,
                dwell_charge_s=dwell_charge_s,
                return_reserve_s=return_reserve_s,
            ),
            6,
        )
        for drone_id, cells in assigned.items()
    }
    sector = {
        "schema": MISSION_SECTOR_SCHEMA,
        "policy_id": MISSION_SECTOR_POLICY,
        "atlas_hash": atlas["atlas_hash"],
        "truth_independent": True,
        "frozen_before_sampling": True,
        "selected_region_ids": sorted(selected_region_ids),
        "selected_cell_ids": sorted(selected_cell_ids),
        # The assignment is public workload metadata, not target metadata.  It
        # makes the capacity certificate independently recomputable instead of
        # trusting a caller-supplied per-drone time claim.
        "cell_assignment_by_drone": {
            drone_id: [str(cell["cell_id"]) for cell in assigned[drone_id]]
            for drone_id in sorted(assigned)
        },
        "region_count": len(selected_region_ids),
        "cell_count": len(selected_cell_ids),
        "altitude_bands": sorted(band_set),
        "region_classes": sorted(
            {str(region_by_id[region_id]["region_class"]) for region_id in selected_region_ids}
        ),
        "represented_area_m2": round(
            sum(
                float(cell_lookup[cell_id]["represented_area_m2"])
                for cell_id in selected_cell_ids
            ),
            6,
        ),
        "capacity_certificate": {
            "model": "public_grouped_safe_sky_scan_dwell_return_lower_bound",
            "episode_duration_s": duration_s,
            "capacity_fraction": capacity_fraction,
            "capacity_limit_s": round(capacity_limit_s, 6),
            "exhaustive_sector_completion_required": False,
            "return_reserve_s": return_reserve_s,
            "discrete_dwell_charge_s": round(dwell_charge_s, 6),
            "horizontal_speed_mps": horizontal_speed,
            "vertical_speed_mps": vertical_speed,
            "per_drone_required_lower_bound_s": per_drone_required,
            "all_lower_bounds_fit": all(
                value <= capacity_limit_s for value in per_drone_required.values()
            ),
            "native_cf2x_validation_required": True,
            "calibration_status": "frozen",
        },
    }
    sector["sector_hash"] = content_hash(sector)
    validate_public_mission_sector(sector, atlas, starts, execution_contract)
    return sector


def validate_public_mission_sector(
    sector: dict[str, Any],
    atlas: dict[str, Any],
    starts: list[dict[str, Any]],
    execution_contract: dict[str, Any],
) -> None:
    """Fail closed on tampered, cross-layout, or over-budget public sectors."""

    validate_public_inspection_atlas(atlas)
    if sector.get("schema") != MISSION_SECTOR_SCHEMA:
        raise ValueError("mission sector schema differs")
    expected_hash = str(sector.get("sector_hash", ""))
    payload = {key: value for key, value in sector.items() if key != "sector_hash"}
    if content_hash(payload) != expected_hash:
        raise ValueError("mission sector hash mismatch")
    _assert_no_private_keys(payload, "mission_sector")
    if (
        sector.get("atlas_hash") != atlas.get("atlas_hash")
        or sector.get("truth_independent") is not True
        or sector.get("frozen_before_sampling") is not True
    ):
        raise ValueError("mission sector is not bound to a target-independent atlas")
    atlas_regions = {str(region["region_id"]): region for region in atlas["regions"]}
    atlas_cells = {
        str(cell["cell_id"]): (str(region["region_id"]), cell)
        for region in atlas["regions"]
        for cell in region["cells"]
    }
    region_ids = [str(value) for value in sector.get("selected_region_ids", [])]
    cell_ids = [str(value) for value in sector.get("selected_cell_ids", [])]
    if (
        not region_ids
        or not cell_ids
        or len(region_ids) != len(set(region_ids))
        or len(cell_ids) != len(set(cell_ids))
        or set(region_ids) - set(atlas_regions)
        or set(cell_ids) - set(atlas_cells)
    ):
        raise ValueError("mission sector references unknown or duplicate atlas obligations")
    if region_ids != sorted(region_ids) or cell_ids != sorted(cell_ids):
        raise ValueError("mission sector obligations must use canonical ordering")
    if {atlas_cells[cell_id][0] for cell_id in cell_ids} != set(region_ids):
        raise ValueError("mission sector region and cell projections differ")
    if int(sector.get("region_count", -1)) != len(region_ids) or int(
        sector.get("cell_count", -1)
    ) != len(cell_ids):
        raise ValueError("mission sector counts differ from selected obligations")
    certificate = sector.get("capacity_certificate")
    expected_certificate_fields = {
        "model",
        "episode_duration_s",
        "capacity_fraction",
        "capacity_limit_s",
        "exhaustive_sector_completion_required",
        "return_reserve_s",
        "discrete_dwell_charge_s",
        "horizontal_speed_mps",
        "vertical_speed_mps",
        "per_drone_required_lower_bound_s",
        "all_lower_bounds_fit",
        "native_cf2x_validation_required",
        "calibration_status",
    }
    if (
        not isinstance(certificate, dict)
        or set(certificate) != expected_certificate_fields
        or certificate.get("all_lower_bounds_fit") is not True
    ):
        raise ValueError("mission sector lacks a fitting public capacity certificate")
    duration = float(execution_contract["episode"]["duration_s"])
    if not math.isclose(float(certificate.get("episode_duration_s", -1.0)), duration):
        raise ValueError("mission sector duration differs from the execution contract")
    if (
        certificate.get("model") != "public_grouped_safe_sky_scan_dwell_return_lower_bound"
        or float(certificate.get("capacity_fraction", -1.0)) != 1.0
        or certificate.get("exhaustive_sector_completion_required") is not False
        or certificate.get("native_cf2x_validation_required") is not True
        or certificate.get("calibration_status") != "frozen"
    ):
        raise ValueError("mission sector capacity policy differs from the public contract")
    expected_drones = {str(start["drone_id"]) for start in starts}
    assignment = sector.get("cell_assignment_by_drone")
    if not isinstance(assignment, dict) or set(assignment) != expected_drones:
        raise ValueError("mission sector cell assignment differs from public starts")
    assigned_cell_ids: set[str] = set()
    for drone_id in sorted(expected_drones):
        assigned = assignment.get(drone_id)
        if not isinstance(assigned, list) or not assigned:
            raise ValueError("mission sector must assign cells to every public vehicle")
        if any(not isinstance(cell_id, str) for cell_id in assigned):
            raise ValueError("mission sector cell assignments must use string IDs")
        if len(assigned) != len(set(assigned)):
            raise ValueError("mission sector cell assignments contain duplicates")
        if assigned_cell_ids.intersection(assigned):
            raise ValueError("mission sector assigns one cell to multiple vehicles")
        unknown = set(assigned) - set(cell_ids)
        if unknown:
            raise ValueError("mission sector assignment references an unselected cell")
        assigned_cell_ids.update(assigned)
    if assigned_cell_ids != set(cell_ids):
        raise ValueError("mission sector assignment does not cover selected cells")
    required = certificate.get("per_drone_required_lower_bound_s")
    if not isinstance(required, dict) or set(required) != expected_drones:
        raise ValueError("mission sector capacity certificate differs from public starts")
    return_reserve_s = _finite_number(
        execution_contract["episode"]["return_reserve_s"],
        "episode.return_reserve_s",
    )
    period_s = _finite_number(execution_contract["control_period_s"], "control_period_s")
    dwell_s = _finite_number(
        execution_contract["observe"]["continuous_dwell_s"],
        "observe.continuous_dwell_s",
    )
    horizontal_speed = min(
        1.5,
        _finite_number(execution_contract["vehicle"]["horizontal_speed_mps"], "horizontal speed"),
    )
    vertical_speed = min(
        1.0,
        _finite_number(execution_contract["vehicle"]["vertical_speed_mps"], "vertical speed"),
    )
    dwell_charge_s = (math.ceil(dwell_s / period_s) + 1) * period_s
    if (
        not math.isclose(float(certificate["return_reserve_s"]), return_reserve_s, abs_tol=1.0e-6)
        or not math.isclose(
            float(certificate["discrete_dwell_charge_s"]), dwell_charge_s, abs_tol=1.0e-6
        )
        or not math.isclose(
            float(certificate["horizontal_speed_mps"]), horizontal_speed, abs_tol=1.0e-6
        )
        or not math.isclose(
            float(certificate["vertical_speed_mps"]), vertical_speed, abs_tol=1.0e-6
        )
    ):
        raise ValueError("mission sector capacity timing differs from the execution contract")
    safe_sky = _finite_number(
        atlas["transit_graph"]["safe_sky_altitude_m"],
        "mission safe-sky altitude",
    )
    start_by_drone = {
        str(start["drone_id"]): _vec3(start["position"], "mission start.position")
        for start in starts
    }
    cell_lookup = {
        cell_id: {**cell, "_mission_region_id": region_id}
        for cell_id, (region_id, cell) in atlas_cells.items()
    }
    recomputed_required = {
        drone_id: round(
            _mission_route_lower_bound_s(
                start_by_drone[drone_id],
                [cell_lookup[cell_id] for cell_id in assignment[drone_id]],
                safe_sky_altitude_m=safe_sky,
                horizontal_speed_mps=horizontal_speed,
                vertical_speed_mps=vertical_speed,
                dwell_charge_s=dwell_charge_s,
                return_reserve_s=return_reserve_s,
            ),
            6,
        )
        for drone_id in sorted(expected_drones)
    }
    if any(
        not math.isclose(float(required[drone_id]), recomputed_required[drone_id], abs_tol=1.0e-6)
        for drone_id in expected_drones
    ):
        raise ValueError("mission sector capacity certificate is not reproducible")
    expected_area = sum(
        float(atlas_cells[cell_id][1]["represented_area_m2"]) for cell_id in cell_ids
    )
    if not math.isclose(
        float(sector.get("represented_area_m2", -1.0)), expected_area, abs_tol=1.0e-5
    ):
        raise ValueError("mission sector represented area differs from selected cells")
    expected_classes = sorted(
        {str(atlas_regions[atlas_cells[cell_id][0]]["region_class"]) for cell_id in cell_ids}
    )
    expected_bands = sorted(
        {str(atlas_regions[atlas_cells[cell_id][0]]["altitude_band"]) for cell_id in cell_ids}
    )
    if (
        sector.get("region_classes") != expected_classes
        or sector.get("altitude_bands") != expected_bands
    ):
        raise ValueError("mission sector public region metadata differs from selected cells")
    expected_limit = duration
    limit = float(certificate["capacity_limit_s"])
    if not math.isclose(limit, expected_limit, abs_tol=1.0e-6) or any(
        value > limit + 1.0e-9 for value in recomputed_required.values()
    ):
        raise ValueError("mission sector exceeds its declared public capacity")


def project_inspection_atlas(
    atlas: dict[str, Any], prior_level: str
) -> dict[str, Any]:
    """Create a versioned coarse/full prior for the mandatory G2-I ablation.

    The coarse projection states which target-independent structural regions
    require inspection, but leaves viewpoint generation and routing to the
    method.  The full projection exposes the existing cell and transit ABI.
    Neither projection is a target witness or a formal score authority.
    """

    validate_public_inspection_atlas(atlas)
    if prior_level not in ATLAS_PRIOR_LEVELS:
        raise ValueError(f"unsupported G2-I atlas prior level: {prior_level}")
    projection: dict[str, Any] = {
        "schema": ATLAS_PROJECTION_SCHEMA,
        "projection_version": ATLAS_PROJECTION_VERSION,
        "prior_level": prior_level,
        "source_atlas_hash": atlas["atlas_hash"],
        "layout_id": atlas["layout_id"],
        "inspection_geometry_hash": atlas["inspection_geometry_hash"],
        "sampling_policy": copy.deepcopy(atlas["sampling_policy"]),
        "geometric_admission": copy.deepcopy(atlas["geometric_admission"]),
        "observation_contract": copy.deepcopy(atlas["observation_contract"]),
        "regions": [],
    }
    if prior_level == ATLAS_PRIOR_COARSE:
        projection["regions"] = [
            {
                "region_id": region["region_id"],
                "region_class": region["region_class"],
                "bounds": copy.deepcopy(region["bounds"]),
                "represented_area_m2": region["represented_area_m2"],
                "altitude_band": region["altitude_band"],
            }
            for region in atlas["regions"]
        ]
    else:
        projection["regions"] = copy.deepcopy(atlas["regions"])
        projection["transit_graph"] = copy.deepcopy(atlas["transit_graph"])
    projection["projection_hash"] = content_hash(projection)
    validate_inspection_atlas_projection(projection)
    return projection


def validate_inspection_atlas_projection(projection: dict[str, Any]) -> None:
    """Fail closed on projection level, content hash, and coarse information limits."""

    common = {
        "schema",
        "projection_version",
        "prior_level",
        "source_atlas_hash",
        "layout_id",
        "inspection_geometry_hash",
        "sampling_policy",
        "geometric_admission",
        "observation_contract",
        "regions",
        "projection_hash",
    }
    prior_level = projection.get("prior_level")
    expected = common if prior_level == ATLAS_PRIOR_COARSE else common | {"transit_graph"}
    if set(projection) != expected:
        raise ValueError("inspection-atlas projection fields differ from its prior level")
    if (
        projection.get("schema") != ATLAS_PROJECTION_SCHEMA
        or projection.get("projection_version") != ATLAS_PROJECTION_VERSION
        or prior_level not in ATLAS_PRIOR_LEVELS
    ):
        raise ValueError("inspection-atlas projection schema/version is unsupported")
    _assert_no_private_keys(
        {key: value for key, value in projection.items() if key != "projection_hash"}
    )
    candidate = copy.deepcopy(projection)
    declared_hash = str(candidate.pop("projection_hash", ""))
    if content_hash(candidate) != declared_hash:
        raise ValueError("inspection-atlas projection hash mismatch")
    if prior_level == ATLAS_PRIOR_COARSE:
        forbidden = {
            "cells",
            "pose",
            "surface_point",
            "surface_normal",
            "pose_envelope",
            "transit_graph",
        }

        def reject_full_fields(node: object) -> None:
            if isinstance(node, dict):
                if set(node) & forbidden:
                    raise ValueError("coarse inspection prior contains full-atlas geometry")
                for value in node.values():
                    reject_full_fields(value)
            elif isinstance(node, list):
                for value in node:
                    reject_full_fields(value)

        reject_full_fields(projection)


def validate_public_inspection_atlas(atlas: dict[str, Any]) -> None:
    """Fail closed on schema, hash, or evaluator-private atlas content."""

    expected_root = {
        "schema",
        "atlas_version",
        "layout_id",
        "inspection_geometry_hash",
        "sampling_policy",
        "geometric_admission",
        "observation_contract",
        "regions",
        "transit_graph",
        "runtime_validation_required",
        "atlas_hash",
    }
    if set(atlas) != expected_root:
        raise ValueError("inspection atlas root fields differ from the public contract")
    if atlas["schema"] != ATLAS_SCHEMA or atlas["atlas_version"] != ATLAS_VERSION:
        raise ValueError("inspection atlas schema/version is unsupported")
    if not isinstance(atlas["layout_id"], str) or not atlas["layout_id"]:
        raise ValueError("inspection atlas layout ID is invalid")
    if (
        not isinstance(atlas["inspection_geometry_hash"], str)
        or len(atlas["inspection_geometry_hash"]) != 64
    ):
        raise ValueError("inspection atlas geometry hash is invalid")
    if atlas["runtime_validation_required"] is not True:
        raise ValueError("inspection atlas must retain runtime validation")
    _assert_no_private_keys({key: value for key, value in atlas.items() if key != "atlas_hash"})
    expected_hash = str(atlas["atlas_hash"])
    candidate = copy.deepcopy(atlas)
    candidate.pop("atlas_hash")
    if content_hash(candidate) != expected_hash:
        raise ValueError("inspection atlas hash mismatch")

    sampling_policy = atlas["sampling_policy"]
    _sampling_policy_candidate(sampling_policy)
    geometric_admission = atlas["geometric_admission"]
    expected_admission = {
        "schema",
        "candidate_cell_count",
        "admitted_cell_count",
        "rejected_cell_count",
        "rejection_counts",
        "rejection_counts_by_region_class",
        "flight_bound_margin_m",
        "admission_semantics",
        "native_cf2x_validation_required",
    }
    if not isinstance(geometric_admission, dict) or set(geometric_admission) != expected_admission:
        raise ValueError("inspection atlas geometric admission fields differ")
    candidate_count = int(geometric_admission["candidate_cell_count"])
    admitted_count = int(geometric_admission["admitted_cell_count"])
    rejected_count = int(geometric_admission["rejected_cell_count"])
    if min(candidate_count, admitted_count, rejected_count) < 0:
        raise ValueError("inspection atlas geometric admission counts cannot be negative")
    if candidate_count != admitted_count + rejected_count or admitted_count <= 0:
        raise ValueError("inspection atlas geometric admission counts are inconsistent")
    if geometric_admission["native_cf2x_validation_required"] is not True:
        raise ValueError("inspection atlas must retain native CF2X validation")
    if _finite_number(
        geometric_admission["flight_bound_margin_m"],
        "inspection atlas flight-bound margin",
    ) <= 0.0:
        raise ValueError("inspection atlas flight-bound margin must be positive")
    observation = atlas["observation_contract"]
    expected_observation = {
        "maximum_range_m",
        "horizontal_fov_deg",
        "vertical_fov_deg",
        "continuous_dwell_s",
        "minimum_clearance_m",
        "nominal_standoff_m",
        "horizontal_cell_spacing_m",
        "vertical_cell_spacing_m",
    }
    if not isinstance(observation, dict) or set(observation) != expected_observation:
        raise ValueError("inspection atlas observation contract differs")
    for key, value in observation.items():
        if _finite_number(value, f"inspection atlas {key}") <= 0.0:
            raise ValueError(f"inspection atlas {key} must be positive")

    if not isinstance(atlas["regions"], list) or not atlas["regions"]:
        raise ValueError("inspection atlas must contain regions")
    region_ids: set[str] = set()
    global_cell_ids: set[str] = set()
    for region in atlas["regions"]:
        expected_region = {
            "region_id",
            "region_class",
            "bounds",
            "represented_area_m2",
            "altitude_band",
            "cells",
        }
        if not isinstance(region, dict) or set(region) != expected_region:
            raise ValueError("inspection atlas region fields differ")
        region_id = str(region["region_id"])
        if not region_id.startswith("atlas-region-") or region_id in region_ids:
            raise ValueError("inspection atlas region IDs must be unique")
        region_ids.add(region_id)
        if region["region_class"] not in REGION_CLASSES:
            raise ValueError("inspection atlas contains an unsupported region class")
        if _finite_number(region["represented_area_m2"], "inspection region area") <= 0.0:
            raise ValueError("inspection atlas region area must be positive")
        if region["altitude_band"] not in {
            "near_ground",
            "lower",
            "mid",
            "elevated",
            "highrise",
        }:
            raise ValueError("inspection atlas altitude band is invalid")
        bounds = region["bounds"]
        if not isinstance(bounds, dict) or set(bounds) != {"minimum", "maximum"}:
            raise ValueError("inspection atlas region bounds differ")
        minimum = _vector(bounds["minimum"], 3, "inspection region minimum")
        maximum = _vector(bounds["maximum"], 3, "inspection region maximum")
        if any(low > high for low, high in zip(minimum, maximum, strict=True)):
            raise ValueError("inspection atlas region bounds are inverted")
        if not isinstance(region["cells"], list) or not region["cells"]:
            raise ValueError("inspection atlas region must contain cells")
        cell_ids: set[str] = set()
        represented_cell_area = 0.0
        for cell in region["cells"]:
            expected_cell = {
                "cell_id",
                "pose",
                "surface_point",
                "surface_normal",
                "represented_area_m2",
                "pose_envelope",
            }
            if not isinstance(cell, dict) or set(cell) != expected_cell:
                raise ValueError("inspection atlas cell fields differ")
            cell_id = str(cell["cell_id"])
            if (
                not cell_id.startswith("atlas-cell-")
                or cell_id in cell_ids
                or cell_id in global_cell_ids
            ):
                raise ValueError("inspection atlas cell IDs must be globally unique")
            cell_ids.add(cell_id)
            global_cell_ids.add(cell_id)
            pose = cell["pose"]
            if not isinstance(pose, dict) or set(pose) != {"position", "yaw_deg", "pitch_deg"}:
                raise ValueError("inspection atlas pose fields differ")
            _vector(pose["position"], 3, "inspection atlas pose position")
            _finite_number(pose["yaw_deg"], "inspection atlas pose yaw")
            _finite_number(pose["pitch_deg"], "inspection atlas pose pitch")
            _vector(cell["surface_point"], 3, "inspection atlas surface point")
            normal = _vector(cell["surface_normal"], 3, "inspection atlas surface normal")
            normal_length = math.sqrt(sum(value * value for value in normal))
            if not math.isclose(normal_length, 1.0, abs_tol=1.0e-4):
                raise ValueError("inspection atlas surface normal must be unit length")
            cell_area = _finite_number(
                cell["represented_area_m2"], "inspection cell area"
            )
            if cell_area <= 0.0:
                raise ValueError("inspection atlas cell area must be positive")
            represented_cell_area += cell_area
            envelope = cell["pose_envelope"]
            if not isinstance(envelope, dict) or set(envelope) != {
                "nominal_standoff_m",
                "lateral_tolerance_m",
                "vertical_tolerance_m",
            }:
                raise ValueError("inspection atlas pose envelope differs")
            for key, value in envelope.items():
                if _finite_number(value, f"inspection atlas envelope {key}") <= 0.0:
                    raise ValueError("inspection atlas pose envelope must be positive")
        area_tolerance = max(1.0e-3, len(region["cells"]) * 5.1e-5)
        if not math.isclose(
            represented_cell_area,
            float(region["represented_area_m2"]),
            abs_tol=area_tolerance,
        ):
            raise ValueError("inspection cell areas do not conserve their region area")

    if len(global_cell_ids) != admitted_count:
        raise ValueError("inspection atlas admitted-cell count differs from serialized cells")

    graph = atlas["transit_graph"]
    expected_graph = {"schema", "safe_sky_altitude_m", "nodes", "edges"}
    if not isinstance(graph, dict) or set(graph) != expected_graph:
        raise ValueError("inspection atlas transit graph fields differ")
    if graph["schema"] != "org.aerocity.bench.inspection-atlas-transit-public.v1":
        raise ValueError("inspection atlas transit graph schema is unsupported")
    if _finite_number(graph["safe_sky_altitude_m"], "safe sky altitude") <= 0.0:
        raise ValueError("inspection atlas safe sky altitude must be positive")
    if not isinstance(graph["nodes"], list) or not graph["nodes"]:
        raise ValueError("inspection atlas transit graph must contain nodes")
    node_ids = set()
    for node in graph["nodes"]:
        if not isinstance(node, dict) or set(node) != {"node_id", "region_id", "position"}:
            raise ValueError("inspection atlas transit node fields differ")
        node_id = str(node["node_id"])
        if not node_id.startswith("atlas-transit-") or node_id in node_ids:
            raise ValueError("inspection atlas transit node IDs must be unique")
        if node["region_id"] not in region_ids:
            raise ValueError("inspection atlas transit node references an unknown region")
        node_ids.add(node_id)
        _vector(node["position"], 3, "inspection atlas transit node position")
    if not isinstance(graph["edges"], list):
        raise ValueError("inspection atlas transit edges must be a list")
    edge_ids = set()
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in graph["edges"]:
        if not isinstance(edge, dict) or set(edge) != {
            "edge_id",
            "start_node_id",
            "end_node_id",
            "safe_sky_distance_m",
        }:
            raise ValueError("inspection atlas transit edge fields differ")
        edge_id = str(edge["edge_id"])
        if not edge_id.startswith("atlas-edge-") or edge_id in edge_ids:
            raise ValueError("inspection atlas transit edge IDs must be unique")
        if edge["start_node_id"] not in node_ids or edge["end_node_id"] not in node_ids:
            raise ValueError("inspection atlas transit edge references an unknown node")
        if edge["start_node_id"] == edge["end_node_id"]:
            raise ValueError("inspection atlas transit graph must not contain self edges")
        if _finite_number(edge["safe_sky_distance_m"], "safe-sky edge distance") <= 0.0:
            raise ValueError("inspection atlas transit edge distance must be positive")
        adjacency[str(edge["start_node_id"])].add(str(edge["end_node_id"]))
        adjacency[str(edge["end_node_id"])].add(str(edge["start_node_id"]))
        edge_ids.add(edge_id)
    reached = set()
    frontier = [min(node_ids)]
    while frontier:
        node_id = frontier.pop()
        if node_id in reached:
            continue
        reached.add(node_id)
        frontier.extend(sorted(adjacency[node_id] - reached))
    if reached != node_ids:
        raise ValueError("inspection atlas transit graph is disconnected")
