"""Fail-closed scientific audits for the target-independent G2-I atlas."""

from __future__ import annotations

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
from .inspection_atlas import validate_public_inspection_atlas

AUDIT_SCHEMA = "org.aerocity.bench.g2-i-atlas-scientific-audit.v1"


def _vec3(value: object, name: str) -> Vec3:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-vector")
    x_value, y_value, z_value = (float(item) for item in value)
    result: Vec3 = (x_value, y_value, z_value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _inside_flight_bounds(point: Vec3, city: dict[str, Any], margin: float) -> bool:
    minimum = _vec3(city["flight_bounds"]["minimum"], "flight_bounds.minimum")
    maximum = _vec3(city["flight_bounds"]["maximum"], "flight_bounds.maximum")
    return all(
        low + margin <= coordinate <= high - margin
        for coordinate, low, high in zip(point, minimum, maximum, strict=True)
    )


def _inspection_camera_pose(pose: Pose3D, execution_contract: dict[str, Any]) -> Pose3D:
    """Use the public bounded gimbal rather than an impossible hover pitch."""

    rig = execution_contract["sensor_rig"]
    body = pose if rig["gimbal_mode"] == "fixed" else Pose3D(pose.position, pose.yaw_deg)
    return sensor_pose(
        body,
        rig["translation_body_m"],
        sensor_pitch_deg=(pose.pitch_deg if rig["gimbal_mode"] == "bounded" else None),
    )


def _graph_mst_distance(graph: dict[str, Any]) -> float:
    node_ids = {str(node["node_id"]) for node in graph["nodes"]}
    if len(node_ids) <= 1:
        return 0.0
    parent = {node_id: node_id for node_id in node_ids}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    total = 0.0
    retained = 0
    for edge in sorted(
        graph["edges"],
        key=lambda item: (float(item["safe_sky_distance_m"]), str(item["edge_id"])),
    ):
        first = find(str(edge["start_node_id"]))
        second = find(str(edge["end_node_id"]))
        if first == second:
            continue
        parent[max(first, second)] = min(first, second)
        total += float(edge["safe_sky_distance_m"])
        retained += 1
        if retained == len(node_ids) - 1:
            break
    if retained != len(node_ids) - 1:
        raise ValueError("cannot compute an MST for a disconnected transit graph")
    return total


def audit_inspection_atlas(
    city: dict[str, Any],
    atlas: dict[str, Any],
    execution_contract: dict[str, Any],
    *,
    fleet_count: int,
    episode_duration_s: float,
    horizontal_speed_mps: float = 1.5,
) -> dict[str, Any]:
    """Audit public atlas geometry without consuming evaluator-private truth.

    This is an AABB/kinematic preflight.  A passing report shortlists cells and
    edges for native CF2X/PhysX replay; it never upgrades an L0 result to L1.
    """

    validate_public_inspection_atlas(atlas)
    if fleet_count <= 0 or episode_duration_s <= 0.0 or horizontal_speed_mps <= 0.0:
        raise ValueError("audit fleet, duration, and speed must be positive")
    colliders = colliders_from_city(city)
    vehicle = execution_contract["vehicle"]
    observe = execution_contract["observe"]
    body_margin = float(vehicle["radius_m"]) + float(vehicle["minimum_clearance_m"])
    safe_sky = float(atlas["transit_graph"]["safe_sky_altitude_m"])
    issues: Counter[str] = Counter()
    duplicate_poses: Counter[tuple[float, ...]] = Counter()
    region_class_counts: Counter[str] = Counter()
    altitude_band_counts: Counter[str] = Counter()
    cell_count = 0
    represented_area = 0.0

    for region in atlas["regions"]:
        region_class_counts[str(region["region_class"])] += 1
        altitude_band_counts[str(region["altitude_band"])] += 1
        for cell in region["cells"]:
            cell_count += 1
            represented_area += float(cell["represented_area_m2"])
            pose = Pose3D.from_dict(cell["pose"])
            normal = _vec3(cell["surface_normal"], "cell.surface_normal")
            surface_point = _vec3(cell["surface_point"], "cell.surface_point")
            duplicate_poses[
                tuple(round(value, 4) for value in (*pose.position, pose.yaw_deg, pose.pitch_deg))
            ] += 1
            if not _inside_flight_bounds(pose.position, city, body_margin):
                issues["cell_body_outside_flight_bounds"] += 1
            clearance, _ = minimum_clearance(pose.position, colliders)
            if clearance + 1.0e-9 < body_margin:
                issues["cell_body_clearance"] += 1
            camera = _inspection_camera_pose(pose, execution_contract)
            if distance(camera.position, surface_point) > float(observe["max_range_m"]):
                issues["cell_sensor_range"] += 1
            in_view, _, _ = in_field_of_view(
                camera,
                surface_point,
                float(observe["horizontal_fov_deg"]),
                float(observe["vertical_fov_deg"]),
            )
            if not in_view:
                issues["cell_sensor_fov"] += 1
            facing, _ = surface_facing(
                camera.position,
                surface_point,
                normal,
                float(observe["surface_facing_min_cosine"]),
            )
            if not facing:
                issues["cell_surface_facing"] += 1
            visible, _ = line_of_sight(camera.position, surface_point, colliders)
            if not visible:
                issues["cell_surface_los"] += 1
            sky_point = (pose.position[0], pose.position[1], safe_sky)
            climb_clearance, _ = minimum_segment_clearance(
                pose.position, sky_point, colliders
            )
            if climb_clearance + 1.0e-9 < body_margin:
                issues["cell_safe_sky_climb_clearance"] += 1

    duplicate_pose_count = sum(count - 1 for count in duplicate_poses.values() if count > 1)
    if duplicate_pose_count:
        issues["duplicate_cell_pose"] += duplicate_pose_count

    graph = atlas["transit_graph"]
    positions = {
        str(node["node_id"]): _vec3(node["position"], "transit node position")
        for node in graph["nodes"]
    }
    for point in positions.values():
        if not _inside_flight_bounds(point, city, body_margin):
            issues["transit_node_outside_flight_bounds"] += 1
        clearance, _ = minimum_clearance(point, colliders)
        if clearance + 1.0e-9 < body_margin:
            issues["transit_node_clearance"] += 1
    for edge in graph["edges"]:
        start = positions[str(edge["start_node_id"])]
        end = positions[str(edge["end_node_id"])]
        clearance, _ = minimum_segment_clearance(start, end, colliders)
        if clearance + 1.0e-9 < body_margin:
            issues["transit_edge_clearance"] += 1
        declared_distance = float(edge["safe_sky_distance_m"])
        if not math.isclose(
            math.dist(start[:2], end[:2]), declared_distance, abs_tol=1.0e-3
        ):
            issues["transit_edge_distance_mismatch"] += 1

    mst_distance = _graph_mst_distance(graph)
    serial_dwell_workload = (
        cell_count * float(observe["continuous_dwell_s"]) / fleet_count
    )
    transit_lower_bound = mst_distance / (fleet_count * horizontal_speed_mps)
    workload_ratio = serial_dwell_workload / episode_duration_s
    geometry_status = "PASS_CPU" if not issues else "FAIL"
    sampling_frozen = atlas["sampling_policy"]["calibration_status"] == "frozen"
    report: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "layout_id_hash": content_hash(str(city["layout_id"])),
        "atlas_hash": atlas["atlas_hash"],
        "execution_contract_hash": content_hash(execution_contract),
        "formal_score_eligible": False,
        "cpu_geometry_status": geometry_status,
        "scientific_gate_status": (
            "NATIVE_AND_CALIBRATION_REQUIRED"
            if geometry_status == "PASS_CPU"
            else "CPU_GEOMETRY_FAILED"
        ),
        "issue_counts": dict(sorted(issues.items())),
        "aggregate": {
            "region_count": len(atlas["regions"]),
            "cell_count": cell_count,
            "represented_area_m2": round(represented_area, 6),
            "region_class_counts": dict(sorted(region_class_counts.items())),
            "altitude_band_counts": dict(sorted(altitude_band_counts.items())),
            "transit_node_count": len(graph["nodes"]),
            "transit_edge_count": len(graph["edges"]),
        },
        "budget_bracket": {
            "episode_duration_s": episode_duration_s,
            "fleet_count": fleet_count,
            "region_transit_mst_lower_bound_m": round(mst_distance, 6),
            "region_transit_mst_lower_bound_s": round(transit_lower_bound, 6),
            "serial_one_cell_per_observe_dwell_workload_s": round(
                serial_dwell_workload, 6
            ),
            "serial_dwell_workload_to_budget_ratio": round(workload_ratio, 6),
            "interpretation": (
                "WORKLOAD_EXCEEDS_BUDGET"
                if workload_ratio > 1.0
                else "WORKLOAD_WITHIN_BUDGET"
            ),
            "not_a_solvability_claim": True,
        },
        "remaining_gates": {
            "sampling_policy_frozen_by_method_independent_calibration": sampling_frozen,
            "native_cf2x_cell_shortlist_replay": False,
            "native_cf2x_climb_and_edge_replay": False,
            "three_to_six_independent_calibration_ancestors": False,
            "four_vehicle_public_g2_i_l1_closure": False,
        },
    }
    report["report_hash"] = content_hash(report)
    return report
