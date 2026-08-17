from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.citylite_scene import (
    AUTHORITY_SHA256,
    CityLiteAuthority,
    ENVIRONMENT_ID,
    CITY_LITE_ROUTE_FAMILY_A_ID,
    CITY_LITE_START_ANCHOR_A_ID,
    CITY_LITE_TARGET_REGION_A_ID,
    EXPECTED_NATIVE_COLLISION_COUNTS,
    EXPECTED_UPSTREAM_PERMISSIONS,
    PUBLIC_ROUTES_W_M,
    PUBLIC_ROUTES_B_W_M,
    ROUTE_CLEARANCE_M,
    SCENE_CONTRACT_GATE_STATUS,
    SCENE_CONTRACT_PAYLOAD_SHA256,
    SCENE_CONTRACT_SCHEMA,
    SCENE_CONTRACT_SHA256,
    SELECTIVE_REFERENCES,
    AABB,
    aabb_geometry_sha256,
    canonical_payload_sha256,
    city_task_obstacle_material_closure_receipt_template,
    make_rivermark_layer_inventory,
)
from rivermark_benchmark.isaac_capture import (
    HOVER_THRUST_PER_ROTOR_N,
    IDENTITY_MARKER_RADIUS_M,
    INITIAL_HOVER_RPS,
    LITERAL_SPAWN_DEFAULT_RPS_TOLERANCE,
    LITERAL_SPAWN_DEFAULT_STATE_TOLERANCE,
    LITERAL_SPAWN_DEFAULT_THRUST_TOLERANCE_N,
    LITERAL_SPAWN_ORIENTATION_TOLERANCE_RAD,
    LITERAL_SPAWN_POSITION_TOLERANCE_M,
    LITERAL_USD_SPAWN_ORIENTATION_TOLERANCE_RAD,
    LITERAL_USD_SPAWN_POSITION_TOLERANCE_M,
    MAX_CF2X_ANGULAR_VELOCITY_RADPS,
    MAX_CF2X_LINEAR_VELOCITY_MPS,
    ONBOARD_CAMERA_CLIPPING_RANGE_M,
    ONBOARD_CONTENT_GATE_SCHEMA,
    OVERVIEW_ARCHIVE_SCHEMA,
    OVERVIEW_ARCHIVE_STRIDE,
    OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS,
    OVERVIEW_WITNESS_TRACKED_AGENT_ID,
    PRIVATE_EVALUATOR_SCHEMA,
    PRIVATE_TARGET_ORIGIN,
    PRIVATE_TARGET_PLACEMENT_SCHEMA,
    SWARM_AGENT_LITERAL_PRIM_PATHS,
    TARGET_COUNT,
    T1_DATA_TRACK_ID,
    T1_OBSERVABILITY_OUTCOME_SCHEMA,
    TASK_VARIANT_ID,
    THRUST_COEFFICIENT_N_PER_RPS_SQUARED,
    VISUAL_INTRUSION_GATE_SCHEMA,
    _city_lite_spawn_states,
    _captured_frame_indices,
    _overview_archive_frame_indices,
    _public_route_witness_schedule,
    _public_route_witness_view_at_time_ns,
    _visual_intrusion_gate_contract,
    _onboard_content_gate_contract,
)
from rivermark_benchmark.isaac_runtime_safety import (
    CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N,
    INTER_AGENT_PAIR_COUNT,
    RUNTIME_SAFETY_FRAME_OUTCOME_CODES,
    RUNTIME_SAFETY_PHASE_CODES,
    RUNTIME_SAFETY_TRACE_RELATIVE_PATH,
    SENSOR_PHASE_EVENT_SEQUENCE,
    SENSOR_PHASE_SENSOR_NAMES,
    SENSOR_PHASE_TRACE_RELATIVE_PATH,
    SENSOR_PHASE_TRACE_SCHEMA,
    finalize_runtime_safety_guard,
    physics_time_ns,
    runtime_safety_receipt_template,
    sensor_phase_array_digest,
)
from rivermark_benchmark.frame_archive import write_chunked_frame_archive
from rivermark_benchmark.citylite_task import (
    sample_private_targets,
    target_visibility_execution_window,
    target_visibility_geometry_contract,
)
from rivermark_benchmark.isaac_validate import (
    AGENT_COUNT,
    CAPTURE_SCHEMA,
    EXPECTED_ARTIFACTS,
    IsaacValidationReport,
    OVERVIEW_CONTENT_GATE_SCHEMA,
    ValidationIssue,
    _overview_content_gate_contract,
    _validate_literal_city_lite_fleet_spawn,
    validate_isaac_capture,
    write_validation_receipt,
)
from rivermark_benchmark.private_evaluator_manifest import (
    PRIVATE_MANIFEST_RETENTION_KIND,
    PRIVATE_MANIFEST_RETENTION_MAX_BYTES,
)
from rivermark_benchmark.video import sha256_file
from rivermark_benchmark.runtime_lock import runtime_lock_sha256


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _private_manifest(path: Path) -> None:
    rows = _structural_aabbs()
    boxes = tuple(
        AABB(
            tuple(row["min"]),  # type: ignore[arg-type]
            tuple(row["max"]),  # type: ignore[arg-type]
            source_prim=str(row["path"]),
            category=str(row["source_kind"]),
        )
        for row in rows
    )
    sampled_targets = sample_private_targets(
        seed=17,
        target_count=TARGET_COUNT,
        target_region_id=CITY_LITE_TARGET_REGION_A_ID,
        visibility_bucket="direct-visible-v1",
        routes_w_m=PUBLIC_ROUTES_W_M,
        structural_aabbs=boxes,
        radius_m=0.14,
        obstacle_clearance_m=ROUTE_CLEARANCE_M,
        minimum_route_separation_m=2.0,
        minimum_pairwise_separation_m=1.5,
    )
    _write_json(
        path,
        {
            "schema": PRIVATE_EVALUATOR_SCHEMA,
            "environment_id": ENVIRONMENT_ID,
            "city_lite_scene_contract_sha256": SCENE_CONTRACT_SHA256,
            "city_lite_scene_payload_sha256": SCENE_CONTRACT_PAYLOAD_SHA256,
            "task_variant_id": TASK_VARIANT_ID,
            "sampled_before_policy_start": True,
            "route_conditioning": "public_only",
            "target_origin": {
                "kind": PRIVATE_TARGET_ORIGIN,
                "candidate_pool_released": False,
                "seed_released": False,
                "coordinates_released": False,
            },
            "target_placement_contract": {
                "schema": PRIVATE_TARGET_PLACEMENT_SCHEMA,
                "obstacle_clearance_m": ROUTE_CLEARANCE_M,
                "minimum_route_separation_m": 2.0,
                "minimum_pairwise_separation_m": 1.5,
            },
            "target_visibility_contract": target_visibility_geometry_contract(
                route_family_id=CITY_LITE_ROUTE_FAMILY_A_ID,
                routes_w_m=PUBLIC_ROUTES_W_M,
                aabb_geometry_sha256=_aabb_hash(rows),
                target_region_id=CITY_LITE_TARGET_REGION_A_ID,
                visibility_bucket="direct-visible-v1",
            ),
            "targets": [
                {
                    "target_id": f"private-object-{index:02d}",
                    "position_w_m": target["position_w_m"],
                    "appearance": "fixture",
                    "radius_m": target["radius_m"],
                    "visibility_bucket": "direct-visible-v1",
                }
                for index, target in enumerate(sampled_targets)
            ],
        },
    )


def _rivermark_layer_inventory(root: Path) -> dict[str, object]:
    """Create a filesystem-backed receipt outside the capture's closed world."""

    authority_root = root.parent / "fixture_city_lite_authority"
    asset_paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for index, filename in enumerate(sorted(AUTHORITY_SHA256), start=1):
        path = authority_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#usda 1.0\n# fixture authority layer {index}\n", encoding="ascii")
        asset_paths[filename] = path
        hashes[filename] = sha256_file(path)

    external_root = root.parent / "RivermarkSrc51"
    external_layer = external_root / "dsready_content" / "scene" / "city_lite.usda"
    external_layer.parent.mkdir(parents=True, exist_ok=True)
    external_layer.write_text("#usda 1.0\n# fixture Rivermark layer\n", encoding="ascii")
    authority = CityLiteAuthority(
        root=authority_root,
        contract_path=authority_root / "contract.json",
        final_scene_path=next(iter(asset_paths.values())),
        asset_paths=asset_paths,
        sha256=hashes,
        contract_sha256="0" * 64,
        contract_payload_sha256="1" * 64,
    )
    local_layers = [asset_paths[filename] for filename in sorted(AUTHORITY_SHA256)]
    return make_rivermark_layer_inventory(
        authority,
        ["anon:fixture-root", *local_layers, external_layer],
        asset_root=external_root,
    )


def _city_lite_routes() -> np.ndarray:
    return np.asarray(PUBLIC_ROUTES_W_M, dtype=np.float32)


def _structural_aabbs() -> list[dict[str, object]]:
    return [
        {
            "path": "/World/StaticScene/City/Rivermark/Buildings/CentralFixture",
            "source_kind": "city_structural_visible_geometry",
            "min": [-4.0, -4.0, 0.0],
            "max": [4.0, 4.0, 19.0],
        },
        {
            "path": "/World/StaticScene/CityTaskObstacles/FixtureObstacle",
            "source_kind": "city_task_obstacle",
            "min": [44.0, 42.0, 0.0],
            "max": [45.0, 43.0, 8.0],
        },
    ]


def _aabb_hash(rows: list[dict[str, object]]) -> str:
    boxes = [
        AABB(
            tuple(row["min"]),  # type: ignore[arg-type]
            tuple(row["max"]),  # type: ignore[arg-type]
            source_prim=str(row["path"]),
            category=str(row["source_kind"]),
        )
        for row in rows
    ]
    return aabb_geometry_sha256(boxes)


def _runtime_safety_guard_fixture(
    rows: list[dict[str, object]], *, steps: int, warmup_steps: int
) -> dict[str, object]:
    boxes = tuple(
        AABB(
            tuple(row["min"]),  # type: ignore[arg-type]
            tuple(row["max"]),  # type: ignore[arg-type]
            source_prim=str(row["path"]),
            category=str(row["source_kind"]),
        )
        for row in rows
    )
    guard = runtime_safety_receipt_template(
        boxes,
        contact_prim_expression="/World/Swarm/Agent_.*/Robot/body",
        physics_dt_s=0.005,
    )
    guard["checks"] = {
        "post_reset_agent_center_checks": AGENT_COUNT,
        "post_reset_point_geometry_checks": AGENT_COUNT,
        "post_reset_inter_agent_pair_checks": INTER_AGENT_PAIR_COUNT,
        "warmup_physics_steps_checked": warmup_steps,
        "rollout_physics_steps_checked": steps,
        "agent_center_checks": AGENT_COUNT * (1 + warmup_steps + steps),
        "swept_segments_checked": AGENT_COUNT * (warmup_steps + steps),
        "inter_agent_pair_checks": INTER_AGENT_PAIR_COUNT * (1 + warmup_steps + steps),
        "minimum_inter_agent_swept_separation_m": 1.0,
        "contact_samples_checked": 1 + warmup_steps + steps,
        "max_contact_force_n": 0.0,
        "contact_abort_count": 0,
    }
    return guard


def _literal_fleet_spawn_fixture() -> dict[str, object]:
    paths = list(SWARM_AGENT_LITERAL_PRIM_PATHS)
    return {
        "literal_prim_paths": paths,
        "authored_usd_transform": {
            "source": "fresh_stage_usd_xform_cache_before_sim_reset",
            "position_tolerance_m": LITERAL_USD_SPAWN_POSITION_TOLERANCE_M,
            "orientation_tolerance_rad": LITERAL_USD_SPAWN_ORIENTATION_TOLERANCE_RAD,
            "per_agent": [
                {
                    "agent_id": agent_id,
                    "prim_path": path,
                    "position_error_m": 0.0,
                    "orientation_error_rad": 0.0,
                    "rigid_transform_determinant": 1.0,
                    "basis_axis_lengths": [1.0, 1.0, 1.0],
                }
                for agent_id, path in enumerate(paths)
            ],
            "max_position_error_m": 0.0,
            "max_orientation_error_rad": 0.0,
        },
        "authored_defaults": {
            "root_state_shape": [AGENT_COUNT, 13],
            "thruster_rps_shape": [AGENT_COUNT, 4],
            "thrust_target_shape": [AGENT_COUNT, 4],
            "root_state_max_abs_error": 0.0,
            "thruster_rps_max_abs_error": 0.0,
            "thrust_target_max_abs_error_n": 0.0,
            "root_state_tolerance": LITERAL_SPAWN_DEFAULT_STATE_TOLERANCE,
            "thruster_rps_tolerance": LITERAL_SPAWN_DEFAULT_RPS_TOLERANCE,
            "thrust_target_tolerance_n": LITERAL_SPAWN_DEFAULT_THRUST_TOLERANCE_N,
        },
        "post_reset_physics_settling": {
            "classification": "observed_after_sim_reset_before_first_command",
            "max_position_delta_m": 0.002452,
            "max_orientation_delta_rad": 0.0,
            "max_linear_velocity_mps": 0.196151,
            "max_angular_velocity_radps": 0.0,
            "position_tolerance_m": LITERAL_SPAWN_POSITION_TOLERANCE_M,
            "orientation_tolerance_rad": LITERAL_SPAWN_ORIENTATION_TOLERANCE_RAD,
            "linear_velocity_hard_limit_mps": MAX_CF2X_LINEAR_VELOCITY_MPS,
            "angular_velocity_hard_limit_radps": MAX_CF2X_ANGULAR_VELOCITY_RADPS,
        },
        "post_reset_root_pose_rewrite": False,
        "post_reset_root_velocity_rewrite": False,
        "anchor_contract": "rivermark_public_route_initial_waypoints",
    }


def _physics_fixture(literal_fleet_spawn: dict[str, object]) -> dict[str, object]:
    """Mirror the public literal-fleet evidence emitted by Isaac capture."""

    return {
        "same_world_agent_count": AGENT_COUNT,
        "multirotor_prim_expression": "/World/Swarm/Agent_.*/Robot",
        "literal_agent_prim_paths": list(SWARM_AGENT_LITERAL_PRIM_PATHS),
        "literal_fleet_spawn": literal_fleet_spawn,
        "cf2x_hover_trim": {
            "source": "md_qd_swarm.qdr_runtime_spawn_final_task",
            "hover_thrust_per_rotor_n": HOVER_THRUST_PER_ROTOR_N,
            "thrust_coefficient_n_per_rps_squared": THRUST_COEFFICIENT_N_PER_RPS_SQUARED,
            "initial_hover_rps": INITIAL_HOVER_RPS,
            "max_linear_velocity_mps": MAX_CF2X_LINEAR_VELOCITY_MPS,
            "max_angular_velocity_radps": MAX_CF2X_ANGULAR_VELOCITY_RADPS,
        },
    }


def _bind_receipt(
    root: Path,
    evaluator_manifest: Path,
    *,
    dirty: bool = False,
    claim_boundary: dict[str, bool] | None = None,
    capture_integrity: dict[str, bool] | None = None,
    command: dict[str, object] | None = None,
    runtime_safety_guard: dict[str, object] | None = None,
    physics: dict[str, object] | None = None,
) -> None:
    artifacts = {}
    for relative in sorted(EXPECTED_ARTIFACTS):
        path = root / relative
        artifacts[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    existing: dict[str, object] = {}
    receipt_path = root / "capture_receipt.json"
    if receipt_path.is_file():
        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    phase_binding = existing.get("sensor_phase_trace")
    phase_path = root / SENSOR_PHASE_TRACE_RELATIVE_PATH
    if phase_path.is_file():
        with np.load(phase_path, allow_pickle=False) as phase_archive:
            phase_count = int(phase_archive["physics_step"].shape[0])
        phase_binding = {
            "schema": SENSOR_PHASE_TRACE_SCHEMA,
            "path": SENSOR_PHASE_TRACE_RELATIVE_PATH,
            "sha256": sha256_file(phase_path),
            "frame_count": phase_count,
            "sensor_names": list(SENSOR_PHASE_SENSOR_NAMES),
            "event_codes": list(SENSOR_PHASE_EVENT_SEQUENCE),
        }
    receipt = {
        "schema": CAPTURE_SCHEMA,
        "status": "captured",
        "ok": True,
        "task_kind": "search3d",
        "information_profile": "multisensor_rgbd_lidar_imu_state",
        "source_revision": "0123456789abcdef",
        "source_tree_sha256": "a" * 64,
        "source_worktree_dirty": dirty,
        "evaluator_manifest_sha256": sha256_file(evaluator_manifest),
        "command": command or existing.get("command") or {"seed": 20260723},
        "provenance": {"legacy_route_target_trace_or_evaluator_migrated": False},
        "runtime_lock": existing.get("runtime_lock"),
        "runtime_live": existing.get("runtime_live"),
        "preflight": existing.get("preflight"),
        "runtime_target_usd_pre_reset": existing.get("runtime_target_usd_pre_reset"),
        "runtime_target_usd_post_reset": existing.get("runtime_target_usd_post_reset"),
        "capture_integrity": capture_integrity
        or {
            "online_capture": True,
            "queue_used": False,
            "queue_overflow": False,
            "silent_frame_drop": False,
            "synchronous_sensor_reads": True,
        },
        "claim_boundary": claim_boundary
        or {
            "formal_benchmark_admission": False,
            "radar_profile_eligible": False,
            "hardware_validated": False,
            "foundation_model_executed": False,
            "semantic_labels_policy_visible": False,
        },
        "runtime_safety_guard": runtime_safety_guard
        or existing.get("runtime_safety_guard"),
        "sensor_phase_trace": phase_binding,
        "physics": physics or existing.get("physics"),
        "artifact_hashes": artifacts,
    }
    _write_json(receipt_path, receipt)
    (root / "capture_receipt.sha256").write_text(
        f"{sha256_file(receipt_path)}  capture_receipt.json\n", encoding="ascii"
    )


def _capture_fixture(
    root: Path,
    evaluator_manifest: Path,
    *,
    steps: int = 4,
    warmup_steps: int = 0,
    capture_stride: int = 1,
    motion_axis: int = 0,
    motion_sign: float = 1.0,
    motion_step: float = 1.60,
) -> None:
    if (
        steps < 2
        or warmup_steps < 0
        or capture_stride < 1
        or motion_axis not in {0, 1, 2}
        or not np.isfinite(motion_sign)
        or motion_sign == 0.0
        or not np.isfinite(motion_step)
        or motion_step <= 0.0
    ):
        raise ValueError("fixture requires at least two rollout steps and a positive stride")
    dt_s = 0.005
    runtime_frame_count = 1 + warmup_steps + steps
    physics_times = np.asarray(
        [physics_time_ns(frame, dt_s) for frame in range(runtime_frame_count)],
        dtype=np.int64,
    )
    command = physics_times[warmup_steps : warmup_steps + steps]
    effective = physics_times[1 + warmup_steps : 1 + warmup_steps + steps]
    state_indices = np.asarray(
        _captured_frame_indices(steps, capture_stride), dtype=np.int64
    )
    timestamps = effective[state_indices]
    sample_count = len(timestamps)
    if sample_count < 2:
        raise ValueError("fixture requires at least two sensor samples")
    root.mkdir(parents=True, exist_ok=True)
    _private_manifest(evaluator_manifest)
    _write_json(
        root / "capture_progress.json",
        {"schema": "org.rivermark.isaac-capture-progress.v1", "stage": "capture_receipt_finalized"},
    )

    routes = _city_lite_routes()
    structural_aabbs = _structural_aabbs()
    geometry_sha256 = _aabb_hash(structural_aabbs)
    runtime_safety_guard = _runtime_safety_guard_fixture(
        structural_aabbs, steps=steps, warmup_steps=warmup_steps
    )
    literal_fleet_spawn = _literal_fleet_spawn_fixture()
    physics = _physics_fixture(literal_fleet_spawn)
    _write_json(
        root / "public_task.json",
        {
            "schema": "org.rivermark.public-search-task.v1",
            "task_kind": "search3d",
            "task_variant_id": TASK_VARIANT_ID,
            "agent_count": AGENT_COUNT,
            "nominal_object_count": TARGET_COUNT,
            "route_generation": "target-free-citylite-static-geometry-v1",
            "route_conditioning": "public_only",
            "route_family_id": CITY_LITE_ROUTE_FAMILY_A_ID,
            "start_anchor_id": CITY_LITE_START_ANCHOR_A_ID,
            "waypoint_segment_seconds": 5.0,
            "waypoint_reached_radius_m": 0.20,
            "routes_w_m": [
                [list(point) for point in route]
                for route in PUBLIC_ROUTES_W_M
            ],
            "route_contract": {
                "geometry_source": "citylite_structural_aabb_v1",
                "clearance_m": ROUTE_CLEARANCE_M,
                "aabb_geometry_sha256": geometry_sha256,
                "routes_sha256": canonical_payload_sha256(PUBLIC_ROUTES_W_M),
                "all_waypoints_in_command_volume": True,
                "all_segments_clear": True,
            },
            "action_abi": {"kind": "position_waypoint_with_velocity_feedforward"},
            "communication_abi": {"kind": "explicit_public_broadcast"},
            "object_coordinates_in_policy_inputs": False,
        },
    )
    _write_json(
        root / "scene.json",
        {
            "schema": "org.rivermark.public-isaac-scene.v1",
            "environment_id": ENVIRONMENT_ID,
            "agent_count": AGENT_COUNT,
            "agent_prim_expression": "/World/Swarm/Agent_.*/Robot",
            "initial_root_poses_wxyz": [
                [*position, *quaternion]
                for position, quaternion in _city_lite_spawn_states()
            ],
            "literal_fleet": literal_fleet_spawn,
            "fresh_stage": True,
            "legacy_route_or_target_imported": False,
            "static_scene_authority_verified": True,
            "unresolved_reference_count": 0,
            "legacy_prim_count": 0,
            "forbidden_decoration_prim_count": 0,
            "city_task_obstacle_material_closure": city_task_obstacle_material_closure_receipt_template(),
            "scene_contract": {
                "sha256": SCENE_CONTRACT_SHA256,
                "payload_sha256": SCENE_CONTRACT_PAYLOAD_SHA256,
                "schema": SCENE_CONTRACT_SCHEMA,
                "gate_status": SCENE_CONTRACT_GATE_STATUS,
                "permissions": dict(EXPECTED_UPSTREAM_PERMISSIONS),
            },
            "authority_assets": {
                filename: {"path": f"authority/{filename}", "sha256": digest}
                for filename, digest in AUTHORITY_SHA256.items()
            },
            "selective_references": [
                {"source_prim": source, "destination_prim": destination}
                for source, destination in SELECTIVE_REFERENCES
            ],
            "rivermark_layer_inventory": _rivermark_layer_inventory(root),
            "stage_units": {
                "meters_per_unit": 1.0,
                "up_axis": "Z",
                "time_codes_per_second": 60.0,
                "frames_per_second": 60.0,
            },
            "native_collision_counts": dict(EXPECTED_NATIVE_COLLISION_COUNTS),
            "flight_volume_m": {"x": [-46.0, 46.0], "y": [-48.0, 44.0], "z": [8.9, 15.0]},
            "command_volume_m": {"x": [-46.0, 46.0], "y": [-48.0, 44.0], "z": [9.0, 14.25]},
            "route_clearance_m": ROUTE_CLEARANCE_M,
            "runtime_safety_guard": runtime_safety_guard,
            "structural_aabbs": structural_aabbs,
            "collision_proxies": {
                "count": len(structural_aabbs),
                "aabb_geometry_sha256": geometry_sha256,
                "source_aabb_geometry_sha256": geometry_sha256,
                "representation": "conservative_world_aabb",
                "prim_root": "/World/StaticScene/CollisionProxies",
                "collision_enabled": True,
                "visible": False,
            },
            "lidar_geometry_coverage": {
                "includes_city": True,
                "includes_city_task_obstacles": True,
                "includes_collision_proxies": True,
                "geometry_aabb_sha256": geometry_sha256,
            },
            "private_evaluator_manifest_sha256": sha256_file(evaluator_manifest),
            "formal_benchmark_admission": False,
            "search_object_prim_count": TARGET_COUNT,
            "search_object_paths_listed": False,
            "object_coordinates_in_policy_inputs": False,
            "identity_markers": [f"/World/Marker_{index}" for index in range(AGENT_COUNT)],
            "identity_marker_provenance": {
                "schema": "org.rivermark.isaac-cf2x-identity-marker.v1",
                "shape": "sphere",
                "radius_m": IDENTITY_MARKER_RADIUS_M,
                "collision_enabled": False,
                "body_relative_translation_m": [-0.045, 0.0, 0.075],
                "root_semantic_tags": [
                    ["class", "cf2x"],
                    ["class", "agent_identity"],
                ],
                "markers": [
                    {
                        "agent_id": agent_id,
                        "prim_path": f"/World/Marker_{agent_id}",
                        "semantic_tags": [
                            ["class", "agent_identity"],
                            ["agent_id", str(agent_id)],
                        ],
                    }
                    for agent_id in range(AGENT_COUNT)
                ],
            },
            "overview_route_witness_schedule": _public_route_witness_schedule(),
            "public_task_sha256": sha256_file(root / "public_task.json"),
        },
    )
    _write_json(
        root / "calibration.json",
        {
            "onboard_camera": {
                "clipping_range_m": list(ONBOARD_CAMERA_CLIPPING_RANGE_M),
                "visual_intrusion_gate": {
                    "schema": VISUAL_INTRUSION_GATE_SCHEMA,
                    "status": "passed",
                    "contract": _visual_intrusion_gate_contract(),
                    "capture_frame_count": sample_count,
                    "capture_frames": [
                        {
                            "schema": VISUAL_INTRUSION_GATE_SCHEMA,
                            "passed": True,
                            "failures": [],
                            "per_agent": [
                                {"agent_id": agent_id} for agent_id in range(AGENT_COUNT)
                            ],
                        }
                        for _ in range(sample_count)
                    ],
                },
                "content_gate": {
                    "schema": ONBOARD_CONTENT_GATE_SCHEMA,
                    "status": "passed",
                    "contract": _onboard_content_gate_contract(),
                    "capture_frame_count": sample_count,
                    "capture_frames": [
                        {
                            "schema": ONBOARD_CONTENT_GATE_SCHEMA,
                            "passed": True,
                            "failures": [],
                            "per_agent": [
                                {"agent_id": agent_id, "passed": True, "failures": []}
                                for agent_id in range(AGENT_COUNT)
                            ],
                        }
                        for _ in range(sample_count)
                    ],
                },
            },
            "lidar": {
                "implementation": "fixture-ray-caster",
                "max_distance_m": 35.0,
            },
            "overview_camera": {
                "route_witness_schedule": _public_route_witness_schedule(),
                "clipping_range_m": [0.05, 200.0],
                "data_types": [
                    "rgb",
                    "distance_to_image_plane",
                    "semantic_segmentation",
                ],
                "content_gate": {
                    "schema": OVERVIEW_CONTENT_GATE_SCHEMA,
                    "status": "passed",
                    "contract": _overview_content_gate_contract(),
                    "initial_post_render": {
                        "schema": OVERVIEW_CONTENT_GATE_SCHEMA,
                        "passed": True,
                        "failures": [],
                    },
                    "capture_frame_count": sample_count,
                    "capture_frames": [
                        {
                            "schema": OVERVIEW_CONTENT_GATE_SCHEMA,
                            "passed": True,
                            "failures": [],
                        }
                        for _ in range(sample_count)
                    ],
                },
                "tracked_agent_visibility_gate": {
                    "schema": "org.rivermark.isaac-route-witness-agent-visibility.v1",
                    "status": "passed",
                    "tracked_agent_id": OVERVIEW_WITNESS_TRACKED_AGENT_ID,
                    "minimum_tracked_agent_pixels": OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS,
                    "initial_post_render": {
                        "schema": "org.rivermark.isaac-route-witness-agent-visibility.v1",
                        "effective_time_ns": 0,
                        "witness_shot_index": 0,
                        "passed": True,
                        "failures": [],
                    },
                    "capture_frame_count": sample_count,
                    "capture_frames": [
                        {
                            "schema": "org.rivermark.isaac-route-witness-agent-visibility.v1",
                            "tracked_agent_id": OVERVIEW_WITNESS_TRACKED_AGENT_ID,
                            "tracked_agent_pixel_count": OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS,
                            "effective_time_ns": int(timestamps[frame_id]),
                            "witness_shot_index": _public_route_witness_view_at_time_ns(
                                int(timestamps[frame_id])
                            )["shot_index"],
                            "passed": True,
                            "failures": [],
                        }
                        for frame_id in range(sample_count)
                    ],
                },
            },
            "radar": {"status": "not_captured", "fail_closed": True},
        },
    )

    positions = np.repeat(routes[:, 0][None, :, :], steps, axis=0)
    # A stride-two capture of the four-step fixture must still prove the
    # route-witness CF2X moved more than its 3 m public-demo minimum.
    positions[:, :, motion_axis] += (
        np.arange(steps, dtype=np.float32)[:, None] * motion_step * motion_sign
    )
    velocities = np.zeros_like(positions)
    velocities[:, :, 0] = 0.30
    quaternion = np.zeros((steps, AGENT_COUNT, 4), dtype=np.float32)
    quaternion[..., 0] = 1.0
    desired = positions.copy()
    desired[:, :, motion_axis] += (
        np.arange(steps, dtype=np.float32)[:, None] * (0.03 * motion_step / 1.60) * motion_sign
    )
    target = np.full((steps, AGENT_COUNT, 4), 0.08, dtype=np.float32)
    applied = target.copy()
    applied[0] = 0.04
    applied[1] = 0.06
    _savez(
        root / "streams/state_action.npz",
        command_time_ns=command,
        effective_time_ns=effective,
        root_pos_w_m=positions,
        root_quat_wxyz=quaternion,
        root_lin_vel_w_mps=velocities,
        root_ang_vel_b_radps=np.zeros_like(positions),
        desired_pos_w_m=desired,
        desired_vel_w_mps=velocities,
        target_thrust_n=target,
        applied_thrust_n=applied,
    )

    waypoint_progression = np.minimum(
        np.arange(1, sample_count + 1, dtype=np.int64), 2
    )[:, None]
    waypoint_index = np.repeat(waypoint_progression, AGENT_COUNT, axis=1)
    desired_waypoint = np.empty((sample_count, AGENT_COUNT, 3), dtype=np.float32)
    for agent_id in range(AGENT_COUNT):
        desired_waypoint[:, agent_id] = routes[agent_id, waypoint_index[:, agent_id]]
    _savez(
        root / "streams/public_task.npz",
        timestamps_ns=timestamps,
        waypoint_index=waypoint_index,
        waypoint_progress=np.full((sample_count, AGENT_COUNT), 0.5, dtype=np.float32),
        desired_waypoint_w_m=desired_waypoint,
        distance_to_waypoint_m=np.full((sample_count, AGENT_COUNT), 0.2, dtype=np.float32),
        waypoint_reached=np.zeros((sample_count, AGENT_COUNT), dtype=np.bool_),
        action_mode=np.zeros((sample_count, AGENT_COUNT), dtype=np.int8),
        coverage_cell_id=np.repeat(np.arange(AGENT_COUNT, dtype=np.int64)[None, :], sample_count, axis=0),
        task_time_s=np.repeat(np.arange(1, sample_count + 1, dtype=np.float32)[:, None], AGENT_COUNT, axis=1),
    )
    _savez(
        root / "streams/public_messages.npz",
        timestamps_ns=timestamps,
        sender_agent_id=np.repeat(np.arange(AGENT_COUNT, dtype=np.int64)[None, :], sample_count, axis=0),
        message_sequence=np.repeat(np.arange(sample_count, dtype=np.int64)[:, None], AGENT_COUNT, axis=1),
        message_waypoint_index=waypoint_index,
        message_position_w_m=positions[state_indices],
        message_velocity_w_mps=velocities[state_indices],
        message_flags=np.ones((sample_count, AGENT_COUNT), dtype=np.uint8),
    )

    camera_vector = positions[state_indices]
    camera_quaternion = quaternion[state_indices]
    closure = np.zeros((sample_count, AGENT_COUNT), dtype=np.float32)
    _savez(
        root / "sensors/camera_poses.npz",
        timestamps_ns=timestamps,
        camera_expected_pos_w_m=camera_vector,
        camera_expected_quat_wxyz=camera_quaternion,
        camera_observed_pos_w_m=camera_vector,
        camera_observed_quat_wxyz=camera_quaternion,
        camera_position_error_m=closure,
        camera_orientation_error_rad=closure,
        camera_usd_position_error_m=closure,
        camera_usd_forward_alignment_cosine=np.ones(
            (sample_count, AGENT_COUNT), dtype=np.float32
        ),
        camera_usd_orientation_error_rad=closure,
        camera_fabric_observed_pos_w_m=camera_vector,
        camera_fabric_observed_quat_wxyz=camera_quaternion,
        camera_fabric_position_error_m=closure,
        camera_fabric_orientation_error_rad=closure,
        camera_render_read_pre_frame_index=np.repeat(
            np.arange(sample_count, dtype=np.int64)[:, None], AGENT_COUNT, axis=1
        ),
        camera_render_read_post_frame_index=np.repeat(
            np.arange(1, sample_count + 1, dtype=np.int64)[:, None], AGENT_COUNT, axis=1
        ),
    )
    rgb = np.zeros((sample_count, AGENT_COUNT, 8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(8, dtype=np.uint8)[None, None, None, :, None]
    _savez(
        root / "sensors/onboard_rgbd.npz",
        timestamps_ns=timestamps,
        rgb=rgb,
        distance_to_image_plane_m=np.ones((sample_count, AGENT_COUNT, 8, 8, 1), dtype=np.float32),
    )
    semantic = np.full((sample_count, AGENT_COUNT, 8, 8, 1), 5, dtype=np.int32)
    for target_index in range(TARGET_COUNT):
        semantic[:, target_index, 0:3, 0:3, 0] = target_index + 1
    _savez(
        root / "learning_labels/semantic_segmentation.npz",
        timestamps_ns=timestamps,
        semantic_segmentation=semantic,
    )
    onboard_frame_mapping = {
        "per_camera": [
            {
                "id_to_labels": {
                    "0": {"class": "BACKGROUND"},
                    "5": {"class": "building"},
                    str(camera_index + 1): {
                        "class": f"search_target_slot_{camera_index:03d}",
                    },
                }
            }
            for camera_index in range(AGENT_COUNT)
        ]
    }
    overview_frame_mapping = {
        "per_camera": [
            {
                "id_to_labels": {
                    "9": {"class": "prop_structure"},
                    "11": {
                        "class": "agent_identity,cf2x",
                        "agent_id": str(OVERVIEW_WITNESS_TRACKED_AGENT_ID),
                    },
                }
            }
        ]
    }
    semantic_metadata_rows = [
        {
            "schema": "org.rivermark.isaac-semantic-frame-metadata.v1",
            "frame_index": frame_index,
            "timestamp_ns": int(timestamp),
            "onboard_replicator_info": onboard_frame_mapping,
            "overview_replicator_info": overview_frame_mapping,
        }
        for frame_index, timestamp in enumerate(timestamps)
    ]
    (root / "learning_labels/semantic_frame_metadata.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in semantic_metadata_rows
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "learning_labels/semantic_metadata.json",
        {
            "schema": "org.rivermark.isaac-semantic-metadata.v2",
            "partition": "learning_labels",
            "policy_visible": False,
            "frame_metadata": {
                "schema": "org.rivermark.isaac-semantic-frame-metadata.v1",
                "path": "learning_labels/semantic_frame_metadata.jsonl",
                "frame_count": sample_count,
                "onboard_camera_count": AGENT_COUNT,
                "overview_camera_count": 1,
                "record_fields": [
                    "schema",
                    "frame_index",
                    "timestamp_ns",
                    "onboard_replicator_info",
                    "overview_replicator_info",
                ],
            },
        },
    )
    ranges = np.full((sample_count, AGENT_COUNT, 32), 35.0, dtype=np.float32)
    ranges[..., 0] = 2.0
    _savez(
        root / "sensors/lidar.npz",
        timestamps_ns=timestamps,
        pos_w_m=camera_vector,
        quat_wxyz=camera_quaternion,
        ranges_m=ranges,
    )
    _savez(
        root / "sensors/imu.npz",
        timestamps_ns=timestamps,
        pos_w_m=camera_vector,
        quat_wxyz=camera_quaternion,
        linear_acceleration_b_mps2=np.zeros((sample_count, AGENT_COUNT, 3), dtype=np.float32),
        angular_velocity_b_radps=np.zeros((sample_count, AGENT_COUNT, 3), dtype=np.float32),
    )
    _savez(
        root / "sensors/contact.npz",
        timestamps_ns=timestamps,
        net_forces_w_n=np.zeros((sample_count, AGENT_COUNT, 1, 3), dtype=np.float32),
    )
    contact_frame = np.zeros((AGENT_COUNT, 1, 3), dtype=np.float32)
    _savez(
        root / SENSOR_PHASE_TRACE_RELATIVE_PATH,
        schema=np.asarray([SENSOR_PHASE_TRACE_SCHEMA]),
        sensor_names=np.asarray(SENSOR_PHASE_SENSOR_NAMES),
        physics_step=np.asarray(
            warmup_steps + state_indices + 1, dtype=np.int64
        ),
        physics_time_ns=timestamps,
        event_codes=np.repeat(
            np.asarray(SENSOR_PHASE_EVENT_SEQUENCE, dtype=np.uint8)[None, :],
            sample_count,
            axis=0,
        ),
        retained_contact_sha256=np.repeat(
            np.frombuffer(sensor_phase_array_digest(contact_frame), dtype=np.uint8)[None, :],
            sample_count,
            axis=0,
        ),
        archive_frame_index=np.arange(sample_count, dtype=np.int64),
    )
    runtime_positions = np.concatenate(
        (np.repeat(positions[:1], 1 + warmup_steps, axis=0), positions),
        axis=0,
    )
    runtime_safety_guard["checks"]["minimum_inter_agent_swept_separation_m"] = float(
        min(
            np.linalg.norm(frame[left] - frame[right])
            for frame in runtime_positions
            for left in range(AGENT_COUNT - 1)
            for right in range(left + 1, AGENT_COUNT)
        )
    )
    runtime_trace_path = root / RUNTIME_SAFETY_TRACE_RELATIVE_PATH
    _savez(
        runtime_trace_path,
        physics_step=np.arange(runtime_frame_count, dtype=np.int64),
        physics_time_ns=physics_times,
        phase_code=np.asarray(
            [RUNTIME_SAFETY_PHASE_CODES["post_reset"]]
            + [RUNTIME_SAFETY_PHASE_CODES["warmup"]] * warmup_steps
            + [RUNTIME_SAFETY_PHASE_CODES["rollout"]] * steps,
            dtype=np.int8,
        ),
        frame_outcome_code=np.full(
            runtime_frame_count,
            RUNTIME_SAFETY_FRAME_OUTCOME_CODES["passed"],
            dtype=np.uint8,
        ),
        root_pos_w_m=runtime_positions,
        net_contact_forces_w_n=np.zeros(
            (runtime_frame_count, AGENT_COUNT, 1, 3), dtype=np.float32
        ),
        max_contact_force_n=np.zeros(runtime_frame_count, dtype=np.float32),
    )
    finalize_runtime_safety_guard(
        runtime_safety_guard,
        trace_sha256=sha256_file(runtime_trace_path),
        physics_frame_count=runtime_frame_count,
    )
    scene_path = root / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["runtime_safety_guard"] = runtime_safety_guard
    _write_json(scene_path, scene)
    overview = np.zeros((sample_count, 16, 16, 3), dtype=np.uint8)
    overview[:, :, ::2, 0] = 255
    overview_depth = np.broadcast_to(
        np.linspace(12.0, 60.0, 16, dtype=np.float32)[None, None, :, None],
        (sample_count, 16, 16, 1),
    ).copy()
    overview_semantic = np.zeros((sample_count, 16, 16, 1), dtype=np.int32)
    overview_semantic[:, 4:12, 4:12] = 9
    overview_semantic[:, 6:12, 6:12] = 11
    witness_views = [
        _public_route_witness_view_at_time_ns(int(timestamp))
        for timestamp in timestamps
    ]
    _savez(
        root / "sensors/overview_rgb.npz",
        timestamps_ns=timestamps,
        rgb=overview,
        distance_to_image_plane_m=overview_depth,
        semantic_segmentation=overview_semantic,
        camera_pos_w_m=np.asarray(
            [view["eye_w_m"] for view in witness_views], dtype=np.float32
        ),
        camera_quat_wxyz=np.asarray(
            [view["orientation_wxyz"] for view in witness_views], dtype=np.float32
        ),
        target_w_m=np.asarray(
            [view["target_w_m"] for view in witness_views], dtype=np.float32
        ),
    )
    _write_json(
        root / "task_outcome.json",
        {
            "schema": T1_OBSERVABILITY_OUTCOME_SCHEMA,
            "track": T1_DATA_TRACK_ID,
            "task_variant_id": TASK_VARIANT_ID,
            "scoring_status": "not_scored",
            "search_score": None,
            "object_count": TARGET_COUNT,
            "target_observability": {
                "schema": "org.rivermark.isaac-target-visibility-summary.v2",
                "target_count": TARGET_COUNT,
                "targets_meeting_visibility": TARGET_COUNT,
                "minimum_visible_sensor_frames_per_target": 1,
                "minimum_visible_instance_pixels": 8,
                "passed": True,
                "failed_target_count": 0,
                "failed_target_slots": [],
                "per_target_slot": {
                    f"search_target_slot_{target_index:03d}": {
                        "max_pixels": 9,
                        "visible_frames": sample_count,
                    }
                    for target_index in range(TARGET_COUNT)
                },
            },
            "observation_rule": "native onboard semantic anonymous-slot-class visibility",
            "policy_confirmation_events_present": False,
            "closed_loop_scoring_eligible": False,
            "private_manifest_commitment_sha256": sha256_file(evaluator_manifest),
            "state_action_sha256": sha256_file(root / "streams/state_action.npz"),
            "private_coordinates_released": False,
        },
    )
    _bind_receipt(
        root,
        evaluator_manifest,
        command={
            "seed": 20260723,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "capture_stride": capture_stride,
            "dt_s": dt_s,
        },
        runtime_safety_guard=runtime_safety_guard,
        physics=physics,
    )


def _rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    mutate(arrays)
    np.savez_compressed(path, **arrays)


def _rewrite_low_rate_overview_fixture(root: Path) -> tuple[int, ...]:
    """Convert the legacy fixture into the new no-depth overview contract."""

    overview_path = root / "sensors/overview_rgb.npz"
    with np.load(overview_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    indices = _overview_archive_frame_indices(len(arrays["timestamps_ns"]))
    selected = np.asarray(indices, dtype=np.int64)
    _savez(
        overview_path,
        timestamps_ns=arrays["timestamps_ns"][selected],
        rgb=arrays["rgb"][selected],
        semantic_segmentation=arrays["semantic_segmentation"][selected],
        camera_pos_w_m=arrays["camera_pos_w_m"][selected],
        camera_quat_wxyz=arrays["camera_quat_wxyz"][selected],
        target_w_m=arrays["target_w_m"][selected],
    )
    calibration_path = root / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    overview_calibration = calibration["overview_camera"]
    overview_calibration["evidence_archive"] = {
        "schema": OVERVIEW_ARCHIVE_SCHEMA,
        "selection_rule": "first_each_fixed_retained_frame_stride_and_final",
        "frame_index_stride": OVERVIEW_ARCHIVE_STRIDE,
        "source_frame_count": len(arrays["timestamps_ns"]),
        "source_frame_indices": list(indices),
        "stored_fields": [
            "rgb",
            "semantic_segmentation",
            "camera_pos_w_m",
            "camera_quat_wxyz",
            "target_w_m",
        ],
        "runtime_only_render_products": ["distance_to_image_plane"],
        "selection_uses_content_or_outcome": False,
    }
    for frame, timestamp in zip(
        overview_calibration["content_gate"]["capture_frames"],
        arrays["timestamps_ns"],
    ):
        frame.update(
            {
                "effective_time_ns": int(timestamp),
                "witness_shot_index": _public_route_witness_view_at_time_ns(
                    int(timestamp)
                )["shot_index"],
                "city_evidence_passed": True,
                "structural_evidence_passed": True,
            }
        )
    _write_json(calibration_path, calibration)
    return indices


def _rebind_state_outcome(root: Path, evaluator_manifest: Path) -> None:
    outcome_path = root / "task_outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["state_action_sha256"] = sha256_file(root / "streams/state_action.npz")
    _write_json(outcome_path, outcome)
    _bind_receipt(root, evaluator_manifest)


def _rebind_evaluator_commitments(root: Path, evaluator_manifest: Path) -> None:
    commitment = sha256_file(evaluator_manifest)
    scene_path = root / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["private_evaluator_manifest_sha256"] = commitment
    _write_json(scene_path, scene)
    outcome_path = root / "task_outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["private_manifest_commitment_sha256"] = commitment
    _write_json(outcome_path, outcome)
    _bind_receipt(root, evaluator_manifest)


def _set_evaluator_manifest_retention(
    root: Path,
    retention: dict[str, object] | None,
) -> None:
    """Set only public retention commitment metadata on a fixture receipt."""

    receipt_path = root / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if retention is None:
        receipt.pop("evaluator_manifest_retention", None)
    else:
        receipt["evaluator_manifest_retention"] = retention
    _write_json(receipt_path, receipt)
    (root / "capture_receipt.sha256").write_text(
        f"{sha256_file(receipt_path)}  capture_receipt.json\n", encoding="ascii"
    )


def _codes(report: IsaacValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


class IsaacIndependentValidationTests(unittest.TestCase):
    def test_complete_search3d_bundle_passes_all_core_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private" / "evaluator.json"
            _capture_fixture(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private, require_clean_source=True)
            self.assertTrue(report.valid, report.issues)

            self.assertEqual(set(json.loads((root / "capture_receipt.json").read_text())["artifact_hashes"]), EXPECTED_ARTIFACTS)
            self.assertEqual(report.checks["physics_steps"], 4)
            self.assertEqual(report.checks["sensor_samples"], 4)
            self.assertTrue(report.checks["evaluator_manifest_verified"])
            self.assertTrue(report.checks["action_causality_audit_passed"])
            self.assertTrue(report.checks["sensor_decode_audit_passed"])
            self.assertTrue(report.checks["policy_leakage_audit_passed"])
            self.assertTrue(report.checks["city_lite_scene_audit_passed"])
            self.assertTrue(report.checks["city_lite_authority_verified"])
            self.assertTrue(report.checks["rivermark_layer_inventory_verified"])
            self.assertTrue(report.checks["trajectory_in_city_lite_flight_volume"])
            self.assertTrue(report.checks["contact_free"])
            self.assertTrue(report.checks["collision_proxy_geometry_verified"])
            self.assertTrue(report.checks["evaluator_binding_verified"])
            self.assertTrue(report.checks["private_target_geometry_verified"])
            self.assertTrue(report.checks["trajectory_segment_clearance_verified"])
            self.assertTrue(report.checks["route_geometry_audit_passed"])
            self.assertNotIn(
                "evaluator_manifest_retention",
                json.loads((root / "capture_receipt.json").read_text(encoding="utf-8")),
            )
            self.assertEqual(report.checks["selective_reference_count"], 2)
            self.assertEqual(report.checks["structural_aabb_count"], 2)
            self.assertFalse(report.checks["radar_captured"])
            declared_outcome = json.loads(
                (root / "task_outcome.json").read_text(encoding="utf-8")
            )["target_observability"]
            self.assertEqual(report.checks["target_observability"], declared_outcome)

    def test_private_manifest_retention_commitment_is_public_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private" / "evaluator.json"
            _capture_fixture(root, private)
            valid = {
                "kind": PRIVATE_MANIFEST_RETENTION_KIND,
                "sha256": sha256_file(private),
                "bytes": private.stat().st_size,
                "path_released": False,
                "payload_released": False,
            }
            self.assertLessEqual(valid["bytes"], PRIVATE_MANIFEST_RETENTION_MAX_BYTES)
            _set_evaluator_manifest_retention(root, valid)
            self.assertTrue(
                validate_isaac_capture(root, evaluator_manifest=private).valid
            )

            invalid_commitments = (
                {**valid, "kind": "unrecognized-retention-kind"},
                {**valid, "sha256": "0" * 64},
                {**valid, "bytes": 0},
                {**valid, "path_released": True},
                {**valid, "payload_released": True},
                {**valid, "path": "operator-private/manifest.json"},
            )
            for invalid in invalid_commitments:
                _set_evaluator_manifest_retention(root, invalid)
                self.assertIn(
                    "evaluator_manifest_retention",
                    _codes(validate_isaac_capture(root, evaluator_manifest=private)),
                )

    def test_low_rate_overview_is_exact_schedule_without_persisted_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            indices = _rewrite_low_rate_overview_fixture(root)
            _bind_receipt(root, private)

            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, report.issues)
            self.assertEqual(report.checks["overview_archive_frame_indices"], list(indices))
            self.assertEqual(report.checks["overview_archive_source_frame_count"], 4)
            self.assertFalse(report.checks["overview_live_depth_gate_persisted"])
            self.assertTrue(report.checks["overview_live_depth_gate_verified"])
            self.assertTrue(report.checks["overview_archive_visual_verified"])

    def test_low_rate_overview_rejects_schedule_tamper_and_extra_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_low_rate_overview_fixture(root)
            _rewrite_npz(
                root / "sensors/overview_rgb.npz",
                lambda arrays: arrays["timestamps_ns"].__setitem__(1, 1),
            )
            _bind_receipt(root, private)
            self.assertIn(
                "overview_timestamp_schedule",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

            _capture_fixture(root, private)
            _rewrite_low_rate_overview_fixture(root)
            _rewrite_npz(
                root / "sensors/overview_rgb.npz",
                lambda arrays: arrays.__setitem__(
                    "distance_to_image_plane_m",
                    np.ones((*arrays["rgb"].shape[:-1], 1), dtype=np.float32),
                ),
            )
            _bind_receipt(root, private)
            self.assertIn(
                "npz_fields",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

    def test_public_task_does_not_expose_private_target_holdout_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            public_task = json.loads(
                (root / "public_task.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("target_region_id", public_task)
            self.assertNotIn("visibility_bucket", public_task)
            self.assertEqual(public_task["route_family_id"], CITY_LITE_ROUTE_FAMILY_A_ID)
            self.assertEqual(public_task["start_anchor_id"], CITY_LITE_START_ANCHOR_A_ID)

    def test_capture_progress_private_target_identifier_and_coordinate_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            private_payload = json.loads(private.read_text(encoding="utf-8"))
            first_target = private_payload["targets"][0]
            _write_json(
                root / "capture_progress.json",
                {
                    "schema": "org.rivermark.isaac-capture-progress.v1",
                    "stage": "failed",
                    "target_id": first_target["target_id"],
                    "position_w_m": first_target["position_w_m"],
                },
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("public_private_leakage", _codes(report))

    def test_post_validation_video_evidence_is_allowed_only_when_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            receipt = json.loads((root / "capture_receipt.json").read_text(encoding="utf-8"))
            video = root / "videos" / "overview.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"fixture-mp4")
            video_sha256 = sha256_file(video)
            video_receipt = {
                "schema": "org.rivermark.isaac-demo-video.v1",
                "ok": True,
                "capture_receipt_sha256": sha256_file(root / "capture_receipt.json"),
                "video_sha256": video_sha256,
                "audit": {"bytes": video.stat().st_size, "sha256": video_sha256},
                "input_artifacts": {
                    "sensors/overview_rgb.npz": receipt["artifact_hashes"]["sensors/overview_rgb.npz"]
                },
            }
            _write_json(video.with_suffix(".mp4.receipt.json"), video_receipt)
            _bind_receipt(root, private)
            validation = root / "independent_validation.json"
            _write_json(
                validation,
                {
                    "schema": "org.rivermark.isaac-independent-validation.v1",
                    "status": "passed",
                    "capture_receipt_sha256": sha256_file(root / "capture_receipt.json"),
                },
            )
            # Re-read the receipt hash after the fixture helper refreshed it.
            video_receipt["capture_receipt_sha256"] = sha256_file(root / "capture_receipt.json")
            video_receipt["independent_validation_sha256"] = sha256_file(validation)
            _write_json(video.with_suffix(".mp4.receipt.json"), video_receipt)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, [issue.code for issue in report.issues])

            video_receipt["independent_validation_sha256"] = "0" * 64
            _write_json(video.with_suffix(".mp4.receipt.json"), video_receipt)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("video_artifact", _codes(report))

            video_receipt["independent_validation_sha256"] = sha256_file(validation)
            _write_json(video.with_suffix(".mp4.receipt.json"), video_receipt)
            video_receipt_path = video.with_suffix(".mp4.receipt.json")
            video_receipt_path.unlink()
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("video_artifact", _codes(report))

    def test_capture_start_control_marker_is_allowed_but_not_required_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private" / "evaluator.json"
            _capture_fixture(root, private)
            marker = {
                "schema": "org.rivermark.isaac-capture-start.v1",
                "attempt_id": "attempt-" + "a" * 32,
                "started_wall_time_ns": 1,
            }
            _write_json(root / "capture_start.json", marker)
            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["artifact_hashes"]["capture_start.json"] = {
                "bytes": (root / "capture_start.json").stat().st_size,
                "sha256": sha256_file(root / "capture_start.json"),
            }
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n", encoding="ascii"
            )
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, report.issues)

    def test_runtime_lock_binding_requires_external_hash_preflight_and_live_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            lock_path = ROOT / "config" / "isaac_runtime.windows-5.1.json"
            if not lock_path.is_file():
                self.skipTest("local runtime lock fixture is unavailable")
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            digest = runtime_lock_sha256(lock)
            simulation = lock["simulation"]
            launcher = lock["launcher"]
            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["runtime_lock"] = {
                "path": str(lock_path.resolve()),
                "profile_id": lock["profile_id"],
                "sha256": digest,
            }
            receipt["capture_integrity"]["sensor_step_order"] = [
                "command_write",
                "simulation_step",
                "state_update",
                "safety_contact_read",
                "camera_pose_update",
                "render",
                "rgbd_lidar_imu_read",
                "retained_contact_read",
                "storage",
            ]
            receipt["capture_integrity"]["per_physics_step_safety_contact_reads"] = True
            receipt["capture_integrity"]["retained_contact_read_in_synchronous_sensor_phase"] = True
            receipt["preflight"] = {
                "checks": [
                    {
                        "name": "runtime_lock",
                        "passed": True,
                        "value": {
                            "status": "passed",
                            "profile_id": lock["profile_id"],
                            "runtime_lock_sha256": digest,
                        },
                    }
                ]
            }
            receipt["runtime_live"] = {
                "device": simulation["device"],
                "physics_dt_s": simulation["dt_s"],
                "rendering_dt_s": simulation["dt_s"] * simulation["render_interval"],
                "gravity_w_mps2": simulation["gravity_w_mps2"],
                "render_interval": simulation["render_interval"],
                "use_fabric": simulation["use_fabric"],
                "enable_scene_query_support": simulation["config_digests"]["fabric"]["settings"]["enable_scene_query_support"],
                "rendering_mode": launcher["rendering_mode"],
                "rtx_sensors_active": True,
                "config_digests": {
                    name: value["sha256"]
                    for name, value in simulation["config_digests"].items()
                },
                "configuration_observation": "public_simulation_context_and_locked_cfg",
            }
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n", encoding="ascii"
            )
            report = validate_isaac_capture(
                root,
                evaluator_manifest=private,
                runtime_lock_path=lock_path,
            )
            self.assertTrue(report.valid, report.issues)
            self.assertTrue(report.checks["runtime_lock_verified"])
            receipt["runtime_live"]["gravity_w_mps2"] = [
                0.0,
                0.0,
                -9.8100004196167,
            ]
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n", encoding="ascii"
            )
            self.assertTrue(
                validate_isaac_capture(
                    root,
                    evaluator_manifest=private,
                    runtime_lock_path=lock_path,
                ).valid
            )
            receipt["runtime_live"]["physics_dt_s"] = 0.01
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n", encoding="ascii"
            )
            self.assertIn(
                "runtime_lock_live",
                _codes(validate_isaac_capture(root, evaluator_manifest=private, runtime_lock_path=lock_path)),
            )
            receipt["runtime_live"]["physics_dt_s"] = simulation["dt_s"]
            receipt["runtime_live"]["gravity_w_mps2"] = [0.0, 0.0, -9.81001]
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n", encoding="ascii"
            )
            self.assertIn(
                "runtime_lock_live",
                _codes(validate_isaac_capture(root, evaluator_manifest=private, runtime_lock_path=lock_path)),
            )

    def test_chunked_sensor_frames_pass_the_same_independent_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)

            def load(path: Path) -> dict[str, np.ndarray]:
                with np.load(path, allow_pickle=False) as archive:
                    return {name: archive[name].copy() for name in archive.files}

            onboard_path = root / "sensors/onboard_rgbd.npz"
            semantic_path = root / "learning_labels/semantic_segmentation.npz"
            overview_path = root / "sensors/overview_rgb.npz"
            onboard = load(onboard_path)
            semantic = load(semantic_path)
            overview = load(overview_path)
            write_chunked_frame_archive(
                onboard_path,
                timestamps_ns=onboard["timestamps_ns"],
                inline_fields={},
                frame_fields={
                    "rgb": onboard["rgb"],
                    "distance_to_image_plane_m": onboard["distance_to_image_plane_m"],
                },
            )
            write_chunked_frame_archive(
                semantic_path,
                timestamps_ns=semantic["timestamps_ns"],
                inline_fields={},
                frame_fields={"semantic_segmentation": semantic["semantic_segmentation"]},
            )
            write_chunked_frame_archive(
                overview_path,
                timestamps_ns=overview["timestamps_ns"],
                inline_fields={
                    "camera_pos_w_m": overview["camera_pos_w_m"],
                    "camera_quat_wxyz": overview["camera_quat_wxyz"],
                    "target_w_m": overview["target_w_m"],
                },
                frame_fields={
                    "rgb": overview["rgb"],
                    "distance_to_image_plane_m": overview["distance_to_image_plane_m"],
                    "semantic_segmentation": overview["semantic_segmentation"],
                },
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, report.issues)
            self.assertTrue(report.checks["visual_intrusion_verified"])
            self.assertTrue(report.checks["overview_city_content_verified"])

    def test_literal_fleet_spawn_evidence_is_required_and_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)

            baseline = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertNotIn("literal_fleet_spawn", _codes(baseline))
            self.assertTrue(baseline.checks["literal_fleet_spawn_verified"])

            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            physics = receipt["physics"]
            assert isinstance(physics, dict)
            literal = physics["literal_fleet_spawn"]
            assert isinstance(literal, dict)
            usd = literal["authored_usd_transform"]
            assert isinstance(usd, dict)
            rows = usd["per_agent"]
            assert isinstance(rows, list)
            first_row = rows[0]
            assert isinstance(first_row, dict)
            first_row["basis_axis_lengths"] = [1.0, 1.1, 1.0]
            _write_json(receipt_path, receipt)

            scene_path = root / "scene.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            scene_literal = scene["literal_fleet"]
            assert isinstance(scene_literal, dict)
            scene_usd = scene_literal["authored_usd_transform"]
            assert isinstance(scene_usd, dict)
            scene_rows = scene_usd["per_agent"]
            assert isinstance(scene_rows, list)
            scene_first_row = scene_rows[0]
            assert isinstance(scene_first_row, dict)
            scene_first_row["basis_axis_lengths"] = [1.0, 1.1, 1.0]
            _write_json(scene_path, scene)
            _bind_receipt(root, private)

            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("literal_fleet_spawn", _codes(report))
            self.assertFalse(report.checks["literal_fleet_spawn_verified"])

    def test_literal_fleet_spawn_reconstructs_route_family_b_starts(self) -> None:
        literal = _literal_fleet_spawn_fixture()
        receipt = {"physics": _physics_fixture(literal)}
        states = _city_lite_spawn_states(PUBLIC_ROUTES_B_W_M)
        scene = {
            "initial_root_poses_wxyz": [
                [*position, *quaternion] for position, quaternion in states
            ],
            "literal_fleet": literal,
        }
        issues: list[ValidationIssue] = []
        checks: dict[str, object] = {}

        self.assertTrue(
            _validate_literal_city_lite_fleet_spawn(
                receipt,
                scene,
                issues,
                checks,
                routes_w_m=PUBLIC_ROUTES_B_W_M,
            )
        )
        self.assertEqual(issues, [])
        self.assertTrue(checks["literal_fleet_spawn_verified"])

    def test_malformed_route_family_type_fails_closed_without_validator_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            public_path = root / "public_task.json"
            public_task = json.loads(public_path.read_text(encoding="utf-8"))
            public_task["route_family_id"] = ["malformed"]
            _write_json(public_path, public_task)
            scene_path = root / "scene.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            scene["public_task_sha256"] = sha256_file(public_path)
            _write_json(scene_path, scene)
            _bind_receipt(root, private)

            report = validate_isaac_capture(root, evaluator_manifest=private)

            self.assertFalse(report.valid)
            self.assertIn("route_start", _codes(report))

    def test_runtime_safety_trace_and_scene_guard_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / RUNTIME_SAFETY_TRACE_RELATIVE_PATH,
                lambda arrays: arrays["phase_code"].__setitem__(1, RUNTIME_SAFETY_PHASE_CODES["post_reset"]),
            )
            _bind_receipt(root, private)
            codes = _codes(validate_isaac_capture(root, evaluator_manifest=private))
            self.assertTrue({"runtime_safety_trace", "runtime_safety_guard"}.issubset(codes))

            _capture_fixture(root, private)
            scene_path = root / "scene.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            guard = scene["runtime_safety_guard"]
            assert isinstance(guard, dict)
            checks = guard["checks"]
            assert isinstance(checks, dict)
            checks["swept_segments_checked"] = int(checks["swept_segments_checked"]) + 1
            _write_json(scene_path, scene)
            _bind_receipt(root, private)
            self.assertIn(
                "runtime_safety_guard",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

    def test_runtime_trace_binds_warmup_state_and_capture_stride_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private, warmup_steps=3, capture_stride=2)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, report.issues)
            self.assertEqual(report.checks["runtime_safety_trace_frames"], 8)
            self.assertTrue(report.checks["runtime_safety_trace_timing_bound"])
            with np.load(root / "streams/state_action.npz", allow_pickle=False) as state:
                np.testing.assert_array_equal(
                    state["command_time_ns"],
                    np.array([15_000_000, 20_000_000, 25_000_000, 30_000_000]),
                )
                np.testing.assert_array_equal(
                    state["effective_time_ns"],
                    np.array([20_000_000, 25_000_000, 30_000_000, 35_000_000]),
                )
            with np.load(root / "sensors/contact.npz", allow_pickle=False) as contact:
                np.testing.assert_array_equal(
                    contact["timestamps_ns"],
                    np.array([25_000_000, 35_000_000]),
                )

    def test_runtime_trace_accepts_retained_trailing_partial_stride(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(
                root,
                private,
                steps=5,
                warmup_steps=2,
                capture_stride=2,
                motion_axis=1,
                motion_sign=-1.0,
                motion_step=1.10,
            )
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, report.issues)
            self.assertTrue(report.checks["runtime_safety_trace_timing_bound"])
            self.assertTrue(report.checks["sensor_phase_trace_verified"])

    def test_timing_tampering_is_rejected_even_when_timestamps_remain_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private, warmup_steps=2, capture_stride=2)
            _rewrite_npz(
                root / "streams/state_action.npz",
                lambda arrays: arrays["command_time_ns"].__setitem__(
                    slice(None), arrays["command_time_ns"] + 1
                ),
            )
            _rebind_state_outcome(root, private)
            codes = _codes(validate_isaac_capture(root, evaluator_manifest=private))
            self.assertTrue({"capture_timing", "runtime_safety_trace_binding"}.issubset(codes))

            _capture_fixture(root, private, warmup_steps=2, capture_stride=2)
            sensor_timestamp_paths = (
                "sensors/camera_poses.npz",
                "sensors/contact.npz",
                "sensors/imu.npz",
                "sensors/lidar.npz",
                "sensors/onboard_rgbd.npz",
                "sensors/overview_rgb.npz",
                "learning_labels/semantic_segmentation.npz",
                "streams/public_task.npz",
                "streams/public_messages.npz",
            )
            for relative in sensor_timestamp_paths:
                _rewrite_npz(
                    root / relative,
                    lambda arrays: arrays["timestamps_ns"].__setitem__(
                        slice(None), arrays["timestamps_ns"] + 1
                    ),
                )
            _bind_receipt(root, private)
            codes = _codes(validate_isaac_capture(root, evaluator_manifest=private))
            self.assertTrue({"capture_timing", "runtime_safety_trace_binding"}.issubset(codes))

    def test_runtime_trace_time_outcome_and_float32_contact_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / RUNTIME_SAFETY_TRACE_RELATIVE_PATH,
                lambda arrays: arrays["physics_time_ns"].__setitem__(2, 10_000_001),
            )
            _bind_receipt(root, private)
            self.assertIn(
                "runtime_safety_timing",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

            _capture_fixture(root, private)
            _rewrite_npz(
                root / RUNTIME_SAFETY_TRACE_RELATIVE_PATH,
                lambda arrays: arrays["frame_outcome_code"].__setitem__(
                    1, RUNTIME_SAFETY_FRAME_OUTCOME_CODES["aborted"]
                ),
            )
            _bind_receipt(root, private)
            self.assertIn(
                "runtime_safety_trace",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

            _capture_fixture(root, private)

            def set_float32_abort_force(arrays: dict[str, np.ndarray]) -> None:
                arrays["net_contact_forces_w_n"][1, 0, 0, 0] = np.float32(
                    CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N
                )
                arrays["max_contact_force_n"][1] = np.float32(
                    CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N
                )

            _rewrite_npz(root / RUNTIME_SAFETY_TRACE_RELATIVE_PATH, set_float32_abort_force)
            _bind_receipt(root, private)
            self.assertIn(
                "runtime_safety_trace",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

    def test_empty_contact_body_dimension_is_rejected_without_validator_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/contact.npz",
                lambda arrays: arrays.__setitem__(
                    "net_forces_w_n",
                    np.zeros((3, AGENT_COUNT, 0, 3), dtype=np.float32),
                ),
            )
            _bind_receipt(root, private)
            self.assertIn(
                "contact_shape",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

    def test_capture_receipt_metadata_is_path_scoped_not_a_policy_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            self.assertTrue(
                validate_isaac_capture(root, evaluator_manifest=private).valid
            )

            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["command"]["evaluator_seed"] = 7
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n",
                encoding="ascii",
            )
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("policy_truth_leakage", _codes(report))

    def test_versioned_visibility_window_is_recomputed_from_the_capture_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            command = receipt["command"]
            receipt["target_visibility_execution_window"] = target_visibility_execution_window(
                dt_s=float(command["dt_s"]),
                warmup_steps=command["warmup_steps"],
                rollout_steps=2400,
                capture_stride=10,
                waypoint_segment_seconds=5.0,
            )
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n",
                encoding="ascii",
            )
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("evaluator_target_geometry", _codes(report))

    def test_collection_binding_requires_the_runtime_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["collection_binding"] = {
                "protocol_id": "citylite-coverage-v1",
                "protocol_sha256": "c" * 64,
                "cell_id": "train-route-0",
                "split": "train",
                "episode_index": 3,
                "episode_seed": receipt["command"]["seed"],
            }
            target_closure = {
                "schema": "org.rivermark.runtime-target-usd-closure.v1",
                "target_count": TARGET_COUNT,
                "all_targets_active": True,
                "all_targets_visible": True,
                "all_targets_renderable": True,
                "all_targets_have_expected_class_label": True,
                "all_target_transforms_rigid": True,
                "maximum_world_position_error_m": 0.0,
                "maximum_radius_error_m": 0.0,
                "maximum_bound_extent_error_m": 0.0,
                "position_tolerance_m": 1.0e-5,
                "radius_tolerance_m": 1.0e-5,
                "bound_extent_tolerance_m": 1.0e-5,
            }
            receipt["runtime_target_usd_pre_reset"] = target_closure
            receipt["runtime_target_usd_post_reset"] = dict(target_closure)
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n",
                encoding="ascii",
            )
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, report.issues)
            self.assertTrue(report.checks["collection_binding_verified"])
            self.assertTrue(report.checks["runtime_target_usd_closure_verified"])

            receipt["collection_binding"]["episode_seed"] += 1
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n",
                encoding="ascii",
            )
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("collection_seed", _codes(report))
            self.assertFalse(report.checks["collection_binding_verified"])

    def test_protocol_bound_capture_requires_two_path_free_target_usd_closures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            receipt_path = root / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["collection_binding"] = {
                "protocol_id": "citylite-coverage-v1",
                "protocol_sha256": "c" * 64,
                "cell_id": "train-route-0",
                "split": "train",
                "episode_index": 3,
                "episode_seed": receipt["command"]["seed"],
            }
            closure = {
                "schema": "org.rivermark.runtime-target-usd-closure.v1",
                "target_count": TARGET_COUNT,
                "all_targets_active": True,
                "all_targets_visible": True,
                "all_targets_renderable": True,
                "all_targets_have_expected_class_label": True,
                "all_target_transforms_rigid": True,
                "maximum_world_position_error_m": 0.0,
                "maximum_radius_error_m": 0.0,
                "maximum_bound_extent_error_m": 0.0,
                "position_tolerance_m": 1.0e-5,
                "radius_tolerance_m": 1.0e-5,
                "bound_extent_tolerance_m": 1.0e-5,
            }
            receipt["runtime_target_usd_pre_reset"] = closure
            receipt["runtime_target_usd_post_reset"] = dict(closure)
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n", encoding="ascii"
            )
            passed = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(passed.valid, passed.issues)
            self.assertTrue(passed.checks["runtime_target_usd_closure_verified"])

            receipt.pop("runtime_target_usd_post_reset")
            _write_json(receipt_path, receipt)
            (root / "capture_receipt.sha256").write_text(
                f"{sha256_file(receipt_path)}  capture_receipt.json\n", encoding="ascii"
            )
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("runtime_target_usd_closure", _codes(report))
            self.assertFalse(report.checks["runtime_target_usd_closure_verified"])

    def test_usd_camera_render_closure_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/camera_poses.npz",
                lambda arrays: arrays["camera_usd_forward_alignment_cosine"].__setitem__(
                    (0, 0), 0.99
                ),
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("camera_usd_orientation_closure", _codes(report))
            self.assertFalse(report.checks["pose_closure_audit_passed"])

    def test_usd_camera_full_orientation_closure_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/camera_poses.npz",
                lambda arrays: arrays["camera_usd_orientation_error_rad"].__setitem__(
                    (0, 0), 0.011
                ),
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("camera_usd_full_orientation_closure", _codes(report))
            self.assertFalse(report.checks["pose_closure_audit_passed"])

    def test_fabric_camera_lag_is_retained_without_rejecting_usd_bound_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/camera_poses.npz",
                lambda arrays: arrays["camera_fabric_position_error_m"].__setitem__(
                    (0, 0), 2.5
                ),
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, report.issues)
            self.assertEqual(report.checks["camera_fabric_pose_authority"], "diagnostic_only")
            self.assertEqual(report.checks["camera_fabric_pose_max_error_m"], 2.5)

    def test_nonfinite_fabric_diagnostic_does_not_reject_usd_bound_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/camera_poses.npz",
                lambda arrays: arrays["camera_fabric_position_error_m"].__setitem__(
                    (0, 0), np.nan
                ),
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, report.issues)
            self.assertFalse(report.checks["camera_fabric_pose_finite"])
            self.assertIsNone(report.checks["camera_fabric_pose_max_error_m"])

    def test_rivermark_layer_inventory_is_required_and_hash_bound(self) -> None:
        for scenario in ("missing", "wrong_type", "tampered_hash"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                root, private = base / "capture", base / "private.json"
                _capture_fixture(root, private)
                scene_path = root / "scene.json"
                scene = json.loads(scene_path.read_text(encoding="utf-8"))
                if scenario == "missing":
                    scene.pop("rivermark_layer_inventory")
                elif scenario == "wrong_type":
                    scene["rivermark_layer_inventory"] = []
                else:
                    scene["rivermark_layer_inventory"]["inventory_sha256"] = "0" * 64
                _write_json(scene_path, scene)
                _bind_receipt(root, private)

                report = validate_isaac_capture(root, evaluator_manifest=private)
                self.assertIn("rivermark_layer_inventory", _codes(report))
                self.assertFalse(report.checks["rivermark_layer_inventory_verified"])
                self.assertFalse(report.checks["city_lite_scene_audit_passed"])

    def test_private_manifest_must_be_external_and_target_geometry_is_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)

            payload = json.loads(private.read_text(encoding="utf-8"))
            payload["target_origin"]["candidate_pool_released"] = True
            _write_json(private, payload)
            _rebind_evaluator_commitments(root, private)
            self.assertIn(
                "evaluator_manifest_contract",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            payload = json.loads(private.read_text(encoding="utf-8"))
            public = json.loads((root / "public_task.json").read_text(encoding="utf-8"))
            payload["targets"][0]["position_w_m"] = public["routes_w_m"][3][3]
            _write_json(private, payload)
            _rebind_evaluator_commitments(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("evaluator_target_geometry", _codes(report))
            self.assertFalse(report.checks["private_target_geometry_verified"])

    def test_public_candidate_or_target_payload_and_actual_trajectory_collision_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            public_path = root / "public_task.json"
            public = json.loads(public_path.read_text(encoding="utf-8"))
            public["candidate_pools"] = [[[1.0, 2.0, 3.0]]]
            _write_json(public_path, public)
            scene_path = root / "scene.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            scene["public_task_sha256"] = sha256_file(public_path)
            _write_json(scene_path, scene)
            _bind_receipt(root, private)
            self.assertIn(
                "policy_truth_leakage",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            state_path = root / "streams/state_action.npz"
            _rewrite_npz(
                state_path,
                lambda arrays: arrays["root_pos_w_m"].__setitem__((1, 0), [0.0, 0.0, 11.0]),
            )
            _rebind_state_outcome(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("trajectory_clearance", _codes(report))
            self.assertFalse(report.checks["trajectory_segment_clearance_verified"])

    def test_city_lite_authority_and_composition_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            scene_path = root / "scene.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            scene["environment_id"] = "RIVERMARK_GRAYBOX_v0"
            first_asset = next(iter(AUTHORITY_SHA256))
            scene["authority_assets"][first_asset]["sha256"] = "0" * 64
            scene["selective_references"] = scene["selective_references"][:1]
            scene["unresolved_reference_count"] = 1
            scene["legacy_prim_count"] = 1
            scene["forbidden_decoration_prim_count"] = 1
            material_closure = scene["city_task_obstacle_material_closure"]
            assert isinstance(material_closure, dict)
            material_closure["post_repair_binding_closure"] = False
            scene["stage_units"]["frames_per_second"] = 30.0
            scene["native_collision_counts"]["structural_props"] = 1
            _write_json(scene_path, scene)
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(
                {
                    "environment_id",
                    "authority_assets",
                    "selective_references",
                    "unresolved_reference",
                    "legacy_prims",
                    "decorative_prims",
                    "city_task_obstacle_material_closure",
                    "stage_units",
                    "native_collision_counts",
                }.issubset(_codes(report))
            )
            self.assertFalse(report.checks["city_lite_scene_audit_passed"])
            self.assertFalse(report.checks["city_task_obstacle_material_closure_verified"])

    def test_route_volume_clearance_proxy_and_lidar_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)

            public_path = root / "public_task.json"
            public = json.loads(public_path.read_text(encoding="utf-8"))
            routes = np.asarray(public["routes_w_m"], dtype=np.float32)
            routes[5, 2] = np.asarray([0.0, 0.0, 11.0])
            routes[0, -1] = np.asarray([47.0, 18.0, 11.0])
            public["routes_w_m"] = routes.tolist()
            public["route_contract"]["aabb_geometry_sha256"] = "1" * 64
            _write_json(public_path, public)

            task_path = root / "streams/public_task.npz"
            with np.load(task_path, allow_pickle=False) as archive:
                task = {name: archive[name].copy() for name in archive.files}
            for agent_id in range(AGENT_COUNT):
                task["desired_waypoint_w_m"][:, agent_id] = routes[
                    agent_id, task["waypoint_index"][:, agent_id]
                ]
            np.savez_compressed(task_path, **task)

            scene_path = root / "scene.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            scene["public_task_sha256"] = sha256_file(public_path)
            scene["route_clearance_m"] = 0.5
            scene["flight_volume_m"]["z"] = [0.0, 15.0]
            scene["command_volume_m"]["x"] = [-50.0, 50.0]
            scene["collision_proxies"]["source_aabb_geometry_sha256"] = "2" * 64
            scene["lidar_geometry_coverage"]["includes_collision_proxies"] = False
            _write_json(scene_path, scene)
            _bind_receipt(root, private)

            codes = _codes(validate_isaac_capture(root, evaluator_manifest=private))
            self.assertTrue(
                {
                    "flight_volume",
                    "command_volume",
                    "route_clearance_contract",
                    "route_command_volume",
                    "route_clearance",
                    "route_contract",
                    "collision_proxies",
                    "lidar_geometry_coverage",
                }.issubset(codes)
            )

    def test_trajectory_contact_and_all_evaluator_bindings_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)

            state_path = root / "streams/state_action.npz"
            _rewrite_npz(
                state_path,
                lambda arrays: arrays["root_pos_w_m"].__setitem__((1, 0, 0), 46.5),
            )
            contact_path = root / "sensors/contact.npz"
            _rewrite_npz(
                contact_path,
                lambda arrays: arrays["net_forces_w_n"].__setitem__((0, 0, 0, 0), 0.01),
            )
            outcome_path = root / "task_outcome.json"
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            outcome["state_action_sha256"] = sha256_file(state_path)
            outcome["private_manifest_commitment_sha256"] = "3" * 64
            _write_json(outcome_path, outcome)
            scene_path = root / "scene.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            scene["private_evaluator_manifest_sha256"] = "4" * 64
            _write_json(scene_path, scene)
            _bind_receipt(root, private)

            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(
                {"trajectory_flight_volume", "contact_abort_threshold", "evaluator_binding"}.issubset(
                    _codes(report)
                )
            )
            binding_paths = {
                issue.path for issue in report.issues if issue.code == "evaluator_binding"
            }
            self.assertEqual(
                binding_paths,
                {
                    "scene.json.private_evaluator_manifest_sha256",
                    "task_outcome.json",
                },
            )

    def test_payload_and_private_manifest_hash_tamper_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            with (root / "sensors/lidar.npz").open("ab") as stream:
                stream.write(b"tamper")
            self.assertIn("artifact_hash", _codes(validate_isaac_capture(root, evaluator_manifest=private)))
            private.write_text("{}\n", encoding="utf-8")
            self.assertIn("evaluator_manifest_hash", _codes(validate_isaac_capture(root, evaluator_manifest=private)))

    def test_private_manifest_inside_capture_is_rejected_and_closed_world(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            inside = root / "evaluator_private" / "manifest.json"
            inside.parent.mkdir(parents=True)
            inside.write_bytes(private.read_bytes())
            codes = _codes(validate_isaac_capture(root, evaluator_manifest=inside))
            self.assertTrue({"evaluator_manifest_location", "closed_world"}.issubset(codes))

    def test_zero_search_movement_and_static_high_level_action_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            path = root / "streams/state_action.npz"
            _rewrite_npz(
                path,
                lambda arrays: (
                    arrays.__setitem__("root_pos_w_m", np.zeros_like(arrays["root_pos_w_m"])),
                    arrays.__setitem__("desired_pos_w_m", np.zeros_like(arrays["desired_pos_w_m"])),
                ),
            )
            _rebind_state_outcome(root, private)
            codes = _codes(validate_isaac_capture(root, evaluator_manifest=private))
            self.assertTrue({"insufficient_search_movement", "static_high_level_action"}.issubset(codes))

    def test_static_waypoint_and_message_misalignment_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            task_path = root / "streams/public_task.npz"
            with np.load(task_path, allow_pickle=False) as archive:
                task = {name: archive[name].copy() for name in archive.files}
            task["waypoint_index"][:] = 1
            routes = np.asarray(json.loads((root / "public_task.json").read_text())["routes_w_m"], dtype=np.float32)
            for agent_id in range(AGENT_COUNT):
                task["desired_waypoint_w_m"][:, agent_id] = routes[agent_id, 1]
            np.savez_compressed(task_path, **task)
            message_path = root / "streams/public_messages.npz"
            _rewrite_npz(message_path, lambda arrays: arrays["message_waypoint_index"].__setitem__((0, 0), 2))
            _bind_receipt(root, private)
            codes = _codes(validate_isaac_capture(root, evaluator_manifest=private))
            self.assertTrue({"waypoint_static", "message_task_alignment"}.issubset(codes))

    def test_semantic_policy_leak_and_bad_metadata_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            camera_path = root / "sensors/onboard_rgbd.npz"
            with np.load(camera_path, allow_pickle=False) as archive:
                camera = {name: archive[name].copy() for name in archive.files}
            camera["semantic_segmentation"] = np.ones((*camera["rgb"].shape[:-1], 1), dtype=np.int32)
            np.savez_compressed(camera_path, **camera)
            _write_json(
                root / "learning_labels/semantic_metadata.json",
                {"schema": "wrong", "partition": "policy_visible", "policy_visible": True, "replicator_info": {}},
            )
            _bind_receipt(
                root,
                private,
                claim_boundary={
                    "formal_benchmark_admission": False,
                    "radar_profile_eligible": False,
                    "hardware_validated": False,
                    "foundation_model_executed": False,
                    "semantic_labels_policy_visible": True,
                },
            )
            codes = _codes(validate_isaac_capture(root, evaluator_manifest=private))
            self.assertTrue({"npz_fields", "semantic_metadata", "claim_boundary"}.issubset(codes))

    def test_target_observability_rejects_generic_or_instance_only_camera_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            metadata_path = root / "learning_labels/semantic_frame_metadata.jsonl"
            rows = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
            ]
            metadata = rows[0]
            per_camera = metadata["onboard_replicator_info"]["per_camera"]
            assert isinstance(per_camera, list)
            per_camera[0]["id_to_labels"]["1"] = {
                "class": "search_target",
                "instance": "search_target_slot_000",
            }
            metadata_path.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            _bind_receipt(root, private)

            report = validate_isaac_capture(root, evaluator_manifest=private)

            self.assertIn("target_observability", _codes(report))
            self.assertIn("target_observability_binding", _codes(report))
            summary = report.checks["target_observability"]
            self.assertEqual(summary["schema"], "org.rivermark.isaac-target-visibility-summary.v2")
            self.assertIn("search_target_slot_000", summary["failed_target_slots"])

    def test_target_observability_rejects_legacy_v1_declared_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            outcome_path = root / "task_outcome.json"
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            outcome["target_observability"] = {
                "schema": "org.rivermark.isaac-target-visibility-summary.v1",
                "target_count": TARGET_COUNT,
                "targets_meeting_visibility": TARGET_COUNT,
                "minimum_visible_sensor_frames_per_target": 1,
                "minimum_visible_instance_pixels": 8,
                "passed": True,
                "failed_target_count": 0,
            }
            _write_json(outcome_path, outcome)
            _bind_receipt(root, private)

            report = validate_isaac_capture(root, evaluator_manifest=private)

            self.assertIn("target_observability_binding", _codes(report))
            self.assertEqual(
                report.checks["target_observability"]["schema"],
                "org.rivermark.isaac-target-visibility-summary.v2",
            )

    def test_overview_city_content_and_structural_semantics_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/overview_rgb.npz",
                lambda arrays: arrays["distance_to_image_plane_m"].fill(200.0),
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("overview_city_content", _codes(report))
            self.assertFalse(report.checks["overview_city_content_verified"])

            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/overview_rgb.npz",
                lambda arrays: arrays["semantic_segmentation"].fill(0),
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("overview_structural_semantics", _codes(report))
            self.assertFalse(report.checks["overview_structural_semantics_verified"])

    def test_raw_onboard_visual_intrusion_is_recomputed_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/onboard_rgbd.npz",
                lambda arrays: arrays["distance_to_image_plane_m"].__setitem__(
                    (0, 1), 0.20
                ),
            )
            _rewrite_npz(
                root / "sensors/lidar.npz",
                lambda arrays: arrays["ranges_m"].__setitem__((0, 1, slice(0, 10)), 0.20),
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("visual_intrusion", _codes(report))
            self.assertFalse(report.checks["visual_intrusion_verified"])

    def test_raw_onboard_scene_content_is_recomputed_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/onboard_rgbd.npz",
                lambda arrays: arrays["distance_to_image_plane_m"].fill(100.0),
            )
            _rewrite_npz(
                root / "learning_labels/semantic_segmentation.npz",
                lambda arrays: arrays["semantic_segmentation"].fill(0),
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("onboard_scene_content", _codes(report))
            self.assertFalse(report.checks["onboard_scene_content_verified"])

    def test_onboard_scene_content_calibration_contract_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            calibration_path = root / "calibration.json"
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration["onboard_camera"]["clipping_range_m"][1] = 50.0
            _write_json(calibration_path, calibration)
            _bind_receipt(root, private)
            self.assertIn(
                "onboard_content_calibration",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

    def test_route_witness_camera_pose_is_rederived_from_the_public_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/overview_rgb.npz",
                lambda arrays: arrays["camera_pos_w_m"].__setitem__((1, 0), 100.0),
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("route_witness_camera_pose", _codes(report))

    def test_route_witness_schedule_and_frame_shot_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            calibration_path = root / "calibration.json"
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration["overview_camera"]["route_witness_schedule"]["shots"][0][
                "target_w_m"
            ][1] = -38.0
            _write_json(calibration_path, calibration)
            _bind_receipt(root, private)
            self.assertIn(
                "route_witness_camera_calibration",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

            _capture_fixture(root, private)
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration["overview_camera"]["tracked_agent_visibility_gate"][
                "capture_frames"
            ][0]["witness_shot_index"] = 1
            _write_json(calibration_path, calibration)
            _bind_receipt(root, private)
            self.assertIn(
                "route_witness_visibility_calibration",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

    def test_route_witness_visibility_calibration_rejects_noninteger_pixel_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            calibration_path = root / "calibration.json"
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration["overview_camera"]["tracked_agent_visibility_gate"][
                "capture_frames"
            ][0]["tracked_agent_pixel_count"] = "36"
            _write_json(calibration_path, calibration)
            _bind_receipt(root, private)

            report = validate_isaac_capture(root, evaluator_manifest=private)

            self.assertIn("route_witness_visibility_calibration", _codes(report))

    def test_batched_camera_semantic_metadata_requires_each_render_product_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            metadata_path = root / "learning_labels/semantic_frame_metadata.jsonl"
            rows = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["onboard_replicator_info"] = {
                "per_camera": [rows[0]["onboard_replicator_info"]]
            }
            rows[0]["overview_replicator_info"] = {
                "per_camera": [rows[0]["overview_replicator_info"]]
            }
            metadata_path.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertIn("target_observability", _codes(report))

    def test_batched_camera_semantic_metadata_keeps_same_numeric_id_camera_local(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            metadata_path = root / "learning_labels/semantic_frame_metadata.jsonl"
            rows = [
                json.loads(line)
                for line in metadata_path.read_text(encoding="utf-8").splitlines()
            ]
            metadata = rows[0]
            per_camera = metadata["onboard_replicator_info"]["per_camera"]
            assert isinstance(per_camera, list)
            per_camera[1]["id_to_labels"]["1"] = {"class": "prop_structure"}
            metadata_path.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, report.issues)

    def test_radar_claim_and_calibration_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _write_json(root / "calibration.json", {"radar": {"status": "captured", "fail_closed": False}})
            _bind_receipt(
                root,
                private,
                claim_boundary={
                    "formal_benchmark_admission": False,
                    "radar_profile_eligible": True,
                    "hardware_validated": False,
                    "foundation_model_executed": False,
                    "semantic_labels_policy_visible": False,
                },
            )
            codes = _codes(validate_isaac_capture(root, evaluator_manifest=private))
            self.assertTrue({"radar_claim", "radar_calibration"}.issubset(codes))

    def test_lidar_uses_capture_bound_max_distance_and_requires_a_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            calibration_path = root / "calibration.json"
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration["lidar"]["max_distance_m"] = 100.0
            _write_json(calibration_path, calibration)
            _rewrite_npz(
                root / "sensors/lidar.npz",
                lambda arrays: (
                    arrays["ranges_m"].fill(100.0),
                    arrays["ranges_m"].__setitem__((0, 0, 0), 42.0),
                ),
            )
            _bind_receipt(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            self.assertTrue(report.valid, report.issues)
            self.assertEqual(report.checks["lidar_max_distance_m"], 100.0)

            _rewrite_npz(
                root / "sensors/lidar.npz",
                lambda arrays: arrays["ranges_m"].fill(100.0),
            )
            _bind_receipt(root, private)
            self.assertIn("lidar_range", _codes(validate_isaac_capture(root, evaluator_manifest=private)))

    def test_capture_integrity_and_timestamp_mutations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            for relative in (
                "sensors/camera_poses.npz",
                "sensors/contact.npz",
                "sensors/imu.npz",
                "sensors/lidar.npz",
                "sensors/onboard_rgbd.npz",
                "sensors/overview_rgb.npz",
                "learning_labels/semantic_segmentation.npz",
                "streams/public_task.npz",
                "streams/public_messages.npz",
            ):
                _rewrite_npz(root / relative, lambda arrays: arrays.__setitem__("timestamps_ns", arrays["timestamps_ns"].astype(np.float32)))
            _bind_receipt(
                root,
                private,
                capture_integrity={
                    "online_capture": False,
                    "queue_used": True,
                    "queue_overflow": True,
                    "silent_frame_drop": True,
                    "synchronous_sensor_reads": False,
                },
            )
            codes = _codes(validate_isaac_capture(root, evaluator_manifest=private))
            self.assertTrue({"capture_integrity", "timestamps"}.issubset(codes))

    def test_sensor_phase_trace_mutations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            trace_path = root / SENSOR_PHASE_TRACE_RELATIVE_PATH
            for field, mutate in (
                (
                    "event_codes",
                    lambda arrays: arrays["event_codes"].__setitem__((0, 0), 99),
                ),
                (
                    "retained_contact_sha256",
                    lambda arrays: arrays["retained_contact_sha256"].__setitem__((0, 0), 1),
                ),
                (
                    "archive_frame_index",
                    lambda arrays: arrays["archive_frame_index"].__setitem__(0, 1),
                ),
            ):
                _capture_fixture(root, private)
                _rewrite_npz(trace_path, mutate)
                _bind_receipt(root, private)
                codes = _codes(validate_isaac_capture(root, evaluator_manifest=private))
                self.assertTrue(
                    {"sensor_phase_order", "sensor_phase_contact_binding", "sensor_phase_binding"}
                    & codes
                    if field != "archive_frame_index"
                    else {"sensor_phase_binding"} & codes,
                    field,
                )

    def test_camera_render_read_fence_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            _rewrite_npz(
                root / "sensors/camera_poses.npz",
                lambda arrays: arrays["camera_render_read_post_frame_index"].__setitem__((0, 0), 99),
            )
            _bind_receipt(root, private)
            self.assertIn(
                "camera_render_read_fence",
                _codes(validate_isaac_capture(root, evaluator_manifest=private)),
            )

    def test_validation_receipt_is_hash_bound_and_invalid_report_cannot_be_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root, private = base / "capture", base / "private.json"
            _capture_fixture(root, private)
            report = validate_isaac_capture(root, evaluator_manifest=private)
            destination = base / "validation.json"
            write_validation_receipt(report, destination)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload["capture_receipt_sha256"], sha256_file(root / "capture_receipt.json"))
            self.assertEqual(payload["checks"]["evaluator_manifest_sha256"], sha256_file(private))
            self.assertFalse(payload["formal_benchmark_admission"])
            invalid = IsaacValidationReport(
                root,
                report.receipt_sha256,
                {},
                (ValidationIssue("tampered", ".", "fixture"),),
            )
            with self.assertRaisesRegex(RuntimeError, "invalid capture"):
                write_validation_receipt(invalid, destination)


if __name__ == "__main__":
    unittest.main()
