"""Capture a fresh eight-CF2X Isaac Lab rollout and its raw video frames.

The module intentionally has no Isaac imports at module import time.  It owns
one AppLauncher, creates a new stage, and never imports legacy MD-QD-Swarm
routes, targets, traces, evaluators, or artifacts.  The capture receipt is
evidence for an Isaac pilot; formal dataset admission is performed separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .citylite_scene import (
    AABB,
    CITY_LITE_COMMAND_VOLUME_W_M,
    CITY_LITE_FLIGHT_VOLUME_W_M,
    CITY_TASK_OBSTACLE_MATERIAL_CONTRACT_SHA256,
    ENVIRONMENT_ID,
    SCENE_CONTRACT_SHA256,
    EXPECTED_NATIVE_COLLISION_COUNTS,
    PUBLIC_ROUTES_W_M,
    ROUTE_CLEARANCE_M,
    SELECTIVE_REFERENCES,
    START_ANCHOR_IDS_BY_ROUTE_FAMILY,
    TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M,
    CITY_LITE_ROUTE_FAMILY_A_ID,
    CITY_LITE_ROUTE_FAMILY_B_ID,
    CITY_LITE_TARGET_REGION_A_ID,
    CITY_LITE_TARGET_REGION_B_ID,
    CityLiteAuthority,
    CityLiteRouteError,
    aabb_geometry_sha256,
    canonical_payload_sha256,
    city_task_obstacle_material_contract_payload,
    forbidden_scene_paths,
    make_rivermark_layer_inventory,
    make_public_route_contract,
    resolve_public_route_family,
    resolve_city_lite_authority,
    validate_rivermark_layer_inventory_receipt,
    validate_public_route_contract,
    validate_public_routes,
    validate_static_scene_receipt,
)
from .eight_cf2x_fleet import EightCF2XFleet
from .citylite_task import (
    LIDAR_CHANNEL_COUNT,
    LIDAR_HORIZONTAL_FOV_RANGE_DEG,
    LIDAR_HORIZONTAL_RESOLUTION_DEG,
    LIDAR_RAY_COUNT,
    LIDAR_VERTICAL_FOV_RANGE_DEG,
    ONBOARD_FOCAL_LENGTH_MM,
    ONBOARD_HORIZONTAL_APERTURE_MM,
    ONBOARD_IMAGE_HEIGHT,
    ONBOARD_IMAGE_WIDTH,
    TARGET_VISIBILITY_GEOMETRY_SCHEMA,
    TARGET_VISIBILITY_BUCKETS,
    TARGET_VISIBILITY_MIN_NATIVE_FRAMES,
    TARGET_VISIBILITY_MIN_NATIVE_PIXELS,
    PUBLIC_ROUTE_WAYPOINT_SEGMENT_SECONDS,
    validate_route_timing_feasibility,
    target_region_for_positions,
    target_visibility_execution_window,
    target_visibility_geometry_contract,
    verify_target_visibility_bucket,
)
from .capture_lease import repository_app_launcher_lease
from .frame_archive import FrameSpool, write_chunked_frame_archive
from .cleanup_history import cleanup_completed_runs
from .failure_ledger import (
    CAPTURE_START_SCHEMA,
    FailureRecord,
    append_failure_record_once,
    recover_crash_left_attempts,
)
from .collection_protocol import (
    CollectionProtocolError,
    NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA,
    NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA,
    is_native_t2_canary_protocol,
    load_collection_protocol,
    native_t2_motion_contract,
    resolve_collection_binding,
    validate_collection_binding,
)
from .condition_realization import condition_request_from_protocol
from .private_evaluator_manifest import (
    NATIVE_T2_V2_TASK_VARIANT_ID,
    NATIVE_T2_V3_TASK_VARIANT_ID,
    PRIVATE_EVALUATOR_SCHEMA,
    PRIVATE_MANIFEST_RETENTION_KIND,
    PRIVATE_TARGET_MAX_RADIUS_M,
    PRIVATE_TARGET_MIN_PAIRWISE_SEPARATION_M,
    PRIVATE_TARGET_MIN_ROUTE_SEPARATION_M,
    PRIVATE_TARGET_OBSTACLE_CLEARANCE_M,
    PRIVATE_TARGET_ORIGIN,
    PRIVATE_TARGET_PLACEMENT_SCHEMA,
    TARGET_COUNT,
    retain_private_evaluator_manifest,
)
from .provenance import detect_source_provenance
from .resource_telemetry import (
    DEFAULT_ABORT_COMMIT_PERCENT,
    DEFAULT_PREFLIGHT_COMMIT_PERCENT,
    FOREIGN_NATIVE_PROCESS_CENSUS_SCHEMA,
    ResourceTelemetry,
    foreign_native_process_census,
)
from .isaac_runtime_safety import (
    RUNTIME_SAFETY_FRAME_OUTCOME_CODES,
    RUNTIME_SAFETY_PHASE_CODES,
    RUNTIME_SAFETY_TRACE_RELATIVE_PATH,
    SENSOR_PHASE_EVENT_CODES,
    SENSOR_PHASE_EVENT_SEQUENCE,
    SENSOR_PHASE_SENSOR_NAMES,
    SENSOR_PHASE_TRACE_RELATIVE_PATH,
    SENSOR_PHASE_TRACE_SCHEMA,
    RuntimeSafetyAbort,
    bind_runtime_safety_trace_evidence,
    evaluate_runtime_safety,
    finalize_runtime_safety_guard,
    record_runtime_safety_abort,
    record_runtime_safety_check,
    runtime_safety_receipt_template,
    sensor_phase_array_digest,
    physics_time_ns,
)
from .isaac_transfer import FixedDecisionCadence, WorldCommandBounds
from .native_t2_canary import (
    NATIVE_T2_EVENTS_SCHEMA,
    NATIVE_T2_TRACE_SCHEMA,
    PublicRouteCoveragePolicy,
    SpatialCandidateDeduplicator,
    bind_native_t2_calibration,
    native_rgbd_world_points,
    native_semantic_rgbd_candidates,
)
from .t2_policy_abi import (
    T2CandidateEventJournal,
    T2NativeStepEvidence,
    T2PolicyRunner,
    T2PublicFleetObservation,
    T2PublicSensorObservation,
)


AGENT_COUNT = 8
_SYSTEM_COMMIT_SNAPSHOT_UNSET = object()
_PREFLIGHT_SYSTEM_COMMIT_PHASES = frozenset(("preflight", "before_app_launcher"))
CAPTURE_STORAGE_BUDGET_SCHEMA = "org.rivermark.isaac-capture-storage-budget.v1"
_SEMANTIC_METADATA_RESERVATION_PER_FRAME_BYTES = 256 * 1024
_RUNTIME_TRACE_RESERVATION_PER_PHYSICS_STEP_BYTES = 8 * 1024
_ARCHIVE_CONTAINER_OVERHEAD_PER_MEMBER_BYTES = 512
_STORAGE_RESERVATION_HEADROOM_NUMERATOR = 6
_STORAGE_RESERVATION_HEADROOM_DENOMINATOR = 5
SWARM_ROOT_PRIM = "/World/Swarm"
SWARM_AGENT_PRIM_EXPRESSION = f"{SWARM_ROOT_PRIM}/Agent_.*/Robot"
SWARM_AGENT_BODY_PRIM_EXPRESSION = f"{SWARM_AGENT_PRIM_EXPRESSION}/body"
SWARM_AGENT_LITERAL_PRIM_PATHS = tuple(
    f"{SWARM_ROOT_PRIM}/Agent_{agent_id}/Robot" for agent_id in range(AGENT_COUNT)
)
THRUSTER_NAMES = ("m1_prop", "m2_prop", "m3_prop", "m4_prop")
HOVER_THRUST_PER_ROTOR_N = 0.06935
THRUST_COEFFICIENT_N_PER_RPS_SQUARED = 1.0e-6
MAX_THRUST_PER_ROTOR_N = 0.18
INITIAL_HOVER_RPS = math.sqrt(
    HOVER_THRUST_PER_ROTOR_N / THRUST_COEFFICIENT_N_PER_RPS_SQUARED
)
MAX_CF2X_LINEAR_VELOCITY_MPS = 12.0
MAX_CF2X_ANGULAR_VELOCITY_RADPS = 60.0
LITERAL_SPAWN_POSITION_TOLERANCE_M = 0.02
LITERAL_SPAWN_ORIENTATION_TOLERANCE_RAD = 0.01
# These prove the configuration that IsaacLab resolved into its own asset
# buffers.  They are deliberately much tighter than the live post-reset pose
# tolerance below: the latter admits Isaac's documented reset-time PhysX
# settling, while these values audit the authored initial conditions exactly.
LITERAL_SPAWN_DEFAULT_STATE_TOLERANCE = 1.0e-5
LITERAL_SPAWN_DEFAULT_RPS_TOLERANCE = 1.0e-3
LITERAL_SPAWN_DEFAULT_THRUST_TOLERANCE_N = 1.0e-5
LITERAL_USD_SPAWN_POSITION_TOLERANCE_M = 1.0e-5
LITERAL_USD_SPAWN_ORIENTATION_TOLERANCE_RAD = 1.0e-5
LITERAL_USD_SPAWN_BASIS_LENGTH_TOLERANCE = 1.0e-6
# Search targets are authored into the active USD stage from evaluator-private
# truth.  These tight authoring checks run before and after reset so a missing,
# hidden, or displaced target cannot consume a full rollout and then be
# diagnosed only from a semantic failure.  The public receipt records aggregates
# only; target paths, IDs, coordinates, and radii remain evaluator-private.
RUNTIME_TARGET_USD_POSITION_TOLERANCE_M = 1.0e-5
RUNTIME_TARGET_USD_RADIUS_TOLERANCE_M = 1.0e-5
RUNTIME_TARGET_USD_BOUND_EXTENT_TOLERANCE_M = 1.0e-5
CAMERA_OFFSET_BODY_M = (0.12, 0.0, 0.04)
# IsaacLab's ``world`` camera convention uses +X as the optical axis.  A
# modest fixed downward pitch keeps the route geometry in view at the native
# City-Lite flight altitude without turning the policy camera into a tracker.
ONBOARD_CAMERA_PITCH_DOWN_RAD = math.radians(15.0)


def _camera_mount_quat_wxyz(pitch_down_rad: float = ONBOARD_CAMERA_PITCH_DOWN_RAD) -> tuple[float, float, float, float]:
    """Return the body-relative WXYZ mount quaternion for a downward pitch.

    In the right-handed body frame, a positive rotation about +Y maps the
    Camera ``world`` optical axis +X to ``(+cos(pitch), 0, -sin(pitch))``.
    Keeping this formula in one place makes the USD/Fabric calibration
    auditable instead of relying on a hand-entered quaternion sign.
    """

    pitch = float(pitch_down_rad)
    if not math.isfinite(pitch) or abs(pitch) >= math.pi / 2.0:
        raise ValueError("onboard camera pitch must be finite and strictly within +/-90 degrees")
    half = 0.5 * pitch
    return (math.cos(half), 0.0, math.sin(half), 0.0)


CAMERA_OFFSET_WXYZ = _camera_mount_quat_wxyz()
# Dynamic CF2X bodies advance through Fabric while render-side USD transforms
# must be authored explicitly. These bounds apply to the renderer-facing USD
# Xforms before every accepted sensor frame.
ONBOARD_CAMERA_USD_POSITION_TOLERANCE_M = 0.05
# A camera attached to a moving Fabric body must agree in all three axes, not
# only its optical axis.  This avoids accepting an otherwise forward-facing
# camera with a stale roll or pitch.
ONBOARD_CAMERA_USD_ORIENTATION_TOLERANCE_RAD = 0.01
ONBOARD_CAMERA_USD_FORWARD_COSINE_MIN = math.cos(ONBOARD_CAMERA_USD_ORIENTATION_TOLERANCE_RAD)
# The Camera Fabric cache is persisted with each RGB-D frame. Match the
# independent validator here so capture cannot emit a self-invalidating bundle.
ONBOARD_CAMERA_FABRIC_POSITION_TOLERANCE_M = 1.0e-4
ONBOARD_CAMERA_FABRIC_ORIENTATION_TOLERANCE_RAD = 2.0e-3
ONBOARD_CAMERA_DEMO_AGENT_IDS = (0, 1, 4, 7)
# The legacy southwest overview remains available as a diagnostic reference.
# The former body-follow presentation camera is retained below for historical
# receipt compatibility, but it is not suitable as public video evidence: a
# camera moving with the CF2X removes the very background parallax a viewer
# needs to see that the aircraft is flying.
OVERVIEW_CAMERA_EYE_W_M = (60.0, -78.0, 42.0)
OVERVIEW_CAMERA_TARGET_W_M = (0.0, -1.0, 8.0)
OVERVIEW_CAMERA_POSITION_TOLERANCE_M = 0.05
OVERVIEW_CAMERA_FORWARD_COSINE_MIN = 0.999
OVERVIEW_CAMERA_CLIPPING_RANGE_M = (0.05, 200.0)
ONBOARD_CAMERA_CLIPPING_RANGE_M = (0.05, 100.0)
ONBOARD_CONTENT_GATE_SCHEMA = "org.rivermark.isaac-onboard-scene-content-gate.v1"
ONBOARD_CONTENT_MIN_FINITE_DEPTH_FRACTION = 0.99
ONBOARD_CONTENT_MIN_GEOMETRY_FRACTION = 0.20
ONBOARD_CONTENT_MAX_BACKGROUND_FRACTION = 0.80
OVERVIEW_FOLLOW_SCHEMA = "org.rivermark.isaac-public-agent-follow-camera.v1"
OVERVIEW_FOLLOW_TRACKED_AGENT_ID = 0
# A trailing, elevated body-frame view leaves enough City-Lite context in the
# frame while keeping the tracked physical CF2X visually readable. These are
# evidence-camera geometry only and never enter policy observations.
# Keep the public evidence view near enough to show the tracked CF2X and
# nearby RiverMark geometry.  The earlier 16 m chase geometry technically
# moved with the vehicle but rendered as an almost static distant panorama.
OVERVIEW_FOLLOW_EYE_OFFSET_BODY_M = (-6.0, -4.0, 3.5)
OVERVIEW_FOLLOW_TARGET_OFFSET_BODY_M = (0.0, 0.0, 0.30)
OVERVIEW_FOLLOW_MIN_CAMERA_DISPLACEMENT_M = 0.50
OVERVIEW_FOLLOW_POSITION_TOLERANCE_M = 0.05
OVERVIEW_FOLLOW_FORWARD_COSINE_MIN = 0.999
# The published Isaac demonstration uses one frozen, world-coordinate tripod
# camera for its entire duration. A time-selected series of otherwise fixed
# cameras is still a camera cut: it can conceal relative motion and makes a
# route witness visually ambiguous. No view is body-following or selected from
# physical state, semantic output, target data, or policy state.
OVERVIEW_WITNESS_SCHEMA = "org.rivermark.isaac-public-route-witness-camera.v3"
OVERVIEW_WITNESS_TRACKED_AGENT_ID = 2
OVERVIEW_WITNESS_SELECTION = "single_frozen_public_world_pose"
OVERVIEW_WITNESS_SHOTS = (
    # A distant south-side view over Agent 2's complete train and validation
    # route families. The closer repaired pose only covered the train family;
    # validation's mirrored north route fell outside the 16:9 frustum. This
    # single fixed pose covers both families while remaining outside the
    # City-Lite flight volume and task geometry.
    (0, None, (0.0, -95.0, 30.0), (0.0, 0.0, 13.0)),
)
# Retained as aliases for static inspection helpers.
OVERVIEW_WITNESS_EYE_W_M = OVERVIEW_WITNESS_SHOTS[0][2]
OVERVIEW_WITNESS_TARGET_W_M = OVERVIEW_WITNESS_SHOTS[0][3]
OVERVIEW_WITNESS_MIN_TRACKED_AGENT_DISPLACEMENT_M = 3.0
OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS = 32
OVERVIEW_WITNESS_POSITION_TOLERANCE_M = 0.05
OVERVIEW_WITNESS_FORWARD_COSINE_MIN = 0.999
OVERVIEW_WITNESS_FOCAL_LENGTH_MM = 45.0
OVERVIEW_WITNESS_IMAGE_WIDTH = 1920
OVERVIEW_WITNESS_IMAGE_HEIGHT = 1080
# The marker is a collision-free semantic witness attached to the physical
# body, not a controller input.  The single fixed camera must cover both the
# near train route and the mirrored validation route; 0.20 m keeps the far
# validation reset marker above the 32-pixel evidence gate at the locked
# 45 mm/1920x1080 overview contract.
IDENTITY_MARKER_RADIUS_M = 0.20
OVERVIEW_CONTENT_GATE_SCHEMA = "org.rivermark.isaac-overview-city-content-gate.v1"
# The fixed-world overview is an audit witness, not an onboard training
# modality.  It is rendered and checked at every retained sensor frame, but
# only this deterministic subset is retained as evidence.  Never derive this
# schedule from semantic content, target visibility, or a pass/fail outcome.
OVERVIEW_ARCHIVE_SCHEMA = "org.rivermark.isaac-overview-evidence-archive.v1"
OVERVIEW_ARCHIVE_STRIDE = 10
# The overview is an evidence camera rather than a policy input. These limits
# reject an environment-only frame or a camera embedded in geometry, while
# keeping semantic labels optional because referenced USD assets can lose them
# on the Replicator path.
OVERVIEW_CONTENT_MIN_FINITE_DEPTH_FRACTION = 0.99
OVERVIEW_CONTENT_MIN_GEOMETRY_FRACTION = 0.03
OVERVIEW_CONTENT_NEAR_SURFACE_M = 2.0
OVERVIEW_CONTENT_MAX_NEAR_SURFACE_FRACTION = 0.20
OVERVIEW_CONTENT_MIN_DEPTH_SPAN_M = 1.5
OVERVIEW_CONTENT_RGB_EDGE_DELTA = 8.0
OVERVIEW_CONTENT_MIN_RGB_EDGE_FRACTION = 0.003
OVERVIEW_CONTENT_MIN_STRUCTURAL_PIXEL_FRACTION = 0.001
OVERVIEW_STRUCTURAL_LABEL_TOKENS = (
    "building",
    "structure",
    "facade",
    "wall",
    "tower",
    "rubble",
    "debris",
)
# A body may be contact-free according to its root sensor while an incorrectly
# mounted camera or a missing static collider is already inside a rendered
# facade.  These raw-sensor limits make that contradiction fail closed. The
# LiDAR target set excludes the CF2X fleet, so a near return is independent
# evidence of scene geometry rather than a self-return.
VISUAL_INTRUSION_GATE_SCHEMA = "org.rivermark.isaac-rgbd-lidar-visual-intrusion-gate.v1"
VISUAL_INTRUSION_NEAR_DISTANCE_M = 0.35
VISUAL_INTRUSION_RGBD_MAX_NEAR_PIXEL_FRACTION = 0.20
VISUAL_INTRUSION_LIDAR_MAX_NEAR_RETURN_FRACTION = 0.02
CAPTURE_SCHEMA = "org.rivermark.isaac-swarm-capture.v1"
PRIVATE_TARGET_VISIBILITY_SCHEMA = TARGET_VISIBILITY_GEOMETRY_SCHEMA
T1_OBSERVABILITY_OUTCOME_SCHEMA = "org.rivermark.t1-target-observability.v1"
T1_DATA_TRACK_ID = "t1-expert-coverage-multisensor-v1"
TASK_VARIANT_ID = "isaac-eight-agent-public-waypoint-search-v1"
CONTROL_MODE_FIXED_PUBLIC_ROUTE = "fixed_public_route"
CONTROL_MODE_SB3_STATE_ONLY_TRANSFER = "sb3_state_only_transfer"
CONTROL_MODE_NATIVE_T2_CANARY = "native_t2_canary"
CONTROL_TRANSFER_TASK_KIND = "state_only_control_transfer_smoke"
CONTROL_TRANSFER_TASK_VARIANT_ID = (
    "isaac-eight-agent-sb3-state-only-control-transfer-smoke-v1"
)
SB3_TRANSFER_TRACE_RELATIVE_PATH = "streams/sb3_state_only_transfer.npz"
SB3_TRANSFER_PROVENANCE_RELATIVE_PATH = (
    "streams/sb3_state_only_transfer_provenance.json"
)
DEFAULT_SB3_DECISION_STRIDE = 40
DEFAULT_SB3_TRANSFER_HORIZONTAL_SPEED_MPS = 0.75
# The pilot's vertical command was trained in a small kinematic world.  Keep a
# conservative physical envelope around the literal City-Lite spawn altitude
# during a development transfer smoke; runtime safety remains authoritative.
DEFAULT_SB3_TRANSFER_VERTICAL_SPEED_MPS = 0.05
DEFAULT_SB3_TRANSFER_YAW_RATE_RADPS = 0.80
NATIVE_T2_TASK_KIND = "native_t2_search_canary"
NATIVE_T2_DECISION_TRACE_RELATIVE_PATH = "streams/native_t2_decisions.jsonl"
NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH = "streams/native_t2_events.json"
NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH = "sensors/native_t2_camera_extrinsics.npz"
DEFAULT_NATIVE_T2_DECISION_STRIDE = 40
# These defaults are the v2 public motion contract.  v1 launch requests are
# rejected before AppLauncher because they did not bind a feasible schedule;
# they are not silently reinterpreted with these newer values.
DEFAULT_NATIVE_T2_HORIZONTAL_SPEED_MPS = 2.0
DEFAULT_NATIVE_T2_VERTICAL_SPEED_MPS = 0.40
DEFAULT_NATIVE_T2_YAW_RATE_RADPS = 0.80
NATIVE_T2_CANDIDATE_MINIMUM_PIXELS = TARGET_VISIBILITY_MIN_NATIVE_PIXELS
NATIVE_T2_CANDIDATE_MERGE_RADIUS_M = 0.75
WAYPOINT_SEGMENT_SECONDS = PUBLIC_ROUTE_WAYPOINT_SEGMENT_SECONDS
WAYPOINT_REACHED_RADIUS_M = 0.75
# These bounds match the SB3 state-only action ABI. They constrain a
# development transfer policy before its command reaches the CF2X allocator.
STATE_ONLY_POLICY_HORIZONTAL_SPEED_MPS = 2.3
STATE_ONLY_POLICY_VERTICAL_SPEED_MPS = 1.25
STATE_ONLY_POLICY_YAW_RATE_RADPS = 1.4
STATE_ONLY_ALTITUDE_HOLD_GAIN = 0.85
TARGET_DETECTION_RADIUS_M = 0.70
COLLISION_PROXY_ROOT = "/World/StaticScene/CollisionProxies"
COLLISION_PROXY_REPRESENTATION = "conservative_world_aabb"
PRIVATE_TARGET_MIN_VISIBLE_SENSOR_FRAMES = TARGET_VISIBILITY_MIN_NATIVE_FRAMES
PRIVATE_TARGET_MIN_VISIBLE_INSTANCE_PIXELS = TARGET_VISIBILITY_MIN_NATIVE_PIXELS
TARGET_SEMANTIC_INSTANCE_PREFIX = "search_target_slot_"

IDENTITY_COLORS = (
    (0.95, 0.15, 0.15),
    (0.10, 0.85, 0.25),
    (0.15, 0.45, 1.00),
    (1.00, 0.75, 0.10),
    (0.95, 0.20, 0.90),
    (0.10, 0.90, 0.90),
    (1.00, 1.00, 1.00),
    (1.00, 0.45, 0.10),
)


class RadarUnavailableError(RuntimeError):
    """Raised when radar is required without a validated RTX or hardware source."""


class SensorPhysicsSmokeReceiptError(ValueError):
    """Raised when an external native sensor/physics smoke cannot bind capture."""


class PrivateEvaluatorManifestError(ValueError):
    """Raised when an external evaluator-owned target manifest is invalid."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_cf2x() -> Path:
    candidates = (
        Path(r"~\IsaacLab\isaac_drone_racer\assets\new\cf2x.usd"),
        _repository_root().parent / "IsaacLab" / "isaac_drone_racer" / "assets" / "new" / "cf2x.usd",
    )
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0])


def _default_city_lite_contract() -> Path:
    local_md_qd_swarm = Path(
        r"~\IsaacLab\isaac_drone_racer\experiments\md_qd_swarm"
    )
    candidates = (
        local_md_qd_swarm
        / "isaaclab_high_fidelity_scenes"
        / "hi_fi_search_rescue_rivermark_city_lite_v1_r2"
        / "rivermark_city_lite_scene_contract_v1.json",
        _repository_root().parent
        / "IsaacLab"
        / "isaac_drone_racer"
        / "experiments"
        / "md_qd_swarm"
        / "isaaclab_high_fidelity_scenes"
        / "hi_fi_search_rescue_rivermark_city_lite_v1_r2"
        / "rivermark_city_lite_scene_contract_v1.json",
    )
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0])


def _module_path_is_under(module: Any, package_root: Path) -> bool:
    """Return whether an already imported module belongs to ``package_root``."""

    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).expanduser().resolve().relative_to(package_root)
    except (OSError, ValueError):
        return False
    return True


def _activate_local_isaaclab_source(source_root: Path | None = None) -> Path | None:
    """Activate one IsaacLab source tree and reject preloaded source drift.

    A locked capture must audit and import the same tree.  In particular, an
    editable site-package or a stale ``sys.modules`` entry must not silently
    win after the command-line path was audited.
    """

    if source_root is None:
        configured = os.environ.get("RIVERMARK_ISAACLAB_SOURCE")
        candidates = [Path(configured)] if configured else []
        candidates.extend(
            (
                _repository_root().parent / "IsaacLab" / "source" / "isaaclab",
                Path(r"~\IsaacLab\source\isaaclab"),
            )
        )
    else:
        candidates = [Path(source_root)]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        package_root = resolved / "isaaclab"
        if not (package_root / "__init__.py").is_file():
            continue
        loaded = sys.modules.get("isaaclab")
        if loaded is not None and not _module_path_is_under(loaded, package_root):
            raise RuntimeError(
                "isaaclab is already imported from a different source tree: "
                f"{getattr(loaded, '__file__', None)!r}; expected {package_root}"
            )
        source = str(resolved)
        if source not in sys.path:
            sys.path.insert(0, source)
        return resolved
    if source_root is not None:
        raise FileNotFoundError(
            f"locked IsaacLab source does not contain isaaclab/__init__.py: "
            f"{Path(source_root).expanduser().resolve()}"
        )
    return None


def _activate_local_isaaclab_contrib_source(source_root: Path) -> Path:
    """Activate the lock-bound contrib extension before importing its modules."""

    resolved = Path(source_root).expanduser().resolve()
    package_root = resolved / "isaaclab_contrib"
    if not (package_root / "__init__.py").is_file():
        raise FileNotFoundError(
            "locked IsaacLab contrib source does not contain "
            f"isaaclab_contrib/__init__.py: {resolved}"
        )
    loaded = sys.modules.get("isaaclab_contrib")
    if loaded is not None and not _module_path_is_under(loaded, package_root):
        raise RuntimeError(
            "isaaclab_contrib is already imported from a different source tree: "
            f"{getattr(loaded, '__file__', None)!r}; expected {package_root}"
        )
    source = str(resolved)
    if source not in sys.path:
        sys.path.insert(0, source)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drone-usd", type=Path, default=_default_cf2x())
    parser.add_argument(
        "--scene-contract",
        type=Path,
        default=_default_city_lite_contract(),
        help="Exact md_qd_swarm City-Lite v1_r2 static-scene contract.",
    )
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--warmup-steps", type=int, default=120)
    parser.add_argument("--capture-stride", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--onboard-width", type=int, default=160)
    parser.add_argument("--onboard-height", type=int, default=120)
    parser.add_argument("--overview-width", type=int, default=OVERVIEW_WITNESS_IMAGE_WIDTH)
    parser.add_argument("--overview-height", type=int, default=OVERVIEW_WITNESS_IMAGE_HEIGHT)
    parser.add_argument("--base-thrust", type=float, default=HOVER_THRUST_PER_ROTOR_N)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260722,
        help="Pilot seed; a collection protocol binding replaces it with the deterministic episode seed.",
    )
    parser.add_argument(
        "--collection-protocol",
        type=Path,
        help="Public collection protocol JSON used to bind this attempt to one predeclared cell.",
    )
    parser.add_argument(
        "--collection-cell-id",
        help="Predeclared public collection cell identifier; requires --collection-protocol.",
    )
    parser.add_argument(
        "--collection-episode-index",
        type=int,
        help="Zero-based episode index within the predeclared collection cell.",
    )
    parser.add_argument(
        "--evaluator-private-manifest",
        type=Path,
        help=(
            "Pre-existing external evaluator-owned target manifest. It is required only "
            "for fixed_public_route Search3D capture, must be outside --output-dir, and "
            "is never copied into the capture bundle."
        ),
    )
    parser.add_argument(
        "--evaluator-private-manifest-retention-root",
        type=Path,
        help=(
            "Existing operator-controlled directory outside the repository and capture. "
            "fixed_public_route stores and then uses an exact content-addressed private "
            "manifest snapshot there; neither the path nor payload enters the capture bundle."
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=(
            CONTROL_MODE_FIXED_PUBLIC_ROUTE,
            CONTROL_MODE_SB3_STATE_ONLY_TRANSFER,
            CONTROL_MODE_NATIVE_T2_CANARY,
        ),
        default=CONTROL_MODE_FIXED_PUBLIC_ROUTE,
        help=(
            "fixed_public_route records the existing public-route Search3D capture; "
            "sb3_state_only_transfer is a development-only physical control smoke with "
            "no evaluator targets or benchmark outcome; native_t2_canary is a calibrated "
            "development-only closed-loop search evidence run."
        ),
    )
    parser.add_argument(
        "--sb3-checkpoint",
        type=Path,
        help="Immutable SB3 checkpoint used only by sb3_state_only_transfer.",
    )
    parser.add_argument(
        "--sb3-metadata",
        type=Path,
        help="Optional explicit v2 .rivermark.json sidecar for --sb3-checkpoint.",
    )
    parser.add_argument(
        "--sb3-decision-stride",
        type=int,
        default=DEFAULT_SB3_DECISION_STRIDE,
        help="Positive fixed physics-step cadence for development SB3 decisions.",
    )
    parser.add_argument(
        "--sb3-max-horizontal-speed-mps",
        type=float,
        default=DEFAULT_SB3_TRANSFER_HORIZONTAL_SPEED_MPS,
    )
    parser.add_argument(
        "--sb3-max-vertical-speed-mps",
        type=float,
        default=DEFAULT_SB3_TRANSFER_VERTICAL_SPEED_MPS,
    )
    parser.add_argument(
        "--sb3-max-yaw-rate-radps",
        type=float,
        default=DEFAULT_SB3_TRANSFER_YAW_RATE_RADPS,
    )
    parser.add_argument(
        "--cf2x-runtime-calibration",
        type=Path,
        help=(
            "Passed external CF2X runtime calibration JSON. Required only by "
            "native_t2_canary; the report path and payload are never copied into capture output."
        ),
    )
    parser.add_argument(
        "--native-t2-decision-stride",
        type=int,
        default=DEFAULT_NATIVE_T2_DECISION_STRIDE,
        help="Positive fixed physics-step cadence for calibrated native T2 decisions.",
    )
    parser.add_argument(
        "--native-t2-max-horizontal-speed-mps",
        type=float,
        default=DEFAULT_NATIVE_T2_HORIZONTAL_SPEED_MPS,
    )
    parser.add_argument(
        "--native-t2-max-vertical-speed-mps",
        type=float,
        default=DEFAULT_NATIVE_T2_VERTICAL_SPEED_MPS,
    )
    parser.add_argument(
        "--native-t2-max-yaw-rate-radps",
        type=float,
        default=DEFAULT_NATIVE_T2_YAW_RATE_RADPS,
    )
    parser.add_argument("--require-radar", action="store_true")
    parser.add_argument(
        "--preflight-commit-percent",
        type=float,
        default=DEFAULT_PREFLIGHT_COMMIT_PERCENT,
        help="Windows system-commit ceiling before AppLauncher may start.",
    )
    parser.add_argument(
        "--abort-commit-percent",
        type=float,
        default=DEFAULT_ABORT_COMMIT_PERCENT,
        help="Windows system-commit ceiling that aborts an active capture.",
    )
    parser.add_argument(
        "--maximum-foreign-native-private-commit-gib",
        type=float,
        default=8.0,
        help=(
            "Fail before or during AppLauncher ownership when another Python/Kit/Isaac "
            "process retains at least this much private commit. The guard cannot be disabled."
        ),
    )
    parser.add_argument(
        "--no-auto-cleanup",
        action="store_true",
        help=(
            "Do not move eligible old sibling runs to the Windows Recycle Bin. "
            "Automatic cleanup is enabled only under a rivermark-runs root."
        ),
    )
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=20.0,
        help="Minimum free space that must remain on the capture volume.",
    )
    parser.add_argument(
        "--estimated-capture-gib",
        type=float,
        default=8.0,
        help="Reserved upper-bound storage budget for this raw capture.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Require an NVIDIA GPU even when --device is not a CUDA device.",
    )
    parser.add_argument(
        "--minimum-gpu-vram-gib",
        type=float,
        default=8.0,
        help="Minimum VRAM required when a GPU-backed capture is requested.",
    )
    parser.add_argument(
        "--minimum-driver-version",
        default=None,
        help="Optional minimum NVIDIA driver version for the preflight probe.",
    )
    parser.add_argument(
        "--isaac-sim-version",
        default=None,
        help="Optional minimum installed isaacsim distribution version.",
    )
    parser.add_argument(
        "--isaaclab-version",
        default=None,
        help="Optional minimum installed isaaclab distribution version.",
    )
    parser.add_argument(
        "--runtime-lock",
        type=Path,
        help=(
            "Exact public Isaac runtime lock. Required with --isaaclab-source for "
            "any collection-protocol-bound capture."
        ),
    )
    parser.add_argument(
        "--isaaclab-source",
        type=Path,
        help=(
            "IsaacLab source tree bound by --runtime-lock. Required with the lock "
            "for any collection-protocol-bound capture."
        ),
    )
    parser.add_argument(
        "--sensor-physics-smoke-receipt",
        type=Path,
        help=(
            "External passed full-profile isaac_smoke_receipt.json produced from the "
            "same clean source, runtime lock, City-Lite contract, and CF2X asset. "
            "Required for collection-protocol-bound capture and never copied into it."
        ),
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Development-only override for the clean-source preflight gate.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.warmup_steps < 0 or args.capture_stride < 1:
        raise ValueError("steps/capture-stride must be positive and warmup-steps must be non-negative")
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("--dt must be finite and positive")
    for name in (
        "minimum_free_gib",
        "estimated_capture_gib",
        "minimum_gpu_vram_gib",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    maximum_foreign_commit_gib = float(args.maximum_foreign_native_private_commit_gib)
    if not math.isfinite(maximum_foreign_commit_gib) or maximum_foreign_commit_gib <= 0.0:
        raise ValueError(
            "--maximum-foreign-native-private-commit-gib must be finite and positive"
        )
    if args.minimum_driver_version is not None and not any(
        character.isdigit() for character in str(args.minimum_driver_version)
    ):
        raise ValueError("--minimum-driver-version must contain a numeric version")
    for name in ("isaac_sim_version", "isaaclab_version"):
        value = getattr(args, name)
        if value is not None and not any(character.isdigit() for character in str(value)):
            raise ValueError(f"--{name.replace('_', '-')} must contain a numeric version")
    for name in ("preflight_commit_percent", "abort_commit_percent"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 1.0 <= value < 100.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and in [1, 100)")
    if args.abort_commit_percent <= args.preflight_commit_percent:
        raise ValueError("--abort-commit-percent must exceed --preflight-commit-percent")
    if isinstance(args.seed, bool) or not 0 <= args.seed <= 0xFFFFFFFF:
        raise ValueError("--seed must be an unsigned 32-bit integer")
    collection_values = (args.collection_protocol, args.collection_cell_id, args.collection_episode_index)
    if any(value is not None for value in collection_values) and not all(
        value is not None for value in collection_values
    ):
        raise ValueError(
            "--collection-protocol, --collection-cell-id, and --collection-episode-index must be provided together"
        )
    if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY and args.collection_protocol is None:
        raise ValueError(
            "native_t2_canary requires --collection-protocol, --collection-cell-id, and "
            "--collection-episode-index"
        )
    if args.collection_episode_index is not None and (
        isinstance(args.collection_episode_index, bool) or args.collection_episode_index < 0
    ):
        raise ValueError("--collection-episode-index must be a non-negative integer")
    if args.collection_protocol is not None and args.control_mode not in (
        CONTROL_MODE_FIXED_PUBLIC_ROUTE,
        CONTROL_MODE_NATIVE_T2_CANARY,
    ):
        raise ValueError(
            "collection protocol binding is only valid for fixed_public_route or native_t2_canary capture"
        )
    if (args.runtime_lock is None) != (args.isaaclab_source is None):
        raise ValueError("--runtime-lock and --isaaclab-source must be provided together")
    if args.collection_protocol is not None and args.runtime_lock is None:
        raise ValueError("collection-protocol-bound capture requires --runtime-lock and --isaaclab-source")
    if args.collection_protocol is not None and args.sensor_physics_smoke_receipt is None:
        raise ValueError(
            "collection-protocol-bound capture requires --sensor-physics-smoke-receipt"
        )
    if args.sensor_physics_smoke_receipt is not None and args.runtime_lock is None:
        raise ValueError(
            "--sensor-physics-smoke-receipt requires --runtime-lock and --isaaclab-source"
        )
    if not math.isfinite(args.base_thrust) or not 0.0 < args.base_thrust <= MAX_THRUST_PER_ROTOR_N:
        raise ValueError(f"--base-thrust must be in (0, {MAX_THRUST_PER_ROTOR_N}] N per rotor")
    if not math.isclose(
        float(args.base_thrust), HOVER_THRUST_PER_ROTOR_N, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "City-Lite capture fixes --base-thrust to the upstream CF2X hover trim "
            f"({HOVER_THRUST_PER_ROTOR_N} N per rotor)"
        )
    for name in ("onboard_width", "onboard_height", "overview_width", "overview_height"):
        if getattr(args, name) < 16:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 16")
    if args.control_mode in (CONTROL_MODE_FIXED_PUBLIC_ROUTE, CONTROL_MODE_NATIVE_T2_CANARY) and (
        args.overview_width != OVERVIEW_WITNESS_IMAGE_WIDTH
        or args.overview_height != OVERVIEW_WITNESS_IMAGE_HEIGHT
    ):
        raise ValueError(
            "fixed_public_route and native_t2_canary require the locked 1920x1080 overview witness "
            "resolution; lower-resolution renders cannot satisfy its 32-pixel "
            "identity-marker evidence contract"
        )
    if args.require_radar:
        raise RadarUnavailableError(
            "No validated RTX radar or hardware radar smoke receipt is configured; refusing radar capture."
        )
    if args.control_mode in (CONTROL_MODE_FIXED_PUBLIC_ROUTE, CONTROL_MODE_NATIVE_T2_CANARY):
        if args.evaluator_private_manifest is None:
            raise ValueError(
                "--evaluator-private-manifest is required for fixed_public_route and native_t2_canary capture"
            )
        if args.evaluator_private_manifest_retention_root is None:
            raise ValueError(
                "--evaluator-private-manifest-retention-root is required for fixed_public_route and native_t2_canary capture"
            )
        if (args.onboard_width, args.onboard_height) != (
            ONBOARD_IMAGE_WIDTH,
            ONBOARD_IMAGE_HEIGHT,
        ):
            raise ValueError(
                "fixed_public_route and native_t2_canary freeze onboard camera resolution at "
                f"{ONBOARD_IMAGE_WIDTH}x{ONBOARD_IMAGE_HEIGHT} for visibility geometry"
            )
        if args.sb3_checkpoint is not None or args.sb3_metadata is not None:
            raise ValueError("SB3 checkpoint options require --control-mode sb3_state_only_transfer")
        if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
            if args.cf2x_runtime_calibration is None:
                raise ValueError("native_t2_canary requires --cf2x-runtime-calibration")
            if not args.cf2x_runtime_calibration.expanduser().resolve().is_file():
                raise FileNotFoundError(
                    "CF2X runtime calibration is missing: "
                    f"{args.cf2x_runtime_calibration.expanduser().resolve()}"
                )
            if args.runtime_lock is None or args.isaaclab_source is None:
                raise ValueError("native_t2_canary requires --runtime-lock and --isaaclab-source")
            if isinstance(args.native_t2_decision_stride, bool) or args.native_t2_decision_stride < 1:
                raise ValueError("--native-t2-decision-stride must be a positive integer")
            for name, maximum in (
                ("native_t2_max_horizontal_speed_mps", STATE_ONLY_POLICY_HORIZONTAL_SPEED_MPS),
                ("native_t2_max_vertical_speed_mps", STATE_ONLY_POLICY_VERTICAL_SPEED_MPS),
                ("native_t2_max_yaw_rate_radps", STATE_ONLY_POLICY_YAW_RATE_RADPS),
            ):
                value = float(getattr(args, name))
                if not math.isfinite(value) or not 0.0 < value <= maximum:
                    raise ValueError(
                        f"--{name.replace('_', '-')} must be finite and in (0, {maximum}]"
                    )
    elif args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
        if args.evaluator_private_manifest is not None:
            raise ValueError(
                "sb3_state_only_transfer forbids --evaluator-private-manifest: "
                "the development control smoke has no private targets"
            )
        if args.evaluator_private_manifest_retention_root is not None:
            raise ValueError(
                "sb3_state_only_transfer forbids --evaluator-private-manifest-retention-root"
            )
        if args.sb3_checkpoint is None:
            raise ValueError("sb3_state_only_transfer requires --sb3-checkpoint")
        if not args.sb3_checkpoint.expanduser().resolve().is_file():
            raise FileNotFoundError(f"SB3 checkpoint is missing: {args.sb3_checkpoint.expanduser().resolve()}")
        if args.sb3_metadata is not None and not args.sb3_metadata.expanduser().resolve().is_file():
            raise FileNotFoundError(f"SB3 metadata is missing: {args.sb3_metadata.expanduser().resolve()}")
        if isinstance(args.sb3_decision_stride, bool) or args.sb3_decision_stride < 1:
            raise ValueError("--sb3-decision-stride must be a positive integer")
        for name, maximum in (
            ("sb3_max_horizontal_speed_mps", STATE_ONLY_POLICY_HORIZONTAL_SPEED_MPS),
            ("sb3_max_vertical_speed_mps", STATE_ONLY_POLICY_VERTICAL_SPEED_MPS),
            ("sb3_max_yaw_rate_radps", STATE_ONLY_POLICY_YAW_RATE_RADPS),
        ):
            value = float(getattr(args, name))
            if not math.isfinite(value) or not 0.0 < value <= maximum:
                raise ValueError(
                    f"--{name.replace('_', '-')} must be finite and in (0, {maximum}]"
                )


    else:
        raise ValueError(f"unknown control mode: {args.control_mode!r}")


def _bind_runtime_lock_to_args(
    args: argparse.Namespace,
    receipt: dict[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Load a lock and reject command-line drift before Isaac is allocated."""

    if args.runtime_lock is None:
        return None
    from .runtime_lock import load_runtime_lock, runtime_lock_sha256

    lock = load_runtime_lock(args.runtime_lock.expanduser().resolve())
    simulation = lock["simulation"]
    expected_device = str(simulation["device"])
    expected_dt = float(simulation["dt_s"])
    if str(args.device) != expected_device:
        raise ValueError(
            f"--device {args.device!r} conflicts with runtime lock {expected_device!r}"
        )
    if not math.isclose(float(args.dt), expected_dt, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            f"--dt {args.dt!r} conflicts with runtime lock {expected_dt!r}"
        )
    launcher = lock["launcher"]
    if launcher.get("enable_cameras") is not True:
        raise ValueError("runtime lock must enable cameras for native capture")
    expected_headless = bool(launcher["headless"])
    if bool(args.headless) != expected_headless:
        raise ValueError(
            f"--headless {bool(args.headless)!r} conflicts with runtime lock "
            f"{expected_headless!r}"
        )
    if receipt is not None:
        receipt["runtime_lock"] = {
            "path": str(args.runtime_lock.expanduser().resolve()),
            "profile_id": str(lock["profile_id"]),
            "sha256": runtime_lock_sha256(lock),
        }
    return lock


def _captured_frame_indices(steps: int, capture_stride: int) -> tuple[int, ...]:
    """Return zero-based rollout steps retained by the capture cadence.

    A trailing partial stride is an accepted physical frame.  Keeping the
    exact indices in one Isaac-free helper prevents the spool allocation,
    capture loop, receipt, and independent validator from drifting apart.
    """

    if steps < 1 or capture_stride < 1:
        raise ValueError("steps and capture_stride must be positive")
    indices = list(range(capture_stride - 1, steps, capture_stride))
    if not indices or indices[-1] != steps - 1:
        indices.append(steps - 1)
    return tuple(indices)


def _captured_frame_count(steps: int, capture_stride: int) -> int:
    """Return the exact number of retained frames for the capture cadence."""

    return len(_captured_frame_indices(steps, capture_stride))


def _overview_archive_frame_indices(
    sensor_frame_count: int,
    archive_stride: int = OVERVIEW_ARCHIVE_STRIDE,
) -> tuple[int, ...]:
    """Return the immutable low-rate overview evidence schedule.

    The first and final retained sensor frames are always included.  Interior
    entries are selected solely by their zero-based retained-frame index, so
    the archive cannot become a quality-selected subset after a run succeeds.
    """

    if sensor_frame_count < 1 or archive_stride < 1:
        raise ValueError("sensor_frame_count and archive_stride must be positive")
    indices = list(range(0, sensor_frame_count, archive_stride))
    if indices[-1] != sensor_frame_count - 1:
        indices.append(sensor_frame_count - 1)
    return tuple(indices)


@dataclass(frozen=True)
class CaptureStorageBudget:
    """Conservative on-volume reservation for one native capture.

    This is deliberately a capacity model, not a prediction of compressed
    artifact size.  It counts raw numeric payloads, the temporary spool-growth
    generation, and a second finalization generation before adding fixed
    headroom.  A caller therefore cannot make preflight pass merely by claiming
    that RGB-D or semantic frames will compress well.
    """

    sensor_frame_count: int
    overview_frame_count: int
    physics_frame_count: int
    sensor_spool_bytes: int
    overview_spool_bytes: int
    metadata_and_trace_bytes: int
    archive_container_overhead_bytes: int
    spool_growth_peak_bytes: int
    finalization_peak_bytes: int
    required_bytes: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "schema": CAPTURE_STORAGE_BUDGET_SCHEMA,
            "sensor_frame_count": self.sensor_frame_count,
            "overview_frame_count": self.overview_frame_count,
            "physics_frame_count": self.physics_frame_count,
            "sensor_spool_bytes": self.sensor_spool_bytes,
            "overview_spool_bytes": self.overview_spool_bytes,
            "metadata_and_trace_bytes": self.metadata_and_trace_bytes,
            "archive_container_overhead_bytes": self.archive_container_overhead_bytes,
            "spool_growth_peak_bytes": self.spool_growth_peak_bytes,
            "finalization_peak_bytes": self.finalization_peak_bytes,
            "required_bytes": self.required_bytes,
        }


def _capture_storage_budget(args: argparse.Namespace) -> CaptureStorageBudget:
    """Derive the minimum acceptable capture reservation from frozen layouts.

    The source keeps the calculation Isaac-free so it can fail before an
    AppLauncher is created.  The explicit dtype widths mirror the arrays
    written in ``_capture``: RGB is uint8, semantic/depth and sensor values are
    32-bit numeric tensors, and timestamp/camera-witness scalars are int64 or
    float64.  If a retained field is added, this model must change in the same
    commit and its tests must be updated.
    """

    dimensions = {
        "steps": getattr(args, "steps", None),
        "warmup_steps": getattr(args, "warmup_steps", None),
        "capture_stride": getattr(args, "capture_stride", None),
        "onboard_width": getattr(args, "onboard_width", None),
        "onboard_height": getattr(args, "onboard_height", None),
        "overview_width": getattr(args, "overview_width", None),
        "overview_height": getattr(args, "overview_height", None),
    }
    for name, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer for storage budgeting")
    if dimensions["steps"] < 1 or dimensions["capture_stride"] < 1:
        raise ValueError("steps and capture_stride must be positive for storage budgeting")
    for name in ("onboard_width", "onboard_height", "overview_width", "overview_height"):
        if dimensions[name] < 1:
            raise ValueError(f"{name} must be positive for storage budgeting")

    sensor_frame_count = _captured_frame_count(
        dimensions["steps"], dimensions["capture_stride"]
    )
    overview_frame_count = len(_overview_archive_frame_indices(sensor_frame_count))
    physics_frame_count = 1 + dimensions["warmup_steps"] + dimensions["steps"]

    onboard_pixels = dimensions["onboard_width"] * dimensions["onboard_height"]
    overview_pixels = dimensions["overview_width"] * dimensions["overview_height"]
    # Per retained onboard frame: RGB [A,H,W,3] uint8; depth and semantic are
    # [A,H,W,1] float32/int32.  The remaining spool tensors are all float32
    # apart from the shared int64 timestamp.
    onboard_image_bytes = AGENT_COUNT * onboard_pixels * (3 + 4 + 4)
    onboard_state_float_count = AGENT_COUNT * (
        # Camera expected/observed poses, closure errors and USD closure.
        3 + 4 + 3 + 4 + 1 + 1 + 1 + 1 + 1
        # LiDAR pose plus the fixed native 16 x 72 range layout.
        + 3 + 4 + LIDAR_RAY_COUNT
        # IMU pose, acceleration and angular velocity.
        + 3 + 4 + 3 + 3
        # Contact net force.
        + 3
    )
    sensor_spool_bytes = sensor_frame_count * (
        onboard_image_bytes + onboard_state_float_count * 4 + 8
    )

    # The overview evidence stores only RGB, native semantics and fixed camera
    # witness scalars.  Runtime-only overview depth is intentionally excluded
    # because it is never retained.
    overview_spool_bytes = overview_frame_count * (
        overview_pixels * (3 + 4) + (3 + 4 + 3) * 8 + 8
    )
    metadata_and_trace_bytes = (
        sensor_frame_count * _SEMANTIC_METADATA_RESERVATION_PER_FRAME_BYTES
        + physics_frame_count * _RUNTIME_TRACE_RESERVATION_PER_PHYSICS_STEP_BYTES
        + 16 * 1024 * 1024
    )
    archive_member_count = 3 * sensor_frame_count + 2 * overview_frame_count + 96
    archive_container_overhead_bytes = (
        archive_member_count * _ARCHIVE_CONTAINER_OVERHEAD_PER_MEMBER_BYTES
    )
    durable_raw_bytes = (
        sensor_spool_bytes
        + overview_spool_bytes
        + metadata_and_trace_bytes
        + archive_container_overhead_bytes
    )
    # On-demand spool growth can briefly contain a complete old and new
    # generation.  Finalization can retain the old spool while atomically
    # writing a complete archive generation.  Treat both as two complete raw
    # generations; this is intentionally larger than the normal path.
    spool_growth_peak_bytes = 2 * (sensor_spool_bytes + overview_spool_bytes)
    finalization_peak_bytes = 2 * durable_raw_bytes
    unheaded_required = max(spool_growth_peak_bytes, finalization_peak_bytes)
    required_bytes = math.ceil(
        unheaded_required
        * _STORAGE_RESERVATION_HEADROOM_NUMERATOR
        / _STORAGE_RESERVATION_HEADROOM_DENOMINATOR
    )
    return CaptureStorageBudget(
        sensor_frame_count=sensor_frame_count,
        overview_frame_count=overview_frame_count,
        physics_frame_count=physics_frame_count,
        sensor_spool_bytes=sensor_spool_bytes,
        overview_spool_bytes=overview_spool_bytes,
        metadata_and_trace_bytes=metadata_and_trace_bytes,
        archive_container_overhead_bytes=archive_container_overhead_bytes,
        spool_growth_peak_bytes=spool_growth_peak_bytes,
        finalization_peak_bytes=finalization_peak_bytes,
        required_bytes=required_bytes,
    )


def _capture_tree_bytes(root: Path) -> int:
    """Return durable regular-file bytes below a capture directory.

    This is intentionally a conservative accounting helper rather than a
    directory-size proxy: Windows allocation-unit size can only increase the
    true demand, while the preflight headroom absorbs normal allocation slack.
    """

    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            # A simultaneous partial artifact is itself a capture failure at
            # the subsequent write boundary.  Do not report a guessed byte
            # total as successful evidence.
            raise RuntimeError(f"cannot account capture storage at {path}") from None
    return total


def _enforce_runtime_storage_guard(
    args: argparse.Namespace,
    receipt: dict[str, Any],
    *,
    phase: str,
    output_dir: Path,
    budget: CaptureStorageBudget,
) -> None:
    """Keep the preflight reservation valid at spool/finalization boundaries."""

    usage = shutil.disk_usage(output_dir)
    captured_bytes = _capture_tree_bytes(output_dir)
    required_free_bytes = int(float(args.minimum_free_gib) * 1024**3)
    required_total_bytes = required_free_bytes + budget.required_bytes
    observed_total_bytes = usage.free + captured_bytes
    event = {
        "phase": phase,
        "free_bytes": usage.free,
        "capture_bytes": captured_bytes,
        "observed_total_bytes": observed_total_bytes,
        "required_total_bytes": required_total_bytes,
        "passed": observed_total_bytes >= required_total_bytes,
    }
    guard = receipt.setdefault(
        "runtime_storage_guard",
        {
            "schema": "org.rivermark.isaac-runtime-storage-guard.v1",
            "budget": budget.as_dict(),
            "minimum_free_bytes": required_free_bytes,
            "events": [],
        },
    )
    events = guard["events"]
    if not isinstance(events, list):
        raise RuntimeError("runtime storage guard receipt is malformed")
    events.append(event)
    if not event["passed"]:
        raise RuntimeError(
            "runtime storage reservation was consumed before "
            f"{phase}: observed {observed_total_bytes} bytes, require {required_total_bytes}"
        )


@dataclass
class _SensorUpdateTimeline:
    """Advance each IsaacLab sensor exactly to an absolute physics timestamp.

    IsaacLab ``SensorBase.update`` increments an internal sensor clock on
    every call.  Safety and retained-contact reads may happen in the same
    physical frame, so a second read must receive a zero increment rather
    than another simulation interval.
    """

    _last_update_time_ns: dict[int, int]

    def __init__(self) -> None:
        self._last_update_time_ns = {}

    def update(self, sensor: Any, *, time_ns: int) -> None:
        if isinstance(time_ns, bool) or not isinstance(time_ns, int) or time_ns < 0:
            raise ValueError("sensor update time must be a non-negative integer nanosecond timestamp")
        key = id(sensor)
        previous = self._last_update_time_ns.get(key, 0)
        if time_ns < previous:
            raise ValueError("sensor update time cannot move backwards")
        sensor.update((time_ns - previous) / 1_000_000_000.0, force_recompute=True)
        self._last_update_time_ns[key] = time_ns


def _windows_system_commit_snapshot() -> dict[str, float | int] | None:
    """Read Windows CommitTotal/CommitLimit without spawning a subprocess."""

    if os.name != "nt":
        return None
    try:
        import ctypes

        class _PerformanceInformation(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("CommitTotal", ctypes.c_size_t),
                ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t),
                ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t),
                ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t),
                ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_uint32),
                ("ProcessCount", ctypes.c_uint32),
                ("ThreadCount", ctypes.c_uint32),
            ]

        performance = _PerformanceInformation()
        performance.cb = ctypes.sizeof(performance)
        api = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
        api.argtypes = [ctypes.POINTER(_PerformanceInformation), ctypes.c_uint32]
        api.restype = ctypes.c_int
        if not api(ctypes.byref(performance), performance.cb):
            return None
        total = int(performance.CommitTotal * performance.PageSize)
        limit = int(performance.CommitLimit * performance.PageSize)
        peak = int(performance.CommitPeak * performance.PageSize)
        if total < 0 or limit <= 0 or peak < total:
            return None
        return {
            "commit_total_bytes": total,
            "commit_limit_bytes": limit,
            "commit_peak_bytes": peak,
            "commit_percent": 100.0 * total / limit,
        }
    except (AttributeError, OSError):
        return None


def _enforce_system_commit_guard(
    args: argparse.Namespace,
    receipt: dict[str, Any],
    *,
    phase: str,
    output_dir: Path | None = None,
    snapshot: Mapping[str, Any] | None | object = _SYSTEM_COMMIT_SNAPSHOT_UNSET,
) -> None:
    """Fail closed before host commit pressure risks Windows process failure.

    Callers which have just sampled :class:`ResourceTelemetry` pass its exact
    ``system_commit`` snapshot here.  This prevents the receipt, telemetry,
    and threshold decision from describing three different host observations.
    The direct query remains for preflight paths that exist before telemetry.
    """

    if snapshot is _SYSTEM_COMMIT_SNAPSHOT_UNSET:
        observed_snapshot = _windows_system_commit_snapshot()
    elif isinstance(snapshot, Mapping):
        observed_snapshot = snapshot
    else:
        # A caller which has already sampled unavailable telemetry must not
        # silently substitute a later host observation for that evidence.
        observed_snapshot = None
    guard = receipt.setdefault(
        "system_commit_guard",
        {
            "schema": "org.rivermark.windows-system-commit-guard.v1",
            "preflight_max_percent": float(args.preflight_commit_percent),
            "abort_max_percent": float(args.abort_commit_percent),
            "status": "unavailable",
        },
    )
    if observed_snapshot is None:
        return
    commit_percent = observed_snapshot.get("commit_percent")
    if (
        not isinstance(commit_percent, (int, float))
        or isinstance(commit_percent, bool)
        or not math.isfinite(float(commit_percent))
    ):
        return
    snapshot_copy = dict(observed_snapshot)
    observed_percent = float(commit_percent)
    guard["status"] = "active"
    guard["last_phase"] = phase
    guard["last_snapshot"] = snapshot_copy
    prior = guard.get("maximum_observed_percent")
    if (
        not isinstance(prior, (int, float))
        or isinstance(prior, bool)
        or not math.isfinite(float(prior))
        or observed_percent > float(prior)
    ):
        guard["maximum_observed_percent"] = observed_percent
        guard["maximum_phase"] = phase
        guard["maximum_snapshot"] = snapshot_copy
    limit = (
        float(args.preflight_commit_percent)
        if phase in _PREFLIGHT_SYSTEM_COMMIT_PHASES
        else float(args.abort_commit_percent)
    )
    if observed_percent >= limit:
        if output_dir is not None:
            _checkpoint(
                output_dir,
                "system_commit_guard_rejected",
                phase=phase,
                threshold_percent=limit,
                snapshot=snapshot_copy,
            )
        raise RuntimeError(
            "refusing Isaac capture because Windows system commit is "
            f"{observed_percent:.2f}% (limit {limit:.2f}%)"
        )


def _enforce_foreign_native_process_guard(
    args: argparse.Namespace,
    receipt: dict[str, Any],
    *,
    phase: str,
    output_dir: Path | None = None,
    census: Mapping[str, Any] | None | object = _SYSTEM_COMMIT_SNAPSHOT_UNSET,
) -> None:
    """Reject whenever another high-commit native runtime owns the host.

    The process census is intentionally anonymous in the receipt.  A human can
    inspect local processes when needed, while a development artifact exposes
    only the aggregate condition that made a run unsafe.  Census unavailability
    or an invalid aggregate is a hard failure: treating either as zero candidates
    would recreate the race this guard exists to prevent.
    """

    threshold_gib = float(getattr(args, "maximum_foreign_native_private_commit_gib", 8.0))
    if not math.isfinite(threshold_gib) or threshold_gib <= 0.0:
        raise RuntimeError(
            "foreign native process guard threshold must be finite and positive"
        )
    threshold_bytes = int(threshold_gib * 1024**3)
    guard = receipt.setdefault(
        "foreign_native_process_guard",
        {
            "schema": "org.rivermark.foreign-native-process-guard.v2",
            "maximum_private_commit_gib": threshold_gib,
            "status": "not_sampled",
            "sample_count": 0,
            "maximum_candidate_count": 0,
            "maximum_candidate_count_phase": None,
            "maximum_candidate_private_commit_bytes": 0,
            "maximum_candidate_private_commit_phase": None,
        },
    )
    if not isinstance(guard, dict):
        raise RuntimeError("foreign native process guard receipt is not mutable")
    if guard.get("schema") != "org.rivermark.foreign-native-process-guard.v2":
        raise RuntimeError("foreign native process guard receipt schema changed during the run")
    if guard.get("maximum_private_commit_gib") != threshold_gib:
        raise RuntimeError("foreign native process guard threshold changed during the run")
    sample_count = guard.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 0:
        raise RuntimeError("foreign native process guard sample count is invalid")
    guard["sample_count"] = sample_count + 1
    guard["last_phase"] = phase
    if census is _SYSTEM_COMMIT_SNAPSHOT_UNSET:
        observed = foreign_native_process_census(
            minimum_private_commit_bytes=threshold_bytes
        )
    elif isinstance(census, Mapping):
        observed = census
    else:
        observed = None
    if observed is None:
        guard["status"] = "rejected"
        guard["last_census_status"] = "unavailable"
        if output_dir is not None:
            _checkpoint(
                output_dir,
                "foreign_native_process_guard_rejected",
                phase=phase,
                reason="census_unavailable",
                threshold_gib=threshold_gib,
            )
        raise RuntimeError(
            f"foreign native process census is unavailable at {phase}; refusing to assume an empty host"
        )

    required_integer_fields = (
        "enumerated_native_process_count",
        "minimum_private_commit_bytes",
        "candidate_count",
        "candidate_private_commit_bytes",
        "maximum_candidate_private_commit_bytes",
    )
    invalid_reason: str | None = None
    if observed.get("schema") != FOREIGN_NATIVE_PROCESS_CENSUS_SCHEMA:
        invalid_reason = "schema"
    elif any(
        not isinstance(observed.get(name), int)
        or isinstance(observed.get(name), bool)
        or int(observed[name]) < 0
        for name in required_integer_fields
    ):
        invalid_reason = "integer_fields"
    else:
        enumerated_count = int(observed["enumerated_native_process_count"])
        observed_threshold = int(observed["minimum_private_commit_bytes"])
        candidate_count = int(observed["candidate_count"])
        candidate_total = int(observed["candidate_private_commit_bytes"])
        candidate_maximum = int(observed["maximum_candidate_private_commit_bytes"])
        if observed_threshold != threshold_bytes:
            invalid_reason = "threshold"
        elif candidate_count > enumerated_count:
            invalid_reason = "candidate_count"
        elif candidate_count == 0 and (candidate_total != 0 or candidate_maximum != 0):
            invalid_reason = "empty_aggregate"
        elif candidate_count > 0 and (
            candidate_maximum < threshold_bytes
            or candidate_total < candidate_maximum
            or candidate_total > candidate_count * candidate_maximum
        ):
            invalid_reason = "candidate_aggregate"

    if invalid_reason is not None:
        guard["status"] = "rejected"
        guard["last_census_status"] = "malformed"
        guard["last_census_error"] = invalid_reason
        if output_dir is not None:
            _checkpoint(
                output_dir,
                "foreign_native_process_guard_rejected",
                phase=phase,
                reason="census_malformed",
                census_error=invalid_reason,
                threshold_gib=threshold_gib,
            )
        raise RuntimeError(
            f"foreign native process census is malformed at {phase} ({invalid_reason})"
        )

    candidate_count = int(observed["candidate_count"])
    candidate_maximum = int(observed["maximum_candidate_private_commit_bytes"])
    guard["status"] = "active"
    guard["last_census_status"] = "available"
    guard["last_census"] = dict(observed)
    if candidate_count > int(guard["maximum_candidate_count"]):
        guard["maximum_candidate_count"] = candidate_count
        guard["maximum_candidate_count_phase"] = phase
    if candidate_maximum > int(guard["maximum_candidate_private_commit_bytes"]):
        guard["maximum_candidate_private_commit_bytes"] = candidate_maximum
        guard["maximum_candidate_private_commit_phase"] = phase
    if candidate_count <= 0:
        return
    guard["status"] = "rejected"
    if output_dir is not None:
        _checkpoint(
            output_dir,
            "foreign_native_process_guard_rejected",
            phase=phase,
            reason="foreign_candidate_present",
            threshold_gib=threshold_gib,
            candidate_count=candidate_count,
            maximum_candidate_private_commit_bytes=candidate_maximum,
        )
    raise RuntimeError(
        "refusing native Isaac run because another high-commit Python/Kit/Isaac "
        f"process is active at {phase} ({candidate_count} process(es) at or above "
        f"{threshold_gib:.2f} GiB)"
    )


def _validate_private_manifest_input(
    output_dir: Path,
    source: Path,
    *,
    repository_root: Path | None = None,
) -> None:
    """Require evaluator truth to pre-exist outside the distributable tree."""

    if _is_within(source, output_dir):
        raise ValueError("--evaluator-private-manifest must be outside --output-dir")
    if repository_root is not None and _is_within(source, repository_root):
        raise ValueError("--evaluator-private-manifest must be outside the repository")
    if source.suffix.lower() != ".json":
        raise ValueError("--evaluator-private-manifest must name a .json file")
    if not source.is_file():
        raise FileNotFoundError(f"external private evaluator manifest is missing: {source}")


def _create_state_only_sb3_transfer(args: argparse.Namespace) -> Any:
    """Load one checked SB3 policy before Isaac allocates a simulation app."""

    if args.control_mode != CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
        raise ValueError("SB3 transfer construction requires sb3_state_only_transfer mode")
    if args.sb3_checkpoint is None:
        raise ValueError("SB3 transfer construction requires --sb3-checkpoint")
    from .isaac_transfer import (
        FixedDecisionCadence,
        WorldCommandBounds,
        create_state_only_sb3_isaac_transfer,
    )

    return create_state_only_sb3_isaac_transfer(
        args.sb3_checkpoint.expanduser().resolve(),
        args.sb3_metadata.expanduser().resolve() if args.sb3_metadata is not None else None,
        cadence=FixedDecisionCadence(args.sb3_decision_stride),
        bounds=WorldCommandBounds(
            max_horizontal_speed_mps=args.sb3_max_horizontal_speed_mps,
            max_vertical_speed_mps=args.sb3_max_vertical_speed_mps,
            max_yaw_rate_rad_s=args.sb3_max_yaw_rate_radps,
        ),
    )


def _load_native_t2_calibration_binding(
    args: argparse.Namespace,
    *,
    runtime_lock: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Read one passed external calibration before AppLauncher allocation."""

    if args.control_mode != CONTROL_MODE_NATIVE_T2_CANARY:
        return None
    if args.cf2x_runtime_calibration is None or runtime_lock is None:
        raise RuntimeError("native T2 calibration prerequisites disappeared after argument validation")
    source = args.cf2x_runtime_calibration.expanduser().resolve()
    try:
        report = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read CF2X runtime calibration: {source}") from exc
    from .runtime_lock import runtime_lock_sha256

    return bind_native_t2_calibration(
        report,
        expected_usd_sha256=_sha256(args.drone_usd.expanduser().resolve()),
        expected_runtime_lock_sha256=runtime_lock_sha256(runtime_lock),
        expected_control_dt_s=float(args.dt),
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrivateEvaluatorManifestError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PrivateEvaluatorManifestError(f"{label} must be a finite number")
    return result


def _normalized_private_targets(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    targets = manifest.get("targets")
    if not isinstance(targets, list) or len(targets) != TARGET_COUNT:
        raise PrivateEvaluatorManifestError(
            f"targets must contain exactly {TARGET_COUNT} external private targets"
        )
    normalized: list[dict[str, Any]] = []
    target_ids: set[str] = set()
    positions: set[tuple[float, float, float]] = set()
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise PrivateEvaluatorManifestError(f"targets[{index}] must be an object")
        target_id = target.get("target_id")
        if not isinstance(target_id, str) or not target_id.strip() or target_id in target_ids:
            raise PrivateEvaluatorManifestError(
                f"targets[{index}].target_id must be a unique nonempty string"
            )
        position = target.get("position_w_m")
        if isinstance(position, (str, bytes)) or not isinstance(position, Sequence) or len(position) != 3:
            raise PrivateEvaluatorManifestError(
                f"targets[{index}].position_w_m must be finite xyz"
            )
        xyz = tuple(
            _finite_float(value, label=f"targets[{index}].position_w_m[{axis}]")
            for axis, value in enumerate(position)
        )
        if xyz in positions:
            raise PrivateEvaluatorManifestError("private target positions must be unique")
        radius_m = _finite_float(target.get("radius_m"), label=f"targets[{index}].radius_m")
        if not 0.0 < radius_m <= PRIVATE_TARGET_MAX_RADIUS_M:
            raise PrivateEvaluatorManifestError(
                f"targets[{index}].radius_m must be in (0, {PRIVATE_TARGET_MAX_RADIUS_M}]"
            )
        target_ids.add(target_id)
        positions.add(xyz)
        visibility_bucket = target.get("visibility_bucket")
        if not isinstance(visibility_bucket, str) or not visibility_bucket.strip():
            raise PrivateEvaluatorManifestError(
                f"targets[{index}].visibility_bucket must be a public difficulty bucket"
            )
        normalized.append(
            {
                "target_id": target_id,
                "position_w_m": xyz,
                "radius_m": radius_m,
                "visibility_bucket": visibility_bucket,
            }
        )
    return tuple(normalized)


def validate_external_private_evaluator_manifest(
    manifest: Mapping[str, Any],
    *,
    city_lite_scene_contract_sha256: str,
    city_lite_scene_payload_sha256: str,
    expected_collection_binding: Mapping[str, Any] | None = None,
    expected_task_variant_id: str = TASK_VARIANT_ID,
    expected_native_t2_motion_contract: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate an evaluator-owned manifest without minting any target truth.

    The selector, its candidate population, and any entropy source are outside
    the public repository.  Capture receives only selected targets from a
    pre-existing private artifact and records only that artifact's SHA-256.
    """

    if not isinstance(manifest, Mapping):
        raise PrivateEvaluatorManifestError("external evaluator manifest must be a JSON object")
    origin = manifest.get("target_origin")
    required_origin = {
        "kind": PRIVATE_TARGET_ORIGIN,
        "candidate_pool_released": False,
        "seed_released": False,
        "coordinates_released": False,
    }
    if not isinstance(origin, Mapping) or (
        origin.get("kind") != required_origin["kind"]
        or origin.get("candidate_pool_released") is not False
        or origin.get("seed_released") is not False
        or origin.get("coordinates_released") is not False
    ):
        raise PrivateEvaluatorManifestError(
            "target_origin must declare an unreleased external private evaluator"
        )
    expected = {
        "schema": PRIVATE_EVALUATOR_SCHEMA,
        "environment_id": ENVIRONMENT_ID,
        "city_lite_scene_contract_sha256": city_lite_scene_contract_sha256,
        "city_lite_scene_payload_sha256": city_lite_scene_payload_sha256,
        "task_variant_id": expected_task_variant_id,
        "sampled_before_policy_start": True,
        "route_conditioning": "public_only",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise PrivateEvaluatorManifestError(f"external evaluator manifest field is invalid: {key}")
    if expected_collection_binding is not None:
        binding = manifest.get("collection_binding")
        issues = validate_collection_binding(binding)
        if issues:
            raise PrivateEvaluatorManifestError(
                "external evaluator manifest collection_binding is invalid: "
                + "; ".join(issue.code for issue in issues)
            )
        if dict(binding) != dict(expected_collection_binding):
            raise PrivateEvaluatorManifestError(
                "external evaluator manifest collection_binding does not match the capture"
            )
    placement = manifest.get("target_placement_contract")
    expected_placement = {
        "schema": PRIVATE_TARGET_PLACEMENT_SCHEMA,
        "obstacle_clearance_m": PRIVATE_TARGET_OBSTACLE_CLEARANCE_M,
        "minimum_route_separation_m": PRIVATE_TARGET_MIN_ROUTE_SEPARATION_M,
        "minimum_pairwise_separation_m": PRIVATE_TARGET_MIN_PAIRWISE_SEPARATION_M,
    }
    if not isinstance(placement, Mapping) or any(
        placement.get(key) != value for key, value in expected_placement.items()
    ):
        raise PrivateEvaluatorManifestError(
            "target_placement_contract must bind the frozen private-placement safety rules"
        )
    visibility = manifest.get("target_visibility_contract")
    if not isinstance(visibility, Mapping):
        raise PrivateEvaluatorManifestError(
            "target_visibility_contract must bind public geometry and native semantic evidence"
        )
    try:
        route_family_id = str(visibility["route_family_id"])
        routes = resolve_public_route_family(route_family_id)
        aabb_hash = str(visibility["aabb_geometry_sha256"])
        target_region_id = str(visibility["target_region_id"])
        visibility_bucket = str(visibility["visibility_bucket"])
        tracking_envelope_m = float(visibility["tracking_envelope_m"])
        execution_window = visibility.get("execution_window")
        if not isinstance(execution_window, Mapping):
            raise ValueError("target visibility contract has no execution window")
        expected_visibility = target_visibility_geometry_contract(
            route_family_id=route_family_id,
            routes_w_m=routes,
            aabb_geometry_sha256=aabb_hash,
            target_region_id=target_region_id,
            visibility_bucket=visibility_bucket,
            tracking_envelope_m=tracking_envelope_m,
            execution_window=execution_window,
            **_target_visibility_heading_kwargs(visibility),
        )
    except (KeyError, TypeError, ValueError, CityLiteRouteError) as exc:
        raise PrivateEvaluatorManifestError(
            f"target_visibility_contract is invalid: {exc}"
        ) from exc
    if dict(visibility) != expected_visibility:
        raise PrivateEvaluatorManifestError(
            "target_visibility_contract differs from the frozen geometry/semantic contract"
        )
    if expected_native_t2_motion_contract is not None:
        _validate_target_visibility_against_native_t2_motion(
            visibility, expected_native_t2_motion_contract
        )
    if any(target["visibility_bucket"] != visibility_bucket for target in _normalized_private_targets(manifest)):
        raise PrivateEvaluatorManifestError(
            "target visibility buckets must agree with the manifest visibility contract"
        )
    return _normalized_private_targets(manifest)


def _validate_target_visibility_against_native_t2_motion(
    visibility_contract: Mapping[str, Any], motion_contract: Mapping[str, Any]
) -> None:
    """Bind a v4 private target contract to the independently loaded protocol.

    The manifest's own SHA only proves self-consistency.  This comparison is
    deliberately against the capture-side protocol content, so changing a
    heading or retained-window field in the private file cannot weaken target
    placement without failing before Isaac starts.
    """

    try:
        expected_window = target_visibility_execution_window(
            dt_s=float(motion_contract["dt_s"]),
            warmup_steps=int(motion_contract["warmup_steps"]),
            rollout_steps=int(motion_contract["rollout_steps"]),
            capture_stride=int(motion_contract["capture_stride"]),
            waypoint_segment_seconds=float(motion_contract["waypoint_segment_seconds"]),
        )
        expected_heading = {
            "model": str(motion_contract["camera_heading_model"]),
            "max_yaw_rate_rad_s": float(motion_contract["max_yaw_rate_rad_s"]),
            "yaw_feedback_gain": float(motion_contract["yaw_feedback_gain"]),
            "yaw_stability_error_rad": float(motion_contract["yaw_stability_error_rad"]),
            "yaw_settle_margin_s": float(motion_contract["yaw_settle_margin_s"]),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PrivateEvaluatorManifestError("native T2 motion contract is malformed") from exc
    if visibility_contract.get("execution_window") != expected_window:
        raise PrivateEvaluatorManifestError(
            "target visibility execution window does not match the frozen native T2 motion contract"
        )
    if visibility_contract.get("camera_heading_contract") != expected_heading:
        raise PrivateEvaluatorManifestError(
            "target visibility heading contract does not match the frozen native T2 motion contract"
        )


def _target_visibility_heading_kwargs(
    visibility_contract: Mapping[str, Any],
) -> dict[str, float | str]:
    """Extract the optional v4 public camera-heading schedule.

    Absence intentionally means the frozen v3 initial-heading model.  A
    malformed v4 field is left to ``target_visibility_geometry_contract`` so
    every manifest validator shares one strict semantic implementation.
    """

    heading = visibility_contract.get("camera_heading_contract")
    if heading is None:
        return {}
    if not isinstance(heading, Mapping):
        raise ValueError("target visibility camera_heading_contract must be an object")
    return {
        "camera_heading_model": str(heading.get("model")),
        "max_yaw_rate_rad_s": float(heading.get("max_yaw_rate_rad_s")),
        "yaw_feedback_gain": float(heading.get("yaw_feedback_gain")),
        "yaw_stability_error_rad": float(heading.get("yaw_stability_error_rad")),
        "yaw_settle_margin_s": float(heading.get("yaw_settle_margin_s")),
    }


def _target_semantic_visibility_evidence(
    semantic: Any,
    semantic_metadata: Any,
    target_slots: Sequence[str],
    *,
    minimum_pixels: int,
) -> dict[str, Any]:
    """Count anonymous target slots in the just-rendered onboard semantic frame.

    Slots are capture-local learning-label identities.  They are deliberately
    unrelated to evaluator-owned target IDs, which must never enter a public
    semantic label or payload.
    """

    semantic_array = np.asarray(_to_numpy(semantic))
    if semantic_array.ndim == 4 and semantic_array.shape[-1] == 1:
        semantic_array = semantic_array[..., 0]
    if semantic_array.ndim == 2:
        semantic_array = semantic_array[None, ...]
    if semantic_array.ndim != 3 or not np.issubdtype(semantic_array.dtype, np.integer):
        return {
            "schema": "org.rivermark.isaac-target-visibility-evidence.v1",
            "passed": False,
            "per_target_slot": {},
            "failures": ["onboard semantic segmentation is unavailable or malformed"],
        }

    if len(set(target_slots)) != len(target_slots) or any(
        not isinstance(slot, str) or not slot.startswith(TARGET_SEMANTIC_INSTANCE_PREFIX)
        for slot in target_slots
    ):
        return {
            "schema": "org.rivermark.isaac-target-visibility-evidence.v1",
            "passed": False,
            "per_target_slot": {},
            "failures": ["capture-local target slots are invalid"],
        }
    per_camera_metadata = (
        semantic_metadata.get("per_camera")
        if isinstance(semantic_metadata, Mapping)
        else None
    )
    if (
        not isinstance(per_camera_metadata, (list, tuple))
        or len(per_camera_metadata) != semantic_array.shape[0]
    ):
        return {
            "schema": "org.rivermark.isaac-target-visibility-evidence.v2",
            "passed": False,
            "per_target_slot": {},
            "failures": [
                "onboard semantic metadata must provide one ID mapping per rendered camera"
            ],
        }

    # Replicator assigns segmentation IDs independently for each render product.
    # A semantic ID from camera A must never be applied to pixels from camera B.
    # The public slot itself is the target's sole class label because the locked
    # Isaac runtime drops auxiliary semantic instances from the ID mapping.
    target_semantic_ids: dict[str, list[set[int]]] = {
        slot: [set() for _ in range(semantic_array.shape[0])]
        for slot in target_slots
    }
    for camera_index, metadata in enumerate(per_camera_metadata):
        if not isinstance(metadata, Mapping):
            continue
        labels = metadata.get("id_to_labels", metadata.get("idToLabels"))
        if not isinstance(labels, Mapping):
            continue
        for raw_id, raw_labels in labels.items():
            try:
                semantic_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            class_value = raw_labels.get("class") if isinstance(raw_labels, Mapping) else None
            class_labels = {
                item.strip().lower()
                for item in str(class_value).split(",")
                if item.strip()
            }
            for target_slot in target_slots:
                if target_slot.lower() in class_labels:
                    target_semantic_ids[target_slot][camera_index].add(semantic_id)

    per_target_slot: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for target_slot in target_slots:
        semantic_ids_by_camera = target_semantic_ids[target_slot]
        pixels = np.asarray(
            [
                np.count_nonzero(
                    np.isin(
                        semantic_array[camera_index],
                        np.asarray(sorted(semantic_ids), dtype=semantic_array.dtype),
                    )
                )
                if semantic_ids
                else 0
                for camera_index, semantic_ids in enumerate(semantic_ids_by_camera)
            ],
            dtype=np.int64,
        )
        visible_frames = int(bool(np.any(pixels >= minimum_pixels)))
        per_target_slot[target_slot] = {
            "semantic_ids_by_camera": [
                sorted(semantic_ids) for semantic_ids in semantic_ids_by_camera
            ],
            "maximum_pixels_in_one_camera": int(np.max(pixels)) if pixels.size else 0,
            "visible_sensor_frames": visible_frames,
            "minimum_visible_sensor_frames": PRIVATE_TARGET_MIN_VISIBLE_SENSOR_FRAMES,
            "minimum_visible_instance_pixels": minimum_pixels,
        }
        if not any(semantic_ids_by_camera):
            failures.append(f"target slot {target_slot} has no native semantic slot-label ID")
        elif visible_frames < PRIVATE_TARGET_MIN_VISIBLE_SENSOR_FRAMES:
            failures.append(f"target slot {target_slot} is not visible in a native onboard frame")
    return {
        "schema": "org.rivermark.isaac-target-visibility-evidence.v2",
        "passed": not failures,
        "per_target_slot": per_target_slot,
        "failures": failures,
    }


def _target_semantic_slots(count: int) -> tuple[str, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("target slot count must be a non-negative integer")
    return tuple(f"{TARGET_SEMANTIC_INSTANCE_PREFIX}{index:03d}" for index in range(count))


def _target_visibility_checkpoint_summary(
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Reduce native target visibility evidence for a frame checkpoint.

    ``per_target_slot`` is the public capture-local ABI.  Keeping this access
    in one helper prevents a stale evaluator-oriented key from aborting a
    physical capture after raw sensor frames have already been written.
    """

    if evidence is None:
        return None
    per_target_slot = evidence.get("per_target_slot")
    if not isinstance(per_target_slot, Mapping):
        raise ValueError("target visibility evidence is missing per_target_slot")
    return {
        "schema": evidence["schema"],
        "passed": evidence["passed"],
        "visible_target_count": sum(
            int(row["visible_sensor_frames"] > 0)
            for row in per_target_slot.values()
        ),
        "target_count": len(per_target_slot),
    }


def _target_visibility_rollout_summary(
    target_slots: Sequence[str],
    evidence_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate native target visibility without exposing evaluator truth.

    Capture-local semantic slots are intentionally the only identities in the
    persisted summary.  Target IDs, positions, private seed material, and the
    evaluator manifest stay outside the capture directory even when a final
    visibility gate fails.
    """

    if len(set(target_slots)) != len(target_slots) or any(
        not isinstance(slot, str) or not slot.startswith(TARGET_SEMANTIC_INSTANCE_PREFIX)
        for slot in target_slots
    ):
        raise ValueError("target visibility summary requires unique capture-local slots")
    observed: dict[str, dict[str, int]] = {
        target_slot: {"max_pixels": 0, "visible_frames": 0}
        for target_slot in target_slots
    }
    for evidence in evidence_samples:
        per_target_slot = evidence.get("per_target_slot")
        if not isinstance(per_target_slot, Mapping):
            continue
        for target_slot, row in per_target_slot.items():
            if target_slot not in observed or not isinstance(row, Mapping):
                continue
            observed[target_slot]["max_pixels"] = max(
                observed[target_slot]["max_pixels"],
                int(row.get("maximum_pixels_in_one_camera", 0)),
            )
            observed[target_slot]["visible_frames"] += int(
                row.get("visible_sensor_frames", 0)
            )
    failed_target_slots = [
        target_slot
        for target_slot, row in observed.items()
        if row["visible_frames"] < PRIVATE_TARGET_MIN_VISIBLE_SENSOR_FRAMES
    ]
    return {
        "schema": "org.rivermark.isaac-target-visibility-summary.v2",
        "target_count": len(target_slots),
        "targets_meeting_visibility": len(target_slots) - len(failed_target_slots),
        "minimum_visible_sensor_frames_per_target": PRIVATE_TARGET_MIN_VISIBLE_SENSOR_FRAMES,
        "minimum_visible_instance_pixels": PRIVATE_TARGET_MIN_VISIBLE_INSTANCE_PIXELS,
        "passed": not failed_target_slots,
        "failed_target_count": len(failed_target_slots),
        "failed_target_slots": failed_target_slots,
        "per_target_slot": observed,
    }


def _redact_private_target_metadata(
    value: Any, *, private_target_ids: Sequence[str] = ()
) -> Any:
    """Remove private evaluator identifiers from public semantic metadata."""

    private_tokens = tuple(
        token.lower()
        for token in private_target_ids
        if isinstance(token, str) and token.strip()
    )

    def contains_private_token(item: Any) -> bool:
        lowered = str(item).lower()
        return any(token in lowered for token in private_tokens)

    if isinstance(value, Mapping):
        return {
            key: _redact_private_target_metadata(
                child, private_target_ids=private_target_ids
            )
            for key, child in value.items()
            if str(key).lower() not in {"target_id", "target_ids"}
            and not contains_private_token(key)
        }
    if isinstance(value, list):
        return [
            _redact_private_target_metadata(child, private_target_ids=private_target_ids)
            for child in value
        ]
    if isinstance(value, tuple):
        return [
            _redact_private_target_metadata(child, private_target_ids=private_target_ids)
            for child in value
        ]
    if isinstance(value, str) and contains_private_token(value):
        return "[private-target-redacted]"
    return value


def _public_capture_failure(
    error: BaseException,
    *,
    private_route: bool,
) -> dict[str, str | bool]:
    """Return failure evidence that cannot disclose evaluator-private inputs.

    A failed private-route capture is still useful locally through its process
    stderr and retained raw artifact directory, but its receipt is a public
    control-plane document.  Exception strings and tracebacks are therefore not
    serialized for that route: validation errors can contain target IDs or
    coordinates even when the normal success receipt is aggregate-only.
    """

    if private_route:
        return {
            "type": type(error).__name__,
            "message": "private-route capture failed; evaluator-owned details redacted",
            "traceback_redacted": True,
            "private_inputs_redacted": True,
        }
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(limit=30),
        "private_inputs_redacted": False,
    }


def _load_external_private_evaluator_manifest(
    path: Path,
    authority: CityLiteAuthority,
    *,
    expected_collection_binding: Mapping[str, Any] | None = None,
    expected_task_variant_id: str = TASK_VARIANT_ID,
    expected_native_t2_motion_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise PrivateEvaluatorManifestError(
            f"cannot read external evaluator manifest {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise PrivateEvaluatorManifestError("external evaluator manifest must be a JSON object")
    validate_external_private_evaluator_manifest(
        payload,
        city_lite_scene_contract_sha256=authority.contract_sha256,
        city_lite_scene_payload_sha256=authority.contract_payload_sha256,
        expected_collection_binding=expected_collection_binding,
        expected_task_variant_id=expected_task_variant_id,
        expected_native_t2_motion_contract=expected_native_t2_motion_contract,
    )
    return payload


def _point_to_segment_distance_m(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    direction = tuple(float(end[axis]) - float(start[axis]) for axis in range(3))
    squared_length = sum(component * component for component in direction)
    if squared_length <= 1.0e-12:
        return math.dist(point, start)
    projection = sum(
        (float(point[axis]) - float(start[axis])) * direction[axis]
        for axis in range(3)
    ) / squared_length
    alpha = min(1.0, max(0.0, projection))
    closest = tuple(float(start[axis]) + alpha * direction[axis] for axis in range(3))
    return math.dist(point, closest)


def validate_private_target_geometry(
    manifest: Mapping[str, Any],
    *,
    structural_aabbs: Sequence[AABB],
    public_routes_w_m: Sequence[Sequence[Sequence[float]]],
    city_lite_scene_contract_sha256: str,
    city_lite_scene_payload_sha256: str,
    execution_window: Mapping[str, Any] | None = None,
    expected_task_variant_id: str = TASK_VARIANT_ID,
    expected_native_t2_motion_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject unsafe/private targets before they can be spawned in Isaac.

    This checks selected truth against runtime City-Lite geometry and public
    routes.  It does not reveal positions, candidate pools, or source seeds.
    """

    targets = validate_external_private_evaluator_manifest(
        manifest,
        city_lite_scene_contract_sha256=city_lite_scene_contract_sha256,
        city_lite_scene_payload_sha256=city_lite_scene_payload_sha256,
        expected_task_variant_id=expected_task_variant_id,
        expected_native_t2_motion_contract=expected_native_t2_motion_contract,
    )
    if not structural_aabbs:
        raise PrivateEvaluatorManifestError("private target placement requires structural City-Lite AABBs")
    route_segments = [
        (start, end)
        for route in public_routes_w_m
        for start, end in zip(route, route[1:])
    ]
    if not route_segments:
        raise PrivateEvaluatorManifestError("private target placement requires public route segments")

    visibility_contract = manifest.get("target_visibility_contract")
    if not isinstance(visibility_contract, Mapping):
        raise PrivateEvaluatorManifestError("private target visibility contract is missing")
    route_family_id = str(visibility_contract.get("route_family_id", ""))
    try:
        expected_routes = resolve_public_route_family(route_family_id)
    except CityLiteRouteError as exc:
        raise PrivateEvaluatorManifestError(str(exc)) from exc
    if canonical_payload_sha256(public_routes_w_m) != canonical_payload_sha256(expected_routes):
        raise PrivateEvaluatorManifestError(
            "private target visibility contract route family does not match the executed public route"
        )
    declared_window = visibility_contract.get("execution_window")
    if not isinstance(declared_window, Mapping):
        raise PrivateEvaluatorManifestError("private target visibility contract has no execution window")
    effective_window = declared_window if execution_window is None else execution_window
    geometry_sha256 = aabb_geometry_sha256(structural_aabbs)
    expected_visibility_contract = target_visibility_geometry_contract(
        route_family_id=route_family_id,
        routes_w_m=expected_routes,
        aabb_geometry_sha256=geometry_sha256,
        target_region_id=str(visibility_contract.get("target_region_id", "")),
        visibility_bucket=str(visibility_contract.get("visibility_bucket", "")),
        tracking_envelope_m=float(visibility_contract.get("tracking_envelope_m")),
        execution_window=effective_window,
        **_target_visibility_heading_kwargs(visibility_contract),
    )
    if dict(visibility_contract) != expected_visibility_contract:
        raise PrivateEvaluatorManifestError(
            "private target visibility contract is not bound to runtime route/AABB geometry"
        )

    minimum_route_distance = math.inf
    for target in targets:
        position = target["position_w_m"]
        radius_m = float(target["radius_m"])
        if not CITY_LITE_FLIGHT_VOLUME_W_M.contains(position, margin_m=radius_m):
            raise PrivateEvaluatorManifestError(
                f"private target {target['target_id']} does not fit inside the City-Lite flight volume"
            )
        for box in structural_aabbs:
            if box.expanded(PRIVATE_TARGET_OBSTACLE_CLEARANCE_M + radius_m).contains(position):
                raise PrivateEvaluatorManifestError(
                    f"private target {target['target_id']} overlaps protected City-Lite geometry {box.source_prim}"
                )
        target_route_distance = min(
            _point_to_segment_distance_m(position, start, end)
            for start, end in route_segments
        )
        minimum_route_distance = min(minimum_route_distance, target_route_distance)
        if target_route_distance < PRIVATE_TARGET_MIN_ROUTE_SEPARATION_M:
            raise PrivateEvaluatorManifestError(
                f"private target {target['target_id']} is too close to a public route segment"
            )

    minimum_pairwise_distance = math.inf
    for left_index, left in enumerate(targets):
        for right in targets[left_index + 1 :]:
            separation = math.dist(left["position_w_m"], right["position_w_m"])
            minimum_pairwise_distance = min(minimum_pairwise_distance, separation)
            required = max(
                PRIVATE_TARGET_MIN_PAIRWISE_SEPARATION_M,
                float(left["radius_m"]) + float(right["radius_m"]) + TARGET_DETECTION_RADIUS_M,
            )
            if separation < required:
                raise PrivateEvaluatorManifestError(
                    f"private targets {left['target_id']} and {right['target_id']} are too close"
                )
    target_region_id = target_region_for_positions(
        [target["position_w_m"] for target in targets]
    )
    if target_region_id != visibility_contract["target_region_id"]:
        raise PrivateEvaluatorManifestError(
            "private targets do not all lie in the declared holdout target region"
        )
    visibility_passed, visibility_evidence = verify_target_visibility_bucket(
        [target["position_w_m"] for target in targets],
        requested_bucket=str(visibility_contract["visibility_bucket"]),
        routes_w_m=public_routes_w_m,
        structural_aabbs=structural_aabbs,
        radii_m=[float(target["radius_m"]) for target in targets],
        tracking_envelope_m=float(visibility_contract["tracking_envelope_m"]),
        execution_window=expected_visibility_contract["execution_window"],
        **_target_visibility_heading_kwargs(visibility_contract),
    )
    if not visibility_passed:
        observed = [item.visibility_bucket for item in visibility_evidence]
        raise PrivateEvaluatorManifestError(
            "private targets do not realize the declared geometric visibility bucket: "
            f"{observed}"
        )
    return {
        "target_count": len(targets),
        "minimum_route_separation_m": float(minimum_route_distance),
        "minimum_pairwise_separation_m": float(minimum_pairwise_distance),
        "target_region_id": target_region_id,
        "visibility_bucket": str(visibility_contract["visibility_bucket"]),
        "minimum_visible_witness_count": min(
            item.visible_witness_count for item in visibility_evidence
        ),
        "maximum_blocked_witness_count": max(
            item.blocked_witness_count for item in visibility_evidence
        ),
        "minimum_maximum_projected_instance_pixels": min(
            item.maximum_projected_instance_pixels for item in visibility_evidence
        ),
    }


def _capture_target_visibility_execution_window(
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    """Return the retained-sensor window implied by one capture command."""

    try:
        return target_visibility_execution_window(
            dt_s=float(args.dt),
            warmup_steps=int(args.warmup_steps),
            rollout_steps=int(args.steps),
            capture_stride=int(args.capture_stride),
            waypoint_segment_seconds=_native_t2_waypoint_segment_seconds(args),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise PrivateEvaluatorManifestError(
            "capture arguments cannot form a valid target-visibility execution window"
        ) from exc


def _native_t2_waypoint_segment_seconds(args: argparse.Namespace) -> float:
    """Use the protocol-bound route clock, legacy constant otherwise."""

    motion = getattr(args, "native_t2_motion_contract", None)
    if isinstance(motion, Mapping):
        try:
            value = float(motion["waypoint_segment_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("native T2 motion contract lacks waypoint_segment_seconds") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("native T2 waypoint_segment_seconds must be finite and positive")
        return value
    return WAYPOINT_SEGMENT_SECONDS


def _native_t2_task_variant_id(args: argparse.Namespace) -> str:
    """Return the preflight-bound task variant; never infer it from defaults."""

    task_variant_id = getattr(args, "native_t2_task_variant_id", None)
    if task_variant_id not in (NATIVE_T2_V2_TASK_VARIANT_ID, NATIVE_T2_V3_TASK_VARIANT_ID):
        raise RuntimeError("native T2 canary requires a preflight-bound task variant")
    return str(task_variant_id)


def validate_private_target_execution_window(
    manifest: Mapping[str, Any],
    *,
    execution_window: Mapping[str, Any],
) -> None:
    """Reject a target manifest that was sampled for another rollout window.

    The full target geometry test runs after the native City-Lite stage is
    composed.  This smaller check runs before AppLauncher so a harmless-looking
    command-line duration change cannot allocate Isaac and then sample only a
    prefix of the manifest's promised witness route.
    """

    visibility_contract = manifest.get("target_visibility_contract")
    if not isinstance(visibility_contract, Mapping):
        raise PrivateEvaluatorManifestError("private target visibility contract is missing")
    declared_window = visibility_contract.get("execution_window")
    if not isinstance(declared_window, Mapping):
        raise PrivateEvaluatorManifestError(
            "private target visibility contract has no execution window"
        )
    if dict(declared_window) != dict(execution_window):
        raise PrivateEvaluatorManifestError(
            "private target visibility execution window does not match capture arguments"
        )


def _json_compatible(value: Any) -> Any:
    """Convert Replicator metadata into strict JSON-compatible values."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_compatible(value.tolist())
    return str(value)


def _camera_semantic_metadata(camera: Any) -> dict[str, Any]:
    """Return non-synthetic semantic metadata with a stable JSON-object ABI."""

    info = getattr(getattr(camera, "data", None), "info", None)
    if isinstance(info, Mapping):
        metadata = _json_compatible(info.get("semantic_segmentation", {}))
        return dict(metadata) if isinstance(metadata, Mapping) and metadata else {}
    if isinstance(info, (list, tuple)):
        per_camera: list[dict[str, Any]] = []
        for entry in info:
            metadata = _json_compatible(
                entry.get("semantic_segmentation", {}) if isinstance(entry, Mapping) else {}
            )
            per_camera.append(dict(metadata) if isinstance(metadata, Mapping) else {})
        return {"per_camera": per_camera} if any(per_camera) else {}
    return {}


def _onboard_semantic_metadata(onboard: Any) -> Any:
    """Compatibility wrapper for the batched onboard camera metadata."""

    return _camera_semantic_metadata(onboard)


def _overview_semantic_metadata(overview: Any) -> Any:
    """Return metadata for the singleton public overview render product."""

    return _camera_semantic_metadata(overview)


SEMANTIC_METADATA_SCHEMA = "org.rivermark.isaac-semantic-metadata.v2"
SEMANTIC_FRAME_METADATA_SCHEMA = "org.rivermark.isaac-semantic-frame-metadata.v1"
SEMANTIC_FRAME_METADATA_RELATIVE_PATH = "learning_labels/semantic_frame_metadata.jsonl"


def _public_semantic_id_mapping(
    metadata: Any, *, private_target_ids: Sequence[str] = ()
) -> dict[str, Any]:
    """Project Replicator metadata to the public per-render semantic ID ABI.

    Replicator's numeric semantic IDs belong to a render product and can be
    reassigned after a camera update.  The capture artifact therefore stores a
    freshly projected mapping for every retained frame.  Restricting it to ID,
    class, and public CF2X identity prevents USD paths, transforms, arbitrary
    annotator attributes, and evaluator-private target identifiers from
    entering a public capture directory.
    """

    redacted = _redact_private_target_metadata(
        metadata, private_target_ids=private_target_ids
    )

    def project_one(value: Any) -> dict[str, Any]:
        labels = (
            value.get("id_to_labels", value.get("idToLabels"))
            if isinstance(value, Mapping)
            else None
        )
        if not isinstance(labels, Mapping):
            return {"id_to_labels": {}}
        projected: dict[str, dict[str, str]] = {}
        for raw_id, raw_labels in labels.items():
            try:
                semantic_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if semantic_id < 0:
                continue
            if isinstance(raw_labels, Mapping):
                class_value = raw_labels.get("class", "")
                agent_id = raw_labels.get("agent_id")
            else:
                class_value = raw_labels
                agent_id = None
            item: dict[str, str] = {"class": str(class_value)}
            if isinstance(agent_id, (str, int)) and not isinstance(agent_id, bool):
                try:
                    agent_index = int(agent_id)
                except (TypeError, ValueError):
                    agent_index = -1
                if 0 <= agent_index < AGENT_COUNT:
                    item["agent_id"] = str(agent_index)
            projected[str(semantic_id)] = item
        return {
            "id_to_labels": {
                key: projected[key]
                for key in sorted(projected, key=lambda item: int(item))
            }
        }

    if isinstance(redacted, Mapping) and isinstance(
        redacted.get("per_camera"), (list, tuple)
    ):
        return {
            "per_camera": [
                project_one(item) for item in redacted["per_camera"]
            ]
        }
    return project_one(redacted)


def _semantic_frame_metadata_record(
    *,
    frame_index: int,
    timestamp_ns: int,
    onboard_metadata: Any,
    overview_metadata: Any,
    private_target_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the immutable public semantic mapping bound to one raw frame."""

    return {
        "schema": SEMANTIC_FRAME_METADATA_SCHEMA,
        "frame_index": int(frame_index),
        "timestamp_ns": int(timestamp_ns),
        "onboard_replicator_info": _public_semantic_id_mapping(
            onboard_metadata, private_target_ids=private_target_ids
        ),
        "overview_replicator_info": _public_semantic_id_mapping(
            overview_metadata, private_target_ids=private_target_ids
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _atomic_write_text(path: Path, content: str, *, encoding: str) -> None:
    """Replace one control-plane file without exposing a partially written value."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _checkpoint(output_dir: Path, stage: str, **details: object) -> None:
    """Persist the last completed native-capture stage for crash diagnosis."""

    _write_json(
        output_dir / "capture_progress.json",
        {"schema": "org.rivermark.isaac-capture-progress.v1", "stage": stage, "wall_time_ns": time.time_ns(), **details},
    )


def _persist_receipt_snapshot(output_dir: Path, receipt: Mapping[str, Any]) -> None:
    """Persist a receipt and its checksum before a Kit process can be created."""

    receipt_path = output_dir / "capture_receipt.json"
    _atomic_write_text(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _atomic_write_text(
        output_dir / "capture_receipt.sha256",
        f"{_sha256(receipt_path)}  capture_receipt.json\n",
        encoding="ascii",
    )


def _persist_terminal_capture_state(output_dir: Path, receipt: dict[str, Any]) -> None:
    """Durably record a terminal receipt and its public denominator entry.

    Isaac/Kit shutdown can terminate the interpreter from ``app.close()``. A
    terminal receipt must therefore reach the append-only failure ledger before
    ``_capture`` enters its resource-closing ``finally`` block. The ledger
    append is idempotent and deliberately best-effort: receipt durability is
    never downgraded if a separate control-plane write temporarily fails, and
    ``recover_crash_left_attempts`` can reconcile that explicit residue.
    """

    receipt["artifact_hashes"] = _artifact_hashes(output_dir)
    _persist_receipt_snapshot(output_dir, receipt)
    try:
        _record_raw_capture_attempt(output_dir, receipt)
    except Exception as error:  # noqa: BLE001 - independent ledger I/O is recoverable.
        # Do not convert successfully captured physical evidence into a false
        # capture failure because its separately recoverable ledger write failed.
        print(f"[WARN] raw capture ledger append skipped: {error}", file=sys.stderr)


def _write_capture_start_marker(output_dir: Path, receipt: Mapping[str, Any]) -> str:
    """Persist the crash-recovery start event before any risky runtime work."""

    created = int(receipt["created_wall_time_ns"])
    source_revision = str(receipt["source_revision"])
    attempt_id = "attempt-" + hashlib.sha256(
        f"{output_dir.name}:{created}:{source_revision}".encode("utf-8")
    ).hexdigest()[:32]
    marker: dict[str, Any] = {
        "schema": CAPTURE_START_SCHEMA,
        "attempt_id": attempt_id,
        "started_wall_time_ns": created,
        "source_revision": source_revision,
        "source_tree_sha256": str(receipt["source_tree_sha256"]),
        "source_worktree_dirty": bool(receipt["source_worktree_dirty"]),
        "task_kind": str(receipt["task_kind"]),
        "control_mode": str(receipt["command"]["control_mode"]),
        "agent_count_requested": int(receipt["agent_count_requested"]),
    }
    binding = receipt.get("collection_binding")
    if isinstance(binding, Mapping):
        marker["collection_binding"] = dict(binding)
    _write_json(output_dir / "capture_start.json", marker)
    return attempt_id


def _resolve_collection_binding(args: argparse.Namespace) -> dict[str, Any] | None:
    """Load and resolve the public protocol without retaining its local path."""

    if args.collection_protocol is None:
        return None
    try:
        protocol = load_collection_protocol(args.collection_protocol.expanduser().resolve())
        if (
            args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY
            and not is_native_t2_canary_protocol(protocol)
        ):
            raise CollectionProtocolError(
                "native T2 canary requires the dedicated native T2 canary protocol"
            )
        if (
            args.control_mode == CONTROL_MODE_FIXED_PUBLIC_ROUTE
            and is_native_t2_canary_protocol(protocol)
        ):
            raise CollectionProtocolError(
                "development native T2 protocol cannot bind fixed_public_route capture"
            )
        return resolve_collection_binding(
            protocol,
            cell_id=str(args.collection_cell_id),
            episode_index=int(args.collection_episode_index),
        )
    except (OSError, CollectionProtocolError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"collection protocol binding rejected: {exc}") from exc


def _native_t2_motion_contract_for_capture(
    args: argparse.Namespace, *, collection_binding: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Bind versioned motion values and route feasibility before AppLauncher starts.

    The old T2 canary had no public binding from its command limits to private
    target placement.  It is retained as failure evidence, but a new native
    launch must use v2 so the camera-witness schedule, action envelope and
    route timing are one auditable contract.
    """

    if args.control_mode != CONTROL_MODE_NATIVE_T2_CANARY:
        return None
    if args.collection_protocol is None or collection_binding is None:
        raise ValueError("native T2 motion contract requires a collection protocol binding")
    try:
        protocol = load_collection_protocol(args.collection_protocol.expanduser().resolve())
        if not is_native_t2_canary_protocol(protocol):
            raise CollectionProtocolError("native T2 requires a dedicated canary protocol")
        motion = native_t2_motion_contract(protocol)
        if motion is None:
            raise CollectionProtocolError(
                "native T2 canary v1 has no route-coupled motion contract and is retained only as historical evidence"
            )
        task_variant_by_schema = {
            NATIVE_T2_CANARY_V2_PROTOCOL_SCHEMA: NATIVE_T2_V2_TASK_VARIANT_ID,
            NATIVE_T2_CANARY_V3_PROTOCOL_SCHEMA: NATIVE_T2_V3_TASK_VARIANT_ID,
        }
        task_variant_id = task_variant_by_schema.get(protocol.get("schema"))
        if task_variant_id is None:
            raise CollectionProtocolError(
                "native T2 motion contract requires a revisioned v2-or-later canary schema"
            )
        cell = next(
            item
            for item in protocol["cells"]
            if item.get("cell_id") == collection_binding.get("cell_id")
        )
        conditions = cell["conditions"]
        routes = resolve_public_route_family(str(conditions["route_family"]))
    except (OSError, CollectionProtocolError, StopIteration, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"native T2 motion contract rejected: {exc}") from exc

    expected_arguments: tuple[tuple[str, str], ...] = (
        ("dt", "dt_s"),
        ("warmup_steps", "warmup_steps"),
        ("steps", "rollout_steps"),
        ("capture_stride", "capture_stride"),
        ("native_t2_decision_stride", "decision_stride_physics_steps"),
        ("native_t2_max_horizontal_speed_mps", "max_horizontal_speed_mps"),
        ("native_t2_max_vertical_speed_mps", "max_vertical_speed_mps"),
        ("native_t2_max_yaw_rate_radps", "max_yaw_rate_rad_s"),
    )
    for argument_name, motion_name in expected_arguments:
        actual = getattr(args, argument_name)
        expected = motion[motion_name]
        if isinstance(expected, int):
            matches = isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
        else:
            matches = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12)
        if not matches:
            raise ValueError(
                "native T2 command argument does not match frozen v2 motion contract: "
                f"--{argument_name.replace('_', '-')}={actual!r}, expected {expected!r}"
            )
    try:
        feasibility = validate_route_timing_feasibility(
            routes,
            waypoint_segment_seconds=float(motion["waypoint_segment_seconds"]),
            max_horizontal_speed_mps=float(motion["max_horizontal_speed_mps"]),
            max_vertical_speed_mps=float(motion["max_vertical_speed_mps"]),
            utilization_limit=float(motion["route_speed_utilization_limit"]),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"native T2 route timing is infeasible before AppLauncher: {exc}") from exc
    return {
        "motion_contract": motion,
        "route_timing_feasibility": feasibility,
        "task_variant_id": task_variant_id,
    }


def _resolve_condition_request(
    args: argparse.Namespace, binding: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Copy the public cell conditions into the existing capture receipt."""

    if args.collection_protocol is None or binding is None:
        return None
    try:
        protocol = load_collection_protocol(args.collection_protocol.expanduser().resolve())
        return condition_request_from_protocol(
            protocol,
            protocol_id=str(binding["protocol_id"]),
            protocol_sha256=str(binding["protocol_sha256"]),
            cell_id=str(binding["cell_id"]),
        )
    except (OSError, CollectionProtocolError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"collection condition request rejected: {exc}") from exc


def _bind_sensor_physics_smoke_receipt(
    args: argparse.Namespace,
    output_dir: Path,
    receipt: dict[str, Any],
    runtime_lock: Mapping[str, Any],
) -> None:
    """Validate and commit to a pre-existing full native smoke receipt.

    The smoke remains outside the capture ownership boundary. Reading its
    bytes once prevents a time-of-check/time-of-use mismatch between semantic
    validation and the SHA-256 committed by the raw capture receipt.
    """

    configured = args.sensor_physics_smoke_receipt
    if configured is None:
        return
    path = configured.expanduser().resolve()
    if _is_within(path, output_dir):
        raise SensorPhysicsSmokeReceiptError(
            "--sensor-physics-smoke-receipt must remain outside --output-dir"
        )
    if path.name != "isaac_smoke_receipt.json" or not path.is_file():
        raise SensorPhysicsSmokeReceiptError(
            "--sensor-physics-smoke-receipt must name an existing "
            "isaac_smoke_receipt.json"
        )
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke SHA-256 sidecar is missing"
        )
    payload_bytes = path.read_bytes()
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    expected_sidecar = f"{payload_sha256}  {path.name}\n"
    try:
        observed_sidecar = sidecar.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke SHA-256 sidecar is not ASCII"
        ) from exc
    if observed_sidecar != expected_sidecar:
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke receipt was modified after its SHA-256 sidecar"
        )
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke receipt is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke receipt must be a JSON object"
        )

    # Local import avoids a module cycle: isaac_smoke deliberately reuses the
    # native capture constructors from this module.
    from .isaac_smoke import validate_smoke_receipt
    from .runtime_lock import runtime_lock_sha256

    errors = validate_smoke_receipt(payload, runtime_lock=runtime_lock)
    if errors:
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke receipt failed validation: " + "; ".join(errors)
        )
    if payload.get("status") != "passed" or payload.get("resource_probe_profile") != "full":
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke receipt must be a passed full-profile smoke"
        )

    source = payload.get("source")
    if not isinstance(source, Mapping) or source.get("source_worktree_dirty") is not False:
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke receipt is not bound to a clean source tree"
        )
    if receipt.get("source_worktree_dirty") is not False:
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke can bind only a clean capture source tree"
        )
    if (
        source.get("source_revision") != receipt.get("source_revision")
        or source.get("source_tree_sha256") != receipt.get("source_tree_sha256")
    ):
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke source revision/tree does not match this capture"
        )

    lock_sha256 = runtime_lock_sha256(runtime_lock)
    profile_id = runtime_lock.get("profile_id")
    if (
        payload.get("runtime_lock_sha256") != lock_sha256
        or payload.get("runtime_profile_id") != profile_id
    ):
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke runtime lock/profile does not match this capture"
        )
    expected_assets = runtime_lock.get("assets")
    runtime_audit = payload.get("runtime_audit")
    observed = runtime_audit.get("observed") if isinstance(runtime_audit, Mapping) else None
    observed_assets = observed.get("assets") if isinstance(observed, Mapping) else None
    if (
        not isinstance(expected_assets, Mapping)
        or not isinstance(observed_assets, Mapping)
        or dict(observed_assets) != dict(expected_assets)
    ):
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke asset hashes do not match the runtime lock"
        )
    scene = payload.get("scene")
    if (
        not isinstance(scene, Mapping)
        or scene.get("contract_sha256")
        != expected_assets.get("city_lite_contract_sha256")
    ):
        raise SensorPhysicsSmokeReceiptError(
            "sensor-physics smoke City-Lite contract does not match this capture"
        )

    build = f"isaaclab:{profile_id}@sha256:{lock_sha256}"
    receipt["capture_backend"] = {
        "kind": "isaaclab",
        "build": build,
        "sensor_physics_smoke_receipt_sha256": payload_sha256,
    }


def _run_capture_preflight(
    args: argparse.Namespace,
    output_dir: Path,
    receipt: dict[str, Any],
) -> None:
    """Run all Isaac-free gates and persist their result before AppLauncher."""

    from .preflight import RuntimePreflightRequirements, run_preflight

    storage_budget = _capture_storage_budget(args)
    declared_capture_bytes = int(float(args.estimated_capture_gib) * 1024**3)
    if declared_capture_bytes < storage_budget.required_bytes:
        receipt["capture_storage_budget"] = storage_budget.as_dict()
        raise RuntimeError(
            "declared --estimated-capture-gib is below the derived capture "
            f"storage reservation ({storage_budget.required_bytes} bytes)"
        )
    receipt["capture_storage_budget"] = storage_budget.as_dict()
    runtime_lock = _bind_runtime_lock_to_args(args, receipt)
    device_name = str(args.device).casefold()
    gpu_requested = bool(args.require_gpu) or device_name.startswith(("cuda", "gpu"))
    minimum_vram_bytes = (
        int(float(args.minimum_gpu_vram_gib) * 1024**3) if gpu_requested else 0
    )
    minimum_driver_version = (
        args.minimum_driver_version
        if args.minimum_driver_version is not None
        else ("545.0" if gpu_requested else None)
    )
    runtime = RuntimePreflightRequirements(
        require_gpu=gpu_requested,
        minimum_gpu_vram_bytes=minimum_vram_bytes,
        minimum_driver_version=minimum_driver_version,
        isaac_sim_version=args.isaac_sim_version,
        isaaclab_version=args.isaaclab_version,
        scene_contract=args.scene_contract,
        scene_contract_sha256=SCENE_CONTRACT_SHA256,
        runtime_lock=args.runtime_lock,
        isaaclab_source=args.isaaclab_source,
        cf2x_usd=args.drone_usd,
    )
    _enforce_foreign_native_process_guard(
        args,
        receipt,
        phase="preflight",
        output_dir=output_dir,
    )
    report = run_preflight(
        output_dir=output_dir,
        minimum_free_bytes=int(float(args.minimum_free_gib) * 1024**3),
        estimated_capture_bytes=declared_capture_bytes,
        source_root=_repository_root(),
        required_assets=(
            (args.drone_usd.expanduser().resolve(), _sha256(args.drone_usd.expanduser().resolve())),
        ),
        require_clean=not args.allow_dirty_source,
        runtime=runtime,
    )
    report_payload = report.as_dict()
    receipt["preflight"] = report_payload
    receipt["command"]["preflight"] = {
        "minimum_free_gib": float(args.minimum_free_gib),
        "estimated_capture_gib": float(args.estimated_capture_gib),
        "derived_storage_required_bytes": storage_budget.required_bytes,
        "gpu_requested": gpu_requested,
        "minimum_gpu_vram_gib": float(args.minimum_gpu_vram_gib) if gpu_requested else 0.0,
        "minimum_driver_version": minimum_driver_version,
        "clean_source_required": not args.allow_dirty_source,
        "runtime_lock_required": args.collection_protocol is not None,
        "runtime_lock_configured": args.runtime_lock is not None,
    }
    _checkpoint(output_dir, "preflight_completed", valid=report.valid, report=report_payload)
    _persist_receipt_snapshot(output_dir, receipt)
    if not report.valid:
        failed = [check["name"] for check in report_payload["checks"] if not check["passed"]]
        raise RuntimeError(
            "Isaac capture preflight rejected the launch: "
            + ", ".join(failed)
        )
    if runtime_lock is not None:
        # Keep the command namespace canonical for every downstream helper.
        # The values were already compared above; assigning the normalized
        # representation prevents float/string formatting drift in receipts.
        args.dt = float(runtime_lock["simulation"]["dt_s"])
        args.device = str(runtime_lock["simulation"]["device"])
    if args.sensor_physics_smoke_receipt is not None:
        if runtime_lock is None:
            raise RuntimeError("sensor-physics smoke binding requires a runtime lock")
        try:
            _bind_sensor_physics_smoke_receipt(
                args,
                output_dir,
                receipt,
                runtime_lock,
            )
        except (OSError, SensorPhysicsSmokeReceiptError) as exc:
            _checkpoint(
                output_dir,
                "sensor_physics_smoke_rejected",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            _persist_receipt_snapshot(output_dir, receipt)
            raise
        _checkpoint(
            output_dir,
            "sensor_physics_smoke_validated",
            capture_backend=receipt["capture_backend"],
        )
        _persist_receipt_snapshot(output_dir, receipt)


def _artifact_hashes(root: Path) -> dict[str, dict[str, Any]]:
    excluded = {"capture_receipt.json", "capture_receipt.sha256"}
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            relative = path.relative_to(root).as_posix()
            result[relative] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _failure_ledger_classification(receipt: Mapping[str, Any]) -> tuple[str, str, str]:
    """Classify one terminal receipt without treating capacity guards as physics failures."""

    if receipt.get("status") == "captured":
        return "quarantined", "quality_failure", "development_evidence_not_formal"

    infrastructure_guards = (
        ("foreign_native_process_guard", "foreign_native_process_guard_rejected"),
        ("system_commit_guard", "system_commit_guard_rejected"),
    )
    for guard_key, reason_code in infrastructure_guards:
        guard = receipt.get(guard_key)
        if isinstance(guard, Mapping) and guard.get("status") == "rejected":
            return "failed", "infrastructure_failure", reason_code

    failure = receipt.get("failure")
    failure_type = failure.get("type") if isinstance(failure, Mapping) else None
    reason_code = (
        str(failure_type).lower()
        if isinstance(failure_type, str) and failure_type
        else "capture_not_completed"
    )
    return "failed", "capture_failure", reason_code


def _record_raw_capture_attempt(output_dir: Path, receipt: Mapping[str, Any]) -> None:
    """Record a public raw-capture denominator without touching formal index data.

    The automatic path is deliberately restricted to a ``rivermark-runs``
    sibling root.  Unit tests and ad-hoc runs outside that root remain isolated;
    production capture attempts get one path-free ledger record after the final
    receipt hash is known.
    """

    if output_dir.parent.name.lower() != "rivermark-runs":
        return
    receipt_path = output_dir / "capture_receipt.json"
    if not receipt_path.is_file():
        return
    receipt_hash = _sha256(receipt_path)
    outcome, category, reason_code = _failure_ledger_classification(receipt)
    start_marker = output_dir / "capture_start.json"
    attempt_id: str | None = None
    try:
        marker = json.loads(start_marker.read_text(encoding="utf-8"))
        if isinstance(marker, Mapping) and isinstance(marker.get("attempt_id"), str):
            attempt_id = marker["attempt_id"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    if attempt_id is None:
        # Compatibility for pre-recovery pilot runs that predate the marker.
        attempt_id = "attempt-" + hashlib.sha256(
            f"{output_dir.name}:{receipt_hash}".encode("utf-8")
        ).hexdigest()[:32]
    binding = receipt.get("collection_binding")
    binding_kwargs: dict[str, Any] = {}
    if isinstance(binding, Mapping):
        binding_kwargs = {
            "collection_protocol_id": binding.get("protocol_id"),
            "collection_protocol_sha256": binding.get("protocol_sha256"),
            "collection_cell_id": binding.get("cell_id"),
            "collection_episode_index": binding.get("episode_index"),
            "episode_seed": binding.get("episode_seed"),
        }
        split = binding.get("split")
    else:
        split = None
    append_failure_record_once(
        output_dir.parent / "failure_ledger.jsonl",
        FailureRecord(
            attempt_id=attempt_id,
            outcome=outcome,
            category=category,
            stage="isaac_capture",
            recorded_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            split=split if isinstance(split, str) else "pilot",
            source_capture_sha256=receipt_hash,
            receipt_sha256=receipt_hash,
            reason_code=reason_code,
            **binding_kwargs,
        ),
    )


def _to_numpy(value: Any) -> Any:
    """Copy Tensor and array sensor values without assuming one backend."""

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, "copy"):
        return value.copy()
    return value


@dataclass(frozen=True)
class _RouteExecutionProfile:
    route_family_id: str
    start_anchor_id: str
    target_region_id: str
    visibility_bucket: str
    routes_w_m: tuple[tuple[tuple[float, float, float], ...], ...]


def _route_execution_profile(receipt: Mapping[str, Any] | None = None) -> _RouteExecutionProfile:
    """Resolve executable holdout conditions, never labels without an executor."""

    conditions: Mapping[str, Any] = {}
    if isinstance(receipt, Mapping):
        request = receipt.get("condition_request")
        if isinstance(request, Mapping) and isinstance(request.get("conditions"), Mapping):
            conditions = request["conditions"]
    route_family_id = str(conditions.get("route_family", CITY_LITE_ROUTE_FAMILY_A_ID))
    routes = resolve_public_route_family(route_family_id)
    expected_geometry = {
        CITY_LITE_ROUTE_FAMILY_A_ID: (
            START_ANCHOR_IDS_BY_ROUTE_FAMILY[CITY_LITE_ROUTE_FAMILY_A_ID],
            CITY_LITE_TARGET_REGION_B_ID,
        ),
        CITY_LITE_ROUTE_FAMILY_B_ID: (
            START_ANCHOR_IDS_BY_ROUTE_FAMILY[CITY_LITE_ROUTE_FAMILY_B_ID],
            CITY_LITE_TARGET_REGION_A_ID,
        ),
    }[route_family_id]
    requested_geometry = (
        conditions.get("start_anchor", expected_geometry[0]),
        conditions.get("target_region", expected_geometry[1]),
    )
    if requested_geometry != expected_geometry:
        raise ValueError(
            "collection holdout conditions do not match an executable City-Lite route profile"
        )
    visibility_bucket = conditions.get("visibility_bucket", "direct-visible-v1")
    if visibility_bucket not in TARGET_VISIBILITY_BUCKETS:
        raise ValueError("collection visibility condition is not executable")
    if conditions and conditions.get("route") != "fixed-public-route-v1":
        raise ValueError("collection route condition is not executable by fixed_public_route")
    return _RouteExecutionProfile(
        route_family_id=route_family_id,
        start_anchor_id=expected_geometry[0],
        target_region_id=expected_geometry[1],
        visibility_bucket=str(visibility_bucket),
        routes_w_m=routes,
    )


def _city_lite_spawn_states(
    routes_w_m: Sequence[Sequence[Sequence[float]]] = PUBLIC_ROUTES_W_M,
) -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float, float]], ...
]:
    """Return task-owned literal initial states for the public route anchors."""

    states: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]] = []
    for route, yaw_rad in zip(
        routes_w_m, _initial_route_heading_yaws_rad(routes_w_m), strict=True
    ):
        states.append(
            (
                tuple(float(value) for value in route[0]),
                (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)),
            )
        )
    if len(states) != AGENT_COUNT:
        raise RuntimeError("City-Lite literal CF2X spawn count must remain eight")
    return tuple(states)


def _city_lite_initial_root_poses(
    torch: Any,
    device: str,
    routes_w_m: Sequence[Sequence[Sequence[float]]] = PUBLIC_ROUTES_W_M,
) -> Any:
    """Build a receipt-only view of states authored by literal spawn configs."""

    poses = torch.zeros((AGENT_COUNT, 7), dtype=torch.float32, device=device)
    for agent_id, (position, quaternion) in enumerate(_city_lite_spawn_states(routes_w_m)):
        poses[agent_id, :3] = torch.tensor(position, dtype=torch.float32, device=device)
        poses[agent_id, 3:] = torch.tensor(quaternion, dtype=torch.float32, device=device)
    return poses


def _city_lite_initial_root_states(
    torch: Any,
    device: str,
    routes_w_m: Sequence[Sequence[Sequence[float]]] = PUBLIC_ROUTES_W_M,
) -> Any:
    """Return the complete literal CF2X default root state expected by IsaacLab.

    ``AssetBase`` authors ``init_state.pos`` and ``init_state.rot`` into the
    USD spawn transform.  The remaining velocity fields are resolved into
    ``Multirotor.data.default_root_state`` and must be zero before the first
    policy command.  This helper makes both parts of that contract explicit.
    """

    states = torch.zeros((AGENT_COUNT, 13), dtype=torch.float32, device=device)
    states[:, :7] = _city_lite_initial_root_poses(torch, device, routes_w_m)
    return states


def _city_lite_initial_thruster_rps(torch: Any, device: str) -> Any:
    """Return the resolved four-rotor hover trim required at every reset."""

    return torch.full(
        (AGENT_COUNT, len(THRUSTER_NAMES)),
        float(INITIAL_HOVER_RPS),
        dtype=torch.float32,
        device=device,
    )


def _initial_route_heading_yaws_rad(
    routes_w_m: Sequence[Sequence[Sequence[float]]] = PUBLIC_ROUTES_W_M,
) -> tuple[float, ...]:
    """Face every CF2X along its first public horizontal route segment."""

    headings: list[float] = []
    for route in routes_w_m:
        for start, end in zip(route, route[1:]):
            delta_x = float(end[0]) - float(start[0])
            delta_y = float(end[1]) - float(start[1])
            if math.hypot(delta_x, delta_y) > 1.0e-6:
                headings.append(math.atan2(delta_y, delta_x))
                break
        else:
            raise RuntimeError("every public City-Lite route requires a horizontal segment")
    if len(headings) != AGENT_COUNT:
        raise RuntimeError("public route count does not match the eight-agent CF2X formation")
    return tuple(headings)


def _axis_volume_payload(box: AABB) -> dict[str, list[float]]:
    """Represent one frozen world-frame volume in the public scene receipt."""

    return {
        "x": [box.minimum[0], box.maximum[0]],
        "y": [box.minimum[1], box.maximum[1]],
        "z": [box.minimum[2], box.maximum[2]],
    }


def _quat_rotate(quat_wxyz: Any, vector: Any, torch: Any) -> Any:
    xyz = quat_wxyz[:, 1:]
    cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quat_wxyz[:, :1] * cross + torch.cross(xyz, cross, dim=-1)


def _quat_multiply(left: Any, right: Any, torch: Any) -> Any:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _expected_onboard_camera_world_poses_from_parent(
    parent_pos_w: Any, parent_quat_w: Any, torch: Any
) -> tuple[Any, Any]:
    """Compose the native camera mount from one live parent-frame pose batch."""

    body_pos = parent_pos_w.detach()
    body_quat = parent_quat_w.detach()
    if body_pos.shape != (AGENT_COUNT, 3) or body_quat.shape != (AGENT_COUNT, 4):
        raise RuntimeError("onboard camera parent pose must be [eight, xyz/wxyz]")
    offset = torch.tensor(CAMERA_OFFSET_BODY_M, device=body_pos.device, dtype=body_pos.dtype).repeat(AGENT_COUNT, 1)
    mount_quat = torch.tensor(CAMERA_OFFSET_WXYZ, device=body_pos.device, dtype=body_pos.dtype).repeat(AGENT_COUNT, 1)
    expected_pos = body_pos + _quat_rotate(body_quat, offset, torch)
    expected_quat = _quat_multiply(body_quat, mount_quat, torch)
    norms = torch.linalg.vector_norm(expected_quat, dim=-1, keepdim=True)
    if not bool(torch.isfinite(expected_pos).all()) or not bool(torch.isfinite(norms).all()) or bool((norms <= 1.0e-8).any()):
        raise RuntimeError("live CF2X state cannot produce a finite onboard-camera USD pose")
    return expected_pos, expected_quat / norms


def _expected_onboard_camera_world_poses(robot: Any, torch: Any) -> tuple[Any, Any]:
    """Derive fixed-extrinsic camera world poses from the legacy root-link view."""

    return _expected_onboard_camera_world_poses_from_parent(
        robot.data.root_pos_w, robot.data.root_quat_w, torch
    )


def _literal_onboard_camera_parent_link_poses(robot: Any, torch: Any) -> tuple[Any, Any, int]:
    """Read the exact ``Robot/body`` link which parents each onboard camera."""

    body_names = list(getattr(robot, "body_names", ()))
    if body_names.count("body") != 1:
        raise RuntimeError("CF2X fleet must expose exactly one literal body parent link")
    body_index = body_names.index("body")
    positions = robot.data.body_link_pos_w.detach()
    quaternions = robot.data.body_link_quat_w.detach()
    if (
        positions.ndim != 3
        or quaternions.ndim != 3
        or positions.shape[0] != AGENT_COUNT
        or quaternions.shape[0] != AGENT_COUNT
        or positions.shape[1] != len(body_names)
        or quaternions.shape[1] != len(body_names)
        or positions.shape[2] != 3
        or quaternions.shape[2] != 4
    ):
        raise RuntimeError("CF2X body-link state does not match the shared literal body order")
    return positions[:, body_index], quaternions[:, body_index], body_index


def _onboard_camera_mount_diagnostics(
    robot: Any,
    camera: Any,
    *,
    root_expected_pos_w: Any,
    root_expected_quat_wxyz: Any,
    previous_root_expected_pos_w: Any | None,
    previous_root_expected_phase: str | None,
    torch: Any,
) -> dict[str, Any]:
    """Emit receipt-only alternatives for the native parent/timing closure audit.

    The data deliberately contains camera/body transforms only. It neither
    relaxes render-facing USD acceptance nor exposes targets, images, semantic
    payloads, or private evaluator information.
    """

    body_pos_w, body_quat_w, body_index = _literal_onboard_camera_parent_link_poses(robot, torch)
    body_expected_pos_w, body_expected_quat_wxyz = _expected_onboard_camera_world_poses_from_parent(
        body_pos_w, body_quat_w, torch
    )
    observed_pos_w = camera.data.pos_w.detach()
    observed_quat_wxyz = camera.data.quat_w_world.detach()
    if observed_pos_w.shape != (AGENT_COUNT, 3) or observed_quat_wxyz.shape != (AGENT_COUNT, 4):
        raise RuntimeError("onboard Camera Fabric pose does not contain eight xyz/wxyz values")
    if root_expected_pos_w.shape != (AGENT_COUNT, 3) or root_expected_quat_wxyz.shape != (AGENT_COUNT, 4):
        raise RuntimeError("root-derived onboard camera expectation does not contain eight xyz/wxyz values")
    if previous_root_expected_pos_w is None:
        if previous_root_expected_phase is not None:
            raise RuntimeError("previous root-derived camera phase requires a pose")
    else:
        if previous_root_expected_pos_w.shape != (AGENT_COUNT, 3):
            raise RuntimeError("previous root-derived onboard camera expectation has an invalid shape")
        if not isinstance(previous_root_expected_phase, str) or not previous_root_expected_phase:
            raise RuntimeError("previous root-derived camera pose requires an auditable phase")

    def _fabric_vector(value: Any) -> list[float | None]:
        return [
            float(component) if math.isfinite(float(component)) else None
            for component in value.detach().cpu().tolist()
        ]

    def _optional_finite(value: Any) -> float | None:
        result = float(value)
        return result if math.isfinite(result) else None

    root_residual = observed_pos_w - root_expected_pos_w
    body_residual = observed_pos_w - body_expected_pos_w
    root_body_delta = root_expected_pos_w - body_expected_pos_w
    rows: list[dict[str, Any]] = []
    root_residual_norms: list[float | None] = []
    body_residual_norms: list[float | None] = []
    for agent_id in range(AGENT_COUNT):
        root_residual_norm = _optional_finite(
            torch.linalg.vector_norm(root_residual[agent_id]).item()
        )
        body_residual_norm = _optional_finite(
            torch.linalg.vector_norm(body_residual[agent_id]).item()
        )
        row: dict[str, Any] = {
            "agent_id": agent_id,
            "root_expected_pos_w_m": root_expected_pos_w[agent_id].detach().cpu().tolist(),
            "root_expected_quat_wxyz": root_expected_quat_wxyz[agent_id].detach().cpu().tolist(),
            "body_link_expected_pos_w_m": body_expected_pos_w[agent_id].detach().cpu().tolist(),
            "body_link_expected_quat_wxyz": body_expected_quat_wxyz[agent_id].detach().cpu().tolist(),
            "observed_camera_fabric_pos_w_m": _fabric_vector(observed_pos_w[agent_id]),
            "observed_camera_fabric_quat_wxyz": _fabric_vector(observed_quat_wxyz[agent_id]),
            "observed_minus_root_expected_m": _fabric_vector(root_residual[agent_id]),
            "observed_minus_body_link_expected_m": _fabric_vector(body_residual[agent_id]),
            "root_expected_minus_body_link_expected_m": root_body_delta[agent_id].detach().cpu().tolist(),
            "root_residual_norm_m": root_residual_norm,
            "body_link_residual_norm_m": body_residual_norm,
            "fabric_pose_finite": bool(
                torch.isfinite(observed_pos_w[agent_id]).all()
                and torch.isfinite(observed_quat_wxyz[agent_id]).all()
            ),
        }
        root_residual_norms.append(root_residual_norm)
        body_residual_norms.append(body_residual_norm)
        if previous_root_expected_pos_w is None:
            row["root_expected_step_delta_m"] = None
            row["observed_minus_previous_root_expected_m"] = None
            row["previous_root_residual_norm_m"] = None
        else:
            root_step_delta = root_expected_pos_w[agent_id] - previous_root_expected_pos_w[agent_id]
            previous_root_residual = observed_pos_w[agent_id] - previous_root_expected_pos_w[agent_id]
            previous_root_residual_norm = _optional_finite(
                torch.linalg.vector_norm(previous_root_residual).item()
            )
            row["previous_root_expected_phase"] = previous_root_expected_phase
            row["root_expected_step_delta_m"] = root_step_delta.detach().cpu().tolist()
            row["observed_minus_previous_root_expected_m"] = _fabric_vector(
                previous_root_residual
            )
            row["previous_root_residual_norm_m"] = previous_root_residual_norm
        rows.append(row)
    return {
        "schema": "org.rivermark.onboard-camera-mount-diagnostic.v1",
        "purpose": "root_link_vs_literal_body_parent_and_previous_step_fabric_closure",
        "literal_parent_link": {"name": "body", "index": body_index},
        "per_agent": rows,
        "maximum_root_residual_norm_m": (
            max(root_residual_norms) if all(value is not None for value in root_residual_norms) else None
        ),
        "maximum_body_link_residual_norm_m": (
            max(body_residual_norms) if all(value is not None for value in body_residual_norms) else None
        ),
    }


def _camera_pose_closure(robot: Any, camera: Any, torch: Any) -> dict[str, Any]:
    """Read Camera's Fabric cache for diagnostics only.

    ``Camera.data`` is populated through a Fabric view.  It is useful for
    detecting a delayed cache, but it is not the authority for a frame that
    was rendered by the USD camera hierarchy.  Callers that bind an RGB-D
    frame must use :func:`_camera_pose_closure_from_usd` after the matching
    render/read fence has completed.
    """

    expected_pos, expected_quat = _expected_onboard_camera_world_poses(robot, torch)
    observed_pos = camera.data.pos_w.detach()
    observed_quat = camera.data.quat_w_world.detach()
    position_error = torch.linalg.vector_norm(expected_pos - observed_pos, dim=-1)
    cosine_half = torch.abs(torch.sum(expected_quat * observed_quat, dim=-1)).clamp(0.0, 1.0)
    orientation_error = 2.0 * torch.acos(cosine_half)
    return {
        "expected_pos_w_m": expected_pos,
        "expected_quat_wxyz": expected_quat,
        "observed_pos_w_m": observed_pos,
        "observed_quat_wxyz": observed_quat,
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
    }


def _world_camera_quat_from_usd_axes(
    *,
    forward: Sequence[float],
    right: Sequence[float],
    up: Sequence[float],
) -> tuple[float, float, float, float]:
    """Convert USD/OpenGL camera axes to IsaacLab's ``world`` convention.

    USD Cameras use ``+X`` right, ``+Y`` up, and ``-Z`` forward.  IsaacLab's
    world convention instead uses ``+X`` forward, ``-Y`` right, and ``+Z``
    up.  The returned quaternion maps that latter local basis into world
    coordinates, matching ``CameraData.quat_w_world`` without reading its
    potentially delayed Fabric cache.
    """

    if any(len(vector) != 3 for vector in (forward, right, up)):
        raise RuntimeError("onboard USD camera axes must be three-dimensional")
    matrix = tuple(
        tuple(
            float(component)
            for component in (forward[axis], -right[axis], up[axis])
        )
        for axis in range(3)
    )
    if not all(math.isfinite(component) for row in matrix for component in row):
        raise RuntimeError("onboard USD camera axes must be finite")
    return _matrix_to_quat_wxyz(matrix)


def _camera_pose_closure_from_usd(
    usd_closure: Mapping[str, Any],
    expected_positions: Any,
    expected_world_quats: Any,
    torch: Any,
) -> dict[str, Any]:
    """Return the frame-authoritative camera closure from the USD hierarchy."""

    rows = usd_closure.get("per_agent")
    if not isinstance(rows, list) or len(rows) != AGENT_COUNT:
        raise RuntimeError("onboard USD camera closure must contain eight agent rows")
    try:
        observed_positions = torch.as_tensor(
            [row["observed_pos_w_m"] for row in rows],
            device=expected_positions.device,
            dtype=expected_positions.dtype,
        )
        observed_quats = torch.as_tensor(
            [row["observed_quat_wxyz"] for row in rows],
            device=expected_world_quats.device,
            dtype=expected_world_quats.dtype,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("onboard USD camera closure has no usable observed world pose") from error
    if (
        observed_positions.shape != (AGENT_COUNT, 3)
        or observed_quats.shape != (AGENT_COUNT, 4)
        or expected_positions.shape != (AGENT_COUNT, 3)
        or expected_world_quats.shape != (AGENT_COUNT, 4)
    ):
        raise RuntimeError("onboard camera USD closure pose shape is invalid")
    norms = torch.linalg.vector_norm(observed_quats, dim=-1, keepdim=True)
    if (
        not bool(torch.isfinite(observed_positions).all())
        or not bool(torch.isfinite(norms).all())
        or bool(torch.any(norms <= 1.0e-8).item())
    ):
        raise RuntimeError("onboard camera USD closure has a non-finite observed pose")
    observed_quats = observed_quats / norms
    expected_norms = torch.linalg.vector_norm(expected_world_quats, dim=-1, keepdim=True)
    if bool(torch.any(expected_norms <= 1.0e-8).item()):
        raise RuntimeError("onboard camera USD closure has a degenerate expected pose")
    expected_quats = expected_world_quats / expected_norms
    position_error = torch.linalg.vector_norm(
        expected_positions - observed_positions, dim=-1
    )
    cosine_half = torch.abs(
        torch.sum(expected_quats * observed_quats, dim=-1)
    ).clamp(0.0, 1.0)
    orientation_error = 2.0 * torch.acos(cosine_half)
    return {
        "expected_pos_w_m": expected_positions.detach(),
        "expected_quat_wxyz": expected_quats.detach(),
        "observed_pos_w_m": observed_positions.detach(),
        "observed_quat_wxyz": observed_quats.detach(),
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "authority": "render_facing_usd_hierarchy",
    }


def _onboard_camera_prim_path(agent_id: int) -> str:
    if not 0 <= int(agent_id) < AGENT_COUNT:
        raise RuntimeError(f"invalid onboard camera agent id: {agent_id}")
    return f"/World/Swarm/Agent_{int(agent_id)}/Robot/body/onboard_camera"


def _onboard_camera_usd_pose_closure(
    stage: Any, expected_positions: Any, expected_world_quats: Any, torch: Any
) -> dict[str, Any]:
    """Verify the complete render-facing USD camera pose for every CF2X."""

    from pxr import Gf, Usd, UsdGeom

    if int(expected_positions.shape[0]) != AGENT_COUNT or int(expected_world_quats.shape[0]) != AGENT_COUNT:
        raise RuntimeError("onboard USD closure requires eight expected camera poses")
    forward_body = torch.tensor(
        (1.0, 0.0, 0.0), device=expected_world_quats.device, dtype=expected_world_quats.dtype
    )
    # In the Camera ``world`` convention, OpenGL's +X/+Y/+Z map to
    # body-camera right/up/back = -Y/+Z/-X.  Checking all three axes catches
    # stale roll and pitch as well as an optical-axis error.
    right_body = torch.tensor(
        (0.0, -1.0, 0.0), device=expected_world_quats.device, dtype=expected_world_quats.dtype
    )
    up_body = torch.tensor(
        (0.0, 0.0, 1.0), device=expected_world_quats.device, dtype=expected_world_quats.dtype
    )
    back_body = torch.tensor(
        (-1.0, 0.0, 0.0), device=expected_world_quats.device, dtype=expected_world_quats.dtype
    )
    expected_forwards = _quat_rotate(
        expected_world_quats,
        forward_body.repeat(AGENT_COUNT, 1),
        torch,
    ).detach().cpu().tolist()
    expected_rights = _quat_rotate(
        expected_world_quats,
        right_body.repeat(AGENT_COUNT, 1),
        torch,
    ).detach().cpu().tolist()
    expected_ups = _quat_rotate(
        expected_world_quats,
        up_body.repeat(AGENT_COUNT, 1),
        torch,
    ).detach().cpu().tolist()
    expected_backs = _quat_rotate(
        expected_world_quats,
        back_body.repeat(AGENT_COUNT, 1),
        torch,
    ).detach().cpu().tolist()
    positions = expected_positions.detach().cpu().tolist()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rows: list[dict[str, Any]] = []
    for agent_id, (expected_position, expected_forward, expected_right, expected_up, expected_back) in enumerate(
        zip(positions, expected_forwards, expected_rights, expected_ups, expected_backs, strict=True)
    ):
        prim = stage.GetPrimAtPath(_onboard_camera_prim_path(agent_id))
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"onboard camera prim is not a USD Camera: {prim.GetPath()}")
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        observed_position = tuple(float(translation[axis]) for axis in range(3))

        def _normalized(vector: Any, label: str) -> tuple[float, float, float]:
            values = tuple(float(component) for component in vector)
            norm = math.sqrt(sum(component * component for component in values))
            if not math.isfinite(norm) or norm <= 1.0e-9:
                raise RuntimeError(f"onboard USD camera closure has a degenerate {label} axis")
            return tuple(component / norm for component in values)

        observed_right = _normalized(matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)), "right")
        observed_up = _normalized(matrix.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)), "up")
        observed_back = _normalized(matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)), "back")
        observed_forward = tuple(-component for component in observed_back)
        normalized_expected_forward = _normalized(expected_forward, "expected forward")
        normalized_expected_right = _normalized(expected_right, "expected right")
        normalized_expected_up = _normalized(expected_up, "expected up")
        normalized_expected_back = _normalized(expected_back, "expected back")
        position_error = math.sqrt(
            sum((float(expected_position[axis]) - observed_position[axis]) ** 2 for axis in range(3))
        )
        forward_cosine = sum(
            normalized_expected_forward[axis] * observed_forward[axis] for axis in range(3)
        )
        up_cosine = sum(normalized_expected_up[axis] * observed_up[axis] for axis in range(3))
        right_cosine = sum(normalized_expected_right[axis] * observed_right[axis] for axis in range(3))
        back_cosine = sum(normalized_expected_back[axis] * observed_back[axis] for axis in range(3))
        orientation_cosine = max(-1.0, min(1.0, (right_cosine + up_cosine + back_cosine - 1.0) / 2.0))
        orientation_error = math.acos(orientation_cosine)
        observed_quat_wxyz = _world_camera_quat_from_usd_axes(
            forward=observed_forward,
            right=observed_right,
            up=observed_up,
        )
        rows.append(
            {
                "agent_id": agent_id,
                "expected_pos_w_m": [float(value) for value in expected_position],
                "observed_pos_w_m": list(observed_position),
                "observed_quat_wxyz": list(observed_quat_wxyz),
                "position_error_m": position_error,
                "forward_alignment_cosine": forward_cosine,
                "up_alignment_cosine": up_cosine,
                "right_alignment_cosine": right_cosine,
                "back_alignment_cosine": back_cosine,
                "orientation_error_rad": orientation_error,
            }
        )
    return {
        "render_transform_authoring": "live_body_pose_plus_fixed_extrinsic_via_camera_world_pose_and_standard_usd_mirror",
        "audit_agent_ids": list(ONBOARD_CAMERA_DEMO_AGENT_IDS),
        "position_tolerance_m": ONBOARD_CAMERA_USD_POSITION_TOLERANCE_M,
        "forward_cosine_min": ONBOARD_CAMERA_USD_FORWARD_COSINE_MIN,
        "orientation_tolerance_rad": ONBOARD_CAMERA_USD_ORIENTATION_TOLERANCE_RAD,
        "per_agent": rows,
        "max_position_error_m": max(float(row["position_error_m"]) for row in rows),
        "min_forward_alignment_cosine": min(float(row["forward_alignment_cosine"]) for row in rows),
        "min_up_alignment_cosine": min(float(row["up_alignment_cosine"]) for row in rows),
        "max_orientation_error_rad": max(float(row["orientation_error_rad"]) for row in rows),
    }


def _require_onboard_camera_usd_pose(closure: Mapping[str, Any]) -> None:
    """Fail closed when any Isaac render camera drifts from its CF2X mount."""

    position_error = float(closure["max_position_error_m"])
    forward_cosine = float(closure["min_forward_alignment_cosine"])
    orientation_error = float(closure["max_orientation_error_rad"])
    if not math.isfinite(position_error) or position_error > ONBOARD_CAMERA_USD_POSITION_TOLERANCE_M:
        raise RuntimeError(
            "onboard camera USD pose does not match the live CF2X mount: "
            f"maximum position error {position_error:.4f} m"
        )
    if not math.isfinite(forward_cosine) or forward_cosine < ONBOARD_CAMERA_USD_FORWARD_COSINE_MIN:
        raise RuntimeError(
            "onboard camera USD optical axis does not match the live CF2X mount: "
            f"minimum forward cosine {forward_cosine:.6f}"
        )
    if not math.isfinite(orientation_error) or orientation_error > ONBOARD_CAMERA_USD_ORIENTATION_TOLERANCE_RAD:
        raise RuntimeError(
            "onboard camera USD full orientation does not match the live CF2X mount: "
            f"maximum orientation error {orientation_error:.6f} rad"
        )


def _onboard_camera_fabric_pose_diagnostic(
    closure: Mapping[str, Any], torch: Any
) -> dict[str, Any]:
    """Classify the non-authoritative Camera Fabric cache without gating RGB-D.

    The render-facing USD transform and its matching render/read fence are the
    acceptance evidence.  A Fabric residual can reveal cache lag or a parent
    binding problem, so it is retained verbatim, but it cannot reject a frame
    whose USD transform, raw image and independent replay all agree.
    """

    position_error = float(torch.max(closure["position_error_m"]).item())
    orientation_error = float(torch.max(closure["orientation_error_rad"]).item())
    finite = math.isfinite(position_error) and math.isfinite(orientation_error)
    within_reference = finite and (
        position_error <= ONBOARD_CAMERA_FABRIC_POSITION_TOLERANCE_M
        and orientation_error <= ONBOARD_CAMERA_FABRIC_ORIENTATION_TOLERANCE_RAD
    )
    return {
        "authority": "diagnostic_only_camera_fabric_cache",
        "acceptance_authority": "render_facing_usd_hierarchy",
        "max_position_error_m": position_error if finite else None,
        "max_orientation_error_rad": orientation_error if finite else None,
        "reference_position_tolerance_m": ONBOARD_CAMERA_FABRIC_POSITION_TOLERANCE_M,
        "reference_orientation_tolerance_rad": ONBOARD_CAMERA_FABRIC_ORIENTATION_TOLERANCE_RAD,
        "finite": finite,
        "within_reference_tolerance": within_reference,
        "status": (
            "within_reference_tolerance"
            if within_reference
            else "lag_or_unverified_non_authoritative"
        ),
    }


def _prepare_onboard_camera_local_mount(
    sim: Any, stage: Any, robot: Any, torch: Any
) -> tuple[Any, Any, dict[str, Any]]:
    """Flush the body transform and close the native parent-relative camera mount.

    ``CameraCfg.offset`` is the local transform from the CF2X body parent to
    the camera.  Do not write a world pose to this dynamic child: IsaacLab's
    Fabric world-pose path and USD hierarchy composition have different
    authorities.  The only runtime operation here is a simulation flush before
    reading the renderer-facing USD hierarchy.
    """

    forward = getattr(sim, "forward", None)
    if not callable(forward):
        raise RuntimeError("SimulationContext Fabric-forward API is unavailable")
    # Flush the post-step PhysX body transform before querying the composed USD
    # hierarchy.  This is not a camera transform write.
    forward()
    expected_positions, expected_world_quats = _expected_onboard_camera_world_poses(robot, torch)
    usd_closure = _onboard_camera_usd_pose_closure(
        stage, expected_positions, expected_world_quats, torch
    )
    _require_onboard_camera_usd_pose(usd_closure)
    return expected_positions, expected_world_quats, usd_closure


def _onboard_camera_frame_counter(camera: Any, torch: Any) -> Any:
    """Return IsaacLab's per-render Camera counter as fail-closed audit data.

    IsaacLab 2.3.2 exposes this counter as ``Camera._frame`` rather than a
    public property.  It is used only to fence a render/read operation; it is
    explicitly not evidence that a particular pixel was rendered at a given
    physics time.  The latter remains subject to native candidate replay.
    """

    counter = getattr(camera, "_frame", None)
    if not torch.is_tensor(counter):
        raise RuntimeError("onboard Camera render-frame counter is unavailable")
    if counter.ndim != 1 or counter.shape[0] != AGENT_COUNT:
        raise RuntimeError("onboard Camera render-frame counter has an invalid agent shape")
    if counter.dtype == torch.bool or torch.is_floating_point(counter):
        raise RuntimeError("onboard Camera render-frame counter must be integral")
    counter = counter.detach().clone().to(dtype=torch.int64)
    if bool(torch.any(counter < 0)):
        raise RuntimeError("onboard Camera render-frame counter must be non-negative")
    return counter


def _require_onboard_camera_render_read_fence(camera: Any, before: Any, torch: Any) -> dict[str, Any]:
    """Require exactly one IsaacLab Camera buffer update after a render call."""

    if not torch.is_tensor(before) or before.ndim != 1 or before.shape[0] != AGENT_COUNT:
        raise RuntimeError("onboard Camera pre-render counter has an invalid agent shape")
    before = before.detach().clone().to(dtype=torch.int64)
    after = _onboard_camera_frame_counter(camera, torch)
    expected_delta = torch.ones_like(before)
    if not torch.equal(after - before, expected_delta):
        raise RuntimeError(
            "onboard Camera render/read frame fence requires exactly one buffer update per agent"
        )
    return {
        "pre_frame_index": before,
        "post_frame_index": after,
    }


def _overview_view_spec() -> dict[str, Any]:
    """Return the fixed public City-Lite overview geometry in world metres."""

    direction = tuple(
        OVERVIEW_CAMERA_TARGET_W_M[axis] - OVERVIEW_CAMERA_EYE_W_M[axis]
        for axis in range(3)
    )
    distance = math.sqrt(sum(component * component for component in direction))
    if not math.isfinite(distance) or distance <= 0.0:
        raise RuntimeError("fixed overview eye and target must be distinct finite world points")
    return {
        "eye_w_m": list(OVERVIEW_CAMERA_EYE_W_M),
        "target_w_m": list(OVERVIEW_CAMERA_TARGET_W_M),
        "view_distance_m": distance,
        "position_tolerance_m": OVERVIEW_CAMERA_POSITION_TOLERANCE_M,
        "forward_cosine_min": OVERVIEW_CAMERA_FORWARD_COSINE_MIN,
    }


def _normalize_vec3(vector: Sequence[float], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(component) ** 2 for component in vector))
    if not math.isfinite(norm) or norm < 1.0e-9:
        return fallback
    return tuple(float(component) / norm for component in vector)  # type: ignore[return-value]


def _cross_vec3(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _matrix_to_quat_wxyz(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to Isaac/USD wxyz quaternion."""

    m = [[float(value) for value in row] for row in matrix]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (0.25 * scale, (m[2][1] - m[1][2]) / scale, (m[0][2] - m[2][0]) / scale, (m[1][0] - m[0][1]) / scale)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        values = ((m[2][1] - m[1][2]) / scale, 0.25 * scale, (m[0][1] + m[1][0]) / scale, (m[0][2] + m[2][0]) / scale)
    elif m[1][1] > m[2][2]:
        scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        values = ((m[0][2] - m[2][0]) / scale, (m[0][1] + m[1][0]) / scale, 0.25 * scale, (m[1][2] + m[2][1]) / scale)
    else:
        scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        values = ((m[1][0] - m[0][1]) / scale, (m[0][2] + m[2][0]) / scale, (m[1][2] + m[2][1]) / scale, 0.25 * scale)
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise RuntimeError("overview look-at rotation cannot be normalized")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _look_at_quat_wxyz(
    eye: Sequence[float], target: Sequence[float]
) -> tuple[float, float, float, float]:
    """Use the same USD/OpenGL look-at basis as md_qd_swarm's scene builder."""

    forward = _normalize_vec3(
        tuple(float(target[axis]) - float(eye[axis]) for axis in range(3)),
        (0.0, 0.0, -1.0),
    )
    up_reference = (0.0, 0.0, 1.0)
    right = _cross_vec3(forward, up_reference)
    if math.sqrt(sum(component * component for component in right)) < 1.0e-7:
        up_reference = (0.0, 1.0, 0.0)
        right = _cross_vec3(forward, up_reference)
    right = _normalize_vec3(right, (1.0, 0.0, 0.0))
    up = _normalize_vec3(_cross_vec3(right, forward), (0.0, 0.0, 1.0))
    backward = tuple(-component for component in forward)
    return _matrix_to_quat_wxyz(
        (
            (right[0], up[0], backward[0]),
            (right[1], up[1], backward[1]),
            (right[2], up[2], backward[2]),
        )
    )


def _author_overview_camera_usd_transform(stage: Any, prim_path: str) -> None:
    """Direct fallback when headless Kit exposes no active viewport API."""

    _author_camera_usd_look_at(
        stage,
        prim_path,
        eye=OVERVIEW_CAMERA_EYE_W_M,
        target=OVERVIEW_CAMERA_TARGET_W_M,
    )


def _author_camera_usd_look_at(
    stage: Any,
    prim_path: str,
    *,
    eye: Sequence[float],
    target: Sequence[float],
) -> None:
    """Author one render-facing Camera transform from public look-at points."""

    from pxr import Gf, Sdf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"overview camera prim is not a USD Camera: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    translate = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    orient = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if translate is None:
        translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if orient is None:
        orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    eye_values = tuple(float(value) for value in eye)
    target_values = tuple(float(value) for value in target)
    if len(eye_values) != 3 or len(target_values) != 3:
        raise RuntimeError("overview camera look-at points must be xyz triples")
    if translate.GetAttr().GetTypeName() == Sdf.ValueTypeNames.Float3:
        translate.Set(Gf.Vec3f(*eye_values))
    else:
        translate.Set(Gf.Vec3d(*eye_values))
    w, x, y, z = _look_at_quat_wxyz(eye_values, target_values)
    if orient.GetAttr().GetTypeName() == Sdf.ValueTypeNames.Quatf:
        orient.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    else:
        orient.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def _set_fixed_overview_view(stage: Any, overview: Any) -> dict[str, Any]:
    """Set then independently verify the render-facing fixed overview transform."""

    spec = _overview_view_spec()
    # Try the exact upstream helper first. In headless Isaac Sim it may return
    # after only logging that no active viewport exists, hence the explicit USD
    # audit and fallback below.
    viewport_api_error: Exception | None = None
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=spec["eye_w_m"],
            target=spec["target_w_m"],
            camera_prim_path=overview.cfg.prim_path,
        )
    except Exception as error:  # noqa: BLE001 - every failure takes the audited USD fallback
        viewport_api_error = error
    closure = _overview_camera_usd_pose_closure(stage, overview.cfg.prim_path)
    try:
        if viewport_api_error is not None:
            raise RuntimeError("Isaac Sim viewport API did not author the witness pose")
        _require_overview_camera_pose(closure)
        spec["transform_authoring"] = "isaacsim_viewport_api"
    except RuntimeError:
        _author_overview_camera_usd_transform(stage, overview.cfg.prim_path)
        closure = _overview_camera_usd_pose_closure(stage, overview.cfg.prim_path)
        _require_overview_camera_pose(closure)
        spec["transform_authoring"] = "direct_usd_look_at_fallback"
        if viewport_api_error is not None:
            spec["viewport_api_error_type"] = type(viewport_api_error).__name__
    return spec


def _overview_camera_usd_pose_closure(stage: Any, prim_path: str) -> dict[str, Any]:
    """Read the render-facing USD transform instead of a possibly stale Fabric pose."""

    return _camera_usd_pose_closure_for_view(
        stage,
        prim_path,
        expected_eye=OVERVIEW_CAMERA_EYE_W_M,
        expected_target=OVERVIEW_CAMERA_TARGET_W_M,
    )


def _camera_usd_pose_closure_for_view(
    stage: Any,
    prim_path: str,
    *,
    expected_eye: Sequence[float],
    expected_target: Sequence[float],
) -> dict[str, Any]:
    """Close a USD camera against an explicit public look-at specification."""

    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"overview camera prim is not a USD Camera: {prim_path}")
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    translation = matrix.ExtractTranslation()
    observed_pos = (float(translation[0]), float(translation[1]), float(translation[2]))
    # USD cameras use OpenGL optical axes: -Z forward, +Y up.
    observed_forward_raw = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
    observed_norm = math.sqrt(sum(float(component) ** 2 for component in observed_forward_raw))
    if not math.isfinite(observed_norm) or observed_norm <= 0.0:
        raise RuntimeError("overview camera USD transform has no finite optical forward direction")
    observed_forward = tuple(float(component) / observed_norm for component in observed_forward_raw)
    eye = tuple(float(value) for value in expected_eye)
    target = tuple(float(value) for value in expected_target)
    if len(eye) != 3 or len(target) != 3:
        raise RuntimeError("camera closure look-at points must be xyz triples")
    expected_forward_raw = tuple(target[axis] - eye[axis] for axis in range(3))
    expected_norm = math.sqrt(sum(component * component for component in expected_forward_raw))
    if not math.isfinite(expected_norm) or expected_norm <= 1.0e-9:
        raise RuntimeError("camera closure look-at points must be distinct finite xyz triples")
    expected_forward = tuple(component / expected_norm for component in expected_forward_raw)
    position_error = math.sqrt(
        sum(
            (eye[axis] - observed_pos[axis]) ** 2
            for axis in range(3)
        )
    )
    return {
        "expected_pos_w_m": list(eye),
        "expected_target_w_m": list(target),
        "observed_pos_w_m": list(observed_pos),
        "position_error_m": position_error,
        "forward_alignment_cosine": sum(
            expected_forward[axis] * observed_forward[axis] for axis in range(3)
        ),
    }


def _require_overview_camera_pose(closure: dict[str, Any]) -> None:
    """Fail before recording video if the rendering camera is not the public view."""

    position_error = float(closure["position_error_m"])
    forward_cosine = float(closure["forward_alignment_cosine"])
    if not math.isfinite(position_error) or position_error > OVERVIEW_CAMERA_POSITION_TOLERANCE_M:
        raise RuntimeError(
            "overview camera USD pose does not match the fixed City-Lite view: "
            f"position error {position_error:.4f} m"
        )
    if not math.isfinite(forward_cosine) or forward_cosine < OVERVIEW_CAMERA_FORWARD_COSINE_MIN:
        raise RuntimeError(
            "overview camera USD orientation does not face the fixed City-Lite view: "
            f"forward cosine {forward_cosine:.6f}"
        )


def _rotate_vec3_wxyz(quaternion: Sequence[float], vector: Sequence[float]) -> tuple[float, float, float]:
    """Rotate one xyz vector without importing Isaac math into unit tests."""

    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise RuntimeError("follow camera requires a finite nonzero CF2X quaternion")
    w, x, y, z = (component / norm for component in (w, x, y, z))
    vx, vy, vz = (float(value) for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _public_follow_view_from_body_pose(
    body_pos_w_m: Sequence[float], body_quat_wxyz: Sequence[float]
) -> dict[str, Any]:
    """Derive the public demo view solely from one measured CF2X body pose."""

    position = tuple(float(value) for value in body_pos_w_m)
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise RuntimeError("follow camera requires a finite CF2X world position")
    eye_offset = _rotate_vec3_wxyz(body_quat_wxyz, OVERVIEW_FOLLOW_EYE_OFFSET_BODY_M)
    target_offset = _rotate_vec3_wxyz(body_quat_wxyz, OVERVIEW_FOLLOW_TARGET_OFFSET_BODY_M)
    eye = tuple(position[axis] + eye_offset[axis] for axis in range(3))
    target = tuple(position[axis] + target_offset[axis] for axis in range(3))
    orientation = _look_at_quat_wxyz(eye, target)
    return {
        "schema": OVERVIEW_FOLLOW_SCHEMA,
        "mode": "public_cf2x_body_follow",
        "tracked_agent_id": OVERVIEW_FOLLOW_TRACKED_AGENT_ID,
        "eye_w_m": list(eye),
        "target_w_m": list(target),
        "orientation_wxyz": list(orientation),
    }


def _set_public_follow_overview_view(
    stage: Any, overview: Any, robot: Any, torch: Any
) -> dict[str, Any]:
    """Author the evidence camera from the live public CF2X pose before render.

    ``Camera.set_world_poses`` uses the Camera Fabric orientation convention,
    which is not the USD Camera OpenGL optical-axis convention used by the
    renderer.  Authoring the look-at transform directly keeps the frame that
    Isaac renders and the USD closure in the same coordinate system.
    """

    agent_id = OVERVIEW_FOLLOW_TRACKED_AGENT_ID
    body_pos = _to_numpy(robot.data.root_pos_w[agent_id]).tolist()
    body_quat = _to_numpy(robot.data.root_quat_w[agent_id]).tolist()
    spec = _public_follow_view_from_body_pose(body_pos, body_quat)
    _author_camera_usd_look_at(
        stage,
        overview.cfg.prim_path,
        eye=spec["eye_w_m"],
        target=spec["target_w_m"],
    )
    closure = _camera_usd_pose_closure_for_view(
        stage,
        overview.cfg.prim_path,
        expected_eye=spec["eye_w_m"],
        expected_target=spec["target_w_m"],
    )
    position_error = float(closure["position_error_m"])
    forward_cosine = float(closure["forward_alignment_cosine"])
    if (
        not math.isfinite(position_error)
        or position_error > OVERVIEW_FOLLOW_POSITION_TOLERANCE_M
    ):
        raise RuntimeError(
            "public follow camera USD position does not match the live CF2X pose: "
            f"position error {position_error:.4f} m"
        )
    if (
        not math.isfinite(forward_cosine)
        or forward_cosine < OVERVIEW_FOLLOW_FORWARD_COSINE_MIN
    ):
        raise RuntimeError(
            "public follow camera USD orientation does not face its live CF2X target: "
            f"forward cosine {forward_cosine:.6f}"
        )
    return {**spec, "pose_closure": closure}


def _public_route_witness_schedule() -> dict[str, Any]:
    """Return the one immutable, state-independent world witness pose."""

    if len(OVERVIEW_WITNESS_SHOTS) != 1:
        raise RuntimeError("route witness must declare exactly one frozen world pose")
    shots: list[dict[str, Any]] = []
    for shot_index, (start_ns, end_ns, raw_eye, raw_target) in enumerate(
        OVERVIEW_WITNESS_SHOTS
    ):
        eye = tuple(float(value) for value in raw_eye)
        target = tuple(float(value) for value in raw_target)
        if start_ns != 0 or end_ns is not None:
            raise RuntimeError("route witness must start at zero and remain open-ended")
        if (
            len(eye) != 3
            or len(target) != 3
            or not all(math.isfinite(value) for value in (*eye, *target))
            or math.dist(eye, target) <= 1.0e-6
        ):
            raise RuntimeError("route-witness schedule contains an invalid fixed camera shot")
        shots.append(
            {
                "shot_index": shot_index,
                "start_time_ns": int(start_ns),
                "end_time_ns": None if end_ns is None else int(end_ns),
                "eye_w_m": list(eye),
                "target_w_m": list(target),
                "orientation_wxyz": list(_look_at_quat_wxyz(eye, target)),
            }
        )
    if len(shots) != 1 or shots[0]["start_time_ns"] != 0 or shots[0]["end_time_ns"] is not None:
        raise RuntimeError("route witness must contain one open-ended frozen world pose")
    return {
        "schema": OVERVIEW_WITNESS_SCHEMA,
        "mode": "public_fixed_route_witness_schedule",
        "selection": OVERVIEW_WITNESS_SELECTION,
        "selection_state_independent": True,
        "tracked_agent_id": OVERVIEW_WITNESS_TRACKED_AGENT_ID,
        "shots": shots,
        "minimum_tracked_agent_displacement_m": OVERVIEW_WITNESS_MIN_TRACKED_AGENT_DISPLACEMENT_M,
        "minimum_tracked_agent_pixels": OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS,
        "position_tolerance_m": OVERVIEW_WITNESS_POSITION_TOLERANCE_M,
        "forward_cosine_min": OVERVIEW_WITNESS_FORWARD_COSINE_MIN,
    }


def _public_route_witness_view_at_time_ns(effective_time_ns: int) -> dict[str, Any]:
    """Return the same frozen public witness pose for every nonnegative time."""

    if isinstance(effective_time_ns, bool) or int(effective_time_ns) < 0:
        raise ValueError("route-witness effective_time_ns must be a non-negative integer")
    timestamp_ns = int(effective_time_ns)
    schedule = _public_route_witness_schedule()
    for shot in schedule["shots"]:
        end_ns = shot["end_time_ns"]
        if end_ns is None or timestamp_ns < int(end_ns):
            return {
                **schedule,
                "effective_time_ns": timestamp_ns,
                "shot_index": int(shot["shot_index"]),
                "start_time_ns": int(shot["start_time_ns"]),
                "end_time_ns": end_ns,
                "eye_w_m": list(shot["eye_w_m"]),
                "target_w_m": list(shot["target_w_m"]),
                "orientation_wxyz": list(shot["orientation_wxyz"]),
            }
    raise RuntimeError("route-witness schedule does not cover its effective timestamp")


def _public_route_witness_view() -> dict[str, Any]:
    """Return the first immutable witness shot for initial-render compatibility."""

    return _public_route_witness_view_at_time_ns(0)


def _set_public_route_witness_overview_view(
    stage: Any, overview: Any, *, effective_time_ns: int = 0
) -> dict[str, Any]:
    """Author and close the selected fixed public witness transform before rendering."""

    spec = _public_route_witness_view_at_time_ns(effective_time_ns)
    eye = tuple(float(value) for value in spec["eye_w_m"])
    target = tuple(float(value) for value in spec["target_w_m"])
    if (
        len(eye) != 3
        or len(target) != 3
        or not all(math.isfinite(value) for value in (*eye, *target))
        or math.dist(eye, target) <= 1.0e-6
    ):
        raise RuntimeError("public route witness camera requires finite distinct xyz points")
    _author_camera_usd_look_at(
        stage,
        overview.cfg.prim_path,
        eye=spec["eye_w_m"],
        target=spec["target_w_m"],
    )
    closure = _camera_usd_pose_closure_for_view(
        stage,
        overview.cfg.prim_path,
        expected_eye=spec["eye_w_m"],
        expected_target=spec["target_w_m"],
    )
    position_error = float(closure["position_error_m"])
    forward_cosine = float(closure["forward_alignment_cosine"])
    if (
        not math.isfinite(position_error)
        or position_error > OVERVIEW_WITNESS_POSITION_TOLERANCE_M
    ):
        raise RuntimeError(
            "public route witness camera USD position differs from its declared world pose: "
            f"position error {position_error:.4f} m"
        )
    if (
        not math.isfinite(forward_cosine)
        or forward_cosine < OVERVIEW_WITNESS_FORWARD_COSINE_MIN
    ):
        raise RuntimeError(
            "public route witness camera USD orientation differs from its declared target: "
            f"forward cosine {forward_cosine:.6f}"
        )
    return {**spec, "pose_closure": closure}


def _overview_city_content_gate_contract() -> dict[str, float]:
    """Return the public thresholds used to accept an overview render frame."""

    return {
        "minimum_finite_depth_fraction": OVERVIEW_CONTENT_MIN_FINITE_DEPTH_FRACTION,
        "minimum_geometry_fraction": OVERVIEW_CONTENT_MIN_GEOMETRY_FRACTION,
        "near_surface_m": OVERVIEW_CONTENT_NEAR_SURFACE_M,
        "maximum_near_surface_fraction": OVERVIEW_CONTENT_MAX_NEAR_SURFACE_FRACTION,
        "minimum_geometry_depth_span_m": OVERVIEW_CONTENT_MIN_DEPTH_SPAN_M,
        "rgb_edge_delta": OVERVIEW_CONTENT_RGB_EDGE_DELTA,
        "minimum_rgb_edge_fraction": OVERVIEW_CONTENT_MIN_RGB_EDGE_FRACTION,
        "minimum_structural_pixel_fraction": OVERVIEW_CONTENT_MIN_STRUCTURAL_PIXEL_FRACTION,
    }


def _semantic_label_text(value: Any) -> str:
    """Flatten a Replicator label payload without relying on one metadata ABI."""

    if isinstance(value, Mapping):
        return " ".join(
            f"{key} {_semantic_label_text(item)}" for key, item in value.items()
        ).lower()
    if isinstance(value, (list, tuple)):
        return " ".join(_semantic_label_text(item) for item in value).lower()
    return str(value).lower()


def _overview_structural_semantic_ids(metadata: Any) -> tuple[tuple[int, ...], bool]:
    """Find structural Replicator IDs from either common metadata spelling."""

    structural_ids: set[int] = set()
    id_labels_seen = False

    def visit(value: Any) -> None:
        nonlocal id_labels_seen
        if isinstance(value, Mapping):
            for key in ("id_to_labels", "idToLabels"):
                labels = value.get(key)
                if not isinstance(labels, Mapping):
                    continue
                id_labels_seen = True
                for raw_id, label in labels.items():
                    try:
                        semantic_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    label_text = _semantic_label_text(label)
                    if any(token in label_text for token in OVERVIEW_STRUCTURAL_LABEL_TOKENS):
                        structural_ids.add(semantic_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(metadata)
    return tuple(sorted(structural_ids)), id_labels_seen


def _overview_render_array(value: Any, *, channels: int) -> np.ndarray | None:
    """Normalize one singleton Camera output to image-major NumPy storage."""

    if value is None:
        return None
    array = np.asarray(_to_numpy(value))
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if channels == 3:
        return array if array.ndim == 3 and array.shape[-1] == 3 else None
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    return array if array.ndim == 2 else None


def _overview_tracked_agent_visibility_evidence(
    semantic: Any, semantic_metadata: Any
) -> dict[str, Any]:
    """Prove that the route-witness image contains its named live CF2X marker.

    The semantic marker is parented to the physical CF2X body.  It is a better
    visual-evidence contract than a global pixel-difference threshold, which
    could be satisfied by rendering noise or unrelated scene motion.
    """

    tracked_agent = str(OVERVIEW_WITNESS_TRACKED_AGENT_ID)
    marker_ids: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in ("id_to_labels", "idToLabels"):
                labels = value.get(key)
                if not isinstance(labels, Mapping):
                    continue
                for raw_id, raw_labels in labels.items():
                    try:
                        semantic_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    label_text = _semantic_label_text(raw_labels)
                    declared_agent = (
                        str(raw_labels.get("agent_id"))
                        if isinstance(raw_labels, Mapping) and "agent_id" in raw_labels
                        else ""
                    )
                    if declared_agent == tracked_agent and "agent_identity" in label_text:
                        marker_ids.add(semantic_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(semantic_metadata)
    semantic_array = _overview_render_array(semantic, channels=1)
    failures: list[str] = []
    pixel_count = 0
    if semantic_array is None or not np.issubdtype(semantic_array.dtype, np.integer):
        failures.append("overview semantic image is unavailable for tracked-CF2X visibility")
    elif not marker_ids:
        failures.append("overview semantic metadata has no tracked CF2X identity marker")
    else:
        pixel_count = int(
            np.count_nonzero(
                np.isin(semantic_array, np.asarray(sorted(marker_ids), dtype=semantic_array.dtype))
            )
        )
        if pixel_count < OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS:
            failures.append(
                "tracked CF2X identity marker is too small or absent from the route-witness frame"
            )
    return {
        "schema": "org.rivermark.isaac-route-witness-agent-visibility.v1",
        "tracked_agent_id": OVERVIEW_WITNESS_TRACKED_AGENT_ID,
        "tracked_agent_marker_semantic_ids": sorted(marker_ids),
        "tracked_agent_pixel_count": pixel_count,
        "minimum_tracked_agent_pixels": OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS,
        "passed": not failures,
        "failures": failures,
    }


def _require_overview_tracked_agent_visibility(evidence: Mapping[str, Any]) -> None:
    """Reject a demo frame whose declared route CF2X is not visibly present."""

    if evidence.get("passed") is True:
        return
    failures = evidence.get("failures")
    raise RuntimeError(
        "public route witness does not visibly contain the tracked CF2X: "
        + "; ".join(str(value) for value in failures if str(value))
    )


def _persist_initial_overview_failure_diagnostics(
    output_dir: Path,
    *,
    rgb: Any,
    depth: Any,
    semantic: Any,
    semantic_metadata: Any,
    content_evidence: Mapping[str, Any],
    agent_visibility_evidence: Mapping[str, Any],
    root_pos_w_m: Any,
    root_quat_wxyz: Any,
    root_lin_vel_w_mps: Any,
    np: Any,
) -> dict[str, Any]:
    """Persist the native initial witness frame before a fail-closed gate raises.

    The overview gate executes before the normal frame spools exist. Retaining its
    real RGB, depth, semantic IDs, and public fleet state makes a camera or
    semantic failure independently diagnosable without manufacturing a frame or
    leaking evaluator-private coordinates.
    """

    relative_root = Path("failure_diagnostics")
    archive_path = output_dir / relative_root / "initial_overview_native.npz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive_path,
        rgb=np.asarray(_to_numpy(rgb)),
        distance_to_image_plane=np.asarray(_to_numpy(depth)),
        semantic_segmentation=np.asarray(_to_numpy(semantic)),
        root_pos_w_m=np.asarray(_to_numpy(root_pos_w_m)),
        root_quat_wxyz=np.asarray(_to_numpy(root_quat_wxyz)),
        root_lin_vel_w_mps=np.asarray(_to_numpy(root_lin_vel_w_mps)),
    )
    metadata_path = archive_path.with_suffix(".json")
    _write_json(
        metadata_path,
        {
            "schema": "org.rivermark.initial-overview-failure-diagnostics.v1",
            "raw_sensor_archive": archive_path.name,
            "raw_sensor_archive_sha256": _sha256(archive_path),
            "semantic_metadata": semantic_metadata,
            "overview_content_evidence": dict(content_evidence),
            "overview_agent_visibility_evidence": dict(agent_visibility_evidence),
            "private_evaluator_coordinates_included": False,
        },
    )
    return {
        "schema": "org.rivermark.initial-overview-failure-diagnostics.v1",
        "archive_relative_path": archive_path.relative_to(output_dir).as_posix(),
        "archive_sha256": _sha256(archive_path),
        "metadata_relative_path": metadata_path.relative_to(output_dir).as_posix(),
        "metadata_sha256": _sha256(metadata_path),
    }


def _overview_array_shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    try:
        return list(np.asarray(_to_numpy(value)).shape)
    except (TypeError, ValueError):
        return None


def _overview_city_content_evidence(
    rgb: Any,
    depth: Any,
    semantic: Any,
    semantic_metadata: Any,
    *,
    far_clip_m: float = OVERVIEW_CAMERA_CLIPPING_RANGE_M[1],
) -> dict[str, Any]:
    """Measure whether one real overview render visibly contains City-Lite.

    Depth proves that the renderer returned scene geometry rather than only an
    environment/background. RGB spatial variation prevents a uniform depth
    buffer from being treated as a useful video frame. Structural semantic
    evidence is mandatory only when the render product supplies structural ID
    metadata, preserving a valid geometric fallback for unlabelled USD assets.
    """

    failures: list[str] = []
    rgb_array = _overview_render_array(rgb, channels=3)
    depth_array = _overview_render_array(depth, channels=1)
    semantic_array = _overview_render_array(semantic, channels=1)
    structural_ids, id_labels_seen = _overview_structural_semantic_ids(
        semantic_metadata
    )
    if not math.isfinite(float(far_clip_m)) or float(far_clip_m) <= 0.0:
        failures.append("overview far clipping distance is not finite and positive")

    rgb_edge_fraction: float | None = None
    image_shape_hw: list[int] | None = None
    if rgb_array is None or rgb_array.dtype != np.uint8:
        failures.append("overview RGB must be uint8 [H,W,3]")
    else:
        height, width = rgb_array.shape[:2]
        image_shape_hw = [int(height), int(width)]
        if height < 2 or width < 2:
            failures.append("overview RGB is too small for spatial evidence")
        else:
            luma = (
                0.2126 * rgb_array[..., 0].astype(np.float32)
                + 0.7152 * rgb_array[..., 1].astype(np.float32)
                + 0.0722 * rgb_array[..., 2].astype(np.float32)
            )
            horizontal = np.abs(np.diff(luma, axis=1)) >= OVERVIEW_CONTENT_RGB_EDGE_DELTA
            vertical = np.abs(np.diff(luma, axis=0)) >= OVERVIEW_CONTENT_RGB_EDGE_DELTA
            rgb_edge_fraction = float(
                (np.count_nonzero(horizontal) + np.count_nonzero(vertical))
                / (horizontal.size + vertical.size)
            )
            if rgb_edge_fraction < OVERVIEW_CONTENT_MIN_RGB_EDGE_FRACTION:
                failures.append(
                    "overview RGB lacks enough spatial structure for City-Lite evidence"
                )

    finite_depth_fraction: float | None = None
    geometry_fraction: float | None = None
    near_surface_fraction: float | None = None
    geometry_depth_span_m: float | None = None
    if depth_array is None or not np.issubdtype(depth_array.dtype, np.floating):
        failures.append("overview depth must be floating [H,W,1]")
    elif image_shape_hw is None or list(depth_array.shape) != image_shape_hw:
        failures.append("overview depth shape does not match RGB")
    else:
        finite_depth = np.isfinite(depth_array)
        finite_depth_fraction = float(np.mean(finite_depth))
        if finite_depth_fraction < OVERVIEW_CONTENT_MIN_FINITE_DEPTH_FRACTION:
            failures.append("overview depth contains too much non-finite background")
        background_margin_m = max(0.05, float(far_clip_m) * 1.0e-3)
        geometry = finite_depth & (depth_array >= 0.0) & (
            depth_array < float(far_clip_m) - background_margin_m
        )
        geometry_fraction = float(np.mean(geometry))
        if geometry_fraction < OVERVIEW_CONTENT_MIN_GEOMETRY_FRACTION:
            failures.append("overview depth has insufficient non-background geometry")
        near_surface_fraction = float(
            np.mean(geometry & (depth_array <= OVERVIEW_CONTENT_NEAR_SURFACE_M))
        )
        if near_surface_fraction > OVERVIEW_CONTENT_MAX_NEAR_SURFACE_FRACTION:
            failures.append("overview camera is dominated by near-surface geometry")
        if np.any(geometry):
            geometry_values = depth_array[geometry]
            geometry_depth_span_m = float(
                np.percentile(geometry_values, 95) - np.percentile(geometry_values, 5)
            )
        else:
            geometry_depth_span_m = 0.0
        if geometry_depth_span_m < OVERVIEW_CONTENT_MIN_DEPTH_SPAN_M:
            failures.append("overview geometry has insufficient depth variation")

    structural_pixel_fraction: float | None = None
    structural_semantics_required = bool(structural_ids)
    if structural_semantics_required:
        if (
            semantic_array is None
            or not np.issubdtype(semantic_array.dtype, np.integer)
            or image_shape_hw is None
            or list(semantic_array.shape) != image_shape_hw
        ):
            failures.append(
                "overview structural metadata exists but segmentation cannot be matched to RGB"
            )
        else:
            structural_pixel_fraction = float(
                np.mean(np.isin(semantic_array, np.asarray(structural_ids)))
            )
            if structural_pixel_fraction < OVERVIEW_CONTENT_MIN_STRUCTURAL_PIXEL_FRACTION:
                failures.append("overview contains no labelled structural City-Lite pixels")

    city_evidence_passed = bool(
        finite_depth_fraction is not None
        and finite_depth_fraction >= OVERVIEW_CONTENT_MIN_FINITE_DEPTH_FRACTION
        and geometry_fraction is not None
        and geometry_fraction >= OVERVIEW_CONTENT_MIN_GEOMETRY_FRACTION
        and near_surface_fraction is not None
        and near_surface_fraction <= OVERVIEW_CONTENT_MAX_NEAR_SURFACE_FRACTION
        and geometry_depth_span_m is not None
        and geometry_depth_span_m >= OVERVIEW_CONTENT_MIN_DEPTH_SPAN_M
        and rgb_edge_fraction is not None
        and rgb_edge_fraction >= OVERVIEW_CONTENT_MIN_RGB_EDGE_FRACTION
    )
    structural_evidence_passed = bool(
        not structural_semantics_required
        or structural_pixel_fraction is not None
        and structural_pixel_fraction >= OVERVIEW_CONTENT_MIN_STRUCTURAL_PIXEL_FRACTION
    )
    return {
        "schema": OVERVIEW_CONTENT_GATE_SCHEMA,
        "far_clip_m": float(far_clip_m),
        "thresholds": _overview_city_content_gate_contract(),
        "rgb_input_shape": _overview_array_shape(rgb),
        "depth_input_shape": _overview_array_shape(depth),
        "semantic_input_shape": _overview_array_shape(semantic),
        "image_shape_hw": image_shape_hw,
        "finite_depth_fraction": finite_depth_fraction,
        "non_background_geometry_fraction": geometry_fraction,
        "near_surface_fraction": near_surface_fraction,
        "geometry_depth_span_m": geometry_depth_span_m,
        "rgb_edge_fraction": rgb_edge_fraction,
        "semantic_id_metadata_available": id_labels_seen,
        "structural_semantic_ids": list(structural_ids),
        "structural_semantics_required": structural_semantics_required,
        "structural_pixel_fraction": structural_pixel_fraction,
        # These two fields bind the live depth gate to the retained low-rate
        # RGB/semantic witness without retaining every overview depth frame.
        "city_evidence_passed": city_evidence_passed,
        "structural_evidence_passed": structural_evidence_passed,
        "passed": not failures,
        "failures": failures,
    }


def _require_overview_city_content(evidence: Mapping[str, Any]) -> None:
    """Reject a render frame that cannot prove it shows the City-Lite scene."""

    if evidence.get("passed") is True:
        return
    failures = evidence.get("failures")
    if isinstance(failures, list) and failures:
        detail = "; ".join(str(item) for item in failures)
    else:
        detail = "unknown City-Lite content evidence failure"
    raise RuntimeError(f"overview City-Lite content gate failed: {detail}")


def _onboard_content_gate_contract() -> dict[str, float]:
    """Return the immutable onboard scene-content thresholds."""

    return {
        "minimum_finite_depth_fraction": ONBOARD_CONTENT_MIN_FINITE_DEPTH_FRACTION,
        "minimum_geometry_fraction": ONBOARD_CONTENT_MIN_GEOMETRY_FRACTION,
        "maximum_background_fraction": ONBOARD_CONTENT_MAX_BACKGROUND_FRACTION,
    }


def _onboard_background_semantic_ids(
    semantic_metadata: Any, agent_id: int
) -> tuple[int, ...]:
    """Extract background IDs for one batched onboard camera metadata entry."""

    metadata = semantic_metadata
    if isinstance(semantic_metadata, Mapping):
        per_camera = semantic_metadata.get("per_camera")
        if isinstance(per_camera, (list, tuple)) and agent_id < len(per_camera):
            metadata = per_camera[agent_id]
        elif isinstance(per_camera, (list, tuple)) and len(per_camera) == 1:
            metadata = per_camera[0]
    background_ids: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in ("id_to_labels", "idToLabels"):
                labels = value.get(key)
                if not isinstance(labels, Mapping):
                    continue
                for raw_id, label in labels.items():
                    try:
                        semantic_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    if "background" in _semantic_label_text(label):
                        background_ids.add(semantic_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(metadata)
    return tuple(sorted(background_ids))


def _onboard_scene_content_evidence(
    depth_m: Any,
    semantic: Any,
    semantic_metadata: Any,
    *,
    far_clip_m: float = ONBOARD_CAMERA_CLIPPING_RANGE_M[1],
) -> dict[str, Any]:
    """Reject onboard frames dominated by far-clip/background pixels.

    Near-geometry intrusion and scene-content quality are separate contracts:
    a camera can be clear of meshes while still looking mostly at the sky.  The
    semantic background fraction is recomputed from the raw label image and
    per-camera Replicator metadata; calibration can only declare the result,
    never make a failing raw frame pass.
    """

    depth = np.asarray(_to_numpy(depth_m))
    labels = np.asarray(_to_numpy(semantic))
    failures: list[str] = []
    if (
        depth.ndim != 4
        or depth.shape[0] != AGENT_COUNT
        or depth.shape[-1] != 1
        or not np.issubdtype(depth.dtype, np.floating)
    ):
        failures.append("onboard depth must be floating [8,H,W,1]")
    if (
        labels.ndim != 4
        or labels.shape[0] != AGENT_COUNT
        or labels.shape[-1] != 1
        or not np.issubdtype(labels.dtype, np.integer)
        or depth.ndim == 4
        and labels.shape != depth.shape
    ):
        failures.append("onboard semantic labels must be integer [8,H,W,1] matching depth")
    if (
        not math.isfinite(float(far_clip_m))
        or float(far_clip_m) <= 0.0
    ):
        failures.append("onboard far clipping distance is not finite and positive")
    if failures:
        return {
            "schema": ONBOARD_CONTENT_GATE_SCHEMA,
            "far_clip_m": float(far_clip_m),
            "thresholds": _onboard_content_gate_contract(),
            "depth_input_shape": list(depth.shape),
            "semantic_input_shape": list(labels.shape),
            "passed": False,
            "failures": failures,
            "per_agent": [],
        }

    per_agent: list[dict[str, Any]] = []
    background_ids_available = True
    margin_m = max(0.05, float(far_clip_m) * 1.0e-3)
    for agent_id in range(AGENT_COUNT):
        depth_values = depth[agent_id, ..., 0]
        label_values = labels[agent_id, ..., 0]
        finite_depth = np.isfinite(depth_values) & (depth_values >= 0.0)
        geometry = finite_depth & (depth_values < float(far_clip_m) - margin_m)
        finite_fraction = float(np.mean(finite_depth))
        geometry_fraction = float(np.mean(geometry))
        background_ids = _onboard_background_semantic_ids(semantic_metadata, agent_id)
        if not background_ids:
            background_ids_available = False
            background_fraction: float | None = None
        else:
            background_fraction = float(
                np.mean(np.isin(label_values, np.asarray(background_ids, dtype=label_values.dtype)))
            )
        agent_failures: list[str] = []
        if finite_fraction < ONBOARD_CONTENT_MIN_FINITE_DEPTH_FRACTION:
            agent_failures.append("onboard depth contains too many invalid pixels")
        if geometry_fraction < ONBOARD_CONTENT_MIN_GEOMETRY_FRACTION:
            agent_failures.append("onboard view is dominated by far-clip/background depth")
        if background_fraction is None:
            agent_failures.append("onboard semantic metadata has no background label IDs")
        elif background_fraction > ONBOARD_CONTENT_MAX_BACKGROUND_FRACTION:
            agent_failures.append("onboard semantic view is dominated by background labels")
        per_agent.append(
            {
                "agent_id": agent_id,
                "finite_depth_fraction": finite_fraction,
                "non_background_geometry_fraction": geometry_fraction,
                "background_semantic_ids": list(background_ids),
                "background_fraction": background_fraction,
                "passed": not agent_failures,
                "failures": agent_failures,
            }
        )
    failed_agents = [item["agent_id"] for item in per_agent if not item["passed"]]
    if failed_agents:
        failures.append(f"onboard scene content failed for agents {failed_agents}")
    return {
        "schema": ONBOARD_CONTENT_GATE_SCHEMA,
        "far_clip_m": float(far_clip_m),
        "thresholds": _onboard_content_gate_contract(),
        "depth_input_shape": list(depth.shape),
        "semantic_input_shape": list(labels.shape),
        "background_semantic_ids_available": background_ids_available,
        "passed": not failures,
        "failures": failures,
        "per_agent": per_agent,
    }


def _onboard_semantic_frame_evidence(
    depth_m: Any,
    semantic: Any,
    semantic_metadata: Any,
    target_slots: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Evaluate both onboard semantic contracts from one frame-local mapping.

    Replicator IDs are camera-local and can change after each update.  Keeping
    both gates behind this small helper prevents a stale or differently scoped
    metadata variable from making one gate observe a different frame.
    """

    scene_content = _onboard_scene_content_evidence(
        depth_m,
        semantic,
        semantic_metadata,
        far_clip_m=ONBOARD_CAMERA_CLIPPING_RANGE_M[1],
    )
    target_visibility = (
        _target_semantic_visibility_evidence(
            semantic,
            semantic_metadata,
            target_slots,
            minimum_pixels=PRIVATE_TARGET_MIN_VISIBLE_INSTANCE_PIXELS,
        )
        if target_slots
        else None
    )
    return scene_content, target_visibility


def _require_onboard_scene_content(evidence: Mapping[str, Any]) -> None:
    """Abort before a sky/background-dominated onboard frame is accepted."""

    if evidence.get("passed") is True:
        return
    failures = evidence.get("failures")
    detail = "; ".join(str(value) for value in failures) if isinstance(failures, list) else "unknown failure"
    raise RuntimeError(f"onboard scene-content gate failed: {detail}")


def _visual_intrusion_gate_contract() -> dict[str, float]:
    """Return the immutable raw RGB-D/LiDAR near-geometry limits."""

    return {
        "near_distance_m": VISUAL_INTRUSION_NEAR_DISTANCE_M,
        "maximum_rgbd_near_pixel_fraction": VISUAL_INTRUSION_RGBD_MAX_NEAR_PIXEL_FRACTION,
        # "return" is reserved by the policy-leakage scanner for evaluator
        # returns. This is a public sensor-density threshold, not a reward.
        "maximum_lidar_near_fraction": VISUAL_INTRUSION_LIDAR_MAX_NEAR_RETURN_FRACTION,
    }


def _onboard_visual_intrusion_evidence(
    depth_m: Any,
    lidar_ranges_m: Any,
    *,
    lidar_max_distance_m: float,
) -> dict[str, Any]:
    """Cross-check raw onboard RGB-D and LiDAR for rendered geometry intrusion.

    This guard is intentionally independent of the root contact sensor and
    conservative AABB sweep. It catches a stale/mis-mounted camera as well as
    a rendered mesh that has no corresponding native collider.
    """

    depth = np.asarray(_to_numpy(depth_m))
    ranges = np.asarray(_to_numpy(lidar_ranges_m))
    failures: list[str] = []
    if (
        depth.ndim != 4
        or depth.shape[0] != AGENT_COUNT
        or depth.shape[-1] != 1
        or not np.issubdtype(depth.dtype, np.floating)
    ):
        failures.append("RGB-D input must be floating [8,H,W,1]")
    if (
        ranges.ndim != 2
        or ranges.shape[0] != AGENT_COUNT
        or ranges.shape[1] < 32
        or not np.issubdtype(ranges.dtype, np.floating)
    ):
        failures.append("LiDAR input must be floating [8,R] with R >= 32")
    if not math.isfinite(float(lidar_max_distance_m)) or float(lidar_max_distance_m) <= 0.0:
        failures.append("LiDAR max distance must be finite and positive")
    if failures:
        return {
            "schema": VISUAL_INTRUSION_GATE_SCHEMA,
            "contract": _visual_intrusion_gate_contract(),
            "passed": False,
            "failures": failures,
            "per_agent": [],
        }

    per_agent: list[dict[str, Any]] = []
    for agent_id in range(AGENT_COUNT):
        depth_values = depth[agent_id, ..., 0]
        range_values = ranges[agent_id]
        finite_depth = np.isfinite(depth_values) & (depth_values >= 0.0)
        finite_ranges = np.isfinite(range_values) & (range_values >= 0.0)
        rgbd_near_fraction = float(
            np.mean(finite_depth & (depth_values <= VISUAL_INTRUSION_NEAR_DISTANCE_M))
        )
        lidar_near_fraction = float(
            np.mean(finite_ranges & (range_values <= VISUAL_INTRUSION_NEAR_DISTANCE_M))
        )
        rgbd_min = float(np.min(depth_values[finite_depth])) if np.any(finite_depth) else math.inf
        lidar_min = float(np.min(range_values[finite_ranges])) if np.any(finite_ranges) else math.inf
        agent_failures: list[str] = []
        if rgbd_near_fraction > VISUAL_INTRUSION_RGBD_MAX_NEAR_PIXEL_FRACTION:
            agent_failures.append("RGB-D frame is dominated by near geometry")
        if lidar_near_fraction > VISUAL_INTRUSION_LIDAR_MAX_NEAR_RETURN_FRACTION:
            agent_failures.append("LiDAR has excessive near-geometry returns")
        per_agent.append(
            {
                "agent_id": agent_id,
                "rgbd_near_pixel_fraction": rgbd_near_fraction,
                "lidar_near_fraction": lidar_near_fraction,
                "rgbd_min_distance_m": rgbd_min,
                "lidar_min_distance_m": lidar_min,
                "passed": not agent_failures,
                "failures": agent_failures,
            }
        )
    failed_agents = [item["agent_id"] for item in per_agent if not item["passed"]]
    if failed_agents:
        failures.append(f"visual intrusion detected for agents {failed_agents}")
    return {
        "schema": VISUAL_INTRUSION_GATE_SCHEMA,
        "contract": _visual_intrusion_gate_contract(),
        "passed": not failures,
        "failures": failures,
        "per_agent": per_agent,
    }


def _require_onboard_visual_integrity(evidence: Mapping[str, Any]) -> None:
    """Abort before an invalid sensor frame can become a dataset artifact."""

    if evidence.get("passed") is True:
        return
    failures = evidence.get("failures")
    detail = "; ".join(str(value) for value in failures) if isinstance(failures, list) else "unknown failure"
    raise RuntimeError(f"onboard RGB-D/LiDAR visual intrusion gate failed: {detail}")


def _make_multirotor_cfgs(
    args: argparse.Namespace,
    sim_utils: Any,
    MultirotorCfg: Any,
    ThrusterCfg: Any,
    routes_w_m: Sequence[Sequence[Sequence[float]]] = PUBLIC_ROUTES_W_M,
) -> tuple[Any, ...]:
    """Author all eight City-Lite CF2X starts before Isaac initializes PhysX.

    The per-agent literal config pattern follows the upstream md_qd_swarm
    runtime-spawn lifecycle.  Rivermark owns the route-anchor positions and
    headings, so this function deliberately does not claim the unrelated
    upstream frozen-start layout.
    """

    arm_m = 0.046
    yaw_ratio = 0.006
    allocation_matrix = (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (-arm_m, arm_m, arm_m, -arm_m),
        (-arm_m, -arm_m, arm_m, arm_m),
        (yaw_ratio, -yaw_ratio, yaw_ratio, -yaw_ratio),
    )
    states = _city_lite_spawn_states(routes_w_m)
    return tuple(
        MultirotorCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(args.drone_usd.resolve()),
                activate_contact_sensors=True,
                semantic_tags=[
                    ("class", "cf2x"),
                    ("class", "agent_identity"),
                    ("agent_id", str(agent_id)),
                ],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    linear_damping=0.02,
                    angular_damping=0.02,
                    max_linear_velocity=MAX_CF2X_LINEAR_VELOCITY_MPS,
                    max_angular_velocity=MAX_CF2X_ANGULAR_VELOCITY_RADPS,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=MultirotorCfg.InitialStateCfg(
                pos=position,
                rot=quaternion,
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
                # The physical motor model has a non-zero response time.
                # Precharging to the upstream hover trim avoids a free-fall
                # transient without ever rewriting a root state after reset.
                rps={name: float(INITIAL_HOVER_RPS) for name in THRUSTER_NAMES},
            ),
            actuators={
                "thrusters": ThrusterCfg(
                    dt=float(args.dt),
                    thrust_range=(0.0, MAX_THRUST_PER_ROTOR_N),
                    max_thrust_rate=100000.0,
                    thrust_const_range=(
                        THRUST_COEFFICIENT_N_PER_RPS_SQUARED,
                        THRUST_COEFFICIENT_N_PER_RPS_SQUARED,
                    ),
                    tau_inc_range=(0.04, 0.06),
                    tau_dec_range=(0.02, 0.03),
                    torque_to_thrust_ratio=yaw_ratio,
                    thruster_names_expr=list(THRUSTER_NAMES),
                )
            },
            allocation_matrix=allocation_matrix,
            rotor_directions=(1, -1, 1, -1),
        )
        for agent_id, (prim_path, (position, quaternion)) in enumerate(
            zip(SWARM_AGENT_LITERAL_PRIM_PATHS, states, strict=True)
        )
    )


@dataclass(frozen=True)
class _LiteralUsdWorldPose:
    """One pre-physics root transform read from the active USD stage."""

    prim_path: str
    position_w_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    rigid_transform_determinant: float
    basis_axis_lengths: tuple[float, float, float]


@dataclass(frozen=True)
class _RuntimeTargetUsdObservation:
    """One evaluator-private target read from the composed live USD stage.

    This object is intentionally local-only.  ``_audit_runtime_target_usd_authoring``
    converts it into a path-free aggregate before putting anything in a public
    receipt or checkpoint.
    """

    prim_path: str
    position_w_m: tuple[float, float, float]
    radius_m: float
    bound_extents_m: tuple[float, float, float]
    active: bool
    visible: bool
    renderable: bool
    semantic_class_labels: tuple[str, ...]
    rigid_transform_determinant: float
    basis_axis_lengths: tuple[float, float, float]


def _runtime_target_sphere_prim(
    prim: Any, sphere_type: Any, descendants: Any | None = None
) -> Any:
    """Return the unique authored sphere under a target root.

    IsaacLab shape spawners use an Xform root with a ``geometry/mesh`` child;
    accepting only the root's concrete type would reject the native authoring
    that the capture itself creates. OpenUSD exposes subtree traversal through
    ``Usd.PrimRange``; ``Usd.Prim`` itself has no ``GetDescendants`` method.
    The optional iterable keeps the hierarchy rule unit-testable without Isaac.
    """

    if prim.IsA(sphere_type):
        return prim
    if descendants is None:
        from pxr import Usd

        descendants = Usd.PrimRange(prim)
    sphere_descendants = tuple(
        descendant
        for descendant in descendants
        if descendant != prim and descendant.IsA(sphere_type)
    )
    if len(sphere_descendants) != 1:
        raise RuntimeError(
            "runtime target USD root must contain exactly one sphere geometry"
        )
    return sphere_descendants[0]


def _runtime_target_class_labels(
    prim: Any, labels_api_type: Any, descendants: Any | None = None
) -> tuple[str, ...]:
    """Read native class labels from one target subtree.

    IsaacLab's shape spawner applies semantic tags to the authored root, but
    USD composition and Replicator authoring can place the corresponding
    ``SemanticsLabelsAPI:class`` on a geometry descendant.  The audit must
    inspect the composed subtree without treating render-product ID maps as
    authoring evidence.  Repeated copies of the same label are harmless;
    conflicting class labels or private ``target_id`` namespaces are not.
    """

    if descendants is None:
        from pxr import Usd

        descendants = Usd.PrimRange(prim)
    class_labels: set[str] = set()
    target_id_namespace_seen = False
    for candidate in descendants:
        applied = candidate.GetAppliedSchemas()
        for schema_name in applied:
            schema_text = str(schema_name)
            if not schema_text.startswith("SemanticsLabelsAPI:"):
                continue
            instance_name = schema_text.split(":", 1)[1]
            if instance_name == "target_id":
                target_id_namespace_seen = True
                continue
            if instance_name != "class":
                continue
            labels_attr = labels_api_type(candidate, instance_name).GetLabelsAttr()
            labels = labels_attr.Get() if labels_attr else None
            if labels is None:
                continue
            if isinstance(labels, (str, bytes)):
                raise RuntimeError("runtime target USD class semantic label is malformed")
            try:
                label_values = tuple(labels)
            except TypeError as exc:
                raise RuntimeError(
                    "runtime target USD class semantic label is malformed"
                ) from exc
            for label in label_values:
                if not isinstance(label, str) or not label.strip():
                    raise RuntimeError("runtime target USD class semantic label is malformed")
                normalized = label.strip()
                if normalized.lower() == "search_target":
                    raise RuntimeError(
                        "runtime target USD class semantic label is generic"
                    )
                class_labels.add(normalized)
    if target_id_namespace_seen:
        raise RuntimeError("runtime target USD authoring contains a private target-id semantic label")
    if len(class_labels) != 1:
        raise RuntimeError(
            "runtime target USD prim must contain exactly one non-conflicting class semantic label"
        )
    return tuple(sorted(class_labels))


def _read_literal_city_lite_usd_world_poses(stage: Any) -> tuple[_LiteralUsdWorldPose, ...]:
    """Read fresh-stage CF2X world transforms through USD, not Fabric or PhysX."""

    from pxr import Usd, UsdGeom

    # Construct after all eight spawns. A cache created before authoring can
    # retain stale transforms from a previous USD composition query.
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rows: list[_LiteralUsdWorldPose] = []
    for path in SWARM_AGENT_LITERAL_PRIM_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive() or not prim.IsA(UsdGeom.Xformable):
            raise RuntimeError(f"literal CF2X USD root is not an active Xform: {path}")
        matrix = cache.GetLocalToWorldTransform(prim)
        # Extracting rotation from scale/shear can hide an incorrect root
        # transform, so reject it before serializing a pose receipt.
        determinant = float(matrix.GetDeterminant3())
        basis_axis_lengths = tuple(
            math.sqrt(
                sum(float(component) ** 2 for component in matrix.GetRow3(axis))
            )
            for axis in range(3)
        )
        if (
            not bool(matrix.HasOrthogonalRows3())
            or not math.isfinite(determinant)
            or not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-6)
            or any(
                not math.isfinite(length)
                or not math.isclose(
                    length,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=LITERAL_USD_SPAWN_BASIS_LENGTH_TOLERANCE,
                )
                for length in basis_axis_lengths
            )
        ):
            raise RuntimeError(f"literal CF2X USD root is not a rigid transform: {path}")
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        rows.append(
            _LiteralUsdWorldPose(
                prim_path=path,
                position_w_m=tuple(float(translation[axis]) for axis in range(3)),
                quaternion_wxyz=(
                    float(rotation.GetReal()),
                    float(imaginary[0]),
                    float(imaginary[1]),
                    float(imaginary[2]),
                ),
                rigid_transform_determinant=determinant,
                basis_axis_lengths=basis_axis_lengths,
            )
        )
    return tuple(rows)


def _audit_literal_city_lite_usd_spawn_poses(
    observed: Sequence[_LiteralUsdWorldPose],
    routes_w_m: Sequence[Sequence[Sequence[float]]] = PUBLIC_ROUTES_W_M,
) -> dict[str, Any]:
    """Pure-Python route-anchor audit for fresh-stage USD transform reads."""

    expected = _city_lite_spawn_states(routes_w_m)
    if len(observed) != AGENT_COUNT:
        raise RuntimeError("literal CF2X USD spawn audit requires exactly eight world poses")
    rows: list[dict[str, Any]] = []
    for agent_id, (row, path, (expected_position, expected_quaternion)) in enumerate(
        zip(observed, SWARM_AGENT_LITERAL_PRIM_PATHS, expected, strict=True)
    ):
        if row.prim_path != path:
            raise RuntimeError("literal CF2X USD spawn audit received an unstable agent-path order")
        values = (
            *row.position_w_m,
            *row.quaternion_wxyz,
            row.rigid_transform_determinant,
            *row.basis_axis_lengths,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError(f"literal CF2X USD root has non-finite transform: {path}")
        if not math.isclose(
            row.rigid_transform_determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise RuntimeError(f"literal CF2X USD root is not a rigid transform: {path}")
        if len(row.basis_axis_lengths) != 3 or any(
            not math.isclose(
                length,
                1.0,
                rel_tol=0.0,
                abs_tol=LITERAL_USD_SPAWN_BASIS_LENGTH_TOLERANCE,
            )
            for length in row.basis_axis_lengths
        ):
            raise RuntimeError(f"literal CF2X USD root has non-unit basis scale: {path}")
        observed_norm = math.sqrt(sum(value * value for value in row.quaternion_wxyz))
        expected_norm = math.sqrt(sum(value * value for value in expected_quaternion))
        if observed_norm <= 1.0e-12 or expected_norm <= 1.0e-12:
            raise RuntimeError(f"literal CF2X USD root has a degenerate orientation: {path}")
        position_error = math.dist(row.position_w_m, expected_position)
        orientation_cosine = abs(
            sum(
                row.quaternion_wxyz[axis] * expected_quaternion[axis]
                for axis in range(4)
            )
            / (observed_norm * expected_norm)
        )
        orientation_error = math.acos(
            max(-1.0, min(1.0, 2.0 * orientation_cosine**2 - 1.0))
        )
        if (
            position_error > LITERAL_USD_SPAWN_POSITION_TOLERANCE_M
            or orientation_error > LITERAL_USD_SPAWN_ORIENTATION_TOLERANCE_RAD
        ):
            raise RuntimeError(
                "literal City-Lite USD root differs from its route anchor "
                f"({path}: position={position_error:.8f} m, "
                f"orientation={orientation_error:.8f} rad)"
            )
        rows.append(
            {
                "agent_id": agent_id,
                "prim_path": path,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "rigid_transform_determinant": row.rigid_transform_determinant,
                "basis_axis_lengths": list(row.basis_axis_lengths),
            }
        )
    return {
        "source": "fresh_stage_usd_xform_cache_before_sim_reset",
        "position_tolerance_m": LITERAL_USD_SPAWN_POSITION_TOLERANCE_M,
        "orientation_tolerance_rad": LITERAL_USD_SPAWN_ORIENTATION_TOLERANCE_RAD,
        "per_agent": rows,
        "max_position_error_m": max(float(row["position_error_m"]) for row in rows),
        "max_orientation_error_rad": max(
            float(row["orientation_error_rad"]) for row in rows
        ),
    }


def _verify_literal_city_lite_usd_spawn(
    stage: Any,
    routes_w_m: Sequence[Sequence[Sequence[float]]] = PUBLIC_ROUTES_W_M,
) -> dict[str, Any]:
    """Read and audit the eight literal USD roots before PhysX initialization."""

    return _audit_literal_city_lite_usd_spawn_poses(
        _read_literal_city_lite_usd_world_poses(stage),
        routes_w_m,
    )


def _read_runtime_target_usd_authoring(
    stage: Any, target_paths: Sequence[str]
) -> tuple[_RuntimeTargetUsdObservation, ...]:
    """Read target authoring from USD without exposing it to the policy path."""

    from pxr import Usd, UsdGeom, UsdSemantics

    if not target_paths:
        raise RuntimeError("runtime target USD audit requires at least one target path")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    observations: list[_RuntimeTargetUsdObservation] = []
    for path in target_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError("runtime target USD prim is missing")
        # IsaacLab's ShapeCfg deliberately authors a rigid-body Xform root and
        # puts the geometry at ``{root}/geometry/mesh``.  The root carries the
        # authored translation, rigid-body APIs and semantic LabelsAPI, while
        # the child carries the sphere radius.  A direct Sphere is still
        # accepted for older/simple stages, but a wrapped target must contain
        # exactly one sphere descendant so a malformed or ambiguous asset
        # fails closed instead of silently using the wrong geometry.
        sphere_prim = _runtime_target_sphere_prim(
            prim, UsdGeom.Sphere, descendants=Usd.PrimRange(prim)
        )
        sphere = UsdGeom.Sphere(sphere_prim)
        radius = sphere.GetRadiusAttr().Get()
        if radius is None:
            raise RuntimeError("runtime target USD sphere has no radius")
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        determinant = float(matrix.GetDeterminant3())
        basis_axis_lengths = tuple(
            math.sqrt(sum(float(component) ** 2 for component in matrix.GetRow3(axis)))
            for axis in range(3)
        )
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if aligned.IsEmpty():
            bound_extents = (float("nan"),) * 3
        else:
            lower, upper = aligned.GetMin(), aligned.GetMax()
            bound_extents = tuple(float(upper[axis] - lower[axis]) for axis in range(3))
        translation = matrix.ExtractTranslation()
        imageable = UsdGeom.Imageable(prim)
        # IsaacLab 2.3 writes ``semantic_tags`` through the Isaac Sim 5
        # LabelsAPI.  Read the composed target subtree using the same applied
        # schema contract instead of inferring labels from render-product ID
        # maps, whose numeric namespace is local to each camera.
        class_labels = _runtime_target_class_labels(prim, UsdSemantics.LabelsAPI)
        observations.append(
            _RuntimeTargetUsdObservation(
                prim_path=str(prim.GetPath()),
                position_w_m=tuple(float(translation[axis]) for axis in range(3)),
                radius_m=float(radius),
                bound_extents_m=bound_extents,
                active=bool(prim.IsActive()),
                visible=(imageable.ComputeVisibility() != UsdGeom.Tokens.invisible),
                renderable=not aligned.IsEmpty(),
                semantic_class_labels=tuple(str(label) for label in class_labels),
                rigid_transform_determinant=determinant,
                basis_axis_lengths=basis_axis_lengths,
            )
        )
    return tuple(observations)


def _audit_runtime_target_usd_authoring(
    observed: Sequence[_RuntimeTargetUsdObservation],
    evaluator_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless USD target authoring exactly matches private truth.

    The returned receipt contains no per-target detail because even an index
    paired with a position, radius, or semantic slot would disclose evaluator
    information.  It proves only that every expected target passed the same
    runtime closure at the named capture phase.
    """

    targets = evaluator_manifest.get("targets")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise RuntimeError("runtime target USD audit has no private target list")
    if len(observed) != len(targets) or not observed:
        raise RuntimeError("runtime target USD audit count differs from private manifest")

    maximum_position_error = 0.0
    maximum_radius_error = 0.0
    maximum_bound_extent_error = 0.0
    all_active = True
    all_visible = True
    all_renderable = True
    all_expected_semantic_labels = True
    all_rigid = True
    for index, (row, target) in enumerate(zip(observed, targets, strict=True)):
        if not isinstance(target, Mapping):
            raise RuntimeError("runtime target USD audit has malformed private target")
        expected_path = f"/World/SearchTargets/Target_{index}"
        if row.prim_path != expected_path:
            raise RuntimeError("runtime target USD audit has unstable target-path order")
        raw_position = target.get("position_w_m")
        raw_radius = target.get("radius_m")
        if (
            not isinstance(raw_position, Sequence)
            or isinstance(raw_position, (str, bytes))
            or len(raw_position) != 3
            or not isinstance(raw_radius, (int, float))
        ):
            raise RuntimeError("runtime target USD audit has malformed private target geometry")
        values = (
            *row.position_w_m,
            row.radius_m,
            *row.bound_extents_m,
            row.rigid_transform_determinant,
            *row.basis_axis_lengths,
            *(float(value) for value in raw_position),
            float(raw_radius),
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("runtime target USD audit found non-finite target geometry")
        expected_position = tuple(float(value) for value in raw_position)
        expected_radius = float(raw_radius)
        expected_semantic_label = _target_semantic_slots(len(targets))[index]
        position_error = math.dist(row.position_w_m, expected_position)
        radius_error = abs(row.radius_m - expected_radius)
        bound_extent_error = max(
            abs(extent - 2.0 * expected_radius) for extent in row.bound_extents_m
        )
        rigid = (
            math.isclose(
                row.rigid_transform_determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-6
            )
            and len(row.basis_axis_lengths) == 3
            and all(
                math.isclose(
                    length,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=LITERAL_USD_SPAWN_BASIS_LENGTH_TOLERANCE,
                )
                for length in row.basis_axis_lengths
            )
        )
        all_active = all_active and row.active
        all_visible = all_visible and row.visible
        all_renderable = all_renderable and row.renderable
        all_expected_semantic_labels = (
            all_expected_semantic_labels
            and tuple(row.semantic_class_labels) == (expected_semantic_label,)
        )
        all_rigid = all_rigid and rigid
        maximum_position_error = max(maximum_position_error, position_error)
        maximum_radius_error = max(maximum_radius_error, radius_error)
        maximum_bound_extent_error = max(maximum_bound_extent_error, bound_extent_error)

    if not all_active:
        raise RuntimeError("runtime target USD audit found an inactive target")
    if not all_visible:
        raise RuntimeError("runtime target USD audit found an invisible target")
    if not all_renderable:
        raise RuntimeError("runtime target USD audit found an unrenderable target")
    if not all_expected_semantic_labels:
        raise RuntimeError("runtime target USD audit found a missing or mismatched class semantic label")
    if not all_rigid:
        raise RuntimeError("runtime target USD audit found a non-rigid target transform")
    if maximum_position_error > RUNTIME_TARGET_USD_POSITION_TOLERANCE_M:
        raise RuntimeError("runtime target USD position differs from private manifest")
    if maximum_radius_error > RUNTIME_TARGET_USD_RADIUS_TOLERANCE_M:
        raise RuntimeError("runtime target USD radius differs from private manifest")
    if maximum_bound_extent_error > RUNTIME_TARGET_USD_BOUND_EXTENT_TOLERANCE_M:
        raise RuntimeError("runtime target USD bound differs from private manifest")
    return {
        "schema": "org.rivermark.runtime-target-usd-closure.v1",
        "target_count": len(observed),
        "all_targets_active": all_active,
        "all_targets_visible": all_visible,
        "all_targets_renderable": all_renderable,
        "all_targets_have_expected_class_label": all_expected_semantic_labels,
        "all_target_transforms_rigid": all_rigid,
        "maximum_world_position_error_m": maximum_position_error,
        "maximum_radius_error_m": maximum_radius_error,
        "maximum_bound_extent_error_m": maximum_bound_extent_error,
        "position_tolerance_m": RUNTIME_TARGET_USD_POSITION_TOLERANCE_M,
        "radius_tolerance_m": RUNTIME_TARGET_USD_RADIUS_TOLERANCE_M,
        "bound_extent_tolerance_m": RUNTIME_TARGET_USD_BOUND_EXTENT_TOLERANCE_M,
    }


def _verify_runtime_target_usd_authoring(
    stage: Any, target_paths: Sequence[str], evaluator_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Read then redact the runtime target closure in one capture-only helper."""

    return _audit_runtime_target_usd_authoring(
        _read_runtime_target_usd_authoring(stage, target_paths), evaluator_manifest
    )


def _verify_literal_city_lite_spawn(
    robot: Any,
    expected_root_states: Any,
    expected_thruster_rps: Any,
    torch: Any,
) -> dict[str, Any]:
    """Audit authored defaults separately from reset-time physical settling.

    ``SimulationContext.reset()`` advances PhysX before the multirotor facade
    can issue its first wrench.  With gravity enabled, that can legitimately
    yield a small downward position delta and non-zero velocity even though the
    USD spawn transform and configured default velocity are correct.  Do not
    conceal that behavior with a root-state rewrite or by treating a live
    velocity as a configuration error.  Instead, prove the resolved defaults
    exactly, retain a tight live pose/orientation closure, record the physical
    settling evidence, and let the immediately following runtime safety guard
    reject contact, geometry, volume, and inter-agent violations.
    """

    observed_position = robot.data.root_pos_w
    observed_quaternion = robot.data.root_quat_w
    linear_velocity = robot.data.root_lin_vel_w
    angular_velocity = robot.data.root_ang_vel_b
    default_root_state = robot.data.default_root_state
    default_thruster_rps = robot.data.default_thruster_rps
    reset_thrust_target = robot.data.thrust_target
    expected_position = expected_root_states[:, :3]
    expected_quaternion = expected_root_states[:, 3:7]
    expected_hover_thrust = (
        expected_thruster_rps.square() * THRUST_COEFFICIENT_N_PER_RPS_SQUARED
    )
    expected_shapes = (
        ("root position", observed_position, (AGENT_COUNT, 3)),
        ("root quaternion", observed_quaternion, (AGENT_COUNT, 4)),
        ("root linear velocity", linear_velocity, (AGENT_COUNT, 3)),
        ("root angular velocity", angular_velocity, (AGENT_COUNT, 3)),
        ("default root state", default_root_state, (AGENT_COUNT, 13)),
        ("default thruster RPS", default_thruster_rps, (AGENT_COUNT, 4)),
        ("reset thrust target", reset_thrust_target, (AGENT_COUNT, 4)),
        ("expected root state", expected_root_states, (AGENT_COUNT, 13)),
        ("expected thruster RPS", expected_thruster_rps, (AGENT_COUNT, 4)),
    )
    for label, value, shape in expected_shapes:
        if tuple(value.shape) != shape:
            raise RuntimeError(
                f"literal CF2X fleet {label} did not expose the expected {shape} ABI"
            )

    root_state_max_abs_error = float(
        torch.max(torch.abs(default_root_state - expected_root_states)).item()
    )
    thruster_rps_max_abs_error = float(
        torch.max(torch.abs(default_thruster_rps - expected_thruster_rps)).item()
    )
    thrust_target_max_abs_error_n = float(
        torch.max(torch.abs(reset_thrust_target - expected_hover_thrust)).item()
    )
    authored_errors = (
        root_state_max_abs_error,
        thruster_rps_max_abs_error,
        thrust_target_max_abs_error_n,
    )
    if (
        not all(math.isfinite(value) for value in authored_errors)
        or root_state_max_abs_error > LITERAL_SPAWN_DEFAULT_STATE_TOLERANCE
        or thruster_rps_max_abs_error > LITERAL_SPAWN_DEFAULT_RPS_TOLERANCE
        or thrust_target_max_abs_error_n > LITERAL_SPAWN_DEFAULT_THRUST_TOLERANCE_N
    ):
        raise RuntimeError(
            "literal City-Lite CF2X authored defaults differ from the frozen init_state "
            f"(root_state_abs_error={root_state_max_abs_error:.8f}, "
            f"thruster_rps_abs_error={thruster_rps_max_abs_error:.8f}, "
            f"reset_thrust_abs_error={thrust_target_max_abs_error_n:.8f} N)"
        )

    position_error = torch.linalg.vector_norm(
        observed_position - expected_position, dim=-1
    )
    observed_unit = observed_quaternion / torch.linalg.vector_norm(
        observed_quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    expected_unit = expected_quaternion / torch.linalg.vector_norm(
        expected_quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    orientation_error = torch.acos(
        torch.clamp(
            2.0 * torch.sum(observed_unit * expected_unit, dim=-1).abs().square()
            - 1.0,
            -1.0,
            1.0,
        )
    )
    linear_velocity_norm = torch.linalg.vector_norm(linear_velocity, dim=-1)
    angular_velocity_norm = torch.linalg.vector_norm(angular_velocity, dim=-1)
    max_position_delta_m = float(torch.max(position_error).item())
    max_orientation_delta_rad = float(torch.max(orientation_error).item())
    max_linear_velocity_mps = float(torch.max(linear_velocity_norm).item())
    max_angular_velocity_radps = float(torch.max(angular_velocity_norm).item())
    settling_values = (
        max_position_delta_m,
        max_orientation_delta_rad,
        max_linear_velocity_mps,
        max_angular_velocity_radps,
    )
    if (
        not all(math.isfinite(value) for value in settling_values)
        or max_position_delta_m > LITERAL_SPAWN_POSITION_TOLERANCE_M
        or max_orientation_delta_rad > LITERAL_SPAWN_ORIENTATION_TOLERANCE_RAD
        or max_linear_velocity_mps > MAX_CF2X_LINEAR_VELOCITY_MPS
        or max_angular_velocity_radps > MAX_CF2X_ANGULAR_VELOCITY_RADPS
    ):
        raise RuntimeError(
            "literal City-Lite CF2X post-reset physics state is outside the spawn contract "
            f"(position_delta={max_position_delta_m:.6f} m, "
            f"orientation_delta={max_orientation_delta_rad:.6f} rad, "
            f"linear_velocity={max_linear_velocity_mps:.6f} m/s, "
            f"angular_velocity={max_angular_velocity_radps:.6f} rad/s)"
        )
    return {
        "literal_prim_paths": list(SWARM_AGENT_LITERAL_PRIM_PATHS),
        "authored_defaults": {
            "root_state_shape": [AGENT_COUNT, 13],
            "thruster_rps_shape": [AGENT_COUNT, 4],
            "thrust_target_shape": [AGENT_COUNT, 4],
            "root_state_max_abs_error": root_state_max_abs_error,
            "thruster_rps_max_abs_error": thruster_rps_max_abs_error,
            "thrust_target_max_abs_error_n": thrust_target_max_abs_error_n,
            "root_state_tolerance": LITERAL_SPAWN_DEFAULT_STATE_TOLERANCE,
            "thruster_rps_tolerance": LITERAL_SPAWN_DEFAULT_RPS_TOLERANCE,
            "thrust_target_tolerance_n": LITERAL_SPAWN_DEFAULT_THRUST_TOLERANCE_N,
        },
        "post_reset_physics_settling": {
            "classification": "observed_after_sim_reset_before_first_command",
            "max_position_delta_m": max_position_delta_m,
            "max_orientation_delta_rad": max_orientation_delta_rad,
            "max_linear_velocity_mps": max_linear_velocity_mps,
            "max_angular_velocity_radps": max_angular_velocity_radps,
            "position_tolerance_m": LITERAL_SPAWN_POSITION_TOLERANCE_M,
            "orientation_tolerance_rad": LITERAL_SPAWN_ORIENTATION_TOLERANCE_RAD,
            "linear_velocity_hard_limit_mps": MAX_CF2X_LINEAR_VELOCITY_MPS,
            "angular_velocity_hard_limit_radps": MAX_CF2X_ANGULAR_VELOCITY_RADPS,
        },
        "post_reset_root_pose_rewrite": False,
        "post_reset_root_velocity_rewrite": False,
        "anchor_contract": "rivermark_public_route_initial_waypoints",
    }


def _collision_enabled(prim: Any) -> bool:
    try:
        if "PhysicsCollisionAPI" in {str(schema) for schema in prim.GetAppliedSchemas()}:
            return True
    except Exception:
        pass
    attribute = prim.GetAttribute("physics:collisionEnabled")
    if attribute and attribute.HasAuthoredValueOpinion():
        try:
            return bool(attribute.Get())
        except Exception:
            return True
    return False


def _city_lite_ground_like(path: str) -> bool:
    lowered = path.lower()
    return any(
        token in lowered
        for token in (
            "ground",
            "terrain",
            "road",
            "street",
            "asphalt",
            "sidewalk",
            "crosswalk",
            "lane",
            "curb",
            "foundation",
            "landscape",
            "grass",
        )
    )


def _city_lite_helper(path: str) -> bool:
    lowered = path.lower()
    return any(
        token in lowered
        for token in (
            "/mission/",
            "/drones/",
            "/materials",
            "/render",
            "/lights",
            "targetmarker",
            "target_marker",
            "debug",
            "lamppost",
            "lamp_post",
            "cylinderlight",
            "rectlight",
            "disklight",
            "telephone_wire",
        )
    )


def _city_lite_structural(path: str) -> bool:
    lowered = path.lower()
    return path.startswith("/World/StaticScene/CityTaskObstacles") or any(
        token in lowered
        for token in (
            "building",
            "facade",
            "wall",
            "tower",
            "roof",
            "structure",
            "block",
            "garage",
            "shed",
            "rubble",
            "debris",
            "skybridge",
        )
    )


def _box_overlaps_city_lite_scan(minimum: tuple[float, float, float], maximum: tuple[float, float, float]) -> bool:
    # Route and collision-proxy source geometry needs a little vertical context
    # around the allowed flight envelope; the command itself remains stricter.
    return not (
        maximum[0] < CITY_LITE_FLIGHT_VOLUME_W_M.minimum[0]
        or minimum[0] > CITY_LITE_FLIGHT_VOLUME_W_M.maximum[0]
        or maximum[1] < CITY_LITE_FLIGHT_VOLUME_W_M.minimum[1]
        or minimum[1] > CITY_LITE_FLIGHT_VOLUME_W_M.maximum[1]
        or maximum[2] < CITY_LITE_FLIGHT_VOLUME_W_M.minimum[2] - 2.0
        or minimum[2] > CITY_LITE_FLIGHT_VOLUME_W_M.maximum[2] + 2.0
    )


def _native_collision_counts(stage: Any) -> dict[str, int]:
    collision_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsActive() and _collision_enabled(prim)
    ]
    return {
        "total": len(collision_paths),
        "drivable_surfaces": sum("/drivable_surfaces/" in path for path in collision_paths),
        "city_task_obstacles": sum(
            path.startswith("/World/StaticScene/CityTaskObstacles/")
            for path in collision_paths
        ),
        "structural_props": sum(
            path.startswith("/World/StaticScene/City/Rivermark/props/")
            for path in collision_paths
        ),
    }


def _author_city_task_obstacle_material_repair(stage: Any) -> None:
    """Author the eight source-faithful local materials before reference loading.

    The final City-Lite USDA binds these meshes to absolute material paths that
    are deliberately outside the two admitted reference scopes.  A local
    explicit binding authored in the stronger root layer prevents the scoped
    geometry from silently falling back to an unbound display color.
    """

    from pxr import Gf, Sdf, UsdGeom, UsdShade

    contract = city_task_obstacle_material_contract_payload()
    UsdGeom.Scope.Define(stage, contract["material_root"])
    for binding in contract["bindings"]:
        local_material_path = binding["local_material_prim"]
        shader_path = binding["shader_prim"]
        material = UsdShade.Material.Define(stage, local_material_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr().Set(binding["shader_id"])
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*binding["diffuse_color"])
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(binding["opacity"])
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
            binding["roughness"]
        )
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        if not material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        ):
            raise RuntimeError(
                f"could not connect local CityTaskObstacle material surface: {local_material_path}"
            )

        # The child can be an unloaded referenced prim at this point.  A local
        # `over` is intentional: it has stronger opinion than the invalid
        # absolute source binding once the selective reference is loaded.
        obstacle = stage.OverridePrim(binding["obstacle_prim"])
        if not obstacle or not obstacle.IsValid():
            raise RuntimeError(
                f"could not create local CityTaskObstacle binding over: {binding['obstacle_prim']}"
            )
        if not UsdShade.MaterialBindingAPI.Apply(obstacle).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        ):
            raise RuntimeError(
                f"could not bind local CityTaskObstacle material: {binding['obstacle_prim']}"
            )


def _validate_city_task_obstacle_material_closure(
    stage: Any,
    composition_diagnostics: Sequence[str],
) -> dict[str, Any]:
    """Verify direct local material resolution after City-Lite composition."""

    from pxr import UsdShade

    contract = city_task_obstacle_material_contract_payload()
    material_root = str(contract["material_root"])
    root_prim = stage.GetPrimAtPath(material_root)
    if not root_prim or not root_prim.IsValid() or not root_prim.IsActive():
        raise RuntimeError("CityTaskObstacle local material root is missing after composition")

    expected_material_paths = {
        str(binding["local_material_prim"]) for binding in contract["bindings"]
    }
    observed_material_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{material_root}/")
        and str(prim.GetTypeName()) == "Material"
    }
    if observed_material_paths != expected_material_paths:
        raise RuntimeError(
            "CityTaskObstacle material repair must contain exactly the eight allow-listed "
            f"local materials; observed={sorted(observed_material_paths)}"
        )
    source_material_root = stage.GetPrimAtPath("/World/Materials")
    if source_material_root and source_material_root.IsValid() and source_material_root.IsActive():
        raise RuntimeError(
            "CityTaskObstacle material repair must not import the legacy /World/Materials root"
        )

    observed_bindings: list[dict[str, Any]] = []
    for binding in contract["bindings"]:
        obstacle_path = str(binding["obstacle_prim"])
        obstacle = stage.GetPrimAtPath(obstacle_path)
        if not obstacle or not obstacle.IsValid() or not obstacle.IsActive():
            raise RuntimeError(f"CityTaskObstacle prim is missing after composition: {obstacle_path}")
        material = UsdShade.Material.Get(stage, binding["local_material_prim"])
        if not material or not material.GetPrim().IsValid() or not material.GetPrim().IsActive():
            raise RuntimeError(
                f"CityTaskObstacle local material is missing after composition: {binding['local_material_prim']}"
            )

        direct_targets = [
            str(path)
            for path in UsdShade.MaterialBindingAPI(obstacle)
            .GetDirectBindingRel()
            .GetTargets()
        ]
        if direct_targets != [binding["local_material_prim"]]:
            raise RuntimeError(
                f"CityTaskObstacle direct material binding is not closed for {obstacle_path}: "
                f"{direct_targets}"
            )
        bound_material, _ = UsdShade.MaterialBindingAPI(obstacle).ComputeBoundMaterial()
        if (
            not bound_material
            or not bound_material.GetPrim().IsValid()
            or str(bound_material.GetPath()) != binding["local_material_prim"]
        ):
            raise RuntimeError(
                f"CityTaskObstacle resolved material is not local for {obstacle_path}"
            )

        shader = UsdShade.Shader.Get(stage, binding["shader_prim"])
        if not shader or not shader.GetPrim().IsValid() or str(shader.GetIdAttr().Get()) != binding[
            "shader_id"
        ]:
            raise RuntimeError(
                f"CityTaskObstacle shader is invalid for {binding['local_material_prim']}"
            )
        color = shader.GetInput("diffuseColor").Get()
        if color is None or any(
            not math.isclose(
                float(color[axis]),
                float(binding["diffuse_color"][axis]),
                rel_tol=0.0,
                abs_tol=1.0e-7,
            )
            for axis in range(3)
        ):
            raise RuntimeError(
                f"CityTaskObstacle diffuse color differs from authority for {obstacle_path}"
            )
        for input_name in ("opacity", "roughness"):
            value = shader.GetInput(input_name).Get()
            if value is None or not math.isclose(
                float(value), float(binding[input_name]), rel_tol=0.0, abs_tol=1.0e-7
            ):
                raise RuntimeError(
                    f"CityTaskObstacle {input_name} differs from authority for {obstacle_path}"
                )
        connections = [
            str(path) for path in material.GetSurfaceOutput().GetAttr().GetConnections()
        ]
        if connections != [binding["surface_output_connection"]]:
            raise RuntimeError(
                f"CityTaskObstacle material surface connection is invalid for {obstacle_path}: "
                f"{connections}"
            )
        observed_bindings.append(
            {
                "obstacle_prim": obstacle_path,
                "local_material_prim": binding["local_material_prim"],
                "surface_output_connection": binding["surface_output_connection"],
                "resolved": True,
            }
        )

    source_scope_warnings = [
        message
        for message in composition_diagnostics
        if "outside the scope of the reference" in message.lower()
        and "/world/materials/" in message.lower()
    ]
    return {
        **contract,
        "contract_sha256": CITY_TASK_OBSTACLE_MATERIAL_CONTRACT_SHA256,
        "post_repair_binding_closure": True,
        "observed_bindings": observed_bindings,
        "source_scope_diagnostics": {
            "known_external_material_binding_count": contract["binding_count"],
            "repair_applied_before_stage_load": True,
            "reported_warning_count": len(source_scope_warnings),
            "reported_warnings": source_scope_warnings,
        },
    }


def _compose_city_lite_stage(authority: CityLiteAuthority) -> tuple[Any, dict[str, Any]]:
    """Compose only the two admitted City-Lite prims into the fresh Kit stage."""

    import omni.usd
    from pxr import Tf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Isaac did not provide a writable USD stage")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetTimeCodesPerSecond(60.0)
    stage.SetFramesPerSecond(60.0)
    UsdGeom.Xform.Define(stage, "/World/StaticScene")

    mark = Tf.Error.Mark()
    mark.SetMark()
    for source_prim, destination_prim in SELECTIVE_REFERENCES:
        destination = UsdGeom.Xform.Define(stage, destination_prim).GetPrim()
        if not destination.GetReferences().AddReference(
            str(authority.final_scene_path), source_prim
        ):
            raise RuntimeError(
                f"could not selectively reference {source_prim} into {destination_prim}"
            )
    _author_city_task_obstacle_material_repair(stage)
    stage.Load()
    diagnostics = [str(error) for error in mark.GetErrors()]
    unresolved = [
        message
        for message in diagnostics
        if any(token in message.lower() for token in ("unresolved", "could not open", "failed to open"))
    ]
    if unresolved:
        raise RuntimeError(
            "active City-Lite selective composition has unresolved references: "
            + " | ".join(unresolved[:4])
        )
    material_closure = _validate_city_task_obstacle_material_closure(stage, diagnostics)

    used_layer_identifiers: list[str] = []
    for layer in stage.GetUsedLayers():
        # `realPath` is empty for Kit's anonymous root/session layers. Preserve
        # their `anon:` identifier so the inventory can count, but not hash,
        # them. Every non-anonymous layer is resolved and hash-bound below.
        identifier = str(
            getattr(layer, "realPath", "") or getattr(layer, "identifier", "")
        ).strip()
        if not identifier:
            raise RuntimeError("City-Lite stage returned an unnamed used USD layer")
        used_layer_identifiers.append(identifier)
    rivermark_layer_inventory = make_rivermark_layer_inventory(
        authority,
        used_layer_identifiers,
    )
    validate_rivermark_layer_inventory_receipt(rivermark_layer_inventory)

    static_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsActive() and str(prim.GetPath()).startswith("/World/StaticScene/")
    ]
    forbidden = forbidden_scene_paths(static_paths)
    if forbidden:
        raise RuntimeError(
            "City-Lite selective composition contains legacy/decorative prims: "
            + ", ".join(forbidden[:8])
        )
    counts = _native_collision_counts(stage)
    if counts != dict(EXPECTED_NATIVE_COLLISION_COUNTS):
        raise RuntimeError(
            "City-Lite native collision audit differs from the pinned v1_r2 authority: "
            f"{counts}"
        )
    return stage, {
        "unresolved_reference_count": len(unresolved),
        "unresolved_reference_diagnostics": unresolved,
        "active_static_prim_paths": static_paths,
        "active_static_prim_count": len(static_paths),
        "legacy_prim_count": 0,
        "forbidden_decoration_prim_count": 0,
        "city_task_obstacle_material_closure": material_closure,
        "native_collision_counts": counts,
        "rivermark_layer_inventory": rivermark_layer_inventory,
    }


def _extract_structural_aabbs(stage: Any) -> tuple[AABB, ...]:
    """Extract conservative active-stage AABBs without claiming mesh fidelity."""

    from pxr import Sdf, Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    all_prims = [prim for prim in stage.Traverse() if prim.IsActive()]
    collision_paths = {
        str(prim.GetPath()) for prim in all_prims if _collision_enabled(prim)
    }
    collision_ancestors: set[str] = set()
    for path in collision_paths:
        parent = Sdf.Path(path).GetParentPath()
        while parent and parent != Sdf.Path.absoluteRootPath:
            text = str(parent)
            if text in collision_paths:
                collision_ancestors.add(text)
            parent = parent.GetParentPath()

    selected: dict[tuple[tuple[float, float, float], tuple[float, float, float]], AABB] = {}
    city_count = task_count = 0
    for prim in all_prims:
        path = str(prim.GetPath())
        if not (
            path.startswith("/World/StaticScene/City/Rivermark/")
            or path.startswith("/World/StaticScene/CityTaskObstacles/")
        ):
            continue
        if not prim.IsA(UsdGeom.Boundable) or _city_lite_ground_like(path) or _city_lite_helper(path):
            continue
        collision_leaf = path in collision_paths and path not in collision_ancestors
        if not (collision_leaf or _city_lite_structural(path)):
            continue
        try:
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if aligned.IsEmpty():
                continue
            lower, upper = aligned.GetMin(), aligned.GetMax()
            minimum = (float(lower[0]), float(lower[1]), float(lower[2]))
            maximum = (float(upper[0]), float(upper[1]), float(upper[2]))
        except Exception as error:
            raise RuntimeError(f"cannot bound City-Lite structural prim {path}: {error}") from error
        if not _box_overlaps_city_lite_scan(minimum, maximum):
            continue
        source_kind = (
            "city_task_obstacle"
            if path.startswith("/World/StaticScene/CityTaskObstacles/")
            else "rivermark_structural_visual"
        )
        box = AABB(minimum, maximum, source_prim=path, category=source_kind)
        key = (box.minimum, box.maximum)
        previous = selected.get(key)
        if previous is None or (box.category, box.source_prim) < (
            previous.category,
            previous.source_prim,
        ):
            selected[key] = box

    boxes = tuple(
        sorted(selected.values(), key=lambda box: (box.category, box.source_prim))
    )
    city_count = sum(box.category == "rivermark_structural_visual" for box in boxes)
    task_count = sum(box.category == "city_task_obstacle" for box in boxes)
    if not boxes or city_count == 0 or task_count != EXPECTED_NATIVE_COLLISION_COUNTS["city_task_obstacles"]:
        raise RuntimeError(
            "City-Lite structural AABB audit is incomplete: "
            f"total={len(boxes)}, city={city_count}, task={task_count}"
        )
    return boxes


def _spawn_collision_proxies(stage: Any, boxes: Sequence[AABB]) -> list[str]:
    """Create one invisible static collision cube per conservative source AABB."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    UsdGeom.Xform.Define(stage, COLLISION_PROXY_ROOT)
    paths: list[str] = []
    for index, box in enumerate(boxes):
        path = f"{COLLISION_PROXY_ROOT}/StructuralAabb_{index:04d}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        center = tuple((box.minimum[axis] + box.maximum[axis]) * 0.5 for axis in range(3))
        size = tuple(box.maximum[axis] - box.minimum[axis] for axis in range(3))
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
        collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision.CreateCollisionEnabledAttr(True)
        cube.GetPrim().CreateAttribute(
            "rivermark:sourcePrim", Sdf.ValueTypeNames.String
        ).Set(box.source_prim)
        cube.GetPrim().CreateAttribute(
            "rivermark:representation", Sdf.ValueTypeNames.String
        ).Set(
            COLLISION_PROXY_REPRESENTATION
        )
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        paths.append(path)
    return paths


def _make_scene(
    sim_utils: Any, evaluator_manifest: Mapping[str, Any] | None
) -> list[str]:
    import isaacsim.core.utils.prims as prim_utils

    prim_utils.create_prim("/World/Swarm", "Xform")
    for index in range(AGENT_COUNT):
        prim_utils.create_prim(f"/World/Swarm/Agent_{index}", "Xform")

    target_paths: list[str] = []
    targets = () if evaluator_manifest is None else evaluator_manifest["targets"]
    target_slots = _target_semantic_slots(len(targets))
    for index, target in enumerate(targets):
        path = f"/World/SearchTargets/Target_{index}"
        color = (0.95, 0.05, 0.75)
        cfg = sim_utils.SphereCfg(
            radius=float(target["radius_m"]),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                emissive_color=color,
                metallic=0.0,
                roughness=0.25,
            ),
            semantic_tags=[
                ("class", target_slots[index]),
            ],
        )
        cfg.func(
            path,
            cfg,
            translation=tuple(float(value) for value in target["position_w_m"]),
        )
        target_paths.append(path)
    light = sim_utils.DomeLightCfg(intensity=3500.0, color=(0.78, 0.84, 0.95))
    light.func("/World/Light", light)
    return target_paths


def _spawn_identity_markers(sim_utils: Any) -> list[str]:
    """Attach a visible, collision-free identity marker to each physical CF2X body."""

    marker_paths: list[str] = []
    for agent_id, color in enumerate(IDENTITY_COLORS):
        path = f"/World/Swarm/Agent_{agent_id}/Robot/body/identity_marker"
        cfg = sim_utils.SphereCfg(
            radius=IDENTITY_MARKER_RADIUS_M,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                emissive_color=tuple(component * 0.25 for component in color),
                metallic=0.0,
                roughness=0.3,
            ),
            semantic_tags=[("class", "agent_identity"), ("agent_id", str(agent_id))],
        )
        cfg.func(path, cfg, translation=(-0.045, 0.0, 0.075))
        marker_paths.append(path)
    return marker_paths


def _make_sensors(
    args: argparse.Namespace,
    sim_utils: Any,
    target_paths: list[str],
    *,
    include_onboard_camera: bool = True,
    include_overview_camera: bool = True,
    use_tiled_onboard_camera: bool = False,
    use_tiled_overview_camera: bool = False,
) -> tuple[Any, ...]:
    from isaaclab.sensors import (
        Camera,
        CameraCfg,
        ContactSensor,
        ContactSensorCfg,
        Imu,
        ImuCfg,
    )
    from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg, patterns

    if use_tiled_onboard_camera or use_tiled_overview_camera:
        from isaaclab.sensors import TiledCamera, TiledCameraCfg

    # CameraCfg.offset is the native local CF2X mount.  Request IsaacLab's
    # documented post-render pose update so pose telemetry and annotator reads
    # are refreshed together by Camera.update().
    onboard_cfg_class = TiledCameraCfg if use_tiled_onboard_camera else CameraCfg
    onboard_cfg = onboard_cfg_class(
        prim_path="/World/Swarm/Agent_.*/Robot/body/onboard_camera",
        offset=onboard_cfg_class.OffsetCfg(
            pos=CAMERA_OFFSET_BODY_M, rot=CAMERA_OFFSET_WXYZ, convention="world"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=ONBOARD_FOCAL_LENGTH_MM,
            horizontal_aperture=ONBOARD_HORIZONTAL_APERTURE_MM,
            clipping_range=ONBOARD_CAMERA_CLIPPING_RANGE_M,
        ),
        width=args.onboard_width,
        height=args.onboard_height,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        depth_clipping_behavior="max",
        update_latest_camera_pose=True,
    )
    # Use a dedicated Camera for the singleton overview. Its transform is
    # authored later with Isaac Sim's render-facing USD viewport utility.
    overview_cfg_class = TiledCameraCfg if use_tiled_overview_camera else CameraCfg
    overview_cfg = overview_cfg_class(
        prim_path="/World/OverviewCamera",
        offset=overview_cfg_class.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=OVERVIEW_WITNESS_FOCAL_LENGTH_MM,
            horizontal_aperture=36.0,
            clipping_range=OVERVIEW_CAMERA_CLIPPING_RANGE_M,
        ),
        width=args.overview_width,
        height=args.overview_height,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        colorize_semantic_segmentation=False,
        depth_clipping_behavior="max",
        update_latest_camera_pose=True,
    )
    # The direct City and task-obstacle roots preserve high-fidelity LiDAR
    # returns. The exact same conservative proxy boxes backstop structures
    # whose visual assets have no native PhysX collision.
    lidar_targets: list[Any] = [
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/StaticScene/City/Rivermark",
            track_mesh_transforms=False,
            merge_prim_meshes=True,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/StaticScene/CityTaskObstacles",
            track_mesh_transforms=False,
            merge_prim_meshes=True,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr=COLLISION_PROXY_ROOT,
            track_mesh_transforms=False,
            merge_prim_meshes=True,
        ),
    ]
    if target_paths:
        lidar_targets.append(
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/SearchTargets",
                track_mesh_transforms=False,
                merge_prim_meshes=True,
            )
        )
    lidar_cfg = MultiMeshRayCasterCfg(
        prim_path="/World/Swarm/Agent_.*/Robot/body",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.02)),
        mesh_prim_paths=lidar_targets,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=LIDAR_CHANNEL_COUNT,
            vertical_fov_range=LIDAR_VERTICAL_FOV_RANGE_DEG,
            horizontal_fov_range=LIDAR_HORIZONTAL_FOV_RANGE_DEG,
            horizontal_res=LIDAR_HORIZONTAL_RESOLUTION_DEG,
        ),
        ray_alignment="base",
        max_distance=100.0,
        # IsaacLab 2.3.2 currently allocates [..., 1] but returns [...] for
        # mesh IDs. Distances and hit points are unaffected, so fail closed on
        # this optional output instead of patching the external installation.
        update_mesh_ids=False,
    )
    imu_cfg = ImuCfg(prim_path="/World/Swarm/Agent_.*/Robot/body")
    contact_cfg = ContactSensorCfg(
        prim_path="/World/Swarm/Agent_.*/Robot/body",
        update_period=float(args.dt),
        track_pose=False,
        track_air_time=True,
        force_threshold=0.01,
        history_length=1,
    )
    return (
        (TiledCamera(onboard_cfg) if use_tiled_onboard_camera else Camera(onboard_cfg))
        if include_onboard_camera
        else None,
        (TiledCamera(overview_cfg) if use_tiled_overview_camera else Camera(overview_cfg))
        if include_overview_camera
        else None,
        MultiMeshRayCaster(lidar_cfg),
        Imu(imu_cfg),
        ContactSensor(contact_cfg),
        onboard_cfg,
        overview_cfg,
        lidar_cfg,
    )


def _waypoint_routes(
    dtype: Any,
    device: str,
    torch: Any,
    routes_w_m: Sequence[Sequence[Sequence[float]]] = PUBLIC_ROUTES_W_M,
) -> Any:
    """Return the frozen absolute City-Lite routes, never local offsets."""

    return torch.tensor(routes_w_m, dtype=dtype, device=device)


def _controller_target(
    robot: Any,
    waypoint_routes: Any,
    base_thrust: float,
    sim_time: float,
    torch: Any,
    math_utils: Any,
) -> tuple[Any, Any, Any, Any, float]:
    segment_count = int(waypoint_routes.shape[1]) - 1
    route_time = max(0.0, float(sim_time)) / WAYPOINT_SEGMENT_SECONDS
    segment_index = min(int(route_time), segment_count - 1)
    segment_progress = min(max(route_time - segment_index, 0.0), 1.0)
    if route_time >= segment_count:
        segment_index = segment_count - 1
        segment_progress = 1.0
    start = waypoint_routes[:, segment_index]
    end = waypoint_routes[:, segment_index + 1]
    desired_pos = start + (end - start) * segment_progress
    desired_vel = (end - start) / WAYPOINT_SEGMENT_SECONDS
    if route_time >= segment_count:
        desired_vel = torch.zeros_like(desired_vel)
    waypoint_index = torch.full(
        (AGENT_COUNT,), segment_index + 1, dtype=torch.int64, device=robot.device
    )
    pos = robot.data.root_pos_w
    vel = robot.data.root_lin_vel_w
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    ang_vel_b = robot.data.root_ang_vel_b
    collective = 4.0 * base_thrust + 0.12 * (desired_pos[:, 2] - pos[:, 2]) + 0.08 * (desired_vel[:, 2] - vel[:, 2])
    acc_xy_w = 1.2 * (desired_pos[:, :2] - pos[:, :2]) + 0.7 * (desired_vel[:, :2] - vel[:, :2])
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    acc_x_b = cos_yaw * acc_xy_w[:, 0] + sin_yaw * acc_xy_w[:, 1]
    acc_y_b = -sin_yaw * acc_xy_w[:, 0] + cos_yaw * acc_xy_w[:, 1]
    desired_pitch = torch.clamp(acc_x_b / 9.81, -0.12, 0.12)
    desired_roll = torch.clamp(-acc_y_b / 9.81, -0.12, 0.12)
    wrench = torch.zeros((AGENT_COUNT, 6), dtype=torch.float32, device=robot.device)
    wrench[:, 2] = torch.clamp(collective, 0.0, 0.72)
    wrench[:, 3] = 0.015 * (desired_roll - roll) - 0.006 * ang_vel_b[:, 0]
    wrench[:, 4] = 0.015 * (desired_pitch - pitch) - 0.006 * ang_vel_b[:, 1]
    wrench[:, 5] = -0.0015 * ang_vel_b[:, 2]
    target = (torch.linalg.pinv(robot.allocation_matrix) @ wrench.T).T.clamp(0.0, 0.18)
    return target, desired_pos, desired_vel, waypoint_index, segment_progress


def _velocity_yaw_controller_target(
    robot: Any,
    desired_velocity_w_mps: Any,
    desired_yaw_rate_radps: Any,
    altitude_reference_w_m: Any,
    base_thrust: float,
    command_hold_s: float,
    torch: Any,
    math_utils: Any,
) -> tuple[Any, Any, Any, Any]:
    """Lower a bounded world-velocity/yaw command to real CF2X thrust.

    This is deliberately separate from the frozen public-route tracker. The
    caller owns policy provenance and the swept geometry guard; this function
    only provides the physical velocity/yaw-to-wrench control law and returns
    the exact clipped command used to produce the thrust target.
    """

    if not math.isfinite(float(base_thrust)) or not math.isfinite(float(command_hold_s)):
        raise ValueError("base thrust and command hold must be finite")
    if command_hold_s <= 0.0:
        raise ValueError("command hold must be positive")
    if tuple(desired_velocity_w_mps.shape) != (AGENT_COUNT, 3):
        raise ValueError("velocity controller requires world velocity [8,3]")
    if tuple(desired_yaw_rate_radps.shape) != (AGENT_COUNT,):
        raise ValueError("velocity controller requires yaw rate [8]")
    if tuple(altitude_reference_w_m.shape) != (AGENT_COUNT,):
        raise ValueError("velocity controller requires altitude reference [8]")
    if not bool(
        torch.isfinite(desired_velocity_w_mps).all()
        and torch.isfinite(desired_yaw_rate_radps).all()
        and torch.isfinite(altitude_reference_w_m).all()
    ):
        raise ValueError("velocity controller received non-finite policy command")

    position = robot.data.root_pos_w
    velocity = robot.data.root_lin_vel_w
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    angular_velocity_b = robot.data.root_ang_vel_b

    requested_xy = desired_velocity_w_mps[:, :2]
    requested_xy_norm = torch.linalg.vector_norm(requested_xy, dim=-1, keepdim=True)
    xy_scale = torch.clamp(STATE_ONLY_POLICY_HORIZONTAL_SPEED_MPS / requested_xy_norm, max=1.0)
    commanded_xy = requested_xy * xy_scale
    altitude_correction = STATE_ONLY_ALTITUDE_HOLD_GAIN * (
        altitude_reference_w_m - position[:, 2]
    )
    commanded_z = torch.clamp(
        desired_velocity_w_mps[:, 2] + altitude_correction,
        -STATE_ONLY_POLICY_VERTICAL_SPEED_MPS,
        STATE_ONLY_POLICY_VERTICAL_SPEED_MPS,
    )
    commanded_velocity = torch.cat((commanded_xy, commanded_z.unsqueeze(-1)), dim=-1)
    commanded_yaw_rate = torch.clamp(
        desired_yaw_rate_radps,
        -STATE_ONLY_POLICY_YAW_RATE_RADPS,
        STATE_ONLY_POLICY_YAW_RATE_RADPS,
    )

    acceleration_xy_w = 1.2 * (commanded_velocity[:, :2] - velocity[:, :2])
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    acceleration_x_b = cos_yaw * acceleration_xy_w[:, 0] + sin_yaw * acceleration_xy_w[:, 1]
    acceleration_y_b = -sin_yaw * acceleration_xy_w[:, 0] + cos_yaw * acceleration_xy_w[:, 1]
    desired_pitch = torch.clamp(acceleration_x_b / 9.81, -0.12, 0.12)
    desired_roll = torch.clamp(-acceleration_y_b / 9.81, -0.12, 0.12)
    collective = 4.0 * base_thrust + 0.08 * (commanded_velocity[:, 2] - velocity[:, 2])

    wrench = torch.zeros((AGENT_COUNT, 6), dtype=torch.float32, device=robot.device)
    wrench[:, 2] = torch.clamp(collective, 0.0, 4.0 * MAX_THRUST_PER_ROTOR_N)
    wrench[:, 3] = 0.015 * (desired_roll - roll) - 0.006 * angular_velocity_b[:, 0]
    wrench[:, 4] = 0.015 * (desired_pitch - pitch) - 0.006 * angular_velocity_b[:, 1]
    wrench[:, 5] = 0.003 * (commanded_yaw_rate - angular_velocity_b[:, 2])
    target = (
        torch.linalg.pinv(robot.allocation_matrix) @ wrench.T
    ).T.clamp(0.0, MAX_THRUST_PER_ROTOR_N)

    desired_position = position + commanded_velocity * command_hold_s
    minimum = torch.tensor(
        CITY_LITE_COMMAND_VOLUME_W_M.minimum, dtype=desired_position.dtype, device=robot.device
    )
    maximum = torch.tensor(
        CITY_LITE_COMMAND_VOLUME_W_M.maximum, dtype=desired_position.dtype, device=robot.device
    )
    desired_position = torch.maximum(torch.minimum(desired_position, maximum), minimum)
    return target, desired_position, commanded_velocity, commanded_yaw_rate


def _append(samples: dict[str, list[Any]], key: str, value: Any) -> None:
    samples[key].append(_to_numpy(value))


def _raw_contact_force_maximum_n(net_contact_forces_w_n: Any) -> float:
    """Return the raw per-body normal-force maximum for a trace frame.

    This runs before the guard decision so an aborted frame retains the same
    raw contact evidence that triggered it.  A non-finite component remains
    non-finite rather than being silently converted to a passing value.
    """

    try:
        rows = net_contact_forces_w_n.reshape((-1, 3))
    except Exception:
        return math.nan
    maximum = 0.0
    for row in rows:
        try:
            norm = math.sqrt(sum(float(component) ** 2 for component in row))
        except (TypeError, ValueError):
            return math.nan
        if not math.isfinite(norm):
            return norm
        maximum = max(maximum, norm)
    return maximum


def _record_runtime_safety_trace_frame(
    runtime_samples: dict[str, list[Any]],
    *,
    physics_step: int,
    physics_dt_s: float,
    phase: str,
    outcome: str,
    root_positions_w_m: Any,
    net_contact_forces_w_n: Any,
    max_contact_force_n: float,
) -> None:
    """Append one immutable raw safety snapshot with a canonical time binding."""

    if phase not in RUNTIME_SAFETY_PHASE_CODES:
        raise ValueError(f"unknown runtime safety phase: {phase}")
    if outcome not in RUNTIME_SAFETY_FRAME_OUTCOME_CODES:
        raise ValueError(f"unknown runtime safety frame outcome: {outcome}")
    runtime_samples["physics_step"].append(int(physics_step))
    runtime_samples["physics_time_ns"].append(
        physics_time_ns(int(physics_step), physics_dt_s)
    )
    runtime_samples["phase_code"].append(int(RUNTIME_SAFETY_PHASE_CODES[phase]))
    runtime_samples["frame_outcome_code"].append(
        int(RUNTIME_SAFETY_FRAME_OUTCOME_CODES[outcome])
    )
    runtime_samples["root_pos_w_m"].append(root_positions_w_m.copy())
    runtime_samples["net_contact_forces_w_n"].append(net_contact_forces_w_n.copy())
    runtime_samples["max_contact_force_n"].append(float(max_contact_force_n))


def _evaluate_and_record_runtime_safety(
    runtime_samples: dict[str, list[Any]],
    guard: dict[str, Any],
    *,
    previous_positions_w_m: Any | None,
    current_positions_w_m: Any,
    net_contact_forces_w_n: Any,
    structural_aabbs: Sequence[AABB],
    phase: str,
    physics_step: int,
    physics_dt_s: float,
) -> Any:
    """Evaluate one CPU snapshot and retain a passing or abort evidence frame.

    The inputs are copied from Fabric in two batched transfers before the
    pure-Python guard runs. This avoids CUDA scalar synchronization for every
    coordinate and makes the previous state an immutable trace snapshot.
    """

    current_positions = _to_numpy(current_positions_w_m)
    current_forces = _to_numpy(net_contact_forces_w_n)
    raw_maximum = _raw_contact_force_maximum_n(current_forces)
    try:
        check = evaluate_runtime_safety(
            previous_positions_w_m,
            current_positions,
            current_forces,
            structural_aabbs,
            phase=phase,
            physics_step=physics_step,
        )
    except RuntimeSafetyAbort:
        _record_runtime_safety_trace_frame(
            runtime_samples,
            physics_step=physics_step,
            physics_dt_s=physics_dt_s,
            phase=phase,
            outcome="aborted",
            root_positions_w_m=current_positions,
            net_contact_forces_w_n=current_forces,
            max_contact_force_n=raw_maximum,
        )
        raise
    record_runtime_safety_check(guard, check, phase=phase)
    _record_runtime_safety_trace_frame(
        runtime_samples,
        physics_step=physics_step,
        physics_dt_s=physics_dt_s,
        phase=phase,
        outcome="passed",
        root_positions_w_m=current_positions,
        net_contact_forces_w_n=current_forces,
        max_contact_force_n=float(check.max_contact_force_n),
    )
    return current_positions.copy()


def _write_runtime_safety_trace(
    output_dir: Path, runtime_samples: Mapping[str, Sequence[Any]], np: Any
) -> Path:
    """Write the evidence trace that independently replays every guard step."""

    expected = {
        "physics_step",
        "physics_time_ns",
        "phase_code",
        "frame_outcome_code",
        "root_pos_w_m",
        "net_contact_forces_w_n",
        "max_contact_force_n",
    }
    if set(runtime_samples) != expected:
        raise RuntimeError("runtime safety trace fields are incomplete")
    count = len(runtime_samples["physics_step"])
    if count < 1 or any(len(runtime_samples[key]) != count for key in expected):
        raise RuntimeError("runtime safety trace samples are incomplete")
    destination = output_dir / RUNTIME_SAFETY_TRACE_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        physics_step=np.asarray(runtime_samples["physics_step"], dtype=np.int64),
        physics_time_ns=np.asarray(runtime_samples["physics_time_ns"], dtype=np.int64),
        phase_code=np.asarray(runtime_samples["phase_code"], dtype=np.int8),
        frame_outcome_code=np.asarray(
            runtime_samples["frame_outcome_code"], dtype=np.uint8
        ),
        root_pos_w_m=np.stack(runtime_samples["root_pos_w_m"], axis=0),
        net_contact_forces_w_n=np.stack(runtime_samples["net_contact_forces_w_n"], axis=0),
        max_contact_force_n=np.asarray(runtime_samples["max_contact_force_n"], dtype=np.float32),
    )
    return destination


def _write_sensor_phase_trace(
    output_dir: Path, phase_samples: Mapping[str, Sequence[Any]], np: Any
) -> Path:
    """Write the retained-frame execution order and contact binding trace."""

    expected = {
        "physics_step",
        "physics_time_ns",
        "event_codes",
        "retained_contact_sha256",
        "archive_frame_index",
    }
    if set(phase_samples) != expected:
        raise RuntimeError("sensor phase trace fields are incomplete")
    count = len(phase_samples["physics_step"])
    if count < 1 or any(len(phase_samples[key]) != count for key in expected):
        raise RuntimeError("sensor phase trace samples are incomplete")
    event_codes = np.asarray(phase_samples["event_codes"], dtype=np.uint8)
    contact_digests = np.asarray(
        phase_samples["retained_contact_sha256"], dtype=np.uint8
    )
    if event_codes.shape != (count, len(SENSOR_PHASE_EVENT_SEQUENCE)):
        raise RuntimeError("sensor phase event trace has an invalid shape")
    if contact_digests.shape != (count, 32):
        raise RuntimeError("sensor phase contact digests have an invalid shape")
    destination = output_dir / SENSOR_PHASE_TRACE_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema=np.asarray([SENSOR_PHASE_TRACE_SCHEMA]),
        sensor_names=np.asarray(SENSOR_PHASE_SENSOR_NAMES),
        physics_step=np.asarray(phase_samples["physics_step"], dtype=np.int64),
        physics_time_ns=np.asarray(phase_samples["physics_time_ns"], dtype=np.int64),
        event_codes=event_codes,
        retained_contact_sha256=contact_digests,
        archive_frame_index=np.asarray(
            phase_samples["archive_frame_index"], dtype=np.int64
        ),
    )
    return destination


def _capture_quality_observations(
    *,
    timestamps_ns: Any,
    camera_position_errors_m: Any,
    camera_orientation_errors_rad: Any,
    onboard_usd_max_position_error_m: float,
    onboard_usd_min_forward_alignment_cosine: float,
    onboard_usd_max_orientation_error_rad: float,
    overview_closure: Mapping[str, Any],
    overview_camera_positions_w_m: Any,
    overview_first_rgb: Any,
    target_thrust_n: Any,
    applied_thrust_n: Any,
    np: Any,
) -> dict[str, Any]:
    """Build scalar receipt observations from already captured raw arrays.

    Route-witness validity is established separately by the public schedule,
    pose-closure, and agent-visibility gates.  The camera displacement here is
    descriptive evidence across that fixed schedule, including its declared
    shot transitions; it is not a substitute for any of those gates.
    """

    positions = np.asarray(overview_camera_positions_w_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] < 1 or positions.shape[1] != 3:
        raise ValueError("overview camera positions must have shape [frame, 3]")
    if not np.isfinite(positions).all():
        raise ValueError("overview camera positions must be finite")
    overview_camera_max_displacement_m = float(
        np.max(np.linalg.norm(positions - positions[0:1], axis=-1))
    )
    return {
        "timestamps_strictly_monotonic": bool(np.all(np.diff(timestamps_ns) > 0)),
        "camera_pose_closure_max_error_m": float(np.max(camera_position_errors_m)),
        "camera_pose_closure_max_error_rad": float(np.max(camera_orientation_errors_rad)),
        "camera_usd_pose_closure_max_error_m": onboard_usd_max_position_error_m,
        "camera_usd_pose_closure_min_forward_alignment_cosine": onboard_usd_min_forward_alignment_cosine,
        "camera_usd_pose_closure_max_orientation_error_rad": onboard_usd_max_orientation_error_rad,
        "overview_camera_position_error_m": float(overview_closure["position_error_m"]),
        "overview_camera_forward_alignment_cosine": float(
            overview_closure["forward_alignment_cosine"]
        ),
        "overview_camera_max_displacement_m": overview_camera_max_displacement_m,
        "onboard_visual_intrusion_gate_passed": True,
        "all_state_values_finite": bool(
            np.isfinite(target_thrust_n).all() and np.isfinite(applied_thrust_n).all()
        ),
        "overview_first_frame_nonconstant": bool(
            np.max(overview_first_rgb) > np.min(overview_first_rgb)
        ),
    }


def _capture(
    args: argparse.Namespace,
    output_dir: Path,
    receipt: dict[str, Any],
    evaluator_manifest: Mapping[str, Any] | None,
    authority: CityLiteAuthority,
    state_only_transfer: Any | None = None,
    resource_telemetry: ResourceTelemetry | None = None,
) -> None:
    route_profile = _route_execution_profile(receipt)
    storage_budget = _capture_storage_budget(args)
    public_routes_w_m = route_profile.routes_w_m
    runtime_lock = _bind_runtime_lock_to_args(args, receipt)
    isaaclab_source = _activate_local_isaaclab_source(
        args.isaaclab_source if runtime_lock is not None else None
    )
    isaaclab_contrib_source: Path | None = None
    if runtime_lock is not None:
        contrib_relative = PurePosixPath(
            str(runtime_lock["isaaclab_contrib_source"]["relative_path"])
        )
        isaaclab_contrib_source = _activate_local_isaaclab_contrib_source(
            args.isaaclab_source.expanduser().resolve().parent.joinpath(
                *contrib_relative.parts
            )
        )

    app = None
    lease = repository_app_launcher_lease(
        _repository_root(),
        metadata={
            "output_dir": str(output_dir),
            "source_revision": receipt.get("source_revision"),
            "owner": "rivermark_benchmark.isaac_capture",
        },
    )
    if resource_telemetry is None:
        resource_telemetry = ResourceTelemetry()
    try:
        _acquire_capture_app_launcher_lease(lease, output_dir, receipt)
        before_launcher_telemetry = resource_telemetry.sample("before_app_launcher")
        _enforce_system_commit_guard(
            args,
            receipt,
            phase="before_app_launcher",
            output_dir=output_dir,
            snapshot=(
                before_launcher_telemetry.get("system_commit")
                if isinstance(before_launcher_telemetry.get("system_commit"), Mapping)
                else None
            ),
        )
        _checkpoint(output_dir, "before_app_launcher")
        launcher_settings: dict[str, Any] = {
            "headless": bool(args.headless),
            "enable_cameras": True,
            "device": args.device,
        }
        if runtime_lock is not None:
            from .runtime_lock import (
                locked_launcher_kwargs,
                validate_locked_launcher_environment,
            )

            validate_locked_launcher_environment(runtime_lock)
            if isaaclab_source is None:
                raise RuntimeError("locked capture did not activate an IsaacLab source")
            launcher_settings = locked_launcher_kwargs(runtime_lock, isaaclab_source)
        from isaaclab.app import AppLauncher

        app = AppLauncher(launcher_settings).app
        _checkpoint(output_dir, "app_launched")
        _enforce_foreign_native_process_guard(
            args,
            receipt,
            phase="after_app_launcher",
            output_dir=output_dir,
        )

        import numpy as np
        import omni.usd
        import torch
        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        from isaaclab_contrib.actuators import ThrusterCfg
        from isaaclab_contrib.assets import Multirotor, MultirotorCfg
        after_launcher_telemetry = resource_telemetry.sample(
            "after_app_launcher", torch_module=torch
        )
        _enforce_system_commit_guard(
            args,
            receipt,
            phase="after_app_launcher",
            output_dir=output_dir,
            snapshot=(
                after_launcher_telemetry.get("system_commit")
                if isinstance(after_launcher_telemetry.get("system_commit"), Mapping)
                else None
            ),
        )

        if runtime_lock is not None:
            if isaaclab_source is None or not _module_path_is_under(
                __import__("isaaclab"), isaaclab_source / "isaaclab"
            ):
                raise RuntimeError("locked capture imported isaaclab from an unbound source")
            if isaaclab_contrib_source is None or not _module_path_is_under(
                __import__("isaaclab_contrib"),
                isaaclab_contrib_source / "isaaclab_contrib",
            ):
                raise RuntimeError(
                    "locked capture imported isaaclab_contrib from an unbound source"
                )

        torch.manual_seed(args.seed)
        omni.usd.get_context().new_stage()
        _checkpoint(output_dir, "new_stage_created")
        stage, static_scene_evidence = _compose_city_lite_stage(authority)
        structural_aabbs = _extract_structural_aabbs(stage)
        route_contract = make_public_route_contract(
            structural_aabbs,
            route_family_id=route_profile.route_family_id,
            routes_w_m=public_routes_w_m,
        )
        route_report = validate_public_route_contract(
            route_contract,
            public_routes_w_m,
            structural_aabbs,
        )
        if args.control_mode in (CONTROL_MODE_FIXED_PUBLIC_ROUTE, CONTROL_MODE_NATIVE_T2_CANARY):
            if evaluator_manifest is None:
                raise RuntimeError("private-target capture requires an evaluator manifest")
            validate_private_target_geometry(
                evaluator_manifest,
                structural_aabbs=structural_aabbs,
                public_routes_w_m=public_routes_w_m,
                city_lite_scene_contract_sha256=authority.contract_sha256,
                city_lite_scene_payload_sha256=authority.contract_payload_sha256,
                execution_window=_capture_target_visibility_execution_window(args),
                expected_task_variant_id=(
                    _native_t2_task_variant_id(args)
                    if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY
                    else TASK_VARIANT_ID
                ),
                expected_native_t2_motion_contract=(
                    getattr(args, "native_t2_motion_contract")
                    if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY
                    else None
                ),
            )
        elif args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
            if evaluator_manifest is not None or state_only_transfer is None:
                raise RuntimeError(
                    "SB3 state-only transfer must have no evaluator manifest and one preflighted bridge"
                )
        else:
            raise RuntimeError(f"unsupported capture control mode: {args.control_mode!r}")
        # Keep an explicit second call here so the capture fails before PhysX
        # initialization even if route-contract validation changes in future.
        validate_public_routes(
            public_routes_w_m,
            structural_aabbs,
            expected_starts_w_m=TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M[
                route_profile.route_family_id
            ],
        )
        proxy_paths = _spawn_collision_proxies(stage, structural_aabbs)
        if len(proxy_paths) != len(structural_aabbs):
            raise RuntimeError("City-Lite collision-proxy count does not match structural AABBs")
        static_scene_receipt = authority.provenance()
        static_scene_receipt.update(static_scene_evidence)
        static_scene_receipt["native_collision_counts"] = static_scene_evidence[
            "native_collision_counts"
        ]
        validate_static_scene_receipt(static_scene_receipt)
        # Persist the verified static evidence immediately.  A later PhysX or
        # sensor failure must not erase the exact City-Lite material-closure
        # audit from an otherwise failed development receipt.
        receipt["city_lite_authority"] = static_scene_receipt
        _checkpoint(
            output_dir,
            "city_lite_static_composed",
            active_static_prim_count=static_scene_evidence["active_static_prim_count"],
            structural_aabb_count=len(structural_aabbs),
            collision_proxy_count=len(proxy_paths),
            route_segment_count=route_report.segment_count,
            private_target_placement_verified=True,
        )
        sim_cfg = sim_utils.SimulationCfg(dt=args.dt, device=args.device)
        sim_cfg.gravity = (0.0, 0.0, -9.81)
        if runtime_lock is not None:
            from .runtime_lock import configure_simulation_cfg

            configure_simulation_cfg(sim_cfg, runtime_lock)
        sim = sim_utils.SimulationContext(sim_cfg)
        _checkpoint(output_dir, "simulation_context_created")
        target_paths = _make_scene(sim_utils, evaluator_manifest)
        _checkpoint(
            output_dir,
            "city_lite_runtime_scene_created",
            target_count=len(target_paths),
        )
        runtime_target_usd_pre_reset: dict[str, Any] | None = None
        if evaluator_manifest is not None:
            runtime_target_usd_pre_reset = _verify_runtime_target_usd_authoring(
                stage, target_paths, evaluator_manifest
            )
            receipt["runtime_target_usd_pre_reset"] = runtime_target_usd_pre_reset
            _checkpoint(
                output_dir,
                "runtime_target_usd_pre_reset_verified",
                **runtime_target_usd_pre_reset,
            )
        literal_cfgs = _make_multirotor_cfgs(
            args, sim_utils, MultirotorCfg, ThrusterCfg, public_routes_w_m
        )
        members = tuple(Multirotor(cfg) for cfg in literal_cfgs)
        _checkpoint(
            output_dir,
            "literal_multirotors_constructed",
            literal_prim_paths=list(SWARM_AGENT_LITERAL_PRIM_PATHS),
        )
        literal_usd_spawn_receipt = _verify_literal_city_lite_usd_spawn(
            stage, public_routes_w_m
        )
        _checkpoint(
            output_dir,
            "literal_city_lite_usd_spawn_verified",
            max_position_error_m=literal_usd_spawn_receipt["max_position_error_m"],
            max_orientation_error_rad=literal_usd_spawn_receipt[
                "max_orientation_error_rad"
            ],
        )
        marker_paths = _spawn_identity_markers(sim_utils)
        onboard, overview, lidar, imu, contact, onboard_cfg, overview_cfg, lidar_cfg = _make_sensors(
            args, sim_utils, target_paths
        )
        _checkpoint(output_dir, "sensors_constructed")
        _enforce_foreign_native_process_guard(
            args,
            receipt,
            phase="sensors_constructed",
            output_dir=output_dir,
        )
        sim.reset()
        _checkpoint(output_dir, "simulation_reset")
        _enforce_foreign_native_process_guard(
            args,
            receipt,
            phase="simulation_reset",
            output_dir=output_dir,
        )
        if evaluator_manifest is not None:
            runtime_target_usd_post_reset = _verify_runtime_target_usd_authoring(
                stage, target_paths, evaluator_manifest
            )
            receipt["runtime_target_usd_post_reset"] = runtime_target_usd_post_reset
            _checkpoint(
                output_dir,
                "runtime_target_usd_post_reset_verified",
                **runtime_target_usd_post_reset,
            )
        after_reset_telemetry = resource_telemetry.sample(
            "after_reset", torch_module=torch
        )
        _enforce_system_commit_guard(
            args,
            receipt,
            phase="after_reset",
            output_dir=output_dir,
            snapshot=(
                after_reset_telemetry.get("system_commit")
                if isinstance(after_reset_telemetry.get("system_commit"), Mapping)
                else None
            ),
        )
        runtime_live_observed: Mapping[str, Any] | None = None
        if runtime_lock is not None:
            from .runtime_lock import compare_live_simulation, observe_live_simulation

            runtime_live_observed = observe_live_simulation(runtime_lock, sim)
            runtime_live_issues = compare_live_simulation(
                runtime_lock, runtime_live_observed
            )
            if runtime_live_issues:
                raise RuntimeError(
                    "live Isaac runtime does not match the locked configuration: "
                    + "; ".join(
                        f"{issue.path}: {issue.message}"
                        for issue in runtime_live_issues
                    )
                )
            receipt["runtime_live"] = dict(runtime_live_observed)
            _checkpoint(
                output_dir,
                "runtime_lock_live_configuration_verified",
                runtime_lock_sha256=receipt["runtime_lock"]["sha256"],
                observed=runtime_live_observed,
            )
        # IsaacLab exposes PhysX views only after reset.  Build the batching
        # facade now; each CF2X was already authored at its route anchor in
        # its literal config, so no reset-time root-state mutation is allowed.
        robot = EightCF2XFleet(
            members,
            prim_expression=SWARM_AGENT_PRIM_EXPRESSION,
            literal_prim_paths=SWARM_AGENT_LITERAL_PRIM_PATHS,
        )
        if not robot.is_initialized or robot.num_instances != AGENT_COUNT or robot.num_thrusters != 4:
            raise RuntimeError(
                f"Expected eight literal one-instance four-thruster CF2X assets; got {robot.num_instances}x{robot.num_thrusters}"
            )
        for name, sensor, count in (
            ("onboard_camera", onboard, AGENT_COUNT),
            ("overview_camera", overview, 1),
            ("multi_mesh_lidar", lidar, AGENT_COUNT),
            ("imu", imu, AGENT_COUNT),
            ("contact", contact, AGENT_COUNT),
        ):
            if not sensor.is_initialized:
                raise RuntimeError(f"{name} did not initialize")
            if hasattr(sensor, "num_instances") and int(sensor.num_instances) != count:
                raise RuntimeError(f"{name} expected {count} instances, got {sensor.num_instances}")
            sensor.reset()
        # `reset()` refreshes actuator state only. It must never be followed by
        # a root-pose or root-velocity write: the literal init states are the
        # sole physical spawn authority for this capture.
        robot.reset()
        robot.update(args.dt)
        root_state = _city_lite_initial_root_states(
            torch, robot.device, public_routes_w_m
        )
        root_pose = root_state[:, :7]
        initial_thruster_rps = _city_lite_initial_thruster_rps(torch, robot.device)
        literal_spawn_receipt = _verify_literal_city_lite_spawn(
            robot, root_state, initial_thruster_rps, torch
        )
        literal_spawn_receipt["authored_usd_transform"] = literal_usd_spawn_receipt
        _checkpoint(
            output_dir,
            "literal_city_lite_spawn_verified",
            **literal_spawn_receipt,
        )
        _checkpoint(output_dir, "assets_initialized")
        runtime_safety_guard = runtime_safety_receipt_template(
            structural_aabbs,
            contact_prim_expression="/World/Swarm/Agent_.*/Robot/body",
            physics_dt_s=args.dt,
        )
        runtime_safety_samples: dict[str, list[Any]] = {
            "physics_step": [],
            "physics_time_ns": [],
            "phase_code": [],
            "frame_outcome_code": [],
            "root_pos_w_m": [],
            "net_contact_forces_w_n": [],
            "max_contact_force_n": [],
        }
        previous_runtime_positions_w_m: Any | None = None

        def _guard_runtime_frame(phase: str, physics_step: int) -> None:
            nonlocal previous_runtime_positions_w_m
            try:
                previous_runtime_positions_w_m = _evaluate_and_record_runtime_safety(
                    runtime_safety_samples,
                    runtime_safety_guard,
                    previous_positions_w_m=previous_runtime_positions_w_m,
                    current_positions_w_m=robot.data.root_pos_w,
                    net_contact_forces_w_n=contact.data.net_forces_w,
                    structural_aabbs=structural_aabbs,
                    phase=phase,
                    physics_step=physics_step,
                    physics_dt_s=args.dt,
                )
            except RuntimeSafetyAbort as error:
                record_runtime_safety_abort(runtime_safety_guard, error)
                if runtime_safety_samples["physics_step"]:
                    trace_path = _write_runtime_safety_trace(
                        output_dir, runtime_safety_samples, np
                    )
                    bind_runtime_safety_trace_evidence(
                        runtime_safety_guard,
                        trace_sha256=_sha256(trace_path),
                        physics_frame_count=len(runtime_safety_samples["physics_step"]),
                    )
                receipt["runtime_safety_guard"] = runtime_safety_guard
                _checkpoint(
                    output_dir,
                    "runtime_safety_abort",
                    phase=phase,
                    physics_step=physics_step,
                    first_violation=error.violation,
                )
                raise

        sensor_timeline = _SensorUpdateTimeline()
        sensor_timeline.update(contact, time_ns=physics_time_ns(0, args.dt))
        _guard_runtime_frame("post_reset", 0)
        _checkpoint(output_dir, "post_reset_runtime_safety_verified")
        waypoint_routes = _waypoint_routes(
            torch.float32, robot.device, torch, public_routes_w_m
        )
        overview_witness_view = _set_public_route_witness_overview_view(
            stage, overview, effective_time_ns=0
        )
        _checkpoint(output_dir, "initial_public_route_witness_view_set")
        (
            onboard_expected_positions,
            onboard_expected_world_quats,
            onboard_usd_closure,
        ) = _prepare_onboard_camera_local_mount(
            sim, stage, robot, torch
        )
        _checkpoint(output_dir, "initial_onboard_camera_local_mount_prepared")
        onboard_mount_prepare_count = 1
        onboard_render_read_fence_count = 0
        onboard_last_render_read_fence: dict[str, Any] | None = None
        onboard_usd_max_position_error_m = float(onboard_usd_closure["max_position_error_m"])
        onboard_usd_min_forward_alignment_cosine = float(
            onboard_usd_closure["min_forward_alignment_cosine"]
        )
        onboard_usd_max_orientation_error_rad = float(
            onboard_usd_closure["max_orientation_error_rad"]
        )
        # Force one render/read after the native local-mount closure.  This
        # catches a broken renderer-facing hierarchy before any rollout frame
        # can be accepted as evidence.
        _enforce_foreign_native_process_guard(
            args,
            receipt,
            phase="before_initial_camera_render",
            output_dir=output_dir,
        )
        onboard_frame_before_render = _onboard_camera_frame_counter(onboard, torch)
        sim.render()
        _checkpoint(output_dir, "initial_camera_render_completed")
        sensor_timeline.update(onboard, time_ns=physics_time_ns(0, args.dt))
        onboard_last_render_read_fence = _require_onboard_camera_render_read_fence(
            onboard, onboard_frame_before_render, torch
        )
        onboard_render_read_fence_count += 1
        sensor_timeline.update(overview, time_ns=physics_time_ns(0, args.dt))
        onboard_fabric_closure = _camera_pose_closure(robot, onboard, torch)
        onboard_usd_closure = _onboard_camera_usd_pose_closure(
            stage, onboard_expected_positions, onboard_expected_world_quats, torch
        )
        _require_onboard_camera_usd_pose(onboard_usd_closure)
        onboard_render_closure = _camera_pose_closure_from_usd(
            onboard_usd_closure,
            onboard_expected_positions,
            onboard_expected_world_quats,
            torch,
        )
        onboard_fabric_diagnostic = _onboard_camera_fabric_pose_diagnostic(
            onboard_fabric_closure, torch
        )
        # Establish a receipt value even if a deliberately short run aborts
        # before its first retained sensor frame.  It remains a Fabric-only
        # diagnostic and cannot become RGB-D acceptance evidence.
        onboard_mount_diagnostic = _onboard_camera_mount_diagnostics(
            robot,
            onboard,
            root_expected_pos_w=onboard_expected_positions,
            root_expected_quat_wxyz=onboard_expected_world_quats,
            previous_root_expected_pos_w=None,
            previous_root_expected_phase=None,
            torch=torch,
        )
        previous_root_camera_expected_pos = onboard_expected_positions.detach().clone()
        previous_root_camera_expected_phase = "initial_post_render_usd_closure"
        onboard_usd_max_position_error_m = max(
            onboard_usd_max_position_error_m,
            float(onboard_usd_closure["max_position_error_m"]),
        )
        onboard_usd_min_forward_alignment_cosine = min(
            onboard_usd_min_forward_alignment_cosine,
            float(onboard_usd_closure["min_forward_alignment_cosine"]),
        )
        onboard_usd_max_orientation_error_rad = max(
            onboard_usd_max_orientation_error_rad,
            float(onboard_usd_closure["max_orientation_error_rad"]),
        )
        overview_closure = overview_witness_view["pose_closure"]
        overview_semantic_metadata: Any = _overview_semantic_metadata(overview)
        overview_initial_content_evidence = _overview_city_content_evidence(
            overview.data.output["rgb"],
            overview.data.output["distance_to_image_plane"],
            overview.data.output["semantic_segmentation"],
            overview_semantic_metadata,
            far_clip_m=OVERVIEW_CAMERA_CLIPPING_RANGE_M[1],
        )
        overview_initial_agent_visibility_evidence = _overview_tracked_agent_visibility_evidence(
            overview.data.output["semantic_segmentation"], overview_semantic_metadata
        )
        overview_initial_agent_visibility_evidence = {
            **overview_initial_agent_visibility_evidence,
            "effective_time_ns": 0,
            "witness_shot_index": int(overview_witness_view["shot_index"]),
        }
        # This checkpoint is overwritten by the verified stage on success. On
        # a fail-closed initial render it preserves enough raw evidence to
        # diagnose a real Isaac camera/semantic mismatch without guessing.
        _checkpoint(
            output_dir,
            "initial_overview_evidence_evaluated",
            overview_content_evidence=overview_initial_content_evidence,
            overview_agent_visibility_evidence=overview_initial_agent_visibility_evidence,
            overview_semantic_metadata=overview_semantic_metadata,
        )
        try:
            _require_overview_city_content(overview_initial_content_evidence)
            _require_overview_tracked_agent_visibility(
                overview_initial_agent_visibility_evidence
            )
        except RuntimeError:
            diagnostics = _persist_initial_overview_failure_diagnostics(
                output_dir,
                rgb=overview.data.output["rgb"],
                depth=overview.data.output["distance_to_image_plane"],
                semantic=overview.data.output["semantic_segmentation"],
                semantic_metadata=overview_semantic_metadata,
                content_evidence=overview_initial_content_evidence,
                agent_visibility_evidence=overview_initial_agent_visibility_evidence,
                root_pos_w_m=robot.data.root_pos_w,
                root_quat_wxyz=robot.data.root_quat_w,
                root_lin_vel_w_mps=robot.data.root_lin_vel_w,
                np=np,
            )
            _checkpoint(
                output_dir,
                "initial_overview_gate_failed",
                overview_content_evidence=overview_initial_content_evidence,
                overview_agent_visibility_evidence=overview_initial_agent_visibility_evidence,
                initial_overview_failure_diagnostics=diagnostics,
            )
            raise
        _checkpoint(
            output_dir,
            "overview_camera_view_and_content_verified",
            position_error_m=float(overview_closure["position_error_m"]),
            forward_alignment_cosine=float(overview_closure["forward_alignment_cosine"]),
            tracked_agent_id=OVERVIEW_WITNESS_TRACKED_AGENT_ID,
            non_background_geometry_fraction=overview_initial_content_evidence[
                "non_background_geometry_fraction"
            ],
            structural_pixel_fraction=overview_initial_content_evidence[
                "structural_pixel_fraction"
            ],
            tracked_agent_pixel_count=overview_initial_agent_visibility_evidence[
                "tracked_agent_pixel_count"
            ],
            onboard_max_position_error_m=onboard_usd_max_position_error_m,
            onboard_min_forward_alignment_cosine=onboard_usd_min_forward_alignment_cosine,
            onboard_max_orientation_error_rad=onboard_usd_max_orientation_error_rad,
        )

        for warmup in range(args.warmup_steps):
            if warmup % args.capture_stride == 0 or warmup == args.warmup_steps - 1:
                _enforce_foreign_native_process_guard(
                    args,
                    receipt,
                    phase=f"warmup_step_{warmup}",
                    output_dir=output_dir,
                )
            if args.control_mode in (
                CONTROL_MODE_SB3_STATE_ONLY_TRANSFER,
                CONTROL_MODE_NATIVE_T2_CANARY,
            ):
                zero_velocity = torch.zeros(
                    (AGENT_COUNT, 3), dtype=torch.float32, device=robot.device
                )
                zero_yaw_rate = torch.zeros(
                    (AGENT_COUNT,), dtype=torch.float32, device=robot.device
                )
                target, _, _, _ = _velocity_yaw_controller_target(
                    robot,
                    zero_velocity,
                    zero_yaw_rate,
                    robot.data.root_pos_w[:, 2].clone(),
                    args.base_thrust,
                    args.dt
                    * (
                        args.native_t2_decision_stride
                        if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY
                        else args.sb3_decision_stride
                    ),
                    torch,
                    math_utils,
                )
            else:
                target, _, _, _, _ = _controller_target(
                    robot, waypoint_routes, args.base_thrust, 0.0, torch, math_utils
                )
            robot.set_thrust_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(args.dt)
            warmup_effective_time_ns = physics_time_ns(warmup + 1, args.dt)
            sensor_timeline.update(contact, time_ns=warmup_effective_time_ns)
            _guard_runtime_frame("warmup", warmup + 1)
            if warmup >= max(0, args.warmup_steps - 3):
                (
                    onboard_expected_positions,
                    onboard_expected_world_quats,
                    onboard_usd_closure,
                ) = _prepare_onboard_camera_local_mount(
                    sim, stage, robot, torch
                )
                onboard_mount_prepare_count += 1
                overview_witness_view = _set_public_route_witness_overview_view(
                    stage,
                    overview,
                    effective_time_ns=warmup_effective_time_ns,
                )
                onboard_frame_before_render = _onboard_camera_frame_counter(onboard, torch)
                sim.render()
                sensor_timeline.update(onboard, time_ns=warmup_effective_time_ns)
                onboard_last_render_read_fence = _require_onboard_camera_render_read_fence(
                    onboard, onboard_frame_before_render, torch
                )
                onboard_render_read_fence_count += 1
                sensor_timeline.update(overview, time_ns=warmup_effective_time_ns)
                onboard_fabric_closure = _camera_pose_closure(robot, onboard, torch)
                onboard_usd_closure = _onboard_camera_usd_pose_closure(
                    stage, onboard_expected_positions, onboard_expected_world_quats, torch
                )
                _require_onboard_camera_usd_pose(onboard_usd_closure)
                onboard_render_closure = _camera_pose_closure_from_usd(
                    onboard_usd_closure,
                    onboard_expected_positions,
                    onboard_expected_world_quats,
                    torch,
                )
                onboard_fabric_diagnostic = _onboard_camera_fabric_pose_diagnostic(
                    onboard_fabric_closure, torch
                )
                previous_root_camera_expected_pos = (
                    onboard_expected_positions.detach().clone()
                )
                previous_root_camera_expected_phase = (
                    f"warmup_step_{warmup + 1}_post_render_usd_closure"
                )
                onboard_usd_max_position_error_m = max(
                    onboard_usd_max_position_error_m,
                    float(onboard_usd_closure["max_position_error_m"]),
                )
                onboard_usd_min_forward_alignment_cosine = min(
                    onboard_usd_min_forward_alignment_cosine,
                    float(onboard_usd_closure["min_forward_alignment_cosine"]),
                )
                onboard_usd_max_orientation_error_rad = max(
                    onboard_usd_max_orientation_error_rad,
                    float(onboard_usd_closure["max_orientation_error_rad"]),
                )
        overview_closure = overview_witness_view["pose_closure"]
        _checkpoint(output_dir, "warmup_completed", warmup_steps=args.warmup_steps)

        state_sample_keys = [
            "command_time_ns", "effective_time_ns", "root_pos_w_m", "root_quat_wxyz", "root_lin_vel_w_mps",
            "root_ang_vel_b_radps", "desired_pos_w_m", "desired_vel_w_mps", "target_thrust_n", "applied_thrust_n"
        ]
        if args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
            state_sample_keys.extend(
                (
                    "pre_command_root_pos_w_m",
                    "pre_command_root_quat_wxyz",
                    "pre_command_root_lin_vel_w_mps",
                    "pre_command_root_ang_vel_b_radps",
                    "emitted_world_velocity_yaw_command",
                )
            )
        elif args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
            state_sample_keys.extend(
                (
                    "pre_command_root_pos_w_m",
                    "pre_command_root_quat_wxyz",
                    "pre_command_root_lin_vel_w_mps",
                    "pre_command_root_ang_vel_b_radps",
                    "emitted_world_velocity_yaw_command",
                )
            )
        state_samples: dict[str, list[Any]] = {key: [] for key in state_sample_keys}
        task_samples: dict[str, list[Any]] = {key: [] for key in (
            "timestamps_ns", "waypoint_index", "waypoint_progress", "desired_waypoint_w_m",
            "distance_to_waypoint_m", "waypoint_reached", "action_mode", "coverage_cell_id", "task_time_s"
        )}
        message_samples: dict[str, list[Any]] = {key: [] for key in (
            "timestamps_ns", "sender_agent_id", "message_sequence", "message_waypoint_index",
            "message_position_w_m", "message_velocity_w_mps", "message_flags"
        )}
        sensor_samples = FrameSpool(
            output_dir / ".sensor_spool_v1",
            frame_capacity=_captured_frame_count(args.steps, args.capture_stride),
        )
        overview_archive_indices = _overview_archive_frame_indices(
            sensor_samples.frame_capacity
        )
        overview_archive_index_set = frozenset(overview_archive_indices)
        overview_samples = FrameSpool(
            output_dir / ".overview_spool_v1",
            frame_capacity=len(overview_archive_indices),
        )
        _enforce_runtime_storage_guard(
            args,
            receipt,
            phase="before_sensor_spool",
            output_dir=output_dir,
            budget=storage_budget,
        )
        sensor_phase_samples: dict[str, list[Any]] = {
            "physics_step": [],
            "physics_time_ns": [],
            "event_codes": [],
            "retained_contact_sha256": [],
            "archive_frame_index": [],
        }
        capture_frame_indices = frozenset(
            _captured_frame_indices(args.steps, args.capture_stride)
        )
        label_dir = output_dir / "learning_labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        semantic_frame_metadata_path = (
            output_dir / SEMANTIC_FRAME_METADATA_RELATIVE_PATH
        )
        semantic_frame_metadata_stream = semantic_frame_metadata_path.open(
            "x", encoding="utf-8", newline="\n"
        )
        semantic_frame_metadata_count = 0
        private_target_ids = tuple(
            str(target["target_id"])
            for target in evaluator_manifest["targets"]
            if isinstance(target, Mapping) and isinstance(target.get("target_id"), str)
        ) if evaluator_manifest is not None else ()
        overview_content_evidence_samples: list[dict[str, Any]] = []
        overview_agent_visibility_evidence_samples: list[dict[str, Any]] = []
        onboard_scene_content_evidence_samples: list[dict[str, Any]] = []
        onboard_visual_intrusion_evidence_samples: list[dict[str, Any]] = []
        target_visibility_evidence_samples: list[dict[str, Any]] = []
        transfer_samples: dict[str, list[Any]] | None = None
        transfer_provenance: Mapping[str, Any] | None = None
        transfer_velocity_w_mps: Any | None = None
        transfer_yaw_rate_radps: Any | None = None
        transfer_altitude_reference_w_m: Any | None = None
        native_t2_runner: T2PolicyRunner | None = None
        native_t2_decision: Any | None = None
        native_t2_velocity_w_mps: Any | None = None
        native_t2_yaw_rate_radps: Any | None = None
        native_t2_altitude_reference_w_m: Any | None = None
        native_t2_trace_stream: Any | None = None
        native_t2_event_journal: T2CandidateEventJournal | None = None
        native_t2_deduplicator: SpatialCandidateDeduplicator | None = None
        native_t2_sensor_observations: list[dict[str, Any]] = []
        native_t2_extrinsics: dict[str, list[Any]] | None = None
        native_t2_decision_count = 0
        native_t2_physical_step_count = 0
        if args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
            if state_only_transfer is None:
                raise RuntimeError("SB3 state-only transfer bridge disappeared after preflight")
            transfer_provenance = state_only_transfer.provenance()
            transfer_samples = {key: [] for key in (
                "rollout_physics_step",
                "command_time_ns",
                "effective_time_ns",
                "decision_index",
                "physical_state_8d",
                "pilot_state_8d",
                "normalized_observation_8d",
                "raw_action",
                "normalized_action",
                "local_velocity_yaw_command",
                "prebound_world_velocity_yaw_command",
                "emitted_world_velocity_yaw_command",
                "altitude_reference_w_m",
            )}
        elif args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
            if evaluator_manifest is None:
                raise RuntimeError("native T2 canary requires an evaluator manifest")
            motion = getattr(args, "native_t2_motion_contract", None)
            if not isinstance(motion, Mapping):
                raise RuntimeError("native T2 canary requires a preflight-bound motion contract")
            native_t2_policy = PublicRouteCoveragePolicy(
                np.asarray(public_routes_w_m, dtype=np.float64),
                waypoint_segment_seconds=_native_t2_waypoint_segment_seconds(args),
                route_start_time_ns=physics_time_ns(args.warmup_steps, args.dt),
                position_feedback_gain=float(motion["position_feedback_gain"]),
                yaw_feedback_gain=float(motion["yaw_feedback_gain"]),
            )
            native_t2_runner = T2PolicyRunner(
                native_t2_policy,
                cadence=FixedDecisionCadence(args.native_t2_decision_stride),
                bounds=WorldCommandBounds(
                    max_horizontal_speed_mps=args.native_t2_max_horizontal_speed_mps,
                    max_vertical_speed_mps=args.native_t2_max_vertical_speed_mps,
                    max_yaw_rate_rad_s=args.native_t2_max_yaw_rate_radps,
                ),
            )
            native_t2_trace_path = output_dir / NATIVE_T2_DECISION_TRACE_RELATIVE_PATH
            native_t2_trace_path.parent.mkdir(parents=True, exist_ok=True)
            native_t2_trace_stream = native_t2_trace_path.open(
                "x", encoding="utf-8", newline="\n"
            )
            native_t2_event_journal = T2CandidateEventJournal(
                episode_id=str(receipt["capture_attempt_id"]),
                event_time_origin_ns=physics_time_ns(args.warmup_steps, args.dt),
            )
            native_t2_deduplicator = SpatialCandidateDeduplicator(
                merge_radius_m=NATIVE_T2_CANDIDATE_MERGE_RADIUS_M
            )
            native_t2_extrinsics = {
                "timestamps_ns": [],
                "pos_w_m": [],
                "quat_w_ros": [],
                "intrinsic_matrices": [],
            }
            policy_provenance = native_t2_policy.provenance()
            runner_provenance = native_t2_runner.provenance()
            native_t2_trace_stream.write(
                json.dumps(
                    {
                        "schema": NATIVE_T2_TRACE_SCHEMA,
                        "record_type": "provenance",
                        "claim_boundary": "development_native_t2_canary_only",
                        "capture_attempt_id": receipt["capture_attempt_id"],
                        "policy": policy_provenance,
                        "policy_abi": runner_provenance,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            native_t2_trace_stream.flush()

        for step in range(args.steps):
            if step % args.capture_stride == 0:
                telemetry_phase = f"rollout_step_{step}"
                _enforce_foreign_native_process_guard(
                    args,
                    receipt,
                    phase=telemetry_phase,
                    output_dir=output_dir,
                )
                rollout_telemetry = resource_telemetry.sample(
                    telemetry_phase, torch_module=torch
                )
                _enforce_system_commit_guard(
                    args,
                    receipt,
                    phase=telemetry_phase,
                    output_dir=output_dir,
                    snapshot=(
                        rollout_telemetry.get("system_commit")
                        if isinstance(rollout_telemetry.get("system_commit"), Mapping)
                        else None
                    ),
                )
            phase_events = [SENSOR_PHASE_EVENT_CODES["command_write"]]
            command_time_ns = physics_time_ns(args.warmup_steps + step, args.dt)
            effective_time_ns = physics_time_ns(
                args.warmup_steps + step + 1, args.dt
            )
            pre_command_position = robot.data.root_pos_w.clone()
            pre_command_quaternion = robot.data.root_quat_w.clone()
            pre_command_linear_velocity = robot.data.root_lin_vel_w.clone()
            pre_command_angular_velocity = robot.data.root_ang_vel_b.clone()
            waypoint_index: Any | None = None
            waypoint_progress: float | None = None
            if args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
                if transfer_samples is None or state_only_transfer is None:
                    raise RuntimeError("SB3 transfer trace is not initialized")
                if step % args.sb3_decision_stride == 0:
                    decision = state_only_transfer.decide(
                        step,
                        _to_numpy(pre_command_position),
                        _to_numpy(pre_command_linear_velocity),
                        _to_numpy(pre_command_quaternion),
                        _to_numpy(pre_command_angular_velocity),
                    )
                    emitted = decision.emitted_world_velocity_yaw_command
                    transfer_velocity_w_mps = torch.as_tensor(
                        emitted[:, :3], dtype=torch.float32, device=robot.device
                    )
                    transfer_yaw_rate_radps = torch.as_tensor(
                        emitted[:, 3], dtype=torch.float32, device=robot.device
                    )
                    if transfer_altitude_reference_w_m is None:
                        transfer_altitude_reference_w_m = pre_command_position[:, 2].clone()
                    for key, value in (
                        ("rollout_physics_step", step),
                        ("command_time_ns", command_time_ns),
                        ("effective_time_ns", effective_time_ns),
                        ("decision_index", decision.decision_index),
                        ("physical_state_8d", decision.physical_state_8d),
                        ("pilot_state_8d", decision.pilot_state_8d),
                        ("normalized_observation_8d", decision.normalized_observation_8d),
                        ("raw_action", decision.raw_action),
                        ("normalized_action", decision.normalized_action),
                        ("local_velocity_yaw_command", decision.local_velocity_yaw_command),
                        ("prebound_world_velocity_yaw_command", decision.prebound_world_velocity_yaw_command),
                        ("emitted_world_velocity_yaw_command", emitted),
                        ("altitude_reference_w_m", transfer_altitude_reference_w_m),
                    ):
                        transfer_samples[key].append(_to_numpy(value))
                if (
                    transfer_velocity_w_mps is None
                    or transfer_yaw_rate_radps is None
                    or transfer_altitude_reference_w_m is None
                ):
                    raise RuntimeError("SB3 transfer did not emit its initial command")
                target, desired_pos, desired_vel, desired_yaw_rate = _velocity_yaw_controller_target(
                    robot,
                    transfer_velocity_w_mps,
                    transfer_yaw_rate_radps,
                    transfer_altitude_reference_w_m,
                    args.base_thrust,
                    args.dt * args.sb3_decision_stride,
                    torch,
                    math_utils,
                )
            elif args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
                if native_t2_runner is None or native_t2_trace_stream is None:
                    raise RuntimeError("native T2 trace is not initialized")
                if step % args.native_t2_decision_stride == 0:
                    public_observation = T2PublicFleetObservation.from_rigid_body_state(
                        physics_step=step,
                        command_time_ns=command_time_ns,
                        position_w_m=_to_numpy(pre_command_position),
                        linear_velocity_w_mps=_to_numpy(pre_command_linear_velocity),
                        quaternion_wxyz=_to_numpy(pre_command_quaternion),
                        angular_velocity_b_radps=_to_numpy(pre_command_angular_velocity),
                    )
                    native_t2_decision = native_t2_runner.decide(public_observation)
                    emitted = native_t2_decision.action.emitted_velocity_yaw_command
                    native_t2_velocity_w_mps = torch.as_tensor(
                        emitted[:, :3], dtype=torch.float32, device=robot.device
                    )
                    native_t2_yaw_rate_radps = torch.as_tensor(
                        emitted[:, 3], dtype=torch.float32, device=robot.device
                    )
                    if native_t2_altitude_reference_w_m is None:
                        native_t2_altitude_reference_w_m = pre_command_position[:, 2].clone()
                    native_t2_trace_stream.write(
                        json.dumps(
                            {
                                "schema": NATIVE_T2_TRACE_SCHEMA,
                                "record_type": "decision",
                                "rollout_physics_step": step,
                                "decision_sha256": native_t2_decision.sha256,
                                "decision": native_t2_decision.public_dict(),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    native_t2_trace_stream.flush()
                    native_t2_decision_count += 1
                if (
                    native_t2_decision is None
                    or native_t2_velocity_w_mps is None
                    or native_t2_yaw_rate_radps is None
                    or native_t2_altitude_reference_w_m is None
                ):
                    raise RuntimeError("native T2 policy did not emit its initial command")
                target, desired_pos, desired_vel, desired_yaw_rate = _velocity_yaw_controller_target(
                    robot,
                    native_t2_velocity_w_mps,
                    native_t2_yaw_rate_radps,
                    native_t2_altitude_reference_w_m,
                    args.base_thrust,
                    args.dt * args.native_t2_decision_stride,
                    torch,
                    math_utils,
                )
            else:
                target, desired_pos, desired_vel, waypoint_index, waypoint_progress = _controller_target(
                    robot,
                    waypoint_routes,
                    args.base_thrust,
                    command_time_ns / 1_000_000_000.0,
                    torch,
                    math_utils,
                )
                desired_yaw_rate = torch.zeros(
                    (AGENT_COUNT,), dtype=torch.float32, device=robot.device
                )
            robot.set_thrust_target(target)
            robot.write_data_to_sim()
            native_t2_requested_thrust: Any | None = None
            native_t2_applied_thrust: Any | None = None
            native_t2_applied_wrench: Any | None = None
            if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
                native_t2_requested_thrust = robot.data.thrust_target.clone()
                native_t2_applied_thrust = robot.data.applied_thrust.clone()
                native_t2_applied_wrench = (
                    robot.allocation_matrix @ native_t2_applied_thrust.transpose(0, 1)
                ).transpose(0, 1).clone()
            sim.step(render=False)
            phase_events.append(SENSOR_PHASE_EVENT_CODES["simulation_step"])
            robot.update(args.dt)
            phase_events.append(SENSOR_PHASE_EVENT_CODES["state_update"])
            sensor_timeline.update(contact, time_ns=effective_time_ns)
            _guard_runtime_frame("rollout", args.warmup_steps + step + 1)
            phase_events.append(SENSOR_PHASE_EVENT_CODES["safety_contact_read"])
            if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
                if (
                    native_t2_decision is None
                    or native_t2_trace_stream is None
                    or native_t2_requested_thrust is None
                    or native_t2_applied_thrust is None
                    or native_t2_applied_wrench is None
                ):
                    raise RuntimeError("native T2 physical evidence is incomplete")
                post_step_state = T2PublicFleetObservation.from_rigid_body_state(
                    physics_step=step + 1,
                    command_time_ns=effective_time_ns,
                    position_w_m=_to_numpy(robot.data.root_pos_w),
                    linear_velocity_w_mps=_to_numpy(robot.data.root_lin_vel_w),
                    quaternion_wxyz=_to_numpy(robot.data.root_quat_w),
                    angular_velocity_b_radps=_to_numpy(robot.data.root_ang_vel_b),
                )
                native_t2_evidence = T2NativeStepEvidence(
                    decision=native_t2_decision,
                    applied_physics_step=step + 1,
                    physical_command_time_ns=command_time_ns,
                    effective_time_ns=effective_time_ns,
                    requested_thrust_n=_to_numpy(native_t2_requested_thrust),
                    applied_thrust_n=_to_numpy(native_t2_applied_thrust),
                    applied_wrench_body=_to_numpy(native_t2_applied_wrench),
                    post_step_state_8d=post_step_state.state.values,
                )
                native_t2_trace_stream.write(
                    json.dumps(
                        {
                            "schema": NATIVE_T2_TRACE_SCHEMA,
                            "record_type": "physical_step",
                            "rollout_physics_step": step,
                            "global_applied_physics_step": args.warmup_steps + step + 1,
                            "evidence": native_t2_evidence.public_dict(),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                native_t2_trace_stream.flush()
                native_t2_physical_step_count += 1
            state_samples["command_time_ns"].append(command_time_ns)
            state_samples["effective_time_ns"].append(effective_time_ns)
            for key, value in (
                ("root_pos_w_m", robot.data.root_pos_w),
                ("root_quat_wxyz", robot.data.root_quat_w),
                ("root_lin_vel_w_mps", robot.data.root_lin_vel_w),
                ("root_ang_vel_b_radps", robot.data.root_ang_vel_b),
                ("desired_pos_w_m", desired_pos),
                ("desired_vel_w_mps", desired_vel),
                (
                    "target_thrust_n",
                    native_t2_requested_thrust
                    if native_t2_requested_thrust is not None
                    else robot.data.thrust_target,
                ),
                (
                    "applied_thrust_n",
                    native_t2_applied_thrust
                    if native_t2_applied_thrust is not None
                    else robot.data.applied_thrust,
                ),
            ):
                _append(state_samples, key, value)
            if args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
                if transfer_velocity_w_mps is None or transfer_yaw_rate_radps is None:
                    raise RuntimeError("SB3 transfer command cache is empty")
                emitted_command = torch.cat(
                    (transfer_velocity_w_mps, transfer_yaw_rate_radps.unsqueeze(-1)),
                    dim=-1,
                )
                for key, value in (
                    ("pre_command_root_pos_w_m", pre_command_position),
                    ("pre_command_root_quat_wxyz", pre_command_quaternion),
                    ("pre_command_root_lin_vel_w_mps", pre_command_linear_velocity),
                    ("pre_command_root_ang_vel_b_radps", pre_command_angular_velocity),
                    ("emitted_world_velocity_yaw_command", emitted_command),
                ):
                    _append(state_samples, key, value)
            elif args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
                if native_t2_velocity_w_mps is None or native_t2_yaw_rate_radps is None:
                    raise RuntimeError("native T2 command cache is empty")
                emitted_command = torch.cat(
                    (native_t2_velocity_w_mps, native_t2_yaw_rate_radps.unsqueeze(-1)),
                    dim=-1,
                )
                for key, value in (
                    ("pre_command_root_pos_w_m", pre_command_position),
                    ("pre_command_root_quat_wxyz", pre_command_quaternion),
                    ("pre_command_root_lin_vel_w_mps", pre_command_linear_velocity),
                    ("pre_command_root_ang_vel_b_radps", pre_command_angular_velocity),
                    ("emitted_world_velocity_yaw_command", emitted_command),
                ):
                    _append(state_samples, key, value)

            if step not in capture_frame_indices:
                continue
            (
                onboard_expected_positions,
                onboard_expected_world_quats,
                onboard_usd_closure,
            ) = _prepare_onboard_camera_local_mount(
                sim, stage, robot, torch
            )
            phase_events.append(SENSOR_PHASE_EVENT_CODES["camera_pose_update"])
            onboard_mount_prepare_count += 1
            overview_witness_view = _set_public_route_witness_overview_view(
                stage, overview, effective_time_ns=effective_time_ns
            )
            onboard_frame_before_render = _onboard_camera_frame_counter(onboard, torch)
            sim.render()
            phase_events.append(SENSOR_PHASE_EVENT_CODES["render"])
            sensor_timeline.update(onboard, time_ns=effective_time_ns)
            onboard_last_render_read_fence = _require_onboard_camera_render_read_fence(
                onboard, onboard_frame_before_render, torch
            )
            onboard_render_read_fence_count += 1
            sensor_timeline.update(overview, time_ns=effective_time_ns)
            sensor_timeline.update(lidar, time_ns=effective_time_ns)
            sensor_timeline.update(imu, time_ns=effective_time_ns)
            phase_events.append(SENSOR_PHASE_EVENT_CODES["rgbd_lidar_imu_read"])
            # The safety guard already advanced ContactSensor to this physical
            # step.  A retained read must refresh the same state with a zero
            # clock increment so SensorBase cannot drift ahead of simulation.
            sensor_timeline.update(contact, time_ns=effective_time_ns)
            phase_events.append(SENSOR_PHASE_EVENT_CODES["retained_contact_read"])
            onboard_fabric_closure = _camera_pose_closure(robot, onboard, torch)
            onboard_usd_closure = _onboard_camera_usd_pose_closure(
                stage, onboard_expected_positions, onboard_expected_world_quats, torch
            )
            _require_onboard_camera_usd_pose(onboard_usd_closure)
            onboard_render_closure = _camera_pose_closure_from_usd(
                onboard_usd_closure,
                onboard_expected_positions,
                onboard_expected_world_quats,
                torch,
            )
            onboard_fabric_diagnostic = _onboard_camera_fabric_pose_diagnostic(
                onboard_fabric_closure, torch
            )
            onboard_mount_diagnostic = _onboard_camera_mount_diagnostics(
                robot,
                onboard,
                root_expected_pos_w=onboard_expected_positions,
                root_expected_quat_wxyz=onboard_expected_world_quats,
                previous_root_expected_pos_w=previous_root_camera_expected_pos,
                previous_root_expected_phase=previous_root_camera_expected_phase,
                torch=torch,
            )
            previous_root_camera_expected_pos = onboard_expected_positions.detach().clone()
            previous_root_camera_expected_phase = (
                f"retained_step_{step + 1}_post_render_usd_closure"
            )
            onboard_usd_max_position_error_m = max(
                onboard_usd_max_position_error_m,
                float(onboard_usd_closure["max_position_error_m"]),
            )
            onboard_usd_min_forward_alignment_cosine = min(
                onboard_usd_min_forward_alignment_cosine,
                float(onboard_usd_closure["min_forward_alignment_cosine"]),
            )
            onboard_usd_max_orientation_error_rad = max(
                onboard_usd_max_orientation_error_rad,
                float(onboard_usd_closure["max_orientation_error_rad"]),
            )
            # Replicator numeric IDs are render-product-local and may change
            # after every Camera.update().  Read and project the mapping for
            # this exact retained frame before any semantic gate consumes it.
            onboard_semantic_metadata = _onboard_semantic_metadata(onboard)
            overview_semantic_metadata = _overview_semantic_metadata(overview)
            frame_semantic_metadata = _semantic_frame_metadata_record(
                frame_index=sensor_samples.frame_count,
                timestamp_ns=effective_time_ns,
                onboard_metadata=onboard_semantic_metadata,
                overview_metadata=overview_semantic_metadata,
                private_target_ids=private_target_ids,
            )
            onboard_semantic_metadata = frame_semantic_metadata[
                "onboard_replicator_info"
            ]
            overview_semantic_metadata = frame_semantic_metadata[
                "overview_replicator_info"
            ]
            # Native T2 events are replayed from these exact retained arrays.
            # Do not derive their public coordinates from the live Torch
            # tensors: GPU and CPU linear algebra can differ by a few ULPs,
            # which would invalidate an otherwise identical event journal.
            retained_onboard_depth_m = _to_numpy(
                onboard.data.output["distance_to_image_plane"]
            )
            retained_onboard_semantic = _to_numpy(
                onboard.data.output["semantic_segmentation"]
            )
            if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
                if (
                    native_t2_event_journal is None
                    or native_t2_deduplicator is None
                    or native_t2_extrinsics is None
                ):
                    raise RuntimeError("native T2 sensor-event state is not initialized")
                native_pos_w_m = _to_numpy(onboard_render_closure["observed_pos_w_m"])
                native_quat_w_ros = _to_numpy(
                    math_utils.convert_camera_frame_orientation_convention(
                        onboard_render_closure["observed_quat_wxyz"],
                        origin="world",
                        target="ros",
                    )
                )
                native_intrinsic_matrices = _to_numpy(onboard.data.intrinsic_matrices)
                points_world = native_rgbd_world_points(
                    retained_onboard_depth_m,
                    native_intrinsic_matrices,
                    native_pos_w_m,
                    native_quat_w_ros,
                )
                native_candidates = native_semantic_rgbd_candidates(
                    retained_onboard_semantic,
                    onboard_semantic_metadata,
                    points_world,
                    minimum_pixels=NATIVE_T2_CANDIDATE_MINIMUM_PIXELS,
                )
                native_sensor_frame_index = sensor_samples.frame_count
                for agent_id, rows in enumerate(native_candidates):
                    observation = T2PublicSensorObservation(
                        agent_id=agent_id,
                        capture_frame_index=native_sensor_frame_index,
                        sensor_time_ns=effective_time_ns,
                    )
                    native_t2_sensor_observations.append(observation.public_dict())
                    native_t2_event_journal.append(
                        observation,
                        native_t2_deduplicator.filter(rows),
                    )
                native_t2_extrinsics["timestamps_ns"].append(effective_time_ns)
                native_t2_extrinsics["pos_w_m"].append(native_pos_w_m)
                native_t2_extrinsics["quat_w_ros"].append(native_quat_w_ros)
                native_t2_extrinsics["intrinsic_matrices"].append(native_intrinsic_matrices)
            overview_content_evidence = _overview_city_content_evidence(
                overview.data.output["rgb"],
                overview.data.output["distance_to_image_plane"],
                overview.data.output["semantic_segmentation"],
                overview_semantic_metadata,
                far_clip_m=OVERVIEW_CAMERA_CLIPPING_RANGE_M[1],
            )
            overview_content_evidence = {
                **overview_content_evidence,
                "effective_time_ns": int(effective_time_ns),
                "witness_shot_index": int(overview_witness_view["shot_index"]),
            }
            overview_agent_visibility_evidence = _overview_tracked_agent_visibility_evidence(
                overview.data.output["semantic_segmentation"], overview_semantic_metadata
            )
            overview_agent_visibility_evidence = {
                **overview_agent_visibility_evidence,
                "effective_time_ns": int(effective_time_ns),
                "witness_shot_index": int(overview_witness_view["shot_index"]),
            }
            # These fields are the actual USD world pose that produced this
            # frame, not the potentially delayed Camera Fabric cache.
            closure = onboard_render_closure
            hits = lidar.data.ray_hits_w
            lidar_ranges = torch.linalg.vector_norm(hits - lidar.data.pos_w.unsqueeze(1), dim=-1)
            lidar_ranges = torch.nan_to_num(lidar_ranges, nan=lidar_cfg.max_distance, posinf=lidar_cfg.max_distance)
            lidar_ranges.clamp_(0.0, lidar_cfg.max_distance)
            visual_intrusion_evidence = _onboard_visual_intrusion_evidence(
                onboard.data.output["distance_to_image_plane"],
                lidar_ranges,
                lidar_max_distance_m=float(lidar_cfg.max_distance),
            )
            onboard_scene_content_evidence, target_visibility_evidence = (
                _onboard_semantic_frame_evidence(
                    onboard.data.output["distance_to_image_plane"],
                    onboard.data.output["semantic_segmentation"],
                    onboard_semantic_metadata,
                    _target_semantic_slots(len(evaluator_manifest["targets"]))
                    if evaluator_manifest is not None
                    else (),
                )
            )
            usd_position_errors = np.asarray(
                [row["position_error_m"] for row in onboard_usd_closure["per_agent"]],
                dtype=np.float64,
            )
            usd_forward_cosines = np.asarray(
                [row["forward_alignment_cosine"] for row in onboard_usd_closure["per_agent"]],
                dtype=np.float64,
            )
            usd_orientation_errors = np.asarray(
                [row["orientation_error_rad"] for row in onboard_usd_closure["per_agent"]],
                dtype=np.float64,
            )
            # Append the complete raw sensor frame before any content gate can
            # abort.  A failing frame remains in the on-disk spool for diagnosis.
            _enforce_runtime_storage_guard(
                args,
                receipt,
                phase=f"before_retained_frame_{sensor_samples.frame_count}",
                output_dir=output_dir,
                budget=storage_budget,
            )
            if onboard_last_render_read_fence is None:
                raise RuntimeError("retained camera frame is missing its render/read fence")
            sensor_samples.append(effective_time_ns, {
                key: _to_numpy(value)
                for key, value in (
                ("onboard_rgb", onboard.data.output["rgb"]),
                ("depth_m", retained_onboard_depth_m),
                ("semantic", retained_onboard_semantic),
                ("camera_expected_pos_w_m", closure["expected_pos_w_m"]),
                ("camera_expected_quat_wxyz", closure["expected_quat_wxyz"]),
                ("camera_observed_pos_w_m", closure["observed_pos_w_m"]),
                ("camera_observed_quat_wxyz", closure["observed_quat_wxyz"]),
                ("camera_position_error_m", closure["position_error_m"]),
                ("camera_orientation_error_rad", closure["orientation_error_rad"]),
                ("camera_usd_position_error_m", usd_position_errors),
                ("camera_usd_forward_alignment_cosine", usd_forward_cosines),
                ("camera_usd_orientation_error_rad", usd_orientation_errors),
                ("camera_fabric_observed_pos_w_m", onboard_fabric_closure["observed_pos_w_m"]),
                ("camera_fabric_observed_quat_wxyz", onboard_fabric_closure["observed_quat_wxyz"]),
                ("camera_fabric_position_error_m", onboard_fabric_closure["position_error_m"]),
                ("camera_fabric_orientation_error_rad", onboard_fabric_closure["orientation_error_rad"]),
                ("camera_render_read_pre_frame_index", onboard_last_render_read_fence["pre_frame_index"]),
                ("camera_render_read_post_frame_index", onboard_last_render_read_fence["post_frame_index"]),
                ("lidar_pos_w_m", lidar.data.pos_w),
                ("lidar_quat_wxyz", lidar.data.quat_w),
                ("lidar_ranges_m", lidar_ranges),
                ("imu_pos_w_m", imu.data.pos_w),
                ("imu_quat_wxyz", imu.data.quat_w),
                ("imu_lin_acc_b_mps2", imu.data.lin_acc_b),
                ("imu_ang_vel_b_radps", imu.data.ang_vel_b),
                ("contact_net_forces_w_n", contact.data.net_forces_w),
                )
            })
            sensor_frame_index = sensor_samples.frame_count - 1
            if sensor_frame_index != frame_semantic_metadata["frame_index"]:
                raise RuntimeError(
                    "semantic frame metadata index diverged from retained sensor spool"
                )
            semantic_frame_metadata_stream.write(
                json.dumps(
                    frame_semantic_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            semantic_frame_metadata_stream.flush()
            semantic_frame_metadata_count += 1
            if sensor_frame_index in overview_archive_index_set:
                # This is deliberately independent of every content-gate
                # result.  The live RGB/depth/semantic gates above still run
                # on every retained frame; the archive keeps only a bounded
                # audit witness and never persists overview depth.
                overview_samples.append(
                    effective_time_ns,
                    {
                        "rgb": _to_numpy(overview.data.output["rgb"])[0],
                        "semantic_segmentation": _to_numpy(
                            overview.data.output["semantic_segmentation"]
                        )[0],
                        "camera_pos_w_m": np.asarray(
                            overview_witness_view["eye_w_m"], dtype=np.float64
                        ),
                        "camera_quat_wxyz": np.asarray(
                            overview_witness_view["orientation_wxyz"],
                            dtype=np.float64,
                        ),
                        "target_w_m": np.asarray(
                            overview_witness_view["target_w_m"], dtype=np.float64
                        ),
                    },
                )
            phase_events.append(SENSOR_PHASE_EVENT_CODES["storage"])
            if tuple(phase_events) != SENSOR_PHASE_EVENT_SEQUENCE:
                raise RuntimeError(
                    "sensor phase event order diverged from the locked execution contract"
                )
            sensor_phase_samples["physics_step"].append(
                args.warmup_steps + step + 1
            )
            sensor_phase_samples["physics_time_ns"].append(effective_time_ns)
            sensor_phase_samples["event_codes"].append(tuple(phase_events))
            sensor_phase_samples["retained_contact_sha256"].append(
                np.frombuffer(
                    sensor_phase_array_digest(contact.data.net_forces_w),
                    dtype=np.uint8,
                ).copy()
            )
            sensor_phase_samples["archive_frame_index"].append(
                sensor_samples.frame_count - 1
            )
            _checkpoint(
                output_dir,
                "capture_frame_evidence_evaluated",
                physics_step=step + 1,
                overview_content_evidence=overview_content_evidence,
                overview_agent_visibility_evidence=overview_agent_visibility_evidence,
                onboard_scene_content_evidence=onboard_scene_content_evidence,
                onboard_visual_intrusion_evidence=visual_intrusion_evidence,
                target_visibility_evidence=_target_visibility_checkpoint_summary(
                    target_visibility_evidence
                ),
                raw_sensor_frame_spooled=True,
            )
            _require_overview_city_content(overview_content_evidence)
            overview_content_evidence_samples.append(overview_content_evidence)
            _require_overview_tracked_agent_visibility(overview_agent_visibility_evidence)
            overview_agent_visibility_evidence_samples.append(overview_agent_visibility_evidence)
            _require_onboard_visual_integrity(visual_intrusion_evidence)
            onboard_visual_intrusion_evidence_samples.append(visual_intrusion_evidence)
            _require_onboard_scene_content(onboard_scene_content_evidence)
            onboard_scene_content_evidence_samples.append(onboard_scene_content_evidence)
            if target_visibility_evidence is not None:
                target_visibility_evidence_samples.append(target_visibility_evidence)
            if args.control_mode == CONTROL_MODE_FIXED_PUBLIC_ROUTE:
                if waypoint_index is None or waypoint_progress is None:
                    raise RuntimeError("fixed public-route controller did not produce waypoint state")
                agent_ids = torch.arange(AGENT_COUNT, dtype=torch.int64, device=robot.device)
                desired_waypoint = waypoint_routes[agent_ids, waypoint_index]
                distance_to_waypoint = torch.linalg.vector_norm(
                    desired_waypoint - robot.data.root_pos_w, dim=-1
                )
                waypoint_reached = distance_to_waypoint <= WAYPOINT_REACHED_RADIUS_M
                route_finished = torch.full(
                    (AGENT_COUNT,),
                    waypoint_progress >= 1.0 and int(waypoint_index[0]) == waypoint_routes.shape[1] - 1,
                    dtype=torch.bool,
                    device=robot.device,
                )
                for key, value in (
                    ("waypoint_index", waypoint_index),
                    ("waypoint_progress", torch.full((AGENT_COUNT,), waypoint_progress, device=robot.device)),
                    ("desired_waypoint_w_m", desired_waypoint),
                    ("distance_to_waypoint_m", distance_to_waypoint),
                    ("waypoint_reached", waypoint_reached),
                    ("action_mode", route_finished.to(dtype=torch.int8)),
                    ("coverage_cell_id", agent_ids),
                    (
                        "task_time_s",
                        torch.full(
                            (AGENT_COUNT,),
                            effective_time_ns / 1_000_000_000.0,
                            device=robot.device,
                        ),
                    ),
                ):
                    _append(task_samples, key, value)
                task_samples["timestamps_ns"].append(effective_time_ns)
                for key, value in (
                    ("sender_agent_id", agent_ids),
                    ("message_sequence", torch.full((AGENT_COUNT,), sensor_samples.frame_count - 1, dtype=torch.int64, device=robot.device)),
                    ("message_waypoint_index", waypoint_index),
                    ("message_position_w_m", robot.data.root_pos_w),
                    ("message_velocity_w_mps", robot.data.root_lin_vel_w),
                    ("message_flags", torch.ones((AGENT_COUNT,), dtype=torch.uint8, device=robot.device)),
                ):
                    _append(message_samples, key, value)
                message_samples["timestamps_ns"].append(effective_time_ns)

        _checkpoint(output_dir, "rollout_completed", physics_steps=args.steps)

        semantic_frame_metadata_stream.flush()
        semantic_frame_metadata_stream.close()
        if semantic_frame_metadata_count != sensor_samples.frame_count:
            raise RuntimeError(
                "semantic frame metadata count does not match retained sensor frames"
            )

        stream_dir = output_dir / "streams"
        sensor_dir = output_dir / "sensors"
        stream_dir.mkdir(parents=True, exist_ok=True)
        sensor_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stream_dir / "state_action.npz", **{
            key: np.asarray(values, dtype=np.int64) if key.endswith("_ns") else np.stack(values, axis=0)
            for key, values in state_samples.items()
        })
        if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
            if (
                native_t2_trace_stream is None
                or native_t2_event_journal is None
                or native_t2_extrinsics is None
                or native_t2_decision_count < 1
                or native_t2_physical_step_count != args.steps
            ):
                raise RuntimeError("native T2 trace finalization is incomplete")
            native_t2_trace_stream.flush()
            native_t2_trace_stream.close()
            native_t2_trace_path = output_dir / NATIVE_T2_DECISION_TRACE_RELATIVE_PATH
            if not native_t2_trace_path.is_file():
                raise RuntimeError("native T2 decision trace was not created")
            if len(native_t2_extrinsics["timestamps_ns"]) != sensor_samples.frame_count:
                raise RuntimeError("native T2 camera extrinsics do not cover every retained sensor frame")
            native_t2_extrinsics_path = output_dir / NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH
            np.savez_compressed(
                native_t2_extrinsics_path,
                timestamps_ns=np.asarray(native_t2_extrinsics["timestamps_ns"], dtype=np.int64),
                pos_w_m=np.stack(native_t2_extrinsics["pos_w_m"], axis=0),
                quat_w_ros=np.stack(native_t2_extrinsics["quat_w_ros"], axis=0),
                intrinsic_matrices=np.stack(native_t2_extrinsics["intrinsic_matrices"], axis=0),
            )
            native_t2_event_path = output_dir / NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH
            native_t2_event_payload = {
                "schema": NATIVE_T2_EVENTS_SCHEMA,
                "claim_boundary": "development_native_t2_canary_only",
                "formal_benchmark_admission": False,
                "capture_attempt_id": receipt["capture_attempt_id"],
                "decision_trace": NATIVE_T2_DECISION_TRACE_RELATIVE_PATH,
                "decision_trace_sha256": _sha256(native_t2_trace_path),
                "source_observations": native_t2_sensor_observations,
                "candidate_event_journal": native_t2_event_journal.public_dict(),
                "event_time_origin_ns": physics_time_ns(args.warmup_steps, args.dt),
                "candidate_detection_is_public_rgbd_semantic_only": True,
                "private_evaluator_payload_released": False,
            }
            _write_json(native_t2_event_path, native_t2_event_payload)
            receipt["native_t2_evidence"] = {
                "task_variant_id": _native_t2_task_variant_id(args),
                "claim_boundary": "development_native_t2_canary_only",
                "decision_trace": {
                    "path": NATIVE_T2_DECISION_TRACE_RELATIVE_PATH,
                    "sha256": _sha256(native_t2_trace_path),
                    "decision_count": native_t2_decision_count,
                    "physical_step_count": native_t2_physical_step_count,
                },
                "candidate_events": {
                    "path": NATIVE_T2_EVENT_JOURNAL_RELATIVE_PATH,
                    "sha256": _sha256(native_t2_event_path),
                    "source_observation_count": len(native_t2_sensor_observations),
                    "event_count": len(
                        native_t2_event_payload["candidate_event_journal"]["submission"]["events"]
                    ),
                },
                "camera_extrinsics": {
                    "path": NATIVE_T2_CAMERA_EXTRINSICS_RELATIVE_PATH,
                    "sha256": _sha256(native_t2_extrinsics_path),
                    "frame_count": len(native_t2_extrinsics["timestamps_ns"]),
                    "world_camera_closure": "T_world_camera_from_verified_render_facing_usd_pose_converted_to_ros",
                },
            }
        target_visibility_summary: dict[str, Any] | None = None
        if evaluator_manifest is not None:
            target_slots = _target_semantic_slots(len(evaluator_manifest["targets"]))
            target_visibility_summary = _target_visibility_rollout_summary(
                target_slots, target_visibility_evidence_samples
            )
            receipt["target_visibility"] = target_visibility_summary
            _checkpoint(
                output_dir,
                "target_native_visibility_evaluated",
                target_visibility=target_visibility_summary,
            )
            if not bool(target_visibility_summary["passed"]):
                raise RuntimeError(
                    "private target visibility contract failed for one or more targets; "
                    "raw semantic frames remain in the capture spool"
                )
        if args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
            if transfer_samples is None or transfer_provenance is None:
                raise RuntimeError("SB3 transfer output cannot be finalized without its trace")
            if not transfer_samples["rollout_physics_step"]:
                raise RuntimeError("SB3 transfer emitted no decision samples")
            transfer_trace_path = output_dir / SB3_TRANSFER_TRACE_RELATIVE_PATH
            transfer_integer_fields = {
                "rollout_physics_step",
                "command_time_ns",
                "effective_time_ns",
                "decision_index",
            }
            np.savez_compressed(
                transfer_trace_path,
                **{
                    key: (
                        np.asarray(values, dtype=np.int64)
                        if key in transfer_integer_fields
                        else np.stack(values, axis=0)
                    )
                    for key, values in transfer_samples.items()
                },
            )
            _write_json(
                output_dir / SB3_TRANSFER_PROVENANCE_RELATIVE_PATH,
                {
                    "schema": "org.rivermark.isaac-sb3-state-transfer-trace.v1",
                    "claim_boundary": "development_state_only_control_wiring_smoke_only",
                    "formal_benchmark_admission": False,
                    "dataset_episode": False,
                    "task_kind": CONTROL_TRANSFER_TASK_KIND,
                    "task_variant_id": CONTROL_TRANSFER_TASK_VARIANT_ID,
                    "control_mode": CONTROL_MODE_SB3_STATE_ONLY_TRANSFER,
                    "state_phase": "pre_sim_command_state",
                    "state_action_state_phase": "pre_sim_command_state",
                    "state_action_path": "streams/state_action.npz",
                    "state_action_sha256": _sha256(stream_dir / "state_action.npz"),
                    "trace_path": SB3_TRANSFER_TRACE_RELATIVE_PATH,
                    "trace_sha256": _sha256(transfer_trace_path),
                    "trace_decision_count": len(transfer_samples["rollout_physics_step"]),
                    "trace_fields": list(transfer_samples),
                    "transfer": dict(transfer_provenance),
                },
            )
        timestamps = sensor_samples.timestamps()
        overview_timestamps = overview_samples.timestamps()
        expected_overview_timestamps = timestamps[np.asarray(overview_archive_indices)]
        if not np.array_equal(overview_timestamps, expected_overview_timestamps):
            raise RuntimeError(
                "overview evidence timestamps diverged from the frozen retained-frame schedule"
            )
        if args.control_mode == CONTROL_MODE_FIXED_PUBLIC_ROUTE:
            np.savez_compressed(
                stream_dir / "public_task.npz",
                **{
                    key: np.asarray(values, dtype=np.int64) if key.endswith("_ns") else np.stack(values, axis=0)
                    for key, values in task_samples.items()
                },
            )
            np.savez_compressed(
                stream_dir / "public_messages.npz",
                **{
                    key: np.asarray(values, dtype=np.int64) if key.endswith("_ns") else np.stack(values, axis=0)
                    for key, values in message_samples.items()
                },
            )
        _enforce_runtime_storage_guard(
            args,
            receipt,
            phase="before_onboard_archive",
            output_dir=output_dir,
            budget=storage_budget,
        )
        write_chunked_frame_archive(
            sensor_dir / "onboard_rgbd.npz",
            timestamps_ns=timestamps,
            inline_fields={},
            frame_fields={
                "rgb": sensor_samples.values("onboard_rgb"),
                "distance_to_image_plane_m": sensor_samples.values("depth_m"),
            },
        )
        sensor_samples.discard_fields_after_archive(("onboard_rgb", "depth_m"))
        _enforce_runtime_storage_guard(
            args,
            receipt,
            phase="before_semantic_archive",
            output_dir=output_dir,
            budget=storage_budget,
        )
        write_chunked_frame_archive(
            label_dir / "semantic_segmentation.npz",
            timestamps_ns=timestamps,
            inline_fields={},
            frame_fields={"semantic_segmentation": sensor_samples.values("semantic")},
        )
        sensor_samples.discard_fields_after_archive(("semantic",))
        _write_json(
            label_dir / "semantic_metadata.json",
            {
                "schema": SEMANTIC_METADATA_SCHEMA,
                "partition": "learning_labels",
                "policy_visible": False,
                "frame_metadata": {
                    "schema": SEMANTIC_FRAME_METADATA_SCHEMA,
                    "path": SEMANTIC_FRAME_METADATA_RELATIVE_PATH,
                    "frame_count": int(semantic_frame_metadata_count),
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
        np.savez_compressed(
            sensor_dir / "camera_poses.npz",
            timestamps_ns=timestamps,
            **{key: sensor_samples.values(key) for key in (
                "camera_expected_pos_w_m", "camera_expected_quat_wxyz", "camera_observed_pos_w_m",
                "camera_observed_quat_wxyz", "camera_position_error_m", "camera_orientation_error_rad",
                "camera_usd_position_error_m", "camera_usd_forward_alignment_cosine",
                "camera_usd_orientation_error_rad", "camera_fabric_observed_pos_w_m",
                "camera_fabric_observed_quat_wxyz", "camera_fabric_position_error_m",
                "camera_fabric_orientation_error_rad", "camera_render_read_pre_frame_index",
                "camera_render_read_post_frame_index"
            )},
        )
        _enforce_runtime_storage_guard(
            args,
            receipt,
            phase="before_overview_archive",
            output_dir=output_dir,
            budget=storage_budget,
        )
        write_chunked_frame_archive(
            sensor_dir / "overview_rgb.npz",
            timestamps_ns=overview_timestamps,
            inline_fields={
                "camera_pos_w_m": overview_samples.values("camera_pos_w_m"),
                "camera_quat_wxyz": overview_samples.values("camera_quat_wxyz"),
                "target_w_m": overview_samples.values("target_w_m"),
            },
            frame_fields={
                "rgb": overview_samples.values("rgb"),
                "semantic_segmentation": overview_samples.values("semantic_segmentation"),
            },
        )
        # The archive is now durable.  Copy the scalar receipt inputs before
        # releasing the low-rate spool, so no finalizer can read a closed
        # mapping and the full-rate primary spool never holds overview frames.
        overview_first_rgb = np.array(
            overview_samples.values("rgb")[0], copy=True
        )
        overview_camera_positions = np.array(
            overview_samples.values("camera_pos_w_m"), copy=True
        )
        overview_camera_targets = np.array(
            overview_samples.values("target_w_m"), copy=True
        )
        overview_camera_quaternions = np.array(
            overview_samples.values("camera_quat_wxyz"), copy=True
        )
        overview_samples.discard_fields_after_archive(
            ("rgb", "semantic_segmentation", "camera_pos_w_m", "camera_quat_wxyz", "target_w_m")
        )
        np.savez_compressed(
            sensor_dir / "lidar.npz",
            timestamps_ns=timestamps,
            pos_w_m=sensor_samples.values("lidar_pos_w_m"),
            quat_wxyz=sensor_samples.values("lidar_quat_wxyz"),
            ranges_m=sensor_samples.values("lidar_ranges_m"),
        )
        np.savez_compressed(
            sensor_dir / "imu.npz",
            timestamps_ns=timestamps,
            pos_w_m=sensor_samples.values("imu_pos_w_m"),
            quat_wxyz=sensor_samples.values("imu_quat_wxyz"),
            linear_acceleration_b_mps2=sensor_samples.values("imu_lin_acc_b_mps2"),
            angular_velocity_b_radps=sensor_samples.values("imu_ang_vel_b_radps"),
        )
        np.savez_compressed(
            sensor_dir / "contact.npz",
            timestamps_ns=timestamps,
            net_forces_w_n=sensor_samples.values("contact_net_forces_w_n"),
        )
        expected_runtime_frames = 1 + args.warmup_steps + args.steps
        if len(runtime_safety_samples["physics_step"]) != expected_runtime_frames:
            raise RuntimeError("runtime safety trace does not cover every physical frame")
        if runtime_safety_guard["checks"]["contact_samples_checked"] != expected_runtime_frames:
            raise RuntimeError("runtime safety guard counters do not match the full physical trace")
        runtime_safety_trace_path = _write_runtime_safety_trace(
            output_dir, runtime_safety_samples, np
        )
        finalize_runtime_safety_guard(
            runtime_safety_guard,
            trace_sha256=_sha256(runtime_safety_trace_path),
            physics_frame_count=expected_runtime_frames,
        )
        receipt["runtime_safety_guard"] = runtime_safety_guard
        if len(sensor_phase_samples["physics_step"]) != len(timestamps):
            raise RuntimeError("sensor phase trace count does not match retained sensor frames")
        sensor_phase_trace_path = _write_sensor_phase_trace(
            output_dir, sensor_phase_samples, np
        )
        receipt["sensor_phase_trace"] = {
            "schema": SENSOR_PHASE_TRACE_SCHEMA,
            "path": SENSOR_PHASE_TRACE_RELATIVE_PATH,
            "sha256": _sha256(sensor_phase_trace_path),
            "frame_count": len(sensor_phase_samples["physics_step"]),
            "sensor_names": list(SENSOR_PHASE_SENSOR_NAMES),
            "event_codes": list(SENSOR_PHASE_EVENT_SEQUENCE),
        }
        camera_errors = sensor_samples.values("camera_position_error_m")
        orientation_errors = sensor_samples.values("camera_orientation_error_rad")
        fabric_camera_errors = sensor_samples.values("camera_fabric_position_error_m")
        fabric_orientation_errors = sensor_samples.values(
            "camera_fabric_orientation_error_rad"
        )
        fabric_camera_error_max = float(np.max(fabric_camera_errors))
        fabric_orientation_error_max = float(np.max(fabric_orientation_errors))
        overview_witness_views = [
            _public_route_witness_view_at_time_ns(int(timestamp))
            for timestamp in overview_timestamps
        ]
        expected_witness_position = np.asarray(
            [view["eye_w_m"] for view in overview_witness_views], dtype=np.float64
        )
        expected_witness_target = np.asarray(
            [view["target_w_m"] for view in overview_witness_views], dtype=np.float64
        )
        expected_witness_quaternion = np.asarray(
            [view["orientation_wxyz"] for view in overview_witness_views], dtype=np.float64
        )
        overview_witness_position_error_m = float(
            np.max(np.linalg.norm(overview_camera_positions - expected_witness_position, axis=-1))
        )
        overview_witness_target_error_m = float(
            np.max(np.linalg.norm(overview_camera_targets - expected_witness_target, axis=-1))
        )
        overview_witness_quaternion_norm = np.linalg.norm(overview_camera_quaternions, axis=-1)
        overview_witness_orientation_min_abs_dot = float(
            np.min(
                np.abs(
                    np.sum(
                        overview_camera_quaternions
                        / overview_witness_quaternion_norm[:, None]
                        * expected_witness_quaternion,
                        axis=-1,
                    )
                )
            )
        )
        if (
            not np.isfinite(overview_witness_quaternion_norm).all()
            or np.any(overview_witness_quaternion_norm <= 1.0e-8)
            or overview_witness_position_error_m > OVERVIEW_WITNESS_POSITION_TOLERANCE_M
            or overview_witness_target_error_m > OVERVIEW_WITNESS_POSITION_TOLERANCE_M
            or overview_witness_orientation_min_abs_dot < math.cos(0.01 / 2.0)
        ):
            raise RuntimeError(
                "public route witness camera does not match its declared Isaac USD pose"
            )
        state_positions = np.stack(state_samples["root_pos_w_m"], axis=0)
        state_target = np.stack(state_samples["target_thrust_n"], axis=0)
        state_applied = np.stack(state_samples["applied_thrust_n"], axis=0)
        path_length_m = np.sum(
            np.linalg.norm(np.diff(state_positions, axis=0), axis=-1), axis=0
        )
        displacement_m = np.linalg.norm(state_positions - state_positions[0:1], axis=-1)
        tracked_agent_displacement_m = float(
            np.max(displacement_m[:, OVERVIEW_WITNESS_TRACKED_AGENT_ID])
        )
        if tracked_agent_displacement_m < OVERVIEW_WITNESS_MIN_TRACKED_AGENT_DISPLACEMENT_M:
            raise RuntimeError(
                "tracked CF2X did not move enough for the public route-witness demonstration"
            )
        tracked_agent_pixel_counts = np.asarray(
            [
                evidence["tracked_agent_pixel_count"]
                for evidence in overview_agent_visibility_evidence_samples
            ],
            dtype=np.int64,
        )
        if (
            tracked_agent_pixel_counts.size != len(timestamps)
            or int(np.min(tracked_agent_pixel_counts)) < OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS
        ):
            raise RuntimeError(
                "route-witness capture does not visibly contain its tracked CF2X in every frame"
            )
        public_task_sha256: str | None = None
        target_observability_passed: bool | None = None
        observable_target_count: int | None = None
        private_target_count = 0
        if args.control_mode == CONTROL_MODE_FIXED_PUBLIC_ROUTE:
            if evaluator_manifest is None:
                raise RuntimeError("Search3D outcome requires an evaluator manifest")
            if target_visibility_summary is None:
                raise RuntimeError("Search3D outcome requires native target-visibility evidence")
            private_target_count = len(evaluator_manifest["targets"])
            observable_target_count = int(
                target_visibility_summary["targets_meeting_visibility"]
            )
            target_observability_passed = bool(target_visibility_summary["passed"])
            public_task = {
                "schema": "org.rivermark.public-search-task.v1",
                "task_kind": "search3d",
                "task_variant_id": TASK_VARIANT_ID,
                "agent_count": AGENT_COUNT,
                "nominal_object_count": TARGET_COUNT,
                "route_generation": "fixed-public-cell-coverage-v1",
                "route_conditioning": "public_only",
                "route_family_id": route_profile.route_family_id,
                "start_anchor_id": route_profile.start_anchor_id,
                "waypoint_segment_seconds": WAYPOINT_SEGMENT_SECONDS,
                "waypoint_reached_radius_m": WAYPOINT_REACHED_RADIUS_M,
                # The public route contract is the frozen task definition. The
                # controller executes its float32 tensor representation, but
                # serialising that tensor here would change hashes for values
                # such as 9.081 and make the route identity platform-dependent.
                "routes_w_m": [
                    [list(point) for point in route]
                    for route in public_routes_w_m
                ],
                "route_contract": {
                    "geometry_source": "citylite_structural_aabb_v1",
                    "clearance_m": ROUTE_CLEARANCE_M,
                    "aabb_geometry_sha256": route_report.aabb_geometry_sha256,
                    "routes_sha256": route_contract["routes_sha256"],
                    "all_waypoints_in_command_volume": True,
                    "all_segments_clear": True,
                },
                "action_abi": {
                    "kind": "position_waypoint_with_velocity_feedforward",
                    "fields": ["desired_position_w_m", "desired_velocity_w_mps", "yaw_rate_radps"],
                    "yaw_rate_radps": 0.0,
                },
                "communication_abi": {
                    "kind": "explicit_public_broadcast",
                    "fields": ["sender_agent_id", "sequence", "waypoint_index", "position_w_m", "velocity_w_mps"],
                    "delivery": "same-sample-team-visible",
                },
                "object_coordinates_in_policy_inputs": False,
            }
            public_task_path = output_dir / "public_task.json"
            _write_json(public_task_path, public_task)
            public_task_sha256 = _sha256(public_task_path)
            _write_json(
                output_dir / "task_outcome.json",
                {
                    "schema": T1_OBSERVABILITY_OUTCOME_SCHEMA,
                    "track": T1_DATA_TRACK_ID,
                    "task_variant_id": TASK_VARIANT_ID,
                    "scoring_status": "not_scored",
                    "search_score": None,
                    "object_count": private_target_count,
                    "target_observability": target_visibility_summary,
                    "observation_rule": "native onboard semantic anonymous-slot-class visibility",
                    "policy_confirmation_events_present": False,
                    "closed_loop_scoring_eligible": False,
                    "private_manifest_commitment_sha256": receipt["evaluator_manifest_sha256"],
                    "state_action_sha256": _sha256(stream_dir / "state_action.npz"),
                    "private_coordinates_released": False,
                },
            )
        elif args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
            if evaluator_manifest is None or target_visibility_summary is None:
                raise RuntimeError("native T2 outcome requires private visibility evidence")
            if "native_t2_evidence" not in receipt:
                raise RuntimeError("native T2 outcome requires finalized native evidence")
            private_target_count = len(evaluator_manifest["targets"])
            observable_target_count = int(target_visibility_summary["targets_meeting_visibility"])
            target_observability_passed = bool(target_visibility_summary["passed"])
            public_task = {
                "schema": "org.rivermark.public-search-task.v1",
                "task_kind": NATIVE_T2_TASK_KIND,
                "task_variant_id": _native_t2_task_variant_id(args),
                "agent_count": AGENT_COUNT,
                "nominal_object_count": TARGET_COUNT,
                "route_generation": "fixed-public-cell-coverage-v1",
                "route_conditioning": "public_only",
                "route_family_id": route_profile.route_family_id,
                "start_anchor_id": route_profile.start_anchor_id,
                "waypoint_segment_seconds": _native_t2_waypoint_segment_seconds(args),
                "motion_contract": dict(getattr(args, "native_t2_motion_contract")),
                "routes_w_m": [
                    [list(point) for point in route]
                    for route in public_routes_w_m
                ],
                "route_contract": {
                    "geometry_source": "citylite_structural_aabb_v1",
                    "clearance_m": ROUTE_CLEARANCE_M,
                    "aabb_geometry_sha256": route_report.aabb_geometry_sha256,
                    "routes_sha256": route_contract["routes_sha256"],
                    "all_waypoints_in_command_volume": True,
                    "all_segments_clear": True,
                },
                "policy_abi": {
                    "kind": "bounded_public_state_velocity_yaw",
                    "state_visible_to_policy": True,
                    "semantic_target_ids_visible_to_policy": False,
                    "private_evaluator_inputs_visible_to_policy": False,
                    "action_fields": [
                        "velocity_x_mps",
                        "velocity_y_mps",
                        "velocity_z_mps",
                        "yaw_rate_radps",
                    ],
                    "decision_stride_physics_steps": args.native_t2_decision_stride,
                },
                "candidate_event_abi": {
                    "kind": "native_rgbd_semantic_anonymous_candidate",
                    "minimum_pixels": NATIVE_T2_CANDIDATE_MINIMUM_PIXELS,
                    "merge_radius_m": NATIVE_T2_CANDIDATE_MERGE_RADIUS_M,
                    "private_target_ids_released": False,
                },
                "evaluator_contract": {
                    "schema": "org.rivermark.native-t2-private-evaluation-contract.v1",
                    "event_time_origin": "post_warmup_physics_time",
                    "time_budget_s": args.steps * args.dt,
                    "match_radius_m": NATIVE_T2_CANDIDATE_MERGE_RADIUS_M,
                    "maximum_false_confirmations": 0,
                    # A canary must prove the complete public-observation ->
                    # event -> private-match path.  A zero-event rollout may
                    # still be useful for diagnosis, but is not a passing T2
                    # acceptance run.
                    "minimum_verified_matches": 1,
                    "observation_time_tolerance_s": 0.0,
                    "target_count_source": "external_private_evaluator_manifest",
                },
                "object_coordinates_in_policy_inputs": False,
            }
            public_task_path = output_dir / "public_task.json"
            _write_json(public_task_path, public_task)
            public_task_sha256 = _sha256(public_task_path)
            _write_json(
                output_dir / "task_outcome.json",
                {
                    "schema": "org.rivermark.native-t2-canary-outcome.v1",
                    "claim_boundary": "development_native_t2_canary_only",
                    "task_variant_id": _native_t2_task_variant_id(args),
                    "scoring_status": "independent_private_evaluation_required",
                    "search_score": None,
                    "object_count": private_target_count,
                    "target_observability": target_visibility_summary,
                    "policy_confirmation_events_present": True,
                    "closed_loop_scoring_eligible": False,
                    "formal_benchmark_admission": False,
                    "private_manifest_commitment_sha256": receipt["evaluator_manifest_sha256"],
                    "state_action_sha256": _sha256(stream_dir / "state_action.npz"),
                    "native_t2_evidence": receipt["native_t2_evidence"],
                    "private_coordinates_released": False,
                },
            )
        calibration = {
            "schema": "org.rivermark.isaac-swarm-calibration.v1",
            "agent_count": AGENT_COUNT,
            "coordinate_frames": {
                "world": "right-handed +Z up",
                "body": "FLU",
                "camera_optical": "OpenCV x-right y-down z-forward",
                "quaternion_order": "wxyz",
            },
            "onboard_camera": {
                "prim_expression": onboard_cfg.prim_path,
                "fixed_parent": "/World/Swarm/Agent_{agent_id}/Robot/body",
                "translation_body_m": list(CAMERA_OFFSET_BODY_M),
                "rotation_body_wxyz": list(CAMERA_OFFSET_WXYZ),
                "pitch_down_rad": ONBOARD_CAMERA_PITCH_DOWN_RAD,
                "rotation_definition": "positive body +Y rotation maps Camera world +X optical axis to (cos(pitch), 0, -sin(pitch))",
                "initial_yaw_alignment": {
                    "source": "first_public_horizontal_route_segment",
                    "yaw_rad_by_agent": list(
                        _initial_route_heading_yaws_rad(public_routes_w_m)
                    ),
                },
                "implementation": "isaaclab.sensors.Camera",
                "render_transform_write_mode": "camera_cfg_parent_relative_offset_no_runtime_world_pose_write",
                "render_transform_writes": 0,
                "mount_prepare_calls": onboard_mount_prepare_count,
                "clipping_range_m": list(ONBOARD_CAMERA_CLIPPING_RANGE_M),
                "fabric_pose_closure": {
                    "sample_phase": "post_render_camera_update",
                    "automatic_post_render_pose_read": True,
                    "authority": "diagnostic_only_camera_fabric_cache",
                    "acceptance_authority": "render_facing_usd_hierarchy",
                    "max_position_error_m": (
                        fabric_camera_error_max
                        if math.isfinite(fabric_camera_error_max)
                        else None
                    ),
                    "max_orientation_error_rad": (
                        fabric_orientation_error_max
                        if math.isfinite(fabric_orientation_error_max)
                        else None
                    ),
                    "reference_position_tolerance_m": ONBOARD_CAMERA_FABRIC_POSITION_TOLERANCE_M,
                    "reference_orientation_tolerance_rad": ONBOARD_CAMERA_FABRIC_ORIENTATION_TOLERANCE_RAD,
                    "last_render": onboard_fabric_diagnostic,
                    "last_render_parent_diagnostic": onboard_mount_diagnostic,
                },
                "render_read_frame_fence": {
                    "counter_source": "isaaclab.Camera._frame (version-pinned diagnostic counter)",
                    "claim_boundary": "proves_one_camera_buffer_update_after_each_render_not_pixel_time_alignment",
                    "verified_render_read_count": onboard_render_read_fence_count,
                    "last_pre_frame_index": (
                        _to_numpy(onboard_last_render_read_fence["pre_frame_index"]).tolist()
                        if onboard_last_render_read_fence is not None
                        else None
                    ),
                    "last_post_frame_index": (
                        _to_numpy(onboard_last_render_read_fence["post_frame_index"]).tolist()
                        if onboard_last_render_read_fence is not None
                        else None
                    ),
                },
                "usd_pose_closure": {
                    "audit_agent_ids": list(ONBOARD_CAMERA_DEMO_AGENT_IDS),
                    "max_position_error_m": onboard_usd_max_position_error_m,
                    "min_forward_alignment_cosine": onboard_usd_min_forward_alignment_cosine,
                    "max_orientation_error_rad": onboard_usd_max_orientation_error_rad,
                    "orientation_tolerance_rad": ONBOARD_CAMERA_USD_ORIENTATION_TOLERANCE_RAD,
                    "last_render": onboard_usd_closure,
                },
                "visual_intrusion_gate": {
                    "schema": VISUAL_INTRUSION_GATE_SCHEMA,
                    "status": "passed",
                    "contract": _visual_intrusion_gate_contract(),
                    "capture_frame_count": len(onboard_visual_intrusion_evidence_samples),
                    "capture_frames": onboard_visual_intrusion_evidence_samples,
                },
                "content_gate": {
                    "schema": ONBOARD_CONTENT_GATE_SCHEMA,
                    "status": "passed",
                    "contract": _onboard_content_gate_contract(),
                    "capture_frame_count": len(onboard_scene_content_evidence_samples),
                    "capture_frames": onboard_scene_content_evidence_samples,
                },
                "intrinsic_matrices": _to_numpy(onboard.data.intrinsic_matrices).tolist(),
                "image_shape_hw": [args.onboard_height, args.onboard_width],
            },
            "overview_camera": {
                "prim_path": overview_cfg.prim_path,
                "role": "Isaac-rendered demo evidence; not policy-visible",
                "implementation": "isaaclab.sensors.Camera",
                "transform_write_mode": "direct_usd_look_at_fixed_public_route_witness",
                "route_witness_schedule": _public_route_witness_schedule(),
                "clipping_range_m": list(OVERVIEW_CAMERA_CLIPPING_RANGE_M),
                "data_types": [
                    "rgb",
                    "distance_to_image_plane",
                    "semantic_segmentation",
                ],
                "evidence_archive": {
                    "schema": OVERVIEW_ARCHIVE_SCHEMA,
                    "selection_rule": "first_each_fixed_retained_frame_stride_and_final",
                    "frame_index_stride": OVERVIEW_ARCHIVE_STRIDE,
                    "source_frame_count": len(timestamps),
                    "source_frame_indices": list(overview_archive_indices),
                    "stored_fields": [
                        "rgb",
                        "semantic_segmentation",
                        "camera_pos_w_m",
                        "camera_quat_wxyz",
                        "target_w_m",
                    ],
                    "runtime_only_render_products": [
                        "distance_to_image_plane",
                    ],
                    "selection_uses_content_or_outcome": False,
                },
                "pose_closure": {
                    "expected_position_w_m": overview_closure["expected_pos_w_m"],
                    "observed_position_w_m": overview_closure["observed_pos_w_m"],
                    "position_error_m": float(overview_closure["position_error_m"]),
                    "forward_alignment_cosine": float(overview_closure["forward_alignment_cosine"]),
                    "maximum_schedule_position_error_m": overview_witness_position_error_m,
                    "maximum_schedule_target_error_m": overview_witness_target_error_m,
                    "minimum_schedule_orientation_abs_dot": overview_witness_orientation_min_abs_dot,
                    "tracked_agent_max_displacement_m": tracked_agent_displacement_m,
                },
                "content_gate": {
                    "schema": OVERVIEW_CONTENT_GATE_SCHEMA,
                    "status": "passed",
                    "contract": _overview_city_content_gate_contract(),
                    "initial_post_render": overview_initial_content_evidence,
                    "capture_frame_count": len(overview_content_evidence_samples),
                    "capture_frames": overview_content_evidence_samples,
                },
                "tracked_agent_visibility_gate": {
                    "schema": "org.rivermark.isaac-route-witness-agent-visibility.v1",
                    "status": "passed",
                    "tracked_agent_id": OVERVIEW_WITNESS_TRACKED_AGENT_ID,
                    "minimum_tracked_agent_pixels": OVERVIEW_WITNESS_MIN_TRACKED_AGENT_PIXELS,
                    "initial_post_render": overview_initial_agent_visibility_evidence,
                    "capture_frame_count": len(overview_agent_visibility_evidence_samples),
                    "capture_frames": overview_agent_visibility_evidence_samples,
                },
                "intrinsic_matrix": _to_numpy(overview.data.intrinsic_matrices)[0].tolist(),
                "image_shape_hw": [args.overview_height, args.overview_width],
            },
            "lidar": {
                "implementation": "isaaclab.sensors.ray_caster.MultiMeshRayCaster",
                "prim_expression": lidar_cfg.prim_path,
                "mesh_prim_paths": [str(getattr(item, "prim_expr", item)) for item in lidar_cfg.mesh_prim_paths],
                "ray_count": int(lidar.num_rays),
                "max_distance_m": float(lidar_cfg.max_distance),
                "mesh_id_output": False,
                "mesh_id_reason": "Disabled due to IsaacLab 2.3.2 optional mesh-ID shape mismatch; ranges are captured.",
            },
            "imu": {
                "implementation": "isaaclab.sensors.Imu",
                "prim_expression": "/World/Swarm/Agent_.*/Robot/body",
                "attachment_frame": "body_flu",
                "measurement_fields": [
                    "angular_velocity_b_radps",
                    "linear_acceleration_b_mps2",
                ],
            },
            "contact": {
                "implementation": "isaaclab.sensors.ContactSensor",
                "prim_expression": "/World/Swarm/Agent_.*/Robot/body",
                "update_period_s": float(args.dt),
                "measurement": "root_body_net_normal_force_w_n",
                "body_count": 1,
                "runtime_abort_threshold_n": 0.01,
            },
            "radar": {
                "status": "not_captured",
                "fail_closed": True,
                "reason": "No independently validated RTX radar or hardware radar source was available.",
            },
        }
        _write_json(output_dir / "calibration.json", calibration)
        structural_aabb_rows = [
            {
                "path": box.source_prim,
                "source_kind": box.category,
                "min": list(box.minimum),
                "max": list(box.maximum),
            }
            for box in structural_aabbs
        ]
        scene_payload = dict(static_scene_receipt)
        scene_payload.update(
            {
                "schema": "org.rivermark.public-isaac-scene.v1",
                "fresh_stage": True,
                "agent_prim_expression": SWARM_AGENT_PRIM_EXPRESSION,
                "agent_count": AGENT_COUNT,
                "initial_root_poses_wxyz": _to_numpy(root_pose).tolist(),
                "literal_fleet": literal_spawn_receipt,
                "flight_volume_m": _axis_volume_payload(CITY_LITE_FLIGHT_VOLUME_W_M),
                "command_volume_m": _axis_volume_payload(CITY_LITE_COMMAND_VOLUME_W_M),
                "route_clearance_m": ROUTE_CLEARANCE_M,
                "runtime_safety_guard": runtime_safety_guard,
                "route_validation": route_report.as_dict(),
                "structural_aabbs": structural_aabb_rows,
                "collision_proxies": {
                    "count": len(proxy_paths),
                    "aabb_geometry_sha256": route_report.aabb_geometry_sha256,
                    "source_aabb_geometry_sha256": route_report.aabb_geometry_sha256,
                    "representation": COLLISION_PROXY_REPRESENTATION,
                    "prim_root": COLLISION_PROXY_ROOT,
                    "collision_enabled": True,
                    "visible": False,
                },
                "lidar_geometry_coverage": {
                    "includes_city": True,
                    "includes_city_task_obstacles": True,
                    "includes_collision_proxies": True,
                    "geometry_aabb_sha256": route_report.aabb_geometry_sha256,
                },
                "capture_control_mode": args.control_mode,
                "identity_markers": marker_paths,
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
                            "prim_path": marker_paths[agent_id],
                            "semantic_tags": [
                                ["class", "agent_identity"],
                                ["agent_id", str(agent_id)],
                            ],
                        }
                        for agent_id in range(AGENT_COUNT)
                    ],
                },
                "overview_route_witness_schedule": _public_route_witness_schedule(),
                "search_object_prim_count": len(target_paths),
                "search_object_paths_listed": False,
                "legacy_route_or_target_imported": False,
                "object_coordinates_in_policy_inputs": False,
            }
        )
        if args.control_mode in (CONTROL_MODE_FIXED_PUBLIC_ROUTE, CONTROL_MODE_NATIVE_T2_CANARY):
            if public_task_sha256 is None:
                raise RuntimeError("private-target capture scene requires a public task hash")
            scene_payload.update(
                {
                    "private_evaluator_manifest_sha256": receipt[
                        "evaluator_manifest_sha256"
                    ],
                    "public_task_sha256": public_task_sha256,
                }
            )
            if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
                native_t2_evidence = receipt.get("native_t2_evidence")
                if not isinstance(native_t2_evidence, Mapping):
                    raise RuntimeError("native T2 scene requires finalized evidence")
                scene_payload.update(
                    {
                        "native_t2_task_kind": NATIVE_T2_TASK_KIND,
                        "native_t2_task_variant_id": _native_t2_task_variant_id(args),
                        "native_t2_decision_trace_sha256": native_t2_evidence[
                            "decision_trace"
                        ]["sha256"],
                        "native_t2_event_journal_sha256": native_t2_evidence[
                            "candidate_events"
                        ]["sha256"],
                        "native_t2_camera_extrinsics_sha256": native_t2_evidence[
                            "camera_extrinsics"
                        ]["sha256"],
                        "native_t2_policy_input": "public_state_only",
                    }
                )
        elif args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
            transfer_trace_path = output_dir / SB3_TRANSFER_TRACE_RELATIVE_PATH
            transfer_provenance_path = output_dir / SB3_TRANSFER_PROVENANCE_RELATIVE_PATH
            scene_payload.update(
                {
                    "control_transfer_task_kind": CONTROL_TRANSFER_TASK_KIND,
                    "control_transfer_task_variant_id": CONTROL_TRANSFER_TASK_VARIANT_ID,
                    "control_transfer_trace_sha256": _sha256(transfer_trace_path),
                    "control_transfer_provenance_sha256": _sha256(
                        transfer_provenance_path
                    ),
                    "control_transfer_state_phase": "pre_sim_command_state",
                    "control_transfer_policy_input": "state_only_8d",
                }
            )
        _write_json(
            output_dir / "scene.json",
            scene_payload,
        )
        if args.control_mode == CONTROL_MODE_FIXED_PUBLIC_ROUTE:
            capture_task = {
                "task_kind": "expert_coverage_dataset",
                "track": T1_DATA_TRACK_ID,
                "task_variant_id": TASK_VARIANT_ID,
                "route_conditioning": "public_only",
                "scoring_status": "not_scored",
                "target_observability_passed": target_observability_passed,
                "observable_target_count": observable_target_count,
                "object_count": private_target_count,
            }
            capture_modalities = {
                "rgb": "captured",
                "distance_to_image_plane": "captured",
                "multi_mesh_raycaster_lidar": "captured",
                "imu": "captured",
                "contact": "captured",
                "body_state": "captured",
                "target_and_applied_thrust": "captured",
                "public_task_state": "captured",
                "high_level_action_history": "captured",
                "public_team_messages": "captured",
                "overview_video_frames": "captured",
                "semantic_segmentation": "captured_learning_labels_not_policy_visible",
                "rtx_radar": "not_captured",
                "hardware_radar": "not_captured",
                "real_flight": "not_captured",
            }
        elif args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
            capture_task = {
                "task_kind": CONTROL_TRANSFER_TASK_KIND,
                "task_variant_id": CONTROL_TRANSFER_TASK_VARIANT_ID,
                "evaluation": "not_a_search_result",
                "private_targets_present": False,
                "decision_trace": SB3_TRANSFER_TRACE_RELATIVE_PATH,
                "decision_trace_sha256": _sha256(
                    output_dir / SB3_TRANSFER_TRACE_RELATIVE_PATH
                ),
            }
            capture_modalities = {
                "rgb": "captured_not_policy_visible",
                "distance_to_image_plane": "captured_not_policy_visible",
                "multi_mesh_raycaster_lidar": "captured_not_policy_visible",
                "imu": "captured_not_policy_visible",
                "contact": "captured",
                "body_state": "captured_state_only_policy_input",
                "target_and_applied_thrust": "captured",
                "state_only_sb3_decision_trace": "captured",
                "public_task_state": "not_applicable_control_smoke",
                "high_level_action_history": "not_policy_input",
                "public_team_messages": "not_applicable_control_smoke",
                "overview_video_frames": "captured",
                "semantic_segmentation": "captured_learning_labels_not_policy_visible",
                "rtx_radar": "not_captured",
                "hardware_radar": "not_captured",
                "real_flight": "not_captured",
            }
        else:
            native_t2_evidence = receipt.get("native_t2_evidence")
            if not isinstance(native_t2_evidence, Mapping):
                raise RuntimeError("native T2 receipt requires finalized evidence")
            capture_task = {
                "task_kind": NATIVE_T2_TASK_KIND,
                "task_variant_id": _native_t2_task_variant_id(args),
                "evaluation": "independent_private_evaluation_required",
                "private_targets_present": True,
                "formal_benchmark_admission": False,
                "decision_trace": native_t2_evidence["decision_trace"]["path"],
                "decision_trace_sha256": native_t2_evidence["decision_trace"]["sha256"],
                "candidate_events": native_t2_evidence["candidate_events"]["path"],
                "candidate_events_sha256": native_t2_evidence["candidate_events"]["sha256"],
            }
            capture_modalities = {
                "rgb": "captured_for_public_candidate_reconstruction",
                "distance_to_image_plane": "captured_for_public_candidate_reconstruction",
                "multi_mesh_raycaster_lidar": "captured_safety_gate_only",
                "imu": "captured",
                "contact": "captured",
                "body_state": "captured_public_policy_input",
                "target_and_applied_thrust": "captured_native_actuation_evidence",
                "native_t2_decision_trace": "captured",
                "native_t2_candidate_events": "captured",
                "native_t2_camera_extrinsics": "captured",
                "public_task_state": "captured",
                "high_level_action_history": "not_applicable_native_t2",
                "public_team_messages": "not_applicable_native_t2",
                "overview_video_frames": "captured",
                "semantic_segmentation": "captured_for_candidate_reconstruction_not_policy_input",
                "rtx_radar": "not_captured",
                "hardware_radar": "not_captured",
                "real_flight": "not_captured",
            }
        receipt.update(
            {
                "status": "captured",
                "ok": True,
                "finished_wall_time_ns": time.time_ns(),
                "simulator": {
                    "isaac_sim": importlib.metadata.version("isaacsim"),
                    "isaaclab_package": importlib.metadata.version("isaaclab"),
                    "isaaclab_runtime": str(getattr(__import__("isaaclab"), "__version__", "unknown")),
                    "isaaclab_module": str(Path(__import__("isaaclab").__file__).resolve()),
                    "isaaclab_source_override": str(isaaclab_source) if isaaclab_source else None,
                    "isaaclab_contrib_module": str(
                        Path(__import__("isaaclab_contrib").__file__).resolve()
                    ),
                    "isaaclab_contrib_source_override": (
                        str(isaaclab_contrib_source)
                        if isaaclab_contrib_source is not None
                        else None
                    ),
                },
                "physics": {
                    "same_world_agent_count": int(robot.num_instances),
                    "multirotor_prim_expression": robot.cfg.prim_path,
                    "literal_agent_prim_paths": list(
                        robot.cfg.literal_prim_paths
                    ),
                    "literal_fleet_spawn": literal_spawn_receipt,
                    "physics_steps": args.steps,
                    "warmup_physics_steps": args.warmup_steps,
                    "cf2x_hover_trim": {
                        "source": "md_qd_swarm.qdr_runtime_spawn_final_task",
                        "hover_thrust_per_rotor_n": HOVER_THRUST_PER_ROTOR_N,
                        "thrust_coefficient_n_per_rps_squared": THRUST_COEFFICIENT_N_PER_RPS_SQUARED,
                        "initial_hover_rps": INITIAL_HOVER_RPS,
                        "max_linear_velocity_mps": MAX_CF2X_LINEAR_VELOCITY_MPS,
                        "max_angular_velocity_radps": MAX_CF2X_ANGULAR_VELOCITY_RADPS,
                    },
                    "sensor_samples": len(timestamps),
                    "target_thrust_max_n": float(np.max(state_target)),
                    "applied_thrust_max_n": float(np.max(state_applied)),
                    "minimum_agent_path_length_m": float(np.min(path_length_m)),
                    "minimum_agent_max_displacement_m": float(np.min(np.max(displacement_m, axis=0))),
                },
                "task": capture_task,
                "city_lite_scene": {
                    "environment_id": ENVIRONMENT_ID,
                    "scene_contract_sha256": authority.contract_sha256,
                    "scene_contract_payload_sha256": authority.contract_payload_sha256,
                    "active_static_prim_count": static_scene_evidence["active_static_prim_count"],
                    "structural_aabb_count": len(structural_aabbs),
                    "collision_proxy_count": len(proxy_paths),
                    "structural_aabb_geometry_sha256": route_report.aabb_geometry_sha256,
                    "rivermark_layer_inventory_sha256": static_scene_evidence[
                        "rivermark_layer_inventory"]["inventory_sha256"]
                    ,
                    "rivermarksrc51_external_layer_count": static_scene_evidence[
                        "rivermark_layer_inventory"
                    ]["rivermarksrc51_external_layer_count"],
                },
                "quality_observations": _capture_quality_observations(
                    timestamps_ns=timestamps,
                    camera_position_errors_m=camera_errors,
                    camera_orientation_errors_rad=orientation_errors,
                    onboard_usd_max_position_error_m=onboard_usd_max_position_error_m,
                    onboard_usd_min_forward_alignment_cosine=onboard_usd_min_forward_alignment_cosine,
                    onboard_usd_max_orientation_error_rad=onboard_usd_max_orientation_error_rad,
                    overview_closure=overview_closure,
                    overview_camera_positions_w_m=overview_camera_positions,
                    overview_first_rgb=overview_first_rgb,
                    target_thrust_n=state_target,
                    applied_thrust_n=state_applied,
                    np=np,
                ),
                "modalities": capture_modalities,
                "resource_telemetry": resource_telemetry.as_dict(),
            }
        )
        # The spool is a capture-time staging area, not an evidence artifact.
        # Retain it on every failure path; discard it only after every bounded
        # raw artifact and receipt field has been constructed successfully.
        del (
            timestamps,
            overview_timestamps,
            camera_errors,
            orientation_errors,
            overview_camera_positions,
            overview_camera_targets,
            overview_camera_quaternions,
        )
        sensor_samples.discard_after_success()
        overview_samples.discard_after_success()
        _checkpoint(output_dir, "capture_receipt_finalized")
        # On Windows, SimulationApp.close() can terminate the interpreter
        # before main() resumes. Finalize both the evidence bundle and its
        # denominator record while this owning process is still alive.
        _persist_terminal_capture_state(output_dir, receipt)
    except BaseException as error:
        # Kit shutdown may terminate the interpreter before control returns to
        # main(), so persist the diagnostic while the application is alive.
        if "sensor_phase_samples" in locals() and sensor_phase_samples["physics_step"]:
            try:
                failed_phase_path = _write_sensor_phase_trace(
                    output_dir, sensor_phase_samples, np
                )
                receipt["sensor_phase_trace"] = {
                    "schema": SENSOR_PHASE_TRACE_SCHEMA,
                    "path": SENSOR_PHASE_TRACE_RELATIVE_PATH,
                    "sha256": _sha256(failed_phase_path),
                    "frame_count": len(sensor_phase_samples["physics_step"]),
                    "sensor_names": list(SENSOR_PHASE_SENSOR_NAMES),
                    "event_codes": list(SENSOR_PHASE_EVENT_SEQUENCE),
                }
            except Exception as trace_error:
                receipt.setdefault("diagnostics", {})["sensor_phase_trace_error"] = str(
                    trace_error
                )
        if "semantic_frame_metadata_stream" in locals():
            try:
                if not semantic_frame_metadata_stream.closed:
                    semantic_frame_metadata_stream.flush()
                    semantic_frame_metadata_stream.close()
            except Exception as metadata_error:
                receipt.setdefault("diagnostics", {})[
                    "semantic_frame_metadata_close_error"
                ] = str(metadata_error)
        if "native_t2_trace_stream" in locals() and native_t2_trace_stream is not None:
            try:
                if not native_t2_trace_stream.closed:
                    native_t2_trace_stream.flush()
                    native_t2_trace_stream.close()
            except Exception as trace_error:
                receipt.setdefault("diagnostics", {})[
                    "native_t2_trace_close_error"
                ] = str(trace_error)
        receipt["resource_telemetry"] = resource_telemetry.as_dict()
        receipt.update(
            {
                "status": "aborted" if isinstance(error, RuntimeSafetyAbort) else "failed",
                "ok": False,
                "finished_wall_time_ns": time.time_ns(),
                "failure": _public_capture_failure(
                    error, private_route=evaluator_manifest is not None
                ),
            }
        )
        _persist_terminal_capture_state(output_dir, receipt)
        traceback.print_exc(limit=30, file=sys.stderr)
        raise
    finally:
        _close_capture_resources(app, lease)


def _close_capture_resources(app: Any | None, lease: Any) -> None:
    """Release the exclusive launcher lease even when Kit shutdown fails."""

    try:
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)
    finally:
        lease.release()


def _acquire_capture_app_launcher_lease(
    lease: Any,
    output_dir: Path,
    receipt: dict[str, Any],
) -> None:
    """Record an exclusive-launch attempt before the lock can reject it."""

    receipt["app_launcher_lease"] = {
        "schema": "org.rivermark.app-launcher-lease.v1",
        "path": ".isaac_app_launcher.lock",
        "owner": "rivermark_benchmark.isaac_capture",
        "exclusive": True,
        "state": "acquiring",
    }
    _checkpoint(output_dir, "app_launcher_lease_acquiring")
    lease.acquire()
    receipt["app_launcher_lease"]["state"] = "acquired"
    _checkpoint(output_dir, "app_launcher_lease_acquired")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    private_manifest_path = (
        args.evaluator_private_manifest.expanduser().resolve()
        if args.evaluator_private_manifest is not None
        else None
    )
    private_manifest_retention_root = (
        args.evaluator_private_manifest_retention_root.expanduser().resolve()
        if args.evaluator_private_manifest_retention_root is not None
        else None
    )
    cleanup_records: tuple[Any, ...] = ()
    recovery_records: tuple[Any, ...] = ()
    cleanup_enabled = (
        not args.no_auto_cleanup and output_dir.parent.name.lower() == "rivermark-runs"
    )
    if cleanup_enabled:
        try:
            recovery_records = recover_crash_left_attempts(
                output_dir.parent,
                min_age_hours=24.0,
            )
            cleanup_records = cleanup_completed_runs(
                output_dir.parent,
                keep_paths=(output_dir,),
            )
        except Exception as error:
            # Housekeeping must never turn a valid capture into a missing run.
            print(f"[WARN] automatic history cleanup skipped: {error}", file=sys.stderr)
    try:
        # A capture root is its immutable ownership boundary.  Creating it
        # atomically prevents two launchers from both observing an empty
        # directory and then interleaving start markers, progress, or receipts.
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"output directory must not already exist; choose a new capture directory: {output_dir}"
        ) from exc
    source = detect_source_provenance()
    revision = source.source_revision
    dirty = source.source_worktree_dirty
    receipt: dict[str, Any] = {
        "schema": CAPTURE_SCHEMA,
        "status": "not_started",
        "ok": False,
        "created_wall_time_ns": time.time_ns(),
        "automatic_cleanup": {
            "enabled": cleanup_enabled,
            "recovered_crash_left_count": sum(
                record.status.startswith("recovered") for record in recovery_records
            ),
            "moved_count": sum(record.action == "recycle_bin" for record in cleanup_records),
            "skipped_count": sum(record.action == "skipped" for record in cleanup_records),
        },
        "source_revision": revision,
        "source_tree_sha256": source.source_tree_sha256,
        "source_worktree_dirty": dirty,
        "task_kind": (
            "search3d"
            if args.control_mode == CONTROL_MODE_FIXED_PUBLIC_ROUTE
            else (
                NATIVE_T2_TASK_KIND
                if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY
                else CONTROL_TRANSFER_TASK_KIND
            )
        ),
        "agent_count_requested": AGENT_COUNT,
        "information_profile": (
            "multisensor_rgbd_lidar_imu_state"
            if args.control_mode == CONTROL_MODE_FIXED_PUBLIC_ROUTE
            else (
                "state_only_control_plus_rgbd_semantic_events"
                if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY
                else "state_only"
            )
        ),
        "capture_integrity": {
            "online_capture": True,
            "queue_used": False,
            "queue_overflow": False,
            "silent_frame_drop": False,
            "synchronous_sensor_reads": True,
            "sensor_step_order": list(SENSOR_PHASE_EVENT_CODES),
            "per_physics_step_safety_contact_reads": True,
            "retained_contact_read_in_synchronous_sensor_phase": True,
        },
        "command": {
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "capture_stride": args.capture_stride,
            "expected_sensor_frames": _captured_frame_count(args.steps, args.capture_stride),
            "preflight_commit_percent": args.preflight_commit_percent,
            "abort_commit_percent": args.abort_commit_percent,
            "dt_s": args.dt,
            "device": args.device,
            "headless": args.headless,
            "seed": args.seed,
            "control_mode": args.control_mode,
            "base_thrust_per_rotor_n": args.base_thrust,
            "cf2x_physics_profile": {
                "hover_thrust_per_rotor_n": HOVER_THRUST_PER_ROTOR_N,
                "thrust_coefficient_n_per_rps_squared": THRUST_COEFFICIENT_N_PER_RPS_SQUARED,
                "initial_hover_rps": INITIAL_HOVER_RPS,
                "max_linear_velocity_mps": MAX_CF2X_LINEAR_VELOCITY_MPS,
                "max_angular_velocity_radps": MAX_CF2X_ANGULAR_VELOCITY_RADPS,
            },
            "drone_usd": str(args.drone_usd.resolve()),
            "drone_usd_sha256": _sha256(args.drone_usd.resolve()) if args.drone_usd.is_file() else None,
            "scene_contract": str(args.scene_contract.expanduser().resolve()),
        },
        "provenance": {
            "capture_owner": "rivermark_benchmark.isaac_capture",
            "legacy_md_qd_swarm_imported": False,
            "legacy_route_target_trace_or_evaluator_migrated": False,
            "cf2x_license": (
                "unresolved_external_asset; isaac_drone_racer configuration "
                "BSD-3-Clause; binary USD provenance not established"
            ),
        },
        "claim_boundary": {
            "formal_benchmark_admission": False,
            "hardware_validated": False,
            "radar_profile_eligible": False,
            "foundation_model_executed": False,
            "semantic_labels_policy_visible": False,
            "development_control_transfer": (
                args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER
            ),
            "development_native_t2_canary": (
                args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY
            ),
            "isaac_training": False,
            "physical_training": False,
        },
    }
    if args.control_mode == CONTROL_MODE_SB3_STATE_ONLY_TRANSFER:
        receipt["command"]["sb3_state_only_transfer"] = {
            "checkpoint": (
                str(args.sb3_checkpoint.expanduser().resolve())
                if args.sb3_checkpoint is not None
                else None
            ),
            "metadata": (
                str(args.sb3_metadata.expanduser().resolve())
                if args.sb3_metadata is not None
                else None
            ),
            "decision_stride_physics_steps": args.sb3_decision_stride,
            "world_command_bounds": {
                "max_horizontal_speed_mps": args.sb3_max_horizontal_speed_mps,
                "max_vertical_speed_mps": args.sb3_max_vertical_speed_mps,
                "max_yaw_rate_rad_s": args.sb3_max_yaw_rate_radps,
            },
        }
    elif args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY:
        receipt["command"]["native_t2_canary"] = {
            "decision_stride_physics_steps": args.native_t2_decision_stride,
            "world_command_bounds": {
                "max_horizontal_speed_mps": args.native_t2_max_horizontal_speed_mps,
                "max_vertical_speed_mps": args.native_t2_max_vertical_speed_mps,
                "max_yaw_rate_rad_s": args.native_t2_max_yaw_rate_radps,
            },
            "candidate_minimum_pixels": NATIVE_T2_CANDIDATE_MINIMUM_PIXELS,
            "candidate_merge_radius_m": NATIVE_T2_CANDIDATE_MERGE_RADIUS_M,
            "calibration_path_released": False,
        }
    resource_telemetry = ResourceTelemetry()
    try:
        _validate_args(args)
        collection_binding = _resolve_collection_binding(args)
        if collection_binding is not None:
            args.seed = int(collection_binding["episode_seed"])
            receipt["command"]["seed"] = args.seed
            receipt["collection_binding"] = collection_binding
            receipt["condition_request"] = _resolve_condition_request(args, collection_binding)
        native_t2_motion = _native_t2_motion_contract_for_capture(
            args, collection_binding=collection_binding
        )
        if native_t2_motion is not None:
            args.native_t2_motion_contract = dict(native_t2_motion["motion_contract"])
            args.native_t2_task_variant_id = str(native_t2_motion["task_variant_id"])
            native_t2_command = receipt["command"].get("native_t2_canary")
            if not isinstance(native_t2_command, dict):
                raise RuntimeError("native T2 command receipt was not initialized")
            native_t2_command["motion_contract"] = dict(native_t2_motion["motion_contract"])
            native_t2_command["route_timing_feasibility"] = dict(
                native_t2_motion["route_timing_feasibility"]
            )
        receipt["capture_attempt_id"] = _write_capture_start_marker(output_dir, receipt)
        preflight_telemetry = resource_telemetry.sample("preflight")
        receipt["resource_telemetry"] = resource_telemetry.as_dict()
        _enforce_system_commit_guard(
            args,
            receipt,
            phase="preflight",
            output_dir=output_dir,
            snapshot=(
                preflight_telemetry.get("system_commit")
                if isinstance(preflight_telemetry.get("system_commit"), Mapping)
                else None
            ),
        )
        if not args.drone_usd.resolve().is_file():
            raise FileNotFoundError(f"CF2X USD not found: {args.drone_usd.resolve()}")
        _run_capture_preflight(args, output_dir, receipt)
        runtime_lock = _bind_runtime_lock_to_args(args, receipt)
        native_t2_calibration = _load_native_t2_calibration_binding(
            args, runtime_lock=runtime_lock
        )
        if native_t2_calibration is not None:
            receipt["cf2x_runtime_calibration"] = native_t2_calibration
        authority = resolve_city_lite_authority(args.scene_contract)
        receipt["city_lite_authority"] = authority.provenance()
        evaluator_manifest: Mapping[str, Any] | None = None
        state_only_transfer: Any | None = None
        if args.control_mode in (CONTROL_MODE_FIXED_PUBLIC_ROUTE, CONTROL_MODE_NATIVE_T2_CANARY):
            if private_manifest_path is None or private_manifest_retention_root is None:
                raise RuntimeError(
                    "fixed_public_route requires an evaluator-private manifest and retention root"
                )
            _validate_private_manifest_input(
                output_dir,
                private_manifest_path,
                repository_root=_repository_root(),
            )
            retained_manifest = retain_private_evaluator_manifest(
                private_manifest_path,
                private_manifest_retention_root,
                forbidden_roots=(output_dir, _repository_root()),
            )
            evaluator_manifest = _load_external_private_evaluator_manifest(
                retained_manifest.path,
                authority,
                expected_collection_binding=collection_binding,
                expected_task_variant_id=(
                    _native_t2_task_variant_id(args)
                    if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY
                    else TASK_VARIANT_ID
                ),
                expected_native_t2_motion_contract=(
                    getattr(args, "native_t2_motion_contract")
                    if args.control_mode == CONTROL_MODE_NATIVE_T2_CANARY
                    else None
                ),
            )
            execution_window = _capture_target_visibility_execution_window(args)
            validate_private_target_execution_window(
                evaluator_manifest, execution_window=execution_window
            )
            receipt["target_visibility_execution_window"] = execution_window
            receipt["evaluator_manifest_sha256"] = retained_manifest.sha256
            receipt["evaluator_manifest_retention"] = {
                "kind": PRIVATE_MANIFEST_RETENTION_KIND,
                "sha256": retained_manifest.sha256,
                "bytes": retained_manifest.byte_count,
                "path_released": False,
                "payload_released": False,
            }
        else:
            state_only_transfer = _create_state_only_sb3_transfer(args)
            receipt["state_only_transfer"] = state_only_transfer.provenance()
        receipt["status"] = "running"
        _capture(
            args,
            output_dir,
            receipt,
            evaluator_manifest,
            authority,
            state_only_transfer,
            resource_telemetry=resource_telemetry,
        )
    except Exception as error:
        receipt["resource_telemetry"] = resource_telemetry.as_dict()
        receipt.update(
            {
                "status": "aborted" if isinstance(error, RuntimeSafetyAbort) else "failed",
                "ok": False,
                "finished_wall_time_ns": time.time_ns(),
                "failure": _public_capture_failure(
                    error, private_route=private_manifest_path is not None
                ),
            }
        )
    _persist_terminal_capture_state(output_dir, receipt)
    receipt_path = output_dir / "capture_receipt.json"
    if cleanup_enabled:
        try:
            # Run housekeeping once more after finalization so a capture that
            # fails before Isaac starts still gets the same cleanup guarantee.
            # The current output is explicitly protected from recycling.
            cleanup_completed_runs(
                output_dir.parent,
                keep_paths=(output_dir,),
            )
        except Exception as error:
            # Cleanup is reversible housekeeping and must not rewrite the
            # scientific capture status or hide a valid receipt.
            print(f"[WARN] final automatic history cleanup skipped: {error}", file=sys.stderr)
    print(json.dumps({"ok": receipt["ok"], "receipt": str(receipt_path), "status": receipt["status"]}))
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
