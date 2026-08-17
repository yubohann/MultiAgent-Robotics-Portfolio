"""Bounded target-free native Isaac smoke for the public City-Lite runtime.

The smoke loads the same City-Lite authority, eight physical CF2X assets, and
sensor constructors as the capture path. It writes only a small receipt: no
episode, target, video, or reusable sensor payload is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping, Sequence

from .citylite_scene import PUBLIC_ROUTES_W_M, make_public_route_contract, resolve_city_lite_authority, validate_public_route_contract, validate_public_routes
from .capture_lease import repository_app_launcher_lease
from .eight_cf2x_fleet import EightCF2XFleet
from .isaac_capture import (
    AGENT_COUNT,
    HOVER_THRUST_PER_ROTOR_N,
    ONBOARD_CAMERA_CLIPPING_RANGE_M,
    OVERVIEW_CAMERA_CLIPPING_RANGE_M,
    SWARM_AGENT_LITERAL_PRIM_PATHS,
    SWARM_AGENT_PRIM_EXPRESSION,
    _activate_local_isaaclab_contrib_source,
    _activate_local_isaaclab_source,
    _camera_pose_closure,
    _camera_pose_closure_from_usd,
    _compose_city_lite_stage,
    _controller_target,
    _city_lite_initial_root_states,
    _city_lite_initial_thruster_rps,
    _expected_onboard_camera_world_poses,
    _extract_structural_aabbs,
    _enforce_foreign_native_process_guard,
    _make_multirotor_cfgs,
    _make_scene,
    _make_sensors,
    _module_path_is_under,
    _onboard_camera_usd_pose_closure,
    _onboard_camera_fabric_pose_diagnostic,
    _onboard_camera_mount_diagnostics,
    _onboard_scene_content_evidence,
    _onboard_semantic_metadata,
    _onboard_visual_intrusion_evidence,
    _onboard_camera_frame_counter,
    _overview_city_content_evidence,
    _overview_semantic_metadata,
    _overview_tracked_agent_visibility_evidence,
    _require_onboard_camera_render_read_fence,
    _require_onboard_camera_usd_pose,
    _require_onboard_scene_content,
    _require_onboard_visual_integrity,
    _require_overview_city_content,
    _require_overview_tracked_agent_visibility,
    _set_public_route_witness_overview_view,
    _SensorUpdateTimeline,
    _spawn_collision_proxies,
    _spawn_identity_markers,
    _prepare_onboard_camera_local_mount,
    _to_numpy,
    _verify_literal_city_lite_spawn,
    _verify_literal_city_lite_usd_spawn,
    _waypoint_routes,
    _windows_system_commit_snapshot,
)
from .isaac_runtime_safety import evaluate_runtime_safety, physics_time_ns
from .provenance import detect_source_provenance
from .resource_telemetry import (
    DEFAULT_ABORT_COMMIT_PERCENT,
    DEFAULT_PREFLIGHT_COMMIT_PERCENT,
    ResourceTelemetry,
)
from .runtime_lock import (
    RUNTIME_AUDIT_SCHEMA,
    RUNTIME_AUDIT_OBSERVATION,
    audit_runtime_lock,
    compare_live_simulation,
    configure_simulation_cfg,
    live_gravity_matches,
    load_runtime_lock,
    locked_launcher_kwargs,
    observe_live_simulation,
    runtime_lock_sha256,
    validate_locked_launcher_environment,
)


ISAAC_SMOKE_SCHEMA = "org.rivermark.benchmark.target-free-isaac-smoke.v1"
STEP_ORDER = (
    "command_write",
    "simulation_step",
    "state_update",
    "safety_contact_read",
    "camera_pose_update",
    "render",
    "rgbd_lidar_imu_read",
    "retained_contact_read",
    "storage",
)
SMOKE_SENSOR_NAMES = ("rgb", "depth", "semantic", "lidar", "imu", "contact")
SMOKE_RESOURCE_PROFILES = frozenset(
    {
        "full",
        "no_cameras",
        "onboard_only",
        "overview_only",
        "onboard_tiled_only",
        "overview_tiled_only",
    }
)
SMOKE_TILED_CAMERA_PROFILES = frozenset({"onboard_tiled_only", "overview_tiled_only"})
_HEX_DIGITS = frozenset("0123456789abcdef")
_SYSTEM_COMMIT_SNAPSHOT_UNSET = object()


class IsaacSmokeError(RuntimeError):
    """Raised when the public smoke fails closed."""


def _sensor_profile_status(profile: str) -> dict[str, bool]:
    """Return the fixed sensor-family contract for one resource profile."""

    if profile == "full":
        return {name: True for name in SMOKE_SENSOR_NAMES}
    if profile == "no_cameras":
        return {
            "rgb": False,
            "depth": False,
            "semantic": False,
            "lidar": True,
            "imu": True,
            "contact": True,
        }
    if profile in {"onboard_only", "overview_only", "onboard_tiled_only", "overview_tiled_only"}:
        return {name: True for name in SMOKE_SENSOR_NAMES}
    raise ValueError(f"unsupported smoke resource profile: {profile}")


def _camera_profile_flags(profile: str) -> tuple[bool, bool]:
    """Return whether a profile constructs onboard and overview Cameras."""

    flags = {
        "full": (True, True),
        "no_cameras": (False, False),
        "onboard_only": (True, False),
        "overview_only": (False, True),
        "onboard_tiled_only": (True, False),
        "overview_tiled_only": (False, True),
    }
    try:
        return flags[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported smoke resource profile: {profile}") from exc


def _camera_profile_tiled_flags(profile: str) -> tuple[bool, bool]:
    """Return whether a profile uses IsaacLab's tiled Camera implementation."""

    flags = {
        "full": (False, False),
        "no_cameras": (False, False),
        "onboard_only": (False, False),
        "overview_only": (False, False),
        "onboard_tiled_only": (True, False),
        "overview_tiled_only": (False, True),
    }
    try:
        return flags[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported smoke resource profile: {profile}") from exc


def _resource_probe_contract(profile: str) -> dict[str, Any]:
    """Bind the exact native camera implementation selected for a probe."""

    onboard_enabled, overview_enabled = _camera_profile_flags(profile)
    onboard_tiled, overview_tiled = _camera_profile_tiled_flags(profile)
    return {
        "kind": "reset_only",
        "camera_render_products": {
            "onboard": onboard_enabled,
            "overview": overview_enabled,
        },
        "camera_sensor_implementations": {
            "onboard": "tiled_camera" if onboard_tiled else ("camera" if onboard_enabled else "none"),
            "overview": "tiled_camera" if overview_tiled else ("camera" if overview_enabled else "none"),
        },
        "not_capture_evidence": True,
    }


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_receipt(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "isaac_smoke_receipt.json"
    path.write_bytes(_canonical_bytes(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "isaac_smoke_receipt.sha256").write_text(f"{digest}  isaac_smoke_receipt.json\n", encoding="ascii")


def _terminalize_running_receipt(
    output_dir: Path,
    early_receipt: Mapping[str, Any],
    error: BaseException,
) -> None:
    """Turn a reserved smoke receipt into durable failed evidence.

    The smoke reserves and signs a ``running`` receipt before importing Kit so
    a crash cannot leave an unaccounted output directory.  A pre-launch error
    must replace that provisional receipt rather than leave it looking like an
    active process.  Existing terminal evidence is deliberately never
    rewritten.
    """

    path = output_dir / "isaac_smoke_receipt.json"
    receipt: dict[str, Any] = dict(early_receipt)
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, Mapping):
            if existing.get("status") in {"passed", "failed"}:
                return
            receipt = dict(existing)
    receipt.update(
        {
            "status": "failed",
            "finished_wall_time_ns": time.time_ns(),
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    )
    _write_receipt(output_dir, receipt)


def _array_digest(value: Any) -> str:
    if hasattr(value, "detach"):
        array = value.detach().cpu().contiguous().numpy()
    else:
        import numpy as np

        array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _record_system_commit_snapshot(
    system_commit: MutableMapping[str, Any],
    *,
    phase: str,
    snapshot: Mapping[str, Any] | None,
) -> None:
    """Bind a real host-commit observation before a threshold can reject it."""

    if snapshot is None:
        return
    commit_percent = snapshot.get("commit_percent")
    if (
        not isinstance(commit_percent, (int, float))
        or isinstance(commit_percent, bool)
        or not math.isfinite(float(commit_percent))
    ):
        return
    snapshot_copy = dict(snapshot)
    system_commit["last_phase"] = phase
    system_commit["last_snapshot"] = snapshot_copy
    maximum = system_commit.get("maximum_observed_percent")
    if (
        not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or not math.isfinite(float(maximum))
        or float(commit_percent) > float(maximum)
    ):
        system_commit["maximum_observed_percent"] = float(commit_percent)
        system_commit["maximum_phase"] = phase
        system_commit["maximum_snapshot"] = snapshot_copy


def _sample_system_commit(
    resource_telemetry: ResourceTelemetry,
    system_commit: MutableMapping[str, Any],
    *,
    phase: str,
    torch_module: Any | None = None,
) -> Mapping[str, Any] | None:
    """Sample telemetry and retain its exact commit observation in the receipt."""

    sample = resource_telemetry.sample(phase, torch_module=torch_module)
    snapshot = sample.get("system_commit")
    if not isinstance(snapshot, Mapping):
        return None
    _record_system_commit_snapshot(system_commit, phase=phase, snapshot=snapshot)
    return snapshot


def _sync_early_receipt_telemetry(
    early_receipt: MutableMapping[str, Any],
    resource_telemetry: ResourceTelemetry,
) -> None:
    """Bind sampled pre-launch telemetry before its guard can reject a smoke."""

    early_receipt["resource_telemetry"] = resource_telemetry.as_dict()


def _check_commit(
    *,
    threshold_percent: float,
    phase: str,
    system_commit: MutableMapping[str, Any] | None = None,
    snapshot: Mapping[str, Any] | None | object = _SYSTEM_COMMIT_SNAPSHOT_UNSET,
) -> Mapping[str, Any] | None:
    if snapshot is _SYSTEM_COMMIT_SNAPSHOT_UNSET:
        observed_snapshot = _windows_system_commit_snapshot()
    elif isinstance(snapshot, Mapping):
        observed_snapshot = snapshot
    else:
        observed_snapshot = None
    if system_commit is not None:
        _record_system_commit_snapshot(
            system_commit,
            phase=phase,
            snapshot=observed_snapshot,
        )
    commit_percent = (
        observed_snapshot.get("commit_percent")
        if observed_snapshot is not None
        else None
    )
    if (
        isinstance(commit_percent, (int, float))
        and not isinstance(commit_percent, bool)
        and math.isfinite(float(commit_percent))
        and float(commit_percent) >= threshold_percent
    ):
        raise IsaacSmokeError(
            f"Windows system commit is {float(commit_percent):.2f}% at {phase}; "
            f"limit is {threshold_percent:.2f}%"
        )
    return observed_snapshot


def _close_smoke_app(app: Any) -> None:
    """Exit a standalone smoke after its terminal receipt is durable.

    A smoke retains no Replicator output or sensor payload.  Isaac Sim 5.1
    documents this close mode as immediate exit without cleanup; using it here
    prevents a failed bounded smoke from holding a large Kit allocation while
    the process performs an irrelevant full teardown.
    """

    app.close(wait_for_replicator=False, skip_cleanup=True)


def _close_smoke_resources(app: Any | None, lease: Any) -> None:
    """Close Kit, but never allow a close failure to retain the lease."""

    try:
        if app is not None:
            _close_smoke_app(app)
    finally:
        lease.release()


def _is_system_commit_snapshot(value: Any) -> bool:
    """Return whether a receipt contains a coherent Windows commit snapshot."""

    if not isinstance(value, Mapping):
        return False
    total = value.get("commit_total_bytes")
    limit = value.get("commit_limit_bytes")
    peak = value.get("commit_peak_bytes")
    percent = value.get("commit_percent")
    if (
        not all(isinstance(item, int) and not isinstance(item, bool) for item in (total, limit, peak))
        or total < 0
        or limit <= 0
        or peak < total
        or not isinstance(percent, (int, float))
        or isinstance(percent, bool)
        or not math.isfinite(float(percent))
    ):
        return False
    return math.isclose(
        float(percent),
        100.0 * float(total) / float(limit),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )


def _system_commit_receipt_is_coherent(value: Any, resource_telemetry: Any) -> bool:
    """Check that the receipt binds a real peak to sampled host telemetry."""

    if not isinstance(value, Mapping):
        return False
    if not isinstance(resource_telemetry, Mapping):
        return False
    samples = resource_telemetry.get("samples")
    if not isinstance(samples, list):
        return False

    def is_bound_snapshot(phase: Any, snapshot: Any) -> bool:
        return (
            isinstance(phase, str)
            and bool(phase)
            and _is_system_commit_snapshot(snapshot)
            and any(
                isinstance(sample, Mapping)
                and sample.get("phase") == phase
                and sample.get("system_commit") == snapshot
                for sample in samples
            )
        )

    preflight = value.get("preflight")
    if preflight is not None and (
        not _is_system_commit_snapshot(preflight)
        or not is_bound_snapshot("preflight", preflight)
    ):
        return False
    maximum = value.get("maximum_observed_percent")
    maximum_phase = value.get("maximum_phase")
    maximum_snapshot = value.get("maximum_snapshot")
    last_phase = value.get("last_phase")
    last_snapshot = value.get("last_snapshot")
    if maximum is None:
        return (
            maximum_phase is None
            and maximum_snapshot is None
            and last_phase is None
            and last_snapshot is None
        )
    if (
        not isinstance(maximum, (int, float))
        or isinstance(maximum, bool)
        or not math.isfinite(float(maximum))
        or not isinstance(maximum_phase, str)
        or not maximum_phase
        or not is_bound_snapshot(maximum_phase, maximum_snapshot)
        or not math.isclose(
            float(maximum),
            float(maximum_snapshot["commit_percent"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not isinstance(last_phase, str)
        or not last_phase
        or not is_bound_snapshot(last_phase, last_snapshot)
    ):
        return False
    return True


def validate_smoke_receipt(
    payload: Any, *, runtime_lock: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ("receipt must be an object",)
    if payload.get("schema") != ISAAC_SMOKE_SCHEMA:
        errors.append("schema mismatch")
    if payload.get("status") not in {"passed", "failed"}:
        errors.append("status must be passed or failed")
    profile = payload.get("resource_probe_profile", "full")
    if profile not in SMOKE_RESOURCE_PROFILES:
        errors.append("resource probe profile is unsupported")
    is_full_profile = profile == "full"
    camera_flags = _camera_profile_flags(str(profile)) if profile in SMOKE_RESOURCE_PROFILES else None
    expected_probe = (
        _resource_probe_contract(str(profile))
        if profile in SMOKE_RESOURCE_PROFILES and not is_full_profile
        else None
    )
    boundary = payload.get("claim_boundary")
    expected_boundary = {
        "formal_episode": False,
        "benchmark_score": False,
        "private_targets_present": False,
        "evaluator_truth_used": False,
        "sensor_payload_retained": False,
    }
    if boundary != expected_boundary:
        errors.append("claim boundary is incomplete")
    if payload.get("status") == "passed":
        if payload.get("step_order") != list(STEP_ORDER):
            errors.append("physics/sensor step order is not frozen")
        if payload.get("agent_count") != AGENT_COUNT:
            errors.append("smoke did not initialize eight agents")
        if payload.get("search_target_prim_count") != 0:
            errors.append("target-free smoke contains search targets")
        if not _system_commit_receipt_is_coherent(
            payload.get("system_commit"),
            payload.get("resource_telemetry"),
        ):
            errors.append("system commit guard evidence is incomplete or incoherent")
        sensors = payload.get("sensors")
        expected_sensors = _sensor_profile_status(str(profile)) if profile in SMOKE_RESOURCE_PROFILES else None
        if not isinstance(sensors, Mapping) or expected_sensors is None or dict(sensors) != expected_sensors:
            errors.append("sensor families do not match the declared resource profile")
        runtime_audit = payload.get("runtime_audit")
        if not isinstance(runtime_audit, Mapping):
            errors.append("runtime lock audit binding is missing")
        else:
            if runtime_audit.get("status") != "passed":
                errors.append("runtime lock audit did not pass")
            if runtime_audit.get("schema") != RUNTIME_AUDIT_SCHEMA:
                errors.append("runtime lock audit schema is not bound")
            if runtime_audit.get("runtime_lock_sha256") != payload.get("runtime_lock_sha256"):
                errors.append("runtime lock audit hash is not bound")
            if runtime_audit.get("profile_id") != payload.get("runtime_profile_id"):
                errors.append("runtime lock audit profile is not bound")
            if runtime_audit.get("configuration_observation") != RUNTIME_AUDIT_OBSERVATION:
                errors.append("runtime lock audit observation provenance is missing")
            audit_observed = runtime_audit.get("observed")
            if (
                not isinstance(audit_observed, Mapping)
                or audit_observed.get("configuration_observation")
                != RUNTIME_AUDIT_OBSERVATION
            ):
                errors.append("runtime lock audit observed environment is not provenance-bound")
            audit_issues = runtime_audit.get("issues")
            if not isinstance(audit_issues, list) or audit_issues:
                errors.append("runtime lock audit contains unresolved issues")
        launcher = payload.get("launcher")
        if (
            not isinstance(launcher, Mapping)
            or launcher.get("headless") is not True
            or launcher.get("enable_cameras") is not True
            or launcher.get("device") not in {"cpu", "cuda", "cuda:0"}
            or launcher.get("rendering_mode") not in {"performance", "balanced", "quality"}
            or launcher.get("livestream") != 0
            or launcher.get("xr") is not False
            or launcher.get("distributed") is not False
            or not isinstance(launcher.get("kit_args"), str)
            or not isinstance(launcher.get("experience"), Mapping)
        ):
            errors.append("launcher configuration is not bound")
        simulation = payload.get("simulation")
        if (
            not isinstance(simulation, Mapping)
            or simulation.get("device") not in {"cpu", "cuda", "cuda:0"}
            or not isinstance(simulation.get("dt_s"), (int, float))
            or isinstance(simulation.get("dt_s"), bool)
            or not math.isfinite(float(simulation.get("dt_s")))
            or float(simulation.get("dt_s")) <= 0.0
            or not isinstance(simulation.get("gravity_w_mps2"), list)
            or len(simulation.get("gravity_w_mps2")) != 3
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in simulation.get("gravity_w_mps2")
            )
            or simulation.get("agent_count") != AGENT_COUNT
            or simulation.get("render_interval") != 1
            or simulation.get("use_fabric") is not True
            or not isinstance(simulation.get("config_digests"), Mapping)
        ):
            errors.append("simulation configuration is not bound")
        if isinstance(launcher, Mapping) and isinstance(simulation, Mapping):
            if launcher.get("device") != simulation.get("device"):
                errors.append("launcher and simulation devices are not bound")
            config_digests = simulation.get("config_digests")
            if not isinstance(config_digests, Mapping) or set(config_digests) != {"render", "fabric", "physx"}:
                errors.append("simulation configuration digests are incomplete")
            else:
                for name, value in config_digests.items():
                    if not isinstance(value, Mapping) or not isinstance(value.get("settings"), Mapping) or not isinstance(value.get("sha256"), str) or len(value.get("sha256")) != 64:
                        errors.append(f"simulation {name} configuration digest is malformed")
        sensor_digests = payload.get("sensor_last_frame_sha256")
        if is_full_profile:
            if (
                not isinstance(sensor_digests, Mapping)
                or set(sensor_digests) != set(SMOKE_SENSOR_NAMES)
                or any(not _is_sha256(sensor_digests.get(name)) for name in SMOKE_SENSOR_NAMES)
            ):
                errors.append("last-frame sensor digests are incomplete or malformed")
        elif sensor_digests != {}:
            errors.append("reset-only resource probes must not retain sensor-frame digests")
        if not is_full_profile and camera_flags is not None and expected_probe is not None:
            legacy_probe = {
                "kind": "reset_only",
                "camera_render_products": {
                    "onboard": camera_flags[0],
                    "overview": camera_flags[1],
                },
                "not_capture_evidence": True,
            }
            accepted_probes = [expected_probe]
            if profile not in SMOKE_TILED_CAMERA_PROFILES:
                accepted_probes.append(legacy_probe)
            if profile == "no_cameras":
                accepted_probes.append(
                    {
                        "kind": "reset_only",
                        "camera_render_products_constructed": False,
                        "not_capture_evidence": True,
                    }
                )
            if payload.get("resource_probe") not in accepted_probes:
                errors.append("resource probe render-product contract is not bound")
            request = payload.get("resource_probe_request")
            if request is not None and request != expected_probe:
                errors.append("resource probe requested camera implementation is not bound")
            if profile in SMOKE_TILED_CAMERA_PROFILES and request != expected_probe:
                errors.append("tiled resource probe implementation request is not bound")
        if runtime_lock is not None:
            try:
                from .runtime_lock import compare_live_simulation, runtime_lock_sha256, validate_runtime_lock

                lock_issues = validate_runtime_lock(runtime_lock)
                if lock_issues:
                    errors.append("runtime lock object is invalid")
                if payload.get("runtime_lock_sha256") != runtime_lock_sha256(runtime_lock):
                    errors.append("runtime lock object hash is not bound")
                if payload.get("runtime_profile_id") != runtime_lock.get("profile_id"):
                    errors.append("runtime lock object profile is not bound")
                if isinstance(launcher, Mapping) and dict(launcher) != dict(runtime_lock.get("launcher", {})):
                    errors.append("launcher configuration does not match the runtime lock")
                if isinstance(simulation, Mapping) and dict(simulation) != dict(runtime_lock.get("simulation", {})):
                    errors.append("simulation configuration does not match the runtime lock")
                observed = payload.get("runtime_observed")
                if isinstance(observed, Mapping):
                    if compare_live_simulation(runtime_lock, observed):
                        errors.append("live runtime observation does not match the runtime lock")
                else:
                    errors.append("live runtime observation is missing")
            except (KeyError, TypeError, ValueError):
                errors.append("runtime lock object could not be compared")
        runtime_observed = payload.get("runtime_observed")
        if not isinstance(runtime_observed, Mapping) or not isinstance(simulation, Mapping):
            errors.append("live simulation device is not bound")
        elif runtime_observed.get("device") != simulation.get("device"):
            errors.append("live simulation device is not bound")
        elif (
            runtime_observed.get("physics_dt_s") != simulation.get("dt_s")
            or runtime_observed.get("rendering_dt_s")
            != simulation.get("dt_s") * simulation.get("render_interval")
            or not live_gravity_matches(
                simulation.get("gravity_w_mps2"),
                runtime_observed.get("gravity_w_mps2"),
            )
            or runtime_observed.get("rendering_mode") != launcher.get("rendering_mode")
            or runtime_observed.get("render_interval") != simulation.get("render_interval")
            or runtime_observed.get("use_fabric") != simulation.get("use_fabric")
            or (camera_flags is not None and any(camera_flags) and runtime_observed.get("rtx_sensors_active") is not True)
            or (camera_flags is not None and not any(camera_flags) and runtime_observed.get("rtx_sensors_active") is not False)
            or runtime_observed.get("config_digests")
            != {
                name: value.get("sha256")
                for name, value in simulation.get("config_digests", {}).items()
            }
        ):
            errors.append("live renderer or simulation configuration is not bound")
        if (
            not isinstance(runtime_observed, Mapping)
            or runtime_observed.get("configuration_observation")
            != "public_simulation_context_and_locked_cfg"
        ):
            errors.append("live runtime observation provenance is missing")
        if payload.get("source", {}).get("source_worktree_dirty") is not False:
            errors.append("passing public smoke requires a clean source tree")
        foreign_guard = payload.get("foreign_native_process_guard")
        if (
            not isinstance(foreign_guard, Mapping)
            or foreign_guard.get("schema")
            != "org.rivermark.foreign-native-process-guard.v2"
            or foreign_guard.get("status") != "active"
            or not isinstance(foreign_guard.get("sample_count"), int)
            or isinstance(foreign_guard.get("sample_count"), bool)
            or int(foreign_guard.get("sample_count")) < 4
            or foreign_guard.get("last_census_status") != "available"
            or foreign_guard.get("maximum_candidate_count") != 0
            or foreign_guard.get("maximum_candidate_private_commit_bytes") != 0
        ):
            errors.append("runtime foreign-native process guard is not bound")
        steps = payload.get("physics_steps")
        trace = payload.get("step_trace")
        if is_full_profile:
            if not isinstance(steps, int) or steps < 1 or not isinstance(trace, list) or len(trace) != steps:
                errors.append("step trace does not cover every physics step")
            elif any(
                not isinstance(row, Mapping)
                or row.get("events") != list(STEP_ORDER)
                or row.get("step") != index + 1
                for index, row in enumerate(trace)
            ):
                errors.append("step trace contains an invalid or out-of-order event")
            frame_fences = payload.get("onboard_camera_render_read_fences")
            if (
                not isinstance(frame_fences, list)
                or len(frame_fences) != steps
                or any(
                    not isinstance(fence, Mapping)
                    or not isinstance(fence.get("pre_frame_index"), list)
                    or not isinstance(fence.get("post_frame_index"), list)
                    or len(fence["pre_frame_index"]) != AGENT_COUNT
                    or len(fence["post_frame_index"]) != AGENT_COUNT
                    or any(
                        isinstance(before, bool)
                        or isinstance(after, bool)
                        or not isinstance(before, int)
                        or not isinstance(after, int)
                        or before < 0
                        or after != before + 1
                        for before, after in zip(
                            fence["pre_frame_index"], fence["post_frame_index"], strict=True
                        )
                    )
                    for fence in frame_fences
                )
            ):
                errors.append("onboard camera render/read frame fence is invalid")
        elif steps != 0 or trace != []:
            errors.append("reset-only resource probe must not report physics-step evidence")
    return tuple(errors)


def run_target_free_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise IsaacSmokeError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    early_receipt: dict[str, Any] = {
        "schema": ISAAC_SMOKE_SCHEMA,
        "status": "failed",
        "created_wall_time_ns": time.time_ns(),
        "claim_boundary": {
            "formal_episode": False,
            "benchmark_score": False,
            "private_targets_present": False,
            "evaluator_truth_used": False,
            "sensor_payload_retained": False,
        },
    }
    try:
        resource_telemetry = ResourceTelemetry()
        system_commit: dict[str, Any] = {
            "preflight": None,
            "maximum_observed_percent": None,
        }
        early_receipt["system_commit"] = system_commit
        _sample_system_commit(
            resource_telemetry,
            system_commit,
            phase="smoke_start",
        )
        _enforce_foreign_native_process_guard(
            args,
            early_receipt,
            phase="smoke_start",
        )
        _sync_early_receipt_telemetry(early_receipt, resource_telemetry)
        return _run_target_free_smoke_checked(
            args,
            output_dir,
            early_receipt,
            resource_telemetry,
            system_commit,
        )
    except BaseException as exc:
        _terminalize_running_receipt(output_dir, early_receipt, exc)
        raise


def _run_target_free_smoke_checked(
    args: argparse.Namespace,
    output_dir: Path,
    early_receipt: dict[str, Any],
    resource_telemetry: ResourceTelemetry,
    system_commit: dict[str, Any],
) -> dict[str, Any]:
    """Run after the output has been reserved for fail-closed evidence."""

    profile = str(getattr(args, "resource_probe_profile", "full"))
    if profile not in SMOKE_RESOURCE_PROFILES:
        raise IsaacSmokeError(
            "--resource-probe-profile must be one of: full, no_cameras, onboard_only, overview_only, "
            "onboard_tiled_only, overview_tiled_only"
        )
    if not 1 <= args.steps <= 32:
        raise IsaacSmokeError("--steps must be in [1, 32]")
    if profile != "full" and args.steps != 1:
        raise IsaacSmokeError("reset-only resource probes require --steps 1")
    if args.preflight_commit_percent <= 0.0 or args.abort_commit_percent <= args.preflight_commit_percent or args.abort_commit_percent > 95.0:
        raise IsaacSmokeError("commit thresholds must satisfy 0 < preflight < abort <= 95")
    maximum_foreign_commit_gib = float(
        getattr(args, "maximum_foreign_native_private_commit_gib", 8.0)
    )
    if not math.isfinite(maximum_foreign_commit_gib) or maximum_foreign_commit_gib <= 0.0:
        raise IsaacSmokeError(
            "--maximum-foreign-native-private-commit-gib must be finite and positive"
        )
    if shutil.disk_usage(output_dir).free < int(args.minimum_free_gib * 1024**3):
        raise IsaacSmokeError("output volume does not meet the smoke free-space budget")

    source = detect_source_provenance(Path(__file__).resolve().parents[2])
    lock = load_runtime_lock(args.runtime_lock)
    runtime_audit = audit_runtime_lock(
        args.runtime_lock,
        isaaclab_source=args.isaaclab_source,
        scene_contract=args.scene_contract,
        cf2x_usd=args.drone_usd,
    )
    base_receipt: dict[str, Any] = {
        **early_receipt,
        "status": "running",
        "runtime_lock_sha256": runtime_lock_sha256(lock),
        "runtime_profile_id": lock["profile_id"],
        "runtime_audit": runtime_audit,
        "source": source.as_dict(),
        "agent_count": AGENT_COUNT,
        "launcher": dict(lock["launcher"]),
        "simulation": dict(lock["simulation"]),
        "resource_probe_profile": profile,
        "physics_steps": int(args.steps) if profile == "full" else 0,
        "step_order": list(STEP_ORDER),
        "budgets": {
            "minimum_free_gib": float(args.minimum_free_gib),
            "preflight_commit_percent": float(args.preflight_commit_percent),
            "abort_commit_percent": float(args.abort_commit_percent),
            "maximum_foreign_native_private_commit_gib": float(
                getattr(args, "maximum_foreign_native_private_commit_gib", 8.0)
            ),
            "raw_sensor_payload_retained": False,
        },
    }
    if profile != "full":
        base_receipt["resource_probe_request"] = _resource_probe_contract(profile)
    if runtime_audit.get("status") != "passed":
        raise IsaacSmokeError("runtime lock audit failed")
    if source.source_worktree_dirty:
        raise IsaacSmokeError("public Isaac smoke requires a clean Git worktree")
    # Share this mapping with the early handler so a reset-time rejection cannot
    # regress to the lower preflight-only observation in its terminal receipt.
    early_receipt["system_commit"] = system_commit
    base_receipt["system_commit"] = system_commit
    preflight_commit = _sample_system_commit(
        resource_telemetry,
        system_commit,
        phase="preflight",
    )
    system_commit["preflight"] = preflight_commit
    _sync_early_receipt_telemetry(early_receipt, resource_telemetry)
    _check_commit(
        threshold_percent=args.preflight_commit_percent,
        phase="preflight",
        system_commit=system_commit,
        snapshot=preflight_commit,
    )
    lease = repository_app_launcher_lease(
        Path(__file__).resolve().parents[2],
        metadata={
            "output_dir": str(output_dir),
            "source_revision": source.source_revision,
            "owner": "rivermark_benchmark.isaac_smoke",
        },
    )
    before_launcher_commit = _sample_system_commit(
        resource_telemetry,
        system_commit,
        phase="before_app_launcher",
    )
    _sync_early_receipt_telemetry(early_receipt, resource_telemetry)
    _check_commit(
        threshold_percent=args.preflight_commit_percent,
        phase="before_app_launcher",
        system_commit=system_commit,
        snapshot=before_launcher_commit,
    )
    _enforce_foreign_native_process_guard(
        args,
        base_receipt,
        phase="before_app_launcher",
    )
    base_receipt["app_launcher_lease"] = {
        "schema": "org.rivermark.app-launcher-lease.v1",
        "path": ".isaac_app_launcher.lock",
        "owner": "rivermark_benchmark.isaac_smoke",
        "exclusive": True,
        "state": "not_attempted",
    }
    _write_receipt(output_dir, base_receipt)

    app = None
    try:
        if bool(args.headless) != bool(lock["launcher"]["headless"]):
            raise IsaacSmokeError("--headless conflicts with the runtime lock")
        isaaclab_source = _activate_local_isaaclab_source(args.isaaclab_source)
        if isaaclab_source is None:
            raise IsaacSmokeError("locked IsaacLab source did not activate")
        contrib_relative = lock["isaaclab_contrib_source"]["relative_path"]
        isaaclab_contrib_source = _activate_local_isaaclab_contrib_source(
            isaaclab_source.parent.joinpath(*str(contrib_relative).split("/"))
        )
        validate_locked_launcher_environment(lock)
        base_receipt["app_launcher_lease"]["state"] = "acquiring"
        _write_receipt(output_dir, base_receipt)
        lease.acquire()
        base_receipt["app_launcher_lease"]["state"] = "acquired"
        _write_receipt(output_dir, base_receipt)
        from isaaclab.app import AppLauncher

        app = AppLauncher(locked_launcher_kwargs(lock, isaaclab_source)).app
        import omni.usd
        import torch
        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        from isaaclab_contrib.actuators import ThrusterCfg
        from isaaclab_contrib.assets import Multirotor, MultirotorCfg
        _enforce_foreign_native_process_guard(
            args,
            base_receipt,
            phase="after_app_launcher",
        )
        after_launcher_commit = _sample_system_commit(
            resource_telemetry,
            system_commit,
            phase="after_app_launcher",
            torch_module=torch,
        )
        _check_commit(
            threshold_percent=args.abort_commit_percent,
            phase="after_app_launcher",
            system_commit=system_commit,
            snapshot=after_launcher_commit,
        )

        if not _module_path_is_under(__import__("isaaclab"), isaaclab_source / "isaaclab"):
            raise IsaacSmokeError("locked smoke imported isaaclab from an unbound source")
        if not _module_path_is_under(
            __import__("isaaclab_contrib"),
            isaaclab_contrib_source / "isaaclab_contrib",
        ):
            raise IsaacSmokeError("locked smoke imported isaaclab_contrib from an unbound source")

        omni.usd.get_context().new_stage()
        authority = resolve_city_lite_authority(args.scene_contract)
        stage, scene_evidence = _compose_city_lite_stage(authority)
        structural_aabbs = _extract_structural_aabbs(stage)
        route_report = validate_public_route_contract(
            make_public_route_contract(structural_aabbs), PUBLIC_ROUTES_W_M, structural_aabbs
        )
        validate_public_routes(PUBLIC_ROUTES_W_M, structural_aabbs)
        proxies = _spawn_collision_proxies(stage, structural_aabbs)
        sim_cfg = sim_utils.SimulationCfg(
            dt=float(lock["simulation"]["dt_s"]),
            device=lock["simulation"]["device"],
        )
        configure_simulation_cfg(sim_cfg, lock)
        sim = sim_utils.SimulationContext(sim_cfg)
        target_paths = _make_scene(sim_utils, None)
        if target_paths or stage.GetPrimAtPath("/World/SearchTargets").IsValid():
            raise IsaacSmokeError("target-free smoke unexpectedly created SearchTargets")
        cfg_args = SimpleNamespace(drone_usd=args.drone_usd, dt=float(lock["simulation"]["dt_s"]))
        members = tuple(Multirotor(cfg) for cfg in _make_multirotor_cfgs(cfg_args, sim_utils, MultirotorCfg, ThrusterCfg))
        literal_usd = _verify_literal_city_lite_usd_spawn(stage)
        markers = _spawn_identity_markers(sim_utils)
        sensor_args = SimpleNamespace(
            dt=float(lock["simulation"]["dt_s"]),
            onboard_width=int(args.onboard_width),
            onboard_height=int(args.onboard_height),
            overview_width=int(args.overview_width),
            overview_height=int(args.overview_height),
        )
        onboard_enabled, overview_enabled = _camera_profile_flags(profile)
        onboard_tiled, overview_tiled = _camera_profile_tiled_flags(profile)
        onboard, overview, lidar, imu, contact, _onboard_cfg, _overview_cfg, lidar_cfg = _make_sensors(
            sensor_args,
            sim_utils,
            [],
            include_onboard_camera=onboard_enabled,
            include_overview_camera=overview_enabled,
            use_tiled_onboard_camera=onboard_tiled,
            use_tiled_overview_camera=overview_tiled,
        )
        _enforce_foreign_native_process_guard(
            args,
            base_receipt,
            phase="sensors_constructed",
        )
        sim.reset()
        _enforce_foreign_native_process_guard(
            args,
            base_receipt,
            phase="simulation_reset",
        )
        after_reset_commit = _sample_system_commit(
            resource_telemetry,
            system_commit,
            phase="after_reset",
            torch_module=torch,
        )
        _check_commit(
            threshold_percent=args.abort_commit_percent,
            phase="after_reset",
            system_commit=system_commit,
            snapshot=after_reset_commit,
        )
        runtime_observed = observe_live_simulation(lock, sim)
        runtime_issues = compare_live_simulation(lock, runtime_observed)
        if runtime_issues or (any((onboard_enabled, overview_enabled)) and runtime_observed["rtx_sensors_active"] is not True):
            raise IsaacSmokeError(
                "live renderer/device configuration does not match the runtime lock: "
                + "; ".join(
                    f"{issue.path}: {issue.message}" for issue in runtime_issues
                )
            )
        robot = EightCF2XFleet(members, prim_expression=SWARM_AGENT_PRIM_EXPRESSION, literal_prim_paths=SWARM_AGENT_LITERAL_PRIM_PATHS)
        initialized_sensors = [("lidar", lidar, AGENT_COUNT), ("imu", imu, AGENT_COUNT), ("contact", contact, AGENT_COUNT)]
        if overview_enabled:
            initialized_sensors[:0] = [("overview", overview, 1)]
        if onboard_enabled:
            initialized_sensors[:0] = [("onboard", onboard, AGENT_COUNT)]
        for name, sensor, expected_count in initialized_sensors:
            if sensor is None:
                raise IsaacSmokeError(f"{name} was not constructed for the declared resource profile")
            if not sensor.is_initialized or (hasattr(sensor, "num_instances") and int(sensor.num_instances) != expected_count):
                raise IsaacSmokeError(f"{name} did not initialize with {expected_count} instances")
            sensor.reset()
        robot.reset()
        robot.update(float(lock["simulation"]["dt_s"]))
        spawn = _verify_literal_city_lite_spawn(
            robot,
            _city_lite_initial_root_states(torch, robot.device),
            _city_lite_initial_thruster_rps(torch, robot.device),
            torch,
        )
        if profile != "full":
            receipt = {
                **base_receipt,
                "status": "passed",
                "finished_wall_time_ns": time.time_ns(),
                "step_trace": [],
                "search_target_prim_count": 0,
                "sensors": _sensor_profile_status(profile),
                "sensor_last_frame_sha256": {},
                "scene": {
                    "scene_id": "RIVERMARK_CITY_LITE_v1",
                    "contract_sha256": authority.contract_sha256,
                    "active_static_prim_count": scene_evidence["active_static_prim_count"],
                    "structural_aabb_count": len(structural_aabbs),
                    "collision_proxy_count": len(proxies),
                    "aabb_geometry_sha256": route_report.aabb_geometry_sha256,
                },
                "cf2x": {
                    "literal_prim_paths": list(SWARM_AGENT_LITERAL_PRIM_PATHS),
                    "identity_marker_count": len(markers),
                    "literal_usd_spawn": literal_usd,
                    "literal_physics_spawn": spawn,
                    "maximum_displacement_m": 0.0,
                },
                "resource_probe": _resource_probe_contract(profile),
                "runtime_observed": runtime_observed,
                "system_commit": dict(system_commit),
                "resource_telemetry": resource_telemetry.as_dict(),
            }
            errors = validate_smoke_receipt(receipt, runtime_lock=lock)
            if errors:
                raise IsaacSmokeError("invalid resource probe receipt: " + "; ".join(errors))
            _write_receipt(output_dir, receipt)
            return receipt
        routes = _waypoint_routes(torch.float32, robot.device, torch)
        initial_position = robot.data.root_pos_w.detach().clone()
        step_trace: list[dict[str, Any]] = []
        sensor_digests: dict[str, str] = {}
        sensor_timeline = _SensorUpdateTimeline()
        previous_position: Any | None = None
        # Preserve the reset-state mount before the first command.  The first
        # render gate can otherwise report a current-pose residual without a
        # reference capable of falsifying one-step Camera telemetry lag.
        previous_root_camera_expected_pos, _ = _expected_onboard_camera_world_poses(robot, torch)
        previous_root_camera_expected_phase: str | None = "post_reset_state_update"
        onboard_camera_render_read_fences: list[dict[str, list[int]]] = []
        for step in range(args.steps):
            target, _desired_pos, _desired_vel, _waypoint, _progress = _controller_target(
                robot, routes, HOVER_THRUST_PER_ROTOR_N, step * float(lock["simulation"]["dt_s"]), torch, math_utils
            )
            events: list[str] = []
            robot.set_thrust_target(target)
            robot.write_data_to_sim()
            events.append("command_write")
            sim.step(render=False)
            events.append("simulation_step")
            robot.update(float(lock["simulation"]["dt_s"]))
            events.append("state_update")
            effective_time_ns = physics_time_ns(
                step + 1, float(lock["simulation"]["dt_s"])
            )
            sensor_timeline.update(contact, time_ns=effective_time_ns)
            current_position = robot.data.root_pos_w.detach().cpu().numpy().copy()
            current_contact = contact.data.net_forces_w.detach().cpu().numpy().copy()
            evaluate_runtime_safety(
                previous_positions_w_m=previous_position,
                current_positions_w_m=current_position,
                net_contact_forces_w_n=current_contact,
                structural_aabbs=structural_aabbs,
                phase="rollout",
                physics_step=step + 1,
            )
            previous_position = current_position
            # This mandatory per-physics-step contact read is a safety guard,
            # not the synchronized retained sensor sample below.
            events.append("safety_contact_read")
            expected_pos, expected_quat, usd_closure = _prepare_onboard_camera_local_mount(
                sim, stage, robot, torch
            )
            events.append("camera_pose_update")
            _set_public_route_witness_overview_view(
                stage, overview, effective_time_ns=effective_time_ns
            )
            _enforce_foreign_native_process_guard(
                args,
                base_receipt,
                phase=f"before_render_step_{step + 1}",
            )
            onboard_frame_before_render = _onboard_camera_frame_counter(onboard, torch)
            sim.render()
            events.append("render")
            sensor_timeline.update(onboard, time_ns=effective_time_ns)
            onboard_render_read_fence = _require_onboard_camera_render_read_fence(
                onboard, onboard_frame_before_render, torch
            )
            onboard_camera_render_read_fences.append(
                {
                    "pre_frame_index": _to_numpy(
                        onboard_render_read_fence["pre_frame_index"]
                    ).astype(int).tolist(),
                    "post_frame_index": _to_numpy(
                        onboard_render_read_fence["post_frame_index"]
                    ).astype(int).tolist(),
                }
            )
            sensor_timeline.update(overview, time_ns=effective_time_ns)
            sensor_timeline.update(lidar, time_ns=effective_time_ns)
            sensor_timeline.update(imu, time_ns=effective_time_ns)
            events.append("rgbd_lidar_imu_read")
            sensor_timeline.update(contact, time_ns=effective_time_ns)
            fabric_closure = _camera_pose_closure(robot, onboard, torch)
            base_receipt["onboard_camera_mount_diagnostic"] = _onboard_camera_mount_diagnostics(
                robot,
                onboard,
                root_expected_pos_w=expected_pos,
                root_expected_quat_wxyz=expected_quat,
                previous_root_expected_pos_w=previous_root_camera_expected_pos,
                previous_root_expected_phase=previous_root_camera_expected_phase,
                torch=torch,
            )
            previous_root_camera_expected_pos = expected_pos.detach().clone()
            previous_root_camera_expected_phase = f"rollout_step_{step + 1}"
            usd_closure = _onboard_camera_usd_pose_closure(stage, expected_pos, expected_quat, torch)
            _require_onboard_camera_usd_pose(usd_closure)
            render_closure = _camera_pose_closure_from_usd(
                usd_closure, expected_pos, expected_quat, torch
            )
            base_receipt["onboard_camera_render_pose"] = {
                "authority": render_closure["authority"],
                "max_position_error_m": float(
                    torch.max(render_closure["position_error_m"]).item()
                ),
                "max_orientation_error_rad": float(
                    torch.max(render_closure["orientation_error_rad"]).item()
                ),
            }
            base_receipt["onboard_camera_fabric_diagnostic"] = (
                _onboard_camera_fabric_pose_diagnostic(fabric_closure, torch)
            )
            hits = lidar.data.ray_hits_w
            lidar_ranges = torch.linalg.vector_norm(hits - lidar.data.pos_w.unsqueeze(1), dim=-1)
            lidar_ranges = torch.nan_to_num(lidar_ranges, nan=lidar_cfg.max_distance, posinf=lidar_cfg.max_distance).clamp(0.0, lidar_cfg.max_distance)
            _require_onboard_visual_integrity(_onboard_visual_intrusion_evidence(onboard.data.output["distance_to_image_plane"], lidar_ranges, lidar_max_distance_m=float(lidar_cfg.max_distance)))
            _require_onboard_scene_content(_onboard_scene_content_evidence(onboard.data.output["distance_to_image_plane"], onboard.data.output["semantic_segmentation"], _onboard_semantic_metadata(onboard), far_clip_m=ONBOARD_CAMERA_CLIPPING_RANGE_M[1]))
            _require_overview_city_content(_overview_city_content_evidence(overview.data.output["rgb"], overview.data.output["distance_to_image_plane"], overview.data.output["semantic_segmentation"], _overview_semantic_metadata(overview), far_clip_m=OVERVIEW_CAMERA_CLIPPING_RANGE_M[1]))
            _require_overview_tracked_agent_visibility(_overview_tracked_agent_visibility_evidence(overview.data.output["semantic_segmentation"], _overview_semantic_metadata(overview)))
            if not bool(torch.isfinite(imu.data.lin_acc_b).all() and torch.isfinite(imu.data.ang_vel_b).all() and torch.isfinite(contact.data.net_forces_w).all()):
                raise IsaacSmokeError("IMU/contact smoke sample is non-finite")
            sensor_digests = {
                "rgb": _array_digest(onboard.data.output["rgb"]),
                "depth": _array_digest(onboard.data.output["distance_to_image_plane"]),
                "semantic": _array_digest(onboard.data.output["semantic_segmentation"]),
                "lidar": _array_digest(lidar_ranges),
                "imu": _array_digest(imu.data.lin_acc_b),
                "contact": _array_digest(contact.data.net_forces_w),
            }
            events.append("retained_contact_read")
            events.append("storage")
            step_trace.append({"step": step + 1, "events": events})
            step_phase = f"step-{step + 1}"
            step_commit = _sample_system_commit(
                resource_telemetry,
                system_commit,
                phase=step_phase,
                torch_module=torch,
            )
            _check_commit(
                threshold_percent=args.abort_commit_percent,
                phase=step_phase,
                system_commit=system_commit,
                snapshot=step_commit,
            )

        displacement = torch.linalg.vector_norm(robot.data.root_pos_w - initial_position, dim=-1)
        receipt = {
            **base_receipt,
            "status": "passed",
            "finished_wall_time_ns": time.time_ns(),
            "step_trace": step_trace,
            "search_target_prim_count": 0,
            "sensors": {key: True for key in sensor_digests},
            "sensor_last_frame_sha256": sensor_digests,
            "onboard_camera_render_read_fences": onboard_camera_render_read_fences,
            "scene": {
                "scene_id": "RIVERMARK_CITY_LITE_v1",
                "contract_sha256": authority.contract_sha256,
                "active_static_prim_count": scene_evidence["active_static_prim_count"],
                "structural_aabb_count": len(structural_aabbs),
                "collision_proxy_count": len(proxies),
                "aabb_geometry_sha256": route_report.aabb_geometry_sha256,
            },
            "cf2x": {
                "literal_prim_paths": list(SWARM_AGENT_LITERAL_PRIM_PATHS),
                "identity_marker_count": len(markers),
                "literal_usd_spawn": literal_usd,
                "literal_physics_spawn": spawn,
                "maximum_displacement_m": float(torch.max(displacement).item()),
            },
            "runtime_observed": runtime_observed,
            "system_commit": dict(system_commit),
            "resource_telemetry": resource_telemetry.as_dict(),
        }
        errors = validate_smoke_receipt(receipt, runtime_lock=lock)
        if errors:
            raise IsaacSmokeError("invalid smoke receipt: " + "; ".join(errors))
        _write_receipt(output_dir, receipt)
        return receipt
    except BaseException as exc:
        _sample_system_commit(
            resource_telemetry,
            system_commit,
            phase="failed",
            torch_module=locals().get("torch"),
        )
        failed = {
            **base_receipt,
            "status": "failed",
            "finished_wall_time_ns": time.time_ns(),
            "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "resource_telemetry": resource_telemetry.as_dict(),
        }
        _write_receipt(output_dir, failed)
        raise
    finally:
        _close_smoke_resources(app, lease)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--isaaclab-source", type=Path, required=True)
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--drone-usd", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--onboard-width", type=int, default=160)
    parser.add_argument("--onboard-height", type=int, default=120)
    # The full smoke shares the production fixed-world witness contract. A
    # lower resolution makes its 0.20 m identity marker physically unable to
    # satisfy the unchanged 32-pixel visibility gate at the far route extent.
    parser.add_argument("--overview-width", type=int, default=1920)
    parser.add_argument("--overview-height", type=int, default=1080)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument(
        "--preflight-commit-percent", type=float, default=DEFAULT_PREFLIGHT_COMMIT_PERCENT
    )
    parser.add_argument(
        "--abort-commit-percent", type=float, default=DEFAULT_ABORT_COMMIT_PERCENT
    )
    parser.add_argument(
        "--maximum-foreign-native-private-commit-gib",
        type=float,
        default=8.0,
        help=(
            "Fail before or during the smoke when another Python/Kit/Isaac process "
            "retains at least this much private commit."
        ),
    )
    parser.add_argument(
        "--resource-probe-profile",
        choices=sorted(SMOKE_RESOURCE_PROFILES),
        default="full",
        help=(
            "full is the normal smoke; no_cameras, onboard_only, overview_only, onboard_tiled_only, and "
            "overview_tiled_only "
            "are reset-only resource diagnoses, never capture evidence"
        ),
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = run_target_free_smoke(args)
    except Exception as exc:
        print(json.dumps({"schema": ISAAC_SMOKE_SCHEMA, "status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
