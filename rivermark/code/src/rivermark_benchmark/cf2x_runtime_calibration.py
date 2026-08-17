"""Fail-closed native Isaac calibration probe for the Rivermark CF2X asset.

The capture configuration contains a CF2X allocation and motor-response
assumption.  This module turns that assumption into an auditable, native Isaac
observation before any T2 policy is allowed to command a vehicle.  It retains
no City-Lite payload, sensor frame, target, or formal episode data.

Isaac imports intentionally occur only after all resource, source, runtime-lock,
and exclusive-AppLauncher gates have passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np

from .capture_lease import repository_app_launcher_lease
from .provenance import detect_source_provenance
from .resource_telemetry import (
    DEFAULT_ABORT_COMMIT_PERCENT,
    DEFAULT_PREFLIGHT_COMMIT_PERCENT,
    ResourceTelemetry,
    foreign_native_process_census,
)
from .runtime_lock import (
    audit_runtime_lock,
    configure_simulation_cfg,
    load_runtime_lock,
    locked_launcher_kwargs,
    runtime_lock_sha256,
    validate_locked_launcher_environment,
)

CF2X_RUNTIME_CALIBRATION_SCHEMA = "org.rivermark.cf2x-runtime-calibration.v1"
CF2X_RUNTIME_CALIBRATION_PRELAUNCH_FAILURE_SCHEMA = (
    "org.rivermark.cf2x-runtime-calibration-prelaunch-failure.v1"
)
_SHA256_HEX = frozenset("0123456789abcdef")
_EXPECTED_ROTOR_COUNT = 4
_EXPECTED_ALLOCATION_ROWS = 6
_PROBE_STEP_ORDER = (
    "set_thrust_target",
    "write_data_to_sim",
    "simulation_step",
    "robot_update",
)


class CF2XRuntimeCalibrationError(RuntimeError):
    """Raised when a calibration cannot prove a native CF2X fact."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_matrix(value: Any, *, rows: int, columns: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == rows
        and all(
            isinstance(row, list)
            and len(row) == columns
            and all(_finite_number(entry) for entry in row)
            for row in value
        )
    )


def _finite_vector(value: Any, *, length: int, positive: bool = False) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(
            _finite_number(entry) and (not positive or float(entry) > 0.0)
            for entry in value
        )
    )


def _finite_range(value: Any, *, allow_zero_lower_bound: bool = False) -> bool:
    """Validate an ordered actuator range without treating zero thrust as invalid."""

    if not isinstance(value, list) or len(value) != 2 or not all(_finite_number(item) for item in value):
        return False
    lower, upper = (float(item) for item in value)
    if allow_zero_lower_bound:
        return lower >= 0.0 and upper > 0.0 and lower <= upper
    return lower > 0.0 and upper > 0.0 and lower <= upper


def _valid_body_physics(row: Any, *, runtime: bool) -> bool:
    if not isinstance(row, Mapping):
        return False
    if not isinstance(row.get("body_name"), str) or not row["body_name"]:
        return False
    if not _finite_number(row.get("mass_kg")) or float(row["mass_kg"]) <= 0.0:
        return False
    if not _finite_vector(
        row.get("diagonal_inertia_kg_m2"), length=3, positive=True
    ):
        return False
    return not runtime or _finite_matrix(
        row.get("inertia_matrix_kg_m2"), rows=3, columns=3
    )


def calibration_report_sha256(report: Mapping[str, Any]) -> str:
    """Hash the report while excluding its self-reference field."""

    canonical = dict(report)
    canonical.pop("report_sha256", None)
    return _sha256_bytes(_canonical_bytes(canonical))


def prelaunch_failure_report_sha256(report: Mapping[str, Any]) -> str:
    """Hash a minimal pre-AppLauncher failure receipt without self-reference."""

    canonical = dict(report)
    canonical.pop("report_sha256", None)
    return _sha256_bytes(_canonical_bytes(canonical))


def _locked_contrib_source_root(isaaclab_source_checkout: Path, lock: Mapping[str, Any]) -> Path:
    """Resolve the lock-bound contrib tree beside the audited IsaacLab checkout."""

    source = lock.get("isaaclab_contrib_source")
    relative = source.get("relative_path") if isinstance(source, Mapping) else None
    if not isinstance(relative, str) or not relative:
        raise CF2XRuntimeCalibrationError("runtime lock has no IsaacLab contrib relative path")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
        raise CF2XRuntimeCalibrationError("runtime lock has an unsafe IsaacLab contrib relative path")
    return Path(isaaclab_source_checkout).expanduser().resolve().parent.joinpath(*parsed.parts)


def _as_json_tensor(value: Any) -> list[Any]:
    """Materialize a Torch-like runtime tensor only after the caller owns Isaac."""

    if value is None:
        raise CF2XRuntimeCalibrationError("runtime tensor query returned no tensor")
    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise CF2XRuntimeCalibrationError("runtime query did not return a tensor")
    materialized = detach()
    cpu = getattr(materialized, "cpu", None)
    if not callable(cpu):
        raise CF2XRuntimeCalibrationError("runtime tensor cannot be materialized on CPU")
    json_value = cpu()
    tolist = getattr(json_value, "tolist", None)
    if not callable(tolist):
        raise CF2XRuntimeCalibrationError("runtime tensor has no list conversion")
    return tolist()


def _as_float_vector(value: Any) -> list[float]:
    """Convert a scalar sequence or CUDA/CPU Torch tensor to finite floats."""

    if callable(getattr(value, "detach", None)):
        value = _as_json_tensor(value)
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise CF2XRuntimeCalibrationError("runtime value contains a non-finite component")
    return [float(item) for item in values.tolist()]


def _read_static_usd_physics(usd_path: Path) -> dict[str, Any]:
    """Read authored MassAPI values from the exact USD whose bytes are hashed."""

    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise CF2XRuntimeCalibrationError("unable to open CF2X USD for static physics audit")
    bodies: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        mass_attribute = prim.GetAttribute("physics:mass")
        inertia_attribute = prim.GetAttribute("physics:diagonalInertia")
        if not mass_attribute.IsValid() and not inertia_attribute.IsValid():
            continue
        mass = mass_attribute.Get() if mass_attribute.IsValid() else None
        inertia = inertia_attribute.Get() if inertia_attribute.IsValid() else None
        inertia_values = (
            [float(inertia[index]) for index in range(3)] if inertia is not None else None
        )
        body = {
            "prim_path": str(prim.GetPath()),
            "body_name": prim.GetName(),
            "mass_kg": float(mass) if mass is not None else None,
            "diagonal_inertia_kg_m2": inertia_values,
        }
        if body["mass_kg"] is not None and not _finite_number(body["mass_kg"]):
            raise CF2XRuntimeCalibrationError("CF2X USD has a non-finite mass")
        if inertia_values is not None and not all(_finite_number(item) for item in inertia_values):
            raise CF2XRuntimeCalibrationError("CF2X USD has a non-finite inertia")
        bodies.append(body)
    if not bodies:
        raise CF2XRuntimeCalibrationError("CF2X USD exposes no authored MassAPI values")
    default_prim = stage.GetDefaultPrim()
    return {
        "usd_sha256": _sha256_file(usd_path),
        "default_prim": str(default_prim.GetPath())
        if default_prim is not None and default_prim.IsValid()
        else None,
        "bodies": sorted(bodies, key=lambda row: str(row["prim_path"])),
    }


def _runtime_body_physics(robot: Any) -> list[dict[str, Any]]:
    """Read body properties from the live PhysX articulation view.

    ``MultirotorData`` intentionally does not populate Articulation's cached
    ``default_mass`` and ``default_inertia`` fields.  The PhysX view is the
    authoritative runtime source and is also the source IsaacLab's
    ``Articulation`` initialization uses for those fields.
    """

    body_names = list(getattr(robot, "body_names", ()))
    if not body_names:
        raise CF2XRuntimeCalibrationError("runtime CF2X exposes no body names")
    physics_view = getattr(robot, "root_physx_view", None)
    if physics_view is None:
        raise CF2XRuntimeCalibrationError("runtime CF2X has no PhysX articulation view")
    get_masses = getattr(physics_view, "get_masses", None)
    get_inertias = getattr(physics_view, "get_inertias", None)
    if not callable(get_masses) or not callable(get_inertias):
        raise CF2XRuntimeCalibrationError("runtime PhysX articulation view cannot read mass and inertia")
    try:
        mass = np.asarray(_as_json_tensor(get_masses()), dtype=np.float64)
        inertia = np.asarray(_as_json_tensor(get_inertias()), dtype=np.float64)
    except CF2XRuntimeCalibrationError:
        raise
    except Exception as exc:
        raise CF2XRuntimeCalibrationError("runtime PhysX mass/inertia query failed") from exc
    if mass.ndim != 2 or mass.shape[0] != 1 or mass.shape[1] != len(body_names):
        raise CF2XRuntimeCalibrationError("runtime PhysX mass shape disagrees with CF2X body names")
    if inertia.ndim != 3 or inertia.shape != (1, len(body_names), 9):
        raise CF2XRuntimeCalibrationError("runtime PhysX inertia shape is not [1, body, 9]")
    if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(inertia)):
        raise CF2XRuntimeCalibrationError("runtime CF2X physics values are non-finite")
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(body_names):
        rows.append(
            {
                "body_name": str(name),
                "mass_kg": float(mass[0, index]),
                "inertia_matrix_kg_m2": [
                    [float(value) for value in inertia[0, index, row * 3 : row * 3 + 3]]
                    for row in range(3)
                ],
                "diagonal_inertia_kg_m2": [
                    float(inertia[0, index, diagonal * 3 + diagonal])
                    for diagonal in range(3)
                ],
            }
        )
    return rows


def _force_axis_from_allocation(allocation: list[list[float]]) -> dict[str, Any]:
    columns = len(allocation[0])
    axes: list[list[float]] = []
    for column in range(columns):
        force = np.asarray([allocation[row][column] for row in range(3)], dtype=np.float64)
        norm = float(np.linalg.norm(force))
        if not math.isfinite(norm) or norm <= 0.0:
            raise CF2XRuntimeCalibrationError("allocation matrix has a zero or invalid force axis")
        axes.append([float(component / norm) for component in force])
    is_positive_body_z = all(
        math.isclose(axis[0], 0.0, abs_tol=1.0e-7)
        and math.isclose(axis[1], 0.0, abs_tol=1.0e-7)
        and axis[2] > 0.999999
        for axis in axes
    )
    return {"per_rotor_unit_vector_body": axes, "all_positive_body_z": is_positive_body_z}


def _actuator_runtime_summary(robot: Any, *, control_dt_s: float) -> dict[str, Any]:
    actuators = getattr(robot, "actuators", None)
    if not isinstance(actuators, Mapping) or set(actuators) != {"thrusters"}:
        raise CF2XRuntimeCalibrationError("CF2X runtime must expose exactly one thrusters actuator")
    actuator = actuators["thrusters"]
    cfg = actuator.cfg
    summary = {
        "control_dt_s": float(control_dt_s),
        "actuator_dt_s": float(cfg.dt),
        "thrust_range_n": [float(value) for value in cfg.thrust_range],
        "max_thrust_rate_n_per_s": float(cfg.max_thrust_rate),
        "thrust_constant_range_n_per_rps_squared": [
            float(value) for value in cfg.thrust_const_range
        ],
        "tau_increase_range_s": [float(value) for value in cfg.tau_inc_range],
        "tau_decrease_range_s": [float(value) for value in cfg.tau_dec_range],
        "torque_to_thrust_ratio_nm_per_n": float(cfg.torque_to_thrust_ratio),
        "use_discrete_approximation": bool(cfg.use_discrete_approximation),
        "integration_scheme": str(cfg.integration_scheme),
        "sampled_tau_increase_s": _as_float_vector(actuator.tau_inc_s),
        "sampled_tau_decrease_s": _as_float_vector(actuator.tau_dec_s),
        "sampled_thrust_constant_n_per_rps_squared": _as_float_vector(actuator.thrust_const),
    }
    if not math.isclose(summary["control_dt_s"], summary["actuator_dt_s"], abs_tol=1.0e-12):
        raise CF2XRuntimeCalibrationError("actuator dt and simulation control dt disagree")
    return summary


def _cross_check_static_and_runtime(
    static_usd: Mapping[str, Any], runtime_bodies: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    static_rows = static_usd.get("bodies")
    if not isinstance(static_rows, list):
        raise CF2XRuntimeCalibrationError("static USD body list is malformed")
    static_by_name = {
        str(row.get("body_name")): row
        for row in static_rows
        if isinstance(row, Mapping) and isinstance(row.get("body_name"), str)
    }
    rows: list[dict[str, Any]] = []
    for runtime in runtime_bodies:
        name = runtime["body_name"]
        static = static_by_name.get(name)
        if static is None:
            rows.append({"body_name": name, "status": "missing_static_body"})
            continue
        static_mass = static.get("mass_kg")
        static_inertia = static.get("diagonal_inertia_kg_m2")
        runtime_mass = runtime.get("mass_kg")
        runtime_inertia = runtime.get("diagonal_inertia_kg_m2")
        mass_error = (
            abs(float(static_mass) - float(runtime_mass))
            if _finite_number(static_mass) and _finite_number(runtime_mass)
            else None
        )
        inertia_error = (
            max(abs(float(left) - float(right)) for left, right in zip(static_inertia, runtime_inertia, strict=True))
            if isinstance(static_inertia, list)
            and isinstance(runtime_inertia, list)
            and len(static_inertia) == 3
            and len(runtime_inertia) == 3
            and all(_finite_number(value) for value in (*static_inertia, *runtime_inertia))
            else None
        )
        status = (
            "matched"
            if mass_error is not None
            and inertia_error is not None
            and mass_error <= 1.0e-7
            and inertia_error <= 1.0e-9
            else "mismatch"
        )
        rows.append(
            {
                "body_name": name,
                "status": status,
                "mass_absolute_error_kg": mass_error,
                "diagonal_inertia_max_absolute_error_kg_m2": inertia_error,
            }
        )
    return {
        "status": "passed" if rows and all(row["status"] == "matched" for row in rows) else "failed",
        "tolerances": {"mass_kg": 1.0e-7, "diagonal_inertia_kg_m2": 1.0e-9},
        "bodies": rows,
    }


def validate_calibration_report(report: Any) -> tuple[str, ...]:
    """Validate a calibration report without Isaac, Torch, or PXR imports."""

    if not isinstance(report, Mapping):
        return ("report must be an object",)
    issues: list[str] = []
    if report.get("schema") != CF2X_RUNTIME_CALIBRATION_SCHEMA:
        issues.append("schema mismatch")
    if report.get("status") not in {"passed", "failed"}:
        issues.append("invalid status")
    claim_boundary = report.get("claim_boundary")
    if claim_boundary != {
        "formal_episode": False,
        "city_lite_capture": False,
        "benchmark_score": False,
        "sensor_payload_retained": False,
    }:
        issues.append("calibration claim boundary is invalid")
    source = report.get("source")
    if (
        not isinstance(source, Mapping)
        or not isinstance(source.get("source_revision"), str)
        or len(source["source_revision"]) not in {40, 64}
        or not set(source["source_revision"]) <= _SHA256_HEX
        or not _is_sha256(source.get("source_tree_sha256"))
        or source.get("source_worktree_dirty") is not False
    ):
        issues.append("calibration source provenance is not clean and hash-bound")
    if not _is_sha256(report.get("runtime_lock_sha256")):
        issues.append("runtime lock hash is missing")
    runtime_audit = report.get("runtime_audit")
    if not isinstance(runtime_audit, Mapping) or runtime_audit.get("status") != "passed":
        issues.append("runtime lock audit did not pass")
    asset = report.get("asset")
    if not isinstance(asset, Mapping) or not _is_sha256(asset.get("usd_sha256")):
        issues.append("asset hash is missing")
    static_usd = report.get("static_usd")
    if (
        not isinstance(static_usd, Mapping)
        or static_usd.get("usd_sha256") != (asset.get("usd_sha256") if isinstance(asset, Mapping) else None)
        or not isinstance(static_usd.get("bodies"), list)
        or not static_usd["bodies"]
        or not all(_valid_body_physics(row, runtime=False) for row in static_usd["bodies"])
    ):
        issues.append("static USD physics evidence is missing or unbound")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping):
        issues.append("runtime section is missing")
    else:
        names = runtime.get("thruster_names")
        if not isinstance(names, list) or len(names) != _EXPECTED_ROTOR_COUNT or len(set(names)) != _EXPECTED_ROTOR_COUNT or not all(isinstance(name, str) and name for name in names):
            issues.append("runtime rotor order is invalid")
        allocation = runtime.get("allocation_matrix")
        if not _finite_matrix(allocation, rows=_EXPECTED_ALLOCATION_ROWS, columns=_EXPECTED_ROTOR_COUNT):
            issues.append("runtime allocation matrix is invalid")
        directions = runtime.get("rotor_directions")
        if not isinstance(directions, list) or len(directions) != _EXPECTED_ROTOR_COUNT or any(value not in {-1, 1} for value in directions):
            issues.append("runtime rotor directions are invalid")
        axis = runtime.get("thrust_axis")
        if not isinstance(axis, Mapping) or axis.get("all_positive_body_z") is not True:
            issues.append("runtime thrust axis is not positive body z")
        actuator = runtime.get("actuator")
        if not isinstance(actuator, Mapping) or not all(_finite_number(actuator.get(key)) and float(actuator[key]) > 0.0 for key in ("control_dt_s", "actuator_dt_s", "max_thrust_rate_n_per_s")):
            issues.append("runtime actuator timing is invalid")
        elif not math.isclose(float(actuator["control_dt_s"]), float(actuator["actuator_dt_s"]), abs_tol=1.0e-12):
            issues.append("runtime actuator dt differs from control dt")
        elif not all(
            _finite_vector(actuator.get(key), length=_EXPECTED_ROTOR_COUNT, positive=True)
            for key in (
                "sampled_tau_increase_s",
                "sampled_tau_decrease_s",
                "sampled_thrust_constant_n_per_rps_squared",
            )
        ) or not _finite_range(actuator.get("thrust_range_n"), allow_zero_lower_bound=True) or not all(
            _finite_range(actuator.get(key))
            for key in (
                "thrust_constant_range_n_per_rps_squared",
                "tau_increase_range_s",
                "tau_decrease_range_s",
            )
        ) or not _finite_number(actuator.get("torque_to_thrust_ratio_nm_per_n")):
            issues.append("runtime actuator response evidence is invalid")
        bodies = runtime.get("bodies")
        if not isinstance(bodies, list) or not bodies or not all(
            _valid_body_physics(row, runtime=True) for row in bodies
        ):
            issues.append("runtime mass/inertia values are missing")
    cross_check = report.get("static_runtime_cross_check")
    runtime_body_count = len(runtime.get("bodies", [])) if isinstance(runtime, Mapping) else 0
    cross_check_bodies = cross_check.get("bodies") if isinstance(cross_check, Mapping) else None
    if (
        not isinstance(cross_check, Mapping)
        or cross_check.get("status") != "passed"
        or not isinstance(cross_check_bodies, list)
        or len(cross_check_bodies) != runtime_body_count
        or any(
            not isinstance(row, Mapping) or row.get("status") != "matched"
            for row in cross_check_bodies
        )
    ):
        issues.append("static/runtime cross-check did not pass")
    probe = report.get("actuation_probe")
    if not isinstance(probe, Mapping):
        issues.append("actuation probe is missing")
    else:
        if probe.get("step_order") != list(_PROBE_STEP_ORDER):
            issues.append("actuation probe step order is invalid")
        if probe.get("command_before_step") is not True:
            issues.append("actuation probe does not prove command-before-step")
        requested = probe.get("requested_thrust_n")
        target_after_set = probe.get("target_thrust_after_set_n")
        applied = probe.get("applied_thrust_after_write_n")
        wrench = probe.get("applied_wrench_after_write_body")
        if not isinstance(requested, list) or len(requested) != _EXPECTED_ROTOR_COUNT or not all(_finite_number(value) and float(value) >= 0.0 for value in requested):
            issues.append("probe requested thrust is invalid")
        if not isinstance(applied, list) or len(applied) != _EXPECTED_ROTOR_COUNT or not all(_finite_number(value) and float(value) >= 0.0 for value in applied):
            issues.append("probe applied thrust is invalid")
        elif not any(float(value) > 0.0 for value in applied):
            issues.append("probe did not apply positive thrust")
        if not isinstance(target_after_set, list) or len(target_after_set) != _EXPECTED_ROTOR_COUNT or not all(_finite_number(value) and float(value) >= 0.0 for value in target_after_set):
            issues.append("probe post-set thrust target is invalid")
        if not isinstance(wrench, list) or len(wrench) != _EXPECTED_ALLOCATION_ROWS or not all(_finite_number(value) for value in wrench):
            issues.append("probe applied wrench is invalid")
        elif float(wrench[2]) <= 0.0:
            issues.append("probe did not apply positive body-z force")
        for field in ("initial_root_position_w_m", "final_root_position_w_m"):
            position = probe.get(field)
            if not isinstance(position, list) or len(position) != 3 or not all(
                _finite_number(value) for value in position
            ):
                issues.append(f"probe {field} is invalid")
        samples = probe.get("samples")
        if not isinstance(samples, list) or not samples or not all(
            isinstance(sample, Mapping)
            and isinstance(sample.get("physics_step"), int)
            and not isinstance(sample.get("physics_step"), bool)
            and sample["physics_step"] > 0
            and _finite_vector(sample.get("root_position_w_m"), length=3)
            and _finite_vector(sample.get("root_linear_velocity_w_mps"), length=3)
            and _finite_vector(
                sample.get("applied_thrust_n"), length=_EXPECTED_ROTOR_COUNT
            )
            for sample in samples
        ):
            issues.append("probe has no post-step state samples")
    if report.get("report_sha256") != calibration_report_sha256(report):
        issues.append("report self-hash does not match")
    return tuple(issues)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _system_commit_percent() -> float | None:
    from .resource_telemetry import _system_commit_snapshot

    snapshot = _system_commit_snapshot()
    value = snapshot.get("commit_percent") if isinstance(snapshot, Mapping) else None
    return float(value) if _finite_number(value) else None


def _require_commit_below(
    telemetry_sample: Mapping[str, Any], *, threshold_percent: float, phase: str
) -> None:
    """Stop before a short probe can amplify a host commit-pressure event."""

    system_commit = telemetry_sample.get("system_commit")
    value = system_commit.get("commit_percent") if isinstance(system_commit, Mapping) else None
    if _finite_number(value) and float(value) >= threshold_percent:
        raise CF2XRuntimeCalibrationError(
            f"Windows system commit is {float(value):.2f}% at {phase}; "
            f"abort limit is {threshold_percent:.2f}%"
        )


def _enforce_native_resource_preflight(args: argparse.Namespace) -> None:
    if args.preflight_commit_percent <= 0.0 or args.abort_commit_percent <= args.preflight_commit_percent or args.abort_commit_percent > 95.0:
        raise CF2XRuntimeCalibrationError("commit thresholds must satisfy 0 < preflight < abort <= 95")
    if args.steps < 1 or args.steps > 64:
        raise CF2XRuntimeCalibrationError("--steps must be in [1, 64]")
    if not 0.0 < args.probe_thrust_n <= 0.18:
        raise CF2XRuntimeCalibrationError("--probe-thrust-n must be in (0, 0.18]")
    if args.maximum_foreign_native_private_commit_gib <= 0.0:
        raise CF2XRuntimeCalibrationError("foreign native private-commit limit must be positive")
    if not args.drone_usd.is_file():
        raise CF2XRuntimeCalibrationError("CF2X USD is missing")
    if not args.scene_contract.is_file():
        raise CF2XRuntimeCalibrationError("City-Lite contract is missing")
    if not args.runtime_lock.is_file():
        raise CF2XRuntimeCalibrationError("runtime lock is missing")
    if not args.isaaclab_source.is_dir():
        raise CF2XRuntimeCalibrationError("IsaacLab source is missing")
    if args.output_dir.exists():
        raise CF2XRuntimeCalibrationError("calibration output directory must not already exist")
    if not args.output_dir.parent.is_dir():
        raise CF2XRuntimeCalibrationError("calibration output parent is not available")
    if args.minimum_free_gib <= 0.0:
        raise CF2XRuntimeCalibrationError("minimum free GiB must be positive")
    import shutil

    if shutil.disk_usage(args.output_dir.parent).free < int(args.minimum_free_gib * 1024**3):
        raise CF2XRuntimeCalibrationError("calibration output volume lacks free-space budget")
    commit_percent = _system_commit_percent()
    if commit_percent is not None and commit_percent >= args.preflight_commit_percent:
        raise CF2XRuntimeCalibrationError(
            f"Windows system commit is {commit_percent:.2f}%; preflight limit is {args.preflight_commit_percent:.2f}%"
        )
    census = foreign_native_process_census(
        minimum_private_commit_bytes=int(args.maximum_foreign_native_private_commit_gib * 1024**3)
    )
    if isinstance(census, Mapping) and int(census.get("candidate_count", 0)) > 0:
        raise CF2XRuntimeCalibrationError(
            "refusing native CF2X calibration because another high-commit Python/Kit/Isaac process is active"
        )


def _make_native_report(args: argparse.Namespace) -> dict[str, Any]:
    """Run a one-CF2X empty-stage probe under the shared AppLauncher lease."""

    from .isaac_capture import (
        _activate_local_isaaclab_contrib_source,
        _activate_local_isaaclab_source,
        _make_multirotor_cfgs,
        _module_path_is_under,
    )

    output_dir = args.output_dir.resolve()
    source = detect_source_provenance(Path(__file__).resolve().parents[2])
    if source.source_worktree_dirty:
        raise CF2XRuntimeCalibrationError("native calibration requires a clean source worktree")
    lock = load_runtime_lock(args.runtime_lock)
    runtime_audit = audit_runtime_lock(
        args.runtime_lock,
        isaaclab_source=args.isaaclab_source,
        scene_contract=args.scene_contract,
        cf2x_usd=args.drone_usd,
    )
    if runtime_audit.get("status") != "passed":
        raise CF2XRuntimeCalibrationError("runtime lock audit failed")
    if bool(args.headless) != bool(lock["launcher"]["headless"]):
        raise CF2XRuntimeCalibrationError("--headless conflicts with the runtime lock")

    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": CF2X_RUNTIME_CALIBRATION_SCHEMA,
        "status": "failed",
        "created_wall_time_ns": time.time_ns(),
        "claim_boundary": {
            "formal_episode": False,
            "city_lite_capture": False,
            "benchmark_score": False,
            "sensor_payload_retained": False,
        },
        "asset": {"usd_sha256": _sha256_file(args.drone_usd.resolve())},
        "source": source.as_dict(),
        "runtime_lock_sha256": runtime_lock_sha256(lock),
        "runtime_audit": runtime_audit,
    }
    lease = repository_app_launcher_lease(
        Path(__file__).resolve().parents[2],
        metadata={
            "owner": "rivermark_benchmark.cf2x_runtime_calibration",
            "output_dir": str(output_dir),
            "source_revision": source.source_revision,
        },
    )
    app = None
    telemetry = ResourceTelemetry()
    try:
        _require_commit_below(
            telemetry.sample("pre_app_launcher"),
            threshold_percent=float(args.preflight_commit_percent),
            phase="pre_app_launcher",
        )
        isaaclab_source = _activate_local_isaaclab_source(args.isaaclab_source)
        if isaaclab_source is None:
            raise CF2XRuntimeCalibrationError("locked IsaacLab source did not activate")
        isaaclab_contrib_source = _activate_local_isaaclab_contrib_source(
            _locked_contrib_source_root(isaaclab_source, lock)
        )
        validate_locked_launcher_environment(lock)
        lease.acquire()
        from isaaclab.app import AppLauncher

        app = AppLauncher(locked_launcher_kwargs(lock, isaaclab_source)).app
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab_contrib.actuators import ThrusterCfg
        from isaaclab_contrib.assets import Multirotor, MultirotorCfg

        _require_commit_below(
            telemetry.sample("post_app_launcher", torch_module=torch),
            threshold_percent=float(args.abort_commit_percent),
            phase="post_app_launcher",
        )
        census = foreign_native_process_census(
            minimum_private_commit_bytes=int(args.maximum_foreign_native_private_commit_gib * 1024**3)
        )
        if isinstance(census, Mapping) and int(census.get("candidate_count", 0)) > 0:
            raise CF2XRuntimeCalibrationError(
                "another high-commit Python/Kit/Isaac process appeared after AppLauncher"
            )
        if not _module_path_is_under(__import__("isaaclab"), isaaclab_source / "isaaclab"):
            raise CF2XRuntimeCalibrationError("calibration imported isaaclab from an unbound source")
        if isaaclab_contrib_source is None or not _module_path_is_under(
            __import__("isaaclab_contrib"), isaaclab_contrib_source / "isaaclab_contrib"
        ):
            raise CF2XRuntimeCalibrationError("calibration imported isaaclab_contrib from an unbound source")

        torch.manual_seed(int(args.seed))
        import omni.usd

        omni.usd.get_context().new_stage()
        sim_cfg = sim_utils.SimulationCfg(
            dt=float(lock["simulation"]["dt_s"]), device=lock["simulation"]["device"]
        )
        configure_simulation_cfg(sim_cfg, lock)
        sim = sim_utils.SimulationContext(sim_cfg)
        cfg_args = SimpleNamespace(drone_usd=args.drone_usd.resolve(), dt=float(sim_cfg.dt))
        cfg = _make_multirotor_cfgs(cfg_args, sim_utils, MultirotorCfg, ThrusterCfg)[0]
        robot = Multirotor(cfg)
        sim.reset()
        robot.update(float(sim.cfg.dt))
        _require_commit_below(
            telemetry.sample("post_simulation_reset", torch_module=torch),
            threshold_percent=float(args.abort_commit_percent),
            phase="post_simulation_reset",
        )
        if not robot.is_initialized or int(robot.num_instances) != 1:
            raise CF2XRuntimeCalibrationError("single CF2X calibration did not initialize exactly one robot")
        if int(robot.num_thrusters) != _EXPECTED_ROTOR_COUNT:
            raise CF2XRuntimeCalibrationError("CF2X calibration did not resolve four rotor actuators")

        allocation = _as_json_tensor(robot.allocation_matrix)
        if not _finite_matrix(allocation, rows=_EXPECTED_ALLOCATION_ROWS, columns=_EXPECTED_ROTOR_COUNT):
            raise CF2XRuntimeCalibrationError("runtime allocation matrix is malformed")
        runtime_bodies = _runtime_body_physics(robot)
        static_usd = _read_static_usd_physics(args.drone_usd.resolve())
        cross_check = _cross_check_static_and_runtime(static_usd, runtime_bodies)
        if cross_check["status"] != "passed":
            raise CF2XRuntimeCalibrationError("CF2X USD static physics disagrees with live runtime physics")
        actuator = _actuator_runtime_summary(robot, control_dt_s=float(sim.cfg.dt))
        force_axis = _force_axis_from_allocation(allocation)
        if force_axis["all_positive_body_z"] is not True:
            raise CF2XRuntimeCalibrationError("CF2X allocation does not apply all thrust along positive body z")

        initial_position = _as_float_vector(robot.data.root_pos_w[0])
        command = torch.full(
            (1, _EXPECTED_ROTOR_COUNT), float(args.probe_thrust_n), device=robot.device
        )
        robot.set_thrust_target(command)
        target_after_set = _as_float_vector(robot.data.thrust_target[0])
        robot.write_data_to_sim()
        applied_after_write = _as_float_vector(robot.data.applied_thrust[0])
        wrench_after_write = _as_float_vector(robot._internal_wrench_target_sim[0])
        samples = []
        for step in range(int(args.steps)):
            sim.step(render=False)
            robot.update(float(sim.cfg.dt))
            _require_commit_below(
                telemetry.sample(f"probe_step_{step + 1}", torch_module=torch),
                threshold_percent=float(args.abort_commit_percent),
                phase=f"probe_step_{step + 1}",
            )
            samples.append(
                {
                    "physics_step": step + 1,
                    "root_position_w_m": _as_float_vector(robot.data.root_pos_w[0]),
                    "root_linear_velocity_w_mps": _as_float_vector(robot.data.root_lin_vel_w[0]),
                    "applied_thrust_n": _as_float_vector(robot.data.applied_thrust[0]),
                }
            )
            if step + 1 < int(args.steps):
                robot.set_thrust_target(command)
                robot.write_data_to_sim()
        final_position = _as_float_vector(robot.data.root_pos_w[0])
        report.update(
            {
                "status": "passed",
                "finished_wall_time_ns": time.time_ns(),
                "static_usd": static_usd,
                "runtime": {
                    "thruster_names": [str(name) for name in robot.thruster_names],
                    "allocation_matrix": allocation,
                    "rotor_directions": [int(value) for value in cfg.rotor_directions],
                    "thrust_axis": force_axis,
                    "bodies": runtime_bodies,
                    "actuator": actuator,
                },
                "static_runtime_cross_check": cross_check,
                "actuation_probe": {
                    "step_order": list(_PROBE_STEP_ORDER),
                    "command_before_step": True,
                    "requested_thrust_n": _as_float_vector(command[0]),
                    "target_thrust_after_set_n": target_after_set,
                    "applied_thrust_after_write_n": applied_after_write,
                    "applied_wrench_after_write_body": wrench_after_write,
                    "initial_root_position_w_m": initial_position,
                    "final_root_position_w_m": final_position,
                    "samples": samples,
                },
                "resource_telemetry": telemetry.as_dict(),
            }
        )
        report["report_sha256"] = calibration_report_sha256(report)
        issues = validate_calibration_report(report)
        if issues:
            raise CF2XRuntimeCalibrationError("invalid completed calibration: " + "; ".join(issues))
        return report
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "finished_wall_time_ns": time.time_ns(),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "resource_telemetry": telemetry.as_dict(),
            }
        )
        raise
    finally:
        report["report_sha256"] = calibration_report_sha256(report)
        _write_json_atomic(output_dir / "cf2x_runtime_calibration.json", report)
        try:
            if app is not None:
                app.close(wait_for_replicator=False, skip_cleanup=True)
        finally:
            lease.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument(
        "--isaaclab-source",
        type=Path,
        required=True,
        help="Lock-bound IsaacLab checkout containing isaaclab/__init__.py; its parent contains isaaclab_contrib.",
    )
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--drone-usd", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--probe-thrust-n", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument(
        "--preflight-commit-percent", type=float, default=DEFAULT_PREFLIGHT_COMMIT_PERCENT
    )
    parser.add_argument(
        "--abort-commit-percent", type=float, default=DEFAULT_ABORT_COMMIT_PERCENT
    )
    parser.add_argument(
        "--maximum-foreign-native-private-commit-gib", type=float, default=8.0
    )
    parser.add_argument("--headless", action="store_true")
    return parser


def _persist_prelaunch_failure(args: argparse.Namespace, error: BaseException) -> Path | None:
    """Preserve a new startup failure without overwriting another run's evidence."""

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() or not output_dir.parent.is_dir():
        return None
    report = {
        "schema": CF2X_RUNTIME_CALIBRATION_PRELAUNCH_FAILURE_SCHEMA,
        "status": "failed",
        "stage": "pre_app_launcher",
        "created_wall_time_ns": time.time_ns(),
        "finished_wall_time_ns": time.time_ns(),
        "claim_boundary": {
            "formal_episode": False,
            "city_lite_capture": False,
            "benchmark_score": False,
            "sensor_payload_retained": False,
            "app_launcher_started": False,
        },
        "failure": {"type": type(error).__name__, "message": str(error)},
    }
    report["report_sha256"] = prelaunch_failure_report_sha256(report)
    try:
        output_dir.mkdir(parents=False, exist_ok=False)
        path = output_dir / "cf2x_runtime_calibration.prelaunch_failure.json"
        _write_json_atomic(path, report)
    except OSError:
        return None
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.runtime_lock = args.runtime_lock.expanduser().resolve()
    args.isaaclab_source = args.isaaclab_source.expanduser().resolve()
    args.scene_contract = args.scene_contract.expanduser().resolve()
    args.drone_usd = args.drone_usd.expanduser().resolve()
    try:
        _enforce_native_resource_preflight(args)
        report = _make_native_report(args)
    except (
        CF2XRuntimeCalibrationError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        failure_path = _persist_prelaunch_failure(args, exc)
        print(
            json.dumps(
                {
                    "schema": CF2X_RUNTIME_CALIBRATION_SCHEMA,
                    "status": "failed",
                    "error": str(exc),
                    "prelaunch_failure_artifact": str(failure_path) if failure_path else None,
                },
                indent=2,
            )
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
