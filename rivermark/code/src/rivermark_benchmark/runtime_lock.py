"""Exact, Isaac-free runtime lock verification for native City-Lite work.

The v2 lock deliberately separates facts which can be observed before Kit is
started (package/source/asset provenance) from settings which must be checked
again after ``AppLauncher`` and ``SimulationContext`` exist.  This keeps the
preflight useful without pretending that a JSON file alone proves the live
renderer or PhysX configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

from .preflight import _probe_nvidia_smi, _version_at_least, sha256_file


RUNTIME_LOCK_SCHEMA = "org.rivermark.benchmark.isaac-runtime-lock.v2"
RUNTIME_AUDIT_SCHEMA = "org.rivermark.benchmark.isaac-runtime-audit.v2"
RUNTIME_AUDIT_OBSERVATION = "public_runtime_environment_and_locked_assets"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SUFFIXES = frozenset({".py", ".toml", ".json", ".yaml", ".yml"})
_SOURCE_FIELDS = frozenset(
    {"relative_path", "tree_sha256", "file_count", "byte_count", "version_file", "extension_version"}
)
_ENVIRONMENT_LOCK_FIELDS = frozenset({"repository_relative_path", "sha256"})
LIVE_GRAVITY_ABS_TOLERANCE_MPS2 = 1.0e-6


@dataclass(frozen=True)
class RuntimeLockIssue:
    code: str
    path: str
    message: str


class RuntimeLockError(ValueError):
    """Raised when a runtime lock cannot be loaded or audited."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def runtime_lock_sha256(lock: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(lock)).hexdigest()


def config_sha256(settings: Mapping[str, Any]) -> str:
    """Return the digest used for a locked render/Fabric/PhysX settings map."""

    if not isinstance(settings, Mapping):
        raise TypeError("configuration settings must be a mapping")
    return hashlib.sha256(_canonical_bytes(settings)).hexdigest()


def environment_lock_sha256(path: Path) -> str:
    """Hash the dependency lock with stable LF semantics on every host."""

    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _finite_number(value: Any) -> bool:
    """Return whether a JSON number can safely be applied to Isaac settings."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _finite_close(observed: Any, expected: Any, *, abs_tol: float) -> bool:
    return (
        _finite_number(observed)
        and _finite_number(expected)
        and math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=abs_tol)
    )


def _json_value_is_finite(value: Any) -> bool:
    """Validate that a direct Python lock value is representable as strict JSON."""

    if value is None or isinstance(value, (str, bool)):
        return True
    if _finite_number(value):
        return True
    if isinstance(value, list):
        return all(_json_value_is_finite(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _json_value_is_finite(item)
            for key, item in value.items()
        )
    return False


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("runtime lock contains a non-finite numeric literal")
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"runtime lock contains a non-standard JSON constant: {value}")


def live_gravity_matches(expected: Any, observed: Any) -> bool:
    """Compare a locked gravity vector with a PhysX float32 readback.

    Isaac's public gravity API returns a direction/magnitude representation
    backed by single-precision USD/PhysX values.  Keep the lock's intended
    decimal values, but allow no more than one float32-scale absolute ULP at
    Earth gravity.  Type, shape, finite-value, and all other runtime checks
    remain fail-closed.
    """

    if not isinstance(expected, list) or not isinstance(observed, list):
        return False
    if len(expected) != 3 or len(observed) != 3:
        return False
    for expected_value, observed_value in zip(expected, observed):
        if (
            not isinstance(expected_value, (int, float))
            or isinstance(expected_value, bool)
            or not isinstance(observed_value, (int, float))
            or isinstance(observed_value, bool)
        ):
            return False
        expected_float = float(expected_value)
        observed_float = float(observed_value)
        if not math.isfinite(expected_float) or not math.isfinite(observed_float):
            return False
        if not math.isclose(
            observed_float,
            expected_float,
            rel_tol=0.0,
            abs_tol=LIVE_GRAVITY_ABS_TOLERANCE_MPS2,
        ):
            return False
    return True


def source_tree_sha256(root: Path) -> tuple[str, int, int]:
    """Hash selected source files by relative POSIX path and content."""

    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimeLockError(f"IsaacLab source root is not a directory: {resolved}")
    files = sorted(
        (
            path
            for path in resolved.rglob("*")
            if path.suffix.lower() in _SOURCE_SUFFIXES
            and path.is_file()
            and "__pycache__" not in path.parts
        ),
        key=lambda path: PurePosixPath(path.relative_to(resolved)).as_posix(),
    )
    if not files:
        raise RuntimeLockError("IsaacLab source root contains no lockable source files")
    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        relative = PurePosixPath(path.relative_to(resolved)).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        byte_count += len(content)
    return digest.hexdigest(), len(files), byte_count


def load_runtime_lock(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeLockError(f"cannot read runtime lock {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeLockError("runtime lock must be an object")
    issues = validate_runtime_lock(payload)
    if issues:
        raise RuntimeLockError("invalid runtime lock: " + "; ".join(f"{issue.path}: {issue.message}" for issue in issues))
    return payload


def validate_runtime_lock(payload: Any) -> tuple[RuntimeLockIssue, ...]:
    issues: list[RuntimeLockIssue] = []

    def issue(code: str, path: str, message: str) -> None:
        issues.append(RuntimeLockIssue(code, path, message))

    if not isinstance(payload, Mapping):
        return (RuntimeLockIssue("type", "$", "runtime lock must be an object"),)
    required = {
        "schema",
        "profile_id",
        "python",
        "host",
        "distributions",
        "isaaclab_source",
        "isaaclab_contrib_source",
        "environment_lock",
        "gpu",
        "launcher",
        "simulation",
        "assets",
    }
    if set(payload) != required:
        issue("fields", "$", f"expected exactly {sorted(required)}")
    if payload.get("schema") != RUNTIME_LOCK_SCHEMA:
        issue("schema", "$.schema", f"expected {RUNTIME_LOCK_SCHEMA}")
    if not isinstance(payload.get("profile_id"), str) or not payload.get("profile_id"):
        issue("profile_id", "$.profile_id", "must be a non-empty string")
    python = payload.get("python")
    if not isinstance(python, Mapping) or set(python) != {"implementation", "version"}:
        issue("python", "$.python", "must contain implementation and exact version")
    elif any(not isinstance(python.get(field), str) or not python.get(field) for field in ("implementation", "version")):
        issue("python", "$.python", "implementation and version must be non-empty strings")
    host = payload.get("host")
    if not isinstance(host, Mapping) or set(host) != {"system", "machine", "minimum_windows_build"}:
        issue("host", "$.host", "must contain system, machine, and minimum_windows_build")
    else:
        if host.get("system") != "Windows":
            issue("host", "$.host.system", "must be Windows")
        if not isinstance(host.get("machine"), str) or not host.get("machine"):
            issue("host", "$.host.machine", "must be a non-empty string")
        if not _positive_integer(host.get("minimum_windows_build")):
            issue("host", "$.host.minimum_windows_build", "must be a positive integer")
    distributions = payload.get("distributions")
    if not isinstance(distributions, Mapping) or not distributions:
        issue("distributions", "$.distributions", "must map package names to exact versions")
    elif any(not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in distributions.items()):
        issue("distributions", "$.distributions", "package names and versions must be non-empty strings")
    for source_name in ("isaaclab_source", "isaaclab_contrib_source"):
        source = payload.get(source_name)
        if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
            issue(source_name, f"$.{source_name}", "must contain the complete source-tree binding")
        else:
            _validate_source_binding(source_name, source, issue)
    environment_lock = payload.get("environment_lock")
    if not isinstance(environment_lock, Mapping) or set(environment_lock) != _ENVIRONMENT_LOCK_FIELDS:
        issue(
            "environment_lock",
            "$.environment_lock",
            "must bind the repository-relative dependency lock path and SHA-256",
        )
    else:
        _validate_relative_path(
            "$.environment_lock.repository_relative_path",
            environment_lock.get("repository_relative_path"),
            issue,
        )
        if not _SHA256.fullmatch(str(environment_lock.get("sha256", ""))):
            issue("sha256", "$.environment_lock.sha256", "must be a lowercase SHA-256")
    gpu = payload.get("gpu")
    if not isinstance(gpu, Mapping) or set(gpu) != {"vendor", "minimum_driver_version", "minimum_vram_bytes"}:
        issue("gpu", "$.gpu", "must declare vendor, minimum driver, and minimum VRAM")
    else:
        if gpu.get("vendor") != "NVIDIA":
            issue("gpu", "$.gpu.vendor", "must be NVIDIA")
        if not isinstance(gpu.get("minimum_driver_version"), str) or not gpu.get("minimum_driver_version"):
            issue("gpu", "$.gpu.minimum_driver_version", "must be a non-empty string")
        if not _positive_integer(gpu.get("minimum_vram_bytes")):
            issue("gpu", "$.gpu.minimum_vram_bytes", "must be a positive integer")
    launcher = payload.get("launcher")
    launcher_fields = {
        "headless", "enable_cameras", "device", "rendering_mode", "livestream",
        "xr", "distributed", "kit_args", "experience",
    }
    if not isinstance(launcher, Mapping) or set(launcher) != launcher_fields:
        issue("launcher", "$.launcher", "must bind AppLauncher settings and experience")
    else:
        if not isinstance(launcher.get("headless"), bool):
            issue("type", "$.launcher.headless", "must be boolean")
        if launcher.get("enable_cameras") is not True:
            issue("cameras", "$.launcher.enable_cameras", "must be true for sensor capture")
        if not isinstance(launcher.get("device"), str) or launcher.get("device") not in {"cpu", "cuda", "cuda:0"}:
            issue("device", "$.launcher.device", "must be cpu, cuda, or cuda:0")
        if not isinstance(launcher.get("rendering_mode"), str) or launcher.get("rendering_mode") not in {
            "performance", "balanced", "quality"
        }:
            issue("rendering_mode", "$.launcher.rendering_mode", "must be a supported rendering preset")
        if (
            not isinstance(launcher.get("livestream"), int)
            or isinstance(launcher.get("livestream"), bool)
            or launcher.get("livestream") not in {0, 1, 2}
        ):
            issue("livestream", "$.launcher.livestream", "must be 0, 1, or 2")
        if not isinstance(launcher.get("xr"), bool):
            issue("xr", "$.launcher.xr", "must be boolean")
        if not isinstance(launcher.get("distributed"), bool):
            issue("distributed", "$.launcher.distributed", "must be boolean")
        if not isinstance(launcher.get("kit_args"), str):
            issue("kit_args", "$.launcher.kit_args", "must be one command-line argument string")
        experience = launcher.get("experience")
        if not isinstance(experience, Mapping) or set(experience) != {"path", "sha256"}:
            issue("experience", "$.launcher.experience", "must contain relative path and SHA-256")
        else:
            _validate_relative_path("$.launcher.experience.path", experience.get("path"), issue)
            if not _SHA256.fullmatch(str(experience.get("sha256", ""))):
                issue("sha256", "$.launcher.experience.sha256", "must be a lowercase SHA-256")
    simulation = payload.get("simulation")
    simulation_fields = {
        "device", "dt_s", "gravity_w_mps2", "agent_count", "render_interval", "use_fabric", "config_digests"
    }
    if not isinstance(simulation, Mapping) or set(simulation) != simulation_fields:
        issue("simulation", "$.simulation", "must bind device, dt, gravity, render interval, Fabric, and config digests")
    else:
        if not isinstance(simulation.get("device"), str) or simulation.get("device") not in {"cpu", "cuda", "cuda:0"}:
            issue("device", "$.simulation.device", "must be cpu, cuda, or cuda:0")
        if not _finite_number(simulation.get("dt_s")) or float(simulation.get("dt_s")) <= 0:
            issue("dt", "$.simulation.dt_s", "must be a positive number")
        gravity = simulation.get("gravity_w_mps2")
        if not isinstance(gravity, list) or len(gravity) != 3 or any(
            not _finite_number(value)
            for value in gravity
        ):
            issue("gravity", "$.simulation.gravity_w_mps2", "must contain three numeric world-frame values")
        if not _positive_integer(simulation.get("agent_count")):
            issue("agent_count", "$.simulation.agent_count", "must be a positive integer")
        if not _positive_integer(simulation.get("render_interval")):
            issue("render_interval", "$.simulation.render_interval", "must be a positive integer")
        if not isinstance(simulation.get("use_fabric"), bool):
            issue("use_fabric", "$.simulation.use_fabric", "must be boolean")
        digests = simulation.get("config_digests")
        if not isinstance(digests, Mapping) or set(digests) != {"render", "fabric", "physx"}:
            issue("config_digests", "$.simulation.config_digests", "must contain render, fabric, and physx digests")
        else:
            for name in ("render", "fabric", "physx"):
                digest = digests.get(name)
                if not isinstance(digest, Mapping) or set(digest) != {"settings", "sha256"}:
                    issue("config_digest", f"$.simulation.config_digests.{name}", "must contain settings and sha256")
                else:
                    settings = digest.get("settings")
                    if not isinstance(settings, Mapping):
                        issue("config_settings", f"$.simulation.config_digests.{name}.settings", "must be an object")
                    elif not _json_value_is_finite(settings):
                        issue(
                            "config_settings",
                            f"$.simulation.config_digests.{name}.settings",
                            "must contain only finite JSON values",
                        )
                    elif digest.get("sha256") != config_sha256(settings):
                        issue("config_digest", f"$.simulation.config_digests.{name}.sha256", "does not match canonical settings digest")
        launcher_values = launcher if isinstance(launcher, Mapping) else {}
        render_digest = digests.get("render") if isinstance(digests, Mapping) else None
        fabric_digest = digests.get("fabric") if isinstance(digests, Mapping) else None
        render_candidate = render_digest.get("settings") if isinstance(render_digest, Mapping) else None
        fabric_candidate = fabric_digest.get("settings") if isinstance(fabric_digest, Mapping) else None
        render_settings = render_candidate if isinstance(render_candidate, Mapping) else {}
        fabric_settings = fabric_candidate if isinstance(fabric_candidate, Mapping) else {}
        if render_settings.get("rendering_mode") != launcher_values.get("rendering_mode"):
            issue("renderer", "$.simulation.config_digests.render.settings.rendering_mode", "must match launcher rendering mode")
        if fabric_settings.get("use_fabric") != simulation.get("use_fabric"):
            issue("fabric", "$.simulation.config_digests.fabric.settings.use_fabric", "must match simulation use_fabric")
        if launcher_values.get("device") != simulation.get("device"):
            issue("device", "$.launcher.device", "must match simulation device")
    assets = payload.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != {"city_lite_contract_sha256", "cf2x_usd_sha256"}:
        issue("assets", "$.assets", "must bind City-Lite and CF2X hashes")
    elif any(not _SHA256.fullmatch(str(value)) for value in assets.values()):
        issue("sha256", "$.assets", "asset hashes must be lowercase SHA-256 values")
    return tuple(issues)


def _validate_relative_path(path: str, value: Any, issue: Any) -> None:
    if not isinstance(value, str) or not value:
        issue("path", path, "must be a non-empty relative POSIX path")
        return
    normalized = PurePosixPath(value)
    if normalized.is_absolute() or ".." in normalized.parts or "\\" in value:
        issue("path", path, "must be a relative POSIX path without parent traversal")


def _validate_source_binding(name: str, source: Mapping[str, Any], issue: Any) -> None:
    _validate_relative_path(f"$.{name}.relative_path", source.get("relative_path"), issue)
    if not _SHA256.fullmatch(str(source.get("tree_sha256", ""))):
        issue("sha256", f"$.{name}.tree_sha256", "must be a lowercase SHA-256")
    for field in ("file_count", "byte_count"):
        if not _positive_integer(source.get(field)):
            issue("count", f"$.{name}.{field}", "must be a positive integer")
    for field in ("version_file", "extension_version"):
        if not isinstance(source.get(field), str) or not source.get(field):
            issue("version", f"$.{name}.{field}", "must be a non-empty string")


def _source_binding(root: Path, *, relative_path: str) -> dict[str, Any]:
    """Measure a source extension without importing Isaac or Kit."""

    resolved = Path(root).expanduser().resolve()
    digest, file_count, byte_count = source_tree_sha256(resolved)
    extension_path = resolved / "config" / "extension.toml"
    extension_version: str | None = None
    if extension_path.is_file():
        match = re.search(
            r'^version\s*=\s*"([^"]+)"',
            extension_path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        extension_version = match.group(1) if match else None
    checkout_version = resolved.parents[1] / "VERSION"
    version_file = (
        checkout_version.read_text(encoding="utf-8").strip()
        if checkout_version.is_file()
        else extension_version
    )
    return {
        "relative_path": relative_path,
        "tree_sha256": digest,
        "file_count": file_count,
        "byte_count": byte_count,
        "version_file": version_file,
        "extension_version": extension_version,
    }


def resolve_locked_experience(lock: Mapping[str, Any], isaaclab_source: Path) -> Path:
    """Resolve the lock's experience relative to the IsaacLab checkout root."""

    relative = PurePosixPath(str(lock["launcher"]["experience"]["path"]))
    source_text = str(Path(isaaclab_source).expanduser())
    if re.match(r"^[A-Za-z]:[\\/]", source_text):
        checkout_root = PureWindowsPath(source_text).parents[1]
        return Path(str(checkout_root.joinpath(*relative.parts)))
    checkout_root = Path(source_text).resolve().parents[1]
    return checkout_root.joinpath(*relative.parts)


def resolve_locked_environment_lock(lock_path: Path, lock: Mapping[str, Any]) -> Path:
    """Resolve the repository-level dependency lock bound by a runtime lock.

    Runtime profiles are checked in at ``<repository>/config/*.json``.  The
    environment lock path is deliberately repository-relative so a clone can
    move as a unit without rewriting machine-specific paths.  Resolution is
    constrained to that repository root and never permits an absolute or
    escaping path.
    """

    runtime_path = Path(lock_path).expanduser().resolve()
    repository_root = runtime_path.parent.parent
    relative = PurePosixPath(str(lock["environment_lock"]["repository_relative_path"]))
    candidate = repository_root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(repository_root)
    except ValueError as exc:
        raise RuntimeLockError("environment lock path escapes the runtime-lock repository") from exc
    return candidate


def locked_launcher_kwargs(lock: Mapping[str, Any], isaaclab_source: Path) -> dict[str, Any]:
    """Build only documented ``AppLauncher`` arguments from a validated lock."""

    launcher = lock["launcher"]
    return {
        "headless": bool(launcher["headless"]),
        "enable_cameras": bool(launcher["enable_cameras"]),
        "device": str(launcher["device"]),
        "rendering_mode": str(launcher["rendering_mode"]),
        "livestream": int(launcher["livestream"]),
        "xr": bool(launcher["xr"]),
        "distributed": bool(launcher["distributed"]),
        "kit_args": str(launcher["kit_args"]),
        "experience": str(resolve_locked_experience(lock, isaaclab_source)),
    }


def validate_locked_launcher_environment(lock: Mapping[str, Any]) -> None:
    """Reject environment variables AppLauncher cannot override safely.

    ``xr=False`` does not override ``XR=1`` in IsaacLab 2.3.2.  The remaining
    locked launch arguments are explicitly passed, so their ambient values are
    recorded by the process but cannot change the resolved launcher state.
    """

    launcher = lock["launcher"]
    xr = os.environ.get("XR", "0")
    if not str(xr).strip().isdigit() or int(xr) not in {0, 1}:
        raise RuntimeLockError("XR must be 0 or 1 before a locked AppLauncher starts")
    if launcher["xr"] is False and int(xr) != 0:
        raise RuntimeLockError("XR=1 conflicts with a locked non-XR AppLauncher")


def configure_simulation_cfg(sim_cfg: Any, lock: Mapping[str, Any]) -> None:
    """Apply every lock-owned SimulationCfg setting before context creation."""

    simulation = lock["simulation"]
    sim_cfg.device = str(simulation["device"])
    sim_cfg.dt = float(simulation["dt_s"])
    sim_cfg.gravity = tuple(float(value) for value in simulation["gravity_w_mps2"])
    sim_cfg.render_interval = int(simulation["render_interval"])
    for name, value in simulation["config_digests"]["fabric"]["settings"].items():
        if not hasattr(sim_cfg, name):
            raise RuntimeLockError(f"locked fabric setting is not supported by SimulationCfg: {name}")
        setattr(sim_cfg, name, value)
    for section in ("render", "physx"):
        target = getattr(sim_cfg, section)
        for name, value in simulation["config_digests"][section]["settings"].items():
            if not hasattr(target, name):
                raise RuntimeLockError(
                    f"locked {section} setting is not supported by this IsaacLab version: {name}"
                )
            setattr(target, name, value)


def _configured_settings(target: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    return {name: getattr(target, name) for name in settings}


def observe_live_simulation(lock: Mapping[str, Any], sim: Any) -> dict[str, Any]:
    """Read public SimulationContext state after Kit has created the context."""

    simulation = lock["simulation"]
    gravity_direction, gravity_magnitude = sim.get_physics_context().get_gravity()
    gravity = [float(value) * float(gravity_magnitude) for value in gravity_direction]
    config_digests = simulation["config_digests"]
    render_settings = _configured_settings(
        sim.cfg.render, config_digests["render"]["settings"]
    )
    fabric_settings = _configured_settings(sim.cfg, config_digests["fabric"]["settings"])
    physx_settings = _configured_settings(
        sim.cfg.physx, config_digests["physx"]["settings"]
    )
    return {
        "device": str(sim.device),
        "physics_dt_s": float(sim.get_physics_dt()),
        "rendering_dt_s": float(sim.get_rendering_dt()),
        "gravity_w_mps2": gravity,
        "render_interval": int(round(float(sim.get_rendering_dt()) / float(sim.get_physics_dt()))),
        "use_fabric": bool(sim.is_fabric_enabled()),
        "enable_scene_query_support": bool(sim.cfg.enable_scene_query_support),
        "rendering_mode": sim.carb_settings.get("/isaaclab/rendering/rendering_mode"),
        "rtx_sensors_active": bool(sim.has_rtx_sensors()),
        "config_digests": {
            "render": config_sha256(render_settings),
            "fabric": config_sha256(fabric_settings),
            "physx": config_sha256(physx_settings),
        },
        "configuration_observation": "public_simulation_context_and_locked_cfg",
    }


def compare_live_simulation(
    lock: Mapping[str, Any], observed: Mapping[str, Any]
) -> tuple[RuntimeLockIssue, ...]:
    """Compare the public live state with configuration already checked pre-Kit."""

    simulation = lock["simulation"]
    issues: list[RuntimeLockIssue] = []

    def mismatch(path: str, expected: Any, actual: Any) -> None:
        issues.append(RuntimeLockIssue("mismatch", path, f"expected {expected!r}, observed {actual!r}"))

    if observed.get("device") != simulation["device"]:
        mismatch("$.simulation.device", simulation["device"], observed.get("device"))
    if not _finite_close(
        observed.get("physics_dt_s"), simulation["dt_s"], abs_tol=1.0e-12
    ):
        mismatch("$.simulation.dt_s", simulation["dt_s"], observed.get("physics_dt_s"))
    expected_render_dt = float(simulation["dt_s"]) * int(simulation["render_interval"])
    if not _finite_close(
        observed.get("rendering_dt_s"), expected_render_dt, abs_tol=1.0e-12
    ):
        mismatch("$.simulation.render_interval", expected_render_dt, observed.get("rendering_dt_s"))
    expected_gravity = simulation["gravity_w_mps2"]
    actual_gravity = observed.get("gravity_w_mps2")
    if not live_gravity_matches(expected_gravity, actual_gravity):
        mismatch("$.simulation.gravity_w_mps2", expected_gravity, actual_gravity)
    if observed.get("render_interval") != simulation["render_interval"]:
        mismatch("$.simulation.render_interval", simulation["render_interval"], observed.get("render_interval"))
    if observed.get("use_fabric") != simulation["use_fabric"]:
        mismatch("$.simulation.use_fabric", simulation["use_fabric"], observed.get("use_fabric"))
    if observed.get("rendering_mode") != lock["launcher"]["rendering_mode"]:
        mismatch("$.launcher.rendering_mode", lock["launcher"]["rendering_mode"], observed.get("rendering_mode"))
    observed_digests = observed.get("config_digests", {})
    for name, payload in simulation["config_digests"].items():
        if observed_digests.get(name) != payload["sha256"]:
            mismatch(
                f"$.simulation.config_digests.{name}.sha256",
                payload["sha256"],
                observed_digests.get(name),
            )
    return tuple(issues)


def observe_runtime(
    lock: Mapping[str, Any],
    *,
    isaaclab_source: Path,
    scene_contract: Path,
    cf2x_usd: Path,
    environment_lock: Path | None = None,
) -> dict[str, Any]:
    distributions: dict[str, str | None] = {}
    for name in lock["distributions"]:
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    source_root = Path(isaaclab_source).expanduser().resolve()
    source_base = source_root.parent
    expected_source = source_base.joinpath(
        *PurePosixPath(str(lock["isaaclab_source"]["relative_path"])).parts
    )
    contrib_root = source_base.joinpath(
        *PurePosixPath(str(lock["isaaclab_contrib_source"]["relative_path"])).parts
    )
    experience = resolve_locked_experience(lock, source_root)
    gpus, gpu_error = _probe_nvidia_smi()
    return {
        "configuration_observation": RUNTIME_AUDIT_OBSERVATION,
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
        "host": {"system": platform.system(), "machine": platform.machine(), "windows_build": sys.getwindowsversion().build if sys.platform == "win32" else None},
        "distributions": distributions,
        "isaaclab_source": _source_binding(
            source_root,
            relative_path=source_root.name,
        ),
        "isaaclab_contrib_source": _source_binding(
            contrib_root,
            relative_path=contrib_root.name,
        ),
        "launcher": {
            "experience": {
                "path": str(lock["launcher"]["experience"]["path"]),
                "sha256": sha256_file(experience) if experience.is_file() else None,
            },
            "isaaclab_source_path_matches": source_root == expected_source,
        },
        "gpu": {"devices": gpus, "probe_error": gpu_error},
        "assets": {
            "city_lite_contract_sha256": sha256_file(scene_contract.resolve()) if scene_contract.is_file() else None,
            "cf2x_usd_sha256": sha256_file(cf2x_usd.resolve()) if cf2x_usd.is_file() else None,
        },
        "environment_lock": {
            "repository_relative_path": lock["environment_lock"]["repository_relative_path"],
            "sha256": environment_lock_sha256(environment_lock) if environment_lock is not None and environment_lock.is_file() else None,
        },
    }


def compare_runtime(lock: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[RuntimeLockIssue, ...]:
    issues: list[RuntimeLockIssue] = []

    def exact(path: str, expected: Any, actual: Any) -> None:
        if actual != expected:
            issues.append(RuntimeLockIssue("mismatch", path, f"expected {expected!r}, observed {actual!r}"))

    for key, expected in lock["python"].items():
        exact(f"$.python.{key}", expected, observed.get("python", {}).get(key))
    host_observed = observed.get("host", {})
    exact("$.host.system", lock["host"]["system"], host_observed.get("system"))
    exact("$.host.machine", lock["host"]["machine"], host_observed.get("machine"))
    build = host_observed.get("windows_build")
    if not isinstance(build, int) or build < int(lock["host"]["minimum_windows_build"]):
        issues.append(RuntimeLockIssue("incompatible", "$.host.minimum_windows_build", f"requires >= {lock['host']['minimum_windows_build']}, observed {build!r}"))
    for name, expected in lock["distributions"].items():
        exact(f"$.distributions.{name}", expected, observed.get("distributions", {}).get(name))
    for source_name in ("isaaclab_source", "isaaclab_contrib_source"):
        for key, expected in lock[source_name].items():
            exact(
                f"$.{source_name}.{key}",
                expected,
                observed.get(source_name, {}).get(key),
            )
    launcher_observed = observed.get("launcher", {})
    if launcher_observed.get("isaaclab_source_path_matches") is not True:
        issues.append(
            RuntimeLockIssue(
                "mismatch",
                "$.isaaclab_source.relative_path",
                "the supplied IsaacLab source is not the locked sibling path",
            )
        )
    exact(
        "$.launcher.experience.path",
        lock["launcher"]["experience"]["path"],
        launcher_observed.get("experience", {}).get("path"),
    )
    exact(
        "$.launcher.experience.sha256",
        lock["launcher"]["experience"]["sha256"],
        launcher_observed.get("experience", {}).get("sha256"),
    )
    for key, expected in lock["assets"].items():
        exact(f"$.assets.{key}", expected, observed.get("assets", {}).get(key))
    environment_observed = observed.get("environment_lock", {})
    exact(
        "$.environment_lock.repository_relative_path",
        lock["environment_lock"]["repository_relative_path"],
        environment_observed.get("repository_relative_path"),
    )
    exact(
        "$.environment_lock.sha256",
        lock["environment_lock"]["sha256"],
        environment_observed.get("sha256"),
    )
    gpu_lock = lock["gpu"]
    devices = observed.get("gpu", {}).get("devices", [])
    capable = [
        item for item in devices
        if isinstance(item, Mapping)
        and isinstance(item.get("vram_bytes"), int)
        and item["vram_bytes"] >= int(gpu_lock["minimum_vram_bytes"])
        and _version_at_least(item.get("driver_version"), str(gpu_lock["minimum_driver_version"]))
    ]
    if gpu_lock["vendor"] != "NVIDIA" or not capable:
        issues.append(RuntimeLockIssue("gpu_incompatible", "$.gpu", "no NVIDIA GPU satisfies the locked driver and VRAM requirements"))
    return tuple(issues)


def audit_runtime_lock(
    lock_path: Path,
    *,
    isaaclab_source: Path,
    scene_contract: Path,
    cf2x_usd: Path,
) -> dict[str, Any]:
    lock = load_runtime_lock(lock_path)
    environment_lock = resolve_locked_environment_lock(lock_path, lock)
    observed = observe_runtime(
        lock,
        isaaclab_source=isaaclab_source,
        scene_contract=scene_contract,
        cf2x_usd=cf2x_usd,
        environment_lock=environment_lock,
    )
    issues = compare_runtime(lock, observed)
    return {
        "schema": RUNTIME_AUDIT_SCHEMA,
        "status": "passed" if not issues else "failed",
        "profile_id": lock["profile_id"],
        "runtime_lock_sha256": runtime_lock_sha256(lock),
        "configuration_observation": RUNTIME_AUDIT_OBSERVATION,
        "observed": observed,
        "issues": [issue.__dict__ for issue in issues],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--isaaclab-source", type=Path, required=True)
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--cf2x-usd", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_runtime_lock(
            args.lock,
            isaaclab_source=args.isaaclab_source,
            scene_contract=args.scene_contract,
            cf2x_usd=args.cf2x_usd,
        )
    except (OSError, RuntimeLockError, ValueError) as exc:
        report = {"schema": RUNTIME_AUDIT_SCHEMA, "status": "failed", "error": str(exc)}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
