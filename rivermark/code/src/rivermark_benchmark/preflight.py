"""Bounded, Isaac-free preflight checks for storage and source provenance.

This module runs before any Isaac AppLauncher is created. It does not promise
that a GPU, scene, or Kit renderer can run; it prevents avoidable launches when
declared storage, provenance, runtime, or authority requirements are invalid.
The existing runtime Windows commit guard remains authoritative during an
active capture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .citylite_scene import resolve_city_lite_authority
from .provenance import SourceProvenance, detect_source_provenance


PREFLIGHT_SCHEMA = "org.rivermark.benchmark.preflight.v1"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    value: Any
    requirement: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]
    source: SourceProvenance | None

    @property
    def valid(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PREFLIGHT_SCHEMA,
            "valid": self.valid,
            "checks": [asdict(check) for check in self.checks],
            "source": self.source.as_dict() if self.source else None,
        }


@dataclass(frozen=True)
class RuntimePreflightRequirements:
    """Optional host/runtime requirements checked before Isaac is imported."""

    require_gpu: bool = False
    minimum_gpu_vram_bytes: int = 0
    minimum_driver_version: str | None = None
    isaac_sim_version: str | None = None
    isaaclab_version: str | None = None
    scene_contract: Path | None = None
    scene_contract_sha256: str | None = None
    python_min_version: tuple[int, int] = (3, 10)
    runtime_lock: Path | None = None
    isaaclab_source: Path | None = None
    cf2x_usd: Path | None = None


def _version_key(value: str) -> tuple[int, ...]:
    numbers = tuple(int(part) for part in re.findall(r"\d+", value))
    if not numbers:
        raise ValueError(f"version has no numeric components: {value!r}")
    return numbers


def _version_at_least(actual: str | None, required: str) -> bool:
    if actual is None:
        return False
    try:
        actual_key = _version_key(actual)
        required_key = _version_key(required)
    except ValueError:
        return False
    width = max(len(actual_key), len(required_key))
    return actual_key + (0,) * (width - len(actual_key)) >= required_key + (0,) * (width - len(required_key))


def _probe_nvidia_smi() -> tuple[list[dict[str, Any]], str | None]:
    """Probe GPU name/VRAM/driver with a bounded, non-shell subprocess."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return [], "nvidia-smi was not found on PATH"
    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=5.0,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"nvidia-smi probe failed: {exc}"
    if completed.returncode != 0:
        detail = completed.stderr.strip()[:256]
        return [], f"nvidia-smi exited with {completed.returncode}: {detail}"
    records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        name, memory_text, driver = fields
        try:
            memory_mib = int(memory_text)
        except ValueError:
            memory_mib = None
        records.append(
            {
                "name": name,
                "vram_bytes": memory_mib * 1024 * 1024 if memory_mib is not None and memory_mib > 0 else None,
                "driver_version": driver or None,
            }
        )
    if not records:
        return [], "nvidia-smi returned no parseable GPU records"
    return records, None


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_checks(requirements: RuntimePreflightRequirements) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    if (
        not isinstance(requirements.python_min_version, tuple)
        or len(requirements.python_min_version) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in requirements.python_min_version
        )
    ):
        raise ValueError("python_min_version must be a (major, minor) tuple")
    if (
        not isinstance(requirements.minimum_gpu_vram_bytes, int)
        or isinstance(requirements.minimum_gpu_vram_bytes, bool)
        or requirements.minimum_gpu_vram_bytes < 0
    ):
        raise ValueError("minimum_gpu_vram_bytes must be non-negative")
    if requirements.scene_contract_sha256 is not None and (
        len(requirements.scene_contract_sha256) != 64
        or any(char not in "0123456789abcdef" for char in requirements.scene_contract_sha256)
    ):
        raise ValueError("scene_contract_sha256 must be 64 lowercase hexadecimal characters")

    actual_python = (sys.version_info.major, sys.version_info.minor)
    python_required = requirements.python_min_version
    checks.append(
        PreflightCheck(
            "python_version",
            actual_python >= python_required,
            {"major": actual_python[0], "minor": actual_python[1]},
            f">={python_required[0]}.{python_required[1]}",
            "Python runtime meets the declared minimum"
            if actual_python >= python_required
            else "refusing launch: Python runtime is below the declared minimum",
        )
    )

    for distribution, expected in (
        ("isaacsim", requirements.isaac_sim_version),
        ("isaaclab", requirements.isaaclab_version),
    ):
        if expected is None:
            continue
        actual = _installed_version(distribution)
        passed = actual is not None and _version_at_least(actual, expected)
        checks.append(
            PreflightCheck(
                f"{distribution}_version",
                passed,
                {"installed": actual, "required_minimum": expected},
                f"{distribution}>={expected}",
                f"{distribution} version meets the declared minimum"
                if passed
                else f"refusing launch: {distribution} is missing or below the declared minimum",
            )
        )

    if requirements.scene_contract is not None:
        contract_path = requirements.scene_contract.expanduser().resolve()
        exists = contract_path.is_file()
        actual_hash = sha256_file(contract_path) if exists else None
        hash_ok = requirements.scene_contract_sha256 is None or actual_hash == requirements.scene_contract_sha256
        checks.append(
            PreflightCheck(
                "scene_contract_file",
                exists and hash_ok,
                {"path": str(contract_path), "sha256": actual_hash},
                "City-Lite contract exists" + (" and SHA-256 matches" if requirements.scene_contract_sha256 else ""),
                "City-Lite contract file is present and hash-bound"
                if exists and hash_ok
                else "refusing launch: City-Lite contract is missing or hash-mismatched",
            )
        )
        authority_ok = False
        authority_value: dict[str, Any] = {"path": str(contract_path)}
        authority_message = "refusing launch: City-Lite authority validation failed"
        if exists and hash_ok:
            try:
                authority = resolve_city_lite_authority(contract_path)
            except (OSError, RuntimeError, ValueError) as exc:
                authority_value["error"] = str(exc)
            else:
                authority_ok = True
                authority_value.update(
                    {
                        "scene_id": "RIVERMARK_CITY_LITE_v1",
                        "contract_sha256": authority.contract_sha256,
                        "asset_count": len(authority.asset_paths),
                    }
                )
                authority_message = "City-Lite static authority and bound assets validate"
        checks.append(
            PreflightCheck(
                "scene_contract_authority",
                authority_ok,
                authority_value,
                "approved City-Lite v1_r2 authority and all bound asset hashes",
                authority_message,
            )
        )
    elif requirements.scene_contract_sha256 is not None:
        raise ValueError("scene_contract is required when scene_contract_sha256 is provided")

    gpu_required = (
        requirements.require_gpu
        or requirements.minimum_gpu_vram_bytes > 0
        or requirements.minimum_driver_version is not None
    )
    if gpu_required:
        gpus, probe_error = _probe_nvidia_smi()
        checks.append(
            PreflightCheck(
                "gpu_probe",
                bool(gpus),
                {"gpus": gpus, "error": probe_error},
                "at least one NVIDIA GPU is discoverable by nvidia-smi",
                "GPU probe returned at least one device"
                if gpus
                else "refusing launch: GPU probe did not return a usable device",
            )
        )
        capable = [
            gpu
            for gpu in gpus
            if isinstance(gpu.get("vram_bytes"), int)
            and gpu["vram_bytes"] >= requirements.minimum_gpu_vram_bytes
            and (
                requirements.minimum_driver_version is None
                or _version_at_least(gpu.get("driver_version"), requirements.minimum_driver_version)
            )
        ]
        checks.append(
            PreflightCheck(
                "gpu_capacity",
                bool(capable),
                {"capable_gpus": capable, "minimum_vram_bytes": requirements.minimum_gpu_vram_bytes, "minimum_driver_version": requirements.minimum_driver_version},
                "one GPU meets declared VRAM and driver requirements",
                "GPU VRAM/driver requirements are satisfied"
                if capable
                else "refusing launch: no GPU meets the declared VRAM/driver requirements",
            )
        )
    if requirements.runtime_lock is not None:
        lock_path = requirements.runtime_lock.expanduser().resolve()
        lock_inputs = {
            "runtime_lock": str(lock_path),
            "isaaclab_source": str(requirements.isaaclab_source) if requirements.isaaclab_source else None,
            "scene_contract": str(requirements.scene_contract) if requirements.scene_contract else None,
            "cf2x_usd": str(requirements.cf2x_usd) if requirements.cf2x_usd else None,
        }
        lock_ok = all(
            value is not None
            for value in (requirements.isaaclab_source, requirements.scene_contract, requirements.cf2x_usd)
        ) and lock_path.is_file()
        lock_report: dict[str, Any] = {"inputs": lock_inputs}
        if lock_ok:
            try:
                # Import lazily to preserve the Isaac-free preflight boundary and
                # avoid a module cycle: runtime_lock reuses helpers from here.
                from .runtime_lock import audit_runtime_lock

                lock_report = audit_runtime_lock(
                    lock_path,
                    isaaclab_source=Path(requirements.isaaclab_source),
                    scene_contract=Path(requirements.scene_contract),
                    cf2x_usd=Path(requirements.cf2x_usd),
                )
                lock_ok = lock_report.get("status") == "passed"
            except (OSError, RuntimeError, ValueError) as exc:
                lock_report = {**lock_report, "status": "failed", "error": str(exc)}
                lock_ok = False
        else:
            lock_report = {
                **lock_report,
                "status": "failed",
                "error": "runtime lock and all bound paths are required",
            }
        checks.append(
            PreflightCheck(
                "runtime_lock",
                lock_ok,
                lock_report,
                "runtime lock, IsaacLab source, City-Lite contract, and CF2X asset audit passes",
                "locked runtime and asset provenance validate"
                if lock_ok
                else "refusing launch: runtime lock or one of its bound inputs failed",
            )
        )
    return checks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_volume_probe(path: Path) -> Path:
    """Return the nearest existing path so nested output dirs are preflightable."""

    candidate = path.expanduser().resolve()
    if candidate.exists():
        return candidate
    for parent in candidate.parents:
        if parent.exists():
            return parent
    raise OSError(f"cannot find an existing volume for output directory: {path}")


def run_preflight(
    *,
    output_dir: Path,
    minimum_free_bytes: int,
    estimated_capture_bytes: int = 0,
    source_root: Path | None = None,
    required_assets: Iterable[tuple[Path, str | None]] = (),
    require_clean: bool = True,
    runtime: RuntimePreflightRequirements | None = None,
) -> PreflightReport:
    """Run deterministic checks without importing Isaac, Torch, or GPU APIs."""

    if minimum_free_bytes < 0 or estimated_capture_bytes < 0:
        raise ValueError("storage byte thresholds must be non-negative")
    checks: list[PreflightCheck] = []
    output_dir = output_dir.expanduser().resolve()
    probe = _existing_volume_probe(output_dir)
    usage = shutil.disk_usage(probe)
    required_free = minimum_free_bytes + estimated_capture_bytes
    checks.append(
        PreflightCheck(
            "disk_free_bytes",
            usage.free >= required_free,
            usage.free,
            f">={required_free}",
            "output volume has enough free space for the requested capture budget"
            if usage.free >= required_free
            else "refusing launch: output volume is below the bounded storage budget",
        )
    )
    if output_dir.exists() and not output_dir.is_dir():
        checks.append(
            PreflightCheck(
                "output_directory",
                False,
                str(output_dir),
                "output path is a directory or does not exist yet",
                "refusing launch: output path is an existing non-directory",
            )
        )
    source: SourceProvenance | None = None
    if source_root is not None:
        source = detect_source_provenance(source_root)
        clean = not source.source_worktree_dirty
        checks.append(
            PreflightCheck(
                "clean_source",
                clean or not require_clean,
                clean,
                "clean Git worktree" if require_clean else "source provenance recorded",
                "source worktree is clean" if clean else "source worktree is dirty",
            )
        )
    for asset, expected_hash in required_assets:
        resolved = asset.expanduser().resolve()
        exists = resolved.is_file()
        actual_hash = sha256_file(resolved) if exists and expected_hash else None
        hash_ok = expected_hash is None or actual_hash == expected_hash
        checks.append(
            PreflightCheck(
                f"asset:{asset}",
                exists and hash_ok,
                {"path": str(resolved), "sha256": actual_hash},
                "file exists" + (" and SHA-256 matches" if expected_hash else ""),
                "asset is present and hash-bound" if exists and hash_ok else "required asset is missing or hash-mismatched",
            )
        )
    if runtime is not None:
        checks.extend(_runtime_checks(runtime))
    return PreflightReport(tuple(checks), source)


def _parse_asset(value: str) -> tuple[Path, str | None]:
    path, separator, digest = value.partition("=")
    if not path:
        raise argparse.ArgumentTypeError("asset path cannot be empty")
    if separator and (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
        raise argparse.ArgumentTypeError("asset hash must be 64 lowercase hexadecimal characters")
    return Path(path), digest or None


def _parse_version(value: str) -> str:
    try:
        _version_key(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--estimated-capture-gib", type=float, default=0.0)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--asset", action="append", type=_parse_asset, default=[])
    parser.add_argument("--scene-contract", type=Path)
    parser.add_argument("--scene-contract-sha256")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--minimum-gpu-vram-gib", type=float, default=0.0)
    parser.add_argument("--minimum-driver-version", type=_parse_version)
    parser.add_argument("--isaac-sim-version", type=_parse_version)
    parser.add_argument("--isaaclab-version", type=_parse_version)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--isaaclab-source", type=Path)
    parser.add_argument("--cf2x-usd", type=Path)
    parser.add_argument("--allow-dirty-source", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if not math.isfinite(args.minimum_gpu_vram_gib) or args.minimum_gpu_vram_gib < 0.0:
            raise ValueError("--minimum-gpu-vram-gib must be finite and non-negative")
        report = run_preflight(
            output_dir=args.output_dir,
            minimum_free_bytes=int(args.minimum_free_gib * 1024**3),
            estimated_capture_bytes=int(args.estimated_capture_gib * 1024**3),
            source_root=args.source_root,
            required_assets=args.asset,
            require_clean=not args.allow_dirty_source,
            runtime=RuntimePreflightRequirements(
                require_gpu=args.require_gpu,
                minimum_gpu_vram_bytes=int(args.minimum_gpu_vram_gib * 1024**3),
                minimum_driver_version=args.minimum_driver_version,
                isaac_sim_version=args.isaac_sim_version,
                isaaclab_version=args.isaaclab_version,
                scene_contract=args.scene_contract,
                scene_contract_sha256=args.scene_contract_sha256,
                runtime_lock=args.runtime_lock,
                isaaclab_source=args.isaaclab_source,
                cf2x_usd=args.cf2x_usd,
            ),
        )
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return 0 if report.valid else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"schema": PREFLIGHT_SCHEMA, "valid": False, "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
