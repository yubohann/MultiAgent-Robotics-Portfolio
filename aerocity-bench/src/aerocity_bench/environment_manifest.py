"""Collect privacy-preserving environment evidence for release reproducibility.

The manifest deliberately records versions and content hashes rather than local
installation paths.  A manifest from a dirty source tree is useful development
evidence, but it cannot bind an official release to a Git commit.
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .canonical import content_hash, file_hash, read_json

ENVIRONMENT_MANIFEST_SCHEMA = "org.aerocity.bench.release-environment-manifest.v1"


def _run_text(command: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, "", f"{type(error).__name__}: {error}"
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_version(module_name: str, attribute: str = "__version__") -> str | None:
    try:
        module = __import__(module_name, fromlist=[attribute])
    except ImportError:
        return None
    value = getattr(module, attribute, None)
    return str(value) if value is not None else None


def _git_source_state(repository_root: Path) -> dict[str, str]:
    status_code, status_stdout, _ = _run_text(
        ["git", "-C", str(repository_root), "status", "--porcelain=v1"]
    )
    commit_code, commit_stdout, _ = _run_text(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"]
    )
    if status_code != 0 or commit_code != 0 or len(commit_stdout) != 40:
        return {
            "state": "UNVERIFIABLE",
            "source_commit": "UNVERIFIABLE",
            "official_release_binding": "REJECTED",
        }
    if status_stdout:
        return {
            "state": "DIRTY",
            "source_commit": "UNCOMMITTED-DEVELOPMENT",
            "official_release_binding": "REJECTED",
        }
    return {
        "state": "CLEAN",
        "source_commit": commit_stdout,
        "official_release_binding": "VALID",
    }


def _gpu_inventory(runner: Callable[[list[str]], tuple[int, str, str]]) -> dict[str, object]:
    code, stdout, stderr = runner(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    if code != 0:
        return {"status": "UNAVAILABLE", "devices": [], "diagnostic": stderr[-240:]}
    devices = [line.strip() for line in stdout.splitlines() if line.strip()]
    return {"status": "AVAILABLE", "devices": devices, "diagnostic": ""}


def _usd_version(
    runner: Callable[[list[str]], tuple[int, str, str]] = _run_text,
) -> dict[str, str | None]:
    """Probe USD in a child interpreter without loading native extensions here.

    Isaac/PXR libraries can retain native process state after simulator tests.  A
    release manifest must not make an otherwise completed test process unsafe
    merely to collect an informational version field.  The child process is the
    fault boundary; probe failure is represented in the manifest rather than
    inferred from a partial import in the caller.
    """

    code = (
        "from pxr import Usd\n"
        "print('AEROCITY_USD_VERSION=' + '.'.join(str(part) for part in Usd.GetVersion()))\n"
    )
    returncode, stdout, stderr = runner([sys.executable, "-c", code])
    if returncode != 0:
        return {
            "status": "UNAVAILABLE",
            "version": None,
            "diagnostic": stderr[-240:] or f"probe exited with code {returncode}",
        }
    match = re.search(r"^AEROCITY_USD_VERSION=([^\r\n]+)$", stdout, flags=re.MULTILINE)
    if match is None:
        return {
            "status": "UNAVAILABLE",
            "version": None,
            "diagnostic": "USD version probe returned no valid version marker",
        }
    return {"status": "AVAILABLE", "version": match.group(1), "diagnostic": ""}


def build_environment_manifest(
    *,
    repository_root: Path,
    cf2x_usd: Path,
    release_config: Path | None = None,
    asset_audit: Path | None = None,
    runner: Callable[[list[str]], tuple[int, str, str]] = _run_text,
) -> dict[str, object]:
    """Create a content-addressed manifest without serializing local paths."""

    root = repository_root.resolve()
    cf2x = cf2x_usd.resolve()
    if not cf2x.is_file():
        raise FileNotFoundError(f"CF2X USD does not exist: {cf2x}")
    if "5_in_drone" in {part.lower() for part in cf2x.parts}:
        raise ValueError("the forbidden 5_in_drone asset cannot enter a manifest")
    if cf2x.name.lower() != "cf2x.usd":
        raise ValueError("the manifest requires assets/new/cf2x.usd")

    inputs: dict[str, object] = {
        "cf2x": {
            "asset_id": "cf2x-local-hash-locked",
            "sha256": file_hash(cf2x),
            "redistributed": False,
        },
        "release_config_sha256": None,
        "asset_audit": None,
    }
    if release_config is not None:
        config = release_config.resolve()
        if not config.is_file():
            raise FileNotFoundError(f"release config does not exist: {config}")
        inputs["release_config_sha256"] = file_hash(config)
    if asset_audit is not None:
        audit = read_json(asset_audit.resolve())
        if audit.get("schema") != "org.aerocity.bench.cc0-release-asset-audit.v1":
            raise ValueError("asset audit schema differs")
        if audit.get("status") != "PASS" or audit.get("formal_score_eligible") is not False:
            raise ValueError("asset audit is not a valid development-only PASS")
        inputs["asset_audit"] = {
            "report_hash": audit.get("report_hash"),
            "registry_hash": audit.get("registry_hash"),
            "asset_count": audit.get("asset_count"),
            "usd_dependency_closure": audit.get("usd_dependency_closure"),
        }

    report: dict[str, object] = {
        "schema": ENVIRONMENT_MANIFEST_SCHEMA,
        "formal_score_eligible": False,
        "scope": "development_or_calibration_reproducibility_only",
        "source_tree": _git_source_state(root),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_sha256": file_hash(Path(sys.executable)),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "runtime_packages": {
            "aerocity_bench": _module_version("aerocity_bench"),
            "isaacsim": _distribution_version("isaacsim"),
            "isaaclab": _distribution_version("isaaclab"),
            "omni_physics": _distribution_version("omni-physics"),
            "pxr_usd": _usd_version(runner),
        },
        "gpu": _gpu_inventory(runner),
        "inputs": inputs,
    }
    report["manifest_hash"] = content_hash(report)
    return report
