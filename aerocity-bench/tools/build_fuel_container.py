"""Validate the external FUEL source lock before a container build.

The tool deliberately has no default source path and never invokes a planner.
It prevents a local zip snapshot, a modified checkout, or a different upstream
revision from being mistaken for the frozen GPL external-method dependency.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPOSITORY_ROOT / "external" / "fuel" / "source-lock.json"


def _run_git(source: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown git failure"
        raise ValueError(f"cannot inspect FUEL source checkout: {detail}")
    return completed.stdout.strip()


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "method_id",
        "upstream_url",
        "upstream_commit",
        "upstream_license",
        "process_boundary",
        "ros_distribution",
        "base_image",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"FUEL source lock is missing fields: {', '.join(missing)}")
    if payload["upstream_license"] != "GPL-3.0-only":
        raise ValueError("FUEL source lock must retain its GPL-3.0-only license boundary")
    if payload["process_boundary"] != "container":
        raise ValueError("FUEL source lock must require a container boundary")
    if len(payload["upstream_commit"]) != 40:
        raise ValueError("FUEL source lock must contain a full 40-character Git revision")
    if "@sha256:" not in payload["base_image"]:
        raise ValueError("FUEL source lock must pin the ROS base image by digest")
    return payload


def verify_source(source: Path, lock: dict[str, Any]) -> dict[str, str]:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"FUEL source directory does not exist: {source}")
    if not (source / ".git").exists():
        raise ValueError("FUEL source must be a Git checkout, not an unversioned snapshot")

    revision = _run_git(source, "rev-parse", "HEAD")
    remote = _run_git(source, "remote", "get-url", "origin")
    status = _run_git(source, "status", "--porcelain")
    if revision != lock["upstream_commit"]:
        raise ValueError(
            f"FUEL revision mismatch: expected {lock['upstream_commit']}, found {revision}"
        )
    if remote != lock["upstream_url"]:
        raise ValueError(f"FUEL origin mismatch: expected {lock['upstream_url']}, found {remote}")
    if status:
        raise ValueError("FUEL source checkout is dirty; use a clean locked checkout")
    return {"source": str(source), "revision": revision, "origin": remote}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="clean locked FUEL checkout")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate provenance only; container invocation is intentionally external",
    )
    arguments = parser.parse_args(argv)

    try:
        lock = load_lock()
        source = verify_source(arguments.source, lock)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FUEL_EXTERNAL_BUILD_REJECTED: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "FUEL_EXTERNAL_SOURCE_VERIFIED",
                "verify_only": arguments.verify_only,
                "lock": lock,
                "source": source,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
