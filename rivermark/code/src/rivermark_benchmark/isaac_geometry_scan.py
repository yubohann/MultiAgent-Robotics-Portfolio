"""Create a small native geometry catalog for the approved City-Lite authority.

This is deliberately not a capture: it spawns no CF2X, targets, sensors,
video, or dataset payload.  Its only purpose is to bind conservative AABBs to
the exact native City-Lite composition used by ``isaac_capture`` before an
external private evaluator samples targets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .capture_lease import repository_app_launcher_lease
from .citylite_scene import (
    AABB,
    CityLiteAuthority,
    resolve_city_lite_authority,
    sha256_file,
)
from .isaac_capture import (
    _activate_local_isaaclab_source,
    _compose_city_lite_stage,
    _extract_structural_aabbs,
    _module_path_is_under,
    _repository_root,
    _windows_system_commit_snapshot,
)
from .private_evaluator_manifest import (
    NATIVE_GEOMETRY_SCAN_EVIDENCE_KIND,
    NATIVE_GEOMETRY_SCAN_GENERATOR,
    NATIVE_GEOMETRY_SCAN_SCHEMA,
    NATIVE_GEOMETRY_SCAN_TOOL_PATH,
    native_geometry_scan_sha256,
)
from .provenance import SourceProvenance, require_clean_source
from .resource_telemetry import (
    DEFAULT_ABORT_COMMIT_PERCENT,
    DEFAULT_PREFLIGHT_COMMIT_PERCENT,
)
from .runtime_lock import (
    audit_runtime_lock,
    load_runtime_lock,
    locked_launcher_kwargs,
    runtime_lock_sha256,
    validate_locked_launcher_environment,
)


class IsaacGeometryScanError(RuntimeError):
    """Raised when a native geometry catalog cannot be safely produced."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _validate_thresholds(preflight_percent: float, abort_percent: float) -> None:
    for name, value in (
        ("preflight_commit_percent", preflight_percent),
        ("abort_commit_percent", abort_percent),
    ):
        if not math.isfinite(float(value)) or not 1.0 <= float(value) < 100.0:
            raise IsaacGeometryScanError(f"{name} must be finite and in [1, 100)")
    if float(abort_percent) <= float(preflight_percent):
        raise IsaacGeometryScanError("abort_commit_percent must exceed preflight_commit_percent")


def _enforce_commit_guard(*, phase: str, threshold_percent: float) -> dict[str, float | int] | None:
    """Apply the same Windows process-creation ceiling as capture and smoke."""

    snapshot = _windows_system_commit_snapshot()
    if snapshot is None:
        return None
    percent = snapshot.get("commit_percent")
    if not isinstance(percent, (int, float)) or isinstance(percent, bool):
        raise IsaacGeometryScanError("Windows system-commit probe returned no finite percentage")
    if not math.isfinite(float(percent)):
        raise IsaacGeometryScanError("Windows system-commit probe returned a non-finite percentage")
    if float(percent) >= float(threshold_percent):
        raise IsaacGeometryScanError(
            "refusing native geometry scan because Windows system commit is "
            f"{float(percent):.2f}% at {phase} (limit {float(threshold_percent):.2f}%)"
        )
    return snapshot


def _domain_rows(structural_aabbs: Sequence[AABB]) -> list[dict[str, Any]]:
    rows = []
    for index, box in enumerate(structural_aabbs):
        rows.append(
            {
                "domain_id": f"rivermark-domain-{index:04d}",
                "structure_type": box.category,
                "aabb": {
                    "min": list(box.minimum),
                    "max": list(box.maximum),
                    "path": box.source_prim,
                    "source_kind": box.category,
                },
            }
        )
    return rows


def build_native_geometry_scan_payload(
    *,
    authority: CityLiteAuthority,
    structural_aabbs: Sequence[AABB],
    source: SourceProvenance,
    tool_sha256: str,
    stage_evidence: Mapping[str, Any],
    runtime_lock_digest: str,
    runtime_profile_id: str,
    system_commit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build one self-hashed scan payload from native composition evidence."""

    if source.source_worktree_dirty:
        raise IsaacGeometryScanError("native geometry scan requires a clean Git worktree")
    if not structural_aabbs:
        raise IsaacGeometryScanError("native geometry scan requires at least one structural AABB")
    for name, value in (("tool_sha256", tool_sha256), ("runtime_lock_sha256", runtime_lock_digest)):
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise IsaacGeometryScanError(f"{name} must be a lowercase SHA-256")
    if not isinstance(runtime_profile_id, str) or not runtime_profile_id:
        raise IsaacGeometryScanError("runtime_profile_id must be non-empty")
    active_count = stage_evidence.get("active_static_prim_count")
    collision_counts = stage_evidence.get("native_collision_counts")
    if not isinstance(active_count, int) or isinstance(active_count, bool) or active_count <= 0:
        raise IsaacGeometryScanError("native City-Lite composition did not report active static prims")
    if not isinstance(collision_counts, Mapping):
        raise IsaacGeometryScanError("native City-Lite composition did not report collision counts")

    payload: dict[str, Any] = {
        "schema": NATIVE_GEOMETRY_SCAN_SCHEMA,
        "status": "passed",
        "formal": False,
        "generator": NATIVE_GEOMETRY_SCAN_GENERATOR,
        "geometry_evidence_kind": NATIVE_GEOMETRY_SCAN_EVIDENCE_KIND,
        "tool_path": NATIVE_GEOMETRY_SCAN_TOOL_PATH,
        "tool_sha256": tool_sha256,
        "source_revision": source.source_revision,
        "source_tree_sha256": source.source_tree_sha256,
        "source_worktree_dirty": False,
        "scene_id": authority.provenance()["environment_id"],
        "scene_content_sha256": authority.contract_payload_sha256,
        "scene_contract_sha256": authority.contract_sha256,
        "runtime_lock": {
            "sha256": runtime_lock_digest,
            "profile_id": runtime_profile_id,
            "audit_status": "passed",
        },
        "domains": _domain_rows(structural_aabbs),
        "stage_evidence": {
            "active_static_prim_count": active_count,
            "native_collision_counts": dict(collision_counts),
            "structural_aabb_count": len(structural_aabbs),
            "selective_reference_destinations": [
                "/World/StaticScene/City/Rivermark",
                "/World/StaticScene/CityTaskObstacles",
            ],
        },
        "resource_guard": {
            "system_commit": dict(system_commit) if system_commit is not None else None,
        },
        "claim_boundary": {
            "native_city_lite_composed": True,
            "cf2x_spawned": False,
            "targets_spawned": False,
            "sensors_captured": False,
            "video_created": False,
            "formal_episode_created": False,
        },
    }
    payload["scan_sha256"] = native_geometry_scan_sha256(payload)
    return payload


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise IsaacGeometryScanError(f"refusing to overwrite existing scan artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_bytes(_canonical_json(payload))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_failure_if_new(path: Path, error: BaseException) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        return
    payload = {
        "schema": "org.rivermark.native-geometry-scan-failure.v1",
        "status": "failed",
        "formal": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    _write_new_json(destination, payload)


def _run_native_scan(args: argparse.Namespace, authority: CityLiteAuthority, source: SourceProvenance) -> dict[str, Any]:
    runtime_lock = load_runtime_lock(args.runtime_lock.expanduser().resolve())
    runtime_audit = audit_runtime_lock(
        args.runtime_lock.expanduser().resolve(),
        isaaclab_source=args.isaaclab_source.expanduser().resolve(),
        scene_contract=args.scene_contract.expanduser().resolve(),
        cf2x_usd=args.drone_usd.expanduser().resolve(),
    )
    if runtime_audit.get("status") != "passed":
        raise IsaacGeometryScanError("runtime lock audit failed")
    isaaclab_source = _activate_local_isaaclab_source(args.isaaclab_source)
    if isaaclab_source is None:
        raise IsaacGeometryScanError("locked native geometry scan could not activate IsaacLab")
    validate_locked_launcher_environment(runtime_lock)
    preflight_commit = _enforce_commit_guard(
        phase="before_app_launcher", threshold_percent=args.preflight_commit_percent
    )
    lease = repository_app_launcher_lease(
        _repository_root(),
        metadata={
            "owner": "rivermark_benchmark.isaac_geometry_scan",
            "output": str(args.output.expanduser().resolve()),
            "source_revision": source.source_revision,
        },
    )
    app: Any | None = None
    try:
        lease.acquire()
        from isaaclab.app import AppLauncher

        app = AppLauncher(locked_launcher_kwargs(runtime_lock, isaaclab_source)).app
        if not _module_path_is_under(__import__("isaaclab"), isaaclab_source / "isaaclab"):
            raise IsaacGeometryScanError("locked geometry scan imported IsaacLab from an unbound source")
        import omni.usd

        omni.usd.get_context().new_stage()
        stage, static_evidence = _compose_city_lite_stage(authority)
        structural_aabbs = _extract_structural_aabbs(stage)
        after_compose_commit = _enforce_commit_guard(
            phase="after_city_lite_composition", threshold_percent=args.abort_commit_percent
        )
        payload = build_native_geometry_scan_payload(
            authority=authority,
            structural_aabbs=structural_aabbs,
            source=source,
            tool_sha256=sha256_file(Path(__file__).resolve()),
            stage_evidence=static_evidence,
            runtime_lock_digest=runtime_lock_sha256(runtime_lock),
            runtime_profile_id=str(runtime_lock["profile_id"]),
            system_commit=after_compose_commit or preflight_commit,
        )
        # Kit shutdown can terminate its Python owner before the caller gets a
        # chance to persist a result.  The native evidence must therefore be
        # durable before closing the only AppLauncher.
        _write_new_json(args.output, payload)
        return payload
    finally:
        try:
            if app is not None:
                app.close(wait_for_replicator=False, skip_cleanup=True)
        finally:
            lease.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--isaaclab-source", type=Path, required=True)
    parser.add_argument(
        "--drone-usd",
        type=Path,
        required=True,
        help="CF2X USD hash-bound by the same runtime lock; it is not spawned by this scan.",
    )
    parser.add_argument(
        "--preflight-commit-percent", type=float, default=DEFAULT_PREFLIGHT_COMMIT_PERCENT
    )
    parser.add_argument(
        "--abort-commit-percent", type=float, default=DEFAULT_ABORT_COMMIT_PERCENT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_thresholds(args.preflight_commit_percent, args.abort_commit_percent)
        source = require_clean_source(_repository_root())
        authority = resolve_city_lite_authority(args.scene_contract)
        payload = _run_native_scan(args, authority, source)
    except (OSError, RuntimeError, ValueError, IsaacGeometryScanError) as error:
        try:
            _write_failure_if_new(args.output, error)
        except (OSError, RuntimeError):
            pass
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=True))
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "scan": str(args.output.expanduser().resolve()),
                "scan_sha256": payload["scan_sha256"],
                "structural_aabb_count": len(payload["domains"]),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
