"""Generate an evidence-labelled manifest for one project run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EVIDENCE_LEVELS = frozenset({"synthetic", "gazebo_simulation", "bag_replay", "host_hardware"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_revision(path: str | Path | None) -> str:
    if not path:
        return "unknown"
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(path)), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _sha256(path: str | Path | None) -> str:
    if not path or not Path(path).is_file():
        return "unknown"
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_source_specs(specs: Iterable[str]) -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"source must use LABEL=PATH: {spec}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"source must use LABEL=PATH: {spec}")
        sources[label] = {"path": raw_path, "git_revision": _git_revision(raw_path)}
    return sources


def _parse_label_values(specs: Iterable[str], option_name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{option_name} must use LABEL=VALUE: {spec}")
        label, value = (part.strip() for part in spec.split("=", 1))
        if not label or not value:
            raise ValueError(f"{option_name} must use LABEL=VALUE: {spec}")
        values[label] = value
    return values


def build_manifest(
    *,
    run_id: str,
    evidence_level: str,
    workspace_path: str | Path | None = None,
    source_specs: Iterable[str] = (),
    source_revision_specs: Iterable[str] = (),
    game_profile: str = "TBD",
    rule_revision: str = "TBD",
    map_path: str | Path | None = None,
    map_id: str = "TBD",
    map_version: str = "TBD",
    source_bag: str = "TBD",
    artifacts: dict[str, str] | None = None,
    hardware: dict[str, str] | None = None,
    network: dict[str, str] | None = None,
    calibration: dict[str, str] | None = None,
    metrics: dict[str, str | float | int] | None = None,
    started_at_utc: str | None = None,
) -> dict[str, object]:
    """Build a manifest without turning missing evidence into measurements."""
    if evidence_level not in EVIDENCE_LEVELS:
        allowed = ", ".join(sorted(EVIDENCE_LEVELS))
        raise ValueError(f"evidence_level must be one of: {allowed}")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")

    sources = _parse_source_specs(source_specs)
    explicit_revisions = _parse_label_values(source_revision_specs, "source revision")
    unknown_labels = set(explicit_revisions).difference(sources)
    if unknown_labels:
        raise ValueError(f"source revision has no matching source: {sorted(unknown_labels)[0]}")
    for label, revision in explicit_revisions.items():
        sources[label]["git_revision"] = revision

    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": started_at_utc or _utc_now(),
        "generated_at_utc": _utc_now(),
        "evidence_level": evidence_level,
        "evidence_policy": {
            "declared_by_operator": True,
            "generator_verifies_metadata_only": True,
            "hardware_and_accuracy_claims_require_external_evidence": True,
        },
        "software": {
            "workspace_path": str(workspace_path) if workspace_path else "TBD",
            "workspace_commit": _git_revision(workspace_path),
            "sources": sources,
            "lio_frontend": "FAST-LIO2",
        },
        "profiles": {
            "localization": "competition",
            "game_profile": game_profile,
            "rule_revision": rule_revision,
        },
        "hardware": {
            "target_computer_product": "TBD",
            "target_computer_kernel": "TBD",
            "lidar_model": "Livox MID-360",
            "lidar_firmware": "TBD",
            "lidar_serial": "TBD",
            **(hardware or {}),
        },
        "network": {
            "target_computer_interface": "TBD",
            "target_computer_ip": "TBD",
            "lidar_ip": "TBD",
            **(network or {}),
        },
        "calibration": {
            "base_to_imu_version": "TBD",
            "imu_to_lidar_version": "TBD",
            "time_offset_version": "TBD",
            "calibration_evidence": "TBD",
            **(calibration or {}),
        },
        "map": {
            "map_id": map_id,
            "map_version": map_version,
            "frame": "map",
            "source_bag": source_bag,
            "path": str(map_path) if map_path else "TBD",
            "sha256": _sha256(map_path),
        },
        "metrics": {
            "pose_rate_hz": "unknown",
            "pose_age_p95_ms": "unknown",
            "position_error_m": "unknown",
            "yaw_error_deg": "unknown",
            "relative_drift_m": "unknown",
            "relative_drift_deg": "unknown",
            **(metrics or {}),
        },
        "artifacts": {
            "bag_path": source_bag,
            "diagnostics_path": "TBD",
            "plot_directory": "TBD",
            **(artifacts or {}),
        },
    }


def write_manifest(path: str | Path, manifest: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=True, indent=2)
        stream.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSON manifest output path")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evidence-level", required=True, choices=sorted(EVIDENCE_LEVELS))
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--source", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument(
        "--source-revision",
        action="append",
        default=[],
        metavar="LABEL=REVISION",
        help="Explicit revision for a vendored source that has no .git directory",
    )
    parser.add_argument("--game-profile", default="TBD")
    parser.add_argument("--rule-revision", default="TBD")
    parser.add_argument("--map", dest="map_path", default=None)
    parser.add_argument("--map-id", default="TBD")
    parser.add_argument("--map-version", default="TBD")
    parser.add_argument("--source-bag", default="TBD")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_manifest(
        run_id=args.run_id,
        evidence_level=args.evidence_level,
        workspace_path=args.workspace,
        source_specs=args.source,
        source_revision_specs=args.source_revision,
        game_profile=args.game_profile,
        rule_revision=args.rule_revision,
        map_path=args.map_path,
        map_id=args.map_id,
        map_version=args.map_version,
        source_bag=args.source_bag,
    )
    write_manifest(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
