"""Aggregate the independent checks required before a public release.

This module is deliberately a control-plane report.  It does not admit an
episode, upload bytes, contact a URL, or grant a license.  It combines the
existing formal-dataset, release-manifest, supply-chain, and Git-index audits
so a maintainer cannot mistake one passing subsystem for a release decision.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .formal_dataset import verify_dataset_integrity
from .release_manifest import (
    ReleaseManifestError,
    load_release_manifest,
    verify_shard_file,
)
from .repository_audit import audit_repository
from .supply_chain import SupplyChainError, verify_supply_chain_manifest


READINESS_SCHEMA = "org.rivermark.benchmark.release-readiness.v1"


@dataclass(frozen=True)
class ReleaseReadinessIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ReleaseReadinessReport:
    dataset_root: Path
    release_manifest: Path
    supply_chain_manifest: Path
    minimum_episodes: int
    episode_count: int
    shard_count: int
    checks: Mapping[str, str]
    issues: tuple[ReleaseReadinessIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def _issue(issues: list[ReleaseReadinessIssue], code: str, path: str, message: str) -> None:
    issues.append(ReleaseReadinessIssue(code, path, message))


def _add_prefixed(
    issues: list[ReleaseReadinessIssue],
    source: Any,
    *,
    prefix: str,
) -> None:
    for item in getattr(source, "issues", ()):
        _issue(
            issues,
            str(getattr(item, "code", "invalid")),
            f"{prefix}.{getattr(item, 'path', '$')}",
            str(getattr(item, "message", "check failed")),
        )


def _read_json(path: Path, *, issues: list[ReleaseReadinessIssue], label: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _issue(issues, f"{label}_read", str(path), str(exc))
        return None
    if not isinstance(value, Mapping):
        _issue(issues, f"{label}_type", str(path), "expected a JSON object")
        return None
    return value


def _verify_local_release_bytes(
    dataset_root: Path,
    payload: Mapping[str, Any],
    *,
    issues: list[ReleaseReadinessIssue],
) -> int:
    """Verify every locally addressable byte named by a release manifest."""

    shard_count = 0
    entries: list[tuple[str, Mapping[str, Any]]] = []
    raw_shards = payload.get("shards")
    if isinstance(raw_shards, list):
        entries.extend((f"$.shards[{index}]", shard) for index, shard in enumerate(raw_shards) if isinstance(shard, Mapping))
    accounting = payload.get("accounting")
    ledger = accounting.get("failure_ledger") if isinstance(accounting, Mapping) else None
    if isinstance(ledger, Mapping):
        entries.append(("$.accounting.failure_ledger", ledger))

    root = dataset_root.resolve()
    for path, entry in entries:
        shard_count += 1
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            _issue(issues, "local_payload_path", f"{path}.path", "release entry has no local relative path")
            continue
        candidate = (root / Path(*relative.replace("\\", "/").split("/"))).resolve()
        if candidate != root and root not in candidate.parents:
            _issue(issues, "local_payload_escape", f"{path}.path", "release entry escapes dataset root")
            continue
        if not candidate.is_file():
            _issue(issues, "local_payload_missing", f"{path}.path", f"payload is not present locally: {relative}")
            continue
        try:
            # The accounting ledger is a release entry but intentionally has
            # no public shard_id.  Give the shared byte verifier an internal
            # diagnostic label without changing the manifest contract.
            verify_shard_file(
                {"shard_id": entry.get("shard_id", path), **dict(entry)},
                candidate,
            )
        except Exception as exc:  # keep the report complete and fail closed
            _issue(issues, "local_payload_hash", f"{path}.path", str(exc))
    return shard_count


def audit_release_readiness(
    dataset_root: Path,
    release_manifest: Path,
    supply_chain_manifest: Path,
    *,
    source_root: Path | None = None,
    minimum_episodes: int = 1,
    require_https: bool = True,
) -> ReleaseReadinessReport:
    """Return one fail-closed report for the public-release boundary."""

    if minimum_episodes < 1:
        raise ValueError("minimum_episodes must be at least one")
    root = dataset_root.resolve()
    release_path = release_manifest.resolve()
    supply_path = supply_chain_manifest.resolve()
    issues: list[ReleaseReadinessIssue] = []
    checks: dict[str, str] = {}

    integrity = verify_dataset_integrity(root)
    _add_prefixed(issues, integrity, prefix="formal_dataset")
    episode_count = int(getattr(integrity, "episode_count", 0))
    if episode_count < minimum_episodes:
        _issue(
            issues,
            "minimum_episode_count",
            "manifests/dataset_index.json",
            f"public release requires at least {minimum_episodes} episode(s), found {episode_count}",
        )
    checks["formal_dataset"] = "passed" if not getattr(integrity, "issues", ()) and episode_count >= minimum_episodes else "failed"

    release_payload: Mapping[str, Any] | None = None
    try:
        release_payload = load_release_manifest(release_path, require_https=require_https)
    except (ReleaseManifestError, OSError, UnicodeDecodeError) as exc:
        _issue(issues, "release_manifest", str(release_path), str(exc))
    checks["release_manifest"] = "passed" if release_payload is not None else "failed"

    try:
        supply_report = verify_supply_chain_manifest(supply_path, require_release=True)
    except (SupplyChainError, OSError, UnicodeDecodeError) as exc:
        supply_report = {
            "status": "invalid",
            "issues": [{"code": "manifest_read", "path": str(supply_path), "message": str(exc)}],
        }
    for item in supply_report.get("issues", ()):
        if isinstance(item, Mapping):
            _issue(issues, str(item.get("code", "invalid")), f"supply_chain.{item.get('path', '$')}", str(item.get("message", "check failed")))
    checks["supply_chain"] = "passed" if supply_report.get("status") == "valid" else "failed"

    shard_count = 0
    if release_payload is not None:
        if not isinstance(release_payload.get("source_revision"), str) or not release_payload.get("source_revision"):
            _issue(issues, "source_revision_required", "$.source_revision", "public release must bind a source revision")
        expected_supply_hash = release_payload.get("supply_chain_manifest_sha256")
        actual_supply_hash = supply_report.get("manifest_sha256")
        if expected_supply_hash != actual_supply_hash:
            _issue(
                issues,
                "supply_chain_binding",
                "$.supply_chain_manifest_sha256",
                "release manifest does not bind the supplied supply-chain manifest",
            )
        if release_payload.get("release_id") != supply_report.get("release_id"):
            _issue(issues, "release_id_binding", "$.release_id", "release and supply-chain release IDs differ")
        index = _read_json(root / "manifests" / "dataset_index.json", issues=issues, label="dataset_index")
        if isinstance(index, Mapping) and index.get("episode_count", 0) > 0:
            if release_payload.get("dataset_version") != index.get("dataset_version"):
                _issue(issues, "dataset_version_binding", "$.dataset_version", "release version differs from formal dataset index")
        shard_count = _verify_local_release_bytes(root, release_payload, issues=issues)
    checks["release_bindings"] = "passed" if release_payload is not None and not any(
        issue.code in {"source_revision_required", "supply_chain_binding", "release_id_binding", "dataset_version_binding", "local_payload_path", "local_payload_escape", "local_payload_missing", "local_payload_hash"}
        for issue in issues
    ) else "failed"

    if source_root is not None:
        repository = audit_repository(source_root.resolve())
        _add_prefixed(issues, repository, prefix="repository_audit")
        checks["repository_audit"] = "passed" if not repository.issues else "failed"
    else:
        checks["repository_audit"] = "not_requested"

    return ReleaseReadinessReport(
        dataset_root=root,
        release_manifest=release_path,
        supply_chain_manifest=supply_path,
        minimum_episodes=minimum_episodes,
        episode_count=episode_count,
        shard_count=shard_count,
        checks=checks,
        issues=tuple(issues),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("release_manifest", type=Path)
    parser.add_argument("supply_chain_manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--minimum-episodes", type=int, default=1)
    parser.add_argument("--allow-non-https", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = audit_release_readiness(
            args.dataset_root,
            args.release_manifest,
            args.supply_chain_manifest,
            source_root=args.source_root,
            minimum_episodes=args.minimum_episodes,
            require_https=not args.allow_non_https,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema": READINESS_SCHEMA, "status": "invalid", "error": str(exc)}, indent=2))
        return 1
    payload = {
        "schema": READINESS_SCHEMA,
        "status": "passed" if report.valid else "failed",
        "dataset_root": str(report.dataset_root),
        "release_manifest": str(report.release_manifest),
        "supply_chain_manifest": str(report.supply_chain_manifest),
        "minimum_episodes": report.minimum_episodes,
        "episode_count": report.episode_count,
        "shard_count": report.shard_count,
        "checks": dict(report.checks),
        "issues": [asdict(issue) for issue in report.issues],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
