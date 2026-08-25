"""License-lock validation and release-local asset staging."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .canonical import content_hash, file_hash, read_json, write_json
from .errors import AssetRegistryError

ACCEPTED_SPDX = frozenset({"CC0-1.0", "CC-BY-4.0", "CC-BY-3.0"})


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    bundle: str
    kind: str
    spdx: str
    role: str
    files: tuple[dict[str, Any], ...]

    @property
    def root_file(self) -> str:
        if not self.files:
            raise AssetRegistryError(f"asset {self.asset_id} has no registered files")
        for entry in self.files:
            suffix = PurePosixPath(str(entry["path"])).suffix.lower()
            if suffix in {".usd", ".usda", ".usdc"}:
                return str(entry["path"])
        return str(self.files[0]["path"])


@dataclass(frozen=True)
class AssetLock:
    bundle: str
    registry_hash: str
    records: dict[str, AssetRecord]


def _safe_file(bundle_root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts:
        raise AssetRegistryError(f"unsafe asset path: {relative}")
    candidate = bundle_root.joinpath(*posix.parts).resolve()
    try:
        candidate.relative_to(bundle_root.resolve())
    except ValueError as exc:
        raise AssetRegistryError(f"asset path escapes bundle: {relative}") from exc
    return candidate


def load_asset_lock(asset_root: Path, bundle: str, requested_ids: set[str]) -> AssetLock:
    bundle_root = (asset_root / bundle).resolve()
    registry_path = bundle_root / "ASSET_REGISTRY.json"
    if not registry_path.is_file():
        raise AssetRegistryError(
            f"missing {registry_path}; rebuild the source registry before using visual assets"
        )
    raw = read_json(registry_path)
    if not isinstance(raw, dict) or not isinstance(raw.get("assets"), list):
        raise AssetRegistryError(f"invalid asset registry: {registry_path}")
    records: dict[str, AssetRecord] = {}
    for node in raw["assets"]:
        if not isinstance(node, dict):
            continue
        asset_id = str(node.get("asset_id", ""))
        if asset_id not in requested_ids:
            continue
        spdx = str(node.get("spdx", ""))
        if spdx not in ACCEPTED_SPDX:
            raise AssetRegistryError(f"asset {asset_id} has inadmissible SPDX identifier {spdx}")
        if node.get("redistribution_allowed") is not True:
            raise AssetRegistryError(f"asset {asset_id} is not registered as redistributable")
        files = tuple(node.get("files", ()))
        for entry in files:
            relative = str(entry.get("path", ""))
            source = _safe_file(bundle_root, relative)
            if not source.is_file():
                raise AssetRegistryError(f"asset {asset_id} is missing {relative}")
            expected = str(entry.get("sha256", "")).lower()
            if len(expected) != 64 or file_hash(source) != expected:
                raise AssetRegistryError(f"asset {asset_id} failed SHA-256 validation: {relative}")
        records[asset_id] = AssetRecord(
            asset_id=asset_id,
            bundle=bundle,
            kind=str(node.get("kind", "unknown")),
            spdx=spdx,
            role=str(node.get("role", "visual_decoration")),
            files=files,
        )
    missing = sorted(requested_ids - records.keys())
    if missing:
        raise AssetRegistryError(f"requested visual assets are absent from the registry: {missing}")
    return AssetLock(bundle=bundle, registry_hash=file_hash(registry_path), records=records)


def stage_assets(lock: AssetLock, asset_root: Path, release_root: Path) -> dict[str, Any]:
    source_root = (asset_root / lock.bundle).resolve()
    destination_root = release_root / "_assets" / lock.bundle
    entries: list[dict[str, Any]] = []
    copied: set[str] = set()
    for asset_id in sorted(lock.records):
        record = lock.records[asset_id]
        for node in record.files:
            relative = str(node["path"])
            if relative not in copied:
                source = _safe_file(source_root, relative)
                destination = destination_root.joinpath(*PurePosixPath(relative).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                if file_hash(destination) != str(node["sha256"]).lower():
                    raise AssetRegistryError(f"staged asset changed while copying: {relative}")
                copied.add(relative)
        entries.append(
            {
                "asset_id": asset_id,
                "kind": record.kind,
                "role": record.role,
                "spdx": record.spdx,
                "root_file": record.root_file,
                "files": [dict(item) for item in record.files],
            }
        )
    manifest = {
        "schema": "org.aerocity.bench.asset-lock.v1",
        "bundle": lock.bundle,
        "source_registry_sha256": lock.registry_hash,
        "assets": entries,
    }
    manifest["asset_lock_hash"] = content_hash(manifest)
    write_json(release_root / "_assets" / "asset_lock.json", manifest)
    return manifest
