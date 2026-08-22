"""Fail-closed CC0 asset and release provenance gates for ordinary-v3."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .assets import AssetLock, AssetRecord
from .canonical import content_hash, file_hash, read_json, write_json
from .errors import AssetRegistryError, ValidationError

SAFE_BUNDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FORBIDDEN_SOURCE_TOKENS = (
    "nvidia",
    "nucleus",
    "omniverse://",
    "isaac sim",
    "isaacsim_assets",
    "rivermark",
    "official_isaacsim_assets",
)
USD_REMOTE_PREFIXES = ("omniverse://", "http://", "https://", "file://")
USD_REFERENCE_PATTERN = re.compile(r"@([^@]+)@")


@dataclass(frozen=True)
class ProvenanceEvidence:
    manifest_path: Path
    manifest_hash: str
    asset_creators: dict[str, tuple[str, ...]]
    asset_official_evidence: dict[str, dict[str, dict[str, Any]]]
    license_snapshot_path: Path
    license_snapshot_hash: str


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative.replace("\\", "/").replace("//", "/"))
    if posix.is_absolute() or ".." in posix.parts:
        raise AssetRegistryError(f"unsafe registered asset path: {relative}")
    candidate = root.joinpath(*posix.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AssetRegistryError(f"registered asset escapes bundle root: {relative}") from exc
    return candidate


def validate_bundle_root(asset_root: Path, bundle: str) -> Path:
    if not SAFE_BUNDLE.fullmatch(bundle):
        raise AssetRegistryError("asset bundle must be a simple controlled name")
    root = asset_root.resolve()
    bundle_root = (root / bundle).resolve()
    try:
        bundle_root.relative_to(root)
    except ValueError as exc:
        raise AssetRegistryError("asset bundle escapes the declared asset root") from exc
    normalized = str(bundle_root).lower()
    if any(token in normalized for token in FORBIDDEN_SOURCE_TOKENS):
        raise AssetRegistryError("asset bundle path matches a prohibited source token")
    return bundle_root


def _resolve_evidence_path(bundle_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        candidate = path.resolve()
    else:
        candidate = (bundle_root / path).resolve()
    try:
        candidate.relative_to(bundle_root)
    except ValueError as exc:
        raise AssetRegistryError(f"evidence path escapes bundle: {value}") from exc
    return candidate


def load_provenance_evidence(bundle_root: Path, required_ids: set[str]) -> ProvenanceEvidence:
    manifest_path = bundle_root / "provenance" / "PROVENANCE_MANIFEST.json"
    hash_path = bundle_root / "provenance" / "PROVENANCE_MANIFEST.sha256"
    if not manifest_path.is_file() or not hash_path.is_file():
        raise AssetRegistryError("OFFICIAL assets require a captured provenance manifest and hash")
    expected_hash = hash_path.read_text(encoding="ascii").split()[0].lower()
    if file_hash(manifest_path) != expected_hash:
        raise AssetRegistryError("asset provenance manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("failures"):
        raise AssetRegistryError("asset provenance capture contains unresolved failures")
    summary = manifest.get("verification_summary", {})
    if summary.get("all_file_and_url_checks_passed") is not True:
        raise AssetRegistryError("asset provenance file/URL verification did not pass")
    creator_map: dict[str, tuple[str, ...]] = {}
    official_evidence_map: dict[str, dict[str, dict[str, Any]]] = {}
    for asset in manifest.get("assets", []):
        asset_id = str(asset.get("asset_id", ""))
        if asset_id not in required_ids:
            continue
        creators = tuple(str(name) for name in asset.get("creator_names", []) if str(name))
        if creators:
            creator_map[asset_id] = creators
        official = asset.get("official_evidence", {})
        normalized: dict[str, dict[str, Any]] = {}
        expected_urls = {
            "source_page": f"https://polyhaven.com/a/{asset_id}",
            "info_api": f"https://api.polyhaven.com/info/{asset_id}",
            "files_api": f"https://api.polyhaven.com/files/{asset_id}",
        }
        for evidence_name, expected_url in expected_urls.items():
            record = official.get(evidence_name, {})
            snapshot_value = str(record.get("snapshot_path", ""))
            snapshot_hash = str(record.get("sha256", "")).lower()
            requested_url = str(record.get("requested_url", ""))
            retrieved_at = str(record.get("retrieved_at_utc", ""))
            if (
                int(record.get("http_status", 0)) != 200
                or requested_url != expected_url
                or not retrieved_at
                or len(snapshot_hash) != 64
                or not snapshot_value
            ):
                raise AssetRegistryError(
                    f"asset lacks complete official {evidence_name} evidence: {asset_id}"
                )
            snapshot_path = _resolve_evidence_path(bundle_root, snapshot_value)
            if not snapshot_path.is_file() or file_hash(snapshot_path) != snapshot_hash:
                raise AssetRegistryError(
                    f"asset official {evidence_name} snapshot failed hash validation: {asset_id}"
                )
            normalized[evidence_name] = {
                "path": snapshot_path,
                "sha256": snapshot_hash,
                "requested_url": requested_url,
                "retrieved_at_utc": retrieved_at,
            }
        official_evidence_map[asset_id] = normalized
    missing_creators = sorted(required_ids - set(creator_map))
    if missing_creators:
        raise AssetRegistryError(f"assets lack official creator evidence: {missing_creators}")
    license_record = manifest.get("global_evidence", {}).get("polyhaven_license", {})
    snapshot_value = str(license_record.get("snapshot_path", ""))
    license_hash = str(license_record.get("sha256", "")).lower()
    if not snapshot_value or len(license_hash) != 64:
        raise AssetRegistryError("Poly Haven license snapshot evidence is incomplete")
    snapshot_path = _resolve_evidence_path(bundle_root, snapshot_value)
    if not snapshot_path.is_file() or file_hash(snapshot_path) != license_hash:
        raise AssetRegistryError("Poly Haven license snapshot failed SHA-256 validation")
    return ProvenanceEvidence(
        manifest_path=manifest_path,
        manifest_hash=expected_hash,
        asset_creators=creator_map,
        asset_official_evidence=official_evidence_map,
        license_snapshot_path=snapshot_path,
        license_snapshot_hash=license_hash,
    )


def _usd_dependencies(path: Path) -> tuple[list[Path], list[str]]:
    if path.suffix.lower() in {".usdc", ".usd"}:
        # Binary USD dependency traversal requires pxr; fail closed rather than guessing.
        try:
            from pxr import UsdUtils  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AssetRegistryError(
                f"cannot close binary USD dependencies without pxr: {path.name}"
            ) from exc
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
        if unresolved:
            return [], [str(item) for item in unresolved]
        dependencies = [Path(str(item.realPath or item.identifier)).resolve() for item in layers]
        dependencies.extend(Path(str(item)).resolve() for item in assets)
        return dependencies, []
    text = path.read_text(encoding="utf-8", errors="strict")
    dependencies: list[Path] = []
    unresolved: list[str] = []
    for reference in USD_REFERENCE_PATTERN.findall(text):
        normalized = reference.replace("\\", "/")
        if normalized.lower().startswith(USD_REMOTE_PREFIXES):
            unresolved.append(reference)
        else:
            dependencies.append((path.parent / PurePosixPath(normalized)).resolve())
    return dependencies, unresolved


def validate_usd_dependency_closure(
    bundle_root: Path, registered_files: set[str], root_files: Iterable[str]
) -> dict[str, Any]:
    registered_paths = {
        _safe_child(bundle_root, relative).resolve(): relative for relative in registered_files
    }
    checked: set[Path] = set()
    unresolved: list[dict[str, str]] = []
    remote: list[dict[str, str]] = []
    queue = [_safe_child(bundle_root, relative).resolve() for relative in root_files]
    while queue:
        current = queue.pop()
        if current in checked:
            continue
        checked.add(current)
        if current.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            continue
        dependencies, unresolved_references = _usd_dependencies(current)
        unresolved.extend(
            {"source": str(current), "reference": reference} for reference in unresolved_references
        )
        for referenced in dependencies:
            if referenced not in registered_paths or not referenced.is_file():
                unresolved.append({"source": str(current), "reference": str(referenced)})
                continue
            queue.append(referenced)
    if remote or unresolved:
        raise AssetRegistryError(
            f"USD dependency closure failed; remote={len(remote)}, unresolved={len(unresolved)}"
        )
    return {
        "checked_usd_layers": len(checked),
        "remote_dependencies": 0,
        "unresolved_dependencies": 0,
    }


def load_official_cc0_lock(
    asset_root: Path, bundle: str, allowlist: list[str]
) -> tuple[AssetLock, ProvenanceEvidence, dict[str, Any]]:
    bundle_root = validate_bundle_root(asset_root, bundle)
    registry_path = bundle_root / "ASSET_REGISTRY.json"
    if not registry_path.is_file():
        raise AssetRegistryError(f"missing official asset registry: {registry_path}")
    registry = read_json(registry_path)
    requested = set(allowlist)
    if len(requested) != len(allowlist) or not requested:
        raise AssetRegistryError("official allowlist must be non-empty and unique")
    records: dict[str, AssetRecord] = {}
    registered_files: set[str] = set()
    root_files: list[str] = []
    for node in registry.get("assets", []):
        asset_id = str(node.get("asset_id", ""))
        if asset_id not in requested:
            continue
        if str(node.get("spdx")) != "CC0-1.0":
            raise AssetRegistryError(f"official asset is not CC0-1.0: {asset_id}")
        if node.get("redistribution_allowed") is not True:
            raise AssetRegistryError(f"official asset is not redistributable: {asset_id}")
        source_text = json.dumps(node, ensure_ascii=False).lower()
        if any(token in source_text for token in FORBIDDEN_SOURCE_TOKENS):
            raise AssetRegistryError(
                f"official asset metadata matches prohibited source: {asset_id}"
            )
        files = tuple(dict(item) for item in node.get("files", []))
        if not files:
            raise AssetRegistryError(f"official asset has no registered files: {asset_id}")
        for entry in files:
            relative = str(entry.get("path", ""))
            source = _safe_child(bundle_root, relative)
            expected_hash = str(entry.get("sha256", "")).lower()
            if (
                not source.is_file()
                or len(expected_hash) != 64
                or file_hash(source) != expected_hash
            ):
                raise AssetRegistryError(f"official asset file/hash failure: {asset_id}/{relative}")
            registered_files.add(relative)
        record = AssetRecord(
            asset_id=asset_id,
            bundle=bundle,
            kind=str(node.get("kind", "unknown")),
            spdx="CC0-1.0",
            role=str(node.get("role", "visual_decoration")),
            files=files,
        )
        records[asset_id] = record
        root_files.append(record.root_file)
    missing = sorted(requested - set(records))
    if missing:
        raise AssetRegistryError(f"official allowlist IDs are absent: {missing}")
    evidence = load_provenance_evidence(bundle_root, requested)
    closure = validate_usd_dependency_closure(bundle_root, registered_files, root_files)
    lock = AssetLock(bundle=bundle, registry_hash=file_hash(registry_path), records=records)
    return lock, evidence, closure


def write_release_legal_materials(
    release_root: Path,
    lock: AssetLock,
    evidence: ProvenanceEvidence,
    closure: dict[str, Any],
    *,
    project_version: str,
    source_commit: str,
) -> dict[str, Any]:
    licenses = release_root / "LICENSES"
    licenses.mkdir(parents=True, exist_ok=True)
    license_destination = licenses / "Poly-Haven-CC0-license-snapshot.html"
    license_destination.write_bytes(evidence.license_snapshot_path.read_bytes())
    records = []
    for asset_id in sorted(lock.records):
        record = lock.records[asset_id]
        packaged_evidence: dict[str, dict[str, Any]] = {}
        for evidence_name, evidence_record in evidence.asset_official_evidence[asset_id].items():
            source_path = Path(evidence_record["path"])
            suffix = source_path.suffix.lower() or ".snapshot"
            destination = licenses / "provenance" / asset_id / f"{evidence_name}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source_path.read_bytes())
            packaged_evidence[evidence_name] = {
                "path": destination.relative_to(release_root).as_posix(),
                "sha256": evidence_record["sha256"],
                "source_url": evidence_record["requested_url"],
                "retrieved_at_utc": evidence_record["retrieved_at_utc"],
            }
        records.append(
            {
                "asset_id": asset_id,
                "spdx": record.spdx,
                "creators": list(evidence.asset_creators[asset_id]),
                "role": record.role,
                "files": [dict(item) for item in record.files],
                "official_evidence": packaged_evidence,
            }
        )
    asset_bom = {
        "schema": "org.aerocity.bench.asset-bom.v1",
        "bundle": lock.bundle,
        "registry_sha256": lock.registry_hash,
        "provenance_manifest_sha256": evidence.manifest_hash,
        "license_snapshot_sha256": evidence.license_snapshot_hash,
        "dependency_closure": closure,
        "assets": records,
    }
    asset_bom["asset_bom_hash"] = content_hash(asset_bom)
    write_json(release_root / "ASSET_BOM.json", asset_bom)
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "aerocity-bench-release",
                "version": project_version,
                "properties": [
                    {"name": "aerocity:source-commit", "value": source_commit},
                    {"name": "aerocity:asset-bom-hash", "value": asset_bom["asset_bom_hash"]},
                ],
            }
        },
        "components": [
            {
                "type": "file",
                "name": record["asset_id"],
                "licenses": [{"license": {"id": "CC0-1.0"}}],
            }
            for record in records
        ],
    }
    write_json(release_root / "SBOM.cdx.json", sbom)
    notice_lines = [
        "# Third-party notices",
        "",
        "AeroCityBench official assets in this release are CC0-1.0 Poly Haven assets.",
        "Attribution is not required by CC0; creator names are preserved for provenance.",
        "The NVIDIA Isaac Sim runtime and NVIDIA/Nucleus content are not redistributed here.",
        "GPL baselines are not included in this BSD core or release data package.",
        "",
        "## Assets",
        "",
    ]
    for record in records:
        notice_lines.append(
            f"- `{record['asset_id']}`; creators: {', '.join(record['creators'])}; "
            "license: CC0-1.0"
        )
    (release_root / "THIRD_PARTY_NOTICES.md").write_text(
        "\n".join(notice_lines) + "\n", encoding="utf-8", newline="\n"
    )
    (release_root / "DATA_LICENSE.md").write_text(
        "# Data license\n\n"
        "AeroCityBench-authored generated metadata and primitive geometry are released under "
        "CC BY 4.0. Third-party asset files retain CC0-1.0 as recorded in ASSET_BOM.json. "
        "Software remains under the repository BSD-3-Clause license. The Isaac Sim runtime "
        "and NVIDIA content are not included.\n",
        encoding="utf-8",
        newline="\n",
    )
    legal_manifest = {
        "schema": "org.aerocity.bench.legal-manifest.v1",
        "data_license": "CC-BY-4.0",
        "software_license": "BSD-3-Clause",
        "asset_policy": "CC0-only",
        "nvidia_content_redistributed": False,
        "gpl_source_in_core": False,
        "asset_bom_hash": asset_bom["asset_bom_hash"],
        "source_commit": source_commit,
    }
    legal_manifest["legal_manifest_hash"] = content_hash(legal_manifest)
    write_json(release_root / "LEGAL_MANIFEST.json", legal_manifest)
    return legal_manifest


def validate_release_legal_materials(release_root: Path) -> dict[str, Any]:
    required = (
        "ASSET_BOM.json",
        "SBOM.cdx.json",
        "THIRD_PARTY_NOTICES.md",
        "DATA_LICENSE.md",
        "LEGAL_MANIFEST.json",
        "LICENSES/Poly-Haven-CC0-license-snapshot.html",
    )
    missing = [name for name in required if not (release_root / name).is_file()]
    if missing:
        raise ValidationError(f"release legal materials are missing: {missing}")
    manifest = read_json(release_root / "LEGAL_MANIFEST.json")
    expected_hash = str(manifest.pop("legal_manifest_hash", ""))
    if content_hash(manifest) != expected_hash:
        raise ValidationError("LEGAL_MANIFEST hash mismatch")
    if (
        manifest.get("asset_policy") != "CC0-only"
        or manifest.get("nvidia_content_redistributed") is not False
        or manifest.get("gpl_source_in_core") is not False
    ):
        raise ValidationError("release legal policy is not the ordinary-v3 fail-closed policy")
    asset_bom = read_json(release_root / "ASSET_BOM.json")
    asset_bom_payload = dict(asset_bom)
    asset_bom_hash = str(asset_bom_payload.pop("asset_bom_hash", ""))
    if content_hash(asset_bom_payload) != asset_bom_hash:
        raise ValidationError("ASSET_BOM hash mismatch")
    if asset_bom_hash != manifest.get("asset_bom_hash"):
        raise ValidationError("ASSET_BOM differs from LEGAL_MANIFEST")
    license_snapshot = release_root / "LICENSES" / "Poly-Haven-CC0-license-snapshot.html"
    if file_hash(license_snapshot) != asset_bom.get("license_snapshot_sha256"):
        raise ValidationError("packaged Poly Haven license snapshot hash mismatch")
    if any(asset.get("spdx") != "CC0-1.0" for asset in asset_bom.get("assets", [])):
        raise ValidationError("ASSET_BOM contains a non-CC0 official asset")
    bundle = str(asset_bom.get("bundle", ""))
    for asset in asset_bom.get("assets", []):
        for file_record in asset.get("files", []):
            relative = PurePosixPath(str(file_record.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValidationError("ASSET_BOM contains an unsafe asset path")
            staged = release_root / "_assets" / bundle / Path(*relative.parts)
            if not staged.is_file() or file_hash(staged) != file_record.get("sha256"):
                raise ValidationError(f"ASSET_BOM staged file differs: {relative.as_posix()}")
        official = asset.get("official_evidence", {})
        if set(official) != {"source_page", "info_api", "files_api"}:
            raise ValidationError("ASSET_BOM lacks complete official source evidence")
        for record in official.values():
            relative = PurePosixPath(str(record.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValidationError("ASSET_BOM contains an unsafe evidence path")
            evidence_path = release_root.joinpath(*relative.parts)
            if not evidence_path.is_file() or file_hash(evidence_path) != record.get("sha256"):
                raise ValidationError("packaged official asset evidence hash mismatch")
    sbom = read_json(release_root / "SBOM.cdx.json")
    expected_components = {str(asset["asset_id"]) for asset in asset_bom.get("assets", [])}
    observed_components = {str(component.get("name")) for component in sbom.get("components", [])}
    if observed_components != expected_components:
        raise ValidationError("software/data BOM asset component sets differ")
    return {
        "status": "PASS",
        "asset_count": len(asset_bom.get("assets", [])),
        "legal_manifest_hash": expected_hash,
        "asset_bom_hash": asset_bom_hash,
        "source_commit": manifest.get("source_commit"),
    }
