"""Bounded, Isaac-free provenance hints for local OpenUSD assets.

The scanner is deliberately evidence-only.  It fingerprints a USD file and
looks for plain-text external-reference markers such as NVIDIA Nucleus paths
inside ASCII or binary USDC metadata.  A clean scan never proves a copyright
or redistribution grant; a detected reference is a reproducible reason to
keep the asset user-installed and outside a public release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ASSET_PROVENANCE_SCHEMA = "org.rivermark.benchmark.asset-provenance.v1"
LOCAL_ASSETS_SCHEMA = "org.rivermark.local-assets.v1"
DEFAULT_MAX_SCAN_BYTES = 64 * 1024 * 1024
_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_REQUIRED = frozenset(
    {
        "schema",
        "asset_root",
        "asset_package_id",
        "asset_package_version",
        "asset_package_manifest",
        "asset_package_sha256",
        "isaaclab_root",
        "isaac_python",
        "city_lite_contract",
        "city_lite_contract_sha256",
        "city_lite_layer",
        "city_lite_layer_sha256",
        "cf2x_usd",
        "cf2x_usd_sha256",
        "cf2x_source_provenance",
        "license_status",
        "public_redistribution",
    }
)
_PUBLIC_REDISTRIBUTION_KEYS = frozenset(
    {
        "raw_nvidia_assets",
        "cf2x_usd",
        "city_lite_composed_layer",
        "rendered_video",
        "derived_sensor_payload",
    }
)

# USDC stores metadata strings in a binary table, so these patterns operate on
# bytes rather than assuming a textual USDA file.  They intentionally cover
# only recognizable external-runtime markers and do not attempt to parse USD.
_REFERENCE_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "nvidia_isaac_nucleus",
        re.compile(rb"isaac-dev\.ov\.nvidia\.com[^\x00\s\"']*", re.IGNORECASE),
    ),
    (
        "nvidia_dsready_content",
        re.compile(rb"dsready_content[^\x00\s\"']*", re.IGNORECASE),
    ),
    (
        "nvidia_dsready_marker",
        # USDC may split a dictionary string with binary table bytes.  The
        # stable marker still identifies the NVIDIA dsready content package.
        re.compile(rb"dsready", re.IGNORECASE),
    ),
    (
        "nvidia_nv_content",
        re.compile(rb"(?:^|[/\\])nv_content[^\x00\s\"']*", re.IGNORECASE),
    ),
    (
        "omniverse_uri",
        re.compile(rb"omniverse://[^\x00\s\"']*", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class AssetProvenanceReport:
    """Reproducible file facts and external-reference hints."""

    schema: str
    path: str
    size_bytes: int
    sha256: str
    usd_format: str
    scan_limit_bytes: int
    scan_complete: bool
    references: tuple[dict[str, str], ...]
    classification: str
    license_status: str

    @property
    def has_external_references(self) -> bool:
        return bool(self.references)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["references"] = [dict(reference) for reference in self.references]
        payload["has_external_references"] = self.has_external_references
        return payload


class AssetProvenanceError(ValueError):
    """Raised when a local asset cannot be fingerprinted safely."""


@dataclass(frozen=True)
class LocalAssetAuditIssue:
    code: str
    path: str
    message: str


def _usd_format(path: Path, header: bytes) -> str:
    if header.startswith(b"#usda"):
        return "usda"
    if header.startswith(b"PXR-USDC"):
        return "usdc"
    return path.suffix.casefold().lstrip(".") or "unknown"


def _decode_reference(value: bytes) -> str:
    # USD metadata is normally ASCII/UTF-8.  Replacement decoding keeps the
    # report deterministic if a binary table contains a non-UTF-8 byte.
    decoded = value.decode("utf-8", errors="replace")
    printable = []
    for character in decoded:
        if ord(character) < 32 and character not in "\t":
            break
        printable.append(character)
    return "".join(printable)[:1024]


def inspect_usd(path: Path, *, max_scan_bytes: int = DEFAULT_MAX_SCAN_BYTES) -> AssetProvenanceReport:
    """Hash and scan one USD file without importing Isaac or OpenUSD.

    The file is streamed in bounded chunks.  Hashing covers the complete file;
    reference scanning is complete only when the whole file fits the declared
    scan budget.  A truncated scan is never treated as proof of self-containment.
    """

    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.casefold() not in _USD_SUFFIXES:
        raise AssetProvenanceError(f"unsupported OpenUSD suffix: {resolved}")
    if not resolved.is_file():
        raise AssetProvenanceError(f"asset file does not exist: {resolved}")
    if isinstance(max_scan_bytes, bool) or not isinstance(max_scan_bytes, int) or max_scan_bytes <= 0:
        raise AssetProvenanceError("max_scan_bytes must be a positive integer")

    before = resolved.stat()
    size = before.st_size
    digest = hashlib.sha256()
    scan_budget = min(size, max_scan_bytes)
    scanned = bytearray()
    remaining = scan_budget
    with resolved.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if remaining:
                take = chunk[:remaining]
                scanned.extend(take)
                remaining -= len(take)

    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise AssetProvenanceError(f"asset changed while hashing: {resolved}")

    found: dict[tuple[str, str], dict[str, str]] = {}
    raw = bytes(scanned)
    for kind, pattern in _REFERENCE_PATTERNS:
        for match in pattern.finditer(raw):
            value = _decode_reference(match.group(0))
            key = (kind, value)
            found[key] = {"kind": kind, "value": value}

    references = tuple(found[key] for key in sorted(found))
    if references:
        classification = "nvidia_or_external_runtime_reference"
    elif size > max_scan_bytes:
        classification = "scan_truncated_unknown"
    else:
        classification = "no_detected_external_reference"

    return AssetProvenanceReport(
        schema=ASSET_PROVENANCE_SCHEMA,
        path=str(resolved),
        size_bytes=size,
        sha256=digest.hexdigest(),
        usd_format=_usd_format(resolved, raw[:16]),
        scan_limit_bytes=max_scan_bytes,
        scan_complete=size <= max_scan_bytes,
        references=references,
        classification=classification,
        # This is intentionally not inferred from a clean scan.  Only a human
        # decision or an explicit upstream license can change this field.
        license_status="unresolved",
    )


def inspect_many(paths: Iterable[Path], *, max_scan_bytes: int = DEFAULT_MAX_SCAN_BYTES) -> tuple[AssetProvenanceReport, ...]:
    reports = tuple(inspect_usd(path, max_scan_bytes=max_scan_bytes) for path in paths)
    if not reports:
        raise AssetProvenanceError("at least one USD path is required")
    return reports


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_issue(issues: list[LocalAssetAuditIssue], code: str, path: str, message: str) -> None:
    issues.append(LocalAssetAuditIssue(code, path, message))


def _local_path(
    payload: dict[str, Any],
    key: str,
    *,
    directory: bool,
    issues: list[LocalAssetAuditIssue],
    repository_root: Path | None,
) -> Path | None:
    raw = payload.get(key)
    if not isinstance(raw, str) or not raw:
        _local_issue(issues, "path", f"$.{key}", "must be a non-empty absolute path")
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        _local_issue(issues, "path", f"$.{key}", "must be an absolute path")
        return None
    resolved = candidate.resolve()
    if directory and not resolved.is_dir():
        _local_issue(issues, "path_missing", f"$.{key}", "directory does not exist")
        return None
    if not directory and not resolved.is_file():
        _local_issue(issues, "path_missing", f"$.{key}", "file does not exist")
        return None
    if repository_root is not None:
        try:
            resolved.relative_to(repository_root)
        except ValueError:
            pass
        else:
            _local_issue(issues, "repository_asset", f"$.{key}", "runtime assets must remain outside the repository root")
    return resolved


def _verify_local_hash(
    path: Path | None,
    expected: Any,
    *,
    issue_path: str,
    issues: list[LocalAssetAuditIssue],
) -> str | None:
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        _local_issue(issues, "sha256", issue_path, "must be 64 lowercase hexadecimal characters")
        return None
    if path is None:
        return None
    actual = _sha256_file(path)
    if actual != expected:
        _local_issue(issues, "hash_mismatch", issue_path, "local bytes do not match the configured SHA-256")
    return actual


def audit_local_assets_config(
    config_path: Path,
    *,
    max_scan_bytes: int = DEFAULT_MAX_SCAN_BYTES,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Audit a user-installed Isaac/City-Lite configuration without copying assets.

    The audit proves local path and byte bindings only. It intentionally keeps
    redistribution disabled and does not infer license clearance from a clean
    hash or an absence of recognizable external references.
    """

    resolved_config = Path(config_path).expanduser().resolve()
    try:
        payload = json.loads(resolved_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssetProvenanceError(f"cannot read local-assets config: {resolved_config}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssetProvenanceError("local-assets config must be an object")
    root = repository_root.expanduser().resolve() if repository_root is not None else None
    issues: list[LocalAssetAuditIssue] = []
    if payload.get("schema") != LOCAL_ASSETS_SCHEMA:
        _local_issue(issues, "schema", "$.schema", f"expected {LOCAL_ASSETS_SCHEMA!r}")
    for key in sorted(set(payload) - _LOCAL_REQUIRED):
        _local_issue(issues, "unknown_field", f"$.{key}", "field is not part of the local-assets contract")
    for key in sorted(_LOCAL_REQUIRED - set(payload)):
        _local_issue(issues, "required_field", f"$.{key}", "field is required")
    for key in ("asset_package_id", "asset_package_version", "cf2x_source_provenance"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            _local_issue(issues, "value", f"$.{key}", "must be a non-empty string")
    if payload.get("license_status") != "internal_only":
        _local_issue(issues, "license_status", "$.license_status", "BYOA configuration must remain internal_only")
    redistribution = payload.get("public_redistribution")
    if not isinstance(redistribution, dict):
        _local_issue(issues, "redistribution", "$.public_redistribution", "must be an object with explicit false values")
    else:
        for key in sorted(set(redistribution) - _PUBLIC_REDISTRIBUTION_KEYS):
            _local_issue(issues, "redistribution", f"$.public_redistribution.{key}", "unknown redistribution scope")
        for key in sorted(_PUBLIC_REDISTRIBUTION_KEYS):
            if redistribution.get(key) is not False:
                _local_issue(issues, "redistribution", f"$.public_redistribution.{key}", "must be false for BYOA assets")

    asset_root = _local_path(payload, "asset_root", directory=True, issues=issues, repository_root=root)
    isaaclab_root = _local_path(payload, "isaaclab_root", directory=True, issues=issues, repository_root=root)
    isaac_python = _local_path(payload, "isaac_python", directory=False, issues=issues, repository_root=root)
    package_manifest = _local_path(payload, "asset_package_manifest", directory=False, issues=issues, repository_root=root)
    city_contract = _local_path(payload, "city_lite_contract", directory=False, issues=issues, repository_root=root)
    city_layer = _local_path(payload, "city_lite_layer", directory=False, issues=issues, repository_root=root)
    cf2x = _local_path(payload, "cf2x_usd", directory=False, issues=issues, repository_root=root)
    digests = {
        "asset_package_manifest": _verify_local_hash(package_manifest, payload.get("asset_package_sha256"), issue_path="$.asset_package_sha256", issues=issues),
        "city_lite_contract": _verify_local_hash(city_contract, payload.get("city_lite_contract_sha256"), issue_path="$.city_lite_contract_sha256", issues=issues),
        "city_lite_layer": _verify_local_hash(city_layer, payload.get("city_lite_layer_sha256"), issue_path="$.city_lite_layer_sha256", issues=issues),
        "cf2x_usd": _verify_local_hash(cf2x, payload.get("cf2x_usd_sha256"), issue_path="$.cf2x_usd_sha256", issues=issues),
    }
    reports: list[dict[str, Any]] = []
    for label, asset in (("city_lite_layer", city_layer), ("cf2x_usd", cf2x)):
        if asset is None:
            continue
        try:
            reports.append({"asset": label, **inspect_usd(asset, max_scan_bytes=max_scan_bytes).as_dict()})
        except (OSError, AssetProvenanceError) as exc:
            _local_issue(issues, "usd_scan", f"$.{label}", str(exc))
    return {
        "schema": LOCAL_ASSETS_SCHEMA,
        "status": "passed" if not issues else "blocked",
        "config": str(resolved_config),
        "repository_root": str(root) if root is not None else None,
        "asset_root": str(asset_root) if asset_root is not None else None,
        "isaaclab_root": str(isaaclab_root) if isaaclab_root is not None else None,
        "isaac_python": str(isaac_python) if isaac_python is not None else None,
        "digests": digests,
        "reports": reports,
        "issues": [issue.__dict__ for issue in issues],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--config", type=Path, help="audit a local BYOA configuration instead of standalone USD files")
    parser.add_argument("--repository-root", type=Path, help="reject runtime assets located under this source repository")
    parser.add_argument(
        "--max-scan-mib",
        type=float,
        default=64.0,
        help="maximum bytes scanned for plain-text references per file (default: 64 MiB)",
    )
    parser.add_argument("--require-no-external-references", action="store_true")
    args = parser.parse_args(argv)
    if args.max_scan_mib <= 0:
        parser.error("--max-scan-mib must be positive")
    if args.config is not None and args.paths:
        parser.error("--config cannot be combined with standalone USD paths")
    if args.config is None and not args.paths:
        parser.error("provide one or more USD paths or --config")
    max_scan_bytes = int(args.max_scan_mib * 1024 * 1024)
    if args.config is not None:
        try:
            payload = audit_local_assets_config(
                args.config,
                max_scan_bytes=max_scan_bytes,
                repository_root=args.repository_root,
            )
        except (OSError, AssetProvenanceError) as exc:
            print(json.dumps({"schema": LOCAL_ASSETS_SCHEMA, "status": "invalid", "error": str(exc)}, indent=2))
            return 1
        print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 2
    try:
        reports = inspect_many(args.paths, max_scan_bytes=max_scan_bytes)
    except (OSError, AssetProvenanceError) as exc:
        print(json.dumps({"schema": ASSET_PROVENANCE_SCHEMA, "status": "invalid", "error": str(exc)}, indent=2))
        return 1
    payload = {
        "schema": ASSET_PROVENANCE_SCHEMA,
        "status": "passed",
        "reports": [report.as_dict() for report in reports],
    }
    if args.require_no_external_references and any(
        report.has_external_references or not report.scan_complete for report in reports
    ):
        payload["status"] = "blocked"
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
