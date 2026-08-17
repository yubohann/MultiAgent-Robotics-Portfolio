"""Fail-closed supply-chain and redistribution manifest validation."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUPPLY_CHAIN_SCHEMA = "org.rivermark.benchmark.supply-chain.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_PRIVATE_TOKENS = ("evaluator", "private", "hidden_target", "target_truth")
_ASSET_KINDS = frozenset({"code", "scene_layer", "robot_asset", "runtime", "checkpoint", "label", "video", "data"})
_LICENSE_STATUSES = frozenset({"pending", "internal_only", "redistribution_cleared", "not_applicable"})
_SBOM_STATUSES = frozenset({"missing", "present", "verified"})
_SIGNATURE_STATUSES = frozenset({"unsigned", "attestation_only", "cryptographically_verified"})
_SIGNATURE_ALGORITHMS = frozenset({"ed25519", "minisign", "sigstore"})
_CYCLONEDX_SPEC_VERSIONS = frozenset({"1.6", "1.7"})
_PACKAGE_NORMALIZE = re.compile(r"[-_.]+")


@dataclass(frozen=True)
class SupplyChainIssue:
    code: str
    path: str
    message: str


class SupplyChainError(ValueError):
    """Raised when a supply-chain manifest is malformed or unsafe."""


def canonical_supply_chain_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return deterministic bytes for external signing and hash binding."""

    normalized = dict(payload)
    signature = normalized.get("signature")
    if isinstance(signature, Mapping):
        detached = dict(signature)
        detached.pop("manifest_sha256", None)
        # A detached signature is calculated from these canonical bytes, so
        # its own digest cannot be part of the signed message.
        detached.pop("sha256", None)
        normalized["signature"] = detached
    return (json.dumps(normalized, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def supply_chain_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_supply_chain_bytes(payload)).hexdigest()


def _issue(issues: list[SupplyChainIssue], code: str, path: str, message: str) -> None:
    issues.append(SupplyChainIssue(code, path, message))


def _safe_public_uri(value: Any, *, path: str, issues: list[SupplyChainIssue]) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        _issue(issues, "uri", path, "must be a non-empty URI")
        return
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        _issue(issues, "uri_scheme", path, "public source URI must use HTTPS and include a host")
        return
    host = parsed.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".local") or any(token in value.lower() for token in _PRIVATE_TOKENS):
        _issue(issues, "private_uri", path, "URI must not reference private/evaluator content")
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (address.is_private or address.is_loopback or address.is_link_local or address.is_reserved or address.is_unspecified):
        _issue(issues, "private_uri", path, "URI must not target a private or local address")


def _safe_relative_path(value: Any, *, path: str, issues: list[SupplyChainIssue]) -> None:
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/") or ".." in value.split("/"):
        _issue(issues, "path", path, "must be a public relative path")


def _validate_hash(value: Any, path: str, issues: list[SupplyChainIssue]) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _issue(issues, "sha256", path, "must be 64 lowercase hexadecimal characters")


def validate_supply_chain_manifest(payload: Any, *, require_release: bool = False) -> tuple[SupplyChainIssue, ...]:
    """Validate structure and, optionally, release-level clearance requirements."""

    issues: list[SupplyChainIssue] = []
    if not isinstance(payload, Mapping):
        return (SupplyChainIssue("type", "$", "manifest must be an object"),)
    allowed = {"schema", "manifest_version", "release_id", "created_at", "assets", "runtime_dependencies", "sbom", "signature"}
    for key in sorted(set(payload) - allowed):
        _issue(issues, "unknown_field", f"$.{key}", "field is not part of supply-chain v1")
    if payload.get("schema") != SUPPLY_CHAIN_SCHEMA:
        _issue(issues, "schema", "$.schema", f"expected {SUPPLY_CHAIN_SCHEMA!r}")
    if not isinstance(payload.get("manifest_version"), str) or not _SEMVER.fullmatch(payload["manifest_version"]):
        _issue(issues, "manifest_version", "$.manifest_version", "must be a semantic version")
    if not isinstance(payload.get("release_id"), str) or not _ID.fullmatch(payload.get("release_id", "")):
        _issue(issues, "release_id", "$.release_id", "invalid release identifier")
    if not isinstance(payload.get("created_at"), str) or not payload["created_at"]:
        _issue(issues, "created_at", "$.created_at", "must be a non-empty timestamp")

    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        _issue(issues, "assets", "$.assets", "must contain at least one asset")
        assets = []
    asset_ids: set[str] = set()
    for index, asset in enumerate(assets):
        path = f"$.assets[{index}]"
        if not isinstance(asset, Mapping):
            _issue(issues, "type", path, "asset must be an object")
            continue
        allowed_asset = {
            "asset_id",
            "kind",
            "path",
            "source_uri",
            "sha256",
            "license_spdx",
            "license_status",
            "redistributable",
            "attribution",
            "decision_record",
        }
        for key in sorted(set(asset) - allowed_asset):
            _issue(issues, "unknown_field", f"{path}.{key}", "field is not part of asset v1")
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or not _ID.fullmatch(asset_id):
            _issue(issues, "asset_id", f"{path}.asset_id", "invalid asset identifier")
        elif asset_id in asset_ids:
            _issue(issues, "duplicate_asset_id", f"{path}.asset_id", "asset_id must be unique")
        else:
            asset_ids.add(asset_id)
        if asset.get("kind") not in _ASSET_KINDS:
            _issue(issues, "kind", f"{path}.kind", "unknown asset kind")
        _validate_hash(asset.get("sha256"), f"{path}.sha256", issues)
        license_spdx = asset.get("license_spdx")
        if not isinstance(license_spdx, str) or not license_spdx:
            _issue(issues, "license_spdx", f"{path}.license_spdx", "must be a non-empty SPDX expression or NOASSERTION")
        status = asset.get("license_status")
        if status not in _LICENSE_STATUSES:
            _issue(issues, "license_status", f"{path}.license_status", "unknown license status")
        redistributable = asset.get("redistributable")
        if not isinstance(redistributable, bool):
            _issue(issues, "redistributable", f"{path}.redistributable", "must be boolean")
        if not isinstance(asset.get("attribution"), str) or not asset["attribution"].strip():
            _issue(issues, "attribution", f"{path}.attribution", "must be a non-empty attribution")
        if status == "redistribution_cleared" and redistributable is not True:
            _issue(issues, "license_consistency", path, "cleared assets must be marked redistributable")
        if status == "redistribution_cleared" and isinstance(license_spdx, str) and license_spdx.upper() in {"NOASSERTION", "UNKNOWN"}:
            _issue(issues, "license_consistency", path, "cleared assets require a concrete license expression")
        decision = asset.get("decision_record")
        if status == "redistribution_cleared":
            if not isinstance(decision, Mapping):
                _issue(issues, "decision_record", f"{path}.decision_record", "cleared assets require a human decision record")
            else:
                allowed_decision = {"record_id", "approved_by", "approved_at", "evidence_sha256"}
                for key in sorted(set(decision) - allowed_decision):
                    _issue(issues, "unknown_field", f"{path}.decision_record.{key}", "field is not part of decision record v1")
                for key in ("record_id", "approved_by", "approved_at"):
                    if not isinstance(decision.get(key), str) or not decision[key].strip():
                        _issue(issues, "decision_record", f"{path}.decision_record.{key}", "must be a non-empty string")
                _validate_hash(decision.get("evidence_sha256"), f"{path}.decision_record.evidence_sha256", issues)
        elif decision is not None:
            _issue(issues, "decision_record", f"{path}.decision_record", "only cleared assets may carry an approval decision")
        if status == "not_applicable" and asset.get("kind") not in {"data", "label"}:
            _issue(issues, "license_consistency", path, "not_applicable is restricted to derived data/labels")
        _safe_public_uri(asset.get("source_uri"), path=f"{path}.source_uri", issues=issues)
        if require_release and not isinstance(asset.get("source_uri"), str):
            _issue(issues, "source_uri_required", f"{path}.source_uri", "release assets require a public source URI")
        if "path" in asset:
            _safe_relative_path(asset["path"], path=f"{path}.path", issues=issues)

    dependencies = payload.get("runtime_dependencies")
    if not isinstance(dependencies, list):
        _issue(issues, "runtime_dependencies", "$.runtime_dependencies", "must be an array")
        dependencies = []
    for index, dependency in enumerate(dependencies):
        path = f"$.runtime_dependencies[{index}]"
        if not isinstance(dependency, Mapping):
            _issue(issues, "type", path, "dependency must be an object")
            continue
        for key in sorted(set(dependency) - {"name", "version", "license_spdx", "source_uri", "sha256"}):
            _issue(issues, "unknown_field", f"{path}.{key}", "field is not part of dependency v1")
        for key in ("name", "version", "license_spdx"):
            if not isinstance(dependency.get(key), str) or not dependency[key]:
                _issue(issues, key, f"{path}.{key}", "must be a non-empty string")
        if "sha256" in dependency:
            _validate_hash(dependency["sha256"], f"{path}.sha256", issues)
        _safe_public_uri(dependency.get("source_uri"), path=f"{path}.source_uri", issues=issues)

    sbom = payload.get("sbom")
    if not isinstance(sbom, Mapping):
        _issue(issues, "sbom", "$.sbom", "must be an object")
        sbom = {}
    if sbom.get("format") not in {"cyclonedx-json", "spdx-json"}:
        _issue(issues, "sbom_format", "$.sbom.format", "unsupported SBOM format")
    if sbom.get("status") not in _SBOM_STATUSES:
        _issue(issues, "sbom_status", "$.sbom.status", "unknown SBOM status")
    _validate_hash(sbom.get("sha256"), "$.sbom.sha256", issues)
    _safe_public_uri(sbom.get("uri"), path="$.sbom.uri", issues=issues)
    if "path" in sbom:
        _safe_relative_path(sbom["path"], path="$.sbom.path", issues=issues)
    if "spec_version" in sbom and sbom.get("spec_version") not in _CYCLONEDX_SPEC_VERSIONS:
        _issue(issues, "sbom_spec_version", "$.sbom.spec_version", "unsupported CycloneDX specification version")
    if "generator" in sbom and (not isinstance(sbom.get("generator"), str) or not sbom["generator"].strip()):
        _issue(issues, "sbom_generator", "$.sbom.generator", "must be a non-empty generator identifier")
    if require_release and sbom.get("status") == "verified" and not isinstance(sbom.get("uri"), str):
        _issue(issues, "sbom_uri_required", "$.sbom.uri", "release SBOM requires a public URI")
    if require_release:
        if not isinstance(sbom.get("path"), str):
            _issue(issues, "sbom_path_required", "$.sbom.path", "release SBOM requires a local relative verification path")
        if sbom.get("format") != "cyclonedx-json":
            _issue(issues, "sbom_format", "$.sbom.format", "release verifier currently supports CycloneDX JSON only")
        if sbom.get("spec_version") not in _CYCLONEDX_SPEC_VERSIONS:
            _issue(issues, "sbom_spec_version", "$.sbom.spec_version", "release SBOM requires a supported specification version")
        if not isinstance(sbom.get("generator"), str) or not sbom["generator"].strip():
            _issue(issues, "sbom_generator", "$.sbom.generator", "release SBOM requires its generator identifier")

    signature = payload.get("signature")
    if not isinstance(signature, Mapping):
        _issue(issues, "signature", "$.signature", "must be an object")
        signature = {}
    if signature.get("status") not in _SIGNATURE_STATUSES:
        _issue(issues, "signature_status", "$.signature.status", "unknown signature status")
    if signature.get("status") == "cryptographically_verified":
        if signature.get("algorithm") not in _SIGNATURE_ALGORITHMS:
            _issue(issues, "signature_algorithm", "$.signature.algorithm", "unsupported signature algorithm")
        for key in (
            "algorithm",
            "key_id",
            "path",
            "uri",
            "sha256",
            "manifest_sha256",
            "public_key_path",
            "public_key_uri",
            "public_key_sha256",
        ):
            if not signature.get(key):
                _issue(issues, "signature_binding", f"$.signature.{key}", "required for cryptographically verified release")
        _validate_hash(signature.get("sha256"), "$.signature.sha256", issues)
        _validate_hash(signature.get("manifest_sha256"), "$.signature.manifest_sha256", issues)
        _validate_hash(signature.get("public_key_sha256"), "$.signature.public_key_sha256", issues)
        if isinstance(signature.get("manifest_sha256"), str) and signature["manifest_sha256"] != supply_chain_sha256(payload):
            _issue(issues, "signature_manifest_mismatch", "$.signature.manifest_sha256", "does not bind the canonical manifest")
        _safe_public_uri(signature.get("uri"), path="$.signature.uri", issues=issues)
        _safe_public_uri(signature.get("public_key_uri"), path="$.signature.public_key_uri", issues=issues)
        _safe_relative_path(signature.get("path"), path="$.signature.path", issues=issues)
        _safe_relative_path(signature.get("public_key_path"), path="$.signature.public_key_path", issues=issues)
    elif require_release:
        _issue(issues, "signature_required", "$.signature.status", "release requires cryptographically_verified signature")
    if require_release:
        if any(
            isinstance(asset, Mapping)
            and str(asset.get("license_spdx", "")).upper() in {"NOASSERTION", "UNKNOWN"}
            for asset in assets
        ):
            _issue(issues, "license_assertion", "$.assets", "released assets cannot use NOASSERTION or UNKNOWN")
        if any(asset.get("license_status") != "redistribution_cleared" or asset.get("redistributable") is not True for asset in assets if isinstance(asset, Mapping)):
            _issue(issues, "license_closure", "$.assets", "every released asset must be redistribution-cleared")
        if sbom.get("status") != "verified":
            _issue(issues, "sbom_required", "$.sbom.status", "release requires a verified SBOM")
    return tuple(issues)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(manifest_path: Path, relative: Any, *, issue_path: str, issues: list[SupplyChainIssue]) -> Path | None:
    if not isinstance(relative, str):
        _issue(issues, "artifact_path", issue_path, "missing relative verification path")
        return None
    root = manifest_path.parent.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        _issue(issues, "artifact_path", issue_path, "resolves outside the manifest directory")
        return None
    if not candidate.is_file():
        _issue(issues, "artifact_missing", issue_path, "referenced verification artifact is not a regular file")
        return None
    return candidate


def _verify_cyclonedx_sbom(
    path: Path,
    payload: Mapping[str, Any],
    *,
    issues: list[SupplyChainIssue],
) -> None:
    sbom = payload.get("sbom")
    dependencies = payload.get("runtime_dependencies")
    if not isinstance(sbom, Mapping) or not isinstance(dependencies, list):
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _issue(issues, "sbom_json", "$.sbom.path", f"cannot parse CycloneDX JSON: {exc}")
        return
    if not isinstance(document, Mapping) or document.get("bomFormat") != "CycloneDX":
        _issue(issues, "sbom_document", "$.sbom.path", "must be a CycloneDX JSON BOM")
        return
    if document.get("specVersion") != sbom.get("spec_version"):
        _issue(issues, "sbom_spec_version", "$.sbom.path", "BOM specification version does not match the manifest")
    components = document.get("components")
    if not isinstance(components, list):
        _issue(issues, "sbom_components", "$.sbom.path", "CycloneDX BOM must contain a component list")
        components = []
    indexed: dict[str, set[str]] = {}
    for component in components:
        if not isinstance(component, Mapping):
            continue
        name = component.get("name")
        version = component.get("version")
        if isinstance(name, str) and isinstance(version, str):
            indexed.setdefault(_PACKAGE_NORMALIZE.sub("-", name).casefold(), set()).add(version)
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, Mapping):
            continue
        name = dependency.get("name")
        version = dependency.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        versions = indexed.get(_PACKAGE_NORMALIZE.sub("-", name).casefold(), set())
        if not versions:
            _issue(issues, "sbom_dependency_missing", f"$.runtime_dependencies[{index}]", "dependency is absent from the CycloneDX BOM")
        elif version not in versions:
            _issue(issues, "sbom_dependency_version", f"$.runtime_dependencies[{index}]", "dependency version does not match the CycloneDX BOM")


def _verify_ed25519_signature(
    public_key_path: Path,
    signature_path: Path,
    payload: Mapping[str, Any],
    *,
    issues: list[SupplyChainIssue],
) -> None:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        _issue(issues, "signature_verifier_unavailable", "$.signature.algorithm", "install the supply-chain extra to verify Ed25519 signatures")
        return
    try:
        public_key = serialization.load_pem_public_key(public_key_path.read_bytes())
    except (OSError, ValueError, TypeError) as exc:
        _issue(issues, "signature_public_key", "$.signature.public_key_path", f"cannot load Ed25519 public key: {exc}")
        return
    if not isinstance(public_key, Ed25519PublicKey):
        _issue(issues, "signature_public_key", "$.signature.public_key_path", "public key is not an Ed25519 key")
        return
    try:
        public_key.verify(signature_path.read_bytes(), canonical_supply_chain_bytes(payload))
    except InvalidSignature:
        _issue(issues, "signature_invalid", "$.signature.path", "detached signature does not verify the canonical manifest")
    except OSError as exc:
        _issue(issues, "signature_read", "$.signature.path", f"cannot read detached signature: {exc}")


def verify_supply_chain_artifacts(path: Path, payload: Mapping[str, Any]) -> tuple[SupplyChainIssue, ...]:
    """Verify local SBOM/signature artifacts referenced by one manifest.

    The artifact files must live below the manifest directory. This binds the
    release builder to bytes it can inspect before it emits any public manifest;
    public URLs are still checked separately by structural validation.
    """

    issues: list[SupplyChainIssue] = []
    manifest_path = Path(path).resolve()
    sbom = payload.get("sbom")
    if isinstance(sbom, Mapping) and sbom.get("status") == "verified":
        sbom_path = _resolve_artifact(manifest_path, sbom.get("path"), issue_path="$.sbom.path", issues=issues)
        if sbom_path is not None:
            if _sha256_file(sbom_path) != sbom.get("sha256"):
                _issue(issues, "sbom_hash", "$.sbom.path", "SBOM bytes do not match the declared SHA-256")
            elif sbom.get("format") == "cyclonedx-json":
                _verify_cyclonedx_sbom(sbom_path, payload, issues=issues)
            else:
                _issue(issues, "sbom_verifier_unavailable", "$.sbom.format", "only CycloneDX JSON has an in-repository artifact verifier")

    signature = payload.get("signature")
    if isinstance(signature, Mapping) and signature.get("status") == "cryptographically_verified":
        signature_path = _resolve_artifact(manifest_path, signature.get("path"), issue_path="$.signature.path", issues=issues)
        public_key_path = _resolve_artifact(manifest_path, signature.get("public_key_path"), issue_path="$.signature.public_key_path", issues=issues)
        if signature_path is not None and _sha256_file(signature_path) != signature.get("sha256"):
            _issue(issues, "signature_hash", "$.signature.path", "signature bytes do not match the declared SHA-256")
        if public_key_path is not None and _sha256_file(public_key_path) != signature.get("public_key_sha256"):
            _issue(issues, "signature_public_key_hash", "$.signature.public_key_path", "public key bytes do not match the declared SHA-256")
        if signature_path is not None and public_key_path is not None:
            if signature.get("algorithm") == "ed25519":
                _verify_ed25519_signature(public_key_path, signature_path, payload, issues=issues)
            else:
                _issue(issues, "signature_verifier_unavailable", "$.signature.algorithm", "no in-repository verifier is available for this signature algorithm")
    return tuple(issues)


def load_supply_chain_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupplyChainError(f"cannot read supply-chain manifest: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SupplyChainError("supply-chain manifest must be an object")
    return payload


def verify_supply_chain_manifest(
    path: Path,
    *,
    require_release: bool = False,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    payload = load_supply_chain_manifest(path)
    issues = list(validate_supply_chain_manifest(payload, require_release=require_release))
    artifact_checked = require_release or verify_artifacts
    if artifact_checked:
        issues.extend(verify_supply_chain_artifacts(path, payload))
    return {
        "schema": SUPPLY_CHAIN_SCHEMA,
        "status": "valid" if not issues else "invalid",
        "manifest": str(Path(path).resolve()),
        "manifest_sha256": supply_chain_sha256(payload),
        "asset_count": len(payload.get("assets", [])) if isinstance(payload.get("assets"), list) else 0,
        "release_id": payload.get("release_id"),
        "issues": [issue.__dict__ for issue in issues],
        "release_requirements": require_release,
        "artifact_verification": "verified" if artifact_checked and not issues else ("failed" if artifact_checked else "not_requested"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--verify-artifacts", action="store_true", help="verify local SBOM/signature bytes without asserting release clearance")
    args = parser.parse_args(argv)
    try:
        report = verify_supply_chain_manifest(
            args.manifest,
            require_release=args.require_release,
            verify_artifacts=args.verify_artifacts,
        )
    except (OSError, UnicodeDecodeError, SupplyChainError) as exc:
        print(json.dumps({"schema": SUPPLY_CHAIN_SCHEMA, "status": "invalid", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "valid" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
