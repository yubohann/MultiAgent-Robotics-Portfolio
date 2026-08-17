from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.supply_chain import (  # noqa: E402
    SUPPLY_CHAIN_SCHEMA,
    canonical_supply_chain_bytes,
    supply_chain_sha256,
    validate_supply_chain_manifest,
    verify_supply_chain_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(*, cleared: bool = False) -> dict:
    status = "redistribution_cleared" if cleared else "internal_only"
    payload = {
        "schema": SUPPLY_CHAIN_SCHEMA,
        "manifest_version": "1.0.0",
        "release_id": "rivermark-dev-audit",
        "created_at": "2026-07-24T00:00:00Z",
        "assets": [
            {
                "asset_id": "project-code",
                "kind": "code",
                "path": "src/rivermark_benchmark",
                "source_uri": "https://github.com/yubohann/rivermark-benchmark",
                "sha256": "a" * 64,
                "license_spdx": "Apache-2.0",
                "license_status": status,
                "redistributable": cleared,
                "attribution": "Rivermark Benchmark maintainers",
            },
            {
                "asset_id": "city-lite-layer",
                "kind": "scene_layer",
                "source_uri": "https://example.org/rivermark/city-lite.usd",
                "sha256": "b" * 64,
                "license_spdx": "CC-BY-4.0",
                "license_status": status,
                "redistributable": cleared,
                "attribution": "City-Lite asset authors",
            },
        ],
        "runtime_dependencies": [
            {
                "name": "numpy",
                "version": "1.24.0",
                "license_spdx": "BSD-3-Clause",
                "source_uri": "https://pypi.org/project/numpy/",
            }
        ],
        "sbom": {
            "format": "cyclonedx-json",
            "status": "verified" if cleared else "present",
            "path": "artifacts/sbom.cdx.json",
            "uri": "https://example.org/rivermark/sbom.json",
            "sha256": "c" * 64,
            "spec_version": "1.7",
            "generator": "cyclonedx-py 7.3.1",
        },
        "signature": {"status": "unsigned"},
    }
    if cleared:
        for index, asset in enumerate(payload["assets"]):
            asset["decision_record"] = {
                "record_id": f"legal-decision-{index + 1}",
                "approved_by": "Release approver",
                "approved_at": "2026-07-24T00:00:00Z",
                "evidence_sha256": str(index + 1) * 64,
            }
    return payload


def _write_verified_release_artifacts(root: Path, payload: dict) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sbom_path = root / "artifacts/sbom.cdx.json"
    signature_path = root / "artifacts/supply-chain.ed25519.sig"
    public_key_path = root / "artifacts/release.ed25519.pub"
    sbom_path.parent.mkdir(parents=True, exist_ok=True)
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "serialNumber": "urn:uuid:00000000-0000-0000-0000-000000000001",
        "version": 1,
        "metadata": {"component": {"type": "application", "name": "rivermark-benchmark", "version": "0.1.0"}},
        "components": [{"type": "library", "name": "numpy", "version": "1.24.0"}],
    }
    sbom_path.write_text(json.dumps(sbom, sort_keys=True), encoding="utf-8")
    private_key = Ed25519PrivateKey.generate()
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    payload["sbom"].update({"sha256": _sha256(sbom_path)})
    payload["signature"] = {
        "status": "cryptographically_verified",
        "algorithm": "ed25519",
        "key_id": "release-key-2026",
        "path": "artifacts/supply-chain.ed25519.sig",
        "uri": "https://example.org/rivermark/release.sig",
        "sha256": "0" * 64,
        "manifest_sha256": "0" * 64,
        "public_key_path": "artifacts/release.ed25519.pub",
        "public_key_uri": "https://example.org/rivermark/release.ed25519.pub",
        "public_key_sha256": _sha256(public_key_path),
    }
    signature_path.write_bytes(private_key.sign(canonical_supply_chain_bytes(payload)))
    payload["signature"]["sha256"] = _sha256(signature_path)
    payload["signature"]["manifest_sha256"] = supply_chain_sha256(payload)


class SupplyChainTests(unittest.TestCase):
    def test_internal_audit_is_structurally_valid_but_release_closed(self) -> None:
        payload = _manifest()
        payload["assets"][1]["license_spdx"] = "NOASSERTION"
        self.assertEqual(validate_supply_chain_manifest(payload), ())
        codes = {issue.code for issue in validate_supply_chain_manifest(payload, require_release=True)}
        self.assertIn("license_assertion", codes)
        self.assertIn("license_closure", codes)
        self.assertIn("sbom_required", codes)
        self.assertIn("signature_required", codes)

    def test_private_uri_and_unknown_field_fail_closed(self) -> None:
        payload = _manifest()
        payload["assets"][0]["source_uri"] = "https://private.example/evaluator/asset"
        payload["unexpected"] = True
        codes = {issue.code for issue in validate_supply_chain_manifest(payload)}
        self.assertIn("private_uri", codes)
        self.assertIn("unknown_field", codes)

    def test_unknown_license_cannot_be_marked_cleared(self) -> None:
        payload = _manifest(cleared=True)
        payload["assets"][1]["license_spdx"] = "NOASSERTION"
        codes = {issue.code for issue in validate_supply_chain_manifest(payload)}
        self.assertIn("license_consistency", codes)

    def test_clearance_requires_human_decision_record(self) -> None:
        payload = _manifest(cleared=True)
        del payload["assets"][0]["decision_record"]
        self.assertIn(
            "decision_record",
            {issue.code for issue in validate_supply_chain_manifest(payload)},
        )

        pending = _manifest()
        pending["assets"][0]["decision_record"] = {
            "record_id": "premature",
            "approved_by": "Nobody",
            "approved_at": "2026-07-24T00:00:00Z",
            "evidence_sha256": "1" * 64,
        }
        self.assertIn(
            "decision_record",
            {issue.code for issue in validate_supply_chain_manifest(pending)},
        )

    def test_verified_release_binds_signature_to_canonical_manifest(self) -> None:
        payload = _manifest(cleared=True)
        payload["signature"] = {
            "status": "cryptographically_verified",
            "algorithm": "ed25519",
            "key_id": "release-key-2026",
            "path": "artifacts/supply-chain.ed25519.sig",
            "uri": "https://example.org/rivermark/release.sig",
            "sha256": "d" * 64,
            "manifest_sha256": "0" * 64,
            "public_key_path": "artifacts/release.ed25519.pub",
            "public_key_uri": "https://example.org/rivermark/release.ed25519.pub",
            "public_key_sha256": "e" * 64,
        }
        payload["signature"]["manifest_sha256"] = supply_chain_sha256(payload)
        self.assertEqual(validate_supply_chain_manifest(payload, require_release=True), ())
        tampered = copy.deepcopy(payload)
        tampered["assets"][0]["sha256"] = "e" * 64
        self.assertIn(
            "signature_manifest_mismatch",
            {issue.code for issue in validate_supply_chain_manifest(tampered, require_release=True)},
        )

    def test_file_verifier_reports_hash_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "supply-chain.json"
            payload = _manifest()
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = verify_supply_chain_manifest(path)
            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["manifest_sha256"], supply_chain_sha256(payload))
            self.assertEqual(report["asset_count"], 2)
            self.assertEqual(report["release_id"], "rivermark-dev-audit")

    def test_release_verifier_reads_hash_bound_sbom_and_ed25519_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _manifest(cleared=True)
            _write_verified_release_artifacts(root, payload)
            manifest_path = root / "supply-chain.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            report = verify_supply_chain_manifest(manifest_path, require_release=True)
            self.assertEqual(report["status"], "valid", report["issues"])
            self.assertEqual(report["artifact_verification"], "verified")

    def test_release_verifier_rejects_tampered_sbom_and_wrong_public_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _manifest(cleared=True)
            _write_verified_release_artifacts(root, payload)
            manifest_path = root / "supply-chain.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            sbom_path = root / "artifacts/sbom.cdx.json"
            sbom_path.write_text("{}", encoding="utf-8")
            report = verify_supply_chain_manifest(manifest_path, require_release=True)
            self.assertIn("sbom_hash", {issue["code"] for issue in report["issues"]})

            _write_verified_release_artifacts(root, payload)
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            key_path = root / "artifacts/release.ed25519.pub"
            key_path.write_text("not a public key", encoding="utf-8")
            report = verify_supply_chain_manifest(manifest_path, require_release=True)
            self.assertIn("signature_public_key_hash", {issue["code"] for issue in report["issues"]})


if __name__ == "__main__":
    unittest.main()
