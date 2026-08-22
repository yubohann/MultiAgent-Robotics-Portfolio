#!/usr/bin/env python3
"""Verify a captured Poly Haven provenance package without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from collect_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from collect_strings(child)


def resolve_evidence_path(bundle: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else bundle / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    provenance = bundle / "provenance"
    manifest_path = provenance / "PROVENANCE_MANIFEST.json"
    hash_path = provenance / "PROVENANCE_MANIFEST.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    expected_manifest_hash = hash_path.read_text(encoding="ascii").split()[0].lower()
    if sha256_file(manifest_path) != expected_manifest_hash:
        failures.append("PROVENANCE_MANIFEST.json SHA-256 mismatch")

    original = manifest["original_registry"]
    original_path = resolve_evidence_path(bundle, original["path"])
    if sha256_file(original_path) != original["sha256"]:
        failures.append("Preserved original registry SHA-256 mismatch")
    if sha256_file(bundle / "ASSET_REGISTRY.json") != original["sha256"]:
        failures.append("Current ASSET_REGISTRY.json differs from preserved original")

    snapshot_count = 0
    for name, evidence in manifest["global_evidence"].items():
        if not isinstance(evidence, dict) or "sha256" not in evidence:
            continue
        path_value = evidence.get("snapshot_path") or evidence.get("path")
        if not path_value:
            failures.append(f"Global evidence has no path: {name}")
            continue
        path = resolve_evidence_path(bundle, path_value)
        if not path.is_file() or sha256_file(path) != evidence["sha256"]:
            failures.append(f"Global evidence mismatch: {name}")
        snapshot_count += 1

    for name, output in manifest["generated_outputs"].items():
        path = resolve_evidence_path(bundle, output["path"])
        if not path.is_file() or sha256_file(path) != output["sha256"]:
            failures.append(f"Generated output mismatch: {name}")

    author_count = 0
    registered_file_count = 0
    exact_historical_time_count = 0
    for asset in manifest["assets"]:
        asset_id = asset["asset_id"]
        if not asset.get("creator_names"):
            failures.append(f"Missing creator: {asset_id}")
        else:
            author_count += 1

        if asset["download_time_evidence"].get("exact_download_time_available"):
            exact_historical_time_count += 1

        official = asset["official_evidence"]
        for evidence_name in ("info_api", "files_api", "source_page"):
            evidence = official.get(evidence_name)
            if not evidence:
                failures.append(f"Missing {evidence_name} evidence: {asset_id}")
                continue
            path = resolve_evidence_path(bundle, evidence["snapshot_path"])
            if not path.is_file() or sha256_file(path) != evidence["sha256"]:
                failures.append(f"Snapshot mismatch: {asset_id}/{evidence_name}")
            snapshot_count += 1

        files_api_path = resolve_evidence_path(bundle, official["files_api"]["snapshot_path"])
        files_api_urls = set(
            collect_strings(json.loads(files_api_path.read_text(encoding="utf-8")))
        )
        for file_entry in asset["original_files"]:
            registered_file_count += 1
            local_path = bundle / file_entry["path"].replace("\\", "/").replace("//", "/").lstrip(
                "/"
            )
            if not local_path.is_file():
                failures.append(f"Missing asset file: {file_entry['path']}")
                continue
            observed_hash = sha256_file(local_path)
            if observed_hash != file_entry["registry_sha256"]:
                failures.append(f"Asset SHA-256 mismatch: {file_entry['path']}")
            if file_entry["source_url"] not in files_api_urls:
                failures.append(
                    f"Source URL absent from official API snapshot: {file_entry['path']}"
                )

    summary = {
        "assets": len(manifest["assets"]),
        "assets_with_authors": author_count,
        "registered_files": registered_file_count,
        "evidence_snapshots_and_documents": snapshot_count,
        "exact_historical_download_times": exact_historical_time_count,
        "failures": len(failures),
        "failure_details": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
