#!/usr/bin/env python3
"""Capture reproducible provenance evidence for a Poly Haven asset bundle.

The original registry remains untouched.  This tool writes an evidence overlay,
an enriched registry, raw official HTTP snapshots, and SHA-256 manifests below
``BUNDLE/provenance``.

Historical download timestamps cannot be recreated exactly.  On Windows this
tool records NTFS creation timestamps as explicitly weak, estimated evidence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

USER_AGENT = "AeroCityBench-Provenance/1.0 (+research asset audit)"
POLYHAVEN_LICENSE_URL = "https://polyhaven.com/license"
CC0_LEGALCODE_URL = "https://creativecommons.org/publicdomain/zero/1.0/legalcode"
POLYHAVEN_CATALOG_URL = "https://api.polyhaven.com/assets"
ARCHIVE_CDX_URL = (
    "https://web.archive.org/cdx/search/cdx?"
    "url=polyhaven.com/license&output=json&filter=statuscode:200&"
    "filter=mimetype:text/html&fl=timestamp,original,statuscode,digest&"
    "collapse=digest&limit=100"
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def iso_utc(timestamp: float) -> str:
    return (
        dt.datetime.fromtimestamp(timestamp, dt.UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def selected_headers(headers: Any) -> dict[str, str]:
    allowed = {
        "cache-control",
        "content-length",
        "content-type",
        "date",
        "etag",
        "expires",
        "last-modified",
        "server",
        "vary",
    }
    return {str(k).lower(): str(v) for k, v in headers.items() if str(k).lower() in allowed}


def fetch(url: str, destination: Path, *, retries: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        retrieved_at = utc_now()
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*", "Accept-Encoding": "identity"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read()
                final_url = response.geturl()
                status = int(getattr(response, "status", 200))
                headers = selected_headers(response.headers)
            atomic_write(destination, body)
            record = {
                "requested_url": url,
                "final_url": final_url,
                "retrieved_at_utc": retrieved_at,
                "http_status": status,
                "response_headers": headers,
                "snapshot_path": str(destination),
                "bytes": len(body),
                "sha256": sha256_bytes(body),
            }
            atomic_write(destination.with_name(destination.name + ".http.json"), json_bytes(record))
            return record
        except (
            OSError,
            http.client.HTTPException,
            urllib.error.URLError,
            urllib.error.HTTPError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def load_json_snapshot(record: dict[str, Any]) -> Any:
    return json.loads(Path(record["snapshot_path"]).read_text(encoding="utf-8"))


def collect_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from collect_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from collect_strings(child)


def path_from_registry(bundle: Path, registry_path: str) -> Path:
    clean = registry_path.replace("\\", "/").replace("//", "/").lstrip("/")
    candidate = (bundle / clean).resolve()
    candidate.relative_to(bundle.resolve())
    return candidate


def download_time_evidence(bundle: Path, asset: dict[str, Any]) -> dict[str, Any]:
    file_records: list[dict[str, Any]] = []
    creation_times: list[float] = []
    for file_entry in asset.get("files", []):
        path = path_from_registry(bundle, file_entry["path"])
        if not path.is_file():
            file_records.append({"path": file_entry["path"], "exists": False})
            continue
        stat = path.stat()
        creation_times.append(stat.st_ctime)
        file_records.append(
            {
                "path": file_entry["path"],
                "exists": True,
                "observed_ntfs_creation_time_utc": iso_utc(stat.st_ctime),
                "observed_last_write_time_utc": iso_utc(stat.st_mtime),
                "bytes_observed": stat.st_size,
            }
        )
    result: dict[str, Any] = {
        "exact_download_time_available": False,
        "status": "estimated_from_local_filesystem_metadata",
        "confidence": "weak",
        "warning_cn": (
            "原下载程序未记录完成时间。NTFS创建时间可能因复制、恢复或迁移而改变，"
            "只能作为本机首次出现时间的旁证，不能表述为精确下载时间。"
        ),
        "method": "minimum_and_maximum_observed_NTFS_creation_time_across_registered_files",
        "files": file_records,
    }
    if creation_times:
        result["estimated_window_utc"] = {
            "earliest": iso_utc(min(creation_times)),
            "latest": iso_utc(max(creation_times)),
        }
    return result


def capture_one_asset(
    bundle: Path,
    snapshots: Path,
    asset: dict[str, Any],
    *,
    capture_pages: bool,
) -> dict[str, Any]:
    asset_id = asset["asset_id"]
    safe_id = urllib.parse.quote(asset_id, safe="_-.")
    asset_dir = snapshots / "polyhaven" / "assets" / asset_id
    info_url = f"https://api.polyhaven.com/info/{safe_id}"
    files_url = f"https://api.polyhaven.com/files/{safe_id}"
    source_page = asset.get("source_page") or f"https://polyhaven.com/a/{safe_id}"

    info_record = fetch(info_url, asset_dir / "info.json")
    files_record = fetch(files_url, asset_dir / "files.json")
    page_record = fetch(source_page, asset_dir / "source_page.html") if capture_pages else None

    info = load_json_snapshot(info_record)
    files_api = load_json_snapshot(files_record)
    official_strings = set(collect_strings(files_api))
    authors_raw = info.get("authors", {}) if isinstance(info, dict) else {}
    if isinstance(authors_raw, dict):
        author_names = sorted(str(name) for name in authors_raw)
    elif isinstance(authors_raw, list):
        author_names = sorted(str(name) for name in authors_raw)
    else:
        author_names = [str(authors_raw)] if authors_raw else []

    original_files: list[dict[str, Any]] = []
    for file_entry in asset.get("files", []):
        local_path = path_from_registry(bundle, file_entry["path"])
        observed_hash = sha256_file(local_path) if local_path.is_file() else None
        source_url = file_entry.get("source_url")
        original_files.append(
            {
                "path": file_entry["path"],
                "source_url": source_url,
                "source_url_present_in_official_files_api_snapshot": source_url in official_strings,
                "registry_sha256": file_entry.get("sha256"),
                "observed_sha256": observed_hash,
                "sha256_matches_registry": observed_hash == file_entry.get("sha256")
                if observed_hash
                else False,
                "bytes_registry": file_entry.get("bytes"),
                "bytes_observed": local_path.stat().st_size if local_path.is_file() else None,
            }
        )

    return {
        "asset_id": asset_id,
        "creator_names": author_names,
        "authors_raw": authors_raw,
        "publisher": "Poly Haven",
        "spdx": asset.get("spdx"),
        "source_page": source_page,
        "official_evidence": {
            "info_api": info_record,
            "files_api": files_record,
            "source_page": page_record,
        },
        "download_time_evidence": download_time_evidence(bundle, asset),
        "original_files": original_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-asset-pages", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    registry_path = bundle / "ASSET_REGISTRY.json"
    if not registry_path.is_file():
        raise SystemExit(f"Missing registry: {registry_path}")

    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes.decode("utf-8-sig"))
    assets = list(registry.get("assets", []))
    if not assets:
        raise SystemExit("Registry contains no assets")

    provenance = bundle / "provenance"
    snapshots = provenance / "snapshots"
    provenance.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    previous_manifest_path = provenance / "PROVENANCE_MANIFEST.json"
    previous_manifest = None
    if args.resume and previous_manifest_path.is_file():
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))

    original_copy = provenance / "original" / "ASSET_REGISTRY.json"
    atomic_write(original_copy, registry_bytes)

    print(f"Capturing global evidence for {len(assets)} assets", flush=True)
    global_evidence: dict[str, Any] = (
        dict(previous_manifest.get("global_evidence", {})) if previous_manifest else {}
    )
    global_targets = [
        ("polyhaven_license", POLYHAVEN_LICENSE_URL, snapshots / "polyhaven" / "license.html"),
        (
            "cc0_legalcode",
            CC0_LEGALCODE_URL,
            snapshots / "creativecommons" / "CC0-1.0-legalcode.html",
        ),
        (
            "polyhaven_catalog",
            POLYHAVEN_CATALOG_URL,
            snapshots / "polyhaven" / "api" / "assets.json",
        ),
    ]
    for name, url, destination in global_targets:
        if name not in global_evidence:
            global_evidence[name] = fetch(url, destination)

    if "internet_archive_cdx" not in global_evidence:
        try:
            global_evidence["internet_archive_cdx"] = fetch(
                ARCHIVE_CDX_URL,
                snapshots / "internet_archive" / "polyhaven_license_cdx.json",
                retries=2,
            )
        except Exception as exc:  # Supplementary evidence must not block official evidence.
            global_evidence["internet_archive_cdx"] = {
                "requested_url": ARCHIVE_CDX_URL,
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "attempted_at_utc": utc_now(),
            }

    local_license = bundle / "licenses" / "CC0-1.0.txt"
    if local_license.is_file():
        global_evidence["bundled_cc0_text"] = {
            "path": relative_posix(local_license, bundle),
            "bytes": local_license.stat().st_size,
            "sha256": sha256_file(local_license),
        }

    results: list[dict[str, Any]] = (
        list(previous_manifest.get("assets", [])) if previous_manifest else []
    )
    completed_ids = {item["asset_id"] for item in results}
    pending_assets = [asset for asset in assets if asset["asset_id"] not in completed_ids]
    failures: list[dict[str, str]] = []
    workers = max(1, min(args.workers, 8))
    if completed_ids:
        resume_message = (
            f"Resume: retaining {len(completed_ids)} completed assets; "
            f"pending={len(pending_assets)}"
        )
        print(
            resume_message,
            flush=True,
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(
                capture_one_asset,
                bundle,
                snapshots,
                asset,
                capture_pages=not args.skip_asset_pages,
            ): asset["asset_id"]
            for asset in pending_assets
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_id):
            asset_id = future_to_id[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"asset_id": asset_id, "error": f"{type(exc).__name__}: {exc}"})
            completed += 1
            if completed % 5 == 0 or completed == len(pending_assets):
                print(
                    f"Captured pending {completed}/{len(pending_assets)}; "
                    f"total={len(results)}/{len(assets)}; failures={len(failures)}",
                    flush=True,
                )

    results.sort(key=lambda item: item["asset_id"])
    failures.sort(key=lambda item: item["asset_id"])

    enriched = json.loads(json.dumps(registry))
    result_by_id = {item["asset_id"]: item for item in results}
    for asset in enriched["assets"]:
        evidence = result_by_id.get(asset["asset_id"])
        if evidence:
            asset["provenance"] = evidence
    enriched["provenance_overlay"] = {
        "generated_at_utc": utc_now(),
        "original_registry_sha256": sha256_bytes(registry_bytes),
        "policy": "Original registry preserved; historical download time is explicitly estimated.",
    }
    enriched_path = provenance / "ASSET_REGISTRY.enriched.json"
    atomic_write(enriched_path, json_bytes(enriched))

    download_report = {
        "schema_version": "aerocity-download-time-recovery-1",
        "generated_at_utc": utc_now(),
        "exact_historical_download_time_recoverable": False,
        "registry_generated_at_utc": registry.get("generated_at_utc"),
        "assets": [
            {"asset_id": item["asset_id"], **item["download_time_evidence"]} for item in results
        ],
    }
    download_report_path = provenance / "DOWNLOAD_TIME_RECOVERY.json"
    atomic_write(download_report_path, json_bytes(download_report))

    authors = {
        "schema_version": "aerocity-polyhaven-authors-1",
        "generated_at_utc": utc_now(),
        "assets": [
            {
                "asset_id": item["asset_id"],
                "creator_names": item["creator_names"],
                "authors_raw": item["authors_raw"],
                "publisher": item["publisher"],
                "evidence_url": item["official_evidence"]["info_api"]["requested_url"],
                "evidence_sha256": item["official_evidence"]["info_api"]["sha256"],
                "evidence_snapshot": relative_posix(
                    Path(item["official_evidence"]["info_api"]["snapshot_path"]), bundle
                ),
            }
            for item in results
        ],
    }
    authors_path = provenance / "AUTHORS.json"
    atomic_write(authors_path, json_bytes(authors))

    all_files_verified = all(
        file_record["sha256_matches_registry"]
        and file_record["source_url_present_in_official_files_api_snapshot"]
        for item in results
        for file_record in item["original_files"]
    )
    manifest = {
        "schema_version": "aerocity-asset-provenance-1",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "bundle": str(bundle),
        "asset_count_registry": len(assets),
        "asset_count_evidenced": len(results),
        "failures": failures,
        "original_registry": {
            "path": relative_posix(original_copy, bundle),
            "sha256": sha256_bytes(registry_bytes),
        },
        "global_evidence": global_evidence,
        "generated_outputs": {
            "enriched_registry": {
                "path": relative_posix(enriched_path, bundle),
                "sha256": sha256_file(enriched_path),
            },
            "authors": {
                "path": relative_posix(authors_path, bundle),
                "sha256": sha256_file(authors_path),
            },
            "download_time_recovery": {
                "path": relative_posix(download_report_path, bundle),
                "sha256": sha256_file(download_report_path),
            },
        },
        "verification_summary": {
            "all_registered_files_exist_and_match_sha256": all(
                file_record["sha256_matches_registry"]
                for item in results
                for file_record in item["original_files"]
            ),
            "all_registered_source_urls_confirmed_by_official_files_api": all(
                file_record["source_url_present_in_official_files_api_snapshot"]
                for item in results
                for file_record in item["original_files"]
            ),
            "all_file_and_url_checks_passed": all_files_verified,
        },
        "assets": results,
    }
    manifest_path = provenance / "PROVENANCE_MANIFEST.json"
    atomic_write(manifest_path, json_bytes(manifest))
    manifest_hash = sha256_file(manifest_path)
    atomic_write(
        provenance / "PROVENANCE_MANIFEST.sha256",
        f"{manifest_hash}  PROVENANCE_MANIFEST.json\n".encode(),
    )

    readme = "# Poly Haven 资产来源证据\n\n"
    readme += f"- 资产数量：{len(assets)}\n"
    readme += f"- 成功取得官方证据：{len(results)}\n"
    readme += f"- 失败：{len(failures)}\n"
    readme += f"- 证据采集完成时间（UTC）：{manifest['completed_at_utc']}\n"
    readme += f"- 原登记表 SHA-256：`{sha256_bytes(registry_bytes)}`\n"
    readme += f"- 证据清单 SHA-256：`{manifest_hash}`\n\n"
    readme += "作者来自 Poly Haven 官方 `info/{asset_id}` API；原始文件 URL 由官方 "
    readme += "`files/{asset_id}` API 快照交叉验证。许可证证据包含 Poly Haven 官方许可页、"
    readme += "Creative Commons CC0 法律文本及其原始 HTTP 快照。\n\n"
    readme += "历史下载程序没有保存精确完成时间。`DOWNLOAD_TIME_RECOVERY.json` 中的时间仅为 "
    readme += "NTFS 创建时间估算，已明确标记为弱证据，不得改写为精确下载时间。\n"
    atomic_write(provenance / "README_CN.md", readme.encode("utf-8"))

    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Manifest SHA-256: {manifest_hash}", flush=True)
    if failures:
        print(json.dumps(failures, ensure_ascii=False, indent=2), flush=True)
        return 2
    if not all_files_verified:
        print("Evidence captured, but one or more file or source URL checks failed", flush=True)
        return 3
    print("All provenance checks passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
