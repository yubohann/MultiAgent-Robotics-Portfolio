from __future__ import annotations

import hashlib
import http.server
import json
import sys
import tempfile
import threading
import urllib.request
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.release_manifest import (
    DownloadError,
    ReleaseManifestError,
    download_shards,
    load_release_manifest,
    plan_download,
    select_shards,
    validate_release_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(source: Path) -> dict:
    return {
        "schema": "org.rivermark.benchmark.release-manifest.v1",
        "dataset_version": "0.1.0",
        "release_id": "pilot-test-001",
        "license_status": "redistribution_cleared",
        "source_revision": "a" * 40,
        "shards": [
            {
                "shard_id": "episode-001-rgb",
                "episode_id": "episode-001",
                "split": "validation",
                "modality": "rgb",
                "agent_id": 0,
                "media_type": "application/octet-stream",
                "compression": "none",
                "schema": "org.rivermark.rgb.v1",
                "path": "validation/episode-001/rgb.bin",
                "url": source.as_uri(),
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
                "source_capture_sha256": "b" * 64,
            }
        ],
    }


class ReleaseManifestTests(unittest.TestCase):
    def _withdrawn_defect(self, payload: dict) -> dict:
        shard = payload["shards"][0]
        return {
            "issue_id": "DATA-001",
            "status": "withdrawn",
            "severity": "high",
            "summary": "The shard contains a decode defect.",
            "affected_shards": [{
                "shard_id": shard["shard_id"],
                "episode_id": shard["episode_id"],
                "path": shard["path"],
                "original_sha256": shard["sha256"],
            }],
            "version_bump_policy": "patch",
            "deprecation_window": {"grace_releases": 0, "replacement_required": True},
            "tombstone": {
                "kind": "withdrawn",
                "reason": "Source stream failed independent decoder validation.",
                "replacement_release_id": "pilot-test-002",
            },
        }

    def test_manifest_validates_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"Rivermark shard\n")
            payload = _manifest(source)
            self.assertEqual(validate_release_manifest(payload), ())
            self.assertEqual(len(select_shards(payload, splits=["validation"], modalities=["rgb"])), 1)
            self.assertEqual(select_shards(payload, splits=["train"]), ())

    def test_defect_index_binds_immutable_shard_and_withdrawn_download_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"immutable defect fixture")
            payload = _manifest(source)
            payload["defects"] = [self._withdrawn_defect(payload)]
            self.assertEqual(validate_release_manifest(payload), ())
            manifest_path = root / "release-manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ReleaseManifestError):
                download_shards(manifest_path, root / "download")

    def test_defect_index_rejects_stale_hash_and_private_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"x")
            payload = _manifest(source)
            defect = self._withdrawn_defect(payload)
            defect["affected_shards"][0]["original_sha256"] = "c" * 64
            defect["affected_shards"][0]["path"] = "private/evaluator.bin"
            payload["defects"] = [defect]
            codes = {issue.code for issue in validate_release_manifest(payload)}
            self.assertIn("unsafe_path", codes)
            self.assertIn("defect_binding", codes)

    def test_resolved_defect_requires_correction_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"x")
            payload = _manifest(source)
            defect = self._withdrawn_defect(payload)
            defect["status"] = "resolved"
            defect.pop("tombstone")
            payload["defects"] = [defect]
            self.assertIn("correction_mapping", {issue.code for issue in validate_release_manifest(payload)})

    def test_download_plan_reports_bytes_without_creating_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"Rivermark shard\n")
            manifest_path = root / "release-manifest.json"
            manifest_path.write_text(json.dumps(_manifest(source)), encoding="utf-8")
            cache = root / "planned-cache"
            plan = plan_download(manifest_path, splits=("validation",), modalities=("rgb",))
            self.assertEqual(plan["status"], "planned")
            self.assertEqual(plan["shard_count"], 1)
            self.assertEqual(plan["total_bytes"], source.stat().st_size)
            self.assertFalse(cache.exists())
            self.assertEqual(plan["shards"][0]["sha256"], _sha256(source))

    def test_frame_range_selects_only_complete_pre_sharded_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"Rivermark shard\n")
            payload = _manifest(source)
            payload["shards"][0].update({"frame_start": 10, "frame_end": 20})
            self.assertEqual(len(select_shards(payload, frame_start=0, frame_end=20)), 1)
            self.assertEqual(select_shards(payload, frame_start=15, frame_end=25), ())
            with self.assertRaises(ReleaseManifestError):
                select_shards(payload, frame_start=10)
            with self.assertRaises(ReleaseManifestError):
                select_shards(payload, frame_start=20, frame_end=20)

    def test_frame_range_download_is_hash_bound_and_does_not_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"one immutable shard")
            payload = _manifest(source)
            payload["shards"][0].update({"frame_start": 100, "frame_end": 120})
            manifest_path = root / "release-manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            results = download_shards(manifest_path, root / "download", frame_start=0, frame_end=120)
            self.assertEqual(results[0].status, "downloaded")
            self.assertEqual((root / "download/validation/episode-001/rgb.bin").read_bytes(), source.read_bytes())
            with self.assertRaises(ReleaseManifestError):
                download_shards(manifest_path, root / "other", frame_start=110, frame_end=130)

    def test_private_url_and_unlicensed_manifest_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"x")
            payload = _manifest(source)
            payload["license_status"] = "internal_only"
            payload["shards"][0]["url"] = "https://example.invalid/private/evaluator.bin"
            codes = {issue.code for issue in validate_release_manifest(payload, require_https=True)}
            self.assertIn("license_status", codes)
            self.assertIn("private_url", codes)

    def test_private_host_and_malformed_metadata_uri_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"x")
            payload = _manifest(source)
            payload["metadata_uri"] = "https://127.0.0.1/manifest.json"
            payload["shards"][0]["url"] = "https://example.com"
            codes = {issue.code for issue in validate_release_manifest(payload, require_https=True)}
            self.assertIn("private_host", codes)

    def test_unsafe_relative_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"x")
            payload = _manifest(source)
            payload["shards"][0]["path"] = "validation//episode/rgb.bin"
            self.assertIn("unsafe_path", {issue.code for issue in validate_release_manifest(payload)})

    def test_download_is_hash_bound_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"one immutable shard")
            manifest_path = root / "release-manifest.json"
            manifest_path.write_text(json.dumps(_manifest(source)), encoding="utf-8")
            destination = root / "download"
            first = download_shards(manifest_path, destination)
            second = download_shards(manifest_path, destination)
            self.assertEqual(first[0].status, "downloaded")
            self.assertEqual(second[0].status, "already_verified")
            self.assertEqual((destination / "validation/episode-001/rgb.bin").read_bytes(), source.read_bytes())

    def test_existing_corrupt_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"correct")
            manifest_path = root / "release-manifest.json"
            manifest_path.write_text(json.dumps(_manifest(source)), encoding="utf-8")
            destination = root / "download/validation/episode-001/rgb.bin"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"corrupt")
            with self.assertRaises(DownloadError):
                download_shards(manifest_path, root / "download")

    def test_complete_verified_partial_is_promoted_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"complete partial")
            manifest_path = root / "release-manifest.json"
            manifest_path.write_text(json.dumps(_manifest(source)), encoding="utf-8")
            partial = root / "download/validation/episode-001/rgb.bin.part"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(source.read_bytes())
            results = download_shards(manifest_path, root / "download")
            self.assertEqual(results[0].status, "resumed_verified")
            self.assertEqual((root / "download/validation/episode-001/rgb.bin").read_bytes(), source.read_bytes())

    def test_http_range_resume_requires_matching_content_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"0123456789abcdef")

            class RangeHandler(http.server.BaseHTTPRequestHandler):
                requests: list[tuple[str, str | None]] = []

                def do_GET(self) -> None:  # noqa: N802
                    RangeHandler.requests.append((self.path, self.headers.get("Range")))
                    body = source.read_bytes()
                    value = self.headers.get("Range")
                    if value == "bytes=5-":
                        self.send_response(206)
                        self.send_header("Content-Range", "bytes 5-15/16")
                        payload = body[5:]
                    else:
                        self.send_response(200)
                        payload = body
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def log_message(self, *_args: object) -> None:
                    return

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                payload = _manifest(source)
                payload["shards"][0]["url"] = "https://download.example.org/source.bin"
                manifest_path = root / "release-manifest.json"
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                partial = root / "download/validation/episode-001/rgb.bin.part"
                partial.parent.mkdir(parents=True)
                partial.write_bytes(source.read_bytes()[:5])

                def local_open(_url: str, offset: int):
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/source.bin",
                        headers={"Range": f"bytes={offset}-"} if offset else {},
                    )
                    # Keep the local fixture independent of any machine proxy.
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    return opener.open(request, timeout=5)

                with patch("rivermark_benchmark.release_manifest._open_download", side_effect=local_open):
                    results = download_shards(manifest_path, root / "download")
                self.assertEqual(results[0].status, "downloaded")
                self.assertEqual((root / "download/validation/episode-001/rgb.bin").read_bytes(), source.read_bytes())
                self.assertEqual(RangeHandler.requests[0][1], "bytes=5-")

                RangeHandler.requests.clear()
                (root / "download/validation/episode-001/rgb.bin").unlink()
                partial.unlink(missing_ok=True)
                partial.write_bytes(source.read_bytes()[:5])
                # A wrong start offset must not be appended. The downloader
                # retries from byte zero and verifies the complete object.
                class WrongRangeHandler(RangeHandler):
                    def do_GET(self) -> None:  # noqa: N802
                        WrongRangeHandler.requests.append((self.path, self.headers.get("Range")))
                        body = source.read_bytes()
                        self.send_response(206 if self.headers.get("Range") else 200)
                        if self.headers.get("Range"):
                            self.send_header("Content-Range", "bytes 0-10/16")
                            body = body[:11]
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)

                server.shutdown()
                server.server_close()
                server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), WrongRangeHandler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                payload["shards"][0]["url"] = "https://download.example.org/source.bin"
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                with patch("rivermark_benchmark.release_manifest._open_download", side_effect=local_open):
                    results = download_shards(manifest_path, root / "download")
                self.assertEqual(results[0].status, "downloaded")
                self.assertEqual((root / "download/validation/episode-001/rgb.bin").read_bytes(), source.read_bytes())
                self.assertEqual(WrongRangeHandler.requests, [("/source.bin", "bytes=5-"), ("/source.bin", None)])
            finally:
                server.shutdown()
                server.server_close()

    def test_https_requirement_rejects_local_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"x")
            payload = _manifest(source)
            self.assertIn("url_scheme", {issue.code for issue in validate_release_manifest(payload, require_https=True)})
            manifest_path = Path(temporary) / "release-manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ReleaseManifestError):
                load_release_manifest(manifest_path, require_https=True)

    def test_accounting_manifest_is_validated_and_downloadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            source.write_bytes(b"one immutable shard")
            ledger = root / "failure_ledger.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "schema": "org.rivermark.benchmark.failure-ledger.v1",
                        "attempt_id": "attempt-accounting-001",
                        "outcome": "admitted",
                        "category": "none",
                        "stage": "formal_admission",
                        "recorded_at": "2026-07-24T00:00:00Z",
                        "split": "validation",
                        "episode_id": "episode-001",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            payload = _manifest(source)
            payload["accounting"] = {
                "failure_ledger": {
                    "path": "manifests/failure_ledger.jsonl",
                    "url": ledger.as_uri(),
                    "size_bytes": ledger.stat().st_size,
                    "sha256": _sha256(ledger),
                    "schema": "org.rivermark.benchmark.failure-ledger.v1",
                    "media_type": "application/x-ndjson",
                    "compression": "none",
                    "license_status": "redistribution_cleared",
                },
                "failure_summary": {
                    "schema": "org.rivermark.benchmark.failure-ledger.v1",
                    "attempt_count": 1,
                    "admitted_count": 1,
                    "quarantined_count": 0,
                    "failed_count": 0,
                    "failure_categories": {},
                    "attempt_ids_sha256": "a" * 64,
                },
            }
            self.assertEqual(validate_release_manifest(payload), ())
            manifest_path = root / "release-manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            results = download_shards(
                manifest_path,
                root / "download",
                include_accounting=True,
            )
            self.assertEqual({result.shard_id for result in results}, {"episode-001-rgb", "release-failure-ledger"})
            self.assertEqual(
                (root / "download/manifests/failure_ledger.jsonl").read_bytes(),
                ledger.read_bytes(),
            )

    def test_accounting_count_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"x")
            payload = _manifest(source)
            payload["accounting"] = {
                "failure_ledger": {
                    "path": "manifests/failure_ledger.jsonl",
                    "url": "https://example.org/rivermark/failure_ledger.jsonl",
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                    "schema": "org.rivermark.benchmark.failure-ledger.v1",
                    "media_type": "application/x-ndjson",
                },
                "failure_summary": {
                    "schema": "org.rivermark.benchmark.failure-ledger.v1",
                    "attempt_count": 1,
                    "admitted_count": 1,
                    "quarantined_count": 1,
                    "failed_count": 0,
                    "failure_categories": {},
                    "attempt_ids_sha256": "a" * 64,
                },
            }
            self.assertIn(
                "accounting_count_mismatch",
                {issue.code for issue in validate_release_manifest(payload, require_https=True)},
            )

    def test_accounting_path_cannot_collide_with_payload_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.bin"
            source.write_bytes(b"x")
            payload = _manifest(source)
            payload["shards"][0]["path"] = "manifests/failure_ledger.jsonl"
            payload["accounting"] = {
                "failure_ledger": {
                    "path": "manifests/failure_ledger.jsonl",
                    "url": "https://example.org/rivermark/failure_ledger.jsonl",
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                    "schema": "org.rivermark.benchmark.failure-ledger.v1",
                    "media_type": "application/x-ndjson",
                },
                "failure_summary": {
                    "schema": "org.rivermark.benchmark.failure-ledger.v1",
                    "attempt_count": 1,
                    "admitted_count": 1,
                    "quarantined_count": 0,
                    "failed_count": 0,
                    "failure_categories": {},
                    "attempt_ids_sha256": "a" * 64,
                },
            }
            self.assertIn(
                "duplicate_path",
                {issue.code for issue in validate_release_manifest(payload, require_https=True)},
            )


if __name__ == "__main__":
    unittest.main()
