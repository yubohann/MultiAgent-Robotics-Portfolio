"""Authenticated, resumable HM3D v0.2 downloader with no secret logging."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import write_json_atomic


def _user_environment(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if os.name != "nt":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except FileNotFoundError:
        return None
    return str(value) if value else None


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "source",
        "license_url",
        "asset_version",
        "credentials",
        "selections",
        "resources",
        "excluded_as_redundant_for_isaac",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("HM3D download selection fields mismatch")
    if payload["schema_version"] != "hm3d-v0.2-download-selection-v1":
        raise ValueError("HM3D download selection schema mismatch")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _authorization(token_id: str, token_secret: str) -> str:
    encoded = base64.b64encode(f"{token_id}:{token_secret}".encode()).decode("ascii")
    return f"Basic {encoded}"


def _download(
    *,
    url: str,
    destination: Path,
    authorization: str,
    advertised_bytes: int,
) -> dict[str, Any]:
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    request.add_header("Authorization", authorization)
    request.add_header("User-Agent", "aerocity-method-hm3d-audit/1.0")
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            raise PermissionError("Matterport API rejected the token ID/secret") from error
        raise
    status = int(getattr(response, "status", response.getcode()))
    append = existing > 0 and status == 206
    if existing > 0 and not append:
        existing = 0
    mode = "ab" if append else "wb"
    downloaded = existing
    last_report = time.monotonic()
    with response, partial.open(mode) as stream:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            stream.write(block)
            downloaded += len(block)
            now = time.monotonic()
            if now - last_report >= 30.0:
                percent = 100.0 * downloaded / max(advertised_bytes, 1)
                print(
                    f"{destination.name}: {downloaded / 2**30:.2f} GiB "
                    f"({percent:.1f}% of advertised size)",
                    flush=True,
                )
                last_report = now
    partial.replace(destination)
    return {
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "elapsed_s": time.monotonic() - started,
        "resumed_from_bytes": existing,
        "http_status": status,
    }


def _tar_audit(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["tar", "-tf", str(path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    members = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not members:
        raise ValueError(f"downloaded archive failed tar audit: {path.name}")
    return {
        "tar_readable": True,
        "member_count": len(members),
        "first_members": members[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "external" / "hm3d_v0.2_download_selection.json",
    )
    parser.add_argument(
        "--selection",
        choices=("development", "formal_exploration_geometry"),
        default="development",
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reserve-free-gib", type=float, default=24.0)
    args = parser.parse_args()
    config = _load_config(args.config)
    credentials = config["credentials"]
    token_id = _user_environment(credentials["token_id_environment_variable"])
    token_secret = _user_environment(credentials["token_secret_environment_variable"])
    if not token_id or not token_secret:
        raise RuntimeError(
            "HM3D API credentials are absent; set the documented user environment variables"
        )
    authorization = _authorization(token_id, token_secret)
    names = config["selections"][args.selection]
    resources = config["resources"]
    required = sum(int(resources[name]["advertised_bytes"]) for name in names)
    args.destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.destination).free
    reserve = int(args.reserve_free_gib * 2**30)
    existing = sum(
        (args.destination / name).stat().st_size
        for name in names
        if (args.destination / name).is_file()
    )
    if free + existing < required + reserve:
        raise OSError("insufficient disk space for selection plus frozen reserve")

    rows = []
    for name in names:
        destination = args.destination / name
        resource = resources[name]
        if destination.is_file():
            measurement = {
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "elapsed_s": 0.0,
                "resumed_from_bytes": destination.stat().st_size,
                "http_status": None,
            }
        else:
            measurement = _download(
                url=resource["url"],
                destination=destination,
                authorization=authorization,
                advertised_bytes=int(resource["advertised_bytes"]),
            )
        rows.append(
            {
                "name": name,
                "url": resource["url"],
                "kind": resource["kind"],
                "split": resource["split"],
                "path": str(destination.resolve()),
                **measurement,
                **_tar_audit(destination),
            }
        )
    report = {
        "schema_version": "hm3d-v0.2-download-report-v1",
        "asset_version": config["asset_version"],
        "selection": args.selection,
        "source": config["source"],
        "license_url": config["license_url"],
        "credentials_present": True,
        "credentials_recorded": False,
        "archives": rows,
        "all_tar_readable": all(row["tar_readable"] for row in rows),
    }
    write_json_atomic(args.report, report)
    print(json.dumps({"status": "PASS", "archives": len(rows), "report": str(args.report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
