"""Attest the public cityspec hash in an already materialized development layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from aerocity_bench.canonical import content_hash, read_json, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.layouts_root.resolve()
    for ancestor in sorted(root.glob("ancestor-*")):
        manifest_path = ancestor / "development_layout_manifest.json"
        city_paths = sorted(
            ancestor.glob("splits/calibration/city-*/scene_authority/cityspec.json")
        )
        if not manifest_path.is_file() or len(city_paths) != 1:
            raise ValueError(f"incomplete materialized layout: {ancestor}")
        manifest = read_json(manifest_path)
        city = read_json(city_paths[0])
        if not isinstance(manifest, dict) or not isinstance(city, dict):
            raise ValueError(f"invalid layout evidence: {ancestor}")
        manifest["materialized_cityspec_sha256"] = content_hash(city)
        write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
