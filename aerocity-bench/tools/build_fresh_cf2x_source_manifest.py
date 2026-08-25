"""Build a source manifest for freshly materialized, development-only layouts.

The historical calibration manifest points at the retired v12 inputs.  This
small helper binds the replacement public-boundary-audited layouts to their
private episode files without changing the task contract or selecting inputs
from replay outcomes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import content_hash, read_json, write_json  # noqa: E402

SCHEMA = "org.aerocity.bench.g2-i-scientific-audit-manifest.v1"


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts-root", type=Path, required=True)
    parser.add_argument(
        "--private-city-root",
        type=Path,
        default=None,
        help="optional root containing calibration-ancestor-XX-city.json authorities",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _relative_to_manifest(path: Path, manifest_root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(manifest_root).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must stay below the source manifest") from error


def build(
    layouts_root: Path,
    output: Path,
    *,
    private_city_root: Path | None = None,
) -> dict[str, Any]:
    layouts_root = layouts_root.resolve()
    output = output.resolve()
    private_city_root = (
        private_city_root.resolve() if private_city_root is not None else None
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source manifest: {output}")
    # The B-gate manifest builder resolves source inputs relative to this
    # manifest.  Keep the serialized paths relative to the output location,
    # rather than to ``layouts_root``: callers commonly keep the manifest one
    # directory above the materialized layouts.
    manifest_root = output.parent
    records: list[dict[str, str]] = []
    for ancestor_dir in sorted(layouts_root.glob("ancestor-*")):
        if not ancestor_dir.is_dir():
            continue
        city_candidates = sorted(
            ancestor_dir.glob("splits/calibration/city-*/scene_authority/cityspec.json")
        )
        private_candidates = sorted(
            ancestor_dir.glob("splits/calibration/city-*/evaluator_private/episodes/episode-0000.json")
        )
        if len(city_candidates) != 1 or len(private_candidates) != 1:
            raise ValueError(f"fresh layout is incomplete: {ancestor_dir}")
        city = read_json(city_candidates[0])
        if not isinstance(city, dict) or not city.get("layout_id"):
            raise ValueError(f"invalid city spec: {city_candidates[0]}")
        development = read_json(ancestor_dir / "development_layout_manifest.json")
        if (
            not isinstance(development, dict)
            or development.get("formal_score_eligible") is not False
            or development.get("public_boundary_audit", {}).get("status") != "PASS"
        ):
            raise ValueError(f"fresh layout is not public-boundary audited: {ancestor_dir}")
        record: dict[str, Any] = {
            "layout_ancestor": f"g2-i-calibration-{ancestor_dir.name}",
            "city_path": _relative_to_manifest(
                city_candidates[0], manifest_root, "public CitySpec"
            ),
            "private_episode_path": _relative_to_manifest(
                private_candidates[0], manifest_root, "private episode"
            ),
            "split_label": "calibration",
        }
        if private_city_root is not None:
            private_city = private_city_root / f"calibration-{ancestor_dir.name}-city.json"
            if not private_city.is_file():
                raise FileNotFoundError(f"private CitySpec is absent: {private_city}")
            private = read_json(private_city)
            required_private = {
                "layout_id",
                "layout_hash",
                "split",
                "spawn_grammar",
                "family_private",
                "generation_seed",
            }
            if not isinstance(private, dict) or not required_private <= set(private):
                raise ValueError(f"private CitySpec is incomplete: {private_city}")
            if (
                private["layout_id"] != city["layout_id"]
                or private["layout_hash"] != city.get("layout_hash")
                or private["split"] != "calibration"
            ):
                raise ValueError(f"private CitySpec does not bind the fresh layout: {private_city}")
            record["private_city_source_path"] = _relative_to_manifest(
                private_city, manifest_root, "private CitySpec"
            )
            record["private_city_source_sha256"] = content_hash(private)
        records.append(record)
    if len(records) != 3:
        raise ValueError(f"expected exactly three fresh calibration layouts, found {len(records)}")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "purpose": "fresh-public-boundary-audited-cf2x-b-gate-source",
        "formal_score_eligible": False,
        "self_method_results_used": False,
        "development_splits": ["calibration"],
        "accepted_ancestor_count": len(records),
        "records": records,
    }
    payload["manifest_hash"] = content_hash(payload)
    write_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    build(
        args.layouts_root,
        args.output,
        private_city_root=args.private_city_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
