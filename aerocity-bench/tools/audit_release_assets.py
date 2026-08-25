"""Write immutable evidence for the fail-closed CC0 asset and USD-closure gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import content_hash, file_hash, write_json  # noqa: E402
from aerocity_bench.ordinary_config import load_ordinary_config  # noqa: E402
from aerocity_bench.supply_chain import load_official_cc0_lock  # noqa: E402


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def audit_release_assets(release_path: Path, asset_root: Path) -> dict[str, object]:
    """Validate the configured official core without publishing local asset paths."""

    config = load_ordinary_config(release_path)
    assets = config.raw["assets"]
    lock, evidence, closure = load_official_cc0_lock(
        asset_root.resolve(), str(assets["bundle"]), list(assets["allowlist"])
    )
    report: dict[str, object] = {
        "schema": "org.aerocity.bench.cc0-release-asset-audit.v1",
        "formal_score_eligible": False,
        "status": "PASS",
        "release_config_sha256": file_hash(release_path.resolve()),
        "asset_bundle": lock.bundle,
        "asset_count": len(lock.records),
        "asset_ids": sorted(lock.records),
        "registry_hash": lock.registry_hash,
        "provenance_manifest_hash": evidence.manifest_hash,
        "license_snapshot_hash": evidence.license_snapshot_hash,
        "usd_dependency_closure": closure,
        "nvidia_content_redistributed": False,
        "cf2x_redistributed": False,
        "scope": "development_release_core_validation_only",
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite asset-audit evidence: {args.output}")
    report = audit_release_assets(args.release, args.asset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"RELEASE_ASSET_AUDIT={report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
