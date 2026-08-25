"""Materialize a hash-bound, development-only authority package for L2 review.

The static scene cohort establishes that the deterministic development layouts
are admissible.  Isaac review must load those exact layouts, not an unrelated
one-layout authority release.  This tool builds that separate review input and
binds it to the cohort's public-safe summary before a capture batch is allowed
to use it.  It never includes a formal split and never creates formal scores.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.builder_v3 import (  # noqa: E402
    build_ordinary_release,
    validate_ordinary_release,
)
from aerocity_bench.canonical import content_hash, read_json, write_json  # noqa: E402
from aerocity_bench.ordinary_config import OrdinaryReleaseConfig, load_ordinary_config  # noqa: E402
from aerocity_bench.scene_audit import (  # noqa: E402
    DEVELOPMENT_AUDIT_SPLITS,
    SCENE_AUDIT_COHORT_SCHEMA,
)

SCHEMA = "org.aerocity.bench.development-l2-review-authority.v1"
DEVELOPMENT_SOURCE = "UNCOMMITTED-DEVELOPMENT"


def development_l2_review_config(
    base: OrdinaryReleaseConfig, *, layouts_per_split: int
) -> OrdinaryReleaseConfig:
    """Derive a review-only config without changing the task contract."""

    if not 4 <= layouts_per_split <= 6:
        raise ValueError("layouts-per-split must be in [4, 6] for a 12--18 city cohort")
    raw = copy.deepcopy(base.raw)
    raw["release_kind"] = "CUSTOM"
    raw["release_version"] = (
        f"{base.version}-development-l2-review-{layouts_per_split}x3"
    )
    for split in DEVELOPMENT_AUDIT_SPLITS:
        raw["split_counts"][split] = layouts_per_split
    return OrdinaryReleaseConfig(path=base.path, raw=raw, config_hash=content_hash(raw))


def _cohort_members(
    summary: dict[str, Any], *, layouts_per_split: int
) -> dict[tuple[str, int], str]:
    if summary.get("schema") != SCENE_AUDIT_COHORT_SCHEMA:
        raise ValueError("scene-audit summary schema is invalid")
    if summary.get("status") != "PASS" or summary.get("formal_score_eligible") is not False:
        raise ValueError("scene-audit summary must be a development-only PASS")
    sampling = summary.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("layouts_per_split") != layouts_per_split:
        raise ValueError("scene-audit cohort layout count differs from L2 review request")
    members = summary.get("members")
    if not isinstance(members, list):
        raise ValueError("scene-audit summary lacks members")
    expected = {
        (split, index)
        for split in DEVELOPMENT_AUDIT_SPLITS
        for index in range(layouts_per_split)
    }
    received: dict[tuple[str, int], str] = {}
    for member in members:
        if not isinstance(member, dict):
            raise ValueError("scene-audit member is invalid")
        split = str(member.get("split"))
        index = member.get("index")
        layout_hash = member.get("layout_hash")
        if split not in DEVELOPMENT_AUDIT_SPLITS or not isinstance(index, int):
            raise ValueError("scene-audit member is outside development cohort")
        if not isinstance(layout_hash, str) or len(layout_hash) != 64:
            raise ValueError("scene-audit member lacks layout hash")
        key = (split, index)
        if key in received:
            raise ValueError("scene-audit cohort repeats a split/index member")
        received[key] = layout_hash
    if set(received) != expected:
        raise ValueError("scene-audit cohort members do not match requested review cohort")
    return received


def build_development_l2_review_authority(
    *,
    base_config_path: Path,
    asset_root: Path,
    scene_audit_summary_path: Path,
    output: Path,
    layouts_per_split: int,
) -> dict[str, Any]:
    """Build and attest a development-only authority input for L2 capture."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite L2 review evidence: {output}")
    base = load_ordinary_config(base_config_path.resolve())
    derived = development_l2_review_config(base, layouts_per_split=layouts_per_split)
    summary = read_json(scene_audit_summary_path.resolve())
    if not isinstance(summary, dict):
        raise ValueError("scene-audit summary must be an object")
    expected_members = _cohort_members(summary, layouts_per_split=layouts_per_split)
    if summary.get("release_config_hash") != base.config_hash:
        raise ValueError("scene-audit summary uses a different base release configuration")

    output.mkdir(parents=True)
    authority = output / "authority"
    try:
        build_ordinary_release(
            derived,
            asset_root.resolve(),
            authority,
            DEVELOPMENT_AUDIT_SPLITS,
            source_commit=DEVELOPMENT_SOURCE,
        )
        validation = validate_ordinary_release(authority)
        index = read_json(authority / "release_index.json")
        observed_members = {
            (str(layout["split"]), position): str(layout["layout_hash"])
            for split in DEVELOPMENT_AUDIT_SPLITS
            for position, layout in enumerate(
                item for item in index["layouts"] if item["split"] == split
            )
        }
        if observed_members != expected_members:
            raise ValueError("materialized review layouts differ from the static scene cohort")
        receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "PASS",
            "scope": "development_only_l2_visual_review_input",
            "formal_score_eligible": False,
            "source_commit": DEVELOPMENT_SOURCE,
            "authority_root": str(authority),
            "authority_release_index_hash": index["release_index_hash"],
            "authority_release_config_hash": derived.config_hash,
            "base_release_config_hash": base.config_hash,
            "scene_audit_summary": str(scene_audit_summary_path.resolve()),
            "scene_audit_summary_hash": summary["report_hash"],
            "selected_splits": list(DEVELOPMENT_AUDIT_SPLITS),
            "layouts_per_split": layouts_per_split,
            "layout_hashes_match_static_cohort": True,
            "authority_validation_status": validation["status"],
        }
        receipt["receipt_hash"] = content_hash(receipt)
        write_json(output / "materialization_receipt.json", receipt)
        return receipt
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "FAIL",
            "scope": "development_only_l2_visual_review_input",
            "formal_score_eligible": False,
            "source_commit": DEVELOPMENT_SOURCE,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        failure["receipt_hash"] = content_hash(failure)
        write_json(output / "materialization_failure.json", failure)
        raise


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--scene-audit-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts-per-split", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    receipt = build_development_l2_review_authority(
        base_config_path=args.base_config,
        asset_root=args.asset_root,
        scene_audit_summary_path=args.scene_audit_summary,
        output=args.output,
        layouts_per_split=args.layouts_per_split,
    )
    print(
        f"L2_DEVELOPMENT_AUTHORITY={receipt['status']} "
        f"layouts={receipt['layouts_per_split'] * len(DEVELOPMENT_AUDIT_SPLITS)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
