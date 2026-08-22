"""Run target-safe G2-I geometry, budget, and statistical leakage audits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from aerocity_bench.atlas_audit import audit_inspection_atlas
from aerocity_bench.atlas_leakage import audit_atlas_leakage
from aerocity_bench.canonical import content_hash, read_json, write_json
from aerocity_bench.inspection_atlas import (
    compile_inspection_atlas,
    validate_public_mission_sector,
)
from aerocity_bench.ordinary_config import FORMAL_SPLITS, load_ordinary_config

MANIFEST_SCHEMA = "org.aerocity.bench.g2-i-scientific-audit-manifest.v1"
REPORT_SCHEMA = "org.aerocity.bench.g2-i-scientific-gate-report.v1"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=64)
    args = parser.parse_args(argv)
    if args.permutations < 16:
        parser.error("--permutations must be at least 16")
    return args


def _local_path(root: Path, value: object, name: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a relative path below the manifest directory")
    resolved = (root / relative).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"{name} escapes the manifest directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {relative}")
    return resolved


def run_audit(manifest_path: Path, *, permutation_count: int) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported G2-I scientific audit manifest")
    required = {
        "schema",
        "purpose",
        "self_method_results_used",
        "release_config_path",
        "records",
    }
    allowed = required | {
        "accepted_ancestor_count",
        "formal_score_eligible",
        "manifest_hash",
        "sector_policy",
        "development_splits",
    }
    if not required.issubset(manifest) or set(manifest) - allowed:
        raise ValueError("G2-I scientific audit manifest fields differ")
    if manifest["purpose"] != "method-independent-task-calibration":
        raise ValueError("audit purpose must be method-independent task calibration")
    if manifest["self_method_results_used"] is not False:
        raise ValueError("task calibration cannot consume the self method's results")
    if "manifest_hash" in manifest:
        payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if content_hash(payload) != manifest["manifest_hash"]:
            raise ValueError("G2-I scientific audit manifest hash mismatch")
    root = manifest_path.parent
    config = load_ordinary_config(
        _local_path(root, manifest["release_config_path"], "release_config_path")
    )
    records_node = manifest["records"]
    if not isinstance(records_node, list) or not records_node:
        raise ValueError("G2-I scientific audit manifest has no records")
    leakage_records = []
    geometry_by_hash: dict[str, dict[str, Any]] = {}
    decision_inputs = []
    for node in records_node:
        if not isinstance(node, dict) or set(node) != {
            "city_path",
            "private_episode_path",
            "layout_ancestor",
            "split_label",
        }:
            raise ValueError("G2-I scientific audit record fields differ")
        split_label = str(node["split_label"])
        if split_label in FORMAL_SPLITS:
            raise ValueError("scientific calibration must not inspect a formal split")
        if split_label not in {"train", "validation", "calibration", "development"}:
            raise ValueError("scientific calibration record uses an unknown development split")
        city_path = _local_path(root, node["city_path"], "city_path")
        episode_path = _local_path(
            root, node["private_episode_path"], "private_episode_path"
        )
        city = read_json(city_path)
        episode = read_json(episode_path)
        if not isinstance(city, dict) or not isinstance(episode, dict):
            raise ValueError("city and private episode inputs must be objects")
        if split_label != "development" and str(city.get("split")) != split_label:
            raise ValueError("manifest split label differs from the CitySpec split")
        if (
            episode.get("layout_id") != city.get("layout_id")
            or episode.get("layout_hash") != city.get("layout_hash")
        ):
            raise ValueError("private episode is not bound to its audit city")
        atlas = compile_inspection_atlas(city, config.raw["execution_contract"])
        mission_sector = episode.get("mission_sector")
        if mission_sector is not None:
            if not isinstance(mission_sector, dict):
                raise ValueError("private episode mission sector must be an object")
            validate_public_mission_sector(
                mission_sector,
                atlas,
                episode["starts"],
                config.raw["execution_contract"],
            )
            if episode.get("mission_sector_hash") != mission_sector.get("sector_hash"):
                raise ValueError("private episode mission-sector binding differs")
        atlas_hash = str(atlas["atlas_hash"])
        if atlas_hash not in geometry_by_hash:
            geometry_by_hash[atlas_hash] = audit_inspection_atlas(
                city,
                atlas,
                config.raw["execution_contract"],
                fleet_count=config.fleet_count,
                episode_duration_s=float(
                    config.raw["execution_contract"]["episode"]["duration_s"]
                ),
            )
        leakage_records.append(
            {
                "atlas": atlas,
                "private_episode": episode,
                "layout_ancestor": str(node["layout_ancestor"]),
                "split_label": split_label,
            }
        )
        decision_inputs.append(
            {
                "city_sha256": content_hash(city),
                "private_episode_sha256": content_hash(episode),
                "atlas_hash": atlas_hash,
                "layout_ancestor_hash": content_hash(str(node["layout_ancestor"])),
            }
        )
    leakage = audit_atlas_leakage(
        leakage_records,
        execution_contract=config.raw["execution_contract"],
        permutation_count=permutation_count,
    )
    ancestors = {str(record["layout_ancestor"]) for record in leakage_records}
    cpu_pass = all(
        report["cpu_geometry_status"] == "PASS_CPU"
        for report in geometry_by_hash.values()
    )
    leakage_status = leakage["paired_label_probe"]["status"]
    sector_process_status = leakage["sector_process_label_probe"]["status"]
    sector_probe_required = manifest.get("sector_policy") is not None
    sector_probe_pass = sector_process_status == "PASS_NO_DETECTED_SIGNAL"
    sampling_policy_frozen = all(
        report["remaining_gates"][
            "sampling_policy_frozen_by_method_independent_calibration"
        ]
        is True
        for report in geometry_by_hash.values()
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "formal_score_eligible": False,
        "overall_status": "FORMAL_NO_GO",
        "manifest_hash": content_hash(manifest),
        "decision_input_set_hash": content_hash(decision_inputs),
        "method_independence": {
            "purpose": manifest["purpose"],
            "self_method_results_used": False,
            "contract_freeze_allowed": False,
        },
        "aggregate": {
            "record_count": len(leakage_records),
            "unique_atlas_count": len(geometry_by_hash),
            "independent_ancestor_count": len(ancestors),
        },
        "geometry_reports": [geometry_by_hash[key] for key in sorted(geometry_by_hash)],
        "leakage_report": leakage,
        "gate_checks": {
            "cpu_geometry_all_pass": cpu_pass,
            "at_least_three_independent_ancestors": len(ancestors) >= 3,
            "paired_leakage_probe_pass": leakage_status == "PASS_NO_DETECTED_SIGNAL",
            "sector_process_leakage_probe_pass": (
                sector_probe_pass
            ),
            "sampling_policy_frozen": sampling_policy_frozen,
            "native_cf2x_reachability_pass": False,
            "public_four_vehicle_l1_closure_pass": False,
            "l0_l1_ranking_consistency_measured": False,
        },
        "next_authorized_step": (
            "FIX_CPU_OR_LEAKAGE_FAILURES"
            if (
                not cpu_pass
                or leakage_status != "PASS_NO_DETECTED_SIGNAL"
                or (sector_probe_required and not sector_probe_pass)
            )
            else "RUN_NATIVE_CF2X_SHORTLIST_AND_PUBLIC_L1_CALIBRATION"
        ),
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite scientific evidence: {args.output}")
    report = run_audit(args.manifest, permutation_count=args.permutations)
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
