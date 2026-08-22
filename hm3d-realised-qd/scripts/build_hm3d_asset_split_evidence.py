"""Lock the locally installed HM3D v0.2 assets and a scene-disjoint split.

The script deliberately records asset paths and hashes only.  It never copies
or converts HM3D assets into the repository.  The split follows the practical
protocol for this project:

* official ``train`` is deterministically divided into train and development
  validation scenes;
* scenes already used for local visual/runtime development are quarantined in
  validation; and
* the remaining official ``val`` scenes are a frozen, untouched final test
  partition.

This is a source/license audit for P01 and P05, not a simulator result.  It
must run before any new HM3D development job uses the locked scene list.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.evaluation.hm3d_preflight import (
    METHOD_CORE,
    PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
    PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    load_preflight_protocol,
)

DATASET_VERSION = "hm3d-v0.2"
LICENSE_ID = "matterport-academic-use-model-data-eula"
SOURCE_URL = "https://github.com/matterport/habitat-matterport-3dresearch"
TERMS_URL = "https://matterport.com/matterport-end-user-license-agreement-academic-use-model-data"
DEFAULT_SPLIT_SEED = "aerocity-hm3d-scene-split-20260801-v1"
# Local previews and A-B-A development work have used these official val scenes.
# Keeping the complete minival range out of final test is conservative: minival is
# a published subset of val and is already installed on this machine.
DEFAULT_QUARANTINED_VAL_PREFIXES = tuple(f"{index:05d}-" for index in range(800, 810))
# This scene was used by the measured A-B-A/calibration/collision development audit
# before this asset lock existed, despite not belonging to minival.
DEFAULT_QUARANTINED_VAL_SCENE_IDS = frozenset({"00821-eF36g7L6Z9M"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _scene_files(root: Path) -> dict[str, Path]:
    """Return canonical scene IDs to GLBs under one official split root."""

    if not root.is_dir():
        raise FileNotFoundError(f"HM3D split directory is missing: {root}")
    rows: dict[str, Path] = {}
    for path in sorted(root.glob("*/*.glb")):
        scene_id = path.parent.name
        if "-" not in scene_id or path.stem != scene_id.split("-", 1)[1]:
            raise ValueError(f"unexpected HM3D GLB layout: {path}")
        if scene_id in rows:
            raise ValueError(f"duplicate HM3D scene ID: {scene_id}")
        rows[scene_id] = path.resolve()
    if not rows:
        raise ValueError(f"no HM3D GLBs found in {root}")
    return rows


def _validation_train_ids(scene_ids: list[str], count: int, seed: str) -> set[str]:
    if not 1 <= count < len(scene_ids):
        raise ValueError("validation scene count must be within the official train split")
    ranked = sorted(
        scene_ids,
        key=lambda scene_id: hashlib.sha256(f"{seed}:{scene_id}".encode()).hexdigest(),
    )
    return set(ranked[:count])


def _quarantined_val_ids(scene_ids: list[str], prefixes: tuple[str, ...]) -> set[str]:
    prefix_selected = {scene_id for scene_id in scene_ids if scene_id.startswith(prefixes)}
    explicit_selected = set(scene_ids) & DEFAULT_QUARANTINED_VAL_SCENE_IDS
    if explicit_selected != DEFAULT_QUARANTINED_VAL_SCENE_IDS:
        missing = sorted(DEFAULT_QUARANTINED_VAL_SCENE_IDS - explicit_selected)
        raise ValueError(f"known development scene is missing from official val: {missing}")
    selected = prefix_selected | explicit_selected
    if not prefix_selected:
        raise ValueError("no published minival scenes were found in official val")
    return selected


def _artifact(*, phase_id: str, kind: str, origin: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
        "phase_id": phase_id,
        "kind": kind,
        "origin": origin,
        "measured": True,
        "synthetic": False,
        "denominator_complete": True,
        "payload": payload,
    }


def build_payloads(
    *,
    train_root: Path,
    val_root: Path,
    conversion_tool: Path,
    license_record_path: Path,
    train_validation_count: int,
    split_seed: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build P01/P05 envelopes and a partial preflight manifest without writing."""

    train_files = _scene_files(train_root.resolve())
    val_files = _scene_files(val_root.resolve())
    overlap = set(train_files) & set(val_files)
    if overlap:
        raise ValueError(f"official train/val asset IDs overlap: {sorted(overlap)[:3]}")
    if not conversion_tool.is_file():
        raise FileNotFoundError(f"conversion tool is missing: {conversion_tool}")
    if not license_record_path.is_file():
        raise FileNotFoundError(f"license record is missing: {license_record_path}")

    validation_train = _validation_train_ids(
        sorted(train_files), train_validation_count, split_seed
    )
    quarantined_val = _quarantined_val_ids(sorted(val_files), DEFAULT_QUARANTINED_VAL_PREFIXES)
    scenes: list[dict[str, str]] = []
    for scene_id, path in sorted(train_files.items()):
        scenes.append(
            {
                "scene_id": scene_id,
                "split": "validation" if scene_id in validation_train else "train",
                "asset_origin": "official_hm3d",
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    for scene_id, path in sorted(val_files.items()):
        scenes.append(
            {
                "scene_id": scene_id,
                "split": "validation" if scene_id in quarantined_val else "test",
                "asset_origin": "official_hm3d",
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    assignments = [
        {"scene_id": row["scene_id"], "split": row["split"], "asset_sha256": row["sha256"]}
        for row in sorted(scenes, key=lambda row: row["scene_id"])
    ]
    split_hash = canonical_sha256(assignments)
    public_contract_placeholder = hashlib.sha256(
        b"P04-not-run-public-observation-contract-not-yet-locked"
    ).hexdigest()
    evaluation_denominator_placeholder = hashlib.sha256(
        b"P04-not-run-evaluator-denominator-not-yet-locked"
    ).hexdigest()
    p01_payload = {
        "evidence_class": "source_license_audit",
        "dataset_version": DATASET_VERSION,
        "license_id": LICENSE_ID,
        "license_record_path": str(license_record_path.resolve()),
        "license_sha256": _sha256(license_record_path),
        "source_url": SOURCE_URL,
        "raw_assets_redistributed": False,
        "repository_included": False,
        "conversion_tool": {
            "tool_id": "aerocity-hm3d-glb-to-isaac-usd",
            "version": "2026-08-01",
            "sha256": _sha256(conversion_tool),
        },
        "scenes": scenes,
    }
    # P05 becomes valid only after P04 supplies the actual public-observation
    # and evaluator hashes.  It
    # is nevertheless recorded now so the asset partition itself is immutable.
    p05_draft = {
        "evidence_class": "source_license_audit",
        "official_split_provenance": (
            "hm3d-v0.2-official-train-val-with-published-minival-quarantine-"
            "and-deterministic-scene-holdout-v1"
        ),
        "dataset_version": DATASET_VERSION,
        "scene_assignments": assignments,
        "split_manifest_sha256": split_hash,
        "public_contract_sha256": public_contract_placeholder,
        "evaluation_denominator_sha256": evaluation_denominator_placeholder,
        "episode_seed_manifest_sha256": hashlib.sha256(
            f"{split_seed}:episode-seeds-not-yet-frozen".encode()
        ).hexdigest(),
        "difficulty_distribution_sha256": hashlib.sha256(
            f"{split_seed}:difficulty-not-yet-frozen".encode()
        ).hexdigest(),
        "run_partition": "development",
        "test_used_for_development": False,
        "test_access_count_before_freeze": 0,
    }
    protocol = load_preflight_protocol(
        ROOT / "configs" / "external" / "hm3d_formal_preflight_protocol.json"
    )
    p01_envelope = _artifact(
        phase_id="P01",
        kind="asset_lock",
        origin="source_license_audit",
        payload=p01_payload,
    )
    p05_envelope = _artifact(
        phase_id="P05",
        kind="scene_split_freeze",
        origin="source_license_audit",
        payload=p05_draft,
    )
    partial_evidence = {
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
        "protocol_hash": protocol.protocol_hash,
        "requested_gate": "formal_experiment_start",
        "method_core": METHOD_CORE,
        "aerocity_bench_accesses": [],
        "artifacts": [],
    }
    return p01_envelope, p05_envelope, partial_evidence


def _write_license_record(path: Path, *, train_root: Path, val_root: Path) -> None:
    content = "\n".join(
        [
            "# HM3D v0.2 local access and licensing record",
            "",
            "Date: 2026-08-01",
            "",
            "- Dataset: Habitat-Matterport 3D Research Dataset (HM3D) v0.2.",
            f"- Official repository: {SOURCE_URL}",
            f"- Terms URL: {TERMS_URL}",
            "- Scope: academic, non-commercial use only; no raw or converted HM3D",
            "  asset is committed or redistributed by aerocity-method.",
            "- Local installation was obtained by an authorized local operator through",
            "  the official Matterport distribution API.",
            f"- Official train GLB root: {train_root.resolve()}",
            f"- Official val GLB root: {val_root.resolve()}",
            "- This record is an audit pointer, not a replacement for Matterport's terms.",
            "  Each operator must independently retain and comply with the accepted agreement.",
            "",
        ]
    )
    if path.exists():
        raise FileExistsError(f"refusing to overwrite license record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-glb-root", type=Path, required=True)
    parser.add_argument("--val-glb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conversion-tool",
        type=Path,
        default=ROOT / "scripts" / "convert_hm3d_glb_to_collision_usd.py",
    )
    parser.add_argument("--train-validation-count", type=int, default=80)
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse evidence directory: {output_dir}")
    train_root = args.train_glb_root.expanduser().resolve()
    val_root = args.val_glb_root.expanduser().resolve()
    license_record = output_dir / "license" / "HM3D_v0.2_access_record.md"
    _write_license_record(license_record, train_root=train_root, val_root=val_root)
    p01, p05, partial = build_payloads(
        train_root=train_root,
        val_root=val_root,
        conversion_tool=args.conversion_tool.expanduser().resolve(),
        license_record_path=license_record,
        train_validation_count=args.train_validation_count,
        split_seed=args.split_seed,
    )
    p01_path = output_dir / "artifacts" / "P01_asset_lock.json"
    p05_path = output_dir / "drafts" / "P05_scene_split_freeze_draft.json"
    _write_json_new(p01_path, p01)
    _write_json_new(p05_path, p05)
    partial["artifacts"] = [
        {
            "phase_id": "P01",
            "kind": "asset_lock",
            "origin": "source_license_audit",
            "path": str(p01_path),
            "sha256": _sha256(p01_path),
        }
    ]
    manifest_path = output_dir / "partial_preflight_evidence.json"
    _write_json_new(manifest_path, partial)
    counts = {split: 0 for split in ("train", "validation", "test")}
    for row in p01["payload"]["scenes"]:
        counts[row["split"]] += 1
    summary = {
        "status": "P01_ASSET_LOCK_READY_P05_DRAFT_AWAITS_P04_PUBLIC_OBSERVATION",
        "output_dir": str(output_dir),
        "asset_counts": counts,
        "p01_artifact": str(p01_path),
        "p05_draft": str(p05_path),
        "partial_evidence": str(manifest_path),
        "test_scenes_are_untouched_official_val_after_minival_quarantine": True,
    }
    _write_json_new(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
