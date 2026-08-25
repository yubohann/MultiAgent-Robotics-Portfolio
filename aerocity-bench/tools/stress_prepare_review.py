"""Build many development layouts and stress the resumable review scheduler."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from aerocity_bench.builder_v3 import build_ordinary_release, validate_ordinary_release
from aerocity_bench.canonical import content_hash
from aerocity_bench.cli import _capture_review_batch
from aerocity_bench.ordinary_config import OrdinaryReleaseConfig, load_ordinary_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-release", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--authority-output", type=Path, required=True)
    parser.add_argument("--batch-output", type=Path, required=True)
    parser.add_argument("--layout-count", type=int, default=100)
    parser.add_argument("--target-count", type=int, default=8)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _stress_config(base_path: Path, layout_count: int) -> OrdinaryReleaseConfig:
    if layout_count < 3:
        raise ValueError("layout-count must be at least three")
    base = load_ordinary_config(base_path)
    raw = copy.deepcopy(base.raw)
    validation_count = max(1, layout_count // 10)
    calibration_count = max(1, layout_count // 10)
    train_count = layout_count - validation_count - calibration_count
    raw["release_kind"] = "CUSTOM"
    raw["release_version"] = f"prepare-scale-stress-{layout_count}"
    raw["master_seed"] = int(raw["master_seed"]) + layout_count
    raw["split_counts"].update(
        {
            "train": train_count,
            "validation": validation_count,
            "calibration": calibration_count,
        }
    )
    return OrdinaryReleaseConfig(
        path=base.path,
        raw=raw,
        config_hash=content_hash(raw),
    )


def main() -> int:
    args = _arguments()
    config = _stress_config(args.base_release, args.layout_count)
    started = time.perf_counter()
    if args.authority_output.exists():
        validate_ordinary_release(args.authority_output)
    else:
        build_ordinary_release(
            config,
            args.asset_root,
            args.authority_output,
            ("train", "validation", "calibration"),
            source_commit=args.source_commit,
        )
    authority_ready_s = time.perf_counter() - started
    batch_started = time.perf_counter()
    report = _capture_review_batch(
        argparse.Namespace(
            authority_root=args.authority_output,
            splits=["train", "validation", "calibration"],
            target_count=args.target_count,
            process="height_stratified",
            output=args.batch_output,
            isaac_python=None,
            width=640,
            height=480,
            timeout_s=240.0,
            max_attempts=2,
            limit=None,
            resume=args.resume,
            prepare_only=True,
        )
    )
    summary = {
        "schema": "org.aerocity.bench.prepare-scale-stress.v1",
        "status": report["status"],
        "requested_layout_count": args.layout_count,
        "prepared_layout_count": report["passed_layout_count"],
        "authority_ready_s": round(authority_ready_s, 3),
        "batch_prepare_s": round(time.perf_counter() - batch_started, 3),
        "resume": bool(args.resume),
        "authority_root": str(args.authority_output.resolve()),
        "batch_root": str(args.batch_output.resolve()),
        "batch_report_hash": report["report_hash"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
