#!/usr/bin/env python3
"""Evaluate recorded detector predictions against explicit frame truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robocon_camera_yolo_adapter.metrics import evaluate_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL records with ground_truth and predictions")
    parser.add_argument("--output", type=Path, help="write metrics JSON to this path")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    args = parser.parse_args()
    result = evaluate_jsonl(
        args.input,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
