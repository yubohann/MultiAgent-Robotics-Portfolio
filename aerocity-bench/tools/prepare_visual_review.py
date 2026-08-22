"""Create a deterministic non-scoring target overlay from an authority layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerocity_bench.builder_v3 import validate_ordinary_release
from aerocity_bench.canonical import read_json, write_json
from aerocity_bench.ordinary_config import load_ordinary_config
from aerocity_bench.targets_v3 import sample_visual_review_episode_v3


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("authority_root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--layout-id")
    parser.add_argument("--target-count", type=int, default=32)
    parser.add_argument("--process", default="height_stratified")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    root = args.authority_root.resolve()
    validate_ordinary_release(root)
    config = load_ordinary_config(root / "authority_private" / "release_config.json")
    candidates = sorted((root / "splits" / args.split).glob("*/scene_authority/cityspec.json"))
    if args.layout_id:
        candidates = [path for path in candidates if path.parent.parent.name == args.layout_id]
    if len(candidates) != 1:
        raise ValueError(f"layout selection must resolve once, found {len(candidates)}")
    city_path = candidates[0]
    layout_root = city_path.parent.parent
    city = read_json(city_path)
    support = read_json(layout_root / "evaluator_private" / "support_sites.json")
    source_episode_path = sorted((layout_root / "evaluator_private" / "episodes").glob("*.json"))[0]
    source_episode = read_json(source_episode_path)
    review = sample_visual_review_episode_v3(
        config,
        city,
        list(support["support_sites"]),
        list(source_episode["starts"]),
        target_count=args.target_count,
        process_name=args.process,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, review)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.resolve()),
                "layout_id": review["layout_id"],
                "target_count": review["target_count"],
                "audit": review["audit"],
                "formal_score_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
