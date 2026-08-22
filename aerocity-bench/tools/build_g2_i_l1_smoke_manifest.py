"""Precommit one public four-CF2X G2-I calibration replay."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from aerocity_bench.canonical import content_hash, read_json, write_json

PANEL_SCHEMA = "org.aerocity.bench.g2-i-l1-smoke-panel.v1"
MANIFEST_SCHEMA = "org.aerocity.bench.g2-i-l1-measurement-evidence-manifest.v1"
PROTOCOL_PATH = Path("configs/g2i-measurement-claim-protocol-v1.json")
_FORBIDDEN_METHOD_MARKERS = ("oracle", "witness", "private", "fixture")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--layout-ancestor", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--episode-name", default="episode-0000.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--purpose", default="single_ancestor_four_cf2x_public_policy_smoke")
    return parser.parse_args(argv)


def _relative_path(root: Path, path: Path, field: str) -> str:
    try:
        relative = Path(os.path.relpath(path.resolve(), root.resolve()))
    except ValueError as exc:
        raise ValueError(f"{field} must share a filesystem volume with output root") from exc
    if relative == Path(".") or not relative.parts:
        raise ValueError(f"{field} cannot be the output root")
    return relative.as_posix()


def _public_episode(layout_root: Path, episode_name: str) -> dict[str, Any]:
    path = layout_root / "method_public" / "episodes" / episode_name
    episode = read_json(path)
    if not isinstance(episode, dict):
        raise ValueError("public episode must be an object")
    if episode.get("schema") != "org.aerocity.bench.episode-public.ordinary.v1":
        raise ValueError("public episode schema is unsupported")
    if (
        episode.get("target_count_public") is not False
        or episode.get("target_process_public") is not False
    ):
        raise ValueError("public episode exposes target information")
    return episode


def build_manifest(
    *,
    layout_root: Path,
    release_config: Path,
    layout_ancestor: str,
    method_id: str,
    episode_name: str,
    output_root: Path,
    purpose: str,
) -> dict[str, Any]:
    if not layout_root.is_dir():
        raise FileNotFoundError(f"layout root is missing: {layout_root}")
    if not release_config.is_file():
        raise FileNotFoundError(f"release config is missing: {release_config}")
    if not layout_ancestor or any(character in layout_ancestor for character in "/\\"):
        raise ValueError("layout ancestor identifier is invalid")
    if not method_id or any(marker in method_id.casefold() for marker in _FORBIDDEN_METHOD_MARKERS):
        raise ValueError("private or fixture methods cannot enter a public smoke panel")
    if not episode_name or Path(episode_name).name != episode_name:
        raise ValueError("episode name must be a single file name")
    episode = _public_episode(layout_root, episode_name)
    task = read_json(layout_root / "method_public" / "task_spec.json")
    if not isinstance(task, dict) or task.get("task_track") != "G2-I":
        raise ValueError("smoke panel requires a public G2-I task")
    if task.get("inspection_prior_level") != "full-cells":
        raise ValueError("L1 measurement evidence requires the full-cell public atlas")
    protocol = read_json(Path(__file__).resolve().parents[1] / PROTOCOL_PATH)
    if not isinstance(protocol, dict) or not isinstance(protocol.get("protocol_hash"), str):
        raise ValueError("measurement protocol is invalid")

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{layout_ancestor}-{method_id}"
    panel: dict[str, Any] = {
        "schema": PANEL_SCHEMA,
        "formal_score_eligible": False,
        "purpose": purpose,
        "precommitted_before_execution": True,
        "layout_ancestors": [layout_ancestor],
        "method_ids": [method_id],
    }
    panel["panel_hash"] = content_hash(panel)
    write_json(output_root / "panel.json", panel)

    episode_record = {
        "layout_ancestor": layout_ancestor,
        "method_id": method_id,
        "episode_id": str(episode["episode_id"]),
        "episode_name": episode_name,
        "layout_root": _relative_path(output_root, layout_root, "layout_root"),
        "release_config": _relative_path(output_root, release_config, "release_config"),
        "public_report": f"{stem}.public.json",
        "private_report": f"{stem}.private.json",
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "formal_score_eligible": False,
        "purpose": "precommitted_calibration_l1_evidence",
        "protocol_hash": str(protocol["protocol_hash"]),
        "panel_manifest_hash": str(panel["panel_hash"]),
        "precommitted_before_execution": True,
        "episodes": [episode_record],
    }
    manifest["manifest_hash"] = content_hash(manifest)
    write_json(output_root / "evidence-manifest.json", manifest)
    return {"panel": panel, "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    result = build_manifest(
        layout_root=args.layout_root.resolve(),
        release_config=args.release_config.resolve(),
        layout_ancestor=args.layout_ancestor,
        method_id=args.method_id,
        episode_name=args.episode_name,
        output_root=args.output_root,
        purpose=args.purpose,
    )
    print(
        {
            "panel_hash": result["panel"]["panel_hash"],
            "manifest_hash": result["manifest"]["manifest_hash"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
