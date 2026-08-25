#!/usr/bin/env python3
"""Plan native Isaac recordings for every frozen Rivermark start layout.

The planner only materializes public route geometry and command templates. It
never reads or copies evaluator-private targets. A recording runner must supply
the private manifest, smoke receipt, and asset paths from outside the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_routes(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = repo_root / "rivermark" / "code" / "src"
    sys.path.insert(0, str(source))
    from rivermark_benchmark.citylite_scene.routes import (  # noqa: PLC0415
        PUBLIC_ROUTE_FAMILIES_W_M,
        TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M,
    )

    return PUBLIC_ROUTE_FAMILIES_W_M, TARGET_FREE_SAFE_STARTS_BY_ROUTE_FAMILY_W_M


def _command_template(
    *,
    repo_root: Path,
    protocol: Path,
    cell_id: str,
    episode_index: int,
    output_dir: str,
) -> list[str]:
    """Return a fail-closed command with external values as placeholders."""

    return [
        "<ISAAC_PYTHON>",
        "-m",
        "rivermark_benchmark.isaac_capture",
        "--output-dir",
        output_dir,
        "--drone-usd",
        "<EXTERNAL_CF2X_USD>",
        "--scene-contract",
        "<EXTERNAL_CITY_LITE_CONTRACT>",
        "--collection-protocol",
        str(protocol),
        "--collection-cell-id",
        cell_id,
        "--collection-episode-index",
        str(episode_index),
        "--evaluator-private-manifest",
        "<EXTERNAL_PRIVATE_MANIFEST>",
        "--evaluator-private-manifest-retention-root",
        "<EXTERNAL_PRIVATE_RETENTION_ROOT>",
        "--runtime-lock",
        str(repo_root / "rivermark" / "code" / "config" / "isaac_runtime.windows-5.1.json"),
        "--isaaclab-source",
        "<ISAACLAB_SOURCE>",
        "--sensor-physics-smoke-receipt",
        "<EXTERNAL_SENSOR_SMOKE_RECEIPT>",
        "--control-mode",
        "fixed_public_route",
        "--headless",
    ]


def build_matrix(
    *,
    repo_root: Path,
    protocol_path: Path,
    episodes_per_cell: int,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or not isinstance(protocol.get("cells"), list):
        raise ValueError("collection protocol must contain a cells list")
    route_families, starts_by_family = _load_routes(repo_root)
    rows: list[dict[str, Any]] = []
    for cell in protocol["cells"]:
        if not isinstance(cell, dict):
            raise ValueError("protocol cell must be an object")
        cell_id = str(cell["cell_id"])
        conditions = cell.get("conditions")
        if not isinstance(conditions, dict):
            raise ValueError(f"cell {cell_id} has no conditions object")
        family_id = str(conditions["route_family"])
        if family_id not in route_families:
            raise ValueError(f"cell {cell_id} references unknown route family {family_id}")
        routes = route_families[family_id]
        starts = starts_by_family[family_id]
        if len(routes) != 8 or len(starts) != 8:
            raise ValueError(f"route family {family_id} must contain eight starts/routes")
        for route, start in zip(routes, starts, strict=True):
            if tuple(route[0]) != tuple(start):
                raise ValueError(f"route family {family_id} has a route/start mismatch")
        for episode_index in range(episodes_per_cell):
            output_dir = f"<OUTPUT_ROOT>/{cell['split']}/{cell_id}/episode-{episode_index:04d}"
            rows.append(
                {
                    "episode_index": episode_index,
                    "split": str(cell["split"]),
                    "cell_id": cell_id,
                    "route_family_id": family_id,
                    "agent_count": 8,
                    "initial_positions_w_m": [list(point) for point in starts],
                    "route_waypoints_w_m": [[list(point) for point in route] for route in routes],
                    "private_manifest_required": True,
                    "command": _command_template(
                        repo_root=repo_root,
                        protocol=protocol_path,
                        cell_id=cell_id,
                        episode_index=episode_index,
                        output_dir=output_dir,
                    ),
                }
            )
    return {
        "schema": "org.rivermark.native-multistart-video-matrix.v1",
        "source_protocol": str(protocol_path),
        "protocol_id": protocol.get("protocol_id"),
        "scene_identity": protocol.get("scene_identity"),
        "claim_boundary": "native Isaac capture commands; no video is created by this planner",
        "episode_count": len(rows),
        "episodes": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "code" / "config" / "collection_protocol.citylite_t1_expert_coverage_v2.json",
    )
    parser.add_argument("--episodes-per-cell", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.episodes_per_cell < 1:
        parser.error("--episodes-per-cell must be positive")
    matrix = build_matrix(
        repo_root=args.repo_root.resolve(),
        protocol_path=args.protocol.resolve(),
        episodes_per_cell=args.episodes_per_cell,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"planned {matrix['episode_count']} native Isaac episodes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
