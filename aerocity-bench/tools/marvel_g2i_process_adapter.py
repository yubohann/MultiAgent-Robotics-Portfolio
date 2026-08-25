"""Run a frozen upstream MARVEL checkpoint behind AeroCityBench's JSONL bridge.

This is a cross-environment transfer diagnostic, not an upstream MARVEL
reimplementation and not a claim that MARVEL was trained for 3-D hidden-target
search.  The adapter consumes only the process bridge's public G2-I payload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.marvel_g2i_projection import (  # noqa: E402
    MarvelG2IProjection,
    MarvelGraphInput,
)

REQUEST_SCHEMA = "org.aerocity.bench.external-planner-request.v1"
RESPONSE_SCHEMA = "org.aerocity.bench.external-planner-response.v1"


class FrozenMarvelPolicy:
    """Thin weight loader which keeps the upstream model implementation external."""

    def __init__(self, marvel_root: Path, checkpoint: Path, device: str) -> None:
        sys.path.insert(0, str(marvel_root.resolve()))
        try:
            import torch
            from test_parameter import EMBEDDING_DIM, NODE_INPUT_DIM, NUM_ANGLES_BIN
            from utils.model import PolicyNet
        except ImportError as exc:  # pragma: no cover - environment integration path.
            raise RuntimeError("MARVEL upstream dependencies cannot be imported") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, NUM_ANGLES_BIN).to(self.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict) or "policy_model" not in payload:
            raise RuntimeError("MARVEL checkpoint lacks its frozen policy_model state")
        self.model.load_state_dict(payload["policy_model"])
        self.model.eval()
        self._warm_up()

    def _warm_up(self) -> None:
        """Pay PyTorch's first-forward cost before the first control deadline.

        This uses a fixed all-zero graph and runs during child-process startup,
        before the public episode reset is accepted.  It therefore cannot
        encode a city, target, evaluator fact, or action decision.  The actual
        policy still performs every episode decision through ``choose_slot``.
        """

        graph = MarvelGraphInput(
            node_inputs=[[0.0] * 6 for _ in range(360)],
            node_padding_mask=[1] * 360,
            edge_mask=[[1] * 360 for _ in range(360)],
            current_edge=[0] * 25,
            edge_padding_mask=[1] * 25,
            frontier_distribution=[[0.0] * 36 for _ in range(360)],
            headings_visited=[[0.0] * 36 for _ in range(360)],
            neighbor_best_headings=[[[0.0] * 36 for _ in range(3)] for _ in range(25)],
            candidate_cell_ids=["warmup-cell"],
        )
        self.choose_slot(graph)

    def choose_slot(self, graph: MarvelGraphInput) -> int:
        torch = self.torch
        node_inputs = torch.tensor(
            graph.node_inputs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        node_padding_mask = torch.tensor(
            graph.node_padding_mask, dtype=torch.int16, device=self.device
        ).view(1, 1, -1)
        edge_mask = torch.tensor(
            graph.edge_mask, dtype=torch.int16, device=self.device
        ).unsqueeze(0)
        current_edge = torch.tensor(
            graph.current_edge, dtype=torch.int64, device=self.device
        ).view(1, -1, 1)
        edge_padding_mask = torch.tensor(
            graph.edge_padding_mask, dtype=torch.int16, device=self.device
        ).view(1, 1, -1)
        frontier_distribution = torch.tensor(
            graph.frontier_distribution, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        headings_visited = torch.tensor(
            graph.headings_visited, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        neighbor_best_headings = torch.tensor(
            graph.neighbor_best_headings, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logp = self.model(
                node_inputs,
                node_padding_mask,
                edge_mask,
                torch.zeros((1, 1, 1), dtype=torch.int64, device=self.device),
                current_edge,
                edge_padding_mask,
                frontier_distribution,
                headings_visited,
                neighbor_best_headings,
            )
        action_index = int(torch.argmax(logp, dim=1).item())
        return action_index // 3 - 1


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marvel-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args(argv)


def _response(request_id: object, **payload: Any) -> str:
    return json.dumps(
        {"schema": RESPONSE_SCHEMA, "request_id": request_id, **payload},
        ensure_ascii=True,
        separators=(",", ":"),
    )


def serve(policy: FrozenMarvelPolicy) -> None:
    projection: MarvelG2IProjection | None = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
                raise ValueError("request schema differs")
            request_id = request.get("request_id")
            kind = request.get("kind")
            if kind == "reset":
                projection = MarvelG2IProjection.from_public_reset(
                    request["public_episode"], request["public_task_spec"]
                )
                output = _response(request_id, status="ok")
            elif kind == "act":
                if projection is None:
                    raise ValueError("act arrived before reset")
                observations = request.get("observations")
                if (
                    not isinstance(observations, dict)
                    or set(observations) != set(projection.starts)
                ):
                    raise ValueError("active observations differ from the public fleet")
                actions = {
                    drone_id: projection.action(drone_id, observation, policy.choose_slot)
                    for drone_id, observation in sorted(observations.items())
                    if isinstance(observation, dict)
                }
                if set(actions) != set(projection.starts):
                    raise ValueError("an observation was not an object")
                output = _response(request_id, status="ok", actions=actions)
            else:
                raise ValueError("request kind is unsupported")
        except Exception as exc:
            request_id = (
                request.get("request_id")
                if "request" in locals() and isinstance(request, dict)
                else None
            )
            output = _response(request_id, status=f"error:{type(exc).__name__}")
        print(output, flush=True)


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    if not args.marvel_root.is_dir() or not args.checkpoint.is_file():
        raise SystemExit("MARVEL root or frozen checkpoint path is invalid")
    serve(FrozenMarvelPolicy(args.marvel_root, args.checkpoint, args.device))


if __name__ == "__main__":
    main()
