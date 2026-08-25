"""Run a source-locked ACO3D public-inspection ordering translation.

The SII 2024 upstream code is a MATLAB implementation of a sequential
three-dimensional inspection tour.  This adapter preserves its published
point-cost formula and fixed ant-colony loop, but does not claim that its
source provides G2-I's public four-UAV assignment or CF2X execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_LOCK_PATH = _REPOSITORY_ROOT / "external" / "aco3d" / "source-lock.json"
UPSTREAM_URL = "https://github.com/duynamrcv/aco_3d_ipp.git"
UPSTREAM_COMMIT = "c395f5b61f6746b2d39310dbc55a7ec3e1eae2d5"
UPSTREAM_LICENSE = "MIT"
ACO_ITERATIONS = 350
ACO_ANTS = 50
ACO_Q = 1.0
ACO_ALPHA = 1.0
ACO_BETA = 1.0
ACO_RHO = 0.05


def _load_public_route_base() -> Any:
    """Load shared public-route execution without importing OR-Tools itself."""

    path = Path(__file__).with_name("ortools_g2i_process_adapter.py")
    spec = importlib.util.spec_from_file_location("aerocity_public_route_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load public inspection route base")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_public_route_base()
REQUEST_SCHEMA = _BASE.REQUEST_SCHEMA
RESPONSE_SCHEMA = _BASE.RESPONSE_SCHEMA


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_lock() -> dict[str, Any]:
    raw = json.loads(_SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ACO3D source lock must be an object")
    upstream = raw.get("upstream")
    adapter = raw.get("adapter")
    if not isinstance(upstream, dict) or not isinstance(adapter, dict):
        raise ValueError("ACO3D source lock is incomplete")
    if (
        upstream.get("url") != UPSTREAM_URL
        or upstream.get("commit") != UPSTREAM_COMMIT
        or upstream.get("license") != UPSTREAM_LICENSE
    ):
        raise ValueError("ACO3D source lock differs from the adapter constants")
    parameters = adapter.get("fixed_parameters")
    expected_parameters = {
        "iterations": ACO_ITERATIONS,
        "ants": ACO_ANTS,
        "q": ACO_Q,
        "alpha": ACO_ALPHA,
        "beta": ACO_BETA,
        "rho": ACO_RHO,
    }
    if parameters != expected_parameters:
        raise ValueError("ACO3D source lock parameters differ from the published loop")
    return raw


def _verify_upstream_source(path: Path) -> None:
    """Verify a clean, sparse or full checkout before a native-source claim."""

    lock = _source_lock()
    upstream = lock["upstream"]
    source_hashes = upstream.get("source_file_sha256")
    if not path.is_dir() or not isinstance(source_hashes, dict):
        raise ValueError("ACO3D upstream source directory is unavailable")
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        remote = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("ACO3D upstream source is not a readable Git checkout") from exc
    if head != UPSTREAM_COMMIT:
        raise ValueError("ACO3D upstream Git revision differs from the source lock")
    if dirty:
        raise ValueError("ACO3D upstream source checkout must be clean")
    if remote.rstrip("/") != UPSTREAM_URL.removesuffix(".git") and remote != UPSTREAM_URL:
        raise ValueError("ACO3D upstream remote differs from the source lock")
    for relative, expected in source_hashes.items():
        source_file = path / str(relative)
        if not source_file.is_file() or _file_hash(source_file) != expected:
            raise ValueError(f"ACO3D upstream source hash differs: {relative}")


def _source_distance_matrix(points: list[tuple[float, float, float]]) -> list[list[float]]:
    """Translate ``CreateModel.m`` without substituting a benchmark route cost."""

    count = len(points)
    matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    for left in range(count - 1):
        for right in range(left + 1, count):
            x0, y0, z0 = points[left]
            x1, y1, z1 = points[right]
            horizontal = math.hypot(x0 - x1, y0 - y1)
            three_dimensional = math.dist(points[left], points[right])
            if horizontal < 50.0:
                value = 1.2 * three_dimensional + horizontal + 2.0 * abs(z0 - z1)
            else:
                value = 1.2 * three_dimensional + 4.0 * horizontal + 2.0 * abs(z0 - z1)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "ACO3D cannot order coincident or non-finite public inspection cells"
                )
            matrix[left][right] = value
            matrix[right][left] = value
    return matrix


def _source_tour_cost(
    tour: list[int], points: list[tuple[float, float, float]], matrix: list[list[float]]
) -> float:
    """Translate ``TourCost.m``'s open-tour objective exactly."""

    return sum(
        (2.0 if points[left][2] != points[right][2] else 1.0) * matrix[left][right]
        for left, right in zip(tour[:-1], tour[1:], strict=True)
    )


def _public_seed(drone_id: str, cell_ids: list[str]) -> int:
    material = f"aco3d-source-translation-v1|{drone_id}|{'|'.join(cell_ids)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], byteorder="big", signed=False)


def _source_aco_order(drone_id: str, cells: list[Any]) -> list[str]:
    """Execute the source-fixed ACO ordering on one public drone sector."""

    if len(cells) <= 1:
        return [cell.cell_id for cell in cells]
    cell_ids = [str(cell.cell_id) for cell in cells]
    points = [tuple(float(value) for value in cell.position) for cell in cells]
    matrix = _source_distance_matrix(points)
    mean_distance = sum(sum(row) for row in matrix) / (len(cells) * len(cells))
    if not math.isfinite(mean_distance) or mean_distance <= 0.0:
        raise ValueError("ACO3D public sector has an invalid source distance mean")
    tau0 = 10.0 * ACO_Q / (len(cells) * mean_distance)
    tau = [[tau0 for _ in cells] for _ in cells]
    eta = [
        [
            0.0 if source == destination else 1.0 / matrix[source][destination]
            for destination in range(len(cells))
        ]
        for source in range(len(cells))
    ]
    generator = random.Random(_public_seed(drone_id, cell_ids))
    best_tour: list[int] | None = None
    best_cost = math.inf
    for _ in range(ACO_ITERATIONS):
        ants: list[tuple[list[int], float]] = []
        for _ in range(ACO_ANTS):
            tour = [0]  # ``main.m`` overwrites its random start with point 1.
            while len(tour) < len(cells):
                current = tour[-1]
                candidates = [candidate for candidate in range(len(cells)) if candidate not in tour]
                weights = [
                    tau[current][candidate] ** ACO_ALPHA * eta[current][candidate] ** ACO_BETA
                    for candidate in candidates
                ]
                total = sum(weights)
                if not math.isfinite(total) or total <= 0.0:
                    raise ValueError("ACO3D source transition weights are invalid")
                threshold = generator.random() * total
                cumulative = 0.0
                selected = candidates[-1]
                for candidate, weight in zip(candidates, weights, strict=True):
                    cumulative += weight
                    if threshold <= cumulative:
                        selected = candidate
                        break
                tour.append(selected)
            cost = _source_tour_cost(tour, points, matrix)
            ants.append((tour, cost))
            if cost < best_cost:
                best_tour, best_cost = list(tour), cost
        for tour, cost in ants:
            if not math.isfinite(cost) or cost <= 0.0:
                raise ValueError("ACO3D source tour cost is invalid")
            for left, right in zip(tour, [*tour[1:], tour[0]], strict=True):
                tau[left][right] += ACO_Q / cost
        tau = [[(1.0 - ACO_RHO) * value for value in row] for row in tau]
    if best_tour is None:
        raise RuntimeError("ACO3D source loop returned no tour")
    return [cell_ids[index] for index in best_tour]


class ACO3DInspectionPlanner(_BASE.ORToolsInspectionPlanner):
    """Use the locked ACO3D ordering inside the existing public execution ABI."""

    def _solve_sector_route(self, drone_id: str) -> list[str]:
        cells = [self.cells[cell_id] for cell_id in self.assignments[drone_id]]
        return _source_aco_order(drone_id, cells)


def _response(request_id: object, **payload: Any) -> str:
    return _BASE._response(request_id, **payload)


def serve(lines: Iterable[str] | None = None) -> None:
    planner: ACO3DInspectionPlanner | None = None
    for line in sys.stdin if lines is None else lines:
        request: object = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
                raise ValueError("request schema differs")
            request_id = request.get("request_id")
            if request.get("kind") == "reset":
                planner = ACO3DInspectionPlanner.from_public_reset(
                    request["public_episode"], request["public_task_spec"]
                )
                output = _response(request_id, status="ok")
            elif request.get("kind") == "act":
                if planner is None:
                    raise ValueError("act arrived before reset")
                observations = request.get("observations")
                if not isinstance(observations, dict) or set(observations) != set(planner.starts):
                    raise ValueError("active observations differ from the public fleet")
                actions = {
                    drone_id: planner.action(drone_id, observation)
                    for drone_id, observation in sorted(observations.items())
                    if isinstance(observation, dict)
                }
                if set(actions) != set(planner.starts):
                    raise ValueError("an observation was not an object")
                output = _response(request_id, status="ok", actions=actions)
            else:
                raise ValueError("request kind is unsupported")
        except Exception as exc:
            request_id = request.get("request_id") if isinstance(request, dict) else None
            output = _response(request_id, status=f"error:{type(exc).__name__}")
        print(output, flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--upstream-source", type=Path)
    args = parser.parse_args(argv)
    if args.upstream_source is not None:
        _verify_upstream_source(args.upstream_source)
    if args.version:
        _source_lock()
        print("aco3d-source-translation-v1")
        return
    serve()


if __name__ == "__main__":
    main()
