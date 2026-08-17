"""Train and execute a real pyribs MAP-Elites archive on public pilot routes.

This is deliberately a route-parameter archive rather than a hidden-target
optimizer.  Fitness and descriptors are derived only from the public mission,
public geometry, and the rollout's public state/safety records.  The resulting
archive is hash-bound and can be executed through the same online recorder as
other pilot methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .provenance import detect_source_provenance
from .runtime import HighLevelAction, PilotRuntimeConfig, PilotSwarmRuntime, PublicMission, PublicObservation


ARCHIVE_SCHEMA = "org.rivermark.pyribs-map-elites.v1"
SOLUTION_DIM = 4
ACTION_SCALE = np.asarray((2.3, 2.3, 1.15, 1.35), dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_ribs() -> tuple[Any, Any, Any]:
    try:
        from ribs.archives import GridArchive
        from ribs.emitters import GaussianEmitter
        from ribs.schedulers import Scheduler
    except ImportError as exc:  # pragma: no cover - verified through CLI behavior.
        raise RuntimeError("pyribs is required for MAP-Elites pilot training") from exc
    return GridArchive, GaussianEmitter, Scheduler


def _atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _action_from_solution(solution: np.ndarray, *, source: str) -> HighLevelAction:
    command = np.clip(np.asarray(solution, dtype=np.float32), -1.0, 1.0) * ACTION_SCALE
    return HighLevelAction(
        velocity_xyz=tuple(float(value) for value in command[:3]),
        yaw_rate_rad_s=float(command[3]),
        mode="transit",
        source=source,
    )


def _evaluate_solution(solution: np.ndarray, *, seed: int, steps: int) -> tuple[float, np.ndarray]:
    """Evaluate one constant action using public trajectory data only."""

    runtime = PilotSwarmRuntime(
        PilotRuntimeConfig(agent_count=1, max_steps=steps, seed=seed),
        information_profile="state_only",
    )
    observations = runtime.reset()
    del observations
    action = _action_from_solution(solution, source="pyribs_map_elites_training")
    positions: list[np.ndarray] = [runtime.current_frame().states[0].position_m.copy()]
    public_safety_events = 0
    while not runtime.done:
        _, frame = runtime.step({0: action})
        positions.append(frame.states[0].position_m.copy())
        public_safety_events += len(frame.safety_events)
    trajectory = np.asarray(positions, dtype=np.float64)
    displacement = float(np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2]))
    altitude_error = float(np.mean(np.abs(trajectory[:, 2] - 2.8)))
    path_length = float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())
    width, height = runtime.mission.bounds_xy_m
    horizontal_extent = float(np.clip((trajectory[:, 0].max() - trajectory[:, 0].min()) / width, 0.0, 1.0))
    vertical_extent = float(np.clip((trajectory[:, 2].max() - trajectory[:, 2].min()) / (runtime.config.max_altitude_m - runtime.config.min_altitude_m), 0.0, 1.0))
    # This public objective favors useful travel without treating evaluator truth
    # or target confirmation as a fitness signal.
    objective = displacement + 0.20 * path_length - 1.5 * altitude_error - 1.0 * public_safety_events
    return objective, np.asarray((horizontal_extent, vertical_extent), dtype=np.float64)


@dataclass(frozen=True)
class QdTrainResult:
    archive_path: Path
    metadata_path: Path
    elite_count: int
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "archive": str(self.archive_path),
            "metadata": str(self.metadata_path),
            "elite_count": self.elite_count,
            "iterations": self.iterations,
        }


def train_map_elites(
    output: Path,
    *,
    iterations: int,
    batch_size: int,
    rollout_steps: int,
    seed: int,
) -> QdTrainResult:
    """Create a genuine pyribs archive with deterministic public rollouts."""

    if iterations < 1 or batch_size < 1 or rollout_steps < 2:
        raise ValueError("iterations, batch_size, and rollout_steps must be positive")
    GridArchive, GaussianEmitter, Scheduler = _require_ribs()
    archive = GridArchive(
        solution_dim=SOLUTION_DIM,
        dims=(8, 5),
        ranges=((0.0, 1.0), (0.0, 1.0)),
        threshold_min=-np.inf,
        seed=seed,
    )
    emitter = GaussianEmitter(
        archive,
        x0=np.zeros(SOLUTION_DIM, dtype=np.float64),
        sigma=0.45,
        bounds=[(-1.0, 1.0)] * SOLUTION_DIM,
        batch_size=batch_size,
        seed=seed + 1,
    )
    scheduler = Scheduler(archive, [emitter])
    for iteration in range(iterations):
        solutions = scheduler.ask()
        objectives = np.empty(len(solutions), dtype=np.float64)
        measures = np.empty((len(solutions), 2), dtype=np.float64)
        for index, solution in enumerate(solutions):
            objective, descriptor = _evaluate_solution(
                solution,
                seed=seed + iteration * 10_000 + index,
                steps=rollout_steps,
            )
            objectives[index] = objective
            measures[index] = descriptor
        scheduler.tell(objectives, measures)
    data = archive.data()
    archive_path = output.resolve()
    _atomic_npz(
        archive_path,
        solutions=np.asarray(data["solution"], dtype=np.float32),
        objectives=np.asarray(data["objective"], dtype=np.float32),
        measures=np.asarray(data["measures"], dtype=np.float32),
    )
    metadata_path = archive_path.with_suffix(".rivermark.json")
    source = detect_source_provenance()
    metadata = {
        "schema": ARCHIVE_SCHEMA,
        "implementation_kind": "trained_pyribs_map_elites_archive",
        "training_backend": "rivermark-kinematic-pilot-v1",
        "formal_benchmark_admission": False,
        "information_profile": "state_only",
        "pyribs_version": "0.8.3",
        "solution_dim": SOLUTION_DIM,
        "descriptor": "public_horizontal_and_vertical_trajectory_extent",
        "objective": "public_displacement_plus_path_length_minus_altitude_and_safety_penalties",
        "objective_uses_evaluator_private_truth": False,
        "iterations": iterations,
        "batch_size": batch_size,
        "rollout_steps": rollout_steps,
        "seed": seed,
        "elite_count": int(len(data["objective"])),
        "source_revision": source.source_revision,
        "source_tree_sha256": source.source_tree_sha256,
        "source_worktree_dirty": source.source_worktree_dirty,
        "archive_sha256": sha256_file(archive_path),
    }
    _atomic_json(metadata_path, metadata)
    return QdTrainResult(archive_path, metadata_path, int(len(data["objective"])), iterations)


class PyribsMapElitesCheckpointPolicy:
    """Run a checked pyribs archive using state-only public observations."""

    method_id = "pyribs_map_elites_checkpoint"

    def __init__(self, archive_path: Path, metadata_path: Path | None = None) -> None:
        self.archive_path = archive_path.resolve()
        if not self.archive_path.is_file():
            raise FileNotFoundError(f"MAP-Elites archive is missing: {self.archive_path}")
        self.metadata_path = (metadata_path or self.archive_path.with_suffix(".rivermark.json")).resolve()
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"MAP-Elites metadata is missing: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != ARCHIVE_SCHEMA:
            raise ValueError("unsupported MAP-Elites metadata schema")
        if self.metadata.get("information_profile") != "state_only":
            raise ValueError("MAP-Elites archive must declare state_only")
        if self.metadata.get("archive_sha256") != sha256_file(self.archive_path):
            raise ValueError("MAP-Elites archive SHA-256 does not match its metadata")
        with np.load(self.archive_path, allow_pickle=False) as payload:
            self.solutions = payload["solutions"].astype(np.float32, copy=True)
            self.objectives = payload["objectives"].astype(np.float32, copy=True)
            self.measures = payload["measures"].astype(np.float32, copy=True)
        if self.solutions.ndim != 2 or self.solutions.shape[1] != SOLUTION_DIM:
            raise ValueError("MAP-Elites archive has malformed solutions")
        if self.objectives.shape != (self.solutions.shape[0],) or self.measures.shape != (self.solutions.shape[0], 2):
            raise ValueError("MAP-Elites archive has inconsistent elite arrays")
        if len(self.solutions) == 0:
            raise ValueError("MAP-Elites archive contains no elites")
        self.mission: PublicMission | None = None
        self.agent_count = 0

    def reset(
        self,
        mission: PublicMission,
        agent_count: int,
        *,
        public_geometry: Mapping[str, Any] | None = None,
    ) -> None:
        del public_geometry
        self.mission = mission
        self.agent_count = agent_count

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            if observation.information_profile != "state_only":
                raise RuntimeError("MAP-Elites archive received a mismatched information profile")
            position = observation.proprioception[:3]
            progress = float(np.clip(position[0] / self.mission.bounds_xy_m[0], 0.0, 1.0)) if self.mission else 0.0
            altitude = float(np.clip((position[2] - 1.0) / 4.0, 0.0, 1.0))
            desired = np.asarray((progress, altitude), dtype=np.float32)
            distance = np.linalg.norm(self.measures - desired, axis=1)
            candidate_indices = np.flatnonzero(distance == distance.min())
            index = int(candidate_indices[np.argmax(self.objectives[candidate_indices])])
            actions[agent_id] = _action_from_solution(self.solutions[index], source=self.method_id)
        return actions

    def provenance(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "implementation_kind": "trained_pyribs_map_elites_archive",
            "external_dependency": "ribs",
            "archive": str(self.archive_path),
            "archive_sha256": sha256_file(self.archive_path),
            "adapter_metadata": str(self.metadata_path),
            "adapter_metadata_sha256": sha256_file(self.metadata_path),
            "elite_count": int(len(self.solutions)),
            "objective_uses_evaluator_private_truth": False,
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = train_map_elites(
        args.output,
        iterations=args.iterations,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
    )
    print(json.dumps({"status": "completed", **result.as_dict()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
