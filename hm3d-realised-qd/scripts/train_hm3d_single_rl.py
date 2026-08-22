"""Train the P07 archive-free single-RL baseline from real train rollouts only."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import read_json_object, write_json_atomic
from aerocity_method.evaluation.hm3d_single_rl_training import (
    sample_from_p07_training_record,
    train_single_rl_baseline,
    training_scene_ids_from_split_manifest,
)


def _write_checkpoint_new(path: Path, payload: dict[str, object]) -> None:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - dependency failure is explicit
        raise RuntimeError("single-RL training requires PyTorch") from error
    if path.exists():
        raise FileExistsError(f"refusing to overwrite single-RL checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p05-split-manifest", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--minimum-transitions", type=int, default=40)
    parser.add_argument("--minimum-scenes", type=int, default=2)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_output.expanduser().resolve()
    provenance_path = args.provenance_output.expanduser().resolve()
    if checkpoint_path == provenance_path:
        raise ValueError("checkpoint and provenance outputs must be different files")
    if checkpoint_path.exists() or provenance_path.exists():
        raise FileExistsError("single-RL trainer refuses to overwrite checkpoint or provenance")
    split_payload = read_json_object(args.p05_split_manifest.expanduser().resolve())
    train_scene_ids = training_scene_ids_from_split_manifest(split_payload)
    root = split_payload.get("payload", split_payload)
    assert isinstance(root, dict)
    split_hash = str(root["split_manifest_sha256"])
    samples = tuple(
        sample
        for path in args.rollout
        for sample in sample_from_p07_training_record(
            read_json_object(path.expanduser().resolve()),
            allowed_train_scene_ids=train_scene_ids,
        )
    )
    checkpoint, provenance = train_single_rl_baseline(
        samples,
        split_manifest_sha256=split_hash,
        updates=args.updates,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        minimum_transitions=args.minimum_transitions,
        minimum_scenes=args.minimum_scenes,
    )
    _write_checkpoint_new(checkpoint_path, checkpoint)
    write_json_atomic(provenance_path, provenance)
    print(
        "SINGLE_RL_TRAINING_COMPLETE "
        f"checkpoint={checkpoint_path} provenance={provenance_path} updates={args.updates}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
