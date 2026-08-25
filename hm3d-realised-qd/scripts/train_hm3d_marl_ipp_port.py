"""Train the author-network MARL-IPP controlled transfer from real train rollouts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import read_json_object, write_json_atomic
from aerocity_method.evaluation.hm3d_marl_ipp_training import (
    sample_from_p07_training_record,
    train_marl_ipp_port_baseline,
    training_scene_ids_from_split_manifest,
)


def _write_checkpoint_new(path: Path, payload: dict[str, object]) -> None:
    import torch

    if path.exists():
        raise FileExistsError(f"refusing to overwrite MARL-IPP checkpoint: {path}")
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
    parser.add_argument(
        "--source-root", type=Path, default=Path(r"E:\github_repos\marl_ipp-main")
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--minimum-transitions", type=int, default=40)
    parser.add_argument("--minimum-scenes", type=int, default=2)
    args = parser.parse_args()
    checkpoint = args.checkpoint_output.expanduser().resolve()
    provenance = args.provenance_output.expanduser().resolve()
    if checkpoint == provenance or checkpoint.exists() or provenance.exists():
        raise FileExistsError("MARL-IPP trainer refuses to overwrite its outputs")
    split = read_json_object(args.p05_split_manifest.expanduser().resolve())
    train_scenes = training_scene_ids_from_split_manifest(split)
    root = split.get("payload", split)
    if not isinstance(root, dict):
        raise ValueError("P05 split manifest payload must be an object")
    split_hash = str(root["split_manifest_sha256"])
    samples = tuple(
        sample
        for path in args.rollout
        for sample in sample_from_p07_training_record(
            read_json_object(path.expanduser().resolve()),
            allowed_train_scene_ids=train_scenes,
        )
    )
    model_checkpoint, model_provenance = train_marl_ipp_port_baseline(
        samples,
        split_manifest_sha256=split_hash,
        source_root=args.source_root.expanduser().resolve(),
        source_checkpoint=args.source_checkpoint.expanduser().resolve(),
        updates=args.updates,
        seed=args.seed,
        minimum_transitions=args.minimum_transitions,
        minimum_scenes=args.minimum_scenes,
    )
    _write_checkpoint_new(checkpoint, model_checkpoint)
    write_json_atomic(provenance, model_provenance)
    print(
        "MARL_IPP_PORT_TRAINING_COMPLETE "
        f"checkpoint={checkpoint} provenance={provenance} updates={args.updates}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
