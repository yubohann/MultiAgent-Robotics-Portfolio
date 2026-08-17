"""Read-only snapshot auditing for the external robotics ecosystems.

The projects listed here inform Rivermark's data product, task protocol,
perception, and baseline work.  They remain external sources: auditing a local
checkout does not import its runtime, copy payloads, create a Rivermark
episode, or establish native Isaac closed-loop evidence.

The manifest intentionally contains only source identifiers, relative key
paths, sizes, and hashes.  It never stores the local checkout path, which keeps
research provenance shareable without exposing a workstation layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

EXTERNAL_SOURCE_SNAPSHOT_SCHEMA = "org.rivermark.external-source-snapshots.v1"


class ExternalSourceError(ValueError):
    """Raised when an external source snapshot cannot be audited safely."""


@dataclass(frozen=True)
class ExternalSourceSpec:
    """Stable, technical description of one external source snapshot."""

    source_id: str
    local_directory: str
    repository: str
    role: str
    integration_level: str
    key_paths: tuple[str, ...]
    resource_constraint: str
    action_boundary: str


_SPECS: tuple[ExternalSourceSpec, ...] = (
    ExternalSourceSpec(
        "droid_policy_learning",
        "droid_policy_learning-master",
        "https://github.com/droid-dataset/droid_policy_learning",
        "offline imitation and RLDS training reference",
        "thin_adapter_candidate",
        ("README.md", "robomimic/utils/rlds_utils.py", "robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds.py"),
        "Use a bounded shuffle buffer and bounded workers; never inherit the upstream high-memory defaults.",
        "The 7D manipulator action is not a CF2X action and must not supervise a flight action head.",
    ),
    ExternalSourceSpec(
        "droid_dataset",
        "droid-main",
        "https://github.com/droid-dataset/droid",
        "collection-context, calibration, and failure-accounting reference",
        "protocol_reference_only",
        ("README.md", "docs/the-droid-dataset.md", "droid/data_loading/trajectory_sampler.py"),
        "Inspect small episodes first; retain failed and successful attempts instead of inheriting success-only sampling.",
        "Robot-arm trajectories and language annotations are external pretraining/context data, not flight controls.",
    ),
    ExternalSourceSpec(
        "embodiedscan",
        "EmbodiedScan-main",
        "https://github.com/OpenRobotLab/EmbodiedScan",
        "multiview 3D perception, grounding, and ontology reference",
        "perception_adapter_candidate",
        ("README.md", "embodiedscan/datasets/embodiedscan_dataset.py", "embodiedscan/eval/metrics/grounding_metric.py"),
        "Do not silently filter empty scenes when reporting detection or grounding failure rates.",
        "Indoor RGB-D boxes and language targets do not constitute UAV search episodes or CF2X actions.",
    ),
    ExternalSourceSpec(
        "habitat_lab",
        "habitat-lab-main",
        "https://github.com/facebookresearch/habitat-lab",
        "task, split, measurement, and multi-agent protocol reference",
        "protocol_reference_only",
        ("README.md", "DATASETS.md", "habitat-lab/habitat/tasks/nav/nav.py"),
        "Reuse task and metric organization without adding Habitat as a Rivermark simulator dependency.",
        "Habitat navigation actions and scores are not native Isaac CF2X evidence.",
    ),
    ExternalSourceSpec(
        "lerobot",
        "lerobot-main",
        "https://github.com/huggingface/lerobot",
        "research-facing Parquet/video dataset product reference",
        "projection_candidate",
        ("README.md", "pyproject.toml", "src/lerobot/datasets/dataset_writer.py", "src/lerobot/datasets/lerobot_dataset.py"),
        "Do not attach its episode-buffering or multi-camera process encoding to a live eight-agent capture; project after validation with explicit byte bounds.",
        "Its generic action field needs a documented T2 flight ABI mapping; no automatic action equivalence exists.",
    ),
    ExternalSourceSpec(
        "omnidrones",
        "OmniDrones-main",
        "https://github.com/btx0424/OmniDrones",
        "multirotor controller and MARL algorithm reference",
        "isolated_controller_port_candidate",
        ("README.md", "omni_drones/controllers/lee_position_controller.py", "omni_drones/robots/drone/crazyflie.py", "omni_drones/robots/assets/usd/crazyflie.yaml"),
        "Keep the port dependency-free until state, frame, rotor order, units, and actuator timing pass CPU parity tests; do not import its simulator stack into the live runtime.",
        "Its asset, dynamics, normalized rotor commands, and scores are not Rivermark CF2X evidence.",
    ),
    ExternalSourceSpec(
        "open_x_embodiment",
        "open_x_embodiment-main",
        "https://github.com/google-deepmind/open_x_embodiment",
        "cross-embodiment RLDS and representation-pretraining reference",
        "external_pretraining_only",
        ("README.md", "LICENSE"),
        "Audit a bounded component dataset before any preprocessing; keep model caches and converted payload outside Rivermark Git.",
        "Cross-robot actions cannot be relabelled as CF2X flight actions.",
    ),
    ExternalSourceSpec(
        "open_aoe",
        "Open-AoE-main",
        "https://github.com/ant-research/Open-AoE",
        "egocentric visual, language, and world-model pretraining source",
        "external_pretraining_only",
        ("README.md", "README_zh.md", "aoe-training-ready/ACTION_SPEC.md", "aoe-training-ready/lerobot/scripts/convert_open_aoe_to_lerobot.py"),
        "Use the existing bounded segment auditor before training; never bulk-convert the corpus into the Rivermark repository.",
        "The 20D hand and optional 6D camera-motion actions are not CF2X flight controls.",
    ),
    ExternalSourceSpec(
        "rlds",
        "rlds-main",
        "https://github.com/google-research/rlds",
        "episode and action-timing semantics reference",
        "semantic_reference_only",
        ("README.md", "rlds/tfds/episode_writer.py"),
        "Use its step semantics, but keep the dependency-light Rivermark interchange until a full TFDS projection is independently read back.",
        "RLDS is a representation, not evidence that an action has the same actuator semantics.",
    ),
    ExternalSourceSpec(
        "robocasa",
        "robocasa-main",
        "https://github.com/robocasa/robocasa",
        "task/split registry and LeRobot dataset-product reference",
        "protocol_and_projection_reference",
        ("README.md", "robocasa/utils/dataset_registry.py", "robocasa/utils/lerobot_utils.py"),
        "Reuse split certificates and dataset-soup accounting; do not use all-at-once statistics code on the full corpus on a memory-constrained host.",
        "Kitchen manipulation data cannot supply a flight action target or City-Lite score.",
    ),
    ExternalSourceSpec(
        "robomimic",
        "robomimic-master",
        "https://github.com/ARISE-Initiative/robomimic",
        "offline imitation-learning experiment reference",
        "offline_baseline_adapter_candidate",
        ("README.md", "robomimic/utils/dataset.py", "robomimic/utils/file_utils.py"),
        "Enable only selected modalities and bounded sequence/cache settings; compute statistics incrementally rather than materializing all image observations.",
        "HDF5 arm demonstrations require an explicit flight-ABI export before they can train a T2 policy.",
    ),
    ExternalSourceSpec(
        "tartanairpy",
        "tartanairpy-main",
        "https://github.com/castacks/tartanairpy",
        "visual odometry, trajectory, and sensor-geometry diagnostic reference",
        "passive_perception_reference",
        ("README.md", "tartanair/dataset.py", "tartanair/evaluator.py"),
        "Construct bounded trajectory indexes; the supplied image loader explicitly rejects IMU and LiDAR modalities and must not be presented as a complete sensor loader.",
        "Trajectory pose/motion labels are passive perception supervision, not a CF2X control policy target.",
    ),
)

EXTERNAL_SOURCE_SPECS: Mapping[str, ExternalSourceSpec] = {spec.source_id: spec for spec in _SPECS}


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_ids(source_ids: Iterable[str] | None) -> tuple[str, ...]:
    if source_ids is None:
        return tuple(EXTERNAL_SOURCE_SPECS)
    requested = tuple(source_ids)
    if not requested or any(not isinstance(source_id, str) or not source_id for source_id in requested):
        raise ExternalSourceError("source_ids must be a non-empty sequence of source identifiers")
    if len(set(requested)) != len(requested):
        raise ExternalSourceError("source_ids must not contain duplicates")
    unknown = sorted(set(requested) - set(EXTERNAL_SOURCE_SPECS))
    if unknown:
        raise ExternalSourceError(f"unknown external source identifiers: {unknown}")
    return requested


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _snapshot_one(root: Path, spec: ExternalSourceSpec) -> dict[str, object]:
    source_root = (root / spec.local_directory).resolve()
    base = {
        "source": {
            "source_id": spec.source_id,
            "repository": spec.repository,
            "role": spec.role,
            "integration_level": spec.integration_level,
            "resource_constraint": spec.resource_constraint,
            "action_boundary": spec.action_boundary,
            "formal_rivermark_admission": False,
            "native_isaac_closed_loop_evidence": False,
        },
        "local_source_path_redacted": True,
    }
    if not source_root.is_dir() or not _is_within(source_root, root):
        return {**base, "status": "missing", "file_count": 0, "total_bytes": 0, "key_files": []}

    files = tuple(sorted((path for path in source_root.rglob("*") if path.is_file()), key=lambda path: path.as_posix()))
    key_files: list[dict[str, object]] = []
    missing: list[str] = []
    for relative in spec.key_paths:
        path = (source_root / relative).resolve()
        if not _is_within(path, source_root) or not path.is_file():
            missing.append(relative)
            continue
        key_files.append({"path": relative.replace("\\", "/"), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return {
        **base,
        "status": "complete" if not missing else "incomplete",
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "key_files": key_files,
        "missing_key_paths": missing,
    }


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def scan_external_source_snapshots(
    source_root: Path,
    *,
    source_ids: Iterable[str] | None = None,
    repository_root: Path | None = None,
    access_basis: str = "user_authorized_local_use",
) -> dict[str, object]:
    """Return a path-free, read-only audit of selected source checkouts.

    ``access_basis`` records the scope asserted for local technical study.  It
    does not make a release or redistribution assertion, and every source is
    frozen as external to formal Rivermark episode accounting.
    """

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise ExternalSourceError(f"external source root does not exist: {root}")
    if repository_root is not None and _is_within(root, Path(repository_root)):
        raise ExternalSourceError("external source snapshots must remain outside the Rivermark repository")
    if access_basis not in {"user_authorized_local_use", "upstream_terms_verified"}:
        raise ExternalSourceError("access_basis must be user_authorized_local_use or upstream_terms_verified")
    records = [_snapshot_one(root, EXTERNAL_SOURCE_SPECS[source_id]) for source_id in _source_ids(source_ids)]
    complete = all(record["status"] == "complete" for record in records)
    manifest: dict[str, object] = {
        "schema": EXTERNAL_SOURCE_SNAPSHOT_SCHEMA,
        "status": "complete" if complete else "incomplete",
        "access_basis": access_basis,
        "source_root_redacted": True,
        "payload_copied_into_rivermark": False,
        "formal_rivermark_admission": False,
        "native_isaac_closed_loop_evidence": False,
        "source_count": len(records),
        "complete_source_count": sum(record["status"] == "complete" for record in records),
        "records": records,
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest


def write_external_source_manifest(path: Path, manifest: Mapping[str, object], *, overwrite: bool = False) -> Path:
    """Atomically write a manifest already bound by ``manifest_sha256``."""

    if manifest.get("schema") != EXTERNAL_SOURCE_SNAPSHOT_SCHEMA:
        raise ExternalSourceError("manifest does not use the external source snapshot schema")
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise ExternalSourceError("manifest hash is missing or does not bind its content")
    output = Path(path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise ExternalSourceError(f"refusing to overwrite existing manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as stream:
            temporary_name = stream.name
            stream.write(_canonical_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a path-free technical audit of external source snapshots.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--access-basis", choices=("user_authorized_local_use", "upstream_terms_verified"), default="user_authorized_local_use")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = scan_external_source_snapshots(
            args.source_root,
            source_ids=args.source_id or None,
            repository_root=args.repository_root,
            access_basis=args.access_basis,
        )
        output = write_external_source_manifest(args.output, manifest, overwrite=args.overwrite)
    except ExternalSourceError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=True, sort_keys=True))
        return 2
    print(json.dumps({"status": manifest["status"], "output": str(output), "manifest_sha256": manifest["manifest_sha256"]}, ensure_ascii=True, sort_keys=True))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXTERNAL_SOURCE_SNAPSHOT_SCHEMA",
    "EXTERNAL_SOURCE_SPECS",
    "ExternalSourceError",
    "ExternalSourceSpec",
    "scan_external_source_snapshots",
    "write_external_source_manifest",
]
