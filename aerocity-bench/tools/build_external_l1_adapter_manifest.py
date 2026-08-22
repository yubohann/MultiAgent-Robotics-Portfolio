"""Create a local, hash-bound external G2-I L1 process manifest.

The manifest deliberately contains workstation paths and is therefore a local
launch input, not public evidence.  The L1 runner publishes only its content
hash and provenance hashes after it verifies the adapter, checkpoint, runtime
environment, and clean upstream Git worktree.
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

from aerocity_bench.adapters import (  # noqa: E402
    EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA,
    load_external_l1_adapter_manifest,
)
from aerocity_bench.canonical import file_hash, write_json  # noqa: E402


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--upstream-license", required=True)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--adapter-source", type=Path, required=True)
    artifact_input = parser.add_mutually_exclusive_group(required=True)
    artifact_input.add_argument("--checkpoint", type=Path)
    artifact_input.add_argument(
        "--solver-lock",
        type=Path,
        help="immutable lock file for an external solver with no learned checkpoint",
    )
    parser.add_argument("--runtime-environment-manifest", type=Path, required=True)
    parser.add_argument(
        "--runtime-python",
        type=Path,
        required=True,
        help="isolated Python executable used to launch the external process",
    )
    parser.add_argument(
        "--task-domain",
        choices=("2d_exploration_transfer", "3d_geometry_search"),
        required=True,
    )
    parser.add_argument(
        "--comparability-claim",
        choices=("transfer_diagnostic", "substantive_3d"),
        required=True,
    )
    command_input = parser.add_mutually_exclusive_group(required=True)
    command_input.add_argument(
        "--command-template-json",
        help="JSON string with verified placeholders, never a shell command",
    )
    command_input.add_argument(
        "--command-template-file",
        type=Path,
        help="UTF-8 JSON string array; avoids shell quote rewriting on Windows",
    )
    return parser.parse_args(argv)


def _relative_or_absolute(path: Path, output_parent: Path) -> str:
    try:
        return path.resolve().relative_to(output_parent.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite external adapter manifest: {output}")
    adapter = args.adapter_source.resolve()
    artifact_kind = "checkpoint" if args.checkpoint is not None else "solver_lock"
    artifact_path = args.checkpoint if args.checkpoint is not None else args.solver_lock
    assert artifact_path is not None
    execution_artifact = artifact_path.resolve()
    upstream = args.upstream_source.resolve()
    runtime_environment = args.runtime_environment_manifest.resolve()
    runtime_python = args.runtime_python.resolve()
    inputs_exist = (
        adapter.is_file()
        and execution_artifact.is_file()
        and upstream.is_dir()
        and runtime_environment.is_file()
        and runtime_python.is_file()
    )
    if not inputs_exist:
        raise ValueError(
            "adapter, execution artifact, upstream source, runtime manifest, and runtime Python "
            "must exist"
        )
    if args.command_template_file is not None:
        template_file = args.command_template_file.resolve()
        if not template_file.is_file():
            raise ValueError("command template file must exist")
        # ``utf-8-sig`` accepts both portable UTF-8 and Windows PowerShell's
        # UTF-8 BOM output without treating the BOM as JSON content.
        command_input = template_file.read_text(encoding="utf-8-sig")
    else:
        command_input = args.command_template_json
    try:
        command = json.loads(command_input)
    except json.JSONDecodeError as exc:
        raise ValueError("command template must be a JSON string array") from exc
    if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
        raise ValueError("command template must be a non-empty JSON string array")
    payload: dict[str, Any] = {
        "schema": EXTERNAL_L1_ADAPTER_MANIFEST_SCHEMA,
        "declaration": {
            "adapter_id": args.adapter_id,
            "method_id": args.method_id,
            "capability_profile": "G2-I",
            "upstream_url": args.upstream_url,
            "upstream_commit": args.upstream_commit,
            "upstream_license": args.upstream_license,
            "process_boundary": "process",
            "training_allowed": False,
            "decentralized_execution": False,
            "runtime_image_digest": None,
        },
        "command": command,
        "adapter_source_path": _relative_or_absolute(adapter, output.parent),
        "adapter_source_sha256": file_hash(adapter),
        "upstream_source_path": _relative_or_absolute(upstream, output.parent),
        "execution_artifact_kind": artifact_kind,
        "execution_artifact_path": _relative_or_absolute(execution_artifact, output.parent),
        "execution_artifact_sha256": file_hash(execution_artifact),
        "runtime_environment_sha256": file_hash(runtime_environment),
        "runtime_python_path": _relative_or_absolute(runtime_python, output.parent),
        "runtime_python_sha256": file_hash(runtime_python),
        "task_domain": args.task_domain,
        "comparability_claim": args.comparability_claim,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    load_external_l1_adapter_manifest(output)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
