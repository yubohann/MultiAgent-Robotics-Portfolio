"""Create a hash-bound external evidence manifest for a clean-source audit.

The generated file intentionally omits the absolute evidence root.  A verifier
must provide both the manifest and a root, and every registered path is checked
against its SHA-256 digest before it can satisfy a missing local receipt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import read_json, write_json  # noqa: E402
from aerocity_bench.experiment_governance import (  # noqa: E402
    build_external_evidence_manifest,
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=_REPOSITORY_ROOT / "configs" / "experiment-governance-v1.json",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="root that currently holds every registered evidence file",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence manifest: {args.output}")
    manifest = build_external_evidence_manifest(
        repository_root=_REPOSITORY_ROOT,
        registry_path=args.registry,
        registry=read_json(args.registry),
        evidence_root=args.evidence_root,
    )
    write_json(args.output, manifest)
    print(f"EXTERNAL_EVIDENCE_MANIFEST={manifest['manifest_hash']}")
    print(f"REGISTERED_PATH_COUNT={len(manifest['evidence'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
