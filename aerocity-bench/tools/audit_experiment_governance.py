"""Audit whether any experimental result was promoted across a frozen boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import write_json  # noqa: E402
from aerocity_bench.experiment_governance import (  # noqa: E402
    load_and_audit_experiment_governance,
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=_REPOSITORY_ROOT / "configs" / "experiment-governance-v1.json",
    )
    parser.add_argument(
        "--external-evidence-root",
        type=Path,
        help="explicit root containing hash-bound, uncommitted registered evidence",
    )
    parser.add_argument(
        "--external-evidence-manifest",
        type=Path,
        help="source-commit- and registry-bound external evidence manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit evidence: {args.output}")
    report = load_and_audit_experiment_governance(
        args.registry,
        repository_root=_REPOSITORY_ROOT,
        external_evidence_root=args.external_evidence_root,
        external_evidence_manifest=args.external_evidence_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"EXPERIMENT_GOVERNANCE={report['overall_status']}")
    return 0 if report["overall_status"] == "CONTAINMENT_PASS_FORMAL_NO_GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
