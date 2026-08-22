"""Write an immutable AeroCityBench development environment manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import write_json  # noqa: E402
from aerocity_bench.environment_manifest import build_environment_manifest  # noqa: E402


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf2x-usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-config", type=Path)
    parser.add_argument("--asset-audit", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite environment evidence: {args.output}")
    report = build_environment_manifest(
        repository_root=_REPOSITORY_ROOT,
        cf2x_usd=args.cf2x_usd,
        release_config=args.release_config,
        asset_audit=args.asset_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"ENVIRONMENT_MANIFEST={report['source_tree']['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
