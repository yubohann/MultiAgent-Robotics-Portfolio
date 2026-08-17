"""Command-line entry point for the coverage report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ..failure_ledger import FailureLedgerError, load_failure_ledger
from .common import CollectionProtocolError
from .constants import COVERAGE_REPORT_SCHEMA
from .coverage import coverage_report
from .validate import load_collection_protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--failure-ledger", type=Path, required=True, help="redacted public failure ledger JSONL")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        protocol = load_collection_protocol(args.protocol)
        report = coverage_report(protocol, load_failure_ledger(args.failure_ledger))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, FailureLedgerError, CollectionProtocolError, ValueError) as exc:
        print(json.dumps({"schema": COVERAGE_REPORT_SCHEMA, "complete": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
