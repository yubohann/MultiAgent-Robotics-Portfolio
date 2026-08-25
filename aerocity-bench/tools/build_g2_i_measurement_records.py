"""Build G2-I measurement rows from precommitted L1 evidence, never by hand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerocity_bench.canonical import read_json, write_json
from aerocity_bench.measurement_aggregator import aggregate_measurement_records


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(
    protocol_path: Path,
    panel_manifest_path: Path,
    evidence_manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite machine-built records: {output_path}")
    protocol = read_json(protocol_path)
    panel = read_json(panel_manifest_path)
    report = aggregate_measurement_records(
        evidence_manifest_path,
        protocol_hash=str(protocol["protocol_hash"]),
        panel_manifest_hash=str(panel["panel_hash"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    report = run(
        args.protocol.resolve(),
        args.panel_manifest.resolve(),
        args.evidence_manifest.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
