"""Aggregate source-locked ACO3D L0 calibration reports without private truth.

This verifier deliberately certifies execution integrity only. It cannot turn a
Python source translation or an L0 replay into native-upstream, CF2X L1, or
Gate C evidence.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from aerocity_bench.canonical import content_hash, read_json, write_json  # noqa: E402

REPORT_SCHEMA = "org.aerocity.bench.aco3d-public-atlas-smoke.v1"
PANEL_SCHEMA = "org.aerocity.bench.aco3d-g2i-l0-calibration-panel.v1"
_ANCESTOR_LABEL = re.compile(r"^ancestor-[0-9]{2}$")
_REQUIRED_ANCESTORS = frozenset({"ancestor-00", "ancestor-01", "ancestor-02"})


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="ANCESTOR=PATH",
        help="One complete calibration replay report per required ancestor.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _bound_reports(values: list[str]) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not _ANCESTOR_LABEL.fullmatch(label) or not raw_path:
            raise ValueError("report binding must be ancestor-NN=PATH")
        path = Path(raw_path).resolve()
        if label in reports or not path.is_file():
            raise ValueError("each report label must be unique and reference a readable file")
        reports[label] = path
    if set(reports) != _REQUIRED_ANCESTORS:
        raise ValueError("panel requires exactly ancestors 00, 01, and 02")
    return reports


def _required_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _fraction(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


def build_panel(report_paths: dict[str, Path]) -> dict[str, Any]:
    """Validate a three-city calibration panel and emit public-safe aggregates."""

    if set(report_paths) != _REQUIRED_ANCESTORS:
        raise ValueError("panel requires exactly ancestors 00, 01, and 02")
    baseline: dict[str, object] | None = None
    city_hashes: set[str] = set()
    entries: list[dict[str, Any]] = []
    for label, path in sorted(report_paths.items()):
        raw = _required_mapping(read_json(path), f"{label} report")
        if raw.get("schema") != REPORT_SCHEMA:
            raise ValueError(f"{label} has an incompatible ACO3D report schema")
        if raw.get("scope") != "calibration_only_source_locked_translation_smoke":
            raise ValueError(f"{label} is not a calibration-only source-translation report")
        if raw.get("formal_score_eligible") is not False:
            raise ValueError(f"{label} must remain ineligible for formal scoring")
        if raw.get("return_closure_required") is not True:
            raise ValueError(f"{label} must explicitly require return closure")
        if raw.get("pass") is not True:
            raise ValueError(f"{label} execution-integrity replay did not pass")
        upstream = _required_mapping(raw.get("upstream"), f"{label}.upstream")
        adapter = _required_mapping(raw.get("adapter"), f"{label}.adapter")
        inputs = _required_mapping(raw.get("public_input_hashes"), f"{label}.public_input_hashes")
        execution = _required_mapping(raw.get("execution"), f"{label}.execution")
        if upstream.get("source_checkout_verified") is not True:
            raise ValueError(f"{label} did not verify its locked upstream checkout")
        if upstream.get("upstream_runtime_executed") is not False:
            raise ValueError(f"{label} cannot claim native upstream execution")
        if execution.get("formal_score_eligible") is not False:
            raise ValueError(f"{label} execution must remain L0-ineligible")
        if execution.get("all_returned_home") is not True:
            raise ValueError(f"{label} did not return every vehicle home")
        if execution.get("failure_categories") != []:
            raise ValueError(f"{label} has a non-empty failure denominator")
        zero_fields = (
            "collision_count",
            "out_of_bounds_actions",
            "deadline_miss_tick_count",
        )
        if any(
            _nonnegative_int(execution.get(field), f"{label}.{field}") != 0
            for field in zero_fields
        ):
            raise ValueError(f"{label} has a safety or deadline failure")
        city_hash = inputs.get("city")
        if not isinstance(city_hash, str) or len(city_hash) != 64:
            raise ValueError(f"{label} has no public city hash")
        if city_hash in city_hashes:
            raise ValueError("calibration ancestors must use distinct public city hashes")
        city_hashes.add(city_hash)
        comparable = {
            "upstream_url": upstream.get("url"),
            "upstream_commit": upstream.get("commit"),
            "upstream_license": upstream.get("license"),
            "source_lock_sha256": upstream.get("source_lock_sha256"),
            "adapter_version": upstream.get("adapter_version"),
            "adapter_source_sha256": adapter.get("adapter_source_sha256"),
            "runner_source_sha256": adapter.get("runner_source_sha256"),
            "release_config_sha256": inputs.get("release_config"),
            "pass_semantics": raw.get("pass_semantics"),
        }
        if baseline is None:
            baseline = comparable
        elif comparable != baseline:
            raise ValueError("calibration reports do not share one locked method and contract")
        coverage = _required_mapping(execution.get("inspection_coverage"), f"{label}.coverage")
        entries.append(
            {
                "ancestor": label,
                "city_sha256": city_hash,
                "report_sha256": content_hash(raw),
                "task_time_s": float(execution.get("task_time_s", 0.0)),
                "receipt_count": _nonnegative_int(
                    execution.get("receipt_count"), f"{label}.receipt_count"
                ),
                "observe_request_count": _nonnegative_int(
                    execution.get("observe_request_count"), f"{label}.observe_request_count"
                ),
                "anonymous_confirmation_count": _nonnegative_int(
                    execution.get("confirmation_count"), f"{label}.confirmation_count"
                ),
                "inspection_area_fraction": _fraction(
                    coverage.get("area_fraction"), f"{label}.area_fraction"
                ),
                "inspection_cell_fraction": _fraction(
                    coverage.get("cell_fraction"), f"{label}.cell_fraction"
                ),
            }
        )
    assert baseline is not None
    area_fractions = [entry["inspection_area_fraction"] for entry in entries]
    cell_fractions = [entry["inspection_cell_fraction"] for entry in entries]
    report = {
        "schema": PANEL_SCHEMA,
        "scope": "three-ancestor-calibration-only-source-translation-l0-panel",
        "formal_score_eligible": False,
        "gate_c_eligible": False,
        "status": "L0_CALIBRATION_PANEL_PASS_NOT_GATE_C",
        "not_established": [
            "native MATLAB or Octave source equivalence",
            "upstream multi-UAV allocation",
            "CF2X Isaac L1 execution",
            "formal leaderboard score",
        ],
        "locked_method_and_contract": baseline,
        "ancestors": entries,
        "aggregate": {
            "minimum_inspection_area_fraction": min(area_fractions),
            "mean_inspection_area_fraction": sum(area_fractions) / len(area_fractions),
            "minimum_inspection_cell_fraction": min(cell_fractions),
            "mean_inspection_cell_fraction": sum(cell_fractions) / len(cell_fractions),
            "all_returned_home": True,
            "total_anonymous_confirmation_count": sum(
                entry["anonymous_confirmation_count"] for entry in entries
            ),
        },
    }
    report["report_hash"] = content_hash(report)
    return report


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    report = build_panel(_bound_reports(args.report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print("ACO3D_G2I_L0_PANEL=PASS")


if __name__ == "__main__":
    main()
