"""Create an auditable comparison report from explicit experiment records.

The tool never searches artifact directories or selects a stronger historical run. Every
record must declare its data revision, split policy, validation-only selection policy,
seed, and validation/test metrics before it can appear in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_ml_engineering.paths import ARTIFACTS_ROOT
from fraud_ml_engineering.run_artifacts import write_json


DEFAULT_OUTPUT_ROOT = ARTIFACTS_ROOT / "experiments" / "auditable_comparison"
REQUIRED_RECORD_FIELDS = (
    "dataset",
    "model",
    "data_revision",
    "split_policy",
    "selection_policy",
    "seed",
    "metrics",
)
DISALLOWED_SOURCE_MARKERS = ("smoke", "debug", "probe", "stagecheck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        action="append",
        required=True,
        help="Explicit JSON result record. Repeat once for each comparison row.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for the comparison JSON and Markdown report.",
    )
    parser.add_argument(
        "--title",
        default="Auditable SplitGNN Comparison",
        help="Report title.",
    )
    return parser.parse_args()


def _read_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Record must be a JSON object: {path}")
    return payload


def _record_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_metric_block(record_path: Path, name: str, value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Record {record_path.name} requires a non-empty metrics.{name} object")
    normalized: dict[str, float] = {}
    for metric_name, metric_value in value.items():
        if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
            raise ValueError(f"Record {record_path.name} has a non-numeric metrics.{name}.{metric_name}")
        metric = float(metric_value)
        if not math.isfinite(metric):
            raise ValueError(f"Record {record_path.name} has a non-finite metrics.{name}.{metric_name}")
        normalized[str(metric_name)] = metric
    return normalized


def validate_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one explicit result record for reporting."""

    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Record {path.name} is missing required fields: {', '.join(missing)}")
    if any(marker in path.name.lower() for marker in DISALLOWED_SOURCE_MARKERS):
        raise ValueError(f"Record {path.name} appears to be a temporary or smoke artifact")

    selection_policy = str(payload["selection_policy"]).strip()
    if not selection_policy.lower().startswith("validation_only"):
        raise ValueError(
            f"Record {path.name} must declare a validation_only selection policy, got {selection_policy!r}"
        )
    try:
        seed = int(payload["seed"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Record {path.name} has an invalid seed") from error

    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError(f"Record {path.name} requires a metrics object")
    validation = _validate_metric_block(path, "validation", metrics.get("validation"))
    test = _validate_metric_block(path, "test", metrics.get("test"))

    for field in ("dataset", "model", "data_revision", "split_policy"):
        if not str(payload[field]).strip():
            raise ValueError(f"Record {path.name} has an empty {field}")

    return {
        "dataset": str(payload["dataset"]).strip(),
        "model": str(payload["model"]).strip(),
        "data_revision": str(payload["data_revision"]).strip(),
        "split_policy": str(payload["split_policy"]).strip(),
        "selection_policy": selection_policy,
        "seed": seed,
        "metrics": {"validation": validation, "test": test},
        "source_record": path.name,
        "source_sha256": _record_digest(path),
    }


def _validate_dataset_revisions(records: list[dict[str, Any]]) -> None:
    revisions: dict[str, set[str]] = {}
    splits: dict[str, set[str]] = {}
    for record in records:
        revisions.setdefault(record["dataset"], set()).add(record["data_revision"])
        splits.setdefault(record["dataset"], set()).add(record["split_policy"])
    inconsistent_revisions = {dataset: values for dataset, values in revisions.items() if len(values) > 1}
    if inconsistent_revisions:
        details = "; ".join(f"{dataset}: {sorted(values)}" for dataset, values in inconsistent_revisions.items())
        raise ValueError(f"Cannot compare mixed data revisions: {details}")
    inconsistent_splits = {dataset: values for dataset, values in splits.items() if len(values) > 1}
    if inconsistent_splits:
        details = "; ".join(f"{dataset}: {sorted(values)}" for dataset, values in inconsistent_splits.items())
        raise ValueError(f"Cannot compare mixed split policies: {details}")


def build_report(title: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a report from records already approved by ``validate_record``."""

    _validate_dataset_revisions(records)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    datasets = sorted({record["dataset"] for record in records})
    return {
        "schema_version": 1,
        "generated_at": timestamp,
        "title": str(title),
        "report_policy": {
            "record_selection": "explicit_input_only",
            "ranking": "not_performed",
            "test_metric_policy": "reported_after_validation_only_selection",
        },
        "datasets": datasets,
        "records": records,
    }


def _metric_value(record: dict[str, Any], partition: str, metric: str) -> str:
    value = record["metrics"][partition].get(metric)
    return "-" if value is None else f"{float(value):.6f}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render a report without ranking or implicitly selecting any supplied row."""

    lines = [
        f"# {report['title']}",
        "",
        f"- generated_at: `{report['generated_at']}`",
        "- record_selection: `explicit_input_only`",
        "- ranking: `not_performed`",
        "- test_metric_policy: `reported_after_validation_only_selection`",
        "",
    ]
    for dataset in report["datasets"]:
        dataset_records = [record for record in report["records"] if record["dataset"] == dataset]
        first = dataset_records[0]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- data_revision: `{first['data_revision']}`",
                f"- split_policy: `{first['split_policy']}`",
                "",
                "| model | seed | selection_policy | validation_auc | test_auc | test_pr_auc | test_f1_macro | source_record |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for record in dataset_records:
            lines.append(
                "| {model} | {seed} | {selection_policy} | {validation_auc} | {test_auc} | {test_pr_auc} | {test_f1_macro} | {source_record} |".format(
                    model=record["model"],
                    seed=record["seed"],
                    selection_policy=record["selection_policy"],
                    validation_auc=_metric_value(record, "validation", "auc"),
                    test_auc=_metric_value(record, "test", "auc"),
                    test_pr_auc=_metric_value(record, "test", "pr_auc"),
                    test_f1_macro=_metric_value(record, "test", "f1_macro"),
                    source_record=record["source_record"],
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.record]
    records = [validate_record(path, _read_record(path)) for path in paths]
    report = build_report(args.title, records)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "auditable_comparison.json"
    markdown_path = output_root / "auditable_comparison.md"
    write_json(json_path, report)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "markdown_path": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
