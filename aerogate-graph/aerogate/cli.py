"""Command-line entry points for AeroGate Graph."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import __version__
from .reproducibility import DEFAULT_REPRODUCIBILITY_SEEDS, verify_reproducibility
from .scenarios import available_scenarios, run_smoke

PROJECT_NAME = "AeroGate Graph"
PROJECT_DESCRIPTION = (
    "A modular 2D drone-racing simulator for graph-based route planning, "
    "formation control, and dynamic gate navigation."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aerogate", description=PROJECT_DESCRIPTION)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info", help="print project and scenario metadata")
    smoke_parser = subparsers.add_parser("smoke", help="step a core environment without training")
    _add_scenario_options(smoke_parser, default_scenario="single-static")
    smoke_parser.add_argument("--seed", type=int, default=7)
    smoke_parser.add_argument("--steps", type=int, default=8)
    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="verify deterministic core rollouts across one or more explicit seeds",
    )
    _add_scenario_options(reproduce_parser, default_scenario="multi-static")
    reproduce_parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_REPRODUCIBILITY_SEEDS)
    reproduce_parser.add_argument("--steps", type=int, default=8)
    reproduce_parser.add_argument(
        "--output", type=Path, default=None, help="optional path for the JSON evidence report"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        print(
            json.dumps(
                {
                    "name": PROJECT_NAME,
                    "description": PROJECT_DESCRIPTION,
                    "scenarios": [
                        {
                            "name": scenario.name,
                            "mode": scenario.mode,
                            "dynamic_gates": scenario.dynamic_gates,
                            "description": scenario.description,
                        }
                        for scenario in available_scenarios()
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "smoke":
        payload = run_smoke(
            args.scenario,
            agents=args.agents,
            seed=args.seed,
            steps=args.steps,
        )
    else:
        payload = _reproducibility_payload(
            scenario=args.scenario,
            agents=args.agents,
            seeds=args.seeds,
            steps=args.steps,
        )
    _write_json(payload, output_path=getattr(args, "output", None))


def _add_scenario_options(parser: argparse.ArgumentParser, *, default_scenario: str) -> None:
    """Attach the shared public scenario arguments to a CLI subcommand."""

    parser.add_argument(
        "--scenario",
        choices=[scenario.name for scenario in available_scenarios()],
        default=default_scenario,
    )
    parser.add_argument("--agents", type=int, default=None, help="team size for multi-agent scenarios")


def _reproducibility_payload(
    *,
    scenario: str,
    agents: int | None,
    seeds: Sequence[int],
    steps: int,
) -> dict[str, object]:
    """Combine a deterministic report with the runtime facts needed to interpret it."""

    report = verify_reproducibility(
        scenario,
        agents=agents,
        seeds=seeds,
        steps=steps,
    ).to_dict()
    return {
        "schema_version": 1,
        "provenance": {
            "aerogate_version": __version__,
            "numpy_version": np.__version__,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        **report,
    }


def _write_json(payload: dict[str, object], *, output_path: Path | None) -> None:
    """Print a stable JSON report and optionally persist the identical text."""

    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
