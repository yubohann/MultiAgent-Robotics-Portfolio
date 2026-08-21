"""Entry point for the FraudGraph engineering toolkit.

The CLI exposes the framework's public command surface. The research pipeline
behind these commands is withheld until the paper is published; invoking a
withheld pipeline prints the notice.
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv=None) -> argparse.Namespace:
    """Set up the CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="fraud_ml_engineering",
        description="Graph-and-sequence fraud detection engineering framework (core implementation withheld).",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset selection for framework runs (dataset names are resolved at runtime).",
    )
    parser.add_argument(
        "--federated_rounds",
        type=int,
        default=24,
        help="Number of federated-style training rounds.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the command surface without running the pipeline.")
    return parser.parse_args(argv)


def _withheld() -> None:
    print("Core research pipeline is withheld until the associated paper is published. See NOTICE.md.")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print("CLI surface OK")
        return 0
    _withheld()
    return 0


if __name__ == "__main__":
    sys.exit(main())