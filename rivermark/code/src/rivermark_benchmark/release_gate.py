"""Fail closed unless a Rivermark dataset is structurally releaseable."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .formal_dataset import verify_dataset_integrity


@dataclass(frozen=True)
class ReleaseGateIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ReleaseGateReport:
    dataset_root: Path
    episode_count: int
    minimum_episodes: int
    issues: tuple[ReleaseGateIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def audit_release_dataset(dataset_root: Path, *, minimum_episodes: int = 1) -> ReleaseGateReport:
    """Combine formal integrity checks with a non-empty release requirement."""

    if minimum_episodes < 1:
        raise ValueError("minimum_episodes must be at least one")
    root = dataset_root.resolve()
    issues: list[ReleaseGateIssue] = []
    integrity = verify_dataset_integrity(root)
    issues.extend(
        ReleaseGateIssue(issue.code, issue.path, issue.message) for issue in integrity.issues
    )

    index_path = root / "manifests" / "dataset_index.json"
    episode_count = 0
    try:
        index: Any = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(ReleaseGateIssue("dataset_index", "manifests/dataset_index.json", str(exc)))
    else:
        if not isinstance(index, dict):
            issues.append(
                ReleaseGateIssue("dataset_index", "manifests/dataset_index.json", "expected an object")
            )
        else:
            raw_count = index.get("episode_count")
            if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count >= 0:
                episode_count = raw_count
            else:
                issues.append(
                    ReleaseGateIssue(
                        "episode_count", "manifests/dataset_index.json", "episode_count must be non-negative"
                    )
                )
    if episode_count < minimum_episodes:
        issues.append(
            ReleaseGateIssue(
                "minimum_episode_count",
                "manifests/dataset_index.json",
                f"formal release requires at least {minimum_episodes} episode(s), found {episode_count}",
            )
        )
    return ReleaseGateReport(root, episode_count, minimum_episodes, tuple(issues))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--minimum-episodes", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_release_dataset(args.dataset_root, minimum_episodes=args.minimum_episodes)
    payload = {
        "valid": report.valid,
        "dataset_root": str(report.dataset_root),
        "episode_count": report.episode_count,
        "minimum_episodes": report.minimum_episodes,
        "issues": [asdict(issue) for issue in report.issues],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
