"""Fail-closed audits for artifacts that must never enter Git history.

The audit inspects the Git index rather than the working tree. Ignored local
captures therefore remain usable for development, while a force-added capture,
asset, model, prompt, or oversized file is rejected before a release commit.
The default audit inspects the current Git index.  The explicit history audit
walks all ref-reachable objects, including paths that were later deleted.  It
does not rewrite history or delete any artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Sequence

REPOSITORY_AUDIT_SCHEMA = "org.rivermark.benchmark.repository-audit.v1"
DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024
_FORBIDDEN_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".usd", ".usda", ".usdc", ".npz",
    ".npy", ".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".parquet", ".arrow",
})
_FORBIDDEN_ROOTS = frozenset({"artifacts", "captures", "checkpoints", "data", "evidence", "models", "private"})
_ALLOWED_DEMO_FILES = frozenset({"demos/readme.md", "demos/manifest.json"})
_PROMPT_NAMES = frozenset({
    "isaac_execution_prompt.md",
    "paper_reading_prompt.md",
})


@dataclass(frozen=True)
class RepositoryAuditIssue:
    code: str
    path: str
    message: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class RepositoryAuditReport:
    root: str
    schema: str
    tracked_file_count: int
    max_file_bytes: int
    status: str
    issues: tuple[RepositoryAuditIssue, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "schema": self.schema,
            "tracked_file_count": self.tracked_file_count,
            "max_file_bytes": self.max_file_bytes,
            "status": self.status,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class GitHistoryAuditReport:
    root: str
    schema: str
    reachable_entry_count: int
    reachable_blob_count: int
    max_file_bytes: int
    status: str
    issues: tuple[RepositoryAuditIssue, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "schema": self.schema,
            "reachable_entry_count": self.reachable_entry_count,
            "reachable_blob_count": self.reachable_blob_count,
            "max_file_bytes": self.max_file_bytes,
            "status": self.status,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class SourceDistributionAuditReport:
    archive: str
    schema: str
    member_count: int
    max_file_bytes: int
    status: str
    issues: tuple[RepositoryAuditIssue, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "archive": self.archive,
            "schema": self.schema,
            "member_count": self.member_count,
            "max_file_bytes": self.max_file_bytes,
            "status": self.status,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _tracked_paths(root: Path) -> tuple[str, ...]:
    try:
        output = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect Git index under {root}: {exc}") from exc
    return tuple(sorted(part.decode("utf-8", errors="surrogateescape") for part in output.split(b"\0") if part))


def _path_issues(path: str) -> tuple[RepositoryAuditIssue, ...]:
    normalized = path.replace("\\", "/").lower()
    parts = normalized.split("/")
    issues: list[RepositoryAuditIssue] = []
    if parts and parts[0] in _FORBIDDEN_ROOTS:
        issues.append(RepositoryAuditIssue("forbidden_path", path, f"tracked artifact root is not allowed: {parts[0]}"))
    if normalized.startswith("demos/") and normalized not in _ALLOWED_DEMO_FILES:
        issues.append(RepositoryAuditIssue("forbidden_path", path, "only the demo README and manifest may be tracked"))
    if Path(normalized).suffix in _FORBIDDEN_EXTENSIONS:
        issues.append(RepositoryAuditIssue("forbidden_extension", path, "binary capture, asset, model, or payload extension is not allowed"))
    if Path(normalized).name in _PROMPT_NAMES or normalized.endswith("/prompt.md"):
        issues.append(RepositoryAuditIssue("operator_prompt", path, "operator prompt files are not release artifacts"))
    return tuple(issues)


def audit_repository(root: Path, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> RepositoryAuditReport:
    """Audit all files currently tracked by Git, without inspecting ignored files."""
    if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be a positive integer")
    resolved = Path(root).expanduser().resolve()
    paths = _tracked_paths(resolved)
    issues: list[RepositoryAuditIssue] = []
    for relative in paths:
        path_issues = _path_issues(relative)
        candidate = resolved / Path(relative)
        size: int | None = None
        try:
            if candidate.is_file():
                size = candidate.stat().st_size
        except OSError as exc:
            issues.append(RepositoryAuditIssue("stat_failed", relative, str(exc)))
            continue
        issues.extend(path_issues)
        if size is None:
            issues.append(RepositoryAuditIssue("missing_tracked_file", relative, "tracked file is absent from the worktree"))
        elif size > max_file_bytes:
            issues.append(RepositoryAuditIssue("oversized_file", relative, f"tracked file exceeds {max_file_bytes} bytes", size))
    return RepositoryAuditReport(str(resolved), REPOSITORY_AUDIT_SCHEMA, len(paths), max_file_bytes, "passed" if not issues else "blocked", tuple(issues))


def _reachable_history_entries(root: Path) -> tuple[tuple[str, str, int], ...]:
    """Return ``(object_id, path, size)`` for ref-reachable file objects.

    ``git rev-list --objects --all`` includes objects from older commits that
    are still reachable through a branch or tag, which is exactly the history
    surface a maintainer must inspect before claiming an artifact was removed.
    Object sizes come from ``cat-file`` instead of reading blobs into memory.
    """

    try:
        listing = subprocess.check_output(
            ["git", "-C", str(root), "rev-list", "--objects", "--all"],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect reachable Git history under {root}: {exc}") from exc
    pairs: list[tuple[str, str]] = []
    object_ids: list[str] = []
    for raw in listing.splitlines():
        object_id, separator, raw_path = raw.partition(b" ")
        if not separator or len(object_id) not in {40, 64} or not raw_path:
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        object_id_text = object_id.decode("ascii")
        pairs.append((object_id_text, path))
        object_ids.append(object_id_text)
    if not object_ids:
        return ()
    try:
        details = subprocess.check_output(
            ["git", "-C", str(root), "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
            input=("\n".join(object_ids) + "\n").encode("ascii"),
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect reachable Git object sizes under {root}: {exc}") from exc
    sizes: dict[str, int] = {}
    for raw in details.splitlines():
        fields = raw.decode("ascii", errors="replace").split()
        if len(fields) == 3 and fields[1] == "blob":
            try:
                sizes[fields[0]] = int(fields[2])
            except ValueError:
                continue
    return tuple((object_id, path, sizes[object_id]) for object_id, path in pairs if object_id in sizes)


def audit_git_history(root: Path, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> GitHistoryAuditReport:
    """Audit ref-reachable history, including files removed in later commits.

    This is intentionally opt-in because an existing repository may have
    historical evidence that requires an owner-approved history rewrite.  A
    blocked result is diagnostic evidence; it never mutates refs or objects.
    """

    if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be a positive integer")
    resolved = Path(root).expanduser().resolve()
    entries = _reachable_history_entries(resolved)
    issues: list[RepositoryAuditIssue] = []
    for object_id, relative, size in entries:
        path_issues = _path_issues(relative)
        for issue in path_issues:
            issues.append(
                RepositoryAuditIssue(
                    f"historical_{issue.code}",
                    relative,
                    f"reachable history retains object {object_id}: {issue.message}",
                    size,
                )
            )
        if size > max_file_bytes:
            issues.append(
                RepositoryAuditIssue(
                    "historical_oversized_object",
                    relative,
                    f"reachable history retains an object larger than {max_file_bytes} bytes",
                    size,
                )
            )
    return GitHistoryAuditReport(
        str(resolved),
        REPOSITORY_AUDIT_SCHEMA,
        len(entries),
        len({object_id for object_id, _, _ in entries}),
        max_file_bytes,
        "passed" if not issues else "blocked",
        tuple(issues),
    )


def audit_source_distribution(
    archive_path: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> SourceDistributionAuditReport:
    """Audit the actual sdist payload rather than only its source Git index.

    Setuptools can reuse a local ``SOURCES.txt`` from an ignored ``egg-info``
    directory.  A clean index is therefore insufficient evidence that an sdist
    does not contain historical video, evidence, prompts, or other excluded
    payloads.  This function never extracts the archive.
    """

    if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be a positive integer")
    archive = Path(archive_path).expanduser().resolve()
    issues: list[RepositoryAuditIssue] = []
    member_count = 0
    try:
        with tarfile.open(archive, "r:*") as source:
            for member in source.getmembers():
                if member.isdir():
                    continue
                member_count += 1
                parts = PurePosixPath(member.name).parts
                if (
                    len(parts) < 2
                    or PurePosixPath(member.name).is_absolute()
                    or any(part in {"", ".", ".."} for part in parts)
                ):
                    issues.append(
                        RepositoryAuditIssue(
                            "unsafe_archive_path",
                            member.name,
                            "source distribution member must be beneath one package root",
                            member.size,
                        )
                    )
                    continue
                relative = PurePosixPath(*parts[1:]).as_posix()
                if not member.isfile():
                    issues.append(
                        RepositoryAuditIssue(
                            "unsafe_archive_member",
                            relative,
                            "source distribution may contain regular files only",
                            member.size,
                        )
                    )
                    continue
                issues.extend(_path_issues(relative))
                if member.size > max_file_bytes:
                    issues.append(
                        RepositoryAuditIssue(
                            "oversized_archive_member",
                            relative,
                            f"source distribution member exceeds {max_file_bytes} bytes",
                            member.size,
                        )
                    )
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"cannot inspect source distribution {archive}: {exc}") from exc
    return SourceDistributionAuditReport(
        str(archive),
        REPOSITORY_AUDIT_SCHEMA,
        member_count,
        max_file_bytes,
        "passed" if not issues else "blocked",
        tuple(issues),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--history",
        action="store_true",
        help="audit all ref-reachable history, including paths deleted in later commits",
    )
    mode.add_argument(
        "--sdist",
        type=Path,
        help="audit exactly one built source-distribution tarball",
    )
    args = parser.parse_args(argv)
    try:
        if args.sdist is not None:
            sdist = args.sdist.expanduser().resolve()
            if sdist.is_dir():
                archives = tuple(sorted(sdist.glob("*.tar.gz")))
                if len(archives) != 1:
                    raise RuntimeError(
                        f"source-distribution directory must contain exactly one .tar.gz archive: {sdist}"
                    )
                sdist = archives[0]
            report = audit_source_distribution(sdist, max_file_bytes=args.max_file_bytes)
        elif args.history:
            report = audit_git_history(args.root, max_file_bytes=args.max_file_bytes)
        else:
            report = audit_repository(args.root, max_file_bytes=args.max_file_bytes)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"schema": REPOSITORY_AUDIT_SCHEMA, "status": "blocked", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report.as_dict(), ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report.status == "passed" else 2


__all__ = [
    "DEFAULT_MAX_FILE_BYTES",
    "REPOSITORY_AUDIT_SCHEMA",
    "RepositoryAuditIssue",
    "RepositoryAuditReport",
    "GitHistoryAuditReport",
    "SourceDistributionAuditReport",
    "audit_repository",
    "audit_git_history",
    "audit_source_distribution",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
