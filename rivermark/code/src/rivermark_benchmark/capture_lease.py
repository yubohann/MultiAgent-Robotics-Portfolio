"""Process-owned lease for the single native Isaac AppLauncher."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


class AppLauncherLeaseError(RuntimeError):
    """The repository already has an AppLauncher owner."""


def _metadata_bytes(metadata: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


class AppLauncherLease:
    """Hold an OS-backed exclusive lock for one AppLauncher lifetime.

    The lock file is deliberately tiny and is never treated as capture data.
    Kernel file-lock ownership means an abrupt process exit releases the lease;
    no stale PID or wall-clock timeout can incorrectly permit two Kit owners.
    """

    def __init__(self, path: Path, *, metadata: Mapping[str, Any]):
        self.path = Path(path).expanduser().resolve()
        self.metadata = dict(metadata)
        self._stream: Any | None = None
        self._locked = False

    @property
    def locked(self) -> bool:
        return self._locked

    def acquire(self) -> None:
        if self._locked:
            raise AppLauncherLeaseError("AppLauncher lease is already held by this owner")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            # msvcrt.locking requires a non-empty byte range.  The byte is only
            # a lock anchor; the human-readable metadata follows after lock.
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b" ")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._stream = stream
            self._locked = True
            payload = {
                "schema": "org.rivermark.app-launcher-lease.v1",
                "pid": os.getpid(),
                "acquired_wall_time_ns": time.time_ns(),
                **self.metadata,
            }
            encoded = _metadata_bytes(payload)
            stream.seek(0)
            stream.truncate()
            stream.write(encoded)
            stream.flush()
        except (OSError, BlockingIOError) as exc:
            stream.close()
            raise AppLauncherLeaseError(
                f"another process owns the AppLauncher lease: {self.path}"
            ) from exc
        except BaseException:
            stream.close()
            raise

    def release(self) -> None:
        stream = self._stream
        self._stream = None
        was_locked = self._locked
        self._locked = False
        if stream is None:
            return
        try:
            if was_locked:
                if os.name == "nt":
                    import msvcrt

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "AppLauncherLease":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def repository_app_launcher_lease(repository_root: Path, *, metadata: Mapping[str, Any]) -> AppLauncherLease:
    """Return the single lease shared by capture and smoke output directories."""

    return AppLauncherLease(
        Path(repository_root).expanduser().resolve() / ".isaac_app_launcher.lock",
        metadata=metadata,
    )
