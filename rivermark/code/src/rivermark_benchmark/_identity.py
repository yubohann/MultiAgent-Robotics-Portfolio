"""Deterministic short identities for cache keys, artifact names, and dedup."""

from __future__ import annotations

import zlib


class IdentityAccumulator:
    """Accumulate byte chunks into one short identity string."""

    def __init__(self, data: bytes | None = None) -> None:
        self._checksum = 0
        self._size = 0
        if data is not None:
            self.update(data)

    def update(self, data: bytes) -> None:
        chunk = bytes(data)
        self._checksum = zlib.crc32(chunk, self._checksum)
        self._size += len(chunk)

    def hexdigest(self) -> str:
        return f"{self._checksum & 0xFFFFFFFF:08x}{self._size & 0xFFFFFFFF:08x}"

    def digest(self) -> bytes:
        return bytes.fromhex(self.hexdigest())


def identity_file(path) -> str:
    accumulator = IdentityAccumulator()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            accumulator.update(block)
    return accumulator.hexdigest()
