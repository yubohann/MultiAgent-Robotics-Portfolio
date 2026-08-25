"""Small, auditable M0 emitters."""

from __future__ import annotations

import random
from dataclasses import dataclass

from aerocity_method.archives.qd import QDArchive
from aerocity_method.contracts.io import require_identifier
from aerocity_method.contracts.models import CandidateFragmentManifest


@dataclass(frozen=True, slots=True)
class Emission:
    manifest: CandidateFragmentManifest
    emitter: str

    def __post_init__(self) -> None:
        require_identifier(self.emitter, "emitter")


def _legal_pool(
    pool: tuple[CandidateFragmentManifest, ...] | list[CandidateFragmentManifest],
) -> tuple[CandidateFragmentManifest, ...]:
    legal = tuple(manifest for manifest in pool if manifest.feasible)
    if not legal:
        raise ValueError("emitter requires at least one feasible candidate")
    return legal


class DeterministicEmitter:
    name = "deterministic"

    def __init__(self) -> None:
        self.cursor = 0

    def ask(
        self,
        pool: tuple[CandidateFragmentManifest, ...] | list[CandidateFragmentManifest],
        count: int = 1,
    ) -> tuple[Emission, ...]:
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        legal = tuple(sorted(_legal_pool(pool), key=lambda item: item.manifest_hash))
        emitted = tuple(
            Emission(legal[(self.cursor + index) % len(legal)], self.name) for index in range(count)
        )
        self.cursor = (self.cursor + count) % len(legal)
        return emitted


class RandomEmitter:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def ask(
        self,
        pool: tuple[CandidateFragmentManifest, ...] | list[CandidateFragmentManifest],
        count: int = 1,
    ) -> tuple[Emission, ...]:
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        legal = _legal_pool(pool)
        return tuple(Emission(self._rng.choice(legal), self.name) for _ in range(count))

    def state(self) -> object:
        return self._rng.getstate()

    def restore(self, state: object) -> None:
        self._rng.setstate(state)


class ArchiveAwareEmitter:
    name = "archive_aware"

    def __init__(self, archive: QDArchive) -> None:
        self.archive = archive

    def ask(
        self,
        pool: tuple[CandidateFragmentManifest, ...] | list[CandidateFragmentManifest],
        count: int = 1,
    ) -> tuple[Emission, ...]:
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        scored: list[tuple[int, float, float, str, CandidateFragmentManifest]] = []
        for manifest in _legal_pool(pool):
            cell = self.archive.spec.cell(manifest.planned_descriptor)
            novelty = 1 if self.archive.get(cell) is None else 0
            scored.append(
                (
                    -novelty,
                    -manifest.quality_hint,
                    manifest.cost_hint,
                    manifest.manifest_hash,
                    manifest,
                )
            )
        scored.sort(key=lambda row: row[:-1])
        ordered = [row[-1] for row in scored]
        return tuple(Emission(ordered[index % len(ordered)], self.name) for index in range(count))
