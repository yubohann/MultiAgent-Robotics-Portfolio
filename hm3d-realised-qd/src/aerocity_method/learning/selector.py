"""Unified selection over archive elites and newly emitted candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from aerocity_method.archives.qd import QDArchive
from aerocity_method.contracts.models import CandidateFragmentManifest


@dataclass(frozen=True, slots=True)
class SelectionOption:
    manifest: CandidateFragmentManifest
    from_archive: bool
    archive_quality: float | None


def build_selection_set(
    candidates: Sequence[CandidateFragmentManifest], archive: QDArchive
) -> tuple[SelectionOption, ...]:
    catalog = {manifest.manifest_hash: manifest for manifest in candidates}
    options: dict[str, SelectionOption] = {
        digest: SelectionOption(manifest, False, None) for digest, manifest in catalog.items()
    }
    for _, elite in archive.items():
        manifest = catalog.get(elite.manifest_hash)
        if manifest is None:
            raise ValueError("archive elite is missing from the executable candidate catalog")
        options[elite.manifest_hash] = SelectionOption(manifest, True, elite.corrected_quality)
    resolved = tuple(sorted(options.values(), key=lambda option: option.manifest.manifest_hash))
    if not resolved or not any(option.manifest.feasible for option in resolved):
        raise ValueError("selection set requires a feasible candidate")
    return resolved


class ProbabilityModel(Protocol):
    def action_probabilities(
        self,
        context: Sequence[float],
        candidates: Sequence[Sequence[float]],
        legal_mask: Sequence[bool],
        preference: Sequence[float] = (),
    ) -> tuple[float, ...]: ...


class UnifiedSelector:
    def __init__(self, model: ProbabilityModel | None = None) -> None:
        self.model = model

    @staticmethod
    def _feature(option: SelectionOption) -> tuple[float, ...]:
        manifest = option.manifest
        return (
            *manifest.planned_descriptor,
            manifest.quality_hint,
            manifest.cost_hint,
            float(option.from_archive),
            0.0 if option.archive_quality is None else option.archive_quality,
        )

    def probabilities(
        self,
        options: Sequence[SelectionOption],
        context: Sequence[float] = (),
        preference: Sequence[float] = (),
    ) -> tuple[float, ...]:
        rows = tuple(options)
        legal = tuple(option.manifest.feasible for option in rows)
        if self.model is not None:
            probabilities = tuple(
                self.model.action_probabilities(
                    context,
                    [self._feature(option) for option in rows],
                    legal,
                    preference,
                )
            )
            if len(probabilities) != len(rows):
                raise ValueError("probability model returned the wrong action count")
            if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
                raise ValueError("probability model returned invalid probabilities")
            if abs(sum(probabilities) - 1.0) > 1e-6:
                raise ValueError("probability model probabilities must sum to one")
            illegal_mass = any(
                probability != 0.0
                for probability, allowed in zip(probabilities, legal, strict=True)
                if not allowed
            )
            if illegal_mass:
                raise ValueError("probability model assigned mass to an illegal candidate")
            return probabilities
        scores = [
            option.manifest.quality_hint
            - option.manifest.cost_hint
            + (0.0 if option.archive_quality is None else option.archive_quality)
            if option.manifest.feasible
            else float("-inf")
            for option in rows
        ]
        best = max(score for score in scores if score != float("-inf"))
        winners = [index for index, score in enumerate(scores) if score == best]
        probabilities = [0.0 for _ in rows]
        for index in winners:
            probabilities[index] = 1.0 / len(winners)
        return tuple(probabilities)

    def select(
        self,
        options: Sequence[SelectionOption],
        context: Sequence[float] = (),
        preference: Sequence[float] = (),
    ) -> int:
        probabilities = self.probabilities(options, context, preference)
        return max(range(len(probabilities)), key=lambda index: probabilities[index])
