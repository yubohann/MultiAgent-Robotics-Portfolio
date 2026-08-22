"""A deterministic outcome-grounded QD archive."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from aerocity_method.contracts.io import (
    canonical_sha256,
    finite_number,
    require_identifier,
    require_sha256,
)
from aerocity_method.contracts.models import ABI_VERSION


@dataclass(frozen=True, slots=True)
class DescriptorAxis:
    name: str
    lower: float
    upper: float
    bins: int

    def __post_init__(self) -> None:
        require_identifier(self.name, "descriptor axis name")
        lower = finite_number(self.lower, f"{self.name}.lower")
        upper = finite_number(self.upper, f"{self.name}.upper")
        if upper <= lower:
            raise ValueError("descriptor axis upper must exceed lower")
        if not isinstance(self.bins, int) or isinstance(self.bins, bool) or self.bins < 2:
            raise ValueError("descriptor axis bins must be an integer >= 2")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def cell(self, value: float) -> int:
        resolved = finite_number(value, self.name)
        if resolved < self.lower or resolved > self.upper:
            raise ValueError(f"descriptor {self.name}={resolved} is outside frozen range")
        if resolved == self.upper:
            return self.bins - 1
        width = (self.upper - self.lower) / self.bins
        return int((resolved - self.lower) / width)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "lower": self.lower, "upper": self.upper, "bins": self.bins}


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    axes: tuple[DescriptorAxis, ...]
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ABI_VERSION:
            raise ValueError("unsupported archive schema version")
        axes = tuple(self.axes)
        if len(axes) < 2:
            raise ValueError("QD archive requires at least two descriptor axes")
        names = [axis.name for axis in axes]
        if len(names) != len(set(names)):
            raise ValueError("descriptor axis names must be unique")
        object.__setattr__(self, "axes", axes)

    def cell(self, descriptor: tuple[float, ...]) -> tuple[int, ...]:
        values = tuple(descriptor)
        if len(values) != len(self.axes):
            raise ValueError("descriptor dimensionality does not match archive spec")
        return tuple(axis.cell(value) for axis, value in zip(self.axes, values, strict=True))

    @property
    def capacity(self) -> int:
        result = 1
        for axis in self.axes:
            result *= axis.bins
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "axes": [axis.to_dict() for axis in self.axes],
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class Elite:
    candidate_id: str
    manifest_hash: str
    behavior_hash: str
    realised_descriptor: tuple[float, ...]
    quality: float
    cost: float
    feasible: bool
    source: str
    evaluation_count: int = 1
    quality_mean: float | None = None
    quality_m2: float = 0.0
    schema_version: str = ABI_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ABI_VERSION:
            raise ValueError("unsupported elite schema version")
        require_identifier(self.candidate_id, "candidate_id")
        require_identifier(self.source, "source")
        require_sha256(self.manifest_hash, "manifest_hash")
        require_sha256(self.behavior_hash, "behavior_hash")
        descriptor = tuple(
            finite_number(value, "realised_descriptor") for value in self.realised_descriptor
        )
        object.__setattr__(self, "realised_descriptor", descriptor)
        quality = finite_number(self.quality, "quality")
        cost = finite_number(self.cost, "cost")
        if cost < 0.0:
            raise ValueError("elite cost must be non-negative")
        if not isinstance(self.evaluation_count, int) or self.evaluation_count < 1:
            raise ValueError("evaluation_count must be a positive integer")
        mean = (
            quality
            if self.quality_mean is None
            else finite_number(self.quality_mean, "quality_mean")
        )
        m2 = finite_number(self.quality_m2, "quality_m2")
        if m2 < 0.0:
            raise ValueError("quality_m2 must be non-negative")
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "quality_mean", mean)
        object.__setattr__(self, "quality_m2", m2)

    @property
    def corrected_quality(self) -> float:
        assert self.quality_mean is not None
        return self.quality_mean

    @property
    def quality_variance(self) -> float:
        if self.evaluation_count < 2:
            return 0.0
        return self.quality_m2 / (self.evaluation_count - 1)

    def reevaluated(self, observed_quality: float, observed_cost: float) -> Elite:
        value = finite_number(observed_quality, "observed_quality")
        cost = finite_number(observed_cost, "observed_cost")
        if cost < 0.0:
            raise ValueError("observed_cost must be non-negative")
        count = self.evaluation_count + 1
        assert self.quality_mean is not None
        delta = value - self.quality_mean
        mean = self.quality_mean + delta / count
        m2 = self.quality_m2 + delta * (value - mean)
        return replace(
            self,
            quality=value,
            cost=(self.cost * self.evaluation_count + cost) / count,
            evaluation_count=count,
            quality_mean=mean,
            quality_m2=m2,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "manifest_hash": self.manifest_hash,
            "behavior_hash": self.behavior_hash,
            "realised_descriptor": self.realised_descriptor,
            "quality": self.quality,
            "cost": self.cost,
            "feasible": self.feasible,
            "source": self.source,
            "evaluation_count": self.evaluation_count,
            "quality_mean": self.quality_mean,
            "quality_m2": self.quality_m2,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Elite:
        return cls(
            candidate_id=payload["candidate_id"],
            manifest_hash=payload["manifest_hash"],
            behavior_hash=payload["behavior_hash"],
            realised_descriptor=tuple(payload["realised_descriptor"]),
            quality=payload["quality"],
            cost=payload["cost"],
            feasible=payload["feasible"],
            source=payload["source"],
            evaluation_count=payload.get("evaluation_count", 1),
            quality_mean=payload.get("quality_mean"),
            quality_m2=payload.get("quality_m2", 0.0),
            schema_version=payload.get("schema_version", ABI_VERSION),
        )


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    cell: tuple[int, ...] | None
    replaced_manifest_hash: str | None
    revision: int


def _is_better(candidate: Elite, incumbent: Elite) -> bool:
    candidate_key = (candidate.corrected_quality, -candidate.cost)
    incumbent_key = (incumbent.corrected_quality, -incumbent.cost)
    if candidate_key != incumbent_key:
        return candidate_key > incumbent_key
    return candidate.manifest_hash < incumbent.manifest_hash


class QDArchive:
    """Deterministic archive keyed only by realised descriptors."""

    def __init__(self, spec: ArchiveSpec) -> None:
        self.spec = spec
        self._cells: dict[tuple[int, ...], Elite] = {}
        self._behavior_cells: dict[str, tuple[int, ...]] = {}
        self._manifest_cells: dict[str, tuple[int, ...]] = {}
        self.revision = 0

    def __len__(self) -> int:
        return len(self._cells)

    def items(self) -> tuple[tuple[tuple[int, ...], Elite], ...]:
        return tuple(sorted(self._cells.items()))

    def get(self, cell: tuple[int, ...]) -> Elite | None:
        return self._cells.get(tuple(cell))

    def _remove_behavior(self, behavior_hash: str) -> Elite | None:
        old_cell = self._behavior_cells.pop(behavior_hash, None)
        if old_cell is None:
            return None
        elite = self._cells.pop(old_cell, None)
        if elite is not None:
            self._manifest_cells.pop(elite.manifest_hash, None)
        return elite

    def add_or_update(self, elite: Elite) -> AdmissionDecision:
        if not elite.feasible:
            return AdmissionDecision(False, "INFEASIBLE", None, None, self.revision)
        cell = self.spec.cell(elite.realised_descriptor)
        old_behavior_cell = self._behavior_cells.get(elite.behavior_hash)
        if old_behavior_cell is not None:
            old_behavior_elite = self._cells[old_behavior_cell]
            if old_behavior_elite.manifest_hash != elite.manifest_hash:
                return AdmissionDecision(
                    False,
                    "DUPLICATE_BEHAVIOR",
                    cell,
                    old_behavior_elite.manifest_hash,
                    self.revision,
                )
        old_manifest_cell = self._manifest_cells.get(elite.manifest_hash)
        if old_manifest_cell == cell:
            old_manifest_elite = self._cells[cell]
            updated = old_manifest_elite.reevaluated(elite.quality, elite.cost)
            updated = replace(
                updated,
                behavior_hash=elite.behavior_hash,
                realised_descriptor=elite.realised_descriptor,
                source=elite.source,
            )
            self._behavior_cells.pop(old_manifest_elite.behavior_hash, None)
            self._behavior_cells[updated.behavior_hash] = cell
            self._cells[cell] = updated
            self.revision += 1
            return AdmissionDecision(True, "REEVALUATED", cell, None, self.revision)
        incumbent = self._cells.get(cell)
        if incumbent is not None and not _is_better(elite, incumbent):
            return AdmissionDecision(
                False, "NOT_BETTER", cell, incumbent.manifest_hash, self.revision
            )
        replaced_hash = None
        if incumbent is not None:
            replaced_hash = incumbent.manifest_hash
            self._behavior_cells.pop(incumbent.behavior_hash, None)
            self._manifest_cells.pop(incumbent.manifest_hash, None)
        if old_manifest_cell is not None:
            old_manifest_elite = self._cells.pop(old_manifest_cell)
            self._behavior_cells.pop(old_manifest_elite.behavior_hash, None)
        self._cells[cell] = elite
        self._behavior_cells[elite.behavior_hash] = cell
        self._manifest_cells[elite.manifest_hash] = cell
        self.revision += 1
        return AdmissionDecision(True, "ADMITTED", cell, replaced_hash, self.revision)

    def best(self) -> Elite | None:
        if not self._cells:
            return None
        return sorted(
            self._cells.values(),
            key=lambda elite: (-elite.corrected_quality, elite.cost, elite.manifest_hash),
        )[0]

    def metrics(self) -> dict[str, float]:
        if not self._cells:
            return {
                "coverage": 0.0,
                "effective_cells": 0.0,
                "qd_score": 0.0,
                "max_quality": 0.0,
                "median_quality": 0.0,
                "feasible_elite_fraction": 0.0,
            }
        qualities = sorted(elite.corrected_quality for elite in self._cells.values())
        middle = len(qualities) // 2
        median = (
            qualities[middle]
            if len(qualities) % 2
            else (qualities[middle - 1] + qualities[middle]) / 2.0
        )
        return {
            "coverage": len(self._cells) / self.spec.capacity,
            "effective_cells": float(len(self._cells)),
            "qd_score": sum(qualities),
            "max_quality": max(qualities),
            "median_quality": median,
            "feasible_elite_fraction": 1.0,
        }

    def state_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": ABI_VERSION,
            "spec": self.spec.to_dict(),
            "revision": self.revision,
            "cells": [{"cell": cell, "elite": elite.to_dict()} for cell, elite in self.items()],
        }
        payload["checkpoint_hash"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> QDArchive:
        if state.get("schema_version") != ABI_VERSION:
            raise ValueError("archive checkpoint schema version mismatch")
        supplied_hash = state.get("checkpoint_hash")
        unsigned = {key: value for key, value in state.items() if key != "checkpoint_hash"}
        if canonical_sha256(unsigned) != supplied_hash:
            raise ValueError("archive checkpoint content hash mismatch")
        spec_payload = state["spec"]
        spec = ArchiveSpec(
            tuple(
                DescriptorAxis(
                    name=axis["name"],
                    lower=axis["lower"],
                    upper=axis["upper"],
                    bins=axis["bins"],
                )
                for axis in spec_payload["axes"]
            ),
            schema_version=spec_payload.get("schema_version", ABI_VERSION),
        )
        archive = cls(spec)
        occupied_cells: set[tuple[int, ...]] = set()
        manifest_hashes: set[str] = set()
        for row in state.get("cells", []):
            cell = tuple(row["cell"])
            elite = Elite.from_dict(row["elite"])
            if archive.spec.cell(elite.realised_descriptor) != cell:
                raise ValueError("checkpoint cell does not match realised descriptor")
            if cell in occupied_cells:
                raise ValueError("checkpoint contains duplicate archive cell")
            if elite.manifest_hash in manifest_hashes:
                raise ValueError("checkpoint contains one manifest in multiple cells")
            archive._cells[cell] = elite
            if elite.behavior_hash in archive._behavior_cells:
                raise ValueError("checkpoint contains duplicate behavior hash")
            archive._behavior_cells[elite.behavior_hash] = cell
            archive._manifest_cells[elite.manifest_hash] = cell
            occupied_cells.add(cell)
            manifest_hashes.add(elite.manifest_hash)
        revision = state.get("revision", len(archive._cells))
        if not isinstance(revision, int) or revision < len(archive._cells):
            raise ValueError("invalid archive revision")
        archive.revision = revision
        return archive

    @property
    def digest(self) -> str:
        return canonical_sha256(self.state_dict())
