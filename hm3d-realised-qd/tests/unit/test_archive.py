from __future__ import annotations

from dataclasses import replace

import pytest

from aerocity_method.archives.emitters import (
    ArchiveAwareEmitter,
    DeterministicEmitter,
    RandomEmitter,
)
from aerocity_method.archives.qd import ArchiveSpec, DescriptorAxis, Elite, QDArchive
from aerocity_method.contracts.io import canonical_sha256


@pytest.fixture
def archive():
    return QDArchive(
        ArchiveSpec(
            (
                DescriptorAxis("x", 0.0, 1.0, 2),
                DescriptorAxis("z", 0.0, 1.0, 2),
            )
        )
    )


def elite(manifest, descriptor, quality=1.0, cost=0.2, behavior=None, feasible=True):
    return Elite(
        candidate_id=manifest.candidate_id,
        manifest_hash=manifest.manifest_hash,
        behavior_hash=behavior or canonical_sha256({"behavior": manifest.candidate_id}),
        realised_descriptor=descriptor,
        quality=quality,
        cost=cost,
        feasible=feasible,
        source="test",
    )


def test_axis_boundaries_and_upper_inclusive():
    axis = DescriptorAxis("x", 0.0, 1.0, 4)
    assert axis.cell(0.0) == 0
    assert axis.cell(0.25) == 1
    assert axis.cell(1.0) == 3
    with pytest.raises(ValueError):
        axis.cell(1.001)


def test_archive_requires_two_axes():
    with pytest.raises(ValueError):
        ArchiveSpec((DescriptorAxis("x", 0, 1, 2),))


def test_infeasible_elite_is_rejected(archive, manifests):
    decision = archive.add_or_update(elite(manifests[0], (0.1, 0.1), feasible=False))
    assert not decision.admitted
    assert len(archive) == 0


def test_quality_replacement_and_cost_tie_break(archive, manifests):
    first = elite(manifests[0], (0.1, 0.1), quality=1.0, cost=0.5)
    second = elite(manifests[1], (0.2, 0.2), quality=1.0, cost=0.2)
    archive.add_or_update(first)
    decision = archive.add_or_update(second)
    assert decision.admitted
    assert archive.best().manifest_hash == second.manifest_hash


def test_manifest_hash_is_final_deterministic_tie_break(archive, manifests):
    candidates = [
        elite(manifest, (0.1, 0.1), behavior=canonical_sha256({"b": i}))
        for i, manifest in enumerate(manifests[:2])
    ]
    for row in sorted(candidates, key=lambda item: item.manifest_hash, reverse=True):
        archive.add_or_update(row)
    assert archive.best().manifest_hash == min(item.manifest_hash for item in candidates)


def test_duplicate_behavior_cannot_occupy_two_candidates(archive, manifests):
    behavior = canonical_sha256({"same": True})
    archive.add_or_update(elite(manifests[0], (0.1, 0.1), behavior=behavior))
    decision = archive.add_or_update(elite(manifests[1], (0.9, 0.9), behavior=behavior))
    assert decision.reason == "DUPLICATE_BEHAVIOR"
    assert len(archive) == 1


def test_same_manifest_is_reevaluated_with_welford_statistics(archive, manifests):
    row = elite(manifests[0], (0.1, 0.1), quality=1.0)
    archive.add_or_update(row)
    decision = archive.add_or_update(replace(row, quality=3.0, cost=0.4))
    stored = archive.get((0, 0))
    assert decision.reason == "REEVALUATED"
    assert stored.evaluation_count == 2
    assert stored.corrected_quality == 2.0
    assert stored.quality_variance == 2.0


def test_realised_descriptor_migration_removes_old_cell(archive, manifests):
    row = elite(manifests[0], (0.1, 0.1))
    archive.add_or_update(row)
    archive.add_or_update(replace(row, realised_descriptor=(0.9, 0.9), quality=2.0))
    assert archive.get((0, 0)) is None
    assert archive.get((1, 1)).manifest_hash == row.manifest_hash


def test_archive_checkpoint_round_trip(archive, manifests):
    archive.add_or_update(elite(manifests[0], (0.1, 0.1)))
    restored = QDArchive.from_state_dict(archive.state_dict())
    assert restored.digest == archive.digest
    assert restored.revision == archive.revision


def test_checkpoint_rejects_cell_descriptor_mismatch(archive, manifests):
    archive.add_or_update(elite(manifests[0], (0.1, 0.1)))
    state = archive.state_dict()
    state["cells"][0]["cell"] = (1, 1)
    with pytest.raises(ValueError):
        QDArchive.from_state_dict(state)


def test_deterministic_emitter_cursor_is_reproducible(manifests):
    left = DeterministicEmitter()
    right = DeterministicEmitter()
    assert [x.manifest.manifest_hash for x in left.ask(manifests, 5)] == [
        x.manifest.manifest_hash for x in right.ask(tuple(reversed(manifests)), 5)
    ]


def test_random_emitter_state_restores(manifests):
    emitter = RandomEmitter(3)
    state = emitter.state()
    first = emitter.ask(manifests, 4)
    emitter.restore(state)
    second = emitter.ask(manifests, 4)
    assert [row.manifest.manifest_hash for row in first] == [
        row.manifest.manifest_hash for row in second
    ]


def test_archive_aware_emitter_prefers_empty_cell(archive, manifests):
    archive.add_or_update(elite(manifests[0], manifests[0].planned_descriptor))
    emitted = ArchiveAwareEmitter(archive).ask(manifests, 1)[0]
    assert emitted.manifest.manifest_hash != manifests[0].manifest_hash


def test_archive_metrics_have_five_evidence_groups_core(archive, manifests):
    archive.add_or_update(elite(manifests[0], (0.1, 0.1)))
    metrics = archive.metrics()
    expected = {"coverage", "qd_score", "max_quality", "median_quality", "effective_cells"}
    assert expected <= metrics.keys()
