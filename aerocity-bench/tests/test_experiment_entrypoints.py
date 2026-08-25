from __future__ import annotations

from pathlib import Path

from aerocity_bench.experiment_entrypoints import audit_experiment_entrypoints


def test_every_public_experiment_entrypoint_has_a_common_boundary_guard() -> None:
    repository = Path(__file__).parents[1]
    report = audit_experiment_entrypoints(repository)

    assert report["status"] == "PASS"
    assert report["entrypoint_count"] >= 13
    assert all(record["status"] == "PASS" for record in report["records"])
