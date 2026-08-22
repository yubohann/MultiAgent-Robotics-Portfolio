from __future__ import annotations

from pathlib import Path

import pytest

import aerocity_bench.cf2x_contract as cf2x
from aerocity_bench.canonical import file_hash
from aerocity_bench.cf2x_native import (
    cf2x_allocation_matrix,
    cf2x_hover_rps,
    cf2x_max_thrust_per_rotor_n,
    cf2x_thrust_constant_n_per_rps2,
    validate_cf2x_runtime_masses_kg,
)
from aerocity_bench.quadrotor_dynamics import project_asset_spec


def _write_fixture_asset(root: Path) -> Path:
    path = root / "cf2x.usd"
    schema = root / "configuration" / "cf2x_robot_schema.usd"
    schema.parent.mkdir(parents=True)
    path.write_bytes(b"cf2x-root-fixture")
    schema.write_bytes(b"cf2x-schema-fixture")
    return path


def test_verified_local_cf2x_requires_exact_root_and_relative_schema_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_fixture_asset(tmp_path)
    schema = tmp_path / "configuration" / "cf2x_robot_schema.usd"
    monkeypatch.setattr(cf2x, "CF2X_SHA256", file_hash(path))
    monkeypatch.setattr(cf2x, "CF2X_SCHEMA_SHA256", file_hash(schema))
    verified = cf2x.verify_local_cf2x_asset(path)
    assert verified.usd_path == path.resolve()
    assert verified.schema_path == schema.resolve()
    assert verified.fingerprint_payload()["redistribution_status"] == (
        "local_runtime_only_license_clearance_pending"
    )


def test_local_cf2x_rejects_old_asset_nucleus_paths_and_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "5_in_drone" / "cf2x.usd"
    old.parent.mkdir()
    old.write_bytes(b"not-allowed")
    with pytest.raises(ValueError, match="prohibited"):
        cf2x.verify_local_cf2x_asset(old)

    path = _write_fixture_asset(tmp_path / "clean")
    schema = path.parent / "configuration" / "cf2x_robot_schema.usd"
    monkeypatch.setattr(cf2x, "CF2X_SHA256", file_hash(path))
    monkeypatch.setattr(cf2x, "CF2X_SCHEMA_SHA256", file_hash(schema))
    path.write_bytes(b"digest-drift")
    with pytest.raises(ValueError, match="SHA-256"):
        cf2x.verify_local_cf2x_asset(path)


def test_cf2x_native_allocation_is_derived_from_the_static_usd_geometry() -> None:
    spec = project_asset_spec()
    allocation = cf2x_allocation_matrix(spec)
    assert len(allocation) == 6
    assert allocation[2] == [1.0] * 4
    assert allocation[3] == pytest.approx([-0.031, -0.031, 0.031, 0.031])
    assert allocation[4] == pytest.approx([-0.031, 0.031, 0.031, -0.031])
    assert cf2x_max_thrust_per_rotor_n(spec) > spec.mass_kg * spec.gravity_mps2 / 4.0
    assert cf2x_thrust_constant_n_per_rps2(spec) > 0.0
    assert cf2x_hover_rps(spec) > 0.0


def test_cf2x_runtime_mass_requires_native_physx_values_to_match_locked_usd_total() -> None:
    values, total = validate_cf2x_runtime_masses_kg(
        (0.025, 0.0008, 0.0008, 0.0008, 0.0008),
        expected_total_mass_kg=0.0282,
    )
    assert values == pytest.approx((0.025, 0.0008, 0.0008, 0.0008, 0.0008))
    assert total == pytest.approx(0.0282)
    with pytest.raises(ValueError, match="differs"):
        validate_cf2x_runtime_masses_kg(
            (0.025, 0.0008, 0.0008, 0.0008, 0.0010), expected_total_mass_kg=0.0282
        )
