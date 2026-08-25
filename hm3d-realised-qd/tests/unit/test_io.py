from __future__ import annotations

import json
import math

import pytest

from aerocity_method.contracts.io import (
    canonical_json_bytes,
    canonical_sha256,
    finite_number,
    read_json_object,
    require_identifier,
    require_sha256,
    validate_finite_diagnostics,
    write_json_atomic,
)


def test_canonical_hash_ignores_mapping_order():
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_canonical_hash_has_utf8_and_trailing_newline():
    encoded = canonical_json_bytes({"名称": "无人机"})
    assert encoded.endswith(b"\n")
    assert "无人机" in encoded.decode("utf-8")


def test_set_normalization_is_deterministic():
    assert canonical_sha256({"x": {3, 1, 2}}) == canonical_sha256({"x": {2, 3, 1}})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(value):
    with pytest.raises(ValueError):
        canonical_sha256({"bad": value})


def test_boolean_is_not_accepted_as_numeric():
    with pytest.raises(ValueError):
        finite_number(True, "value")


def test_identifier_rejects_empty_and_control_characters():
    with pytest.raises(ValueError):
        require_identifier("", "id")
    with pytest.raises(ValueError):
        require_identifier("bad\nvalue", "id")


def test_sha256_contract_is_lowercase_and_exact_length():
    digest = canonical_sha256({"ok": 1})
    assert require_sha256(digest, "digest") == digest
    with pytest.raises(ValueError):
        require_sha256(digest.upper(), "digest")


def test_finite_diagnostics_normalize_numbers():
    assert validate_finite_diagnostics({"loss": 1, "gain": 0.5}) == {
        "loss": 1.0,
        "gain": 0.5,
    }
    with pytest.raises(ValueError):
        validate_finite_diagnostics({"bad": "1"})


def test_atomic_json_write_and_read_round_trip(tmp_path):
    destination = tmp_path / "nested" / "artifact.json"
    write_json_atomic(destination, {"名称": "测试", "value": 1.0})
    assert read_json_object(destination) == {"名称": "测试", "value": 1.0}
    assert not list(destination.parent.glob("*.tmp"))


def test_read_json_requires_object(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError):
        read_json_object(path)


def test_unsupported_object_is_rejected():
    with pytest.raises(ValueError):
        canonical_sha256(object())
