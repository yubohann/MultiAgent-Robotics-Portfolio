from __future__ import annotations

import json
import math

import pytest

from aerocity_method.contracts.io import (
    canonical_json_bytes,
    finite_number,
    payload_label,
    read_json_object,
    require_identifier,
    validate_finite_diagnostics,
    write_json_atomic,
)


def test_canonical_json_ignores_mapping_order():
    assert canonical_json_bytes({"b": 2, "a": 1}) == canonical_json_bytes({"a": 1, "b": 2})


def test_canonical_json_has_utf8_and_trailing_newline():
    encoded = canonical_json_bytes({"名称": "无人机"})
    assert encoded.endswith(b"\n")
    assert "无人机" in encoded.decode("utf-8")


def test_set_normalization_is_deterministic():
    assert canonical_json_bytes({"x": {3, 1, 2}}) == canonical_json_bytes({"x": {2, 3, 1}})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(value):
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": value})


def test_boolean_is_not_accepted_as_numeric():
    with pytest.raises(ValueError):
        finite_number(True, "value")


def test_identifier_rejects_empty_and_control_characters():
    with pytest.raises(ValueError):
        require_identifier("", "id")
    with pytest.raises(ValueError):
        require_identifier("bad\nvalue", "id")


def test_payload_label_is_explicit_and_deterministic():
    payload = {"scene_id": "scene0", "episode_id": "episode0", "value": 1}
    assert payload_label(payload, prefix="episode") == payload_label(
        dict(reversed(tuple(payload.items()))), prefix="episode"
    )
    assert payload_label(payload, prefix="episode").startswith("episode:")
    assert "scene_id=scene0" in payload_label(payload, prefix="episode")


def test_payload_label_varies_with_identity_fields():
    assert payload_label({"scene_id": "a"}, prefix="x") != payload_label(
        {"scene_id": "b"}, prefix="x"
    )


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
        canonical_json_bytes(object())
