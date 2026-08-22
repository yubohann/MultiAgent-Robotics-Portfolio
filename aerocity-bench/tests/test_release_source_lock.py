from __future__ import annotations

import argparse

import pytest

from aerocity_bench import cli


class _Config:
    raw = {"release_kind": "OFFICIAL"}


def _args(*, allow_uncommitted: bool) -> argparse.Namespace:
    return argparse.Namespace(
        release="ordinary.json",
        splits=None,
        source_commit=None,
        allow_uncommitted_development=allow_uncommitted,
        asset_root="assets",
        output="output",
    )


def test_official_cli_build_refuses_dirty_worktree(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_is_ordinary_config", lambda _: True)
    monkeypatch.setattr(cli, "load_ordinary_config", lambda _: _Config())
    monkeypatch.setattr(cli, "_git_commit", lambda _: "a" * 40)
    monkeypatch.setattr(cli, "_git_worktree_clean", lambda _: False)

    with pytest.raises(ValueError, match="clean Git worktree"):
        cli._build(_args(allow_uncommitted=False))


def test_dirty_development_build_is_explicitly_uncommitted(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "_is_ordinary_config", lambda _: True)
    monkeypatch.setattr(cli, "load_ordinary_config", lambda _: _Config())
    monkeypatch.setattr(cli, "_git_commit", lambda _: "a" * 40)
    monkeypatch.setattr(cli, "_git_worktree_clean", lambda _: False)
    monkeypatch.setattr(
        cli,
        "build_ordinary_release",
        lambda *_, **kwargs: captured.update(kwargs) or {"status": "PASS"},
    )

    assert cli._build(_args(allow_uncommitted=True))["status"] == "PASS"
    assert captured["source_commit"] == "UNCOMMITTED-DEVELOPMENT"
    assert captured["allow_uncommitted_development"] is True


def test_clean_official_build_keeps_frozen_commit(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "_is_ordinary_config", lambda _: True)
    monkeypatch.setattr(cli, "load_ordinary_config", lambda _: _Config())
    monkeypatch.setattr(cli, "_git_commit", lambda _: "b" * 40)
    monkeypatch.setattr(cli, "_git_worktree_clean", lambda _: True)
    monkeypatch.setattr(
        cli,
        "build_ordinary_release",
        lambda *_, **kwargs: captured.update(kwargs) or {"status": "PASS"},
    )

    cli._build(_args(allow_uncommitted=False))
    assert captured["source_commit"] == "b" * 40
