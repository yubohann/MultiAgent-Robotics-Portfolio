"""Regression coverage for P03--P05 immutable-artifact assembly."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "assemble_hm3d_p03_p05_admission.py"


def _load_assembler():
    spec = importlib.util.spec_from_file_location("assemble_hm3d_p03_p05_admission", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_digest_is_persistable_sha256_text():
    assembler = _load_assembler()

    digest = assembler._command_sha256(["python", "assemble.py", "--p03-output", "P03.json"])

    assert isinstance(digest, str)
    assert len(digest) == 64
    assert int(digest, 16) >= 0
