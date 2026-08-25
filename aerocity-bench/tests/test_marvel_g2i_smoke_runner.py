from __future__ import annotations

import importlib.util
from pathlib import Path


def test_marvel_smoke_runner_locks_the_real_upstream_revision() -> None:
    path = Path("tools/run_marvel_g2i_l0_smoke.py")
    spec = importlib.util.spec_from_file_location("run_marvel_g2i_l0_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.UPSTREAM_URL == "https://github.com/marmotlab/MARVEL.git"
    assert module.UPSTREAM_COMMIT == "318c2a6016d0f2d1dbb0dd08b3f8f8224b361e4c"
    assert module.UPSTREAM_LICENSE == "MIT"
    assert module.MAXIMUM_RESET_BYTES == 2_000_000
    assert "arbitrate_public_fleet_actions" in path.read_text(encoding="utf-8")


def test_smoke_summary_keeps_return_closure_separate_from_execution_safety() -> None:
    path = Path("tools/run_marvel_g2i_l0_smoke.py")
    spec = importlib.util.spec_from_file_location("run_marvel_g2i_l0_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module._summary(
        {
            "task_time_s": 2.4,
            "execution_receipts": [{}, {}],
            "confirmations": [{"confirmation_id": "opaque"}],
            "failures": [],
            "budget_ledger": {
                "collisions": 0,
                "out_of_bounds_actions": 0,
                "deadline_misses": 0,
            },
            "returned_home": {"uav-00": True, "uav-01": False},
            "formal_score_eligible": False,
        }
    )

    assert summary["confirmation_count"] == 1
    assert summary["all_returned_home"] is False
