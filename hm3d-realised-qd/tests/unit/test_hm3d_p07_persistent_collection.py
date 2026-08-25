"""Protocol tests for persistent P07 collection planning."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_hm3d_p07_persistent_collection.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("persistent_collection_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_persistent_plan_requires_unique_outputs_and_keeps_isaac_flags_outer(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "run.json"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": module.PLAN_SCHEMA_VERSION,
                "runs": [
                    {
                        "run_id": "run0",
                        "runner_arguments": ["--output", str(output), "--headless"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="shared Isaac application"):
        module._load_plan(plan)


def test_persistent_plan_reads_multiple_worker_runs(tmp_path: Path) -> None:
    module = _module()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": module.PLAN_SCHEMA_VERSION,
                "runs": [
                    {"run_id": "run0", "runner_arguments": ["--output", str(tmp_path / "a")]},
                    {"run_id": "run1", "runner_arguments": ["--output", str(tmp_path / "b")]},
                ],
            }
        ),
        encoding="utf-8",
    )

    runs = module._load_plan(plan)

    assert [run.run_id for run in runs] == ["run0", "run1"]
    assert len({run.output_path for run in runs}) == 2


def test_persistent_plan_rejects_cross_scene_process_by_default(tmp_path: Path) -> None:
    module = _module()
    plan = tmp_path / "cross-scene.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": module.PLAN_SCHEMA_VERSION,
                "runs": [
                    {
                        "run_id": "scene-a",
                        "runner_arguments": [
                            "--scene-id",
                            "scene-a",
                            "--output",
                            str(tmp_path / "a.json"),
                        ],
                    },
                    {
                        "run_id": "scene-b",
                        "runner_arguments": [
                            "--scene-id",
                            "scene-b",
                            "--output",
                            str(tmp_path / "b.json"),
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple scene_id values"):
        module._load_plan(plan)

    runs = module._load_plan(plan, allow_cross_scene_process=True)
    assert [run.run_id for run in runs] == ["scene-a", "scene-b"]


def test_persistent_plan_rejects_retired_fixed_decision_cadence(tmp_path: Path) -> None:
    module = _module()
    plan = tmp_path / "fixed_cadence.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": module.PLAN_SCHEMA_VERSION,
                "runs": [
                    {
                        "run_id": "old-run",
                        "runner_arguments": [
                            "--output",
                            str(tmp_path / "old.json"),
                            "--decision-count",
                            "8",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="event-driven P07 plans"):
        module._load_plan(plan)


def test_persistent_collection_fails_fast_after_a_worker_error() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "if exit_code != 0:" in source
    assert '"aborted_episode_count": len(runs) - len(rows)' in source
    assert '"status": "COLLECTION_RUN_STARTED"' in source


def test_worker_completion_audit_rejects_outcome_backed_terminal_failure() -> None:
    module = _module()

    completed, reasons = module._worker_completion_audit(
        {
            "status": module.P07_EXECUTION_SMOKE_FAILED_STATUS,
            "terminal_outcome": "executed_terminal_safety_failure",
            "execution": {"failed_fragment_count": 1},
        }
    )

    assert completed is False
    assert reasons == [
        "worker_status_not_complete",
        "terminal_outcome_not_budget_exhausted",
        "failed_fragments_present",
    ]


def test_worker_completion_audit_requires_clean_budget_exhaustion() -> None:
    module = _module()

    assert module._worker_completion_audit(
        {
            "status": module.P07_EXECUTION_SMOKE_COMPLETE_STATUS,
            "terminal_outcome": "budget_exhausted",
            "execution": {"failed_fragment_count": 0},
        }
    ) == (True, [])
    assert module._worker_completion_audit(
        {
            "status": module.P07_EXECUTION_SMOKE_COMPLETE_STATUS,
            "terminal_outcome": "executed_terminal_safety_failure",
            "execution": {"failed_fragment_count": 1},
        }
    )[0] is False


def test_completed_collection_uses_a_process_boundary_after_final_manifest() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "if final:" in source
    assert "write_json_atomic(manifest_output, manifest)" in source
    assert source.index("exit_code = main(args, app.app)") < source.index("os._exit(exit_code)")
    assert source.index("app.app.close()") < source.index("os._exit(exit_code)")


def test_incremental_manifest_is_resume_aware() -> None:
    module = _module()
    run = module.CollectionRun("run0", ("--output", "unused"), Path("unused"))

    manifest = module._build_manifest(
        plan_sha256="a" * 64,
        runs=(run,),
        rows=[],
        failed=0,
        started=module.time.perf_counter(),
        final=False,
        allow_cross_scene_process=False,
    )

    assert manifest["status"] == "COLLECTION_IN_PROGRESS"
    assert manifest["aborted_episode_count"] == 1
    assert manifest["process_boundary_policy"] == "one_scene_per_isaac_process"
    assert manifest["cross_scene_process_allowed"] is False
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument(\n        "--resume"' in source
    assert 'summary["resumed_existing"] = True' in source
    assert "worker output already exists; inspect it or pass --resume" in source


def test_final_manifest_is_fail_closed_for_missing_worker_rows() -> None:
    module = _module()
    run = module.CollectionRun("run0", ("--output", "unused"), Path("unused"))

    manifest = module._build_manifest(
        plan_sha256="a" * 64,
        runs=(run,),
        rows=[],
        failed=0,
        started=module.time.perf_counter(),
        final=True,
        allow_cross_scene_process=False,
    )

    assert manifest["status"] == module.COLLECTION_HAS_FAILURES_STATUS
    assert manifest["clean_worker_count"] == 0
    assert manifest["completion_audit"]["all_planned_runs_present"] is False


def test_final_manifest_requires_every_worker_to_be_clean() -> None:
    module = _module()
    run = module.CollectionRun("run0", ("--output", "unused"), Path("unused"))

    manifest = module._build_manifest(
        plan_sha256="a" * 64,
        runs=(run,),
        rows=[{"completed": False}],
        failed=1,
        started=module.time.perf_counter(),
        final=True,
        allow_cross_scene_process=False,
    )

    assert manifest["status"] == module.COLLECTION_HAS_FAILURES_STATUS
    assert manifest["completion_audit"]["all_workers_clean"] is False


def test_final_manifest_is_complete_only_for_a_clean_full_plan() -> None:
    module = _module()
    run = module.CollectionRun("run0", ("--output", "unused"), Path("unused"))

    manifest = module._build_manifest(
        plan_sha256="a" * 64,
        runs=(run,),
        rows=[{"completed": True}],
        failed=0,
        started=module.time.perf_counter(),
        final=True,
        allow_cross_scene_process=False,
    )

    assert manifest["status"] == module.COLLECTION_COMPLETE_STATUS
    assert manifest["clean_worker_count"] == 1
    assert manifest["completion_audit"] == {
        "all_planned_runs_present": True,
        "all_workers_clean": True,
        "requires": (
            "final=true, failed_episode_count=0, every planned run present, "
            "and every worker summary completed=true"
        ),
    }


def test_coverage_audit_counts_scenes_separately_from_repeated_episodes() -> None:
    module = _module()
    rows = [
        {"scene_id": "scene-a", "strategy": "random", "transition_count": 8},
        {"scene_id": "scene-a", "strategy": "auction", "transition_count": 7},
        {"scene_id": "scene-b", "strategy": "random", "transition_count": 8},
    ]

    audit = module._coverage_audit(rows)

    assert audit["independent_scene_count"] == 2
    assert audit["scene_episode_counts"] == {"scene-a": 2, "scene-b": 1}
    assert audit["scene_transition_counts"] == {"scene-a": 15, "scene-b": 8}
    assert audit["strategy_episode_counts"] == {"auction": 1, "random": 2}
    assert audit["scene_strategy_episode_counts"]["scene-a::auction"] == 1


@pytest.mark.parametrize(
    "transition_key",
    [
        "marvel_training_transitions",
        "marvel_supplementary_reference_training_transitions",
    ],
)
def test_persistent_collection_rejects_retired_marvel_transition_family(
    tmp_path: Path, transition_key: str
) -> None:
    module = _module()
    output = tmp_path / "legacy-marvel.json"
    output.write_text(
        json.dumps(
            {
                "runtime_record_sha256": "a" * 64,
                "decisions": [{}],
                "task_reservation": {
                    "schema_version": module.PUBLIC_TASK_RESERVATION_SCHEMA_VERSION,
                },
                transition_key: [],
            }
        ),
        encoding="utf-8",
    )
    module.require_sha256 = lambda value, _name: value
    module.canonical_sha256 = lambda _payload: "a" * 64
    module.require_current_public_schema = lambda _payload, **_kwargs: None
    run = module.CollectionRun("legacy-marvel", (), output)

    with pytest.raises(ValueError, match="rejects retired MARVEL transition family"):
        module._validated_worker_summary(run)
