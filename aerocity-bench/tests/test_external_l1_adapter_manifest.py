from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aerocity_bench.adapters import (
    AdapterDeclaration,
    ExternalProcessPlannerBridge,
    load_external_l1_adapter_manifest,
)
from aerocity_bench.canonical import file_hash, write_json
from aerocity_bench.cf2x_fleet_preflight_contract import (
    COMPLETE_CALIBRATION_PURPOSE,
    EXTERNAL_PROCESS_POLICY_MODE,
    FROZEN_COMPLETE_CALIBRATION_DURATION_S,
    SHORT_PREFLIGHT_PURPOSE,
    public_policy_progress_status,
    validate_native_run_purpose,
)
from tools.build_external_l1_adapter_manifest import _args


def _manifest(tmp_path, *, task_domain: str, comparability_claim: str):
    adapter = tmp_path / "adapter.py"
    checkpoint = tmp_path / "checkpoint.bin"
    upstream = tmp_path / "upstream"
    adapter.write_text("print('adapter')\n", encoding="utf-8")
    checkpoint.write_bytes(b"frozen-checkpoint")
    upstream.mkdir()
    import subprocess

    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    (upstream / "source.py").write_text("print('upstream')\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(upstream), "add", "source.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(upstream),
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    upstream_commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest_path = tmp_path / "adapter-manifest.json"
    write_json(
        manifest_path,
        {
            "schema": "org.aerocity.bench.external-l1-adapter-manifest.v1",
            "declaration": {
                "adapter_id": "test-external-process-v1",
                "method_id": "test-external-method",
                "capability_profile": "G2-I",
                "upstream_url": "https://example.invalid/upstream.git",
                "upstream_commit": upstream_commit,
                "upstream_license": "MIT",
                "process_boundary": "process",
                "training_allowed": False,
                "decentralized_execution": False,
                "runtime_image_digest": None,
            },
            "command": [
                "{python_executable}",
                "-u",
                "{adapter_source}",
                "--upstream-source",
                "{upstream_source}",
                "--checkpoint",
                "{checkpoint}",
            ],
            "adapter_source_path": "adapter.py",
            "adapter_source_sha256": file_hash(adapter),
            "upstream_source_path": "upstream",
            "checkpoint_path": "checkpoint.bin",
            "checkpoint_sha256": file_hash(checkpoint),
            "runtime_environment_sha256": "b" * 64,
            "task_domain": task_domain,
            "comparability_claim": comparability_claim,
        },
    )
    return manifest_path, adapter


def test_external_l1_manifest_binds_files_without_publishing_runtime_paths(tmp_path) -> None:
    manifest_path, _ = _manifest(
        tmp_path,
        task_domain="2d_exploration_transfer",
        comparability_claim="transfer_diagnostic",
    )

    manifest = load_external_l1_adapter_manifest(manifest_path)
    provenance = manifest.public_provenance()

    assert provenance["task_domain"] == "2d_exploration_transfer"
    assert provenance["comparability_claim"] == "transfer_diagnostic"
    assert "adapter_source_path" not in provenance
    assert "checkpoint_path" not in provenance
    assert "command" not in provenance
    assert manifest.launch_command("python.exe")[0] == "python.exe"


def test_external_solver_lock_manifest_has_no_checkpoint_and_cannot_train(tmp_path) -> None:
    manifest_path, _ = _manifest(
        tmp_path,
        task_domain="3d_geometry_search",
        comparability_claim="transfer_diagnostic",
    )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["schema"] = "org.aerocity.bench.external-l1-adapter-manifest.v2"
    raw["command"] = [
        "{python_executable}",
        "-u",
        "{adapter_source}",
        "--upstream-source",
        "{upstream_source}",
    ]
    raw["execution_artifact_kind"] = "solver_lock"
    raw["execution_artifact_path"] = raw.pop("checkpoint_path")
    raw["execution_artifact_sha256"] = raw.pop("checkpoint_sha256")
    from aerocity_bench.canonical import write_json

    write_json(manifest_path, raw)

    manifest = load_external_l1_adapter_manifest(manifest_path)
    provenance = manifest.public_provenance()
    assert manifest.execution_artifact_kind == "solver_lock"
    assert "{checkpoint}" not in manifest.command
    assert provenance["execution_artifact_kind"] == "solver_lock"
    assert "execution_artifact_path" not in provenance

    raw["declaration"]["training_allowed"] = True
    write_json(manifest_path, raw)
    with pytest.raises(ValueError, match="solver lock cannot declare training"):
        load_external_l1_adapter_manifest(manifest_path)


def test_v3_manifest_binds_its_isolated_python_without_publishing_path(tmp_path) -> None:
    manifest_path, _ = _manifest(
        tmp_path,
        task_domain="3d_geometry_search",
        comparability_claim="transfer_diagnostic",
    )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["schema"] = "org.aerocity.bench.external-l1-adapter-manifest.v3"
    raw["execution_artifact_kind"] = "solver_lock"
    raw["execution_artifact_path"] = raw.pop("checkpoint_path")
    raw["execution_artifact_sha256"] = raw.pop("checkpoint_sha256")
    raw["command"] = [
        "{python_executable}",
        "-u",
        "{adapter_source}",
        "--upstream-source",
        "{upstream_source}",
    ]
    raw["runtime_python_path"] = sys.executable
    raw["runtime_python_sha256"] = file_hash(Path(sys.executable))
    write_json(manifest_path, raw)

    manifest = load_external_l1_adapter_manifest(manifest_path)
    provenance = manifest.public_provenance()

    assert manifest.launch_command()[0] == str(Path(sys.executable).resolve())
    assert provenance["runtime_python_sha256"] == file_hash(Path(sys.executable))
    assert "runtime_python_path" not in provenance
    with pytest.raises(ValueError, match="differs from the manifest lock"):
        manifest.launch_command("different-python.exe")


def test_v3_manifest_rejects_tampered_runtime_python(tmp_path) -> None:
    manifest_path, _ = _manifest(
        tmp_path,
        task_domain="3d_geometry_search",
        comparability_claim="transfer_diagnostic",
    )
    runtime_python = tmp_path / "python.exe"
    runtime_python.write_bytes(b"isolated-python-before")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["schema"] = "org.aerocity.bench.external-l1-adapter-manifest.v3"
    raw["execution_artifact_kind"] = "solver_lock"
    raw["execution_artifact_path"] = raw.pop("checkpoint_path")
    raw["execution_artifact_sha256"] = raw.pop("checkpoint_sha256")
    raw["command"] = [
        "{python_executable}",
        "-u",
        "{adapter_source}",
        "--upstream-source",
        "{upstream_source}",
    ]
    raw["runtime_python_path"] = "python.exe"
    raw["runtime_python_sha256"] = file_hash(runtime_python)
    write_json(manifest_path, raw)
    runtime_python.write_bytes(b"isolated-python-after")

    with pytest.raises(ValueError, match="runtime Python hash differs"):
        load_external_l1_adapter_manifest(manifest_path)


def test_external_process_separates_initialization_from_action_deadline(tmp_path) -> None:
    server = tmp_path / "slow_reset_server.py"
    server.write_text(
        """import json
import sys
import time

for line in sys.stdin:
    request = json.loads(line)
    if request[\"kind\"] in {\"reset\", \"act\"}:
        time.sleep(0.05)
    response = {
        \"schema\": \"org.aerocity.bench.external-planner-response.v1\",
        \"request_id\": request[\"request_id\"],
        \"status\": \"ok\",
    }
    print(json.dumps(response), flush=True)
""",
        encoding="utf-8",
    )
    declaration = AdapterDeclaration(
        adapter_id="initialization-timeout-test",
        method_id="test-external-method",
        capability_profile="G1",
        upstream_url="https://example.invalid/upstream.git",
        upstream_commit="a" * 40,
        upstream_license="MIT",
        process_boundary="process",
        training_allowed=False,
        decentralized_execution=False,
    )
    with ExternalProcessPlannerBridge(
        declaration,
        [sys.executable, "-u", str(server)],
        cwd=tmp_path,
        response_timeout_s=0.01,
        initialization_timeout_s=0.5,
    ) as bridge:
        bridge.reset({"episode_id": "public-only"})
        initialization = bridge.initialization_report()
        with pytest.raises(TimeoutError, match="act JSONL response"):
            bridge._request("act")

    assert initialization["completed"] is True
    assert initialization["deadline_s"] == 0.5
    assert float(initialization["elapsed_s"]) >= 0.05


def test_external_l1_manifest_rejects_tampered_adapter_source(tmp_path) -> None:
    manifest_path, adapter = _manifest(
        tmp_path,
        task_domain="3d_geometry_search",
        comparability_claim="substantive_3d",
    )
    adapter.write_text("print('tampered')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="adapter source hash differs"):
        load_external_l1_adapter_manifest(manifest_path)


def test_external_l1_manifest_rejects_dirty_or_mismatched_upstream_worktree(tmp_path) -> None:
    manifest_path, _ = _manifest(
        tmp_path,
        task_domain="3d_geometry_search",
        comparability_claim="substantive_3d",
    )
    (tmp_path / "upstream" / "source.py").write_text("print('modified')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="worktree must be clean"):
        load_external_l1_adapter_manifest(manifest_path)


def test_external_l1_manifest_allows_only_python_bytecode_caches(tmp_path) -> None:
    manifest_path, _ = _manifest(
        tmp_path,
        task_domain="3d_geometry_search",
        comparability_claim="substantive_3d",
    )
    cache = tmp_path / "upstream" / "__pycache__"
    cache.mkdir()
    (cache / "runtime.cpython-311.pyc").write_bytes(b"cache")

    manifest = load_external_l1_adapter_manifest(manifest_path)
    assert manifest.declaration.method_id == "test-external-method"


def test_2d_external_method_cannot_claim_substantive_3d_comparability(tmp_path) -> None:
    manifest_path, _ = _manifest(
        tmp_path,
        task_domain="2d_exploration_transfer",
        comparability_claim="substantive_3d",
    )

    with pytest.raises(ValueError, match="non-3-D external method"):
        load_external_l1_adapter_manifest(manifest_path)


def test_external_process_mode_uses_frozen_duration_and_fails_closed_on_adapter_error() -> None:
    validate_native_run_purpose(
        purpose=COMPLETE_CALIBRATION_PURPOSE,
        execution_mode=EXTERNAL_PROCESS_POLICY_MODE,
        requested_sim_time_s=FROZEN_COMPLETE_CALIBRATION_DURATION_S,
        frozen_episode_duration_s=FROZEN_COMPLETE_CALIBRATION_DURATION_S,
    )
    assert (
        public_policy_progress_status(
            purpose=SHORT_PREFLIGHT_PURPOSE,
            observe_action_count=1,
            confirmation_receipt_count=0,
            return_action_count=0,
            all_returned_home=True,
            episode_budget_completed=False,
            safe_completion=True,
            deadline_miss_tick_count=0,
            adapter_failure_count=1,
        )
        == "ADAPTER_FAILED"
    )


def test_manifest_builder_accepts_command_template_file_for_windows_shell_safety(tmp_path) -> None:
    template = tmp_path / "command.json"
    template.write_text('["{python_executable}"]\n', encoding="utf-8")

    parsed = _args(
        [
            "--output",
            "manifest.json",
            "--adapter-id",
            "adapter",
            "--method-id",
            "method",
            "--upstream-url",
            "https://example.invalid/upstream.git",
            "--upstream-commit",
            "a" * 40,
            "--upstream-license",
            "MIT",
            "--upstream-source",
            "upstream",
            "--adapter-source",
            "adapter.py",
            "--checkpoint",
            "checkpoint.bin",
            "--runtime-environment-manifest",
            "environment.json",
            "--runtime-python",
            "python.exe",
            "--task-domain",
            "2d_exploration_transfer",
            "--comparability-claim",
            "transfer_diagnostic",
            "--command-template-file",
            str(template),
        ]
    )

    assert parsed.command_template_file == template
    assert parsed.command_template_json is None
