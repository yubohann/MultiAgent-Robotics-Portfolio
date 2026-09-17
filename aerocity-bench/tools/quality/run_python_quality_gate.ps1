[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$testGroups = @(
    @(
        "tests/pipeline/test_benchmark.py",
        "tests/unit/test_cf2x_contract.py",
        "tests/unit/test_hover_stability.py",
        "tests/pipeline/test_l1_measurement_evidence.py",
        "tests/unit/test_measurement_claim.py",
        "tests/unit/test_statistical_protocol.py",
        "tests/unit/test_public_boundary.py"
    ),
    @(
        "tests/pipeline/test_cf2x_fixture_aggregate.py",
        "tests/pipeline/test_cf2x_b_gate.py",
        "tests/pipeline/test_cf2x_l0_pairing.py",
        "tests/pipeline/test_g2i_a_gate.py",
        "tests/pipeline/test_g2i_mission_sector.py",
        "tests/pipeline/test_g2i_risk_gates.py",
        "tests/pipeline/test_inspection_atlas.py",
        "tests/unit/test_quadrotor_guidance.py"
    ),
    @(
        "tests/pipeline/test_ordinary_v3.py",
        "-k",
        "not external_process_bridge and not guarded_process and not windows_tree_stop and not host_mutex"
    ),
    @("tests/pipeline/test_ordinary_v3.py::test_external_process_bridge_binds_public_requests_and_canonical_actions"),
    @("tests/pipeline/test_ordinary_v3.py::test_external_process_bridge_rejects_private_wire_payload_and_false_boundaries"),
    @("tests/pipeline/test_ordinary_v3.py::test_external_process_bridge_rejects_mismatched_response_and_timeout"),
    @("tests/pipeline/test_ordinary_v3.py::test_windows_1344_and_commit_pressure_are_host_failures"),
    @("tests/pipeline/test_ordinary_v3.py::test_guarded_process_classifies_exit_timeout_commit_and_1344"),
    @("tests/pipeline/test_ordinary_v3.py::test_guarded_process_rejects_a_residual_isaac_runtime_after_success"),
    @("tests/pipeline/test_ordinary_v3.py::test_guarded_process_writes_preflight_and_monitor_failure_receipts"),
    @("tests/pipeline/test_ordinary_v3.py::test_windows_tree_stop_falls_back_when_taskkill_fails"),
    @("tests/pipeline/test_ordinary_v3.py::test_host_mutex_is_exclusive_and_releasable"),
    @(
        "tests/unit/test_quadrotor_dynamics.py",
        "tests/unit/test_quadrotor_preflight_batch.py",
        "tests/unit/test_cf2x_fleet_preflight_contract.py"
    ),
    @("tests/unit/test_sensor_profiles.py", "tests/unit/test_vertical_slice_contract.py")
)

& $Python -c "import jsonschema, pytest, ruff"
if ($LASTEXITCODE -ne 0) {
    throw "Python environment lacks the required .[dev] quality dependencies."
}

& $Python -m ruff check src tests tools
if ($LASTEXITCODE -ne 0) {
    throw "Ruff quality gate failed."
}

for ($index = 0; $index -lt $testGroups.Count; $index++) {
    Write-Host "Running AeroCityBench Python test group $($index + 1)/$($testGroups.Count)..."
    & $Python -m pytest -q @($testGroups[$index])
    if ($LASTEXITCODE -ne 0) {
        throw "Python test group $($index + 1) failed."
    }
}

Write-Host "AeroCityBench Python quality gate passed."
