[CmdletBinding()]
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$testGroups = @(
    @(
        "tests/test_benchmark.py",
        "tests/test_cf2x_contract.py",
        "tests/test_hover_stability.py",
        "tests/test_l1_measurement_evidence.py",
        "tests/test_measurement_claim.py",
        "tests/test_statistical_protocol.py",
        "tests/test_public_boundary.py"
    ),
    @(
        "tests/test_cf2x_fixture_aggregate.py",
        "tests/test_cf2x_b_gate.py",
        "tests/test_cf2x_l0_pairing.py",
        "tests/test_g2_i_a_gate.py",
        "tests/test_g2_i_mission_sector.py",
        "tests/test_g2_i_risk_gates.py",
        "tests/test_inspection_atlas.py",
        "tests/test_quadrotor_guidance.py"
    ),
    @(
        "tests/test_ordinary_v3.py",
        "-k",
        "not external_process_bridge and not guarded_process and not windows_tree_stop and not host_mutex"
    ),
    @("tests/test_ordinary_v3.py::test_external_process_bridge_binds_public_requests_and_canonical_actions"),
    @("tests/test_ordinary_v3.py::test_external_process_bridge_rejects_private_wire_payload_and_false_boundaries"),
    @("tests/test_ordinary_v3.py::test_external_process_bridge_rejects_mismatched_response_and_timeout"),
    @("tests/test_ordinary_v3.py::test_windows_1344_and_commit_pressure_are_host_failures"),
    @("tests/test_ordinary_v3.py::test_guarded_process_classifies_exit_timeout_commit_and_1344"),
    @("tests/test_ordinary_v3.py::test_guarded_process_rejects_a_residual_isaac_runtime_after_success"),
    @("tests/test_ordinary_v3.py::test_guarded_process_writes_preflight_and_monitor_failure_receipts"),
    @("tests/test_ordinary_v3.py::test_windows_tree_stop_falls_back_when_taskkill_fails"),
    @("tests/test_ordinary_v3.py::test_host_mutex_is_exclusive_and_releasable"),
    @(
        "tests/test_quadrotor_dynamics.py",
        "tests/test_quadrotor_preflight_batch.py",
        "tests/test_cf2x_fleet_preflight_contract.py"
    ),
    @("tests/test_sensor_profiles.py", "tests/test_vertical_slice_contract.py")
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
