$ErrorActionPreference = "Stop"
$PackRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
& (Join-Path $PackRoot "run_gate_scenarios.ps1") -Scenario single_static @args
