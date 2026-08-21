param(
    [ValidateSet("single_static", "single_dynamic", "multi_static", "multi_dynamic", "all")]
    [string]$Scenario = "all",
    [string]$Python = "",
    [string]$Device = "cpu",
    [int]$Episodes = 1,
    [int[]]$Seeds = @(0),
    [int]$Workers = 1,
    [int]$SingleStaticGateCount = 6,
    [int]$SingleDynamicGateCount = 12,
    [int]$MultiStaticGateCount = 60,
    [int]$MultiDynamicGateCount = 24,
    [int]$TeamSize = 8,
    [string]$ModelRoot = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

$PackRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $PackRoot
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $PackRoot "outputs"
}

function Resolve-PythonExecutable {
    param([string]$RequestedPython)
    if ($RequestedPython -and (Test-Path -LiteralPath $RequestedPython)) {
        return (Resolve-Path -LiteralPath $RequestedPython).Path
    }
    if ($RequestedPython) {
        $cmd = Get-Command $RequestedPython -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    $fallback = Get-Command python -ErrorAction SilentlyContinue
    if ($fallback) { return $fallback.Source }
    throw "No Python executable found. Pass -Python C:\path\to\python.exe"
}

function Invoke-Checked {
    param([string[]]$CommandArgs)
    Write-Host "[run] $($CommandArgs -join ' ')"
    & $CommandArgs[0] @($CommandArgs[1..($CommandArgs.Count - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Resolve-ModelRoot {
    param([string]$RequestedRoot)
    if ($RequestedRoot) {
        return (Resolve-Path -LiteralPath $RequestedRoot).Path
    }
    if ($env:AEROGATE_MODEL_ROOT) {
        return (Resolve-Path -LiteralPath $env:AEROGATE_MODEL_ROOT).Path
    }
    return (Join-Path $PackRoot "models")
}

function Ensure-ModelPack {
    param([string]$ResolvedModelRoot)
    $Required = @(
        "single_gate_density\best_agent.pt",
        "multi_static_gate60\best_agent.pt",
        "multi_dynamic_c4a_24gate\best_agent.pt"
    )
    $Missing = @()
    foreach ($rel in $Required) {
        $path = Join-Path $ResolvedModelRoot $rel
        if (-not (Test-Path -LiteralPath $path)) {
            $Missing += $rel
        }
    }
    if ($Missing.Count -gt 0) {
        throw "Required model files are missing under ${ResolvedModelRoot}: $($Missing -join ', ')"
    }
}

function Add-DeviceArg {
    param([string[]]$CommandList)
    if ($Device) {
        return $CommandList + @("--device", $Device)
    }
    return $CommandList
}

function Add-CommonArgs {
    param([string[]]$CommandList)
    $seedText = @()
    foreach ($seed in $Seeds) { $seedText += [string]$seed }
    return $CommandList + @("--seeds") + $seedText + @("--episodes", [string]$Episodes, "--workers", [string]$Workers, "--overwrite")
}

$PythonExe = Resolve-PythonExecutable -RequestedPython $Python
$ModelRoot = Resolve-ModelRoot -RequestedRoot $ModelRoot
$env:PYTHONPATH = $ProjectRoot
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

Write-Host "[run] pack root: $PackRoot"
Write-Host "[run] project root: $ProjectRoot"
Write-Host "[run] python: $PythonExe"
Write-Host "[run] model root: $ModelRoot"
Write-Host "[run] scenario: $Scenario"
Write-Host "[run] output root: $OutputRoot"

Ensure-ModelPack -ResolvedModelRoot $ModelRoot

$SingleCheckpoint = Join-Path $ModelRoot "single_gate_density\best_agent.pt"
$MultiStaticCheckpoint = Join-Path $ModelRoot "multi_static_gate60\best_agent.pt"
$MultiDynamicCheckpoint = Join-Path $ModelRoot "multi_dynamic_c4a_24gate\best_agent.pt"

$Scenarios = if ($Scenario -eq "all") {
    @("single_static", "single_dynamic", "multi_static", "multi_dynamic")
} else {
    @($Scenario)
}

Push-Location $ProjectRoot
try {
    foreach ($item in $Scenarios) {
        $ScenarioOutput = Join-Path $OutputRoot $item
        New-Item -ItemType Directory -Force -Path $ScenarioOutput | Out-Null
        if ($item -eq "single_static") {
            $ScenarioCommand = @(
                $PythonExe,
                (Join-Path $ProjectRoot "gate_density_single\scripts\run_paper_gate_density_single_eval.py"),
                "--checkpoint", $SingleCheckpoint,
                "--python", $PythonExe,
                "--experiments", "E1_static_single_gate_density",
                "--methods", "full",
                "--gate-counts", [string]$SingleStaticGateCount,
                "--output-root", $ScenarioOutput
            )
            Invoke-Checked -CommandArgs (Add-DeviceArg -CommandList (Add-CommonArgs -CommandList $ScenarioCommand))
        }
        elseif ($item -eq "single_dynamic") {
            $ScenarioCommand = @(
                $PythonExe,
                (Join-Path $ProjectRoot "gate_density_single\scripts\run_paper_gate_density_single_eval.py"),
                "--checkpoint", $SingleCheckpoint,
                "--python", $PythonExe,
                "--experiments", "E2_dynamic_single_gate_density",
                "--methods", "full",
                "--gate-counts", [string]$SingleDynamicGateCount,
                "--output-root", $ScenarioOutput
            )
            Invoke-Checked -CommandArgs (Add-DeviceArg -CommandList (Add-CommonArgs -CommandList $ScenarioCommand))
        }
        elseif ($item -eq "multi_static") {
            $ScenarioCommand = @(
                $PythonExe,
                (Join-Path $ProjectRoot "multi_gate\scripts\run_paper_multi_gate_density_eval.py"),
                "--checkpoint", $MultiStaticCheckpoint,
                "--python", $PythonExe,
                "--experiments", "E4_static_multi_8d",
                "--methods", "full",
                "--gate-counts", [string]$MultiStaticGateCount,
                "--team-sizes", [string]$TeamSize,
                "--output-root", $ScenarioOutput
            )
            Invoke-Checked -CommandArgs (Add-DeviceArg -CommandList (Add-CommonArgs -CommandList $ScenarioCommand))
        }
        elseif ($item -eq "multi_dynamic") {
            $ScenarioCommand = @(
                $PythonExe,
                (Join-Path $ProjectRoot "multi_gate\scripts\run_paper_multi_gate_density_eval.py"),
                "--checkpoint", $MultiDynamicCheckpoint,
                "--python", $PythonExe,
                "--experiments", "E5_dynamic_multi_8d",
                "--methods", "full",
                "--gate-counts", [string]$MultiDynamicGateCount,
                "--team-sizes", [string]$TeamSize,
                "--output-root", $ScenarioOutput
            )
            Invoke-Checked -CommandArgs (Add-DeviceArg -CommandList (Add-CommonArgs -CommandList $ScenarioCommand))
        }
    }
}
finally {
    Pop-Location
}

Write-Host "[run] complete. Outputs are under: $OutputRoot"
