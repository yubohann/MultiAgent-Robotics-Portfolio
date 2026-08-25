param(
    [string]$Python = "",
    [switch]$CreateVenv,
    [string]$VenvPath = "",
    [bool]$InstallTorchIfMissing = $true,
    [switch]$SkipInstall,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

$PackRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $PackRoot
$RequirementsFile = Join-Path $PackRoot "requirements-runtime.txt"

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

function Test-PythonImport {
    param(
        [string]$PythonExe,
        [string]$ModuleName
    )
    & $PythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

if (-not $VenvPath) {
    $VenvPath = Join-Path $PackRoot ".venv"
}

$PythonExe = Resolve-PythonExecutable -RequestedPython $Python

if ($CreateVenv) {
    if (-not (Test-Path -LiteralPath $VenvPath)) {
        Write-Host "[setup] creating venv: $VenvPath"
        & $PythonExe -m venv $VenvPath
    }
    $VenvPython = Join-Path $VenvPath "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Venv Python was not created: $VenvPython"
    }
    $PythonExe = $VenvPython
}

Write-Host "[setup] pack root: $PackRoot"
Write-Host "[setup] project root: $ProjectRoot"
Write-Host "[setup] python: $PythonExe"
& $PythonExe --version

if (-not $SkipInstall) {
    Write-Host "[setup] upgrading pip"
    & $PythonExe -m pip install --upgrade pip

    Write-Host "[setup] installing runtime requirements"
    & $PythonExe -m pip install -r $RequirementsFile

    if ($InstallTorchIfMissing -and -not (Test-PythonImport -PythonExe $PythonExe -ModuleName "torch")) {
        Write-Host "[setup] torch is missing; installing the default torch package for this Python"
        & $PythonExe -m pip install torch
    }
}

$env:PYTHONPATH = $ProjectRoot
Write-Host "[setup] PYTHONPATH=$env:PYTHONPATH"

if (-not $SkipTests) {
    Write-Host "[setup] running smoke tests"
    Push-Location $ProjectRoot
    try {
        & $PythonExe -m pytest tests -q
    }
    finally {
        Pop-Location
    }
}

Write-Host "[setup] complete"
Write-Host "[setup] scenario runner: $PackRoot\run_gate_scenarios.ps1"
