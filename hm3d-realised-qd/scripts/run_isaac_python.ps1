[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $PythonArgs
)

$ErrorActionPreference = "Stop"
$DefaultIsaacPython = Join-Path $env:USERPROFILE "anaconda3\envs\env_isaaclab\python.exe"
$PythonExe = if ($env:AEROCITY_ISAAC_PYTHON) {
    $env:AEROCITY_ISAAC_PYTHON
} else {
    $DefaultIsaacPython
}

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Isaac Python is missing: $PythonExe"
}

$ResolvedPython = (& $PythonExe -c "import pathlib,sys; print(pathlib.Path(sys.executable).resolve())").Trim()
if ($LASTEXITCODE -ne 0 -or $ResolvedPython -ne (Resolve-Path -LiteralPath $PythonExe).Path) {
    throw "Isaac Python identity check failed: expected $PythonExe, got $ResolvedPython"
}

& $PythonExe @PythonArgs
exit $LASTEXITCODE
