[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $PythonArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Protocol Python is missing: $PythonExe"
}

$ResolvedPython = (& $PythonExe -c "import pathlib,sys; print(pathlib.Path(sys.executable).resolve())").Trim()
if ($LASTEXITCODE -ne 0 -or $ResolvedPython -ne (Resolve-Path -LiteralPath $PythonExe).Path) {
    throw "Protocol Python identity check failed: expected $PythonExe, got $ResolvedPython"
}

& $PythonExe @PythonArgs
exit $LASTEXITCODE
