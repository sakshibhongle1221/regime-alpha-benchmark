param(
    [ValidateSet("artifact", "full")]
    [string]$Mode = "artifact"
)

$ErrorActionPreference = "Stop"

$PythonExe = Join-Path $PSScriptRoot "env\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

& $PythonExe "src\research_grade_checks.py" --mode $Mode
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
