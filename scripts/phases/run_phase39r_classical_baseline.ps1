param(
    [int]$MaxFolds = 0,
    [string]$RunName = "phase39r_classical_full",
    [string]$OutputDir = "models",
    [string]$OutputPrefix = "crypto20_repaired_classical_",
    [switch]$Resume
)
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot


$ErrorActionPreference = "Stop"
$PythonExe = Join-Path $RepoRoot "env\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$Arguments = @(
    "src\repaired_classical_baseline.py",
    "--run-name", $RunName,
    "--output-dir", $OutputDir,
    "--output-prefix", $OutputPrefix
)
if ($MaxFolds -gt 0) {
    $Arguments += @("--max-folds", $MaxFolds)
}
if ($Resume) {
    $Arguments += "--resume"
}

& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
