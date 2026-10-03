param(
    [int]$Epochs = 30,
    [int]$BatchSize = 128,
    [int]$MaxWindows = 5000,
    [int]$MaxFolds = 0,
    [string]$RunName = "phase39r_neural_full_v1",
    [string]$OutputDir = "models",
    [string]$OutputPrefix = "crypto20_repaired_fold_local_",
    [string]$ReportPath = "reports\phase39r_neural_fold_local_results.md",
    [switch]$Resume,
    [switch]$CalendarAuditOnly
)
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot


$ErrorActionPreference = "Stop"

$PythonExe = Join-Path $RepoRoot "env\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$Arguments = @(
    "src\fold_local_encoder_walkforward.py",
    "--universe", "crypto20",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--max-windows", $MaxWindows,
    "--run-name", $RunName,
    "--output-dir", $OutputDir,
    "--output-prefix", $OutputPrefix,
    "--report-path", $ReportPath
)

if ($MaxFolds -gt 0) {
    $Arguments += @("--max-folds", $MaxFolds)
}

if ($Resume) {
    $Arguments += "--resume"
}

if ($CalendarAuditOnly) {
    $Arguments += "--calendar-audit-only"
}

& $PythonExe @Arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
