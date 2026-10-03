param(
    [int]$Epochs = 30,
    [int]$BatchSize = 128,
    [int]$MaxWindows = 0,
    [switch]$TrainOnly
)
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot


$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\env\Scripts\python.exe")) {
    throw "Missing .\env\Scripts\python.exe. Activate or recreate the project virtual environment first."
}

if (-not (Test-Path ".\models\crypto20_regime_assignments.csv")) {
    throw "Missing models\crypto20_regime_assignments.csv. Run src\crypto20_regime_benchmark.py --universe crypto20 first."
}

New-Item -ItemType Directory -Force -Path ".tmp" | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = ".tmp\phase35_crypto20_guided_$timestamp.log"

$argsList = @(
    "src\guided_encoder.py",
    "--universe", "crypto20",
    "--eligible-only",
    "--hmm-assignment-path", "models\crypto20_regime_assignments.csv",
    "--output-prefix", "crypto20_guided_encoder",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize"
)

if ($MaxWindows -gt 0) {
    $argsList += @("--max-windows", "$MaxWindows")
}

if ($TrainOnly) {
    $argsList += @("--train-only")
}

Write-Host "Starting Phase 35 Crypto-20 guided encoder run..."
Write-Host "Log: $logPath"
Write-Host "Command: .\env\Scripts\python.exe $($argsList -join ' ')"

& .\env\Scripts\python.exe @argsList 2>&1 | Tee-Object -FilePath $logPath

Write-Host "Phase 35 command finished. Log saved to $logPath"
