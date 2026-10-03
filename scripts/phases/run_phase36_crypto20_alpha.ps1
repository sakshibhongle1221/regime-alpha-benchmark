param(
    [int]$MaxFolds = 0
)
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot


$ErrorActionPreference = "Stop"

$PythonExe = Join-Path $RepoRoot "env\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$argsList = @(
    "src/walkforward_regimes.py",
    "--universe", "crypto20",
    "--skip-contrastive",
    "--output-prefix", "crypto20_walkforward_",
    "--guided-assignment-path", "models/crypto20_guided_encoder_assignments.csv",
    "--guided-embedding-path", "models/crypto20_guided_encoder_embeddings.npy"
)

if ($MaxFolds -gt 0) {
    $argsList += @("--max-folds", "$MaxFolds")
}

& $PythonExe @argsList
