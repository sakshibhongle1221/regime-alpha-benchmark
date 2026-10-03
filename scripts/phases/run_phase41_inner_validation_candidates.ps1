param(
  [string]$PythonExe = ".\env\Scripts\python.exe",
  [int]$MaxFolds = 1,
  [string]$OutputPrefix = "phase41_classical_",
  [string]$ReportPath = "reports\phase41_inner_validation_candidate_run.md",
  [string[]]$Methods = @(
    "global_lgbm",
    "regime_lgbm_hmm",
    "regime_lgbm_kmeans",
    "regime_lgbm_vol_bucket"
  )
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

)

& $PythonExe src\phase41_inner_validation_candidates.py `
  --universe crypto20 `
  --max-folds $MaxFolds `
  --output-prefix $OutputPrefix `
  --report-path $ReportPath `
  --methods $Methods

exit $LASTEXITCODE
