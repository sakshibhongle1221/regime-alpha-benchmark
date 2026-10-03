$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$ErrorActionPreference = "Stop"

$python = ".\env\Scripts\python.exe"
if (-not (Test-Path $python)) {
  $python = "python"
}

& $python src\phase43_locked_holdout_freeze.py
