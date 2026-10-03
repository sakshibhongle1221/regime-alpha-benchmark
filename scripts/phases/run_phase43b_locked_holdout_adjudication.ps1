$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$ErrorActionPreference = "Stop"
.\env\Scripts\python.exe src\phase43b_locked_holdout_adjudication.py
