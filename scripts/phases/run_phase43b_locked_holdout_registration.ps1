param(
  [string]$Config = "configs\phase43b_locked_holdout_registration_v1.json",
  [string]$DbPath = ""
)
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot


$ErrorActionPreference = "Stop"

if ($DbPath -eq "") {
  .\env\Scripts\python.exe src\phase43b_locked_holdout_registration.py --config $Config
} else {
  .\env\Scripts\python.exe src\phase43b_locked_holdout_registration.py --config $Config --db-path $DbPath
}
