param(
  [switch]$DryRun
)
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot


$ErrorActionPreference = "Stop"

$ArgsList = @("src\phase44_paper_readiness_package.py")
if ($DryRun) {
  $ArgsList += "--dry-run"
}

& ".\env\Scripts\python.exe" @ArgsList
