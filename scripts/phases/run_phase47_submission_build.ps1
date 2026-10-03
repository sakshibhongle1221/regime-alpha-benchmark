param(
  [switch]$DryRun
)
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot


$ErrorActionPreference = "Stop"

$Python = ".\env\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $Python = "python"
}

$ArgsList = @("src\phase47_submission_build.py")
if ($DryRun) {
  $ArgsList += "--dry-run"
}

& $Python @ArgsList
