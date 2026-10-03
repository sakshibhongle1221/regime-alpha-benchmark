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

$ArgsList = @("src\phase45_venue_manuscript_package.py")
if ($DryRun) {
  $ArgsList += "--dry-run"
}

& $Python @ArgsList
