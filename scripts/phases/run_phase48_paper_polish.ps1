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

$ArgsList = @("src\phase48_paper_polish.py")
if ($DryRun) {
  $ArgsList += "--dry-run"
}

& $Python @ArgsList
