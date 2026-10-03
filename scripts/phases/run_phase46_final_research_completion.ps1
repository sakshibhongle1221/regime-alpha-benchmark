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

$ArgsList = @("src\phase46_final_research_completion.py")
if ($DryRun) {
  $ArgsList += "--dry-run"
}

& $Python @ArgsList
