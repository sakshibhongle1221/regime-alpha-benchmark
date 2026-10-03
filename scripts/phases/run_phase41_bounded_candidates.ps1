param(
  [string]$PythonExe = ".\env\Scripts\python.exe"
)
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot


& $PythonExe src\phase41_bounded_candidates.py
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

& $PythonExe -m unittest tests.test_phase41_bounded_candidates -v
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host "OK: Phase 41 bounded candidate protocol artifacts and tests completed."
