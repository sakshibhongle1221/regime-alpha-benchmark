# Phase runner scripts

This folder contains phase-specific reproduction and artifact-generation runners.

The repository root intentionally keeps only the main entry points, such as `reproduce.ps1`, `reproduce.sh`, `run_research_grade_checks.ps1`, and the dashboard files. Individual phase runners live here to keep the GitHub root clean while preserving reproducibility.

All runners resolve the repository root before execution, so they can be launched from the root, for example:

```powershell
.\scripts\phases\run_phase47_submission_build.ps1 -DryRun
```

Current paper-package runner:

```powershell
.\scripts\phases\run_phase49_submission_closure.ps1 -DryRun
```
