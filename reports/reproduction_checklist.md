# Reproduction Checklist

## Before Running

- Use Python 3.11.
- Create and activate `env`.
- Install `requirements-research.txt`.
- Confirm local data access or a populated `data/market.duckdb`.
- Confirm `.env` is local only and not tracked.

## Smoke Reproduction

Run:

```powershell
.\reproduce.ps1 -Mode smoke
```

Linux/macOS equivalent:

```bash
bash reproduce.sh --mode smoke
```

Expected checks:

- Python files compile.
- Existing human-reviewed paper artifacts are preserved; missing paper artifacts initialize.
- `models/validation_audit.csv` updates.
- Validation audit has no critical failures.

The expected methodological warning is that legacy `regime_assignments.csv` is an offline/global artifact. Predictive claims should use fold-local artifacts.

## Full Reproduction

Run:

```powershell
.\reproduce.ps1 -Mode full
```

Linux/macOS equivalent:

```bash
bash reproduce.sh --mode full
```

Optional archival run:

```powershell
.\reproduce.ps1 -Mode full -Archive
```

Linux/macOS equivalent:

```bash
bash reproduce.sh --mode full --archive
```

The full run retrains encoders and can take a long time on CPU.

## Dashboard Reproduction

Run:

```powershell
.\reproduce.ps1 -Mode dashboard
```

Linux/macOS equivalent:

```bash
bash reproduce.sh --mode dashboard
```

Then open the Streamlit local URL printed by the command.

## Evidence Checks

After reproduction, verify:

- `models/validation_audit.csv` has no critical failures.
- `models/walkforward_experiment_results.csv` has equal test rows across methods.
- `models/paper_statistical_summary.csv` still marks guided-HMM versus raw-feature HMM as directional, not statistically dominant.
- `paper/main.md` does not claim profitability, true regimes, or cross-asset alpha generalization.

## Git Safety Checks

Before committing:

```powershell
git status --short
git diff --check
git ls-files | Select-String ".env|data/|market.duckdb|encoder.pt|guided_encoder.pt|embeddings.npy|posteriors.npy|labels.npy"
```

The final command should print nothing.
