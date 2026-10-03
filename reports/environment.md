# Regime Alpha Benchmark Environment

## Purpose

This file documents the expected runtime environment for reproducing the research pipeline and running the dashboard.

## Python Version

Recommended local research runtime:

```text
Python 3.11.x
```

The project has been developed and tested on Windows with a local virtual environment named `env`.

## Dependency Files

| File | Purpose |
|---|---|
| `requirements.txt` | Minimal Streamlit Cloud dashboard dependencies |
| `requirements-research.txt` | Full local research pipeline dependencies |

Use `requirements.txt` for the deployed dashboard. Use `requirements-research.txt` for experiments, validation audit, paper generation, robustness, statistics, and interpretability.

## Local Setup

```powershell
py -3.11 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements-research.txt
```

Or use the reproduction helper:

```powershell
.\reproduce.ps1 -Mode smoke -CreateEnv
```

On Linux/macOS:

```bash
bash reproduce.sh --mode smoke --create-env
```

## Data Requirements

Raw OHLCV data and DuckDB databases are not committed to GitHub.

The local research pipeline expects:

```text
data/market.duckdb
```

or a configured environment that can regenerate the feature store through the ingestion pipeline. Curated summary artifacts are committed for the dashboard and paper narrative; raw row-level data, model weights, embeddings, posteriors, and large prediction files remain ignored.

## Hardware Notes

The current project is CPU-friendly for smoke checks and dashboard use. Full encoder training and full pipeline regeneration can take substantially longer, especially for:

- vanilla contrastive encoder training
- HMM-guided encoder training
- time-frequency encoder prototype
- fold-local regime benchmarking
- interpretability diagnostics

## Public Reproduction Modes

| Mode | Command | Purpose |
|---|---|---|
| Smoke | `.\reproduce.ps1 -Mode smoke` | Compile code, verify or initialize paper artifacts, run validation audit |
| Full | `.\reproduce.ps1 -Mode full` | Regenerate the full local research artifact stack |
| Dashboard | `.\reproduce.ps1 -Mode dashboard` | Launch Streamlit dashboard locally |

Linux/macOS equivalents are `bash reproduce.sh --mode smoke`, `bash reproduce.sh --mode full`, and `bash reproduce.sh --mode dashboard`.

## Reproduction Caveats

- The repository is a research benchmark, not a live trading system.
- Backtest metrics are diagnostics, not deployable trading claims.
- HMM states are proxy/reference states, not ground-truth market regimes.
- Current public artifacts support a BTC/ETH downstream-alpha benchmark plus Crypto-20 structural diagnostics; cross-asset alpha generalization is not claimed.
