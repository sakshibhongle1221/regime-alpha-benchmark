# Regime Alpha Benchmark

> A research-grade quant ML benchmark platform that tests whether learned market regimes improve alpha modeling versus global baselines and classical regime methods, using proper financial labels, purged walk-forward validation, and transaction-cost-aware evaluation.

**Maintainer:** [sakshibhongle1221](https://github.com/sakshibhongle1221)

**Run locally:** `python -m streamlit run streamlit_app.py`. See the setup instructions below.

**Deploy:** Connect your own GitHub repository to Streamlit Community Cloud and select `streamlit_app.py` as the entry point.

## Key Finding

The strongest current result is not that learned regimes beat every baseline. They do not. The true Gaussian HMM baseline remains the best IC and Sharpe method in the latest BTC+ETH run.

Phase 11 tested the natural research response: fit an HMM on learned contrastive embeddings. The contrastive-HMM hybrid improves the learned-regime path from `IC=-0.0165` to `IC=0.0035` and from `Sharpe=-0.902` to `Sharpe=-0.382`, but it still does not beat the raw-feature HMM. That is the current project insight: HMM-style temporal state structure helps learned embeddings, but representation learning alone is not yet enough.

## Research Question

Does learning market regimes from raw financial time-series features improve alpha-model IC, drawdown, and Sharpe compared with:

- a global model with no regime awareness
- Gaussian HMM regimes
- contrastive-HMM hybrid regimes
- KMeans regimes
- volatility-bucket regimes

The project is intentionally framed as a research benchmark, not a live trading bot. Honest weak or mixed results are part of the contribution.

## Architecture

```text
Binance OHLCV
    -> DuckDB feature store
    -> 22 engineered market/microstructure features
    -> Multi-horizon financial labels and triple-barrier targets
    -> Contrastive temporal regime encoder + contrastive-HMM hybrid + classical baselines
    -> Global and regime-conditioned LightGBM models
    -> Purged walk-forward validation with transaction costs
    -> Research dashboard and report artifacts
```

## Current Capabilities

- Incremental Binance OHLCV ingestion for BTCUSDT and ETHUSDT.
- DuckDB feature store with 22 engineered features.
- Contrastive Transformer-style temporal encoder trained with NT-Xent loss.
- GMM regime posteriors and UMAP/timeline visualizations.
- Multi-horizon targets: 4h, 8h, and 24h.
- Triple-barrier labels with neutral class.
- Regime baselines: contrastive, Gaussian HMM, KMeans, volatility buckets.
- Contrastive-HMM hybrid regimes fitted on learned embeddings.
- Regime stability diagnostics: switch rate, duration, transition entropy, and stable-vs-transition IC.
- Global LightGBM baseline and regime-conditioned LightGBM models.
- 5-day embargoed walk-forward validation.
- Validation audit for fold separation, target horizon leakage, coverage parity, and prediction alignment.
- Fold-local regime refitting benchmark for stricter predictive regime evaluation.
- Phase 14A robustness matrix across BTC-only, ETH-only, and BTC+ETH scopes at 4h, 8h, and 24h horizons.
- Phase 14B stress robustness across transaction costs, signal thresholds, and bull/sideways/bear market periods.
- Phase 15.0 run registry for timestamped artifact snapshots and frozen baselines.
- Transaction-cost-aware experiment result table.
- Streamlit dashboard shell and research note.

## Run Order

Recommended local environment:

```powershell
py -3.11 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements-research.txt
```

```powershell
python src/check.py
python src/targets.py --symbols BTCUSDT ETHUSDT
python src/train_encoder.py --symbols BTCUSDT ETHUSDT
python src/visualize_regimes.py --symbols BTCUSDT ETHUSDT
python src/baselines.py --symbols BTCUSDT ETHUSDT
python src/alpha_models.py --symbols BTCUSDT ETHUSDT
python src/regime_stability.py --symbols BTCUSDT ETHUSDT
python src/walkforward_regimes.py --symbols BTCUSDT ETHUSDT
python src/robustness.py
python src/robustness_stress.py
python src/validation_audit.py --symbols BTCUSDT ETHUSDT
python src/archive_run.py --phase phase14b_baseline --run-id 20260522_phase14b_baseline --source-ref v1.3-phase14b --notes "Frozen Phase 14B baseline before Phase 15 statistical and encoder work."
python src/backtest.py
python -m compileall src dashboard.py streamlit_app.py
```

Optional dashboard:

```powershell
python -m streamlit run streamlit_app.py
```

For Streamlit Cloud deployment, the root `requirements.txt` is intentionally minimal and installs only dashboard dependencies.

If local research dependencies are not installed, run:

```powershell
python -m pip install -r requirements-research.txt
```

## Key Artifacts

| Artifact | Purpose |
|---|---|
| `models/target_distribution.csv` | Label balance across horizons and label types |
| `models/target_quality.csv` | Missing-row, horizon-loss, and neutral-class checks |
| `models/regime_assignments.csv` | Aligned regime labels/posteriors for all methods |
| `models/regime_benchmark_summary.csv` | Regime-level comparison table |
| `models/regime_stability_summary.csv` | Persistence, switch-rate, confidence, and stable-vs-transition IC diagnostics |
| `models/per_regime_stats.csv` | Volatility, return, liquidity, and IC diagnostics by regime |
| `models/experiment_results.csv` | Master model comparison table |
| `models/alpha_oos_predictions.csv` | Out-of-sample model predictions |
| `models/validation_audit.csv` | Leakage, embargo, fold, coverage, and prediction-alignment audit |
| `models/fold_audit.csv` | Fold-level train/test boundary and embargo checks |
| `models/walkforward_experiment_results.csv` | Strict fold-local regime-refit alpha benchmark |
| `models/walkforward_comparison.csv` | Offline/global versus fold-local regime result comparison |
| `models/walkforward_regime_summary.csv` | Fold-local regime coverage, duration, and confidence diagnostics |
| `models/robustness_results.csv` | Phase 14A full horizon/symbol-scope robustness matrix |
| `models/robustness_summary.csv` | Best method per robustness grid cell |
| `models/robustness_wins.csv` | Win counts by metric across robustness grid cells |
| `models/robustness_heatmap.png` | Visual summary of robustness winners |
| `models/robustness_stress_results.csv` | Phase 14B full cost/threshold/market-period stress grid |
| `models/robustness_stress_summary.csv` | Best method per stress grid cell |
| `models/robustness_stress_wins.csv` | Stress-test win counts by metric |
| `models/robustness_stress_heatmap.png` | Visual summary of stress-test sensitivity |
| `runs/run_index.csv` | Versioned run registry |
| `runs/20260522_phase14b_baseline/manifest.json` | Frozen Phase 14B baseline manifest |
| `reports/model_card.md` | Reproducible model-card snapshot |
| `reports/compute_budget.md` | Compute-aware experiment plan and multi-asset gate |
| `models/regime_stability.png` | Stability and transition-period IC dashboard panel |
| `models/phase4_dashboard.png` | Static research backtest dashboard |
| `reports/regime_alpha_benchmark_report.md` | Research note |

## Project Structure

```text
regime-alpha-benchmark/
├── dashboard.py
├── streamlit_app.py
├── reports/
│   └── regime_alpha_benchmark_report.md
├── src/
│   ├── ingestion.py
│   ├── features.py
│   ├── targets.py
│   ├── dataset.py
│   ├── encoder.py
│   ├── train_encoder.py
│   ├── visualize_regimes.py
│   ├── baselines.py
│   ├── regime_stability.py
│   ├── validation_audit.py
│   ├── walkforward_regimes.py
│   ├── robustness.py
│   ├── robustness_stress.py
│   ├── archive_run.py
│   ├── alpha_models.py
│   ├── backtest.py
│   └── check.py
├── models/
│   └── generated research artifacts
└── runs/
    └── versioned curated run snapshots
```

## Methodology Notes

The primary target is `tb_label_8h`, an 8-hour triple-barrier label with classes `-1`, `0`, and `+1`. The alpha score is `P(+1) - P(-1)`. A prediction becomes a trade only when the neutral class is not dominant and the absolute score exceeds the threshold.

The validation scheme uses expanding walk-forward folds with a 5-day embargo gap between train and test windows. This reduces leakage from overlapping financial labels and makes the model comparison more defensible.

Phase 12 adds a validation audit. The current audit passes all critical checks: required tables, feature/target schema, finite joined rows, 24-row target horizon loss, 18 walk-forward folds, 120-bar embargo, 8-bar label-horizon purge, equal method coverage, prediction/test-fold alignment, experiment-result row counts, fold-local artifact coverage, Phase 14A robustness artifact completeness, Phase 14B stress-grid completeness, and run-registry snapshot completeness.

The audit also records one methodological warning: the legacy `regime_assignments.csv` artifact is offline/global. Paper-grade predictive regime claims should use the Phase 13 `walkforward_experiment_results.csv` artifact instead.

Phase 13 adds that stricter benchmark. In `walkforward_regimes.py`, the regime assignment layer is refit inside each fold using only training-history rows, then applied to test rows. HMM-based test assignments use an online filtering pass initialized from the training sequence and advanced through the embargo gap. The contrastive encoder remains a frozen offline representation in this phase, but GMM/HMM assignment models on top of those embeddings are fit fold-locally.

Phase 14A adds a compact robustness matrix. The fold-local benchmark is repeated across BTC-only, ETH-only, and BTC+ETH scopes, and across 4h, 8h, and 24h triple-barrier targets. This tests whether the main result is stable across assets and prediction horizons instead of being a single-configuration artifact.

Phase 14B adds stress robustness on top of the fold-local prediction file. It does not retrain the models; it re-scores the same out-of-sample predictions under different signal thresholds, transaction costs, and market-period slices. Bull, sideways, and bear periods are defined from rolling 30-day feature-store returns. This tests whether the benchmark conclusion survives realistic deployment assumptions.

Phase 15.0 adds artifact versioning. Latest files in `models/` are still kept for the dashboard, but frozen baselines are copied into timestamped `runs/` directories with a manifest, SHA-256 hashes, git source ref, and run-index entry. The current frozen baseline is `runs/20260522_phase14b_baseline/`.

Dense contrastive regime inference uses stride 1 after the encoder window warmup, so the learned-regime method is compared on the same BTC+ETH row universe as HMM-style, KMeans, and volatility-bucket baselines.

## Latest Benchmark Snapshot

Latest run: BTCUSDT + ETHUSDT, `tb_label_8h`, 25,920 out-of-sample rows per method.

| Method | IC | Sharpe | Drawdown | Note |
|---|---:|---:|---:|---|
| global_lgbm | 0.0024 | -0.506 | -0.688 | no-regime baseline |
| regime_lgbm_contrastive | -0.0165 | -0.902 | -0.909 | learned-regime benchmark |
| regime_lgbm_contrastive_hmm | 0.0035 | -0.382 | -0.852 | learned embeddings with HMM state dynamics |
| regime_lgbm_hmm | 0.0051 | -0.229 | -0.825 | best IC and Sharpe; true hmmlearn HMM |
| regime_lgbm_kmeans | -0.0013 | -1.180 | -0.920 | classical clustering baseline |
| regime_lgbm_vol_bucket | -0.0030 | -0.988 | -0.896 | simple volatility baseline |

The current result is intentionally presented as research evidence, not a profitable trading claim. The true HMM baseline is strongest in this run, while the contrastive-HMM hybrid shows that adding temporal state dynamics substantially improves the learned-regime approach.

The main interpretation is that dense contrastive inference fixed the earlier coverage problem, but unconstrained GMM clustering over high-density embeddings appears misaligned with the alpha target. HMM smoothing on embeddings repairs much of that weakness, but raw-feature HMM still produces the best downstream IC and Sharpe. This suggests useful temporal state structure matters more than embedding capacity or persistence alone.

## Fold-Local Regime Benchmark

Phase 13 reruns the regime-aware benchmark with regime assignment models refit inside each fold. This is stricter than the offline/global regime benchmark above.

| Method | IC | Sharpe | Drawdown | Note |
|---|---:|---:|---:|---|
| global_lgbm | 0.0024 | -0.506 | -0.688 | no-regime baseline |
| regime_lgbm_contrastive | -0.0110 | -0.834 | -0.926 | fold-local GMM on frozen embeddings |
| regime_lgbm_contrastive_hmm | -0.0026 | -0.548 | -0.778 | fold-local HMM on frozen embeddings |
| regime_lgbm_hmm | 0.0051 | -0.340 | -0.710 | best fold-local Sharpe |
| regime_lgbm_kmeans | 0.0072 | -0.728 | -0.860 | best fold-local IC |
| regime_lgbm_vol_bucket | -0.0020 | -0.820 | -0.854 | volatility threshold baseline |

The stricter benchmark changes the story in a useful way: fold-local KMeans has the strongest IC, while raw-feature HMM still has the strongest Sharpe and drawdown profile among regime methods. The learned contrastive-HMM path remains better than contrastive-GMM but is weaker under strict fold-local assignment than it looked in the offline/global regime benchmark.

## Phase 14A Robustness Matrix

Phase 14A repeats the strict fold-local benchmark across 9 grid cells: 3 symbol scopes by 3 target horizons.

| Scope | Target | Best IC Method | Best IC | Best Sharpe Method | Best Sharpe | Lowest Drawdown Method |
|---|---|---|---:|---|---:|---|
| BTCUSDT | tb_label_4h | regime_lgbm_hmm | 0.0016 | regime_lgbm_hmm | -0.993 | regime_lgbm_contrastive |
| BTCUSDT | tb_label_8h | regime_lgbm_kmeans | -0.0034 | regime_lgbm_contrastive | -0.546 | regime_lgbm_contrastive |
| BTCUSDT | tb_label_24h | regime_lgbm_kmeans | 0.0175 | regime_lgbm_contrastive_hmm | -0.211 | regime_lgbm_kmeans |
| ETHUSDT | tb_label_4h | regime_lgbm_vol_bucket | 0.0208 | regime_lgbm_vol_bucket | 0.315 | regime_lgbm_vol_bucket |
| ETHUSDT | tb_label_8h | global_lgbm | 0.0095 | regime_lgbm_contrastive | -0.201 | regime_lgbm_hmm |
| ETHUSDT | tb_label_24h | global_lgbm | 0.0348 | global_lgbm | 0.354 | regime_lgbm_vol_bucket |
| BTCUSDT+ETHUSDT | tb_label_4h | regime_lgbm_vol_bucket | 0.0103 | regime_lgbm_hmm | -0.205 | regime_lgbm_hmm |
| BTCUSDT+ETHUSDT | tb_label_8h | regime_lgbm_kmeans | 0.0072 | regime_lgbm_hmm | -0.340 | global_lgbm |
| BTCUSDT+ETHUSDT | tb_label_24h | regime_lgbm_contrastive_hmm | 0.0311 | regime_lgbm_vol_bucket | 0.321 | regime_lgbm_vol_bucket |

Across the grid, KMeans wins IC most often, HMM wins Sharpe most often, and volatility buckets most often win drawdown. This is stronger research evidence than a single headline result: it shows regime awareness can help, but the best regime method depends on the asset, horizon, and metric being optimized.

## Phase 14B Stress Robustness

Phase 14B reuses the strict fold-local `tb_label_8h` prediction file and stresses the trading layer across:

| Dimension | Values |
|---|---|
| Signal threshold | 0.03, 0.05, 0.07, 0.10 |
| Transaction cost | 5 bps, 10 bps, 20 bps |
| Market period | all, bull, sideways, bear |

That creates 48 stress cells and 288 method/cell rows. The goal is not to find a new best backtest; it is to see whether the existing conclusion breaks when practical assumptions change.

| Metric | Most Frequent Winner | Wins |
|---|---|---:|
| Signal IC | regime_lgbm_hmm | 24 |
| Sharpe | regime_lgbm_hmm | 22 |
| Drawdown | global_lgbm | 24 |
| Total return | regime_lgbm_hmm | 18 |

The stress test strengthens the interpretation around HMM regimes: raw-feature HMM is the most robust winner for signal IC, Sharpe, and total return across practical cost/threshold/market-period settings. The global model is the most defensive drawdown winner, especially when transaction costs rise. Contrastive-HMM remains useful in sideways regimes, but it is not yet the dominant learned-regime method.

## Current Status

The codebase now produces offline/global and fold-local regime benchmarks, a validation audit, Phase 14A symbol/horizon robustness, Phase 14B cost/threshold/period stress robustness, a frozen baseline run registry, and a Streamlit research dashboard. The next important work is not live trading; it is statistical confidence intervals, regime-quality metrics, and testing whether better learned embeddings can close the remaining gap against simple fold-local baselines.

