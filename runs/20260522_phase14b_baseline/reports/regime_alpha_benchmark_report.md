# Regime Alpha Benchmark Research Note

## Problem Statement

Financial markets are non-stationary: a signal that works in one market state can fail in another. Regime Alpha Benchmark tests whether learned market regimes improve alpha modeling compared with a global no-regime model and classical regime baselines under financial labels, purged walk-forward validation, and transaction costs.

The central question is:

> Do learned regimes improve IC, drawdown, Sharpe, or turnover versus global and classical-regime baselines?

## Data And Features

The current benchmark uses hourly Binance OHLCV data for BTCUSDT and ETHUSDT. Each symbol has 17,520 OHLCV bars and 17,460 feature rows after indicator warmup. The feature store contains 22 engineered technical and microstructure-inspired features, including multi-horizon returns, realized volatility, volatility of volatility, Amihud illiquidity, volume z-score, return autocorrelation, spread proxy, order-flow proxy, RSI, Garman-Klass volatility, ATR, close-vs-VWAP, log volume trend, and return dispersion.

## Target Labeling

The primary modeling target is `tb_label_8h`, an 8-hour triple-barrier label with classes `-1`, `0`, and `+1`. The neutral class captures periods where neither the profit barrier nor the stop barrier is hit before time expiry. Direction and volatility-adjusted labels are also generated at 4-hour, 8-hour, and 24-hour horizons.

The final target table contains 34,872 rows across BTCUSDT and ETHUSDT. For the primary 8-hour triple-barrier target, the neutral class is about 58.3% for both symbols, with up/down classes each near 20.6-21.0%.

## Regime Methods

The benchmark compares five regime methods on a common BTC+ETH universe:

| Method | Implementation | Rows | Silhouette | Avg Duration |
|---|---|---:|---:|---:|
| contrastive | contrastive encoder + GMM | 34,754 | 0.0959 | 30.51 |
| contrastive_hmm | contrastive embeddings + HMM | 34,754 | 0.1016 | 42.18 |
| hmm | hmmlearn Gaussian HMM | 34,754 | 0.0790 | 6.98 |
| kmeans | sklearn KMeans | 34,754 | 0.0967 | 4.05 |
| vol_bucket | realized-volatility quantiles | 34,754 | -0.0393 | 10.44 |

The contrastive method now uses dense stride-1 inference for every valid feature row after the 60-bar encoder window. This fixes the earlier sparse-coverage issue and makes downstream alpha comparisons fair.

Phase 11 adds a contrastive-HMM hybrid. Instead of clustering learned embeddings with GMM only, it fits a Gaussian HMM directly on the learned contrastive embedding sequence. This tests whether the weakness in contrastive regimes comes from representation learning itself or from the lack of temporal state dynamics in the assignment layer.

## Regime Stability Diagnostics

Phase 10 adds explicit stability diagnostics to separate regime persistence from downstream usefulness. This matters because a regime method can look visually smooth while still being weak for alpha conditioning.

| Method | Switches / 1k Bars | Avg Duration | Transition Diagonal | Stable IC | Transition IC |
|---|---:|---:|---:|---:|---:|
| contrastive | 32.72 | 30.51 | 0.967 | -0.0200 | 0.0482 |
| contrastive_hmm | 23.65 | 42.18 | 0.976 | 0.0033 | 0.0100 |
| hmm | 143.24 | 6.98 | 0.857 | 0.0084 | -0.0061 |
| kmeans | 246.60 | 4.05 | 0.753 | 0.0083 | -0.0137 |
| vol_bucket | 95.71 | 10.44 | 0.904 | 0.0001 | -0.0168 |

The important finding is not simply that more stable regimes are better. Contrastive-HMM produces the longest-lived regimes and repairs the negative stable-period IC seen in contrastive-GMM. However, the raw-feature HMM still has the strongest stable-period IC. This suggests that HMM-style temporal structure helps learned embeddings, but the current embedding space is still not as alpha-aligned as the simpler raw-feature HMM state space.

## Validation Setup

Alpha models use expanding walk-forward validation with:

- 6-month initial training window
- 1-month test step
- 5-day embargo between train and test
- multiclass LightGBM over labels `-1`, `0`, `+1`
- transaction cost of 10 bps per trade

The global baseline trains one LightGBM model across both symbols. Regime-aware models train separate LightGBM classifiers per regime and combine predictions through posterior weights where available. The alpha score is `P(+1) - P(-1)`, and trades are taken only when the neutral class is not dominant and the score clears the threshold.

## Validation Audit

Phase 12 adds an explicit validation audit so the benchmark can be evaluated as research evidence rather than a collection of backtest claims. The audit checks database tables, feature/target schema, finite joined rows, target horizon tail loss, common benchmark coverage, fold separation, embargo spacing, label-horizon purging, row-level prediction alignment, duplicate predictions, and consistency between `alpha_oos_predictions.csv` and `experiment_results.csv`.

The audit result is:

| Status | Count | Interpretation |
|---|---:|---|
| PASS | 21 | All critical data, fold, target, coverage, prediction-alignment, fold-local artifact, robustness artifact, stress-grid, and run-registry checks passed |
| WARN | 1 | Legacy `regime_assignments.csv` is an offline/global artifact |
| FAIL | 0 | No critical validation failure was detected |

The most important positive result is that all 18 folds satisfy row separation, the 120-bar embargo, and the 8-bar primary label-horizon purge. All six alpha methods also have equal out-of-sample prediction coverage of 25,920 rows. The audit also confirms that the Phase 14A robustness matrix contains all 54 expected method/cell rows across 9 grid cells, that the Phase 14B stress matrix contains all 288 expected method/cell rows across 48 stress cells, and that the frozen run registry points to a complete archived baseline.

The warning is methodological rather than a code failure: the legacy `regime_assignments.csv` file is generated as an offline/global artifact before alpha-model validation. This is acceptable for descriptive regime analysis and exploratory benchmarking. Phase 13 addresses the predictive version of this concern by adding a separate fold-local regime refit benchmark.

## Reproducibility Snapshot

Phase 15.0 adds artifact versioning before any statistical testing or encoder changes. The current frozen baseline is:

| Field | Value |
|---|---|
| Run id | `20260522_phase14b_baseline` |
| Source tag | `v1.3-phase14b` |
| Source commit | `f0902b21057d3dc464ce9e31d6be718a70531c63` |
| Manifest | `runs/20260522_phase14b_baseline/manifest.json` |
| Curated artifacts | 39 |
| Missing artifacts | 0 |

The `models/` directory remains the latest-artifact location used by the dashboard. The `runs/` directory stores immutable research snapshots with SHA-256 hashes, artifact manifests, git source references, and dependency files. Future statistical tests and encoder ablations should cite a `run_id` rather than relying on mutable latest CSV files.

## Fold-Local Regime Refit Benchmark

Phase 13 adds a stricter predictive benchmark. For each walk-forward fold, regime assignment models are refit using training-history rows only and are then applied to the test window. HMM-based methods use an online filtering pass for test assignments, initialized from the training sequence and advanced through the embargo gap. This avoids using a single full-sample regime assignment file for predictive alpha claims.

The contrastive methods still use frozen embeddings from the existing contrastive encoder, so this phase does not yet claim fully fold-local encoder training. The assignment layer on top of those embeddings, however, is fit fold-locally.

| Method | IC | Accuracy | Balanced Accuracy | Sharpe | Drawdown | Turnover | Total Return |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_lgbm | 0.0024 | 0.5736 | 0.3625 | -0.506 | -0.688 | 0.050 | -0.557 |
| regime_lgbm_contrastive | -0.0110 | 0.5623 | 0.3708 | -0.834 | -0.926 | 0.074 | -0.823 |
| regime_lgbm_contrastive_hmm | -0.0026 | 0.5623 | 0.3730 | -0.548 | -0.778 | 0.077 | -0.685 |
| regime_lgbm_hmm | 0.0051 | 0.5623 | 0.3698 | -0.340 | -0.710 | 0.079 | -0.536 |
| regime_lgbm_kmeans | 0.0072 | 0.5672 | 0.3704 | -0.728 | -0.860 | 0.081 | -0.797 |
| regime_lgbm_vol_bucket | -0.0020 | 0.5570 | 0.3679 | -0.820 | -0.854 | 0.083 | -0.820 |

The stricter benchmark changes the interpretation. Fold-local KMeans has the strongest IC, while raw-feature HMM has the strongest Sharpe and drawdown profile among regime-aware methods. Contrastive-HMM remains better than contrastive-GMM, but it no longer looks as strong as it did in the offline/global assignment benchmark. This is important evidence: some of the apparent offline benefit of learned regimes weakens when the assignment layer is refit inside each fold.

## Phase 14A Robustness Matrix

Phase 14A tests whether the fold-local result is stable across assets and horizons. The same benchmark is repeated across BTC-only, ETH-only, and BTC+ETH scopes at 4-hour, 8-hour, and 24-hour triple-barrier horizons. Regime assignment models are still refit inside each fold; the matrix changes the target horizon and symbol scope, not the validation discipline.

| Scope | Target | Best IC Method | Best IC | Best Sharpe Method | Best Sharpe | Lowest Drawdown Method | Lowest Drawdown |
|---|---|---|---:|---|---:|---|---:|
| BTCUSDT | tb_label_4h | regime_lgbm_hmm | 0.0016 | regime_lgbm_hmm | -0.993 | regime_lgbm_contrastive | -0.755 |
| BTCUSDT | tb_label_8h | regime_lgbm_kmeans | -0.0034 | regime_lgbm_contrastive | -0.546 | regime_lgbm_contrastive | -0.812 |
| BTCUSDT | tb_label_24h | regime_lgbm_kmeans | 0.0175 | regime_lgbm_contrastive_hmm | -0.211 | regime_lgbm_kmeans | -0.990 |
| ETHUSDT | tb_label_4h | regime_lgbm_vol_bucket | 0.0208 | regime_lgbm_vol_bucket | 0.315 | regime_lgbm_vol_bucket | -0.449 |
| ETHUSDT | tb_label_8h | global_lgbm | 0.0095 | regime_lgbm_contrastive | -0.201 | regime_lgbm_hmm | -0.874 |
| ETHUSDT | tb_label_24h | global_lgbm | 0.0348 | global_lgbm | 0.354 | regime_lgbm_vol_bucket | -0.987 |
| BTCUSDT+ETHUSDT | tb_label_4h | regime_lgbm_vol_bucket | 0.0103 | regime_lgbm_hmm | -0.205 | regime_lgbm_hmm | -0.436 |
| BTCUSDT+ETHUSDT | tb_label_8h | regime_lgbm_kmeans | 0.0072 | regime_lgbm_hmm | -0.340 | global_lgbm | -0.688 |
| BTCUSDT+ETHUSDT | tb_label_24h | regime_lgbm_contrastive_hmm | 0.0311 | regime_lgbm_vol_bucket | 0.321 | regime_lgbm_vol_bucket | -0.915 |

The robustness matrix weakens any simple claim that one regime method is universally best. KMeans wins IC most often, HMM wins Sharpe most often, and volatility buckets win drawdown most often. The contrastive-HMM hybrid wins the BTC+ETH 24-hour IC cell, which is useful but not enough to claim learned regimes dominate globally.

The research conclusion becomes more precise: regime conditioning can help, but the choice of regime model is metric-, horizon-, and asset-dependent. This is a stronger and more honest result than a single headline Sharpe because it shows where each method is fragile.

## Phase 14B Stress Robustness

Phase 14B tests whether the fold-local `tb_label_8h` conclusion survives practical trading assumptions. Instead of retraining models, it re-scores the same out-of-sample prediction file across transaction costs, signal thresholds, and market-period slices.

| Stress Dimension | Values |
|---|---|
| Signal threshold | 0.03, 0.05, 0.07, 0.10 |
| Transaction cost | 5 bps, 10 bps, 20 bps |
| Market period | all, bull, sideways, bear |

Bull, sideways, and bear periods are defined from rolling 30-day returns in the feature store. This creates 48 stress cells and 288 method/cell rows.

| Metric | Most Frequent Winner | Wins |
|---|---|---:|
| Signal IC | regime_lgbm_hmm | 24 |
| Sharpe | regime_lgbm_hmm | 22 |
| Drawdown | global_lgbm | 24 |
| Total return | regime_lgbm_hmm | 18 |

The stress grid strengthens the HMM interpretation. Raw-feature HMM wins the most signal-IC, Sharpe, and total-return cells across cost, threshold, and market-period settings. The global model is the most defensive drawdown winner, which is expected because fewer regime-conditioned switches can reduce downside under higher costs. Contrastive-HMM remains useful in sideways regimes, but it is not the dominant learned-regime method.

The important result is not that a strategy is profitable. The important result is that the relative conclusion is stress-tested: HMM-style temporal state structure is the most robust regime-aware layer in the current implementation, while learned embeddings still need a stronger objective or stricter fold-local representation training to dominate.

## Model Comparison

| Method | IC | Accuracy | Balanced Accuracy | Sharpe | Drawdown | Turnover | Total Return |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_lgbm | 0.0024 | 0.5736 | 0.3625 | -0.506 | -0.688 | 0.050 | -0.557 |
| regime_lgbm_contrastive | -0.0165 | 0.5616 | 0.3724 | -0.902 | -0.909 | 0.077 | -0.855 |
| regime_lgbm_contrastive_hmm | 0.0035 | 0.5585 | 0.3720 | -0.382 | -0.852 | 0.079 | -0.588 |
| regime_lgbm_hmm | 0.0051 | 0.5622 | 0.3727 | -0.229 | -0.825 | 0.081 | -0.446 |
| regime_lgbm_kmeans | -0.0013 | 0.5647 | 0.3670 | -1.180 | -0.920 | 0.079 | -0.912 |
| regime_lgbm_vol_bucket | -0.0030 | 0.5566 | 0.3670 | -0.988 | -0.896 | 0.083 | -0.872 |

All methods are evaluated on 25,920 out-of-sample rows. This equal test coverage is important: it prevents a regime method from looking better simply because it was tested on a smaller or easier subset.

## Results Interpretation

The real Gaussian HMM baseline produces the strongest IC in this run, improving from 0.0024 for the global baseline to 0.0051. It also has the least negative Sharpe among the tested methods, although drawdown remains large and total return remains negative after transaction costs.

The honest conclusion is mixed but stronger than before. The contrastive-HMM hybrid improves the learned-regime path substantially: IC moves from -0.0165 to 0.0035, and Sharpe improves from -0.902 to -0.382. That validates the Phase 11 hypothesis that temporal state dynamics help learned embeddings. However, the raw-feature HMM still beats the hybrid on IC and Sharpe, so the current learned representation is not yet superior to the simpler classical state model.

This is still a useful research result because it separates representation learning, regime diagnostics, and deployable alpha instead of overclaiming profitability. The project now has a clear scientific progression: dense contrastive regimes underperform, stability diagnostics identify the assignment-layer weakness, and contrastive-HMM partially fixes it.

## Limitations

- Hourly OHLCV is a noisy signal source.
- The current contrastive encoder is not a true Temporal Fusion Transformer.
- The contrastive encoder is still trained as an offline/frozen representation; a future paper-grade upgrade is fold-local encoder retraining.
- Offline/global regime results and fold-local regime results should be interpreted separately.
- HMM states are not ground truth market regimes. They are a classical proxy/reference state sequence used for comparison and weak supervision.
- Phase 14B stress testing re-scores existing predictions; it does not retrain models under each cost or threshold assumption.
- Backtest returns are research diagnostics, not deployable trading evidence.
- The project intentionally excludes live trading, RL, online retraining, and order-book data in this phase.

## Next Steps

1. Add block-bootstrap confidence intervals for IC and paired method differences.
2. Add feature importance and SHAP summaries for global and regime-aware LightGBM models.
3. Improve the learned encoder objective and retest the contrastive-HMM hybrid.
4. Add fold-local or expanding-window encoder retraining for the learned-regime methods.
5. Add multiple-testing controls before making any strong performance claim.
