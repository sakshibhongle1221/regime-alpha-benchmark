# Regime Alpha Benchmark Research Note

## Problem Statement

Financial markets are non-stationary: a signal that works in one market state can fail in another. Regime Alpha Benchmark tests whether learned market regimes improve alpha modeling compared with a global no-regime model and classical regime baselines under financial labels, purged walk-forward validation, and transaction costs.

The central question is:

> Do learned regimes improve IC, drawdown, Sharpe, or turnover versus global and classical-regime baselines?

## Related Work And Contribution Positioning

Phase 19A maps the project against four literature clusters:

1. Contrastive time-series representation learning: TS2Vec, TNC, CoST, TS-TCC, and TF-C show that unlabeled time-series windows can be embedded through carefully designed temporal, contextual, and frequency-aware objectives.
2. Classical financial regime switching: Hamilton-style Markov switching, Gaussian HMMs, and Markov-switching GARCH motivate sequential latent-state baselines with explicit transition structure.
3. Financial ML validation: triple-barrier labeling, purging, embargoing, fold-level tests, and multiple-testing controls define the evidence standard for financial prediction.
4. Regime-conditioned alpha modeling: predictors can be trained or weighted differently across market states, but the comparison must use the same labels, folds, costs, and test universe.

The paper contribution is therefore not a claim that HMM states are true regimes or that the project is a profitable trading system. The contribution is a controlled benchmark testing whether learned regime embeddings can beat, match, or explain classical sequential regimes under financial validation. Phase 18 then becomes a natural model-side response: use HMM states as weak supervision to improve contrastive representation learning, while still treating the HMM as a proxy rather than ground truth.

The narrative related-work note is `reports/related_work.md`; the compact source matrix is `reports/literature_matrix.csv`.

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

Phase 16 adds a separate structural regime-quality layer. It measures whether regimes are balanced, persistent, confident, and mutually consistent before evaluating whether they improve downstream alpha. The raw-feature HMM sequence is used as a classical reference proxy, not as ground truth.

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
| PASS | 33 | All critical data, fold, target, coverage, prediction-alignment, Phase 20 fold-local artifact, robustness artifact, stress-grid, statistical-test artifact, regime-quality artifact, compute-plan artifact, guided-encoder full-run artifact, Phase 22A time-frequency artifact, Phase 23 interpretability artifact, Phase 24 protocol artifact, Phase 25 ablation artifact, Phase 26 paper-statistical artifact, Phase 27/29 paper-draft artifact, Phase 28 reproducibility artifact, literature-positioning artifact, and run-registry checks passed |
| WARN | 1 | Legacy `regime_assignments.csv` is an offline/global artifact |
| FAIL | 0 | No critical validation failure was detected |

The most important positive result is that all 18 folds satisfy row separation, the 120-bar embargo, and the 8-bar primary label-horizon purge. The legacy offline alpha artifact still has equal coverage across six methods, while the Phase 20 fold-local artifact has equal coverage across eight methods, including the two guided-regime methods, with 25,920 rows each. The audit also confirms that the Phase 21 refreshed robustness matrix contains all 72 expected method/cell rows across 9 grid cells, that the Phase 21 refreshed stress matrix contains all 384 expected method/cell rows across 48 stress cells, that the Phase 15A/15B statistical artifacts are complete, that the Phase 16 regime-quality artifacts are complete, that the Phase 17 compute-plan artifacts are complete, that the Phase 19B guided-encoder full-run artifacts are complete, that the Phase 22A time-frequency encoder artifacts are complete, that the Phase 23 interpretability artifacts are complete, that the Phase 24 paper-protocol artifacts are complete, that the Phase 25 ablation artifacts are complete, that the Phase 26 paper-statistical artifacts are complete, that the Phase 27 paper-draft artifacts are complete, that the Phase 28 reproducibility artifacts are complete, that the Phase 19A literature-positioning artifacts are complete, and that the frozen run registry points to a complete archived baseline.

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

Phase 20 extends this same strict benchmark with two HMM-guided learned-regime methods. The guided embeddings are frozen from the Phase 19B encoder run, but the GMM/HMM assignment layer on top of those embeddings is fit fold-locally.

| Method | IC | Accuracy | Balanced Accuracy | Sharpe | Drawdown | Turnover | Total Return |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_lgbm | 0.0024 | 0.5736 | 0.3625 | -0.506 | -0.688 | 0.050 | -0.557 |
| regime_lgbm_contrastive | -0.0110 | 0.5623 | 0.3708 | -0.834 | -0.926 | 0.074 | -0.823 |
| regime_lgbm_contrastive_hmm | -0.0026 | 0.5623 | 0.3730 | -0.548 | -0.778 | 0.077 | -0.685 |
| regime_lgbm_hmm | 0.0051 | 0.5623 | 0.3698 | -0.340 | -0.710 | 0.079 | -0.536 |
| regime_lgbm_kmeans | 0.0072 | 0.5672 | 0.3704 | -0.728 | -0.860 | 0.081 | -0.797 |
| regime_lgbm_vol_bucket | -0.0020 | 0.5570 | 0.3679 | -0.820 | -0.854 | 0.083 | -0.820 |
| regime_lgbm_hmm_guided_gmm | -0.0092 | 0.5600 | 0.3669 | -0.976 | -0.900 | 0.079 | -0.854 |
| regime_lgbm_hmm_guided_hmm | 0.0094 | 0.5623 | 0.3743 | 0.099 | -0.614 | 0.084 | 0.031 |

The Phase 20 benchmark changes the interpretation. The original fold-local learned-regime path remains weak, and guided embeddings with GMM also fail. However, guided embeddings with an HMM assignment layer become the strongest point-estimate method: they beat raw-feature HMM by `+0.0043` IC, `+0.439` Sharpe, `+0.095` drawdown, and `+0.567` total return. This supports a precise mechanism: HMM-guided representation learning helps only when the downstream assignment layer also enforces sequential state dynamics.

## Phase 14A Robustness Matrix

Phase 14A tests whether the fold-local result is stable across assets and horizons. The same benchmark is repeated across BTC-only, ETH-only, and BTC+ETH scopes at 4-hour, 8-hour, and 24-hour triple-barrier horizons. Regime assignment models are still refit inside each fold; the matrix changes the target horizon and symbol scope, not the validation discipline.

| Scope | Target | Best IC Method | Best IC | Best Sharpe Method | Best Sharpe | Lowest Drawdown Method | Lowest Drawdown |
|---|---|---|---:|---|---:|---|---:|
| BTCUSDT | tb_label_4h | regime_lgbm_hmm_guided_hmm | 0.0058 | regime_lgbm_hmm_guided_hmm | -0.837 | regime_lgbm_hmm_guided_hmm | -0.702 |
| BTCUSDT | tb_label_8h | regime_lgbm_hmm_guided_gmm | 0.0004 | regime_lgbm_contrastive | -0.546 | regime_lgbm_contrastive | -0.812 |
| BTCUSDT | tb_label_24h | regime_lgbm_kmeans | 0.0175 | regime_lgbm_contrastive_hmm | -0.211 | regime_lgbm_kmeans | -0.990 |
| ETHUSDT | tb_label_4h | regime_lgbm_vol_bucket | 0.0208 | regime_lgbm_vol_bucket | 0.315 | regime_lgbm_vol_bucket | -0.449 |
| ETHUSDT | tb_label_8h | global_lgbm | 0.0095 | regime_lgbm_contrastive | -0.201 | regime_lgbm_hmm | -0.874 |
| ETHUSDT | tb_label_24h | global_lgbm | 0.0348 | global_lgbm | 0.354 | regime_lgbm_vol_bucket | -0.987 |
| BTCUSDT+ETHUSDT | tb_label_4h | regime_lgbm_hmm_guided_hmm | 0.0106 | regime_lgbm_hmm | -0.205 | regime_lgbm_hmm | -0.436 |
| BTCUSDT+ETHUSDT | tb_label_8h | regime_lgbm_hmm_guided_hmm | 0.0094 | regime_lgbm_hmm_guided_hmm | 0.099 | regime_lgbm_hmm_guided_hmm | -0.614 |
| BTCUSDT+ETHUSDT | tb_label_24h | regime_lgbm_contrastive_hmm | 0.0311 | regime_lgbm_vol_bucket | 0.321 | regime_lgbm_vol_bucket | -0.915 |

Phase 21 refreshes this matrix with both guided methods included, expanding the robustness artifact from 54 to 72 method/cell rows. `regime_lgbm_hmm_guided_hmm` now wins the most IC cells (3 of 9), including BTC 4h, BTC+ETH 4h, and the primary BTC+ETH 8h setting. It also wins 2 Sharpe cells and 2 drawdown cells. `regime_lgbm_hmm_guided_gmm` wins only one IC cell, so the useful guided result remains specific to guided embeddings plus sequential HMM assignment.

The research conclusion becomes more precise: guided-HMM is now the strongest learned-regime candidate, but not a universal winner. ETH-only cells still favor global or volatility-bucket models, and BTC 24h still favors KMeans on IC. This is stronger and more honest evidence than a single headline Sharpe because it shows exactly where the new method is robust and where it remains fragile.

## Phase 14B Stress Robustness

Phase 14B tests whether the fold-local `tb_label_8h` conclusion survives practical trading assumptions. Instead of retraining models, it re-scores the same out-of-sample prediction file across transaction costs, signal thresholds, and market-period slices.

| Stress Dimension | Values |
|---|---|
| Signal threshold | 0.03, 0.05, 0.07, 0.10 |
| Transaction cost | 5 bps, 10 bps, 20 bps |
| Market period | all, bull, sideways, bear |

Bull, sideways, and bear periods are defined from rolling 30-day returns in the feature store. Phase 21 refreshes this grid with the guided methods included, creating 48 stress cells and 384 method/cell rows.

| Metric | Most Frequent Winner | Wins |
|---|---|---:|
| Signal IC | regime_lgbm_hmm_guided_hmm | 30 |
| Sharpe | regime_lgbm_hmm_guided_hmm | 36 |
| Drawdown | regime_lgbm_hmm_guided_hmm | 28 |
| Total return | regime_lgbm_hmm_guided_hmm | 34 |

The refreshed stress grid is the strongest support for the guided-HMM path. Under practical cost, threshold, and market-period perturbations, `regime_lgbm_hmm_guided_hmm` wins the most cells for signal IC, Sharpe, drawdown, and total return. This is a meaningful upgrade from Phase 14B: the method that looked best in the primary Phase 20 point estimate also survives the primary stress grid.

The important result is not that a strategy is profitable. The important result is that the relative conclusion is stress-tested: HMM-guided representation learning plus HMM assignment is now the most stress-robust regime-aware layer on the primary prediction file. The symbol/horizon robustness matrix remains mixed, so the broad claim should be "stress-robust on the primary BTC+ETH 8h benchmark" rather than "universally dominant across all assets and horizons."

## Phase 15A/15B Statistical Rigor

Phase 15A tests whether the fold-local method differences are statistically reliable. The primary unit for IC and Sharpe significance is the walk-forward fold, not individual rows, because adjacent hourly labels overlap and row-level samples are not independent. For calibration-oriented forecast quality, the phase also runs a Newey-West DM-style test on per-row multiclass negative log-likelihood.

Phase 15B then applies multiple-testing controls. It uses Benjamini-Hochberg false-discovery-rate correction and Holm family-wise correction so that a single attractive raw p-value is not treated as a publishable claim after many method/metric comparisons.

| Method | Mean Fold IC | 95% CI Low | 95% CI High | Positive IC Folds | Mean Fold Sharpe |
|---|---:|---:|---:|---:|---:|
| regime_lgbm_hmm_guided_hmm | 0.0080 | -0.0122 | 0.0278 | 9 | -0.026 |
| regime_lgbm_hmm | 0.0058 | -0.0135 | 0.0247 | 11 | -0.561 |
| regime_lgbm_kmeans | 0.0035 | -0.0202 | 0.0282 | 8 | -0.720 |
| regime_lgbm_vol_bucket | 0.0004 | -0.0230 | 0.0241 | 10 | -0.818 |
| global_lgbm | -0.0005 | -0.0207 | 0.0209 | 9 | -0.583 |
| regime_lgbm_contrastive_hmm | -0.0063 | -0.0305 | 0.0196 | 7 | -0.908 |
| regime_lgbm_hmm_guided_gmm | -0.0075 | -0.0336 | 0.0189 | 8 | -1.058 |
| regime_lgbm_contrastive | -0.0147 | -0.0373 | 0.0095 | 7 | -0.990 |

The confidence intervals are wide and mostly overlap zero. That means the project should not claim that the current guided-HMM edge is statistically decisive from IC alone. The strongest point-estimate result is positive: `regime_lgbm_hmm_guided_hmm` has the highest mean fold IC and the least negative mean fold Sharpe. Against raw-feature HMM, however, the IC improvement is small relative to fold noise (`mean difference = 0.0022`, paired fold `p = 0.801`). The strongest fold-level significant result remains a negative finding: contrastive-GMM is worse than raw-feature HMM on IC (`mean difference = -0.0205`, paired fold `p = 0.035`). After Phase 15B correction, this becomes `raw_only_suggestive` rather than a hard corrected claim. Most other IC and Sharpe differences are not significant at the 5% level.

The row-level DM-style negative-log-likelihood tests show that the global model is often better calibrated than the regime-conditioned models. This does not contradict the IC result; it separates directional/ranking usefulness from probability calibration. For a paper, this is valuable because it prevents overclaiming: the regime methods may improve some alpha diagnostics, but their probability estimates are not automatically better calibrated.

Probabilistic Sharpe Ratio diagnostics improve materially after Phase 20: `regime_lgbm_hmm_guided_hmm` has `PSR(SR > 0) = 0.633`, while raw-feature HMM is about `0.121`. This is useful support for the guided-HMM direction, but it remains a diagnostic on overlapping portfolio returns, not a deployable performance claim.

## Phase 16 Regime Quality

Phase 16 evaluates regime methods structurally before asking whether they help trading. This is important because a regime method can look smooth or produce a better backtest by chance without actually discovering meaningful state partitions.

The metrics include:

- regime balance entropy: whether one regime dominates all rows
- switch rate and transition diagonal probability: how persistent the state sequence is
- posterior confidence and entropy: how uncertain soft-assignment methods are
- pairwise NMI/ARI: how much two regime methods agree after ignoring arbitrary label names
- HMM-reference NMI/purity: agreement with the classical raw-feature HMM proxy

| Method | Balance Entropy | Switches / 1k Bars | Transition Diagonal | Avg Duration | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| contrastive | 0.999 | 32.72 | 0.967 | 30.51 | 0.032 | 0.379 |
| contrastive_hmm | 0.999 | 23.65 | 0.976 | 42.18 | 0.020 | 0.377 |
| hmm | 0.959 | 143.24 | 0.857 | 6.98 | 1.000 | 1.000 |
| kmeans | 0.866 | 246.60 | 0.753 | 4.05 | 0.182 | 0.459 |
| vol_bucket | 1.000 | 95.71 | 0.904 | 10.44 | 0.333 | 0.599 |

The main finding is that learned regimes are structurally smooth but not strongly aligned with the HMM reference. Contrastive-HMM has the longest average duration and highest transition diagonal, but its HMM-reference NMI is only 0.020. Volatility buckets have the strongest non-HMM agreement with the HMM reference (`NMI = 0.333`, purity `0.599`), even though the method is much simpler.

This supports the paper's core diagnostic interpretation: persistence alone is not enough. The learned encoder currently creates stable partitions, but those partitions are not yet aligned with the state structure that helps the downstream alpha benchmark. The next encoder phase should therefore change the representation-learning objective, not merely increase model size.

## Phase 17 Compute Plan

Phase 17 makes the encoder-upgrade roadmap compute-aware. Before launching HMM-guided losses, time-frequency views, or broader ablation grids, the project now records a local training-cost estimate and a capped experiment queue.

The current profile uses synthetic forward/backward timing for the existing `TemporalEncoder` on the local CPU-only environment.

| Metric | Value |
|---|---:|
| Encoder parameters | 139,408 |
| Training windows | 34,798 |
| Batches per epoch | 271 |
| Profiled step time | 0.739 seconds |
| Estimated epoch time | 3.34 minutes |
| Estimated 30-epoch retrain | 100.10 minutes |
| Estimated 12-run initial grid | 21.62 hours |
| Local budget | 24 hours |
| Budget status | green |

The initial ablation cap remains 12 runs: 3 losses by 2 augmentations by 2 assignment methods. The queue is now staged rather than wide-open:

| Priority | Loss | Augmentation | Assignment | Decision |
|---:|---|---|---|---|
| 1 | `hmm_guided` | `time_only` | `hmm` | complete |
| 2 | `hmm_guided` | `time_only` | `gmm` | complete |
| 3 | `hmm_guided` | `time_frequency` | `hmm` | 3-epoch prototype complete |

This matters for paper execution because it prevents an uncontrolled ablation explosion. The time-only guided runs are complete, and the first time-frequency prototype now exists as a capped structural test. The remaining grid should stay on hold until a full-length time-frequency run produces evidence worth downstream alpha compute.

## Phase 18 HMM-Guided Encoder

Phase 18 implements the first encoder-objective upgrade. The current contrastive encoder treats adjacent windows as positives, which can encourage local smoothness even when the market is crossing a regime boundary. The Phase 18 encoder instead uses the raw-feature HMM sequence as weak supervision:

- same HMM state and distant in time: positive pair
- different HMM state and nearby in the same symbol: hard negative pair
- HMM is treated as a proxy/reference sequence, not market-regime ground truth

The implementation writes separate guided artifacts and does not overwrite the existing canonical encoder or benchmark assignments. This keeps the experiment reversible and prevents Phase 18 from contaminating the Phase 15/16 baseline.

The first run is intentionally a one-epoch smoke test to validate the training and artifact path before spending compute on a 30-epoch run.

| Method | Epochs | Silhouette | Avg Duration | Transition Diagonal | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| `hmm_guided_gmm` | 1 | 0.341 | 15.17 | 0.934 | 0.387 | 0.652 |
| `hmm_guided_hmm` | 1 | 0.353 | 18.55 | 0.946 | 0.389 | 0.620 |

This is already directionally useful: the Phase 16 contrastive-GMM method had only `HMM NMI = 0.032`, while the one-epoch HMM-guided variant reaches about `0.39`. That does not prove alpha improvement, but it confirms that the guided objective changes the embedding geometry in the intended direction. The next step is a full 30-epoch guided run followed by fold-local alpha and statistical re-testing.

## Phase 19A Literature Positioning

Phase 19A adds a paper-facing related-work layer before expanding the encoder roadmap. This phase does not change model outputs. It changes the research framing so later experiments answer a precise question instead of becoming a collection of model variants.

The project is positioned against:

| Cluster | Key References | Why It Matters Here |
|---|---|---|
| Time-series contrastive learning | TS2Vec, TNC, CoST, TS-TCC, TF-C | Defines the representation-learning family that the encoder belongs to |
| Financial regime switching | Hamilton 1989, Gaussian HMMs, Markov-switching GARCH | Defines the classical sequential-state baseline the learned regimes must beat |
| Financial ML validation | triple-barrier labels, purging, embargoing, multiple-testing discipline | Defines what counts as credible evidence in a financial prediction benchmark |
| Regime-conditioned alpha | global versus state-conditioned predictors | Defines the downstream use case and evaluation surface |

This positioning clarifies the contribution statement: the project is an empirical bridge between classical sequential regime models and deep time-series representation learning. Its strongest current finding is not that deep regimes dominate, but that sequential state discipline matters and that HMM-guided weak supervision may repair part of the contrastive encoder's regime-structure weakness.

## Phase 19B Full Guided Encoder Run

Phase 19B runs the HMM-guided encoder for the planned 30 epochs. This converts the Phase 18 smoke test into a real structural representation experiment. The training loss falls steadily from `0.9284` at epoch 1 to `0.0949` at epoch 30, while the valid-anchor percentage remains `1.000`.

| Method | Epochs | Silhouette | Avg Duration | Transition Diagonal | Mean Confidence | HMM NMI | HMM ARI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hmm_guided_gmm` | 30 | 0.384 | 5.09 | 0.804 | 0.991 | 0.609 | 0.434 | 0.759 |
| `hmm_guided_hmm` | 30 | 0.629 | 5.72 | 0.825 | 0.998 | 0.869 | 0.887 | 0.957 |

The comparison artifact shows the structural jump clearly:

| Method | Source | HMM NMI | HMM Purity | NMI Delta vs Contrastive |
|---|---|---:|---:|---:|
| `contrastive` | Phase 16 | 0.032 | 0.379 | 0.000 |
| `hmm_guided_gmm` | Phase 19B | 0.609 | 0.759 | 0.577 |
| `hmm_guided_hmm` | Phase 19B | 0.869 | 0.957 | 0.837 |

This is the first strong model-side evidence that the weakly supervised objective is doing what it was designed to do. The learned representation path no longer merely creates smooth partitions; when paired with an HMM assignment layer, it becomes strongly aligned with the classical sequential reference. Phase 20 then tests whether that structural improvement survives contact with the downstream alpha benchmark.

## Phase 20 Guided Alpha Retest

Phase 20 adds the paper-critical downstream test for the Phase 19B encoder. The guided embeddings are kept frozen, but the regime assignment layer on top of them is refit inside each walk-forward fold. This avoids the invalid shortcut of using one full-sample guided assignment file for predictive claims.

Two learned-regime variants are evaluated:

1. `hmm_guided_gmm`: fold-local GMM assignment on HMM-guided embeddings.
2. `hmm_guided_hmm`: fold-local online HMM assignment on HMM-guided embeddings.

Both variants are evaluated on the same 25,920 out-of-sample rows as every other method.

## Model Comparison

| Method | IC | Accuracy | Balanced Accuracy | Sharpe | Drawdown | Turnover | Total Return |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_lgbm | 0.0024 | 0.5736 | 0.3625 | -0.506 | -0.688 | 0.050 | -0.557 |
| regime_lgbm_contrastive | -0.0110 | 0.5623 | 0.3708 | -0.834 | -0.926 | 0.074 | -0.823 |
| regime_lgbm_contrastive_hmm | -0.0026 | 0.5623 | 0.3730 | -0.548 | -0.778 | 0.077 | -0.685 |
| regime_lgbm_hmm | 0.0051 | 0.5623 | 0.3698 | -0.340 | -0.710 | 0.079 | -0.536 |
| regime_lgbm_kmeans | 0.0072 | 0.5672 | 0.3704 | -0.728 | -0.860 | 0.081 | -0.797 |
| regime_lgbm_vol_bucket | -0.0020 | 0.5570 | 0.3679 | -0.820 | -0.854 | 0.083 | -0.820 |
| regime_lgbm_hmm_guided_gmm | -0.0092 | 0.5600 | 0.3669 | -0.976 | -0.900 | 0.079 | -0.854 |
| regime_lgbm_hmm_guided_hmm | 0.0094 | 0.5623 | 0.3743 | 0.099 | -0.614 | 0.084 | 0.031 |

All methods are evaluated on 25,920 out-of-sample rows. This equal test coverage is important: it prevents a regime method from looking better simply because it was tested on a smaller or easier subset.

## Results Interpretation

Phase 20 changes the headline result. The strongest point-estimate method is now `regime_lgbm_hmm_guided_hmm`, with IC `0.0094`, Sharpe `0.099`, drawdown `-0.614`, and total return `0.031`. It beats the raw-feature HMM on the main point estimates while using the same walk-forward folds, costs, target, and test rows.

The mechanism is specific. Guided embeddings with GMM assignment still fail, while guided embeddings with HMM assignment become the best point-estimate method. That supports the main research thesis: representation learning helps only when the assignment layer preserves sequential state dynamics. Rich embeddings alone are not enough.

The statistical interpretation stays cautious. At the fold level, the guided-HMM IC advantage over raw-feature HMM is positive but not significant at the 5% level (`p = 0.801`). The guided-HMM method also has a stronger Probabilistic Sharpe read than raw HMM (`PSR(SR>0) = 0.633` versus about `0.121`), but calibration/NLL diagnostics still favor simpler references in some comparisons. The paper claim should therefore be phrased as a promising, controlled empirical finding rather than proof of a dominant trading strategy.

The project now has a clean scientific progression: dense contrastive regimes underperform, stability diagnostics identify the assignment-layer weakness, contrastive-HMM partly fixes temporal consistency, HMM-guided weak supervision repairs structural alignment, and Phase 20 shows that the guided-HMM path can become the strongest downstream point-estimate method under strict fold-local validation.

## Phase 22A Time-Frequency Guided Encoder

Phase 22A adds the first augmentation-side encoder test. The guided encoder still uses HMM states as weak supervision, but each 60-bar feature window now includes both the original time-domain features and six low-frequency FFT magnitude bands per feature. This increases the encoder input from 22 to 154 features.

The run is intentionally capped at 3 epochs. Its purpose is to test whether the time-frequency view is structurally promising enough to justify a full 30-epoch run and downstream alpha retest.

| Method | Epochs | Input Features | Silhouette | Avg Duration | Transition Diagonal | Mean Confidence | HMM NMI | HMM ARI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tf_hmm_guided_gmm` | 3 | 154 | 0.326 | 6.47 | 0.845 | 0.967 | 0.504 | 0.372 | 0.682 |
| `tf_hmm_guided_hmm` | 3 | 154 | 0.338 | 8.39 | 0.881 | 0.986 | 0.528 | 0.416 | 0.704 |

This is a useful but mixed result. Compared with the original vanilla contrastive path from Phase 16 (`HMM NMI = 0.032`, `HMM purity = 0.379`), the time-frequency guided encoder is clearly more aligned with the raw-feature HMM reference. Compared with the full 30-epoch time-only HMM-guided encoder (`HMM NMI = 0.869`, `HMM purity = 0.957`), it is still weaker.

The paper-safe interpretation is therefore cautious: frequency-domain information may be useful as an ablation, but the current 3-epoch prototype does not justify a downstream alpha claim. A full time-frequency run is only worth doing if the next compute budget allows it, and it should be evaluated against the frozen Phase 19B/20/21 guided-HMM baseline rather than against the old vanilla contrastive baseline.

## Phase 23 Fold-Local Interpretability

Phase 23 adds interpretability evidence for the alpha models. The goal is not to prove causality. The goal is to check whether the features driving the best guided-HMM downstream model are economically plausible and whether the regime-conditioned models are reacting to coherent market-state information.

The script `src/interpretability.py` trains explanation models inside the same expanding walk-forward folds used by the alpha benchmark. It aggregates LightGBM gain/split importance and capped SHAP summaries for the global LightGBM, raw-feature HMM regime LightGBM, and guided-HMM regime LightGBM. This avoids explaining a single full-sample model that has seen all dates.

Top guided-HMM regime-conditioned alpha features:

| Guided-HMM Regime | Top Features | Interpretation |
|---|---|---|
| 0 | `vol_20h`, `vol_of_vol`, `kurtosis`, `atr_14`, `ret_autocorr` | volatility persistence with distribution-shape sensitivity |
| 1 | `vol_20h`, `vol_of_vol`, `kurtosis`, `atr_14`, `ret_autocorr` | volatility persistence with distribution-shape sensitivity |
| 2 | `vol_20h`, `vol_of_vol`, `atr_14`, `kurtosis`, `ret_60h` | volatility state plus medium-horizon return context |
| 3 | `vol_20h`, `vol_of_vol`, `ret_autocorr`, `skewness`, `ret_60h` | volatility state plus autocorrelation and skew-sensitive momentum |

The feature-family summary is consistent with the regime narrative. Across guided-HMM regimes, volatility features explain roughly 35%-39% of SHAP share, momentum features roughly 28%-32%, and distribution-shape features roughly 11%-12%. Liquidity/volume and microstructure proxies contribute smaller but nonzero shares. This is paper-useful because it shows that the guided-HMM alpha model is not driven by arbitrary identifiers alone; its predictions are mostly tied to volatility state, volatility-of-volatility, return autocorrelation, and distribution shape.

The limitation is equally important: feature importance is diagnostic and model-specific, not causal evidence. These results support economic plausibility, but they do not prove that the regimes are true latent market states.

## Phase 24 Paper Protocol Freeze

Phase 24 freezes the paper protocol before additional ablations. This is a scope-control phase, not a model-performance phase. It defines what the paper asks, what evidence is already available, what claims are allowed, and what experiments remain in the queue.

The new protocol artifacts are:

| Artifact | Role |
|---|---|
| `reports/paper_protocol.md` | Freezes the research question, contribution boundary, dataset, features, targets, validation setup, methods, metrics, and decision gates |
| `reports/hypotheses.md` | Maps H1-H7 to current evidence and next tests |
| `reports/claim_registry.md` | Separates supported, directional, diagnostic, open, and forbidden claims |
| `reports/experiment_manifest.md` | Lists completed experiment families, future experiment queue, minimal ablation definition, and submission-readiness checklist |

The most important protocol decision is that the current strongest paper claim is directional rather than decisive: guided-HMM produces the best primary point estimates and strongest primary stress robustness, but the fold-level IC edge over raw-feature HMM is not significant at the 5% level. The next paper-critical experiment is therefore not broad feature expansion; it is a minimal ablation suite that tests whether the mechanism survives controlled model changes.

## Phase 25 Minimal Ablation Suite

Phase 25 runs the minimal paper-critical ablation suite from already completed artifacts. It is intentionally narrow: it tests the mechanism needed for the paper without launching an uncontrolled retraining grid.

| Ablation Family | Paper Question | Current Evidence |
|---|---|---|
| Objective guidance | Does HMM-guided contrastive supervision improve the learned regime path? | Supported for guided-HMM downstream alpha; mixed structurally because the diagnostic duration metric does not always move in one direction |
| Assignment layer | Does HMM assignment improve learned embeddings over GMM assignment? | Strongly supported for guided embeddings, the time-frequency prototype, and downstream alpha |
| Augmentation view | Does the 3-epoch time-frequency prototype improve over the full time-only guided encoder? | Not yet; it should not receive full downstream alpha compute without better structural evidence |
| Classical reference | Does guided-HMM outperform raw-feature HMM on the primary alpha benchmark? | Directionally supported on point estimates, but still requires a refreshed Phase 26 statistical test |

The main finding is that sequential assignment is the most reliable mechanism in the current project. Learned embeddings become most useful when their assignment layer respects temporal state persistence. The ablation suite also prevents overclaiming: the current time-frequency prototype is a useful negative result, not a new winner.

## Phase 26 Paper Statistical Evidence Refresh

Phase 26 maps the Phase 25 ablation rows onto fold-level paper claim tests. The goal is not to search for a new best result. The goal is to translate the mechanism table into paper-safe language with p-values, correction status, and allowed claims.

The main paper-facing results are:

| Claim Comparison | Status | Statistical Read |
|---|---|---|
| `guided_hmm_alpha_vs_guided_gmm_alpha` | raw-suggestive | HMM assignment improves the guided path on all focused metrics; IC `p=0.075`, not corrected significant |
| `guided_hmm_alpha_vs_raw_feature_hmm_alpha` | directionally supported | Guided-HMM still beats raw-feature HMM on point estimates, but fold-level IC remains inconclusive |
| `guided_hmm_alpha_vs_vanilla_contrastive_hmm_alpha` | directionally supported | HMM-guided supervision improves the vanilla contrastive-HMM path on focused point estimates |
| `time_frequency_guided_hmm_vs_time_only_guided_hmm` | do not expand yet | Current 3-epoch time-frequency prototype should not be treated as a downstream alpha result |

This phase strengthens the paper because it separates three levels of evidence: mechanism support, directional alpha support, and statistical support. The current result is strong enough to write as a disciplined benchmark finding, but not strong enough to claim statistical dominance over raw-feature HMM.

## Phase 27 Paper Skeleton

Phase 27 converts the evidence stack into a manuscript scaffold. It adds `src/paper_skeleton.py`, which generates:

| Artifact | Role |
|---|---|
| `paper/main.md` | Structured manuscript draft with abstract, related work, data, methods, validation, results, robustness, interpretability, ablations, limitations, and conclusion sections |
| `reports/paper_artifact_map.csv` | Maps each paper section to the artifact that supports it |
| `reports/paper_submission_checklist.md` | Tracks what is ready, what still needs human writing, optional experiments, and forbidden claims |

This phase is important because it changes the project from "many good artifacts" into a paper-shaped research package. It does not add a new model or a new performance claim. Instead, it makes the existing claim discipline visible in the paper itself: guided-HMM is presented as the strongest current point-estimate and stress-robust method, while statistical dominance over raw-feature HMM remains inconclusive.

## Phase 28 Reproducibility Package

Phase 28 adds the reviewer-facing reproduction layer. It includes:

| Artifact | Role |
|---|---|
| `reproduce.ps1` | PowerShell helper with smoke, full, and dashboard modes |
| `reports/environment.md` | Python, dependency, data, hardware, and deployment notes |
| `reports/artifact_manifest.md` | Separates committed curated artifacts from regenerated local and ignored heavy artifacts |
| `reports/reproduction_checklist.md` | Reviewer checklist for smoke/full/dashboard reproduction and git safety checks |

The most important distinction is between dashboard reproduction and full research reproduction. The dashboard runs from curated committed artifacts and minimal dependencies. Full research reproduction requires the local research environment, raw/local data access, and substantially more compute.

## Phase 29 Paper Prose Pass

Phase 29 turns the generated paper scaffold into stronger manuscript-style prose. The abstract, introduction, related-work positioning, data description, methods, validation protocol, results interpretation, robustness, interpretability, ablations, limitations, reproducibility, and conclusion are now written as a coherent paper draft rather than a checklist.

The phase keeps the same scientific conclusion: sequential HMM assignment is the strongest supported mechanism; HMM-guided learned regimes are promising and stress-robust on the BTC/ETH 8h benchmark; and the fold-level IC advantage over raw-feature HMM remains statistically inconclusive.

## Reviewer Defense Notes

Three likely reviewer objections are now handled directly in the paper draft:

| Reviewer concern | Paper response |
|---|---|
| Only BTC/ETH are tested | The paper frames BTC/ETH as a controlled crypto setting with relatively uniform trading microstructure; multi-asset generalization is future work |
| Fold-level tests have only 18 folds | The paper acknowledges low statistical power and argues that 18 embargoed walk-forward folds are more defensible than row-level independence assumptions on overlapping labels |
| Main guided-HMM vs raw HMM comparison has `p=0.801` | The paper leads with the mechanism: sequential assignment is the key driver, and HMM-guided representation learning makes that assignment more effective; alpha dominance remains directional |

## Phase 31 Multi-Asset Expansion Protocol

The next publication upgrade is a pre-specified multi-asset crypto universe, not an ad hoc expansion after seeing results. Phase 31 adds `configs/crypto_universe_candidates.csv`, `src/multiasset_universe.py`, and generated Crypto-20/Crypto-50 universe artifacts. After the combined Phase 32 data run, all 20 selected Crypto-20 assets are quality-eligible locally; the remaining Crypto-50 candidates stay pending until the larger expansion is explicitly run.

This protects the research narrative: BTC/ETH remains the controlled pilot, while Crypto-20 and Crypto-50 become pre-specified generalization tests. The multi-asset gate is split deliberately. The Phase 20 fold-level result blocks a downstream alpha generalization claim, but it does not block structural generalization diagnostics. In other words, a Crypto-20 guided encoder run can test whether the representation objective transfers structurally, while a separate fold-local Crypto-20 alpha retest is still required before claiming predictive improvement on the wider universe.

## Phase 32 Crypto-20 Data Pipeline And Quality Gate

Phase 32 combines the earlier data-readiness and ingestion-quality steps into one phase. `src/ingestion.py`, `src/features.py`, `src/targets.py`, and `src/check.py` can resolve symbols from `models/asset_universe_crypto20.csv` using `--universe crypto20`. `src/crypto20_quality_gate.py` then writes per-symbol OHLCV, feature, target, and gap checks to `models/crypto20_data_quality.csv` and summarizes whether the Crypto-20 universe is ready for multi-asset regime benchmarking. The current gate passes with 20/20 selected assets.

## Phase 33 Crypto-20 Classical Regime Benchmark

Phase 33 establishes the first multi-asset regime baseline before retraining learned encoders. The Crypto-20 regime benchmark summary covers 349,786 joined feature/target rows across 20 assets and compares raw-feature HMM, KMeans, and volatility buckets on the same universe. The current result preserves the mechanism story: KMeans has higher silhouette but more imbalanced and less persistent regimes, while HMM has lower silhouette but stronger sequential persistence across assets. This freezes the classical Crypto-20 baseline before the next learned-regime experiment.

## Phase 34 Crypto-20 Guided Encoder Readiness

Phase 34 checks whether the HMM-guided learned-regime path is ready to scale from the BTC/ETH pilot to Crypto-20. The readiness audit uses the Phase 33 raw-feature HMM states as weak-supervision labels and verifies that the expanded universe has enough eligible encoder windows, same-state positives, and in-trajectory hard negatives near regime boundaries. The current gate passes with 348,606 eligible HMM-labeled windows, all four regimes represented globally, 100% positive-anchor coverage, and 6,278,476 directed hard-negative pairs.

The compute estimate is heavy but feasible: using the Phase 17 measured CPU step time, a full 30-epoch Crypto-20 guided encoder run is estimated at about 16.77 hours with the current batch size, under the 24-hour planning cap. This does not claim the learned method has beaten the classical Crypto-20 HMM yet. It says the next experiment is justified, pre-gated, and not a blind compute spend.

## Phase 35 Crypto-20 Guided Encoder Training

Phase 35 executes the full learned-regime expansion beyond the BTC/ETH pilot. `src/guided_encoder.py` resolves `--universe crypto20`, reads `models/crypto20_regime_assignments.csv` as the HMM weak-supervision source, and writes separate `crypto20_guided_encoder_*` artifacts without overwriting the BTC/ETH guided encoder outputs. The helper script `scripts/phases/run_phase35_crypto20_guided.ps1` records the long training command and logs output under `.tmp/`.

The completed CPU run trains for 30 epochs on 348,606 eligible windows across 20 Crypto-20 symbols. The training loss falls from `0.3258` at epoch 1 to `0.0864` at epoch 30, while valid-anchor coverage stays at `1.000`. This indicates that the HMM-guided positive-pair and hard-negative mining setup remains usable at the multi-asset scale rather than collapsing into sparse or invalid pair selection.

The strongest assignment path is `hmm_guided_hmm`, not `hmm_guided_gmm`. It reaches silhouette `0.399`, transition diagonal probability `0.890`, HMM-reference NMI `0.694`, HMM-reference ARI `0.627`, and HMM-reference purity `0.814`. The GMM assignment on the same learned embeddings is weaker on the HMM-reference structure, with silhouette `0.230`, NMI `0.506`, ARI `0.384`, and purity `0.721`.

This is a structural generalization result, not a final alpha-performance claim. The Phase 35 evidence supports the mechanism that HMM-guided representation learning can scale from BTC/ETH to Crypto-20 and produce regime geometry that is more aligned with sequential state structure. The next required experiment is a fold-local Crypto-20 downstream alpha retest using these guided assignments against the frozen Phase 33 classical baseline.

## Phase 36 Crypto-20 Downstream Alpha Retest

Phase 36 is the predictive follow-up to the Phase 35 structural result. The protocol uses the same expanding walk-forward setup as the BTC/ETH paper benchmark: 6-month initial training, 1-month test step, 5-day embargo, fold-local regime refits, LightGBM alpha models, and transaction costs. The key difference is scope: the benchmark resolves the pre-specified Crypto-20 universe and writes prefixed artifacts so the original BTC/ETH results remain frozen.

The experiment compares a global LightGBM model, raw-feature HMM, raw-feature KMeans, volatility buckets, HMM-guided embedding GMM, and HMM-guided embedding HMM. Vanilla contrastive methods are skipped in the default Phase 36 run because dense vanilla Crypto-20 contrastive artifacts are not part of the Phase 35 structural expansion. This keeps the claim narrow and reviewer-safe: Phase 36 tests whether HMM-guided learned regimes improve downstream Crypto-20 alpha against classical and global references.

The completed run produces equal OOS coverage across all methods (`230,400` rows each). The strongest guided method, `regime_lgbm_hmm_guided_hmm`, reaches IC `0.0226`, beating the global model (`0.0175`) and raw-feature HMM (`0.0214`) on predictive correlation. However, raw-feature HMM still has stronger Sharpe (`0.1479` versus `-0.0573`) and total return (`0.1386` versus `-0.1257`). The paper interpretation is therefore deliberately mixed: HMM-guided regimes transfer structurally and directionally improve Crypto-20 IC, but do not yet dominate classical HMM at the portfolio-risk level.

## Phase 37 Crypto-20 Statistical Adjudication

Phase 37 tests whether the Phase 36 point-estimate differences survive fold uncertainty, dependence correction, and multiple comparisons. The primary tests pair the same 16 walk-forward folds across methods. A separate DM-style calibration test averages negative-log-likelihood differences across assets at each timestamp before applying a lag-7 Newey-West correction. Per-asset results are secondary because the 20 crypto assets are cross-correlated.

Guided-HMM has the highest mean fold IC (`0.0117`), but its 95% bootstrap interval crosses zero (`-0.0132` to `0.0404`). Relative to raw-feature HMM, its mean fold IC difference is only `+0.00065` (`p=0.840`). Relative to global LightGBM, the difference is `+0.00712` with raw `p=0.094`, but it does not survive correction (`BH q=0.322`). Relative to KMeans, the difference is `+0.00243` (`p=0.631`). No guided-HMM fold-level Sharpe or total-return comparison is significant.

The calibration evidence is negative. Guided-HMM has higher, therefore worse, multiclass NLL than global LightGBM (`DM p=1.51e-12`, surviving Holm correction) and raw-feature HMM (`DM p=0.0148`, surviving BH correction within the NLL family). The asset-level view is directionally stronger against global LightGBM: mean IC difference `+0.00521`, bootstrap interval `0.00037` to `0.00995`, with wins in 13 of 20 assets. Because those assets are correlated and the sign test is not significant, this remains descriptive rather than confirmatory.

The Phase 37 conclusion is therefore narrower than the Phase 36 point-estimate result. Structural transfer is supported, and there is weak directional evidence that guided-HMM improves return ranking versus a global model. Statistical dominance over raw HMM or KMeans, superior risk-adjusted performance, and superior probability calibration are not supported.

The result is also informative because it is weaker but consistent relative to the BTC/ETH pilot. The assignment-layer mechanism replicates: `hmm_guided_hmm` beats `hmm_guided_gmm` on NMI, ARI, purity, and silhouette. But the absolute HMM-reference alignment is lower than in the two-asset pilot (`NMI 0.694` versus `0.869`, purity `0.814` versus `0.957`), which is consistent with greater regime heterogeneity across 20 assets. The transition diagonal is higher on Crypto-20 (`0.890` versus about `0.825` in the BTC/ETH guided-HMM run), so regime persistence and HMM-reference agreement appear to decouple at larger scale. That nuance should be kept in the generalization section rather than hidden.

## Phase 38 Research-Control Reset

Phase 38 does not add a favorable model result. It protects the integrity of the next one. All inspected BTC/ETH and Crypto-20 outcomes are now classified as development-observed in `reports/data_role_registry.csv`, and every completed or planned experiment family is recorded in `reports/experiment_ledger.csv`. The ordered gates in `reports/publication_acceptance_gates.md` block candidate selection, locked evaluation, publication claims, and product expansion until their prerequisites pass.

The most important protocol change is that fold-local assignment is no longer considered enough for the final learned-regime claim. The encoder itself must be trained inside each outer walk-forward fold. Scaling, weak-supervision HMM fitting, pair mining, encoder fitting, embedding assignment, calibration, threshold selection, and alpha fitting must respect the outer training boundary, while model decisions use only inner chronological validation.

The next baseline must also restore the full mechanism ladder on Crypto-20: global and classical methods, vanilla contrastive-GMM/HMM, and guided contrastive-GMM/HMM with identical folds and coverage. Calibration, soft posterior gating, and partially pooled experts are authorized only after that validity baseline passes. Existing Phase 36/37 results remain useful development evidence but cannot be called an untouched final test.

## Limitations

- Hourly OHLCV is a noisy signal source.
- The current contrastive encoder is not a true Temporal Fusion Transformer.
- The contrastive encoder is still trained as an offline/frozen representation; Phase 38 makes fully fold-local encoder retraining a critical validity requirement rather than an optional upgrade.
- Offline/global regime results and fold-local regime results should be interpreted separately.
- HMM states are not ground truth market regimes. They are a classical proxy/reference state sequence used for comparison and weak supervision.
- Regime-quality agreement metrics are diagnostic; agreement with HMM does not prove economic correctness.
- Phase 20 tests guided embeddings downstream, but the guided encoder itself is still trained as a frozen/offline representation. A stricter future version would retrain or update the encoder inside each walk-forward fold.
- The Phase 20 guided-HMM edge over raw-feature HMM is promising but not statistically significant at the 5% level on fold-level IC.
- Phase 21 adds guided-method robustness and stress coverage. Guided-HMM is stress-robust on the primary BTC+ETH 8h prediction file, but symbol/horizon robustness remains mixed.
- Phase 22A is a 3-epoch structural prototype. It does not replace the full 30-epoch time-only guided baseline and has not been retested downstream.
- Phase 23 interpretability is diagnostic. Fold-local SHAP and feature-importance summaries improve economic plausibility, but they are not causal explanations.
- Phase 24 freezes claim language but does not add new empirical evidence.
- Phase 25 aggregates completed artifacts into a minimal ablation table; it does not create new retrained encoder variants beyond already completed runs.
- Phase 26 refreshes paper-facing claim tests, but the main guided-HMM versus raw-feature HMM IC edge remains directionally supported rather than statistically significant.
- Phase 29 improves the manuscript prose, but it still needs final citations, venue formatting, and figure numbering.
- Phase 28 documents reproducibility paths but does not make raw data or model weights public.
- Calibration/NLL diagnostics do not uniformly favor the guided methods, so probability quality and trading-score quality should be discussed separately.
- Phase 19A is a positioning phase, not an empirical result. It clarifies contribution language but does not prove a new model improvement.
- Literature positioning depends on describing HMM states as proxy/reference states, not ground truth labels.
- Phase 14B stress testing re-scores existing predictions; it does not retrain models under each cost or threshold assumption.
- Several method differences are not statistically significant at the 5% level under fold-level tests, and the strongest IC finding does not survive multiple-testing correction.
- Phase 17 timings are synthetic planning estimates, not formal hardware benchmarks.
- Backtest returns are research diagnostics, not deployable trading evidence.
- The project intentionally excludes live trading, RL, online retraining, and order-book data in this phase.

## Next Steps

1. Implement the fully fold-local encoder baseline and leakage tests.
2. Restore vanilla contrastive-GMM/HMM to the equal-coverage Crypto-20 mechanism ladder.
3. Use inner validation to test only the pre-authorized calibration, soft-gating, and pooled-expert candidates.
4. Freeze one candidate before a single registered external evaluation.
5. Complete final dependence-aware statistics before venue formatting and product expansion.
