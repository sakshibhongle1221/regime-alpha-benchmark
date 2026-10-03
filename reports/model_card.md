# Regime Alpha Benchmark Model Card

## Frozen Baseline

| Field | Value |
|---|---|
| Frozen run id | `20260522_phase14b_baseline` |
| Source ref | `v1.3-phase14b` |
| Source commit | `f0902b21057d3dc464ce9e31d6be718a70531c63` |
| Archive manifest | `runs/20260522_phase14b_baseline/manifest.json` |
| Artifact count | 39 curated artifacts |
| Missing artifacts | 0 |

This model card documents the current Phase 14B baseline before statistical testing and encoder upgrades. The frozen run is the reference point for later significance tests, regime-quality metrics, encoder changes, and ablations.

## Research Positioning

Phase 19A adds paper-facing literature positioning.

| Field | Value |
|---|---|
| Narrative note | `reports/related_work.md` |
| Source matrix | `reports/literature_matrix.csv` |
| Literature clusters | time-series contrastive learning, financial regime switching, financial ML validation, regime-conditioned alpha modeling |
| Contribution framing | benchmark learned regimes versus classical sequential regimes under financial validation |
| Key caution | HMM states are reference/proxy states, not ground-truth labels |

The model-card contribution statement is: Regime Alpha Benchmark evaluates whether
learned market-regime representations can beat, match, or explain classical
sequential regime models when both are tested under equal targets, walk-forward
folds, transaction costs, robustness checks, and statistical claim controls.

## Data

| Field | Value |
|---|---|
| Universe | `BTCUSDT`, `ETHUSDT` |
| Source | Binance OHLCV |
| Interval | 1 hour |
| Lookback | 730 days |
| OHLCV rows | 17,520 per symbol |
| Feature rows | 17,460 per symbol |
| Target rows | 17,436 per symbol |
| Database | `data/market.duckdb` locally; not committed |

The committed repository keeps curated research artifacts only. Raw data, DuckDB files, model weights, dense row-level predictions, and numpy arrays remain ignored.

## Feature Set

The feature matrix contains 22 engineered technical and microstructure-inspired features:

```text
ret_1h, ret_5h, ret_15h, ret_60h,
vol_5h, vol_20h, vol_of_vol,
amihud, volume_zscore, ret_autocorr,
spread_proxy, ofi_proxy, rsi_14, gk_vol,
skewness, kurtosis, macd_signal,
bband_pct_b, atr_14, close_vs_vwap,
log_vol_trend, ret_dispersion
```

## Target Labels

| Field | Value |
|---|---|
| Primary target | `tb_label_8h` |
| Classes | `-1`, `0`, `+1` |
| Return column | `forward_return_8h` |
| Additional horizons | 4h, 8h, 24h |
| Labeling style | Directional, triple-barrier, forward return, volatility-adjusted return |

The primary alpha benchmark uses the 8-hour triple-barrier target. Phase 14A also evaluates 4-hour and 24-hour targets for robustness.

## Encoder Architecture

| Field | Value |
|---|---|
| Encoder class | `TemporalEncoder` |
| Input shape | `(batch, 60, 22)` |
| Window size | 60 bars |
| Input projection | 22 to 64 |
| Positional encoding | Learnable, max length 512 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Feed-forward dimension | 256 |
| Dropout | 0.1 |
| Pooling | Mean pooling over time |
| Projection head | Linear 64 to 64, ReLU, Linear 64 to 16 |
| Embedding dimension | 16 |
| Output normalization | L2 normalization |
| Parameter count | 139,408 |

## Encoder Training

| Field | Value |
|---|---|
| Current loss | NT-Xent |
| Temperature | 0.07 |
| Positive pair | Adjacent/augmented temporal window from `WindowDataset` |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| Scheduler | CosineAnnealingLR |
| Default epochs | 30 |
| Sparse artifact stride | 4 |
| Dense inference stride | 1 in `visualize_regimes.py` |
| Random state | 42 where supported |

This encoder is the main candidate for future upgrades. The current research finding is that vanilla contrastive embeddings improve when HMM state dynamics are added, but they do not yet dominate raw-feature HMM regimes.

## Regime Methods

| Method | Description |
|---|---|
| `contrastive` | GMM over contrastive encoder embeddings |
| `contrastive_hmm` | Gaussian HMM over contrastive encoder embeddings |
| `hmm` | Gaussian HMM over raw selected features |
| `kmeans` | KMeans over raw feature space |
| `vol_bucket` | Realized-volatility quantile buckets |

HMM states are not ground-truth regimes. They are a classical reference proxy and a competitive baseline. Agreement with HMM states should be described as proxy agreement, not accuracy.

## Regime Quality Diagnostics

Phase 16 evaluates regime structure independently of alpha-model returns.

| Field | Value |
|---|---|
| Main artifact | `models/regime_quality_summary.csv` |
| Pairwise artifact | `models/regime_agreement_matrix.csv` |
| Visual artifacts | `models/regime_quality_heatmap.png`, `models/regime_agreement_heatmap.png` |
| Reference method | raw-feature Gaussian HMM |
| Reference interpretation | classical proxy, not ground truth |
| Metrics | balance entropy, switch rate, transition diagonal, duration, posterior confidence, NMI, ARI, purity |

Latest BTC+ETH structural read:

| Method | Balance Entropy | Avg Duration | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|
| contrastive | 0.999 | 30.51 | 0.032 | 0.379 |
| contrastive_hmm | 0.999 | 42.18 | 0.020 | 0.377 |
| hmm | 0.959 | 6.98 | 1.000 | 1.000 |
| kmeans | 0.866 | 4.05 | 0.182 | 0.459 |
| vol_bucket | 1.000 | 10.44 | 0.333 | 0.599 |

The learned regimes are balanced and persistent, but weakly aligned with the HMM reference. This suggests the current encoder objective creates smooth partitions without yet producing the most alpha-relevant state structure.

## Alpha Models

| Field | Value |
|---|---|
| Model family | LightGBM multiclass classifier |
| Classes | `-1`, `0`, `+1` mapped to 3 classes |
| Alpha score | `P(+1) - P(-1)` |
| Default trade threshold | 0.05 |
| Default transaction cost | 10 bps |
| Global model | One LightGBM over all rows |
| Regime model | One LightGBM per regime, posterior-weighted predictions |

LightGBM parameters:

```text
objective=multiclass
num_class=3
n_estimators=250
learning_rate=0.04
max_depth=4
num_leaves=15
min_child_samples=25
subsample=0.8
colsample_bytree=0.8
random_state=42
```

## Validation

| Field | Value |
|---|---|
| Validation style | Expanding walk-forward |
| Initial training window | 6 months, 4,320 hourly bars |
| Test step | 1 month, 720 hourly bars |
| Embargo | 5 days, 120 hourly bars |
| Primary label horizon purge | 8 bars |
| Main OOS rows | 25,920 per method |
| Critical audit status | 31 PASS, 1 methodological WARN, 0 FAIL |

The current methodological warning is that the legacy `regime_assignments.csv` artifact is offline/global. Predictive regime claims should use the fold-local Phase 13 artifacts.

## Robustness Coverage

| Phase | Coverage |
|---|---|
| Phase 14A | BTC-only, ETH-only, BTC+ETH across 4h, 8h, 24h |
| Phase 14B | Thresholds 0.03, 0.05, 0.07, 0.10; costs 5, 10, 20 bps; all/bull/sideways/bear periods |
| Phase 15A | Fold-level bootstrap confidence intervals, paired tests, and DM-style NLL forecast-loss checks |
| Phase 15B | Benjamini-Hochberg/Holm corrections, corrected claim status, and Probabilistic Sharpe diagnostics |
| Phase 16 | Regime quality and pairwise agreement diagnostics independent of alpha performance |
| Phase 17 | Encoder compute profile and 12-run ablation budget |
| Phase 18 | HMM-guided contrastive encoder prototype and structural diagnostics |
| Phase 19A | Literature positioning and paper contribution map |
| Phase 19B | Full 30-epoch HMM-guided encoder run and baseline structural comparison |
| Phase 20 | Fold-local downstream alpha retest for HMM-guided embeddings |
| Phase 21 | Guided-method refresh of symbol/horizon robustness and cost/threshold/period stress robustness |
| Phase 22A | 3-epoch time-frequency HMM-guided encoder prototype |
| Phase 23 | Fold-local LightGBM feature-importance and SHAP interpretability diagnostics |
| Phase 24 | Paper protocol freeze with hypotheses, claim registry, and experiment manifest |
| Phase 25 | Minimal ablation suite for objective, assignment-layer, augmentation, and classical-reference mechanisms |
| Phase 26 | Paper-facing statistical claim tests mapped to the Phase 25 ablation suite |
| Phase 27 | Generated manuscript skeleton, paper artifact map, and submission checklist |
| Phase 28 | Reproducibility package with smoke/full/dashboard modes and artifact policy docs |
| Phase 29 | Paper prose pass for submission-style Markdown while preserving conservative claims |

Phase 14B re-scores existing fold-local predictions. It does not retrain models for every cost or threshold setting.

## Compute Plan

Phase 17 records local compute estimates before starting encoder-upgrade work.

| Field | Value |
|---|---|
| Artifact | `models/compute_profile.csv` |
| Ablation queue | `models/ablation_budget.csv` |
| Visual plan | `models/compute_budget_plan.png` |
| Device measured | CPU |
| Synthetic step time | 0.739 seconds |
| Estimated encoder retrain | 100.10 minutes |
| Estimated 12-run grid | 21.62 hours |
| Budget status | green |

The completed guided-encoder experiments now include a structural run, a fold-local downstream alpha retest, a guided-method robustness/stress refresh, and a capped time-frequency prototype. Broader ablations should expand only if the time-frequency path earns additional compute.

## HMM-Guided Encoder Variant

Phase 18/19B adds a separate encoder variant. It does not overwrite the current baseline encoder.

| Field | Value |
|---|---|
| Script | `src/guided_encoder.py` |
| Weak supervision source | raw-feature HMM states from `regime_assignments.csv` |
| Positive pairs | same HMM state, distant in time or cross-symbol |
| Hard negatives | different HMM state, nearby in the same symbol |
| Default min positive gap | 24 bars |
| Default hard negative gap | 24 bars |
| Default hard negative weight | 2.0 |
| Output model | `models/guided_encoder.pt` locally; ignored by Git |
| Committed diagnostics | `guided_encoder_summary.csv`, `guided_encoder_loss.csv`, `guided_encoder_comparison.csv`, `guided_alpha_comparison.csv`, guided plots |

Full 30-epoch Phase 19B diagnostics:

| Method | Silhouette | Avg Duration | Transition Diagonal | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|
| `hmm_guided_gmm` | 0.384 | 5.09 | 0.804 | 0.609 | 0.759 |
| `hmm_guided_hmm` | 0.629 | 5.72 | 0.825 | 0.869 | 0.957 |

The full run confirms that HMM-guided weak supervision strongly improves structural alignment versus the old contrastive encoder path.

## Time-Frequency Guided Encoder Variant

Phase 22A extends `src/guided_encoder.py` with an optional time-frequency input view. For each 60-bar window, the encoder keeps the original normalized time-domain features and appends six low-frequency FFT magnitude bands per feature.

| Field | Value |
|---|---|
| CLI | `python src/guided_encoder.py --symbols BTCUSDT ETHUSDT --augmentation time_frequency --epochs 3` |
| Augmentation | time-domain features plus FFT magnitude bands |
| FFT bins | 6 |
| Input features | 154 |
| Output model | `models/time_frequency_encoder_model.pt` locally; ignored by Git |
| Committed diagnostics | `time_frequency_encoder_summary.csv`, `time_frequency_encoder_loss.csv`, `time_frequency_encoder_comparison.csv`, time-frequency plots |

Phase 22A diagnostics:

| Method | Epochs | Silhouette | Avg Duration | Transition Diagonal | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| `tf_hmm_guided_gmm` | 3 | 0.326 | 6.47 | 0.845 | 0.504 | 0.682 |
| `tf_hmm_guided_hmm` | 3 | 0.338 | 8.39 | 0.881 | 0.528 | 0.704 |

The prototype is stronger than the original vanilla contrastive regime path but weaker than the full 30-epoch time-only guided-HMM baseline. It should be treated as an ablation candidate, not as a replacement model.

## Guided Alpha Retest

Phase 20 evaluates the Phase 19B guided embeddings inside the strict fold-local alpha benchmark. The embeddings are frozen, but the GMM/HMM assignment layer on top of those embeddings is fit inside each walk-forward fold.

| Method | IC | Sharpe | Drawdown | Total Return | OOS Rows |
|---|---:|---:|---:|---:|---:|
| `regime_lgbm_hmm` | 0.0051 | -0.340 | -0.710 | -0.536 | 25,920 |
| `regime_lgbm_hmm_guided_gmm` | -0.0092 | -0.976 | -0.900 | -0.854 | 25,920 |
| `regime_lgbm_hmm_guided_hmm` | 0.0094 | 0.099 | -0.614 | 0.031 | 25,920 |

The useful finding is narrow but real: guided embeddings help downstream alpha only when paired with an HMM assignment layer. The GMM assignment layer on the same guided embedding space remains weak. Fold-level IC improvement versus raw-feature HMM is positive but not statistically significant at the 5% level.

## Interpretability Outputs

Phase 23 adds fold-local interpretation for the alpha models. The script trains explanation models inside the same expanding walk-forward folds and aggregates LightGBM gain/split importance plus capped SHAP summaries.

| Field | Value |
|---|---|
| CLI | `python src/interpretability.py --symbols BTCUSDT ETHUSDT` |
| Explained methods | `global_lgbm`, `regime_lgbm_hmm`, `regime_lgbm_hmm_guided_hmm` |
| SHAP cap | 256 training rows per fold/model by default |
| Main committed outputs | `feature_importance_global.csv`, `feature_importance_by_regime.csv`, `feature_family_summary.csv`, interpretability PNGs |
| Raw local output | `models/feature_importance_raw.csv`; ignored by Git |

Top guided-HMM regime-conditioned features are dominated by `vol_20h`, `vol_of_vol`, `atr_14`, `kurtosis`, `skewness`, `ret_autocorr`, and `ret_60h`. At the family level, guided-HMM regimes are mostly volatility driven, with momentum/autocorrelation and distribution-shape features providing the next largest attribution shares.

Interpretation boundary: these diagnostics show which features the trained LightGBM models used inside the validation design. They do not establish causal feature effects or prove that HMM reference states are ground-truth regimes.

## Statistical Testing

Phase 15A uses `models/walkforward_alpha_oos_predictions.csv` as the prediction source and writes:

```text
models/statistical_fold_metrics.csv
models/statistical_method_summary.csv
models/statistical_pairwise_tests.csv
models/statistical_test_summary.csv
models/statistical_ic_confidence_intervals.png
models/statistical_multiple_testing.csv
models/statistical_claims.csv
models/statistical_sharpe_diagnostics.csv
models/statistical_multiple_testing.png
models/statistical_sharpe_diagnostics.png
```

The primary statistical unit is the walk-forward fold. Row-level samples are not treated as independent for IC/Sharpe claims because adjacent financial labels overlap. The row-level DM-style test is used only as a forecast-loss and calibration diagnostic over multiclass negative log-likelihood.

Phase 15B corrected-claim output should be used in paper language. A raw p-value below 0.05 is not enough for a strong claim unless it survives the multiple-testing correction family being cited.

## Current Baseline Finding

The current frozen baseline supports a cautious conclusion:

```text
HMM-guided embeddings with fold-local HMM assignment are the strongest current
point-estimate method for IC, Sharpe, drawdown, and total return, and Phase 21
shows that the same method is the strongest stress-grid winner on the primary
BTC+ETH 8h prediction file. The result is promising but not yet a statistically
significant dominance claim over the raw-feature HMM, and symbol/horizon
robustness remains mixed. The GMM assignment layer on guided embeddings remains
weak.
```

Future encoder phases must be evaluated against this frozen run, not against overwritten latest CSVs.

## Paper Protocol Freeze

Phase 24 adds four paper-control documents:

```text
reports/paper_protocol.md
reports/hypotheses.md
reports/claim_registry.md
reports/experiment_manifest.md
```

These files define the current research question, hypothesis table, claim boundaries, completed experiment families, future experiment queue, and submission-readiness checklist. They should be treated as the control layer for Phase 25+ work: new experiments should map to a frozen hypothesis and should not expand the paper claim set without updating the protocol.

## Phase 38 Research-Control Reset

Phase 38 reclassifies every inspected BTC/ETH and Crypto-20 outcome as development-observed and adds four enforced control artifacts:

```text
reports/phase38_master_protocol.md
reports/data_role_registry.csv
reports/experiment_ledger.csv
reports/publication_acceptance_gates.md
```

The current learned-regime evidence uses an offline encoder followed by fold-local assignment and downstream fitting. That is sufficient for the completed historical diagnostic, but not for the final predictive claim. The next critical implementation must fit scaling, weak-supervision HMMs, contrastive pairs, encoder weights, assignment layers, calibration, thresholds, and alpha models inside each outer fold, using only an inner chronological validation block for decisions.

Phase 38 also blocks Crypto-50 expansion, unrestricted architecture search, and product deployment until the fold-local validity and baseline-completeness gates pass. The next complete baseline ladder must include global LightGBM, classical regime baselines, vanilla contrastive-GMM/HMM, and guided contrastive-GMM/HMM with equal folds and coverage.

## Minimal Ablation Suite

Phase 25 adds the first paper-facing ablation layer.

| Field | Value |
|---|---|
| CLI | `python src/ablation_suite.py` |
| Main outputs | `ablation_results.csv`, `ablation_summary.csv`, `ablation_heatmap.png` |
| Families tested | objective guidance, assignment layer, augmentation view, classical reference |
| Main positive result | HMM assignment is consistently stronger than GMM assignment for the guided learned-regime path |
| Main negative result | The current 3-epoch time-frequency prototype does not justify full downstream alpha expansion |

The ablation suite does not claim a new retrained model variant. It consolidates completed structural and downstream artifacts into a mechanism-level decision table so the next statistical refresh can test only the claims the paper actually needs.

## Paper Statistical Claim Tests

Phase 26 converts the Phase 25 mechanism table into paper-facing statistical evidence.

| Field | Value |
|---|---|
| CLI | `python src/paper_claim_tests.py` |
| Main outputs | `paper_claim_tests.csv`, `paper_statistical_summary.csv`, `paper_claim_tests.png` |

## Paper Skeleton

Phase 27 created the audited manuscript scaffold, Phase 29 strengthened the prose, and Phases 37/38 synchronized the human-reviewed paper with the completed multi-asset evidence. The initializer now preserves existing reviewed paper artifacts.

| Field | Value |
|---|---|
| CLI | `python src/paper_skeleton.py` |
| Main outputs | `paper/main.md`, `reports/paper_artifact_map.csv`, `reports/paper_submission_checklist.md` |
| Purpose | Initialize missing paper artifacts while preserving human-reviewed manuscript, artifact-map, and checklist edits |
| Claim discipline | Keeps HMM states as proxy states, permits Crypto-20 structural transfer, records directional IC evidence, and blocks profitability or statistical dominance claims |

## Reproducibility Package

Phase 28 documents the exact public reproduction paths.

| Field | Value |
|---|---|
| CLI | `.\reproduce.ps1 -Mode smoke`, `.\reproduce.ps1 -Mode full`, `.\reproduce.ps1 -Mode dashboard` |
| Main outputs | `reports/environment.md`, `reports/artifact_manifest.md`, `reports/reproduction_checklist.md` |
| Dashboard dependencies | `requirements.txt` |
| Research dependencies | `requirements-research.txt` |
| Data policy | Curated summary artifacts are committed; raw data, DuckDB databases, model weights, embeddings, and row-level prediction files stay ignored |
| Main positive read | HMM assignment improves the guided path on all focused point-estimate metrics and is raw-suggestive on IC (`p=0.075`) |
| Main caution | Guided-HMM versus raw-feature HMM remains directionally supported, not statistically significant |
| Paper-safe language | Sequential assignment is the strongest supported mechanism; guided learned regimes are promising versus raw-feature HMM but not statistically dominant |
