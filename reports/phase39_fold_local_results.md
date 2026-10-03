# Phase 39 Fully Fold-Local Encoder Results

> **Invalidated development artifact:** the metrics below came from per-symbol positional folds that overlapped in calendar time across assets. A later audit found that 95% of test rows in every fold were exposed to pooled training data from the same or a later calendar date. Retain these numbers only for debugging history; they are not predictive evidence.

## Run Status

This is a **full development-observed benchmark** over 16 fold(s) and 20 symbol(s), using at most 5,000 deterministic windows per encoder stage. All outcomes remain development-observed and cannot be used as an untouched final test.

## Validity Contract

- Scalers, weak-supervision HMMs, contrastive pairs, encoder weights, assignment layers, and alpha models are fit inside each outer fold.
- Epochs are selected on an inner chronological validation block.
- The selected epoch count is refit on the full authorized outer-training interval.
- Outer-test metrics do not influence training or model selection.
- Vanilla and guided learned methods share the same outer folds and test coverage as classical and global baselines.

## Method Results

| method | IC | Sharpe | drawdown | total_return | turnover | n_test_rows |
| --- | --- | --- | --- | --- | --- | --- |
| global_lgbm | 0.0175 | 0.1087 | -0.5478 | 0.0724 | 0.0233 | 230400 |
| regime_lgbm_contrastive | 0.0222 | 0.0490 | -0.5461 | -0.0156 | 0.0331 | 230400 |
| regime_lgbm_contrastive_hmm | 0.0192 | -0.2271 | -0.6155 | -0.2543 | 0.0349 | 230400 |
| regime_lgbm_hmm | 0.0214 | 0.1479 | -0.5428 | 0.1386 | 0.0386 | 230400 |
| regime_lgbm_hmm_guided_gmm | 0.0173 | 0.1742 | -0.5008 | 0.1891 | 0.0397 | 230400 |
| regime_lgbm_hmm_guided_hmm | 0.0155 | 0.3131 | -0.6035 | 0.4483 | 0.0392 | 230400 |
| regime_lgbm_kmeans | 0.0229 | 0.1238 | -0.5696 | 0.0970 | 0.0378 | 230400 |
| regime_lgbm_vol_bucket | 0.0159 | -0.1596 | -0.6539 | -0.2106 | 0.0440 | 230400 |

## Selected Epochs

| encoder_method | min | median | max |
| --- | --- | --- | --- |
| guided | 2 | 3.5000 | 27 |
| vanilla | 5 | 11.0000 | 24 |

The two vanilla encoders required 5.09 aggregate training hours across folds; the guided encoders required 3.01 hours. Epochs were selected independently inside each fold and were not chosen from outer-test performance.

## Fold-Level Statistical Result

| Method | Mean fold IC | 95% bootstrap CI | Positive IC folds | Mean fold Sharpe |
| --- | ---: | ---: | ---: | ---: |
| Vanilla contrastive-HMM | 0.0110 | [-0.0097, 0.0338] | 9/16 | -0.1626 |
| Raw-feature HMM | 0.0110 | [-0.0111, 0.0378] | 8/16 | -0.1119 |
| KMeans | 0.0093 | [-0.0157, 0.0393] | 8/16 | -0.2254 |
| Vanilla contrastive-GMM | 0.0076 | [-0.0162, 0.0345] | 6/16 | -0.3901 |
| Guided-GMM | 0.0067 | [-0.0155, 0.0314] | 8/16 | -0.3913 |
| Volatility buckets | 0.0060 | [-0.0148, 0.0299] | 8/16 | -0.3318 |
| Guided-HMM | 0.0057 | [-0.0172, 0.0316] | 8/16 | -0.0409 |
| Global LightGBM | 0.0046 | [-0.0205, 0.0318] | 7/16 | 0.0590 |

Every IC interval crosses zero. No IC or Sharpe comparison survives multiple-testing correction.

## Primary Guided-HMM Comparisons

| Reference | Mean fold IC difference | Paired p-value | Mean fold Sharpe difference | Conclusion |
| --- | ---: | ---: | ---: | --- |
| Raw-feature HMM | -0.0054 | 0.111 | +0.0710 | No dominance; IC direction favors raw HMM |
| Global LightGBM | +0.0011 | 0.815 | -0.0999 | No meaningful difference |
| Vanilla contrastive-HMM | -0.0054 | 0.488 | +0.1217 | Mixed and non-significant |
| KMeans | -0.0036 | 0.476 | +0.1844 | Mixed and non-significant |

Guided-HMM has the strongest aggregate Sharpe point estimate (`0.313`) and aggregate return (`0.448`), but its drawdown is `-0.603`, its fold-average Sharpe is slightly negative, and its return distribution is extremely heavy-tailed. These diagnostics do not establish a stable or deployable trading advantage.

## Calibration

Guided-HMM has worse multiclass negative log-likelihood than global LightGBM by `0.00854`; the dependence-aware DM-style p-value is `9.17e-11` and the result survives Holm correction across all tests. The weaker calibration result is therefore more statistically defensible than any ranking or Sharpe advantage.

## Phase 39 Decision

No scientific method comparison can be made from this run because its global calendar boundary was invalid. The repaired common-calendar pipeline must be rerun before reassessing ranking, calibration, or portfolio outcomes.

H8 remains open after the validity repair. Phase 40 model changes are paused until the repaired baseline is completed, and the invalidated outer-fold outcomes must not be used for tuning.

## Interpretation Rule

A smoke run validates code paths, leakage boundaries, artifacts, and coverage only. A full run is still development evidence. Model changes require a new registered candidate family, and confirmatory claims require a frozen configuration evaluated once on a locked holdout.
