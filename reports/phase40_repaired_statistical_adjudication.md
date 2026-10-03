# Phase 40 Repaired Statistical Adjudication

## Purpose

Phase 40 asks whether the repaired Phase 39R method differences are statistically reliable. It does not train a new model, tune a threshold, select a winner, or touch locked/final test data.

The input is the completed repaired fold-local prediction file:

```text
models/crypto20_repaired_fold_local_alpha_oos_predictions.csv
```

The run uses the repaired `crypto20-development-v1` development panel. This means the results are valid development evidence, not an untouched final-test claim.

## Command

```powershell
.\env\Scripts\python.exe src\statistical_tests.py `
  --predictions models\crypto20_repaired_fold_local_alpha_oos_predictions.csv `
  --experiment-results models\crypto20_repaired_fold_local_experiment_results.csv `
  --output-prefix crypto20_repaired_fold_local_ `
  --bootstrap-samples 5000 `
  --reference-methods global_lgbm regime_lgbm_hmm regime_lgbm_kmeans
```

## Main Artifacts

```text
models/crypto20_repaired_fold_local_statistical_method_summary.csv
models/crypto20_repaired_fold_local_statistical_pairwise_tests.csv
models/crypto20_repaired_fold_local_statistical_asset_pairwise_tests.csv
models/crypto20_repaired_fold_local_statistical_multiple_testing.csv
models/crypto20_repaired_fold_local_statistical_claims.csv
models/crypto20_repaired_fold_local_statistical_sharpe_diagnostics.csv
models/crypto20_repaired_fold_local_statistical_ic_confidence_intervals.png
models/crypto20_repaired_fold_local_statistical_multiple_testing.png
models/crypto20_repaired_fold_local_statistical_sharpe_diagnostics.png
```

## Result Summary

The repaired statistical result does not support a robust positive-alpha or method-dominance claim.

Fold-level mean IC is negative for every method:

| Method | Mean fold IC | 95% bootstrap CI | Mean fold Sharpe | Positive IC folds |
|---|---:|---:|---:|---:|
| `regime_lgbm_contrastive` | -0.0116 | [-0.0343, 0.0107] | -1.1981 | 6 / 16 |
| `regime_lgbm_hmm_guided_gmm` | -0.0158 | [-0.0299, -0.0011] | -1.6734 | 6 / 16 |
| `regime_lgbm_hmm` | -0.0164 | [-0.0357, 0.0029] | -1.3215 | 6 / 16 |
| `regime_lgbm_hmm_guided_hmm` | -0.0177 | [-0.0295, -0.0055] | -1.4322 | 4 / 16 |
| `regime_lgbm_vol_bucket` | -0.0188 | [-0.0366, -0.0015] | -0.8376 | 6 / 16 |
| `global_lgbm` | -0.0205 | [-0.0431, 0.0021] | -0.4985 | 7 / 16 |
| `regime_lgbm_contrastive_hmm` | -0.0218 | [-0.0363, -0.0071] | -1.1996 | 5 / 16 |
| `regime_lgbm_kmeans` | -0.0223 | [-0.0414, -0.0029] | -0.7614 | 7 / 16 |

The only raw IC comparison that looks mildly encouraging is vanilla contrastive versus global LightGBM:

```text
regime_lgbm_contrastive vs global_lgbm:
mean IC difference = +0.008914
paired t-test p = 0.0447
BH q by metric = 0.4959
claim status = raw_only_suggestive
```

Because the corrected q-value is not significant, this is not a publishable superiority claim.

## Guided-Method Finding

The repaired guided methods do not win the repaired statistical adjudication.

Key guided-HMM comparisons:

| Comparison | Metric | Mean difference | Primary p-value | Corrected status |
|---|---:|---:|---:|---|
| `regime_lgbm_hmm_guided_hmm` vs `global_lgbm` | IC | +0.002826 | 0.7034 | not significant |
| `regime_lgbm_hmm_guided_hmm` vs `regime_lgbm_hmm` | IC | -0.001288 | 0.8415 | not significant |
| `regime_lgbm_hmm_guided_hmm` vs `regime_lgbm_kmeans` | IC | +0.004636 | 0.3570 | not significant |

The asset-level diagnostic is also inconclusive:

| Comparison | Metric | Mean difference | 95% bootstrap CI | Wins |
|---|---:|---:|---:|---:|
| `regime_lgbm_hmm_guided_hmm` vs `global_lgbm` | IC | -0.000546 | [-0.006764, 0.005513] | 11 / 20 |
| `regime_lgbm_hmm_guided_hmm` vs `regime_lgbm_hmm` | IC | -0.002557 | [-0.008250, 0.002769] | 10 / 20 |
| `regime_lgbm_hmm_guided_hmm` vs `regime_lgbm_kmeans` | IC | +0.002014 | [-0.004515, 0.010013] | 7 / 20 |

This means Phase 40 does not rescue the older guided-HMM alpha story under repaired validation.

## Calibration and Portfolio Diagnostics

Negative log-likelihood diagnostics mostly favor the simpler references. Versus global LightGBM, all regime-conditioned methods have worse NLL, and these differences survive correction.

Probabilistic Sharpe diagnostics are also negative. The best PSR is still below 0.5:

| Method | Annualized Sharpe | PSR(SR > 0) |
|---|---:|---:|
| `global_lgbm` | -0.0884 | 0.4598 |
| `regime_lgbm_kmeans` | -0.2091 | 0.4063 |
| `regime_lgbm_hmm` | -0.2600 | 0.3836 |
| `regime_lgbm_vol_bucket` | -0.2954 | 0.3683 |
| `regime_lgbm_hmm_guided_hmm` | -0.7620 | 0.1955 |

This is not evidence of deployable trading performance.

## Paper-Safe Interpretation

Phase 40 strengthens the project by making the repaired conclusion explicit:

```text
Under leakage-safe common-calendar validation, the repaired Crypto-20 methods do not produce robust positive alpha or statistically corrected method dominance. The project remains valuable as a rigorous benchmark showing that structural regime quality, ranking IC, calibration, and transaction-cost-aware portfolio behavior are separate outcomes.
```

## What Phase 40 Allows

Allowed:

- say the repaired Phase 39R outputs were statistically adjudicated;
- say no repaired method currently establishes robust dominance;
- say all methods have weak/negative development alpha under the repaired protocol;
- use the result to motivate bounded development-only improvements.

Forbidden:

- claiming guided-HMM beats HMM on repaired Crypto-20;
- claiming the strategy is profitable;
- using the repaired Crypto-20 development panel as an untouched final test;
- tuning directly against the Phase 40 outer-test results.

## Next Step

The correct next phase is Phase 41: bounded calibration and soft-gating candidates using only inner-development evidence. Phase 41 must not tune directly against Phase 40 outer-test outcomes.
