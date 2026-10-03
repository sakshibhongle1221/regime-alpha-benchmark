# Phase 37 Crypto-20 Statistical Protocol

## Research Question

Does the Phase 36 HMM-guided regime model improve predictive or economic performance on Crypto-20 beyond global LightGBM, raw-feature HMM, and KMeans after accounting for fold uncertainty, temporal dependence, and multiple comparisons?

## Pre-Specified Primary Evidence

The primary inferential unit is the walk-forward fold. Phase 36 contains 16 paired folds with identical test coverage across all methods. The primary comparisons are:

1. `regime_lgbm_hmm_guided_hmm` versus `regime_lgbm_hmm`.
2. `regime_lgbm_hmm_guided_hmm` versus `regime_lgbm_kmeans`.
3. `regime_lgbm_hmm_guided_hmm` versus `global_lgbm`.

Primary metrics are fold IC, Sharpe, total return, drawdown, and turnover. Paired t-tests, Wilcoxon tests, sign tests, paired-fold bootstrap confidence intervals, Cohen's dz, Benjamini-Hochberg correction, and Holm correction are reported. No row-level independence assumption is used for the primary claim.

## Secondary Evidence

- DM-style multiclass negative-log-likelihood tests average loss differences across assets at each timestamp before applying a Newey-West HAC correction with lag 7.
- Per-asset diagnostics report whether improvements are broad or concentrated. Crypto assets are cross-correlated, so asset-level p-values are descriptive and cannot replace the paired-fold primary tests.
- Probabilistic Sharpe Ratio is reported as a diagnostic, not proof of strategy profitability.

## Interpretation Rules

- A broad superiority claim requires a positive effect, a confidence interval excluding zero, and corrected `p < 0.05` on the relevant primary fold-level metric.
- An uncorrected `p < 0.05` with a non-significant corrected result is labeled suggestive only.
- A positive point estimate with a confidence interval crossing zero is directional evidence, not statistical support.
- Better IC with weaker Sharpe or return is reported as predictive-ranking improvement without portfolio-performance dominance.
- Negative or inconclusive findings remain valid and must not trigger post-hoc epoch tuning on the same OOS folds.

## Expected Artifacts

- `models/crypto20_statistical_fold_metrics.csv`
- `models/crypto20_statistical_asset_metrics.csv`
- `models/crypto20_statistical_method_summary.csv`
- `models/crypto20_statistical_pairwise_tests.csv`
- `models/crypto20_statistical_asset_pairwise_tests.csv`
- `models/crypto20_statistical_test_summary.csv`
- `models/crypto20_statistical_multiple_testing.csv`
- `models/crypto20_statistical_claims.csv`
- `models/crypto20_statistical_sharpe_diagnostics.csv`
- `models/crypto20_statistical_ic_confidence_intervals.png`
- `models/crypto20_statistical_multiple_testing.png`
- `models/crypto20_statistical_sharpe_diagnostics.png`

## Command

```powershell
.\scripts\phases\run_phase37_crypto20_statistics.ps1
```

## Completed Findings

The full Phase 37 analysis used 16 paired walk-forward folds, 20 assets, six methods, and `230,400` OOS rows per method.

`regime_lgbm_hmm_guided_hmm` has the highest mean fold IC (`0.0117`), but its bootstrap interval crosses zero (`[-0.0132, 0.0404]`). Its fold-level IC differences are:

| Reference | Mean IC difference | 95% bootstrap CI | Paired p-value | BH q-value | Interpretation |
|---|---:|---:|---:|---:|---|
| `global_lgbm` | +0.00712 | [-0.00027, 0.01500] | 0.0939 | 0.3216 | Directional only |
| `regime_lgbm_hmm` | +0.00065 | [-0.00578, 0.00639] | 0.8404 | 0.8404 | Inconclusive |
| `regime_lgbm_kmeans` | +0.00243 | [-0.00673, 0.01209] | 0.6311 | 0.8347 | Inconclusive |

No guided-HMM fold-level Sharpe or total-return comparison is statistically significant. The time-block DM diagnostic finds that guided-HMM has worse multiclass negative log-likelihood than global LightGBM (`p=1.51e-12`, surviving Holm correction) and worse NLL than raw HMM (`p=0.0148`, surviving BH correction within the metric family). This separates weak ranking evidence from probability calibration: the guided score can rank returns directionally without producing better calibrated class probabilities.

Across assets, guided-HMM improves IC over global LightGBM by an average `0.00521`, with a bootstrap interval `[0.00037, 0.00995]` and wins in 13 of 20 assets. This remains secondary evidence because the assets are cross-correlated and the sign test is not significant (`p=0.263`).

The pre-specified decision is therefore **no broad superiority claim and no automatic epoch expansion**. Phase 37 supports a structural-transfer finding and weak directional IC evidence, while rejecting claims of statistically proven alpha or calibration dominance on Crypto-20.
