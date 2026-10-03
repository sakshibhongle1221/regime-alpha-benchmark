# Phase 41B Inner-Validation Candidate Run

## Status

This is a **full development-observed run**. Candidate parameters are selected on inner chronological validation only, then evaluated once on the outer fold. This is still development-observed evidence and is not a locked final test.

## Scope

- Folds evaluated: 16
- Methods: global LightGBM, raw HMM, KMeans, volatility buckets
- Executed candidate families: probability calibration and soft regime gating
- Deferred registered candidates: score-threshold execution control (`p41_score_threshold`)
- Neural/guided candidates: deferred until this classical/global calibration layer is verified
- Selection input: inner validation only
- Forbidden input: Phase 40 outer-test statistical results

Score-threshold candidates are registered but deferred in this Phase 41B run. They change trade execution/signals rather than probability calibration, so they require a separate execution-focused run instead of being mixed into this probability/NLL adjudication.

The v1 grid also contains baseline-equivalent no-op values such as temperature `1.0`, prior blend `0.0`, shrinkage `0.0`, and posterior temperature `1.0`. These are retained as explicit controls for the recorded Phase 41B run; a future v2 registration should remove duplicate no-op variants before any new full run.

## Outer-Fold Diagnostic Results

| Method | Mean asset IC | NLL | Sharpe | Rows |
|---|---:|---:|---:|---:|
| `global_lgbm` | -0.011821 | 0.952094 | -0.2605 | 230400 |
| `regime_lgbm_hmm` | -0.011421 | 0.952616 | -0.5863 | 230400 |
| `regime_lgbm_kmeans` | -0.013547 | 0.952921 | -0.5231 | 230400 |
| `regime_lgbm_vol_bucket` | -0.017234 | 0.955362 | -0.4043 | 230400 |

## Selected Candidates

| Method | Candidate | Parameters | Folds |
|---|---|---|---:|
| `global_lgbm` | `baseline` | `none` | 4 |
| `global_lgbm` | `p41_prior_blend` | `prior_blend_weight=0.05` | 1 |
| `global_lgbm` | `p41_prior_blend` | `prior_blend_weight=0.1` | 2 |
| `global_lgbm` | `p41_prior_blend` | `prior_blend_weight=0.2` | 2 |
| `global_lgbm` | `p41_prob_temperature` | `temperature=0.75` | 4 |
| `global_lgbm` | `p41_prob_temperature` | `temperature=1.25` | 3 |
| `regime_lgbm_hmm` | `p41_global_regime_shrinkage` | `global_regime_shrinkage=0.5` | 9 |
| `regime_lgbm_hmm` | `p41_prior_blend` | `prior_blend_weight=0.2` | 2 |
| `regime_lgbm_hmm` | `p41_prob_temperature` | `temperature=0.75` | 3 |
| `regime_lgbm_hmm` | `p41_prob_temperature` | `temperature=1.25` | 1 |
| `regime_lgbm_hmm` | `p41_prob_temperature` | `temperature=1.5` | 1 |
| `regime_lgbm_kmeans` | `p41_global_regime_shrinkage` | `global_regime_shrinkage=0.5` | 9 |
| `regime_lgbm_kmeans` | `p41_prior_blend` | `prior_blend_weight=0.2` | 2 |
| `regime_lgbm_kmeans` | `p41_prob_temperature` | `temperature=0.75` | 3 |
| `regime_lgbm_kmeans` | `p41_prob_temperature` | `temperature=1.25` | 1 |
| `regime_lgbm_kmeans` | `p41_prob_temperature` | `temperature=1.5` | 1 |
| `regime_lgbm_vol_bucket` | `p41_global_regime_shrinkage` | `global_regime_shrinkage=0.5` | 11 |
| `regime_lgbm_vol_bucket` | `p41_prior_blend` | `prior_blend_weight=0.2` | 1 |
| `regime_lgbm_vol_bucket` | `p41_prob_temperature` | `temperature=0.75` | 2 |
| `regime_lgbm_vol_bucket` | `p41_prob_temperature` | `temperature=1.5` | 2 |

## Interpretation

This run is not a performance claim. Its purpose is to evaluate whether Phase 41 candidates selected only on inner validation improve the repaired global/classical development ladder. Statistical adjudication is required before making any development-level comparison.

## Comparison Against Repaired Classical Baseline

Compared with the repaired classical baseline, Phase 41B does not improve the overall alpha story. Mean asset IC remains weak/negative, and Sharpe/return generally deteriorate:

| Method | Baseline mean asset IC | Phase 41B mean asset IC | Baseline Sharpe | Phase 41B Sharpe |
|---|---:|---:|---:|---:|
| `global_lgbm` | -0.011348 | -0.011821 | -0.0884 | -0.2605 |
| `regime_lgbm_hmm` | -0.009337 | -0.011421 | -0.2600 | -0.5863 |
| `regime_lgbm_kmeans` | -0.013909 | -0.013547 | -0.2091 | -0.5231 |
| `regime_lgbm_vol_bucket` | -0.015342 | -0.017234 | -0.2954 | -0.4043 |

## Statistical Adjudication

The Phase 41B statistical run finds no corrected IC or Sharpe superiority claim.

| Method | Mean fold IC | 95% bootstrap CI | Mean fold Sharpe | Positive IC folds |
|---|---:|---:|---:|---:|
| `regime_lgbm_hmm` | -0.019603 | [-0.039748, 0.000222] | -1.1630 | 7 / 16 |
| `global_lgbm` | -0.019908 | [-0.041683, 0.001952] | -0.4645 | 7 / 16 |
| `regime_lgbm_vol_bucket` | -0.021542 | [-0.040617, -0.002982] | -1.0300 | 8 / 16 |
| `regime_lgbm_kmeans` | -0.023126 | [-0.043191, -0.003349] | -0.8982 | 7 / 16 |

Corrected IC/Sharpe claims:

```text
0
```

The calibration read is also not a clean win. Volatility-bucket calibration is worse than the global, raw-HMM, and KMeans references after correction. Global LightGBM remains the least-bad PSR diagnostic among the Phase 41B global/classical methods.

## Paper-Safe Conclusion

Controlled negative result: Phase 41B is useful because it shows that bounded inner-validation calibration and soft-gating candidates can be run without leaking Phase 40 outer-test information, but the first full global/classical candidate run does not produce a stronger alpha claim.

Allowed wording:

```text
Phase 41B tested bounded inner-validation-selected calibration and soft-gating candidates on the repaired global/classical ladder. The result did not improve the development alpha conclusion; corrected IC/Sharpe dominance remains unsupported.
```

Forbidden wording:

```text
Phase 41B improves the model.
Calibration fixes the alpha problem.
Soft gating produces tradable performance.
```
