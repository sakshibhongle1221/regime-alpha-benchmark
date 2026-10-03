# Phase 42 Interpretation And Execution Hardening

## Status

This is a development-observed diagnostic phase. It does not tune models, does not modify locked/final test data, and does not create a new performance claim.

## What Phase 42 Does

Phase 42 explains why the repaired alpha conclusion remains weak by checking:

1. execution sensitivity across signal thresholds and transaction costs,
2. regime transition behavior and stable-vs-transition alpha,
3. cross-asset alpha fragility,
4. feature-family target alignment on the frozen Crypto-20 development panel.

## Execution Stress Summary

Default means threshold `0.0` and transaction cost `10 bps`. Positive cells count how many threshold/cost settings produced positive total return.

| Benchmark | Method | Default Sharpe | Default total return | Default turnover | Positive stress cells |
|---|---|---:|---:|---:|---:|
| `phase39r_repaired_neural` | `global_lgbm` | -0.2953 | -0.0278 | 0.0559 | 5 / 16 |
| `phase39r_repaired_neural` | `regime_lgbm_hmm` | -0.5957 | -0.0601 | 0.0781 | 3 / 16 |
| `phase39r_repaired_neural` | `regime_lgbm_hmm_guided_hmm` | -0.6025 | -0.0537 | 0.0823 | 2 / 16 |
| `phase39r_repaired_neural` | `regime_lgbm_kmeans` | -0.6394 | -0.0553 | 0.0751 | 3 / 16 |
| `phase39r_repaired_neural` | `regime_lgbm_vol_bucket` | -0.8750 | -0.0814 | 0.0885 | 2 / 16 |
| `phase39r_repaired_neural` | `regime_lgbm_hmm_guided_gmm` | -1.0022 | -0.0886 | 0.0813 | 0 / 16 |
| `phase39r_repaired_neural` | `regime_lgbm_contrastive_hmm` | -1.2451 | -0.0958 | 0.0738 | 0 / 16 |
| `phase39r_repaired_neural` | `regime_lgbm_contrastive` | -1.4455 | -0.1130 | 0.0696 | 0 / 16 |
| `phase41b_classical_candidates` | `global_lgbm` | -0.2605 | -0.0229 | 0.0484 | 5 / 16 |
| `phase41b_classical_candidates` | `regime_lgbm_vol_bucket` | -0.4043 | -0.0382 | 0.0703 | 3 / 16 |
| `phase41b_classical_candidates` | `regime_lgbm_kmeans` | -0.5231 | -0.0438 | 0.0608 | 4 / 16 |
| `phase41b_classical_candidates` | `regime_lgbm_hmm` | -0.5863 | -0.0536 | 0.0629 | 1 / 16 |

## Regime Transition Diagnostics

| Regime method | Switch rate | Average duration | Mean confidence | Balance entropy |
|---|---:|---:|---:|---:|
| `contrastive` | 0.0463 | 20.97 | 0.8220 | 0.9941 |
| `contrastive_hmm` | 0.0658 | 14.89 | 0.8792 | 0.9958 |
| `hmm` | 0.2034 | 4.88 | 0.9278 | 0.9842 |
| `hmm_guided_gmm` | 0.0745 | 13.17 | 0.9689 | 0.9974 |
| `hmm_guided_hmm` | 0.0683 | 14.35 | 0.9885 | 0.9927 |
| `kmeans` | 0.2133 | 4.66 | 1.0000 | 0.9917 |
| `vol_bucket` | 0.0902 | 10.92 | 1.0000 | 0.9766 |

Stable-vs-transition mean IC diagnostic:

```text
state_bucket
stable       -0.011090
transition   -0.002957
```

## Cross-Asset Alpha Diagnostic

| Benchmark | Method | Mean asset IC | Positive IC assets | Mean asset Sharpe | Median turnover |
|---|---|---:|---:|---:|---:|
| `phase39r_repaired_neural` | `regime_lgbm_contrastive` | -0.003098 | 7 / 20 | -0.3325 | 0.0505 |
| `phase39r_repaired_neural` | `regime_lgbm_hmm_guided_gmm` | -0.008719 | 6 / 20 | -0.4730 | 0.0622 |
| `phase39r_repaired_neural` | `regime_lgbm_hmm` | -0.009337 | 5 / 20 | -0.2223 | 0.0557 |
| `phase39r_repaired_neural` | `global_lgbm` | -0.011348 | 5 / 20 | -0.2157 | 0.0312 |
| `phase39r_repaired_neural` | `regime_lgbm_hmm_guided_hmm` | -0.011895 | 5 / 20 | -0.4352 | 0.0646 |
| `phase39r_repaired_neural` | `regime_lgbm_kmeans` | -0.013909 | 3 / 20 | -0.1948 | 0.0503 |
| `phase39r_repaired_neural` | `regime_lgbm_vol_bucket` | -0.015342 | 4 / 20 | -0.1805 | 0.0675 |
| `phase39r_repaired_neural` | `regime_lgbm_contrastive_hmm` | -0.017791 | 4 / 20 | -0.4010 | 0.0535 |
| `phase41b_classical_candidates` | `regime_lgbm_hmm` | -0.011421 | 5 / 20 | -0.3697 | 0.0604 |
| `phase41b_classical_candidates` | `global_lgbm` | -0.011821 | 5 / 20 | -0.2297 | 0.0427 |
| `phase41b_classical_candidates` | `regime_lgbm_kmeans` | -0.013547 | 3 / 20 | -0.3129 | 0.0578 |
| `phase41b_classical_candidates` | `regime_lgbm_vol_bucket` | -0.017234 | 3 / 20 | -0.2410 | 0.0660 |

## Feature-Family Target Alignment

This is not model superiority evidence. It is a descriptive check of which feature families have the strongest development-observed one-feature target alignment.

| Feature family | Features | Mean absolute asset IC | Top feature |
|---|---:|---:|---|
| `volatility` | 5 | 0.033679 | `gk_vol` |
| `microstructure` | 2 | 0.029568 | `spread_proxy` |
| `momentum` | 8 | 0.021481 | `ret_dispersion` |
| `technical_state` | 2 | 0.016300 | `rsi_14` |
| `distribution_shape` | 2 | 0.014224 | `skewness` |
| `liquidity_volume` | 3 | 0.010460 | `volume_zscore` |

## Paper-Safe Interpretation

Phase 42 supports the current cautious paper story: the repaired pipeline is valid and informative, but the alpha layer remains fragile. The weak result is not explained away by one simple calibration fix. Execution assumptions, regime transitions, and cross-asset heterogeneity all matter.

Allowed wording:

```text
Phase 42 shows that the repaired alpha weakness is robust to execution diagnostics and is better treated as a modeling/market-structure limitation than as a single calibration bug.
```

Forbidden wording:

```text
Phase 42 proves the strategy is tradable.
Phase 42 rescues the alpha result.
Phase 42 should be used to tune a final-test candidate.
```
