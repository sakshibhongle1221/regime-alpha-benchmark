# Phase 36 Crypto-20 Downstream Alpha Retest

## Purpose

Phase 35 showed that the HMM-guided learned assignment mechanism transfers structurally from BTC/ETH to a pre-specified Crypto-20 universe. That was not a downstream alpha claim. Phase 36 is the predictive retest: it asks whether the same guided regimes improve transaction-cost-aware fold-local alpha results on Crypto-20.

## Protocol

- Universe: the pre-registered `crypto20` asset set from `models/asset_universe_crypto20.csv`.
- Target: `tb_label_8h`, the same primary target used in the BTC/ETH benchmark.
- Validation: expanding walk-forward validation with a 6-month initial train window, 1-month test step, and 5-day embargo.
- Regime fitting: all regime assignment models are refit inside each fold using training-history rows only.
- Methods: global LightGBM, raw-feature HMM, raw-feature KMeans, volatility bucket, HMM-guided embedding GMM, and HMM-guided embedding HMM.
- Vanilla contrastive embeddings are skipped for this phase unless separate dense Crypto-20 contrastive artifacts are generated. The Phase 36 claim is about guided-vs-classical downstream generalization, not vanilla contrastive re-training.

## Artifacts

The runner writes local row-level files for auditability, but only compact summaries are intended for Git:

- `models/crypto20_walkforward_experiment_results.csv`
- `models/crypto20_walkforward_regime_summary.csv`
- `models/crypto20_walkforward_guided_alpha_comparison.csv`
- `models/crypto20_walkforward_equity_curve.png`

Row-level outputs remain ignored:

- `models/crypto20_walkforward_alpha_oos_predictions.csv`
- `models/crypto20_walkforward_regime_assignments.csv`

## Claim Boundary

This phase may support a Crypto-20 downstream alpha claim only if the guided methods survive the same fold-local, embargoed, transaction-cost-aware benchmark as the classical methods. If guided methods do not beat raw-feature HMM, the correct paper conclusion is still useful: structural transfer does not automatically imply predictive transfer in a heterogeneous multi-asset universe.

## Completed Result

The full Phase 36 run completed with equal OOS coverage across all methods: `230,400` test rows per method across the pre-specified 20-symbol universe.

| Method | IC | Sharpe | Drawdown | Total return |
|---|---:|---:|---:|---:|
| `global_lgbm` | 0.0175 | 0.1087 | -0.5478 | 0.0724 |
| `regime_lgbm_hmm` | 0.0214 | 0.1479 | -0.5428 | 0.1386 |
| `regime_lgbm_kmeans` | 0.0229 | 0.1238 | -0.5696 | 0.0970 |
| `regime_lgbm_vol_bucket` | 0.0159 | -0.1596 | -0.6539 | -0.2106 |
| `regime_lgbm_hmm_guided_gmm` | 0.0169 | -0.1901 | -0.6659 | -0.2412 |
| `regime_lgbm_hmm_guided_hmm` | 0.0226 | -0.0573 | -0.6033 | -0.1257 |

The strongest guided method, `regime_lgbm_hmm_guided_hmm`, improves IC versus the global baseline by `+0.0050` and versus raw-feature HMM by `+0.0012`, but it has weaker Sharpe, drawdown, and total return than raw-feature HMM. The paper-safe conclusion is therefore mixed: the guided assignment mechanism transfers structurally and directionally improves predictive ranking quality, but portfolio-level risk-adjusted performance does not yet dominate the classical HMM on Crypto-20.

## Command

```powershell
.\scripts\phases\run_phase36_crypto20_alpha.ps1
```

For a quick smoke check:

```powershell
.\scripts\phases\run_phase36_crypto20_alpha.ps1 -MaxFolds 1
```
