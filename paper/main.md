# HMM-Guided Contrastive Representations for Regime-Conditioned Financial Alpha Modeling

## Paper Status

Phase 44 paper-readiness draft. The project has completed the repaired common-calendar development benchmark and one registered locked external holdout. This draft is paper-facing prose, not a new experiment.

## Abstract

Regime-conditioned financial alpha models are attractive because predictive relationships can change across trend, stress, and choppy market states. Yet regime labels are latent, unstable, and easy to overfit. This project studies whether contrastive time-series representations become more useful when guided by classical Hidden Markov Model state structure. The benchmark compares global LightGBM, raw-feature HMM regimes, KMeans regimes, volatility buckets, vanilla contrastive regimes, and HMM-guided contrastive regimes under triple-barrier labels, purged walk-forward validation, transaction costs, statistical adjudication, and reproducibility gates. A key validation repair invalidated earlier positional-fold evidence and replaced it with strict common-calendar fold-local evaluation. Under the repaired Crypto-20 development benchmark, alpha evidence is weak and does not support a broad positive-performance claim. Under the registered Phase 43B external holdout, the frozen guided-HMM candidate satisfies the prewritten relative IC/Sharpe rule against global LightGBM and raw-feature HMM references: mean asset IC is 0.0007 versus -0.0042 and -0.0024, and Sharpe is -0.3691 versus -1.2810 and -0.9538. However, the final candidate still has negative locked total return (-6.6%) and negative Sharpe, so the paper does not claim a tradable strategy. The contribution is a research-grade benchmark and mechanism boundary: sequential regime discipline can help learned representations, but structural improvement does not automatically become profitable alpha.

## 1. Introduction

Financial markets are non-stationary. A signal that works during calm trend-following periods can fail during stress, high volatility, or transition periods. Regime-conditioned alpha modeling tries to address this by allowing models to behave differently across latent market states. The hard part is that those states are not directly observed.

Classical HMMs offer sequential discipline and persistent state assignments, but they depend on raw features and distributional assumptions. Contrastive encoders can learn richer nonlinear representations, but memoryless clustering of embeddings can produce regimes that look smooth while adding little downstream alpha value. This project studies a hybrid: use HMM state sequences as weak supervision for a contrastive encoder, then test whether sequential assignment on the learned representation improves regime-conditioned alpha modeling.

The important research lesson is not simply whether one method wins a backtest. Earlier versions of this project produced exciting results, but a later audit found that positional multi-asset folds overlapped in calendar time. Those results are now retained only as audit history. The repaired pipeline uses common-calendar fold-local validation, data-role separation, frozen candidate rules, and a locked external holdout.

## 2. Research Question

Does HMM-guided contrastive regime learning produce more useful regime-conditioned alpha models than global LightGBM, raw-feature HMM regimes, vanilla contrastive regimes, and simple clustering/volatility baselines under leakage-safe financial validation?

The paper-safe answer is conditional:

- Yes, the locked holdout gives limited support to the frozen guided-HMM candidate under the prewritten relative rule.
- No, the evidence does not support a profitable or deployable trading strategy.
- No, the project may not switch to a secondary diagnostic method after seeing the locked holdout, even though guided-GMM has higher locked IC (0.0072) but worse Sharpe (-1.7041) and return (-27.2%).

## 3. Contributions

1. A repaired common-calendar regime-conditioned alpha benchmark for crypto assets.
2. A documented validation failure and repair, showing how a plausible multi-asset backtest can become invalid through calendar overlap.
3. An HMM-guided contrastive regime-learning path that treats HMM states as weak proxy supervision, not ground truth.
4. A one-shot locked external holdout with prewritten claim rules.
5. A claim-control layer that separates relative method evidence from tradable-alpha claims.

## 4. Data and Validation

The development benchmark uses a repaired Crypto-20 common-calendar panel. The locked external holdout uses 10 registered external crypto symbols selected before model outcome inspection. Phase 43B evaluates 18 folds and 129,600 out-of-sample rows per method. The validation protocol uses fold-local fitting, purged walk-forward splits, embargo spacing, transaction costs, and equal method coverage.

The paper treats development-observed evidence and locked-holdout evidence differently. Development evidence can motivate the mechanism and explain failure modes. Locked evidence is the only confirmatory claim source, and it has already been spent once.

## 5. Methods

The benchmark includes:

- Global LightGBM with no regimes.
- Raw-feature HMM regimes.
- KMeans regimes.
- Volatility buckets.
- Vanilla contrastive embeddings with GMM or HMM assignment.
- HMM-guided contrastive embeddings with GMM or HMM assignment.

The frozen final candidate is `regime_lgbm_hmm_guided_hmm`.

## 6. Main Locked Result

| Method | Mean Asset IC | Sharpe | Total Return | Drawdown | Rows |
| --- | --- | --- | --- | --- | --- |
| HMM-guided contrastive-HMM + regime LightGBM | 0.0007 | -0.3691 | -6.6% | -11.4% | 129600 |
| HMM-guided contrastive-GMM + regime LightGBM | 0.0072 | -1.7041 | -27.2% | -29.4% | 129600 |
| Raw-feature HMM + regime LightGBM | -0.0024 | -0.9538 | -16.0% | -18.0% | 129600 |
| Global LightGBM | -0.0042 | -1.2810 | -16.4% | -16.7% | 129600 |
| Vanilla contrastive-GMM + regime LightGBM | -0.0002 | 0.2726 | 3.6% | -9.9% | 129600 |
| Vanilla contrastive-HMM + regime LightGBM | -0.0012 | -0.7277 | -10.9% | -15.4% | 129600 |
| KMeans + regime LightGBM | -0.0020 | -0.3003 | -5.5% | -13.8% | 129600 |
| Volatility buckets + regime LightGBM | -0.0017 | -0.6418 | -11.1% | -18.3% | 129600 |

The locked result satisfies the prewritten relative IC/Sharpe rule against the two primary references. This means the frozen candidate is better than the chosen references on the specific registered comparison. It does not mean the strategy is profitable.

## 7. Evidence Interpretation

| evidence_block | data_role | finding | claim_boundary |
| --- | --- | --- | --- |
| validation_repair | development_observed | Original positional-fold evidence is retained only as audit history; repaired common-calendar fold-local evidence is the valid development benchmark. | Do not cite invalidated positional-fold runs as predictive evidence. |
| repaired_crypto20_development | development_observed | Final candidate development mean asset IC=-0.0119, Sharpe=-0.7620; best development mean asset IC method is regime_lgbm_contrastive at -0.0031. | Development results motivate interpretation, not final-test confirmation. |
| development_statistical_adjudication | development_observed | Final candidate has 16 folds, IC bootstrap CI [-0.0295, -0.0055], Sharpe CI [-3.1788, 0.2107]. | No corrected dominance or robust positive-alpha claim. |
| execution_and_mechanism_diagnostics | development_observed | Final candidate has 2/16 positive-return stress cells across Phase 42 diagnostics. | Diagnostics explain fragility; they are not a new tuned model. |
| locked_external_holdout | locked_registered_unobserved | Frozen final candidate mean asset IC=0.0007, Sharpe=-0.3691, return=-6.6%; global reference IC=-0.0042, Sharpe=-1.2810; raw-HMM reference IC=-0.0024, Sharpe=-0.9538. | Tradable-alpha claim is not_supported; no candidate switching or same-holdout retuning. |

The most honest interpretation is a limited-support mechanism paper:

- Sequential assignment matters.
- HMM-guided representation learning can improve the relative ranking behavior of regimes.
- Development and locked evidence still show fragile economic performance.
- A negative/limited result is scientifically valuable because it prevents an invalid positive trading claim.

## 8. Limitations

- HMM states are proxy states, not ground truth.
- Crypto markets are not equities, FX, or options markets.
- The locked holdout supports only the prewritten relative rule, not a broad dominance claim.
- The final candidate has negative locked Sharpe and negative locked total return.
- The same locked holdout cannot be reused for model rescue.
- Row-level financial observations are overlapping and should not be treated as independent statistical evidence.

## 9. Conclusion

The project now has a defensible paper story: HMM-guided contrastive regimes receive limited locked-holdout support as a relative modeling mechanism, while the larger trading-alpha claim is not supported. This is not a failed project. It is a stronger research contribution than an overfit backtest because it shows the full chain: build, audit, invalidate, repair, freeze, evaluate once, and report honestly.
