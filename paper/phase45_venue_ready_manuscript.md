# HMM-Guided Contrastive Representations for Leakage-Safe Regime-Conditioned Crypto Alpha Evaluation

## Venue Note

Phase 45 targets an ACM acmart-style proceedings manuscript. This file is a structured manuscript source, not the final LaTeX/PDF conversion. It keeps all venue-facing claims synchronized with the frozen Phase 43B locked-holdout evidence.

Before submission, Phase 46 must verify the current venue call. The conservative working assumptions are: ACM `acmart`/`sigconf` formatting, double-blind anonymity, compact self-contained manuscript, no dependency on supplementary material, and no author-identifying information in the review PDF.

## Abstract

Regime-conditioned alpha models are attractive because financial relationships can change across trend, stress, and transition periods. However, regime labels are latent and easy to overfit, especially in multi-asset crypto backtests. This paper studies whether Hidden Markov Model guided contrastive representations improve regime-conditioned alpha modeling under leakage-safe validation. The project compares global LightGBM, raw-feature HMM regimes, KMeans regimes, volatility buckets, vanilla contrastive regimes, and HMM-guided contrastive regimes. A central contribution is a validation repair: earlier positional-fold evidence was invalidated after discovering cross-asset calendar overlap, then replaced with common-calendar fold-local validation, frozen-candidate rules, and a one-shot locked external holdout. On the registered 10-asset external holdout, the frozen HMM-guided contrastive-HMM candidate improves mean asset IC versus global LightGBM (0.0007 versus -0.0042) and raw-feature HMM (0.0007 versus -0.0024), with non-worse Sharpe versus both references (-0.3691 versus -1.2810 and -0.9538). The finding is limited: the locked final candidate still has negative Sharpe and negative total return (-6.6%), so the paper does not claim a tradable strategy. The contribution is a research-grade evaluation framework and a mechanism boundary for sequentially guided regime learning.

## 1. Introduction

Financial machine-learning papers often fail quietly because the validation protocol is easier to overfit than the model. This project takes the opposite path: it preserves the full audit trail, including an initially exciting result that was later invalidated. The corrected contribution is not "we found a profitable crypto strategy." The corrected contribution is "we built a regime-learning benchmark that can detect when its own evidence is not strong enough."

The core modeling question is whether classical sequential structure can improve neural regime representations. HMMs are useful because they impose temporal persistence, but they are limited by raw-feature assumptions. Contrastive encoders can learn richer representations, but unconstrained clustering may not produce sequentially meaningful regimes. HMM-guided contrastive learning combines these ideas by using HMM states as weak proxy supervision, then testing downstream alpha under strict walk-forward validation.

## 2. Related Work Positioning

The manuscript should position the project at the intersection of:

- financial machine learning and purged walk-forward validation;
- latent regime models and Hidden Markov Models;
- contrastive representation learning for time series;
- benchmark reproducibility and artifact-centered empirical finance.

The paper should be framed as a validation-and-mechanism study. That framing is stronger than a profitability claim because the locked evidence explicitly blocks positive tradable alpha.

## 3. Data, Labels, and Data Roles

The project separates data roles:

- development-observed Crypto-20 evidence is used for repair, diagnosis, and candidate freezing;
- the Phase 43B external holdout is registered and frozen before outcome inspection;
- the same locked holdout cannot be reused for model rescue.

The locked external holdout contains 10 external assets, 18 folds, and 129,600 out-of-sample rows per method. The final manuscript should report data quality, symbol selection, target construction, transaction costs, and fold-local fitting clearly enough for reviewer reproduction.

## 4. Methods

The benchmark compares a no-regime global model, simple regime baselines, classical HMM regimes, vanilla contrastive regimes, and HMM-guided contrastive regimes. The frozen final candidate is `regime_lgbm_hmm_guided_hmm`. It was frozen before locked-holdout outcomes were inspected.

## 5. Validation Protocol

The key validation repair is common-calendar fold-local evaluation. Feature scaling, HMM weak-supervision fitting, contrastive pair construction, encoder training, regime assignment, and downstream alpha models are all fit inside authorized training intervals. This prevents the previous calendar-overlap problem from becoming predictive evidence.

## 6. Results

### 6.1 Locked External Holdout

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

The frozen final candidate satisfies the prewritten relative IC/Sharpe rule against the two primary references. This is limited locked relative support. It is not a tradable-alpha claim.

### 6.2 Why the Higher-IC Guided-GMM Row Does Not Replace the Final Candidate

The guided-GMM diagnostic row has higher locked mean asset IC (0.0072), but it was not the frozen final candidate and has worse locked Sharpe (-1.7041) and total return (-27.2%). Replacing the candidate after seeing locked outcomes would be post-hoc selection.

## 7. Evidence and Claim Boundaries

| evidence_block | data_role | finding | claim_boundary |
| --- | --- | --- | --- |
| validation_repair | development_observed | Original positional-fold evidence is retained only as audit history; repaired common-calendar fold-local evidence is the valid development benchmark. | Do not cite invalidated positional-fold runs as predictive evidence. |
| repaired_crypto20_development | development_observed | Final candidate development mean asset IC=-0.0119, Sharpe=-0.7620; best development mean asset IC method is regime_lgbm_contrastive at -0.0031. | Development results motivate interpretation, not final-test confirmation. |
| development_statistical_adjudication | development_observed | Final candidate has 16 folds, IC bootstrap CI [-0.0295, -0.0055], Sharpe CI [-3.1788, 0.2107]. | No corrected dominance or robust positive-alpha claim. |
| execution_and_mechanism_diagnostics | development_observed | Final candidate has 2/16 positive-return stress cells across Phase 42 diagnostics. | Diagnostics explain fragility; they are not a new tuned model. |
| locked_external_holdout | locked_registered_unobserved | Frozen final candidate mean asset IC=0.0007, Sharpe=-0.3691, return=-6.6%; global reference IC=-0.0042, Sharpe=-1.2810; raw-HMM reference IC=-0.0024, Sharpe=-0.9538. | Tradable-alpha claim is not_supported; no candidate switching or same-holdout retuning. |

## 8. Limitations

- The final candidate has negative locked Sharpe and negative locked total return.
- HMM states are weak proxy labels, not ground-truth market regimes.
- The locked result supports only the prewritten relative rule.
- Row-level overlapping financial samples are not independent evidence units.
- The same locked holdout cannot be reused for labels, thresholds, features, architecture search, or candidate switching.
- The project does not claim a profitable or deployable trading strategy.

## 9. Reproducibility and Artifact Availability

The reproducibility package should include the code, curated summary CSV files, run scripts, claim registry, artifact manifest, and research-grade check report. Bulky row-level final predictions and raw data are intentionally excluded from GitHub when they are reproducible or too large.

The project is close to ACM-style artifact functionality because it includes an inventory, executable checks, curated artifacts, and validation reports. It should not claim permanent artifact availability until a frozen release is archived in a persistent repository with a DOI or equivalent identifier.

## 10. Conclusion

HMM-guided contrastive regimes receive limited locked-holdout relative support, but not profitable-alpha support. The strongest paper contribution is the disciplined empirical framework: the project found an exciting result, invalidated it, repaired the validation protocol, froze a final candidate, spent one locked holdout, and reported the boundary honestly.
