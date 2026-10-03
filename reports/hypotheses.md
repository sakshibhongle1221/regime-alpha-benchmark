# Regime Alpha Benchmark Hypotheses

## Purpose

This file freezes the paper hypotheses after Phase 23. Every future experiment should map to one or more rows in this table.

## Hypothesis Table

| ID | Hypothesis | Current Evidence | Current Status | Next Test |
|---|---|---|---|---|
| H1 | Sequential consistency matters for useful regime assignment. | Raw-feature HMM beats vanilla contrastive-GMM on the original benchmark; contrastive-HMM improves over contrastive-GMM; guided embeddings need HMM assignment to become useful; Phase 25/26 support the assignment-layer mechanism. | Supported | Include as the central mechanism claim in the paper draft. |
| H2 | Vanilla contrastive regimes are structurally stable but not alpha-relevant enough. | Phase 16 shows low HMM agreement for vanilla contrastive regimes; downstream IC is worse than raw-feature HMM. | Supported | Include vanilla contrastive as a fixed ablation reference. |
| H3 | HMM-guided contrastive supervision improves learned regime structure. | Phase 19B guided-HMM reaches `HMM NMI = 0.869` and `HMM purity = 0.957`, far above vanilla contrastive alignment; Phase 25 keeps objective guidance as supported for the guided-HMM downstream path but mixed structurally. | Strongly supported structurally | Include as structural evidence; avoid calling HMM states ground truth. |
| H4 | Better guided regime structure improves downstream alpha. | Phase 20 guided-HMM has the best primary point estimates and positive Sharpe; Phase 26 keeps the edge over raw HMM directional but statistically inconclusive. | Directionally supported, statistically inconclusive | Treat as promising evidence, not a dominance claim. |
| H5 | Guided-HMM robustness survives realistic trading assumptions. | Phase 21 stress grid shows guided-HMM as the most frequent winner across IC, Sharpe, drawdown, and total return under threshold/cost/period variation. | Supported on primary BTC+ETH 8h setup | Keep stress result as robustness evidence; do not overgeneralize. |
| H6 | Guided-HMM regimes are economically interpretable. | Phase 23 shows fold-local feature attribution dominated by volatility state, momentum/autocorrelation, and distribution-shape features. | Supported diagnostically | Include interpretability in paper; avoid causal language. |
| H7 | The structural regime-learning mechanism transfers beyond BTC/ETH. | Phase 31-35 support structural transfer. Phase 36/37 give guided-HMM the highest mean fold IC, but its edge over raw HMM is non-significant (`p=0.840`) and its probability calibration is worse than global LightGBM. | Structurally supported, predictive dominance not supported | Discuss structural transfer separately from portfolio-level and calibration claims. |
| H8 | The structural and assignment-layer findings survive fully fold-local encoder training. | The original Phase 39 run failed the later cross-asset calendar audit. The repaired full development run and Phase 40 adjudication are complete and show weak/inconclusive downstream alpha rather than robust guided-method dominance. | Diagnostic / not supported as an alpha claim | Do not reuse invalidated metrics as evidence; use Phase 41 only for bounded development improvements selected inside training/validation boundaries. |
| H9 | Validation-only calibration and soft posterior gating improve probability quality without sacrificing ranking IC. | Phase 37 identifies worse guided-HMM NLL despite weak positive IC direction. | Open | Test only after the fold-local baseline passes; select using inner validation. |
| H10 | The selected conclusion transfers to an untouched asset or temporal holdout. | Phase 43B locked external evaluation gives limited relative support to the frozen guided-HMM candidate versus global LightGBM and raw-feature HMM, but locked Sharpe and total return remain negative. | Limited support; no tradable-alpha claim | Use as the final confirmatory evidence for the current paper package; do not retune on the same locked holdout. |

## Phase 25 Status Update

The minimal ablation suite supports the assignment-layer mechanism most strongly: HMM assignment improves the guided learned-regime path more reliably than GMM assignment. Objective guidance is useful for the guided-HMM downstream path, while the current time-frequency prototype remains a negative or inconclusive ablation.

## Phase 26 Status Update

The paper statistical refresh keeps the mechanism claim strong and the alpha dominance claim cautious. HMM assignment on guided embeddings is raw-suggestive on fold-level IC versus guided-GMM (`p=0.075`) and improves all focused point-estimate metrics. Guided-HMM versus raw-feature HMM remains directionally supported, not statistically significant.

## Primary Paper Hypothesis

The primary paper hypothesis is H4:

```text
HMM-guided contrastive regime representations, when paired with sequential HMM assignment, improve regime-conditioned alpha modeling versus raw-feature HMM and vanilla learned-regime baselines under purged walk-forward validation.
```

Current status:

```text
Directionally supported by point estimates and stress robustness, not yet statistically conclusive at fold level.
```

## Secondary Hypotheses

H1, H2, H3, and H6 are already strong enough to support the paper narrative.

H5 supports robustness on the primary setup.

H7 is now part of the paper's multi-asset mechanism evidence, with predictive dominance explicitly unsupported.

## Phase 38 Hypothesis Hierarchy

H8 is the next validity hypothesis and must be resolved before a new model-improvement claim. H9 is the bounded development hypothesis motivated by the Phase 37 calibration failure. H10 is the confirmatory generalization hypothesis and cannot be tested until one configuration is frozen.

The historical H4 result remains directionally supported in the BTC/ETH pilot and unsupported as a broad Crypto-20 dominance claim. A failed H8, H9, or H10 test must remain in the final evidence record and must not trigger tuning on the same outer or locked data.

## Phase 39R/40 Status Update

The original Phase 39 run is invalidated by cross-asset calendar overlap caused by per-symbol positional folds. The repaired implementation aligns all 20 symbols to one timestamp index, passes all 16 global calendar boundaries, and completed the repaired 16-fold classical plus neural/guided development benchmarks with equal coverage across eight methods.

Phase 40 statistically adjudicates those repaired outputs. H8 is now resolved diagnostically: the fully fold-local repaired benchmark does not support robust guided-method dominance or positive-alpha claims. It remains useful because it gives the project a clean, reviewer-safe development baseline and a disciplined reason for bounded Phase 41 improvements.

## Phase 41 Status Update

Phase 41 registers H9 candidates but does not yet claim that H9 is supported. The registered families are probability temperature scaling, class-prior blending, posterior-temperature soft gating, global-regime shrinkage, and score-threshold control. Candidate parameters must be selected only on inner chronological validation inside each outer fold. Phase 40 outer-test metrics and repaired statistical outputs are explicitly forbidden as selection inputs. Score-threshold control is registered as execution-control infrastructure but deferred from the Phase 41B probability-calibration/soft-gating run.

Phase 41B adds and completes the first full H9 implementation on the global/classical ladder for probability calibration and soft gating. Candidate parameters are selected only by inner validation across all 16 folds. H9 is not supported by this first run: corrected IC/Sharpe dominance remains unsupported, and the repaired alpha conclusion remains weak/negative. This does not close all possible calibration or execution-control research, but it blocks any claim that the first bounded calibration/soft-gating pass fixed the project's alpha problem.

## Phase 42 Status Update

Phase 42 does not introduce a new model hypothesis. It explains the weak repaired-alpha result using development-observed diagnostics: execution-cost sensitivity, threshold sensitivity, regime transition behavior, cross-asset fragility, and feature-family target alignment. The result supports a diagnostic interpretation: the current alpha weakness is not a single calibration bug, and it should not be converted into a tradability claim.

## Phase 43A Status Update

Phase 43A freezes H10 before any locked outcome is inspected. The final candidate is `regime_lgbm_hmm_guided_hmm`, compared primarily against `global_lgbm` and `regime_lgbm_hmm`. Phase 41B calibration, soft-gating, score-threshold control, new model search, new feature selection, and new label/horizon selection are excluded. The next step is a single locked holdout evaluation; success or failure must be reported without same-holdout retuning.

## Phase 43B Registration Status Update

Phase 43B adds the locked-holdout registration gate. It checks whether the next pre-ranked external assets have enough OHLCV, feature, and target coverage before any frozen-model outcome is evaluated. If the gate is blocked, the hypothesis remains open and the next action is data ingestion/build only, not model tuning.

## Phase 43B Locked Evaluation Status Update

The locked external evaluation is complete. H10 receives limited confirmatory support under the pre-specified relative rule: the frozen `regime_lgbm_hmm_guided_hmm` candidate improves over `global_lgbm` and `regime_lgbm_hmm` on locked mean asset IC and has non-worse Sharpe with equal coverage. The result does not support a profitable/tradable alpha claim because locked Sharpe and total return remain negative.

## Phase 44 Paper-Readiness Status Update

Phase 44 packages the H10 result for writing. It creates the paper-facing evidence matrix, submission risk register, and updated manuscript draft. It is not a new model-improvement phase and does not authorize candidate switching, threshold rescue, or reuse of the same locked holdout.

## Phase 45 Venue Manuscript Status Update

Phase 45 does not introduce a new hypothesis. It packages the current evidence into a venue-facing manuscript source, table/figure plan, venue requirement audit, external research audit, reproducibility appendix, and claim-to-section map. It preserves the H10 interpretation: limited locked relative support is allowed, but profitable/tradable alpha, candidate switching, and same-holdout retuning remain unsupported or forbidden. It also records that artifact availability should not be claimed before a persistent archive or DOI exists.

## Phase 46 Final Research Completion Status Update

Phase 46 does not introduce a new hypothesis. It closes the repository-side research package by creating the final submission manuscript source, ACM skeleton, section budget, anonymity audit, reviewer response pack, final claim audit, and submission gate. H10 remains limited: the frozen guided-HMM candidate receives limited locked relative support, while profitable/tradable alpha remains unsupported. Phase 46 is not an external-submission claim; final PDF formatting, citations, figures, metadata/anonymity audit, and artifact DOI/archive decisions remain before blind submission.

## Phase 47 Submission Build Status Update

Phase 47 does not introduce a new hypothesis. It converts the Phase 46 package into a build-oriented anonymous LaTeX draft, BibTeX reference file, compiled local PDF, table/figure manifests, source anonymity audit, manuscript build audit, and camera-ready gap list. H10 remains unchanged: limited locked relative support is allowed, while profitable/tradable alpha, candidate switching, and same-holdout retuning remain unsupported or forbidden. Phase 47 is not an external-submission claim; venue-template verification, LaTeX/BibTeX warning cleanup, final figure/citation polish, PDF/metadata anonymity audit, artifact archive/DOI decision, and human review remain before blind submission.

## Phase 48 Paper Polish Status Update

Phase 48 does not introduce a new hypothesis. It polishes the compiled draft into a review-ready manuscript package with a Phase 48 LaTeX source, BibTeX file, compiled local PDF, claim traceability audit, section budget, LaTeX/PDF warning audit, and reviewer reading guide. H10 remains unchanged: limited locked relative support is allowed, while profitable/tradable alpha, deployment, candidate switching, and same-holdout rescue claims remain blocked. Phase 48 is still not an external-submission claim; final venue-template verification, PDF/metadata anonymity audit, artifact archive/DOI decision, and human/advisor review remain before blind submission.

## Phase 49 Final Submission Closure Status Update

Phase 49 does not introduce a new hypothesis. It closes the remaining repository-side submission gaps by adding final F1/F2 conceptual figures, expanded time-series representation-learning citations, a compiled submission-ready manuscript target, PDF metadata audit, LaTeX warning-state audit, and an artifact-release checklist. H10 remains unchanged: limited locked relative support is allowed, while profitable/tradable alpha, deployment, candidate switching, and same-holdout rescue claims remain blocked. Phase 49 is still not an external-submission claim; human/advisor visual layout review, active venue-template verification, and the artifact DOI decision remain before blind submission.

## Claim Language

Use:

```text
The guided-HMM method produced the strongest point estimates and stress robustness in the BTC/ETH benchmark, while fold-level statistical dominance over raw-feature HMM remains inconclusive.
```

Avoid:

```text
The guided-HMM method beats HMM.
The learned regimes are true market regimes.
The strategy is profitable.
The method generalizes to all markets.
```
