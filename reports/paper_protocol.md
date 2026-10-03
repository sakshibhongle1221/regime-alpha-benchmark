# Regime Alpha Benchmark Paper Protocol

## Protocol Status

This document began as the paper-facing protocol after Phase 23 and is synchronized in Phase 44 with the repaired development benchmark and the completed locked external holdout. Future experiments should change it only through a named protocol phase, not casually while running models.

Protocol version: Phase 44

Primary working title:

```text
HMM-Guided Contrastive Representations for Regime-Conditioned Financial Alpha Modeling
```

## Central Research Question

Do HMM-guided contrastive regime representations improve regime-conditioned financial alpha modeling compared with vanilla contrastive regimes, classical raw-feature HMM regimes, and a global no-regime LightGBM baseline under financial labels, purged walk-forward validation, transaction costs, robustness checks, statistical tests, and fold-local interpretability?

## Contribution Boundary

The project does not claim to invent HMMs, contrastive learning, LightGBM, triple-barrier labels, or market-regime modeling.

The contribution is a controlled empirical benchmark and model-side intervention:

1. A fair regime-conditioned alpha benchmark using common financial labels, common folds, common transaction costs, and common test rows.
2. Evidence that vanilla contrastive regimes are structurally stable but weakly aligned with classical sequential regimes and downstream alpha usefulness.
3. An HMM-guided contrastive training objective that uses classical state sequences as weak supervision while still treating them as proxy states, not ground truth.
4. A repaired downstream fold-local test showing that guided embeddings plus HMM assignment do not currently support robust positive alpha, while the later locked holdout gives limited relative IC/Sharpe support under a prewritten rule.
5. Robustness, statistical, interpretability, and locked-holdout artifacts that make the finding auditable rather than purely anecdotal.

## Dataset Freeze

Current paper dataset:

| Field | Value |
|---|---|
| Symbols | BTCUSDT, ETHUSDT |
| Bar frequency | 1 hour |
| Earliest OHLCV timestamp | 2024-04-26 16:30:00+05:30 |
| Latest OHLCV timestamp | 2026-04-26 15:30:00+05:30 |
| OHLCV rows per symbol | 17,520 |
| Feature rows per symbol | 17,460 |
| Primary label | `tb_label_8h` |
| Primary return column | `forward_return_8h` |
| Primary universe | BTCUSDT+ETHUSDT |

Raw data files and DuckDB databases are local research inputs and are not committed.

## Feature Freeze

The paper benchmark uses 22 engineered hourly features:

```text
ret_1h, ret_5h, ret_15h, ret_60h,
vol_5h, vol_20h, vol_of_vol,
amihud, volume_zscore,
ret_autocorr, spread_proxy, ofi_proxy,
rsi_14, gk_vol, skewness, kurtosis,
macd_signal, bband_pct_b, atr_14,
close_vs_vwap, log_vol_trend, ret_dispersion
```

Feature changes after this protocol should be treated as a new experiment family because they change both regime discovery and alpha modeling.

## Target Freeze

Primary paper target:

```text
tb_label_8h
```

Secondary diagnostic targets:

```text
dir_label_4h, dir_label_8h, dir_label_24h
tb_label_4h, tb_label_8h, tb_label_24h
forward_return_4h, forward_return_8h, forward_return_24h
vol_adj_return_4h, vol_adj_return_8h, vol_adj_return_24h
```

The paper should report the primary target first. Secondary horizons belong in robustness or appendix tables.

## Validation Freeze

Predictive claims must use:

| Field | Value |
|---|---|
| Validation style | Expanding walk-forward |
| Initial training | 6 months, 4,320 hourly bars |
| Test step | 1 month, 720 hourly bars |
| Embargo | 5 days, 120 hourly bars |
| Primary label-horizon purge | 8 bars |
| Main OOS rows | 25,920 per method |
| Transaction cost | 10 bps per trade unless stress-testing |

Offline/global regime files may be used for descriptive plots. They must not be used as predictive evidence unless explicitly labeled as offline/global.

## Methods Freeze

Primary comparison methods:

| Method | Role |
|---|---|
| `global_lgbm` | No-regime baseline |
| `regime_lgbm_hmm` | Classical raw-feature HMM baseline |
| `regime_lgbm_kmeans` | Non-sequential clustering baseline |
| `regime_lgbm_vol_bucket` | Simple volatility-state baseline |
| `regime_lgbm_contrastive` | Vanilla learned-regime baseline |
| `regime_lgbm_contrastive_hmm` | Vanilla embedding plus sequential assignment |
| `regime_lgbm_hmm_guided_gmm` | HMM-guided embedding plus GMM assignment |
| `regime_lgbm_hmm_guided_hmm` | Main proposed method |

The current proposed method is `regime_lgbm_hmm_guided_hmm`.

## Metrics Freeze

Primary predictive/economic metrics:

```text
IC, Sharpe, drawdown, total_return, turnover, n_trades, n_test_rows
```

Required statistical diagnostics:

```text
fold-level paired tests, bootstrap confidence intervals,
DM-style NLL forecast-loss diagnostics, multiple-testing correction,
Probabilistic Sharpe Ratio
```

Required regime-quality diagnostics:

```text
silhouette, average duration, transition diagonal, entropy,
NMI/ARI versus raw-feature HMM reference, purity versus raw-feature HMM reference
```

Required interpretability diagnostics:

```text
fold-local LightGBM gain/split importance, capped SHAP summary,
feature-family attribution by method/regime
```

## Permitted Claims

The paper may claim:

1. Vanilla contrastive regimes underperform raw-feature HMM regimes in the current benchmark.
2. Sequential assignment matters: HMM assignment is more useful than GMM assignment on learned embeddings.
3. HMM-guided weak supervision strongly improves structural agreement with the raw-feature HMM reference.
4. The frozen guided-HMM candidate satisfies the Phase 43B locked relative IC/Sharpe rule against the registered global LightGBM and raw-feature HMM references.
5. The repaired and locked evidence does not support a profitable or deployable strategy.
6. Fold-local interpretability indicates economically plausible drivers dominated by volatility state, momentum/autocorrelation, and distribution shape.
7. Crypto-20 structural diagnostics and repaired fold-local alpha results must be separated: structural transfer is useful, but repaired downstream alpha remains weak/negative.

## Forbidden Claims

The paper must not claim:

1. HMM states are ground-truth market regimes.
2. The strategy is profitable or deployable.
3. The guided-HMM method is statistically dominant over raw-feature HMM at 5% significance.
4. The Crypto-20 downstream result establishes statistically proven alpha, calibration, or portfolio-performance dominance.
5. The interpretability results are causal explanations.
6. Offline/global regime artifacts prove predictive performance.

## Decision Gates

Phase 25 ablations and Phase 26 paper claim tests are mandatory before paper submission. Both are now complete in the current protocol.

Historical Phase 27/29 draft work created the paper skeleton before the later validation repair. The current Phase 44 draft supersedes the earlier positive-leaning skeleton and must be treated as the active paper path.

The historical Phase 26 read did not show statistically decisive guided-HMM dominance over raw-feature HMM. Phases 36 and 37 completed the first Crypto-20 predictive test and confirmed that structural transfer does not establish predictive, calibration, or portfolio dominance. Phase 39R/40 repaired the calendar-overlap issue and produced a weak/negative development benchmark. Phase 43B then completed a single locked external holdout: the frozen guided-HMM candidate satisfies the prewritten relative IC/Sharpe rule, but negative Sharpe and total return block any tradable-strategy claim. The default paper is therefore a validation-and-mechanism paper with limited locked relative support, not a profitable-alpha paper.

The multi-asset gate is a downstream alpha-claim gate. It does not prohibit structural generalization experiments that test whether the representation-learning objective transfers to a pre-specified wider universe. Those experiments must be labeled as structural diagnostics, must not be used as evidence of predictive alpha improvement, and must still be followed by a fold-local downstream alpha benchmark before the paper can claim multi-asset alpha generalization.

Phase 44 is represented by `paper/main.md`, `reports/phase44_paper_readiness_package.md`, `models/phase44_paper_evidence_matrix.csv`, `models/phase44_submission_risk_register.csv`, `reports/paper_artifact_map.csv`, and `reports/paper_submission_checklist.md`.

Phase 45 is represented by `paper/phase45_venue_ready_manuscript.md`, `reports/phase45_venue_manuscript_package.md`, `reports/phase45_external_research_audit.md`, `reports/phase45_reproducibility_appendix.md`, `reports/phase45_submission_checklist.md`, `models/phase45_table_plan.csv`, `models/phase45_figure_plan.csv`, `models/phase45_claim_to_section_map.csv`, and `models/phase45_venue_requirement_audit.csv`.

Phase 46 is represented by `paper/phase46_submission_manuscript.md`, `paper/phase46_acm_sigconf_skeleton.tex`, `reports/phase46_final_research_completion.md`, `reports/phase46_submission_readiness_audit.md`, `reports/phase46_anonymity_audit.md`, `reports/phase46_reviewer_response_pack.md`, `reports/phase46_final_submission_todo.md`, `models/phase46_section_budget.csv`, `models/phase46_submission_gate_matrix.csv`, `models/phase46_anonymity_audit.csv`, `models/phase46_final_claim_audit.csv`, and `models/phase46_reviewer_objection_matrix.csv`.

The repository evidence package is now research-complete. The next default step is human-facing submission production: final LaTeX/PDF formatting, current venue-rule verification, citation cleanup, figure drawing, anonymity audit, artifact archive/DOI decision, and advisor/reviewer feedback, not broad experiment expansion.

Reviewer-facing caveats must stay explicit in the paper draft:

1. Development-observed evidence and locked-holdout evidence have different claim rights.
2. The locked holdout is already spent once and cannot be reused for model rescue.
3. The frozen guided-HMM candidate has negative locked Sharpe and negative locked total return.
4. A secondary diagnostic method cannot replace the frozen final candidate after locked-holdout inspection.

## Phase 38 Evidence And Data-Role Reset

Phases 36 and 37 completed the first Crypto-20 downstream and statistical evaluation, but those inspected results later became development-observed audit history. The repaired Phase 39R/40 evidence is the valid development benchmark. It is weak/negative and does not support robust guided-method dominance or positive alpha.

All BTC/ETH and Crypto-20 results inspected through Phase 37 are now `development_observed` according to `reports/data_role_registry.csv`. They may support model development and historical comparison, but they may not be described as an untouched final test.

The fully fold-local learned pipeline and locked external holdout are now complete for the current paper path. Crypto-50 expansion, unrestricted architecture search, product deployment, and same-holdout rescue tuning remain blocked unless a new pre-registered replication protocol is created first.

## Phase 44 Locked-Holdout And Paper Package Update

Phase 44 is the current paper-readiness layer. The allowed locked claim is narrow: on the registered external crypto holdout, the frozen `regime_lgbm_hmm_guided_hmm` candidate satisfies the prewritten relative IC/Sharpe rule against `global_lgbm` and `regime_lgbm_hmm`. The locked result does not support positive tradable alpha, candidate switching, or same-holdout retuning.

## Phase 45 Venue Manuscript Update

Phase 45 converts the Phase 44 evidence into a venue-facing manuscript package. It is still a packaging layer, not a model-improvement layer. The manuscript may use the phrase "limited locked relative support" but must continue to state that the paper does not claim a tradable strategy and that the same locked holdout cannot be reused for model rescue.

The Phase 45 external audit adds conservative ICAIF/ACM-style constraints: compact self-contained paper, double-blind anonymity, official `acmart`/`sigconf` formatting unless the current venue overrides it, and no artifact-availability claim before a persistent archive or DOI exists.

## Phase 46 Final Research Completion Update

Phase 46 closes the repository-side research package. It creates a final submission manuscript source, ACM skeleton, eight-page section budget, anonymity audit, reviewer objection pack, submission gate, and final TODO list. It records that the package is not ready for blind external submission until the compiled PDF, citations, figures, metadata/anonymity audit, and artifact DOI/archive decision are finished.

Future work should focus on final formatting, current venue rule re-check, citations, figure/table cleanup, anonymity audit, artifact archive/DOI decision, and reviewer feedback. It should not run new model search unless a new pre-registered external dataset or replication protocol is created first.

