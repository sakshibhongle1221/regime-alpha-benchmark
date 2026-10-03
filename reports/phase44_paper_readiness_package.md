# Phase 44 Paper-Readiness Package

## Status

Phase 44 converts the completed repaired-development and locked-external-holdout evidence into a paper-facing package. It does **not** tune models, change labels, change candidate choice, or touch final/locked evaluation data.

## Paper Thesis

The strongest paper path is a research-grade negative/limited-support paper:

> Sequential HMM discipline can make contrastive regime representations more useful, but the repaired and locked evidence does not support a profitable trading claim. The contribution is the validation repair, the benchmark, the mechanism boundary, and the one-shot locked-holdout adjudication.

This is stronger and safer than trying to rescue a positive result after the locked holdout.

## Locked-Holdout Result Snapshot

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

## Evidence Matrix

| evidence_block | data_role | main_artifacts | finding | paper_use | claim_boundary |
| --- | --- | --- | --- | --- | --- |
| validation_repair | development_observed | reports/phase39r_neural_fold_local_results.md; reports/publication_acceptance_gates.md | Original positional-fold evidence is retained only as audit history; repaired common-calendar fold-local evidence is the valid development benchmark. | Use as research-integrity story and validation contribution. | Do not cite invalidated positional-fold runs as predictive evidence. |
| repaired_crypto20_development | development_observed | models/crypto20_repaired_fold_local_experiment_results.csv; models/crypto20_repaired_fold_local_statistical_method_summary.csv | Final candidate development mean asset IC=-0.0119, Sharpe=-0.7620; best development mean asset IC method is regime_lgbm_contrastive at -0.0031. | Show that repaired development evidence is weak/negative and prevents a broad positive-alpha claim. | Development results motivate interpretation, not final-test confirmation. |
| development_statistical_adjudication | development_observed | models/crypto20_repaired_fold_local_statistical_method_summary.csv; reports/phase40_repaired_statistical_adjudication.md | Final candidate has 16 folds, IC bootstrap CI [-0.0295, -0.0055], Sharpe CI [-3.1788, 0.2107]. | Frame statistical power and uncertainty explicitly. | No corrected dominance or robust positive-alpha claim. |
| execution_and_mechanism_diagnostics | development_observed | reports/phase42_interpretation_execution_hardening.md; models/phase42_execution_stress_summary.csv | Final candidate has 2/16 positive-return stress cells across Phase 42 diagnostics. | Explain why the weak alpha result is not rescued by simple execution/calibration tweaks. | Diagnostics explain fragility; they are not a new tuned model. |
| locked_external_holdout | locked_registered_unobserved | models/phase43b_locked_external_experiment_results.csv; reports/phase43b_locked_external_adjudication.md | Frozen final candidate mean asset IC=0.0007, Sharpe=-0.3691, return=-6.6%; global reference IC=-0.0042, Sharpe=-1.2810; raw-HMM reference IC=-0.0024, Sharpe=-0.9538. | Allowed locked claim: relative rule is satisfied. | Tradable-alpha claim is not_supported; no candidate switching or same-holdout retuning. |

## Submission Risk Register

| risk | severity | mitigation | owner_artifact |
| --- | --- | --- | --- |
| Overclaiming profitability | critical | Phase 43B explicitly marks positive tradable alpha as not_supported. | models/phase43b_locked_external_claims.csv |
| Switching from frozen final candidate to higher-IC diagnostic method | critical | Candidate switching after holdout is forbidden; guided-GMM cannot replace guided-HMM after seeing locked outcomes. | reports/phase43b_locked_external_adjudication.md |
| Reintroducing calendar leakage | critical | Common-calendar fold-local validation and research-grade checks must pass before push. | models/research_grade_check_report.md |
| Treating development evidence as final test evidence | high | Data-role language separates development_observed from locked_registered_unobserved. | reports/data_role_registry.csv; reports/publication_acceptance_gates.md |
| Weak/negative result framed as failure instead of contribution | medium | Paper thesis focuses on validation, mechanism boundaries, and limited locked relative support. | reports/phase44_paper_readiness_package.md |
| Venue formatting and citations incomplete | medium | Next phase should convert Markdown to target venue template and finalize related work. | paper/main.md; reports/paper_submission_checklist.md |

## Allowed Claim

The paper may say:

> On a registered external crypto holdout, the frozen guided-HMM candidate satisfies the prewritten relative IC/Sharpe rule against the global LightGBM and raw-feature HMM references, but it does not establish positive tradable alpha.

## Forbidden Claims

- Do not claim a profitable or deployable trading strategy.
- Do not switch the final candidate after seeing the locked holdout.
- Do not tune thresholds, labels, features, architecture, or method choice on the same locked holdout.
- Do not cite invalidated positional-fold runs as predictive evidence.
- Do not describe development-observed results as untouched final-test results.

## Recommended Next Phase

Phase 45 should be venue-formatting and reviewer package work: convert `paper/main.md` to the target template, tighten related work, create final tables/figures, and prepare a reproducibility appendix. It should not add new model search unless a new pre-registered dataset or external replication is created first.
