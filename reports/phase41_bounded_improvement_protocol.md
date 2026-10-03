# Phase 41 Bounded Calibration And Soft-Gating Protocol

## Purpose

Phase 41 is a controlled improvement phase. It registers calibration and soft-gating candidates motivated by Phase 40, but it does **not** tune against Phase 40 outer-test results.

The current data role remains `development_observed`. Candidate selection is restricted to `inner_chronological_validation_only`.

## Why This Phase Exists

Phase 40 found no corrected IC/Sharpe superiority claim and showed weak probability/portfolio behavior. The right response is not to search the already-inspected outer-test table for a nicer result. The right response is to define bounded candidates, select them inside each outer fold using inner validation, and then evaluate the frozen choices once on the outer fold.

Execution-control score-threshold candidates are registered but deferred. They are not part of the Phase 41B probability-calibration/soft-gating run because they alter trade execution signals rather than probability calibration; they require a separate execution-focused protocol.

## Forbidden Selection Inputs

- `models/crypto20_repaired_fold_local_alpha_oos_predictions.csv`
- `models/crypto20_repaired_fold_local_statistical_claims.csv`
- `models/crypto20_repaired_fold_local_statistical_method_summary.csv`
- `locked_final_test`

## Registered Candidates

| Candidate | Family | Primary selector | Status |
|---|---|---|---|
| `p41_prob_temperature` | probability_calibration | inner_validation_nll | registered_not_selected |
| `p41_prior_blend` | probability_calibration | inner_validation_nll | registered_not_selected |
| `p41_posterior_temperature` | soft_regime_gating | inner_validation_nll | registered_not_selected |
| `p41_global_regime_shrinkage` | soft_regime_gating | inner_validation_nll | registered_not_selected |
| `p41_score_threshold` | execution_control | inner_validation_nll_then_turnover | registered_deferred |

## Mandatory Rules

| Rule | Requirement | Status |
|---|---|---|
| `p41_no_outer_test_selection` | Do not select candidate parameters from repaired outer-test predictions or Phase 40 statistical outputs. | mandatory |
| `p41_inner_nll_primary` | Use inner_validation_nll as the primary candidate selector. | mandatory |
| `p41_turnover_guardrail` | Reject candidates that increase turnover by more than 25% versus the fold baseline. | mandatory |
| `p41_equal_coverage` | Reject candidates whose outer-test coverage differs from the repaired baseline. | mandatory |
| `p41_freeze_before_locked_test` | Freeze exactly one candidate before any future locked external evaluation. | mandatory |

## Paper-Safe Interpretation

This phase is infrastructure and protocol, not a performance claim. It makes future improvement attempts auditable by separating:

1. candidate definition,
2. inner-validation selection,
3. outer-fold evaluation,
4. final locked-test evaluation.

If Phase 41 candidates do not improve inner-validation calibration or repaired outer-fold diagnostics, that negative result remains part of the research record.
