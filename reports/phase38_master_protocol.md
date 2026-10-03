# Phase 38 Master Research Protocol

## Protocol Status

Phase 38 is the control-layer reset after the completed Crypto-20 statistical adjudication. It does not introduce a new model or reinterpret the Phase 37 result. It freezes what is already known, defines which data may be used for development, and sets the gates that must pass before a new predictive claim is evaluated.

Protocol version: Phase 38

## Project Objective

Regime Alpha Benchmark has two connected outcomes:

1. A publication-grade empirical study of HMM-guided contrastive regime representations.
2. A reproducible BTech-grade research system for multi-asset data, regime learning, alpha evaluation, uncertainty diagnostics, and experiment review.

The research outcome takes priority over product expansion until the final model and evaluation protocol are frozen.

## Current Evidence Freeze

The following results are historical evidence and must not be changed by later tuning:

- The BTC/ETH pilot supports sequential assignment and HMM-guided structural alignment, while downstream dominance remains statistically inconclusive.
- The pre-specified Crypto-20 extension supports structural transfer.
- Phase 36 gives guided-HMM the highest full-sample guided IC but negative risk-adjusted portfolio behavior.
- Phase 37 gives guided-HMM the highest mean fold IC (`0.0117`), but not a significant edge over raw HMM (`p=0.840`).
- Phase 37 finds worse guided-HMM multiclass NLL than global LightGBM after Holm correction.

These are valid completed findings. A later method may be compared with them on development data, but the Phase 37 artifacts themselves are immutable.

## Data Roles

Every dataset must have exactly one paper-facing role recorded in `reports/data_role_registry.csv`:

- `development_observed`: results have already been inspected; model design, diagnostics, and inner validation are allowed.
- `locked_unobserved`: features and labels may exist, but no model-selection result may be inspected before the final model freeze.
- `future_uncollected`: data does not yet exist at protocol time and is reserved for a later temporal test.
- `descriptive_only`: may support plots or data diagnostics but not predictive claims.

All BTC/ETH and Crypto-20 data available through the Phase 37 checkpoint are `development_observed`. They must never again be described as an untouched final test.

## Non-Negotiable Validation Rules

For every outer walk-forward fold:

1. Feature transformations are fit on outer-training rows only.
2. HMM weak-supervision states are fit or inferred without using outer-test rows for parameter estimation.
3. Contrastive pair mining uses outer-training windows only.
4. Encoder weights are trained on outer-training windows only.
5. Epoch choice, calibration, thresholds, and candidate selection use an inner chronological validation split only.
6. Embedding assignment models and alpha models are fit on outer-training rows only.
7. Outer-test predictions are generated once after all fold decisions are frozen.
8. No outer-test metric may trigger retraining, epoch expansion, feature changes, or threshold changes.

Fold-local assignment with an offline encoder is not sufficient for the final paper claim. The encoder itself must be fold-local.

## Phase 39 Model Contract

The next implementation phase must first reproduce the scientific baseline ladder under the fully fold-local pipeline:

1. `global_lgbm`
2. `regime_lgbm_vol_bucket`
3. `regime_lgbm_kmeans`
4. `regime_lgbm_hmm`
5. `regime_lgbm_contrastive`
6. `regime_lgbm_contrastive_hmm`
7. `regime_lgbm_hmm_guided_gmm`
8. `regime_lgbm_hmm_guided_hmm`

All methods must use the same folds, targets, costs, and outer-test rows. Missing dense vanilla-contrastive Crypto-20 artifacts must be generated rather than silently omitted from the final mechanism comparison.

## Development-Only Improvement Budget

Phase 38 authorizes only the following motivated candidate families after the fold-local baseline is working:

1. Validation-only probability calibration.
2. Soft posterior-weighted regime gating with a global-model fallback.
3. Partially pooled multi-asset experts.
4. Explicit separation of ranking score, calibrated confidence, and position sizing.
5. Confidence-based abstention when expected benefit does not clear estimated cost.

More encoder epochs, unrestricted architecture search, Crypto-50 expansion, reinforcement learning, new feature families, and live-money execution are out of scope until the acceptance gates justify them.

## Endpoint Hierarchy

Primary predictive endpoint:

- paired outer-fold IC difference against raw-feature HMM and global LightGBM.

Primary calibration endpoint:

- dependence-aware multiclass NLL or Brier-score difference.

Secondary economic endpoints:

- net Sharpe,
- total return,
- drawdown,
- turnover.

Structural alignment, interpretability, and per-asset results are mechanism or heterogeneity evidence. They cannot replace the primary predictive and calibration endpoints.

## Locked Evaluation Policy

The final candidate must be selected before any locked result is inspected. The first feasible locked asset holdout will use the next pre-ranked eligible assets from `configs/crypto_universe_candidates.csv`, selected by the written coverage and quality rules rather than by model performance. A later temporal holdout will use data strictly after the frozen local development endpoint.

The locked evaluation is run once. A failure is reported and does not trigger same-holdout tuning.

## Paper Decision Paths

### Method-improvement path

Use this path only if the fully fold-local method improves the pre-specified endpoint, remains calibrated, and survives the locked evaluation.

### Mechanism and negative-result path

Use this path if structural alignment survives but predictive or economic dominance does not. The paper contribution becomes the documented separation between regime structure, ranking, calibration, and portfolio behavior.

Both paths are legitimate. Hiding a failed locked result is not.

## Product Boundary

Before the final research model is frozen, engineering work is limited to configuration, tests, artifact lineage, reproducibility, and research visualization. FastAPI deployment, historical replay, paper trading, monitoring, and dashboard expansion begin only after the scientific pipeline passes the model-freeze gate.

## Required Phase 38 Artifacts

- `reports/phase38_master_protocol.md`
- `reports/data_role_registry.csv`
- `reports/experiment_ledger.csv`
- `reports/publication_acceptance_gates.md`
- `reports/phase39_fold_local_encoder_design.md`
- synchronized paper protocol, hypotheses, claim registry, experiment manifest, model card, submission checklist, README, and validation audit

## Phase 38 Exit Condition

Phase 38 is complete when all control artifacts exist, the validation audit enforces them, all existing evidence is classified as development-observed, the next experiment family is bounded, and the repository passes its critical checks without modifying any prediction or final-evaluation input.
