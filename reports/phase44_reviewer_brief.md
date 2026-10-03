# Phase 44 Reviewer Brief

## One-Sentence Paper Position

This is a validation-and-mechanism paper: HMM-guided contrastive regimes receive limited locked relative support, but the evidence does not support a profitable or deployable trading strategy.

## What Is Confirmed?

- The frozen final candidate is `regime_lgbm_hmm_guided_hmm`.
- The locked external holdout was registered before model outcome inspection.
- The frozen candidate satisfies the prewritten relative rule against `global_lgbm` and `regime_lgbm_hmm`.
- Locked mean asset IC is 0.0007, versus -0.0042 for global LightGBM and -0.0024 for raw-feature HMM.
- Locked Sharpe is -0.3691, versus -1.2810 for global LightGBM and -0.9538 for raw-feature HMM.

## What Is Not Confirmed?

- The project does not confirm positive tradable alpha.
- The project does not confirm broad method dominance.
- The project does not prove HMM states are true market regimes.
- The project does not authorize switching to another method after the locked result.

## Why This Is Still Publishable

The paper contributes a complete research audit trail: an initially exciting result was invalidated, repaired, re-evaluated, frozen, and tested once on a registered external holdout. That sequence is valuable because it shows how fragile quant-ML evidence can be, and it gives a reproducible template for separating real signal from validation artifacts.

## Likely Reviewer Questions

| Reviewer Question | Paper-Safe Answer |
|---|---|
| Is this a trading strategy? | No. The locked candidate has negative Sharpe and negative total return. |
| Did the final method change after seeing the holdout? | No. Candidate switching after locked evaluation is forbidden. |
| Why not use the higher locked-IC guided-GMM result? | It was not the frozen final candidate and has worse Sharpe/return; replacing the candidate after seeing outcomes would be post-hoc selection. |
| Are HMM labels ground truth? | No. They are weak proxy supervision and a classical sequential reference. |
| Why publish a weak/negative result? | The validation repair, common-calendar benchmark, locked-holdout discipline, and claim-control framework are the main scientific contribution. |
| What should future work do? | Use a new pre-registered external replication dataset; do not reuse the same locked holdout for model rescue. |
