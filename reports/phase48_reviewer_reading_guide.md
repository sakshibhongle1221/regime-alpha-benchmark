# Phase 48 Reviewer Reading Guide

## What to read first

1. Start with the abstract and introduction for the validation-repair story.
2. Read Table 1 for evidence roles and claim boundaries.
3. Read Tables 2 and 3 for the locked external holdout and prewritten primary rule.
4. Read the discussion before judging the result as a trading system: the paper explicitly does not make that claim.

## Allowed claims

| claim_id | claim | evidence_artifact | status | safe_wording |
| --- | --- | --- | --- | --- |
| C1 | The frozen guided-HMM candidate has limited locked relative IC/Sharpe support against the two primary references. | models/phase43b_locked_external_primary_comparison.csv | allowed | limited locked relative support |
| C2 | The project demonstrates a leakage-safe validation repair path. | reports/phase39r_neural_fold_local_results.md; reports/phase44_paper_readiness_package.md | allowed | validation repair and audit discipline |

## Blocked claims

| claim_id | claim | evidence_artifact | status | safe_wording |
| --- | --- | --- | --- | --- |
| C3 | The method is a profitable or deployable trading strategy. | models/phase43b_locked_external_claims.csv | blocked | no tradable-alpha claim |
| C4 | A different locked-holdout winner may replace the frozen candidate after seeing the holdout. | models/phase43b_locked_external_claims.csv | blocked | no candidate switching after locked evaluation |
| C5 | The locked holdout may be reused for rescue tuning. | models/phase43b_locked_external_claims.csv | blocked | no same-holdout retuning |

## Reviewer-facing one-sentence summary

The paper reports a leakage-safe regime-learning benchmark where HMM-guided contrastive regimes receive limited locked relative IC/Sharpe support, while profitable/tradable alpha remains unsupported.
