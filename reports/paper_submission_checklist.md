# Paper Submission Checklist

## Phase 44 Status

The project has completed the repaired development benchmark and the Phase 43B locked external holdout. The next work is paper packaging, not model rescue.

## Ready

- Repaired common-calendar development benchmark is complete.
- Phase 40 repaired statistical adjudication is complete.
- Phase 42 interpretation/execution diagnostics are complete.
- Phase 43A frozen final candidate and holdout rules are complete.
- Phase 43B locked external holdout is complete.
- Full research-grade checks pass.
- Positive tradable-alpha claim is explicitly blocked.
- Candidate switching after locked evaluation is explicitly blocked.

## Critical Paper Work

- Convert `paper/main.md` into the final venue template.
- Replace placeholder prose with final citations and related-work positioning.
- Add final figure and table numbers.
- Decide whether the target is ICAIF main track, workshop, student research track, or institutional BTech evaluation.
- Write a reproducibility appendix using the artifact manifest and research-grade gate report.

## Must Not Claim

- Do not claim HMM states are ground truth.
- Do not claim a profitable or deployable trading strategy.
- Do not claim guided-HMM statistically dominates raw-feature HMM at 5%.
- Do not claim statistically proven Crypto-20 alpha, calibration, or portfolio-performance dominance.
- Do not treat development-observed results as untouched final-test evidence.
- Do not switch from the frozen guided-HMM final candidate to a secondary diagnostic method after locked-holdout inspection.
- Do not retune thresholds, labels, features, architecture, or method choice on the same locked holdout.
