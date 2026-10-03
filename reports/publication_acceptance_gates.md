# Publication And Project Acceptance Gates

## Purpose

These gates prevent scope growth, test-set tuning, and paper claims that are stronger than the evidence. A later gate cannot pass while an earlier critical gate is open.

## Gate 0: Evidence Freeze

Pass criteria:

- Phase 37 artifacts are archived in Git history and tagged.
- Existing BTC/ETH and Crypto-20 results are marked `development_observed`.
- The experiment ledger records inspected and uninspected families.
- No historical prediction artifact is silently replaced.

Failure action: stop new modeling and repair the registry.

## Gate 1: Fold-Local Validity

Pass criteria:

- Scaling, HMM fitting, pair mining, encoder training, assignment, calibration, thresholding, and alpha fitting are bounded by the outer-training interval.
- Epoch and candidate selection use inner chronological validation only.
- Automated tests reject future-row contamination.
- Every fold records train, validation, embargo, purge, and test boundaries.

Failure action: results are diagnostic only and cannot support the final predictive claim.

## Gate 2: Baseline Completeness And Parity

Pass criteria:

- Global, classical, vanilla learned, and guided learned methods are present.
- GMM and HMM assignment isolate the sequential-assignment effect.
- Methods share target, folds, costs, and test coverage.
- Missing predictions, duplicate rows, and coverage mismatches are zero.

Failure action: do not interpret method rankings.

## Gate 3: Candidate Selection

Pass criteria:

- Candidate family was authorized before results were inspected.
- Selection uses development and inner-validation evidence only.
- Primary IC and calibration endpoints are reported together.
- Rejected candidates remain in the experiment ledger.
- Exactly one configuration is frozen for external evaluation.

Failure action: keep the Phase 37 mechanism/negative-result paper path.

## Gate 4: Locked External Evaluation

Pass criteria:

- Holdout membership and hashes are recorded before execution.
- The candidate configuration is frozen before any holdout outcome is viewed.
- Evaluation is executed once.
- Failure or contradiction is reported without same-holdout retuning.

Failure action: prohibit broad predictive-generalization and superiority claims.

## Gate 5: Statistical Adjudication

Pass criteria:

- Primary tests operate on defensible dependent-data units.
- Confidence intervals and practical effect sizes accompany p-values.
- Multiple-testing families are declared before correction.
- Time and asset dependence are acknowledged or modeled.
- Economic metrics include realistic costs and execution timing.

Failure action: downgrade the affected claim to directional or diagnostic.

## Gate 6: ICAIF Paper Readiness

Pass criteria:

- Anonymous ACM `sigconf` PDF is at most eight total pages including references.
- The paper is self-contained and does not rely on an appendix.
- Literature is current through the submission year.
- Every quantitative claim maps to a generated artifact.
- The abstract states the primary negative or inconclusive result when applicable.
- Anonymity, citation, formatting, and claim audits pass.

Failure action: do not submit a known-invalid or over-length manuscript.

## Gate 7: BTech Project Readiness

Pass criteria:

- One-command research smoke run succeeds.
- API, dashboard, and historical replay use the frozen model contract.
- No real-money execution is enabled.
- Tests, environment lock, artifact lineage, architecture documentation, demonstration video, report, and viva deck exist.
- The demo can run without private credentials or live-network dependence.

Failure action: treat the repository as a research prototype rather than a completed product.

## Current Gate State

| Gate | Current state | Current action |
|---|---|---|
| 0 Evidence Freeze | Complete | Keep invalidated and repaired evidence separated |
| 1 Fold-Local Validity | Complete | Preserve repaired common-calendar fold-local protocol |
| 2 Baseline Completeness | Complete | Preserve equal method coverage and repaired statistical adjudication |
| 3 Candidate Selection | Complete via Phase 43A | Frozen candidate is `regime_lgbm_hmm_guided_hmm`; no calibration/threshold rescue tuning |
| 4 Locked Evaluation | Complete | Phase 43B locked external evaluation is complete: frozen guided-HMM satisfies the prewritten relative IC/Sharpe rule, but negative Sharpe/return forbid a tradable-strategy claim |
| 5 Statistical Adjudication | Complete for current evidence | Development and locked-holdout claims are adjudicated; no same-holdout retuning is authorized |
| 6 Paper Readiness | Phase 49 repository-side submission closure complete | Anonymous ACM source/PDF target, final F1/F2 figures, expanded citation coverage, claim traceability, PDF metadata audit, LaTeX warning-state audit, and artifact DOI boundary now reflect repaired evidence plus limited locked relative support/no tradable-alpha claim |
| 7 BTech Project Readiness | Research package complete; final presentation/submission stage | Project is technically strong; remaining work is human/advisor visual layout review, active venue-template verification, artifact archive/DOI decision, presentation, advisor/reviewer feedback cleanup, and final submission mechanics |
