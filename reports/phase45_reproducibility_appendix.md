# Phase 45 Reproducibility Appendix

## Scope

This appendix explains how a reviewer should reproduce the evidence package without modifying locked final/evaluation data.

## Reproduction Order

1. Verify the development data freeze.
2. Run the calendar audit.
3. Run the unit suite.
4. Run the research-grade artifact gate.
5. Rebuild Phase 44 and Phase 45 packaging artifacts from existing evidence.

## Safe Commands

```powershell
.\env\Scripts\python.exe src\freeze_development_dataset.py --verify-only
.\env\Scripts\python.exe src\fold_local_encoder_walkforward.py --universe crypto20 --calendar-audit-only
.\env\Scripts\python.exe -m unittest discover -s tests -q
.\run_research_grade_checks.ps1 -Mode full
.\scripts\phases\run_phase45_venue_manuscript_package.ps1
```

## Artifact Policy

Summary CSVs, reports, scripts, and check outputs are paper artifacts. Raw data, bulky predictions, local caches, and row-level locked/final outputs remain excluded when they are too large or reproducible.

ACM-style artifact functionality requires the package to be documented, consistent, complete enough for review, and exercisable. Phase 45 satisfies this directionally through the artifact manifest, reproduction commands, and research-grade gate. ACM-style artifact availability is not yet claimed because a DOI or persistent archival repository has not yet been created.

## Table Artifacts

| table_id | paper_section | artifact | purpose | status |
| --- | --- | --- | --- | --- |
| T1 | Data and validation | reports/phase43b_locked_holdout_data_freeze.md | Describe development versus locked-holdout data roles without exposing row-level final predictions. | ready_from_existing_artifacts |
| T2 | Benchmark design | reports/paper_artifact_map.csv | List methods and explain equal-coverage evaluation. | ready_from_existing_artifacts |
| T3 | Repaired development evidence | models/crypto20_repaired_fold_local_statistical_method_summary.csv | Show weak development-observed evidence and uncertainty intervals. | ready_from_existing_artifacts |
| T4 | Locked external holdout | models/phase43b_locked_external_experiment_results.csv | Show the one-shot locked method comparison. | ready_from_existing_artifacts |
| T5 | Claim adjudication | models/phase43b_locked_external_claims.csv | Separate limited locked relative support from blocked tradable-alpha claims. | ready_from_existing_artifacts |
| T6 | Reproducibility appendix | models/research_grade_check_report.md | Report regression gate coverage and test status. | ready_from_existing_artifacts |

## Figure Artifacts

| figure_id | paper_section | artifact | purpose | status |
| --- | --- | --- | --- | --- |
| F1 | Validation protocol | reports/phase39r_neural_fold_local_results.md | Show invalidated positional folds versus repaired common-calendar fold-local design. | needs_final_drawing |
| F2 | System overview | reports/phase44_paper_readiness_package.md | Diagram features, HMM weak labels, contrastive encoder, regime assignment, and downstream LightGBM. | needs_final_drawing |
| F3 | Locked result | models/phase43b_locked_external_primary_comparison.csv | Plot candidate versus primary references on mean asset IC and Sharpe. | ready_from_existing_artifacts |
| F4 | Mechanism boundary | models/phase42_execution_stress_summary.csv | Show execution sensitivity and why the result is not a tradable-alpha claim. | ready_from_existing_artifacts |

## Venue And Artifact Requirement Audit

| requirement_id | source | source_url | requirement | phase45_status | phase45_action |
| --- | --- | --- | --- | --- | --- |
| icaif_scope_fit | ICAIF'24 official call for papers | https://ai-finance.org/call-for-papers/ | Paper should connect AI and Finance, with topic fit for representation learning, financial time series, validation/calibration, robustness, crypto, or trading. | satisfied_by_project_scope | Frame as validation-and-mechanism work in AI/Finance, not a trading-profit paper. |
| icaif_page_budget | ICAIF'24 official call for papers | https://ai-finance.org/call-for-papers/ | Recent ICAIF instructions used eight total pages including figures and references, PDF only, self-contained, and no supplementary material. | not_yet_final_pdf | Phase 46 must convert this source into a compact venue PDF and move nonessential detail into GitHub artifacts, not submission supplement. |
| icaif_double_blind | ICAIF'24 official call for papers | https://ai-finance.org/call-for-papers/ | Recent ICAIF instructions used double-blind review and prohibited author-identifying information in submitted papers. | manuscript_has_no_author_block | Phase 46 must run an anonymity audit over manuscript text, acknowledgements, links, repository names, and self-citations. |
| acm_template | ACM Primary Article Template | https://www.acm.org/publications/proceedings-template | ACM proceedings authors should use the official acmart LaTeX workflow and the sigconf proceedings template unless the venue says otherwise. | structured_markdown_only | Phase 46 must produce acmart/sigconf LaTeX or verify the current venue-specific template before submission. |
| artifact_functional | ACM Artifact Review and Badging | https://www.acm.org/publications/policies/artifact-review-and-badging-current | Artifact package should be documented, consistent, complete enough for review, and exercisable. | mostly_satisfied | Keep artifact manifest, reproduction appendix, tests, and full research-grade gate synchronized with the paper claims. |
| artifact_available | ACM Artifact Review and Badging | https://www.acm.org/publications/policies/artifact-review-and-badging-current | For an availability-style artifact claim, artifacts should be on a public archival repository with a DOI or persistent identifier; personal pages are not enough. | not_yet_satisfied | Before submission or camera-ready, archive a frozen release on Zenodo/OSF/Figshare or an institutional repository if artifact availability is claimed. |
| locked_holdout_integrity | Project claim-control protocol | reports/claim_registry.md | The locked holdout has been spent once and cannot be reused for model rescue. | satisfied | Keep same-holdout retuning, candidate switching, and profitability claims explicitly forbidden. |

## Reviewer Safety Notes

- The locked holdout has already been spent once.
- The final candidate is frozen as `regime_lgbm_hmm_guided_hmm`.
- same-holdout retuning is forbidden.
- Positive tradable alpha is not supported.
