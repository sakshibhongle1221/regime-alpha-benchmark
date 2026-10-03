# Phase 45 Venue-Ready Manuscript Package

## Status

Phase 45 converts the Phase 44 paper-readiness story into a venue-facing manuscript package. It does not tune models, change labels, change candidate choice, rerun the locked holdout, or touch final evaluation data.

## Venue Target

Working target: ACM acmart-style proceedings manuscript.

Practical implication: the paper should be compact, double-blind safe, self-contained, table/figure driven, and centered on reproducibility and claim discipline. Phase 46 must verify the current venue-specific call before submission.

## Main Paper Thesis

This is a validation-and-mechanism paper, not a profitability paper.

Allowed headline:

> HMM-guided contrastive regimes receive limited locked-holdout relative support under a prewritten comparison rule, but the evidence does not support a profitable or deployable trading strategy.

## Required Tables

| table_id | paper_section | artifact | purpose | status |
| --- | --- | --- | --- | --- |
| T1 | Data and validation | reports/phase43b_locked_holdout_data_freeze.md | Describe development versus locked-holdout data roles without exposing row-level final predictions. | ready_from_existing_artifacts |
| T2 | Benchmark design | reports/paper_artifact_map.csv | List methods and explain equal-coverage evaluation. | ready_from_existing_artifacts |
| T3 | Repaired development evidence | models/crypto20_repaired_fold_local_statistical_method_summary.csv | Show weak development-observed evidence and uncertainty intervals. | ready_from_existing_artifacts |
| T4 | Locked external holdout | models/phase43b_locked_external_experiment_results.csv | Show the one-shot locked method comparison. | ready_from_existing_artifacts |
| T5 | Claim adjudication | models/phase43b_locked_external_claims.csv | Separate limited locked relative support from blocked tradable-alpha claims. | ready_from_existing_artifacts |
| T6 | Reproducibility appendix | models/research_grade_check_report.md | Report regression gate coverage and test status. | ready_from_existing_artifacts |

## Required Figures

| figure_id | paper_section | artifact | purpose | status |
| --- | --- | --- | --- | --- |
| F1 | Validation protocol | reports/phase39r_neural_fold_local_results.md | Show invalidated positional folds versus repaired common-calendar fold-local design. | needs_final_drawing |
| F2 | System overview | reports/phase44_paper_readiness_package.md | Diagram features, HMM weak labels, contrastive encoder, regime assignment, and downstream LightGBM. | needs_final_drawing |
| F3 | Locked result | models/phase43b_locked_external_primary_comparison.csv | Plot candidate versus primary references on mean asset IC and Sharpe. | ready_from_existing_artifacts |
| F4 | Mechanism boundary | models/phase42_execution_stress_summary.csv | Show execution sensitivity and why the result is not a tradable-alpha claim. | ready_from_existing_artifacts |

## Claim-To-Section Map

| claim_id | allowed_wording | section | required_artifact | forbidden_extension |
| --- | --- | --- | --- | --- |
| validation_repair_contribution | The original positional-fold evidence is invalidated and retained only as audit history. | Validation protocol | reports/phase39r_neural_fold_local_results.md | Do not use invalidated positional-fold results as predictive evidence. |
| locked_relative_support | The frozen guided-HMM candidate satisfies the prewritten locked relative IC/Sharpe rule versus global LightGBM and raw-feature HMM. | Locked external holdout | models/phase43b_locked_external_primary_comparison.csv | Do not claim broad method dominance or statistical proof of profitability. |
| positive_tradable_alpha | Positive tradable alpha is not supported by the locked result. | Discussion and limitations | models/phase43b_locked_external_claims.csv | Do not claim a deployable trading strategy. |
| same_holdout_retuning | The same locked holdout cannot be reused for model rescue. | Limitations and future work | reports/phase43b_locked_external_adjudication.md | Do not tune thresholds, labels, features, architectures, or candidate choice on the spent holdout. |
| venue_framing | This is a validation-and-mechanism paper, not a profitability paper. | Abstract, introduction, conclusion | reports/phase45_venue_manuscript_package.md | Do not let venue-facing prose soften the negative Sharpe/return limitation. |

## Venue Requirement Audit

| requirement_id | source | source_url | requirement | phase45_status | phase45_action |
| --- | --- | --- | --- | --- | --- |
| icaif_scope_fit | ICAIF'24 official call for papers | https://ai-finance.org/call-for-papers/ | Paper should connect AI and Finance, with topic fit for representation learning, financial time series, validation/calibration, robustness, crypto, or trading. | satisfied_by_project_scope | Frame as validation-and-mechanism work in AI/Finance, not a trading-profit paper. |
| icaif_page_budget | ICAIF'24 official call for papers | https://ai-finance.org/call-for-papers/ | Recent ICAIF instructions used eight total pages including figures and references, PDF only, self-contained, and no supplementary material. | not_yet_final_pdf | Phase 46 must convert this source into a compact venue PDF and move nonessential detail into GitHub artifacts, not submission supplement. |
| icaif_double_blind | ICAIF'24 official call for papers | https://ai-finance.org/call-for-papers/ | Recent ICAIF instructions used double-blind review and prohibited author-identifying information in submitted papers. | manuscript_has_no_author_block | Phase 46 must run an anonymity audit over manuscript text, acknowledgements, links, repository names, and self-citations. |
| acm_template | ACM Primary Article Template | https://www.acm.org/publications/proceedings-template | ACM proceedings authors should use the official acmart LaTeX workflow and the sigconf proceedings template unless the venue says otherwise. | structured_markdown_only | Phase 46 must produce acmart/sigconf LaTeX or verify the current venue-specific template before submission. |
| artifact_functional | ACM Artifact Review and Badging | https://www.acm.org/publications/policies/artifact-review-and-badging-current | Artifact package should be documented, consistent, complete enough for review, and exercisable. | mostly_satisfied | Keep artifact manifest, reproduction appendix, tests, and full research-grade gate synchronized with the paper claims. |
| artifact_available | ACM Artifact Review and Badging | https://www.acm.org/publications/policies/artifact-review-and-badging-current | For an availability-style artifact claim, artifacts should be on a public archival repository with a DOI or persistent identifier; personal pages are not enough. | not_yet_satisfied | Before submission or camera-ready, archive a frozen release on Zenodo/OSF/Figshare or an institutional repository if artifact availability is claimed. |
| locked_holdout_integrity | Project claim-control protocol | reports/claim_registry.md | The locked holdout has been spent once and cannot be reused for model rescue. | satisfied | Keep same-holdout retuning, candidate switching, and profitability claims explicitly forbidden. |

## Stop Rules

- Do not reuse the same locked holdout for model rescue.
- Do not replace the frozen final candidate with a diagnostic row after seeing outcomes.
- Do not describe negative Sharpe/return as tradable alpha.
- Do not use invalidated positional-fold evidence as predictive evidence.
- Do not weaken the limitation that same-holdout retuning is forbidden.
- Do not claim ACM artifact availability until a DOI or persistent archive exists.
- Do not submit before the current venue page limit, anonymity, and supplementary-material rules are rechecked.

## Next Work After Phase 45

The next step should be writing and formatting: final LaTeX conversion, citation polishing, compact figures, reviewer-facing appendix, and optional slide/poster material. New modeling should happen only on a newly registered external replication dataset.
