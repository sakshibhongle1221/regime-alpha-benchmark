# Phase 47 Submission Manuscript Build Report

## Status

Phase 47 converts the Phase 46 research-completion package into a stronger anonymous LaTeX submission draft and a build-readiness audit. It does not tune models, change labels, select a new candidate, rerun the locked holdout, or reinterpret locked/final evaluation data.

## Output

- `paper/phase47_submission_draft.tex`
- `paper/phase47_references.bib`
- `paper/phase47_submission_draft.pdf` when locally compiled
- `models/phase47_manuscript_build_audit.csv`
- `models/phase47_table_manifest.csv`
- `models/phase47_figure_manifest.csv`
- `models/phase47_anonymity_source_audit.csv`
- `models/phase47_reference_manifest.csv`

## Build Audit

| check_id | status | detail |
| --- | --- | --- |
| anonymous_acm_review_mode | pass | ACM sigconf anonymous review class is present. |
| bibliography_present | pass | Draft uses phase47_references.bib. |
| tables_present | pass | Draft includes evidence, locked result, and primary-rule tables. |
| no_profitable_strategy_claim | pass | Draft blocks profitability/tradability claims. |
| same_holdout_rescue_blocked | pass | Draft blocks same-holdout retuning. |
| source_anonymity | pass | Source anonymity audit must pass before external review. |
| pdf_compilation | pass | Compiled PDF exists at paper/phase47_submission_draft.pdf; pages=2; bytes=363597. |
| pdf_page_budget | pass | Compiled draft page estimate is 2; verify against the active venue limit. |
| pdf_warning_review | review_required | LaTeX/BibTeX completed with warning markers requiring human layout review: BibTeX Warning, Class acmart Warning, Overfull \hbox, Package balance Warning. |
| artifact_archive | not_claimed | No artifact availability claim until DOI/permanent archive exists. |

## Table Manifest

| table_id | title | source_artifact | status | paper_purpose | phase45_plan_rows |
| --- | --- | --- | --- | --- | --- |
| T1 | Evidence blocks and allowed claim boundaries | models/phase44_paper_evidence_matrix.csv | included_in_phase47_draft | Separates development-observed evidence from locked confirmatory evidence. | 6 |
| T2 | Locked external holdout method comparison | models/phase43b_locked_external_experiment_results.csv | included_in_phase47_draft | Reports all locked methods with equal coverage and no candidate switching. | 6 |
| T3 | Prewritten primary locked comparison | models/phase43b_locked_external_primary_comparison.csv | included_in_phase47_draft | Shows the exact relative IC/Sharpe rule for the frozen final candidate. | 6 |
| T4 | Final claim audit | models/phase46_final_claim_audit.csv | appendix_or_reviewer_pack | Keeps final manuscript wording inside safe claim boundaries. | 6 |

## Figure Manifest

| figure_id | title | source_artifact | phase47_status | action |
| --- | --- | --- | --- | --- |
| F1 | Validation protocol: Show invalidated positional folds versus repaired common-calendar fold-local design. | reports/phase39r_neural_fold_local_results.md | needs_final_drawing | Draw compact publication figure or keep as table if page budget is tight. |
| F2 | System overview: Diagram features, HMM weak labels, contrastive encoder, regime assignment, and downstream LightGBM. | reports/phase44_paper_readiness_package.md | needs_final_drawing | Draw compact publication figure or keep as table if page budget is tight. |
| F3 | Locked result: Plot candidate versus primary references on mean asset IC and Sharpe. | models/phase43b_locked_external_primary_comparison.csv | ready_from_existing_artifacts | Draw compact publication figure or keep as table if page budget is tight. |
| F4 | Mechanism boundary: Show execution sensitivity and why the result is not a tradable-alpha claim. | models/phase42_execution_stress_summary.csv | ready_from_existing_artifacts | Draw compact publication figure or keep as table if page budget is tight. |

## Reference Manifest

| citation_key | cluster | paper_role |
| --- | --- | --- |
| lopezdeprado2018afml | financial_ml_validation | Purged/embargoed financial ML validation framing. |
| hamilton1989regime | regime_switching | Classical regime-switching model foundation. |
| rabiner1989hmm | hidden_markov_models | HMM tutorial/reference for sequential latent states. |
| oord2018cpc | contrastive_learning | Contrastive predictive coding representation learning. |
| chen2020simclr | contrastive_learning | Modern contrastive objective and augmentation framing. |
| tsfresh2018 | time_series_features | Time-series feature extraction context. |
| lightgbm2017 | gradient_boosting | LightGBM baseline model citation. |
| acmart | venue_formatting | ACM proceedings template placeholder; verify current venue files before submission. |

## Decision

The manuscript source is now build-oriented and paper-safe. A local PDF can be included when compiled, but the package is still not a submitted paper: current venue-template verification, source/PDF metadata anonymity review, final figure drawing, warning cleanup, citation polish, and artifact archive/DOI decision remain before external submission.
