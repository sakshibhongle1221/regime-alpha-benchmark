# Phase 47 Camera-Ready / Submission Gap List

## Still Open

1. Recompile `paper/phase47_submission_draft.tex` with the current venue ACM template before submission.
2. Re-measure final PDF length against the active call-for-papers page limit after all edits.
3. Replace any placeholder or weak citations after a final related-work pass.
4. Draw or intentionally omit each planned figure from `models/phase47_figure_manifest.csv`.
5. Run source, PDF content, and PDF metadata anonymity audits.
6. Decide whether to create a DOI archive before claiming artifact availability.
7. Get a human/advisor review of framing, claims, and formatting.
8. Clean remaining LaTeX/BibTeX warnings where they affect layout or venue compliance.

## Current Build Audit Snapshot

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
