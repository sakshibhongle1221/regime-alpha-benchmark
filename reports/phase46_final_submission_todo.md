# Phase 46 Final Submission TODO

## Must Complete Before External Submission

- Convert `paper/phase46_acm_sigconf_skeleton.tex` into the final venue PDF.
- Verify the current venue call for page limit, anonymity, author list, supplement, and template rules.
- Keep the paper self-contained if supplementary material is not accepted.
- Draw final compact figures F1-F4.
- Replace related-work placeholders with final citations.
- Run an anonymity audit on the compiled PDF and metadata.
- Run `.un_research_grade_checks.ps1 -Mode full` immediately before submission.
- Create a Zenodo/OSF/Figshare/institutional archive if artifact availability is claimed.
- Do not reuse the same locked holdout for model rescue.

## Current Gate Snapshot

| gate | status | evidence_or_action |
| --- | --- | --- |
| venue_scope | pass | AI+Finance, representation learning, validation/calibration, robustness, crypto/time-series fit. |
| claim_control | pass | Profitable/deployable strategy claim is blocked; limited locked relative support only. |
| locked_holdout_integrity | pass | Final candidate frozen before holdout; same-holdout retuning forbidden. |
| research_gate | pass | Full research-grade gate must pass immediately before submission. |
| double_blind | conditional_pass | No author block in generated source; Phase 46 records a final anonymity audit requirement. |
| page_budget | conditional_pass | Section budget targets eight total pages; final PDF still must be measured. |
| acm_template | conditional_pass | ACM/sigconf skeleton exists; final compile must use current venue template. |
| artifact_functionality | pass | Repro commands, tests, manifests, and gate reports are present. |
| artifact_availability | not_claimed | No DOI/persistent archive yet; availability badge must not be claimed. |
| final_submission | not_yet | Requires final PDF, citations, figures, anonymity audit, and advisor/reviewer pass. |
