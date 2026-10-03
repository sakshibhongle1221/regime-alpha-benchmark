# Phase 45 External Research Audit

## Purpose

This audit compares the Phase 45 manuscript package against current or recent primary venue and artifact requirements. It is a guardrail document: it does not change model evidence, does not tune results, and does not touch locked/final evaluation data.

## Primary Sources Used

- ICAIF official call for papers page: `https://ai-finance.org/call-for-papers/`
- ACM Primary Article Template: `https://www.acm.org/publications/proceedings-template`
- ACM Artifact Review and Badging: `https://www.acm.org/publications/policies/artifact-review-and-badging-current`

## Audit Table

| requirement_id | source | source_url | requirement | phase45_status | phase45_action |
| --- | --- | --- | --- | --- | --- |
| icaif_scope_fit | ICAIF'24 official call for papers | https://ai-finance.org/call-for-papers/ | Paper should connect AI and Finance, with topic fit for representation learning, financial time series, validation/calibration, robustness, crypto, or trading. | satisfied_by_project_scope | Frame as validation-and-mechanism work in AI/Finance, not a trading-profit paper. |
| icaif_page_budget | ICAIF'24 official call for papers | https://ai-finance.org/call-for-papers/ | Recent ICAIF instructions used eight total pages including figures and references, PDF only, self-contained, and no supplementary material. | not_yet_final_pdf | Phase 46 must convert this source into a compact venue PDF and move nonessential detail into GitHub artifacts, not submission supplement. |
| icaif_double_blind | ICAIF'24 official call for papers | https://ai-finance.org/call-for-papers/ | Recent ICAIF instructions used double-blind review and prohibited author-identifying information in submitted papers. | manuscript_has_no_author_block | Phase 46 must run an anonymity audit over manuscript text, acknowledgements, links, repository names, and self-citations. |
| acm_template | ACM Primary Article Template | https://www.acm.org/publications/proceedings-template | ACM proceedings authors should use the official acmart LaTeX workflow and the sigconf proceedings template unless the venue says otherwise. | structured_markdown_only | Phase 46 must produce acmart/sigconf LaTeX or verify the current venue-specific template before submission. |
| artifact_functional | ACM Artifact Review and Badging | https://www.acm.org/publications/policies/artifact-review-and-badging-current | Artifact package should be documented, consistent, complete enough for review, and exercisable. | mostly_satisfied | Keep artifact manifest, reproduction appendix, tests, and full research-grade gate synchronized with the paper claims. |
| artifact_available | ACM Artifact Review and Badging | https://www.acm.org/publications/policies/artifact-review-and-badging-current | For an availability-style artifact claim, artifacts should be on a public archival repository with a DOI or persistent identifier; personal pages are not enough. | not_yet_satisfied | Before submission or camera-ready, archive a frozen release on Zenodo/OSF/Figshare or an institutional repository if artifact availability is claimed. |
| locked_holdout_integrity | Project claim-control protocol | reports/claim_registry.md | The locked holdout has been spent once and cannot be reused for model rescue. | satisfied | Keep same-holdout retuning, candidate switching, and profitability claims explicitly forbidden. |

## Reviewer-Risk Findings

1. Phase 45 is correctly framed as a validation-and-mechanism paper, not a profitability paper.
2. Phase 45 now records the conservative ICAIF-style constraints: compact self-contained PDF, double-blind review, no author-identifying information, and no reliance on supplementary material unless the current call permits it.
3. Phase 45 now records the ACM template constraint: final submission should use the official `acmart`/`sigconf` path or the current venue-specific override.
4. Phase 45 now records the ACM artifact distinction: the repository may aim for functional artifact review, but it should not claim artifact availability until a persistent archived release exists.
5. The biggest remaining paper risk is not the model result; it is final writing compression, citation quality, figure quality, and anonymity/format compliance.
