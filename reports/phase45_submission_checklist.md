# Phase 45 Submission Checklist

## Manuscript Structure

- Abstract states limited locked relative support and blocks tradable-alpha language.
- Introduction explains the validation repair without overselling the model.
- Related work connects financial ML validation, HMM regimes, contrastive time-series learning, and reproducibility.
- Methods describe the frozen final candidate and all baselines.
- Results separate development-observed evidence from locked-holdout evidence.
- Limitations explicitly state negative locked Sharpe and negative locked total return.
- Reproducibility appendix lists safe commands and artifact scope.

## Claim-Control Checks

- The paper does not claim a tradable strategy.
- The paper does not switch away from the frozen final candidate.
- The paper states that the same locked holdout cannot be reused for model rescue.
- The paper states that invalidated positional-fold results are audit history only.
- The paper uses "limited locked relative support" instead of "dominance" or "profitability."

## Before Submission

- Convert Markdown to the final venue template.
- Verify the current ICAIF or target-venue call for page limit, anonymity, author-list, supplement, and submission-system rules.
- Keep the review manuscript self-contained if the selected venue does not accept supplementary material.
- Replace placeholder related-work notes with final citations.
- Create compact final figures F1-F4.
- Confirm double-blind requirements for the selected venue.
- Run an anonymity audit over title page, acknowledgements, repository links, self-citations, metadata, and filenames.
- Archive a frozen artifact release with a DOI or persistent identifier before claiming artifact availability.
- Run `.un_research_grade_checks.ps1 -Mode full` immediately before pushing/submission.
