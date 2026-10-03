# Phase 46 Submission Manuscript Source

## Title

HMM-Guided Contrastive Representations for Leakage-Safe Regime-Conditioned Crypto Alpha Evaluation

## Anonymous Submission Status

This is an anonymous, self-contained manuscript source for final formatting. It contains no author block. Before submission, run the final anonymity audit and remove repository links, acknowledgements, identifying filenames, and institution-specific wording from the review PDF.

## Paper Claim

The paper makes one narrow positive claim:

> The frozen guided-HMM candidate satisfies a prewritten locked relative IC/Sharpe comparison against global LightGBM and raw-feature HMM references.

The paper also makes one critical negative claim:

> It does not claim a tradable strategy because the locked final candidate has negative Sharpe and negative total return.

## Eight-Page Section Budget

| section | page_budget | purpose |
| --- | --- | --- |
| Abstract | 0.35 | State limited locked relative support and no tradable-alpha claim. |
| Introduction | 0.9 | Motivate regime learning, validation failure, and corrected contribution. |
| Related Work | 0.75 | Position against financial ML validation, HMM regimes, contrastive time-series learning, and reproducibility. |
| Data and Validation | 1.0 | Explain data roles, common-calendar fold-local validation, labels, costs, and locked holdout. |
| Methods | 1.0 | Summarize baselines and frozen HMM-guided candidate without excessive implementation detail. |
| Results | 1.1 | Show repaired development evidence and locked external comparison in compact tables. |
| Discussion and Limitations | 0.9 | Explain mechanism boundary, negative Sharpe/return, and no same-holdout rescue. |
| Reproducibility | 0.45 | Summarize artifact commands and archive policy. |
| References | 1.55 | Reserve enough space for high-quality citations inside eight total pages. |

## Final Claim Audit

| claim_id | status | safe_wording | blocked_wording |
| --- | --- | --- | --- |
| limited_locked_relative_support | satisfied | The frozen guided-HMM candidate satisfies the prewritten locked relative IC/Sharpe rule versus the two primary references. | The method broadly dominates all baselines. |
| positive_tradable_alpha | not_supported | Positive tradable alpha is not supported because locked Sharpe=-0.3691 and locked total return=-6.6%. | The strategy is profitable or deployable. |
| same_holdout_retuning | forbidden | The same locked holdout cannot be reused for model rescue. | Tune thresholds, labels, features, architecture, or candidate choice on the spent holdout. |
| artifact_availability | not_claimed | Artifacts are functional and documented; persistent artifact availability is deferred until DOI/archive exists. | Claim ACM artifact availability before a DOI or persistent archive exists. |

## Final Writing Instruction

Write the paper as a compact validation-and-mechanism paper. Do not write it as a trading-system paper. The abstract, introduction, results, limitations, and conclusion must all preserve limited locked relative support and must not soften the negative locked Sharpe/return limitation.
