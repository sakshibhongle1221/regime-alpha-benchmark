from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from config import BASE_DIR, SAVE_DIR


MODELS_DIR = Path(SAVE_DIR)
REPORTS_DIR = Path(BASE_DIR) / "reports"
PAPER_DIR = Path(BASE_DIR) / "paper"

LOCKED_RESULTS_PATH = MODELS_DIR / "phase43b_locked_external_experiment_results.csv"
LOCKED_PRIMARY_PATH = MODELS_DIR / "phase43b_locked_external_primary_comparison.csv"
LOCKED_CLAIMS_PATH = MODELS_DIR / "phase43b_locked_external_claims.csv"
PHASE44_EVIDENCE_PATH = MODELS_DIR / "phase44_paper_evidence_matrix.csv"
PHASE44_RISK_PATH = MODELS_DIR / "phase44_submission_risk_register.csv"

MANUSCRIPT_PATH = PAPER_DIR / "phase45_venue_ready_manuscript.md"
PACKAGE_REPORT_PATH = REPORTS_DIR / "phase45_venue_manuscript_package.md"
REPRO_APPENDIX_PATH = REPORTS_DIR / "phase45_reproducibility_appendix.md"
SUBMISSION_CHECKLIST_PATH = REPORTS_DIR / "phase45_submission_checklist.md"
TABLE_PLAN_PATH = MODELS_DIR / "phase45_table_plan.csv"
FIGURE_PLAN_PATH = MODELS_DIR / "phase45_figure_plan.csv"
CLAIM_SECTION_MAP_PATH = MODELS_DIR / "phase45_claim_to_section_map.csv"
VENUE_REQUIREMENT_AUDIT_PATH = MODELS_DIR / "phase45_venue_requirement_audit.csv"
EXTERNAL_AUDIT_PATH = REPORTS_DIR / "phase45_external_research_audit.md"

FINAL_CANDIDATE = "regime_lgbm_hmm_guided_hmm"
PRIMARY_REFERENCES = {"global_lgbm", "regime_lgbm_hmm"}
TEMPLATE_TARGET = "ACM acmart-style proceedings manuscript"

METHOD_LABELS = {
    "global_lgbm": "Global LightGBM",
    "regime_lgbm_hmm": "Raw-feature HMM + regime LightGBM",
    "regime_lgbm_kmeans": "KMeans + regime LightGBM",
    "regime_lgbm_vol_bucket": "Volatility buckets + regime LightGBM",
    "regime_lgbm_contrastive": "Vanilla contrastive-GMM + regime LightGBM",
    "regime_lgbm_contrastive_hmm": "Vanilla contrastive-HMM + regime LightGBM",
    "regime_lgbm_hmm_guided_gmm": "HMM-guided contrastive-GMM + regime LightGBM",
    "regime_lgbm_hmm_guided_hmm": "HMM-guided contrastive-HMM + regime LightGBM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Phase 45 venue-ready manuscript package from frozen evidence."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and generated text without writing artifacts.",
    )
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 45 input is missing: {path}")
    return pd.read_csv(path)


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def pct(value: object, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return str(value)


def method_result(results: pd.DataFrame, method: str) -> pd.Series:
    rows = results[results["method"].astype(str) == method]
    if rows.empty:
        raise ValueError(f"Missing method in locked results: {method}")
    return rows.iloc[0]


def claim_status(claims: pd.DataFrame, claim_id: str) -> str:
    rows = claims[claims["claim_id"].astype(str) == claim_id]
    if rows.empty:
        raise ValueError(f"Missing locked claim row: {claim_id}")
    return str(rows.iloc[0]["claim_status"])


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def validate_inputs(
    locked_results: pd.DataFrame,
    locked_primary: pd.DataFrame,
    locked_claims: pd.DataFrame,
    phase44_evidence: pd.DataFrame,
) -> None:
    required_methods = {FINAL_CANDIDATE, *PRIMARY_REFERENCES}
    missing_methods = required_methods - set(locked_results["method"].astype(str))
    if missing_methods:
        raise ValueError(f"Locked results are missing required methods: {sorted(missing_methods)}")

    if claim_status(locked_claims, "locked_relative_success_rule") != "satisfied":
        raise ValueError("The locked relative rule must already be satisfied before Phase 45.")
    if claim_status(locked_claims, "positive_tradable_alpha") != "not_supported":
        raise ValueError("Phase 45 requires the positive tradable-alpha claim to remain blocked.")
    if claim_status(locked_claims, "same_holdout_retuning") != "forbidden":
        raise ValueError("Phase 45 requires same-holdout retuning to remain forbidden.")

    references = set(locked_primary["reference_method"].astype(str))
    if references != PRIMARY_REFERENCES:
        raise ValueError(f"Primary references drifted: {sorted(references)}")
    candidates = set(locked_primary["final_candidate"].astype(str))
    if candidates != {FINAL_CANDIDATE}:
        raise ValueError(f"Final candidate drifted: {sorted(candidates)}")
    required_rule_cols = {"ic_improved", "sharpe_non_worse", "coverage_equal"}
    missing_rule_cols = sorted(required_rule_cols - set(locked_primary.columns))
    if missing_rule_cols:
        raise ValueError(f"Locked primary comparison missing columns: {missing_rule_cols}")
    if not locked_primary[list(required_rule_cols)].astype(bool).all().all():
        raise ValueError("Locked primary rule no longer passes for every primary reference.")

    evidence_blocks = set(phase44_evidence["evidence_block"].astype(str))
    required_blocks = {"validation_repair", "locked_external_holdout"}
    if not required_blocks.issubset(evidence_blocks):
        raise ValueError(f"Phase 44 evidence matrix is missing blocks: {sorted(required_blocks - evidence_blocks)}")


def locked_result_table(locked_results: pd.DataFrame) -> pd.DataFrame:
    order = {
        FINAL_CANDIDATE: 0,
        "regime_lgbm_hmm_guided_gmm": 1,
        "regime_lgbm_hmm": 2,
        "global_lgbm": 3,
        "regime_lgbm_contrastive": 4,
        "regime_lgbm_contrastive_hmm": 5,
        "regime_lgbm_kmeans": 6,
        "regime_lgbm_vol_bucket": 7,
    }
    rows = locked_results.copy()
    rows["_order"] = rows["method"].map(order).fillna(99)
    rows = rows.sort_values(["_order", "method"]).drop(columns=["_order"])
    return pd.DataFrame(
        {
            "Method": rows["method"].map(METHOD_LABELS).fillna(rows["method"]),
            "Mean Asset IC": rows["mean_asset_IC"].map(fmt),
            "Sharpe": rows["Sharpe"].map(fmt),
            "Total Return": rows["total_return"].map(lambda value: pct(value, 1)),
            "Drawdown": rows["drawdown"].map(lambda value: pct(value, 1)),
            "Rows": rows["n_test_rows"].astype(int).astype(str),
        }
    )


def build_table_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table_id": "T1",
                "paper_section": "Data and validation",
                "artifact": "reports/phase43b_locked_holdout_data_freeze.md",
                "purpose": "Describe development versus locked-holdout data roles without exposing row-level final predictions.",
                "status": "ready_from_existing_artifacts",
            },
            {
                "table_id": "T2",
                "paper_section": "Benchmark design",
                "artifact": "reports/paper_artifact_map.csv",
                "purpose": "List methods and explain equal-coverage evaluation.",
                "status": "ready_from_existing_artifacts",
            },
            {
                "table_id": "T3",
                "paper_section": "Repaired development evidence",
                "artifact": "models/crypto20_repaired_fold_local_statistical_method_summary.csv",
                "purpose": "Show weak development-observed evidence and uncertainty intervals.",
                "status": "ready_from_existing_artifacts",
            },
            {
                "table_id": "T4",
                "paper_section": "Locked external holdout",
                "artifact": "models/phase43b_locked_external_experiment_results.csv",
                "purpose": "Show the one-shot locked method comparison.",
                "status": "ready_from_existing_artifacts",
            },
            {
                "table_id": "T5",
                "paper_section": "Claim adjudication",
                "artifact": "models/phase43b_locked_external_claims.csv",
                "purpose": "Separate limited locked relative support from blocked tradable-alpha claims.",
                "status": "ready_from_existing_artifacts",
            },
            {
                "table_id": "T6",
                "paper_section": "Reproducibility appendix",
                "artifact": "models/research_grade_check_report.md",
                "purpose": "Report regression gate coverage and test status.",
                "status": "ready_from_existing_artifacts",
            },
        ]
    )


def build_figure_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure_id": "F1",
                "paper_section": "Validation protocol",
                "artifact": "reports/phase39r_neural_fold_local_results.md",
                "purpose": "Show invalidated positional folds versus repaired common-calendar fold-local design.",
                "status": "needs_final_drawing",
            },
            {
                "figure_id": "F2",
                "paper_section": "System overview",
                "artifact": "reports/phase44_paper_readiness_package.md",
                "purpose": "Diagram features, HMM weak labels, contrastive encoder, regime assignment, and downstream LightGBM.",
                "status": "needs_final_drawing",
            },
            {
                "figure_id": "F3",
                "paper_section": "Locked result",
                "artifact": "models/phase43b_locked_external_primary_comparison.csv",
                "purpose": "Plot candidate versus primary references on mean asset IC and Sharpe.",
                "status": "ready_from_existing_artifacts",
            },
            {
                "figure_id": "F4",
                "paper_section": "Mechanism boundary",
                "artifact": "models/phase42_execution_stress_summary.csv",
                "purpose": "Show execution sensitivity and why the result is not a tradable-alpha claim.",
                "status": "ready_from_existing_artifacts",
            },
        ]
    )


def build_claim_section_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "validation_repair_contribution",
                "allowed_wording": "The original positional-fold evidence is invalidated and retained only as audit history.",
                "section": "Validation protocol",
                "required_artifact": "reports/phase39r_neural_fold_local_results.md",
                "forbidden_extension": "Do not use invalidated positional-fold results as predictive evidence.",
            },
            {
                "claim_id": "locked_relative_support",
                "allowed_wording": "The frozen guided-HMM candidate satisfies the prewritten locked relative IC/Sharpe rule versus global LightGBM and raw-feature HMM.",
                "section": "Locked external holdout",
                "required_artifact": "models/phase43b_locked_external_primary_comparison.csv",
                "forbidden_extension": "Do not claim broad method dominance or statistical proof of profitability.",
            },
            {
                "claim_id": "positive_tradable_alpha",
                "allowed_wording": "Positive tradable alpha is not supported by the locked result.",
                "section": "Discussion and limitations",
                "required_artifact": "models/phase43b_locked_external_claims.csv",
                "forbidden_extension": "Do not claim a deployable trading strategy.",
            },
            {
                "claim_id": "same_holdout_retuning",
                "allowed_wording": "The same locked holdout cannot be reused for model rescue.",
                "section": "Limitations and future work",
                "required_artifact": "reports/phase43b_locked_external_adjudication.md",
                "forbidden_extension": "Do not tune thresholds, labels, features, architectures, or candidate choice on the spent holdout.",
            },
            {
                "claim_id": "venue_framing",
                "allowed_wording": "This is a validation-and-mechanism paper, not a profitability paper.",
                "section": "Abstract, introduction, conclusion",
                "required_artifact": "reports/phase45_venue_manuscript_package.md",
                "forbidden_extension": "Do not let venue-facing prose soften the negative Sharpe/return limitation.",
            },
        ]
    )


def build_venue_requirement_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "requirement_id": "icaif_scope_fit",
                "source": "ICAIF'24 official call for papers",
                "source_url": "https://ai-finance.org/call-for-papers/",
                "requirement": "Paper should connect AI and Finance, with topic fit for representation learning, financial time series, validation/calibration, robustness, crypto, or trading.",
                "phase45_status": "satisfied_by_project_scope",
                "phase45_action": "Frame as validation-and-mechanism work in AI/Finance, not a trading-profit paper.",
            },
            {
                "requirement_id": "icaif_page_budget",
                "source": "ICAIF'24 official call for papers",
                "source_url": "https://ai-finance.org/call-for-papers/",
                "requirement": "Recent ICAIF instructions used eight total pages including figures and references, PDF only, self-contained, and no supplementary material.",
                "phase45_status": "not_yet_final_pdf",
                "phase45_action": "Phase 46 must convert this source into a compact venue PDF and move nonessential detail into GitHub artifacts, not submission supplement.",
            },
            {
                "requirement_id": "icaif_double_blind",
                "source": "ICAIF'24 official call for papers",
                "source_url": "https://ai-finance.org/call-for-papers/",
                "requirement": "Recent ICAIF instructions used double-blind review and prohibited author-identifying information in submitted papers.",
                "phase45_status": "manuscript_has_no_author_block",
                "phase45_action": "Phase 46 must run an anonymity audit over manuscript text, acknowledgements, links, repository names, and self-citations.",
            },
            {
                "requirement_id": "acm_template",
                "source": "ACM Primary Article Template",
                "source_url": "https://www.acm.org/publications/proceedings-template",
                "requirement": "ACM proceedings authors should use the official acmart LaTeX workflow and the sigconf proceedings template unless the venue says otherwise.",
                "phase45_status": "structured_markdown_only",
                "phase45_action": "Phase 46 must produce acmart/sigconf LaTeX or verify the current venue-specific template before submission.",
            },
            {
                "requirement_id": "artifact_functional",
                "source": "ACM Artifact Review and Badging",
                "source_url": "https://www.acm.org/publications/policies/artifact-review-and-badging-current",
                "requirement": "Artifact package should be documented, consistent, complete enough for review, and exercisable.",
                "phase45_status": "mostly_satisfied",
                "phase45_action": "Keep artifact manifest, reproduction appendix, tests, and full research-grade gate synchronized with the paper claims.",
            },
            {
                "requirement_id": "artifact_available",
                "source": "ACM Artifact Review and Badging",
                "source_url": "https://www.acm.org/publications/policies/artifact-review-and-badging-current",
                "requirement": "For an availability-style artifact claim, artifacts should be on a public archival repository with a DOI or persistent identifier; personal pages are not enough.",
                "phase45_status": "not_yet_satisfied",
                "phase45_action": "Before submission or camera-ready, archive a frozen release on Zenodo/OSF/Figshare or an institutional repository if artifact availability is claimed.",
            },
            {
                "requirement_id": "locked_holdout_integrity",
                "source": "Project claim-control protocol",
                "source_url": "reports/claim_registry.md",
                "requirement": "The locked holdout has been spent once and cannot be reused for model rescue.",
                "phase45_status": "satisfied",
                "phase45_action": "Keep same-holdout retuning, candidate switching, and profitability claims explicitly forbidden.",
            },
        ]
    )


def build_manuscript(locked_results: pd.DataFrame, evidence: pd.DataFrame) -> str:
    final = method_result(locked_results, FINAL_CANDIDATE)
    global_ref = method_result(locked_results, "global_lgbm")
    hmm_ref = method_result(locked_results, "regime_lgbm_hmm")
    guided_gmm = method_result(locked_results, "regime_lgbm_hmm_guided_gmm")
    locked_table = locked_result_table(locked_results)
    compact_evidence = evidence[["evidence_block", "data_role", "finding", "claim_boundary"]]
    return f"""# HMM-Guided Contrastive Representations for Leakage-Safe Regime-Conditioned Crypto Alpha Evaluation

## Venue Note

Phase 45 targets an {TEMPLATE_TARGET}. This file is a structured manuscript source, not the final LaTeX/PDF conversion. It keeps all venue-facing claims synchronized with the frozen Phase 43B locked-holdout evidence.

Before submission, Phase 46 must verify the current venue call. The conservative working assumptions are: ACM `acmart`/`sigconf` formatting, double-blind anonymity, compact self-contained manuscript, no dependency on supplementary material, and no author-identifying information in the review PDF.

## Abstract

Regime-conditioned alpha models are attractive because financial relationships can change across trend, stress, and transition periods. However, regime labels are latent and easy to overfit, especially in multi-asset crypto backtests. This paper studies whether Hidden Markov Model guided contrastive representations improve regime-conditioned alpha modeling under leakage-safe validation. The project compares global LightGBM, raw-feature HMM regimes, KMeans regimes, volatility buckets, vanilla contrastive regimes, and HMM-guided contrastive regimes. A central contribution is a validation repair: earlier positional-fold evidence was invalidated after discovering cross-asset calendar overlap, then replaced with common-calendar fold-local validation, frozen-candidate rules, and a one-shot locked external holdout. On the registered 10-asset external holdout, the frozen HMM-guided contrastive-HMM candidate improves mean asset IC versus global LightGBM ({fmt(final['mean_asset_IC'])} versus {fmt(global_ref['mean_asset_IC'])}) and raw-feature HMM ({fmt(final['mean_asset_IC'])} versus {fmt(hmm_ref['mean_asset_IC'])}), with non-worse Sharpe versus both references ({fmt(final['Sharpe'])} versus {fmt(global_ref['Sharpe'])} and {fmt(hmm_ref['Sharpe'])}). The finding is limited: the locked final candidate still has negative Sharpe and negative total return ({pct(final['total_return'])}), so the paper does not claim a tradable strategy. The contribution is a research-grade evaluation framework and a mechanism boundary for sequentially guided regime learning.

## 1. Introduction

Financial machine-learning papers often fail quietly because the validation protocol is easier to overfit than the model. This project takes the opposite path: it preserves the full audit trail, including an initially exciting result that was later invalidated. The corrected contribution is not "we found a profitable crypto strategy." The corrected contribution is "we built a regime-learning benchmark that can detect when its own evidence is not strong enough."

The core modeling question is whether classical sequential structure can improve neural regime representations. HMMs are useful because they impose temporal persistence, but they are limited by raw-feature assumptions. Contrastive encoders can learn richer representations, but unconstrained clustering may not produce sequentially meaningful regimes. HMM-guided contrastive learning combines these ideas by using HMM states as weak proxy supervision, then testing downstream alpha under strict walk-forward validation.

## 2. Related Work Positioning

The manuscript should position the project at the intersection of:

- financial machine learning and purged walk-forward validation;
- latent regime models and Hidden Markov Models;
- contrastive representation learning for time series;
- benchmark reproducibility and artifact-centered empirical finance.

The paper should be framed as a validation-and-mechanism study. That framing is stronger than a profitability claim because the locked evidence explicitly blocks positive tradable alpha.

## 3. Data, Labels, and Data Roles

The project separates data roles:

- development-observed Crypto-20 evidence is used for repair, diagnosis, and candidate freezing;
- the Phase 43B external holdout is registered and frozen before outcome inspection;
- the same locked holdout cannot be reused for model rescue.

The locked external holdout contains 10 external assets, 18 folds, and 129,600 out-of-sample rows per method. The final manuscript should report data quality, symbol selection, target construction, transaction costs, and fold-local fitting clearly enough for reviewer reproduction.

## 4. Methods

The benchmark compares a no-regime global model, simple regime baselines, classical HMM regimes, vanilla contrastive regimes, and HMM-guided contrastive regimes. The frozen final candidate is `{FINAL_CANDIDATE}`. It was frozen before locked-holdout outcomes were inspected.

## 5. Validation Protocol

The key validation repair is common-calendar fold-local evaluation. Feature scaling, HMM weak-supervision fitting, contrastive pair construction, encoder training, regime assignment, and downstream alpha models are all fit inside authorized training intervals. This prevents the previous calendar-overlap problem from becoming predictive evidence.

## 6. Results

### 6.1 Locked External Holdout

{markdown_table(locked_table, list(locked_table.columns))}

The frozen final candidate satisfies the prewritten relative IC/Sharpe rule against the two primary references. This is limited locked relative support. It is not a tradable-alpha claim.

### 6.2 Why the Higher-IC Guided-GMM Row Does Not Replace the Final Candidate

The guided-GMM diagnostic row has higher locked mean asset IC ({fmt(guided_gmm['mean_asset_IC'])}), but it was not the frozen final candidate and has worse locked Sharpe ({fmt(guided_gmm['Sharpe'])}) and total return ({pct(guided_gmm['total_return'])}). Replacing the candidate after seeing locked outcomes would be post-hoc selection.

## 7. Evidence and Claim Boundaries

{markdown_table(compact_evidence, list(compact_evidence.columns))}

## 8. Limitations

- The final candidate has negative locked Sharpe and negative locked total return.
- HMM states are weak proxy labels, not ground-truth market regimes.
- The locked result supports only the prewritten relative rule.
- Row-level overlapping financial samples are not independent evidence units.
- The same locked holdout cannot be reused for labels, thresholds, features, architecture search, or candidate switching.
- The project does not claim a profitable or deployable trading strategy.

## 9. Reproducibility and Artifact Availability

The reproducibility package should include the code, curated summary CSV files, run scripts, claim registry, artifact manifest, and research-grade check report. Bulky row-level final predictions and raw data are intentionally excluded from GitHub when they are reproducible or too large.

The project is close to ACM-style artifact functionality because it includes an inventory, executable checks, curated artifacts, and validation reports. It should not claim permanent artifact availability until a frozen release is archived in a persistent repository with a DOI or equivalent identifier.

## 10. Conclusion

HMM-guided contrastive regimes receive limited locked-holdout relative support, but not profitable-alpha support. The strongest paper contribution is the disciplined empirical framework: the project found an exciting result, invalidated it, repaired the validation protocol, froze a final candidate, spent one locked holdout, and reported the boundary honestly.
"""


def build_package_report(
    table_plan: pd.DataFrame,
    figure_plan: pd.DataFrame,
    claim_map: pd.DataFrame,
    venue_audit: pd.DataFrame,
) -> str:
    return f"""# Phase 45 Venue-Ready Manuscript Package

## Status

Phase 45 converts the Phase 44 paper-readiness story into a venue-facing manuscript package. It does not tune models, change labels, change candidate choice, rerun the locked holdout, or touch final evaluation data.

## Venue Target

Working target: {TEMPLATE_TARGET}.

Practical implication: the paper should be compact, double-blind safe, self-contained, table/figure driven, and centered on reproducibility and claim discipline. Phase 46 must verify the current venue-specific call before submission.

## Main Paper Thesis

This is a validation-and-mechanism paper, not a profitability paper.

Allowed headline:

> HMM-guided contrastive regimes receive limited locked-holdout relative support under a prewritten comparison rule, but the evidence does not support a profitable or deployable trading strategy.

## Required Tables

{markdown_table(table_plan, list(table_plan.columns))}

## Required Figures

{markdown_table(figure_plan, list(figure_plan.columns))}

## Claim-To-Section Map

{markdown_table(claim_map, list(claim_map.columns))}

## Venue Requirement Audit

{markdown_table(venue_audit, list(venue_audit.columns))}

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
"""


def build_repro_appendix(
    table_plan: pd.DataFrame,
    figure_plan: pd.DataFrame,
    venue_audit: pd.DataFrame,
) -> str:
    return f"""# Phase 45 Reproducibility Appendix

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
.\\env\\Scripts\\python.exe src\\freeze_development_dataset.py --verify-only
.\\env\\Scripts\\python.exe src\\fold_local_encoder_walkforward.py --universe crypto20 --calendar-audit-only
.\\env\\Scripts\\python.exe -m unittest discover -s tests -q
.\\run_research_grade_checks.ps1 -Mode full
.\\scripts\\phases\\run_phase45_venue_manuscript_package.ps1
```

## Artifact Policy

Summary CSVs, reports, scripts, and check outputs are paper artifacts. Raw data, bulky predictions, local caches, and row-level locked/final outputs remain excluded when they are too large or reproducible.

ACM-style artifact functionality requires the package to be documented, consistent, complete enough for review, and exercisable. Phase 45 satisfies this directionally through the artifact manifest, reproduction commands, and research-grade gate. ACM-style artifact availability is not yet claimed because a DOI or persistent archival repository has not yet been created.

## Table Artifacts

{markdown_table(table_plan, list(table_plan.columns))}

## Figure Artifacts

{markdown_table(figure_plan, list(figure_plan.columns))}

## Venue And Artifact Requirement Audit

{markdown_table(venue_audit, list(venue_audit.columns))}

## Reviewer Safety Notes

- The locked holdout has already been spent once.
- The final candidate is frozen as `regime_lgbm_hmm_guided_hmm`.
- same-holdout retuning is forbidden.
- Positive tradable alpha is not supported.
"""


def build_submission_checklist() -> str:
    return """# Phase 45 Submission Checklist

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
- Run `.\run_research_grade_checks.ps1 -Mode full` immediately before pushing/submission.
"""


def build_external_audit(venue_audit: pd.DataFrame) -> str:
    return f"""# Phase 45 External Research Audit

## Purpose

This audit compares the Phase 45 manuscript package against current or recent primary venue and artifact requirements. It is a guardrail document: it does not change model evidence, does not tune results, and does not touch locked/final evaluation data.

## Primary Sources Used

- ICAIF official call for papers page: `https://ai-finance.org/call-for-papers/`
- ACM Primary Article Template: `https://www.acm.org/publications/proceedings-template`
- ACM Artifact Review and Badging: `https://www.acm.org/publications/policies/artifact-review-and-badging-current`

## Audit Table

{markdown_table(venue_audit, list(venue_audit.columns))}

## Reviewer-Risk Findings

1. Phase 45 is correctly framed as a validation-and-mechanism paper, not a profitability paper.
2. Phase 45 now records the conservative ICAIF-style constraints: compact self-contained PDF, double-blind review, no author-identifying information, and no reliance on supplementary material unless the current call permits it.
3. Phase 45 now records the ACM template constraint: final submission should use the official `acmart`/`sigconf` path or the current venue-specific override.
4. Phase 45 now records the ACM artifact distinction: the repository may aim for functional artifact review, but it should not claim artifact availability until a persistent archived release exists.
5. The biggest remaining paper risk is not the model result; it is final writing compression, citation quality, figure quality, and anonymity/format compliance.
"""


def main() -> None:
    args = parse_args()
    locked_results = read_required_csv(LOCKED_RESULTS_PATH)
    locked_primary = read_required_csv(LOCKED_PRIMARY_PATH)
    locked_claims = read_required_csv(LOCKED_CLAIMS_PATH)
    phase44_evidence = read_required_csv(PHASE44_EVIDENCE_PATH)
    read_required_csv(PHASE44_RISK_PATH)

    validate_inputs(locked_results, locked_primary, locked_claims, phase44_evidence)
    table_plan = build_table_plan()
    figure_plan = build_figure_plan()
    claim_map = build_claim_section_map()
    venue_audit = build_venue_requirement_audit()
    manuscript = build_manuscript(locked_results, phase44_evidence)
    package_report = build_package_report(table_plan, figure_plan, claim_map, venue_audit)
    repro_appendix = build_repro_appendix(table_plan, figure_plan, venue_audit)
    checklist = build_submission_checklist()
    external_audit = build_external_audit(venue_audit)

    generated_text = "\n".join([manuscript, package_report, repro_appendix, checklist, external_audit])
    required_phrases = [
        "does not claim a tradable strategy",
        "same locked holdout cannot be reused",
        "frozen final candidate",
        "limited locked relative support",
        "validation repair",
        "self-contained",
        "double-blind",
        "persistent archive",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in generated_text]
    if missing:
        raise ValueError(f"Generated Phase 45 text is missing guardrail phrases: {missing}")

    if args.dry_run:
        print("OK: Phase 45 inputs and text guardrails validated.")
        return

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(TABLE_PLAN_PATH, table_plan)
    write_csv(FIGURE_PLAN_PATH, figure_plan)
    write_csv(CLAIM_SECTION_MAP_PATH, claim_map)
    write_csv(VENUE_REQUIREMENT_AUDIT_PATH, venue_audit)
    MANUSCRIPT_PATH.write_text(manuscript, encoding="utf-8")
    PACKAGE_REPORT_PATH.write_text(package_report, encoding="utf-8")
    REPRO_APPENDIX_PATH.write_text(repro_appendix, encoding="utf-8")
    SUBMISSION_CHECKLIST_PATH.write_text(checklist, encoding="utf-8")
    EXTERNAL_AUDIT_PATH.write_text(external_audit, encoding="utf-8")

    for path in [
        MANUSCRIPT_PATH,
        PACKAGE_REPORT_PATH,
        REPRO_APPENDIX_PATH,
        SUBMISSION_CHECKLIST_PATH,
        EXTERNAL_AUDIT_PATH,
        TABLE_PLAN_PATH,
        FIGURE_PLAN_PATH,
        CLAIM_SECTION_MAP_PATH,
        VENUE_REQUIREMENT_AUDIT_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
