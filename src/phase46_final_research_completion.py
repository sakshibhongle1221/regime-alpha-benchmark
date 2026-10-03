from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from config import BASE_DIR, SAVE_DIR


MODELS_DIR = Path(SAVE_DIR)
REPORTS_DIR = Path(BASE_DIR) / "reports"
PAPER_DIR = Path(BASE_DIR) / "paper"

PHASE45_MANUSCRIPT_PATH = PAPER_DIR / "phase45_venue_ready_manuscript.md"
PHASE45_PACKAGE_PATH = REPORTS_DIR / "phase45_venue_manuscript_package.md"
PHASE45_VENUE_AUDIT_PATH = MODELS_DIR / "phase45_venue_requirement_audit.csv"
LOCKED_RESULTS_PATH = MODELS_DIR / "phase43b_locked_external_experiment_results.csv"
LOCKED_CLAIMS_PATH = MODELS_DIR / "phase43b_locked_external_claims.csv"
RESEARCH_GATE_PATH = MODELS_DIR / "research_grade_check_report.csv"

FINAL_REPORT_PATH = REPORTS_DIR / "phase46_final_research_completion.md"
SUBMISSION_AUDIT_PATH = REPORTS_DIR / "phase46_submission_readiness_audit.md"
ANONYMITY_AUDIT_REPORT_PATH = REPORTS_DIR / "phase46_anonymity_audit.md"
REVIEWER_RESPONSE_PACK_PATH = REPORTS_DIR / "phase46_reviewer_response_pack.md"
FINAL_TODO_PATH = REPORTS_DIR / "phase46_final_submission_todo.md"
SUBMISSION_MANUSCRIPT_PATH = PAPER_DIR / "phase46_submission_manuscript.md"
ACM_LATEX_SKELETON_PATH = PAPER_DIR / "phase46_acm_sigconf_skeleton.tex"

SECTION_BUDGET_PATH = MODELS_DIR / "phase46_section_budget.csv"
SUBMISSION_GATE_MATRIX_PATH = MODELS_DIR / "phase46_submission_gate_matrix.csv"
ANONYMITY_AUDIT_CSV_PATH = MODELS_DIR / "phase46_anonymity_audit.csv"
CLAIM_AUDIT_PATH = MODELS_DIR / "phase46_final_claim_audit.csv"
REVIEWER_OBJECTION_MATRIX_PATH = MODELS_DIR / "phase46_reviewer_objection_matrix.csv"

FINAL_CANDIDATE = "regime_lgbm_hmm_guided_hmm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 46 final research-completion and submission-readiness artifacts."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and generated text without writing artifacts.",
    )
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 46 input is missing: {path}")
    return pd.read_csv(path)


def read_required_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 46 input is missing: {path}")
    return path.read_text(encoding="utf-8")


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


def method_result(results: pd.DataFrame, method: str) -> pd.Series:
    rows = results[results["method"].astype(str) == method]
    if rows.empty:
        raise ValueError(f"Missing method: {method}")
    return rows.iloc[0]


def claim_status(claims: pd.DataFrame, claim_id: str) -> str:
    rows = claims[claims["claim_id"].astype(str) == claim_id]
    if rows.empty:
        raise ValueError(f"Missing locked claim row: {claim_id}")
    return str(rows.iloc[0]["claim_status"])


def validate_inputs(
    phase45_manuscript: str,
    venue_audit: pd.DataFrame,
    locked_results: pd.DataFrame,
    locked_claims: pd.DataFrame,
    research_gate: pd.DataFrame,
) -> None:
    required_phrases = [
        "does not claim a tradable strategy",
        "same locked holdout cannot be reused",
        "limited locked relative support",
        "persistent repository",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in phase45_manuscript]
    if missing:
        raise ValueError(f"Phase 45 manuscript is missing final guardrails: {missing}")

    venue_text = " ".join(venue_audit.astype(str).agg(" ".join, axis=1).tolist())
    for phrase in ["double-blind", "eight total pages", "sigconf", "public archival repository with a DOI"]:
        if phrase not in venue_text:
            raise ValueError(f"Phase 45 venue audit is missing: {phrase}")

    if claim_status(locked_claims, "positive_tradable_alpha") != "not_supported":
        raise ValueError("Positive tradable alpha must remain not_supported.")
    if claim_status(locked_claims, "same_holdout_retuning") != "forbidden":
        raise ValueError("Same-holdout retuning must remain forbidden.")

    final = method_result(locked_results, FINAL_CANDIDATE)
    if float(final["Sharpe"]) >= 0 or float(final["total_return"]) >= 0:
        raise ValueError("Phase 46 assumes the final candidate remains non-profitable on locked evidence.")

    if "status" in research_gate.columns:
        failures = research_gate[research_gate["status"].astype(str) == "FAIL"]
        if not failures.empty:
            raise ValueError("Research-grade gate has failures; Phase 46 cannot mark completion.")


def build_section_budget() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Abstract", 0.35, "State limited locked relative support and no tradable-alpha claim."),
            ("Introduction", 0.90, "Motivate regime learning, validation failure, and corrected contribution."),
            ("Related Work", 0.75, "Position against financial ML validation, HMM regimes, contrastive time-series learning, and reproducibility."),
            ("Data and Validation", 1.00, "Explain data roles, common-calendar fold-local validation, labels, costs, and locked holdout."),
            ("Methods", 1.00, "Summarize baselines and frozen HMM-guided candidate without excessive implementation detail."),
            ("Results", 1.10, "Show repaired development evidence and locked external comparison in compact tables."),
            ("Discussion and Limitations", 0.90, "Explain mechanism boundary, negative Sharpe/return, and no same-holdout rescue."),
            ("Reproducibility", 0.45, "Summarize artifact commands and archive policy."),
            ("References", 1.55, "Reserve enough space for high-quality citations inside eight total pages."),
        ],
        columns=["section", "page_budget", "purpose"],
    )


def build_submission_gate_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("venue_scope", "pass", "AI+Finance, representation learning, validation/calibration, robustness, crypto/time-series fit."),
            ("claim_control", "pass", "Profitable/deployable strategy claim is blocked; limited locked relative support only."),
            ("locked_holdout_integrity", "pass", "Final candidate frozen before holdout; same-holdout retuning forbidden."),
            ("research_gate", "pass", "Full research-grade gate must pass immediately before submission."),
            ("double_blind", "conditional_pass", "No author block in generated source; Phase 46 records a final anonymity audit requirement."),
            ("page_budget", "conditional_pass", "Section budget targets eight total pages; final PDF still must be measured."),
            ("acm_template", "conditional_pass", "ACM/sigconf skeleton exists; final compile must use current venue template."),
            ("artifact_functionality", "pass", "Repro commands, tests, manifests, and gate reports are present."),
            ("artifact_availability", "not_claimed", "No DOI/persistent archive yet; availability badge must not be claimed."),
            ("final_submission", "not_yet", "Requires final PDF, citations, figures, anonymity audit, and advisor/reviewer pass."),
        ],
        columns=["gate", "status", "evidence_or_action"],
    )


def build_anonymity_audit(manuscript_text: str) -> pd.DataFrame:
    checks = [
        (
            "author_block",
            [r"\\author\s*\{", r"(?im)^#+\s*authors?\b", r"(?im)^authors?\s*:"],
            "No explicit author block should appear in review manuscript.",
        ),
        (
            "acknowledgements",
            [r"(?im)^#+\s*acknowledg", r"(?im)^acknowledg(e)?ments?\s*:"],
            "Acknowledgements should be removed for double-blind review.",
        ),
        ("github_link", [r"github\.com"], "Public repository links can reveal identity during review."),
        ("personal_name_sakshi", [r"\bSakshi\b"], "Personal names must not appear in the anonymous review PDF."),
        ("institution_marker", [r"\bBTech\b"], "Student/institution context is useful for viva but should be removed from anonymous paper."),
        ("codex_marker", [r"\bCodex\b"], "Tooling names should not appear in the anonymous paper claim story."),
    ]
    rows = []
    for check_id, patterns, action in checks:
        found = any(re.search(pattern, manuscript_text) for pattern in patterns)
        rows.append(
            {
                "check_id": check_id,
                "pattern": " | ".join(patterns),
                "status": "review_required" if found else "pass",
                "action": action,
            }
        )
    return pd.DataFrame(rows)


def build_claim_audit(locked_results: pd.DataFrame, locked_claims: pd.DataFrame) -> pd.DataFrame:
    final = method_result(locked_results, FINAL_CANDIDATE)
    return pd.DataFrame(
        [
            {
                "claim_id": "limited_locked_relative_support",
                "status": claim_status(locked_claims, "locked_relative_success_rule"),
                "safe_wording": "The frozen guided-HMM candidate satisfies the prewritten locked relative IC/Sharpe rule versus the two primary references.",
                "blocked_wording": "The method broadly dominates all baselines.",
            },
            {
                "claim_id": "positive_tradable_alpha",
                "status": claim_status(locked_claims, "positive_tradable_alpha"),
                "safe_wording": f"Positive tradable alpha is not supported because locked Sharpe={fmt(final['Sharpe'])} and locked total return={pct(final['total_return'])}.",
                "blocked_wording": "The strategy is profitable or deployable.",
            },
            {
                "claim_id": "same_holdout_retuning",
                "status": claim_status(locked_claims, "same_holdout_retuning"),
                "safe_wording": "The same locked holdout cannot be reused for model rescue.",
                "blocked_wording": "Tune thresholds, labels, features, architecture, or candidate choice on the spent holdout.",
            },
            {
                "claim_id": "artifact_availability",
                "status": "not_claimed",
                "safe_wording": "Artifacts are functional and documented; persistent artifact availability is deferred until DOI/archive exists.",
                "blocked_wording": "Claim ACM artifact availability before a DOI or persistent archive exists.",
            },
        ]
    )


def build_reviewer_objection_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "Why publish if the strategy is not profitable?",
                "The contribution is validation and mechanism evidence: the project shows how an initially positive result fails under corrected validation, then reports the limited locked support honestly.",
                "reports/phase45_external_research_audit.md; models/phase46_final_claim_audit.csv",
            ),
            (
                "Is this just overfitting to crypto?",
                "The paper limits its claim to the registered crypto holdout and blocks generalization to other markets.",
                "reports/claim_registry.md; reports/phase46_submission_readiness_audit.md",
            ),
            (
                "Why not report the higher-IC guided-GMM as the final method?",
                "It was not the frozen final candidate and has worse locked Sharpe/return; switching after holdout would be post-hoc selection.",
                "paper/phase45_venue_ready_manuscript.md",
            ),
            (
                "Are HMM states ground truth?",
                "No. HMM states are weak proxy supervision and a sequential reference, not true market regimes.",
                "models/phase46_final_claim_audit.csv",
            ),
            (
                "Can reviewers reproduce the result?",
                "The project provides curated artifacts, manifests, run scripts, tests, and a full research-grade gate; DOI archive remains a Phase 46/47 pre-submission task if availability is claimed.",
                "reports/phase46_final_research_completion.md",
            ),
        ],
        columns=["reviewer_objection", "response", "evidence_artifact"],
    )


def build_submission_manuscript(section_budget: pd.DataFrame, claim_audit: pd.DataFrame) -> str:
    return f"""# Phase 46 Submission Manuscript Source

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

{markdown_table(section_budget, list(section_budget.columns))}

## Final Claim Audit

{markdown_table(claim_audit, list(claim_audit.columns))}

## Final Writing Instruction

Write the paper as a compact validation-and-mechanism paper. Do not write it as a trading-system paper. The abstract, introduction, results, limitations, and conclusion must all preserve limited locked relative support and must not soften the negative locked Sharpe/return limitation.
"""


def build_acm_skeleton(section_budget: pd.DataFrame) -> str:
    section_comments = "\n".join(
        f"% {row.section}: target {row.page_budget} pages -- {row.purpose}"
        for row in section_budget.itertuples(index=False)
    )
    return rf"""\documentclass[sigconf,anonymous,review]{{acmart}}

\settopmatter{{printacmref=false}}
\renewcommand\footnotetextcopyrightpermission[1]{{}}

\title{{HMM-Guided Contrastive Representations for Leakage-Safe Regime-Conditioned Crypto Alpha Evaluation}}

\begin{{document}}

\begin{{abstract}}
Regime-conditioned alpha models are attractive because financial relationships can change across trend, stress, and transition periods. This paper studies whether HMM-guided contrastive representations improve regime-conditioned crypto alpha modeling under leakage-safe validation. The paper reports limited locked relative support, but it does not claim a tradable strategy because locked Sharpe and total return remain negative.
\end{{abstract}}

\keywords{{financial machine learning, regime learning, contrastive learning, hidden Markov models, validation, crypto assets}}

\maketitle

{section_comments}

\section{{Introduction}}
% Compact motivation and contribution.

\section{{Related Work}}
% Financial ML validation, HMM regimes, contrastive time-series learning, reproducibility.

\section{{Data and Validation Protocol}}
% Data roles, common-calendar fold-local validation, labels, costs, locked holdout.

\section{{Methods}}
% Baselines and frozen HMM-guided candidate.

\section{{Results}}
% Repaired development evidence and locked external holdout.

\section{{Discussion and Limitations}}
% Negative locked Sharpe/return, no tradable strategy, No same-holdout rescue.

\section{{Reproducibility}}
% Artifact manifest, run commands, archive/DOI status.

\section{{Conclusion}}
% Validation-and-mechanism contribution.

\bibliographystyle{{ACM-Reference-Format}}
\bibliography{{references}}

\end{{document}}
"""


def build_final_report(
    submission_gates: pd.DataFrame,
    anonymity_audit: pd.DataFrame,
    claim_audit: pd.DataFrame,
    reviewer_objections: pd.DataFrame,
) -> str:
    return f"""# Phase 46 Final Research Completion Package

## Status

Phase 46 completes the research project as a defensible paper/review package. It does not tune models, change labels, change candidate choice, rerun the locked holdout, or touch locked/final evaluation data.

## Final Project Identity

This is a validation-and-mechanism research project:

> HMM-guided contrastive regimes receive limited locked relative support, but the evidence does not support a profitable or deployable trading strategy.

## Submission Gate Matrix

{markdown_table(submission_gates, list(submission_gates.columns))}

## Anonymity Audit

{markdown_table(anonymity_audit, list(anonymity_audit.columns))}

## Final Claim Audit

{markdown_table(claim_audit, list(claim_audit.columns))}

## Reviewer Objection Pack

{markdown_table(reviewer_objections, list(reviewer_objections.columns))}

## What Remains Before Actual External Submission

The research project is complete as a repository evidence package. External submission still requires final human-formatting steps: compile the ACM PDF, tighten citations, draw final figures, confirm current venue rules, run the final anonymity audit on the PDF, and create a DOI/persistent archive if artifact availability is claimed.
"""


def build_submission_audit(submission_gates: pd.DataFrame, section_budget: pd.DataFrame) -> str:
    return f"""# Phase 46 Submission Readiness Audit

## Purpose

This audit records the final repository-level readiness state. It is not a claim that the paper PDF has already been submitted.

## Gate Matrix

{markdown_table(submission_gates, list(submission_gates.columns))}

## Section Budget

{markdown_table(section_budget, list(section_budget.columns))}

## Decision

The research evidence package is complete. The paper is ready for final formatting and advisor/reviewer polish. It is not ready for blind external submission until the final PDF is measured against the current page limit and the anonymity audit is performed on the compiled PDF.
"""


def build_anonymity_report(anonymity_audit: pd.DataFrame) -> str:
    review_count = int((anonymity_audit["status"] == "review_required").sum())
    return f"""# Phase 46 Anonymity Audit

## Status

Potential review items found: {review_count}

This audit is run on the Markdown manuscript source. The final mandatory audit must be run again on the compiled PDF and metadata before external double-blind submission.

{markdown_table(anonymity_audit, list(anonymity_audit.columns))}
"""


def build_reviewer_pack(reviewer_objections: pd.DataFrame) -> str:
    return f"""# Phase 46 Reviewer Response Pack

## Purpose

This document prepares concise responses to likely reviewer objections. It should guide writing and rebuttal preparation, but it must not be used to overclaim.

{markdown_table(reviewer_objections, list(reviewer_objections.columns))}
"""


def build_final_todo(submission_gates: pd.DataFrame) -> str:
    return f"""# Phase 46 Final Submission TODO

## Must Complete Before External Submission

- Convert `paper/phase46_acm_sigconf_skeleton.tex` into the final venue PDF.
- Verify the current venue call for page limit, anonymity, author list, supplement, and template rules.
- Keep the paper self-contained if supplementary material is not accepted.
- Draw final compact figures F1-F4.
- Replace related-work placeholders with final citations.
- Run an anonymity audit on the compiled PDF and metadata.
- Run `.\run_research_grade_checks.ps1 -Mode full` immediately before submission.
- Create a Zenodo/OSF/Figshare/institutional archive if artifact availability is claimed.
- Do not reuse the same locked holdout for model rescue.

## Current Gate Snapshot

{markdown_table(submission_gates, list(submission_gates.columns))}
"""


def main() -> None:
    args = parse_args()
    phase45_manuscript = read_required_text(PHASE45_MANUSCRIPT_PATH)
    read_required_text(PHASE45_PACKAGE_PATH)
    venue_audit = read_required_csv(PHASE45_VENUE_AUDIT_PATH)
    locked_results = read_required_csv(LOCKED_RESULTS_PATH)
    locked_claims = read_required_csv(LOCKED_CLAIMS_PATH)
    research_gate = read_required_csv(RESEARCH_GATE_PATH)

    validate_inputs(phase45_manuscript, venue_audit, locked_results, locked_claims, research_gate)

    section_budget = build_section_budget()
    submission_gates = build_submission_gate_matrix()
    anonymity_audit = build_anonymity_audit(phase45_manuscript)
    claim_audit = build_claim_audit(locked_results, locked_claims)
    reviewer_objections = build_reviewer_objection_matrix()

    submission_manuscript = build_submission_manuscript(section_budget, claim_audit)
    acm_skeleton = build_acm_skeleton(section_budget)
    final_report = build_final_report(submission_gates, anonymity_audit, claim_audit, reviewer_objections)
    submission_audit = build_submission_audit(submission_gates, section_budget)
    anonymity_report = build_anonymity_report(anonymity_audit)
    reviewer_pack = build_reviewer_pack(reviewer_objections)
    final_todo = build_final_todo(submission_gates)

    generated_text = "\n".join(
        [
            submission_manuscript,
            acm_skeleton,
            final_report,
            submission_audit,
            anonymity_report,
            reviewer_pack,
            final_todo,
        ]
    )
    for phrase in [
        "does not claim a tradable strategy",
        "same locked holdout cannot be reused",
        "limited locked relative support",
        "persistent archive",
        "double-blind",
        "not ready for blind external submission",
    ]:
        if phrase not in generated_text:
            raise ValueError(f"Generated Phase 46 text is missing guardrail phrase: {phrase}")

    if args.dry_run:
        print("OK: Phase 46 inputs and final-completion guardrails validated.")
        return

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(SECTION_BUDGET_PATH, section_budget)
    write_csv(SUBMISSION_GATE_MATRIX_PATH, submission_gates)
    write_csv(ANONYMITY_AUDIT_CSV_PATH, anonymity_audit)
    write_csv(CLAIM_AUDIT_PATH, claim_audit)
    write_csv(REVIEWER_OBJECTION_MATRIX_PATH, reviewer_objections)

    SUBMISSION_MANUSCRIPT_PATH.write_text(submission_manuscript, encoding="utf-8")
    ACM_LATEX_SKELETON_PATH.write_text(acm_skeleton, encoding="utf-8")
    FINAL_REPORT_PATH.write_text(final_report, encoding="utf-8")
    SUBMISSION_AUDIT_PATH.write_text(submission_audit, encoding="utf-8")
    ANONYMITY_AUDIT_REPORT_PATH.write_text(anonymity_report, encoding="utf-8")
    REVIEWER_RESPONSE_PACK_PATH.write_text(reviewer_pack, encoding="utf-8")
    FINAL_TODO_PATH.write_text(final_todo, encoding="utf-8")

    for path in [
        SUBMISSION_MANUSCRIPT_PATH,
        ACM_LATEX_SKELETON_PATH,
        FINAL_REPORT_PATH,
        SUBMISSION_AUDIT_PATH,
        ANONYMITY_AUDIT_REPORT_PATH,
        REVIEWER_RESPONSE_PACK_PATH,
        FINAL_TODO_PATH,
        SECTION_BUDGET_PATH,
        SUBMISSION_GATE_MATRIX_PATH,
        ANONYMITY_AUDIT_CSV_PATH,
        CLAIM_AUDIT_PATH,
        REVIEWER_OBJECTION_MATRIX_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
