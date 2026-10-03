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

LOCKED_RESULTS_PATH = MODELS_DIR / "phase43b_locked_external_experiment_results.csv"
LOCKED_PRIMARY_PATH = MODELS_DIR / "phase43b_locked_external_primary_comparison.csv"
LOCKED_CLAIMS_PATH = MODELS_DIR / "phase43b_locked_external_claims.csv"
EVIDENCE_MATRIX_PATH = MODELS_DIR / "phase44_paper_evidence_matrix.csv"
PHASE47_AUDIT_PATH = MODELS_DIR / "phase47_manuscript_build_audit.csv"
RESEARCH_GATE_PATH = MODELS_DIR / "research_grade_check_report.csv"

PHASE48_TEX_PATH = PAPER_DIR / "phase48_review_ready_manuscript.tex"
PHASE48_BIB_PATH = PAPER_DIR / "phase48_references.bib"
PHASE48_PDF_PATH = PAPER_DIR / "phase48_review_ready_manuscript.pdf"
PHASE48_LOG_PATH = PAPER_DIR / "phase48_review_ready_manuscript.log"
PHASE48_BLG_PATH = PAPER_DIR / "phase48_review_ready_manuscript.blg"

PHASE48_REPORT_PATH = REPORTS_DIR / "phase48_paper_polish_report.md"
PHASE48_REVIEWER_GUIDE_PATH = REPORTS_DIR / "phase48_reviewer_reading_guide.md"
PHASE48_WARNING_PLAN_PATH = REPORTS_DIR / "phase48_latex_warning_resolution.md"

PHASE48_QUALITY_AUDIT_PATH = MODELS_DIR / "phase48_paper_quality_audit.csv"
PHASE48_CLAIM_TRACE_PATH = MODELS_DIR / "phase48_claim_traceability.csv"
PHASE48_WARNING_AUDIT_PATH = MODELS_DIR / "phase48_latex_warning_audit.csv"
PHASE48_SECTION_BUDGET_PATH = MODELS_DIR / "phase48_section_budget.csv"

FINAL_CANDIDATE = "regime_lgbm_hmm_guided_hmm"
PRIMARY_REFERENCES = ["global_lgbm", "regime_lgbm_hmm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 48 review-ready paper polish package.")
    parser.add_argument("--dry-run", action="store_true", help="Validate generated artifacts without writing files.")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Phase 48 input: {path}")
    return pd.read_csv(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def pct(value: object, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}\\%"
    except Exception:
        return str(value)


def latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


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


def method_label(method: str) -> str:
    labels = {
        "global_lgbm": "Global LGBM",
        "regime_lgbm_hmm": "Raw-HMM LGBM",
        "regime_lgbm_hmm_guided_hmm": "Guided-HMM LGBM",
        "regime_lgbm_hmm_guided_gmm": "Guided-GMM LGBM",
        "regime_lgbm_contrastive": "Contrastive-GMM LGBM",
        "regime_lgbm_contrastive_hmm": "Contrastive-HMM LGBM",
        "regime_lgbm_kmeans": "KMeans LGBM",
        "regime_lgbm_vol_bucket": "Vol-bucket LGBM",
    }
    return labels.get(method, method)


def validate_inputs(
    locked_results: pd.DataFrame,
    locked_primary: pd.DataFrame,
    locked_claims: pd.DataFrame,
    evidence_matrix: pd.DataFrame,
    phase47_audit: pd.DataFrame,
    research_gate: pd.DataFrame,
) -> None:
    methods = set(locked_results.get("method", pd.Series(dtype=str)).astype(str))
    required_methods = {FINAL_CANDIDATE, *PRIMARY_REFERENCES}
    if not required_methods.issubset(methods):
        raise ValueError(f"Locked results missing required methods: {sorted(required_methods - methods)}")

    primary_refs = set(locked_primary.get("reference_method", pd.Series(dtype=str)).astype(str))
    if not set(PRIMARY_REFERENCES).issubset(primary_refs):
        raise ValueError("Primary locked comparison must contain global_lgbm and regime_lgbm_hmm references.")

    claim_map = locked_claims.set_index("claim_id")["claim_status"].astype(str).to_dict()
    if claim_map.get("positive_tradable_alpha") != "not_supported":
        raise ValueError("Phase 48 cannot proceed when positive tradable alpha is supported.")
    if claim_map.get("same_holdout_retuning") != "forbidden":
        raise ValueError("Phase 48 requires same-holdout retuning to remain forbidden.")

    if evidence_matrix.empty:
        raise ValueError("Phase 48 requires the Phase 44 evidence matrix.")

    audit_map = phase47_audit.set_index("check_id")["status"].astype(str).to_dict()
    if audit_map.get("pdf_compilation") != "pass":
        raise ValueError("Phase 48 starts only after the Phase 47 PDF compilation gate passes.")
    if audit_map.get("pdf_warning_review") != "review_required":
        raise ValueError("Phase 48 expects unresolved PDF warning review to polish, not a hidden pass.")

    if "status" in research_gate.columns and (research_gate["status"].astype(str) == "FAIL").any():
        raise ValueError("Research-grade gate has failures; do not build a polished paper package.")


def build_claim_traceability(locked_primary: pd.DataFrame, locked_claims: pd.DataFrame) -> pd.DataFrame:
    claim_map = locked_claims.set_index("claim_id")["claim_status"].astype(str).to_dict()
    return pd.DataFrame(
        [
            {
                "claim_id": "C1",
                "claim": "The frozen guided-HMM candidate has limited locked relative IC/Sharpe support against the two primary references.",
                "evidence_artifact": "models/phase43b_locked_external_primary_comparison.csv",
                "status": "allowed" if claim_map.get("locked_relative_success_rule") == "satisfied" and len(locked_primary) >= 2 else "blocked",
                "safe_wording": "limited locked relative support",
            },
            {
                "claim_id": "C2",
                "claim": "The project demonstrates a leakage-safe validation repair path.",
                "evidence_artifact": "reports/phase39r_neural_fold_local_results.md; reports/phase44_paper_readiness_package.md",
                "status": "allowed",
                "safe_wording": "validation repair and audit discipline",
            },
            {
                "claim_id": "C3",
                "claim": "The method is a profitable or deployable trading strategy.",
                "evidence_artifact": "models/phase43b_locked_external_claims.csv",
                "status": "blocked",
                "safe_wording": "no tradable-alpha claim",
            },
            {
                "claim_id": "C4",
                "claim": "A different locked-holdout winner may replace the frozen candidate after seeing the holdout.",
                "evidence_artifact": "models/phase43b_locked_external_claims.csv",
                "status": "blocked",
                "safe_wording": "no candidate switching after locked evaluation",
            },
            {
                "claim_id": "C5",
                "claim": "The locked holdout may be reused for rescue tuning.",
                "evidence_artifact": "models/phase43b_locked_external_claims.csv",
                "status": "blocked",
                "safe_wording": "no same-holdout retuning",
            },
        ]
    )


def build_section_budget() -> pd.DataFrame:
    rows = [
        ("Abstract", 0.25, "Clear limited-support result and no-tradable-alpha boundary."),
        ("Introduction", 0.75, "Motivate validation repair as the central contribution."),
        ("Related Work", 0.50, "Financial ML validation, HMMs, contrastive learning, boosting baselines."),
        ("Data and Validation Protocol", 0.85, "Common-calendar fold-local repair, evidence roles, locked-holdout rules."),
        ("Methods", 0.70, "Regime assignment families and frozen guided-HMM mechanism candidate."),
        ("Results", 1.05, "Locked result table, primary rule, and diagnostic non-switching boundary."),
        ("Discussion and Limitations", 0.85, "Negative/limited result interpretation and future-test boundary."),
        ("Reproducibility", 0.35, "Artifacts, scripts, gates, and archive limitation."),
        ("References", 0.70, "Tight starter bibliography; final venue pass still required."),
    ]
    return pd.DataFrame(rows, columns=["section", "target_pages", "purpose"])


def locked_result_rows(locked_results: pd.DataFrame) -> str:
    order = [
        FINAL_CANDIDATE,
        "regime_lgbm_hmm_guided_gmm",
        "regime_lgbm_hmm",
        "global_lgbm",
        "regime_lgbm_contrastive",
        "regime_lgbm_contrastive_hmm",
        "regime_lgbm_kmeans",
        "regime_lgbm_vol_bucket",
    ]
    indexed = locked_results.set_index("method")
    rows = []
    for method in order:
        if method not in indexed.index:
            continue
        row = indexed.loc[method]
        rows.append(
            " & ".join(
                [
                    latex_escape(method_label(method)),
                    fmt(row["mean_asset_IC"], 4),
                    fmt(row["Sharpe"], 3),
                    pct(row["total_return"], 1),
                    pct(row["drawdown"], 1),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def primary_rows(locked_primary: pd.DataFrame) -> str:
    rows = []
    for _, row in locked_primary.iterrows():
        rows.append(
            " & ".join(
                [
                    latex_escape(method_label(str(row["reference_method"]))),
                    fmt(row["delta_mean_asset_IC"], 4),
                    fmt(row["delta_Sharpe"], 3),
                    "yes" if bool(row["coverage_equal"]) else "no",
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def evidence_rows(evidence_matrix: pd.DataFrame) -> str:
    rows = []
    for _, row in evidence_matrix.head(5).iterrows():
        rows.append(
            " & ".join(
                [
                    latex_escape(row.get("evidence_block", "evidence")),
                    latex_escape(row.get("data_role", "unknown")),
                    latex_escape(row.get("claim_boundary", "paper-safe boundary")),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def build_tex(locked_results: pd.DataFrame, locked_primary: pd.DataFrame, evidence_matrix: pd.DataFrame) -> str:
    final = locked_results[locked_results["method"] == FINAL_CANDIDATE].iloc[0]
    global_ref = locked_results[locked_results["method"] == "global_lgbm"].iloc[0]
    hmm_ref = locked_results[locked_results["method"] == "regime_lgbm_hmm"].iloc[0]
    contrastive = locked_results[locked_results["method"] == "regime_lgbm_contrastive"].iloc[0]
    return rf"""\documentclass[sigconf,anonymous,review]{{acmart}}

\settopmatter{{printfolios=true,printccs=false,printacmref=true}}
\acmConference[Anonymous Review]{{Anonymous Review}}{{2026}}{{}}
\acmYear{{2026}}
\acmISBN{{}}
\acmDOI{{}}

\usepackage{{array}}
\usepackage{{tabularx}}
\newcolumntype{{Y}}{{>{{\raggedright\arraybackslash}}X}}

\title{{Leakage-Safe Regime-Conditioned Crypto Alpha Evaluation with HMM-Guided Contrastive Representations}}

\begin{{document}}

\begin{{abstract}}
Regime-conditioned financial machine learning can look compelling when latent-state labels accidentally encode validation artifacts. This paper studies HMM-guided contrastive regime learning for multi-asset crypto alpha evaluation under a deliberately leakage-safe protocol. The central contribution is not a profitable trading rule; it is a complete validation repair and claim-control path. An initially positive positional-fold result is retained only as audit history after discovering cross-asset calendar overlap. The benchmark is then rebuilt with common-calendar fold-local scaling, HMM fitting, pair construction, encoder training, regime assignment, and downstream LightGBM fitting. A single guided-HMM candidate is frozen before a registered 10-asset external holdout is evaluated once. On that locked holdout, the frozen candidate improves mean asset IC over global LightGBM ({fmt(final["mean_asset_IC"], 4)} vs. {fmt(global_ref["mean_asset_IC"], 4)}) and raw-feature HMM ({fmt(final["mean_asset_IC"], 4)} vs. {fmt(hmm_ref["mean_asset_IC"], 4)}), with non-worse Sharpe than both references ({fmt(final["Sharpe"], 3)} vs. {fmt(global_ref["Sharpe"], 3)} and {fmt(hmm_ref["Sharpe"], 3)}). However, its Sharpe and total return remain negative ({fmt(final["Sharpe"], 3)}, {pct(final["total_return"], 1)}). The paper therefore supports only a limited locked relative mechanism claim and explicitly blocks profitable-alpha, deployment, candidate-switching, and same-holdout rescue claims.
\end{{abstract}}

\keywords{{financial machine learning, crypto assets, hidden Markov models, contrastive learning, leakage-safe validation, negative results}}

\maketitle

\section{{Introduction}}
Backtests in financial machine learning are unusually vulnerable to validation mistakes because labels overlap in time, assets share market-wide shocks, and small protocol choices can dominate model differences. Regime-conditioned models add another layer of risk: latent regimes can become attractive post-hoc explanations even when the validation split is flawed.

This paper evaluates a concrete regime-learning pipeline while making the validation audit part of the contribution. The original development result appeared directionally favorable for HMM-guided regimes, but a later audit found that per-symbol positional folds overlapped in calendar time across assets. That result is not used as predictive evidence. It is retained only to document the failure mode. The repaired protocol uses common-calendar fold-local validation and later spends a single registered locked external holdout.

The paper asks a narrower question than whether the system is profitable: does sequentially guided representation learning produce regimes that satisfy a prewritten relative IC/Sharpe rule against global and raw-HMM baselines under leakage-safe validation? The answer is mixed. The frozen guided-HMM candidate satisfies that limited locked relative rule, but locked Sharpe and total return remain negative. The scientific result is therefore a mechanism boundary and a reproducible validation discipline, not a trading claim.

\section{{Related Work}}
The evaluation combines financial ML validation, regime-switching models, contrastive representation learning, and tabular boosting baselines. Leakage-safe financial validation is critical because neighboring observations and overlapping outcomes can create optimistic estimates \cite{{lopezdeprado2018afml}}. Classical HMM and regime-switching models provide temporally persistent latent states \cite{{hamilton1989regime,rabiner1989hmm}}. Contrastive learning provides a representation objective for learning invariances and predictive structure from sequences \cite{{oord2018cpc,chen2020simclr}}. LightGBM remains a strong tabular learner and is used here as the downstream alpha model across all regime families \cite{{lightgbm2017}}.

\section{{Data and Validation Protocol}}
The development universe is Crypto-20; the locked external holdout contains ten additional assets selected and frozen before outcome inspection. The target is the same eight-hour triple-barrier-style label used throughout the repaired project. The locked holdout has 18 folds and {int(final["n_test_rows"]):,} out-of-sample rows per method.

The validation rule is common-calendar and fold-local. Inside each fold, scaling, weak-supervision HMM fitting, pair construction, encoder training, regime assignment, and downstream LightGBM fitting use only the training window. This prevents the cross-asset calendar overlap that invalidated the earlier positional-fold run.

\begin{{table*}}[t]
\caption{{Evidence roles and allowed claim boundaries. Development evidence can explain mechanisms; locked evidence controls final claims.}}
\label{{tab:evidence-boundaries}}
\small
\begin{{tabularx}}{{\textwidth}}{{p{{0.25\textwidth}}p{{0.20\textwidth}}Y}}
\toprule
Evidence block & Role & Claim boundary \\
\midrule
{evidence_rows(evidence_matrix)}
\bottomrule
\end{{tabularx}}
\end{{table*}}

\section{{Methods}}
All methods share the same features, target, transaction-cost assumption, walk-forward structure, and downstream learner class. The compared regimes are global LightGBM, raw-feature HMM, KMeans, volatility buckets, vanilla contrastive, contrastive-HMM, HMM-guided GMM, and HMM-guided HMM.

The frozen final candidate is the HMM-guided contrastive-HMM representation followed by regime-conditioned LightGBM. HMM states are treated as weak sequential supervision rather than ground-truth market labels. The higher-IC guided-GMM row is reported as a diagnostic comparator only; it cannot replace the frozen candidate after the locked holdout is observed.

\section{{Results}}
Table~\ref{{tab:locked-results}} reports the locked external holdout. The frozen guided-HMM candidate has mean asset IC {fmt(final["mean_asset_IC"], 4)}, Sharpe {fmt(final["Sharpe"], 3)}, and total return {pct(final["total_return"], 1)}. Table~\ref{{tab:primary-rule}} shows the exact prewritten primary comparison: the candidate improves mean asset IC and has non-worse Sharpe against global LightGBM and raw-feature HMM with equal coverage.

\begin{{table}}[t]
\caption{{Locked external holdout performance. All methods have {int(final["n_test_rows"]):,} rows.}}
\label{{tab:locked-results}}
\scriptsize
\begin{{tabular}}{{lrrrr}}
\toprule
Method & IC & Sharpe & Return & DD \\
\midrule
{locked_result_rows(locked_results)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\caption{{Prewritten primary rule for the frozen candidate.}}
\label{{tab:primary-rule}}
\small
\begin{{tabular}}{{lrrc}}
\toprule
Reference & $\Delta$IC & $\Delta$Sharpe & Equal \\
\midrule
{primary_rows(locked_primary)}
\bottomrule
\end{{tabular}}
\end{{table}}

The result should not be overstated. A vanilla contrastive row has positive locked Sharpe ({fmt(contrastive["Sharpe"], 3)}) and positive total return ({pct(contrastive["total_return"], 1)}), but it is not the frozen final candidate and is not part of the prewritten guided-HMM mechanism claim. Switching to it after seeing the locked holdout would be post-hoc model selection. If a reader asks whether this paper presents a deployable trading strategy, it does not make that claim.

\section{{Discussion and Limitations}}
The strongest interpretation is that HMM-guided contrastive learning can produce limited relative signal structure under strict validation, but that structure does not automatically become a deployable trading edge. The locked candidate's negative Sharpe and total return block any profitability or deployment claim. The positive contribution is the research process: an attractive result was invalidated, the protocol was repaired, a candidate was frozen, one locked holdout was spent, and the claim was bounded.

Limitations include a crypto-only universe, weak proxy regime labels, overlapping financial outcomes, one spent locked holdout, and incomplete external artifact archival. Future improvements must use development-only evidence or a newly registered confirmatory test. The locked holdout used here cannot be reused for model rescue.

\section{{Reproducibility and Artifact Status}}
The repository contains phase scripts, curated result tables, claim-control reports, and a research-grade regression gate. Bulky row-level predictions and raw data are excluded when they are too large or reproducible. Artifact availability is not claimed until a permanent archive or DOI exists. The current manuscript is anonymous-review oriented and still requires final venue-template verification, PDF metadata review, and human/advisor review before external submission.

\section{{Conclusion}}
Under leakage-safe validation, the frozen guided-HMM regime learner earns limited locked relative IC/Sharpe support but not profitable-alpha support. The paper's contribution is a transparent benchmark and claim-control workflow for separating real evidence from validation artifacts in regime-conditioned crypto alpha research.

\bibliographystyle{{ACM-Reference-Format}}
\bibliography{{phase48_references}}

\end{{document}}
"""


def build_bib() -> str:
    return r"""@book{lopezdeprado2018afml,
  title={Advances in Financial Machine Learning},
  author={L{\'o}pez de Prado, Marcos},
  year={2018},
  publisher={Wiley},
  address={Hoboken, NJ}
}

@article{hamilton1989regime,
  title={A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle},
  author={Hamilton, James D.},
  journal={Econometrica},
  volume={57},
  number={2},
  pages={357--384},
  year={1989}
}

@article{rabiner1989hmm,
  title={A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition},
  author={Rabiner, Lawrence R.},
  journal={Proceedings of the IEEE},
  volume={77},
  number={2},
  pages={257--286},
  year={1989}
}

@misc{oord2018cpc,
  title={Representation Learning with Contrastive Predictive Coding},
  author={van den Oord, Aaron and Li, Yazhe and Vinyals, Oriol},
  year={2018},
  eprint={1807.03748},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@inproceedings{chen2020simclr,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={Proceedings of the 37th International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={119},
  pages={1597--1607},
  year={2020},
  publisher={PMLR},
  address={Online}
}

@article{tsfresh2018,
  title={tsfresh -- A Python package for the automatic extraction of relevant features from time series},
  author={Christ, Maximilian and Braun, Nils and Neuffer, Julius and Kempa-Liehr, Andreas W.},
  journal={Neurocomputing},
  volume={307},
  pages={72--77},
  year={2018}
}

@inproceedings{lightgbm2017,
  title={LightGBM: A Highly Efficient Gradient Boosting Decision Tree},
  author={Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  pages={3146--3154},
  year={2017},
  publisher={Curran Associates, Inc.},
  address={Long Beach, CA}
}
"""


def estimate_pdf_pages(pdf_path: Path) -> int | None:
    if not pdf_path.exists():
        return None
    data = pdf_path.read_bytes()
    matches = re.findall(rb"/Type\s*/Page\b", data)
    return len(matches) if matches else None


def read_build_status() -> dict[str, object]:
    status: dict[str, object] = {
        "pdf_exists": PHASE48_PDF_PATH.exists(),
        "pdf_bytes": PHASE48_PDF_PATH.stat().st_size if PHASE48_PDF_PATH.exists() else None,
        "pdf_pages": estimate_pdf_pages(PHASE48_PDF_PATH),
        "latex_log_exists": PHASE48_LOG_PATH.exists(),
        "bibtex_log_exists": PHASE48_BLG_PATH.exists(),
        "warning_markers": [],
    }
    warnings: list[str] = []
    if PHASE48_LOG_PATH.exists():
        log_text = PHASE48_LOG_PATH.read_text(encoding="utf-8", errors="replace")
        page_match = re.search(r"Output written on .+\.pdf \((\d+) pages?,\s*(\d+) bytes\)", log_text)
        if status["pdf_pages"] is None and page_match:
            status["pdf_pages"] = int(page_match.group(1))
        for marker in ["Overfull \\hbox", "Class acmart Warning", "LaTeX Warning", "Package balance Warning"]:
            if marker in log_text:
                warnings.append(marker)
    if PHASE48_BLG_PATH.exists():
        blg_text = PHASE48_BLG_PATH.read_text(encoding="utf-8", errors="replace")
        if "Warning--" in blg_text:
            warnings.append("BibTeX Warning")
    status["warning_markers"] = sorted(set(warnings))
    return status


def build_warning_audit(build_status: dict[str, object]) -> pd.DataFrame:
    pdf_exists = bool(build_status["pdf_exists"])
    warnings = list(build_status["warning_markers"])
    logs_exist = bool(build_status["latex_log_exists"]) and bool(build_status["bibtex_log_exists"])
    return pd.DataFrame(
        [
            {
                "check_id": "pdf_compilation",
                "status": "pass" if pdf_exists else "conditional_pass",
                "detail": f"pdf_exists={pdf_exists}; pages={build_status['pdf_pages']}; bytes={build_status['pdf_bytes']}",
            },
            {
                "check_id": "build_logs_available",
                "status": "pass" if logs_exist else "review_required",
                "detail": f"latex_log={build_status['latex_log_exists']}; bibtex_log={build_status['bibtex_log_exists']}",
            },
            {
                "check_id": "warning_cleanup",
                "status": "pass" if pdf_exists and logs_exist and not warnings else "review_required",
                "detail": "warnings=" + ",".join(warnings) if warnings else "No warning markers detected after log inspection.",
            },
            {
                "check_id": "page_budget",
                "status": "pass" if isinstance(build_status["pdf_pages"], int) and int(build_status["pdf_pages"]) <= 8 else "review_required",
                "detail": f"pages={build_status['pdf_pages']}; active venue limit must still be verified.",
            },
        ]
    )


def build_quality_audit(tex_text: str, claim_trace: pd.DataFrame, warning_audit: pd.DataFrame) -> pd.DataFrame:
    blocked_claims = claim_trace[claim_trace["status"] == "blocked"]
    forbidden_phrases = [
        "is a profitable trading strategy",
        "presents a profitable trading strategy",
        "beats all baselines",
        "guarantees",
    ]
    forbidden_found = [phrase for phrase in forbidden_phrases if phrase in tex_text.lower()]
    return pd.DataFrame(
        [
            {
                "check_id": "anonymous_review_mode",
                "status": "pass" if r"\documentclass[sigconf,anonymous,review]{acmart}" in tex_text else "fail",
                "detail": "Anonymous ACM review class is present.",
            },
            {
                "check_id": "claim_boundaries",
                "status": "pass" if blocked_claims.shape[0] >= 3 and not forbidden_found else "fail",
                "detail": f"blocked_claims={blocked_claims.shape[0]}; forbidden_found={forbidden_found}",
            },
            {
                "check_id": "locked_holdout_boundary",
                "status": "pass" if "cannot be reused for model rescue" in tex_text else "fail",
                "detail": "Same-holdout rescue is blocked in manuscript text.",
            },
            {
                "check_id": "candidate_switching_boundary",
                "status": "pass" if "post-hoc model selection" in tex_text else "fail",
                "detail": "Candidate switching boundary is explicit.",
            },
            {
                "check_id": "negative_result_framing",
                "status": "pass" if "not profitable-alpha support" in tex_text or "not a trading claim" in tex_text else "fail",
                "detail": "Paper frames result as limited/negative rather than deployable alpha.",
            },
            {
                "check_id": "pdf_warning_state_recorded",
                "status": "pass" if "warning_cleanup" in set(warning_audit["check_id"].astype(str)) else "fail",
                "detail": "Warning audit is present.",
            },
        ]
    )


def build_report(
    quality_audit: pd.DataFrame,
    warning_audit: pd.DataFrame,
    claim_trace: pd.DataFrame,
    section_budget: pd.DataFrame,
) -> str:
    return f"""# Phase 48 Paper Polish Report

## Purpose

Phase 48 converts the Phase 47 compiled draft into a stronger review-ready manuscript package. It does not tune models, change labels, select a new candidate, rerun the locked holdout, or reinterpret locked/final evaluation data.

## Outputs

- `paper/phase48_review_ready_manuscript.tex`
- `paper/phase48_references.bib`
- `paper/phase48_review_ready_manuscript.pdf` after local compilation
- `models/phase48_paper_quality_audit.csv`
- `models/phase48_claim_traceability.csv`
- `models/phase48_latex_warning_audit.csv`
- `models/phase48_section_budget.csv`
- `reports/phase48_reviewer_reading_guide.md`
- `reports/phase48_latex_warning_resolution.md`

## Quality Audit

{markdown_table(quality_audit, list(quality_audit.columns))}

## LaTeX / PDF Warning Audit

{markdown_table(warning_audit, list(warning_audit.columns))}

## Claim Traceability

{markdown_table(claim_trace, list(claim_trace.columns))}

## Section Budget

{markdown_table(section_budget, list(section_budget.columns))}

## Decision

The Phase 48 manuscript is a stronger review draft, not a submitted paper. The allowed contribution is limited locked relative support plus validation repair discipline. Profitable-alpha, deployment, candidate-switching, and same-holdout retuning claims remain blocked.
"""


def build_reviewer_guide(claim_trace: pd.DataFrame) -> str:
    allowed = claim_trace[claim_trace["status"] == "allowed"]
    blocked = claim_trace[claim_trace["status"] == "blocked"]
    return f"""# Phase 48 Reviewer Reading Guide

## What to read first

1. Start with the abstract and introduction for the validation-repair story.
2. Read Table 1 for evidence roles and claim boundaries.
3. Read Tables 2 and 3 for the locked external holdout and prewritten primary rule.
4. Read the discussion before judging the result as a trading system: the paper explicitly does not make that claim.

## Allowed claims

{markdown_table(allowed, list(allowed.columns))}

## Blocked claims

{markdown_table(blocked, list(blocked.columns))}

## Reviewer-facing one-sentence summary

The paper reports a leakage-safe regime-learning benchmark where HMM-guided contrastive regimes receive limited locked relative IC/Sharpe support, while profitable/tradable alpha remains unsupported.
"""


def build_warning_plan(warning_audit: pd.DataFrame) -> str:
    return f"""# Phase 48 LaTeX / PDF Warning Resolution Plan

## Current warning audit

{markdown_table(warning_audit, list(warning_audit.columns))}

## Required before external submission

1. Recompile from a clean LaTeX build directory.
2. Inspect the PDF visually page by page.
3. Remove or resolve any overfull table/text boxes.
4. Confirm ACM reference/copyright requirements for the active venue.
5. Check PDF metadata for author, institution, local path, and software identity leakage.
6. Re-run the research-grade gate after any manuscript source change.

This plan is intentionally conservative: a compiled PDF is not the same as an externally submissible PDF.
"""


def main() -> None:
    args = parse_args()
    locked_results = read_csv(LOCKED_RESULTS_PATH)
    locked_primary = read_csv(LOCKED_PRIMARY_PATH)
    locked_claims = read_csv(LOCKED_CLAIMS_PATH)
    evidence_matrix = read_csv(EVIDENCE_MATRIX_PATH)
    phase47_audit = read_csv(PHASE47_AUDIT_PATH)
    research_gate = read_csv(RESEARCH_GATE_PATH)

    validate_inputs(locked_results, locked_primary, locked_claims, evidence_matrix, phase47_audit, research_gate)

    tex_text = build_tex(locked_results, locked_primary, evidence_matrix)
    bib_text = build_bib()
    build_status = read_build_status()
    warning_audit = build_warning_audit(build_status)
    claim_trace = build_claim_traceability(locked_primary, locked_claims)
    section_budget = build_section_budget()
    quality_audit = build_quality_audit(tex_text, claim_trace, warning_audit)
    report = build_report(quality_audit, warning_audit, claim_trace, section_budget)
    reviewer_guide = build_reviewer_guide(claim_trace)
    warning_plan = build_warning_plan(warning_audit)

    generated_text = "\n".join([tex_text, bib_text, report, reviewer_guide, warning_plan])
    for phrase in [
        "does not make that claim",
        "no same-holdout retuning",
        "not a submitted paper",
        "Profitable-alpha",
        "limited locked relative support",
    ]:
        if phrase not in generated_text:
            raise ValueError(f"Generated Phase 48 artifacts are missing guardrail phrase: {phrase}")

    if args.dry_run:
        print("OK: Phase 48 paper polish inputs and guardrails validated.")
        return

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    PHASE48_TEX_PATH.write_text(tex_text, encoding="utf-8")
    PHASE48_BIB_PATH.write_text(bib_text, encoding="utf-8")
    PHASE48_REPORT_PATH.write_text(report, encoding="utf-8")
    PHASE48_REVIEWER_GUIDE_PATH.write_text(reviewer_guide, encoding="utf-8")
    PHASE48_WARNING_PLAN_PATH.write_text(warning_plan, encoding="utf-8")
    write_csv(PHASE48_QUALITY_AUDIT_PATH, quality_audit)
    write_csv(PHASE48_CLAIM_TRACE_PATH, claim_trace)
    write_csv(PHASE48_WARNING_AUDIT_PATH, warning_audit)
    write_csv(PHASE48_SECTION_BUDGET_PATH, section_budget)

    for path in [
        PHASE48_TEX_PATH,
        PHASE48_BIB_PATH,
        PHASE48_REPORT_PATH,
        PHASE48_REVIEWER_GUIDE_PATH,
        PHASE48_WARNING_PLAN_PATH,
        PHASE48_QUALITY_AUDIT_PATH,
        PHASE48_CLAIM_TRACE_PATH,
        PHASE48_WARNING_AUDIT_PATH,
        PHASE48_SECTION_BUDGET_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
