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

PHASE46_MANUSCRIPT_PATH = PAPER_DIR / "phase46_submission_manuscript.md"
PHASE46_SKELETON_PATH = PAPER_DIR / "phase46_acm_sigconf_skeleton.tex"
PHASE46_CLAIM_AUDIT_PATH = MODELS_DIR / "phase46_final_claim_audit.csv"
PHASE46_GATE_MATRIX_PATH = MODELS_DIR / "phase46_submission_gate_matrix.csv"
PHASE45_TABLE_PLAN_PATH = MODELS_DIR / "phase45_table_plan.csv"
PHASE45_FIGURE_PLAN_PATH = MODELS_DIR / "phase45_figure_plan.csv"
LOCKED_RESULTS_PATH = MODELS_DIR / "phase43b_locked_external_experiment_results.csv"
LOCKED_CLAIMS_PATH = MODELS_DIR / "phase43b_locked_external_claims.csv"
LOCKED_PRIMARY_PATH = MODELS_DIR / "phase43b_locked_external_primary_comparison.csv"
EVIDENCE_MATRIX_PATH = MODELS_DIR / "phase44_paper_evidence_matrix.csv"
RESEARCH_GATE_PATH = MODELS_DIR / "research_grade_check_report.csv"

SUBMISSION_DRAFT_PATH = PAPER_DIR / "phase47_submission_draft.tex"
REFERENCES_PATH = PAPER_DIR / "phase47_references.bib"
COMPILED_PDF_PATH = PAPER_DIR / "phase47_submission_draft.pdf"
LATEX_LOG_PATH = PAPER_DIR / "phase47_submission_draft.log"
BIBTEX_LOG_PATH = PAPER_DIR / "phase47_submission_draft.blg"
BUILD_REPORT_PATH = REPORTS_DIR / "phase47_submission_build_report.md"
BLIND_REVIEW_REPORT_PATH = REPORTS_DIR / "phase47_blind_review_hardening.md"
GAP_LIST_PATH = REPORTS_DIR / "phase47_camera_ready_gap_list.md"

BUILD_AUDIT_PATH = MODELS_DIR / "phase47_manuscript_build_audit.csv"
TABLE_MANIFEST_PATH = MODELS_DIR / "phase47_table_manifest.csv"
FIGURE_MANIFEST_PATH = MODELS_DIR / "phase47_figure_manifest.csv"
ANONYMITY_AUDIT_PATH = MODELS_DIR / "phase47_anonymity_source_audit.csv"
REFERENCE_MANIFEST_PATH = MODELS_DIR / "phase47_reference_manifest.csv"

FINAL_CANDIDATE = "regime_lgbm_hmm_guided_hmm"
PRIMARY_REFERENCES = ["global_lgbm", "regime_lgbm_hmm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 47 submission manuscript and blind-review hardening artifacts."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and generated artifacts without writing files.",
    )
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 47 input is missing: {path}")
    return pd.read_csv(path)


def read_required_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 47 input is missing: {path}")
    return path.read_text(encoding="utf-8")


def estimate_pdf_pages(pdf_path: Path) -> int | None:
    """Estimate page count from a generated PDF without requiring extra PDF packages."""
    if not pdf_path.exists():
        return None
    try:
        data = pdf_path.read_bytes()
    except OSError:
        return None
    matches = re.findall(rb"/Type\s*/Page\b", data)
    return len(matches) if matches else None


def read_pdf_build_status(
    pdf_path: Path = COMPILED_PDF_PATH,
    log_path: Path = LATEX_LOG_PATH,
    bibtex_log_path: Path = BIBTEX_LOG_PATH,
) -> dict[str, object]:
    """Return conservative local PDF build evidence for the submission audit."""
    status: dict[str, object] = {
        "exists": pdf_path.exists(),
        "bytes": pdf_path.stat().st_size if pdf_path.exists() else None,
        "pages": estimate_pdf_pages(pdf_path),
        "warnings": [],
        "log_available": log_path.exists(),
        "bibtex_log_available": bibtex_log_path.exists(),
    }
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        log_page_match = re.search(r"Output written on .+\.pdf \((\d+) pages?,\s*(\d+) bytes\)", log_text)
        if status["pages"] is None and log_page_match:
            status["pages"] = int(log_page_match.group(1))
        warnings = []
        for marker in ["Overfull \\hbox", "Class acmart Warning", "Package balance Warning", "BibTeX warning"]:
            if marker in log_text:
                warnings.append(marker)
        status["warnings"] = sorted(set(warnings))
    if bibtex_log_path.exists():
        bibtex_log_text = bibtex_log_path.read_text(encoding="utf-8", errors="replace")
        if "Warning--" in bibtex_log_text:
            status["warnings"] = sorted(set([*list(status["warnings"]), "BibTeX Warning"]))
    return status


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


def markdown_pct(value: object, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return str(value)


def latex_escape(value: object) -> str:
    text = str(value)
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
    return "".join(replacements.get(char, char) for char in text)


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


def validate_inputs(
    phase46_manuscript: str,
    phase46_skeleton: str,
    locked_results: pd.DataFrame,
    locked_claims: pd.DataFrame,
    locked_primary: pd.DataFrame,
    claim_audit: pd.DataFrame,
    gate_matrix: pd.DataFrame,
    research_gate: pd.DataFrame,
) -> None:
    required_text = [
        "does not claim a tradable strategy",
        "limited locked relative support",
        "negative Sharpe",
        "negative total return",
    ]
    missing = [phrase for phrase in required_text if phrase not in phase46_manuscript]
    if missing:
        raise ValueError(f"Phase 46 manuscript is missing required claim boundaries: {missing}")

    if r"\documentclass[sigconf,anonymous,review]{acmart}" not in phase46_skeleton:
        raise ValueError("Phase 46 ACM skeleton is not anonymous ACM sigconf review mode.")

    methods = set(locked_results.get("method", pd.Series(dtype=str)).astype(str))
    required_methods = {FINAL_CANDIDATE, *PRIMARY_REFERENCES}
    if not required_methods.issubset(methods):
        raise ValueError(f"Locked results missing required methods: {sorted(required_methods - methods)}")

    claim_map = locked_claims.set_index("claim_id")["claim_status"].astype(str).to_dict()
    if claim_map.get("positive_tradable_alpha") != "not_supported":
        raise ValueError("Positive tradable alpha must remain not_supported.")
    if claim_map.get("same_holdout_retuning") != "forbidden":
        raise ValueError("Same-holdout retuning must remain forbidden.")

    primary_refs = set(locked_primary.get("reference_method", pd.Series(dtype=str)).astype(str))
    if not set(PRIMARY_REFERENCES).issubset(primary_refs):
        raise ValueError("Locked primary comparison must include global_lgbm and regime_lgbm_hmm.")

    claim_text = " ".join(claim_audit.astype(str).agg(" ".join, axis=1).tolist())
    for phrase in ["Positive tradable alpha is not supported", "The strategy is profitable or deployable"]:
        if phrase not in claim_text:
            raise ValueError(f"Phase 46 claim audit missing guardrail phrase: {phrase}")

    gate_map = gate_matrix.set_index("gate")["status"].astype(str).to_dict()
    if gate_map.get("final_submission") != "not_yet":
        raise ValueError("Phase 47 starts from a not-yet-submitted Phase 46 gate.")

    if "status" in research_gate.columns:
        failures = research_gate[research_gate["status"].astype(str) == "FAIL"]
        if not failures.empty:
            raise ValueError("Research-grade gate has failures; do not build submission draft.")


def build_table_manifest(
    locked_results: pd.DataFrame,
    locked_primary: pd.DataFrame,
    evidence_matrix: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table_id": "T1",
                "title": "Evidence blocks and allowed claim boundaries",
                "source_artifact": EVIDENCE_MATRIX_PATH.relative_to(BASE_DIR).as_posix(),
                "status": "included_in_phase47_draft" if not evidence_matrix.empty else "source_missing",
                "paper_purpose": "Separates development-observed evidence from locked confirmatory evidence.",
            },
            {
                "table_id": "T2",
                "title": "Locked external holdout method comparison",
                "source_artifact": LOCKED_RESULTS_PATH.relative_to(BASE_DIR).as_posix(),
                "status": "included_in_phase47_draft" if not locked_results.empty else "source_missing",
                "paper_purpose": "Reports all locked methods with equal coverage and no candidate switching.",
            },
            {
                "table_id": "T3",
                "title": "Prewritten primary locked comparison",
                "source_artifact": LOCKED_PRIMARY_PATH.relative_to(BASE_DIR).as_posix(),
                "status": "included_in_phase47_draft" if not locked_primary.empty else "source_missing",
                "paper_purpose": "Shows the exact relative IC/Sharpe rule for the frozen final candidate.",
            },
            {
                "table_id": "T4",
                "title": "Final claim audit",
                "source_artifact": PHASE46_CLAIM_AUDIT_PATH.relative_to(BASE_DIR).as_posix(),
                "status": "appendix_or_reviewer_pack",
                "paper_purpose": "Keeps final manuscript wording inside safe claim boundaries.",
            },
        ]
    )


def build_figure_manifest(phase45_figure_plan: pd.DataFrame) -> pd.DataFrame:
    planned = []
    for _, row in phase45_figure_plan.iterrows():
        section = str(row.get("paper_section", row.get("section", ""))).strip()
        purpose = str(row.get("purpose", "")).strip()
        title = str(row.get("title", row.get("figure", ""))).strip()
        if not title or title == "nan":
            title = f"{section}: {purpose}" if section and purpose else section or purpose or "planned figure"
        planned.append(
            {
                "figure_id": str(row.get("figure_id", row.get("id", "unknown"))),
                "title": title,
                "source_artifact": str(row.get("source_artifact", row.get("artifact", "phase45_figure_plan"))),
                "phase47_status": str(row.get("status", "planned_not_drawn")),
                "action": "Draw compact publication figure or keep as table if page budget is tight.",
            }
        )
    if not planned:
        planned = [
            {
                "figure_id": "F1",
                "title": "Validation repair timeline",
                "source_artifact": "reports/phase39r_neural_fold_local_results.md",
                "phase47_status": "planned_not_drawn",
                "action": "Draw a compact timeline of invalidated -> repaired -> locked evaluation.",
            },
            {
                "figure_id": "F2",
                "title": "Locked holdout primary comparison",
                "source_artifact": "models/phase43b_locked_external_primary_comparison.csv",
                "phase47_status": "table_substitute_available",
                "action": "Use T3 if figure space is unavailable.",
            },
        ]
    return pd.DataFrame(planned)


def build_reference_manifest() -> pd.DataFrame:
    rows = [
        ("lopezdeprado2018afml", "financial_ml_validation", "Purged/embargoed financial ML validation framing."),
        ("hamilton1989regime", "regime_switching", "Classical regime-switching model foundation."),
        ("rabiner1989hmm", "hidden_markov_models", "HMM tutorial/reference for sequential latent states."),
        ("oord2018cpc", "contrastive_learning", "Contrastive predictive coding representation learning."),
        ("chen2020simclr", "contrastive_learning", "Modern contrastive objective and augmentation framing."),
        ("tsfresh2018", "time_series_features", "Time-series feature extraction context."),
        ("lightgbm2017", "gradient_boosting", "LightGBM baseline model citation."),
        ("acmart", "venue_formatting", "ACM proceedings template placeholder; verify current venue files before submission."),
    ]
    return pd.DataFrame(rows, columns=["citation_key", "cluster", "paper_role"])


def build_anonymity_audit(tex_text: str, bib_text: str) -> pd.DataFrame:
    checks = [
        ("author_block", [r"\\author\s*\{", r"\\affiliation\s*\{"], "No author or affiliation block in review draft."),
        ("acknowledgements", [r"(?i)acknowledg"], "Acknowledgements must be absent for double-blind review."),
        ("github_link", [r"github\.com"], "Repository links can reveal identity during review."),
        ("personal_name_sakshi", [r"\bSakshi\b"], "Personal name must not appear in anonymous source."),
        ("institution_marker", [r"\bBTech\b", r"\binstitution\b"], "Institution/student identity wording should not appear."),
        ("tool_marker", [r"\bCodex\b"], "Tool names should not appear in the anonymous paper story."),
    ]
    combined = f"{tex_text}\n{bib_text}"
    rows = []
    for check_id, patterns, action in checks:
        found = any(re.search(pattern, combined) for pattern in patterns)
        rows.append(
            {
                "check_id": check_id,
                "status": "review_required" if found else "pass",
                "action": action,
            }
        )
    return pd.DataFrame(rows)


def method_label(method: str) -> str:
    labels = {
        "global_lgbm": "Global LightGBM",
        "regime_lgbm_hmm": "Raw HMM + regime LightGBM",
        "regime_lgbm_hmm_guided_hmm": "Guided-HMM + regime LightGBM",
        "regime_lgbm_hmm_guided_gmm": "Guided-GMM + regime LightGBM",
        "regime_lgbm_contrastive": "Vanilla contrastive-GMM + regime LightGBM",
        "regime_lgbm_contrastive_hmm": "Vanilla contrastive-HMM + regime LightGBM",
        "regime_lgbm_kmeans": "KMeans + regime LightGBM",
        "regime_lgbm_vol_bucket": "Volatility buckets + regime LightGBM",
    }
    return labels.get(method, method)


def latex_locked_results_table(locked_results: pd.DataFrame) -> str:
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
    rows = []
    indexed = locked_results.set_index("method")
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
                    str(int(row["n_test_rows"])),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def latex_primary_table(locked_primary: pd.DataFrame) -> str:
    rows = []
    for _, row in locked_primary.iterrows():
        rows.append(
            " & ".join(
                [
                    latex_escape(method_label(str(row["reference_method"]))),
                    fmt(row["delta_mean_asset_IC"], 4),
                    fmt(row["delta_Sharpe"], 3),
                    str(bool(row["coverage_equal"])),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def latex_evidence_table(evidence_matrix: pd.DataFrame) -> str:
    selected = evidence_matrix.head(5).copy()
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            " & ".join(
                [
                    latex_escape(row.get("evidence_block", row.get("block", "evidence"))),
                    latex_escape(row.get("data_role", "unknown")),
                    latex_escape(row.get("claim_boundary", "paper-safe boundary")),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def build_submission_tex(
    locked_results: pd.DataFrame,
    locked_primary: pd.DataFrame,
    evidence_matrix: pd.DataFrame,
) -> str:
    final = locked_results[locked_results["method"] == FINAL_CANDIDATE].iloc[0]
    global_ref = locked_results[locked_results["method"] == "global_lgbm"].iloc[0]
    hmm_ref = locked_results[locked_results["method"] == "regime_lgbm_hmm"].iloc[0]
    return rf"""\documentclass[sigconf,anonymous,review]{{acmart}}

\settopmatter{{printacmref=false}}
\renewcommand\footnotetextcopyrightpermission[1]{{}}
\acmConference[Anonymous Review]{{Anonymous Review}}{{}}{{}}
\acmYear{{2026}}

\title{{HMM-Guided Contrastive Representations for Leakage-Safe Regime-Conditioned Crypto Alpha Evaluation}}

\begin{{document}}

\begin{{abstract}}
Regime-conditioned alpha models are attractive because financial relationships can change across trend, stress, and transition periods. However, regime labels are latent and easy to overfit, especially in multi-asset crypto backtests. This paper studies whether Hidden Markov Model guided contrastive representations improve regime-conditioned crypto alpha modeling under leakage-safe validation. The central contribution is a validation repair: an initially positive-looking positional-fold result is retained only as audit history after discovering cross-asset calendar overlap, then replaced with common-calendar fold-local validation, frozen-candidate rules, and a one-shot locked external holdout. On the registered 10-asset external holdout, the frozen guided-HMM candidate improves mean asset IC versus global LightGBM ({fmt(final["mean_asset_IC"], 4)} versus {fmt(global_ref["mean_asset_IC"], 4)}) and raw-feature HMM ({fmt(final["mean_asset_IC"], 4)} versus {fmt(hmm_ref["mean_asset_IC"], 4)}), with non-worse Sharpe versus both references ({fmt(final["Sharpe"], 3)} versus {fmt(global_ref["Sharpe"], 3)} and {fmt(hmm_ref["Sharpe"], 3)}). The finding is deliberately limited: the locked final candidate still has negative Sharpe and negative total return ({pct(final["total_return"], 1)}), so the paper does not claim a tradable strategy.
\end{{abstract}}

\keywords{{financial machine learning, regime learning, contrastive learning, hidden Markov models, validation, crypto assets}}

\maketitle

\section{{Introduction}}
Financial machine-learning papers often fail because the validation protocol is easier to overfit than the model. This paper studies a regime-learning system under a deliberately strict empirical protocol. The original development path produced an exciting guided-regime result, but a later audit found that per-symbol positional folds overlapped in calendar time. Rather than treating that result as evidence, the project invalidated it, kept it as audit history, and rebuilt the benchmark around common-calendar fold-local evaluation.

The resulting paper is not a trading-system paper. It asks a narrower mechanism question: can sequentially guided representation learning produce regimes that are more useful than global models and raw-feature HMM regimes under leakage-safe validation? The answer is mixed and paper-safe. The frozen guided-HMM candidate satisfies a prewritten locked relative IC/Sharpe comparison against two primary references, but locked Sharpe and total return remain negative. The contribution is therefore a research-grade evaluation framework and a mechanism boundary for HMM-guided contrastive regime learning.

\section{{Related Work}}
The project sits at the intersection of financial ML validation, latent regime modeling, contrastive representation learning, and reproducible empirical benchmarking. Purged and walk-forward validation are central in financial ML because adjacent labels and overlapping horizons can create optimistic backtests \cite{{lopezdeprado2018afml}}. Regime-switching models and HMMs provide a classical way to impose temporal persistence on latent market states \cite{{hamilton1989regime,rabiner1989hmm}}. Contrastive learning provides a flexible representation-learning objective for sequential data \cite{{oord2018cpc,chen2020simclr}}, while gradient boosting remains a strong tabular baseline in finance-style feature sets \cite{{lightgbm2017}}.

\section{{Data and Validation Protocol}}
The repository separates evidence into development-observed and locked-confirmatory roles. Crypto-20 development evidence is used for validation repair, diagnosis, and candidate freezing. The external locked holdout is registered before model outcomes are inspected and contains 10 external assets, 18 folds, and 129,600 out-of-sample rows per method. The same locked holdout cannot be reused for threshold tuning, feature selection, label selection, architecture search, or candidate switching.

All predictive comparisons use common-calendar fold-local validation. Feature scaling, weak-supervision HMM fitting, contrastive pair construction, encoder training, regime assignment, and downstream LightGBM fitting are authorized only inside the training interval for each fold. This design directly addresses the earlier cross-asset calendar-overlap failure.

\section{{Methods}}
The benchmark compares a global LightGBM model with regime-conditioned variants based on raw-feature HMM states, KMeans states, volatility buckets, vanilla contrastive regimes, contrastive-HMM regimes, HMM-guided contrastive-GMM regimes, and HMM-guided contrastive-HMM regimes. The frozen final candidate is the HMM-guided contrastive-HMM representation followed by regime-conditioned LightGBM. HMM states are treated as weak proxy supervision and a sequential reference, not as ground-truth market regimes.

\section{{Results}}
\begin{{table*}}[t]
\caption{{Evidence roles and claim boundaries.}}
\label{{tab:evidence-boundaries}}
\begin{{tabular}}{{lll}}
\toprule
Evidence block & Data role & Claim boundary \\
\midrule
{latex_evidence_table(evidence_matrix)}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{table*}}[t]
\caption{{Locked external holdout performance. All methods have equal locked coverage; rows are not independent statistical units.}}
\label{{tab:locked-results}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Method & Mean asset IC & Sharpe & Total return & Drawdown & Rows \\
\midrule
{latex_locked_results_table(locked_results)}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{table}}[t]
\caption{{Prewritten primary comparison for the frozen final candidate.}}
\label{{tab:primary-rule}}
\begin{{tabular}}{{lrrc}}
\toprule
Reference & $\Delta$ IC & $\Delta$ Sharpe & Equal coverage \\
\midrule
{latex_primary_table(locked_primary)}
\bottomrule
\end{{tabular}}
\end{{table}}

Table~\ref{{tab:locked-results}} shows that the frozen guided-HMM candidate has mean asset IC {fmt(final["mean_asset_IC"], 4)}, Sharpe {fmt(final["Sharpe"], 3)}, and total return {pct(final["total_return"], 1)}. Table~\ref{{tab:primary-rule}} records the prewritten relative comparison against global LightGBM and raw-feature HMM. This is limited locked relative support. It is not a profitable-alpha result.

The higher-IC guided-GMM diagnostic row does not replace the frozen final candidate. It was not the pre-registered final candidate and has worse locked Sharpe and total return. Switching to it after seeing the locked holdout would be post-hoc selection.

\section{{Discussion and Limitations}}
The strongest interpretation is empirical discipline rather than trading profitability. The method can improve a narrow locked relative rule while still failing to produce positive locked Sharpe or positive locked return. HMM guidance may improve sequential structure, but that structure is not automatically an exploitable trading edge after transaction costs and fold-local validation. The paper therefore blocks broad dominance, deployment, and profitability claims.

Limitations include the crypto-only universe, weak proxy regime labels, overlapping financial outcomes, and the single spent locked external holdout. The locked holdout cannot be reused for model rescue. Any future model improvement must use new development evidence or a newly registered confirmatory test.

\section{{Reproducibility}}
The repository includes run scripts, curated result tables, claim-control reports, artifact manifests, and a research-grade regression gate. Bulky row-level predictions and raw data are excluded from Git when they are too large or reproducible. Persistent artifact availability should not be claimed until the release is archived in a repository with a DOI or equivalent permanent identifier.

\section{{Conclusion}}
HMM-guided contrastive regimes receive limited locked-holdout relative support, but not profitable-alpha support. The contribution is a leakage-safe benchmark path: identify an attractive result, invalidate it when the validation audit fails, repair the protocol, freeze a candidate, spend one locked holdout, and report the boundary honestly.

\bibliographystyle{{ACM-Reference-Format}}
\bibliography{{phase47_references}}

\end{{document}}
"""


def build_references_bib() -> str:
    return r"""@book{lopezdeprado2018afml,
  title={Advances in Financial Machine Learning},
  author={L{\'o}pez de Prado, Marcos},
  year={2018},
  publisher={Wiley}
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

@inproceedings{oord2018cpc,
  title={Representation Learning with Contrastive Predictive Coding},
  author={van den Oord, Aaron and Li, Yazhe and Vinyals, Oriol},
  booktitle={arXiv preprint arXiv:1807.03748},
  year={2018}
}

@inproceedings{chen2020simclr,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  booktitle={International Conference on Machine Learning},
  year={2020}
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
  year={2017}
}

@misc{acmart,
  title={ACM Master Article Template},
  author={{Association for Computing Machinery}},
  year={2026},
  note={Venue template placeholder; verify the current conference call and template before submission}
}
"""


def build_build_audit(
    tex_text: str,
    references_text: str,
    anonymity_audit: pd.DataFrame,
    pdf_status: dict[str, object] | None = None,
) -> pd.DataFrame:
    pdf_status = pdf_status or {"exists": False, "bytes": None, "pages": None, "warnings": []}
    pdf_exists = bool(pdf_status.get("exists"))
    pdf_pages = pdf_status.get("pages")
    pdf_bytes = pdf_status.get("bytes")
    pdf_warnings = list(pdf_status.get("warnings", []))
    log_available = bool(pdf_status.get("log_available", False))
    bibtex_log_available = bool(pdf_status.get("bibtex_log_available", False))
    if pdf_exists:
        pdf_detail = f"Compiled PDF exists at paper/phase47_submission_draft.pdf; pages={pdf_pages}; bytes={pdf_bytes}."
    else:
        pdf_detail = "Source is build-ready, but final PDF compilation/page count must be run with the current venue template."

    page_budget_ok = isinstance(pdf_pages, int) and pdf_pages <= 8
    warning_review_required = pdf_exists and (bool(pdf_warnings) or not log_available or not bibtex_log_available)
    if pdf_warnings:
        warning_detail = f"LaTeX/BibTeX completed with warning markers requiring human layout review: {', '.join(pdf_warnings)}."
    elif pdf_exists and (not log_available or not bibtex_log_available):
        warning_detail = "Compiled PDF exists, but local LaTeX/BibTeX logs are unavailable; rerun compilation before external submission and inspect warnings."
    else:
        warning_detail = "Local LaTeX and BibTeX logs were inspected and no warning markers were detected."
    rows = [
        {
            "check_id": "anonymous_acm_review_mode",
            "status": "pass" if r"\documentclass[sigconf,anonymous,review]{acmart}" in tex_text else "fail",
            "detail": "ACM sigconf anonymous review class is present.",
        },
        {
            "check_id": "bibliography_present",
            "status": "pass" if r"\bibliography{phase47_references}" in tex_text and "@book" in references_text else "fail",
            "detail": "Draft uses phase47_references.bib.",
        },
        {
            "check_id": "tables_present",
            "status": "pass" if tex_text.count(r"\begin{table") >= 3 else "fail",
            "detail": "Draft includes evidence, locked result, and primary-rule tables.",
        },
        {
            "check_id": "no_profitable_strategy_claim",
            "status": "pass" if "does not claim a tradable strategy" in tex_text and "profitable-alpha support" in tex_text else "fail",
            "detail": "Draft blocks profitability/tradability claims.",
        },
        {
            "check_id": "same_holdout_rescue_blocked",
            "status": "pass" if "locked holdout cannot be reused for model rescue" in tex_text else "fail",
            "detail": "Draft blocks same-holdout retuning.",
        },
        {
            "check_id": "source_anonymity",
            "status": "pass" if set(anonymity_audit["status"]) == {"pass"} else "review_required",
            "detail": "Source anonymity audit must pass before external review.",
        },
        {
            "check_id": "pdf_compilation",
            "status": "pass" if pdf_exists else "conditional_pass",
            "detail": pdf_detail,
        },
        {
            "check_id": "pdf_page_budget",
            "status": "pass" if page_budget_ok else ("review_required" if pdf_exists else "conditional_pass"),
            "detail": (
                f"Compiled draft page estimate is {pdf_pages}; verify against the active venue limit."
                if pdf_exists
                else "Page budget cannot be measured until a PDF is compiled."
            ),
        },
        {
            "check_id": "pdf_warning_review",
            "status": "review_required" if warning_review_required else ("pass" if pdf_exists else "conditional_pass"),
            "detail": warning_detail,
        },
        {
            "check_id": "artifact_archive",
            "status": "not_claimed",
            "detail": "No artifact availability claim until DOI/permanent archive exists.",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    build_audit: pd.DataFrame,
    table_manifest: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    reference_manifest: pd.DataFrame,
) -> str:
    return f"""# Phase 47 Submission Manuscript Build Report

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

{markdown_table(build_audit, list(build_audit.columns))}

## Table Manifest

{markdown_table(table_manifest, list(table_manifest.columns))}

## Figure Manifest

{markdown_table(figure_manifest, list(figure_manifest.columns))}

## Reference Manifest

{markdown_table(reference_manifest, list(reference_manifest.columns))}

## Decision

The manuscript source is now build-oriented and paper-safe. A local PDF can be included when compiled, but the package is still not a submitted paper: current venue-template verification, source/PDF metadata anonymity review, final figure drawing, warning cleanup, citation polish, and artifact archive/DOI decision remain before external submission.
"""


def build_blind_review_report(anonymity_audit: pd.DataFrame) -> str:
    return f"""# Phase 47 Blind-Review Hardening

## Purpose

This report checks the Phase 47 LaTeX source and BibTeX source for obvious double-blind risks. It is a source-level audit only; the compiled PDF content and metadata must still be checked before external submission.

{markdown_table(anonymity_audit, list(anonymity_audit.columns))}

## Required Human Check Before Submission

- Compile the PDF and inspect title page, headers, footers, metadata, and links.
- Remove acknowledgements from the review version.
- Avoid public repository links in the anonymous paper body.
- Verify that artifact archive links are allowed by the venue's double-blind policy before adding them.
"""


def build_gap_list(build_audit: pd.DataFrame) -> str:
    return f"""# Phase 47 Camera-Ready / Submission Gap List

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

{markdown_table(build_audit, list(build_audit.columns))}
"""


def main() -> None:
    args = parse_args()
    phase46_manuscript = read_required_text(PHASE46_MANUSCRIPT_PATH)
    phase46_skeleton = read_required_text(PHASE46_SKELETON_PATH)
    locked_results = read_required_csv(LOCKED_RESULTS_PATH)
    locked_claims = read_required_csv(LOCKED_CLAIMS_PATH)
    locked_primary = read_required_csv(LOCKED_PRIMARY_PATH)
    claim_audit = read_required_csv(PHASE46_CLAIM_AUDIT_PATH)
    gate_matrix = read_required_csv(PHASE46_GATE_MATRIX_PATH)
    table_plan = read_required_csv(PHASE45_TABLE_PLAN_PATH)
    figure_plan = read_required_csv(PHASE45_FIGURE_PLAN_PATH)
    evidence_matrix = read_required_csv(EVIDENCE_MATRIX_PATH)
    research_gate = read_required_csv(RESEARCH_GATE_PATH)

    validate_inputs(
        phase46_manuscript,
        phase46_skeleton,
        locked_results,
        locked_claims,
        locked_primary,
        claim_audit,
        gate_matrix,
        research_gate,
    )

    table_manifest = build_table_manifest(locked_results, locked_primary, evidence_matrix)
    if not table_plan.empty:
        table_manifest["phase45_plan_rows"] = len(table_plan)
    figure_manifest = build_figure_manifest(figure_plan)
    reference_manifest = build_reference_manifest()
    tex_text = build_submission_tex(locked_results, locked_primary, evidence_matrix)
    references_text = build_references_bib()
    anonymity_audit = build_anonymity_audit(tex_text, references_text)
    pdf_status = read_pdf_build_status()
    build_audit = build_build_audit(tex_text, references_text, anonymity_audit, pdf_status)
    report = build_report(build_audit, table_manifest, figure_manifest, reference_manifest)
    blind_review_report = build_blind_review_report(anonymity_audit)
    gap_list = build_gap_list(build_audit)

    generated_text = "\n".join([tex_text, references_text, report, blind_review_report, gap_list])
    for phrase in [
        "does not claim a tradable strategy",
        "limited locked relative support",
        "locked holdout cannot be reused for model rescue",
        "not a submitted paper",
        "artifact archive",
    ]:
        if phrase not in generated_text:
            raise ValueError(f"Generated Phase 47 text is missing guardrail phrase: {phrase}")

    if args.dry_run:
        print("OK: Phase 47 inputs and build-readiness guardrails validated.")
        return

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    SUBMISSION_DRAFT_PATH.write_text(tex_text, encoding="utf-8")
    REFERENCES_PATH.write_text(references_text, encoding="utf-8")
    BUILD_REPORT_PATH.write_text(report, encoding="utf-8")
    BLIND_REVIEW_REPORT_PATH.write_text(blind_review_report, encoding="utf-8")
    GAP_LIST_PATH.write_text(gap_list, encoding="utf-8")
    write_csv(BUILD_AUDIT_PATH, build_audit)
    write_csv(TABLE_MANIFEST_PATH, table_manifest)
    write_csv(FIGURE_MANIFEST_PATH, figure_manifest)
    write_csv(ANONYMITY_AUDIT_PATH, anonymity_audit)
    write_csv(REFERENCE_MANIFEST_PATH, reference_manifest)

    for path in [
        SUBMISSION_DRAFT_PATH,
        REFERENCES_PATH,
        BUILD_REPORT_PATH,
        BLIND_REVIEW_REPORT_PATH,
        GAP_LIST_PATH,
        BUILD_AUDIT_PATH,
        TABLE_MANIFEST_PATH,
        FIGURE_MANIFEST_PATH,
        ANONYMITY_AUDIT_PATH,
        REFERENCE_MANIFEST_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
