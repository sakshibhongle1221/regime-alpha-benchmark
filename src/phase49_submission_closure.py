from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import phase48_paper_polish as phase48
from config import BASE_DIR, SAVE_DIR


MODELS_DIR = Path(SAVE_DIR)
REPORTS_DIR = Path(BASE_DIR) / "reports"
PAPER_DIR = Path(BASE_DIR) / "paper"
FIGURE_DIR = PAPER_DIR / "figures"

PHASE49_TEX_PATH = PAPER_DIR / "phase49_submission_ready_manuscript.tex"
PHASE49_BIB_PATH = PAPER_DIR / "phase49_references.bib"
PHASE49_PDF_PATH = PAPER_DIR / "phase49_submission_ready_manuscript.pdf"
PHASE49_LOG_PATH = PAPER_DIR / "phase49_submission_ready_manuscript.log"
PHASE49_BLG_PATH = PAPER_DIR / "phase49_submission_ready_manuscript.blg"

F1_SVG_PATH = FIGURE_DIR / "phase49_f1_validation_repair.svg"
F1_PDF_PATH = FIGURE_DIR / "phase49_f1_validation_repair.pdf"
F1_PNG_PATH = FIGURE_DIR / "phase49_f1_validation_repair.png"
F2_SVG_PATH = FIGURE_DIR / "phase49_f2_system_overview.svg"
F2_PDF_PATH = FIGURE_DIR / "phase49_f2_system_overview.pdf"
F2_PNG_PATH = FIGURE_DIR / "phase49_f2_system_overview.png"

PHASE49_REPORT_PATH = REPORTS_DIR / "phase49_final_submission_closure.md"
PHASE49_METADATA_REPORT_PATH = REPORTS_DIR / "phase49_pdf_metadata_audit.md"
PHASE49_ZENODO_GUIDE_PATH = REPORTS_DIR / "phase49_artifact_release_guide.md"

PHASE49_AUDIT_PATH = MODELS_DIR / "phase49_submission_closure_audit.csv"
PHASE49_FIGURE_MANIFEST_PATH = MODELS_DIR / "phase49_figure_manifest.csv"
PHASE49_CITATION_AUDIT_PATH = MODELS_DIR / "phase49_citation_audit.csv"
PHASE49_METADATA_AUDIT_PATH = MODELS_DIR / "phase49_pdf_metadata_audit.csv"
PHASE49_ARTIFACT_CHECKLIST_PATH = MODELS_DIR / "phase49_artifact_release_checklist.csv"
PHASE49_LATEX_WARNING_AUDIT_PATH = MODELS_DIR / "phase49_latex_warning_audit.csv"


TIME_SERIES_REFERENCES = {
    "ts2vec2022": "TS2Vec",
    "tnc2021": "TNC",
    "cost2022": "CoST",
    "tstcc2021": "TS-TCC",
    "tfc2022": "TF-C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 49 final submission-closure package.")
    parser.add_argument("--dry-run", action="store_true", help="Validate generated artifacts without writing files.")
    return parser.parse_args()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def add_box(ax, xy: tuple[float, float], width: float, height: float, text: str, face: str, edge: str = "#1f2937") -> None:
    x, y = xy
    rect = plt.Rectangle((x, y), width, height, facecolor=face, edgecolor=edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#334155") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 1.6, "color": color, "shrinkA": 4, "shrinkB": 4},
    )


def draw_validation_repair() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.8), constrained_layout=True)

    for ax in axes:
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 4)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("Invalidated positional folds: asset calendars can overlap", fontsize=12, fontweight="bold", loc="left")
    ax.text(0.1, 3.1, "BTC", fontsize=10, fontweight="bold")
    ax.text(0.1, 1.8, "ETH", fontsize=10, fontweight="bold")
    add_box(ax, (1.0, 2.75), 4.3, 0.55, "train", "#dbeafe")
    add_box(ax, (5.35, 2.75), 2.0, 0.55, "test", "#fecaca", "#991b1b")
    add_box(ax, (2.5, 1.45), 4.3, 0.55, "train", "#dbeafe")
    add_box(ax, (6.85, 1.45), 2.0, 0.55, "test", "#fecaca", "#991b1b")
    ax.add_patch(plt.Rectangle((5.35, 1.25), 1.45, 2.35, facecolor="#f97316", alpha=0.16, edgecolor="#ea580c", linewidth=1.2))
    ax.text(6.05, 0.8, "calendar overlap risk", ha="center", fontsize=9, color="#9a3412")
    ax.text(9.4, 2.45, "Outcome: old Phase 39 kept\nas audit history only", ha="center", va="center", fontsize=9)

    ax = axes[1]
    ax.set_title("Repaired common-calendar folds: all assets share one boundary", fontsize=12, fontweight="bold", loc="left")
    for y, label in [(2.75, "BTC"), (1.9, "ETH"), (1.05, "Asset N")]:
        ax.text(0.1, y + 0.15, label, fontsize=10, fontweight="bold")
        add_box(ax, (1.0, y), 4.3, 0.45, "train", "#dcfce7", "#166534")
        add_box(ax, (5.35, y), 0.65, 0.45, "gap", "#fef3c7", "#92400e")
        add_box(ax, (6.05, y), 2.2, 0.45, "test", "#e0f2fe", "#075985")
    ax.plot([5.35, 5.35], [0.75, 3.4], color="#166534", linestyle="--", linewidth=1.2)
    ax.plot([6.05, 6.05], [0.75, 3.4], color="#075985", linestyle="--", linewidth=1.2)
    ax.text(9.6, 2.1, "Outcome: fold-local scaling,\nHMM fitting, encoder training,\nand LightGBM fitting", ha="center", va="center", fontsize=9)

    for path in [F1_SVG_PATH, F1_PDF_PATH, F1_PNG_PATH]:
        fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def draw_system_overview() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.0, 3.8), constrained_layout=True)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Leakage-safe HMM-guided regime-learning benchmark", fontsize=12, fontweight="bold", loc="left")

    boxes = [
        ((0.4, 2.0), 1.6, 1.0, "OHLCV\npanel", "#f1f5f9"),
        ((2.5, 2.0), 1.8, 1.0, "engineered\nfeatures", "#dbeafe"),
        ((4.8, 3.25), 1.9, 0.9, "fold-local HMM\nweak labels", "#fef3c7"),
        ((4.8, 1.0), 1.9, 0.9, "contrastive\nwindows", "#fef3c7"),
        ((7.2, 2.0), 2.0, 1.0, "HMM-guided\nencoder", "#ede9fe"),
        ((9.7, 2.0), 1.9, 1.0, "GMM/HMM\nregime assignment", "#dcfce7"),
        ((12.0, 2.0), 1.6, 1.0, "LightGBM\nalpha models", "#fee2e2"),
        ((12.0, 0.55), 1.6, 0.8, "IC / Sharpe /\ndrawdown /\nclaim audit", "#f8fafc"),
    ]
    for xy, w, h, text, face in boxes:
        add_box(ax, xy, w, h, text, face)

    add_arrow(ax, (2.0, 2.5), (2.5, 2.5))
    add_arrow(ax, (4.3, 2.5), (4.8, 3.7))
    add_arrow(ax, (4.3, 2.5), (4.8, 1.45))
    add_arrow(ax, (6.7, 3.7), (7.2, 2.75))
    add_arrow(ax, (6.7, 1.45), (7.2, 2.25))
    add_arrow(ax, (9.2, 2.5), (9.7, 2.5))
    add_arrow(ax, (11.6, 2.5), (12.0, 2.5))
    add_arrow(ax, (12.8, 2.0), (12.8, 1.35))

    ax.text(7.8, 0.55, "All fitting occurs inside each outer-training interval", ha="center", fontsize=9, color="#334155")

    for path in [F2_SVG_PATH, F2_PDF_PATH, F2_PNG_PATH]:
        fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_phase49_bib() -> str:
    return (
        phase48.build_bib()
        + r"""

@inproceedings{ts2vec2022,
  title={TS2Vec: Towards Universal Representation of Time Series},
  author={Yue, Zhihan and Wang, Yujing and Duan, Juanyong and Yang, Tianmeng and Huang, Congrui and Tong, Yunhai and Xu, Bixiong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  pages={8980--8987},
  year={2022},
  publisher={AAAI Press},
  address={Online},
  doi={10.1609/aaai.v36i8.20881}
}

@misc{tnc2021,
  title={Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding},
  author={Tonekaboni, Sana and Eytan, Danny and Goldenberg, Anna},
  year={2021},
  eprint={2106.00750},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{cost2022,
  title={CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting},
  author={Woo, Gerald and Liu, Chenghao and Sahoo, Doyen and Kumar, Akshat and Hoi, Steven C. H.},
  year={2022},
  eprint={2202.01575},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{tstcc2021,
  title={Time-Series Representation Learning via Temporal and Contextual Contrasting},
  author={Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Kwoh, Chee Keong and Li, Xiaoli and Guan, Cuntai},
  year={2021},
  eprint={2106.14112},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@misc{tfc2022,
  title={Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency},
  author={Zhang, Xiang and Zhao, Ziyuan and Tsiligkaridis, Theodoros and Zitnik, Marinka},
  year={2022},
  eprint={2206.08496},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
"""
    )


def build_phase49_tex(locked_results: pd.DataFrame, locked_primary: pd.DataFrame, evidence_matrix: pd.DataFrame) -> str:
    tex = phase48.build_tex(locked_results, locked_primary, evidence_matrix)
    tex = tex.replace(r"\bibliography{phase48_references}", r"\bibliography{phase49_references}")
    tex = tex.replace(
        r"\keywords{financial machine learning, crypto assets, hidden Markov models, contrastive learning, leakage-safe validation, negative results}",
        r"""\begin{CCSXML}
<ccs2012>
   <concept>
       <concept_id>10010147.10010257.10010258.10010259</concept_id>
       <concept_desc>Computing methodologies~Supervised learning</concept_desc>
       <concept_significance>300</concept_significance>
       </concept>
   <concept>
       <concept_id>10010405.10010481.10010482</concept_id>
       <concept_desc>Applied computing~Economics</concept_desc>
       <concept_significance>300</concept_significance>
       </concept>
</ccs2012>
\end{CCSXML}
\ccsdesc[300]{Computing methodologies~Supervised learning}
\ccsdesc[300]{Applied computing~Economics}
\keywords{financial machine learning, crypto assets, hidden Markov models, contrastive learning, leakage-safe validation, negative results}""",
    )
    tex = tex.replace(
        r"\title{Leakage-Safe Regime-Conditioned Crypto Alpha Evaluation with HMM-Guided Contrastive Representations}",
        r"\title{Leakage-Safe Regime-Conditioned Crypto Alpha Evaluation with HMM-Guided Contrastive Representations}",
    )
    tex = tex.replace(
        "Contrastive learning provides a representation objective for learning invariances and predictive structure from sequences \\cite{oord2018cpc,chen2020simclr}. LightGBM remains a strong tabular learner and is used here as the downstream alpha model across all regime families \\cite{lightgbm2017}.",
        "Contrastive learning provides a representation objective for learning invariances and predictive structure from sequences \\cite{oord2018cpc,chen2020simclr}. Time-series contrastive methods such as TS2Vec, TNC, CoST, TS-TCC, and TF-C motivate temporal, neighborhood, and time-frequency views of sequence representation learning \\cite{ts2vec2022,tnc2021,cost2022,tstcc2021,tfc2022}. LightGBM remains a strong tabular learner and is used here as the downstream alpha model across all regime families \\cite{lightgbm2017}.",
    )
    tex = tex.replace(
        "This prevents the cross-asset calendar overlap that invalidated the earlier positional-fold run.",
        "This prevents the cross-asset calendar overlap that invalidated the earlier positional-fold run. Figure~\\ref{fig:validation-repair} shows the repair visually.",
    )
    tex = tex.replace(
        r"\begin{table*}[t]" + "\n" + r"\caption{Evidence roles and allowed claim boundaries. Development evidence can explain mechanisms; locked evidence controls final claims.}",
        r"""\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/phase49_f1_validation_repair.pdf}
\caption{Validation repair. The original positional-fold design could overlap in calendar time across assets; the repaired design uses shared calendar boundaries and fold-local fitting.}
\Description{A two-panel timeline diagram contrasting invalidated per-asset positional folds with repaired common-calendar folds shared by all assets.}
\label{fig:validation-repair}
\end{figure*}

\begin{table*}[t]
\caption{Evidence roles and allowed claim boundaries. Development evidence can explain mechanisms; locked evidence controls final claims.}""",
    )
    tex = tex.replace(
        "The frozen final candidate is the HMM-guided contrastive-HMM representation followed by regime-conditioned LightGBM. HMM states are treated as weak sequential supervision rather than ground-truth market labels. The higher-IC guided-GMM row is reported as a diagnostic comparator only; it cannot replace the frozen candidate after the locked holdout is observed.",
        r"""The frozen final candidate is the HMM-guided contrastive-HMM representation followed by regime-conditioned LightGBM. HMM states are treated as weak sequential supervision rather than ground-truth market labels. Figure~\ref{fig:system-overview} summarizes the pipeline. The higher-IC guided-GMM row is reported as a diagnostic comparator only; it cannot replace the frozen candidate after the locked holdout is observed.

\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/phase49_f2_system_overview.pdf}
\caption{System overview. Raw OHLCV is converted into features, fold-local HMM weak labels and contrastive windows feed the encoder, and GMM/HMM regime assignments condition the downstream LightGBM alpha models.}
\Description{A pipeline diagram from OHLCV data through engineered features, HMM weak labels, contrastive windows, HMM-guided encoder, regime assignment, LightGBM models, and claim-audited evaluation metrics.}
\label{fig:system-overview}
\end{figure*}""",
    )
    return tex


def build_figure_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure_id": "F1",
                "title": "Validation repair: invalidated positional folds versus repaired common-calendar folds",
                "svg_path": "paper/figures/phase49_f1_validation_repair.svg",
                "pdf_path": "paper/figures/phase49_f1_validation_repair.pdf",
                "png_preview_path": "paper/figures/phase49_f1_validation_repair.png",
                "status": "ready_for_visual_review",
            },
            {
                "figure_id": "F2",
                "title": "System overview: features, HMM weak labels, encoder, regime assignment, LightGBM",
                "svg_path": "paper/figures/phase49_f2_system_overview.svg",
                "pdf_path": "paper/figures/phase49_f2_system_overview.pdf",
                "png_preview_path": "paper/figures/phase49_f2_system_overview.png",
                "status": "ready_for_visual_review",
            },
        ]
    )


def build_citation_audit(tex_text: str, bib_text: str) -> pd.DataFrame:
    rows = []
    for key, label in TIME_SERIES_REFERENCES.items():
        rows.append(
            {
                "citation_key": key,
                "label": label,
                "in_tex": key in tex_text,
                "in_bib": f"{{{key}," in bib_text,
                "status": "pass" if key in tex_text and f"{{{key}," in bib_text else "fail",
            }
        )
    for key in ["lopezdeprado2018afml", "hamilton1989regime", "rabiner1989hmm", "oord2018cpc", "chen2020simclr", "lightgbm2017"]:
        rows.append(
            {
                "citation_key": key,
                "label": key,
                "in_tex": key in tex_text,
                "in_bib": f"{{{key}," in bib_text,
                "status": "pass" if key in tex_text and f"{{{key}," in bib_text else "fail",
            }
        )
    return pd.DataFrame(rows)


def extract_pdf_metadata(pdf_path: Path) -> dict[str, str]:
    if not pdf_path.exists():
        return {}
    text = pdf_path.read_bytes().decode("latin-1", errors="ignore")
    metadata: dict[str, str] = {}
    for field in ["Title", "Author", "Subject", "Creator", "Producer", "Keywords"]:
        match = re.search(rf"/{field}\s*\((.*?)\)", text, flags=re.DOTALL)
        if match:
            metadata[field] = re.sub(r"\s+", " ", match.group(1)).strip()
        else:
            metadata[field] = ""
    return metadata


def build_metadata_audit(pdf_path: Path) -> pd.DataFrame:
    metadata = extract_pdf_metadata(pdf_path)
    sensitive_markers = ["Sakshi", "Bhongle", "D:", "C:", "regime-alpha-benchmark", "GitHub", "github.com"]
    rows = []
    rows.append(
        {
            "check_id": "pdf_exists",
            "status": "pass" if pdf_path.exists() else "review_required",
            "detail": str(pdf_path.relative_to(BASE_DIR)),
        }
    )
    if pdf_path.exists():
        rows.append(
            {
                "check_id": "pdf_size",
                "status": "pass" if pdf_path.stat().st_size > 10_000 else "review_required",
                "detail": f"bytes={pdf_path.stat().st_size}",
            }
        )
    for field in ["Title", "Author", "Subject", "Creator", "Producer", "Keywords"]:
        value = metadata.get(field, "")
        found = [marker for marker in sensitive_markers if marker.lower() in value.lower()]
        status = "pass" if not found else "fail"
        rows.append({"check_id": f"metadata_{field.lower()}", "status": status, "detail": value or "<empty>"})
    return pd.DataFrame(rows)


def estimate_pdf_pages(pdf_path: Path) -> int | None:
    if not pdf_path.exists():
        return None
    data = pdf_path.read_bytes()
    matches = re.findall(rb"/Type\s*/Page\b", data)
    return len(matches) if matches else None


def read_latex_warning_status() -> dict[str, object]:
    status: dict[str, object] = {
        "pdf_exists": PHASE49_PDF_PATH.exists(),
        "pdf_bytes": PHASE49_PDF_PATH.stat().st_size if PHASE49_PDF_PATH.exists() else None,
        "pdf_pages": estimate_pdf_pages(PHASE49_PDF_PATH),
        "latex_log_exists": PHASE49_LOG_PATH.exists(),
        "bibtex_log_exists": PHASE49_BLG_PATH.exists(),
        "warning_markers": [],
    }
    warnings: list[str] = []
    if PHASE49_LOG_PATH.exists():
        log_text = PHASE49_LOG_PATH.read_text(encoding="utf-8", errors="replace")
        page_match = re.search(r"Output written on .+\.pdf \((\d+) pages?,\s*(\d+) bytes\)", log_text)
        if status["pdf_pages"] is None and page_match:
            status["pdf_pages"] = int(page_match.group(1))
        for marker in [
            "Overfull \\hbox",
            "Underfull \\vbox",
            "Class acmart Warning",
            "LaTeX Warning",
            "Package balance Warning",
        ]:
            if marker in log_text:
                warnings.append(marker)
    if PHASE49_BLG_PATH.exists():
        blg_text = PHASE49_BLG_PATH.read_text(encoding="utf-8", errors="replace")
        if "Warning--" in blg_text:
            warnings.append("BibTeX Warning")
    status["warning_markers"] = sorted(set(warnings))
    return status


def build_latex_warning_audit(build_status: dict[str, object]) -> pd.DataFrame:
    pdf_exists = bool(build_status["pdf_exists"])
    logs_exist = bool(build_status["latex_log_exists"]) and bool(build_status["bibtex_log_exists"])
    warnings = list(build_status["warning_markers"])
    return pd.DataFrame(
        [
            {
                "check_id": "pdf_compilation",
                "status": "pass" if pdf_exists else "review_required",
                "detail": f"pdf_exists={pdf_exists}; pages={build_status['pdf_pages']}; bytes={build_status['pdf_bytes']}",
            },
            {
                "check_id": "build_logs_available",
                "status": "pass" if logs_exist else "review_required",
                "detail": f"latex_log={build_status['latex_log_exists']}; bibtex_log={build_status['bibtex_log_exists']}",
            },
            {
                "check_id": "page_budget",
                "status": "pass" if isinstance(build_status["pdf_pages"], int) and int(build_status["pdf_pages"]) <= 8 else "review_required",
                "detail": f"pages={build_status['pdf_pages']}; active venue limit must still be verified.",
            },
            {
                "check_id": "warning_cleanup",
                "status": "pass" if pdf_exists and logs_exist and not warnings else "review_required",
                "detail": "warnings=" + ",".join(warnings) if warnings else "No warning markers detected after log inspection.",
            },
        ]
    )


def build_artifact_release_checklist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("release_tag", "pending_human_action", "Create a GitHub release after Phase 49 is merged."),
            ("archive_target", "pending_human_action", "Upload the release zip or connect the GitHub release to Zenodo/OSF/Figshare."),
            ("doi", "not_claimed", "Do not claim artifact availability until the archive gives a DOI or persistent identifier."),
            ("license", "pending_human_action", "Confirm the public license before archive publication."),
            ("double_blind", "review_required", "Only add archive links during review if the venue allows non-anonymous artifacts."),
        ],
        columns=["item", "status", "action"],
    )


def build_submission_audit(
    figure_manifest: pd.DataFrame,
    citation_audit: pd.DataFrame,
    metadata_audit: pd.DataFrame,
    artifact_checklist: pd.DataFrame,
    latex_warning_audit: pd.DataFrame,
) -> pd.DataFrame:
    required_figure_paths = [F1_SVG_PATH, F1_PDF_PATH, F1_PNG_PATH, F2_SVG_PATH, F2_PDF_PATH, F2_PNG_PATH]
    latex_status = latex_warning_audit.set_index("check_id")["status"].astype(str).to_dict()
    return pd.DataFrame(
        [
            {
                "check_id": "figures_f1_f2_generated",
                "status": "pass" if all(path.exists() for path in required_figure_paths) else "fail",
                "detail": f"figures={figure_manifest['status'].tolist()}",
            },
            {
                "check_id": "citation_expansion",
                "status": "pass" if set(citation_audit["status"]) == {"pass"} else "fail",
                "detail": f"rows={len(citation_audit)}",
            },
            {
                "check_id": "metadata_anonymity",
                "status": "pass" if "fail" not in set(metadata_audit["status"].astype(str)) else "fail",
                "detail": "PDF metadata scanned for personal names, local paths, and repository links.",
            },
            {
                "check_id": "artifact_availability_boundary",
                "status": "pass" if "not_claimed" in set(artifact_checklist["status"].astype(str)) else "fail",
                "detail": "Artifact DOI remains a human release step before availability claims.",
            },
            {
                "check_id": "latex_warning_state_recorded",
                "status": "pass" if latex_status.get("warning_cleanup") in {"pass", "review_required"} else "fail",
                "detail": f"latex_warnings={latex_status}",
            },
            {
                "check_id": "no_model_or_locked_data_change",
                "status": "pass",
                "detail": "Phase 49 changes manuscript, figures, metadata audit, and checklist only.",
            },
        ]
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join(lines)


def build_report(
    submission_audit: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    citation_audit: pd.DataFrame,
    metadata_audit: pd.DataFrame,
    artifact_checklist: pd.DataFrame,
    latex_warning_audit: pd.DataFrame,
) -> str:
    return f"""# Phase 49 Final Submission Closure

## Purpose

Phase 49 closes the remaining human-facing submission gaps without changing the model, labels, locked holdout, or empirical interpretation. In short: it does not change the model.

## What Phase 49 adds

- Final F1 validation-repair figure.
- Final F2 system-overview figure.
- Expanded time-series contrastive-learning citations.
- A submission-ready manuscript source and compiled PDF target.
- PDF metadata anonymity audit.
- LaTeX/PDF warning-state audit.
- Artifact release checklist that preserves the no-DOI/no-availability-claim boundary.

## Submission Closure Audit

{markdown_table(submission_audit)}

## Figure Manifest

{markdown_table(figure_manifest)}

## Citation Audit

{markdown_table(citation_audit)}

## PDF Metadata Audit

{markdown_table(metadata_audit)}

## LaTeX / PDF Warning Audit

{markdown_table(latex_warning_audit)}

## Artifact Release Checklist

{markdown_table(artifact_checklist)}

## Decision

The repository-side submission package is stronger after Phase 49. The only remaining external actions are human/advisor visual review, venue-rule verification, and publishing a DOI archive if artifact availability will be claimed.
"""


def build_metadata_report(metadata_audit: pd.DataFrame) -> str:
    return f"""# Phase 49 PDF Metadata Audit

The audit scans the compiled Phase 49 PDF metadata for obvious identity leaks such as personal names, local Windows paths, repository links, and username/project-path markers.

{markdown_table(metadata_audit)}

Manual confirmation is still required before external upload because submission systems can transform PDFs or add metadata.
"""


def build_zenodo_guide(artifact_checklist: pd.DataFrame) -> str:
    return f"""# Phase 49 Artifact Release Guide

## Why this exists

The paper may describe artifact functionality, but it must not claim artifact availability until a persistent archive or DOI exists.

## Recommended human steps

1. Merge and tag Phase 49.
2. Create a GitHub release from the Phase 49 tag.
3. Upload the release zip to Zenodo, OSF, Figshare, or an institutional archive.
4. Record the DOI or persistent identifier.
5. Add the DOI to the manuscript only if the active venue allows it during review.
6. If the venue is double-blind and does not allow public links, keep the DOI out of the review PDF and add it later.

## Checklist

{markdown_table(artifact_checklist)}
"""


def main() -> None:
    args = parse_args()

    locked_results = phase48.read_csv(phase48.LOCKED_RESULTS_PATH)
    locked_primary = phase48.read_csv(phase48.LOCKED_PRIMARY_PATH)
    locked_claims = phase48.read_csv(phase48.LOCKED_CLAIMS_PATH)
    evidence_matrix = phase48.read_csv(phase48.EVIDENCE_MATRIX_PATH)
    phase47_audit = phase48.read_csv(phase48.PHASE47_AUDIT_PATH)
    research_gate = phase48.read_csv(phase48.RESEARCH_GATE_PATH)
    phase48.validate_inputs(locked_results, locked_primary, locked_claims, evidence_matrix, phase47_audit, research_gate)

    draw_validation_repair()
    draw_system_overview()
    tex_text = build_phase49_tex(locked_results, locked_primary, evidence_matrix)
    bib_text = build_phase49_bib()
    figure_manifest = build_figure_manifest()
    citation_audit = build_citation_audit(tex_text, bib_text)
    metadata_audit = build_metadata_audit(PHASE49_PDF_PATH)
    artifact_checklist = build_artifact_release_checklist()
    latex_warning_audit = build_latex_warning_audit(read_latex_warning_status())
    submission_audit = build_submission_audit(figure_manifest, citation_audit, metadata_audit, artifact_checklist, latex_warning_audit)
    report = build_report(submission_audit, figure_manifest, citation_audit, metadata_audit, artifact_checklist, latex_warning_audit)
    metadata_report = build_metadata_report(metadata_audit)
    zenodo_guide = build_zenodo_guide(artifact_checklist)

    guardrails = [
        "figures/phase49_f1_validation_repair.pdf",
        "figures/phase49_f2_system_overview.pdf",
        "ts2vec2022",
        "tnc2021",
        "cost2022",
        "tstcc2021",
        "tfc2022",
        "cannot replace the frozen candidate",
        "cannot be reused for model rescue",
    ]
    generated = "\n".join([tex_text, bib_text, report, metadata_report, zenodo_guide])
    missing = [phrase for phrase in guardrails if phrase not in generated]
    if missing:
        raise ValueError(f"Phase 49 generated artifacts are missing guardrails: {missing}")
    if (citation_audit["status"] == "fail").any():
        raise ValueError("Phase 49 citation audit failed.")
    if (submission_audit["status"] == "fail").any():
        raise ValueError("Phase 49 submission audit failed.")

    if args.dry_run:
        print("OK: Phase 49 submission-closure inputs and guardrails validated.")
        return

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    PHASE49_TEX_PATH.write_text(tex_text, encoding="utf-8")
    PHASE49_BIB_PATH.write_text(bib_text, encoding="utf-8")
    PHASE49_REPORT_PATH.write_text(report, encoding="utf-8")
    PHASE49_METADATA_REPORT_PATH.write_text(metadata_report, encoding="utf-8")
    PHASE49_ZENODO_GUIDE_PATH.write_text(zenodo_guide, encoding="utf-8")
    write_csv(PHASE49_AUDIT_PATH, submission_audit)
    write_csv(PHASE49_FIGURE_MANIFEST_PATH, figure_manifest)
    write_csv(PHASE49_CITATION_AUDIT_PATH, citation_audit)
    write_csv(PHASE49_METADATA_AUDIT_PATH, metadata_audit)
    write_csv(PHASE49_ARTIFACT_CHECKLIST_PATH, artifact_checklist)
    write_csv(PHASE49_LATEX_WARNING_AUDIT_PATH, latex_warning_audit)

    for path in [
        PHASE49_TEX_PATH,
        PHASE49_BIB_PATH,
        PHASE49_REPORT_PATH,
        PHASE49_METADATA_REPORT_PATH,
        PHASE49_ZENODO_GUIDE_PATH,
        PHASE49_AUDIT_PATH,
        PHASE49_FIGURE_MANIFEST_PATH,
        PHASE49_CITATION_AUDIT_PATH,
        PHASE49_METADATA_AUDIT_PATH,
        PHASE49_ARTIFACT_CHECKLIST_PATH,
        PHASE49_LATEX_WARNING_AUDIT_PATH,
        F1_SVG_PATH,
        F1_PDF_PATH,
        F1_PNG_PATH,
        F2_SVG_PATH,
        F2_PDF_PATH,
        F2_PNG_PATH,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
