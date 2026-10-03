import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SAVE_DIR
from statistical_tests import (
    adjust_benjamini_hochberg,
    adjust_holm,
    cohen_dz,
    effect_size_label,
    paired_t_test,
    sign_test_p_value,
)


METRICS = [
    ("IC", True),
    ("Sharpe", True),
    ("total_return", True),
    ("drawdown", True),
    ("turnover", False),
]

PAPER_CLAIMS = {
    "guided_hmm_alpha_vs_guided_gmm_alpha": (
        "Sequential HMM assignment improves the HMM-guided learned-regime alpha path."
    ),
    "vanilla_hmm_alpha_vs_vanilla_gmm_alpha": (
        "Sequential HMM assignment is also useful on the original vanilla contrastive embedding path."
    ),
    "guided_hmm_alpha_vs_raw_feature_hmm_alpha": (
        "HMM-guided learned regimes outperform the raw-feature HMM baseline on the primary alpha benchmark."
    ),
    "guided_hmm_alpha_vs_vanilla_contrastive_hmm_alpha": (
        "HMM-guided supervision improves downstream alpha versus vanilla contrastive-HMM regimes."
    ),
    "guided_gmm_alpha_vs_vanilla_contrastive_gmm_alpha": (
        "HMM-guided supervision improves downstream alpha when the assignment layer remains GMM-based."
    ),
}


def ablation_text(row: object) -> str:
    for name in ("description", "interpretation"):
        if hasattr(row, name):
            value = getattr(row, name)
            if pd.notna(value):
                return str(value)
    return ""


def ablation_win_rate(row: object) -> float:
    for name in ("metric_win_rate", "candidate_win_rate"):
        if hasattr(row, name):
            value = getattr(row, name)
            if pd.notna(value):
                return float(value)
    return np.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 26 paper-facing statistical claim tests from the Phase 25 ablation map."
    )
    parser.add_argument("--models-dir", default=SAVE_DIR)
    return parser.parse_args()


def load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise SystemExit(f"Missing required artifact: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def direct_fold_tests(folds: pd.DataFrame, ablations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    downstream = ablations[ablations["evidence_type"] == "downstream_alpha"].copy()

    for ablation in downstream.itertuples(index=False):
        candidate = str(ablation.candidate)
        reference = str(ablation.reference)
        candidate_folds = folds[folds["method"] == candidate]
        reference_folds = folds[folds["method"] == reference]
        if candidate_folds.empty or reference_folds.empty:
            rows.append(
                {
                    "ablation_family": ablation.ablation_family,
                    "comparison": ablation.comparison,
                    "candidate": candidate,
                    "reference": reference,
                    "metric": "all",
                    "paper_claim": PAPER_CLAIMS.get(ablation.comparison, ablation_text(ablation)),
                    "phase25_decision": ablation.phase25_decision,
                    "metric_win_rate": ablation_win_rate(ablation),
                    "statistical_status": "missing_fold_metrics",
                    "allowed_paper_language": "Not testable from current fold metrics.",
                }
            )
            continue

        paired = candidate_folds.merge(reference_folds, on="fold", suffixes=("", "_reference"))
        for metric, higher_is_better in METRICS:
            diffs = paired[metric].to_numpy(dtype=float) - paired[f"{metric}_reference"].to_numpy(dtype=float)
            diffs = diffs[np.isfinite(diffs)]
            stat, p_value = paired_t_test(diffs)
            wins, non_ties, sign_p = sign_test_p_value(diffs, higher_is_better)
            mean_difference = float(np.mean(diffs)) if len(diffs) else np.nan
            row_direction = "candidate_better" if (
                mean_difference > 0 if higher_is_better else mean_difference < 0
            ) else "reference_better"
            rows.append(
                {
                    "ablation_family": ablation.ablation_family,
                    "comparison": ablation.comparison,
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "mean_difference": mean_difference,
                    "direction": row_direction,
                    "paired_t_stat": stat,
                    "paired_t_p_value": p_value,
                    "cohen_dz": cohen_dz(diffs),
                    "effect_size": effect_size_label(cohen_dz(diffs)),
                    "sign_test_wins": wins,
                    "sign_test_non_ties": non_ties,
                    "sign_test_p_value": sign_p,
                    "n_paired_folds": int(len(diffs)),
                    "paper_claim": PAPER_CLAIMS.get(ablation.comparison, ablation_text(ablation)),
                    "phase25_decision": ablation.phase25_decision,
                    "metric_win_rate": ablation_win_rate(ablation),
                    "phase25_mean_delta": getattr(ablation, f"mean_delta_{metric}", np.nan),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    testable = out["metric"] != "all"
    out["primary_p_value"] = out["paired_t_p_value"]
    out["bh_q_all_paper_tests"] = np.nan
    out["holm_p_all_paper_tests"] = np.nan
    out.loc[testable, "bh_q_all_paper_tests"] = adjust_benjamini_hochberg(out.loc[testable, "primary_p_value"])
    out.loc[testable, "holm_p_all_paper_tests"] = adjust_holm(out.loc[testable, "primary_p_value"])
    out["bh_q_by_metric"] = np.nan
    out["holm_p_by_metric"] = np.nan
    for _, idx in out[testable].groupby("metric").groups.items():
        out.loc[idx, "bh_q_by_metric"] = adjust_benjamini_hochberg(out.loc[idx, "primary_p_value"])
        out.loc[idx, "holm_p_by_metric"] = adjust_holm(out.loc[idx, "primary_p_value"])

    out["statistical_status"] = out.apply(classify_statistical_row, axis=1)
    out["allowed_paper_language"] = out.apply(language_for_row, axis=1)
    return out.sort_values(["ablation_family", "comparison", "metric"]).reset_index(drop=True)


def classify_statistical_row(row: pd.Series) -> str:
    if row.get("statistical_status") == "missing_fold_metrics":
        return "missing_fold_metrics"
    p_value = row.get("primary_p_value", np.nan)
    if pd.isna(p_value):
        return "not_tested"
    candidate_better = row.get("direction") == "candidate_better"
    if candidate_better and row.get("holm_p_all_paper_tests", 1.0) < 0.05:
        return "statistically_supported_holm_all"
    if candidate_better and row.get("bh_q_all_paper_tests", 1.0) < 0.05:
        return "statistically_supported_bh_all"
    if candidate_better and row.get("bh_q_by_metric", 1.0) < 0.05:
        return "metric_family_supported"
    if candidate_better and p_value < 0.05:
        return "raw_suggestive"
    if candidate_better:
        return "directional_only"
    if p_value < 0.05:
        return "significant_against_candidate"
    return "not_significant"


def language_for_row(row: pd.Series) -> str:
    metric = row.get("metric", "metric")
    status = row.get("statistical_status", "not_tested")
    comparison = row.get("comparison", "")
    if status.startswith("statistically_supported"):
        return f"For {metric}, {comparison} is statistically supported after paper-level correction."
    if status == "metric_family_supported":
        return f"For {metric}, {comparison} survives correction within the metric family only."
    if status == "raw_suggestive":
        return f"For {metric}, {comparison} is suggestive before correction; treat as exploratory."
    if status == "directional_only":
        return f"For {metric}, {comparison} has the right sign but is not statistically significant."
    if status == "significant_against_candidate":
        return f"For {metric}, the evidence significantly favors the reference over the candidate."
    if status == "missing_fold_metrics":
        return "No fold-level statistical test was available for this ablation row."
    return f"For {metric}, no statistically reliable difference is detected."


def structural_claim_rows(ablations: pd.DataFrame) -> pd.DataFrame:
    structural = ablations[ablations["evidence_type"] == "structural"].copy()
    if structural.empty:
        return pd.DataFrame()
    rows = []
    for row in structural.itertuples(index=False):
        decision = str(row.phase25_decision)
        if decision == "supported":
            status = "mechanism_supported_no_fold_p_value"
            language = (
                f"{row.comparison} is supported as a structural mechanism result, "
                "but it is not a fold-level alpha significance claim."
            )
        elif decision == "do_not_expand_yet":
            status = "do_not_expand_yet"
            language = f"{row.comparison} is not strong enough to justify downstream expansion yet."
        elif decision == "not_supported":
            status = "not_supported"
            language = f"{row.comparison} is not supported by the current structural ablation metrics."
        else:
            status = "mixed_structural_evidence"
            language = f"{row.comparison} has mixed structural evidence and should be described cautiously."
        rows.append(
            {
                "ablation_family": row.ablation_family,
                "comparison": row.comparison,
                "evidence_type": row.evidence_type,
                "candidate": row.candidate,
                "reference": row.reference,
                "metric_win_rate": ablation_win_rate(row),
                "phase25_decision": row.phase25_decision,
                "paper_claim": ablation_text(row),
                "paper_status": status,
                "allowed_paper_language": language,
            }
        )
    return pd.DataFrame(rows)


def build_summary(ablations: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    downstream_rows = []
    if not tests.empty:
        focus_metrics = {"IC", "Sharpe", "total_return", "drawdown"}
        focused = tests[tests["metric"].isin(focus_metrics)].copy()
        for comparison, group in focused.groupby("comparison", sort=False):
            primary = group[group["metric"] == "IC"]
            primary_row = primary.iloc[0] if not primary.empty else group.iloc[0]
            candidate_better = int((group["direction"] == "candidate_better").sum())
            corrected = group[group["statistical_status"].astype(str).str.startswith("statistically_supported")]
            metric_supported = group[group["statistical_status"] == "metric_family_supported"]
            raw_suggestive = group[group["statistical_status"] == "raw_suggestive"]
            against = group[group["statistical_status"] == "significant_against_candidate"]
            if not corrected.empty:
                paper_status = "statistically_supported"
            elif not metric_supported.empty:
                paper_status = "metric_family_supported"
            elif not raw_suggestive.empty:
                paper_status = "raw_suggestive"
            elif not against.empty:
                paper_status = "evidence_against_candidate"
            elif candidate_better == len(group):
                paper_status = "directionally_supported"
            elif candidate_better > 0:
                paper_status = "mixed_directional"
            else:
                paper_status = "not_supported"

            downstream_rows.append(
                {
                    "ablation_family": primary_row["ablation_family"],
                    "comparison": comparison,
                    "evidence_type": "downstream_alpha",
                    "candidate": primary_row["candidate"],
                    "reference": primary_row["reference"],
                    "metric_win_rate": primary_row["metric_win_rate"],
                    "phase25_decision": primary_row["phase25_decision"],
                    "paper_claim": primary_row["paper_claim"],
                    "paper_status": paper_status,
                    "ic_mean_difference": primary_row["mean_difference"],
                    "ic_p_value": primary_row["primary_p_value"],
                    "ic_bh_q_all_paper_tests": primary_row["bh_q_all_paper_tests"],
                    "ic_holm_p_all_paper_tests": primary_row["holm_p_all_paper_tests"],
                    "candidate_better_metrics": candidate_better,
                    "tested_metrics": int(len(group)),
                    "allowed_paper_language": language_for_summary(paper_status, comparison),
                }
            )

    structural = structural_claim_rows(ablations)
    summary = pd.concat([pd.DataFrame(downstream_rows), structural], ignore_index=True, sort=False)
    if summary.empty:
        return summary
    return summary.sort_values(["ablation_family", "comparison"]).reset_index(drop=True)


def language_for_summary(status: str, comparison: str) -> str:
    if status == "statistically_supported":
        return f"{comparison} is statistically supported after paper-level correction."
    if status == "metric_family_supported":
        return f"{comparison} is supported within at least one metric family, but not across all paper tests."
    if status == "raw_suggestive":
        return f"{comparison} is suggestive before correction and should be framed as exploratory."
    if status == "directionally_supported":
        return f"{comparison} improves all focused point-estimate metrics but lacks statistical significance."
    if status == "mixed_directional":
        return f"{comparison} has mixed point-estimate support and should not be stated as a clear win."
    if status == "evidence_against_candidate":
        return f"{comparison} has statistically significant evidence against the candidate on at least one metric."
    return f"{comparison} is not supported by the current paper claim tests."


def plot_claim_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    plot_df = summary.copy()
    plot_df["metric_win_rate"] = pd.to_numeric(plot_df["metric_win_rate"], errors="coerce").fillna(0.0)
    plot_df = plot_df.sort_values("metric_win_rate", ascending=True)
    colors = {
        "statistically_supported": "#1b9e77",
        "metric_family_supported": "#66a61e",
        "directionally_supported": "#7570b3",
        "mechanism_supported_no_fold_p_value": "#1f78b4",
        "raw_suggestive": "#e6ab02",
        "mixed_directional": "#a6761d",
        "mixed_structural_evidence": "#a6761d",
        "do_not_expand_yet": "#d95f02",
        "not_supported": "#b2182b",
        "evidence_against_candidate": "#b2182b",
    }
    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(plot_df))))
    ax.barh(
        plot_df["comparison"],
        plot_df["metric_win_rate"],
        color=[colors.get(status, "#666666") for status in plot_df["paper_status"]],
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Phase 25 metric win rate")
    ax.set_title("Phase 26 Paper Claim Test Summary")
    ax.grid(axis="x", alpha=0.25)
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            min(float(row.metric_win_rate) + 0.02, 1.0),
            idx,
            str(row.paper_status),
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    ablations = load_csv(models_dir / "ablation_summary.csv")
    folds = load_csv(models_dir / "statistical_fold_metrics.csv")

    tests = direct_fold_tests(folds, ablations)
    summary = build_summary(ablations, tests)

    tests_path = models_dir / "paper_claim_tests.csv"
    summary_path = models_dir / "paper_statistical_summary.csv"
    plot_path = models_dir / "paper_claim_tests.png"

    tests.to_csv(tests_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_claim_summary(summary, plot_path)

    print("\nPhase 26 paper statistical summary:")
    cols = [
        "comparison",
        "evidence_type",
        "paper_status",
        "metric_win_rate",
        "ic_mean_difference",
        "ic_p_value",
        "candidate_better_metrics",
        "tested_metrics",
    ]
    display_cols = [col for col in cols if col in summary.columns]
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved: {tests_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
