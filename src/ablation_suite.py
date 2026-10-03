import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SAVE_DIR


STRUCTURAL_METRICS = [
    "hmm_reference_nmi",
    "hmm_reference_purity",
    "transition_diagonal_probability",
    "avg_regime_duration",
]

DOWNSTREAM_METRICS = [
    "IC",
    "Sharpe",
    "drawdown",
    "total_return",
]

HIGHER_IS_BETTER = {
    "hmm_reference_nmi": True,
    "hmm_reference_purity": True,
    "transition_diagonal_probability": True,
    "avg_regime_duration": None,
    "IC": True,
    "Sharpe": True,
    "drawdown": True,
    "total_return": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Phase 25 minimal ablation suite from completed research artifacts."
    )
    parser.add_argument("--models-dir", default=SAVE_DIR)
    return parser.parse_args()


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required artifact: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def method_row(frame: pd.DataFrame, method: str) -> pd.Series | None:
    if frame.empty or "method" not in frame.columns:
        return None
    match = frame[frame["method"] == method]
    if match.empty:
        return None
    return match.iloc[0]


def downstream_row(frame: pd.DataFrame, method: str) -> pd.Series | None:
    if frame.empty or "method" not in frame.columns:
        return None
    match = frame[frame["method"] == method]
    if match.empty:
        return None
    return match.iloc[0]


def add_metric_rows(
    rows: list[dict],
    *,
    ablation_family: str,
    comparison: str,
    evidence_type: str,
    candidate_label: str,
    reference_label: str,
    candidate: pd.Series | None,
    reference: pd.Series | None,
    metrics: list[str],
    source_artifact: str,
    interpretation: str,
) -> None:
    if candidate is None or reference is None:
        rows.append(
            {
                "ablation_family": ablation_family,
                "comparison": comparison,
                "evidence_type": evidence_type,
                "candidate": candidate_label,
                "reference": reference_label,
                "metric": "missing_artifact_row",
                "candidate_value": np.nan,
                "reference_value": np.nan,
                "delta": np.nan,
                "higher_is_better": np.nan,
                "candidate_wins": False,
                "source_artifact": source_artifact,
                "interpretation": "Missing method row; rerun upstream artifacts before using this ablation.",
            }
        )
        return

    for metric in metrics:
        if metric not in candidate.index or metric not in reference.index:
            continue
        candidate_value = float(candidate[metric])
        reference_value = float(reference[metric])
        delta = candidate_value - reference_value
        direction = HIGHER_IS_BETTER.get(metric)
        if direction is None:
            candidate_wins = np.nan
        elif direction:
            candidate_wins = bool(delta > 0)
        else:
            candidate_wins = bool(delta < 0)
        rows.append(
            {
                "ablation_family": ablation_family,
                "comparison": comparison,
                "evidence_type": evidence_type,
                "candidate": candidate_label,
                "reference": reference_label,
                "metric": metric,
                "candidate_value": candidate_value,
                "reference_value": reference_value,
                "delta": delta,
                "higher_is_better": direction,
                "candidate_wins": candidate_wins,
                "source_artifact": source_artifact,
                "interpretation": interpretation,
            }
        )


def build_ablation_rows(models_dir: Path) -> pd.DataFrame:
    guided_struct = read_csv(models_dir / "guided_encoder_comparison.csv")
    tf_struct = read_csv(models_dir / "time_frequency_encoder_comparison.csv")
    downstream = read_csv(models_dir / "walkforward_experiment_results.csv")

    rows: list[dict] = []

    structural = pd.concat([guided_struct, tf_struct], ignore_index=True)
    structural = structural.drop_duplicates(subset=["method"], keep="first")

    structural_pairs = [
        {
            "ablation_family": "objective_guidance",
            "comparison": "hmm_guided_hmm_vs_vanilla_contrastive_hmm",
            "candidate": "hmm_guided_hmm",
            "reference": "contrastive_hmm",
            "interpretation": "Tests whether HMM-guided weak supervision improves learned regime structure over vanilla contrastive embeddings when both use HMM assignment.",
        },
        {
            "ablation_family": "objective_guidance",
            "comparison": "hmm_guided_gmm_vs_vanilla_contrastive_gmm",
            "candidate": "hmm_guided_gmm",
            "reference": "contrastive",
            "interpretation": "Tests whether HMM-guided weak supervision improves learned regime structure over vanilla contrastive embeddings when both use GMM assignment.",
        },
        {
            "ablation_family": "assignment_layer",
            "comparison": "vanilla_hmm_assignment_vs_vanilla_gmm_assignment",
            "candidate": "contrastive_hmm",
            "reference": "contrastive",
            "interpretation": "Tests whether sequential HMM assignment improves the vanilla contrastive regime path over GMM assignment.",
        },
        {
            "ablation_family": "assignment_layer",
            "comparison": "guided_hmm_assignment_vs_guided_gmm_assignment",
            "candidate": "hmm_guided_hmm",
            "reference": "hmm_guided_gmm",
            "interpretation": "Tests whether sequential HMM assignment improves the HMM-guided embedding path over GMM assignment.",
        },
        {
            "ablation_family": "assignment_layer",
            "comparison": "time_frequency_hmm_assignment_vs_time_frequency_gmm_assignment",
            "candidate": "tf_hmm_guided_hmm",
            "reference": "tf_hmm_guided_gmm",
            "interpretation": "Tests whether HMM assignment remains preferable on the time-frequency guided prototype.",
        },
        {
            "ablation_family": "augmentation_view",
            "comparison": "time_frequency_guided_hmm_vs_time_only_guided_hmm",
            "candidate": "tf_hmm_guided_hmm",
            "reference": "hmm_guided_hmm",
            "interpretation": "Tests whether the current 3-epoch time-frequency view improves the guided-HMM structural result over the full time-only guided run.",
        },
        {
            "ablation_family": "augmentation_view",
            "comparison": "time_frequency_guided_gmm_vs_time_only_guided_gmm",
            "candidate": "tf_hmm_guided_gmm",
            "reference": "hmm_guided_gmm",
            "interpretation": "Tests whether the current 3-epoch time-frequency view improves the guided-GMM structural result over the full time-only guided run.",
        },
    ]

    for pair in structural_pairs:
        add_metric_rows(
            rows,
            ablation_family=pair["ablation_family"],
            comparison=pair["comparison"],
            evidence_type="structural",
            candidate_label=pair["candidate"],
            reference_label=pair["reference"],
            candidate=method_row(structural, pair["candidate"]),
            reference=method_row(structural, pair["reference"]),
            metrics=STRUCTURAL_METRICS,
            source_artifact="guided_encoder_comparison.csv + time_frequency_encoder_comparison.csv",
            interpretation=pair["interpretation"],
        )

    downstream_pairs = [
        {
            "ablation_family": "objective_guidance",
            "comparison": "guided_hmm_alpha_vs_vanilla_contrastive_hmm_alpha",
            "candidate": "regime_lgbm_hmm_guided_hmm",
            "reference": "regime_lgbm_contrastive_hmm",
            "interpretation": "Tests whether the HMM-guided learned-regime alpha model improves downstream performance over the vanilla contrastive-HMM alpha path.",
        },
        {
            "ablation_family": "objective_guidance",
            "comparison": "guided_gmm_alpha_vs_vanilla_contrastive_gmm_alpha",
            "candidate": "regime_lgbm_hmm_guided_gmm",
            "reference": "regime_lgbm_contrastive",
            "interpretation": "Tests whether HMM-guided embeddings help downstream alpha when assignment remains GMM-based.",
        },
        {
            "ablation_family": "assignment_layer",
            "comparison": "vanilla_hmm_alpha_vs_vanilla_gmm_alpha",
            "candidate": "regime_lgbm_contrastive_hmm",
            "reference": "regime_lgbm_contrastive",
            "interpretation": "Tests whether HMM assignment improves downstream alpha for vanilla contrastive embeddings.",
        },
        {
            "ablation_family": "assignment_layer",
            "comparison": "guided_hmm_alpha_vs_guided_gmm_alpha",
            "candidate": "regime_lgbm_hmm_guided_hmm",
            "reference": "regime_lgbm_hmm_guided_gmm",
            "interpretation": "Tests whether HMM assignment improves downstream alpha for HMM-guided embeddings.",
        },
        {
            "ablation_family": "classical_reference",
            "comparison": "guided_hmm_alpha_vs_raw_feature_hmm_alpha",
            "candidate": "regime_lgbm_hmm_guided_hmm",
            "reference": "regime_lgbm_hmm",
            "interpretation": "Tests the main paper comparison: guided learned regimes versus the classical raw-feature HMM alpha baseline.",
        },
    ]

    for pair in downstream_pairs:
        add_metric_rows(
            rows,
            ablation_family=pair["ablation_family"],
            comparison=pair["comparison"],
            evidence_type="downstream_alpha",
            candidate_label=pair["candidate"],
            reference_label=pair["reference"],
            candidate=downstream_row(downstream, pair["candidate"]),
            reference=downstream_row(downstream, pair["reference"]),
            metrics=DOWNSTREAM_METRICS,
            source_artifact="walkforward_experiment_results.csv",
            interpretation=pair["interpretation"],
        )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    valid = results[results["metric"] != "missing_artifact_row"].copy()
    valid["candidate_win_numeric"] = valid["candidate_wins"].map({True: 1.0, False: 0.0})

    summary = (
        valid.groupby(["ablation_family", "comparison", "evidence_type", "candidate", "reference"], dropna=False)
        .agg(
            metrics_tested=("metric", "nunique"),
            metrics_with_direction=("candidate_win_numeric", "count"),
            candidate_wins=("candidate_win_numeric", "sum"),
            mean_delta=("delta", "mean"),
            source_artifact=("source_artifact", "first"),
            interpretation=("interpretation", "first"),
        )
        .reset_index()
    )
    summary["candidate_win_rate"] = np.where(
        summary["metrics_with_direction"] > 0,
        summary["candidate_wins"] / summary["metrics_with_direction"],
        np.nan,
    )

    pivot_metrics = valid.pivot_table(
        index=["ablation_family", "comparison"],
        columns="metric",
        values="delta",
        aggfunc="first",
    ).reset_index()
    summary = summary.merge(pivot_metrics, on=["ablation_family", "comparison"], how="left")

    def decision(row: pd.Series) -> str:
        if row["ablation_family"] == "augmentation_view":
            if row.get("hmm_reference_nmi", 0.0) > 0 and row.get("hmm_reference_purity", 0.0) > 0:
                return "promising_structural"
            return "do_not_expand_yet"
        if row["ablation_family"] == "classical_reference":
            if row.get("IC", 0.0) > 0 and row.get("Sharpe", 0.0) > 0:
                return "directional_support_statistical_refresh_needed"
            return "not_supported"
        if row["candidate_win_rate"] >= 0.75:
            return "supported"
        if row["candidate_win_rate"] >= 0.5:
            return "mixed"
        return "not_supported"

    summary["phase25_decision"] = summary.apply(decision, axis=1)
    return summary.sort_values(["ablation_family", "comparison"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = summary.copy()
    plot_data["label"] = plot_data["comparison"].str.replace("_", "\n")
    values = plot_data[["candidate_win_rate"]].to_numpy(dtype=float)

    fig_height = max(5, 0.55 * len(plot_data))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks([0])
    ax.set_xticklabels(["Win rate"])
    ax.set_yticks(np.arange(len(plot_data)))
    ax.set_yticklabels(plot_data["label"], fontsize=8)
    ax.set_title("Phase 25 Minimal Ablation Suite")

    for row, value in enumerate(plot_data["candidate_win_rate"]):
        if pd.isna(value):
            text = "n/a"
        else:
            text = f"{value:.2f}"
        ax.text(0, row, text, ha="center", va="center", color="black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Candidate metric win rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    results = build_ablation_rows(models_dir)
    summary = summarize(results)

    results_path = models_dir / "ablation_results.csv"
    summary_path = models_dir / "ablation_summary.csv"
    heatmap_path = models_dir / "ablation_heatmap.png"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, heatmap_path)

    cols = [
        "ablation_family",
        "comparison",
        "evidence_type",
        "candidate",
        "reference",
        "candidate_win_rate",
        "phase25_decision",
    ]
    print("\nPhase 25 ablation summary:")
    print(summary[cols].to_string(index=False))
    print(f"\nSaved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {heatmap_path}")


if __name__ == "__main__":
    main()
