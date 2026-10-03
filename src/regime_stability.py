import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import N_REGIMES, SAVE_DIR, SYMBOLS


PRIMARY_TARGET = "tb_label_8h"
POST_COLS = [f"post_{i}" for i in range(N_REGIMES)]
METHOD_ORDER = ["contrastive", "contrastive_hmm", "hmm", "kmeans", "vol_bucket"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify regime persistence, transition churn, and stable-vs-transition IC."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
        help="Symbols to include in the stability analysis.",
    )
    parser.add_argument(
        "--transition-bars",
        type=int,
        default=2,
        help="Rows after a regime switch counted as transition rows.",
    )
    return parser.parse_args()


def normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size <= 1:
        return 0.0
    probs = values / values.sum()
    return float(-(probs * np.log(probs)).sum() / np.log(len(probs)))


def add_run_state(assignments: pd.DataFrame, transition_bars: int) -> pd.DataFrame:
    df = assignments.sort_values(["method", "symbol", "feat_idx"]).copy()
    keys = ["method", "symbol"]
    df["prev_regime"] = df.groupby(keys)["regime"].shift()
    df["is_switch"] = df["prev_regime"].notna() & (df["regime"] != df["prev_regime"])
    df["run_id"] = df.groupby(keys)["is_switch"].cumsum().astype(int)
    df["run_position"] = df.groupby(keys + ["run_id"]).cumcount() + 1
    df["run_length"] = df.groupby(keys + ["run_id"])["regime"].transform("size")
    df["period_type"] = np.where(
        (df["run_id"] > 0) & (df["run_position"] <= transition_bars),
        "transition",
        "stable",
    )
    return df


def transition_counts(frame: pd.DataFrame) -> np.ndarray:
    counts = np.zeros((N_REGIMES, N_REGIMES), dtype=float)
    for _, symbol_frame in frame.groupby("symbol", sort=False):
        ordered = symbol_frame.sort_values("feat_idx")
        prev_regime = ordered["regime"].shift()
        valid = prev_regime.notna()
        for prev, curr in zip(prev_regime[valid].astype(int), ordered.loc[valid, "regime"].astype(int)):
            if 0 <= prev < N_REGIMES and 0 <= curr < N_REGIMES:
                counts[prev, curr] += 1
    return counts


def weighted_transition_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return np.nan
    row_sums = counts.sum(axis=1)
    entropies = []
    weights = []
    for row, row_total in zip(counts, row_sums):
        if row_total == 0:
            continue
        probs = row / row_total
        probs = probs[probs > 0]
        entropy = -(probs * np.log(probs)).sum() / np.log(N_REGIMES)
        entropies.append(entropy)
        weights.append(row_total)
    if not entropies:
        return np.nan
    return float(np.average(entropies, weights=weights))


def posterior_metrics(frame: pd.DataFrame) -> dict[str, float]:
    post_cols = [col for col in POST_COLS if col in frame.columns]
    if not post_cols:
        return {
            "mean_confidence": np.nan,
            "mean_posterior_entropy": np.nan,
            "low_confidence_pct": np.nan,
        }

    probs = frame[post_cols].astype(float).to_numpy()
    row_sums = probs.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    if not valid.any():
        return {
            "mean_confidence": np.nan,
            "mean_posterior_entropy": np.nan,
            "low_confidence_pct": np.nan,
        }

    probs = probs[valid] / row_sums[valid]
    confidence = probs.max(axis=1)
    safe_probs = np.clip(probs, 1e-12, 1.0)
    row_entropy = -(probs * np.log(safe_probs)).sum(axis=1)
    row_entropy = row_entropy / np.log(probs.shape[1])

    return {
        "mean_confidence": float(confidence.mean()),
        "mean_posterior_entropy": float(row_entropy.mean()),
        "low_confidence_pct": float((confidence < 0.60).mean()),
    }


def stability_metrics(frame: pd.DataFrame, method: str, symbol_scope: str) -> dict[str, float | int | str]:
    frame = frame.sort_values(["symbol", "feat_idx"]).copy()
    runs = (
        frame.groupby(["symbol", "run_id"], as_index=False)
        .agg(regime=("regime", "first"), run_length=("regime", "size"))
    )
    counts = transition_counts(frame)
    total_transitions = counts.sum()
    diagonal_prob = np.trace(counts) / total_transitions if total_transitions else np.nan
    regime_counts = frame["regime"].value_counts().reindex(range(N_REGIMES), fill_value=0)
    posterior = posterior_metrics(frame)

    return {
        "method": method,
        "symbol_scope": symbol_scope,
        "n_rows": int(len(frame)),
        "n_regimes": int(frame["regime"].nunique()),
        "switch_count": int(frame["is_switch"].sum()),
        "switches_per_1000_bars": float(frame["is_switch"].sum() / max(total_transitions, 1) * 1000),
        "avg_regime_duration": float(runs["run_length"].mean()),
        "median_regime_duration": float(runs["run_length"].median()),
        "p90_regime_duration": float(runs["run_length"].quantile(0.90)),
        "transition_diagonal_probability": float(diagonal_prob),
        "transition_entropy": weighted_transition_entropy(counts),
        "regime_balance_entropy": normalized_entropy(regime_counts.to_numpy()),
        **posterior,
    }


def load_predictions(models_dir: Path) -> pd.DataFrame:
    for name in ["alpha_oos_predictions.csv", "oos_predictions.csv"]:
        path = models_dir / name
        if path.exists():
            preds = pd.read_csv(path)
            if "target" in preds.columns:
                preds = preds[preds["target"] == PRIMARY_TARGET].copy()
            return preds
    return pd.DataFrame()


def add_ic_diagnostics(summary: pd.DataFrame, assignments: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    ic_cols = [
        "stable_oos_rows",
        "transition_oos_rows",
        "stable_ic",
        "transition_ic",
        "ic_stable_minus_transition",
    ]
    for col in ic_cols:
        summary[col] = np.nan

    if predictions.empty:
        return summary

    for idx, row in summary.iterrows():
        method = row["method"]
        symbol_scope = row["symbol_scope"]
        assignment_slice = assignments[assignments["method"] == method]
        pred_slice = predictions[predictions["regime_method"] == method]

        if symbol_scope != "ALL":
            assignment_slice = assignment_slice[assignment_slice["symbol"] == symbol_scope]
            pred_slice = pred_slice[pred_slice["symbol"] == symbol_scope]

        if assignment_slice.empty or pred_slice.empty:
            continue

        merged = pred_slice.merge(
            assignment_slice[
                ["symbol", "feat_idx", "period_type", "run_position", "run_length"]
            ],
            on=["symbol", "feat_idx"],
            how="inner",
        )
        if merged.empty:
            continue

        stable = merged[merged["period_type"] == "stable"]
        transition = merged[merged["period_type"] == "transition"]
        stable_ic = stable["score"].corr(stable["target_return"]) if len(stable) >= 30 else np.nan
        transition_ic = (
            transition["score"].corr(transition["target_return"])
            if len(transition) >= 30
            else np.nan
        )

        summary.loc[idx, "stable_oos_rows"] = int(len(stable))
        summary.loc[idx, "transition_oos_rows"] = int(len(transition))
        summary.loc[idx, "stable_ic"] = stable_ic
        summary.loc[idx, "transition_ic"] = transition_ic
        summary.loc[idx, "ic_stable_minus_transition"] = stable_ic - transition_ic

    return summary


def plot_stability(summary: pd.DataFrame, output_path: Path) -> None:
    aggregate = summary[summary["symbol_scope"] == "ALL"].copy()
    aggregate["method"] = pd.Categorical(aggregate["method"], METHOD_ORDER, ordered=True)
    aggregate = aggregate.sort_values("method")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    methods = aggregate["method"].astype(str).to_list()
    colors = ["#4C78A8", "#00A6A6", "#59A14F", "#F28E2B", "#B07AA1"]

    axes[0, 0].bar(methods, aggregate["avg_regime_duration"], color=colors)
    axes[0, 0].set_title("Average Regime Duration")
    axes[0, 0].set_ylabel("bars")

    axes[0, 1].bar(methods, aggregate["switches_per_1000_bars"], color=colors)
    axes[0, 1].set_title("Switch Rate")
    axes[0, 1].set_ylabel("switches / 1,000 bars")

    axes[1, 0].bar(methods, aggregate["mean_posterior_entropy"], color=colors)
    axes[1, 0].set_title("Mean Posterior Entropy")
    axes[1, 0].set_ylabel("0=confident, 1=diffuse")

    x = np.arange(len(methods))
    width = 0.36
    axes[1, 1].bar(
        x - width / 2,
        aggregate["stable_ic"],
        width=width,
        label="stable",
        color="#4C78A8",
    )
    axes[1, 1].bar(
        x + width / 2,
        aggregate["transition_ic"],
        width=width,
        label="transition",
        color="#E15759",
    )
    axes[1, 1].axhline(0, color="#333333", linewidth=0.8)
    axes[1, 1].set_title("OOS IC: Stable vs Transition Rows")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Regime Alpha Benchmark - Regime Stability Diagnostics", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    models_dir = Path(SAVE_DIR)
    assignments_path = models_dir / "regime_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(
            "Missing models/regime_assignments.csv. Run visualize_regimes.py and baselines.py first."
        )

    assignments = pd.read_csv(assignments_path)
    assignments = assignments[assignments["symbol"].isin(args.symbols)].copy()
    assignments["feat_idx"] = assignments["feat_idx"].astype(int)
    assignments["regime"] = assignments["regime"].astype(int)
    assignments = add_run_state(assignments, args.transition_bars)

    rows = []
    for method in METHOD_ORDER:
        method_frame = assignments[assignments["method"] == method]
        if method_frame.empty:
            continue
        rows.append(stability_metrics(method_frame, method, "ALL"))
        for symbol in args.symbols:
            symbol_frame = method_frame[method_frame["symbol"] == symbol]
            if not symbol_frame.empty:
                rows.append(stability_metrics(symbol_frame, method, symbol))

    summary = pd.DataFrame(rows)
    predictions = load_predictions(models_dir)
    summary = add_ic_diagnostics(summary, assignments, predictions)

    method_order = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary = summary.assign(_method_order=method_order)
    summary = summary.sort_values(["_method_order", "symbol_scope"]).drop(columns="_method_order")

    output_csv = models_dir / "regime_stability_summary.csv"
    output_png = models_dir / "regime_stability.png"
    summary.to_csv(output_csv, index=False)
    plot_stability(summary, output_png)

    aggregate = summary[summary["symbol_scope"] == "ALL"]
    print("\nRegime stability summary:")
    print(
        aggregate[
            [
                "method",
                "n_rows",
                "switches_per_1000_bars",
                "avg_regime_duration",
                "transition_diagonal_probability",
                "mean_confidence",
                "stable_ic",
                "transition_ic",
            ]
        ].to_string(index=False)
    )
    if predictions.empty:
        print("\nWARNING: alpha_oos_predictions.csv not found; IC split columns were left empty.")
    print(f"\nOK: saved {output_csv} and {output_png}")


if __name__ == "__main__":
    main()
