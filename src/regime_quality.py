import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import N_REGIMES, SAVE_DIR, SYMBOLS


METHOD_ORDER = ["contrastive", "contrastive_hmm", "hmm", "kmeans", "vol_bucket"]
POST_COLS = [f"post_{idx}" for idx in range(N_REGIMES)]
KEY_COLS = ["symbol", "feat_idx"]
REFERENCE_METHOD = "hmm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Phase 16 regime quality and agreement metrics."
    )
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument(
        "--assignments",
        default=os.path.join(SAVE_DIR, "regime_assignments.csv"),
        help="Canonical regime assignment file from baselines.py.",
    )
    return parser.parse_args()


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[np.isfinite(counts) & (counts > 0)]
    if counts.size <= 1:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum() / np.log(len(counts)))


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
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / np.log(probs.shape[1])
    return {
        "mean_confidence": float(confidence.mean()),
        "mean_posterior_entropy": float(entropy.mean()),
        "low_confidence_pct": float((confidence < 0.60).mean()),
    }


def transition_matrix(frame: pd.DataFrame) -> np.ndarray:
    counts = np.zeros((N_REGIMES, N_REGIMES), dtype=float)
    for _, group in frame.sort_values(["symbol", "feat_idx"]).groupby("symbol", sort=False):
        prev = group["regime"].shift()
        valid = prev.notna()
        for source, target in zip(prev[valid].astype(int), group.loc[valid, "regime"].astype(int)):
            if 0 <= source < N_REGIMES and 0 <= target < N_REGIMES:
                counts[source, target] += 1.0
    return counts


def transition_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return np.nan
    row_totals = counts.sum(axis=1)
    values = []
    weights = []
    for row, row_total in zip(counts, row_totals):
        if row_total <= 0:
            continue
        probs = row / row_total
        probs = probs[probs > 0]
        values.append(float(-(probs * np.log(probs)).sum() / np.log(N_REGIMES)))
        weights.append(row_total)
    if not values:
        return np.nan
    return float(np.average(values, weights=weights))


def run_lengths(frame: pd.DataFrame) -> pd.Series:
    pieces = []
    for _, group in frame.sort_values(["symbol", "feat_idx"]).groupby("symbol", sort=False):
        switches = group["regime"].ne(group["regime"].shift()).fillna(True)
        run_id = switches.cumsum()
        pieces.append(group.groupby(run_id)["regime"].size())
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces, ignore_index=True)


def reference_alignment(method_frame: pd.DataFrame, reference_frame: pd.DataFrame) -> dict[str, float]:
    if method_frame["method"].iloc[0] == REFERENCE_METHOD:
        return {
            "hmm_reference_nmi": 1.0,
            "hmm_reference_ari": 1.0,
            "hmm_reference_purity": 1.0,
            "hmm_reference_conditional_entropy": 0.0,
        }

    merged = method_frame[KEY_COLS + ["regime"]].merge(
        reference_frame[KEY_COLS + ["regime"]],
        on=KEY_COLS,
        suffixes=("", "_reference"),
        how="inner",
    )
    if merged.empty:
        return {
            "hmm_reference_nmi": np.nan,
            "hmm_reference_ari": np.nan,
            "hmm_reference_purity": np.nan,
            "hmm_reference_conditional_entropy": np.nan,
        }

    labels = merged["regime"].astype(int)
    reference = merged["regime_reference"].astype(int)
    contingency = pd.crosstab(labels, reference)
    purity = contingency.max(axis=1).sum() / contingency.to_numpy().sum()
    conditional_entropies = []
    weights = []
    for _, row in contingency.iterrows():
        total = row.sum()
        if total <= 0:
            continue
        conditional_entropies.append(normalized_entropy(row.to_numpy()))
        weights.append(total)

    return {
        "hmm_reference_nmi": float(normalized_mutual_info_score(reference, labels)),
        "hmm_reference_ari": float(adjusted_rand_score(reference, labels)),
        "hmm_reference_purity": float(purity),
        "hmm_reference_conditional_entropy": float(np.average(conditional_entropies, weights=weights))
        if conditional_entropies
        else np.nan,
    }


def summarize_method(frame: pd.DataFrame, reference_frame: pd.DataFrame, symbol_scope: str) -> dict:
    frame = frame.sort_values(["symbol", "feat_idx"]).copy()
    method = str(frame["method"].iloc[0])
    counts = frame["regime"].value_counts().reindex(range(N_REGIMES), fill_value=0)
    transitions = transition_matrix(frame)
    transition_total = transitions.sum()
    runs = run_lengths(frame)
    switch_count = int((transitions.sum() - np.trace(transitions)) if transition_total else 0)
    posterior = posterior_metrics(frame)
    reference = reference_alignment(frame, reference_frame)

    return {
        "method": method,
        "symbol_scope": symbol_scope,
        "reference_method": REFERENCE_METHOD,
        "n_rows": int(len(frame)),
        "n_symbols": int(frame["symbol"].nunique()),
        "n_regimes": int(frame["regime"].nunique()),
        "min_regime_pct": float(counts.min() / max(counts.sum(), 1)),
        "max_regime_pct": float(counts.max() / max(counts.sum(), 1)),
        "regime_balance_entropy": normalized_entropy(counts.to_numpy()),
        "switch_count": switch_count,
        "switch_rate_per_1000": float(switch_count / max(transition_total, 1) * 1000.0),
        "transition_diagonal_probability": float(np.trace(transitions) / transition_total) if transition_total else np.nan,
        "transition_entropy": transition_entropy(transitions),
        "avg_regime_duration": float(runs.mean()) if len(runs) else np.nan,
        "median_regime_duration": float(runs.median()) if len(runs) else np.nan,
        "p90_regime_duration": float(runs.quantile(0.90)) if len(runs) else np.nan,
        **posterior,
        **reference,
    }


def load_assignments(path: str, symbols: list[str]) -> pd.DataFrame:
    assignments = pd.read_csv(path)
    required = {"method", "symbol", "open_time", "feat_idx", "regime"}
    missing = required - set(assignments.columns)
    if missing:
        raise SystemExit(f"Regime assignment file is missing columns: {sorted(missing)}")
    assignments = assignments[assignments["symbol"].isin(symbols)].copy()
    assignments["feat_idx"] = assignments["feat_idx"].astype(int)
    assignments["regime"] = assignments["regime"].astype(int)
    return assignments.sort_values(["method", "symbol", "feat_idx"]).reset_index(drop=True)


def common_universe(assignments: pd.DataFrame) -> pd.DataFrame:
    method_count = assignments["method"].nunique()
    counts = assignments.groupby(KEY_COLS)["method"].nunique().reset_index(name="method_count")
    common = counts[counts["method_count"] == method_count][KEY_COLS]
    return assignments.merge(common, on=KEY_COLS, how="inner")


def build_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reference_all = assignments[assignments["method"] == REFERENCE_METHOD]
    for method in METHOD_ORDER:
        method_frame = assignments[assignments["method"] == method]
        if method_frame.empty:
            continue
        rows.append(summarize_method(method_frame, reference_all, "ALL"))
        for symbol, symbol_frame in method_frame.groupby("symbol", sort=False):
            reference_symbol = reference_all[reference_all["symbol"] == symbol]
            rows.append(summarize_method(symbol_frame, reference_symbol, symbol))
    return pd.DataFrame(rows)


def agreement_for_pair(left: pd.DataFrame, right: pd.DataFrame, symbol_scope: str) -> dict:
    merged = left[KEY_COLS + ["regime"]].merge(
        right[KEY_COLS + ["regime"]],
        on=KEY_COLS,
        suffixes=("_a", "_b"),
        how="inner",
    )
    method_a = str(left["method"].iloc[0])
    method_b = str(right["method"].iloc[0])
    if merged.empty:
        return {
            "symbol_scope": symbol_scope,
            "method_a": method_a,
            "method_b": method_b,
            "n_rows": 0,
            "nmi": np.nan,
            "ari": np.nan,
            "same_label_pct": np.nan,
        }
    labels_a = merged["regime_a"].astype(int)
    labels_b = merged["regime_b"].astype(int)
    return {
        "symbol_scope": symbol_scope,
        "method_a": method_a,
        "method_b": method_b,
        "n_rows": int(len(merged)),
        "nmi": float(normalized_mutual_info_score(labels_a, labels_b)),
        "ari": float(adjusted_rand_score(labels_a, labels_b)),
        "same_label_pct": float((labels_a == labels_b).mean()),
    }


def build_agreement(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", assignments)]
    scopes.extend((symbol, frame) for symbol, frame in assignments.groupby("symbol", sort=False))

    for symbol_scope, frame in scopes:
        for method_a in METHOD_ORDER:
            left = frame[frame["method"] == method_a]
            if left.empty:
                continue
            for method_b in METHOD_ORDER:
                right = frame[frame["method"] == method_b]
                if right.empty:
                    continue
                rows.append(agreement_for_pair(left, right, symbol_scope))
    return pd.DataFrame(rows)


def bounded_quality_table(summary: pd.DataFrame) -> pd.DataFrame:
    all_rows = summary[summary["symbol_scope"] == "ALL"].copy()
    all_rows["posterior_confidence_score"] = 1.0 - all_rows["mean_posterior_entropy"]
    columns = [
        "regime_balance_entropy",
        "transition_diagonal_probability",
        "posterior_confidence_score",
        "hmm_reference_nmi",
        "hmm_reference_purity",
    ]
    return all_rows.set_index("method")[columns].reindex(METHOD_ORDER)


def plot_quality(summary: pd.DataFrame, output_path: Path) -> None:
    matrix = bounded_quality_table(summary)
    fig, ax = plt.subplots(figsize=(10, 5))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = values[row, col]
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", color="white" if np.isfinite(value) and value < 0.55 else "black")
    ax.set_title("Phase 16 - Regime Quality Metrics")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_agreement(agreement: pd.DataFrame, output_path: Path) -> None:
    overall = agreement[agreement["symbol_scope"] == "ALL"]
    pivot = overall.pivot(index="method_a", columns="method_b", values="nmi").reindex(index=METHOD_ORDER, columns=METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(7, 6))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "NA" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", color="white" if np.isfinite(value) and value < 0.55 else "black")
    ax.set_title("Phase 16 - Pairwise Regime Agreement (NMI)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(SAVE_DIR)
    output_dir.mkdir(exist_ok=True)

    assignments = load_assignments(args.assignments, args.symbols)
    common = common_universe(assignments)
    summary = build_summary(common)
    agreement = build_agreement(common)

    summary.to_csv(output_dir / "regime_quality_summary.csv", index=False)
    agreement.to_csv(output_dir / "regime_agreement_matrix.csv", index=False)
    plot_quality(summary, output_dir / "regime_quality_heatmap.png")
    plot_agreement(agreement, output_dir / "regime_agreement_heatmap.png")

    print("\nRegime quality summary (ALL):")
    columns = [
        "method",
        "n_rows",
        "regime_balance_entropy",
        "switch_rate_per_1000",
        "transition_diagonal_probability",
        "mean_confidence",
        "hmm_reference_nmi",
        "hmm_reference_ari",
        "hmm_reference_purity",
    ]
    print(summary[summary["symbol_scope"] == "ALL"][columns].to_string(index=False))

    print("\nPairwise NMI agreement (ALL):")
    pivot = (
        agreement[agreement["symbol_scope"] == "ALL"]
        .pivot(index="method_a", columns="method_b", values="nmi")
        .reindex(index=METHOD_ORDER, columns=METHOD_ORDER)
    )
    print(pivot.to_string())
    print("\nOK: regime quality artifacts saved.")


if __name__ == "__main__":
    main()
