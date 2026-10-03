import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import (
    ASSIGNMENT_COLS,
    N_REGIMES,
    RegimeOutput,
    avg_regime_duration,
    fit_hmm_like,
    fit_kmeans,
    fit_vol_buckets,
    load_research_frame,
    per_regime_stats,
    summarize_method,
    transition_matrix,
)
from config import SAVE_DIR
from universe import add_symbol_args, resolve_symbols


METHOD_ORDER = ["hmm", "kmeans", "vol_bucket"]
POST_COLS = [f"post_{idx}" for idx in range(N_REGIMES)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact Crypto-20 regime benchmark diagnostics."
    )
    add_symbol_args(parser)
    parser.set_defaults(universe="crypto20")
    parser.add_argument("--prefix", default="crypto20_")
    return parser.parse_args()


def regime_entropy(labels: pd.Series) -> float:
    counts = labels.value_counts(normalize=True).reindex(range(N_REGIMES), fill_value=0.0)
    probs = counts[counts > 0].to_numpy(dtype=float)
    if len(probs) == 0:
        return np.nan
    return float(-(probs * np.log(probs)).sum() / np.log(N_REGIMES))


def transition_diagonal(assignments: pd.DataFrame) -> float:
    matrix = transition_matrix(assignments)
    total = float(matrix.sum())
    if total <= 0:
        return np.nan
    return float(np.trace(matrix) / total)


def simple_feature_ic(frame: pd.DataFrame) -> float:
    numeric = frame[["vol_20h", "ret_1h", "ret_5h", "ret_15h", "ret_60h", "target_return"]].copy()
    if numeric["target_return"].nunique() <= 1:
        return np.nan
    scores = []
    for col in ["vol_20h", "ret_1h", "ret_5h", "ret_15h", "ret_60h"]:
        if numeric[col].nunique() <= 1:
            continue
        scores.append(abs(float(numeric[col].corr(numeric["target_return"], method="spearman"))))
    return float(np.nanmax(scores)) if scores else np.nan


def build_symbol_summary(df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    joined = assignments.merge(
        df[["symbol", "feat_idx", "forward_return_8h", "vol_adj_return_8h"] + ["vol_20h", "ret_1h", "ret_5h", "ret_15h", "ret_60h"]],
        on=["symbol", "feat_idx"],
        how="left",
    ).rename(columns={"forward_return_8h": "target_return"})
    rows = []
    for (method, symbol), group in joined.groupby(["method", "symbol"], sort=False):
        regime_counts = group["regime"].value_counts(normalize=True)
        rows.append(
            {
                "method": method,
                "symbol": symbol,
                "n_rows": int(len(group)),
                "n_regimes": int(group["regime"].nunique()),
                "avg_regime_duration": avg_regime_duration(group[ASSIGNMENT_COLS]),
                "transition_diagonal_probability": transition_diagonal(group[ASSIGNMENT_COLS]),
                "regime_entropy": regime_entropy(group["regime"]),
                "dominant_regime_pct": float(regime_counts.max()) if not regime_counts.empty else np.nan,
                "avg_forward_return_8h": float(group["target_return"].mean()),
                "avg_vol_adj_return_8h": float(group["vol_adj_return_8h"].mean()),
                "simple_feature_ic": simple_feature_ic(group),
            }
        )
    return pd.DataFrame(rows)


def save_transition_plot(prefix: str, method: str, assignments: pd.DataFrame) -> None:
    matrix = transition_matrix(assignments)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Crypto-20 Transition Matrix - {method}")
    ax.set_xlabel("Next regime")
    ax.set_ylabel("Current regime")
    ax.set_xticks(range(N_REGIMES))
    ax.set_yticks(range(N_REGIMES))
    for i in range(N_REGIMES):
        for j in range(N_REGIMES):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(Path(SAVE_DIR) / f"{prefix}transition_matrix_{method}.png", dpi=150)
    plt.close(fig)


def write_outputs(prefix: str, df: pd.DataFrame, outputs: list[RegimeOutput]) -> None:
    out_dir = Path(SAVE_DIR)
    out_dir.mkdir(exist_ok=True)

    assignments = pd.concat([output.assignments for output in outputs], ignore_index=True)
    assignments = assignments.sort_values(["method", "symbol", "feat_idx"]).reset_index(drop=True)
    assignments.to_csv(out_dir / f"{prefix}regime_assignments.csv", index=False)

    summary = pd.DataFrame([summarize_method(output) for output in outputs])
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method").astype({"method": str})
    summary.to_csv(out_dir / f"{prefix}regime_benchmark_summary.csv", index=False)

    stats = per_regime_stats(df, assignments)
    stats.to_csv(out_dir / f"{prefix}per_regime_stats.csv", index=False)

    symbol_summary = build_symbol_summary(df, assignments)
    symbol_summary.to_csv(out_dir / f"{prefix}regime_symbol_summary.csv", index=False)

    for output in outputs:
        save_transition_plot(prefix, output.method, output.assignments)

    print("\nCrypto-20 regime benchmark summary:")
    print(summary.to_string(index=False))
    print("\nCrypto-20 per-symbol summary:")
    print(symbol_summary.groupby("method")[["n_rows", "transition_diagonal_probability", "dominant_regime_pct"]].mean().to_string())
    print(f"\nOK: Crypto-20 regime artifacts saved with prefix '{prefix}'.")


def main() -> None:
    args = parse_args()
    symbols = resolve_symbols(args)
    df = load_research_frame(symbols)
    print(f"Crypto-20 regime frame: {len(df):,} rows | symbols={len(symbols)}")

    outputs = [
        fit_hmm_like(df),
        fit_kmeans(df),
        fit_vol_buckets(df),
    ]
    write_outputs(args.prefix, df, outputs)


if __name__ == "__main__":
    main()
