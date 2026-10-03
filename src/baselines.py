import argparse
import os
from dataclasses import dataclass

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import DB_PATH, FEATURE_COLS, N_REGIMES, SAVE_DIR, SYMBOLS


PRIMARY_TARGET = "vol_adj_return_8h"
HMM_FEATURES = ["ret_1h", "vol_20h", "volume_zscore", "spread_proxy", "rsi_14"]
RANDOM_STATE = 42
POST_COLS = [f"post_{k}" for k in range(N_REGIMES)]
ASSIGNMENT_COLS = ["method", "symbol", "open_time", "feat_idx", "regime"] + POST_COLS
METHOD_ORDER = ["contrastive", "contrastive_hmm", "hmm", "kmeans", "vol_bucket"]


@dataclass
class RegimeOutput:
    method: str
    assignments: pd.DataFrame
    feature_matrix: np.ndarray
    labels: np.ndarray
    implementation: str = ""


def load_research_frame(symbols: list[str]) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    frames = []
    for symbol in symbols:
        df = con.execute(
            f"""
            SELECT
                f.open_time,
                f.symbol,
                {", ".join("f." + c for c in FEATURE_COLS)},
                t.forward_return_8h,
                t.vol_adj_return_8h,
                t.tb_label_8h
            FROM features f
            JOIN targets t
              ON f.symbol = t.symbol
             AND f.open_time = t.open_time
            WHERE f.symbol = ?
            ORDER BY f.open_time
            """,
            [symbol],
        ).df()
        if df.empty:
            print(f"NOTE: no joined feature/target rows for {symbol}; skipping.")
            continue
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.reset_index(drop=True)
        df["feat_idx"] = df.index.astype(int)
        frames.append(df)
    con.close()

    if not frames:
        raise RuntimeError("No joined feature/target rows. Run targets.py first.")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)


def scaled_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    return StandardScaler().fit_transform(x)


def one_hot_posts(labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({f"post_{k}": (labels == k).astype(float) for k in range(N_REGIMES)})


def probability_posts(probs: np.ndarray) -> pd.DataFrame:
    posts = np.zeros((len(probs), N_REGIMES), dtype=float)
    cols = min(probs.shape[1], N_REGIMES)
    posts[:, :cols] = probs[:, :cols]
    row_sum = posts.sum(axis=1, keepdims=True)
    posts = np.divide(posts, row_sum, out=np.full_like(posts, 1.0 / N_REGIMES), where=row_sum != 0)
    return pd.DataFrame({f"post_{k}": posts[:, k] for k in range(N_REGIMES)})


def load_contrastive(df: pd.DataFrame) -> RegimeOutput:
    path = os.path.join(SAVE_DIR, "regime_posteriors.csv")
    if not os.path.exists(path):
        raise RuntimeError("regime_posteriors.csv missing. Run visualize_regimes.py first.")

    reg = pd.read_csv(path)
    if "symbol" not in reg.columns:
        raise RuntimeError("regime_posteriors.csv is missing symbol. Rerun visualize_regimes.py.")
    reg["open_time"] = pd.to_datetime(reg["open_time"])
    reg["feat_idx"] = reg["feat_idx"].astype(int)

    keys = df[["symbol", "feat_idx"]].drop_duplicates()
    reg = reg.merge(keys, on=["symbol", "feat_idx"], how="inner")
    reg["method"] = "contrastive"
    reg = reg[ASSIGNMENT_COLS + ["embedding_idx"]].sort_values(["symbol", "open_time"])

    embedding_path = os.path.join(SAVE_DIR, "embeddings.npy")
    if os.path.exists(embedding_path) and "embedding_idx" in reg.columns:
        embeddings = np.load(embedding_path)
        valid_idx = reg["embedding_idx"].astype(int).to_numpy()
        feature_matrix = embeddings[valid_idx]
    else:
        merged = reg.merge(df[["symbol", "feat_idx"] + FEATURE_COLS], on=["symbol", "feat_idx"])
        feature_matrix = scaled_matrix(merged, FEATURE_COLS)

    assignments = reg[ASSIGNMENT_COLS].reset_index(drop=True)
    return RegimeOutput(
        "contrastive",
        assignments,
        feature_matrix,
        assignments["regime"].to_numpy(dtype=int),
        "contrastive_encoder_gmm",
    )


def fit_contrastive_hmm(contrastive: RegimeOutput) -> RegimeOutput:
    assignments = contrastive.assignments.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    x = StandardScaler().fit_transform(contrastive.feature_matrix)
    lengths = assignments.groupby("symbol", sort=False).size().astype(int).tolist()
    implementation = "contrastive_embedding_hmm"

    try:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(
            n_components=N_REGIMES,
            covariance_type="diag",
            n_iter=250,
            random_state=RANDOM_STATE,
            min_covar=1e-4,
        )
        model.fit(x, lengths)
        labels = model.predict(x, lengths).astype(int)
        posts = probability_posts(model.predict_proba(x, lengths))
    except Exception as exc:
        implementation = "contrastive_embedding_gmm_fallback"
        print(f"NOTE: contrastive-HMM failed ({exc}). Using embedding GaussianMixture fallback.")
        model = GaussianMixture(
            n_components=N_REGIMES,
            covariance_type="full",
            n_init=10,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(x).astype(int)
        posts = probability_posts(model.predict_proba(x))

    hybrid = assignments[["symbol", "open_time", "feat_idx"]].copy()
    hybrid["method"] = "contrastive_hmm"
    hybrid["regime"] = labels
    hybrid = pd.concat([hybrid, posts], axis=1)
    return RegimeOutput(
        "contrastive_hmm",
        hybrid[ASSIGNMENT_COLS],
        x,
        labels,
        implementation,
    )


def fit_hmm_like(df: pd.DataFrame) -> RegimeOutput:
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    x = scaled_matrix(df, HMM_FEATURES)
    lengths = df.groupby("symbol", sort=False).size().astype(int).tolist()
    implementation = "hmmlearn_gaussian_hmm"

    try:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(
            n_components=N_REGIMES,
            covariance_type="full",
            n_iter=200,
            random_state=RANDOM_STATE,
        )
        model.fit(x, lengths)
        labels = model.predict(x, lengths)
        posts = probability_posts(model.predict_proba(x, lengths))
    except Exception as exc:
        implementation = "gmm_fallback"
        print(f"NOTE: hmmlearn unavailable or failed ({exc}). Using GaussianMixture fallback.")
        model = GaussianMixture(
            n_components=N_REGIMES,
            covariance_type="full",
            n_init=10,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(x)
        posts = probability_posts(model.predict_proba(x))

    assignments = df[["symbol", "open_time", "feat_idx"]].copy()
    assignments["method"] = "hmm"
    assignments["regime"] = labels.astype(int)
    assignments = pd.concat([assignments, posts], axis=1)
    return RegimeOutput("hmm", assignments[ASSIGNMENT_COLS], x, labels, implementation)


def fit_kmeans(df: pd.DataFrame) -> RegimeOutput:
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    x = scaled_matrix(df, FEATURE_COLS)
    labels = KMeans(n_clusters=N_REGIMES, n_init=20, random_state=RANDOM_STATE).fit_predict(x)
    assignments = df[["symbol", "open_time", "feat_idx"]].copy()
    assignments["method"] = "kmeans"
    assignments["regime"] = labels.astype(int)
    assignments = pd.concat([assignments, one_hot_posts(labels)], axis=1)
    return RegimeOutput("kmeans", assignments[ASSIGNMENT_COLS], x, labels, "sklearn_kmeans")


def fit_vol_buckets(df: pd.DataFrame) -> RegimeOutput:
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    labels = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("symbol", sort=False).groups.items():
        symbol_labels = pd.qcut(
            df.loc[idx, "vol_20h"].rank(method="first"),
            q=N_REGIMES,
            labels=False,
        ).astype(int)
        labels[np.asarray(idx)] = symbol_labels.to_numpy()

    x = scaled_matrix(df, FEATURE_COLS)
    assignments = df[["symbol", "open_time", "feat_idx"]].copy()
    assignments["method"] = "vol_bucket"
    assignments["regime"] = labels
    assignments = pd.concat([assignments, one_hot_posts(labels)], axis=1)
    return RegimeOutput("vol_bucket", assignments[ASSIGNMENT_COLS], x, labels, "volatility_quantiles")


def avg_regime_duration(assignments: pd.DataFrame) -> float:
    lengths = []
    for _, group in assignments.sort_values(["symbol", "open_time"]).groupby("symbol"):
        labels = group["regime"].to_numpy(dtype=int)
        if len(labels) == 0:
            continue
        current = labels[0]
        count = 1
        for label in labels[1:]:
            if label == current:
                count += 1
            else:
                lengths.append(count)
                current = label
                count = 1
        lengths.append(count)
    return float(np.mean(lengths)) if lengths else 0.0


def transition_matrix(assignments: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((N_REGIMES, N_REGIMES), dtype=float)
    for _, group in assignments.sort_values(["symbol", "open_time"]).groupby("symbol"):
        labels = group["regime"].to_numpy(dtype=int)
        for a, b in zip(labels[:-1], labels[1:]):
            matrix[int(a), int(b)] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


def save_transition_plot(method: str, assignments: pd.DataFrame) -> None:
    matrix = transition_matrix(assignments)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Transition Matrix - {method}")
    ax.set_xlabel("Next regime")
    ax.set_ylabel("Current regime")
    ax.set_xticks(range(N_REGIMES))
    ax.set_yticks(range(N_REGIMES))
    for i in range(N_REGIMES):
        for j in range(N_REGIMES):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"transition_matrix_{method}.png"), dpi=150)
    plt.close(fig)


def summarize_method(output: RegimeOutput) -> dict:
    labels = output.labels.astype(int)
    if len(np.unique(labels)) > 1 and len(labels) > N_REGIMES:
        sil = silhouette_score(
            output.feature_matrix,
            labels,
            sample_size=min(2000, len(labels)),
            random_state=RANDOM_STATE,
        )
    else:
        sil = np.nan

    counts = pd.Series(labels).value_counts(normalize=True)
    return {
        "method": output.method,
        "implementation": output.implementation,
        "n_rows": int(len(labels)),
        "n_symbols": int(output.assignments["symbol"].nunique()),
        "n_regimes": int(len(np.unique(labels))),
        "silhouette": float(sil) if np.isfinite(sil) else np.nan,
        "avg_regime_duration": avg_regime_duration(output.assignments),
        "min_regime_pct": float(counts.min()),
        "max_regime_pct": float(counts.max()),
    }


def per_regime_stats(df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    joined = assignments.merge(
        df[
            [
                "symbol",
                "feat_idx",
                "ret_1h",
                "vol_20h",
                "amihud",
                "volume_zscore",
                "ret_autocorr",
                "forward_return_8h",
                "vol_adj_return_8h",
            ]
        ],
        on=["symbol", "feat_idx"],
        how="left",
    )

    rows = []
    for (method, symbol, regime), group in joined.groupby(["method", "symbol", "regime"]):
        ic = group["ret_1h"].corr(group["vol_adj_return_8h"])
        rows.append(
            {
                "method": method,
                "symbol": symbol,
                "regime": int(regime),
                "n_rows": int(len(group)),
                "pct_rows": float(len(group) / len(joined)),
                "avg_vol_20h": float(group["vol_20h"].mean()),
                "avg_forward_return_8h": float(group["forward_return_8h"].mean()),
                "avg_vol_adj_return_8h": float(group["vol_adj_return_8h"].mean()),
                "avg_amihud": float(group["amihud"].mean()),
                "avg_volume_zscore": float(group["volume_zscore"].mean()),
                "avg_ret_autocorr": float(group["ret_autocorr"].mean()),
                "feature_ic_vs_target": float(ic) if pd.notna(ic) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def common_keys_for_outputs(outputs: list[RegimeOutput]) -> pd.DataFrame:
    keys = pd.concat(
        [
            output.assignments[["symbol", "feat_idx"]].drop_duplicates().assign(method=output.method)
            for output in outputs
        ],
        ignore_index=True,
    )
    counts = (
        keys.groupby(["symbol", "feat_idx"])["method"]
        .nunique()
        .reset_index(name="n_methods")
    )
    common = counts[counts["n_methods"] == len(outputs)][["symbol", "feat_idx"]]
    if common.empty:
        raise RuntimeError("No common regime-assignment rows across benchmark methods.")
    return common


def filter_output_to_keys(output: RegimeOutput, common_keys: pd.DataFrame) -> RegimeOutput:
    marked = (
        output.assignments.reset_index(names="_pos")
        .merge(common_keys, on=["symbol", "feat_idx"], how="inner")
        .sort_values("_pos")
    )
    positions = marked["_pos"].to_numpy(dtype=int)
    assignments = marked.drop(columns=["_pos"]).reset_index(drop=True)
    return RegimeOutput(
        output.method,
        assignments,
        output.feature_matrix[positions],
        assignments["regime"].to_numpy(dtype=int),
        output.implementation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regime benchmark baselines.")
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    args = parser.parse_args()

    df = load_research_frame(args.symbols)
    contrastive = load_contrastive(df)
    outputs = [
        contrastive,
        fit_contrastive_hmm(contrastive),
        fit_hmm_like(df),
        fit_kmeans(df),
        fit_vol_buckets(df),
    ]
    common_keys = common_keys_for_outputs(outputs)
    outputs = [filter_output_to_keys(output, common_keys) for output in outputs]
    df = df.merge(common_keys, on=["symbol", "feat_idx"], how="inner")

    assignments = pd.concat([o.assignments for o in outputs], ignore_index=True)
    assignments = assignments.sort_values(["method", "symbol", "feat_idx"]).reset_index(drop=True)
    assignments.to_csv(os.path.join(SAVE_DIR, "regime_assignments.csv"), index=False)

    summary = pd.DataFrame([summarize_method(o) for o in outputs])
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method").astype({"method": str})
    summary.to_csv(os.path.join(SAVE_DIR, "regime_benchmark_summary.csv"), index=False)

    stats = per_regime_stats(df, assignments)
    stats.to_csv(os.path.join(SAVE_DIR, "per_regime_stats.csv"), index=False)

    for output in outputs:
        save_transition_plot(output.method, output.assignments)

    print("\nRegime benchmark summary:")
    print(summary.to_string(index=False))
    print("\nOK: regime benchmark artifacts saved.")


if __name__ == "__main__":
    main()
