import argparse
import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from config import DB_PATH, LATENT_DIM, N_FEATURES, N_REGIMES, SAVE_DIR, SYMBOLS, WINDOW_SIZE
from dataset import load_feature_matrix, normalize
from encoder import TemporalEncoder


DENSE_STRIDE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42


def normalize_for_encoder(x_raw: np.ndarray) -> np.ndarray:
    mean_path = os.path.join(SAVE_DIR, "norm_mean.npy")
    std_path = os.path.join(SAVE_DIR, "norm_std.npy")
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)
        if len(mean) == x_raw.shape[1] and len(std) == x_raw.shape[1]:
            return (x_raw - mean) / (std + 1e-8)
    x, _, _ = normalize(x_raw)
    return x


def load_times_and_prices(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(DB_PATH, read_only=True)
    times = con.execute(
        """
        SELECT open_time FROM features
        WHERE symbol = ? ORDER BY open_time
        """,
        [symbol],
    ).df()
    price_df = con.execute(
        """
        SELECT open_time, close FROM ohlcv
        WHERE symbol = ? ORDER BY open_time
        """,
        [symbol],
    ).df()
    con.close()
    times["open_time"] = pd.to_datetime(times["open_time"])
    price_df["open_time"] = pd.to_datetime(price_df["open_time"])
    return times, price_df


def extract_symbol_embeddings(
    model: TemporalEncoder,
    symbol: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    x_raw = load_feature_matrix(symbol)
    x = normalize_for_encoder(x_raw)
    times, price_df = load_times_and_prices(symbol)

    embeddings = []
    rows = []
    print(f"Extracting dense embeddings for {symbol} (stride={DENSE_STRIDE})...")
    with torch.no_grad():
        for start_idx in range(0, len(x) - WINDOW_SIZE, DENSE_STRIDE):
            feat_idx = start_idx + WINDOW_SIZE - 1
            window = torch.tensor(
                x[start_idx : start_idx + WINDOW_SIZE], dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)
            z = model(window)
            embeddings.append(z.cpu().numpy())
            rows.append(
                {
                    "symbol": symbol,
                    "start_idx": start_idx,
                    "feat_idx": feat_idx,
                    "open_time": times["open_time"].iloc[feat_idx],
                }
            )

    if not embeddings:
        raise RuntimeError(f"No embeddings extracted for {symbol}.")
    return np.vstack(embeddings), pd.DataFrame(rows), price_df


def save_regime_artifacts(embeddings: np.ndarray, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    print("Fitting GMM on dense multi-symbol embeddings...")
    print("NOTE: This overwrites embeddings.npy, labels.npy, posteriors.npy, and regime_posteriors.csv.")
    print("Always run visualize_regimes.py before baselines.py and alpha_models.py.")

    gmm = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="full",
        n_init=10,
        random_state=RANDOM_STATE,
    )
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)
    posteriors = gmm.predict_proba(embeddings)

    meta = meta.reset_index(drop=True).copy()
    meta.insert(0, "embedding_idx", np.arange(len(meta), dtype=int))
    meta["regime"] = labels.astype(int)
    for k in range(N_REGIMES):
        meta[f"post_{k}"] = posteriors[:, k]

    np.save(os.path.join(SAVE_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(SAVE_DIR, "labels.npy"), labels)
    np.save(os.path.join(SAVE_DIR, "posteriors.npy"), posteriors)
    meta.to_csv(os.path.join(SAVE_DIR, "regime_posteriors.csv"), index=False)
    print(f"Saved aligned dense regime posteriors: {len(meta):,} rows")
    return labels, posteriors


def safe_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(
        silhouette_score(
            embeddings,
            labels,
            sample_size=min(2000, len(labels)),
            random_state=RANDOM_STATE,
        )
    )


def save_umap_plot(embeddings: np.ndarray, labels: np.ndarray, posteriors: np.ndarray, sil: float) -> None:
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=50,
        min_dist=0.1,
        random_state=RANDOM_STATE,
    )
    emb2d = reducer.fit_transform(embeddings)
    colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    for k in range(N_REGIMES):
        mask = labels == k
        ax.scatter(
            emb2d[mask, 0],
            emb2d[mask, 1],
            c=colors[k],
            label=f"Regime {k} ({mask.mean()*100:.0f}%)",
            alpha=0.5,
            s=5,
        )
    ax.set_title(f"UMAP - Dense Regime Labels\nSilhouette: {sil:.3f}", fontsize=12)
    ax.legend(markerscale=3, fontsize=9)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    certainty = posteriors.max(axis=1)
    ax2 = axes[1]
    sc = ax2.scatter(
        emb2d[:, 0],
        emb2d[:, 1],
        c=certainty,
        cmap="RdYlGn",
        alpha=0.5,
        s=5,
        vmin=0.4,
        vmax=1.0,
    )
    plt.colorbar(sc, ax=ax2, label="Regime certainty")
    ax2.set_title("UMAP - Regime Certainty\n(green=confident, red=uncertain)", fontsize=12)
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")

    plt.suptitle("Regime Alpha Benchmark - Dense Learned Regime Structure", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "umap_improved.png"), dpi=150)
    plt.close(fig)


def save_timeline_plot(
    primary_symbol: str,
    price_df: pd.DataFrame,
    meta: pd.DataFrame,
    labels: np.ndarray,
) -> None:
    colors = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B"]
    plot_meta = meta.copy()
    plot_meta["regime"] = labels.astype(int)
    plot_meta = plot_meta[plot_meta["symbol"] == primary_symbol].copy()
    plot_meta["open_time"] = pd.to_datetime(plot_meta["open_time"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    ax1.plot(price_df["open_time"], price_df["close"], color="#1e293b", linewidth=0.6)
    ax1.set_ylabel(f"{primary_symbol} price")
    ax1.set_title(f"{primary_symbol} Price - Hourly")
    ax1.grid(True, alpha=0.2)

    for k in range(N_REGIMES):
        mask = plot_meta["regime"] == k
        ax2.scatter(
            plot_meta.loc[mask, "open_time"],
            np.ones(mask.sum()) * k,
            c=colors[k],
            s=2,
            alpha=0.55,
            label=f"Regime {k}",
        )
    ax2.set_ylabel("Detected regime")
    ax2.set_title(f"Regime Timeline - {primary_symbol}")
    ax2.set_yticks(range(N_REGIMES))
    ax2.legend(markerscale=4, fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "regime_timeline.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dense contrastive regime artifacts.")
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--primary-symbol", default="BTCUSDT")
    args = parser.parse_args()

    print("Loading encoder...")
    model = TemporalEncoder(n_features=N_FEATURES, latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "encoder.pt"), map_location=DEVICE))
    model.eval()

    all_embeddings = []
    all_meta = []
    price_by_symbol = {}
    for symbol in args.symbols:
        embeddings, meta, price_df = extract_symbol_embeddings(model, symbol)
        all_embeddings.append(embeddings)
        all_meta.append(meta)
        price_by_symbol[symbol] = price_df

    embeddings = np.vstack(all_embeddings)
    meta = pd.concat(all_meta, ignore_index=True)
    labels, posteriors = save_regime_artifacts(embeddings, meta)

    sil = safe_silhouette(embeddings, labels)
    print(f"Silhouette score: {sil:.4f}")
    save_umap_plot(embeddings, labels, posteriors, sil)

    primary_symbol = args.primary_symbol if args.primary_symbol in price_by_symbol else args.symbols[0]
    save_timeline_plot(primary_symbol, price_by_symbol[primary_symbol], meta, labels)

    print("\n-- Regime Statistics ---------------------------------")
    for k in range(N_REGIMES):
        mask = labels == k
        avg_certainty = posteriors[mask, k].mean() if mask.any() else np.nan
        print(
            f"  Regime {k}: {mask.sum():5d} windows | "
            f"{mask.mean()*100:5.1f}% | avg certainty: {avg_certainty:.3f}"
        )
    print("OK: Dense visualizations and regime posteriors saved.")


if __name__ == "__main__":
    main()
