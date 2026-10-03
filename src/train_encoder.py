import argparse
import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import ConcatDataset, DataLoader

from config import (
    DB_PATH,
    LATENT_DIM,
    N_FEATURES,
    N_REGIMES,
    SAVE_DIR,
    STRIDE,
    SYMBOLS,
    WINDOW_SIZE,
)
from dataset import WindowDataset, load_feature_matrix
from encoder import NTXentLoss, TemporalEncoder


BATCH_SIZE = 128
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42
os.makedirs(SAVE_DIR, exist_ok=True)


def load_raw_matrices(symbols: list[str]) -> dict[str, np.ndarray]:
    matrices = {}
    for symbol in symbols:
        x_raw = load_feature_matrix(symbol)
        if len(x_raw) <= WINDOW_SIZE + 1:
            print(f"NOTE: {symbol} has too few rows for encoder windows; skipping.")
            continue
        matrices[symbol] = x_raw
        print(f"  {symbol}: {x_raw.shape}")
    if not matrices:
        raise RuntimeError("No feature matrices loaded. Run features.py first.")
    return matrices


def fit_shared_scaler(matrices: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack(list(matrices.values()))
    mean = combined.mean(axis=0)
    std = combined.std(axis=0) + 1e-8
    np.save(os.path.join(SAVE_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(SAVE_DIR, "norm_std.npy"), std)
    return mean, std


def normalize_matrices(
    matrices: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, np.ndarray]:
    return {symbol: (x_raw - mean) / std for symbol, x_raw in matrices.items()}


def load_feature_times(symbol: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    times = con.execute(
        """
        SELECT open_time FROM features
        WHERE symbol = ?
        ORDER BY open_time
        """,
        [symbol],
    ).df()
    con.close()
    times["open_time"] = pd.to_datetime(times["open_time"])
    return times


def build_dataloader(normalized: dict[str, np.ndarray]) -> DataLoader:
    datasets = [WindowDataset(x, window=WINDOW_SIZE) for x in normalized.values()]
    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset):,} windows | {len(dataloader)} batches/epoch")
    return dataloader


def train_model(dataloader: DataLoader, epochs: int) -> tuple[TemporalEncoder, list[float]]:
    model = TemporalEncoder(n_features=N_FEATURES, latent_dim=LATENT_DIM).to(DEVICE)
    criterion = NTXentLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print("\nTraining encoder...")

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for anchor, positive in dataloader:
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)

            loss = criterion(model(anchor), model(positive))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "encoder.pt"))
    print(f"\nModel saved to {os.path.join(SAVE_DIR, 'encoder.pt')}")
    return model, loss_history


def extract_sparse_embeddings(
    model: TemporalEncoder,
    normalized: dict[str, np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame]:
    print(f"\nExtracting sparse embeddings (stride={STRIDE})...")
    model.eval()
    all_embeddings = []
    rows = []

    with torch.no_grad():
        for symbol, x in normalized.items():
            times = load_feature_times(symbol)
            for start_idx in range(0, len(x) - WINDOW_SIZE, STRIDE):
                feat_idx = start_idx + WINDOW_SIZE - 1
                window = torch.tensor(
                    x[start_idx : start_idx + WINDOW_SIZE], dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)
                z = model(window)
                all_embeddings.append(z.cpu().numpy())
                rows.append(
                    {
                        "symbol": symbol,
                        "start_idx": start_idx,
                        "feat_idx": feat_idx,
                        "open_time": times["open_time"].iloc[feat_idx],
                    }
                )

    embeddings = np.vstack(all_embeddings)
    meta = pd.DataFrame(rows)
    np.save(os.path.join(SAVE_DIR, "embeddings.npy"), embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, meta


def save_sparse_regimes(embeddings: np.ndarray, meta: pd.DataFrame) -> None:
    print(f"\nFitting GMM with K={N_REGIMES} regimes...")
    gmm = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="full",
        n_init=5,
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

    np.save(os.path.join(SAVE_DIR, "labels.npy"), labels)
    np.save(os.path.join(SAVE_DIR, "posteriors.npy"), posteriors)
    meta.to_csv(os.path.join(SAVE_DIR, "regime_posteriors.csv"), index=False)
    print(f"Saved sparse regime posteriors: {len(meta):,} rows")

    print("Regime distribution:")
    for k in range(N_REGIMES):
        pct = (labels == k).mean() * 100
        print(f"  Regime {k}: {pct:.1f}% of windows")


def save_loss_curve(loss_history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, color="#3B82F6", linewidth=2)
    ax.set_title("Contrastive Training Loss (NT-Xent)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train contrastive temporal regime encoder.")
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print(f"Training on: {DEVICE}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Sparse artifact stride: {STRIDE}; visualize_regimes.py exports dense stride-1 regimes.")

    print("\nLoading features...")
    matrices = load_raw_matrices(args.symbols)
    mean, std = fit_shared_scaler(matrices)
    normalized = normalize_matrices(matrices, mean, std)

    dataloader = build_dataloader(normalized)
    model, loss_history = train_model(dataloader, args.epochs)
    embeddings, meta = extract_sparse_embeddings(model, normalized)
    save_sparse_regimes(embeddings, meta)
    save_loss_curve(loss_history)

    print("\nOK: Phase 2 training complete.")
    print("Run visualize_regimes.py next to overwrite sparse regimes with dense stride-1 assignments.")


if __name__ == "__main__":
    main()
