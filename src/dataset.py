import numpy as np
import pandas as pd
import duckdb
import torch
from pathlib import Path
from torch.utils.data import Dataset

from config import DB_PATH, WINDOW_SIZE, FEATURE_COLS


def load_feature_frame(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Load timestamped features for a symbol in strict chronological order."""
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(
            f"Market database not found at {DB_PATH}. "
            "Run ingestion.py and features.py first, or move market.duckdb into data/."
        )

    con = duckdb.connect(DB_PATH, read_only=True)
    cols = ", ".join(FEATURE_COLS)
    df = con.execute(f"""
        SELECT open_time, {cols}
        FROM features
        WHERE symbol = ?
        ORDER BY open_time
    """, [symbol]).df()
    con.close()
    df["open_time"] = pd.to_datetime(df["open_time"])
    return df


def load_feature_matrix(symbol: str = "BTCUSDT") -> np.ndarray:
    """Load all 22 features for a symbol as a numpy array, time-ordered."""
    return load_feature_frame(symbol)[FEATURE_COLS].to_numpy(dtype=np.float32)


def normalize(X: np.ndarray):
    """Standardize each feature to zero mean, unit variance."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


class WindowDataset(Dataset):
    """
    Sliding window dataset for contrastive learning.
    Each item is an (anchor, positive) pair:
      anchor:   window starting at index i
      positive: window starting at index i+1 (adjacent = same market context)
    Negatives are all other items in the batch (handled by NT-Xent loss).
    """
    def __init__(self, X: np.ndarray, window: int = WINDOW_SIZE):
        self.X      = X
        self.window = window
        self.indices = list(range(len(X) - window - 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i        = self.indices[idx]
        anchor   = self.X[i     : i + self.window]
        positive = self.X[i + 1 : i + self.window + 1]
        return (
            torch.tensor(anchor,   dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
        )
