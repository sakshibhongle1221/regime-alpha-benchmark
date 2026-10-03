import os
from pathlib import Path

from dotenv import load_dotenv


# Absolute project paths. Keep all scripts independent of the current shell cwd.
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
DATA_DIR = BASE_DIR / "data"
LEGACY_DATA_DIR = SRC_DIR / "data"
SAVE_DIR = BASE_DIR / "models"
DB_FILENAME = "market.duckdb"

load_dotenv(BASE_DIR / ".env")

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")


def _resolve_db_path() -> Path:
    """Prefer the project data folder, but support the existing src/data DB."""
    project_db = DATA_DIR / DB_FILENAME
    legacy_db = LEGACY_DATA_DIR / DB_FILENAME

    if project_db.exists():
        return project_db
    if legacy_db.exists():
        return legacy_db
    return project_db


DATA_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(exist_ok=True)

DB_PATH = str(_resolve_db_path())
SAVE_DIR = str(SAVE_DIR)

SYMBOLS       = ["BTCUSDT", "ETHUSDT"]
INTERVAL      = "1h"
LOOKBACK_DAYS = 730

WINDOW_SIZE = 60
STRIDE      = 4
N_REGIMES   = 4
LATENT_DIM  = 16
N_FEATURES  = 22

FEATURE_COLS = [
    "ret_1h", "ret_5h", "ret_15h", "ret_60h",
    "vol_5h", "vol_20h", "vol_of_vol",
    "amihud", "volume_zscore", "ret_autocorr",
    "spread_proxy", "ofi_proxy", "rsi_14", "gk_vol",
    "skewness", "kurtosis", "macd_signal",
    "bband_pct_b", "atr_14", "close_vs_vwap",
    "log_vol_trend", "ret_dispersion"
]
