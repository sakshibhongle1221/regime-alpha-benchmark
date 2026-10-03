import argparse

import duckdb
import pandas as pd
import numpy as np
from config import DB_PATH, FEATURE_COLS
from universe import add_symbol_args, resolve_symbols


def load_ohlcv(symbol: str, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT open_time, open, high, low, close, volume,
               taker_buy_base_vol
        FROM ohlcv
        WHERE symbol = ?
        ORDER BY open_time
    """, [symbol]).df()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.set_index("open_time")
    return df


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    # 1-4. Multi-horizon log returns
    for h in [1, 5, 15, 60]:
        f[f"ret_{h}h"] = np.log(df["close"] / df["close"].shift(h))

    # 5-6. Realized volatility
    f["vol_5h"]  = f["ret_1h"].rolling(5).std()
    f["vol_20h"] = f["ret_1h"].rolling(20).std()

    # 7. Volatility of volatility
    f["vol_of_vol"] = f["vol_20h"].rolling(20).std()

    # 8. Amihud illiquidity
    f["amihud"] = f["ret_1h"].abs() / (df["volume"] + 1e-8)

    # 9. Volume Z-score
    vol_mean = df["volume"].rolling(20).mean()
    vol_std  = df["volume"].rolling(20).std()
    f["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-8)

    # 10. Return autocorrelation lag-1
    f["ret_autocorr"] = (
        f["ret_1h"]
        .rolling(20)
        .apply(lambda x: x.autocorr(lag=1), raw=False)
    )

    # 11. Bid-ask spread proxy
    f["spread_proxy"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

    # 12. Order flow imbalance proxy
    hl = df["high"] - df["low"] + 1e-8
    f["ofi_proxy"] = (df["close"] - df["open"]) / hl

    # 13. RSI 14-bar
    f["rsi_14"] = _rsi(df["close"], 14)

    # 14. Garman-Klass volatility
    f["gk_vol"] = _garman_klass(df)

    # 15. Rolling skewness
    f["skewness"] = f["ret_1h"].rolling(20).skew()

    # 16. Rolling kurtosis
    f["kurtosis"] = f["ret_1h"].rolling(20).kurt()

    # 17. MACD signal line
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    macd  = ema12 - ema26
    f["macd_signal"] = macd - macd.ewm(span=9).mean()

    # 18. Bollinger %B
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    f["bband_pct_b"] = (df["close"] - (sma20 - 2 * std20)) / (4 * std20 + 1e-8)

    # 19. ATR 14-bar
    f["atr_14"] = _atr(df, 14)

    # 20. Close vs VWAP proxy
    vwap = (df["close"] * df["volume"]).rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-8)
    f["close_vs_vwap"] = (df["close"] - vwap) / (vwap + 1e-8)

    # 21. Log volume trend 5-bar slope
    log_vol = np.log(df["volume"] + 1)
    f["log_vol_trend"] = log_vol.rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    # 22. Return dispersion across horizons
    f["ret_dispersion"] = pd.concat([
        f["ret_1h"], f["ret_5h"], f["ret_15h"], f["ret_60h"]
    ], axis=1).std(axis=1)

    return f


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _garman_klass(df: pd.DataFrame) -> pd.Series:
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    return np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift(1)).abs()
    lpc = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute engineered features from OHLCV bars.")
    add_symbol_args(parser)
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Replace features only for selected symbols instead of rebuilding the whole features table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = resolve_symbols(args)
    con = duckdb.connect(args.db_path)
    all_features = []

    for symbol in symbols:
        print(f"\nComputing features for {symbol}...")
        raw      = load_ohlcv(symbol, con)
        if raw.empty:
            print(f"  Skipping {symbol}: no OHLCV rows found")
            continue
        features = compute_features(raw)
        features["symbol"] = symbol
        features = features.reset_index().dropna()
        all_features.append(features)
        print(f"  Computed {len(features):,} rows for {symbol}")

    if not all_features:
        con.close()
        raise RuntimeError("No feature rows were built. Run ingestion.py first.")

    combined = pd.concat(all_features, ignore_index=True)
    con.register("feat_data", combined)
    if args.append and table_exists(con, "features"):
        con.execute(
            f"DELETE FROM features WHERE symbol IN ({','.join(['?'] * len(symbols))})",
            symbols,
        )
        con.execute("INSERT INTO features SELECT * FROM feat_data")
    else:
        con.execute("DROP TABLE IF EXISTS features")
        con.execute("CREATE TABLE features AS SELECT * FROM feat_data")
    con.unregister("feat_data")
    print(f"\n  Total saved: {len(combined):,} rows")

    result = con.execute("""
        SELECT symbol, COUNT(*) as rows,
               ROUND(AVG(vol_20h), 6) as avg_vol,
               ROUND(AVG(rsi_14), 2)  as avg_rsi
        FROM features
        GROUP BY symbol
    """).df()
    print("\nFeature summary:")
    print(result.to_string(index=False))

    con.close()
    print(f"\nOK: {len(FEATURE_COLS)} features saved to DuckDB. Ready for Phase 2.")


if __name__ == "__main__":
    main()
