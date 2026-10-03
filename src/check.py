import argparse

import duckdb
import pandas as pd
from config import DB_PATH, FEATURE_COLS
from universe import add_symbol_args, resolve_symbols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OHLCV/features quality for selected symbols.")
    add_symbol_args(parser)
    parser.add_argument("--db-path", default=DB_PATH)
    return parser.parse_args()


args = parse_args()
symbols = resolve_symbols(args)

con = duckdb.connect(args.db_path, read_only=True)

# ── OHLCV summary ─────────────────────────────────────────────────────────────
print("=" * 55)
print("OHLCV SUMMARY")
print("=" * 55)
ohlcv = con.execute("""
    SELECT
        symbol,
        COUNT(*)       AS total_bars,
        MIN(open_time) AS earliest,
        MAX(open_time) AS latest
    FROM ohlcv
    WHERE symbol IN ({})
    GROUP BY symbol
    ORDER BY symbol
""".format(",".join(["?"] * len(symbols))), symbols).df()
print(ohlcv.to_string(index=False))
missing_ohlcv = sorted(set(symbols) - set(ohlcv["symbol"].tolist()))
if missing_ohlcv:
    print(f"\nMissing OHLCV symbols ({len(missing_ohlcv)}): {', '.join(missing_ohlcv)}")

# ── Feature summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("FEATURE SUMMARY")
print("=" * 55)
feat = con.execute("""
    SELECT
        symbol,
        COUNT(*)            AS total_rows,
        MIN(open_time)      AS earliest,
        MAX(open_time)      AS latest,
        ROUND(AVG(rsi_14), 2) AS avg_rsi,
        ROUND(AVG(vol_20h), 6) AS avg_vol
    FROM features
    WHERE symbol IN ({})
    GROUP BY symbol
    ORDER BY symbol
""".format(",".join(["?"] * len(symbols))), symbols).df()
print(feat.to_string(index=False))
missing_features = sorted(set(symbols) - set(feat["symbol"].tolist()))
if missing_features:
    print(f"\nMissing feature symbols ({len(missing_features)}): {', '.join(missing_features)}")
print(f"\nFeature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── Gap check ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("GAP CHECK")
print("=" * 55)
for symbol in symbols:
    df = con.execute("""
        SELECT open_time FROM ohlcv
        WHERE symbol = ?
        ORDER BY open_time
    """, [symbol]).df()
    if df.empty:
        print(f"  {symbol}: missing OHLCV rows")
        continue
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["diff"]      = df["open_time"].diff()
    gaps = df[df["diff"] > pd.Timedelta("1h")].dropna()
    if gaps.empty:
        print(f"  {symbol}: no gaps found")
    else:
        print(f"  {symbol}: {len(gaps)} gaps found")
        print(gaps[["open_time", "diff"]].head(5).to_string(index=False))

con.close()
print("\nOK: Data check complete.")
