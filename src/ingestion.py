import argparse
import time
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import duckdb
from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET,
    SYMBOLS, INTERVAL, DB_PATH, LOOKBACK_DAYS
)
from universe import add_symbol_args, resolve_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

MAX_BARS_PER_REQUEST = 1000


def get_client() -> Client:
    """Return Binance client. Works without keys for public endpoints."""
    if BINANCE_API_KEY:
        return Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    return Client()


def fetch_klines(
    client: Client,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Pull all klines between start_ms and end_ms, handling pagination."""
    all_klines = []
    current_start = start_ms
    interval_ms = INTERVAL_MS[interval]

    while current_start < end_ms:
        try:
            raw = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                endTime=end_ms,
                limit=MAX_BARS_PER_REQUEST,
            )
        except BinanceAPIException as e:
            log.error(f"Binance API error for {symbol}: {e}")
            raise

        if not raw:
            break

        all_klines.extend(raw)
        log.info(f"  {symbol}: fetched {len(raw)} bars, total: {len(all_klines)}")

        last_open_time = raw[-1][0]
        current_start = last_open_time + interval_ms
        time.sleep(0.1)

        if len(raw) < MAX_BARS_PER_REQUEST:
            break

    if not all_klines:
        return pd.DataFrame()

    return _parse_klines(all_klines, symbol)


def _parse_klines(raw: list, symbol: str) -> pd.DataFrame:
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["symbol"] = symbol
    df = df.drop(columns=["ignore"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def init_db(db_path: str) -> duckdb.DuckDBPyConnection:
    import os
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol               VARCHAR,
            open_time            TIMESTAMPTZ,
            close_time           TIMESTAMPTZ,
            open                 DOUBLE,
            high                 DOUBLE,
            low                  DOUBLE,
            close                DOUBLE,
            volume               DOUBLE,
            quote_volume         DOUBLE,
            num_trades           INTEGER,
            taker_buy_base_vol   DOUBLE,
            taker_buy_quote_vol  DOUBLE,
            PRIMARY KEY (symbol, open_time)
        )
    """)
    return con


def get_latest_timestamp(con: duckdb.DuckDBPyConnection, symbol: str):
    result = con.execute(
        "SELECT MAX(open_time) FROM ohlcv WHERE symbol = ?", [symbol]
    ).fetchone()[0]
    if result is None:
        return None
    return int(result.timestamp() * 1000)


def upsert_ohlcv(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    con.register("new_data", df)
    con.execute("""
        INSERT OR IGNORE INTO ohlcv
        SELECT
            symbol, open_time, close_time,
            open, high, low, close, volume,
            quote_volume, num_trades,
            taker_buy_base_vol, taker_buy_quote_vol
        FROM new_data
    """)
    con.unregister("new_data")
    return len(df)


def ingest(symbols=SYMBOLS, interval=INTERVAL, db_path=DB_PATH):
    client = get_client()
    con    = init_db(db_path)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for symbol in symbols:
        log.info(f"Ingesting {symbol} [{interval}]")
        latest_ms = get_latest_timestamp(con, symbol)

        if latest_ms is None:
            start_ms = int(
                (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS))
                .timestamp() * 1000
            )
            log.info(f"  First run - fetching {LOOKBACK_DAYS} days of history")
        else:
            start_ms = latest_ms + INTERVAL_MS[interval]
            log.info(f"  Incremental update from {pd.Timestamp(latest_ms, unit='ms', tz='UTC')}")

        df   = fetch_klines(client, symbol, interval, start_ms, now_ms)
        if df.empty:
            log.info(f"  {symbol}: already up to date")
            continue

        rows = upsert_ohlcv(con, df)
        log.info(f"  {symbol}: inserted {rows} new bars")

    con.close()
    log.info("Ingestion complete.")


def verify(db_path=DB_PATH):
    con = duckdb.connect(db_path, read_only=True)
    summary = con.execute("""
        SELECT
            symbol,
            COUNT(*)        AS bars,
            MIN(open_time)  AS earliest,
            MAX(open_time)  AS latest,
            ROUND(AVG(volume), 2) AS avg_volume
        FROM ohlcv
        GROUP BY symbol
        ORDER BY symbol
    """).df()
    con.close()
    print(summary.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Binance OHLCV bars into DuckDB.")
    add_symbol_args(parser)
    parser.add_argument("--interval", default=INTERVAL)
    parser.add_argument("--db-path", default=DB_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_symbols = resolve_symbols(args)
    ingest(symbols=selected_symbols, interval=args.interval, db_path=args.db_path)
    verify(args.db_path)
