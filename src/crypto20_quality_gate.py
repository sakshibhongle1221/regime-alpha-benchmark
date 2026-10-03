import argparse
from pathlib import Path

import duckdb
import pandas as pd

from config import DB_PATH, SAVE_DIR
from universe import add_symbol_args, resolve_symbols


DEFAULT_MIN_BARS = 12_000
DEFAULT_MAX_GAP_HOURS = 6.0
DEFAULT_MIN_PASS_RATE = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Crypto-20 data quality-gate artifacts after ingestion/features/targets."
    )
    add_symbol_args(parser)
    parser.set_defaults(universe="crypto20")
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--min-bars", type=int, default=DEFAULT_MIN_BARS)
    parser.add_argument("--max-gap-hours", type=float, default=DEFAULT_MAX_GAP_HOURS)
    parser.add_argument("--min-pass-rate", type=float, default=DEFAULT_MIN_PASS_RATE)
    return parser.parse_args()


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


def empty_quality(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"symbol": symbols})


def load_ohlcv_quality(con: duckdb.DuckDBPyConnection, symbols: list[str]) -> pd.DataFrame:
    if not table_exists(con, "ohlcv"):
        return empty_quality(symbols)
    return con.execute(
        f"""
        WITH ordered AS (
            SELECT
                symbol,
                open_time,
                quote_volume,
                open_time - LAG(open_time) OVER (
                    PARTITION BY symbol ORDER BY open_time
                ) AS gap
            FROM ohlcv
            WHERE symbol IN ({",".join(["?"] * len(symbols))})
        )
        SELECT
            symbol,
            COUNT(*) AS ohlcv_rows,
            MIN(open_time) AS ohlcv_earliest,
            MAX(open_time) AS ohlcv_latest,
            MEDIAN(quote_volume) AS median_quote_volume,
            SUM(CASE WHEN gap > INTERVAL 1 HOUR THEN 1 ELSE 0 END) AS gap_count,
            MAX(EXTRACT(EPOCH FROM gap) / 3600.0) AS max_gap_hours
        FROM ordered
        GROUP BY symbol
        ORDER BY symbol
        """,
        symbols,
    ).df()


def load_table_counts(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    symbols: list[str],
    row_col: str,
    earliest_col: str,
    latest_col: str,
) -> pd.DataFrame:
    if not table_exists(con, table_name):
        return empty_quality(symbols)
    return con.execute(
        f"""
        SELECT
            symbol,
            COUNT(*) AS {row_col},
            MIN(open_time) AS {earliest_col},
            MAX(open_time) AS {latest_col}
        FROM {table_name}
        WHERE symbol IN ({",".join(["?"] * len(symbols))})
        GROUP BY symbol
        ORDER BY symbol
        """,
        symbols,
    ).df()


def build_quality(symbols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    base = pd.DataFrame({"symbol": symbols})
    con = duckdb.connect(args.db_path, read_only=True)
    try:
        quality = base.merge(load_ohlcv_quality(con, symbols), on="symbol", how="left")
        feature_counts = load_table_counts(
            con,
            "features",
            symbols,
            "feature_rows",
            "feature_earliest",
            "feature_latest",
        )
        target_counts = load_table_counts(
            con,
            "targets",
            symbols,
            "target_rows",
            "target_earliest",
            "target_latest",
        )
    finally:
        con.close()

    quality = quality.merge(feature_counts, on="symbol", how="left")
    quality = quality.merge(target_counts, on="symbol", how="left")
    for col in ["ohlcv_rows", "feature_rows", "target_rows", "gap_count"]:
        quality[col] = quality[col].fillna(0).astype(int)
    for col in ["median_quote_volume", "max_gap_hours"]:
        quality[col] = quality[col].fillna(0.0).astype(float)

    quality["ohlcv_pass"] = quality["ohlcv_rows"] >= args.min_bars
    quality["feature_pass"] = quality["feature_rows"] > 0
    quality["target_pass"] = quality["target_rows"] > 0
    quality["gap_pass"] = quality["max_gap_hours"] <= args.max_gap_hours
    quality["quality_status"] = "pass"
    quality.loc[
        ~(quality["ohlcv_pass"] & quality["feature_pass"] & quality["target_pass"] & quality["gap_pass"]),
        "quality_status",
    ] = "fail"
    quality["failure_reason"] = ""
    quality.loc[~quality["ohlcv_pass"], "failure_reason"] += "insufficient_ohlcv;"
    quality.loc[~quality["feature_pass"], "failure_reason"] += "missing_features;"
    quality.loc[~quality["target_pass"], "failure_reason"] += "missing_targets;"
    quality.loc[~quality["gap_pass"], "failure_reason"] += "excessive_gaps;"
    quality["failure_reason"] = quality["failure_reason"].str.rstrip(";")
    return quality


def build_summary(quality: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    pass_count = int((quality["quality_status"] == "pass").sum())
    total = int(len(quality))
    pass_rate = pass_count / total if total else 0.0
    gate_status = "pass" if pass_rate >= args.min_pass_rate else "fail"
    return pd.DataFrame(
        [
            {"metric": "symbols_checked", "value": total, "notes": "Crypto-20 symbols resolved from universe artifact."},
            {"metric": "symbols_passed", "value": pass_count, "notes": "Symbols passing OHLCV, feature, target, and gap checks."},
            {"metric": "symbols_failed", "value": total - pass_count, "notes": "Symbols requiring exclusion, replacement, or re-ingestion."},
            {"metric": "pass_rate", "value": round(pass_rate, 4), "notes": "Pass fraction across selected universe."},
            {"metric": "min_required_pass_rate", "value": args.min_pass_rate, "notes": "Gate threshold before regime benchmarking."},
            {"metric": "min_bars", "value": args.min_bars, "notes": "Minimum OHLCV bars per symbol."},
            {"metric": "max_gap_hours", "value": args.max_gap_hours, "notes": "Maximum allowed adjacent-bar gap."},
            {"metric": "gate_status", "value": gate_status, "notes": "Proceed only if this is pass."},
        ]
    )


def main() -> None:
    args = parse_args()
    symbols = resolve_symbols(args)
    quality = build_quality(symbols, args)
    summary = build_summary(quality, args)

    out_dir = Path(SAVE_DIR)
    out_dir.mkdir(exist_ok=True)
    quality.to_csv(out_dir / "crypto20_data_quality.csv", index=False)
    summary.to_csv(out_dir / "crypto20_pipeline_summary.csv", index=False)

    print("Crypto-20 quality gate:")
    print(summary.to_string(index=False))
    failures = quality[quality["quality_status"] != "pass"]
    if not failures.empty:
        print("\nFailed symbols:")
        print(failures[["symbol", "failure_reason"]].to_string(index=False))
    print(f"\nSaved: {out_dir / 'crypto20_data_quality.csv'}")
    print(f"Saved: {out_dir / 'crypto20_pipeline_summary.csv'}")


if __name__ == "__main__":
    main()
