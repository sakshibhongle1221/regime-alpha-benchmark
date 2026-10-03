import argparse
from pathlib import Path

import duckdb
import pandas as pd

from config import BASE_DIR, DB_PATH, INTERVAL, LOOKBACK_DAYS, SAVE_DIR


CANDIDATE_PATH = Path(BASE_DIR) / "configs" / "crypto_universe_candidates.csv"
REPORT_PATH = Path(BASE_DIR) / "reports" / "multiasset_universe_plan.md"
STABLE_OR_SYNTHETIC_BASES = {
    "BUSD",
    "DAI",
    "FDUSD",
    "TUSD",
    "USDC",
    "USDP",
    "USDT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reproducible Crypto-20/Crypto-50 universe selection artifacts."
    )
    parser.add_argument("--candidate-path", default=str(CANDIDATE_PATH))
    parser.add_argument("--top-n", nargs="*", type=int, default=[20, 50])
    parser.add_argument("--min-bars", type=int, default=12_000)
    parser.add_argument("--max-gap-hours", type=float, default=6.0)
    parser.add_argument("--min-median-quote-volume", type=float, default=0.0)
    parser.add_argument(
        "--require-db-history",
        action="store_true",
        help="Only select symbols that already pass local DuckDB coverage and liquidity checks.",
    )
    return parser.parse_args()


def load_candidates(path: str) -> pd.DataFrame:
    candidates = pd.read_csv(path)
    required = {"design_rank", "symbol", "base_asset", "universe_group", "notes"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate file missing columns: {missing}")
    candidates["symbol"] = candidates["symbol"].str.upper().str.strip()
    candidates["base_asset"] = candidates["base_asset"].str.upper().str.strip()
    candidates = candidates.drop_duplicates("symbol").sort_values("design_rank").reset_index(drop=True)
    return candidates


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


def load_local_quality(symbols: list[str]) -> pd.DataFrame:
    if not Path(DB_PATH).exists():
        return pd.DataFrame({"symbol": symbols})

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        if not table_exists(con, "ohlcv"):
            return pd.DataFrame({"symbol": symbols})

        quality = con.execute(
            f"""
            WITH ordered AS (
                SELECT
                    symbol,
                    open_time,
                    quote_volume,
                    ABS(high - low) / NULLIF(close, 0) AS range_proxy,
                    open_time - LAG(open_time) OVER (
                        PARTITION BY symbol ORDER BY open_time
                    ) AS gap
                FROM ohlcv
                WHERE symbol IN ({",".join(["?"] * len(symbols))})
            )
            SELECT
                symbol,
                COUNT(*) AS bars,
                MIN(open_time) AS earliest,
                MAX(open_time) AS latest,
                MEDIAN(quote_volume) AS median_quote_volume,
                MEDIAN(range_proxy) AS median_range_proxy,
                SUM(CASE WHEN gap > INTERVAL 1 HOUR THEN 1 ELSE 0 END) AS gap_count,
                MAX(EXTRACT(EPOCH FROM gap) / 3600.0) AS max_gap_hours
            FROM ordered
            GROUP BY symbol
            ORDER BY symbol
            """,
            symbols,
        ).df()
    finally:
        con.close()

    return quality


def score_candidates(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    quality = load_local_quality(candidates["symbol"].tolist())
    scored = candidates.merge(quality, on="symbol", how="left")
    scored["bars"] = scored["bars"].fillna(0).astype(int)
    scored["gap_count"] = scored["gap_count"].fillna(0).astype(int)
    scored["max_gap_hours"] = scored["max_gap_hours"].fillna(0.0).astype(float)
    scored["median_quote_volume"] = scored["median_quote_volume"].fillna(0.0).astype(float)
    scored["median_range_proxy"] = scored["median_range_proxy"].fillna(0.0).astype(float)

    scored["is_stable_or_synthetic"] = scored["base_asset"].isin(STABLE_OR_SYNTHETIC_BASES)
    scored["has_local_history"] = scored["bars"] > 0
    scored["coverage_pass"] = scored["bars"] >= args.min_bars
    scored["gap_pass"] = scored["max_gap_hours"] <= args.max_gap_hours
    scored["liquidity_pass"] = scored["median_quote_volume"] >= args.min_median_quote_volume
    scored["eligible"] = (
        ~scored["is_stable_or_synthetic"]
        & scored["coverage_pass"]
        & scored["gap_pass"]
        & scored["liquidity_pass"]
    )

    scored["selection_status"] = "pending_ingestion"
    scored.loc[scored["is_stable_or_synthetic"], "selection_status"] = "excluded_stable_or_synthetic"
    scored.loc[scored["has_local_history"] & ~scored["eligible"], "selection_status"] = "failed_quality_gate"
    scored.loc[scored["eligible"], "selection_status"] = "eligible"
    return scored


def select_universe(scored: pd.DataFrame, n: int, require_db_history: bool) -> pd.DataFrame:
    if require_db_history:
        pool = scored[scored["eligible"]].copy()
        pool = pool.sort_values(["median_quote_volume", "design_rank"], ascending=[False, True])
    else:
        pool = scored[~scored["is_stable_or_synthetic"]].copy()
        pool = pool.sort_values("design_rank")

    selected = pool.head(n).copy()
    selected.insert(0, "universe", f"crypto{n}")
    selected["target_size"] = n
    selected["selection_mode"] = "quality_gated" if require_db_history else "design_pending_quality_gate"
    selected["selected_by_design"] = True
    return selected


def build_exclusions(scored: pd.DataFrame, selected_symbols: set[str]) -> pd.DataFrame:
    excluded = scored[~scored["symbol"].isin(selected_symbols)].copy()
    reasons = []
    for row in excluded.itertuples(index=False):
        if row.is_stable_or_synthetic:
            reasons.append("stable_or_synthetic")
        elif row.has_local_history and not row.coverage_pass:
            reasons.append("insufficient_history")
        elif row.has_local_history and not row.gap_pass:
            reasons.append("excessive_gaps")
        elif row.has_local_history and not row.liquidity_pass:
            reasons.append("insufficient_liquidity")
        else:
            reasons.append("outside_requested_top_n")
    excluded["exclusion_reason"] = reasons
    return excluded


def build_summary(scored: pd.DataFrame, selections: dict[int, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {
            "metric": "candidate_rows",
            "value": len(scored),
            "notes": "Rows in configs/crypto_universe_candidates.csv",
        },
        {
            "metric": "eligible_with_local_history",
            "value": int(scored["eligible"].sum()),
            "notes": "Symbols already passing local DB quality gates.",
        },
        {
            "metric": "pending_ingestion",
            "value": int((scored["selection_status"] == "pending_ingestion").sum()),
            "notes": "Candidate symbols not yet locally ingested.",
        },
        {
            "metric": "min_bars",
            "value": args.min_bars,
            "notes": "Coverage gate used for quality-gated selection.",
        },
        {
            "metric": "max_gap_hours",
            "value": args.max_gap_hours,
            "notes": "Maximum allowed OHLCV gap between adjacent hourly bars.",
        },
        {
            "metric": "lookback_days",
            "value": LOOKBACK_DAYS,
            "notes": "Configured ingestion lookback.",
        },
        {
            "metric": "interval",
            "value": INTERVAL,
            "notes": "Configured bar interval.",
        },
    ]
    for n, selected in selections.items():
        rows.append(
            {
                "metric": f"crypto{n}_rows",
                "value": len(selected),
                "notes": f"Selected rows for Crypto-{n} universe.",
            }
        )
        rows.append(
            {
                "metric": f"crypto{n}_eligible_now",
                "value": int((selected["selection_status"] == "eligible").sum()),
                "notes": "Selected symbols already passing local quality gates.",
            }
        )
    return pd.DataFrame(rows)


def write_outputs(scored: pd.DataFrame, args: argparse.Namespace) -> None:
    out_dir = Path(SAVE_DIR)
    out_dir.mkdir(exist_ok=True)

    selections: dict[int, pd.DataFrame] = {}
    selected_symbols: set[str] = set()
    for n in sorted(set(args.top_n)):
        selected = select_universe(scored, n, args.require_db_history)
        selections[n] = selected
        selected_symbols.update(selected["symbol"].tolist())
        selected.to_csv(out_dir / f"asset_universe_crypto{n}.csv", index=False)

    exclusions = build_exclusions(scored, selected_symbols)
    summary = build_summary(scored, selections, args)
    scored.to_csv(out_dir / "asset_universe_candidates_scored.csv", index=False)
    exclusions.to_csv(out_dir / "asset_universe_exclusions.csv", index=False)
    summary.to_csv(out_dir / "asset_universe_summary.csv", index=False)

    print("Multi-asset universe artifacts saved:")
    for n in sorted(selections):
        selected = selections[n]
        eligible = int((selected["selection_status"] == "eligible").sum())
        pending = int((selected["selection_status"] == "pending_ingestion").sum())
        print(f"  Crypto-{n}: {len(selected)} symbols | eligible now={eligible} | pending ingestion={pending}")
    print(f"  Scored candidates: {out_dir / 'asset_universe_candidates_scored.csv'}")
    print(f"  Exclusions: {out_dir / 'asset_universe_exclusions.csv'}")
    print(f"  Summary: {out_dir / 'asset_universe_summary.csv'}")
    print(f"  Protocol: {REPORT_PATH}")


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.candidate_path)
    scored = score_candidates(candidates, args)
    write_outputs(scored, args)


if __name__ == "__main__":
    main()
