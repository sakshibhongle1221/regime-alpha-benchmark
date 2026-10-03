import argparse
from pathlib import Path

import pandas as pd

from config import SAVE_DIR, SYMBOLS


def universe_path(name: str) -> Path:
    return Path(SAVE_DIR) / f"asset_universe_{name}.csv"


def load_universe_symbols(
    universe: str,
    path: str | None = None,
    statuses: set[str] | None = None,
) -> list[str]:
    source = Path(path) if path else universe_path(universe)
    if not source.exists():
        raise FileNotFoundError(f"Universe file not found: {source}")

    df = pd.read_csv(source)
    if "symbol" not in df.columns:
        raise ValueError(f"Universe file must contain a symbol column: {source}")
    if statuses is not None and "selection_status" in df.columns:
        df = df[df["selection_status"].isin(statuses)].copy()

    symbols = df["symbol"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    if not symbols:
        raise ValueError(f"No symbols resolved from universe file: {source}")
    return symbols


def add_symbol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbols", nargs="*", default=None, help="Explicit symbols to process.")
    parser.add_argument(
        "--universe",
        default=None,
        help="Universe name from models/asset_universe_<name>.csv, e.g. crypto20 or crypto50.",
    )
    parser.add_argument("--universe-path", default=None, help="Explicit universe CSV path.")
    parser.add_argument(
        "--eligible-only",
        action="store_true",
        help="Only use symbols already marked eligible in the universe artifact.",
    )


def resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [symbol.upper() for symbol in args.symbols]
    if args.universe or args.universe_path:
        statuses = {"eligible"} if args.eligible_only else None
        return load_universe_symbols(args.universe or "crypto20", args.universe_path, statuses)
    return list(SYMBOLS)
