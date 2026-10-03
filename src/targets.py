import argparse
import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DB_PATH, SAVE_DIR
from universe import add_symbol_args, resolve_symbols


HORIZONS = [4, 8, 24]
PRIMARY_HORIZON = 8
DIR_THRESHOLD = 0.25
BARRIER_MULT = 1.0


def load_symbol_frame(con: duckdb.DuckDBPyConnection, symbol: str) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT
            f.open_time,
            f.symbol,
            f.ret_1h,
            f.vol_20h,
            o.close
        FROM features f
        JOIN ohlcv o
          ON f.symbol = o.symbol
         AND f.open_time = o.open_time
        WHERE f.symbol = ?
        ORDER BY f.open_time
        """,
        [symbol],
    ).df()
    df["open_time"] = pd.to_datetime(df["open_time"])
    return df


def triple_barrier_labels(
    close: pd.Series,
    rolling_vol: pd.Series,
    horizon: int,
    barrier_mult: float = BARRIER_MULT,
) -> pd.Series:
    labels = np.full(len(close), np.nan)
    close_values = close.to_numpy(dtype=float)
    vol_values = rolling_vol.to_numpy(dtype=float)

    for i in range(len(close_values) - horizon):
        start_price = close_values[i]
        vol = vol_values[i]
        if not np.isfinite(start_price) or not np.isfinite(vol) or vol <= 0:
            continue

        up = barrier_mult * vol * np.sqrt(horizon)
        down = -barrier_mult * vol * np.sqrt(horizon)
        path_returns = close_values[i + 1 : i + horizon + 1] / start_price - 1.0

        label = 0
        for ret in path_returns:
            if ret >= up:
                label = 1
                break
            if ret <= down:
                label = -1
                break
        labels[i] = label

    return pd.Series(labels, index=close.index)


def build_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = df[["open_time", "symbol"]].copy()
    rolling_vol = df["vol_20h"].replace([np.inf, -np.inf], np.nan)

    for horizon in horizons:
        fwd = df["close"].shift(-horizon) / df["close"] - 1.0
        vol_scale = rolling_vol * np.sqrt(horizon) + 1e-8
        vol_adj = fwd / vol_scale

        out[f"forward_return_{horizon}h"] = fwd
        out[f"vol_adj_return_{horizon}h"] = vol_adj
        out[f"dir_label_{horizon}h"] = np.select(
            [vol_adj > DIR_THRESHOLD, vol_adj < -DIR_THRESHOLD],
            [1, -1],
            default=0,
        )
        out[f"tb_label_{horizon}h"] = triple_barrier_labels(
            df["close"], rolling_vol, horizon
        )

    target_cols = [c for c in out.columns if c not in {"open_time", "symbol"}]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=target_cols)

    for col in out.columns:
        if col.startswith("dir_label_") or col.startswith("tb_label_"):
            out[col] = out[col].astype(int)

    return out


def target_distribution(targets: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for symbol, symbol_df in targets.groupby("symbol"):
        for horizon in horizons:
            for label_type in ["dir", "tb"]:
                col = f"{label_type}_label_{horizon}h"
                counts = symbol_df[col].value_counts().sort_index()
                total = counts.sum()
                for label, count in counts.items():
                    rows.append(
                        {
                            "symbol": symbol,
                            "horizon": f"{horizon}h",
                            "label_type": label_type,
                            "label": int(label),
                            "count": int(count),
                            "pct": float(count / total),
                        }
                    )
    return pd.DataFrame(rows)


def target_quality_rows(
    symbol: str,
    source_rows: int,
    targets: pd.DataFrame,
    horizons: list[int],
) -> list[dict]:
    rows = []
    target_rows = len(targets)
    for horizon in horizons:
        for label_type in ["dir", "tb"]:
            col = f"{label_type}_label_{horizon}h"
            counts = targets[col].value_counts(normalize=True)
            rows.append(
                {
                    "symbol": symbol,
                    "horizon": f"{horizon}h",
                    "label_type": label_type,
                    "source_rows": int(source_rows),
                    "target_rows": int(target_rows),
                    "missing_rows": int(source_rows - target_rows),
                    "expected_horizon_loss": int(max(horizons)),
                    "neutral_pct": float(counts.get(0, 0.0)),
                    "down_pct": float(counts.get(-1, 0.0)),
                    "up_pct": float(counts.get(1, 0.0)),
                }
            )
    return rows


def save_distribution_plot(dist: pd.DataFrame) -> None:
    if dist.empty:
        return

    plot_df = dist.copy()
    plot_df["bucket"] = (
        plot_df["symbol"] + " " + plot_df["label_type"] + " " + plot_df["horizon"]
    )
    pivot = (
        plot_df.pivot_table(index="bucket", columns="label", values="pct", fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_title("Target Label Distribution")
    ax.set_ylabel("Share of rows")
    ax.set_xlabel("")
    ax.legend(title="Label", loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "target_distribution.png"), dpi=150)
    plt.close(fig)


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


def save_targets(
    con: duckdb.DuckDBPyConnection,
    targets: pd.DataFrame,
    symbols: list[str],
    append: bool = False,
) -> None:
    con.register("target_data", targets)
    if append and table_exists(con, "targets"):
        con.execute(
            f"DELETE FROM targets WHERE symbol IN ({','.join(['?'] * len(symbols))})",
            symbols,
        )
        con.execute("INSERT INTO targets SELECT * FROM target_data")
    else:
        con.execute("DROP TABLE IF EXISTS targets")
        con.execute("CREATE TABLE targets AS SELECT * FROM target_data")
    con.unregister("target_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-horizon financial labels.")
    add_symbol_args(parser)
    parser.add_argument(
        "--artifact-prefix",
        default="",
        help="Optional prefix for target diagnostic artifacts, e.g. crypto20_.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Replace targets only for selected symbols instead of rebuilding the whole targets table.",
    )
    args = parser.parse_args()
    symbols = resolve_symbols(args)

    con = duckdb.connect(DB_PATH)
    frames = []
    quality = []

    for symbol in symbols:
        print(f"Building targets for {symbol}...")
        raw = load_symbol_frame(con, symbol)
        if raw.empty:
            print(f"  skipped: no feature/price rows found")
            continue
        built = build_targets(raw, HORIZONS)
        frames.append(built)
        quality.extend(target_quality_rows(symbol, len(raw), built, HORIZONS))
        print(f"  rows: {len(built):,}")

    if not frames:
        con.close()
        raise RuntimeError("No targets were built. Run ingestion.py and features.py first.")

    targets = pd.concat(frames, ignore_index=True)
    save_targets(con, targets, symbols, append=args.append)
    con.close()

    dist = target_distribution(targets, HORIZONS)
    prefix = args.artifact_prefix
    dist.to_csv(os.path.join(SAVE_DIR, f"{prefix}target_distribution.csv"), index=False)
    pd.DataFrame(quality).to_csv(os.path.join(SAVE_DIR, f"{prefix}target_quality.csv"), index=False)
    if prefix:
        plot_path = os.path.join(SAVE_DIR, f"{prefix}target_distribution.png")
        plt.figure(figsize=(14, 7))
        plot_df = dist.copy()
        plot_df["name"] = (
            plot_df["symbol"] + " " + plot_df["label_type"] + " " + plot_df["horizon"].astype(str)
        )
        pivot = plot_df.pivot_table(index="name", columns="label", values="pct", fill_value=0)
        pivot.plot(kind="bar", stacked=True, colormap="viridis", figsize=(14, 7))
        plt.title("Target Label Distribution")
        plt.ylabel("Share of rows")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
    else:
        save_distribution_plot(dist)

    primary_col = f"tb_label_{PRIMARY_HORIZON}h"
    print("\nPrimary target distribution:")
    print(
        targets.groupby("symbol")[primary_col]
        .value_counts(normalize=True)
        .rename("pct")
        .reset_index()
        .to_string(index=False)
    )
    print(f"\nOK: targets table saved with {len(targets):,} rows.")


if __name__ == "__main__":
    main()
