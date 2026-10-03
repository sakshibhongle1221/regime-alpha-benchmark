import argparse
import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from alpha_models import HORIZON_HOURS, PRIMARY_TARGET, SAVE_DIR
from config import DB_PATH
from evaluation import (
    information_coefficients,
    non_overlapping_returns,
    portfolio_metrics_from_returns,
)


DEFAULT_THRESHOLDS = [0.03, 0.05, 0.07, 0.10]
DEFAULT_COST_BPS = [5, 10, 20]
MARKET_PERIODS = ["all", "bull", "sideways", "bear"]
CLASS_LABELS = np.array([-1, 0, 1], dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 14B stress robustness over costs, signal thresholds, and market periods."
    )
    parser.add_argument("--thresholds", nargs="*", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--cost-bps", nargs="*", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument(
        "--predictions",
        default=os.path.join(SAVE_DIR, "walkforward_alpha_oos_predictions.csv"),
        help="Fold-local row-level prediction CSV from walkforward_regimes.py.",
    )
    return parser.parse_args()


def load_predictions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing fold-local prediction file: {path}. "
            "Run python src/walkforward_regimes.py --symbols BTCUSDT ETHUSDT first."
        )
    pred = pd.read_csv(path)
    required = {
        "symbol",
        "open_time",
        "method",
        "regime_method",
        "target",
        "horizon",
        "target_label",
        "target_return",
        "prob_down",
        "prob_neutral",
        "prob_up",
    }
    missing = sorted(required - set(pred.columns))
    if missing:
        raise RuntimeError(f"Prediction file is missing required columns: {missing}")
    pred["open_time"] = pd.to_datetime(pred["open_time"])
    return pred.sort_values(["method", "symbol", "open_time"]).reset_index(drop=True)


def load_feature_returns(symbols: list[str]) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        features = con.execute(
            f"""
            SELECT symbol, open_time, ret_1h
            FROM features
            WHERE symbol IN ({",".join(["?"] * len(symbols))})
            ORDER BY symbol, open_time
            """,
            symbols,
        ).df()
    finally:
        con.close()

    if features.empty:
        raise RuntimeError("No feature returns found for market-period labeling.")
    features["open_time"] = pd.to_datetime(features["open_time"])
    features["ret_1h"] = features["ret_1h"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


def add_market_periods(pred: pd.DataFrame) -> pd.DataFrame:
    features = load_feature_returns(sorted(pred["symbol"].unique()))
    frames = []
    for _, group in features.groupby("symbol", sort=False):
        group = group.sort_values("open_time").copy()
        log_ret = np.log1p(group["ret_1h"].clip(lower=-0.99))
        group["rolling_30d_return"] = np.expm1(log_ret.rolling(24 * 30, min_periods=24 * 10).sum())
        frames.append(group)
    features = pd.concat(frames, ignore_index=True)

    valid = features["rolling_30d_return"].dropna()
    if valid.empty:
        raise RuntimeError("Could not compute rolling 30-day market period labels.")
    bear_cut = float(valid.quantile(0.25))
    bull_cut = float(valid.quantile(0.75))

    features["market_period"] = "sideways"
    features.loc[features["rolling_30d_return"] <= bear_cut, "market_period"] = "bear"
    features.loc[features["rolling_30d_return"] >= bull_cut, "market_period"] = "bull"

    out = pred.merge(
        features[["symbol", "open_time", "rolling_30d_return", "market_period"]],
        on=["symbol", "open_time"],
        how="left",
    )
    out["market_period"] = out["market_period"].fillna("sideways")
    out["rolling_30d_return"] = out["rolling_30d_return"].fillna(0.0)
    return out


def apply_threshold(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = pred.copy()
    probs = out[["prob_down", "prob_neutral", "prob_up"]].to_numpy(dtype=float)
    score = probs[:, 2] - probs[:, 0]
    best_idx = np.argmax(probs, axis=1)
    pred_label = CLASS_LABELS[best_idx].copy()
    pred_label[(best_idx == 1) | (np.abs(score) < threshold)] = 0

    signal = np.zeros(len(out), dtype=int)
    signal[(score > threshold) & (best_idx != 1)] = 1
    signal[(score < -threshold) & (best_idx != 1)] = -1

    out["threshold"] = threshold
    out["score"] = score
    out["pred_label"] = pred_label
    out["signal"] = signal
    return out


def add_net_returns(pred: pd.DataFrame, transaction_cost: float) -> pd.DataFrame:
    return non_overlapping_returns(
        pred,
        horizon_hours=HORIZON_HOURS,
        transaction_cost=transaction_cost,
    )


def summarize_subset(
    pred: pd.DataFrame,
    method: str,
    regime_method: str,
    market_period: str,
    threshold: float,
    cost_bps: float,
) -> dict:
    if market_period == "all":
        subset = pred.copy()
    else:
        subset = pred[pred["market_period"] == market_period].copy()

    base = {
        "symbol_scope": "+".join(sorted(pred["symbol"].unique())),
        "target": PRIMARY_TARGET,
        "horizon": f"{HORIZON_HOURS}h",
        "market_period": market_period,
        "threshold": threshold,
        "transaction_cost_bps": cost_bps,
        "method": method,
        "regime_method": regime_method,
    }
    if subset.empty:
        return {
            **base,
            "score_IC": np.nan,
            "signal_IC": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "Sharpe": np.nan,
            "drawdown": np.nan,
            "turnover": np.nan,
            "total_return": np.nan,
            "n_trades": 0,
            "n_test_rows": 0,
        }

    transaction_cost = cost_bps / 10000.0
    with_returns = add_net_returns(pred, transaction_cost)
    if market_period != "all":
        with_returns = with_returns[with_returns["market_period"] == market_period].copy()
    portfolio = portfolio_metrics_from_returns(with_returns, HORIZON_HOURS)
    ic = information_coefficients(subset)
    unique_labels = subset["target_label"].nunique()
    balanced = (
        balanced_accuracy_score(subset["target_label"], subset["pred_label"])
        if unique_labels > 1
        else np.nan
    )

    return {
        **base,
        "score_IC": ic["IC"],
        "mean_asset_IC": ic["mean_asset_IC"],
        "median_asset_IC": ic["median_asset_IC"],
        "cross_sectional_IC": ic["cross_sectional_IC"],
        "pooled_IC": ic["pooled_IC"],
        "signal_IC": ic["signal_IC"],
        "accuracy": float(accuracy_score(subset["target_label"], subset["pred_label"])),
        "balanced_accuracy": float(balanced) if pd.notna(balanced) else np.nan,
        **portfolio,
        "n_test_rows": int(len(subset)),
    }


def run_stress_grid(pred: pd.DataFrame, thresholds: list[float], cost_bps_values: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        thresholded = apply_threshold(pred, threshold)
        for cost_bps in cost_bps_values:
            for method, group in thresholded.groupby("method", sort=False):
                regime_method = str(group["regime_method"].iloc[0])
                for period in MARKET_PERIODS:
                    rows.append(
                        summarize_subset(
                            group,
                            method=method,
                            regime_method=regime_method,
                            market_period=period,
                            threshold=threshold,
                            cost_bps=cost_bps,
                        )
                    )
    return pd.DataFrame(rows)


def build_summary(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    group_cols = ["market_period", "threshold", "transaction_cost_bps"]
    for key, group in results.groupby(group_cols, sort=False):
        best_signal_ic = group.sort_values("signal_IC", ascending=False).iloc[0]
        best_sharpe = group.sort_values("Sharpe", ascending=False).iloc[0]
        lowest_drawdown = group.sort_values("drawdown", ascending=False).iloc[0]
        best_return = group.sort_values("total_return", ascending=False).iloc[0]
        rows.append(
            {
                "market_period": key[0],
                "threshold": key[1],
                "transaction_cost_bps": key[2],
                "best_signal_ic_method": best_signal_ic["method"],
                "best_signal_ic": best_signal_ic["signal_IC"],
                "best_sharpe_method": best_sharpe["method"],
                "best_sharpe": best_sharpe["Sharpe"],
                "lowest_drawdown_method": lowest_drawdown["method"],
                "lowest_drawdown": lowest_drawdown["drawdown"],
                "best_return_method": best_return["method"],
                "best_total_return": best_return["total_return"],
                "methods_tested": int(len(group)),
                "n_test_rows_per_method": int(group["n_test_rows"].max()),
            }
        )
    summary = pd.DataFrame(rows)

    wins = []
    metric_columns = [
        ("signal_IC", "best_signal_ic_method"),
        ("Sharpe", "best_sharpe_method"),
        ("drawdown", "lowest_drawdown_method"),
        ("total_return", "best_return_method"),
    ]
    for metric, column in metric_columns:
        for method, count in summary[column].value_counts().items():
            wins.append({"metric": metric, "method": method, "wins": int(count)})
    return summary, pd.DataFrame(wins)


def save_stress_plot(results: pd.DataFrame, wins: pd.DataFrame) -> None:
    if results.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    default = results[
        (results["market_period"] == "all")
        & (results["threshold"] == 0.05)
    ]
    if not default.empty:
        pivot = default.pivot_table(index="transaction_cost_bps", columns="method", values="Sharpe", aggfunc="mean")
        pivot.plot(ax=axes[0, 0], marker="o", linewidth=1.4)
        axes[0, 0].set_title("Sharpe Sensitivity To Transaction Costs")
        axes[0, 0].set_ylabel("Sharpe")
        axes[0, 0].grid(True, alpha=0.25)
        axes[0, 0].legend(fontsize=7)

    default_cost = results[
        (results["market_period"] == "all")
        & (results["transaction_cost_bps"] == 10)
    ]
    if not default_cost.empty:
        pivot = default_cost.pivot_table(index="threshold", columns="method", values="total_return", aggfunc="mean")
        pivot.plot(ax=axes[0, 1], marker="o", linewidth=1.4)
        axes[0, 1].set_title("Total Return Sensitivity To Signal Threshold")
        axes[0, 1].set_ylabel("Total return")
        axes[0, 1].grid(True, alpha=0.25)
        axes[0, 1].legend(fontsize=7)

    for ax, metric, title in [
        (axes[1, 0], "Sharpe", "Sharpe Win Counts"),
        (axes[1, 1], "drawdown", "Drawdown Win Counts"),
    ]:
        subset = wins[wins["metric"] == metric].sort_values("wins", ascending=True)
        ax.barh(subset["method"].str.replace("regime_lgbm_", "", regex=False), subset["wins"], color="#2563EB")
        ax.set_title(title)
        ax.set_xlabel("Wins")
        ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Regime Alpha Benchmark - Phase 14B Stress Robustness", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "robustness_stress_heatmap.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pred = load_predictions(args.predictions)
    pred = add_market_periods(pred)
    results = run_stress_grid(pred, args.thresholds, args.cost_bps)
    summary, wins = build_summary(results)

    results.to_csv(os.path.join(SAVE_DIR, "robustness_stress_results.csv"), index=False)
    summary.to_csv(os.path.join(SAVE_DIR, "robustness_stress_summary.csv"), index=False)
    wins.to_csv(os.path.join(SAVE_DIR, "robustness_stress_wins.csv"), index=False)
    save_stress_plot(results, wins)

    print("\nStress robustness summary:")
    print(summary.to_string(index=False))
    print("\nStress robustness win counts:")
    print(wins.to_string(index=False))
    print("\nOK: Phase 14B stress robustness artifacts saved.")


if __name__ == "__main__":
    main()
