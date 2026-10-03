import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SAVE_DIR
from alpha_models import HORIZON_HOURS, TC_PER_TRADE
from evaluation import non_overlapping_returns, portfolio_return_series


def load_predictions() -> pd.DataFrame:
    path = os.path.join(SAVE_DIR, "alpha_oos_predictions.csv")
    if not os.path.exists(path):
        raise RuntimeError("alpha_oos_predictions.csv missing. Run alpha_models.py first.")
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"])
    return df


def method_curves(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in predictions.groupby("method"):
        method_df = non_overlapping_returns(group, HORIZON_HOURS, TC_PER_TRADE)
        curve = portfolio_return_series(method_df).rename("net_return").reset_index()
        curve["method"] = method
        curve["equity"] = (1.0 + curve["net_return"]).cumprod()
        curve["drawdown"] = (curve["equity"] - curve["equity"].cummax()) / curve["equity"].cummax()
        rows.append(curve)
    return pd.concat(rows, ignore_index=True)


def save_dashboard_plot(curves: pd.DataFrame, results: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, :])
    for method, group in curves.groupby("method"):
        ax1.plot(group["equity"].values, label=method, linewidth=1.2)
    ax1.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    ax1.set_title("Regime Alpha Benchmark - Equity Curves")
    ax1.set_ylabel("Portfolio value")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 0])
    for method, group in curves.groupby("method"):
        ax2.plot(group["drawdown"].values, label=method, linewidth=1.0)
    ax2.set_title("Drawdown by Method")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 1])
    if not results.empty:
        ordered = results.sort_values("IC", ascending=False)
        ax3.barh(ordered["method"], ordered["IC"], color="#3B82F6")
    ax3.axvline(0, color="black", linewidth=0.6)
    ax3.set_title("Information Coefficient")
    ax3.grid(True, axis="x", alpha=0.25)

    ax4 = fig.add_subplot(gs[2, 0])
    if not results.empty:
        ordered = results.sort_values("Sharpe", ascending=False)
        ax4.barh(ordered["method"], ordered["Sharpe"], color="#10B981")
    ax4.axvline(0, color="black", linewidth=0.6)
    ax4.set_title("Sharpe by Method")
    ax4.grid(True, axis="x", alpha=0.25)

    ax5 = fig.add_subplot(gs[2, 1])
    predictions = load_predictions()
    global_rows = predictions[predictions["method"] == "global_lgbm"].sort_values(["open_time", "symbol"])
    if not global_rows.empty:
        by_time = global_rows.groupby("open_time", as_index=False)[["score", "target_return"]].mean()
        rolling_ic = by_time["score"].rolling(200).corr(by_time["target_return"])
        ax5.plot(rolling_ic.values, color="#F59E0B", linewidth=1.0)
    ax5.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax5.set_title("Rolling IC - Global Baseline")
    ax5.grid(True, alpha=0.25)

    plt.suptitle("Regime Alpha Benchmark - Research Backtest Dashboard", fontsize=14, y=0.99)
    plt.savefig(os.path.join(SAVE_DIR, "phase4_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    predictions = load_predictions()
    curves = method_curves(predictions)
    curves.to_csv(os.path.join(SAVE_DIR, "backtest_curves.csv"), index=False)

    result_path = os.path.join(SAVE_DIR, "experiment_results.csv")
    results = pd.read_csv(result_path) if os.path.exists(result_path) else pd.DataFrame()
    save_dashboard_plot(curves, results)

    print("Backtest methods:")
    if not results.empty:
        print(results.to_string(index=False))
    else:
        print(curves["method"].value_counts().to_string())
    print("\nOK: backtest_curves.csv and phase4_dashboard.png saved.")


if __name__ == "__main__":
    main()
