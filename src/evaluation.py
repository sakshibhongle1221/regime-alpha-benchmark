from __future__ import annotations

import numpy as np
import pandas as pd


HOURS_PER_YEAR = 24 * 365
PRIMARY_REBALANCE_OFFSET_UTC = 0


def safe_corr(left: pd.Series, right: pd.Series) -> float:
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 3:
        return np.nan
    if left[valid].std(ddof=0) == 0 or right[valid].std(ddof=0) == 0:
        return np.nan
    return float(left[valid].corr(right[valid]))


def grouped_correlations(
    frame: pd.DataFrame,
    group_column: str,
    left_column: str,
    right_column: str,
    minimum_rows: int = 3,
) -> pd.Series:
    work = frame[[group_column, left_column, right_column]].copy()
    work[left_column] = pd.to_numeric(work[left_column], errors="coerce")
    work[right_column] = pd.to_numeric(work[right_column], errors="coerce")
    work = work.dropna(subset=[left_column, right_column])
    if work.empty:
        return pd.Series(dtype=float)
    grouped = work.groupby(group_column, sort=False)
    left_centered = work[left_column] - grouped[left_column].transform("mean")
    right_centered = work[right_column] - grouped[right_column].transform("mean")
    numerator = (left_centered * right_centered).groupby(work[group_column]).sum()
    left_ss = left_centered.pow(2).groupby(work[group_column]).sum()
    right_ss = right_centered.pow(2).groupby(work[group_column]).sum()
    denominator = np.sqrt(left_ss * right_ss)
    counts = grouped.size()
    correlations = numerator / denominator.replace(0, np.nan)
    return correlations[counts >= int(minimum_rows)].dropna()


def information_coefficients(pred: pd.DataFrame) -> dict[str, float | int]:
    """Separate time-series, cross-sectional, and pooled IC definitions."""
    if pred.empty:
        return {
            "IC": np.nan,
            "mean_asset_IC": np.nan,
            "median_asset_IC": np.nan,
            "cross_sectional_IC": np.nan,
            "median_cross_sectional_IC": np.nan,
            "pooled_IC": np.nan,
            "signal_IC": np.nan,
            "n_asset_IC": 0,
            "n_cross_sectional_IC": 0,
        }

    asset_ics = grouped_correlations(pred, "symbol", "score", "target_return")
    asset_signal_ics = grouped_correlations(pred, "symbol", "signal", "target_return")
    cross_sectional = grouped_correlations(
        pred, "open_time", "score", "target_return", minimum_rows=3
    )
    mean_asset = float(asset_ics.mean()) if len(asset_ics) else np.nan
    return {
        "IC": mean_asset,
        "mean_asset_IC": mean_asset,
        "median_asset_IC": float(asset_ics.median()) if len(asset_ics) else np.nan,
        "cross_sectional_IC": float(cross_sectional.mean()) if len(cross_sectional) else np.nan,
        "median_cross_sectional_IC": (
            float(cross_sectional.median()) if len(cross_sectional) else np.nan
        ),
        "pooled_IC": safe_corr(pred["score"], pred["target_return"]),
        "signal_IC": (
            float(asset_signal_ics.mean()) if len(asset_signal_ics) else np.nan
        ),
        "n_asset_IC": int(len(asset_ics)),
        "n_cross_sectional_IC": int(len(cross_sectional)),
    }


def non_overlapping_returns(
    pred: pd.DataFrame,
    horizon_hours: int,
    transaction_cost: float,
    rebalance_offset_utc: int = PRIMARY_REBALANCE_OFFSET_UTC,
) -> pd.DataFrame:
    """Evaluate one pre-specified UTC rebalance grid with non-overlapping labels."""
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be positive.")
    required = {"symbol", "open_time", "signal", "target_return"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"Return evaluation is missing columns: {missing}")
    if pred.empty:
        out = pred.copy()
        out["execution_signal"] = pd.Series(dtype=float)
        out["trade"] = pd.Series(dtype=float)
        out["net_return"] = pd.Series(dtype=float)
        return out

    out = pred.copy()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    grouping = ["symbol"] + (["fold"] if "fold" in out.columns else [])
    out = out.sort_values([*grouping, "open_time"]).reset_index(drop=True)
    out["execution_signal"] = (
        out.groupby(grouping, sort=False)["signal"].shift(1).fillna(0).astype(float)
    )
    utc = out["open_time"]
    epoch_hours = (
        (utc - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() // 3_600
    ).astype("int64")
    cohort = np.mod(epoch_hours, horizon_hours)
    out = out[cohort == int(rebalance_offset_utc) % horizon_hours].copy()
    if out.empty:
        raise RuntimeError("The selected non-overlapping rebalance grid has no rows.")

    previous = out.groupby(grouping, sort=False)["execution_signal"].shift(1).fillna(0)
    out["trade"] = (out["execution_signal"] - previous).abs() / 2.0
    out["net_return"] = (
        out["execution_signal"] * pd.to_numeric(out["target_return"], errors="coerce")
        - out["trade"] * float(transaction_cost)
    )

    for key, group in out.groupby(grouping, sort=False):
        gaps = group["open_time"].sort_values().diff().dropna()
        if len(gaps) and not (gaps >= pd.Timedelta(hours=horizon_hours)).all():
            raise RuntimeError(f"Overlapping return intervals remain for {key}.")
    return out.sort_values(["open_time", "symbol"]).reset_index(drop=True)


def portfolio_return_series(with_returns: pd.DataFrame) -> pd.Series:
    if with_returns.empty:
        return pd.Series(dtype=float)
    return with_returns.groupby("open_time")["net_return"].mean().sort_index()


def portfolio_metrics(
    pred: pd.DataFrame,
    horizon_hours: int,
    transaction_cost: float,
    rebalance_offset_utc: int = PRIMARY_REBALANCE_OFFSET_UTC,
) -> dict[str, float | int]:
    with_returns = non_overlapping_returns(
        pred,
        horizon_hours=horizon_hours,
        transaction_cost=transaction_cost,
        rebalance_offset_utc=rebalance_offset_utc,
    )
    return portfolio_metrics_from_returns(
        with_returns,
        horizon_hours=horizon_hours,
        rebalance_offset_utc=rebalance_offset_utc,
    )


def portfolio_metrics_from_returns(
    with_returns: pd.DataFrame,
    horizon_hours: int,
    rebalance_offset_utc: int = PRIMARY_REBALANCE_OFFSET_UTC,
) -> dict[str, float | int]:
    returns = portfolio_return_series(with_returns)
    cumulative = (1.0 + returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    std = returns.std()
    annualize = np.sqrt(HOURS_PER_YEAR / horizon_hours)
    return {
        "Sharpe": (
            float(returns.mean() / (std + 1e-8) * annualize)
            if len(returns)
            else np.nan
        ),
        "drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "turnover": float(with_returns["trade"].mean()) if len(with_returns) else np.nan,
        "total_return": float(cumulative.iloc[-1] - 1.0) if len(cumulative) else np.nan,
        "n_trades": int((with_returns["trade"] > 0).sum()),
        "n_return_observations": int(len(returns)),
        "rebalance_offset_utc": int(rebalance_offset_utc) % horizon_hours,
    }


def evaluation_metrics(
    pred: pd.DataFrame,
    horizon_hours: int,
    transaction_cost: float,
    rebalance_offset_utc: int = PRIMARY_REBALANCE_OFFSET_UTC,
) -> dict[str, float | int]:
    return {
        **information_coefficients(pred),
        **portfolio_metrics(
            pred,
            horizon_hours=horizon_hours,
            transaction_cost=transaction_cost,
            rebalance_offset_utc=rebalance_offset_utc,
        ),
        "n_test_rows": int(len(pred)),
    }
