import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_models import HORIZON_HOURS, TC_PER_TRADE
from evaluation import evaluation_metrics, non_overlapping_returns, safe_corr
from config import SAVE_DIR


RANDOM_STATE = 42
DEFAULT_PREDICTIONS = os.path.join(SAVE_DIR, "walkforward_alpha_oos_predictions.csv")
DEFAULT_EXPERIMENT_RESULTS = os.path.join(SAVE_DIR, "walkforward_experiment_results.csv")
REFERENCE_METHODS = ["global_lgbm", "regime_lgbm_hmm"]
CLASS_PROB_COLS = {-1: "prob_down", 0: "prob_neutral", 1: "prob_up"}


try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - fallback keeps the script usable in slim envs.
    scipy_stats = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 15A/15B statistical tests on fold-local OOS predictions."
    )
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--experiment-results", default=DEFAULT_EXPERIMENT_RESULTS)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--dm-lag", type=int, default=7, help="Newey-West lag for DM-style row-loss tests.")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix for generated artifacts, e.g. crypto20_ for Phase 37.",
    )
    parser.add_argument(
        "--reference-methods",
        nargs="+",
        default=REFERENCE_METHODS,
        help="Reference methods used in paired comparisons.",
    )
    return parser.parse_args()


def artifact_path(output_dir: Path, prefix: str, stem: str, suffix: str = ".csv") -> Path:
    return output_dir / f"{prefix}{stem}{suffix}"


def balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    recalls = []
    for label in sorted(pd.Series(y_true).dropna().unique()):
        mask = y_true == label
        if mask.sum() > 0:
            recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else np.nan


def add_net_returns(pred: pd.DataFrame) -> pd.DataFrame:
    return non_overlapping_returns(
        pred,
        horizon_hours=HORIZON_HOURS,
        transaction_cost=TC_PER_TRADE,
    )


def summarize_group(group: pd.DataFrame) -> dict:
    metrics = evaluation_metrics(
        group,
        horizon_hours=HORIZON_HOURS,
        transaction_cost=TC_PER_TRADE,
    )

    return {
        **metrics,
        "accuracy": float((group["target_label"] == group["pred_label"]).mean()),
        "balanced_accuracy": balanced_accuracy(group["target_label"], group["pred_label"]),
    }


def fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["method", "regime_method", "target", "horizon", "fold"]
    for key, group in predictions.groupby(group_cols, sort=False):
        method, regime_method, target, horizon, fold = key
        metrics = summarize_group(group)
        rows.append(
            {
                "method": method,
                "regime_method": regime_method,
                "target": target,
                "horizon": horizon,
                "fold": int(fold),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "fold"]).reset_index(drop=True)


def asset_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["method", "regime_method", "target", "horizon", "symbol"]
    for key, group in predictions.groupby(group_cols, sort=False):
        method, regime_method, target, horizon, symbol = key
        metrics = summarize_group(group)
        rows.append(
            {
                "method": method,
                "regime_method": regime_method,
                "target": target,
                "horizon": horizon,
                "symbol": symbol,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "symbol"]).reset_index(drop=True)


def bootstrap_ci(values: pd.Series, rng: np.random.Generator, samples: int) -> tuple[float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(arr, size=(samples, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def method_summary(folds: pd.DataFrame, experiment_results: pd.DataFrame, samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for method, group in folds.groupby("method", sort=False):
        row = {
            "method": method,
            "regime_method": str(group["regime_method"].iloc[0]),
            "target": str(group["target"].iloc[0]),
            "horizon": str(group["horizon"].iloc[0]),
            "n_folds": int(group["fold"].nunique()),
            "mean_fold_IC": float(group["IC"].mean()),
            "median_fold_IC": float(group["IC"].median()),
            "std_fold_IC": float(group["IC"].std(ddof=1)),
            "positive_ic_folds": int((group["IC"] > 0).sum()),
            "mean_fold_Sharpe": float(group["Sharpe"].mean()),
            "median_fold_Sharpe": float(group["Sharpe"].median()),
            "positive_sharpe_folds": int((group["Sharpe"] > 0).sum()),
            "mean_fold_total_return": float(group["total_return"].mean()),
            "median_fold_total_return": float(group["total_return"].median()),
        }
        for metric in ["IC", "Sharpe", "total_return"]:
            mean, low, high = bootstrap_ci(group[metric], rng, samples)
            row[f"{metric}_bootstrap_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        if not experiment_results.empty:
            match = experiment_results[experiment_results["method"] == method]
            if not match.empty:
                for col in ["IC", "Sharpe", "drawdown", "total_return", "n_test_rows"]:
                    row[f"full_sample_{col}"] = match.iloc[0][col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_fold_IC", ascending=False).reset_index(drop=True)


def normal_two_sided_p(statistic: float) -> float:
    if not np.isfinite(statistic):
        return np.nan
    return float(math.erfc(abs(statistic) / math.sqrt(2.0)))


def normal_cdf(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(0.5 * (1.0 + math.erf(value / math.sqrt(2.0))))


def paired_t_test(diffs: np.ndarray) -> tuple[float, float]:
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return np.nan, np.nan
    if np.isclose(diffs.std(ddof=1), 0):
        stat = np.inf if not np.isclose(diffs.mean(), 0) else 0.0
        p_value = 0.0 if np.isinf(stat) else 1.0
        return float(stat), float(p_value)
    if scipy_stats is not None:
        result = scipy_stats.ttest_1samp(diffs, popmean=0.0, nan_policy="omit")
        return float(result.statistic), float(result.pvalue)
    stat = float(diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs))))
    return stat, normal_two_sided_p(stat)


def cohen_dz(diffs: np.ndarray) -> float:
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return np.nan
    std = diffs.std(ddof=1)
    if np.isclose(std, 0):
        return np.nan
    return float(diffs.mean() / std)


def effect_size_label(value: float) -> str:
    if not np.isfinite(value):
        return "not_available"
    abs_value = abs(value)
    if abs_value < 0.2:
        return "tiny"
    if abs_value < 0.5:
        return "small"
    if abs_value < 0.8:
        return "medium"
    return "large"


def wilcoxon_p_value(diffs: np.ndarray) -> float:
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[~np.isclose(diffs, 0)]
    if len(diffs) < 2 or scipy_stats is None:
        return np.nan
    try:
        return float(scipy_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def sign_test_p_value(diffs: np.ndarray, higher_is_better: bool) -> tuple[int, int, float]:
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[~np.isclose(diffs, 0)]
    if len(diffs) == 0:
        return 0, 0, np.nan
    wins = int((diffs > 0).sum()) if higher_is_better else int((diffs < 0).sum())
    n = int(len(diffs))
    tail = sum(math.comb(n, k) for k in range(0, min(wins, n - wins) + 1)) / (2**n)
    return wins, n, float(min(1.0, 2.0 * tail))


def fold_pairwise_tests(
    folds: pd.DataFrame,
    samples: int,
    seed: int,
    reference_methods: list[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000)
    rows = []
    metrics = [
        ("IC", True),
        ("signal_IC", True),
        ("Sharpe", True),
        ("total_return", True),
        ("drawdown", True),
        ("turnover", False),
    ]
    seen_pairs: set[tuple[str, str]] = set()
    for reference in reference_methods:
        ref = folds[folds["method"] == reference]
        if ref.empty:
            continue
        for method in folds["method"].unique():
            if method == reference:
                continue
            pair_key = tuple(sorted((str(method), str(reference))))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            cur = folds[folds["method"] == method]
            paired = cur.merge(ref, on="fold", suffixes=("", "_reference"))
            for metric, higher_is_better in metrics:
                diffs = paired[metric].to_numpy(dtype=float) - paired[f"{metric}_reference"].to_numpy(dtype=float)
                stat, p_value = paired_t_test(diffs)
                wins, n_non_tie, sign_p = sign_test_p_value(diffs, higher_is_better)
                mean, low, high = bootstrap_ci(pd.Series(diffs), rng, samples)
                dz = cohen_dz(diffs)
                rows.append(
                    {
                        "comparison": f"{method} vs {reference}",
                        "method": method,
                        "reference_method": reference,
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        "mean_difference": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "cohen_dz": dz,
                        "effect_size": effect_size_label(dz),
                        "paired_t_stat": stat,
                        "paired_t_p_value": p_value,
                        "wilcoxon_p_value": wilcoxon_p_value(diffs),
                        "sign_test_wins": wins,
                        "sign_test_non_ties": n_non_tie,
                        "sign_test_p_value": sign_p,
                        "n_paired_folds": int(len(paired)),
                    }
                )
    return pd.DataFrame(rows)


def asset_pairwise_tests(
    assets: pd.DataFrame,
    samples: int,
    seed: int,
    reference_methods: list[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3000)
    rows = []
    metrics = [
        ("IC", True),
        ("signal_IC", True),
        ("Sharpe", True),
        ("total_return", True),
        ("drawdown", True),
        ("turnover", False),
    ]
    seen_pairs: set[tuple[str, str]] = set()
    for reference in reference_methods:
        ref = assets[assets["method"] == reference]
        if ref.empty:
            continue
        for method in assets["method"].unique():
            if method == reference:
                continue
            pair_key = tuple(sorted((str(method), str(reference))))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            cur = assets[assets["method"] == method]
            paired = cur.merge(ref, on="symbol", suffixes=("", "_reference"))
            for metric, higher_is_better in metrics:
                diffs = paired[metric].to_numpy(dtype=float) - paired[f"{metric}_reference"].to_numpy(dtype=float)
                stat, p_value = paired_t_test(diffs)
                wins, n_non_tie, sign_p = sign_test_p_value(diffs, higher_is_better)
                mean, low, high = bootstrap_ci(pd.Series(diffs), rng, samples)
                dz = cohen_dz(diffs)
                rows.append(
                    {
                        "comparison": f"{method} vs {reference}",
                        "method": method,
                        "reference_method": reference,
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        "mean_difference": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "cohen_dz": dz,
                        "effect_size": effect_size_label(dz),
                        "paired_t_stat": stat,
                        "paired_t_p_value": p_value,
                        "wilcoxon_p_value": wilcoxon_p_value(diffs),
                        "sign_test_wins": wins,
                        "sign_test_non_ties": n_non_tie,
                        "sign_test_p_value": sign_p,
                        "n_paired_assets": int(len(paired)),
                        "test_note": (
                            "Secondary paired asset diagnostic. Crypto assets are cross-correlated, "
                            "so asset-level p-values are descriptive and not the primary paper test."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def row_negative_log_likelihood(predictions: pd.DataFrame) -> pd.Series:
    prob = np.full(len(predictions), np.nan, dtype=float)
    labels = predictions["target_label"].to_numpy()
    for label, col in CLASS_PROB_COLS.items():
        mask = labels == label
        prob[mask] = predictions.loc[mask, col].to_numpy(dtype=float)
    return pd.Series(-np.log(np.clip(prob, 1e-12, 1.0)), index=predictions.index)


def newey_west_dm(loss_diff: np.ndarray, lag: int) -> tuple[float, float, float]:
    diff = loss_diff[np.isfinite(loss_diff)]
    n = len(diff)
    if n < 3:
        return np.nan, np.nan, np.nan
    mean_diff = float(diff.mean())
    centered = diff - mean_diff
    max_lag = min(max(lag, 0), n - 1)
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for k in range(1, max_lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - k / (max_lag + 1)
        variance += 2.0 * weight * gamma
    se = math.sqrt(max(variance / n, 1e-18))
    statistic = mean_diff / se
    return mean_diff, float(statistic), normal_two_sided_p(statistic)


def dm_style_loss_tests(
    predictions: pd.DataFrame,
    lag: int,
    reference_methods: list[str],
) -> pd.DataFrame:
    key_cols = ["row_id", "symbol", "feat_idx", "open_time", "fold"]
    work = predictions.copy()
    work["nll_loss"] = row_negative_log_likelihood(work)
    rows = []
    seen_pairs: set[tuple[str, str]] = set()
    for reference in reference_methods:
        ref = work[work["method"] == reference][key_cols + ["nll_loss"]].rename(columns={"nll_loss": "reference_loss"})
        if ref.empty:
            continue
        for method in work["method"].unique():
            if method == reference:
                continue
            pair_key = tuple(sorted((str(method), str(reference))))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            cur = work[work["method"] == method][key_cols + ["nll_loss"]]
            paired = cur.merge(ref, on=key_cols)
            loss_diff = paired["nll_loss"].to_numpy(dtype=float) - paired["reference_loss"].to_numpy(dtype=float)
            paired["loss_diff"] = loss_diff
            time_loss_diff = (
                paired.groupby("open_time", sort=True)["loss_diff"]
                .mean()
                .to_numpy(dtype=float)
            )
            mean_diff, dm_stat, p_value = newey_west_dm(time_loss_diff, lag=lag)
            rows.append(
                {
                    "comparison": f"{method} vs {reference}",
                    "method": method,
                    "reference_method": reference,
                    "metric": "nll_loss",
                    "higher_is_better": False,
                    "mean_difference": mean_diff,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "cohen_dz": np.nan,
                    "effect_size": "row_loss",
                    "paired_t_stat": np.nan,
                    "paired_t_p_value": np.nan,
                    "wilcoxon_p_value": np.nan,
                    "sign_test_wins": int((time_loss_diff < 0).sum()),
                    "sign_test_non_ties": int((~np.isclose(time_loss_diff, 0)).sum()),
                    "sign_test_p_value": np.nan,
                    "dm_stat": dm_stat,
                    "dm_p_value": p_value,
                    "n_paired_rows": int(len(paired)),
                    "n_time_blocks": int(len(time_loss_diff)),
                    "test_note": (
                        "Secondary Newey-West DM-style test on multiclass negative log-likelihood. "
                        "Loss differences are averaged across assets per timestamp before HAC correction."
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_tests(
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    samples: int,
    seed: int,
    dm_lag: int,
    reference_methods: list[str],
) -> pd.DataFrame:
    fold_tests = fold_pairwise_tests(
        folds,
        samples=samples,
        seed=seed,
        reference_methods=reference_methods,
    )
    fold_tests["dm_stat"] = np.nan
    fold_tests["dm_p_value"] = np.nan
    fold_tests["n_paired_rows"] = np.nan
    fold_tests["test_note"] = "Paired fold-level metric difference; bootstrap CI resamples folds."
    dm_tests = dm_style_loss_tests(
        predictions,
        lag=dm_lag,
        reference_methods=reference_methods,
    )
    out = pd.concat([fold_tests, dm_tests], ignore_index=True, sort=False)
    return out.sort_values(["reference_method", "method", "metric"]).reset_index(drop=True)


def significance_label(row: pd.Series) -> str:
    p_cols = ["paired_t_p_value", "dm_p_value", "wilcoxon_p_value"]
    p_values = [row[col] for col in p_cols if col in row and pd.notna(row[col])]
    if not p_values:
        return "not_tested"
    best_p = min(p_values)
    if best_p < 0.01:
        return "p<0.01"
    if best_p < 0.05:
        return "p<0.05"
    if best_p < 0.10:
        return "p<0.10"
    return "not_significant"


def test_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    focus = pairwise[pairwise["metric"].isin(["IC", "Sharpe", "nll_loss"])]
    for _, row in focus.iterrows():
        better = row["mean_difference"] > 0 if row["higher_is_better"] else row["mean_difference"] < 0
        rows.append(
            {
                "comparison": row["comparison"],
                "metric": row["metric"],
                "mean_difference": row["mean_difference"],
                "direction": "method_better" if better else "reference_better",
                "significance": significance_label(row),
                "paired_t_p_value": row.get("paired_t_p_value", np.nan),
                "dm_p_value": row.get("dm_p_value", np.nan),
                "n_paired_folds": row.get("n_paired_folds", np.nan),
                "n_paired_rows": row.get("n_paired_rows", np.nan),
            }
        )
    return pd.DataFrame(rows)


def adjust_benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return adjusted
    ranks = np.arange(1, m + 1, dtype=float)
    raw = valid.to_numpy(dtype=float) * m / ranks
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted.loc[valid.index] = np.clip(monotone, 0.0, 1.0)
    return adjusted


def adjust_holm(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return adjusted
    scaled = valid.to_numpy(dtype=float) * np.arange(m, 0, -1, dtype=float)
    monotone = np.maximum.accumulate(scaled)
    adjusted.loc[valid.index] = np.clip(monotone, 0.0, 1.0)
    return adjusted


def add_multiple_testing(pairwise: pd.DataFrame) -> pd.DataFrame:
    out = pairwise.copy()
    out["primary_p_value"] = out["paired_t_p_value"]
    out["p_value_source"] = "paired_t"
    loss_mask = out["metric"] == "nll_loss"
    out.loc[loss_mask, "primary_p_value"] = out.loc[loss_mask, "dm_p_value"]
    out.loc[loss_mask, "p_value_source"] = "dm_newey_west"

    out["bh_q_all_tests"] = adjust_benjamini_hochberg(out["primary_p_value"])
    out["holm_p_all_tests"] = adjust_holm(out["primary_p_value"])
    out["bh_q_by_metric"] = np.nan
    out["holm_p_by_metric"] = np.nan
    for _, idx in out.groupby("metric").groups.items():
        out.loc[idx, "bh_q_by_metric"] = adjust_benjamini_hochberg(out.loc[idx, "primary_p_value"])
        out.loc[idx, "holm_p_by_metric"] = adjust_holm(out.loc[idx, "primary_p_value"])

    out["raw_significant_05"] = out["primary_p_value"] < 0.05
    out["bh_metric_significant_05"] = out["bh_q_by_metric"] < 0.05
    out["holm_metric_significant_05"] = out["holm_p_by_metric"] < 0.05
    out["bh_all_significant_05"] = out["bh_q_all_tests"] < 0.05
    out["holm_all_significant_05"] = out["holm_p_all_tests"] < 0.05
    out["claim_status"] = out.apply(claim_status, axis=1)
    return out


def claim_status(row: pd.Series) -> str:
    if pd.isna(row.get("primary_p_value", np.nan)):
        return "not_tested"
    if bool(row.get("holm_all_significant_05", False)):
        return "survives_holm_all_tests"
    if bool(row.get("bh_all_significant_05", False)):
        return "survives_bh_all_tests"
    if bool(row.get("holm_metric_significant_05", False)):
        return "survives_holm_metric_family"
    if bool(row.get("bh_metric_significant_05", False)):
        return "survives_bh_metric_family"
    if bool(row.get("raw_significant_05", False)):
        return "raw_only_suggestive"
    return "not_significant"


def build_claims(corrected: pd.DataFrame) -> pd.DataFrame:
    focus = corrected[corrected["metric"].isin(["IC", "Sharpe", "nll_loss"])].copy()
    focus["direction"] = np.where(
        np.where(focus["higher_is_better"], focus["mean_difference"] > 0, focus["mean_difference"] < 0),
        "method_better",
        "reference_better",
    )
    focus["claim_status"] = focus.apply(claim_status, axis=1)
    columns = [
        "comparison",
        "metric",
        "direction",
        "mean_difference",
        "cohen_dz",
        "effect_size",
        "primary_p_value",
        "bh_q_by_metric",
        "holm_p_by_metric",
        "bh_q_all_tests",
        "holm_p_all_tests",
        "claim_status",
        "p_value_source",
        "n_paired_folds",
        "n_paired_rows",
    ]
    return focus[columns].sort_values(["metric", "claim_status", "primary_p_value"]).reset_index(drop=True)


def method_portfolio_returns(predictions: pd.DataFrame) -> dict[str, pd.Series]:
    returns = {}
    for method, group in predictions.groupby("method", sort=False):
        with_returns = add_net_returns(group)
        returns[method] = with_returns.groupby("open_time")["net_return"].mean().sort_index()
    return returns


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> dict:
    arr = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(arr)
    if n < 3:
        return {
            "n_periods": n,
            "mean_return": np.nan,
            "return_std": np.nan,
            "per_period_sharpe": np.nan,
            "annualized_sharpe": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "psr_gt_0": np.nan,
            "psr_stat": np.nan,
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    per_period_sr = mean / std if std > 0 else np.nan
    if scipy_stats is not None:
        skew = float(scipy_stats.skew(arr, bias=False))
        kurtosis = float(scipy_stats.kurtosis(arr, fisher=False, bias=False))
    else:
        series = pd.Series(arr)
        skew = float(series.skew())
        kurtosis = float(series.kurtosis() + 3.0)
    denominator = math.sqrt(max(1.0 - skew * per_period_sr + ((kurtosis - 1.0) / 4.0) * per_period_sr**2, 1e-12))
    psr_stat = (per_period_sr - benchmark_sr) * math.sqrt(n - 1.0) / denominator
    annualize = math.sqrt(8760 / HORIZON_HOURS)
    return {
        "n_periods": n,
        "mean_return": mean,
        "return_std": std,
        "per_period_sharpe": per_period_sr,
        "annualized_sharpe": per_period_sr * annualize,
        "skew": skew,
        "kurtosis": kurtosis,
        "psr_gt_0": normal_cdf(psr_stat),
        "psr_stat": psr_stat,
    }


def sharpe_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, returns in method_portfolio_returns(predictions).items():
        diagnostics = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
        diagnostics["method"] = method
        diagnostics["benchmark_sr"] = 0.0
        diagnostics["note"] = "Diagnostic PSR on overlapping portfolio returns; not a deployable performance claim."
        rows.append(diagnostics)
    return pd.DataFrame(rows).sort_values("psr_gt_0", ascending=False).reset_index(drop=True)


def plot_multiple_testing(claims: pd.DataFrame, output_path: Path, title_prefix: str = "Phase 15B") -> None:
    plot_df = claims[claims["metric"].isin(["IC", "Sharpe", "nll_loss"])].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values(["metric", "primary_p_value"]).reset_index(drop=True)
    labels = plot_df["comparison"].str.replace("regime_lgbm_", "", regex=False)
    labels = plot_df["metric"] + ": " + labels
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.24)))
    ax.scatter(plot_df["primary_p_value"], y, label="raw p-value", color="#2563EB", s=24)
    ax.scatter(plot_df["bh_q_by_metric"], y, label="BH q-value by metric", color="#F59E0B", s=24)
    ax.axvline(0.05, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("p or q value, log scale")
    ax.set_title(f"{title_prefix} - Multiple-Testing Correction")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_psr(diagnostics: pd.DataFrame, output_path: Path, title_prefix: str = "Phase 15B") -> None:
    if diagnostics.empty:
        return
    plot_df = diagnostics.sort_values("psr_gt_0", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot_df["method"], plot_df["psr_gt_0"], color="#10B981")
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilistic Sharpe Ratio: P(SR > 0)")
    ax.set_title(f"{title_prefix} - Probabilistic Sharpe Diagnostics")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ic_ci(summary: pd.DataFrame, output_path: Path, title_prefix: str = "Phase 15A") -> None:
    plot_df = summary.sort_values("IC_bootstrap_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(plot_df))
    means = plot_df["IC_bootstrap_mean"].to_numpy(dtype=float)
    low = plot_df["IC_ci_low"].to_numpy(dtype=float)
    high = plot_df["IC_ci_high"].to_numpy(dtype=float)
    ax.errorbar(
        means,
        y,
        xerr=[means - low, high - means],
        fmt="o",
        color="#2563EB",
        ecolor="#93C5FD",
        capsize=4,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["method"])
    ax.set_xlabel("Mean fold IC with 95% bootstrap CI")
    ax.set_title(f"{title_prefix} - Fold-Level IC Confidence Intervals")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def validate_predictions(predictions: pd.DataFrame) -> None:
    required = {
        "row_id",
        "symbol",
        "feat_idx",
        "open_time",
        "target_label",
        "target_return",
        "method",
        "regime_method",
        "target",
        "horizon",
        "fold",
        "prob_down",
        "prob_neutral",
        "prob_up",
        "score",
        "pred_label",
        "signal",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise SystemExit(f"Predictions file is missing required columns: {sorted(missing)}")


def main() -> None:
    args = parse_args()
    output_dir = Path(SAVE_DIR)
    output_dir.mkdir(exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    validate_predictions(predictions)
    experiment_results = pd.read_csv(args.experiment_results) if Path(args.experiment_results).exists() else pd.DataFrame()

    folds = fold_metrics(predictions)
    assets = asset_metrics(predictions)
    summary = method_summary(folds, experiment_results, samples=args.bootstrap_samples, seed=args.seed)
    pairwise = build_pairwise_tests(
        folds,
        predictions,
        samples=args.bootstrap_samples,
        seed=args.seed,
        dm_lag=args.dm_lag,
        reference_methods=args.reference_methods,
    )
    asset_pairwise = asset_pairwise_tests(
        assets,
        samples=args.bootstrap_samples,
        seed=args.seed,
        reference_methods=args.reference_methods,
    )
    compact = test_summary(pairwise)
    corrected = add_multiple_testing(pairwise)
    claims = build_claims(corrected)
    psr = sharpe_diagnostics(predictions)
    if args.output_prefix.startswith("crypto20_repaired_fold_local_"):
        title_prefix = "Phase 40 Repaired Crypto-20"
    elif args.output_prefix.startswith("crypto20_fold_local_"):
        title_prefix = "Phase 39 Fold-Local Crypto-20"
    elif args.output_prefix == "crypto20_":
        title_prefix = "Phase 37 Crypto-20"
    else:
        title_prefix = "Phase 15"

    folds.to_csv(artifact_path(output_dir, args.output_prefix, "statistical_fold_metrics"), index=False)
    assets.to_csv(artifact_path(output_dir, args.output_prefix, "statistical_asset_metrics"), index=False)
    summary.to_csv(artifact_path(output_dir, args.output_prefix, "statistical_method_summary"), index=False)
    pairwise.to_csv(artifact_path(output_dir, args.output_prefix, "statistical_pairwise_tests"), index=False)
    asset_pairwise.to_csv(
        artifact_path(output_dir, args.output_prefix, "statistical_asset_pairwise_tests"),
        index=False,
    )
    compact.to_csv(artifact_path(output_dir, args.output_prefix, "statistical_test_summary"), index=False)
    corrected.to_csv(artifact_path(output_dir, args.output_prefix, "statistical_multiple_testing"), index=False)
    claims.to_csv(artifact_path(output_dir, args.output_prefix, "statistical_claims"), index=False)
    psr.to_csv(artifact_path(output_dir, args.output_prefix, "statistical_sharpe_diagnostics"), index=False)
    plot_ic_ci(
        summary,
        artifact_path(output_dir, args.output_prefix, "statistical_ic_confidence_intervals", ".png"),
        title_prefix=title_prefix,
    )
    plot_multiple_testing(
        claims,
        artifact_path(output_dir, args.output_prefix, "statistical_multiple_testing", ".png"),
        title_prefix=title_prefix,
    )
    plot_psr(
        psr,
        artifact_path(output_dir, args.output_prefix, "statistical_sharpe_diagnostics", ".png"),
        title_prefix=title_prefix,
    )

    print("\nStatistical method summary:")
    display_cols = [
        "method",
        "mean_fold_IC",
        "IC_ci_low",
        "IC_ci_high",
        "positive_ic_folds",
        "mean_fold_Sharpe",
        "positive_sharpe_folds",
    ]
    print(summary[display_cols].to_string(index=False))

    print("\nFocused pairwise tests:")
    focused = compact[compact["metric"].isin(["IC", "Sharpe", "nll_loss"])]
    print(focused.to_string(index=False))

    print("\nSecondary asset-level diagnostics:")
    asset_focus = asset_pairwise[
        asset_pairwise["metric"].isin(["IC", "Sharpe", "total_return"])
        & asset_pairwise["method"].str.contains("hmm_guided_hmm", regex=False)
    ]
    asset_cols = [
        "comparison",
        "metric",
        "mean_difference",
        "ci_low",
        "ci_high",
        "sign_test_wins",
        "n_paired_assets",
    ]
    print(asset_focus[asset_cols].to_string(index=False))

    print("\nMultiple-testing claim summary:")
    claim_cols = [
        "comparison",
        "metric",
        "direction",
        "primary_p_value",
        "bh_q_by_metric",
        "holm_p_by_metric",
        "claim_status",
    ]
    print(claims[claim_cols].to_string(index=False))

    print("\nProbabilistic Sharpe diagnostics:")
    psr_cols = ["method", "annualized_sharpe", "psr_gt_0", "skew", "kurtosis", "n_periods"]
    print(psr[psr_cols].to_string(index=False))
    print("\nOK: statistical artifacts saved.")


if __name__ == "__main__":
    main()
