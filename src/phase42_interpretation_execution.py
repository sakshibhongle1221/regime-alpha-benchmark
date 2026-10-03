from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_models import FEATURE_COLS, HORIZON_HOURS
from config import BASE_DIR, SAVE_DIR
from evaluation import portfolio_metrics, safe_corr
from fold_local_encoder_walkforward import build_common_frame
from universe import add_symbol_args, resolve_symbols


DEFAULT_THRESHOLDS = [0.0, 0.025, 0.05, 0.075]
DEFAULT_COST_BPS = [5.0, 10.0, 20.0, 40.0]
DEFAULT_PHASE41_PREDICTIONS = Path(SAVE_DIR) / "phase41_classical_alpha_oos_predictions.csv"
DEFAULT_REPAIRED_PREDICTIONS = Path(SAVE_DIR) / "crypto20_repaired_fold_local_alpha_oos_predictions.csv"
DEFAULT_REPAIRED_ASSIGNMENTS = Path(SAVE_DIR) / "crypto20_repaired_fold_local_regime_assignments.csv"
DEFAULT_OUTPUT_PREFIX = "phase42_"
DEFAULT_REPORT_PATH = Path(BASE_DIR) / "reports" / "phase42_interpretation_execution_hardening.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 42 interpretation and execution-hardening diagnostics."
    )
    add_symbol_args(parser)
    parser.add_argument("--phase41-predictions", default=str(DEFAULT_PHASE41_PREDICTIONS))
    parser.add_argument("--repaired-predictions", default=str(DEFAULT_REPAIRED_PREDICTIONS))
    parser.add_argument("--repaired-assignments", default=str(DEFAULT_REPAIRED_ASSIGNMENTS))
    parser.add_argument("--thresholds", nargs="*", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--cost-bps", nargs="*", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument(
        "--skip-feature-diagnostics",
        action="store_true",
        help="Skip feature-family target-alignment diagnostics for faster smoke checks.",
    )
    return parser.parse_args()


def output_path(prefix: str, stem: str, suffix: str = ".csv") -> Path:
    return Path(SAVE_DIR) / f"{prefix}{stem}{suffix}"


def feature_family(feature: str) -> str:
    if feature in {"amihud", "volume_zscore", "log_vol_trend"}:
        return "liquidity_volume"
    if feature in {"spread_proxy", "ofi_proxy"}:
        return "microstructure"
    if feature.startswith("ret_") or feature in {"macd_signal", "close_vs_vwap"}:
        return "momentum"
    if "vol" in feature or feature in {"atr_14", "gk_vol", "ret_dispersion"}:
        return "volatility"
    if feature in {"rsi_14", "bband_pct_b"}:
        return "technical_state"
    if feature in {"skewness", "kurtosis", "ret_autocorr"}:
        return "distribution_shape"
    return "other"


def load_predictions(path: Path, benchmark: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the corresponding repaired/Phase 41B experiment first."
        )
    required = [
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
        "signal",
    ]
    pred = pd.read_csv(path, usecols=required)
    pred["benchmark"] = benchmark
    return pred


def apply_score_threshold(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = pred.copy()
    probs = out[["prob_down", "prob_neutral", "prob_up"]].to_numpy(dtype=float)
    best_idx = probs.argmax(axis=1)
    score = pd.to_numeric(out["score"], errors="coerce").to_numpy(dtype=float)
    signal = np.zeros(len(out), dtype=int)
    signal[(score > threshold) & (best_idx != 1)] = 1
    signal[(score < -threshold) & (best_idx != 1)] = -1
    out["signal"] = signal
    out["threshold"] = float(threshold)
    return out


def stress_grid(predictions: pd.DataFrame, thresholds: list[float], cost_bps_values: list[float]) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["benchmark", "method"]
    for threshold in thresholds:
        thresholded = apply_score_threshold(predictions, threshold)
        for cost_bps in cost_bps_values:
            transaction_cost = float(cost_bps) / 10_000.0
            for (benchmark, method), group in thresholded.groupby(group_cols, sort=False):
                metrics = portfolio_metrics(
                    group,
                    horizon_hours=HORIZON_HOURS,
                    transaction_cost=transaction_cost,
                )
                metrics["signal_IC"] = safe_corr(group["signal"], group["target_return"])
                metrics["n_test_rows"] = int(len(group))
                rows.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "threshold": float(threshold),
                        "transaction_cost_bps": float(cost_bps),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def execution_summary(stress: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    default = stress[
        (stress["threshold"] == 0.0)
        & (stress["transaction_cost_bps"] == 10.0)
    ]
    high_cost = stress[
        (stress["threshold"] == 0.05)
        & (stress["transaction_cost_bps"] == 40.0)
    ]
    for (benchmark, method), group in stress.groupby(["benchmark", "method"], sort=False):
        default_row = default[
            (default["benchmark"] == benchmark) & (default["method"] == method)
        ]
        high_cost_row = high_cost[
            (high_cost["benchmark"] == benchmark) & (high_cost["method"] == method)
        ]
        rows.append(
            {
                "benchmark": benchmark,
                "method": method,
                "default_sharpe": float(default_row["Sharpe"].iloc[0]) if len(default_row) else np.nan,
                "default_total_return": (
                    float(default_row["total_return"].iloc[0]) if len(default_row) else np.nan
                ),
                "default_turnover": float(default_row["turnover"].iloc[0]) if len(default_row) else np.nan,
                "high_cost_threshold_sharpe": (
                    float(high_cost_row["Sharpe"].iloc[0]) if len(high_cost_row) else np.nan
                ),
                "high_cost_threshold_total_return": (
                    float(high_cost_row["total_return"].iloc[0]) if len(high_cost_row) else np.nan
                ),
                "best_sharpe": float(group["Sharpe"].max()),
                "worst_sharpe": float(group["Sharpe"].min()),
                "best_total_return": float(group["total_return"].max()),
                "worst_total_return": float(group["total_return"].min()),
                "positive_return_cells": int((group["total_return"] > 0).sum()),
                "stress_cells": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["benchmark", "default_sharpe"], ascending=[True, False])


def transition_entropy(labels: pd.Series) -> float:
    counts = labels.value_counts(normalize=True)
    if counts.empty:
        return np.nan
    entropy = -float((counts * np.log(counts + 1e-12)).sum())
    return entropy / np.log(max(len(counts), 2))


def transition_diagnostics(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    assignments = assignments.sort_values(["method", "fold", "symbol", "feat_idx"]).copy()
    for method, method_frame in assignments.groupby("method", sort=False):
        transition_flags = []
        durations = []
        for _, group in method_frame.groupby(["fold", "symbol"], sort=False):
            regimes = group["regime"].astype(int).to_numpy()
            if len(regimes) == 0:
                continue
            changes = np.r_[False, regimes[1:] != regimes[:-1]]
            transition_flags.extend(changes.tolist())
            run_length = 1
            for idx in range(1, len(regimes)):
                if regimes[idx] == regimes[idx - 1]:
                    run_length += 1
                else:
                    durations.append(run_length)
                    run_length = 1
            durations.append(run_length)
        confidence = method_frame[[c for c in method_frame.columns if c.startswith("post_")]].max(axis=1)
        rows.append(
            {
                "regime_method": method,
                "n_rows": int(len(method_frame)),
                "n_folds": int(method_frame["fold"].nunique()),
                "n_symbols": int(method_frame["symbol"].nunique()),
                "n_regimes": int(method_frame["regime"].nunique()),
                "switch_rate": float(np.mean(transition_flags)) if transition_flags else np.nan,
                "avg_duration": float(np.mean(durations)) if durations else np.nan,
                "median_duration": float(np.median(durations)) if durations else np.nan,
                "mean_confidence": float(confidence.mean()) if len(confidence) else np.nan,
                "regime_balance_entropy": transition_entropy(method_frame["regime"]),
            }
        )
    return pd.DataFrame(rows).sort_values("regime_method").reset_index(drop=True)


def stable_transition_alpha(predictions: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    assignment_parts = []
    work = assignments.sort_values(["method", "fold", "symbol", "feat_idx"]).copy()
    for _, group in work.groupby(["method", "fold", "symbol"], sort=False):
        group = group.copy()
        group["is_transition"] = group["regime"].astype(int).ne(group["regime"].astype(int).shift(1))
        assignment_parts.append(group[["method", "fold", "row_id", "regime", "is_transition"]])
    assignment_marks = pd.concat(assignment_parts, ignore_index=True)

    rows: list[dict] = []
    repaired = predictions[predictions["benchmark"] == "phase39r_repaired_neural"].copy()
    for regime_method in sorted(assignment_marks["method"].unique()):
        method = f"regime_lgbm_{regime_method}"
        pred = repaired[repaired["method"] == method]
        if pred.empty:
            continue
        marks = assignment_marks[assignment_marks["method"] == regime_method]
        merged = pred.merge(marks, on=["fold", "row_id"], how="inner")
        for bucket, group in merged.groupby("is_transition", sort=False):
            rows.append(
                {
                    "method": method,
                    "regime_method": regime_method,
                    "state_bucket": "transition" if bool(bucket) else "stable",
                    "n_rows": int(len(group)),
                    "mean_target_return": float(pd.to_numeric(group["target_return"], errors="coerce").mean()),
                    "mean_score": float(pd.to_numeric(group["score"], errors="coerce").mean()),
                    "ic": safe_corr(group["score"], group["target_return"]),
                    "signal_ic": safe_corr(group["signal"], group["target_return"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "state_bucket"]).reset_index(drop=True)


def alpha_diagnostics(asset_metrics: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict] = []
    for benchmark, frame in asset_metrics:
        for method, group in frame.groupby("method", sort=False):
            rows.append(
                {
                    "benchmark": benchmark,
                    "method": method,
                    "mean_asset_ic": float(group["IC"].mean()),
                    "median_asset_ic": float(group["IC"].median()),
                    "positive_ic_assets": int((group["IC"] > 0).sum()),
                    "n_assets": int(group["symbol"].nunique()),
                    "mean_sharpe_by_asset": float(group["Sharpe"].mean()),
                    "median_turnover_by_asset": float(group["turnover"].median()),
                    "mean_total_return_by_asset": float(group["total_return"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["benchmark", "mean_asset_ic"], ascending=[True, False])


def feature_family_diagnostics(symbols: list[str]) -> pd.DataFrame:
    frame = build_common_frame(symbols)
    rows: list[dict] = []
    for feature in FEATURE_COLS:
        asset_ics = []
        for _, group in frame.groupby("symbol", sort=False):
            corr = safe_corr(group[feature], group["target_return"])
            if np.isfinite(corr):
                asset_ics.append(corr)
        pooled = safe_corr(frame[feature], frame["target_return"])
        rows.append(
            {
                "feature": feature,
                "feature_family": feature_family(feature),
                "pooled_feature_target_ic": pooled,
                "mean_abs_asset_feature_target_ic": float(np.mean(np.abs(asset_ics))) if asset_ics else np.nan,
                "mean_asset_feature_target_ic": float(np.mean(asset_ics)) if asset_ics else np.nan,
                "positive_asset_ics": int(np.sum(np.array(asset_ics) > 0)) if asset_ics else 0,
                "n_assets": int(len(asset_ics)),
            }
        )
    feature_rows = pd.DataFrame(rows)
    family = (
        feature_rows.groupby("feature_family", as_index=False)
        .agg(
            n_features=("feature", "nunique"),
            mean_abs_asset_feature_target_ic=("mean_abs_asset_feature_target_ic", "mean"),
            median_abs_asset_feature_target_ic=("mean_abs_asset_feature_target_ic", "median"),
            mean_pooled_feature_target_ic=("pooled_feature_target_ic", "mean"),
            max_abs_asset_feature_target_ic=("mean_abs_asset_feature_target_ic", "max"),
        )
        .sort_values("mean_abs_asset_feature_target_ic", ascending=False)
        .reset_index(drop=True)
    )
    top_feature = (
        feature_rows.sort_values("mean_abs_asset_feature_target_ic", ascending=False)
        .groupby("feature_family", as_index=False)
        .head(1)[["feature_family", "feature", "mean_abs_asset_feature_target_ic"]]
        .rename(
            columns={
                "feature": "top_feature",
                "mean_abs_asset_feature_target_ic": "top_feature_mean_abs_asset_ic",
            }
        )
    )
    return family.merge(top_feature, on="feature_family", how="left")


def report_text(
    stress_summary: pd.DataFrame,
    transition: pd.DataFrame,
    stable_alpha: pd.DataFrame,
    alpha: pd.DataFrame,
    feature_family: pd.DataFrame | None,
) -> str:
    default_rows = "\n".join(
        f"| `{row.benchmark}` | `{row.method}` | {row.default_sharpe:.4f} | {row.default_total_return:.4f} | {row.default_turnover:.4f} | {row.positive_return_cells} / {row.stress_cells} |"
        for row in stress_summary.itertuples(index=False)
    )
    transition_rows = "\n".join(
        f"| `{row.regime_method}` | {row.switch_rate:.4f} | {row.avg_duration:.2f} | {row.mean_confidence:.4f} | {row.regime_balance_entropy:.4f} |"
        for row in transition.itertuples(index=False)
    )
    alpha_rows = "\n".join(
        f"| `{row.benchmark}` | `{row.method}` | {row.mean_asset_ic:.6f} | {row.positive_ic_assets} / {row.n_assets} | {row.mean_sharpe_by_asset:.4f} | {row.median_turnover_by_asset:.4f} |"
        for row in alpha.itertuples(index=False)
    )
    if feature_family is not None and not feature_family.empty:
        feature_rows = "\n".join(
            f"| `{row.feature_family}` | {row.n_features} | {row.mean_abs_asset_feature_target_ic:.6f} | `{row.top_feature}` |"
            for row in feature_family.itertuples(index=False)
        )
    else:
        feature_rows = "| skipped | 0 | nan | skipped |"

    transition_alpha_note = "not available"
    if not stable_alpha.empty:
        transition_alpha_note = (
            stable_alpha.groupby("state_bucket")["ic"].mean().to_string()
        )

    return f"""# Phase 42 Interpretation And Execution Hardening

## Status

This is a development-observed diagnostic phase. It does not tune models, does not modify locked/final test data, and does not create a new performance claim.

## What Phase 42 Does

Phase 42 explains why the repaired alpha conclusion remains weak by checking:

1. execution sensitivity across signal thresholds and transaction costs,
2. regime transition behavior and stable-vs-transition alpha,
3. cross-asset alpha fragility,
4. feature-family target alignment on the frozen Crypto-20 development panel.

## Execution Stress Summary

Default means threshold `0.0` and transaction cost `10 bps`. Positive cells count how many threshold/cost settings produced positive total return.

| Benchmark | Method | Default Sharpe | Default total return | Default turnover | Positive stress cells |
|---|---|---:|---:|---:|---:|
{default_rows}

## Regime Transition Diagnostics

| Regime method | Switch rate | Average duration | Mean confidence | Balance entropy |
|---|---:|---:|---:|---:|
{transition_rows}

Stable-vs-transition mean IC diagnostic:

```text
{transition_alpha_note}
```

## Cross-Asset Alpha Diagnostic

| Benchmark | Method | Mean asset IC | Positive IC assets | Mean asset Sharpe | Median turnover |
|---|---|---:|---:|---:|---:|
{alpha_rows}

## Feature-Family Target Alignment

This is not model superiority evidence. It is a descriptive check of which feature families have the strongest development-observed one-feature target alignment.

| Feature family | Features | Mean absolute asset IC | Top feature |
|---|---:|---:|---|
{feature_rows}

## Paper-Safe Interpretation

Phase 42 supports the current cautious paper story: the repaired pipeline is valid and informative, but the alpha layer remains fragile. The weak result is not explained away by one simple calibration fix. Execution assumptions, regime transitions, and cross-asset heterogeneity all matter.

Allowed wording:

```text
Phase 42 shows that the repaired alpha weakness is robust to execution diagnostics and is better treated as a modeling/market-structure limitation than as a single calibration bug.
```

Forbidden wording:

```text
Phase 42 proves the strategy is tradable.
Phase 42 rescues the alpha result.
Phase 42 should be used to tune a final-test candidate.
```
"""


def main() -> None:
    args = parse_args()
    symbols = resolve_symbols(args)
    phase41 = load_predictions(Path(args.phase41_predictions), "phase41b_classical_candidates")
    repaired = load_predictions(Path(args.repaired_predictions), "phase39r_repaired_neural")
    predictions = pd.concat([phase41, repaired], ignore_index=True)

    stress = stress_grid(predictions, args.thresholds, args.cost_bps)
    stress_summary = execution_summary(stress)

    assignments = pd.read_csv(args.repaired_assignments)
    transition = transition_diagnostics(assignments)
    stable_alpha = stable_transition_alpha(predictions, assignments)

    asset_metrics = [
        (
            "phase39r_repaired_neural",
            pd.read_csv(Path(SAVE_DIR) / "crypto20_repaired_fold_local_statistical_asset_metrics.csv"),
        ),
        (
            "phase41b_classical_candidates",
            pd.read_csv(Path(SAVE_DIR) / "phase41_classical_statistical_asset_metrics.csv"),
        ),
    ]
    alpha = alpha_diagnostics(asset_metrics)

    feature_family_table = None if args.skip_feature_diagnostics else feature_family_diagnostics(symbols)

    stress.to_csv(output_path(args.output_prefix, "execution_stress_results"), index=False)
    stress_summary.to_csv(output_path(args.output_prefix, "execution_stress_summary"), index=False)
    transition.to_csv(output_path(args.output_prefix, "regime_transition_diagnostics"), index=False)
    stable_alpha.to_csv(output_path(args.output_prefix, "stable_transition_alpha"), index=False)
    alpha.to_csv(output_path(args.output_prefix, "cross_asset_alpha_diagnostics"), index=False)
    if feature_family_table is not None:
        feature_family_table.to_csv(output_path(args.output_prefix, "feature_family_diagnostics"), index=False)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        report_text(stress_summary, transition, stable_alpha, alpha, feature_family_table),
        encoding="utf-8",
    )
    print("OK: Phase 42 interpretation and execution-hardening diagnostics complete.")


if __name__ == "__main__":
    main()
