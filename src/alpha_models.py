import argparse
import os
import warnings
from dataclasses import dataclass

import duckdb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from config import DB_PATH, FEATURE_COLS, N_REGIMES, SAVE_DIR, SYMBOLS
from evaluation import (
    evaluation_metrics,
    non_overlapping_returns,
    portfolio_return_series,
)


warnings.filterwarnings("ignore", message="X does not have valid feature names")

PRIMARY_TARGET = "tb_label_8h"
PRIMARY_RETURN = "forward_return_8h"
HORIZON_HOURS = 8
INITIAL_TRAIN = 24 * 30 * 6
STEP_SIZE = 24 * 30
EMBARGO = 24 * 5
SCORE_THRESHOLD = 0.05
TC_PER_TRADE = 0.001
RANDOM_STATE = 42
REGIME_METHODS = ["contrastive", "contrastive_hmm", "hmm", "kmeans", "vol_bucket"]
LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {v: k for k, v in LABEL_TO_CLASS.items()}
POST_COLS = [f"post_{k}" for k in range(N_REGIMES)]


@dataclass
class PredictionSet:
    method: str
    regime_method: str
    frame: pd.DataFrame


def model_params() -> dict:
    return {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": 250,
        "learning_rate": 0.04,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 25,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }


def load_model_frame(symbols: list[str]) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    frames = []
    for symbol in symbols:
        df = con.execute(
            f"""
            SELECT
                f.open_time,
                f.symbol,
                {", ".join("f." + c for c in FEATURE_COLS)},
                t.{PRIMARY_TARGET} AS target_label,
                t.{PRIMARY_RETURN} AS target_return
            FROM features f
            JOIN targets t
              ON f.symbol = t.symbol
             AND f.open_time = t.open_time
            WHERE f.symbol = ?
            ORDER BY f.open_time
            """,
            [symbol],
        ).df()
        if df.empty:
            print(f"NOTE: no alpha modeling rows for {symbol}; skipping.")
            continue
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.reset_index(drop=True)
        df["feat_idx"] = df.index.astype(int)
        frames.append(df)
    con.close()

    if not frames:
        raise RuntimeError("No alpha modeling rows. Run targets.py first.")
    df = pd.concat(frames, ignore_index=True).sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["target_class"] = df["target_label"].map(LABEL_TO_CLASS).astype(int)
    return df


def load_assignments(symbols: list[str]) -> pd.DataFrame:
    path = os.path.join(SAVE_DIR, "regime_assignments.csv")
    if not os.path.exists(path):
        raise RuntimeError("regime_assignments.csv missing. Run baselines.py first.")

    assignments = pd.read_csv(path)
    assignments["open_time"] = pd.to_datetime(assignments["open_time"])
    assignments["feat_idx"] = assignments["feat_idx"].astype(int)
    assignments = assignments[assignments["symbol"].isin(symbols)].copy()

    missing = sorted(set(REGIME_METHODS) - set(assignments["method"].unique()))
    if missing:
        raise RuntimeError(f"Missing regime methods in regime_assignments.csv: {missing}")
    return assignments


def restrict_to_common_universe(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_counts = (
        assignments[["symbol", "feat_idx", "method"]]
        .drop_duplicates()
        .groupby(["symbol", "feat_idx"])["method"]
        .nunique()
        .reset_index(name="n_methods")
    )
    common_keys = method_counts[method_counts["n_methods"] == len(REGIME_METHODS)][
        ["symbol", "feat_idx"]
    ]
    if common_keys.empty:
        raise RuntimeError("No common symbol/feat_idx universe across all regime methods.")

    original_rows = len(df)
    df = df.merge(common_keys, on=["symbol", "feat_idx"], how="inner")
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=int)

    assignments = assignments.merge(common_keys, on=["symbol", "feat_idx"], how="inner")
    assignments = assignments.merge(df[["symbol", "feat_idx", "row_id"]], on=["symbol", "feat_idx"], how="inner")
    assignments = assignments.sort_values(["method", "symbol", "feat_idx"]).reset_index(drop=True)

    coverage = len(df) / original_rows
    if coverage < 0.90:
        raise RuntimeError(
            f"Common benchmark universe kept only {coverage:.1%} of rows. "
            "Rerun visualize_regimes.py to create dense contrastive assignments."
        )
    print(f"Common benchmark universe: {len(df):,}/{original_rows:,} rows ({coverage:.1%})")
    return df, assignments


def fold_ranges(df: pd.DataFrame) -> list[tuple[int, int, int]]:
    validate_calendar_aligned_panel(df)
    max_common_feat_idx = int(df.groupby("symbol")["feat_idx"].max().min())
    folds = []
    train_end = INITIAL_TRAIN
    while train_end + EMBARGO + STEP_SIZE <= max_common_feat_idx:
        test_start = train_end + EMBARGO
        test_end = min(test_start + STEP_SIZE, max_common_feat_idx + 1)
        folds.append((train_end, test_start, test_end))
        train_end += STEP_SIZE
    return folds


def validate_calendar_aligned_panel(df: pd.DataFrame) -> None:
    """Reject positional multi-asset folds whose indices represent different times."""
    required = {"symbol", "feat_idx", "open_time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cannot validate calendar alignment; missing columns: {missing}")
    if df.empty:
        raise ValueError("Cannot build folds from an empty frame.")

    timestamps_per_index = df.groupby("feat_idx")["open_time"].nunique()
    misaligned = timestamps_per_index[timestamps_per_index > 1]
    if not misaligned.empty:
        example_idx = int(misaligned.index[0])
        examples = (
            df.loc[df["feat_idx"] == example_idx, ["symbol", "open_time"]]
            .sort_values("symbol")
            .astype(str)
            .to_dict("records")
        )
        raise RuntimeError(
            "Unsafe multi-asset positional folds: the same feat_idx maps to different "
            f"calendar timestamps. Example feat_idx={example_idx}: {examples}. "
            "Align the assets to a common timestamp panel before creating folds."
        )


def align_to_common_calendar(
    df: pd.DataFrame, symbols: list[str], warmup_rows: int = 0
) -> pd.DataFrame:
    """Return a strict balanced panel with a shared timestamp-index mapping."""
    requested = list(dict.fromkeys(symbols))
    present = set(df["symbol"].astype(str).unique())
    missing_symbols = sorted(set(requested) - present)
    if missing_symbols:
        raise RuntimeError(f"Calendar alignment is missing symbols: {missing_symbols}")

    aligned = df[df["symbol"].isin(requested)].copy()
    aligned["source_feat_idx"] = aligned["feat_idx"].astype(int)
    common_times = (
        aligned.groupby("open_time")["symbol"]
        .nunique()
        .loc[lambda values: values == len(requested)]
        .index
    )
    aligned = aligned[aligned["open_time"].isin(common_times)].copy()
    if aligned.empty:
        raise RuntimeError("Requested symbols have no common calendar timestamps.")
    aligned = aligned.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    aligned["feat_idx"] = aligned.groupby("symbol").cumcount().astype(int)
    validate_calendar_aligned_panel(aligned)

    counts = aligned.groupby("symbol").size()
    if counts.nunique() != 1:
        raise RuntimeError(f"Common calendar produced unequal symbol rows: {counts.to_dict()}")
    if warmup_rows:
        aligned = aligned[aligned["feat_idx"] >= int(warmup_rows)].copy()
    aligned = aligned.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    aligned["row_id"] = np.arange(len(aligned), dtype=int)
    return aligned


def row_ids_for_fold(df: pd.DataFrame, train_end: int, test_start: int, test_end: int) -> tuple[list[int], list[int]]:
    train = df[df["feat_idx"] < train_end]["row_id"].astype(int).tolist()
    test = df[(df["feat_idx"] >= test_start) & (df["feat_idx"] < test_end)]["row_id"].astype(int).tolist()
    if train and test and "open_time" in df.columns:
        train_latest = pd.Timestamp(df.loc[df["row_id"].isin(train), "open_time"].max())
        test_earliest = pd.Timestamp(df.loc[df["row_id"].isin(test), "open_time"].min())
        if train_latest >= test_earliest:
            raise RuntimeError(
                "Calendar leakage detected: latest pooled training timestamp "
                f"{train_latest} is not earlier than earliest pooled test timestamp "
                f"{test_earliest}."
            )
    return train, test


def fit_lgbm(x_train: np.ndarray, y_train: np.ndarray):
    if len(np.unique(y_train)) < 2:
        return None
    model = lgb.LGBMClassifier(**model_params())
    model.fit(x_train, y_train)
    return model


def aligned_proba(model, x_test: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x_test)
    out = np.zeros((len(x_test), 3), dtype=float)
    for i, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, i]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.zeros_like(out), where=row_sum != 0)


def signal_from_probs(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = probs[:, LABEL_TO_CLASS[1]] - probs[:, LABEL_TO_CLASS[-1]]
    neutral_prob = probs[:, LABEL_TO_CLASS[0]]
    best_class = np.argmax(probs, axis=1)

    signal = np.zeros(len(probs), dtype=int)
    signal[(score > SCORE_THRESHOLD) & (best_class != LABEL_TO_CLASS[0])] = 1
    signal[(score < -SCORE_THRESHOLD) & (best_class != LABEL_TO_CLASS[0])] = -1

    pred_label = np.array([CLASS_TO_LABEL[int(cls)] for cls in best_class], dtype=int)
    pred_label[neutral_prob == probs.max(axis=1)] = 0
    pred_label[np.abs(score) < SCORE_THRESHOLD] = 0
    return score, pred_label, signal


def add_net_returns(pred: pd.DataFrame) -> pd.DataFrame:
    return non_overlapping_returns(
        pred,
        horizon_hours=HORIZON_HOURS,
        transaction_cost=TC_PER_TRADE,
    )


def portfolio_net_returns(pred: pd.DataFrame) -> pd.Series:
    return portfolio_return_series(add_net_returns(pred))


def build_prediction_frame(
    df_by_row: pd.DataFrame,
    row_ids: list[int],
    probs: np.ndarray,
    method: str,
    regime_method: str,
    fold: int,
) -> pd.DataFrame:
    score, pred_label, signal = signal_from_probs(probs)
    out = df_by_row.loc[
        row_ids,
        ["row_id", "symbol", "feat_idx", "open_time", "target_label", "target_return"],
    ].copy()
    out["method"] = method
    out["regime_method"] = regime_method
    out["target"] = PRIMARY_TARGET
    out["horizon"] = f"{HORIZON_HOURS}h"
    out["fold"] = fold
    out["prob_down"] = probs[:, LABEL_TO_CLASS[-1]]
    out["prob_neutral"] = probs[:, LABEL_TO_CLASS[0]]
    out["prob_up"] = probs[:, LABEL_TO_CLASS[1]]
    out["score"] = score
    out["pred_label"] = pred_label
    out["signal"] = signal
    return out


def run_global_model(df_by_row: pd.DataFrame, folds: list[tuple[int, int, int]]) -> PredictionSet:
    predictions = []
    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        if not train_ids or not test_ids:
            continue
        model = fit_lgbm(
            df_by_row.loc[train_ids, FEATURE_COLS].values,
            df_by_row.loc[train_ids, "target_class"].values,
        )
        if model is None:
            continue
        probs = aligned_proba(model, df_by_row.loc[test_ids, FEATURE_COLS].values)
        predictions.append(build_prediction_frame(df_by_row, test_ids, probs, "global_lgbm", "none", fold))

    return PredictionSet("global_lgbm", "none", pd.concat(predictions, ignore_index=True))


def train_regime_models(
    df_by_row: pd.DataFrame,
    assignment_df: pd.DataFrame,
    train_ids: list[int],
) -> dict[int, object]:
    models = {}
    train_assign = assignment_df[assignment_df["row_id"].isin(train_ids)]

    for regime, group in train_assign.groupby("regime"):
        row_ids = group["row_id"].astype(int).tolist()
        if len(row_ids) < 40:
            continue
        model = fit_lgbm(
            df_by_row.loc[row_ids, FEATURE_COLS].values,
            df_by_row.loc[row_ids, "target_class"].values,
        )
        if model is not None:
            models[int(regime)] = model
    return models


def predict_regime_models(
    df_by_row: pd.DataFrame,
    assignment_df: pd.DataFrame,
    models: dict[int, object],
    test_ids: list[int],
    method: str,
    fold: int,
) -> pd.DataFrame | None:
    test_assign = assignment_df[assignment_df["row_id"].isin(test_ids)].copy()
    if test_assign.empty or not models:
        return None

    test_assign = test_assign.sort_values(["symbol", "feat_idx"])
    row_ids = test_assign["row_id"].astype(int).tolist()
    x_test = df_by_row.loc[row_ids, FEATURE_COLS].values
    combined = np.zeros((len(row_ids), 3), dtype=float)
    weight_sum = np.zeros((len(row_ids), 1), dtype=float)

    for regime, model in models.items():
        probs = aligned_proba(model, x_test)
        col = f"post_{regime}"
        if col in test_assign.columns:
            weights = test_assign[col].to_numpy(dtype=float).reshape(-1, 1)
        else:
            weights = (test_assign["regime"].to_numpy(dtype=int) == regime).astype(float).reshape(-1, 1)
        combined += probs * weights
        weight_sum += weights

    combined = np.divide(
        combined,
        weight_sum,
        out=np.full_like(combined, 1.0 / 3.0),
        where=weight_sum != 0,
    )
    return build_prediction_frame(df_by_row, row_ids, combined, f"regime_lgbm_{method}", method, fold)


def run_regime_model(
    df_by_row: pd.DataFrame,
    assignments: pd.DataFrame,
    method: str,
    folds: list[tuple[int, int, int]],
) -> PredictionSet:
    method_assign = assignments[assignments["method"] == method].copy()
    predictions = []

    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        if not train_ids or not test_ids:
            continue
        models = train_regime_models(df_by_row, method_assign, train_ids)
        pred = predict_regime_models(df_by_row, method_assign, models, test_ids, method, fold)
        if pred is not None:
            predictions.append(pred)

    frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return PredictionSet(f"regime_lgbm_{method}", method, frame)


def summarize_predictions(pred: pd.DataFrame, method: str, regime_method: str, symbol_scope: str) -> dict:
    if pred.empty:
        return {
            "method": method,
            "target": PRIMARY_TARGET,
            "horizon": f"{HORIZON_HOURS}h",
            "regime_method": regime_method,
            "symbol_scope": symbol_scope,
            "IC": np.nan,
            "mean_asset_IC": np.nan,
            "median_asset_IC": np.nan,
            "cross_sectional_IC": np.nan,
            "median_cross_sectional_IC": np.nan,
            "pooled_IC": np.nan,
            "signal_IC": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "Sharpe": np.nan,
            "drawdown": np.nan,
            "turnover": np.nan,
            "total_return": np.nan,
            "n_trades": 0,
            "n_return_observations": 0,
            "n_test_rows": 0,
        }

    metrics = evaluation_metrics(
        pred,
        horizon_hours=HORIZON_HOURS,
        transaction_cost=TC_PER_TRADE,
    )

    return {
        "method": method,
        "target": PRIMARY_TARGET,
        "horizon": f"{HORIZON_HOURS}h",
        "regime_method": regime_method,
        "symbol_scope": symbol_scope,
        **metrics,
        "accuracy": float(accuracy_score(pred["target_label"], pred["pred_label"])),
        "balanced_accuracy": float(balanced_accuracy_score(pred["target_label"], pred["pred_label"])),
    }


def validate_test_coverage(results: pd.DataFrame) -> None:
    counts = results.set_index("method")["n_test_rows"]
    expected = int(counts.max())
    low = counts[counts < expected]
    if not low.empty:
        raise RuntimeError(
            "Alpha methods have different test coverage: "
            + ", ".join(f"{method}={int(rows)}" for method, rows in low.items())
            + f"; expected {expected}. Check regime assignment coverage."
        )


def save_equity_plot(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    for method, group in predictions.groupby("method"):
        curve = (1.0 + portfolio_net_returns(group)).cumprod()
        axes[0].plot(curve.values, label=method, linewidth=1.2)
    axes[0].axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    axes[0].set_title("Regime Alpha Benchmark - Experiment Equity Curves")
    axes[0].set_ylabel("Portfolio value")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    global_rows = predictions[predictions["method"] == "global_lgbm"].sort_values(["open_time", "symbol"])
    if not global_rows.empty:
        rolling_ic = global_rows["score"].rolling(200).corr(global_rows["target_return"])
        axes[1].plot(rolling_ic.values, color="#10B981", linewidth=1.0)
    axes[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
    axes[1].set_title("Rolling IC - Global Baseline")
    axes[1].set_ylabel("IC")
    axes[1].grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "equity_curve.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run purged alpha model benchmarks.")
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    args = parser.parse_args()

    symbol_scope = "+".join(args.symbols)
    df = load_model_frame(args.symbols)
    assignments = load_assignments(args.symbols)
    df, assignments = restrict_to_common_universe(df, assignments)
    df_by_row = df.set_index("row_id", drop=False)
    folds = fold_ranges(df)
    if not folds:
        raise RuntimeError("Not enough rows for walk-forward validation.")

    print(f"Rows: {len(df):,} | symbols: {symbol_scope} | folds: {len(folds)} | embargo: {EMBARGO} bars")

    outputs = [run_global_model(df_by_row, folds)]
    for method in REGIME_METHODS:
        outputs.append(run_regime_model(df_by_row, assignments, method, folds))

    predictions = pd.concat([o.frame for o in outputs if not o.frame.empty], ignore_index=True)
    predictions = predictions.sort_values(["method", "open_time", "symbol"]).reset_index(drop=True)
    predictions.to_csv(os.path.join(SAVE_DIR, "alpha_oos_predictions.csv"), index=False)
    predictions.to_csv(os.path.join(SAVE_DIR, "oos_predictions.csv"), index=False)

    summaries = [summarize_predictions(o.frame, o.method, o.regime_method, symbol_scope) for o in outputs]
    results = pd.DataFrame(summaries)
    validate_test_coverage(results)
    results.to_csv(os.path.join(SAVE_DIR, "experiment_results.csv"), index=False)
    results.to_csv(os.path.join(SAVE_DIR, "strategy_comparison.csv"), index=False)
    save_equity_plot(predictions)

    print("\nExperiment results:")
    print(results.to_string(index=False))
    print("\nOK: alpha benchmark artifacts saved.")


if __name__ == "__main__":
    main()
