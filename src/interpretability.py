import argparse
import os
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_models import (
    FEATURE_COLS,
    N_REGIMES,
    PRIMARY_TARGET,
    RANDOM_STATE,
    SAVE_DIR,
    SYMBOLS,
    fit_lgbm,
    fold_ranges,
    row_ids_for_fold,
)
from baselines import HMM_FEATURES
from walkforward_regimes import (
    finite_matrix,
    fit_hmm_assignments,
    load_dense_contrastive_universe,
    load_guided_embedding_matrix,
)


TARGET_METHODS = ["global_lgbm", "regime_lgbm_hmm", "regime_lgbm_hmm_guided_hmm"]
REGIME_METHOD_BY_MODEL = {
    "global_lgbm": "none",
    "regime_lgbm_hmm": "hmm",
    "regime_lgbm_hmm_guided_hmm": "hmm_guided_hmm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate fold-local LightGBM feature importance for paper interpretability."
    )
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--methods", nargs="*", default=TARGET_METHODS)
    parser.add_argument("--max-shap-rows", type=int, default=256)
    parser.add_argument("--skip-shap", action="store_true")
    return parser.parse_args()


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


def load_fold_local_assignments(symbols: list[str]) -> pd.DataFrame:
    path = Path(SAVE_DIR) / "walkforward_regime_assignments.csv"
    if not path.exists():
        raise RuntimeError("walkforward_regime_assignments.csv missing. Run src/walkforward_regimes.py first.")
    assignments = pd.read_csv(path)
    required = {"fold", "split", "method", "symbol", "row_id", "regime"}
    missing = required - set(assignments.columns)
    if missing:
        raise RuntimeError(f"walkforward_regime_assignments.csv missing columns: {sorted(missing)}")
    assignments = assignments[assignments["symbol"].isin(symbols)].copy()
    assignments["row_id"] = assignments["row_id"].astype(int)
    assignments["fold"] = assignments["fold"].astype(int)
    assignments["regime"] = assignments["regime"].astype(int)
    return assignments


def booster_importance(model: lgb.LGBMClassifier, importance_type: str) -> np.ndarray:
    return model.booster_.feature_importance(importance_type=importance_type).astype(float)


def shap_importance(
    model: lgb.LGBMClassifier,
    x_train: np.ndarray,
    max_rows: int,
) -> np.ndarray:
    import shap

    if len(x_train) > max_rows:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(len(x_train), size=max_rows, replace=False)
        x_sample = x_train[sample_idx]
    else:
        x_sample = x_train

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_sample)
    if isinstance(values, list):
        arr = np.stack(values, axis=0)
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            arr = np.moveaxis(arr, -1, 0)
        elif arr.ndim == 2:
            arr = arr[None, :, :]
    return np.abs(arr).mean(axis=(0, 1))


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        return np.zeros_like(values)
    return values / total


def collect_model_importance(
    model: lgb.LGBMClassifier,
    x_train: np.ndarray,
    method: str,
    regime_method: str,
    fold: int,
    regime: int | str,
    n_train_rows: int,
    max_shap_rows: int,
    use_shap: bool,
) -> list[dict]:
    gain = booster_importance(model, "gain")
    split = booster_importance(model, "split")
    gain_norm = normalized(gain)
    split_norm = normalized(split)
    shap_raw = np.full(len(FEATURE_COLS), np.nan)
    shap_norm = np.full(len(FEATURE_COLS), np.nan)

    if use_shap:
        try:
            shap_raw = shap_importance(model, x_train, max_shap_rows)
            shap_norm = normalized(shap_raw)
        except Exception as exc:
            print(f"NOTE: SHAP skipped for {method} fold={fold} regime={regime}: {exc}")

    rows = []
    for idx, feature in enumerate(FEATURE_COLS):
        rows.append(
            {
                "method": method,
                "regime_method": regime_method,
                "fold": fold,
                "regime": regime,
                "feature": feature,
                "feature_family": feature_family(feature),
                "gain_importance": gain[idx],
                "gain_share": gain_norm[idx],
                "split_importance": split[idx],
                "split_share": split_norm[idx],
                "mean_abs_shap": shap_raw[idx],
                "shap_share": shap_norm[idx],
                "n_train_rows": n_train_rows,
                "target": PRIMARY_TARGET,
                "protocol": "fold_local_train_models",
            }
        )
    return rows


def global_importance_rows(
    df: pd.DataFrame,
    folds: list[tuple[int, int, int]],
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, _ = row_ids_for_fold(df, train_end, test_start, test_end)
        if not train_ids:
            continue
        x_train = df.loc[train_ids, FEATURE_COLS].to_numpy(dtype=float)
        y_train = df.loc[train_ids, "target_class"].to_numpy(dtype=int)
        model = fit_lgbm(x_train, y_train)
        if model is None:
            continue
        rows.extend(
            collect_model_importance(
                model,
                x_train,
                "global_lgbm",
                "none",
                fold,
                "all",
                len(train_ids),
                args.max_shap_rows,
                not args.skip_shap,
            )
        )
    return rows


def fold_local_assignment_for_method(
    df: pd.DataFrame,
    method: str,
    train_ids: list[int],
    test_ids: list[int],
    fold: int,
    hmm_matrix: np.ndarray,
    guided_embeddings: np.ndarray | None,
) -> pd.DataFrame:
    if method == "regime_lgbm_hmm":
        output = fit_hmm_assignments(
            df,
            hmm_matrix,
            train_ids,
            test_ids,
            fold,
            "hmm",
            covariance_type="full",
            implementation="interpretability_raw_feature_hmm",
        )
        return output.assignments
    if method == "regime_lgbm_hmm_guided_hmm":
        if guided_embeddings is None:
            raise RuntimeError("guided embeddings are required for regime_lgbm_hmm_guided_hmm.")
        output = fit_hmm_assignments(
            df,
            guided_embeddings,
            train_ids,
            test_ids,
            fold,
            "hmm_guided_hmm",
            covariance_type="diag",
            implementation="interpretability_guided_embedding_hmm",
        )
        return output.assignments
    raise ValueError(f"Unsupported regime interpretability method: {method}")


def regime_importance_rows(
    df: pd.DataFrame,
    folds: list[tuple[int, int, int]],
    method: str,
    args: argparse.Namespace,
    hmm_matrix: np.ndarray,
    guided_embeddings: np.ndarray | None,
) -> list[dict]:
    regime_method = REGIME_METHOD_BY_MODEL[method]
    rows = []

    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, test_ids = row_ids_for_fold(df, train_end, test_start, test_end)
        if not train_ids or not test_ids:
            continue
        assignments = fold_local_assignment_for_method(
            df,
            method,
            train_ids,
            test_ids,
            fold,
            hmm_matrix,
            guided_embeddings,
        )
        fold_train = assignments[(assignments["fold"] == fold) & (assignments["split"] == "train")]
        if fold_train.empty:
            continue
        for regime, group in fold_train.groupby("regime"):
            row_ids = group["row_id"].astype(int).tolist()
            if len(row_ids) < 40:
                continue
            x_train = df.loc[row_ids, FEATURE_COLS].to_numpy(dtype=float)
            y_train = df.loc[row_ids, "target_class"].to_numpy(dtype=int)
            model = fit_lgbm(x_train, y_train)
            if model is None:
                continue
            rows.extend(
                collect_model_importance(
                    model,
                    x_train,
                    method,
                    regime_method,
                    fold,
                    int(regime),
                    len(row_ids),
                    args.max_shap_rows,
                    not args.skip_shap,
                )
            )
    return rows


def summarize_importance(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["method", "regime_method", "regime", "feature", "feature_family"]
    summary = (
        raw.groupby(group_cols, dropna=False)
        .agg(
            mean_gain_share=("gain_share", "mean"),
            std_gain_share=("gain_share", "std"),
            mean_split_share=("split_share", "mean"),
            mean_shap_share=("shap_share", "mean"),
            folds_seen=("fold", "nunique"),
            mean_train_rows=("n_train_rows", "mean"),
        )
        .reset_index()
    )

    metric = "mean_shap_share" if summary["mean_shap_share"].notna().any() else "mean_gain_share"
    summary["rank_within_model_regime"] = (
        summary.groupby(["method", "regime"])[metric]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    global_summary = summary[summary["method"] == "global_lgbm"].copy()
    regime_summary = summary[summary["method"] != "global_lgbm"].copy()

    family_summary = (
        summary.groupby(["method", "regime_method", "regime", "feature_family"], dropna=False)
        .agg(
            mean_gain_share=("mean_gain_share", "sum"),
            mean_split_share=("mean_split_share", "sum"),
            mean_shap_share=("mean_shap_share", "sum"),
        )
        .reset_index()
    )
    return global_summary, regime_summary, family_summary


def plot_regime_importance(regime_summary: pd.DataFrame, output_path: Path) -> None:
    method = "regime_lgbm_hmm_guided_hmm"
    plot_data = regime_summary[regime_summary["method"] == method].copy()
    if plot_data.empty:
        return
    metric = "mean_shap_share" if plot_data["mean_shap_share"].notna().any() else "mean_gain_share"
    top_features = (
        plot_data.groupby("feature")[metric]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .index
        .tolist()
    )
    heat = plot_data[plot_data["feature"].isin(top_features)].pivot_table(
        index="regime",
        columns="feature",
        values=metric,
        aggfunc="mean",
        fill_value=0.0,
    )
    heat = heat[top_features]

    fig, ax = plt.subplots(figsize=(12, 5))
    image = ax.imshow(heat.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_title("Guided-HMM Regime Feature Importance")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Regime")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([str(idx) for idx in heat.index])
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label=metric)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_family_importance(family_summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = family_summary[family_summary["method"] == "regime_lgbm_hmm_guided_hmm"].copy()
    if plot_data.empty:
        return
    metric = "mean_shap_share" if plot_data["mean_shap_share"].notna().any() else "mean_gain_share"
    pivot = plot_data.pivot_table(
        index="regime",
        columns="feature_family",
        values=metric,
        aggfunc="sum",
        fill_value=0.0,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(pivot))
    for family in pivot.columns:
        values = pivot[family].to_numpy(dtype=float)
        ax.bar([str(idx) for idx in pivot.index], values, bottom=bottom, label=family)
        bottom += values
    ax.set_title("Guided-HMM Feature Family Importance by Regime")
    ax.set_xlabel("Regime")
    ax.set_ylabel(metric)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)

    df, _ = load_dense_contrastive_universe(args.symbols)
    folds = fold_ranges(df)
    if not folds:
        raise RuntimeError("No walk-forward folds found for interpretability.")
    hmm_matrix = finite_matrix(df.sort_index(), HMM_FEATURES)
    guided_embeddings = load_guided_embedding_matrix(df) if "regime_lgbm_hmm_guided_hmm" in args.methods else None

    rows: list[dict] = []
    requested = set(args.methods)
    if "global_lgbm" in requested:
        print("Collecting global LightGBM importance...")
        rows.extend(global_importance_rows(df, folds, args))

    for method in ["regime_lgbm_hmm", "regime_lgbm_hmm_guided_hmm"]:
        if method in requested:
            print(f"Collecting {method} regime-conditioned importance...")
            rows.extend(regime_importance_rows(df, folds, method, args, hmm_matrix, guided_embeddings))

    if not rows:
        raise RuntimeError("No interpretability rows produced.")

    raw = pd.DataFrame(rows)
    global_summary, regime_summary, family_summary = summarize_importance(raw)

    raw.to_csv(Path(SAVE_DIR) / "feature_importance_raw.csv", index=False)
    global_summary.to_csv(Path(SAVE_DIR) / "feature_importance_global.csv", index=False)
    regime_summary.to_csv(Path(SAVE_DIR) / "feature_importance_by_regime.csv", index=False)
    family_summary.to_csv(Path(SAVE_DIR) / "feature_family_summary.csv", index=False)

    plot_regime_importance(regime_summary, Path(SAVE_DIR) / "feature_importance_by_regime.png")
    plot_family_importance(family_summary, Path(SAVE_DIR) / "feature_family_importance.png")

    print("\nTop guided-HMM regime features:")
    display_cols = [
        "method",
        "regime",
        "feature",
        "feature_family",
        "mean_gain_share",
        "mean_shap_share",
        "rank_within_model_regime",
    ]
    top = regime_summary[
        (regime_summary["method"] == "regime_lgbm_hmm_guided_hmm")
        & (regime_summary["rank_within_model_regime"] <= 5)
    ].sort_values(["regime", "rank_within_model_regime"])
    print(top[display_cols].to_string(index=False))
    print("\nOK: interpretability artifacts saved.")


if __name__ == "__main__":
    main()
