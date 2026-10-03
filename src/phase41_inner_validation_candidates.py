from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_models import (
    CLASS_TO_LABEL,
    FEATURE_COLS,
    HORIZON_HOURS,
    LABEL_TO_CLASS,
    PRIMARY_TARGET,
    TC_PER_TRADE,
    aligned_proba,
    fit_lgbm,
    row_ids_for_fold,
    signal_from_probs,
)
from baselines import HMM_FEATURES
from config import BASE_DIR, SAVE_DIR
from evaluation import evaluation_metrics
from fold_local_encoder_walkforward import build_common_frame, output_path
from phase41_bounded_candidates import (
    blend_with_prior,
    load_config,
    negative_log_likelihood,
    posterior_temperature_weights,
    temperature_scale_probabilities,
)
from universe import add_symbol_args, resolve_symbols
from walkforward_regimes import POST_COLS, finite_matrix, fit_fold_assignments


REPORT_PATH = Path(BASE_DIR) / "reports" / "phase41_inner_validation_candidate_run.md"
DEFAULT_OUTPUT_PREFIX = "phase41_classical_"
EXPECTED_METHODS = {
    "global_lgbm",
    "regime_lgbm_hmm",
    "regime_lgbm_kmeans",
    "regime_lgbm_vol_bucket",
}
PHASE41B_EXECUTED_CANDIDATES = {
    "baseline",
    "p41_prob_temperature",
    "p41_prior_blend",
    "p41_posterior_temperature",
    "p41_global_regime_shrinkage",
}
PHASE41B_DEFERRED_CANDIDATES = {
    "p41_score_threshold",
}


@dataclass
class PredictionBundle:
    method: str
    regime_method: str
    frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 41B inner-validation selected calibration/gating candidates."
    )
    add_symbol_args(parser)
    parser.add_argument("--max-folds", type=int, default=1)
    parser.add_argument("--inner-validation-bars", type=int, default=720)
    parser.add_argument("--inner-embargo-bars", type=int, default=120)
    parser.add_argument("--label-purge-bars", type=int, default=8)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--output-dir", default=SAVE_DIR)
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=sorted(EXPECTED_METHODS),
        choices=sorted(EXPECTED_METHODS),
        help="Subset of Phase 41B methods to run. Use global_lgbm for a fast smoke.",
    )
    return parser.parse_args()


def class_prior(labels: pd.Series) -> np.ndarray:
    counts = labels.map(LABEL_TO_CLASS).value_counts(normalize=True)
    return np.array([counts.get(i, 0.0) for i in range(3)], dtype=float)


def apply_threshold(probs: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = probs[:, LABEL_TO_CLASS[1]] - probs[:, LABEL_TO_CLASS[-1]]
    neutral_prob = probs[:, LABEL_TO_CLASS[0]]
    best_class = np.argmax(probs, axis=1)
    signal = np.zeros(len(probs), dtype=int)
    signal[(score > threshold) & (best_class != LABEL_TO_CLASS[0])] = 1
    signal[(score < -threshold) & (best_class != LABEL_TO_CLASS[0])] = -1
    pred_label = np.array([CLASS_TO_LABEL[int(cls)] for cls in best_class], dtype=int)
    pred_label[neutral_prob == probs.max(axis=1)] = 0
    pred_label[np.abs(score) < threshold] = 0
    return score, pred_label, signal


def prediction_frame(
    df_by_row: pd.DataFrame,
    row_ids: list[int],
    probs: np.ndarray,
    method: str,
    regime_method: str,
    fold: int,
    threshold: float = 0.0,
) -> pd.DataFrame:
    score, pred_label, signal = apply_threshold(probs, threshold)
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


def evaluate_frame(frame: pd.DataFrame) -> dict:
    metrics = evaluation_metrics(frame, horizon_hours=HORIZON_HOURS, transaction_cost=TC_PER_TRADE)
    probs = frame[["prob_down", "prob_neutral", "prob_up"]].to_numpy(dtype=float)
    metrics["nll_loss"] = negative_log_likelihood(
        probs,
        frame["target_label"].to_numpy(dtype=int),
        {-1: 0, 0: 1, 1: 2},
    )
    return metrics


def split_inner_validation(
    df_by_row: pd.DataFrame,
    train_ids: list[int],
    validation_bars: int,
    embargo_bars: int,
    purge_bars: int,
) -> tuple[list[int], list[int]]:
    train_frame = df_by_row.loc[train_ids].sort_values(["open_time", "symbol"])
    times = pd.Series(train_frame["open_time"].drop_duplicates()).sort_values().reset_index(drop=True)
    if len(times) <= validation_bars + embargo_bars + purge_bars:
        raise RuntimeError("Not enough outer-training calendar bars for Phase 41 inner validation.")
    validation_start = times.iloc[-validation_bars]
    train_cutoff = validation_start - pd.Timedelta(hours=embargo_bars + purge_bars)
    inner_train = train_frame[train_frame["open_time"] <= train_cutoff]["row_id"].astype(int).tolist()
    inner_validation = train_frame[train_frame["open_time"] >= validation_start]["row_id"].astype(int).tolist()
    if not inner_train or not inner_validation:
        raise RuntimeError("Phase 41 inner split produced empty train or validation rows.")
    return inner_train, inner_validation


def fit_global_prediction(
    df_by_row: pd.DataFrame,
    train_ids: list[int],
    row_ids: list[int],
    fold: int,
    method: str = "global_lgbm",
) -> pd.DataFrame:
    model = fit_lgbm(
        df_by_row.loc[train_ids, FEATURE_COLS].values,
        df_by_row.loc[train_ids, "target_class"].values,
    )
    if model is None:
        raise RuntimeError(f"Could not fit {method} for fold {fold}.")
    probs = aligned_proba(model, df_by_row.loc[row_ids, FEATURE_COLS].values)
    return prediction_frame(df_by_row, row_ids, probs, method, "none", fold)


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


def transform_assignment_posteriors(assignments: pd.DataFrame, temperature: float) -> pd.DataFrame:
    out = assignments.copy()
    if temperature == 1.0:
        return out
    posts = out[POST_COLS].to_numpy(dtype=float)
    out.loc[:, POST_COLS] = posterior_temperature_weights(posts, temperature)
    out["regime"] = out[POST_COLS].to_numpy(dtype=float).argmax(axis=1)
    return out


def predict_regime(
    df_by_row: pd.DataFrame,
    assignments: pd.DataFrame,
    models: dict[int, object],
    row_ids: list[int],
    method: str,
    fold: int,
    posterior_temperature: float = 1.0,
) -> pd.DataFrame:
    assignments = transform_assignment_posteriors(assignments, posterior_temperature)
    subset = assignments[assignments["row_id"].isin(row_ids)].copy()
    if subset.empty or not models:
        raise RuntimeError(f"No regime prediction rows for {method} fold {fold}.")
    subset = subset.sort_values(["symbol", "feat_idx"])
    ordered_ids = subset["row_id"].astype(int).tolist()
    x = df_by_row.loc[ordered_ids, FEATURE_COLS].values
    combined = np.zeros((len(ordered_ids), 3), dtype=float)
    weight_sum = np.zeros((len(ordered_ids), 1), dtype=float)
    for regime, model in models.items():
        probs = aligned_proba(model, x)
        col = f"post_{regime}"
        weights = subset[col].to_numpy(dtype=float).reshape(-1, 1) if col in subset else (
            subset["regime"].to_numpy(dtype=int) == regime
        ).astype(float).reshape(-1, 1)
        combined += probs * weights
        weight_sum += weights
    combined = np.divide(
        combined,
        weight_sum,
        out=np.full_like(combined, 1.0 / 3.0),
        where=weight_sum != 0,
    )
    return prediction_frame(df_by_row, ordered_ids, combined, f"regime_lgbm_{method}", method, fold)


def replace_probabilities(frame: pd.DataFrame, probs: np.ndarray, threshold: float = 0.0) -> pd.DataFrame:
    out = frame.copy()
    score, pred_label, signal = apply_threshold(probs, threshold)
    out["prob_down"] = probs[:, LABEL_TO_CLASS[-1]]
    out["prob_neutral"] = probs[:, LABEL_TO_CLASS[0]]
    out["prob_up"] = probs[:, LABEL_TO_CLASS[1]]
    out["score"] = score
    out["pred_label"] = pred_label
    out["signal"] = signal
    return out


def candidate_frames(
    base: pd.DataFrame,
    prior: np.ndarray,
    config: dict,
    global_reference: pd.DataFrame | None = None,
) -> list[tuple[str, str, pd.DataFrame]]:
    probs = base[["prob_down", "prob_neutral", "prob_up"]].to_numpy(dtype=float)
    rows = [("baseline", "none", base)]
    for temp in config["candidate_grids"]["probability_temperature"]:
        rows.append(("p41_prob_temperature", f"temperature={temp}", replace_probabilities(base, temperature_scale_probabilities(probs, temp))))
    for weight in config["candidate_grids"]["prior_blend_weight"]:
        rows.append(("p41_prior_blend", f"prior_blend_weight={weight}", replace_probabilities(base, blend_with_prior(probs, prior, weight))))
    if global_reference is not None:
        if "row_id" not in base or "row_id" not in global_reference:
            raise ValueError("Global-regime shrinkage requires row_id alignment.")
        aligned_global = global_reference.set_index("row_id").reindex(base["row_id"].to_numpy())
        if aligned_global[["prob_down", "prob_neutral", "prob_up"]].isna().any().any():
            raise ValueError("Global reference is missing rows required for shrinkage alignment.")
        global_probs = aligned_global[["prob_down", "prob_neutral", "prob_up"]].to_numpy(dtype=float)
        if len(global_probs) == len(probs):
            for weight in config["candidate_grids"]["global_regime_shrinkage"]:
                blended = (1.0 - weight) * probs + weight * global_probs
                rows.append(("p41_global_regime_shrinkage", f"global_regime_shrinkage={weight}", replace_probabilities(base, blended)))
    return rows


def select_candidate(
    method: str,
    fold: int,
    candidates: list[tuple[str, str, pd.DataFrame]],
    baseline_metrics: dict,
) -> tuple[pd.Series, list[dict]]:
    rows = []
    baseline_turnover = float(baseline_metrics.get("turnover", 0.0) or 0.0)
    for candidate_id, params, frame in candidates:
        metrics = evaluate_frame(frame)
        turnover = float(metrics.get("turnover", 0.0) or 0.0)
        turnover_increase = 0.0 if baseline_turnover == 0 else (turnover - baseline_turnover) / abs(baseline_turnover)
        rows.append(
            {
                "fold": fold,
                "method": method,
                "candidate_id": candidate_id,
                "candidate_params": params,
                "inner_validation_nll": metrics["nll_loss"],
                "inner_validation_mean_asset_ic": metrics["mean_asset_IC"],
                "inner_validation_turnover": turnover,
                "turnover_increase_vs_baseline": turnover_increase,
                "coverage_ok": len(frame) == len(candidates[0][2]),
            }
        )
    table = pd.DataFrame(rows)
    valid = table[
        table["coverage_ok"].astype(bool)
        & (pd.to_numeric(table["turnover_increase_vs_baseline"], errors="coerce") <= 0.25)
    ].copy()
    if valid.empty:
        valid = table[table["candidate_id"] == "baseline"].copy()
    valid = valid.sort_values(["inner_validation_nll", "candidate_id", "candidate_params"])
    return valid.iloc[0], rows


def summarize_method(frame: pd.DataFrame, method: str, regime_method: str, symbol_scope: str) -> dict:
    metrics = evaluate_frame(frame)
    return {
        "method": method,
        "target": PRIMARY_TARGET,
        "horizon": f"{HORIZON_HOURS}h",
        "regime_method": regime_method,
        "symbol_scope": symbol_scope,
        **metrics,
        "n_test_rows": int(len(frame)),
    }


def run_phase41(args: argparse.Namespace) -> None:
    config = load_config()
    symbols = resolve_symbols(args)
    frame = build_common_frame(symbols)
    df_by_row = frame.set_index("row_id", drop=False)
    raw_matrix = finite_matrix(df_by_row.sort_index(), FEATURE_COLS)
    hmm_matrix = finite_matrix(df_by_row.sort_index(), HMM_FEATURES)
    requested_methods = set(args.methods)
    folds = []
    from alpha_models import fold_ranges  # local import avoids widening public surface

    for fold, (train_end, test_start, test_end) in enumerate(fold_ranges(df_by_row), start=1):
        if args.max_folds and fold > args.max_folds:
            break
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        if train_ids and test_ids:
            folds.append((fold, train_ids, test_ids))

    prediction_parts: dict[str, list[pd.DataFrame]] = {method: [] for method in requested_methods}
    candidate_rows: list[dict] = []
    selected_rows: list[dict] = []
    fold_metric_rows: list[dict] = []

    for fold, train_ids, test_ids in folds:
        inner_train, inner_val = split_inner_validation(
            df_by_row,
            train_ids,
            args.inner_validation_bars,
            args.inner_embargo_bars,
            args.label_purge_bars,
        )
        prior = class_prior(df_by_row.loc[inner_train, "target_label"])

        inner_global = fit_global_prediction(df_by_row, inner_train, inner_val, fold)
        outer_global = fit_global_prediction(df_by_row, train_ids, test_ids, fold)
        selected, rows = select_candidate(
            "global_lgbm",
            fold,
            candidate_frames(inner_global, prior, config),
            evaluate_frame(inner_global),
        )
        candidate_rows.extend(rows)
        selected_rows.append(selected.to_dict())
        chosen_frame = dict((cid + "|" + params, frame) for cid, params, frame in candidate_frames(outer_global, prior, config))[
            f"{selected['candidate_id']}|{selected['candidate_params']}"
        ]
        if "global_lgbm" in requested_methods:
            prediction_parts["global_lgbm"].append(chosen_frame)

        requested_regime_methods = [
            method.removeprefix("regime_lgbm_")
            for method in requested_methods
            if method.startswith("regime_lgbm_")
        ]
        if requested_regime_methods:
            for scope_name, split_train, split_eval in [
                ("inner", inner_train, inner_val),
                ("outer", train_ids, test_ids),
            ]:
                outputs = fit_fold_assignments(
                    df_by_row,
                    None,
                    raw_matrix,
                    hmm_matrix,
                    split_train,
                    split_eval,
                    fold,
                    guided_embeddings=None,
                )
                if scope_name == "inner":
                    inner_outputs = {output.method: output.assignments for output in outputs}
                else:
                    outer_outputs = {output.method: output.assignments for output in outputs}

        for regime_method in requested_regime_methods:
            method = f"regime_lgbm_{regime_method}"
            inner_models = train_regime_models(df_by_row, inner_outputs[regime_method], inner_train)
            outer_models = train_regime_models(df_by_row, outer_outputs[regime_method], train_ids)
            inner_base = predict_regime(df_by_row, inner_outputs[regime_method], inner_models, inner_val, regime_method, fold)
            outer_base = predict_regime(df_by_row, outer_outputs[regime_method], outer_models, test_ids, regime_method, fold)

            candidates = candidate_frames(inner_base, prior, config, global_reference=inner_global)
            for temp in config["candidate_grids"]["posterior_temperature"]:
                frame_temp = predict_regime(
                    df_by_row,
                    inner_outputs[regime_method],
                    inner_models,
                    inner_val,
                    regime_method,
                    fold,
                    posterior_temperature=temp,
                )
                candidates.append(("p41_posterior_temperature", f"posterior_temperature={temp}", frame_temp))
            selected, rows = select_candidate(method, fold, candidates, evaluate_frame(inner_base))
            candidate_rows.extend(rows)
            selected_rows.append(selected.to_dict())

            if selected["candidate_id"] == "p41_posterior_temperature":
                temp = float(str(selected["candidate_params"]).split("=")[1])
                final_frame = predict_regime(
                    df_by_row,
                    outer_outputs[regime_method],
                    outer_models,
                    test_ids,
                    regime_method,
                    fold,
                    posterior_temperature=temp,
                )
            else:
                final_candidates = candidate_frames(outer_base, prior, config, global_reference=outer_global)
                final_frame = dict((cid + "|" + params, frame) for cid, params, frame in final_candidates)[
                    f"{selected['candidate_id']}|{selected['candidate_params']}"
                ]
            prediction_parts[method].append(final_frame)

        for method, parts in prediction_parts.items():
            if parts:
                current = parts[-1]
                fold_metric_rows.append(
                    summarize_method(
                        current,
                        method,
                        "none" if method == "global_lgbm" else method.removeprefix("regime_lgbm_"),
                        "+".join(symbols),
                    )
                    | {"fold": fold}
                )
        print(f"Fold {fold:02d}: selected Phase 41 candidates using inner validation only.")

    outputs = []
    for method in sorted(requested_methods):
        frame_out = pd.concat(prediction_parts[method], ignore_index=True)
        regime_method = "none" if method == "global_lgbm" else method.removeprefix("regime_lgbm_")
        outputs.append(PredictionBundle(method, regime_method, frame_out))
    predictions = pd.concat([output.frame for output in outputs], ignore_index=True)
    results = pd.DataFrame(
        [
            summarize_method(output.frame, output.method, output.regime_method, "+".join(symbols))
            for output in outputs
        ]
    )
    candidate_table = pd.DataFrame(candidate_rows)
    selected_table = pd.DataFrame(selected_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path(out_dir, args.output_prefix, "experiment_results"), index=False)
    fold_metrics.to_csv(output_path(out_dir, args.output_prefix, "fold_metrics"), index=False)
    candidate_table.to_csv(output_path(out_dir, args.output_prefix, "inner_candidate_results"), index=False)
    selected_table.to_csv(output_path(out_dir, args.output_prefix, "selected_candidates"), index=False)
    predictions.to_csv(output_path(out_dir, args.output_prefix, "alpha_oos_predictions"), index=False)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text(results, selected_table, len(folds), args.max_folds), encoding="utf-8")
    print(f"OK: Phase 41B wrote {len(folds)} fold(s) with inner-validation-selected candidates.")


def report_text(results: pd.DataFrame, selected: pd.DataFrame, folds: int, max_folds: int | None) -> str:
    status = "smoke/development check" if max_folds and max_folds < 16 else "full development-observed run"
    result_rows = "\n".join(
        f"| `{row.method}` | {row.mean_asset_IC:.6f} | {row.nll_loss:.6f} | {row.Sharpe:.4f} | {int(row.n_test_rows)} |"
        for row in results.itertuples(index=False)
    )
    selected_counts = selected.groupby(["method", "candidate_id", "candidate_params"]).size().reset_index(name="folds")
    selected_rows = "\n".join(
        f"| `{row.method}` | `{row.candidate_id}` | `{row.candidate_params}` | {int(row.folds)} |"
        for row in selected_counts.itertuples(index=False)
    )
    interpretation = (
        "This run is not a performance claim. Its purpose is to evaluate whether "
        "Phase 41 candidates selected only on inner validation improve the repaired "
        "global/classical development ladder. Statistical adjudication is required "
        "before making any development-level comparison."
        if not (max_folds and max_folds < 16)
        else "This run is not a performance claim. Its purpose is to verify that Phase 41 candidate selection can be performed without using outer-test results for tuning. A full 16-fold development run is required before making any development-level comparison."
    )
    return f"""# Phase 41B Inner-Validation Candidate Run

## Status

This is a **{status}**. Candidate parameters are selected on inner chronological validation only, then evaluated once on the outer fold. This is still development-observed evidence and is not a locked final test.

## Scope

- Folds evaluated: {folds}
- Methods: global LightGBM, raw HMM, KMeans, volatility buckets
- Executed candidate families: probability calibration and soft regime gating
- Deferred registered candidates: score-threshold execution control (`p41_score_threshold`)
- Neural/guided candidates: deferred until this classical/global calibration layer is verified
- Selection input: inner validation only
- Forbidden input: Phase 40 outer-test statistical results

Score-threshold candidates are registered but deferred in this Phase 41B run. They change trade execution/signals rather than probability calibration, so they require a separate execution-focused run instead of being mixed into this probability/NLL adjudication.

The v1 grid also contains baseline-equivalent no-op values such as temperature `1.0`, prior blend `0.0`, shrinkage `0.0`, and posterior temperature `1.0`. These are retained as explicit controls for the recorded Phase 41B run; a future v2 registration should remove duplicate no-op variants before any new full run.

## Outer-Fold Diagnostic Results

| Method | Mean asset IC | NLL | Sharpe | Rows |
|---|---:|---:|---:|---:|
{result_rows}

## Selected Candidates

| Method | Candidate | Parameters | Folds |
|---|---|---|---:|
{selected_rows}

## Interpretation

{interpretation}
"""


def main() -> None:
    run_phase41(parse_args())


if __name__ == "__main__":
    main()
