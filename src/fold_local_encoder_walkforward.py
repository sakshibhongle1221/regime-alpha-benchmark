from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from alpha_models import (
    EMBARGO,
    FEATURE_COLS,
    PredictionSet,
    build_prediction_frame,
    fit_lgbm,
    aligned_proba,
    align_to_common_calendar,
    fold_ranges,
    load_model_frame,
    predict_regime_models,
    row_ids_for_fold,
    summarize_predictions,
    train_regime_models,
    validate_test_coverage,
)
from baselines import HMM_FEATURES
from config import SAVE_DIR, WINDOW_SIZE
from dataset import load_feature_frame
from fold_checkpoint import (
    canonical_hash,
    checkpoint_exists,
    initialize_run_state,
    load_fold_checkpoint,
    sha256_file,
    write_fold_checkpoint,
)
from fold_local_encoder import (
    FoldEncoderConfig,
    encode_causal_rows,
    fit_fold_encoder,
    make_fold_bounds,
)
from universe import add_symbol_args, resolve_symbols
from walkforward_regimes import (
    POST_COLS,
    build_guided_alpha_comparison,
    finite_matrix,
    fit_fold_assignments,
    summarize_assignments,
)


EXPECTED_METHODS = {
    "global_lgbm",
    "regime_lgbm_vol_bucket",
    "regime_lgbm_kmeans",
    "regime_lgbm_hmm",
    "regime_lgbm_contrastive",
    "regime_lgbm_contrastive_hmm",
    "regime_lgbm_hmm_guided_gmm",
    "regime_lgbm_hmm_guided_hmm",
}
EXPECTED_FULL_FOLDS = 16
FULL_PROTOCOL_EPOCHS = 30


def is_smoke_manifest(manifest: pd.DataFrame) -> bool:
    if manifest.empty:
        return True
    if "run_kind" in manifest and set(manifest["run_kind"].astype(str)) == {"full"}:
        return False
    folds = int(manifest["fold"].nunique())
    max_epochs = int(pd.to_numeric(manifest["max_epochs"], errors="coerce").min())
    window_cap = 0
    if "window_cap" in manifest:
        window_cap = int(pd.to_numeric(manifest["window_cap"], errors="coerce").fillna(0).max())
    return folds < EXPECTED_FULL_FOLDS or max_epochs < FULL_PROTOCOL_EPOCHS or window_cap > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fully fold-local vanilla/guided encoder benchmark."
    )
    add_symbol_args(parser)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=5000,
        help="Deterministic per-stage encoder window budget; 0 uses every eligible window.",
    )
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--inner-validation-bars", type=int, default=720)
    parser.add_argument("--inner-embargo-bars", type=int, default=120)
    parser.add_argument("--label-purge-bars", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--minimum-selection-epochs", type=int, default=3)
    parser.add_argument("--output-prefix", default="crypto20_fold_local_")
    parser.add_argument(
        "--output-dir",
        default=SAVE_DIR,
        help="Directory for compact CSV outputs; use .tmp for isolated smoke runs.",
    )
    parser.add_argument(
        "--report-path",
        default=str(Path("reports") / "phase39_fold_local_results.md"),
        help="Markdown report destination; use an isolated path for smoke runs.",
    )
    parser.add_argument(
        "--run-name",
        default="phase39_full",
        help="Isolated checkpoint namespace below --heavy-dir.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate lineage and reuse completed fold checkpoints.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the compact Markdown report from existing compact CSV artifacts.",
    )
    parser.add_argument(
        "--calendar-audit-only",
        action="store_true",
        help="Validate the common panel and every global train/test timestamp boundary without training or writing artifacts.",
    )
    parser.add_argument(
        "--heavy-dir",
        default=str(Path(".tmp") / "phase39_fold_local"),
        help="Ignored directory for fold weights and run metadata.",
    )
    parser.add_argument(
        "--freeze-manifest",
        default=str(Path(SAVE_DIR) / "crypto20_development_freeze_manifest.json"),
        help="Required development-data freeze manifest for training and calendar audits.",
    )
    return parser.parse_args()


def output_path(output_dir: str | Path, prefix: str, stem: str) -> Path:
    path = Path(output_dir) / f"{prefix}{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def verify_frozen_dataset(
    manifest_path: str | Path,
    symbols: list[str],
    frame: pd.DataFrame,
    folds: list[tuple[int, int, int]],
    data_hash: str,
) -> dict:
    path = Path(manifest_path)
    if not path.exists():
        raise RuntimeError(
            f"Development freeze manifest is missing: {path}. "
            "Run src/freeze_development_dataset.py first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    allowed_roles = {"development_observed", "locked_registered_unobserved"}
    if manifest.get("data_role") not in allowed_roles:
        failures.append(f"data role is not one of {sorted(allowed_roles)}")
    if manifest.get("symbols") != symbols:
        failures.append("symbol order differs from frozen universe")
    if manifest.get("experiment_data_sha256") != data_hash:
        failures.append("experiment data hash differs from frozen snapshot")
    if int(manifest.get("prediction_rows", -1)) != len(frame):
        failures.append("prediction row count differs from frozen snapshot")
    if int(manifest.get("fold_count", -1)) != len(folds):
        failures.append("fold count differs from frozen snapshot")
    if failures:
        raise RuntimeError("Frozen dataset verification failed: " + "; ".join(failures))
    return manifest


def load_raw_matrices(
    symbols: list[str], aligned_frame: pd.DataFrame | None = None
) -> dict[str, np.ndarray]:
    feature_frames = {symbol: load_feature_frame(symbol) for symbol in symbols}
    full_matrices = {
        symbol: feature_frames[symbol][FEATURE_COLS].to_numpy(dtype=np.float32)
        for symbol in symbols
    }
    if aligned_frame is None:
        matrices = full_matrices
    else:
        matrices = {}
        required = {"symbol", "source_feat_idx", "feat_idx"}
        missing = sorted(required - set(aligned_frame.columns))
        if missing:
            raise ValueError(f"Aligned frame is missing matrix lineage columns: {missing}")
        for symbol in symbols:
            rows = aligned_frame[aligned_frame["symbol"] == symbol].sort_values("feat_idx")
            source_indices = rows["source_feat_idx"].astype(int).drop_duplicates().to_numpy()
            if len(source_indices) == 0:
                raise RuntimeError(f"No aligned feature rows for {symbol}.")
            expected = np.arange(source_indices[0], source_indices[-1] + 1, dtype=int)
            if not np.array_equal(source_indices, expected):
                raise RuntimeError(f"Aligned feature history for {symbol} is not contiguous.")
            offsets = (
                rows["source_feat_idx"].astype(int) - rows["feat_idx"].astype(int)
            ).unique()
            if len(offsets) != 1:
                raise RuntimeError(f"Aligned index lineage for {symbol} is inconsistent.")
            source_start = int(offsets[0])
            source_timestamps = (
                feature_frames[symbol]
                .iloc[rows["source_feat_idx"].astype(int).to_numpy()]["open_time"]
                .reset_index(drop=True)
            )
            aligned_timestamps = rows["open_time"].reset_index(drop=True)
            if not source_timestamps.equals(aligned_timestamps):
                raise RuntimeError(
                    f"Timestamp/index lineage mismatch between features and model frame for {symbol}."
                )
            matrices[symbol] = np.ascontiguousarray(
                full_matrices[symbol][source_start : source_indices[-1] + 1]
            )
    short = {symbol: len(matrix) for symbol, matrix in matrices.items() if len(matrix) <= WINDOW_SIZE}
    if short:
        raise RuntimeError(f"Symbols with insufficient encoder history: {short}")
    return matrices


def hash_experiment_data(
    symbols: list[str], raw_matrices: dict[str, np.ndarray], frame: pd.DataFrame
) -> str:
    digest = hashlib.sha256()
    for symbol in symbols:
        matrix = np.ascontiguousarray(raw_matrices[symbol])
        digest.update(symbol.encode("utf-8"))
        digest.update(str(matrix.shape).encode("ascii"))
        digest.update(str(matrix.dtype).encode("ascii"))
        digest.update(matrix.tobytes())
    lineage_columns = [
        "symbol", "source_feat_idx", "feat_idx", "open_time", "target_class",
        "target_return", *FEATURE_COLS
    ]
    frame_hashes = pd.util.hash_pandas_object(
        frame[lineage_columns], index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    digest.update(frame_hashes.tobytes())
    return digest.hexdigest()


def hash_protocol_sources() -> str:
    source_dir = Path(__file__).resolve().parent
    paths = [
        source_dir / name
        for name in [
            "fold_local_encoder_walkforward.py",
            "fold_local_encoder.py",
            "fold_checkpoint.py",
            "alpha_models.py",
            "walkforward_regimes.py",
            "encoder.py",
            "guided_encoder.py",
        ]
    ]
    return canonical_hash({path.name: sha256_file(path) for path in paths})


def absorb_fold_frames(
    frames: dict[str, pd.DataFrame],
    prediction_parts: dict[str, list[pd.DataFrame]],
    test_assignments: list[pd.DataFrame],
    implementation_rows: list[dict],
    manifest_rows: list[dict],
    loss_rows: list[dict],
    coverage_rows: list[dict],
) -> None:
    predictions = frames["predictions"]
    for method, part in predictions.groupby("method", sort=False):
        if method not in prediction_parts:
            raise RuntimeError(f"Checkpoint contains unexpected method: {method}")
        prediction_parts[method].append(part.copy())
    test_assignments.append(frames["assignments"])
    implementation_rows.extend(frames["implementations"].to_dict("records"))
    manifest_rows.extend(frames["manifest"].to_dict("records"))
    loss_rows.extend(frames["losses"].to_dict("records"))
    coverage_rows.extend(frames["coverage"].to_dict("records"))


def build_common_frame(symbols: list[str]) -> pd.DataFrame:
    frame = load_model_frame(symbols)
    counts = frame.groupby("symbol")["feat_idx"].agg(["min", "max", "count"])
    if counts.empty or len(counts) != len(symbols):
        raise RuntimeError("Fold-local encoder universe is missing requested symbols.")
    return align_to_common_calendar(frame, symbols, warmup_rows=WINDOW_SIZE - 1)


def run_global_fold(
    df_by_row: pd.DataFrame,
    train_ids: list[int],
    test_ids: list[int],
    fold: int,
) -> pd.DataFrame:
    model = fit_lgbm(
        df_by_row.loc[train_ids, FEATURE_COLS].values,
        df_by_row.loc[train_ids, "target_class"].values,
    )
    if model is None:
        raise RuntimeError(f"Global LightGBM could not fit fold {fold}.")
    probabilities = aligned_proba(model, df_by_row.loc[test_ids, FEATURE_COLS].values)
    return build_prediction_frame(
        df_by_row, test_ids, probabilities, "global_lgbm", "none", fold
    )


def validate_complete_coverage(results: pd.DataFrame, predictions: pd.DataFrame, folds: int) -> None:
    methods = set(results["method"].astype(str))
    missing = sorted(EXPECTED_METHODS - methods)
    extra = sorted(methods - EXPECTED_METHODS)
    if missing or extra:
        raise RuntimeError(f"Phase 39 method mismatch: missing={missing}, extra={extra}")
    validate_test_coverage(results)
    counts = predictions.groupby("method").size()
    if counts.nunique() != 1:
        raise RuntimeError(f"Unequal row-level method coverage: {counts.to_dict()}")
    fold_counts = predictions.groupby("method")["fold"].nunique()
    if not (fold_counts == folds).all():
        raise RuntimeError(f"Method fold coverage mismatch: {fold_counts.to_dict()}")
    duplicated = predictions.duplicated(["method", "symbol", "feat_idx", "fold"]).sum()
    if duplicated:
        raise RuntimeError(f"Phase 39 produced {int(duplicated)} duplicate predictions.")


def build_historical_comparison(results: pd.DataFrame) -> pd.DataFrame:
    path = Path(SAVE_DIR) / "crypto20_walkforward_experiment_results.csv"
    if not path.exists():
        return pd.DataFrame()
    historical = pd.read_csv(path)
    metrics = ["IC", "Sharpe", "drawdown", "total_return", "turnover", "n_test_rows"]
    merged = results.merge(historical, on="method", how="left", suffixes=("_fold_local_encoder", "_phase36"))
    for metric in metrics:
        left = f"{metric}_fold_local_encoder"
        right = f"{metric}_phase36"
        if left in merged and right in merged:
            merged[f"delta_{metric}"] = merged[left] - merged[right]
    return merged


def write_results_report(
    path: Path,
    results: pd.DataFrame,
    manifest: pd.DataFrame,
    folds: int,
    symbols: list[str],
    smoke: bool,
    data_role: str = "development_observed",
) -> None:
    cols = ["method", "IC", "Sharpe", "drawdown", "total_return", "turnover", "n_test_rows"]
    def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
        display = frame.copy()
        for column in display.columns:
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].map(lambda value: f"{value:.{digits}f}")
        headers = [str(column) for column in display.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend(
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in display.itertuples(index=False, name=None)
        )
        return "\n".join(lines)

    table = markdown_table(results[cols])
    selected_frame = (
        manifest.groupby("encoder_method")["selected_epoch"]
        .agg(["min", "median", "max"])
        .reset_index()
    )
    selected = markdown_table(selected_frame)
    window_budget = int(pd.to_numeric(manifest.get("window_cap", pd.Series([0])), errors="coerce").fillna(0).max())
    window_budget_text = "all eligible windows" if window_budget == 0 else f"at most {window_budget:,} deterministic windows per encoder stage"
    if data_role == "locked_registered_unobserved":
        status = "locked confirmatory evaluation"
        evidence_sentence = (
            "This is the frozen locked-holdout evaluation; the result must be reported once "
            "without same-holdout retuning."
        )
        title = "Phase 43B Locked External Holdout Evaluation"
        interpretation = (
            "This locked run evaluates the Phase 43A frozen candidate and references on the "
            "Phase 43B registered external holdout. The result may confirm or fail to confirm "
            "the guided-HMM mechanism, but it must not be used to tune thresholds, candidates, "
            "features, labels, or architecture."
        )
    else:
        status = "smoke validation only; no performance claim" if smoke else "full development-observed benchmark"
        evidence_sentence = (
            "All outcomes remain development-observed and cannot be used as an untouched final test."
        )
        title = "Phase 39 Fully Fold-Local Encoder Results"
        interpretation = (
            "A smoke run validates code paths, leakage boundaries, artifacts, and coverage only. "
            "A full run is still development evidence. Model changes require a new registered "
            "candidate family, and confirmatory claims require a frozen configuration evaluated "
            "once on a locked holdout."
        )
    text = f"""# {title}

## Run Status

This is a **{status}** over {folds} fold(s) and {len(symbols)} symbol(s), using {window_budget_text}. {evidence_sentence}

## Validity Contract

- Scalers, weak-supervision HMMs, contrastive pairs, encoder weights, assignment layers, and alpha models are fit inside each outer fold.
- Epochs are selected on an inner chronological validation block.
- The selected epoch count is refit on the full authorized outer-training interval.
- Outer-test metrics do not influence training or model selection.
- Vanilla and guided learned methods share the same outer folds and test coverage as classical and global baselines.

## Method Results

{table}

## Selected Epochs

{selected}

## Interpretation Rule

{interpretation}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.report_only:
        results_path = output_path(args.output_dir, args.output_prefix, "experiment_results")
        manifest_path = output_path(args.output_dir, args.output_prefix, "encoder_manifest")
        if not results_path.exists() or not manifest_path.exists():
            raise RuntimeError("Report-only mode requires existing result and manifest CSV files.")
        results = pd.read_csv(results_path)
        manifest = pd.read_csv(manifest_path)
        folds = int(manifest["fold"].nunique())
        symbol_scope = str(results.iloc[0].get("symbol_scope", "")) if not results.empty else ""
        symbols = [symbol for symbol in symbol_scope.split("+") if symbol]
        write_results_report(
            Path(args.report_path),
            results,
            manifest,
            folds,
            symbols,
            smoke=is_smoke_manifest(manifest),
            data_role="development_observed",
        )
        print("OK: Phase 39 compact report rebuilt from existing artifacts.")
        return
    symbols = resolve_symbols(args)
    config = FoldEncoderConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_windows=args.max_windows,
        inner_validation_bars=args.inner_validation_bars,
        inner_embargo_bars=args.inner_embargo_bars,
        label_purge_bars=args.label_purge_bars,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        minimum_selection_epochs=args.minimum_selection_epochs,
    )
    if config.epochs < 1 or config.batch_size < 2:
        raise ValueError("Epochs must be >=1 and batch size must be >=2.")

    frame = build_common_frame(symbols)
    df_by_row = frame.set_index("row_id", drop=False)
    raw_matrices = load_raw_matrices(symbols, frame)
    all_folds = fold_ranges(frame)
    if not all_folds:
        raise RuntimeError("No eligible walk-forward folds.")
    data_hash = hash_experiment_data(symbols, raw_matrices, frame)
    freeze_manifest = verify_frozen_dataset(
        args.freeze_manifest, symbols, frame, all_folds, data_hash
    )
    folds = all_folds[: args.max_folds] if args.max_folds is not None else all_folds
    if not folds:
        raise RuntimeError("No folds remain after applying --max-folds.")

    if args.calendar_audit_only:
        audit_rows = []
        for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
            train_ids, test_ids = row_ids_for_fold(
                df_by_row, train_end, test_start, test_end
            )
            audit_rows.append(
                {
                    "fold": fold,
                    "latest_train": df_by_row.loc[train_ids, "open_time"].max(),
                    "earliest_test": df_by_row.loc[test_ids, "open_time"].min(),
                    "train_rows": len(train_ids),
                    "test_rows": len(test_ids),
                }
            )
        print(pd.DataFrame(audit_rows).to_string(index=False))
        print(
            f"OK: {len(symbols)} symbols share one calendar index and all "
            f"{len(folds)} folds have strict global train/test separation under "
            f"{freeze_manifest['freeze_id']}."
        )
        return

    raw_matrix = finite_matrix(df_by_row.sort_index(), FEATURE_COLS)
    hmm_matrix = finite_matrix(df_by_row.sort_index(), HMM_FEATURES)
    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        raise ValueError("--run-name must be one safe directory name without path separators.")
    run_dir = Path(args.heavy_dir) / args.run_name
    source_hash = hash_protocol_sources()
    fold_specification = [list(map(int, fold_range)) for fold_range in folds]
    config_hash = canonical_hash(
        {
            "config": asdict(config),
            "symbols": symbols,
            "output_prefix": args.output_prefix,
            "output_dir": str(Path(args.output_dir)),
            "expected_methods": sorted(EXPECTED_METHODS),
            "freeze_id": freeze_manifest["freeze_id"],
            "freeze_manifest_sha256": sha256_file(Path(args.freeze_manifest)),
        }
    )
    fold_hash = canonical_hash(fold_specification)
    protocol_hash = canonical_hash(
        {
            "config_hash": config_hash,
            "data_hash": data_hash,
            "source_hash": source_hash,
            "fold_hash": fold_hash,
        }
    )
    run_state = {
        "schema_version": 1,
        "protocol_hash": protocol_hash,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "source_hash": source_hash,
        "fold_hash": fold_hash,
        "config": asdict(config),
        "symbols": symbols,
        "folds": fold_specification,
        "expected_methods": sorted(EXPECTED_METHODS),
        "output_prefix": args.output_prefix,
        "output_dir": str(Path(args.output_dir)),
        "run_name": args.run_name,
        "freeze_id": freeze_manifest["freeze_id"],
        "freeze_manifest_sha256": sha256_file(Path(args.freeze_manifest)),
    }
    initialize_run_state(run_dir, run_state, resume=args.resume)

    prediction_parts: dict[str, list[pd.DataFrame]] = {
        method: [] for method in EXPECTED_METHODS
    }
    test_assignments = []
    implementation_rows = []
    manifest_rows = []
    loss_rows = []
    coverage_rows = []

    print(
        f"Phase 39 fold-local encoder benchmark: symbols={len(symbols)} folds={len(folds)} "
        f"epochs={config.epochs} max_windows={config.max_windows or 'all'} device={config.device}"
    )
    smoke_run = len(folds) < EXPECTED_FULL_FOLDS or config.epochs < FULL_PROTOCOL_EPOCHS
    run_kind = "smoke" if smoke_run else "full"
    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        print(f"\nFold {fold:02d}: train_end={train_end} test=[{test_start}, {test_end})")
        bounds = make_fold_bounds(fold, train_end, test_start, test_end, config)
        calendar_train = frame[frame["feat_idx"] < train_end]
        calendar_test = frame[
            (frame["feat_idx"] >= test_start) & (frame["feat_idx"] < test_end)
        ]
        if calendar_train.empty or calendar_test.empty:
            raise RuntimeError(f"Fold {fold} has empty calendar train or test rows.")
        calendar_bounds = {
            "calendar_train_start": pd.Timestamp(calendar_train["open_time"].min()).isoformat(),
            "calendar_train_end": pd.Timestamp(calendar_train["open_time"].max()).isoformat(),
            "calendar_test_start": pd.Timestamp(calendar_test["open_time"].min()).isoformat(),
            "calendar_test_end": pd.Timestamp(calendar_test["open_time"].max()).isoformat(),
        }
        if pd.Timestamp(calendar_bounds["calendar_train_end"]) >= pd.Timestamp(
            calendar_bounds["calendar_test_start"]
        ):
            raise RuntimeError(f"Fold {fold} fails global calendar separation.")
        bounds_dict = {**asdict(bounds), **calendar_bounds}
        if checkpoint_exists(run_dir, fold):
            if not args.resume:
                raise RuntimeError(
                    f"Completed checkpoint exists for fold {fold}. Re-run with --resume or use a new --run-name."
                )
            checkpoint_frames = load_fold_checkpoint(
                run_dir, fold, protocol_hash, bounds_dict
            )
            absorb_fold_frames(
                checkpoint_frames,
                prediction_parts,
                test_assignments,
                implementation_rows,
                manifest_rows,
                loss_rows,
                coverage_rows,
            )
            print(f"Fold {fold:02d}: validated and loaded completed checkpoint.")
            continue
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        if not train_ids or not test_ids:
            raise RuntimeError(f"Fold {fold} has empty train or test rows.")

        fold_predictions: list[pd.DataFrame] = []
        fold_assignments: list[pd.DataFrame] = []
        fold_implementation_rows: list[dict] = []
        fold_manifest_rows: list[dict] = []
        fold_loss_rows: list[dict] = []
        fold_coverage_rows: list[dict] = []

        vanilla = fit_fold_encoder("vanilla", raw_matrices, bounds, config)
        guided = fit_fold_encoder("guided", raw_matrices, bounds, config)
        fold_frame = frame[frame["row_id"].isin(train_ids + test_ids)].copy()
        vanilla_embeddings = encode_causal_rows(
            vanilla.model,
            raw_matrices,
            vanilla.scaler,
            fold_frame,
            config.device,
            total_rows=len(frame),
        )
        guided_embeddings = encode_causal_rows(
            guided.model,
            raw_matrices,
            guided.scaler,
            fold_frame,
            config.device,
            total_rows=len(frame),
        )

        fold_dir = run_dir / "weights" / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(vanilla.model.state_dict(), fold_dir / "vanilla_encoder.pt")
        torch.save(guided.model.state_dict(), fold_dir / "guided_encoder.pt")
        (fold_dir / "bounds.json").write_text(
            json.dumps(bounds_dict, indent=2, sort_keys=True), encoding="utf-8"
        )

        for result in [vanilla, guided]:
            fold_manifest_rows.append(
                {
                    "fold": fold,
                    "encoder_method": result.method,
                    **asdict(bounds),
                    **calendar_bounds,
                    "seed": config.seed,
                    "run_kind": run_kind,
                    "window_cap": config.max_windows,
                    "selected_epoch": result.selected_epoch,
                    "max_epochs": config.epochs,
                    "train_windows": result.train_windows,
                    "validation_windows": result.validation_windows,
                    "scaler_training_rows": result.scaler.training_rows,
                    "device": config.device,
                    "runtime_seconds": result.runtime_seconds,
                    "input_hash": result.input_hash,
                    "model_hash": result.model_hash,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "python_version": platform.python_version(),
                    "torch_version": torch.__version__,
                }
            )
            for stage, history in [
                ("inner_selection", result.selection_history),
                ("outer_refit", result.refit_history),
            ]:
                history = history.copy()
                history.insert(0, "stage", stage)
                history.insert(0, "encoder_method", result.method)
                history.insert(0, "fold", fold)
                fold_loss_rows.extend(history.to_dict("records"))

        fold_outputs = fit_fold_assignments(
            df_by_row,
            vanilla_embeddings,
            raw_matrix,
            hmm_matrix,
            train_ids,
            test_ids,
            fold,
            guided_embeddings=guided_embeddings,
        )
        global_prediction = run_global_fold(df_by_row, train_ids, test_ids, fold)
        fold_predictions.append(global_prediction)
        fold_coverage_rows.append(
            {
                "fold": fold,
                "method": "global_lgbm",
                "train_rows": len(train_ids),
                "test_assignment_rows": len(test_ids),
                "test_prediction_rows": len(global_prediction),
            }
        )

        for output in fold_outputs:
            fold_implementation_rows.append(
                {
                    "fold": fold,
                    "method": output.method,
                    "implementation": output.implementation,
                    "train_rows": int((output.assignments["split"] == "train").sum()),
                    "test_rows": int((output.assignments["split"] == "test").sum()),
                }
            )
            fold_assignments.append(output.assignments[output.assignments["split"] == "test"])
            models = train_regime_models(df_by_row, output.assignments, train_ids)
            prediction = predict_regime_models(
                df_by_row, output.assignments, models, test_ids, output.method, fold
            )
            if prediction is None:
                raise RuntimeError(f"No prediction for fold {fold} method {output.method}.")
            method_name = f"regime_lgbm_{output.method}"
            fold_predictions.append(prediction)
            fold_coverage_rows.append(
                {
                    "fold": fold,
                    "method": method_name,
                    "train_rows": int((output.assignments["split"] == "train").sum()),
                    "test_assignment_rows": int((output.assignments["split"] == "test").sum()),
                    "test_prediction_rows": len(prediction),
                }
            )

        checkpoint_frames = {
            "predictions": pd.concat(fold_predictions, ignore_index=True),
            "assignments": pd.concat(fold_assignments, ignore_index=True),
            "implementations": pd.DataFrame(fold_implementation_rows),
            "manifest": pd.DataFrame(fold_manifest_rows),
            "losses": pd.DataFrame(fold_loss_rows),
            "coverage": pd.DataFrame(fold_coverage_rows),
        }
        checkpoint_path = write_fold_checkpoint(
            run_dir,
            fold,
            protocol_hash,
            bounds_dict,
            checkpoint_frames,
        )
        absorb_fold_frames(
            checkpoint_frames,
            prediction_parts,
            test_assignments,
            implementation_rows,
            manifest_rows,
            loss_rows,
            coverage_rows,
        )
        print(f"Fold {fold:02d}: checkpoint committed atomically at {checkpoint_path}.")

    outputs = []
    for method in sorted(EXPECTED_METHODS):
        prediction = pd.concat(prediction_parts[method], ignore_index=True)
        regime_method = "none" if method == "global_lgbm" else method.removeprefix("regime_lgbm_")
        outputs.append(PredictionSet(method, regime_method, prediction))
    predictions = pd.concat([output.frame for output in outputs], ignore_index=True)
    predictions = predictions.sort_values(["method", "fold", "open_time", "symbol"]).reset_index(drop=True)
    results = pd.DataFrame(
        [
            summarize_predictions(
                output.frame, output.method, output.regime_method, "+".join(symbols)
            )
            for output in outputs
        ]
    )
    fold_metric_rows = []
    for output in outputs:
        for fold, fold_predictions in output.frame.groupby("fold", sort=True):
            row = summarize_predictions(
                fold_predictions, output.method, output.regime_method, "+".join(symbols)
            )
            row["fold"] = int(fold)
            fold_metric_rows.append(row)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    validate_complete_coverage(results, predictions, len(folds))

    assignments = pd.concat(test_assignments, ignore_index=True)
    implementations = pd.DataFrame(implementation_rows)
    regime_summary = summarize_assignments(assignments, implementations)
    manifest = pd.DataFrame(manifest_rows)
    losses = pd.DataFrame(loss_rows)
    coverage = pd.DataFrame(coverage_rows)
    comparison = build_historical_comparison(results)
    guided_comparison = build_guided_alpha_comparison(results)

    manifest.to_csv(output_path(args.output_dir, args.output_prefix, "encoder_manifest"), index=False)
    losses.to_csv(output_path(args.output_dir, args.output_prefix, "encoder_loss"), index=False)
    coverage.to_csv(output_path(args.output_dir, args.output_prefix, "encoder_coverage"), index=False)
    results.to_csv(output_path(args.output_dir, args.output_prefix, "experiment_results"), index=False)
    fold_metrics.to_csv(output_path(args.output_dir, args.output_prefix, "fold_metrics"), index=False)
    comparison.to_csv(output_path(args.output_dir, args.output_prefix, "method_comparison"), index=False)
    regime_summary.to_csv(output_path(args.output_dir, args.output_prefix, "regime_summary"), index=False)
    guided_comparison.to_csv(output_path(args.output_dir, args.output_prefix, "guided_comparison"), index=False)
    predictions.to_csv(output_path(args.output_dir, args.output_prefix, "alpha_oos_predictions"), index=False)
    assignments.to_csv(output_path(args.output_dir, args.output_prefix, "regime_assignments"), index=False)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_results_report(
        report_path,
        results,
        manifest,
        len(folds),
        symbols,
        smoke=smoke_run,
        data_role=str(freeze_manifest.get("data_role", "development_observed")),
    )
    run_metadata = {
        "args": vars(args),
        "config": asdict(config),
        "symbols": symbols,
        "folds": len(folds),
        "expected_methods": sorted(EXPECTED_METHODS),
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "protocol_hash": protocol_hash,
        "data_hash": data_hash,
        "source_hash": source_hash,
        "fold_hash": fold_hash,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nPhase 39 fold-local experiment results:")
    print(results.to_string(index=False))
    print(f"\nSaved compact report: {report_path}")
    print("OK: Phase 39 fold-local validity and equal-coverage checks passed.")


if __name__ == "__main__":
    main()
