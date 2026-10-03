from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_models import (
    FEATURE_COLS,
    predict_regime_models,
    row_ids_for_fold,
    summarize_predictions,
    train_regime_models,
)
from baselines import HMM_FEATURES
from config import BASE_DIR, SAVE_DIR
from fold_checkpoint import (
    canonical_hash,
    checkpoint_exists,
    initialize_run_state,
    load_fold_checkpoint,
    sha256_file,
    write_fold_checkpoint,
)
from fold_local_encoder_walkforward import (
    build_common_frame,
    hash_experiment_data,
    load_raw_matrices,
    run_global_fold,
    verify_frozen_dataset,
)
from statistical_tests import fold_metrics
from walkforward_regimes import finite_matrix, fit_fold_assignments


EXPECTED_METHODS = {
    "global_lgbm",
    "regime_lgbm_hmm",
    "regime_lgbm_kmeans",
    "regime_lgbm_vol_bucket",
}
DEFAULT_FREEZE = Path(SAVE_DIR) / "crypto20_development_freeze_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the repaired, frozen, calendar-safe classical Crypto-20 baseline."
    )
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--run-name", default="phase39r_classical_full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--heavy-dir", default=str(Path(".tmp") / "phase39r_classical")
    )
    parser.add_argument("--output-dir", default=SAVE_DIR)
    parser.add_argument("--output-prefix", default="crypto20_repaired_classical_")
    parser.add_argument("--freeze-manifest", default=str(DEFAULT_FREEZE))
    parser.add_argument(
        "--finalize-from-checkpoints",
        action="store_true",
        help=(
            "Load and aggregate an already completed checkpoint run without retraining. "
            "This preserves the checkpoint protocol hash while allowing evaluation-only "
            "finalization fixes."
        ),
    )
    return parser.parse_args()


def source_hash() -> str:
    paths = [
        Path(__file__),
        Path(__file__).with_name("alpha_models.py"),
        Path(__file__).with_name("evaluation.py"),
        Path(__file__).with_name("walkforward_regimes.py"),
        Path(__file__).with_name("fold_checkpoint.py"),
    ]
    return canonical_hash({path.name: sha256_file(path) for path in paths})


def output_path(args: argparse.Namespace, stem: str) -> Path:
    path = Path(args.output_dir) / f"{args.output_prefix}{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def calendar_bounds(
    frame: pd.DataFrame, train_end: int, test_start: int, test_end: int
) -> dict[str, str]:
    train = frame[frame["feat_idx"] < train_end]
    test = frame[(frame["feat_idx"] >= test_start) & (frame["feat_idx"] < test_end)]
    latest_train = pd.Timestamp(train["open_time"].max())
    earliest_test = pd.Timestamp(test["open_time"].min())
    if latest_train >= earliest_test:
        raise RuntimeError("Classical baseline fold has global calendar overlap.")
    return {
        "calendar_train_start": pd.Timestamp(train["open_time"].min()).isoformat(),
        "calendar_train_end": latest_train.isoformat(),
        "calendar_test_start": earliest_test.isoformat(),
        "calendar_test_end": pd.Timestamp(test["open_time"].max()).isoformat(),
    }


def validate_fold_frames(frames: dict[str, pd.DataFrame], fold: int) -> None:
    predictions = frames["predictions"]
    coverage = frames["coverage"]
    if set(predictions["method"].astype(str)) != EXPECTED_METHODS:
        raise RuntimeError(f"Fold {fold} prediction method set is incomplete.")
    if set(coverage["method"].astype(str)) != EXPECTED_METHODS:
        raise RuntimeError(f"Fold {fold} coverage method set is incomplete.")
    counts = predictions.groupby("method").size()
    if counts.nunique() != 1:
        raise RuntimeError(f"Fold {fold} has unequal prediction coverage: {counts.to_dict()}")
    if predictions.duplicated(["method", "symbol", "feat_idx", "fold"]).any():
        raise RuntimeError(f"Fold {fold} contains duplicate predictions.")


def main() -> None:
    args = parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        raise ValueError("--run-name must be one safe directory name.")

    freeze = json.loads(Path(args.freeze_manifest).read_text(encoding="utf-8"))
    symbols = list(freeze["symbols"])
    frame = build_common_frame(symbols)
    indexed = frame.set_index("row_id", drop=False)
    matrices = load_raw_matrices(symbols, frame)
    from alpha_models import fold_ranges

    all_folds = fold_ranges(frame)
    data_hash = hash_experiment_data(symbols, matrices, frame)
    verified_freeze = verify_frozen_dataset(
        args.freeze_manifest, symbols, frame, all_folds, data_hash
    )
    folds = all_folds[: args.max_folds] if args.max_folds is not None else all_folds
    if not folds:
        raise RuntimeError("No folds selected for the classical baseline.")

    raw_matrix = finite_matrix(indexed.sort_index(), FEATURE_COLS)
    hmm_matrix = finite_matrix(indexed.sort_index(), HMM_FEATURES)
    fold_specification = [list(map(int, values)) for values in folds]
    source_lineage = source_hash()
    config_hash = canonical_hash(
        {
            "freeze_id": verified_freeze["freeze_id"],
            "freeze_manifest_sha256": sha256_file(Path(args.freeze_manifest)),
            "methods": sorted(EXPECTED_METHODS),
            "folds": fold_specification,
        }
    )
    protocol_hash = canonical_hash(
        {
            "config_hash": config_hash,
            "data_hash": data_hash,
            "source_hash": source_lineage,
        }
    )
    run_dir = Path(args.heavy_dir) / args.run_name
    run_state = {
        "schema_version": 1,
        "protocol_hash": protocol_hash,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "source_hash": source_lineage,
        "freeze_id": verified_freeze["freeze_id"],
        "symbols": symbols,
        "folds": fold_specification,
        "methods": sorted(EXPECTED_METHODS),
    }
    if args.finalize_from_checkpoints:
        state_path = run_dir / "run_state.json"
        if not state_path.exists():
            raise RuntimeError(
                f"Cannot finalize from checkpoints because {state_path} is missing."
            )
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        mismatches = [
            key
            for key in ["config_hash", "data_hash", "freeze_id", "symbols", "folds", "methods"]
            if saved_state.get(key) != run_state.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "Checkpoint finalization rejected because immutable run inputs changed: "
                f"{mismatches}."
            )
        checkpoint_protocol_hash = saved_state["protocol_hash"]
        checkpoint_source_hash = saved_state.get("source_hash")
    else:
        saved_state = initialize_run_state(
            run_dir,
            run_state,
            resume=args.resume,
        )
        checkpoint_protocol_hash = saved_state["protocol_hash"]
        checkpoint_source_hash = saved_state.get("source_hash")

    prediction_parts = []
    assignment_parts = []
    implementation_parts = []
    manifest_parts = []
    coverage_parts = []
    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        bounds = {
            "fold": fold,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            **calendar_bounds(frame, train_end, test_start, test_end),
        }
        if checkpoint_exists(run_dir, fold):
            if not (args.resume or args.finalize_from_checkpoints):
                raise RuntimeError(
                    f"Checkpoint exists for fold {fold}; use --resume or a new run name."
                )
            frames = load_fold_checkpoint(
                run_dir, fold, checkpoint_protocol_hash, bounds
            )
            validate_fold_frames(frames, fold)
            print(f"Fold {fold:02d}: validated and loaded checkpoint.")
        else:
            if args.finalize_from_checkpoints:
                raise RuntimeError(
                    f"Cannot finalize from checkpoints because fold {fold} is missing."
                )
            train_ids, test_ids = row_ids_for_fold(
                indexed, train_end, test_start, test_end
            )
            predictions = [run_global_fold(indexed, train_ids, test_ids, fold)]
            assignments = []
            implementations = []
            coverage = [
                {
                    "fold": fold,
                    "method": "global_lgbm",
                    "train_rows": len(train_ids),
                    "test_rows": len(test_ids),
                }
            ]
            outputs = fit_fold_assignments(
                indexed,
                None,
                raw_matrix,
                hmm_matrix,
                train_ids,
                test_ids,
                fold,
            )
            for output in outputs:
                test_assignments = output.assignments[
                    output.assignments["split"] == "test"
                ].copy()
                assignments.append(test_assignments)
                implementations.append(
                    {
                        "fold": fold,
                        "method": output.method,
                        "implementation": output.implementation,
                    }
                )
                models = train_regime_models(
                    indexed, output.assignments, train_ids
                )
                prediction = predict_regime_models(
                    indexed,
                    output.assignments,
                    models,
                    test_ids,
                    output.method,
                    fold,
                )
                if prediction is None:
                    raise RuntimeError(
                        f"Fold {fold} produced no prediction for {output.method}."
                    )
                predictions.append(prediction)
                coverage.append(
                    {
                        "fold": fold,
                        "method": f"regime_lgbm_{output.method}",
                        "train_rows": len(train_ids),
                        "test_rows": len(prediction),
                    }
                )
            frames = {
                "predictions": pd.concat(predictions, ignore_index=True),
                "assignments": pd.concat(assignments, ignore_index=True),
                "implementations": pd.DataFrame(implementations),
                "manifest": pd.DataFrame(
                    [
                        {
                            **bounds,
                            "freeze_id": verified_freeze["freeze_id"],
                            "data_hash": data_hash,
                            "source_hash": source_lineage,
                        }
                    ]
                ),
                "losses": pd.DataFrame(
                    [{"fold": fold, "stage": "not_applicable_classical"}]
                ),
                "coverage": pd.DataFrame(coverage),
            }
            validate_fold_frames(frames, fold)
            write_fold_checkpoint(
                run_dir, fold, checkpoint_protocol_hash, bounds, frames
            )
            print(f"Fold {fold:02d}: classical checkpoint complete.")

        prediction_parts.append(frames["predictions"])
        assignment_parts.append(frames["assignments"])
        implementation_parts.append(frames["implementations"])
        manifest_parts.append(frames["manifest"])
        coverage_parts.append(frames["coverage"])

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["method", "fold", "open_time", "symbol"]
    )
    coverage = pd.concat(coverage_parts, ignore_index=True)
    if predictions.groupby("method").size().nunique() != 1:
        raise RuntimeError("Complete classical run has unequal method coverage.")
    results = pd.DataFrame(
        [
            summarize_predictions(
                group,
                method,
                str(group["regime_method"].iloc[0]),
                "+".join(symbols),
            )
            for method, group in predictions.groupby("method", sort=False)
        ]
    )
    folds_summary = fold_metrics(predictions)
    manifest = pd.concat(manifest_parts, ignore_index=True)
    implementations = pd.concat(implementation_parts, ignore_index=True)

    results.to_csv(output_path(args, "experiment_results"), index=False)
    folds_summary.to_csv(output_path(args, "fold_metrics"), index=False)
    coverage.to_csv(output_path(args, "coverage"), index=False)
    manifest.to_csv(output_path(args, "manifest"), index=False)
    implementations.to_csv(output_path(args, "implementations"), index=False)
    predictions.to_csv(run_dir / "oos_predictions.csv", index=False)
    pd.concat(assignment_parts, ignore_index=True).to_csv(
        run_dir / "test_assignments.csv", index=False
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "freeze_id": verified_freeze["freeze_id"],
                "folds": len(folds),
                "methods": sorted(EXPECTED_METHODS),
                "checkpoint_protocol_hash": checkpoint_protocol_hash,
                "checkpoint_source_hash": checkpoint_source_hash,
                "finalizer_protocol_hash": protocol_hash,
                "finalizer_source_hash": source_lineage,
                "finalize_from_checkpoints": bool(args.finalize_from_checkpoints),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(),
                "command": " ".join(sys.argv),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print("\nRepaired classical results (development smoke/full as configured):")
    print(results.to_string(index=False))
    print(f"OK: {len(folds)} fold(s), four methods, equal coverage, frozen data.")


if __name__ == "__main__":
    main()
