from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_models import (
    EMBARGO,
    HORIZON_HOURS,
    INITIAL_TRAIN,
    PRIMARY_RETURN,
    PRIMARY_TARGET,
    STEP_SIZE,
    fold_ranges,
    row_ids_for_fold,
)
from config import BASE_DIR, DB_PATH, FEATURE_COLS, SAVE_DIR, WINDOW_SIZE
from fold_checkpoint import sha256_file
from fold_local_encoder_walkforward import (
    build_common_frame,
    hash_experiment_data,
    load_raw_matrices,
)


DEFAULT_CONFIG = Path(BASE_DIR) / "configs" / "crypto20_development_freeze_v1.json"
SUMMARY_PATH = Path(SAVE_DIR) / "crypto20_development_freeze_manifest.json"
SYMBOL_PATH = Path(SAVE_DIR) / "crypto20_development_symbol_manifest.csv"
FOLD_PATH = Path(SAVE_DIR) / "crypto20_development_fold_calendar.csv"
UNIVERSE_PATH = Path(SAVE_DIR) / "crypto20_development_universe_frozen.csv"
REPORT_PATH = Path(BASE_DIR) / "reports" / "crypto20_development_data_freeze.md"
SOURCE_UNIVERSE_PATH = Path(SAVE_DIR) / "asset_universe_crypto20.csv"
SUMMARY_DATABASE_PROVENANCE_FIELDS = {"database_sha256"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the calendar-aligned Crypto-20 development dataset."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    hashes = pd.util.hash_pandas_object(
        frame[columns], index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "freeze_id",
        "data_role",
        "symbols",
        "feature_context_start",
        "prediction_start",
        "prediction_end",
        "expected_feature_context_rows_per_symbol",
        "expected_prediction_rows_per_symbol",
        "expected_folds",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Freeze configuration is missing fields: {missing}")
    symbols = [str(symbol).upper() for symbol in config["symbols"]]
    if len(symbols) != len(set(symbols)) or len(symbols) != 20:
        raise ValueError("Development freeze must contain exactly 20 unique symbols.")
    config["symbols"] = symbols
    return config


def validate_protocol_constants(config: dict) -> None:
    observed = {
        "window_size": WINDOW_SIZE,
        "initial_train_bars": INITIAL_TRAIN,
        "step_bars": STEP_SIZE,
        "embargo_bars": EMBARGO,
        "label_horizon_bars": HORIZON_HOURS,
        "primary_target": PRIMARY_TARGET,
        "primary_return": PRIMARY_RETURN,
    }
    drift = {
        key: {"expected": config.get(key), "observed": value}
        for key, value in observed.items()
        if config.get(key) != value
    }
    if drift:
        raise RuntimeError(f"Frozen protocol constants drifted: {drift}")


def build_symbol_manifest(
    frame: pd.DataFrame, matrices: dict[str, np.ndarray], config: dict
) -> pd.DataFrame:
    rows = []
    lineage_columns = [
        "open_time",
        "source_feat_idx",
        "feat_idx",
        "target_class",
        "target_return",
        *FEATURE_COLS,
    ]
    for order, symbol in enumerate(config["symbols"], start=1):
        group = frame[frame["symbol"] == symbol].sort_values("feat_idx").reset_index(drop=True)
        matrix = matrices[symbol]
        gaps = group["open_time"].diff().dropna()
        rows.append(
            {
                "freeze_id": config["freeze_id"],
                "data_role": config["data_role"],
                "universe_order": order,
                "symbol": symbol,
                "feature_context_start": pd.Timestamp(config["feature_context_start"]).isoformat(),
                "prediction_start": pd.Timestamp(group["open_time"].min()).isoformat(),
                "prediction_end": pd.Timestamp(group["open_time"].max()).isoformat(),
                "feature_context_rows": int(len(matrix)),
                "prediction_rows": int(len(group)),
                "gap_count": int((gaps != pd.Timedelta(hours=1)).sum()),
                "feature_matrix_sha256": array_hash(matrix),
                "prediction_frame_sha256": frame_hash(group, lineage_columns),
            }
        )
    manifest = pd.DataFrame(rows)
    expected_context = int(config["expected_feature_context_rows_per_symbol"])
    expected_prediction = int(config["expected_prediction_rows_per_symbol"])
    failures = []
    if not (manifest["feature_context_rows"] == expected_context).all():
        failures.append("feature context row counts drifted")
    if not (manifest["prediction_rows"] == expected_prediction).all():
        failures.append("prediction row counts drifted")
    if not (manifest["gap_count"] == 0).all():
        failures.append("hourly gaps were found")
    if set(manifest["prediction_start"]) != {
        pd.Timestamp(config["prediction_start"]).isoformat()
    }:
        failures.append("prediction start drifted")
    if set(manifest["prediction_end"]) != {
        pd.Timestamp(config["prediction_end"]).isoformat()
    }:
        failures.append("prediction end drifted")
    if failures:
        raise RuntimeError("; ".join(failures))
    return manifest


def build_fold_manifest(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    indexed = frame.set_index("row_id", drop=False)
    folds = fold_ranges(frame)
    if len(folds) != int(config["expected_folds"]):
        raise RuntimeError(
            f"Fold count drifted: expected={config['expected_folds']} observed={len(folds)}"
        )
    rows = []
    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, test_ids = row_ids_for_fold(
            indexed, train_end, test_start, test_end
        )
        latest_train = pd.Timestamp(indexed.loc[train_ids, "open_time"].max())
        earliest_test = pd.Timestamp(indexed.loc[test_ids, "open_time"].min())
        rows.append(
            {
                "freeze_id": config["freeze_id"],
                "fold": fold,
                "train_end_index": train_end,
                "test_start_index": test_start,
                "test_end_index": test_end,
                "latest_train": latest_train.isoformat(),
                "earliest_test": earliest_test.isoformat(),
                "latest_test": pd.Timestamp(
                    indexed.loc[test_ids, "open_time"].max()
                ).isoformat(),
                "calendar_gap_hours": float(
                    (earliest_test - latest_train) / pd.Timedelta(hours=1)
                ),
                "train_rows": len(train_ids),
                "test_rows": len(test_ids),
            }
        )
    return pd.DataFrame(rows)


def build_frozen_universe(config: dict) -> pd.DataFrame:
    source = pd.read_csv(SOURCE_UNIVERSE_PATH)
    source["symbol"] = source["symbol"].astype(str).str.upper()
    indexed = source.drop_duplicates("symbol").set_index("symbol")
    missing = sorted(set(config["symbols"]) - set(indexed.index))
    if missing:
        raise RuntimeError(f"Source universe is missing frozen symbols: {missing}")
    frozen = indexed.loc[config["symbols"]].reset_index()
    frozen.insert(0, "universe_order", np.arange(1, len(frozen) + 1))
    frozen.insert(0, "data_role", config["data_role"])
    frozen.insert(0, "freeze_id", config["freeze_id"])
    frozen["selection_provenance"] = config["selection_provenance"]
    return frozen


def build_artifacts(config_path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    validate_protocol_constants(config)
    frame = build_common_frame(config["symbols"])
    matrices = load_raw_matrices(config["symbols"], frame)
    symbols = build_symbol_manifest(frame, matrices, config)
    folds = build_fold_manifest(frame, config)
    universe = build_frozen_universe(config)
    summary = {
        "schema_version": 1,
        "freeze_id": config["freeze_id"],
        "freeze_date": config["freeze_date"],
        "data_role": config["data_role"],
        "selection_provenance": config["selection_provenance"],
        "symbols": config["symbols"],
        "symbol_count": len(config["symbols"]),
        "feature_context_start": config["feature_context_start"],
        "prediction_start": config["prediction_start"],
        "prediction_end": config["prediction_end"],
        "prediction_rows": int(len(frame)),
        "fold_count": int(len(folds)),
        "config_sha256": sha256_file(config_path),
        "source_universe_sha256": sha256_file(SOURCE_UNIVERSE_PATH),
        "database_sha256": sha256_file(Path(DB_PATH)),
        "experiment_data_sha256": hash_experiment_data(
            config["symbols"], matrices, frame
        ),
        "symbol_manifest_sha256": canonical_hash(symbols.to_dict("records")),
        "fold_calendar_sha256": canonical_hash(folds.to_dict("records")),
        "frozen_universe_sha256": canonical_hash(universe.to_dict("records")),
        "evaluation_status": "development_only_not_locked_test",
    }
    return summary, symbols, folds, universe


def report_text(summary: dict) -> str:
    return f"""# Crypto-20 Development Data Freeze

## Status

`{summary['freeze_id']}` is an immutable **development-observed** dataset snapshot. It is not an untouched final test and is not claimed to be a historical top-liquidity universe.

## Frozen Scope

- Assets: {summary['symbol_count']}
- Feature context start: `{summary['feature_context_start']}`
- Prediction start: `{summary['prediction_start']}`
- Prediction end: `{summary['prediction_end']}`
- Prediction rows: {summary['prediction_rows']:,}
- Walk-forward folds: {summary['fold_count']}

## Provenance

{summary['selection_provenance']}

## Integrity

- Database SHA-256: `{summary['database_sha256']}`
- Experiment-data SHA-256: `{summary['experiment_data_sha256']}`
- Symbol-manifest SHA-256: `{summary['symbol_manifest_sha256']}`
- Fold-calendar SHA-256: `{summary['fold_calendar_sha256']}`

The database SHA-256 is file-level provenance for the original freeze moment. In later phases the shared local database may contain additional data roles, so verification enforces the frozen experiment-data, symbol, fold, universe, protocol, and configuration hashes rather than unrelated database-file growth. Any change to the frozen asset slice, timestamp range, row count, protocol, or experiment hash creates a different dataset version and must not be silently resumed under this freeze ID.
"""


def comparable_summary_for_verification(summary: dict) -> dict:
    return {
        key: value
        for key, value in summary.items()
        if key not in SUMMARY_DATABASE_PROVENANCE_FIELDS
    }


def verify_existing(
    summary: dict, symbols: pd.DataFrame, folds: pd.DataFrame, universe: pd.DataFrame
) -> None:
    required = [SUMMARY_PATH, SYMBOL_PATH, FOLD_PATH, UNIVERSE_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Frozen artifacts are missing: {missing}")
    existing_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if comparable_summary_for_verification(existing_summary) != comparable_summary_for_verification(summary):
        raise RuntimeError("Frozen summary drifted from the current database or configuration.")
    if existing_summary.get("database_sha256") != summary.get("database_sha256"):
        print(
            "NOTE: database_sha256 changed because the shared local database now contains "
            "additional data roles. Frozen experiment-data, symbol, fold, and universe "
            "hashes still match."
        )
    comparisons = [
        (SYMBOL_PATH, symbols),
        (FOLD_PATH, folds),
        (UNIVERSE_PATH, universe),
    ]
    for path, expected in comparisons:
        observed = pd.read_csv(path)
        if canonical_hash(observed.to_dict("records")) != canonical_hash(
            expected.to_dict("records")
        ):
            raise RuntimeError(f"Frozen artifact drifted: {path}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    summary, symbols, folds, universe = build_artifacts(config_path)
    if args.verify_only:
        verify_existing(summary, symbols, folds, universe)
        print(
            f"OK: {summary['freeze_id']} matches its configuration, database, "
            "symbol manifest, and fold calendar."
        )
        return

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    symbols.to_csv(SYMBOL_PATH, index=False)
    folds.to_csv(FOLD_PATH, index=False)
    universe.to_csv(UNIVERSE_PATH, index=False)
    REPORT_PATH.write_text(report_text(summary), encoding="utf-8")
    print(
        f"OK: froze {summary['symbol_count']} development symbols, "
        f"{summary['prediction_rows']:,} prediction rows, and "
        f"{summary['fold_count']} calendar-safe folds as {summary['freeze_id']}."
    )


if __name__ == "__main__":
    main()
