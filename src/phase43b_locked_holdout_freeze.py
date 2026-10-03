from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_models import EMBARGO, HORIZON_HOURS, INITIAL_TRAIN, PRIMARY_RETURN, PRIMARY_TARGET, STEP_SIZE, fold_ranges
from config import BASE_DIR, FEATURE_COLS, SAVE_DIR, WINDOW_SIZE
from fold_local_encoder_walkforward import build_common_frame, hash_experiment_data, load_raw_matrices
from phase43b_locked_holdout_registration import MANIFEST_PATH as REGISTRATION_MANIFEST_PATH


FREEZE_ID = "phase43b-locked-external-holdout-v1"
REGISTRATION_ID = "phase43b-locked-holdout-registration-v1"
MANIFEST_PATH = Path(SAVE_DIR) / "phase43b_locked_holdout_freeze_manifest.json"
SYMBOL_PATH = Path(SAVE_DIR) / "phase43b_locked_holdout_symbol_manifest.csv"
FOLD_PATH = Path(SAVE_DIR) / "phase43b_locked_holdout_fold_calendar.csv"
UNIVERSE_PATH = Path(SAVE_DIR) / "phase43b_locked_holdout_universe_frozen.csv"
REPORT_PATH = Path(BASE_DIR) / "reports" / "phase43b_locked_holdout_data_freeze.md"


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    hashed = pd.util.hash_pandas_object(frame[columns], index=False, categorize=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def load_registered_symbols(path: Path = REGISTRATION_MANIFEST_PATH) -> list[str]:
    manifest = pd.read_csv(path)
    values = manifest.set_index("item")["value"].astype(str).to_dict()
    if values.get("registration_status") != "registered_ready":
        raise RuntimeError("Phase 43B holdout cannot be frozen until registration_status is registered_ready.")
    symbols = [symbol for symbol in values.get("selected_symbols", "").split(",") if symbol]
    if len(symbols) < 10:
        raise RuntimeError(f"Phase 43B holdout requires at least 10 symbols; observed={len(symbols)}.")
    return symbols


def build_symbol_manifest(frame: pd.DataFrame, matrices: dict[str, np.ndarray], symbols: list[str]) -> pd.DataFrame:
    rows = []
    lineage_columns = [
        "open_time",
        "source_feat_idx",
        "feat_idx",
        "target_class",
        "target_return",
        *FEATURE_COLS,
    ]
    for order, symbol in enumerate(symbols, start=1):
        group = frame[frame["symbol"] == symbol].sort_values("feat_idx").reset_index(drop=True)
        gaps = group["open_time"].diff().dropna()
        rows.append(
            {
                "freeze_id": FREEZE_ID,
                "registration_id": REGISTRATION_ID,
                "data_role": "locked_registered_unobserved",
                "universe_order": order,
                "symbol": symbol,
                "prediction_start": pd.Timestamp(group["open_time"].min()).isoformat(),
                "prediction_end": pd.Timestamp(group["open_time"].max()).isoformat(),
                "feature_context_rows": int(len(matrices[symbol])),
                "prediction_rows": int(len(group)),
                "gap_count": int((gaps != pd.Timedelta(hours=1)).sum()),
                "feature_matrix_sha256": array_hash(matrices[symbol]),
                "prediction_frame_sha256": frame_hash(group, lineage_columns),
            }
        )
    return pd.DataFrame(rows)


def build_fold_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index("row_id", drop=False)
    rows = []
    for fold, (train_end, test_start, test_end) in enumerate(fold_ranges(frame), start=1):
        train = indexed[indexed["feat_idx"] < train_end]
        test = indexed[(indexed["feat_idx"] >= test_start) & (indexed["feat_idx"] < test_end)]
        rows.append(
            {
                "freeze_id": FREEZE_ID,
                "fold": fold,
                "train_end_index": int(train_end),
                "test_start_index": int(test_start),
                "test_end_index": int(test_end),
                "latest_train": pd.Timestamp(train["open_time"].max()).isoformat(),
                "earliest_test": pd.Timestamp(test["open_time"].min()).isoformat(),
                "latest_test": pd.Timestamp(test["open_time"].max()).isoformat(),
                "calendar_gap_hours": float(
                    (pd.Timestamp(test["open_time"].min()) - pd.Timestamp(train["open_time"].max()))
                    / pd.Timedelta(hours=1)
                ),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            }
        )
    return pd.DataFrame(rows)


def build_universe(symbol_manifest: pd.DataFrame) -> pd.DataFrame:
    registered = pd.read_csv(Path(SAVE_DIR) / "phase43b_registered_holdout_symbols.csv")
    registered["symbol"] = registered["symbol"].astype(str).str.upper()
    frozen = registered.merge(
        symbol_manifest[["symbol", "prediction_start", "prediction_end", "prediction_rows"]],
        on="symbol",
        how="left",
    )
    frozen.insert(0, "freeze_id", FREEZE_ID)
    frozen["data_role"] = "locked_registered_unobserved"
    frozen["selection_provenance"] = "Phase 43B pre-outcome registration; selected by design rank and quality gates only."
    return frozen


def report_text(summary: dict, symbols: pd.DataFrame, folds: pd.DataFrame) -> str:
    symbol_rows = "\n".join(
        f"| {row.universe_order} | `{row.symbol}` | {row.prediction_rows} | {row.prediction_start} | {row.prediction_end} |"
        for row in symbols.itertuples(index=False)
    )
    return f"""# Phase 43B Locked Holdout Data Freeze

## Status

The Phase 43B external holdout is frozen before model evaluation.

- Freeze ID: `{summary['freeze_id']}`
- Registration ID: `{summary['registration_id']}`
- Data role: `{summary['data_role']}`
- Symbols: {len(summary['symbols'])}
- Prediction rows: {summary['prediction_rows']}
- Fold count: {summary['fold_count']}
- Experiment-data hash: `{summary['experiment_data_sha256']}`

No predictions, alpha metrics, method rankings, or threshold choices are used to create this freeze.

## Frozen Symbols

| Order | Symbol | Prediction rows | Start | End |
|---:|---|---:|---|---|
{symbol_rows}

## Fold Calendar

- Earliest test: `{folds['earliest_test'].min()}`
- Latest test: `{folds['latest_test'].max()}`
- Minimum calendar gap hours: {folds['calendar_gap_hours'].min():.1f}

## Interpretation

This artifact only freezes the locked-holdout data. The next step is the one-shot frozen evaluation. If that evaluation fails, the result must be reported without same-holdout tuning.
"""


def write_artifacts() -> None:
    symbols = load_registered_symbols()
    frame = build_common_frame(symbols)
    matrices = load_raw_matrices(symbols, frame)
    folds = fold_ranges(frame)
    data_hash = hash_experiment_data(symbols, matrices, frame)
    symbol_manifest = build_symbol_manifest(frame, matrices, symbols)
    fold_manifest = build_fold_manifest(frame)
    universe = build_universe(symbol_manifest)
    summary = {
        "schema_version": 1,
        "freeze_id": FREEZE_ID,
        "registration_id": REGISTRATION_ID,
        "data_role": "locked_registered_unobserved",
        "symbols": symbols,
        "experiment_data_sha256": data_hash,
        "prediction_rows": int(len(frame)),
        "fold_count": int(len(folds)),
        "window_size": WINDOW_SIZE,
        "initial_train_bars": INITIAL_TRAIN,
        "step_bars": STEP_SIZE,
        "embargo_bars": EMBARGO,
        "label_horizon_bars": HORIZON_HOURS,
        "primary_target": PRIMARY_TARGET,
        "primary_return": PRIMARY_RETURN,
    }
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    symbol_manifest.to_csv(SYMBOL_PATH, index=False)
    fold_manifest.to_csv(FOLD_PATH, index=False)
    universe.to_csv(UNIVERSE_PATH, index=False)
    REPORT_PATH.write_text(report_text(summary, symbol_manifest, fold_manifest), encoding="utf-8")
    print(f"Saved: {MANIFEST_PATH}")
    print(f"Saved: {SYMBOL_PATH}")
    print(f"Saved: {FOLD_PATH}")
    print(f"Saved: {UNIVERSE_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"OK: froze {len(symbols)} locked symbols with {len(frame):,} prediction rows and {len(folds)} folds.")


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Freeze Phase 43B registered locked holdout data.").parse_args()


def main() -> None:
    parse_args()
    write_artifacts()


if __name__ == "__main__":
    main()
