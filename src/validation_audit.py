import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from alpha_models import (
    EMBARGO,
    HORIZON_HOURS,
    PRIMARY_TARGET,
    REGIME_METHODS,
    STEP_SIZE,
    fold_ranges,
    load_assignments,
    load_model_frame,
    restrict_to_common_universe,
    row_ids_for_fold,
)
from config import BASE_DIR, DB_PATH, FEATURE_COLS, SAVE_DIR, SYMBOLS
from targets import HORIZONS


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
GUIDED_REGIME_METHODS = ["hmm_guided_gmm", "hmm_guided_hmm"]
TIME_FREQUENCY_METHODS = ["tf_hmm_guided_gmm", "tf_hmm_guided_hmm"]
FOLD_LOCAL_REGIME_METHODS = REGIME_METHODS + GUIDED_REGIME_METHODS
FOLD_LOCAL_METHODS = ["global_lgbm"] + [f"regime_lgbm_{method}" for method in FOLD_LOCAL_REGIME_METHODS]


@dataclass
class AuditRecord:
    check: str
    status: str
    severity: str
    rows_checked: int
    rows_failed: int
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit leakage, embargo, target horizon, and benchmark coverage assumptions."
    )
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    return parser.parse_args()


def record(
    rows: list[AuditRecord],
    check: str,
    status: str,
    severity: str,
    detail: str,
    rows_checked: int = 0,
    rows_failed: int = 0,
) -> None:
    rows.append(
        AuditRecord(
            check=check,
            status=status,
            severity=severity,
            rows_checked=int(rows_checked),
            rows_failed=int(rows_failed),
            detail=detail,
        )
    )


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [name],
        ).fetchone()[0]
    )


def audit_database_tables(rows: list[AuditRecord], symbols: list[str]) -> None:
    if not os.path.exists(DB_PATH):
        record(rows, "database_exists", FAIL, "critical", f"Missing DuckDB database: {DB_PATH}")
        return

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        required = ["ohlcv", "features", "targets"]
        missing = [table for table in required if not table_exists(con, table)]
        record(
            rows,
            "required_tables_exist",
            FAIL if missing else PASS,
            "critical",
            f"Missing tables: {missing}" if missing else "ohlcv, features, and targets tables exist.",
        )
        if missing:
            return

        for table in required:
            counts = con.execute(
                f"""
                SELECT symbol, COUNT(*) AS n_rows
                FROM {table}
                WHERE symbol IN ({",".join(["?"] * len(symbols))})
                GROUP BY symbol
                ORDER BY symbol
                """,
                symbols,
            ).df()
            absent = sorted(set(symbols) - set(counts["symbol"].tolist()))
            zero_symbols = counts[counts["n_rows"] <= 0]["symbol"].tolist()
            failed = len(absent) + len(zero_symbols)
            detail = (
                f"{table} counts: "
                + ", ".join(f"{r.symbol}={int(r.n_rows)}" for r in counts.itertuples())
            )
            if absent:
                detail += f"; absent={absent}"
            record(
                rows,
                f"{table}_symbol_rows",
                FAIL if failed else PASS,
                "critical",
                detail,
                rows_checked=len(symbols),
                rows_failed=failed,
            )
    finally:
        con.close()


def audit_feature_source(rows: list[AuditRecord]) -> None:
    path = Path(BASE_DIR) / "src" / "features.py"
    text = path.read_text(encoding="utf-8")
    future_patterns = [".shift(-", "center=True"]
    found = [pattern for pattern in future_patterns if pattern in text]
    record(
        rows,
        "feature_source_no_obvious_future_ops",
        FAIL if found else PASS,
        "critical",
        f"Found future-looking feature patterns: {found}" if found else "No shift(-h) or centered rolling windows found in features.py.",
    )


def audit_feature_target_schema(rows: list[AuditRecord], symbols: list[str]) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        feature_cols = con.execute("DESCRIBE features").df()["column_name"].tolist()
        missing_features = sorted(set(FEATURE_COLS) - set(feature_cols))
        record(
            rows,
            "feature_columns_present",
            FAIL if missing_features else PASS,
            "critical",
            f"Missing feature columns: {missing_features}" if missing_features else f"All {len(FEATURE_COLS)} configured feature columns exist.",
            rows_checked=len(FEATURE_COLS),
            rows_failed=len(missing_features),
        )

        target_cols = con.execute("DESCRIBE targets").df()["column_name"].tolist()
        required_targets = [PRIMARY_TARGET, f"forward_return_{HORIZON_HOURS}h"]
        missing_targets = sorted(set(required_targets) - set(target_cols))
        record(
            rows,
            "primary_target_columns_present",
            FAIL if missing_targets else PASS,
            "critical",
            f"Missing target columns: {missing_targets}" if missing_targets else f"Primary target columns exist: {required_targets}.",
            rows_checked=len(required_targets),
            rows_failed=len(missing_targets),
        )

        joined = con.execute(
            f"""
            SELECT
                f.symbol,
                f.open_time,
                {", ".join("f." + c for c in FEATURE_COLS)},
                t.{PRIMARY_TARGET} AS target_label,
                t.forward_return_{HORIZON_HOURS}h AS target_return
            FROM features f
            JOIN targets t
              ON f.symbol = t.symbol
             AND f.open_time = t.open_time
            WHERE f.symbol IN ({",".join(["?"] * len(symbols))})
            ORDER BY f.symbol, f.open_time
            """,
            symbols,
        ).df()
        joined["open_time"] = pd.to_datetime(joined["open_time"])
        joined = joined.reset_index(drop=True)
        joined["feat_idx"] = joined.groupby("symbol").cumcount().astype(int)
    finally:
        con.close()

    bad_feature_rows = joined[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    bad_target_rows = joined[["target_label", "target_return"]].replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    record(
        rows,
        "joined_model_rows_are_finite",
        FAIL if (bad_feature_rows.any() or bad_target_rows.any()) else PASS,
        "critical",
        "Joined feature/target rows contain no NaN/inf values."
        if not (bad_feature_rows.any() or bad_target_rows.any())
        else "NaN/inf detected in joined feature/target rows.",
        rows_checked=len(joined),
        rows_failed=int((bad_feature_rows | bad_target_rows).sum()),
    )
    return joined


def audit_target_horizon_loss(rows: list[AuditRecord], symbols: list[str]) -> None:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        counts = con.execute(
            f"""
            SELECT
                f.symbol,
                COUNT(*) AS feature_rows,
                COUNT(t.open_time) AS target_rows,
                MAX(f.open_time) AS feature_latest,
                MAX(t.open_time) AS target_latest
            FROM features f
            LEFT JOIN targets t
              ON f.symbol = t.symbol
             AND f.open_time = t.open_time
            WHERE f.symbol IN ({",".join(["?"] * len(symbols))})
            GROUP BY f.symbol
            ORDER BY f.symbol
            """,
            symbols,
        ).df()
    finally:
        con.close()

    expected_loss = max(HORIZONS)
    failures = 0
    details = []
    for row in counts.itertuples(index=False):
        loss = int(row.feature_rows - row.target_rows)
        ok = loss == expected_loss and pd.Timestamp(row.target_latest) < pd.Timestamp(row.feature_latest)
        failures += int(not ok)
        details.append(
            f"{row.symbol}: feature_rows={int(row.feature_rows)}, target_rows={int(row.target_rows)}, "
            f"lost={loss}, expected_loss={expected_loss}, target_latest={row.target_latest}"
        )
    record(
        rows,
        "target_horizon_tail_loss",
        FAIL if failures else PASS,
        "critical",
        "; ".join(details),
        rows_checked=len(counts),
        rows_failed=failures,
    )


def audit_common_universe(rows: list[AuditRecord], symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, int, int]]]:
    df = load_model_frame(symbols)
    assignments = load_assignments(symbols)
    df, assignments = restrict_to_common_universe(df, assignments)
    folds = fold_ranges(df)

    method_counts = assignments.groupby("method").size().reindex(REGIME_METHODS, fill_value=0)
    missing = method_counts[method_counts == 0].index.tolist()
    unequal = int(method_counts.nunique() > 1)
    record(
        rows,
        "regime_assignment_common_coverage",
        FAIL if missing or unequal else PASS,
        "critical",
        "Regime assignment rows by method: "
        + ", ".join(f"{method}={int(count)}" for method, count in method_counts.items()),
        rows_checked=len(method_counts),
        rows_failed=len(missing) + unequal,
    )

    record(
        rows,
        "walk_forward_folds_exist",
        PASS if folds else FAIL,
        "critical",
        f"{len(folds)} folds generated with initial_train, step, and embargo settings.",
        rows_checked=len(folds),
        rows_failed=0 if folds else 1,
    )

    record(
        rows,
        "regime_discovery_protocol",
        WARN,
        "methodological",
        "Legacy regime_assignments.csv is an offline/global artifact. Use walkforward_experiment_results.csv for fold-local predictive regime claims.",
        rows_checked=len(assignments),
        rows_failed=0,
    )
    return df, assignments, folds


def audit_folds(
    rows: list[AuditRecord],
    df: pd.DataFrame,
    folds: list[tuple[int, int, int]],
) -> pd.DataFrame:
    df_by_row = df.set_index("row_id", drop=False)
    fold_rows = []
    hard_failures = 0

    for fold, (train_end, test_start, test_end) in enumerate(folds, start=1):
        train_ids, test_ids = row_ids_for_fold(df_by_row, train_end, test_start, test_end)
        train = df_by_row.loc[train_ids]
        test = df_by_row.loc[test_ids]
        train_max_idx = int(train["feat_idx"].max()) if not train.empty else -1
        test_min_idx = int(test["feat_idx"].min()) if not test.empty else -1
        test_max_idx = int(test["feat_idx"].max()) if not test.empty else -1
        embargo_bars = test_min_idx - train_max_idx - 1
        train_label_end = train_max_idx + HORIZON_HOURS
        label_gap_bars = test_min_idx - train_label_end
        row_overlap = len(set(train_ids).intersection(test_ids))

        embargo_ok = embargo_bars >= EMBARGO
        label_overlap_ok = train_label_end < test_min_idx
        row_overlap_ok = row_overlap == 0
        fold_ok = embargo_ok and label_overlap_ok and row_overlap_ok
        hard_failures += int(not fold_ok)

        fold_rows.append(
            {
                "fold": fold,
                "train_end": int(train_end),
                "test_start": int(test_start),
                "test_end": int(test_end),
                "n_train_rows": int(len(train_ids)),
                "n_test_rows": int(len(test_ids)),
                "train_max_feat_idx": train_max_idx,
                "test_min_feat_idx": test_min_idx,
                "test_max_feat_idx": test_max_idx,
                "embargo_required_bars": int(EMBARGO),
                "embargo_observed_bars": int(embargo_bars),
                "primary_label_horizon_bars": int(HORIZON_HOURS),
                "train_label_end_feat_idx": int(train_label_end),
                "label_gap_before_test_bars": int(label_gap_bars),
                "row_overlap_count": int(row_overlap),
                "status": PASS if fold_ok else FAIL,
            }
        )

    record(
        rows,
        "fold_embargo_and_label_purge",
        FAIL if hard_failures else PASS,
        "critical",
        f"{len(folds) - hard_failures}/{len(folds)} folds satisfy row separation, {EMBARGO}-bar embargo, and {HORIZON_HOURS}-bar label-horizon purge.",
        rows_checked=len(folds),
        rows_failed=hard_failures,
    )
    return pd.DataFrame(fold_rows)


def audit_predictions(rows: list[AuditRecord], fold_audit: pd.DataFrame) -> None:
    path = Path(SAVE_DIR) / "alpha_oos_predictions.csv"
    results_path = Path(SAVE_DIR) / "experiment_results.csv"
    if not path.exists():
        record(
            rows,
            "oos_prediction_alignment",
            WARN,
            "artifact",
            "alpha_oos_predictions.csv is missing locally; run alpha_models.py before full row-level prediction audit.",
        )
        return

    pred = pd.read_csv(path)
    required = {"method", "regime_method", "symbol", "feat_idx", "fold", "score", "target_return"}
    missing = sorted(required - set(pred.columns))
    if missing:
        record(rows, "oos_prediction_schema", FAIL, "critical", f"Missing prediction columns: {missing}")
        return
    record(rows, "oos_prediction_schema", PASS, "critical", "alpha_oos_predictions.csv has required audit columns.")

    fold_bounds = fold_audit.set_index("fold")[["test_start", "test_end"]].to_dict("index")
    aligned = []
    for row in pred.itertuples(index=False):
        bounds = fold_bounds.get(int(row.fold))
        ok = bool(bounds and int(row.feat_idx) >= bounds["test_start"] and int(row.feat_idx) < bounds["test_end"])
        aligned.append(ok)
    aligned = np.asarray(aligned, dtype=bool)
    record(
        rows,
        "oos_prediction_rows_match_test_folds",
        FAIL if (~aligned).any() else PASS,
        "critical",
        "All prediction rows fall inside their recorded fold test windows."
        if aligned.all()
        else "Some prediction rows fall outside their recorded fold test windows.",
        rows_checked=len(pred),
        rows_failed=int((~aligned).sum()),
    )

    dupes = pred.duplicated(["method", "symbol", "feat_idx"]).sum()
    record(
        rows,
        "oos_prediction_no_duplicate_method_rows",
        FAIL if dupes else PASS,
        "critical",
        "No duplicated method/symbol/feat_idx predictions."
        if dupes == 0
        else f"Found {int(dupes)} duplicated method/symbol/feat_idx prediction rows.",
        rows_checked=len(pred),
        rows_failed=int(dupes),
    )

    method_rows = pred.groupby("method").size()
    unequal = int(method_rows.nunique() > 1)
    record(
        rows,
        "oos_prediction_equal_method_coverage",
        FAIL if unequal else PASS,
        "critical",
        "Prediction rows by method: " + ", ".join(f"{m}={int(n)}" for m, n in method_rows.items()),
        rows_checked=len(method_rows),
        rows_failed=unequal,
    )

    if results_path.exists():
        results = pd.read_csv(results_path)
        expected = results.set_index("method")["n_test_rows"].to_dict()
        mismatches = []
        for method, observed in method_rows.items():
            if int(expected.get(method, -1)) != int(observed):
                mismatches.append(f"{method}: pred={int(observed)}, results={expected.get(method)}")
        record(
            rows,
            "experiment_results_match_predictions",
            FAIL if mismatches else PASS,
            "critical",
            "experiment_results.csv n_test_rows matches alpha_oos_predictions.csv."
            if not mismatches
            else "; ".join(mismatches),
            rows_checked=len(method_rows),
            rows_failed=len(mismatches),
        )


def audit_walkforward_artifacts(rows: list[AuditRecord]) -> None:
    results_path = Path(SAVE_DIR) / "walkforward_experiment_results.csv"
    comparison_path = Path(SAVE_DIR) / "walkforward_comparison.csv"
    summary_path = Path(SAVE_DIR) / "walkforward_regime_summary.csv"
    guided_comparison_path = Path(SAVE_DIR) / "guided_alpha_comparison.csv"

    if not results_path.exists():
        record(
            rows,
            "fold_local_regime_refit_artifacts",
            WARN,
            "artifact",
            "walkforward_experiment_results.csv is missing; run walkforward_regimes.py for Phase 20 guided alpha retest results.",
        )
        return

    results = pd.read_csv(results_path)
    required_methods = FOLD_LOCAL_METHODS
    missing = sorted(set(required_methods) - set(results["method"]))
    row_counts = results.set_index("method")["n_test_rows"]
    unequal = int(row_counts.nunique() > 1)
    missing_files = [
        str(path.name)
        for path in [comparison_path, summary_path, guided_comparison_path]
        if not path.exists()
    ]
    failures = len(missing) + unequal + len(missing_files)

    detail = (
        "Fold-local rows by method: "
        + ", ".join(f"{method}={int(rows)}" for method, rows in row_counts.items())
    )
    if missing:
        detail += f"; missing_methods={missing}"
    if missing_files:
        detail += f"; missing_artifacts={missing_files}"

    record(
        rows,
        "fold_local_regime_refit_artifacts",
        FAIL if failures else PASS,
        "critical",
        detail,
        rows_checked=len(required_methods),
        rows_failed=failures,
    )


def audit_robustness_artifacts(rows: list[AuditRecord]) -> None:
    results_path = Path(SAVE_DIR) / "robustness_results.csv"
    summary_path = Path(SAVE_DIR) / "robustness_summary.csv"
    wins_path = Path(SAVE_DIR) / "robustness_wins.csv"

    if not results_path.exists():
        record(
            rows,
            "robustness_matrix_artifacts",
            WARN,
            "artifact",
            "robustness_results.csv is missing; run robustness.py for Phase 14A horizon/symbol robustness results.",
        )
        return

    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    required_methods = FOLD_LOCAL_METHODS
    expected_cells = 9
    expected_rows = expected_cells * len(required_methods)
    missing_files = [str(path.name) for path in [summary_path, wins_path] if not path.exists()]
    missing_cols = sorted(
        {"symbol_scope", "target", "horizon", "method", "IC", "Sharpe", "drawdown", "n_test_rows"}
        - set(results.columns)
    )

    failures = len(missing_files) + len(missing_cols)
    detail_parts = [f"robustness rows={len(results)}, expected={expected_rows}"]

    if len(results) != expected_rows:
        failures += 1
        detail_parts.append("unexpected_result_row_count")

    if not missing_cols:
        cell_counts = results.groupby(["symbol_scope", "target", "horizon"])["method"].nunique()
        incomplete_cells = int((cell_counts != len(required_methods)).sum())
        failures += incomplete_cells
        detail_parts.append(f"grid_cells={len(cell_counts)}")
        if incomplete_cells:
            detail_parts.append(f"incomplete_cells={incomplete_cells}")

    if not summary.empty:
        detail_parts.append(f"summary_cells={len(summary)}")
        if len(summary) != expected_cells:
            failures += 1
            detail_parts.append("unexpected_summary_cell_count")

    if missing_files:
        detail_parts.append(f"missing_artifacts={missing_files}")
    if missing_cols:
        detail_parts.append(f"missing_columns={missing_cols}")

    record(
        rows,
        "robustness_matrix_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=expected_rows,
        rows_failed=failures,
    )


def audit_robustness_stress_artifacts(rows: list[AuditRecord]) -> None:
    results_path = Path(SAVE_DIR) / "robustness_stress_results.csv"
    summary_path = Path(SAVE_DIR) / "robustness_stress_summary.csv"
    wins_path = Path(SAVE_DIR) / "robustness_stress_wins.csv"

    if not results_path.exists():
        record(
            rows,
            "robustness_stress_artifacts",
            WARN,
            "artifact",
            "robustness_stress_results.csv is missing; run robustness_stress.py for Phase 14B cost/threshold/period stress results.",
        )
        return

    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    required_methods = FOLD_LOCAL_METHODS
    expected_cells = 4 * 3 * 4
    expected_rows = expected_cells * len(required_methods)
    missing_files = [str(path.name) for path in [summary_path, wins_path] if not path.exists()]
    missing_cols = sorted(
        {
            "market_period",
            "threshold",
            "transaction_cost_bps",
            "method",
            "signal_IC",
            "Sharpe",
            "drawdown",
            "total_return",
            "n_test_rows",
        }
        - set(results.columns)
    )

    failures = len(missing_files) + len(missing_cols)
    detail_parts = [f"stress rows={len(results)}, expected={expected_rows}"]

    if len(results) != expected_rows:
        failures += 1
        detail_parts.append("unexpected_result_row_count")

    if not missing_cols:
        cell_counts = results.groupby(["market_period", "threshold", "transaction_cost_bps"])["method"].nunique()
        incomplete_cells = int((cell_counts != len(required_methods)).sum())
        failures += incomplete_cells
        detail_parts.append(f"stress_cells={len(cell_counts)}")
        if incomplete_cells:
            detail_parts.append(f"incomplete_cells={incomplete_cells}")

    if not summary.empty:
        detail_parts.append(f"summary_cells={len(summary)}")
        if len(summary) != expected_cells:
            failures += 1
            detail_parts.append("unexpected_summary_cell_count")

    if missing_files:
        detail_parts.append(f"missing_artifacts={missing_files}")
    if missing_cols:
        detail_parts.append(f"missing_columns={missing_cols}")

    record(
        rows,
        "robustness_stress_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=expected_rows,
        rows_failed=failures,
    )


def audit_regime_quality_artifacts(rows: list[AuditRecord]) -> None:
    summary_path = Path(SAVE_DIR) / "regime_quality_summary.csv"
    agreement_path = Path(SAVE_DIR) / "regime_agreement_matrix.csv"
    quality_plot_path = Path(SAVE_DIR) / "regime_quality_heatmap.png"
    agreement_plot_path = Path(SAVE_DIR) / "regime_agreement_heatmap.png"

    if not summary_path.exists():
        record(
            rows,
            "regime_quality_artifacts",
            WARN,
            "artifact",
            "regime_quality_summary.csv is missing; run regime_quality.py for Phase 16 structural regime diagnostics.",
        )
        return

    summary = pd.read_csv(summary_path)
    agreement = pd.read_csv(agreement_path) if agreement_path.exists() else pd.DataFrame()
    required_methods = ["contrastive", "contrastive_hmm", "hmm", "kmeans", "vol_bucket"]
    expected_scopes = {"ALL", "BTCUSDT", "ETHUSDT"}
    expected_summary_rows = len(required_methods) * len(expected_scopes)
    expected_agreement_rows = len(required_methods) * len(required_methods) * len(expected_scopes)
    missing_files = [
        str(path.name)
        for path in [agreement_path, quality_plot_path, agreement_plot_path]
        if not path.exists()
    ]
    missing_summary_cols = sorted(
        {
            "method",
            "symbol_scope",
            "n_rows",
            "regime_balance_entropy",
            "transition_diagonal_probability",
            "mean_confidence",
            "hmm_reference_nmi",
            "hmm_reference_ari",
            "hmm_reference_purity",
        }
        - set(summary.columns)
    )
    missing_agreement_cols = sorted(
        {"symbol_scope", "method_a", "method_b", "n_rows", "nmi", "ari", "same_label_pct"}
        - set(agreement.columns)
    )

    failures = len(missing_files) + len(missing_summary_cols) + len(missing_agreement_cols)
    detail_parts = [f"summary_rows={len(summary)}, expected={expected_summary_rows}"]

    if len(summary) != expected_summary_rows:
        failures += 1
        detail_parts.append("unexpected_summary_row_count")

    if not missing_summary_cols:
        all_scope = summary[summary["symbol_scope"] == "ALL"]
        missing_methods = sorted(set(required_methods) - set(all_scope["method"]))
        missing_scopes = sorted(expected_scopes - set(summary["symbol_scope"]))
        failures += len(missing_methods) + len(missing_scopes)
        detail_parts.append(f"all_scope_methods={len(all_scope)}")
        if missing_methods:
            detail_parts.append(f"missing_all_scope_methods={missing_methods}")
        if missing_scopes:
            detail_parts.append(f"missing_scopes={missing_scopes}")
        bad_ranges = int(
            (
                (summary["regime_balance_entropy"] < 0)
                | (summary["regime_balance_entropy"] > 1)
                | (summary["transition_diagonal_probability"] < 0)
                | (summary["transition_diagonal_probability"] > 1)
                | (summary["mean_confidence"] < 0)
                | (summary["mean_confidence"] > 1)
                | (summary["hmm_reference_nmi"] < 0)
                | (summary["hmm_reference_nmi"] > 1)
                | (summary["hmm_reference_purity"] < 0)
                | (summary["hmm_reference_purity"] > 1)
            ).sum()
        )
        failures += bad_ranges
        if bad_ranges:
            detail_parts.append(f"bounded_metric_rows_out_of_range={bad_ranges}")

    if not agreement.empty:
        detail_parts.append(f"agreement_rows={len(agreement)}, expected={expected_agreement_rows}")
        if len(agreement) != expected_agreement_rows:
            failures += 1
            detail_parts.append("unexpected_agreement_row_count")
        if not missing_agreement_cols:
            bad_agreement_ranges = int(
                (
                    (agreement["nmi"] < 0)
                    | (agreement["nmi"] > 1)
                    | (agreement["same_label_pct"] < 0)
                    | (agreement["same_label_pct"] > 1)
                ).sum()
            )
            failures += bad_agreement_ranges
            if bad_agreement_ranges:
                detail_parts.append(f"agreement_metric_rows_out_of_range={bad_agreement_ranges}")
    else:
        detail_parts.append("agreement_rows=0")

    if missing_files:
        detail_parts.append(f"missing_artifacts={missing_files}")
    if missing_summary_cols:
        detail_parts.append(f"missing_summary_columns={missing_summary_cols}")
    if missing_agreement_cols:
        detail_parts.append(f"missing_agreement_columns={missing_agreement_cols}")

    record(
        rows,
        "regime_quality_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(expected_summary_rows + expected_agreement_rows, 1),
        rows_failed=failures,
    )


def audit_compute_plan_artifacts(rows: list[AuditRecord]) -> None:
    profile_path = Path(SAVE_DIR) / "compute_profile.csv"
    budget_path = Path(SAVE_DIR) / "ablation_budget.csv"
    summary_path = Path(SAVE_DIR) / "compute_budget_summary.csv"
    plot_path = Path(SAVE_DIR) / "compute_budget_plan.png"

    if not profile_path.exists():
        record(
            rows,
            "compute_plan_artifacts",
            WARN,
            "artifact",
            "compute_profile.csv is missing; run compute_plan.py for Phase 17 compute planning.",
        )
        return

    profile = pd.read_csv(profile_path)
    budget = pd.read_csv(budget_path) if budget_path.exists() else pd.DataFrame()
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    missing_files = [
        str(path.name)
        for path in [budget_path, summary_path, plot_path]
        if not path.exists()
    ]
    missing_profile_cols = sorted(
        {
            "symbols",
            "device",
            "encoder_parameters",
            "batch_size",
            "epochs",
            "training_windows",
            "batches_per_epoch",
            "profile_steps",
            "measured_step_seconds",
            "estimated_full_train_minutes",
            "planned_ablation_runs",
            "estimated_ablation_hours",
            "recommended_initial_runs",
            "budget_status",
        }
        - set(profile.columns)
    )
    missing_budget_cols = sorted(
        {
            "priority",
            "planned_phase",
            "loss",
            "augmentation",
            "assignment_method",
            "estimated_total_minutes",
            "estimated_cumulative_hours",
            "decision",
        }
        - set(budget.columns)
    )
    missing_summary_cols = sorted({"metric", "value", "unit", "notes"} - set(summary.columns))

    failures = len(missing_files) + len(missing_profile_cols) + len(missing_budget_cols) + len(missing_summary_cols)
    detail_parts = [f"profile_rows={len(profile)}", f"budget_rows={len(budget)}"]

    if len(profile) != 1:
        failures += 1
        detail_parts.append("unexpected_profile_row_count")

    if not missing_budget_cols:
        expected_rows = 12
        if len(budget) != expected_rows:
            failures += 1
            detail_parts.append(f"unexpected_budget_row_count={len(budget)} expected={expected_rows}")
        active_next_count = int((budget["decision"] == "active_next").sum())
        prototype_pending_count = int((budget["decision"] == "prototype_complete_full_pending").sum())
        complete_count = int((budget["decision"] == "complete").sum())
        priority_count = active_next_count + complete_count + prototype_pending_count
        detail_parts.append(
            f"complete={complete_count}; active_next={active_next_count}; "
            f"prototype_pending={prototype_pending_count}"
        )
        if active_next_count + prototype_pending_count < 1:
            failures += 1
            detail_parts.append("no_active_or_pending_run_marked")
        if priority_count < 3:
            failures += 1
            detail_parts.append("missing_completed_or_priority_runs")

    if not missing_profile_cols and not profile.empty:
        numeric_checks = [
            "encoder_parameters",
            "batch_size",
            "epochs",
            "training_windows",
            "batches_per_epoch",
            "estimated_full_train_minutes",
            "planned_ablation_runs",
            "recommended_initial_runs",
        ]
        bad_numeric = 0
        for column in numeric_checks:
            values = pd.to_numeric(profile[column], errors="coerce")
            bad_numeric += int((values <= 0).sum() + values.isna().sum())
        failures += bad_numeric
        if bad_numeric:
            detail_parts.append(f"non_positive_profile_metrics={bad_numeric}")
        detail_parts.append(f"budget_status={profile.iloc[0].get('budget_status', 'missing')}")

    if not summary.empty:
        detail_parts.append(f"summary_rows={len(summary)}")

    if missing_files:
        detail_parts.append(f"missing_artifacts={missing_files}")
    if missing_profile_cols:
        detail_parts.append(f"missing_profile_columns={missing_profile_cols}")
    if missing_budget_cols:
        detail_parts.append(f"missing_budget_columns={missing_budget_cols}")
    if missing_summary_cols:
        detail_parts.append(f"missing_summary_columns={missing_summary_cols}")

    record(
        rows,
        "compute_plan_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(len(profile) + len(budget), 1),
        rows_failed=failures,
    )


def audit_ablation_artifacts(rows: list[AuditRecord]) -> None:
    results_path = Path(SAVE_DIR) / "ablation_results.csv"
    summary_path = Path(SAVE_DIR) / "ablation_summary.csv"
    heatmap_path = Path(SAVE_DIR) / "ablation_heatmap.png"

    if not results_path.exists() or not summary_path.exists() or not heatmap_path.exists():
        missing = [
            str(path.relative_to(Path(SAVE_DIR)))
            for path in [results_path, summary_path, heatmap_path]
            if not path.exists()
        ]
        record(
            rows,
            "ablation_artifacts",
            WARN,
            "artifact",
            f"Missing Phase 25 ablation artifacts: {missing}",
        )
        return

    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path)
    failures = 0
    required_result_cols = {
        "ablation_family",
        "comparison",
        "evidence_type",
        "candidate",
        "reference",
        "metric",
        "candidate_value",
        "reference_value",
        "delta",
        "candidate_wins",
        "source_artifact",
    }
    required_summary_cols = {
        "ablation_family",
        "comparison",
        "evidence_type",
        "candidate",
        "reference",
        "candidate_win_rate",
        "phase25_decision",
    }
    missing_result_cols = sorted(required_result_cols - set(results.columns))
    missing_summary_cols = sorted(required_summary_cols - set(summary.columns))
    expected_families = {
        "objective_guidance",
        "assignment_layer",
        "augmentation_view",
        "classical_reference",
    }
    families = set(summary.get("ablation_family", pd.Series(dtype=str)).dropna().unique())
    missing_families = sorted(expected_families - families)

    if missing_result_cols:
        failures += 1
    if missing_summary_cols:
        failures += 1
    if missing_families:
        failures += 1
    if len(summary) < 10:
        failures += 1
    if "phase25_decision" in summary.columns and not (summary["phase25_decision"] == "supported").any():
        failures += 1

    detail_parts = [
        f"result_rows={len(results)}",
        f"summary_rows={len(summary)}",
        f"families={len(families)}",
    ]
    if "phase25_decision" in summary.columns:
        decision_counts = summary["phase25_decision"].value_counts().to_dict()
        detail_parts.append(f"decisions={decision_counts}")
    if missing_result_cols:
        detail_parts.append(f"missing_result_columns={missing_result_cols}")
    if missing_summary_cols:
        detail_parts.append(f"missing_summary_columns={missing_summary_cols}")
    if missing_families:
        detail_parts.append(f"missing_families={missing_families}")

    record(
        rows,
        "ablation_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(len(results) + len(summary), 1),
        rows_failed=failures,
    )


def audit_paper_statistical_artifacts(rows: list[AuditRecord]) -> None:
    tests_path = Path(SAVE_DIR) / "paper_claim_tests.csv"
    summary_path = Path(SAVE_DIR) / "paper_statistical_summary.csv"
    plot_path = Path(SAVE_DIR) / "paper_claim_tests.png"

    if not tests_path.exists() or not summary_path.exists() or not plot_path.exists():
        missing = [
            str(path.relative_to(Path(SAVE_DIR)))
            for path in [tests_path, summary_path, plot_path]
            if not path.exists()
        ]
        record(
            rows,
            "paper_statistical_artifacts",
            WARN,
            "artifact",
            f"Missing Phase 26 paper statistical artifacts: {missing}",
        )
        return

    tests = pd.read_csv(tests_path)
    summary = pd.read_csv(summary_path)
    failures = 0
    required_test_cols = {
        "ablation_family",
        "comparison",
        "candidate",
        "reference",
        "metric",
        "mean_difference",
        "primary_p_value",
        "statistical_status",
        "allowed_paper_language",
    }
    required_summary_cols = {
        "ablation_family",
        "comparison",
        "evidence_type",
        "candidate",
        "reference",
        "paper_status",
        "allowed_paper_language",
    }
    missing_test_cols = sorted(required_test_cols - set(tests.columns))
    missing_summary_cols = sorted(required_summary_cols - set(summary.columns))
    if missing_test_cols:
        failures += 1
    if missing_summary_cols:
        failures += 1
    if len(summary) < 10:
        failures += 1
    if "guided_hmm_alpha_vs_raw_feature_hmm_alpha" not in set(summary.get("comparison", pd.Series(dtype=str))):
        failures += 1
    supported_statuses = {
        "statistically_supported",
        "metric_family_supported",
        "raw_suggestive",
        "directionally_supported",
        "mechanism_supported_no_fold_p_value",
    }
    status_values = set(summary.get("paper_status", pd.Series(dtype=str)).dropna())
    if not status_values.intersection(supported_statuses):
        failures += 1

    detail_parts = [
        f"test_rows={len(tests)}",
        f"summary_rows={len(summary)}",
    ]
    if "paper_status" in summary.columns:
        detail_parts.append(f"statuses={summary['paper_status'].value_counts().to_dict()}")
    if missing_test_cols:
        detail_parts.append(f"missing_test_columns={missing_test_cols}")
    if missing_summary_cols:
        detail_parts.append(f"missing_summary_columns={missing_summary_cols}")

    record(
        rows,
        "paper_statistical_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(len(tests) + len(summary), 1),
        rows_failed=failures,
    )


def audit_guided_encoder_artifacts(rows: list[AuditRecord]) -> None:
    summary_path = Path(SAVE_DIR) / "guided_encoder_summary.csv"
    loss_path = Path(SAVE_DIR) / "guided_encoder_loss.csv"
    comparison_path = Path(SAVE_DIR) / "guided_encoder_comparison.csv"
    loss_plot_path = Path(SAVE_DIR) / "guided_encoder_loss_curve.png"
    transition_gmm_path = Path(SAVE_DIR) / "guided_encoder_transition_hmm_guided_gmm.png"
    transition_hmm_path = Path(SAVE_DIR) / "guided_encoder_transition_hmm_guided_hmm.png"

    if not summary_path.exists():
        record(
            rows,
            "guided_encoder_artifacts",
            WARN,
            "artifact",
            "guided_encoder_summary.csv is missing; run guided_encoder.py for Phase 19B HMM-guided encoder diagnostics.",
        )
        return

    summary = pd.read_csv(summary_path)
    loss = pd.read_csv(loss_path) if loss_path.exists() else pd.DataFrame()
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    missing_files = [
        str(path.name)
        for path in [loss_path, comparison_path, loss_plot_path, transition_gmm_path, transition_hmm_path]
        if not path.exists()
    ]
    missing_summary_cols = sorted(
        {
            "method",
            "loss",
            "augmentation",
            "assignment_method",
            "epochs",
            "n_rows",
            "silhouette",
            "avg_regime_duration",
            "transition_diagonal_probability",
            "mean_confidence",
            "final_loss",
            "final_valid_anchor_pct",
            "hmm_reference_nmi",
            "hmm_reference_ari",
            "hmm_reference_purity",
        }
        - set(summary.columns)
    )
    missing_loss_cols = sorted(
        {"epoch", "loss", "valid_anchor_pct", "positive_pairs_per_batch", "hard_negative_pairs_per_batch"}
        - set(loss.columns)
    )
    missing_comparison_cols = sorted(
        {
            "method",
            "source_phase",
            "n_rows",
            "hmm_reference_nmi",
            "hmm_reference_ari",
            "hmm_reference_purity",
            "hmm_nmi_vs_contrastive_delta",
            "hmm_purity_vs_contrastive_delta",
        }
        - set(comparison.columns)
    )

    required_methods = {"hmm_guided_gmm", "hmm_guided_hmm"}
    failures = len(missing_files) + len(missing_summary_cols) + len(missing_loss_cols) + len(missing_comparison_cols)
    detail_parts = [f"summary_rows={len(summary)}", f"loss_rows={len(loss)}", f"comparison_rows={len(comparison)}"]

    if not missing_summary_cols:
        missing_methods = sorted(required_methods - set(summary["method"]))
        failures += len(missing_methods)
        if missing_methods:
            detail_parts.append(f"missing_methods={missing_methods}")

        bounded_cols = [
            "transition_diagonal_probability",
            "mean_confidence",
            "final_valid_anchor_pct",
            "hmm_reference_nmi",
            "hmm_reference_purity",
        ]
        bad_ranges = 0
        for column in bounded_cols:
            values = pd.to_numeric(summary[column], errors="coerce")
            bad_ranges += int(((values < 0) | (values > 1) | values.isna()).sum())
        failures += bad_ranges
        if bad_ranges:
            detail_parts.append(f"bounded_metric_rows_out_of_range={bad_ranges}")

        if not summary.empty:
            max_epochs = int(summary["epochs"].max())
            if max_epochs < 30:
                failures += 1
                detail_parts.append(f"epochs_below_phase19b_full_run={max_epochs}")
            detail_parts.append(f"epochs={max_epochs}")
            detail_parts.append(f"best_hmm_reference_nmi={summary['hmm_reference_nmi'].max():.3f}")

    if not loss.empty and not missing_loss_cols:
        bad_loss = int((pd.to_numeric(loss["loss"], errors="coerce") <= 0).sum())
        failures += bad_loss
        if bad_loss:
            detail_parts.append(f"non_positive_loss_rows={bad_loss}")

    if not comparison.empty and not missing_comparison_cols:
        comparison_methods = set(comparison["method"])
        missing_comparison_methods = sorted(required_methods - comparison_methods)
        failures += len(missing_comparison_methods)
        if missing_comparison_methods:
            detail_parts.append(f"missing_comparison_methods={missing_comparison_methods}")
        if "contrastive" in comparison_methods:
            best_guided_nmi = comparison[
                comparison["method"].isin(required_methods)
            ]["hmm_reference_nmi"].max()
            contrastive_nmi = comparison.loc[
                comparison["method"] == "contrastive", "hmm_reference_nmi"
            ].iloc[0]
            detail_parts.append(f"guided_nmi_minus_contrastive={best_guided_nmi - contrastive_nmi:.3f}")

    if missing_files:
        detail_parts.append(f"missing_artifacts={missing_files}")
    if missing_summary_cols:
        detail_parts.append(f"missing_summary_columns={missing_summary_cols}")
    if missing_loss_cols:
        detail_parts.append(f"missing_loss_columns={missing_loss_cols}")
    if missing_comparison_cols:
        detail_parts.append(f"missing_comparison_columns={missing_comparison_cols}")

    record(
        rows,
        "guided_encoder_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(len(summary) + len(loss), 1),
        rows_failed=failures,
    )


def audit_time_frequency_encoder_artifacts(rows: list[AuditRecord]) -> None:
    summary_path = Path(SAVE_DIR) / "time_frequency_encoder_summary.csv"
    loss_path = Path(SAVE_DIR) / "time_frequency_encoder_loss.csv"
    comparison_path = Path(SAVE_DIR) / "time_frequency_encoder_comparison.csv"
    loss_plot_path = Path(SAVE_DIR) / "time_frequency_encoder_loss_curve.png"
    transition_gmm_path = Path(SAVE_DIR) / "time_frequency_encoder_transition_tf_hmm_guided_gmm.png"
    transition_hmm_path = Path(SAVE_DIR) / "time_frequency_encoder_transition_tf_hmm_guided_hmm.png"

    if not summary_path.exists():
        record(
            rows,
            "time_frequency_encoder_artifacts",
            WARN,
            "artifact",
            "time_frequency_encoder_summary.csv is missing; run guided_encoder.py --augmentation time_frequency for Phase 22 diagnostics.",
        )
        return

    summary = pd.read_csv(summary_path)
    loss = pd.read_csv(loss_path) if loss_path.exists() else pd.DataFrame()
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    missing_files = [
        str(path.name)
        for path in [loss_path, comparison_path, loss_plot_path, transition_gmm_path, transition_hmm_path]
        if not path.exists()
    ]
    missing_summary_cols = sorted(
        {
            "method",
            "loss",
            "augmentation",
            "assignment_method",
            "epochs",
            "input_features",
            "fft_bins",
            "n_rows",
            "silhouette",
            "avg_regime_duration",
            "transition_diagonal_probability",
            "mean_confidence",
            "final_loss",
            "final_valid_anchor_pct",
            "hmm_reference_nmi",
            "hmm_reference_ari",
            "hmm_reference_purity",
        }
        - set(summary.columns)
    )
    missing_loss_cols = sorted(
        {"epoch", "loss", "valid_anchor_pct", "positive_pairs_per_batch", "hard_negative_pairs_per_batch"}
        - set(loss.columns)
    )
    missing_comparison_cols = sorted(
        {
            "method",
            "source_phase",
            "n_rows",
            "hmm_reference_nmi",
            "hmm_reference_ari",
            "hmm_reference_purity",
            "hmm_nmi_vs_contrastive_delta",
            "hmm_purity_vs_contrastive_delta",
        }
        - set(comparison.columns)
    )

    required_methods = set(TIME_FREQUENCY_METHODS)
    failures = len(missing_files) + len(missing_summary_cols) + len(missing_loss_cols) + len(missing_comparison_cols)
    detail_parts = [f"summary_rows={len(summary)}", f"loss_rows={len(loss)}", f"comparison_rows={len(comparison)}"]

    if not missing_summary_cols:
        missing_methods = sorted(required_methods - set(summary["method"]))
        failures += len(missing_methods)
        if missing_methods:
            detail_parts.append(f"missing_methods={missing_methods}")

        if set(summary["augmentation"].unique()) != {"time_frequency"}:
            failures += 1
            detail_parts.append(f"unexpected_augmentations={sorted(summary['augmentation'].unique())}")

        expected_features = len(FEATURE_COLS) * (1 + int(summary["fft_bins"].iloc[0]))
        input_features = pd.to_numeric(summary["input_features"], errors="coerce")
        bad_input_features = int((input_features != expected_features).sum())
        failures += bad_input_features
        if bad_input_features:
            detail_parts.append(f"unexpected_input_features={bad_input_features}; expected={expected_features}")

        bounded_cols = [
            "transition_diagonal_probability",
            "mean_confidence",
            "final_valid_anchor_pct",
            "hmm_reference_nmi",
            "hmm_reference_purity",
        ]
        bad_ranges = 0
        for column in bounded_cols:
            values = pd.to_numeric(summary[column], errors="coerce")
            bad_ranges += int(((values < 0) | (values > 1) | values.isna()).sum())
        failures += bad_ranges
        if bad_ranges:
            detail_parts.append(f"bounded_metric_rows_out_of_range={bad_ranges}")

        if not summary.empty:
            max_epochs = int(summary["epochs"].max())
            detail_parts.append(f"epochs={max_epochs}")
            detail_parts.append(f"best_hmm_reference_nmi={summary['hmm_reference_nmi'].max():.3f}")

    if not loss.empty and not missing_loss_cols:
        bad_loss = int((pd.to_numeric(loss["loss"], errors="coerce") <= 0).sum())
        failures += bad_loss
        if bad_loss:
            detail_parts.append(f"non_positive_loss_rows={bad_loss}")

    if not comparison.empty and not missing_comparison_cols:
        comparison_methods = set(comparison["method"])
        missing_comparison_methods = sorted(required_methods - comparison_methods)
        failures += len(missing_comparison_methods)
        if missing_comparison_methods:
            detail_parts.append(f"missing_comparison_methods={missing_comparison_methods}")

    if missing_files:
        detail_parts.append(f"missing_artifacts={missing_files}")
    if missing_summary_cols:
        detail_parts.append(f"missing_summary_columns={missing_summary_cols}")
    if missing_loss_cols:
        detail_parts.append(f"missing_loss_columns={missing_loss_cols}")
    if missing_comparison_cols:
        detail_parts.append(f"missing_comparison_columns={missing_comparison_cols}")

    record(
        rows,
        "time_frequency_encoder_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(len(summary) + len(loss), 1),
        rows_failed=failures,
    )


def audit_interpretability_artifacts(rows: list[AuditRecord]) -> None:
    global_path = Path(SAVE_DIR) / "feature_importance_global.csv"
    regime_path = Path(SAVE_DIR) / "feature_importance_by_regime.csv"
    family_path = Path(SAVE_DIR) / "feature_family_summary.csv"
    regime_plot_path = Path(SAVE_DIR) / "feature_importance_by_regime.png"
    family_plot_path = Path(SAVE_DIR) / "feature_family_importance.png"

    if not global_path.exists() or not regime_path.exists() or not family_path.exists():
        missing = [
            path.name
            for path in [global_path, regime_path, family_path]
            if not path.exists()
        ]
        record(
            rows,
            "interpretability_artifacts",
            WARN,
            "artifact",
            f"Missing Phase 23 interpretability artifacts: {missing}",
        )
        return

    global_imp = pd.read_csv(global_path)
    regime_imp = pd.read_csv(regime_path)
    family_imp = pd.read_csv(family_path)
    required_cols = {
        "method",
        "regime_method",
        "regime",
        "feature",
        "feature_family",
        "mean_gain_share",
        "mean_split_share",
        "mean_shap_share",
        "folds_seen",
        "rank_within_model_regime",
    }
    missing_global_cols = sorted(required_cols - set(global_imp.columns))
    missing_regime_cols = sorted(required_cols - set(regime_imp.columns))
    missing_family_cols = sorted(
        {"method", "regime_method", "regime", "feature_family", "mean_gain_share", "mean_shap_share"}
        - set(family_imp.columns)
    )
    missing_files = [
        path.name
        for path in [regime_plot_path, family_plot_path]
        if not path.exists()
    ]

    failures = len(missing_global_cols) + len(missing_regime_cols) + len(missing_family_cols) + len(missing_files)
    detail_parts = [
        f"global_rows={len(global_imp)}",
        f"regime_rows={len(regime_imp)}",
        f"family_rows={len(family_imp)}",
    ]

    if not missing_global_cols:
        global_features = set(global_imp["feature"])
        missing_features = sorted(set(FEATURE_COLS) - global_features)
        failures += len(missing_features)
        if missing_features:
            detail_parts.append(f"missing_global_features={missing_features}")

    if not missing_regime_cols:
        required_methods = {"regime_lgbm_hmm", "regime_lgbm_hmm_guided_hmm"}
        method_set = set(regime_imp["method"])
        missing_methods = sorted(required_methods - method_set)
        failures += len(missing_methods)
        if missing_methods:
            detail_parts.append(f"missing_methods={missing_methods}")

        guided = regime_imp[regime_imp["method"] == "regime_lgbm_hmm_guided_hmm"]
        if guided.empty:
            failures += 1
            detail_parts.append("missing_guided_hmm_rows")
        else:
            regimes = sorted(guided["regime"].astype(str).unique())
            detail_parts.append(f"guided_regimes={regimes}")
            top = guided.sort_values(["regime", "rank_within_model_regime"]).groupby("regime").head(1)
            top_features = ",".join(top["feature"].astype(str).tolist())
            detail_parts.append(f"guided_top_features={top_features}")

    if missing_global_cols:
        detail_parts.append(f"missing_global_columns={missing_global_cols}")
    if missing_regime_cols:
        detail_parts.append(f"missing_regime_columns={missing_regime_cols}")
    if missing_family_cols:
        detail_parts.append(f"missing_family_columns={missing_family_cols}")
    if missing_files:
        detail_parts.append(f"missing_plots={missing_files}")

    record(
        rows,
        "interpretability_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(len(global_imp) + len(regime_imp) + len(family_imp), 1),
        rows_failed=failures,
    )


def audit_literature_positioning_artifacts(rows: list[AuditRecord]) -> None:
    related_work_path = Path(BASE_DIR) / "reports" / "related_work.md"
    matrix_path = Path(BASE_DIR) / "reports" / "literature_matrix.csv"

    if not related_work_path.exists() or not matrix_path.exists():
        missing = [
            str(path.relative_to(BASE_DIR))
            for path in [related_work_path, matrix_path]
            if not path.exists()
        ]
        record(
            rows,
            "literature_positioning_artifacts",
            WARN,
            "artifact",
            f"Missing Phase 19A literature artifacts: {missing}",
        )
        return

    matrix = pd.read_csv(matrix_path)
    required_cols = {
        "cluster",
        "reference",
        "year",
        "main_idea",
        "relevance_to_regime_alpha_benchmark",
        "gap_or_caution",
        "source_url",
    }
    required_clusters = {
        "contrastive_time_series",
        "financial_regimes",
        "financial_ml_validation",
        "regime_conditioned_alpha",
        "project_contribution",
    }
    missing_cols = sorted(required_cols - set(matrix.columns))
    missing_clusters = sorted(required_clusters - set(matrix["cluster"])) if not missing_cols else []
    empty_required = int(matrix[list(required_cols & set(matrix.columns))].isna().any(axis=1).sum()) if not matrix.empty else 1
    related_text = related_work_path.read_text(encoding="utf-8")
    required_phrases = [
        "HMM states are not ground truth",
        "Contribution Statement",
        "Reviewer Risk Register",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in related_text]

    failures = len(missing_cols) + len(missing_clusters) + empty_required + len(missing_phrases)
    detail_parts = [
        f"matrix_rows={len(matrix)}",
        f"clusters={matrix['cluster'].nunique() if 'cluster' in matrix.columns else 0}",
    ]
    if missing_cols:
        detail_parts.append(f"missing_cols={missing_cols}")
    if missing_clusters:
        detail_parts.append(f"missing_clusters={missing_clusters}")
    if empty_required:
        detail_parts.append(f"rows_with_empty_required_fields={empty_required}")
    if missing_phrases:
        detail_parts.append(f"missing_phrases={missing_phrases}")

    record(
        rows,
        "literature_positioning_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(len(matrix), 1),
        rows_failed=failures,
    )


def audit_paper_protocol_artifacts(rows: list[AuditRecord]) -> None:
    reports_dir = Path(BASE_DIR) / "reports"
    required_files = {
        "paper_protocol.md": [
            "Central Research Question",
            "Contribution Boundary",
            "Dataset Freeze",
            "Validation Freeze",
            "Permitted Claims",
            "Forbidden Claims",
            "Decision Gates",
            "downstream alpha-claim gate",
            "structural generalization experiments",
        ],
        "hypotheses.md": [
            "Hypothesis Table",
            "Primary Paper Hypothesis",
            "Claim Language",
        ],
        "claim_registry.md": [
            "Supported Claims",
            "Directional Claims",
            "Open Claims",
            "Forbidden Claims",
            "Resume-Safe Language",
        ],
        "experiment_manifest.md": [
            "Frozen Baseline",
            "Completed Experiment Families",
            "Future Experiment Queue",
            "Minimal Ablation Definition",
            "Submission Readiness Checklist",
        ],
        "phase38_master_protocol.md": [
            "Current Evidence Freeze",
            "Data Roles",
            "Non-Negotiable Validation Rules",
            "Phase 39 Model Contract",
            "Locked Evaluation Policy",
            "Phase 38 Exit Condition",
        ],
        "publication_acceptance_gates.md": [
            "Gate 0: Evidence Freeze",
            "Gate 1: Fold-Local Validity",
            "Gate 2: Baseline Completeness And Parity",
            "Gate 4: Locked External Evaluation",
            "Gate 6: ICAIF Paper Readiness",
            "Gate 7: BTech Project Readiness",
        ],
    }

    missing_files = []
    missing_sections = []
    total_checks = 0

    for filename, sections in required_files.items():
        path = reports_dir / filename
        if not path.exists():
            missing_files.append(filename)
            continue
        text = path.read_text(encoding="utf-8")
        for section in sections:
            total_checks += 1
            if section not in text:
                missing_sections.append(f"{filename}:{section}")

    failures = len(missing_files) + len(missing_sections)
    detail_parts = [
        f"protocol_files={len(required_files) - len(missing_files)}/{len(required_files)}",
        f"section_checks={total_checks}",
    ]
    if missing_files:
        detail_parts.append(f"missing_files={missing_files}")
    if missing_sections:
        detail_parts.append(f"missing_sections={missing_sections}")

    record(
        rows,
        "paper_protocol_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(total_checks + len(required_files), 1),
        rows_failed=failures,
    )


def audit_phase38_control_artifacts(rows: list[AuditRecord]) -> None:
    reports_dir = Path(BASE_DIR) / "reports"
    paths = {
        "master": reports_dir / "phase38_master_protocol.md",
        "roles": reports_dir / "data_role_registry.csv",
        "ledger": reports_dir / "experiment_ledger.csv",
        "gates": reports_dir / "publication_acceptance_gates.md",
        "paper_initializer": Path(BASE_DIR) / "src" / "paper_skeleton.py",
        "fold_local_design": reports_dir / "phase39_fold_local_encoder_design.md",
    }
    missing = [str(path.relative_to(BASE_DIR)) for path in paths.values() if not path.exists()]
    if missing:
        record(
            rows,
            "phase38_control_artifacts",
            FAIL,
            "critical",
            f"Missing Phase 38 control artifacts: {missing}",
            rows_checked=len(paths),
            rows_failed=len(missing),
        )
        return

    roles = pd.read_csv(paths["roles"])
    ledger = pd.read_csv(paths["ledger"])
    master_text = paths["master"].read_text(encoding="utf-8")
    gates_text = paths["gates"].read_text(encoding="utf-8")
    paper_initializer_text = paths["paper_initializer"].read_text(encoding="utf-8")
    fold_local_design_text = paths["fold_local_design"].read_text(encoding="utf-8")

    required_role_cols = {
        "dataset_id",
        "scope",
        "observed_endpoint",
        "role",
        "result_inspected",
        "allowed_uses",
        "forbidden_uses",
        "lock_rule",
        "notes",
    }
    required_ledger_cols = {
        "experiment_id",
        "phase_family",
        "scope",
        "method_family",
        "selection_role",
        "outcome_inspected",
        "paper_role",
        "status",
        "next_action",
    }
    allowed_roles = {
        "development_observed",
        "locked_unobserved",
        "future_uncollected",
        "descriptive_only",
    }
    allowed_selection_roles = {"development", "final_locked"}
    required_master_phrases = [
        "All BTC/ETH and Crypto-20 data available through the Phase 37 checkpoint are `development_observed`",
        "The encoder itself must be fold-local",
        "The locked evaluation is run once",
        "More encoder epochs, unrestricted architecture search, Crypto-50 expansion",
    ]
    required_gate_phrases = [
        "A later gate cannot pass while an earlier critical gate is open",
        "Failure or contradiction is reported without same-holdout retuning",
        "Exactly one configuration is frozen for external evaluation",
        "No real-money execution is enabled",
    ]

    failures = 0
    details = [f"role_rows={len(roles)}", f"ledger_rows={len(ledger)}"]
    missing_role_cols = sorted(required_role_cols - set(roles.columns))
    missing_ledger_cols = sorted(required_ledger_cols - set(ledger.columns))
    failures += len(missing_role_cols) + len(missing_ledger_cols)

    if not missing_role_cols:
        invalid_roles = sorted(set(roles["role"].astype(str)) - allowed_roles)
        duplicate_datasets = int(roles["dataset_id"].astype(str).duplicated().sum())
        inspected = roles["result_inspected"].astype(str).str.lower().eq("true")
        invalid_inspected_locks = int(
            (inspected & roles["role"].isin({"locked_unobserved", "future_uncollected"})).sum()
        )
        development_rows = int((roles["role"] == "development_observed").sum())
        locked_rows = int(roles["role"].isin({"locked_unobserved", "future_uncollected"}).sum())
        failures += (
            len(invalid_roles)
            + duplicate_datasets
            + invalid_inspected_locks
            + int(development_rows < 2)
            + int(locked_rows < 2)
        )
        details.extend(
            [
                f"development_rows={development_rows}",
                f"locked_or_future_rows={locked_rows}",
                f"invalid_inspected_locks={invalid_inspected_locks}",
            ]
        )
        if invalid_roles:
            details.append(f"invalid_roles={invalid_roles}")
        if duplicate_datasets:
            details.append(f"duplicate_datasets={duplicate_datasets}")

    if not missing_ledger_cols:
        invalid_selection_roles = sorted(
            set(ledger["selection_role"].astype(str)) - allowed_selection_roles
        )
        duplicate_experiments = int(ledger["experiment_id"].astype(str).duplicated().sum())
        locked = ledger["selection_role"].astype(str).eq("final_locked")
        inspected = ledger["outcome_inspected"].astype(str).str.lower().eq("true")
        inspected_final_rows = int((locked & inspected).sum())
        planned_fold_local = int(
            ledger["experiment_id"].astype(str).eq("fold_local_encoder_baseline").sum()
        )
        planned_confirmation = int(
            ledger["experiment_id"].astype(str).eq("external_asset_confirmation").sum()
        )
        failures += (
            len(invalid_selection_roles)
            + duplicate_experiments
            + inspected_final_rows
            + int(planned_fold_local != 1)
            + int(planned_confirmation != 1)
        )
        details.extend(
            [
                f"final_locked_rows={int(locked.sum())}",
                f"inspected_final_rows={inspected_final_rows}",
            ]
        )
        if invalid_selection_roles:
            details.append(f"invalid_selection_roles={invalid_selection_roles}")
        if duplicate_experiments:
            details.append(f"duplicate_experiments={duplicate_experiments}")

    missing_master_phrases = [
        phrase for phrase in required_master_phrases if phrase not in master_text
    ]
    missing_gate_phrases = [
        phrase for phrase in required_gate_phrases if phrase not in gates_text
    ]
    required_initializer_phrases = [
        "Preserved human-reviewed paper draft",
        "Preserved human-reviewed artifact map",
        "Preserved human-reviewed submission checklist",
    ]
    missing_initializer_phrases = [
        phrase for phrase in required_initializer_phrases if phrase not in paper_initializer_text
    ]
    required_design_phrases = [
        "fit_training_scaler",
        "fit_training_hmm",
        "build_training_windows",
        "encode_causal_rows",
        "Outer-test rows never enter pair mining or epoch selection",
        "all eight methods have equal outer-test coverage",
    ]
    missing_design_phrases = [
        phrase for phrase in required_design_phrases if phrase not in fold_local_design_text
    ]
    failures += (
        len(missing_master_phrases)
        + len(missing_gate_phrases)
        + len(missing_initializer_phrases)
        + len(missing_design_phrases)
    )
    if missing_role_cols:
        details.append(f"missing_role_cols={missing_role_cols}")
    if missing_ledger_cols:
        details.append(f"missing_ledger_cols={missing_ledger_cols}")
    if missing_master_phrases:
        details.append(f"missing_master_phrases={missing_master_phrases}")
    if missing_gate_phrases:
        details.append(f"missing_gate_phrases={missing_gate_phrases}")
    if missing_initializer_phrases:
        details.append(f"missing_initializer_phrases={missing_initializer_phrases}")
    if missing_design_phrases:
        details.append(f"missing_design_phrases={missing_design_phrases}")

    record(
        rows,
        "phase38_control_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(details),
        rows_checked=max(
            len(roles)
            + len(ledger)
            + len(required_master_phrases)
            + len(required_gate_phrases)
            + len(required_initializer_phrases)
            + len(required_design_phrases),
            1,
        ),
        rows_failed=failures,
    )


def audit_paper_draft_artifacts(rows: list[AuditRecord]) -> None:
    paper_path = Path(BASE_DIR) / "paper" / "main.md"
    artifact_map_path = Path(BASE_DIR) / "reports" / "paper_artifact_map.csv"
    checklist_path = Path(BASE_DIR) / "reports" / "paper_submission_checklist.md"
    required_paths = [paper_path, artifact_map_path, checklist_path]

    missing = [str(path.relative_to(BASE_DIR)) for path in required_paths if not path.exists()]
    if missing:
        record(
            rows,
            "paper_draft_artifacts",
            FAIL,
            "critical",
            f"Missing Phase 27 paper draft artifacts: {missing}",
            rows_checked=len(required_paths),
            rows_failed=len(missing),
        )
        return

    paper_text = paper_path.read_text(encoding="utf-8")
    checklist_text = checklist_path.read_text(encoding="utf-8")
    artifact_map = pd.read_csv(artifact_map_path)

    required_sections = [
        "Abstract",
        "Related Work",
        "Data and Labels",
        "Methods",
        "Validation and Statistical Protocol",
        "Results",
        "Robustness",
        "Interpretability",
        "Ablations",
        "Limitations",
        "Reproducibility",
        "Conclusion",
        "Figure and Table Plan",
    ]
    required_paper_phrases = [
        "HMM states are proxy states",
        "not statistically significant at 5%",
        "controlled pilot",
        "18 pilot folds and 16 Crypto-20 folds",
        "p=0.801",
        "The headline result is mechanistic",
        "structural improvement does not guarantee",
        "fold-level",
    ]
    required_columns = {"paper_section", "artifact_type", "artifact", "paper_role"}
    required_checklist_phrases = [
        "Must Not Claim",
        "Do not claim HMM states are ground truth",
        "Do not claim guided-HMM statistically dominates raw-feature HMM at 5%",
    ]

    missing_sections = [section for section in required_sections if section not in paper_text]
    missing_paper_phrases = [phrase for phrase in required_paper_phrases if phrase not in paper_text]
    missing_columns = sorted(required_columns - set(artifact_map.columns))
    artifact_rows = 0 if missing_columns else len(artifact_map)
    missing_checklist_phrases = [
        phrase for phrase in required_checklist_phrases if phrase not in checklist_text
    ]
    too_few_artifacts = artifact_rows < 8

    failures = (
        len(missing_sections)
        + len(missing_paper_phrases)
        + len(missing_columns)
        + len(missing_checklist_phrases)
        + int(too_few_artifacts)
    )
    detail_parts = [
        "paper_files=3/3",
        f"section_checks={len(required_sections)}",
        f"artifact_map_rows={artifact_rows}",
    ]
    if missing_sections:
        detail_parts.append(f"missing_sections={missing_sections}")
    if missing_paper_phrases:
        detail_parts.append(f"missing_paper_phrases={missing_paper_phrases}")
    if missing_columns:
        detail_parts.append(f"missing_columns={missing_columns}")
    if missing_checklist_phrases:
        detail_parts.append(f"missing_checklist_phrases={missing_checklist_phrases}")
    if too_few_artifacts:
        detail_parts.append("artifact_map_rows_below_8")

    record(
        rows,
        "paper_draft_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(len(required_paths) + len(required_sections) + artifact_rows, 1),
        rows_failed=failures,
    )


def audit_reproducibility_artifacts(rows: list[AuditRecord]) -> None:
    required_files = {
        "reproduce.ps1": [
            "Mode",
            "smoke",
            "full",
            "dashboard",
            "validation_audit.py",
            "paper_skeleton.py",
            "archive_run.py",
        ],
        "reproduce.sh": [
            "MODE",
            "smoke",
            "full",
            "dashboard",
            "validation_audit.py",
            "paper_skeleton.py",
            "archive_run.py",
        ],
        "scripts/phases/run_phase35_crypto20_guided.ps1": [
            "Epochs",
            "BatchSize",
            "crypto20_regime_assignments.csv",
            "crypto20_guided_encoder",
        ],
        "scripts/phases/run_phase35_crypto20_guided.sh": [
            "EPOCHS",
            "BATCH_SIZE",
            "crypto20_regime_assignments.csv",
            "crypto20_guided_encoder",
        ],
        "reports/environment.md": [
            "Python 3.11",
            "requirements.txt",
            "requirements-research.txt",
            "data/market.duckdb",
            "HMM states are proxy/reference states",
        ],
        "reports/artifact_manifest.md": [
            "Committed Curated Artifacts",
            "Regenerated Local Artifacts",
            "Ignored Private or Heavy Artifacts",
            "models/*.pt",
            "data/",
        ],
        "reports/reproduction_checklist.md": [
            "Smoke Reproduction",
            "Full Reproduction",
            "Dashboard Reproduction",
            "Git Safety Checks",
            "should print nothing",
        ],
        "reports/compute_budget.md": [
            "downstream alpha claims",
            "Structural generalization tests",
            "p = 0.801",
            "structural generalization testing: permitted",
        ],
    }

    missing_files = []
    missing_phrases = []
    total_checks = 0
    for relative_path, phrases in required_files.items():
        path = Path(BASE_DIR) / relative_path
        if not path.exists():
            missing_files.append(relative_path)
            continue
        text = path.read_text(encoding="utf-8")
        for phrase in phrases:
            total_checks += 1
            if phrase not in text:
                missing_phrases.append(f"{relative_path}:{phrase}")

    failures = len(missing_files) + len(missing_phrases)
    detail_parts = [
        f"repro_files={len(required_files) - len(missing_files)}/{len(required_files)}",
        f"phrase_checks={total_checks}",
    ]
    if missing_files:
        detail_parts.append(f"missing_files={missing_files}")
    if missing_phrases:
        detail_parts.append(f"missing_phrases={missing_phrases}")

    record(
        rows,
        "reproducibility_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(total_checks + len(required_files), 1),
        rows_failed=failures,
    )


def audit_multiasset_universe_artifacts(rows: list[AuditRecord]) -> None:
    required_files = [
        Path(BASE_DIR) / "configs" / "crypto_universe_candidates.csv",
        Path(SAVE_DIR) / "asset_universe_candidates_scored.csv",
        Path(SAVE_DIR) / "asset_universe_crypto20.csv",
        Path(SAVE_DIR) / "asset_universe_crypto50.csv",
        Path(SAVE_DIR) / "asset_universe_exclusions.csv",
        Path(SAVE_DIR) / "asset_universe_summary.csv",
        Path(SAVE_DIR) / "crypto20_data_quality.csv",
        Path(SAVE_DIR) / "crypto20_pipeline_summary.csv",
        Path(BASE_DIR) / "reports" / "multiasset_universe_plan.md",
        Path(BASE_DIR) / "reports" / "crypto20_pipeline_plan.md",
    ]
    missing_files = [str(path.relative_to(BASE_DIR)) for path in required_files if not path.exists()]
    required_candidate_cols = {"design_rank", "symbol", "base_asset", "universe_group", "notes"}
    required_selection_cols = {
        "universe",
        "symbol",
        "design_rank",
        "selection_status",
        "target_size",
        "selection_mode",
        "selected_by_design",
    }

    failures = len(missing_files)
    details = [f"files_present={len(required_files) - len(missing_files)}/{len(required_files)}"]
    rows_checked = len(required_files)

    if not missing_files:
        candidates = pd.read_csv(required_files[0])
        scored = pd.read_csv(Path(SAVE_DIR) / "asset_universe_candidates_scored.csv")
        crypto20 = pd.read_csv(Path(SAVE_DIR) / "asset_universe_crypto20.csv")
        crypto50 = pd.read_csv(Path(SAVE_DIR) / "asset_universe_crypto50.csv")
        summary = pd.read_csv(Path(SAVE_DIR) / "asset_universe_summary.csv")
        crypto20_quality = pd.read_csv(Path(SAVE_DIR) / "crypto20_data_quality.csv")
        crypto20_pipeline = pd.read_csv(Path(SAVE_DIR) / "crypto20_pipeline_summary.csv")
        report_text = (Path(BASE_DIR) / "reports" / "multiasset_universe_plan.md").read_text(encoding="utf-8")

        missing_candidate_cols = sorted(required_candidate_cols - set(candidates.columns))
        missing_scored_cols = sorted(required_candidate_cols - set(scored.columns))
        missing_20_cols = sorted(required_selection_cols - set(crypto20.columns))
        missing_50_cols = sorted(required_selection_cols - set(crypto50.columns))
        missing_summary_cols = sorted({"metric", "value", "notes"} - set(summary.columns))
        missing_quality_cols = sorted(
            {"symbol", "ohlcv_rows", "feature_rows", "target_rows", "quality_status", "failure_reason"}
            - set(crypto20_quality.columns)
        )
        missing_pipeline_cols = sorted({"metric", "value", "notes"} - set(crypto20_pipeline.columns))
        missing_report_phrases = [
            phrase
            for phrase in [
                "Crypto-20 pilot",
                "Crypto-50 final",
                "pre-registered crypto universe protocol",
                "downstream alpha gate",
                "structural diagnostics gate",
            ]
            if phrase not in report_text
        ]
        pipeline_text = (Path(BASE_DIR) / "reports" / "crypto20_pipeline_plan.md").read_text(encoding="utf-8")
        missing_report_phrases.extend(
            phrase
            for phrase in ["python src/ingestion.py --universe crypto20", "Completion Gate"]
            if phrase not in pipeline_text
        )
        bad_sizes = int(len(crypto20) != 20) + int(len(crypto50) != 50)
        failures += (
            len(missing_candidate_cols)
            + len(missing_scored_cols)
            + len(missing_20_cols)
            + len(missing_50_cols)
            + len(missing_summary_cols)
            + len(missing_quality_cols)
            + len(missing_pipeline_cols)
            + len(missing_report_phrases)
            + bad_sizes
        )
        rows_checked += len(candidates) + len(crypto20) + len(crypto50) + len(crypto20_quality)
        details.extend(
            [
                f"candidates={len(candidates)}",
                f"crypto20={len(crypto20)}",
                f"crypto50={len(crypto50)}",
                "crypto20_eligible_now="
                + str(int((crypto20["selection_status"] == "eligible").sum()) if "selection_status" in crypto20 else 0),
                "crypto50_eligible_now="
                + str(int((crypto50["selection_status"] == "eligible").sum()) if "selection_status" in crypto50 else 0),
                "crypto20_quality_pass="
                + str(int((crypto20_quality["quality_status"] == "pass").sum()) if "quality_status" in crypto20_quality else 0),
            ]
        )
        if missing_candidate_cols:
            details.append(f"missing_candidate_cols={missing_candidate_cols}")
        if missing_scored_cols:
            details.append(f"missing_scored_cols={missing_scored_cols}")
        if missing_20_cols:
            details.append(f"missing_crypto20_cols={missing_20_cols}")
        if missing_50_cols:
            details.append(f"missing_crypto50_cols={missing_50_cols}")
        if missing_summary_cols:
            details.append(f"missing_summary_cols={missing_summary_cols}")
        if missing_quality_cols:
            details.append(f"missing_quality_cols={missing_quality_cols}")
        if missing_pipeline_cols:
            details.append(f"missing_pipeline_cols={missing_pipeline_cols}")
        if missing_report_phrases:
            details.append(f"missing_report_phrases={missing_report_phrases}")
        if bad_sizes:
            details.append("unexpected_universe_size")
    elif missing_files:
        details.append(f"missing_files={missing_files}")

    record(
        rows,
        "multiasset_universe_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(details),
        rows_checked=max(rows_checked, 1),
        rows_failed=failures,
    )


def audit_crypto20_regime_artifacts(rows: list[AuditRecord]) -> None:
    required_files = [
        Path(SAVE_DIR) / "crypto20_regime_benchmark_summary.csv",
        Path(SAVE_DIR) / "crypto20_per_regime_stats.csv",
        Path(SAVE_DIR) / "crypto20_regime_symbol_summary.csv",
        Path(SAVE_DIR) / "crypto20_transition_matrix_hmm.png",
        Path(SAVE_DIR) / "crypto20_transition_matrix_kmeans.png",
        Path(SAVE_DIR) / "crypto20_transition_matrix_vol_bucket.png",
        Path(BASE_DIR) / "reports" / "crypto20_regime_benchmark_plan.md",
    ]
    missing_files = [str(path.relative_to(BASE_DIR)) for path in required_files if not path.exists()]
    failures = len(missing_files)
    details = [f"files_present={len(required_files) - len(missing_files)}/{len(required_files)}"]
    rows_checked = len(required_files)

    if not missing_files:
        summary = pd.read_csv(Path(SAVE_DIR) / "crypto20_regime_benchmark_summary.csv")
        stats = pd.read_csv(Path(SAVE_DIR) / "crypto20_per_regime_stats.csv")
        symbol_summary = pd.read_csv(Path(SAVE_DIR) / "crypto20_regime_symbol_summary.csv")
        report_text = (Path(BASE_DIR) / "reports" / "crypto20_regime_benchmark_plan.md").read_text(
            encoding="utf-8"
        )

        expected_methods = {"hmm", "kmeans", "vol_bucket"}
        missing_methods = sorted(expected_methods - set(summary.get("method", [])))
        unexpected_methods = sorted(set(summary.get("method", [])) - expected_methods)
        missing_summary_cols = sorted(
            {"method", "implementation", "n_rows", "n_symbols", "n_regimes", "silhouette"}
            - set(summary.columns)
        )
        missing_stats_cols = sorted(
            {"method", "symbol", "regime", "n_rows", "avg_forward_return_8h", "feature_ic_vs_target"}
            - set(stats.columns)
        )
        missing_symbol_cols = sorted(
            {
                "method",
                "symbol",
                "n_rows",
                "transition_diagonal_probability",
                "dominant_regime_pct",
                "simple_feature_ic",
            }
            - set(symbol_summary.columns)
        )
        report_phrases = ["raw-feature Gaussian HMM", "Crypto-20 regime benchmark summary", "Gate To Next Phase"]
        missing_report_phrases = [phrase for phrase in report_phrases if phrase not in report_text]

        bad_rows = 0
        if not missing_summary_cols:
            bad_rows += int(len(summary) != 3)
            bad_rows += int((summary["n_symbols"].astype(int) != 20).any())
            bad_rows += int((summary["n_regimes"].astype(int) != 4).any())
            hmm_impl = summary.set_index("method").get("implementation", pd.Series(dtype=str)).get("hmm", "")
            bad_rows += int(hmm_impl != "hmmlearn_gaussian_hmm")
        if not missing_symbol_cols:
            symbol_method_counts = symbol_summary.groupby("method")["symbol"].nunique()
            bad_rows += int((symbol_method_counts.reindex(sorted(expected_methods), fill_value=0) != 20).any())
        if not missing_stats_cols:
            bad_rows += int(stats.groupby(["method", "symbol"])["regime"].nunique().min() < 1)

        failures += (
            len(missing_methods)
            + len(unexpected_methods)
            + len(missing_summary_cols)
            + len(missing_stats_cols)
            + len(missing_symbol_cols)
            + len(missing_report_phrases)
            + bad_rows
        )
        rows_checked += len(summary) + len(stats) + len(symbol_summary)
        details.extend(
            [
                f"summary_rows={len(summary)}",
                f"stats_rows={len(stats)}",
                f"symbol_summary_rows={len(symbol_summary)}",
                f"methods={sorted(set(summary.get('method', [])))}",
            ]
        )
        if missing_methods:
            details.append(f"missing_methods={missing_methods}")
        if unexpected_methods:
            details.append(f"unexpected_methods={unexpected_methods}")
        if missing_summary_cols:
            details.append(f"missing_summary_cols={missing_summary_cols}")
        if missing_stats_cols:
            details.append(f"missing_stats_cols={missing_stats_cols}")
        if missing_symbol_cols:
            details.append(f"missing_symbol_cols={missing_symbol_cols}")
        if missing_report_phrases:
            details.append(f"missing_report_phrases={missing_report_phrases}")
        if bad_rows:
            details.append(f"bad_row_checks={bad_rows}")
    else:
        details.append(f"missing_files={missing_files}")

    record(
        rows,
        "crypto20_regime_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(details),
        rows_checked=max(rows_checked, 1),
        rows_failed=failures,
    )


def audit_crypto20_guided_readiness_artifacts(rows: list[AuditRecord]) -> None:
    required_files = [
        Path(SAVE_DIR) / "crypto20_guided_symbol_readiness.csv",
        Path(SAVE_DIR) / "crypto20_guided_pair_summary.csv",
        Path(SAVE_DIR) / "crypto20_guided_compute_plan.csv",
        Path(SAVE_DIR) / "crypto20_guided_gate.csv",
        Path(SAVE_DIR) / "crypto20_guided_compute_gate.png",
        Path(BASE_DIR) / "reports" / "crypto20_guided_readiness.md",
    ]
    missing_files = [str(path.relative_to(BASE_DIR)) for path in required_files if not path.exists()]
    failures = len(missing_files)
    details = [f"files_present={len(required_files) - len(missing_files)}/{len(required_files)}"]
    rows_checked = len(required_files)

    if not missing_files:
        symbol_readiness = pd.read_csv(Path(SAVE_DIR) / "crypto20_guided_symbol_readiness.csv")
        pair_summary = pd.read_csv(Path(SAVE_DIR) / "crypto20_guided_pair_summary.csv")
        compute_plan = pd.read_csv(Path(SAVE_DIR) / "crypto20_guided_compute_plan.csv")
        gate = pd.read_csv(Path(SAVE_DIR) / "crypto20_guided_gate.csv")
        report_text = (Path(BASE_DIR) / "reports" / "crypto20_guided_readiness.md").read_text(
            encoding="utf-8"
        )

        missing_symbol_cols = sorted(
            {
                "symbol",
                "eligible_windows",
                "regimes_present",
                "min_regime_windows",
                "transition_rate",
                "directed_hard_negative_pairs",
            }
            - set(symbol_readiness.columns)
        )
        missing_pair_cols = sorted({"metric", "value", "status"} - set(pair_summary.columns))
        missing_compute_cols = sorted(
            {
                "scenario",
                "eligible_windows",
                "epochs",
                "estimated_train_hours",
                "decision",
            }
            - set(compute_plan.columns)
        )
        missing_gate_cols = sorted({"gate", "status", "recommendation", "hard_failures"} - set(gate.columns))
        report_phrases = [
            "Phase 34 Crypto-20 Guided Encoder Readiness",
            "weak-supervision signal",
            "Gate recommendation",
        ]
        missing_report_phrases = [phrase for phrase in report_phrases if phrase not in report_text]

        bad_rows = 0
        if not missing_symbol_cols:
            bad_rows += int(len(symbol_readiness) != 20)
            bad_rows += int((symbol_readiness["eligible_windows"].astype(int) <= 0).any())
            bad_rows += int((symbol_readiness["regimes_present"].astype(int) < 2).any())
        if not missing_pair_cols:
            metrics = set(pair_summary["metric"].astype(str))
            required_metrics = {
                "symbols",
                "eligible_windows",
                "regimes_present_global",
                "positive_anchor_coverage_pct",
                "directed_hard_negative_pairs",
            }
            bad_rows += len(required_metrics - metrics)
            metric_values = pair_summary.set_index("metric")["value"]
            if "symbols" in metric_values:
                bad_rows += int(int(metric_values["symbols"]) != 20)
            if "positive_anchor_coverage_pct" in metric_values:
                bad_rows += int(float(metric_values["positive_anchor_coverage_pct"]) < 0.99)
        if not missing_compute_cols:
            scenarios = set(compute_plan["scenario"].astype(str))
            bad_rows += int("full_crypto20_guided_encoder" not in scenarios)
        if not missing_gate_cols:
            bad_rows += int(gate.empty)
            if not gate.empty:
                bad_rows += int(str(gate["status"].iloc[0]) != "pass")

        failures += (
            len(missing_symbol_cols)
            + len(missing_pair_cols)
            + len(missing_compute_cols)
            + len(missing_gate_cols)
            + len(missing_report_phrases)
            + bad_rows
        )
        rows_checked += len(symbol_readiness) + len(pair_summary) + len(compute_plan) + len(gate)
        details.extend(
            [
                f"symbol_rows={len(symbol_readiness)}",
                f"pair_metrics={len(pair_summary)}",
                f"compute_scenarios={len(compute_plan)}",
                "gate_status=" + (str(gate["status"].iloc[0]) if not gate.empty and "status" in gate else "missing"),
            ]
        )
        if missing_symbol_cols:
            details.append(f"missing_symbol_cols={missing_symbol_cols}")
        if missing_pair_cols:
            details.append(f"missing_pair_cols={missing_pair_cols}")
        if missing_compute_cols:
            details.append(f"missing_compute_cols={missing_compute_cols}")
        if missing_gate_cols:
            details.append(f"missing_gate_cols={missing_gate_cols}")
        if missing_report_phrases:
            details.append(f"missing_report_phrases={missing_report_phrases}")
        if bad_rows:
            details.append(f"bad_row_checks={bad_rows}")
    else:
        details.append(f"missing_files={missing_files}")

    record(
        rows,
        "crypto20_guided_readiness_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(details),
        rows_checked=max(rows_checked, 1),
        rows_failed=failures,
    )


def audit_crypto20_guided_encoder_artifacts(rows: list[AuditRecord]) -> None:
    summary_path = Path(SAVE_DIR) / "crypto20_guided_encoder_summary.csv"
    loss_path = Path(SAVE_DIR) / "crypto20_guided_encoder_loss.csv"
    comparison_path = Path(SAVE_DIR) / "crypto20_guided_encoder_comparison.csv"
    loss_plot_path = Path(SAVE_DIR) / "crypto20_guided_encoder_loss_curve.png"
    transition_gmm_path = Path(SAVE_DIR) / "crypto20_guided_encoder_transition_hmm_guided_gmm.png"
    transition_hmm_path = Path(SAVE_DIR) / "crypto20_guided_encoder_transition_hmm_guided_hmm.png"
    required_files = [
        summary_path,
        loss_path,
        comparison_path,
        loss_plot_path,
        transition_gmm_path,
        transition_hmm_path,
    ]
    missing_files = [str(path.relative_to(BASE_DIR)) for path in required_files if not path.exists()]
    failures = len(missing_files)
    details = [f"files_present={len(required_files) - len(missing_files)}/{len(required_files)}"]
    rows_checked = len(required_files)

    if not summary_path.exists():
        record(
            rows,
            "crypto20_guided_encoder_artifacts",
            WARN,
            "artifact",
            "crypto20_guided_encoder_summary.csv is missing; run scripts/phases/run_phase35_crypto20_guided.ps1 after Phase 34 readiness passes.",
        )
        return

    summary = pd.read_csv(summary_path)
    loss = pd.read_csv(loss_path) if loss_path.exists() else pd.DataFrame()
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    rows_checked += len(summary) + len(loss) + len(comparison)

    required_summary_cols = {
        "method",
        "loss",
        "augmentation",
        "assignment_method",
        "epochs",
        "batch_size",
        "n_rows",
        "n_symbols",
        "n_regimes",
        "silhouette",
        "avg_regime_duration",
        "transition_diagonal_probability",
        "mean_confidence",
        "final_loss",
        "final_valid_anchor_pct",
        "hmm_reference_nmi",
        "hmm_reference_ari",
        "hmm_reference_purity",
    }
    required_loss_cols = {
        "epoch",
        "loss",
        "valid_anchor_pct",
        "positive_pairs_per_batch",
        "hard_negative_pairs_per_batch",
    }
    required_comparison_cols = {
        "method",
        "source_phase",
        "n_rows",
        "n_symbols",
        "n_regimes",
        "avg_regime_duration",
        "hmm_reference_nmi",
        "hmm_reference_ari",
        "hmm_reference_purity",
    }
    missing_summary_cols = sorted(required_summary_cols - set(summary.columns))
    missing_loss_cols = sorted(required_loss_cols - set(loss.columns))
    missing_comparison_cols = sorted(required_comparison_cols - set(comparison.columns))
    required_methods = {"hmm_guided_gmm", "hmm_guided_hmm"}

    failures += len(missing_summary_cols) + len(missing_loss_cols) + len(missing_comparison_cols)
    if not missing_summary_cols:
        method_set = set(summary["method"].astype(str))
        missing_methods = sorted(required_methods - method_set)
        failures += len(missing_methods)

        numeric_summary = summary.copy()
        for column in required_summary_cols - {"method", "loss", "augmentation", "assignment_method"}:
            numeric_summary[column] = pd.to_numeric(numeric_summary[column], errors="coerce")

        bad_rows = 0
        bad_rows += int(numeric_summary["epochs"].min() < 30)
        bad_rows += int(numeric_summary["n_symbols"].min() != 20)
        bad_rows += int(numeric_summary["n_rows"].min() < 300_000)
        bad_rows += int(numeric_summary["n_regimes"].min() != 4)
        bad_rows += int(numeric_summary["final_valid_anchor_pct"].min() < 0.99)
        bad_rows += int(numeric_summary["final_loss"].isna().any())
        bounded_cols = [
            "transition_diagonal_probability",
            "mean_confidence",
            "final_valid_anchor_pct",
            "hmm_reference_nmi",
            "hmm_reference_ari",
            "hmm_reference_purity",
        ]
        for column in bounded_cols:
            bad_rows += int(
                (
                    (numeric_summary[column] < 0)
                    | (numeric_summary[column] > 1)
                    | numeric_summary[column].isna()
                ).sum()
            )

        if required_methods.issubset(method_set):
            guided = numeric_summary.set_index("method")
            gmm_nmi = float(guided.loc["hmm_guided_gmm", "hmm_reference_nmi"])
            hmm_nmi = float(guided.loc["hmm_guided_hmm", "hmm_reference_nmi"])
            gmm_silhouette = float(guided.loc["hmm_guided_gmm", "silhouette"])
            hmm_silhouette = float(guided.loc["hmm_guided_hmm", "silhouette"])
            bad_rows += int(hmm_nmi <= gmm_nmi)
            bad_rows += int(hmm_silhouette <= gmm_silhouette)
            details.extend(
                [
                    f"hmm_guided_hmm_nmi={hmm_nmi:.3f}",
                    f"hmm_guided_hmm_silhouette={hmm_silhouette:.3f}",
                ]
            )
        if missing_methods:
            details.append(f"missing_methods={missing_methods}")
        failures += bad_rows
        if bad_rows:
            details.append(f"bad_summary_checks={bad_rows}")

    if not loss.empty and not missing_loss_cols:
        loss_values = pd.to_numeric(loss["loss"], errors="coerce")
        first_loss = float(loss_values.iloc[0])
        final_loss = float(loss_values.iloc[-1])
        loss_failures = int(len(loss) < 30) + int(not np.isfinite(final_loss)) + int(final_loss >= first_loss)
        failures += loss_failures
        details.extend([f"loss_rows={len(loss)}", f"loss_drop={first_loss:.4f}->{final_loss:.4f}"])
        if loss_failures:
            details.append(f"loss_checks_failed={loss_failures}")

    if not comparison.empty and not missing_comparison_cols:
        source_phases = set(comparison["source_phase"].astype(str))
        methods = set(comparison["method"].astype(str))
        comparison_failures = len(required_methods - methods)
        comparison_failures += int("phase35_crypto20_guided_encoder" not in source_phases)
        failures += comparison_failures
        details.append(f"comparison_rows={len(comparison)}")
        if comparison_failures:
            details.append(f"comparison_checks_failed={comparison_failures}")

    if missing_files:
        details.append(f"missing_files={missing_files}")
    if missing_summary_cols:
        details.append(f"missing_summary_cols={missing_summary_cols}")
    if missing_loss_cols:
        details.append(f"missing_loss_cols={missing_loss_cols}")
    if missing_comparison_cols:
        details.append(f"missing_comparison_cols={missing_comparison_cols}")

    record(
        rows,
        "crypto20_guided_encoder_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(details),
        rows_checked=max(rows_checked, 1),
        rows_failed=failures,
    )


def audit_crypto20_alpha_retest_artifacts(rows: list[AuditRecord]) -> None:
    results_path = Path(SAVE_DIR) / "crypto20_walkforward_experiment_results.csv"
    summary_path = Path(SAVE_DIR) / "crypto20_walkforward_regime_summary.csv"
    guided_path = Path(SAVE_DIR) / "crypto20_walkforward_guided_alpha_comparison.csv"
    equity_path = Path(SAVE_DIR) / "crypto20_walkforward_equity_curve.png"
    report_path = Path(BASE_DIR) / "reports" / "crypto20_alpha_generalization.md"
    required_files = [results_path, summary_path, guided_path, equity_path, report_path]
    missing_files = [str(path.relative_to(BASE_DIR)) for path in required_files if not path.exists()]

    if missing_files:
        record(
            rows,
            "crypto20_alpha_retest_artifacts",
            WARN,
            "artifact",
            f"Missing Phase 36 Crypto-20 downstream alpha artifacts: {missing_files}",
            rows_checked=len(required_files),
            rows_failed=len(missing_files),
        )
        return

    results = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path)
    guided = pd.read_csv(guided_path)
    report_text = report_path.read_text(encoding="utf-8")

    required_result_cols = {
        "method",
        "target",
        "horizon",
        "regime_method",
        "symbol_scope",
        "IC",
        "accuracy",
        "balanced_accuracy",
        "Sharpe",
        "drawdown",
        "turnover",
        "total_return",
        "n_trades",
        "n_test_rows",
    }
    expected_methods = {
        "global_lgbm",
        "regime_lgbm_hmm",
        "regime_lgbm_kmeans",
        "regime_lgbm_vol_bucket",
        "regime_lgbm_hmm_guided_gmm",
        "regime_lgbm_hmm_guided_hmm",
    }
    required_summary_cols = {
        "method",
        "protocol",
        "implementation",
        "n_test_assignment_rows",
        "n_folds",
        "n_symbols",
        "n_regimes",
        "avg_regime_duration",
        "min_regime_pct",
        "max_regime_pct",
        "mean_confidence",
    }
    required_guided_cols = {
        "guided_method",
        "reference_method",
        "delta_IC",
        "delta_Sharpe",
        "delta_drawdown",
        "equal_test_coverage",
    }

    missing_result_cols = sorted(required_result_cols - set(results.columns))
    missing_summary_cols = sorted(required_summary_cols - set(summary.columns))
    missing_guided_cols = sorted(required_guided_cols - set(guided.columns))
    failures = len(missing_result_cols) + len(missing_summary_cols) + len(missing_guided_cols)
    details = [
        f"result_rows={len(results)}",
        f"summary_rows={len(summary)}",
        f"guided_comparison_rows={len(guided)}",
    ]

    if not missing_result_cols:
        method_set = set(results["method"].astype(str))
        missing_methods = sorted(expected_methods - method_set)
        failures += len(missing_methods)
        if missing_methods:
            details.append(f"missing_methods={missing_methods}")

        n_test = pd.to_numeric(results["n_test_rows"], errors="coerce")
        coverage_failures = int(n_test.isna().sum()) + int(n_test.nunique(dropna=True) != 1)
        failures += coverage_failures
        if coverage_failures:
            details.append("unequal_or_invalid_n_test_rows")

        scopes = results["symbol_scope"].dropna().astype(str).unique().tolist()
        symbol_counts = [len(scope.split("+")) for scope in scopes if scope]
        bad_scope = int(not symbol_counts or max(symbol_counts) != 20)
        failures += bad_scope
        details.append(f"symbol_counts={symbol_counts}")

        if "regime_lgbm_hmm_guided_hmm" in method_set:
            best_ic = results.sort_values("IC", ascending=False).iloc[0]
            guided_row = results[results["method"] == "regime_lgbm_hmm_guided_hmm"].iloc[0]
            details.extend(
                [
                    f"best_ic_method={best_ic['method']}",
                    f"hmm_guided_hmm_ic={float(guided_row['IC']):.4f}",
                    f"hmm_guided_hmm_sharpe={float(guided_row['Sharpe']):.4f}",
                ]
            )

    if not missing_summary_cols:
        summary_symbols = pd.to_numeric(summary["n_symbols"], errors="coerce")
        bad_summary = int((summary_symbols != 20).sum()) + int((summary["protocol"] != "fold_local_regime_refit").sum())
        failures += bad_summary
        if bad_summary:
            details.append(f"bad_summary_rows={bad_summary}")

    if not missing_guided_cols:
        references = set(guided["reference_method"].astype(str))
        missing_references = sorted({"regime_lgbm_hmm", "global_lgbm"} - references)
        equal_coverage = guided["equal_test_coverage"].astype(str).str.lower().isin({"true", "1", "yes"})
        guided_failures = len(missing_references) + int((~equal_coverage).sum())
        failures += guided_failures
        if missing_references:
            details.append(f"missing_guided_references={missing_references}")
        if guided_failures and bool((~equal_coverage).any()):
            details.append("guided_comparison_has_unequal_coverage")

    required_phrases = [
        "not a downstream alpha claim",
        "fold-local",
        "structural transfer does not automatically imply predictive transfer",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in report_text]
    failures += len(missing_phrases)
    if missing_phrases:
        details.append(f"missing_report_phrases={missing_phrases}")

    if missing_result_cols:
        details.append(f"missing_result_cols={missing_result_cols}")
    if missing_summary_cols:
        details.append(f"missing_summary_cols={missing_summary_cols}")
    if missing_guided_cols:
        details.append(f"missing_guided_cols={missing_guided_cols}")

    record(
        rows,
        "crypto20_alpha_retest_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(details),
        rows_checked=max(len(results) + len(summary) + len(guided), 1),
        rows_failed=failures,
    )


def audit_crypto20_statistical_artifacts(rows: list[AuditRecord]) -> None:
    prefix = Path(SAVE_DIR) / "crypto20_statistical_"
    paths = {
        "folds": Path(f"{prefix}fold_metrics.csv"),
        "assets": Path(f"{prefix}asset_metrics.csv"),
        "summary": Path(f"{prefix}method_summary.csv"),
        "pairwise": Path(f"{prefix}pairwise_tests.csv"),
        "asset_pairwise": Path(f"{prefix}asset_pairwise_tests.csv"),
        "compact": Path(f"{prefix}test_summary.csv"),
        "corrected": Path(f"{prefix}multiple_testing.csv"),
        "claims": Path(f"{prefix}claims.csv"),
        "psr": Path(f"{prefix}sharpe_diagnostics.csv"),
        "ic_plot": Path(f"{prefix}ic_confidence_intervals.png"),
        "correction_plot": Path(f"{prefix}multiple_testing.png"),
        "psr_plot": Path(f"{prefix}sharpe_diagnostics.png"),
        "protocol": Path(BASE_DIR) / "reports" / "crypto20_statistical_protocol.md",
    }
    missing = [str(path.relative_to(BASE_DIR)) for path in paths.values() if not path.exists()]
    if missing:
        record(
            rows,
            "crypto20_statistical_artifacts",
            WARN,
            "artifact",
            f"Missing Phase 37 artifacts: {missing}",
            rows_checked=len(paths),
            rows_failed=len(missing),
        )
        return

    folds = pd.read_csv(paths["folds"])
    assets = pd.read_csv(paths["assets"])
    summary = pd.read_csv(paths["summary"])
    pairwise = pd.read_csv(paths["pairwise"])
    asset_pairwise = pd.read_csv(paths["asset_pairwise"])
    corrected = pd.read_csv(paths["corrected"])
    claims = pd.read_csv(paths["claims"])
    protocol_text = paths["protocol"].read_text(encoding="utf-8")

    expected_methods = {
        "global_lgbm",
        "regime_lgbm_hmm",
        "regime_lgbm_kmeans",
        "regime_lgbm_vol_bucket",
        "regime_lgbm_hmm_guided_gmm",
        "regime_lgbm_hmm_guided_hmm",
    }
    expected_references = {"global_lgbm", "regime_lgbm_hmm", "regime_lgbm_kmeans"}
    required_metric_cols = {
        "method",
        "regime_method",
        "target",
        "horizon",
        "IC",
        "mean_asset_IC",
        "median_asset_IC",
        "cross_sectional_IC",
        "pooled_IC",
        "Sharpe",
        "total_return",
        "drawdown",
        "turnover",
        "n_return_observations",
        "n_test_rows",
    }
    required_pairwise_cols = {
        "method",
        "reference_method",
        "metric",
        "mean_difference",
        "ci_low",
        "ci_high",
        "paired_t_p_value",
        "wilcoxon_p_value",
        "sign_test_p_value",
        "dm_p_value",
        "test_note",
    }
    required_corrected_cols = {
        "primary_p_value",
        "bh_q_all_tests",
        "holm_p_all_tests",
        "bh_q_by_metric",
        "holm_p_by_metric",
        "claim_status",
    }

    failures = 0
    details = [
        f"fold_rows={len(folds)}",
        f"asset_rows={len(assets)}",
        f"summary_rows={len(summary)}",
        f"pairwise_rows={len(pairwise)}",
    ]

    missing_fold_cols = sorted((required_metric_cols | {"fold"}) - set(folds.columns))
    missing_asset_cols = sorted((required_metric_cols | {"symbol"}) - set(assets.columns))
    missing_pairwise_cols = sorted(required_pairwise_cols - set(pairwise.columns))
    missing_corrected_cols = sorted(required_corrected_cols - set(corrected.columns))
    failures += len(missing_fold_cols) + len(missing_asset_cols)
    failures += len(missing_pairwise_cols) + len(missing_corrected_cols)

    if not missing_fold_cols:
        fold_methods = set(folds["method"].astype(str))
        method_failures = len(expected_methods - fold_methods)
        fold_counts = folds.groupby("method")["fold"].nunique()
        coverage_failures = int((fold_counts != 16).sum())
        row_failures = int(len(folds) != 16 * len(expected_methods))
        failures += method_failures + coverage_failures + row_failures
        details.extend(
            [
                f"fold_methods={len(fold_methods)}",
                f"fold_count_range={int(fold_counts.min())}-{int(fold_counts.max())}",
            ]
        )

    if not missing_asset_cols:
        asset_counts = assets.groupby("method")["symbol"].nunique()
        asset_failures = int((asset_counts != 20).sum()) + int(len(assets) != 20 * len(expected_methods))
        failures += asset_failures
        details.append(f"asset_count_range={int(asset_counts.min())}-{int(asset_counts.max())}")

    if not missing_pairwise_cols:
        primary = pairwise[
            (pairwise["method"] == "regime_lgbm_hmm_guided_hmm")
            & pairwise["reference_method"].isin(expected_references)
            & pairwise["metric"].isin({"IC", "Sharpe", "total_return", "nll_loss"})
        ]
        present_refs = set(primary["reference_method"].astype(str))
        present_metrics = set(primary["metric"].astype(str))
        primary_failures = len(expected_references - present_refs)
        primary_failures += len({"IC", "Sharpe", "total_return", "nll_loss"} - present_metrics)
        nll = pairwise[pairwise["metric"] == "nll_loss"]
        time_block_failures = int(nll.empty)
        time_block_failures += int(
            not nll["test_note"].astype(str).str.contains("averaged across assets per timestamp").all()
        )
        time_block_failures += int("n_time_blocks" not in pairwise.columns)
        failures += primary_failures + time_block_failures
        details.append(f"primary_comparison_rows={len(primary)}")

    asset_note_failures = int(
        not asset_pairwise["test_note"].astype(str).str.contains("cross-correlated").all()
    )
    failures += asset_note_failures

    required_protocol_phrases = [
        "primary inferential unit is the walk-forward fold",
        "asset-level p-values are descriptive",
        "no broad superiority claim",
        "no automatic epoch expansion",
    ]
    missing_phrases = [phrase for phrase in required_protocol_phrases if phrase not in protocol_text]
    failures += len(missing_phrases)

    if missing_fold_cols:
        details.append(f"missing_fold_cols={missing_fold_cols}")
    if missing_asset_cols:
        details.append(f"missing_asset_cols={missing_asset_cols}")
    if missing_pairwise_cols:
        details.append(f"missing_pairwise_cols={missing_pairwise_cols}")
    if missing_corrected_cols:
        details.append(f"missing_corrected_cols={missing_corrected_cols}")
    if missing_phrases:
        details.append(f"missing_protocol_phrases={missing_phrases}")

    guided_ic = corrected[
        (corrected["method"] == "regime_lgbm_hmm_guided_hmm")
        & (corrected["reference_method"] == "regime_lgbm_hmm")
        & (corrected["metric"] == "IC")
    ]
    if not guided_ic.empty:
        details.append(f"guided_vs_hmm_ic_p={float(guided_ic.iloc[0]['primary_p_value']):.3f}")

    record(
        rows,
        "crypto20_statistical_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(details),
        rows_checked=max(len(folds) + len(assets) + len(pairwise) + len(claims), 1),
        rows_failed=failures,
    )


def audit_statistical_artifacts(rows: list[AuditRecord]) -> None:
    fold_path = Path(SAVE_DIR) / "statistical_fold_metrics.csv"
    summary_path = Path(SAVE_DIR) / "statistical_method_summary.csv"
    pairwise_path = Path(SAVE_DIR) / "statistical_pairwise_tests.csv"
    compact_path = Path(SAVE_DIR) / "statistical_test_summary.csv"
    corrected_path = Path(SAVE_DIR) / "statistical_multiple_testing.csv"
    claims_path = Path(SAVE_DIR) / "statistical_claims.csv"
    psr_path = Path(SAVE_DIR) / "statistical_sharpe_diagnostics.csv"
    plot_path = Path(SAVE_DIR) / "statistical_ic_confidence_intervals.png"
    correction_plot_path = Path(SAVE_DIR) / "statistical_multiple_testing.png"
    psr_plot_path = Path(SAVE_DIR) / "statistical_sharpe_diagnostics.png"

    if not fold_path.exists():
        record(
            rows,
            "statistical_test_artifacts",
            WARN,
            "artifact",
            "statistical_fold_metrics.csv is missing; run statistical_tests.py for Phase 15A significance tests.",
        )
        return

    folds = pd.read_csv(fold_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    pairwise = pd.read_csv(pairwise_path) if pairwise_path.exists() else pd.DataFrame()
    compact = pd.read_csv(compact_path) if compact_path.exists() else pd.DataFrame()
    corrected = pd.read_csv(corrected_path) if corrected_path.exists() else pd.DataFrame()
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    psr = pd.read_csv(psr_path) if psr_path.exists() else pd.DataFrame()

    required_methods = FOLD_LOCAL_METHODS
    expected_fold_rows = 18 * len(required_methods)
    missing_files = [
        str(path.name)
        for path in [
            summary_path,
            pairwise_path,
            compact_path,
            corrected_path,
            claims_path,
            psr_path,
            plot_path,
            correction_plot_path,
            psr_plot_path,
        ]
        if not path.exists()
    ]
    missing_fold_cols = sorted(
        {
            "method", "fold", "IC", "mean_asset_IC", "median_asset_IC",
            "cross_sectional_IC", "pooled_IC", "signal_IC", "Sharpe",
            "total_return", "n_return_observations", "n_test_rows",
        }
        - set(folds.columns)
    )
    missing_summary_cols = sorted(
        {"method", "mean_fold_IC", "IC_ci_low", "IC_ci_high", "positive_ic_folds"} - set(summary.columns)
    )
    missing_pairwise_cols = sorted(
        {"method", "reference_method", "metric", "mean_difference", "paired_t_p_value"} - set(pairwise.columns)
    )
    missing_corrected_cols = sorted(
        {"metric", "primary_p_value", "bh_q_by_metric", "holm_p_by_metric", "claim_status"}
        - set(corrected.columns)
    )
    missing_claim_cols = sorted(
        {"comparison", "metric", "primary_p_value", "bh_q_by_metric", "holm_p_by_metric", "claim_status"}
        - set(claims.columns)
    )
    missing_psr_cols = sorted(
        {"method", "annualized_sharpe", "psr_gt_0", "skew", "kurtosis", "n_periods"} - set(psr.columns)
    )

    failures = (
        len(missing_files)
        + len(missing_fold_cols)
        + len(missing_summary_cols)
        + len(missing_pairwise_cols)
        + len(missing_corrected_cols)
        + len(missing_claim_cols)
        + len(missing_psr_cols)
    )
    detail_parts = [f"fold_metric_rows={len(folds)}, expected={expected_fold_rows}"]

    if len(folds) != expected_fold_rows:
        failures += 1
        detail_parts.append("unexpected_fold_metric_row_count")

    if not missing_fold_cols:
        methods = set(folds["method"])
        missing_methods = sorted(set(required_methods) - methods)
        failures += len(missing_methods)
        detail_parts.append(f"methods={len(methods)}")
        if missing_methods:
            detail_parts.append(f"missing_methods={missing_methods}")
        fold_counts = folds.groupby("method")["fold"].nunique()
        incomplete = int((fold_counts != 18).sum())
        failures += incomplete
        if incomplete:
            detail_parts.append(f"incomplete_fold_methods={incomplete}")

    if not summary.empty:
        detail_parts.append(f"summary_rows={len(summary)}")
        if len(summary) != len(required_methods):
            failures += 1
            detail_parts.append("unexpected_summary_row_count")

    if not pairwise.empty:
        detail_parts.append(f"pairwise_rows={len(pairwise)}")
        focus = pairwise[pairwise["metric"].isin(["IC", "Sharpe", "nll_loss"])] if "metric" in pairwise else pd.DataFrame()
        if focus.empty:
            failures += 1
            detail_parts.append("missing_focus_pairwise_tests")

    if not compact.empty:
        detail_parts.append(f"compact_rows={len(compact)}")
    if not corrected.empty:
        detail_parts.append(f"corrected_rows={len(corrected)}")
    if not claims.empty:
        detail_parts.append(f"claim_rows={len(claims)}")
    if not psr.empty:
        detail_parts.append(f"psr_rows={len(psr)}")
        if len(psr) != len(required_methods):
            failures += 1
            detail_parts.append("unexpected_psr_row_count")

    if missing_files:
        detail_parts.append(f"missing_artifacts={missing_files}")
    if missing_fold_cols:
        detail_parts.append(f"missing_fold_columns={missing_fold_cols}")
    if missing_summary_cols:
        detail_parts.append(f"missing_summary_columns={missing_summary_cols}")
    if missing_pairwise_cols:
        detail_parts.append(f"missing_pairwise_columns={missing_pairwise_cols}")
    if missing_corrected_cols:
        detail_parts.append(f"missing_corrected_columns={missing_corrected_cols}")
    if missing_claim_cols:
        detail_parts.append(f"missing_claim_columns={missing_claim_cols}")
    if missing_psr_cols:
        detail_parts.append(f"missing_psr_columns={missing_psr_cols}")

    record(
        rows,
        "statistical_test_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(expected_fold_rows, 1),
        rows_failed=failures,
    )


def audit_run_registry(rows: list[AuditRecord]) -> None:
    runs_dir = Path(BASE_DIR) / "runs"
    index_path = runs_dir / "run_index.csv"
    latest_path = runs_dir / "latest_run.json"

    if not index_path.exists():
        record(
            rows,
            "run_registry_artifacts",
            WARN,
            "artifact",
            "runs/run_index.csv is missing; run archive_run.py before freezing a baseline.",
        )
        return

    index = pd.read_csv(index_path)
    failures = 0
    detail_parts = [f"registered_runs={len(index)}"]
    if index.empty:
        failures += 1
        detail_parts.append("run_index_empty")

    latest_run_id = ""
    if latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            latest_run_id = str(latest.get("run_id", ""))
        except (ValueError, OSError):
            failures += 1
            detail_parts.append("latest_run_json_unreadable")
    else:
        failures += 1
        detail_parts.append("latest_run_json_missing")

    if not latest_run_id and not index.empty:
        latest_run_id = str(index.iloc[-1]["run_id"])

    manifest_path = runs_dir / latest_run_id / "manifest.json"
    artifact_manifest_path = runs_dir / latest_run_id / "artifact_manifest.csv"
    artifact_count = 0
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact_count = int(manifest.get("artifact_count", 0))
            missing = manifest.get("missing_artifacts", [])
            if isinstance(missing, list) and missing:
                failures += len(missing)
                detail_parts.append(f"missing_artifacts={len(missing)}")
            detail_parts.append(f"latest_run={latest_run_id}")
            detail_parts.append(f"latest_artifacts={artifact_count}")
        except (ValueError, OSError):
            failures += 1
            detail_parts.append("manifest_unreadable")
    else:
        failures += 1
        detail_parts.append(f"manifest_missing={latest_run_id}")

    if artifact_manifest_path.exists():
        artifacts = pd.read_csv(artifact_manifest_path)
        missing_files = []
        bad_hashes = []
        text_hash_drifts = []
        for item in artifacts.itertuples(index=False):
            path = Path(BASE_DIR) / item.archived_path
            if not path.exists():
                missing_files.append(item.archived_path)
                continue
            expected_hash = str(item.sha256)
            if expected_hash and expected_hash != "nan":
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected_hash:
                    suffix = Path(str(item.archived_path)).suffix.lower()
                    if suffix in {".md", ".txt"}:
                        text_hash_drifts.append(item.archived_path)
                    else:
                        bad_hashes.append(item.archived_path)
        if artifact_count and len(artifacts) != artifact_count:
            failures += 1
            detail_parts.append(f"artifact_count_mismatch={len(artifacts)} vs {artifact_count}")
        if missing_files:
            failures += len(missing_files)
            detail_parts.append(f"missing_archived_files={len(missing_files)}")
        if bad_hashes:
            failures += len(bad_hashes)
            detail_parts.append(f"bad_hashes={len(bad_hashes)}")
        if text_hash_drifts:
            detail_parts.append(f"text_hash_drifts={len(text_hash_drifts)}")
    else:
        failures += 1
        detail_parts.append("artifact_manifest_missing")

    record(
        rows,
        "run_registry_artifacts",
        FAIL if failures else PASS,
        "critical",
        "; ".join(detail_parts),
        rows_checked=max(artifact_count, 1),
        rows_failed=failures,
    )


def audit_phase39_fold_local_artifacts(rows: list[AuditRecord]) -> None:
    model_dir = Path(SAVE_DIR)
    required = {
        "manifest": model_dir / "crypto20_fold_local_encoder_manifest.csv",
        "coverage": model_dir / "crypto20_fold_local_encoder_coverage.csv",
        "results": model_dir / "crypto20_fold_local_experiment_results.csv",
        "loss": model_dir / "crypto20_fold_local_encoder_loss.csv",
        "fold_metrics": model_dir / "crypto20_fold_local_fold_metrics.csv",
        "statistical_summary": model_dir / "crypto20_fold_local_statistical_method_summary.csv",
        "statistical_claims": model_dir / "crypto20_fold_local_statistical_claims.csv",
        "report": Path(BASE_DIR) / "reports" / "phase39_fold_local_results.md",
    }
    missing = [str(path.relative_to(BASE_DIR)) for path in required.values() if not path.exists()]
    if missing:
        record(
            rows,
            "phase39_fold_local_artifacts",
            WARN,
            "research",
            f"Phase 39 has not produced a complete artifact set: {missing}",
            rows_checked=len(required),
            rows_failed=len(missing),
        )
        return

    manifest = pd.read_csv(required["manifest"])
    coverage = pd.read_csv(required["coverage"])
    results = pd.read_csv(required["results"])
    fold_metrics = pd.read_csv(required["fold_metrics"])
    statistical_summary = pd.read_csv(required["statistical_summary"])
    statistical_claims = pd.read_csv(required["statistical_claims"])
    expected_methods = {
        "global_lgbm",
        "regime_lgbm_vol_bucket",
        "regime_lgbm_kmeans",
        "regime_lgbm_hmm",
        "regime_lgbm_contrastive",
        "regime_lgbm_contrastive_hmm",
        "regime_lgbm_hmm_guided_gmm",
        "regime_lgbm_hmm_guided_hmm",
    }
    failures: list[str] = []
    manifest_columns = {
        "fold", "encoder_method", "outer_train_end", "inner_train_end",
        "inner_validation_start", "inner_validation_end", "outer_test_start",
        "outer_test_end", "selected_epoch", "max_epochs", "scaler_training_rows",
        "calendar_train_start", "calendar_train_end", "calendar_test_start",
        "calendar_test_end", "input_hash", "model_hash",
    }
    coverage_columns = {
        "fold", "method", "train_rows", "test_assignment_rows", "test_prediction_rows",
    }
    if not manifest_columns.issubset(manifest.columns):
        failures.append(f"manifest columns missing={sorted(manifest_columns - set(manifest.columns))}")
    if not coverage_columns.issubset(coverage.columns):
        failures.append(f"coverage columns missing={sorted(coverage_columns - set(coverage.columns))}")
    if "method" not in results:
        failures.append("results method column missing")

    if not failures:
        encoders_by_fold = manifest.groupby("fold")["encoder_method"].agg(lambda x: set(x.astype(str)))
        bad_encoders = encoders_by_fold[encoders_by_fold != {"vanilla", "guided"}]
        if len(bad_encoders):
            failures.append(f"encoder lineage incomplete for folds={bad_encoders.index.tolist()}")
        boundary_bad = manifest[
            (manifest["inner_train_end"] > manifest["inner_validation_start"])
            | (manifest["inner_validation_start"] >= manifest["inner_validation_end"])
            | (manifest["inner_validation_end"] > manifest["outer_train_end"])
            | (manifest["outer_train_end"] >= manifest["outer_test_start"])
            | (manifest["outer_test_start"] >= manifest["outer_test_end"])
        ]
        if len(boundary_bad):
            failures.append(f"invalid temporal boundary rows={len(boundary_bad)}")
        calendar_train_end = pd.to_datetime(manifest["calendar_train_end"], errors="coerce")
        calendar_test_start = pd.to_datetime(manifest["calendar_test_start"], errors="coerce")
        calendar_bad = manifest[
            calendar_train_end.isna()
            | calendar_test_start.isna()
            | (calendar_train_end >= calendar_test_start)
        ]
        if len(calendar_bad):
            failures.append(f"cross-asset calendar overlap rows={len(calendar_bad)}")
        epoch_bad = manifest[
            (manifest["selected_epoch"] < 1)
            | (manifest["selected_epoch"] > manifest["max_epochs"])
        ]
        if len(epoch_bad):
            failures.append(f"invalid selected epochs={len(epoch_bad)}")
        hash_bad = manifest[
            (manifest["input_hash"].astype(str).str.len() != 64)
            | (manifest["model_hash"].astype(str).str.len() != 64)
        ]
        if len(hash_bad):
            failures.append(f"invalid lineage hashes={len(hash_bad)}")

        for fold, fold_coverage in coverage.groupby("fold"):
            methods = set(fold_coverage["method"].astype(str))
            if methods != expected_methods:
                failures.append(f"fold {fold} method set mismatch")
            if fold_coverage["train_rows"].nunique() != 1:
                failures.append(f"fold {fold} unequal training coverage")
            if fold_coverage["test_assignment_rows"].nunique() != 1:
                failures.append(f"fold {fold} unequal assignment coverage")
            if fold_coverage["test_prediction_rows"].nunique() != 1:
                failures.append(f"fold {fold} unequal prediction coverage")
            if not (fold_coverage["test_assignment_rows"] == fold_coverage["test_prediction_rows"]).all():
                failures.append(f"fold {fold} assignment/prediction mismatch")
        if set(results["method"].astype(str)) != expected_methods:
            failures.append("result method set mismatch")
        phase39_metric_columns = {
            "fold", "method", "IC", "mean_asset_IC", "median_asset_IC",
            "cross_sectional_IC", "pooled_IC", "Sharpe",
            "n_return_observations", "n_test_rows",
        }
        if not phase39_metric_columns.issubset(fold_metrics.columns):
            failures.append("fold metrics schema mismatch")
        else:
            expected_fold_metric_rows = manifest["fold"].nunique() * len(expected_methods)
            if len(fold_metrics) != expected_fold_metric_rows:
                failures.append(
                    f"fold metric rows={len(fold_metrics)} expected={expected_fold_metric_rows}"
                )
            if set(fold_metrics["method"].astype(str)) != expected_methods:
                failures.append("fold metric method set mismatch")
        if set(statistical_summary.get("method", pd.Series(dtype=str)).astype(str)) != expected_methods:
            failures.append("statistical summary method set mismatch")
        if "n_folds" not in statistical_summary or not (
            pd.to_numeric(statistical_summary["n_folds"], errors="coerce")
            == manifest["fold"].nunique()
        ).all():
            failures.append("statistical summary fold coverage mismatch")
        if statistical_claims.empty or not {
            "comparison", "metric", "claim_status"
        }.issubset(statistical_claims.columns):
            failures.append("statistical claims missing or incomplete")

    record(
        rows,
        "phase39_fold_local_validity",
        FAIL if failures else PASS,
        "critical",
        "; ".join(failures) if failures else (
            f"Validated temporal boundaries, hashes, encoder lineage, and equal coverage for "
            f"{manifest['fold'].nunique()} fold(s) and all eight methods."
        ),
        rows_checked=len(manifest) + len(coverage) + len(statistical_summary) + len(statistical_claims),
        rows_failed=len(failures),
    )

    folds = int(manifest["fold"].nunique()) if "fold" in manifest else 0
    full = folds == 16 and int(manifest["max_epochs"].min()) >= 30
    if "run_kind" in manifest:
        full = full and set(manifest["run_kind"].astype(str)) == {"full"}
    record(
        rows,
        "phase39_full_development_run",
        PASS if full else WARN,
        "research",
        (
            "Full 16-fold, 30-epoch-cap frozen-budget development protocol is present."
            if full
            else f"Only a smoke/development gate is present ({folds} fold(s)); no Phase 39 performance claim is authorized."
        ),
        rows_checked=folds,
        rows_failed=0 if full else max(1, 16 - folds),
    )


def write_outputs(rows: list[AuditRecord], fold_audit: pd.DataFrame) -> pd.DataFrame:
    out_dir = Path(SAVE_DIR)
    out_dir.mkdir(exist_ok=True)
    audit = pd.DataFrame([row.__dict__ for row in rows])
    status_order = pd.Categorical(audit["status"], [FAIL, WARN, PASS], ordered=True)
    audit = audit.assign(_status_order=status_order).sort_values(["_status_order", "check"]).drop(columns="_status_order")
    audit.to_csv(out_dir / "validation_audit.csv", index=False)
    fold_audit.to_csv(out_dir / "fold_audit.csv", index=False)
    return audit


def audit_development_freeze(rows: list[AuditRecord]) -> None:
    try:
        from freeze_development_dataset import (
            DEFAULT_CONFIG,
            build_artifacts,
            verify_existing,
        )

        summary, symbols, folds, universe = build_artifacts(DEFAULT_CONFIG)
        verify_existing(summary, symbols, folds, universe)
        record(
            rows,
            "crypto20_development_freeze",
            PASS,
            "critical",
            f"{summary['freeze_id']} verified: {summary['symbol_count']} symbols, "
            f"{summary['prediction_rows']} rows, {summary['fold_count']} folds, "
            "and matching database/experiment-data hashes.",
            rows_checked=int(summary["prediction_rows"]),
            rows_failed=0,
        )
    except Exception as exc:
        record(
            rows,
            "crypto20_development_freeze",
            FAIL,
            "critical",
            str(exc),
            rows_checked=1,
            rows_failed=1,
        )


def audit_repaired_classical_baseline(rows: list[AuditRecord]) -> None:
    prefix = Path(SAVE_DIR) / "crypto20_repaired_classical_"
    paths = {
        "results": Path(f"{prefix}experiment_results.csv"),
        "folds": Path(f"{prefix}fold_metrics.csv"),
        "coverage": Path(f"{prefix}coverage.csv"),
        "manifest": Path(f"{prefix}manifest.csv"),
    }
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        record(
            rows,
            "repaired_classical_baseline",
            WARN,
            "artifact",
            f"Full repaired classical run is pending; missing={missing}",
            rows_checked=1,
            rows_failed=len(missing),
        )
        return
    results = pd.read_csv(paths["results"])
    folds = pd.read_csv(paths["folds"])
    coverage = pd.read_csv(paths["coverage"])
    manifest = pd.read_csv(paths["manifest"])
    expected = {
        "global_lgbm",
        "regime_lgbm_hmm",
        "regime_lgbm_kmeans",
        "regime_lgbm_vol_bucket",
    }
    failures = []
    if set(results.get("method", pd.Series(dtype=str)).astype(str)) != expected:
        failures.append("result method set mismatch")
    required_metrics = {
        "IC", "mean_asset_IC", "median_asset_IC", "cross_sectional_IC",
        "pooled_IC", "Sharpe", "n_return_observations", "n_test_rows",
    }
    if not required_metrics.issubset(folds.columns):
        failures.append("repaired metric columns missing")
    if len(folds) != 16 * len(expected) or folds["fold"].nunique() != 16:
        failures.append("fold metric coverage mismatch")
    if coverage.groupby("fold")["test_rows"].nunique().max() != 1:
        failures.append("unequal method coverage")
    if set(manifest.get("freeze_id", pd.Series(dtype=str)).astype(str)) != {
        "crypto20-development-v1"
    }:
        failures.append("freeze ID mismatch")
    train_end = pd.to_datetime(manifest.get("calendar_train_end"), errors="coerce")
    test_start = pd.to_datetime(manifest.get("calendar_test_start"), errors="coerce")
    if train_end.isna().any() or test_start.isna().any() or (train_end >= test_start).any():
        failures.append("calendar separation failure")
    record(
        rows,
        "repaired_classical_baseline",
        FAIL if failures else PASS,
        "critical",
        "; ".join(failures) if failures else (
            "Validated 16 frozen calendar folds, four classical methods, equal coverage, "
            "and repaired evaluation metrics."
        ),
        rows_checked=len(folds) + len(coverage) + len(manifest),
        rows_failed=len(failures),
    )


def main() -> None:
    args = parse_args()
    rows: list[AuditRecord] = []

    audit_database_tables(rows, args.symbols)
    if any(row.status == FAIL and row.severity == "critical" for row in rows):
        audit = write_outputs(rows, pd.DataFrame())
        print(audit.to_string(index=False))
        raise SystemExit("Validation audit failed before data-level checks.")

    audit_feature_source(rows)
    audit_feature_target_schema(rows, args.symbols)
    audit_target_horizon_loss(rows, args.symbols)
    df, _, folds = audit_common_universe(rows, args.symbols)
    fold_audit = audit_folds(rows, df, folds)
    audit_predictions(rows, fold_audit)
    audit_walkforward_artifacts(rows)
    audit_robustness_artifacts(rows)
    audit_robustness_stress_artifacts(rows)
    audit_regime_quality_artifacts(rows)
    audit_compute_plan_artifacts(rows)
    audit_ablation_artifacts(rows)
    audit_paper_statistical_artifacts(rows)
    audit_guided_encoder_artifacts(rows)
    audit_time_frequency_encoder_artifacts(rows)
    audit_interpretability_artifacts(rows)
    audit_literature_positioning_artifacts(rows)
    audit_paper_protocol_artifacts(rows)
    audit_phase38_control_artifacts(rows)
    audit_development_freeze(rows)
    audit_repaired_classical_baseline(rows)
    audit_phase39_fold_local_artifacts(rows)
    audit_paper_draft_artifacts(rows)
    audit_reproducibility_artifacts(rows)
    audit_multiasset_universe_artifacts(rows)
    audit_crypto20_regime_artifacts(rows)
    audit_crypto20_guided_readiness_artifacts(rows)
    audit_crypto20_guided_encoder_artifacts(rows)
    audit_crypto20_alpha_retest_artifacts(rows)
    audit_crypto20_statistical_artifacts(rows)
    audit_statistical_artifacts(rows)
    audit_run_registry(rows)
    audit = write_outputs(rows, fold_audit)

    print("\nValidation audit summary:")
    print(audit[["check", "status", "severity", "rows_checked", "rows_failed", "detail"]].to_string(index=False))

    hard_fails = audit[(audit["status"] == FAIL) & (audit["severity"] == "critical")]
    if not hard_fails.empty:
        raise SystemExit(f"Validation audit failed: {len(hard_fails)} critical checks failed.")

    warn_count = int((audit["status"] == WARN).sum())
    print(f"\nOK: validation audit completed with {warn_count} warning(s).")
    print(f"Saved: {Path(SAVE_DIR) / 'validation_audit.csv'}")
    print(f"Saved: {Path(SAVE_DIR) / 'fold_audit.csv'}")


if __name__ == "__main__":
    main()
