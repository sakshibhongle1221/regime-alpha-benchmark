from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from config import BASE_DIR, DB_PATH, SAVE_DIR


DEFAULT_CONFIG = Path(BASE_DIR) / "configs" / "phase43b_locked_holdout_registration_v1.json"
PHASE43A_CONFIG = Path(BASE_DIR) / "configs" / "phase43_locked_holdout_freeze_v1.json"
QUALITY_PATH = Path(SAVE_DIR) / "phase43b_holdout_candidate_quality.csv"
SYMBOLS_PATH = Path(SAVE_DIR) / "phase43b_registered_holdout_symbols.csv"
MANIFEST_PATH = Path(SAVE_DIR) / "phase43b_locked_holdout_registration_manifest.csv"
REPORT_PATH = Path(BASE_DIR) / "reports" / "phase43b_locked_holdout_registration.md"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "empty"
    normalized = frame[columns].copy()
    for col in normalized.columns:
        normalized[col] = normalized[col].astype(str)
    hashed = pd.util.hash_pandas_object(normalized, index=False, categorize=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "registration_id",
        "parent_freeze_id",
        "data_role",
        "source_universe",
        "development_universe",
        "minimum_assets",
        "quality_gates",
        "forbidden_inputs",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Phase 43B registration config missing keys: {missing}")
    if config["data_role"] != "locked_unobserved_registration_only":
        raise ValueError("Phase 43B registration must not mark locked outcomes as inspected.")
    forbidden = set(config["forbidden_inputs"])
    required_forbidden = {
        "model_predictions",
        "alpha_metrics",
        "threshold_search_on_holdout",
        "candidate_selection_on_holdout",
        "rerun_after_failure",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("Phase 43B registration must forbid model-outcome and selection inputs.")
    return config


def validate_parent_freeze(config: dict[str, Any]) -> dict[str, Any]:
    parent = json.loads(PHASE43A_CONFIG.read_text(encoding="utf-8"))
    if parent.get("freeze_id") != config["parent_freeze_id"]:
        raise ValueError("Phase 43B registration does not match the Phase 43A freeze ID.")
    if parent.get("final_candidate", {}).get("method") != "regime_lgbm_hmm_guided_hmm":
        raise ValueError("Phase 43B must inherit the frozen guided-HMM candidate.")
    return parent


def load_universes(config: dict[str, Any]) -> tuple[pd.DataFrame, set[str]]:
    candidates = pd.read_csv(Path(BASE_DIR) / config["source_universe"])
    development = pd.read_csv(Path(BASE_DIR) / config["development_universe"])
    candidates["symbol"] = candidates["symbol"].astype(str).str.upper()
    development_symbols = set(development["symbol"].astype(str).str.upper())
    if "design_rank" not in candidates:
        raise ValueError("Source universe must contain design_rank.")
    return candidates.sort_values("design_rank").reset_index(drop=True), development_symbols


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def load_table_counts(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    symbols: list[str],
    row_col: str,
    earliest_col: str,
    latest_col: str,
) -> pd.DataFrame:
    if not symbols or not table_exists(con, table_name):
        return pd.DataFrame({"symbol": symbols})
    return con.execute(
        f"""
        SELECT
            symbol,
            COUNT(*) AS {row_col},
            MIN(open_time) AS {earliest_col},
            MAX(open_time) AS {latest_col}
        FROM {table_name}
        WHERE symbol IN ({",".join(["?"] * len(symbols))})
        GROUP BY symbol
        ORDER BY symbol
        """,
        symbols,
    ).df()


def load_ohlcv_quality(con: duckdb.DuckDBPyConnection, symbols: list[str]) -> pd.DataFrame:
    if not symbols or not table_exists(con, "ohlcv"):
        return pd.DataFrame({"symbol": symbols})
    return con.execute(
        f"""
        WITH ordered AS (
            SELECT
                symbol,
                open_time,
                quote_volume,
                high,
                low,
                close,
                open_time - LAG(open_time) OVER (
                    PARTITION BY symbol ORDER BY open_time
                ) AS gap
            FROM ohlcv
            WHERE symbol IN ({",".join(["?"] * len(symbols))})
        )
        SELECT
            symbol,
            COUNT(*) AS ohlcv_rows,
            MIN(open_time) AS ohlcv_earliest,
            MAX(open_time) AS ohlcv_latest,
            MEDIAN(quote_volume) AS median_quote_volume,
            MEDIAN(CASE WHEN close != 0 THEN (high - low) / close ELSE NULL END) AS median_range_proxy,
            SUM(CASE WHEN gap > INTERVAL 1 HOUR THEN 1 ELSE 0 END) AS gap_count,
            COALESCE(MAX(EXTRACT(EPOCH FROM gap) / 3600.0), 1.0) AS max_gap_hours
        FROM ordered
        GROUP BY symbol
        ORDER BY symbol
        """,
        symbols,
    ).df()


def build_candidate_quality(config: dict[str, Any], db_path: str = DB_PATH) -> pd.DataFrame:
    candidates, development_symbols = load_universes(config)
    candidates["symbol"] = candidates["symbol"].astype(str).str.upper()
    base = candidates.copy()
    base["in_development_universe"] = base["symbol"].isin(development_symbols)
    base["external_candidate"] = ~base["in_development_universe"]

    symbols = base["symbol"].tolist()
    con = duckdb.connect(db_path, read_only=True)
    try:
        quality = base.merge(load_ohlcv_quality(con, symbols), on="symbol", how="left")
        quality = quality.merge(
            load_table_counts(con, "features", symbols, "feature_rows", "feature_earliest", "feature_latest"),
            on="symbol",
            how="left",
        )
        quality = quality.merge(
            load_table_counts(con, "targets", symbols, "target_rows", "target_earliest", "target_latest"),
            on="symbol",
            how="left",
        )
    finally:
        con.close()

    for col in ["ohlcv_rows", "feature_rows", "target_rows", "gap_count"]:
        quality[col] = quality[col].fillna(0).astype(int)
    for col in ["median_quote_volume", "median_range_proxy", "max_gap_hours"]:
        quality[col] = quality[col].fillna(0.0).astype(float)

    gates = config["quality_gates"]
    quality["coverage_pass"] = quality["ohlcv_rows"] >= int(gates["minimum_hourly_bars"])
    quality["gap_pass"] = quality["max_gap_hours"] <= float(gates["maximum_gap_hours"])
    quality["feature_pass"] = quality["feature_rows"] > 0 if gates["require_features"] else True
    quality["target_pass"] = quality["target_rows"] > 0 if gates["require_targets"] else True
    quality["instrument_pass"] = ~quality.get("is_stable_or_synthetic", pd.Series(False, index=quality.index)).fillna(False).astype(bool)
    quality["holdout_eligible"] = (
        quality["external_candidate"]
        & quality["coverage_pass"]
        & quality["gap_pass"]
        & quality["feature_pass"]
        & quality["target_pass"]
        & quality["instrument_pass"]
    )
    quality["failure_reason"] = ""
    quality.loc[quality["in_development_universe"], "failure_reason"] += "development_universe;"
    quality.loc[~quality["coverage_pass"], "failure_reason"] += "insufficient_ohlcv;"
    quality.loc[~quality["gap_pass"], "failure_reason"] += "excessive_gaps;"
    quality.loc[~quality["feature_pass"], "failure_reason"] += "missing_features;"
    quality.loc[~quality["target_pass"], "failure_reason"] += "missing_targets;"
    quality.loc[~quality["instrument_pass"], "failure_reason"] += "forbidden_instrument;"
    quality["failure_reason"] = quality["failure_reason"].str.rstrip(";")
    quality.loc[quality["holdout_eligible"], "failure_reason"] = ""
    return quality.sort_values("design_rank").reset_index(drop=True)


def select_registered_symbols(config: dict[str, Any], quality: pd.DataFrame) -> pd.DataFrame:
    selected = quality[quality["holdout_eligible"]].sort_values("design_rank").head(int(config["minimum_assets"])).copy()
    columns = [
        "design_rank",
        "symbol",
        "base_asset",
        "universe_group",
        "ohlcv_rows",
        "feature_rows",
        "target_rows",
        "ohlcv_earliest",
        "ohlcv_latest",
        "median_quote_volume",
        "median_range_proxy",
        "max_gap_hours",
    ]
    for col in columns:
        if col not in selected:
            selected[col] = pd.NA
    selected = selected[columns]
    selected.insert(0, "registration_id", config["registration_id"])
    selected.insert(1, "data_role", config["data_role"])
    return selected


def build_manifest(config: dict[str, Any], parent: dict[str, Any], quality: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_symbols = selected["symbol"].astype(str).tolist() if "symbol" in selected else []
    external_candidates = quality[quality["external_candidate"]]
    eligible_count = int(external_candidates["holdout_eligible"].sum()) if "holdout_eligible" in external_candidates else 0
    status = "registered_ready" if len(selected_symbols) >= int(config["minimum_assets"]) else "blocked_not_ready"
    missing_count = max(0, int(config["minimum_assets"]) - len(selected_symbols))
    rows = [
        {
            "registration_id": config["registration_id"],
            "parent_freeze_id": config["parent_freeze_id"],
            "item": "registration_status",
            "value": status,
            "notes": "No model outcomes inspected; this is a data-membership gate only.",
        },
        {
            "registration_id": config["registration_id"],
            "parent_freeze_id": config["parent_freeze_id"],
            "item": "final_candidate",
            "value": parent["final_candidate"]["method"],
            "notes": "Inherited from Phase 43A; not reselected here.",
        },
        {
            "registration_id": config["registration_id"],
            "parent_freeze_id": config["parent_freeze_id"],
            "item": "selected_asset_count",
            "value": str(len(selected_symbols)),
            "notes": f"Minimum required: {config['minimum_assets']}.",
        },
        {
            "registration_id": config["registration_id"],
            "parent_freeze_id": config["parent_freeze_id"],
            "item": "eligible_external_candidate_count",
            "value": str(eligible_count),
            "notes": "Count before truncating to the registered holdout size.",
        },
        {
            "registration_id": config["registration_id"],
            "parent_freeze_id": config["parent_freeze_id"],
            "item": "missing_assets_before_evaluation",
            "value": str(missing_count),
            "notes": "If positive, ingest/build feature/target data before evaluation.",
        },
        {
            "registration_id": config["registration_id"],
            "parent_freeze_id": config["parent_freeze_id"],
            "item": "selected_symbols",
            "value": ",".join(selected_symbols),
            "notes": "Empty unless the minimum locked-holdout asset count is met.",
        },
        {
            "registration_id": config["registration_id"],
            "parent_freeze_id": config["parent_freeze_id"],
            "item": "selected_symbols_sha256",
            "value": sha256_text(",".join(selected_symbols)),
            "notes": "Hash of ordered registered symbols.",
        },
        {
            "registration_id": config["registration_id"],
            "parent_freeze_id": config["parent_freeze_id"],
            "item": "candidate_quality_sha256",
            "value": frame_hash(
                quality,
                [
                    "design_rank",
                    "symbol",
                    "external_candidate",
                    "ohlcv_rows",
                    "feature_rows",
                    "target_rows",
                    "max_gap_hours",
                    "holdout_eligible",
                    "failure_reason",
                ],
            ),
            "notes": "Hash of non-outcome candidate quality state.",
        },
        {
            "registration_id": config["registration_id"],
            "parent_freeze_id": config["parent_freeze_id"],
            "item": "forbidden_inputs_confirmed",
            "value": ",".join(config["forbidden_inputs"]),
            "notes": "Registration does not read predictions, alpha metrics, or method rankings.",
        },
    ]
    return pd.DataFrame(rows)


def report_text(config: dict[str, Any], manifest: pd.DataFrame, quality: pd.DataFrame, selected: pd.DataFrame) -> str:
    status = manifest.set_index("item").loc["registration_status", "value"]
    selected_asset_count = manifest.set_index("item").loc["selected_asset_count", "value"]
    missing = manifest.set_index("item").loc["missing_assets_before_evaluation", "value"]
    eligible = quality[quality["holdout_eligible"]].sort_values("design_rank")
    pending = quality[(quality["external_candidate"]) & (~quality["holdout_eligible"])].sort_values("design_rank").head(15)

    selected_rows = "\n".join(
        f"| {row.design_rank} | `{row.symbol}` | {row.ohlcv_rows} | {row.feature_rows} | {row.target_rows} |"
        for row in selected.itertuples(index=False)
    ) or "| - | - | - | - | - |"
    pending_rows = "\n".join(
        f"| {row.design_rank} | `{row.symbol}` | {row.ohlcv_rows} | {row.feature_rows} | {row.target_rows} | {row.failure_reason} |"
        for row in pending.itertuples(index=False)
    ) or "| - | - | - | - | - | - |"

    return f"""# Phase 43B Locked Holdout Registration

## Status

Phase 43B registration checks whether the external locked holdout is ready before any frozen-model outcome is evaluated.

- Registration ID: `{config['registration_id']}`
- Parent freeze: `{config['parent_freeze_id']}`
- Data role: `{config['data_role']}`
- Registration status: `{status}`
- Selected locked assets: {selected_asset_count}
- Minimum required assets: {config['minimum_assets']}
- Missing assets before evaluation: {missing}

No model predictions, alpha metrics, method rankings, threshold search, or locked-holdout performance outcomes are read in this phase.

## Registered Symbols

| Design rank | Symbol | OHLCV rows | Feature rows | Target rows |
|---:|---|---:|---:|---:|
{selected_rows}

## First External Candidates Still Not Registered

These rows explain why the local machine is or is not ready for locked evaluation. This is a data-readiness table, not a model-performance table.

| Design rank | Symbol | OHLCV rows | Feature rows | Target rows | Reason |
|---:|---|---:|---:|---:|---|
{pending_rows}

## Interpretation

If status is `registered_ready`, the next valid action is one locked evaluation of the Phase 43A frozen candidate.

If status is `blocked_not_ready`, the next valid action is to ingest/build feature and target data for the next pre-ranked external candidates, rerun this registration, and only then evaluate.

Forbidden wording:

```text
Phase 43B registration proves generalization.
Phase 43B registration selected assets using model performance.
Phase 43B registration allows retrying the locked evaluation after failure.
```
"""


def write_artifacts(config_path: Path = DEFAULT_CONFIG, db_path: str = DB_PATH) -> None:
    config = load_config(config_path)
    parent = validate_parent_freeze(config)
    quality = build_candidate_quality(config, db_path=db_path)
    selected = select_registered_symbols(config, quality)
    manifest = build_manifest(config, parent, quality, selected)

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(QUALITY_PATH, index=False)
    selected.to_csv(SYMBOLS_PATH, index=False)
    manifest.to_csv(MANIFEST_PATH, index=False)
    REPORT_PATH.write_text(report_text(config, manifest, quality, selected), encoding="utf-8")

    status = manifest.set_index("item").loc["registration_status", "value"]
    print(f"Saved: {QUALITY_PATH}")
    print(f"Saved: {SYMBOLS_PATH}")
    print(f"Saved: {MANIFEST_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"Phase 43B registration status: {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register locked holdout membership before Phase 43B evaluation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--db-path", default=DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_artifacts(Path(args.config), db_path=args.db_path)


if __name__ == "__main__":
    main()
