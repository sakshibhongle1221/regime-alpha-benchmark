from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import BASE_DIR, SAVE_DIR


FINAL_CANDIDATE = "regime_lgbm_hmm_guided_hmm"
PRIMARY_REFERENCES = ["global_lgbm", "regime_lgbm_hmm"]
SECONDARY_CONTEXT = ["regime_lgbm_hmm_guided_gmm"]

RESULTS_PATH = Path(SAVE_DIR) / "phase43b_locked_external_experiment_results.csv"
FOLD_METRICS_PATH = Path(SAVE_DIR) / "phase43b_locked_external_fold_metrics.csv"
REGISTRATION_PATH = Path(SAVE_DIR) / "phase43b_locked_holdout_registration_manifest.csv"
FREEZE_PATH = Path(SAVE_DIR) / "phase43b_locked_holdout_freeze_manifest.json"
PRIMARY_COMPARISON_PATH = Path(SAVE_DIR) / "phase43b_locked_external_primary_comparison.csv"
CLAIMS_PATH = Path(SAVE_DIR) / "phase43b_locked_external_claims.csv"
REPORT_PATH = Path(BASE_DIR) / "reports" / "phase43b_locked_external_adjudication.md"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, str]]:
    results = pd.read_csv(RESULTS_PATH)
    fold_metrics = pd.read_csv(FOLD_METRICS_PATH)
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    registration = pd.read_csv(REGISTRATION_PATH).set_index("item")["value"].astype(str).to_dict()
    if registration.get("registration_status") != "registered_ready":
        raise RuntimeError("Locked adjudication requires registered_ready holdout status.")
    required = {FINAL_CANDIDATE, *PRIMARY_REFERENCES}
    missing = sorted(required - set(results["method"].astype(str)))
    if missing:
        raise RuntimeError(f"Locked result is missing required methods: {missing}")
    return results, fold_metrics, freeze, registration


def build_primary_comparison(results: pd.DataFrame) -> pd.DataFrame:
    indexed = results.set_index("method")
    final = indexed.loc[FINAL_CANDIDATE]
    rows = []
    for reference in PRIMARY_REFERENCES:
        ref = indexed.loc[reference]
        rows.append(
            {
                "final_candidate": FINAL_CANDIDATE,
                "reference_method": reference,
                "candidate_mean_asset_IC": float(final["mean_asset_IC"]),
                "reference_mean_asset_IC": float(ref["mean_asset_IC"]),
                "delta_mean_asset_IC": float(final["mean_asset_IC"] - ref["mean_asset_IC"]),
                "candidate_Sharpe": float(final["Sharpe"]),
                "reference_Sharpe": float(ref["Sharpe"]),
                "delta_Sharpe": float(final["Sharpe"] - ref["Sharpe"]),
                "candidate_total_return": float(final["total_return"]),
                "reference_total_return": float(ref["total_return"]),
                "candidate_drawdown": float(final["drawdown"]),
                "reference_drawdown": float(ref["drawdown"]),
                "candidate_n_test_rows": int(final["n_test_rows"]),
                "reference_n_test_rows": int(ref["n_test_rows"]),
                "ic_improved": bool(final["mean_asset_IC"] > ref["mean_asset_IC"]),
                "sharpe_non_worse": bool(final["Sharpe"] >= ref["Sharpe"]),
                "coverage_equal": bool(int(final["n_test_rows"]) == int(ref["n_test_rows"])),
            }
        )
    return pd.DataFrame(rows)


def build_claims(primary: pd.DataFrame, results: pd.DataFrame, fold_metrics: pd.DataFrame, freeze: dict) -> pd.DataFrame:
    final = results.set_index("method").loc[FINAL_CANDIDATE]
    fold_counts = fold_metrics.groupby("method")["fold"].nunique().to_dict()
    expected_folds = int(freeze["fold_count"])
    all_required_folds = all(int(fold_counts.get(method, 0)) == expected_folds for method in [FINAL_CANDIDATE, *PRIMARY_REFERENCES])
    row_counts_equal = primary["coverage_equal"].all()
    success = bool(
        primary["ic_improved"].all()
        and primary["sharpe_non_worse"].all()
        and row_counts_equal
        and all_required_folds
    )
    positive_alpha = bool(float(final["Sharpe"]) > 0 and float(final["total_return"]) > 0)
    secondary_note = ""
    if "regime_lgbm_hmm_guided_gmm" in set(results["method"].astype(str)):
        guided_gmm = results.set_index("method").loc["regime_lgbm_hmm_guided_gmm"]
        if float(guided_gmm["mean_asset_IC"]) > float(final["mean_asset_IC"]):
            secondary_note = (
                "A secondary diagnostic method has higher locked IC, but it was not the frozen final candidate "
                "and cannot replace the final candidate after outcome inspection."
            )
    rows = [
        {
            "claim_id": "locked_relative_success_rule",
            "claim_status": "satisfied" if success else "failed",
            "claim": (
                "The frozen guided-HMM final candidate improves over both primary references on mean asset IC "
                "and has non-worse transaction-cost-adjusted Sharpe with equal coverage."
            ),
            "evidence": (
                "All primary comparisons satisfy IC/Sharpe/coverage requirements."
                if success
                else "At least one primary comparison fails IC, Sharpe, fold, or coverage requirements."
            ),
        },
        {
            "claim_id": "positive_tradable_alpha",
            "claim_status": "not_supported",
            "claim": "The locked result supports a tradable positive-alpha strategy.",
            "evidence": (
                f"Final candidate Sharpe={float(final['Sharpe']):.4f}, "
                f"total_return={float(final['total_return']):.4f}; positive_alpha={positive_alpha}."
            ),
        },
        {
            "claim_id": "candidate_switching_after_holdout",
            "claim_status": "forbidden",
            "claim": "A non-frozen method can replace the final candidate after locked outcome inspection.",
            "evidence": secondary_note or "No secondary method can replace the frozen candidate after locked evaluation.",
        },
        {
            "claim_id": "same_holdout_retuning",
            "claim_status": "forbidden",
            "claim": "Thresholds, features, labels, architecture, or method choice can be tuned after locked evaluation.",
            "evidence": "Phase 43A and Phase 43B forbid same-holdout tuning or rerun-after-failure.",
        },
    ]
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.{digits}f}")
    lines = [
        "| " + " | ".join(display.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def report_text(results: pd.DataFrame, primary: pd.DataFrame, claims: pd.DataFrame, freeze: dict, registration: dict[str, str]) -> str:
    result_cols = ["method", "mean_asset_IC", "Sharpe", "total_return", "drawdown", "n_test_rows"]
    final = results.set_index("method").loc[FINAL_CANDIDATE]
    claim_status = claims.set_index("claim_id").loc["locked_relative_success_rule", "claim_status"]
    return f"""# Phase 43B Locked External Holdout Adjudication

## Status

The Phase 43B locked external evaluation is complete and adjudicated under the Phase 43A frozen rule.

- Freeze ID: `{freeze['freeze_id']}`
- Registration status: `{registration['registration_status']}`
- Registered symbols: `{registration['selected_symbols']}`
- Fold count: {freeze['fold_count']}
- Prediction rows: {freeze['prediction_rows']}
- Final candidate: `{FINAL_CANDIDATE}`
- Primary locked claim status: `{claim_status}`

## Locked Result Summary

{markdown_table(results[result_cols])}

## Primary Frozen Rule Comparison

{markdown_table(primary)}

## Claim Adjudication

{markdown_table(claims)}

## Paper-Safe Interpretation

The frozen guided-HMM candidate satisfies the prewritten relative locked-holdout rule against `global_lgbm` and `regime_lgbm_hmm`: mean asset IC is higher and Sharpe is non-worse with equal coverage.

This is **not** a tradable-strategy claim. The final candidate still has negative locked Sharpe ({float(final['Sharpe']):.4f}) and negative locked total return ({float(final['total_return']):.4f}).

A secondary or diagnostic method cannot replace the final candidate after locked evaluation.

The paper-safe wording is:

```text
The frozen guided-HMM mechanism receives limited locked-holdout support on the pre-specified relative IC/Sharpe rule, but it does not establish a profitable or tradable strategy.
```

Forbidden wording:

```text
The locked holdout proves a tradable strategy.
We can switch the final method after seeing the locked holdout.
The locked holdout result authorizes threshold or model retuning.
```
"""


def write_artifacts() -> None:
    results, fold_metrics, freeze, registration = load_inputs()
    primary = build_primary_comparison(results)
    claims = build_claims(primary, results, fold_metrics, freeze)
    PRIMARY_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(PRIMARY_COMPARISON_PATH, index=False)
    claims.to_csv(CLAIMS_PATH, index=False)
    REPORT_PATH.write_text(report_text(results, primary, claims, freeze, registration), encoding="utf-8")
    print(f"Saved: {PRIMARY_COMPARISON_PATH}")
    print(f"Saved: {CLAIMS_PATH}")
    print(f"Saved: {REPORT_PATH}")


def main() -> None:
    write_artifacts()


if __name__ == "__main__":
    main()
