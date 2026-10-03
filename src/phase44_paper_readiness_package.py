from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from config import BASE_DIR, SAVE_DIR


MODELS_DIR = Path(SAVE_DIR)
REPORTS_DIR = Path(BASE_DIR) / "reports"
PAPER_DIR = Path(BASE_DIR) / "paper"

LOCKED_RESULTS_PATH = MODELS_DIR / "phase43b_locked_external_experiment_results.csv"
LOCKED_CLAIMS_PATH = MODELS_DIR / "phase43b_locked_external_claims.csv"
LOCKED_PRIMARY_PATH = MODELS_DIR / "phase43b_locked_external_primary_comparison.csv"
DEVELOPMENT_RESULTS_PATH = MODELS_DIR / "crypto20_repaired_fold_local_experiment_results.csv"
DEVELOPMENT_STATS_PATH = MODELS_DIR / "crypto20_repaired_fold_local_statistical_method_summary.csv"
STRESS_SUMMARY_PATH = MODELS_DIR / "phase42_execution_stress_summary.csv"

EVIDENCE_MATRIX_PATH = MODELS_DIR / "phase44_paper_evidence_matrix.csv"
RISK_REGISTER_PATH = MODELS_DIR / "phase44_submission_risk_register.csv"
REPORT_PATH = REPORTS_DIR / "phase44_paper_readiness_package.md"
REVIEWER_BRIEF_PATH = REPORTS_DIR / "phase44_reviewer_brief.md"
CHECKLIST_PATH = REPORTS_DIR / "paper_submission_checklist.md"
ARTIFACT_MAP_PATH = REPORTS_DIR / "paper_artifact_map.csv"
PAPER_PATH = PAPER_DIR / "main.md"

FINAL_CANDIDATE = "regime_lgbm_hmm_guided_hmm"
PRIMARY_REFERENCES = ["global_lgbm", "regime_lgbm_hmm"]

METHOD_LABELS = {
    "global_lgbm": "Global LightGBM",
    "regime_lgbm_hmm": "Raw-feature HMM + regime LightGBM",
    "regime_lgbm_kmeans": "KMeans + regime LightGBM",
    "regime_lgbm_vol_bucket": "Volatility buckets + regime LightGBM",
    "regime_lgbm_contrastive": "Vanilla contrastive-GMM + regime LightGBM",
    "regime_lgbm_contrastive_hmm": "Vanilla contrastive-HMM + regime LightGBM",
    "regime_lgbm_hmm_guided_gmm": "HMM-guided contrastive-GMM + regime LightGBM",
    "regime_lgbm_hmm_guided_hmm": "HMM-guided contrastive-HMM + regime LightGBM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Phase 44 paper-readiness package from repaired development "
            "and locked external-holdout artifacts."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate artifacts without writing generated paper files.",
    )
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Phase 44 input is missing: {path}")
    return pd.read_csv(path)


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def pct(value: object, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def method_result(results: pd.DataFrame, method: str) -> pd.Series:
    rows = results[results["method"] == method]
    if rows.empty:
        raise ValueError(f"Missing method in results: {method}")
    return rows.iloc[0]


def claim_status(claims: pd.DataFrame, claim_id: str) -> str:
    rows = claims[claims["claim_id"] == claim_id]
    if rows.empty:
        raise ValueError(f"Missing locked claim: {claim_id}")
    return str(rows.iloc[0]["claim_status"])


def validate_inputs(locked_results: pd.DataFrame, locked_claims: pd.DataFrame, locked_primary: pd.DataFrame) -> None:
    required_methods = {FINAL_CANDIDATE, *PRIMARY_REFERENCES}
    missing_methods = required_methods - set(locked_results["method"].astype(str))
    if missing_methods:
        raise ValueError(f"Locked results are missing primary methods: {sorted(missing_methods)}")
    if claim_status(locked_claims, "locked_relative_success_rule") != "satisfied":
        raise ValueError("Phase 44 expects the frozen relative locked rule to be satisfied.")
    if claim_status(locked_claims, "positive_tradable_alpha") != "not_supported":
        raise ValueError("Phase 44 must not proceed if locked tradable-alpha status is unsupported.")
    reference_col = "reference_method" if "reference_method" in locked_primary.columns else "reference"
    if reference_col not in locked_primary.columns:
        raise ValueError("Locked primary comparison is missing a reference column.")
    references = set(locked_primary[reference_col].astype(str))
    if references != set(PRIMARY_REFERENCES):
        raise ValueError(f"Locked primary comparison references drifted: {sorted(references)}")
    if "final_candidate" in locked_primary.columns:
        candidates = set(locked_primary["final_candidate"].astype(str))
        if candidates != {FINAL_CANDIDATE}:
            raise ValueError(f"Locked final candidate drifted: {sorted(candidates)}")
    required_rule_columns = {"ic_improved", "sharpe_non_worse", "coverage_equal"}
    if not required_rule_columns.issubset(locked_primary.columns):
        missing = sorted(required_rule_columns - set(locked_primary.columns))
        raise ValueError(f"Locked primary comparison is missing rule columns: {missing}")
    failed_rules = {
        column: locked_primary[~locked_primary[column].astype(bool)][reference_col].astype(str).tolist()
        for column in sorted(required_rule_columns)
    }
    failed_rules = {column: refs for column, refs in failed_rules.items() if refs}
    if failed_rules:
        raise ValueError(f"Locked relative rule is not satisfied for all references: {failed_rules}")


def build_locked_result_table(locked_results: pd.DataFrame) -> pd.DataFrame:
    order = [
        FINAL_CANDIDATE,
        "regime_lgbm_hmm_guided_gmm",
        "regime_lgbm_hmm",
        "global_lgbm",
        "regime_lgbm_contrastive",
        "regime_lgbm_contrastive_hmm",
        "regime_lgbm_kmeans",
        "regime_lgbm_vol_bucket",
    ]
    rows = locked_results.copy()
    rows["_order"] = rows["method"].map({method: idx for idx, method in enumerate(order)}).fillna(99)
    rows = rows.sort_values(["_order", "method"]).drop(columns="_order")
    return pd.DataFrame(
        {
            "Method": rows["method"].map(METHOD_LABELS).fillna(rows["method"]),
            "Mean Asset IC": rows["mean_asset_IC"].map(fmt),
            "Sharpe": rows["Sharpe"].map(fmt),
            "Total Return": rows["total_return"].map(lambda value: pct(value, 1)),
            "Drawdown": rows["drawdown"].map(lambda value: pct(value, 1)),
            "Rows": rows["n_test_rows"].astype(int).astype(str),
        }
    )


def build_evidence_matrix(
    locked_results: pd.DataFrame,
    locked_claims: pd.DataFrame,
    development_results: pd.DataFrame,
    development_stats: pd.DataFrame,
    stress_summary: pd.DataFrame,
) -> pd.DataFrame:
    locked_final = method_result(locked_results, FINAL_CANDIDATE)
    locked_global = method_result(locked_results, "global_lgbm")
    locked_hmm = method_result(locked_results, "regime_lgbm_hmm")
    development_final = method_result(development_results, FINAL_CANDIDATE)
    development_best = development_results.sort_values("mean_asset_IC", ascending=False).iloc[0]
    final_stats = method_result(development_stats.rename(columns={"full_sample_IC": "IC"}), FINAL_CANDIDATE)
    stress_rows = stress_summary[stress_summary["method"] == FINAL_CANDIDATE]
    positive_stress_cells = int(stress_rows["positive_return_cells"].sum()) if not stress_rows.empty else 0
    total_stress_cells = int(stress_rows["stress_cells"].sum()) if not stress_rows.empty else 0
    relative_status = claim_status(locked_claims, "locked_relative_success_rule")
    tradable_status = claim_status(locked_claims, "positive_tradable_alpha")
    return pd.DataFrame(
        [
            {
                "evidence_block": "validation_repair",
                "data_role": "development_observed",
                "main_artifacts": "reports/phase39r_neural_fold_local_results.md; reports/publication_acceptance_gates.md",
                "finding": "Original positional-fold evidence is retained only as audit history; repaired common-calendar fold-local evidence is the valid development benchmark.",
                "paper_use": "Use as research-integrity story and validation contribution.",
                "claim_boundary": "Do not cite invalidated positional-fold runs as predictive evidence.",
            },
            {
                "evidence_block": "repaired_crypto20_development",
                "data_role": "development_observed",
                "main_artifacts": "models/crypto20_repaired_fold_local_experiment_results.csv; models/crypto20_repaired_fold_local_statistical_method_summary.csv",
                "finding": (
                    f"Final candidate development mean asset IC={fmt(development_final['mean_asset_IC'])}, "
                    f"Sharpe={fmt(development_final['Sharpe'])}; best development mean asset IC method is "
                    f"{development_best['method']} at {fmt(development_best['mean_asset_IC'])}."
                ),
                "paper_use": "Show that repaired development evidence is weak/negative and prevents a broad positive-alpha claim.",
                "claim_boundary": "Development results motivate interpretation, not final-test confirmation.",
            },
            {
                "evidence_block": "development_statistical_adjudication",
                "data_role": "development_observed",
                "main_artifacts": "models/crypto20_repaired_fold_local_statistical_method_summary.csv; reports/phase40_repaired_statistical_adjudication.md",
                "finding": (
                    f"Final candidate has {int(final_stats['n_folds'])} folds, "
                    f"IC bootstrap CI [{fmt(final_stats['IC_ci_low'])}, {fmt(final_stats['IC_ci_high'])}], "
                    f"Sharpe CI [{fmt(final_stats['Sharpe_ci_low'])}, {fmt(final_stats['Sharpe_ci_high'])}]."
                ),
                "paper_use": "Frame statistical power and uncertainty explicitly.",
                "claim_boundary": "No corrected dominance or robust positive-alpha claim.",
            },
            {
                "evidence_block": "execution_and_mechanism_diagnostics",
                "data_role": "development_observed",
                "main_artifacts": "reports/phase42_interpretation_execution_hardening.md; models/phase42_execution_stress_summary.csv",
                "finding": f"Final candidate has {positive_stress_cells}/{total_stress_cells} positive-return stress cells across Phase 42 diagnostics.",
                "paper_use": "Explain why the weak alpha result is not rescued by simple execution/calibration tweaks.",
                "claim_boundary": "Diagnostics explain fragility; they are not a new tuned model.",
            },
            {
                "evidence_block": "locked_external_holdout",
                "data_role": "locked_registered_unobserved",
                "main_artifacts": "models/phase43b_locked_external_experiment_results.csv; reports/phase43b_locked_external_adjudication.md",
                "finding": (
                    f"Frozen final candidate mean asset IC={fmt(locked_final['mean_asset_IC'])}, "
                    f"Sharpe={fmt(locked_final['Sharpe'])}, return={pct(locked_final['total_return'])}; "
                    f"global reference IC={fmt(locked_global['mean_asset_IC'])}, Sharpe={fmt(locked_global['Sharpe'])}; "
                    f"raw-HMM reference IC={fmt(locked_hmm['mean_asset_IC'])}, Sharpe={fmt(locked_hmm['Sharpe'])}."
                ),
                "paper_use": f"Allowed locked claim: relative rule is {relative_status}.",
                "claim_boundary": f"Tradable-alpha claim is {tradable_status}; no candidate switching or same-holdout retuning.",
            },
        ]
    )


def build_risk_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "risk": "Overclaiming profitability",
                "severity": "critical",
                "mitigation": "Phase 43B explicitly marks positive tradable alpha as not_supported.",
                "owner_artifact": "models/phase43b_locked_external_claims.csv",
            },
            {
                "risk": "Switching from frozen final candidate to higher-IC diagnostic method",
                "severity": "critical",
                "mitigation": "Candidate switching after holdout is forbidden; guided-GMM cannot replace guided-HMM after seeing locked outcomes.",
                "owner_artifact": "reports/phase43b_locked_external_adjudication.md",
            },
            {
                "risk": "Reintroducing calendar leakage",
                "severity": "critical",
                "mitigation": "Common-calendar fold-local validation and research-grade checks must pass before push.",
                "owner_artifact": "models/research_grade_check_report.md",
            },
            {
                "risk": "Treating development evidence as final test evidence",
                "severity": "high",
                "mitigation": "Data-role language separates development_observed from locked_registered_unobserved.",
                "owner_artifact": "reports/data_role_registry.csv; reports/publication_acceptance_gates.md",
            },
            {
                "risk": "Weak/negative result framed as failure instead of contribution",
                "severity": "medium",
                "mitigation": "Paper thesis focuses on validation, mechanism boundaries, and limited locked relative support.",
                "owner_artifact": "reports/phase44_paper_readiness_package.md",
            },
            {
                "risk": "Venue formatting and citations incomplete",
                "severity": "medium",
                "mitigation": "Next phase should convert Markdown to target venue template and finalize related work.",
                "owner_artifact": "paper/main.md; reports/paper_submission_checklist.md",
            },
        ]
    )


def build_artifact_map() -> pd.DataFrame:
    rows = [
        ("Abstract and thesis", "claim control", "reports/phase44_paper_readiness_package.md", "Defines the paper-safe story after locked holdout."),
        ("Reviewer brief", "reviewer response", "reports/phase44_reviewer_brief.md", "Pre-writes safe answers to expected reviewer objections."),
        ("Validation repair", "audit", "reports/phase39r_neural_fold_local_results.md; models/research_grade_check_report.md", "Documents the repaired common-calendar validation protocol."),
        ("Development benchmark", "result table", "models/crypto20_repaired_fold_local_experiment_results.csv", "Shows repaired development-observed alpha behavior."),
        ("Development statistics", "statistical test", "models/crypto20_repaired_fold_local_statistical_method_summary.csv; reports/phase40_repaired_statistical_adjudication.md", "Quantifies uncertainty and prevents corrected dominance claims."),
        ("Mechanism diagnostics", "diagnostic", "reports/phase42_interpretation_execution_hardening.md; models/phase42_execution_stress_summary.csv", "Explains alpha fragility without tuning a new model."),
        ("Locked external holdout", "confirmatory result", "models/phase43b_locked_external_experiment_results.csv; reports/phase43b_locked_external_adjudication.md", "One-shot locked external evaluation of the frozen final candidate."),
        ("Reproducibility", "artifact gate", "reports/artifact_manifest.md; reports/reproduction_checklist.md; models/research_grade_check_report.md", "Documents what is committed, ignored, and checked."),
        ("Submission risk", "risk register", "models/phase44_submission_risk_register.csv", "Lists reviewer-facing failure modes and mitigations."),
        ("Pre-push hardening", "audit", "reports/phase44_prepush_hardening_audit.md", "Records stale-claim cleanup, reviewer-risk checks, and final pre-push gates."),
    ]
    return pd.DataFrame(rows, columns=["paper_section", "artifact_type", "artifact", "paper_role"])


def build_report(evidence: pd.DataFrame, risk: pd.DataFrame, locked_results: pd.DataFrame) -> str:
    locked_table = build_locked_result_table(locked_results)
    return f"""# Phase 44 Paper-Readiness Package

## Status

Phase 44 converts the completed repaired-development and locked-external-holdout evidence into a paper-facing package. It does **not** tune models, change labels, change candidate choice, or touch final/locked evaluation data.

## Paper Thesis

The strongest paper path is a research-grade negative/limited-support paper:

> Sequential HMM discipline can make contrastive regime representations more useful, but the repaired and locked evidence does not support a profitable trading claim. The contribution is the validation repair, the benchmark, the mechanism boundary, and the one-shot locked-holdout adjudication.

This is stronger and safer than trying to rescue a positive result after the locked holdout.

## Locked-Holdout Result Snapshot

{markdown_table(locked_table, list(locked_table.columns))}

## Evidence Matrix

{markdown_table(evidence, list(evidence.columns))}

## Submission Risk Register

{markdown_table(risk, list(risk.columns))}

## Allowed Claim

The paper may say:

> On a registered external crypto holdout, the frozen guided-HMM candidate satisfies the prewritten relative IC/Sharpe rule against the global LightGBM and raw-feature HMM references, but it does not establish positive tradable alpha.

## Forbidden Claims

- Do not claim a profitable or deployable trading strategy.
- Do not switch the final candidate after seeing the locked holdout.
- Do not tune thresholds, labels, features, architecture, or method choice on the same locked holdout.
- Do not cite invalidated positional-fold runs as predictive evidence.
- Do not describe development-observed results as untouched final-test results.

## Recommended Next Phase

Phase 45 should be venue-formatting and reviewer package work: convert `paper/main.md` to the target template, tighten related work, create final tables/figures, and prepare a reproducibility appendix. It should not add new model search unless a new pre-registered dataset or external replication is created first.
"""


def build_reviewer_brief(locked_results: pd.DataFrame) -> str:
    locked_final = method_result(locked_results, FINAL_CANDIDATE)
    locked_global = method_result(locked_results, "global_lgbm")
    locked_hmm = method_result(locked_results, "regime_lgbm_hmm")
    return f"""# Phase 44 Reviewer Brief

## One-Sentence Paper Position

This is a validation-and-mechanism paper: HMM-guided contrastive regimes receive limited locked relative support, but the evidence does not support a profitable or deployable trading strategy.

## What Is Confirmed?

- The frozen final candidate is `{FINAL_CANDIDATE}`.
- The locked external holdout was registered before model outcome inspection.
- The frozen candidate satisfies the prewritten relative rule against `global_lgbm` and `regime_lgbm_hmm`.
- Locked mean asset IC is {fmt(locked_final['mean_asset_IC'])}, versus {fmt(locked_global['mean_asset_IC'])} for global LightGBM and {fmt(locked_hmm['mean_asset_IC'])} for raw-feature HMM.
- Locked Sharpe is {fmt(locked_final['Sharpe'])}, versus {fmt(locked_global['Sharpe'])} for global LightGBM and {fmt(locked_hmm['Sharpe'])} for raw-feature HMM.

## What Is Not Confirmed?

- The project does not confirm positive tradable alpha.
- The project does not confirm broad method dominance.
- The project does not prove HMM states are true market regimes.
- The project does not authorize switching to another method after the locked result.

## Why This Is Still Publishable

The paper contributes a complete research audit trail: an initially exciting result was invalidated, repaired, re-evaluated, frozen, and tested once on a registered external holdout. That sequence is valuable because it shows how fragile quant-ML evidence can be, and it gives a reproducible template for separating real signal from validation artifacts.

## Likely Reviewer Questions

| Reviewer Question | Paper-Safe Answer |
|---|---|
| Is this a trading strategy? | No. The locked candidate has negative Sharpe and negative total return. |
| Did the final method change after seeing the holdout? | No. Candidate switching after locked evaluation is forbidden. |
| Why not use the higher locked-IC guided-GMM result? | It was not the frozen final candidate and has worse Sharpe/return; replacing the candidate after seeing outcomes would be post-hoc selection. |
| Are HMM labels ground truth? | No. They are weak proxy supervision and a classical sequential reference. |
| Why publish a weak/negative result? | The validation repair, common-calendar benchmark, locked-holdout discipline, and claim-control framework are the main scientific contribution. |
| What should future work do? | Use a new pre-registered external replication dataset; do not reuse the same locked holdout for model rescue. |
"""


def build_checklist() -> str:
    return """# Paper Submission Checklist

## Phase 44 Status

The project has completed the repaired development benchmark and the Phase 43B locked external holdout. The next work is paper packaging, not model rescue.

## Ready

- Repaired common-calendar development benchmark is complete.
- Phase 40 repaired statistical adjudication is complete.
- Phase 42 interpretation/execution diagnostics are complete.
- Phase 43A frozen final candidate and holdout rules are complete.
- Phase 43B locked external holdout is complete.
- Full research-grade checks pass.
- Positive tradable-alpha claim is explicitly blocked.
- Candidate switching after locked evaluation is explicitly blocked.

## Critical Paper Work

- Convert `paper/main.md` into the final venue template.
- Replace placeholder prose with final citations and related-work positioning.
- Add final figure and table numbers.
- Decide whether the target is ICAIF main track, workshop, student research track, or institutional BTech evaluation.
- Write a reproducibility appendix using the artifact manifest and research-grade gate report.

## Must Not Claim

- Do not claim HMM states are ground truth.
- Do not claim a profitable or deployable trading strategy.
- Do not claim guided-HMM statistically dominates raw-feature HMM at 5%.
- Do not claim statistically proven Crypto-20 alpha, calibration, or portfolio-performance dominance.
- Do not treat development-observed results as untouched final-test evidence.
- Do not switch from the frozen guided-HMM final candidate to a secondary diagnostic method after locked-holdout inspection.
- Do not retune thresholds, labels, features, architecture, or method choice on the same locked holdout.
"""


def build_paper(locked_results: pd.DataFrame, evidence: pd.DataFrame) -> str:
    locked_final = method_result(locked_results, FINAL_CANDIDATE)
    locked_global = method_result(locked_results, "global_lgbm")
    locked_hmm = method_result(locked_results, "regime_lgbm_hmm")
    locked_gmm = method_result(locked_results, "regime_lgbm_hmm_guided_gmm")
    locked_table = build_locked_result_table(locked_results)
    return f"""# HMM-Guided Contrastive Representations for Regime-Conditioned Financial Alpha Modeling

## Paper Status

Phase 44 paper-readiness draft. The project has completed the repaired common-calendar development benchmark and one registered locked external holdout. This draft is paper-facing prose, not a new experiment.

## Abstract

Regime-conditioned financial alpha models are attractive because predictive relationships can change across trend, stress, and choppy market states. Yet regime labels are latent, unstable, and easy to overfit. This project studies whether contrastive time-series representations become more useful when guided by classical Hidden Markov Model state structure. The benchmark compares global LightGBM, raw-feature HMM regimes, KMeans regimes, volatility buckets, vanilla contrastive regimes, and HMM-guided contrastive regimes under triple-barrier labels, purged walk-forward validation, transaction costs, statistical adjudication, and reproducibility gates. A key validation repair invalidated earlier positional-fold evidence and replaced it with strict common-calendar fold-local evaluation. Under the repaired Crypto-20 development benchmark, alpha evidence is weak and does not support a broad positive-performance claim. Under the registered Phase 43B external holdout, the frozen guided-HMM candidate satisfies the prewritten relative IC/Sharpe rule against global LightGBM and raw-feature HMM references: mean asset IC is {fmt(locked_final['mean_asset_IC'])} versus {fmt(locked_global['mean_asset_IC'])} and {fmt(locked_hmm['mean_asset_IC'])}, and Sharpe is {fmt(locked_final['Sharpe'])} versus {fmt(locked_global['Sharpe'])} and {fmt(locked_hmm['Sharpe'])}. However, the final candidate still has negative locked total return ({pct(locked_final['total_return'])}) and negative Sharpe, so the paper does not claim a tradable strategy. The contribution is a research-grade benchmark and mechanism boundary: sequential regime discipline can help learned representations, but structural improvement does not automatically become profitable alpha.

## 1. Introduction

Financial markets are non-stationary. A signal that works during calm trend-following periods can fail during stress, high volatility, or transition periods. Regime-conditioned alpha modeling tries to address this by allowing models to behave differently across latent market states. The hard part is that those states are not directly observed.

Classical HMMs offer sequential discipline and persistent state assignments, but they depend on raw features and distributional assumptions. Contrastive encoders can learn richer nonlinear representations, but memoryless clustering of embeddings can produce regimes that look smooth while adding little downstream alpha value. This project studies a hybrid: use HMM state sequences as weak supervision for a contrastive encoder, then test whether sequential assignment on the learned representation improves regime-conditioned alpha modeling.

The important research lesson is not simply whether one method wins a backtest. Earlier versions of this project produced exciting results, but a later audit found that positional multi-asset folds overlapped in calendar time. Those results are now retained only as audit history. The repaired pipeline uses common-calendar fold-local validation, data-role separation, frozen candidate rules, and a locked external holdout.

## 2. Research Question

Does HMM-guided contrastive regime learning produce more useful regime-conditioned alpha models than global LightGBM, raw-feature HMM regimes, vanilla contrastive regimes, and simple clustering/volatility baselines under leakage-safe financial validation?

The paper-safe answer is conditional:

- Yes, the locked holdout gives limited support to the frozen guided-HMM candidate under the prewritten relative rule.
- No, the evidence does not support a profitable or deployable trading strategy.
- No, the project may not switch to a secondary diagnostic method after seeing the locked holdout, even though guided-GMM has higher locked IC ({fmt(locked_gmm['mean_asset_IC'])}) but worse Sharpe ({fmt(locked_gmm['Sharpe'])}) and return ({pct(locked_gmm['total_return'])}).

## 3. Contributions

1. A repaired common-calendar regime-conditioned alpha benchmark for crypto assets.
2. A documented validation failure and repair, showing how a plausible multi-asset backtest can become invalid through calendar overlap.
3. An HMM-guided contrastive regime-learning path that treats HMM states as weak proxy supervision, not ground truth.
4. A one-shot locked external holdout with prewritten claim rules.
5. A claim-control layer that separates relative method evidence from tradable-alpha claims.

## 4. Data and Validation

The development benchmark uses a repaired Crypto-20 common-calendar panel. The locked external holdout uses 10 registered external crypto symbols selected before model outcome inspection. Phase 43B evaluates 18 folds and 129,600 out-of-sample rows per method. The validation protocol uses fold-local fitting, purged walk-forward splits, embargo spacing, transaction costs, and equal method coverage.

The paper treats development-observed evidence and locked-holdout evidence differently. Development evidence can motivate the mechanism and explain failure modes. Locked evidence is the only confirmatory claim source, and it has already been spent once.

## 5. Methods

The benchmark includes:

- Global LightGBM with no regimes.
- Raw-feature HMM regimes.
- KMeans regimes.
- Volatility buckets.
- Vanilla contrastive embeddings with GMM or HMM assignment.
- HMM-guided contrastive embeddings with GMM or HMM assignment.

The frozen final candidate is `regime_lgbm_hmm_guided_hmm`.

## 6. Main Locked Result

{markdown_table(locked_table, list(locked_table.columns))}

The locked result satisfies the prewritten relative IC/Sharpe rule against the two primary references. This means the frozen candidate is better than the chosen references on the specific registered comparison. It does not mean the strategy is profitable.

## 7. Evidence Interpretation

{markdown_table(evidence[["evidence_block", "data_role", "finding", "claim_boundary"]], ["evidence_block", "data_role", "finding", "claim_boundary"])}

The most honest interpretation is a limited-support mechanism paper:

- Sequential assignment matters.
- HMM-guided representation learning can improve the relative ranking behavior of regimes.
- Development and locked evidence still show fragile economic performance.
- A negative/limited result is scientifically valuable because it prevents an invalid positive trading claim.

## 8. Limitations

- HMM states are proxy states, not ground truth.
- Crypto markets are not equities, FX, or options markets.
- The locked holdout supports only the prewritten relative rule, not a broad dominance claim.
- The final candidate has negative locked Sharpe and negative locked total return.
- The same locked holdout cannot be reused for model rescue.
- Row-level financial observations are overlapping and should not be treated as independent statistical evidence.

## 9. Conclusion

The project now has a defensible paper story: HMM-guided contrastive regimes receive limited locked-holdout support as a relative modeling mechanism, while the larger trading-alpha claim is not supported. This is not a failed project. It is a stronger research contribution than an overfit backtest because it shows the full chain: build, audit, invalidate, repair, freeze, evaluate once, and report honestly.
"""


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    locked_results = read_required_csv(LOCKED_RESULTS_PATH)
    locked_claims = read_required_csv(LOCKED_CLAIMS_PATH)
    locked_primary = read_required_csv(LOCKED_PRIMARY_PATH)
    development_results = read_required_csv(DEVELOPMENT_RESULTS_PATH)
    development_stats = read_required_csv(DEVELOPMENT_STATS_PATH)
    stress_summary = read_required_csv(STRESS_SUMMARY_PATH)

    validate_inputs(locked_results, locked_claims, locked_primary)
    evidence = build_evidence_matrix(
        locked_results,
        locked_claims,
        development_results,
        development_stats,
        stress_summary,
    )
    risk = build_risk_register()
    artifact_map = build_artifact_map()

    if args.dry_run:
        print("OK: Phase 44 inputs validated.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(EVIDENCE_MATRIX_PATH, evidence)
    write_csv(RISK_REGISTER_PATH, risk)
    write_csv(ARTIFACT_MAP_PATH, artifact_map)
    REPORT_PATH.write_text(build_report(evidence, risk, locked_results), encoding="utf-8")
    REVIEWER_BRIEF_PATH.write_text(build_reviewer_brief(locked_results), encoding="utf-8")
    CHECKLIST_PATH.write_text(build_checklist(), encoding="utf-8")
    PAPER_PATH.write_text(build_paper(locked_results, evidence), encoding="utf-8")
    print(f"Saved: {EVIDENCE_MATRIX_PATH}")
    print(f"Saved: {RISK_REGISTER_PATH}")
    print(f"Saved: {ARTIFACT_MAP_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"Saved: {REVIEWER_BRIEF_PATH}")
    print(f"Saved: {CHECKLIST_PATH}")
    print(f"Saved: {PAPER_PATH}")


if __name__ == "__main__":
    main()
