import argparse
import csv
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
PAPER_DIR = BASE_DIR / "paper"

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
        description="Initialize missing paper artifacts without overwriting the human-reviewed Phase 38 manuscript."
    )
    parser.add_argument("--output-dir", default=str(PAPER_DIR), help="Directory for paper/main.md.")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def pct(value: object, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_Artifact not available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def build_main_result_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    preferred = [
        "regime_lgbm_hmm_guided_hmm",
        "regime_lgbm_hmm",
        "regime_lgbm_contrastive_hmm",
        "regime_lgbm_hmm_guided_gmm",
        "regime_lgbm_contrastive",
        "global_lgbm",
    ]
    ordered = results.copy()
    ordered["_order"] = ordered["method"].map({name: idx for idx, name in enumerate(preferred)}).fillna(99)
    ordered = ordered.sort_values(["_order", "method"]).drop(columns="_order")
    return pd.DataFrame(
        {
            "Method": ordered["method"].map(METHOD_LABELS).fillna(ordered["method"]),
            "IC": ordered["IC"].map(fmt),
            "Sharpe": ordered["Sharpe"].map(fmt),
            "Drawdown": ordered["drawdown"].map(lambda value: pct(value, 1)),
            "Total Return": ordered["total_return"].map(lambda value: pct(value, 1)),
            "Turnover": ordered["turnover"].map(fmt),
            "Rows": ordered["n_test_rows"].astype(str),
        }
    )


def build_claim_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    focus = summary[
        summary["comparison"].isin(
            [
                "guided_hmm_alpha_vs_raw_feature_hmm_alpha",
                "guided_hmm_alpha_vs_guided_gmm_alpha",
                "guided_hmm_alpha_vs_vanilla_contrastive_hmm_alpha",
                "time_frequency_guided_hmm_vs_time_only_guided_hmm",
            ]
        )
    ].copy()
    if focus.empty:
        focus = summary.copy()
    return pd.DataFrame(
        {
            "Comparison": focus["comparison"],
            "Status": focus["paper_status"],
            "IC Diff": focus["ic_mean_difference"].map(fmt),
            "IC p": focus["ic_p_value"].map(fmt),
            "Win Rate": focus["metric_win_rate"].map(lambda value: pct(value, 0)),
            "Allowed Language": focus["allowed_paper_language"],
        }
    )


def build_artifact_map() -> list[dict[str, str]]:
    return [
        {
            "paper_section": "Abstract and contributions",
            "artifact_type": "claim control",
            "artifact": "reports/claim_registry.md",
            "paper_role": "Prevents unsupported novelty, profitability, and downstream multi-asset alpha claims.",
        },
        {
            "paper_section": "Related work",
            "artifact_type": "literature",
            "artifact": "reports/related_work.md; reports/literature_matrix.csv",
            "paper_role": "Positions the project against regime models, time-series contrastive learning, validation, and alpha modeling.",
        },
        {
            "paper_section": "Data and labels",
            "artifact_type": "data diagnostic",
            "artifact": "models/target_distribution.csv; models/target_quality.csv",
            "paper_role": "Documents class balance, neutral share, and horizon-tail loss for financial labels.",
        },
        {
            "paper_section": "Validation",
            "artifact_type": "audit",
            "artifact": "models/validation_audit.csv; models/fold_audit.csv",
            "paper_role": "Documents embargo, label-horizon purge, common coverage, and artifact availability.",
        },
        {
            "paper_section": "Methods",
            "artifact_type": "model card",
            "artifact": "reports/model_card.md; reports/paper_protocol.md",
            "paper_role": "Records architecture, frozen protocol, methods, metrics, and forbidden claims.",
        },
        {
            "paper_section": "Main results",
            "artifact_type": "result table",
            "artifact": "models/walkforward_experiment_results.csv; models/guided_alpha_comparison.csv",
            "paper_role": "Primary fold-local alpha comparison for global, classical, vanilla learned, and guided learned regimes.",
        },
        {
            "paper_section": "Statistical evidence",
            "artifact_type": "statistical test",
            "artifact": "models/statistical_test_summary.csv; models/paper_statistical_summary.csv",
            "paper_role": "Separates point-estimate wins from fold-level significance and multiple-testing-safe claims.",
        },
        {
            "paper_section": "Mechanism",
            "artifact_type": "regime quality",
            "artifact": "models/regime_quality_summary.csv; models/guided_encoder_comparison.csv",
            "paper_role": "Measures structural alignment, persistence, entropy, NMI, ARI, and purity.",
        },
        {
            "paper_section": "Ablations",
            "artifact_type": "ablation",
            "artifact": "models/ablation_summary.csv; models/paper_claim_tests.csv",
            "paper_role": "Tests objective guidance, assignment layer, augmentation view, and classical-reference comparisons.",
        },
        {
            "paper_section": "Robustness",
            "artifact_type": "stress test",
            "artifact": "models/robustness_summary.csv; models/robustness_stress_summary.csv",
            "paper_role": "Checks symbol/horizon robustness and cost/threshold/period sensitivity.",
        },
        {
            "paper_section": "Interpretability",
            "artifact_type": "feature attribution",
            "artifact": "models/feature_importance_by_regime.csv; models/feature_family_summary.csv",
            "paper_role": "Shows fold-local feature drivers by method/regime without making causal claims.",
        },
        {
            "paper_section": "Reproducibility",
            "artifact_type": "reproduction package",
            "artifact": "reproduce.ps1; reports/environment.md; reports/artifact_manifest.md; reports/reproduction_checklist.md",
            "paper_role": "Documents smoke/full/dashboard reproduction paths, environment split, and artifact policy.",
        },
    ]


def write_artifact_map(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["paper_section", "artifact_type", "artifact", "paper_role"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_checklist() -> str:
    return """# Paper Submission Checklist

## Phase 29 Status

This checklist tracks what remains before the project should be treated as a submission-ready paper package.

## Ready

- Central research question is frozen in `reports/paper_protocol.md`.
- Claim boundaries are frozen in `reports/claim_registry.md`.
- Literature positioning exists in `reports/related_work.md` and `reports/literature_matrix.csv`.
- Main fold-local benchmark artifacts exist.
- Phase 25 ablations and Phase 26 paper claim tests exist.
- Phase 29 manuscript prose pass is complete.
- Validation audit has no critical failures.

## Needs Human Writing

- Replace paper-planning source names with the final venue citation style.
- Add final figure numbers after choosing the paper template.
- Tune the abstract to the final submission venue length.
- Decide whether the appendix includes the Streamlit dashboard screenshots.

## Optional Before Submission

- Crypto-20 downstream alpha retest only if the paper needs a predictive generalization section.
- Crypto-50 expansion only if Crypto-20 downstream evidence and compute budget justify it.
- Fold-local encoder retraining only if the paper needs a stronger leakage-resistance appendix.
- Full time-frequency encoder run only if Phase 25/26 gates are reopened.

## Must Not Claim

- Do not claim HMM states are ground truth.
- Do not claim a profitable or deployable trading strategy.
- Do not claim guided-HMM statistically dominates raw-feature HMM at 5%.
- Do not claim downstream alpha generalization outside BTC/ETH before the Crypto-20 fold-local alpha retest.
- Do not treat Phase 35 structural Crypto-20 results as trading or alpha evidence.
"""


def build_paper(results: pd.DataFrame, claims: pd.DataFrame, artifact_rows: list[dict[str, str]]) -> str:
    main_results = build_main_result_table(results)
    claim_table = build_claim_table(claims)
    artifact_df = pd.DataFrame(artifact_rows)
    best_method = "not available"
    if not results.empty and "IC" in results.columns:
        row = results.sort_values("IC", ascending=False).iloc[0]
        best_method = f"{row['method']} (IC={fmt(row['IC'])}, Sharpe={fmt(row['Sharpe'])})"

    return f"""# HMM-Guided Contrastive Representations for Regime-Conditioned Financial Alpha Modeling

## Paper Status

Phase 29 manuscript prose pass. This Markdown draft is generated from current artifacts and is intended to be converted into a venue template in the next phase.

## Abstract

Financial alpha models often behave differently across market regimes, yet regime labels are rarely observed and may be unstable during transitions. Classical Hidden Markov Models impose useful temporal state discipline, while contrastive time-series encoders can learn richer representations from raw features. This paper studies whether those two ideas can be combined for regime-conditioned financial alpha modeling. Regime Alpha Benchmark benchmarks global LightGBM, raw-feature HMM regimes, clustering and volatility baselines, vanilla contrastive regimes, and an HMM-guided contrastive encoder on BTCUSDT and ETHUSDT hourly data. The two-asset scope is intentional: BTC and ETH provide a controlled crypto setting with relatively similar trading venue structure, letting the benchmark isolate regime-method behavior before claiming cross-asset alpha generalization. The evaluation uses triple-barrier labels, expanding purged walk-forward validation, transaction costs, robustness grids, fold-level statistical tests, ablation summaries, and fold-local interpretability. The strongest current point-estimate method is {best_method}. The evidence supports a clear mechanism: learned embeddings are more useful when paired with sequential HMM assignment, and HMM-guided weak supervision improves the learned-regime path. A pre-specified Crypto-20 extension supports structural transfer of the guided objective, but the Crypto-20 downstream alpha retest remains future work. The contribution is therefore a reproducible empirical benchmark and cautious model-side intervention, not a claim of profitable trading or statistically dominant alpha.

## 1. Introduction

Market behavior is non-stationary. Momentum signals, volatility filters, and liquidity-sensitive features can all change their predictive value when the market moves from persistent trends into choppy or stressed periods. Regime-conditioned alpha modeling tries to address this by training or weighting predictors differently across market states. The difficulty is that financial regimes are latent: they are not directly labeled, and hard regime assignments can be unreliable near transitions.

The classical answer is to estimate regimes with a sequential model such as a Gaussian HMM. This gives persistent state assignments and transition probabilities, but it depends heavily on the chosen feature set and distributional assumptions. A newer answer is to learn time-series embeddings through contrastive objectives and then cluster the embedding space. This can discover nonlinear representations, but the original Regime Alpha Benchmark benchmark found that vanilla contrastive-GMM regimes were smooth yet weak for downstream alpha.

This paper studies a hybrid question: can a contrastive encoder become more useful if its training objective is guided by a classical sequential reference? The proposed path uses raw-feature HMM states as weak supervision for contrastive representation learning, then evaluates both GMM and HMM assignment layers on the learned embeddings. HMM states are treated as proxy states, not ground truth. The goal is to test whether sequential structure improves learned regimes under fair financial validation.

### Contributions

1. A reproducible regime-conditioned alpha benchmark using common labels, folds, transaction costs, and test rows across global, classical, vanilla learned, and HMM-guided learned methods.
2. An HMM-guided contrastive representation path that uses classical state sequences as weak supervision while explicitly rejecting the idea that HMM states are ground truth.
3. A fold-local downstream alpha test showing that the guided embedding path only becomes useful when paired with sequential HMM assignment rather than memoryless GMM assignment.
4. A validation and evidence layer that includes leakage audit, robustness checks, fold-level statistics, multiple-testing-aware claim language, ablations, and fold-local interpretability.
5. A cautious empirical finding: guided-HMM produces the strongest current point estimates and stress robustness, while statistical dominance over raw-feature HMM remains unproven.

## 2. Related Work

This work sits between four research areas.

- Classical financial regime detection: HMMs, Gaussian mixtures, volatility regimes, and regime-switching models.
- Contrastive time-series representation learning: vanilla temporal contrastive objectives, TS2Vec/TNC/CoST-style representation learning, and recent time-frequency or hard-negative variants.
- Financial ML validation: triple-barrier labels, purging, embargoing, walk-forward evaluation, multiple-testing caution, and Probabilistic Sharpe diagnostics.
- Regime-conditioned alpha modeling: using state information to train, select, or weight predictive models.

The central distinction is that this paper does not treat HMM states as true labels. They are a classical sequential reference that can discipline learned embeddings and provide a competitive baseline. This framing allows the benchmark to ask whether learned regimes benefit from sequential structure without claiming that any fitted state sequence is the real market ontology.

## 3. Data and Labels

The current downstream-alpha dataset contains BTCUSDT and ETHUSDT hourly bars from 2024-04-26 to 2026-04-26. This narrow predictive universe is an experimental-control decision rather than an unnoticed generalization claim: both assets trade continuously, share similar crypto-market microstructure, and allow the benchmark to compare regime methods without mixing asset classes, exchange schedules, or equity/FX-specific effects. Separately, the project pre-specifies a Crypto-20 universe and runs structural guided-encoder diagnostics on it; those diagnostics test representation transfer, not multi-asset alpha improvement.

The feature store contains 22 engineered hourly features covering returns, realized volatility, volatility-of-volatility, liquidity proxies, order-flow proxy behavior, RSI/MACD/Bollinger-style technical state, distribution shape, and volume behavior. The primary target is `tb_label_8h`, an 8-hour triple-barrier label with down, neutral, and up classes. Secondary directional, triple-barrier, forward-return, and volatility-adjusted-return labels are retained for diagnostics and robustness, but the paper reports the primary target first.

The label diagnostics are intentionally part of the artifact set. They document class balance, neutral share, missing target rows, and the expected tail loss from horizon shifting. This prevents the benchmark from hiding label imbalance or silently comparing models on different prediction universes.

## 4. Methods

### 4.1 Baselines

- `global_lgbm`: no-regime LightGBM baseline.
- `regime_lgbm_hmm`: raw-feature Gaussian HMM regime baseline.
- `regime_lgbm_kmeans`: non-sequential clustering baseline.
- `regime_lgbm_vol_bucket`: simple volatility-state baseline.
- `regime_lgbm_contrastive`: vanilla contrastive embedding with GMM assignment.
- `regime_lgbm_contrastive_hmm`: vanilla contrastive embedding with HMM assignment.

### 4.2 Proposed Method

The proposed method is `regime_lgbm_hmm_guided_hmm`: an HMM-guided contrastive encoder followed by sequential HMM assignment and regime-conditioned LightGBM alpha models. The encoder uses the HMM state sequence to define weak same-state positives and boundary-aware negatives, but the downstream regime assignment is still evaluated as a model component rather than assumed correct.

### 4.3 Alpha Modeling

All alpha models use the same primary target, `tb_label_8h`, and the same walk-forward folds. LightGBM outputs multiclass probabilities over down, neutral, and up labels. The alpha score is `P(up) - P(down)`. A position is taken only when the score exceeds the threshold and the neutral class is not dominant. Transaction costs are applied in the evaluation layer rather than treated as an afterthought.

## 5. Validation and Statistical Protocol

The predictive benchmark uses expanding walk-forward validation, a six-month initial training window, one-month test steps, a five-day embargo, and an eight-bar label-horizon purge. This produces 18 walk-forward test folds. That is a limited sample for fold-level significance testing, and the paper treats it as such. The tradeoff is deliberate: 18 embargoed walk-forward folds are statistically low-power, but they are more defensible than treating thousands of overlapping hourly labels as independent observations.

Predictive paper claims use fold-local regime assignment artifacts, not the older offline/global regime files used for descriptive plots. The validation audit checks row separation, embargo spacing, target-horizon purge, coverage parity, duplicate predictions, and consistency between predictions and result summaries.

Statistical interpretation is fold-level first. Row-level diagnostics are useful for forecast-loss and calibration discussion, but adjacent financial labels overlap and should not be treated as independent evidence for IC or Sharpe claims. The paper therefore separates point estimates, fold-level tests, multiple-testing diagnostics, and Probabilistic Sharpe diagnostics.

## 6. Results

### 6.1 Main Fold-Local Alpha Results

{markdown_table(main_results, list(main_results.columns)) if not main_results.empty else "_Main result artifact missing._"}

### 6.2 Paper Claim Tests

{markdown_table(claim_table, list(claim_table.columns)) if not claim_table.empty else "_Paper claim artifact missing._"}

### 6.3 Current Interpretation

The headline result is mechanistic rather than promotional: sequential structure in the assignment layer is the key driver, and HMM-guided representation learning makes that assignment layer more effective. HMM assignment improves the guided learned-regime path relative to guided-GMM on all focused point-estimate metrics and is raw-suggestive on fold-level IC, but it is not corrected significant. Guided-HMM also improves all focused point-estimate metrics versus raw-feature HMM. However, the main guided-HMM versus raw-feature HMM IC comparison has `p=0.801`, so it cannot be framed as a statistical win. This is still useful because it separates a promising mechanism from an overclaimed victory.

## 7. Robustness

The robustness evidence has two layers:

- Symbol/horizon robustness: useful for showing where the conclusion is stable or mixed.
- Cost/threshold/market-period stress robustness: stronger for the primary BTC+ETH 8h setup.

The safe wording is that guided-HMM is stress-robust on the primary BTC+ETH 8h benchmark, not universally dominant across all assets and horizons. This distinction matters because a method can be strong under transaction-cost and threshold stress while still showing mixed behavior across target horizons or single-symbol subsets.

## 8. Interpretability

Fold-local feature attribution shows which feature families matter inside each method and regime. This is diagnostic interpretability, not causal explanation. Its strongest use is plausibility: the guided-HMM regime-conditioned models rely heavily on volatility state, volatility-of-volatility, momentum/autocorrelation, and distribution-shape features, which is more convincing than a regime layer driven by arbitrary identifiers.

## 9. Ablations

The ablation suite tests objective guidance, assignment layer, augmentation view, and classical-reference comparisons. The strongest evidence is for the assignment layer: guided embeddings become useful when assignment respects temporal state persistence. The current time-frequency prototype is informative but negative; it should not receive a full downstream expansion until its structural evidence improves.

## 10. Limitations

- HMM states are proxy states, not ground truth.
- The downstream alpha benchmark intentionally covers BTCUSDT and ETHUSDT as a controlled crypto setting; the Crypto-20 evidence is structural only and does not claim cross-asset alpha generalization.
- The fold-level statistical tests use 18 walk-forward folds, which is defensible but low-power.
- The guided-HMM edge over raw-feature HMM is directionally supported but not statistically significant at 5%; the main IC comparison has `p=0.801`.
- Backtest results are research diagnostics, not live trading claims.
- Interpretability results are model-specific and not causal.
- The encoder is trained offline; fold-local encoder retraining remains a possible appendix experiment if compute allows.

## 11. Reproducibility

The public repository separates dashboard reproduction from full research reproduction. The dashboard runs from curated committed summary artifacts and minimal Streamlit dependencies. Full research reproduction uses `requirements-research.txt`, local data access, encoder training, fold-local validation, robustness checks, statistical tests, and validation audit.

The reproduction helper supports three modes: smoke, full, and dashboard. Raw data, DuckDB databases, model weights, embeddings, and row-level prediction files remain excluded from GitHub.

## 12. Conclusion

Regime Alpha Benchmark shows that learned market-regime representations need sequential discipline to become useful in this benchmark. Vanilla contrastive-GMM regimes are weak downstream, while HMM-guided embeddings paired with HMM assignment produce the strongest current BTC/ETH point estimates and stress robustness. The Crypto-20 guided-encoder run adds structural evidence that the representation objective scales beyond the pilot, while predictive multi-asset alpha remains unclaimed until a fold-local Crypto-20 retest is complete. The central publishable finding is not a claim of profitable trading or statistical dominance. It is a controlled empirical result: classical sequential structure can improve deep learned regime representations, and the assignment layer is a major driver of downstream usefulness.

## Figure and Table Plan

{markdown_table(artifact_df, ["paper_section", "artifact_type", "artifact", "paper_role"])}
"""


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = read_csv(MODELS_DIR / "walkforward_experiment_results.csv")
    claims = read_csv(MODELS_DIR / "paper_statistical_summary.csv")
    artifact_rows = build_artifact_map()

    paper_path = output_dir / "main.md"
    artifact_map_path = REPORTS_DIR / "paper_artifact_map.csv"
    checklist_path = REPORTS_DIR / "paper_submission_checklist.md"

    if paper_path.exists():
        print(f"Preserved human-reviewed paper draft: {paper_path}")
    else:
        paper_path.write_text(build_paper(results, claims, artifact_rows), encoding="utf-8")
        print(f"Initialized paper draft: {paper_path}")

    if artifact_map_path.exists():
        print(f"Preserved human-reviewed artifact map: {artifact_map_path}")
    else:
        write_artifact_map(artifact_map_path, artifact_rows)
        print(f"Initialized artifact map: {artifact_map_path}")

    if checklist_path.exists():
        print(f"Preserved human-reviewed submission checklist: {checklist_path}")
    else:
        checklist_path.write_text(build_checklist(), encoding="utf-8")
        print(f"Initialized submission checklist: {checklist_path}")


if __name__ == "__main__":
    main()
